"""
hoops_totals.engine — framework-free core for basketball over/under predictions.

Two phases (set in config.json -> "phase"):
  * "offseason": train()  re-fits the machine-learning weights from historical games.
                 Rosters can also be edited.
  * "inseason" : weights are FROZEN (train() refuses to run). Only rosters change,
                 via update_roster() / apply_daily_update(), and predictions are
                 computed from the frozen weights + the latest rosters.

Everything here is plain Python + numpy, so it drops into a CLI, a web server,
a serverless function, or a notebook unchanged.
"""
from __future__ import annotations
import csv, json, math, os, threading
from datetime import date
from pathlib import Path
import numpy as np

DATA = Path(os.environ.get("HOOPS_DATA_DIR", Path(__file__).parent / "data"))
CONFIG_PATH = DATA / "config.json"
WEIGHTS_PATH = DATA / "weights.json"
ROSTERS_PATH = DATA / "rosters.json"

FEATURES = [
    "pace_sum",        # sum of both teams' possessions per 48 min
    "h_off_vs_a_def",  # home offensive rating + away defensive rating (pts/100 poss)
    "a_off_vs_h_def",  # away offensive rating + home defensive rating
    "h_rest", "a_rest",            # days of rest (capped at 4)
    "h_b2b", "a_b2b",              # back-to-back flags
    "h_missing", "a_missing",      # share of rotation minutes unavailable (0-1)
    "formula_total",   # physics baseline: pace x combined efficiency (possessions x pts/possession)
]

_lock = threading.Lock()


# ----------------------------------------------------------------- config / io
def _read(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))

def _write(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)                       # atomic swap, safe under web traffic

def get_config() -> dict:
    return _read(CONFIG_PATH, {"phase": "offseason", "ridge_alpha": 5.0})

def set_phase(phase: str) -> None:
    if phase not in ("offseason", "inseason"):
        raise ValueError("phase must be 'offseason' or 'inseason'")
    cfg = get_config(); cfg["phase"] = phase
    with _lock:
        _write(CONFIG_PATH, cfg)

def phase() -> str:
    return get_config()["phase"]


# --------------------------------------------------------------------- rosters
# rosters.json: {"BOS": {"players":[{"name","min","ortg","drtg","status"}],
#                        "pace": 99.1, "last_game": "2026-10-28"}, ...}
# status: "active" | "out" | "questionable" (questionable counts as 50% available)
AVAIL = {"active": 1.0, "questionable": 0.5, "out": 0.0}

def load_rosters() -> dict:
    return _read(ROSTERS_PATH, {})

def team_snapshot(team: dict) -> dict:
    """Collapse a roster into minutes-weighted offense/defense and missing-minutes share."""
    players = team["players"]
    total_min = sum(p["min"] for p in players) or 1.0
    avail_min = sum(p["min"] * AVAIL.get(p.get("status", "active"), 1.0) for p in players)
    if avail_min <= 0:
        avail_min = 1e-9
    off = sum(p["min"] * AVAIL.get(p.get("status", "active"), 1.0) * p["ortg"] for p in players) / avail_min
    dfn = sum(p["min"] * AVAIL.get(p.get("status", "active"), 1.0) * p["drtg"] for p in players) / avail_min
    return {"off": off, "def": dfn, "pace": team["pace"],
            "missing": max(0.0, 1.0 - avail_min / total_min)}

def update_roster(team: str, players: list[dict] | None = None, pace: float | None = None,
                  statuses: dict[str, str] | None = None, last_game: str | None = None) -> dict:
    """Allowed in BOTH phases (rosters are the only thing that moves in-season)."""
    with _lock:
        rosters = load_rosters()
        t = rosters.setdefault(team.upper(), {"players": [], "pace": 98.0})
        if players is not None:
            t["players"] = players
        if pace is not None:
            t["pace"] = pace
        if last_game is not None:
            t["last_game"] = last_game
        for name, status in (statuses or {}).items():
            if status not in AVAIL:
                raise ValueError(f"bad status {status!r}")
            for p in t["players"]:
                if p["name"].lower() == name.lower():
                    p["status"] = status
        _write(ROSTERS_PATH, rosters)
        return t

def apply_daily_update(update: dict) -> int:
    """update = {"BOS": {"statuses": {"J. Tatum": "out"}, "last_game": "2026-10-29"}, ...}"""
    for team, u in update.items():
        update_roster(team, statuses=u.get("statuses"), players=u.get("players"),
                      pace=u.get("pace"), last_game=u.get("last_game"))
    return len(update)


# -------------------------------------------------------------------- features
def rest_days(team: dict, game_date: date) -> int:
    lg = team.get("last_game")
    if not lg:
        return 3
    return max(0, min(4, (game_date - date.fromisoformat(lg)).days - 1))

def build_features(home: dict, away: dict, h_rest: int, a_rest: int) -> list[float]:
    h, a = team_snapshot(home), team_snapshot(away)
    return [h["pace"] + a["pace"],
            h["off"] + a["def"], a["off"] + h["def"],
            h_rest, a_rest, float(h_rest == 0), float(a_rest == 0),
            h["missing"], a["missing"],
            (h["pace"] + a["pace"]) * ((h["off"] + a["def"]) + (a["off"] + h["def"])) / 400.0]


# -------------------------------------------------------------------- training
def train(csv_path: str | Path, alpha: float | None = None) -> dict:
    """OFF-SEASON ONLY. CSV needs the FEATURES columns plus 'total' (final combined score)."""
    if phase() != "offseason":
        raise PermissionError("Weights are frozen in-season. Switch phase to 'offseason' to retrain.")
    alpha = get_config().get("ridge_alpha", 5.0) if alpha is None else alpha
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) < 50:
        raise ValueError("Need at least 50 historical games to train.")
    X = np.array([[float(r[c]) for c in FEATURES] for r in rows])
    y = np.array([float(r["total"]) for r in rows])

    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    # ridge regression, closed form (intercept unpenalised)
    A = Z.T @ Z + alpha * np.eye(Z.shape[1])
    w = np.linalg.solve(A, Z.T @ (y - y.mean()))
    b = float(y.mean())

    # 5-fold CV error + residual spread (used to turn a prediction into P(over))
    idx = np.random.default_rng(0).permutation(len(y)); folds = np.array_split(idx, 5); errs = []
    for k in range(5):
        te = folds[k]; tr = np.concatenate([folds[j] for j in range(5) if j != k])
        wk = np.linalg.solve(Z[tr].T @ Z[tr] + alpha * np.eye(Z.shape[1]), Z[tr].T @ (y[tr] - y[tr].mean()))
        errs.extend(y[te] - (Z[te] @ wk + y[tr].mean()))
    errs = np.array(errs)

    weights = {"features": FEATURES, "mean": mu.tolist(), "std": sd.tolist(),
               "coef": w.tolist(), "intercept": b,
               "resid_sd": float(errs.std()), "cv_mae": float(np.abs(errs).mean()),
               "n_games": len(y), "alpha": alpha}
    with _lock:
        _write(WEIGHTS_PATH, weights)
    return weights


# ------------------------------------------------------------------ prediction
def load_weights() -> dict:
    w = _read(WEIGHTS_PATH)
    if not w:
        raise FileNotFoundError("No trained weights yet. Run training in the off-season first.")
    return w

def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def predict(home_team: str, away_team: str, game_date: str | None = None,
            book_line: float | None = None) -> dict:
    """Works in both phases. Uses frozen weights + CURRENT rosters."""
    rosters, w = load_rosters(), load_weights()
    ht, at = home_team.upper(), away_team.upper()
    for t in (ht, at):
        if t not in rosters:
            raise KeyError(f"Unknown team {t!r}")
    gd = date.fromisoformat(game_date) if game_date else date.today()
    x = np.array(build_features(rosters[ht], rosters[at],
                                rest_days(rosters[ht], gd), rest_days(rosters[at], gd)))
    z = (x - np.array(w["mean"])) / np.array(w["std"])
    total = float(z @ np.array(w["coef"]) + w["intercept"])
    sd = w["resid_sd"]
    out = {"home": ht, "away": at, "date": gd.isoformat(), "phase": phase(),
           "predicted_total": round(total, 1),
           "suggested_line": round(total * 2) / 2,        # nearest half point
           "typical_error_pts": round(w["cv_mae"], 1),
           "drivers": {f: round(float(c * zi), 2) for f, c, zi in zip(w["features"], w["coef"], z)}}
    if book_line is not None:
        p_over = 1 - _norm_cdf((book_line - total) / sd)
        out.update({"book_line": book_line, "p_over": round(p_over, 3), "p_under": round(1 - p_over, 3),
                    "lean": "OVER" if p_over > 0.5 else "UNDER"})
    return out
