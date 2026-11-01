"""
Refreshes data/rosters.json (the engine's live team data) from nba_api.

Pulls, for every team:
  * players: minutes per game, offensive/defensive rating (points per 100 possessions)
  * team pace
  * date of the team's most recent game (for rest / back-to-back)
Early in the season, current-season numbers are blended with last season's so a
3-game sample doesn't swing the ratings (weight shifts to this season as games pile up).

Injuries: nba_api has no injury feed, so statuses come from an optional injuries.csv
that you keep up to date (see below). Each run is a FULL refresh: everyone starts
"active", then injuries.csv is applied -- so players who healed clear automatically.

    injuries.csv            (header required; status = out | questionable | active)
    player,status
    Jayson Tatum,out
    Jalen Brunson,questionable

Usage:
    python update_rosters.py                       # current season, reads injuries.csv if present
    python update_rosters.py --injuries C:\\path\\injuries.csv
    python update_rosters.py --dry-run             # fetch + summarize, write nothing
    python update_rosters.py --season 2026-27

Allowed in BOTH phases (rosters are what changes in-season). Weights are never touched.
"""
import argparse, csv, sys, time, unicodedata
from datetime import date
from pathlib import Path
import pandas as pd
import engine

K_GAMES = 20          # prior-season weight, in "games"; larger = trust last season longer
MAX_PLAYERS = 12      # rotation size kept per team
MIN_MPG = 4.0         # ignore players below this blended minutes/game
PAUSE, RETRIES = 2.0, 4
STATUS_MAP = {"out": "out", "doubtful": "out", "questionable": "questionable",
              "day-to-day": "questionable", "probable": "active", "available": "active", "active": "active"}


# ------------------------------------------------------------------ seasons
def current_season(today: date | None = None) -> str:
    t = today or date.today()
    y = t.year if t.month >= 10 else t.year - 1
    return f"{y}-{(y + 1) % 100:02d}"

def prior_season(season: str) -> str:
    y = int(season[:4]) - 1
    return f"{y}-{(y + 1) % 100:02d}"


# ------------------------------------------------------------------ network
def _call(fn):
    last = None
    for attempt in range(1, RETRIES + 1):
        try:
            out = fn(); time.sleep(PAUSE); return out
        except Exception as e:
            last = e; wait = 5 * attempt
            print(f"  attempt {attempt}/{RETRIES} failed ({type(e).__name__}); retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"nba_api request failed: {last}")

def fetch_players(season):
    from nba_api.stats.endpoints import leaguedashplayerstats as ep
    return _call(lambda: ep.LeagueDashPlayerStats(
        season=season, measure_type_detailed_defense="Advanced", per_mode_detailed="PerGame",
        season_type_all_star="Regular Season", timeout=60).get_data_frames()[0])

def fetch_teams(season):
    from nba_api.stats.endpoints import leaguedashteamstats as ep
    return _call(lambda: ep.LeagueDashTeamStats(
        season=season, measure_type_detailed_defense="Advanced", per_mode_detailed="PerGame",
        season_type_all_star="Regular Season", timeout=60).get_data_frames()[0])

def fetch_last_games(season) -> dict:
    from nba_api.stats.endpoints import leaguegamelog
    df = _call(lambda: leaguegamelog.LeagueGameLog(
        season=season, player_or_team_abbreviation="T",
        season_type_all_star="Regular Season", timeout=60).get_data_frames()[0])
    if df.empty:
        return {}
    df["d"] = pd.to_datetime(df["GAME_DATE"]).dt.strftime("%Y-%m-%d")
    return df.groupby("TEAM_ABBREVIATION")["d"].max().to_dict()

def team_id_map() -> dict:
    from nba_api.stats.static import teams
    return {t["id"]: t["abbreviation"] for t in teams.get_teams()}


# ------------------------------------------------------------------ pure logic
def _norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return s.lower().replace(".", "").replace("'", "").replace("-", " ").strip()

def _w_current(gp_c, gp_p) -> float:
    """Weight on this season's number. No prior season -> 1.0; no games yet -> 0.0."""
    gp_c = 0.0 if pd.isna(gp_c) else float(gp_c)
    if pd.isna(gp_p):
        return 1.0 if gp_c > 0 else 0.0
    prior_strength = K_GAMES * min(1.0, float(gp_p) / 30.0)      # thin prior samples count less
    return gp_c / (gp_c + prior_strength) if (gp_c + prior_strength) > 0 else 1.0

def _blend(w, cur, pri):
    if pd.isna(pri): return cur
    if pd.isna(cur): return pri
    return w * cur + (1 - w) * pri

def build_rosters(cur_p, pri_p, cur_t, pri_t, last_games, id_to_abbr, injuries=None):
    """Returns ({team: {"players": [...], "pace": x, "last_game": d}}, unmatched_injury_names)."""
    injuries = injuries or {}
    cols = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "GP", "MIN", "OFF_RATING", "DEF_RATING"]
    base = cur_p[cols] if len(cur_p) else pri_p[cols].assign(GP=0)   # preseason fallback: last year's roster
    pri = pri_p[["PLAYER_ID", "GP", "MIN", "OFF_RATING", "DEF_RATING"]].rename(
        columns={c: c + "_p" for c in ["GP", "MIN", "OFF_RATING", "DEF_RATING"]})
    df = base.merge(pri, on="PLAYER_ID", how="left")

    teams, seen = {}, set()
    for r in df.itertuples(index=False):
        w = _w_current(r.GP, r.GP_p)
        mpg, off, dfn = (_blend(w, getattr(r, a), getattr(r, a + "_p")) for a in ("MIN", "OFF_RATING", "DEF_RATING"))
        if pd.isna(mpg) or pd.isna(off) or pd.isna(dfn) or mpg < MIN_MPG:
            continue
        key = _norm(r.PLAYER_NAME)
        status = STATUS_MAP.get(str(injuries.get(key, "active")).lower(), "active")
        if key in injuries: seen.add(key)
        teams.setdefault(r.TEAM_ABBREVIATION, []).append(
            {"name": r.PLAYER_NAME, "min": round(float(mpg), 1), "ortg": round(float(off), 1),
             "drtg": round(float(dfn), 1), "status": status})

    # team pace, blended the same way
    tp = cur_t[["TEAM_ID", "GP", "PACE"]] if len(cur_t) else pri_t[["TEAM_ID", "GP", "PACE"]].assign(GP=0)
    tpp = pri_t[["TEAM_ID", "GP", "PACE"]].rename(columns={"GP": "GP_p", "PACE": "PACE_p"})
    tp = tp.merge(tpp, on="TEAM_ID", how="left")
    pace = {}
    for r in tp.itertuples(index=False):
        abbr = id_to_abbr.get(r.TEAM_ID)
        if abbr: pace[abbr] = float(_blend(_w_current(r.GP, r.GP_p), r.PACE, r.PACE_p))

    out = {}
    for abbr, plist in teams.items():
        if abbr not in pace:
            continue
        plist.sort(key=lambda p: p["min"], reverse=True)
        out[abbr] = {"players": plist[:MAX_PLAYERS], "pace": round(pace[abbr], 1)}
        if abbr in last_games:
            out[abbr]["last_game"] = last_games[abbr]
    unmatched = sorted(set(injuries) - seen)
    return out, unmatched

def load_injuries(path) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, newline="", encoding="utf-8-sig") as f:
        return {_norm(r["player"]): r["status"].strip().lower() for r in csv.DictReader(f) if r.get("player")}


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", default=current_season())
    ap.add_argument("--injuries", default="injuries.csv")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    prev = prior_season(a.season)
    print(f"Season {a.season} (blending with {prev})")

    print("Fetching player ratings..."); cur_p, pri_p = fetch_players(a.season), fetch_players(prev)
    print("Fetching team pace...");      cur_t, pri_t = fetch_teams(a.season), fetch_teams(prev)
    print("Fetching latest game dates..."); last = fetch_last_games(a.season)
    if cur_p.empty:
        print("NOTE: no games played yet this season -- using last season's numbers and rosters.")

    injuries = load_injuries(a.injuries)
    rosters, unmatched = build_rosters(cur_p, pri_p, cur_t, pri_t, last, team_id_map(), injuries)
    if len(rosters) != 30:
        print(f"WARNING: expected 30 teams, built {len(rosters)}.")

    print(f"\n{'team':5} {'pace':>6} {'off':>6} {'def':>6} {'players':>8} {'out':>4}  last game")
    offs = []
    for t in sorted(rosters):
        s = engine.team_snapshot(rosters[t]); offs.append(s["off"])
        n_out = sum(p["status"] == "out" for p in rosters[t]["players"])
        print(f"{t:5} {s['pace']:6.1f} {s['off']:6.1f} {s['def']:6.1f} {len(rosters[t]['players']):8d} {n_out:4d}  {rosters[t].get('last_game', '-')}")
    if offs:
        print(f"\nLeague-average offensive rating: {sum(offs) / len(offs):.1f} (typically ~110-118)")
    if unmatched:
        print("Injury names not found on any roster (check spelling):", ", ".join(unmatched))

    if a.dry_run:
        print("\nDry run: nothing written."); return
    with engine._lock:                           # full snapshot replaces the old file (clears demo/stale teams)
        engine._write(engine.ROSTERS_PATH, rosters)
    print(f"\nWrote {len(rosters)} teams to {engine.ROSTERS_PATH}  (phase: {engine.phase()}, weights untouched)")

if __name__ == "__main__":
    main()
