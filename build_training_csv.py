"""
Builds data/history.csv for engine.train() from raw historical box-score data,
using ONLY information that existed before each game (no future leakage).

INPUT 1 (required): team_games.csv — one row per TEAM per game
    game_id, season, date(YYYY-MM-DD), team, home(1 or 0), pts, opp_pts, poss
    (poss = possessions; if you only have box scores: FGA + 0.44*FTA - OREB + TOV)

INPUT 2 (optional): player_games.csv — one row per player who PLAYED in a game
    date, team, player, min
    Used to compute the "missing" feature (share of rotation minutes absent).
    Without it, "missing" is 0 in training and the model can't learn what injuries do.

Run:  python build_training_csv.py team_games.csv [player_games.csv]
"""
import csv, os, sys
from collections import defaultdict, deque
from datetime import date
import engine

WINDOW, MIN_GAMES, ROTATION = 15, 10, 8     # rolling games, warm-up games, rotation size

team_file = sys.argv[1]
player_file = sys.argv[2] if len(sys.argv) > 2 else None

games = defaultdict(dict)
for r in csv.DictReader(open(team_file, newline="", encoding="utf-8")):
    games[r["game_id"]]["home" if r["home"] == "1" else "away"] = r
order = sorted((g for g in games.values() if "home" in g and "away" in g),
               key=lambda g: g["home"]["date"])              # oldest first

pg = defaultdict(list)                                       # (date, team) -> [(player, min)]
if player_file:
    for r in csv.DictReader(open(player_file, newline="", encoding="utf-8")):
        pg[(r["date"], r["team"])].append((r["player"], float(r["min"])))

hist = defaultdict(lambda: deque(maxlen=WINDOW))             # (season, team) -> recent (off, def, pace)
last = {}                                                    # (season, team) -> date of previous game
pstats = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))  # (season, team) -> player -> [minutes, games]

def rating(K):
    h = hist[K]; n = len(h)
    return sum(x[0] for x in h) / n, sum(x[1] for x in h) / n, sum(x[2] for x in h) / n

def rest(K, gd):
    return 3 if K not in last else min(4, max(0, (gd - last[K]).days - 1))

def missing(K, d, team):
    if (d, team) not in pg or not pstats[K]:
        return 0.0
    avg = {p: m / g for p, (m, g) in pstats[K].items()}
    rot = sorted(avg, key=avg.get, reverse=True)[:ROTATION]
    played = {p for p, _ in pg[(d, team)]}
    tot = sum(avg[p] for p in rot)
    return sum(avg[p] for p in rot if p not in played) / tot if tot else 0.0

rows = []
for g in order:
    H, A = g["home"], g["away"]
    kh, ka = (H["season"], H["team"]), (A["season"], A["team"])
    gd = date.fromisoformat(H["date"])

    # 1) features from history BEFORE this game
    if len(hist[kh]) >= MIN_GAMES and len(hist[ka]) >= MIN_GAMES:
        ho, hd, hp = rating(kh); ao, ad, ap = rating(ka)
        hr, ar = rest(kh, gd), rest(ka, gd)
        rows.append([hp + ap, ho + ad, ao + hd, hr, ar, int(hr == 0), int(ar == 0),
                     missing(kh, H["date"], H["team"]), missing(ka, A["date"], A["team"]),
                     (hp + ap) * ((ho + ad) + (ao + hd)) / 400.0,
                     float(H["pts"]) + float(A["pts"])])      # target: final combined score

    # 2) only now fold this game into the history
    for T, K in ((H, kh), (A, ka)):
        poss = float(T["poss"])
        hist[K].append((100 * float(T["pts"]) / poss, 100 * float(T["opp_pts"]) / poss, poss))
        last[K] = gd
        for p, m in pg.get((T["date"], T["team"]), []):
            pstats[K][p][0] += m; pstats[K][p][1] += 1

out = engine.DATA / "history.csv"
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(engine.FEATURES + ["total"])
    w.writerows([[round(v, 3) for v in r] for r in rows])
print(f"wrote {len(rows)} training rows to {out}")
