"""Creates demo rosters + synthetic history so you can run the project end-to-end.
Replace with real data (see README)."""
import csv, json, random, engine
random.seed(7)
TEAMS = ["BOS","NYK","MIL","DEN","LAL","GSW","PHX","MIA"]
rosters = {}
for t in TEAMS:
    base_o, base_d = random.uniform(108,119), random.uniform(108,119)
    rosters[t] = {"pace": round(random.uniform(96,102),1), "last_game": "2026-10-27",
                  "players": [{"name": f"{t} Player {i+1}", "min": m,
                               "ortg": round(base_o+random.uniform(-6,6),1),
                               "drtg": round(base_d+random.uniform(-6,6),1), "status": "active"}
                              for i,m in enumerate([34,33,30,28,26,22,18,15,12,10])]}
engine.DATA.mkdir(exist_ok=True)
engine._write(engine.ROSTERS_PATH, rosters)
engine._write(engine.CONFIG_PATH, {"phase": "offseason", "ridge_alpha": 5.0})

with open(engine.DATA / "history_sample.csv", "w", newline="") as f:
    wr = csv.writer(f); wr.writerow(engine.FEATURES + ["total"])
    for _ in range(1200):
        h, a = random.sample(TEAMS, 2)
        hr, ar = random.randint(0,4), random.randint(0,4)
        hs = dict(rosters[h]); as_ = dict(rosters[a])
        x = engine.build_features(hs, as_, hr, ar)
        x[7] = random.choice([0,0,0,.1,.25]); x[8] = random.choice([0,0,0,.1,.25])
        total = x[0]*(x[1]+x[2])/400 - 25*x[7] - 25*x[8] - 2.5*(x[5]+x[6]) + random.gauss(0,11)
        wr.writerow([round(v,3) for v in x] + [round(total,1)])
print("sample data written")
