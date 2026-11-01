"""Command line for the operator. The public website only ever calls predict()."""
import argparse, json, engine

p = argparse.ArgumentParser(prog="hoops")
s = p.add_subparsers(dest="cmd", required=True)
s.add_parser("phase").add_argument("value", nargs="?", choices=["offseason", "inseason"])
t = s.add_parser("train");   t.add_argument("csv")
d = s.add_parser("daily");   d.add_argument("json_file", help="roster/injury update file")
g = s.add_parser("predict"); g.add_argument("home"); g.add_argument("away")
g.add_argument("--date"); g.add_argument("--line", type=float)
a = p.parse_args()

if a.cmd == "phase":
    if a.value: engine.set_phase(a.value)
    print(engine.phase())
elif a.cmd == "train":
    w = engine.train(a.csv); print(f"trained on {w['n_games']} games, CV MAE {w['cv_mae']:.2f} pts")
elif a.cmd == "daily":
    print(engine.apply_daily_update(json.load(open(a.json_file))), "teams updated")
else:
    print(json.dumps(engine.predict(a.home, a.away, a.date, a.line), indent=2))
