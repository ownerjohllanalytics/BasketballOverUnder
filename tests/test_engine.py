import os, sys, tempfile
os.environ["HOOPS_DATA_DIR"] = tempfile.mkdtemp()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import engine, make_sample_data   # writes demo rosters + history into the temp dir

def test_offseason_trains_then_inseason_freezes():
    w = engine.train(engine.DATA / "history_sample.csv")
    assert w["cv_mae"] < 12, w["cv_mae"]
    engine.set_phase("inseason")
    try:
        engine.train(engine.DATA / "history_sample.csv"); raise AssertionError("train ran in-season")
    except PermissionError:
        pass
    before = engine.WEIGHTS_PATH.read_text()
    engine.update_roster("BOS", statuses={"BOS Player 1": "out"})
    assert engine.WEIGHTS_PATH.read_text() == before

def test_injury_changes_prediction_not_weights():
    engine.update_roster("BOS", statuses={"BOS Player 1": "active", "BOS Player 2": "active"})
    a = engine.predict("BOS", "NYK", "2026-11-01")["predicted_total"]
    engine.apply_daily_update({"BOS": {"statuses": {"BOS Player 1": "out", "BOS Player 2": "out"}}})
    b = engine.predict("BOS", "NYK", "2026-11-01")["predicted_total"]
    assert a != b, (a, b)

def test_book_line_probabilities():
    r = engine.predict("DEN", "LAL", "2026-11-01", book_line=200.0)
    assert abs(r["p_over"] + r["p_under"] - 1) < 1e-6

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("PASS", n)
