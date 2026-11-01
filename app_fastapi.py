"""Optional FastAPI version of the same API:  uvicorn app_fastapi:app"""
import os
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
import engine

app = FastAPI(title="Hoops Totals")

@app.get("/")
def home(): return FileResponse("static/index.html")

@app.get("/api/teams")
def teams(): return sorted(engine.load_rosters())

@app.get("/api/predict")
def predict(home: str, away: str, date: str | None = None, line: float | None = None):
    try: return engine.predict(home, away, date, line)
    except (KeyError, ValueError, FileNotFoundError) as e: raise HTTPException(400, str(e))

@app.post("/api/admin/roster")
def roster(update: dict, x_admin_token: str = Header("")):
    if not os.environ.get("HOOPS_ADMIN_TOKEN") or x_admin_token != os.environ["HOOPS_ADMIN_TOKEN"]:
        raise HTTPException(403, "forbidden")
    return {"teams_updated": engine.apply_daily_update(update)}
