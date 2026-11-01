"""
Zero-dependency web server (stdlib only) — serves the page and a JSON API.
  GET  /api/teams
  GET  /api/predict?home=BOS&away=NYK&date=2026-11-02&line=224.5
  POST /api/admin/roster   (header X-Admin-Token; roster/injury updates; works both phases)
  GET  /api/status
For production, put this behind nginx/gunicorn-style hosting or use app_fastapi.py.
"""
import json, os, hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
from pathlib import Path
import engine

ADMIN_TOKEN = os.environ.get("HOOPS_ADMIN_TOKEN", "")
STATIC = Path(__file__).parent / "static"
# Sites allowed to call the JSON API from a browser (e.g. the Squarespace page).
ALLOWED_ORIGINS = {o.strip() for o in os.environ.get(
    "HOOPS_ALLOWED_ORIGINS", "https://www.johllanalytics.com,https://johllanalytics.com").split(",") if o.strip()}

class H(BaseHTTPRequestHandler):
    def _cors(self):
        origin = self.headers.get("Origin", "")
        if origin in ALLOWED_ORIGINS:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Admin-Token")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def _send(self, code, obj, ctype="application/json"):
        body = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", ctype)
        self._cors()
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204); self._cors()
        self.send_header("Content-Length", "0"); self.end_headers()

    def do_GET(self):
        u = urlparse(self.path); q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path == "/":
                return self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            if u.path == "/api/teams":
                return self._send(200, sorted(engine.load_rosters()))
            if u.path == "/api/status":
                w = engine.load_weights()
                return self._send(200, {"phase": engine.phase(), "trained_on_games": w["n_games"],
                                        "typical_error_pts": round(w["cv_mae"], 1)})
            if u.path == "/api/predict":
                line = float(q["line"]) if q.get("line") else None
                return self._send(200, engine.predict(q["home"], q["away"], q.get("date"), line))
        except (KeyError, ValueError, FileNotFoundError) as e:
            return self._send(400, {"error": str(e)})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/api/admin/roster":
            return self._send(404, {"error": "not found"})
        tok = self.headers.get("X-Admin-Token", "")
        if not ADMIN_TOKEN or not hmac.compare_digest(tok, ADMIN_TOKEN):
            return self._send(403, {"error": "forbidden"})
        try:
            n = engine.apply_daily_update(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self._send(200, {"teams_updated": n})
        except Exception as e:
            self._send(400, {"error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"http://localhost:{port}"); ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()
