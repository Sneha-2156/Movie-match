"""
web_app.py - MovieMatch in your browser (standard library only).

    python web_app.py              # opens http://127.0.0.1:8000
    python web_app.py --port 9000 --no-browser
"""
from __future__ import annotations

import argparse
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from catalog import Lookup, load_catalog
from evaluate import persona_benchmark
from recommender import RecommendationError, Recommender

HERE = Path(__file__).resolve().parent
INDEX_HTML = HERE / "static" / "index.html"
MAX_BODY = 20_000
LOCAL_HOSTS = {"localhost", "127.0.0.1", "[::1]"}


class App:
    """Shared state for the server: catalog, recommender and a cached evaluation."""

    def __init__(self, movies=None):
        self.movies = movies or load_catalog()
        self.lookup = Lookup(self.movies)
        self.rec = Recommender(self.movies)
        self._evaluation = None
        self._lock = threading.Lock()

    def catalog_payload(self) -> dict:
        return {"movies": [{"id": m.id, "title": m.title, "year": m.year, "genres": list(m.genres),
                            "language": m.language} for m in self.movies],
                "genres": self.lookup.genres, "languages": self.lookup.languages}

    def evaluation(self) -> dict:
        with self._lock:
            if self._evaluation is None:
                self._evaluation = persona_benchmark(self.movies)
            return self._evaluation

    def recommend(self, data: dict) -> list[dict]:
        """Validate a request body and return recommendation dicts."""
        ratings = {}
        for key, value in (data.get("ratings") or {}).items():
            mid, stars = int(key), float(value)
            if mid not in self.rec.by_id or not 1 <= stars <= 5:
                raise RecommendationError("Invalid rating data.")
            ratings[mid] = stars
        genres = [g for g in (self.lookup.genre(str(x)) for x in data.get("genres") or []) if g]
        excluded = [g for g in (self.lookup.genre(str(x)) for x in data.get("exclude_genres") or []) if g]
        languages = [l for l in (self.lookup.language(str(x)) for x in data.get("languages") or []) if l]
        year = lambda v: int(v) if v not in (None, "") else None
        n = max(1, min(int(data.get("n", 10)), 30))
        recs = self.rec.recommend(ratings, genres, n=n, year_min=year(data.get("year_min")),
                                  year_max=year(data.get("year_max")), languages=languages,
                                  exclude_genres=excluded, diverse=bool(data.get("diverse")))
        return [r.to_dict() for r in recs]


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        server_version = "MovieMatch/1.0"

        def log_message(self, *args):
            pass

        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, payload):
            self._send(code, json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")

        def _host_ok(self) -> bool:
            raw = self.headers.get("Host", "")
            host = raw.split("]")[0] + "]" if raw.startswith("[") else raw.rsplit(":", 1)[0]
            return host in LOCAL_HOSTS

        def do_GET(self):
            if not self._host_ok():
                return self._json(403, {"error": "forbidden"})
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                return self._send(200, INDEX_HTML.read_bytes(), "text/html; charset=utf-8")
            if path == "/api/catalog":
                return self._json(200, app.catalog_payload())
            if path == "/api/evaluate":
                return self._json(200, app.evaluation())
            self._json(404, {"error": "not found"})

        def do_POST(self):
            if not self._host_ok():
                return self._json(403, {"error": "forbidden"})
            if self.path.split("?")[0] != "/api/recommend":
                return self._json(404, {"error": "not found"})
            if not self.headers.get("Content-Type", "").startswith("application/json"):
                return self._json(415, {"error": "JSON only"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = -1
            if not 0 <= length <= MAX_BODY:
                return self._json(413, {"error": "request too large"})
            try:
                data = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(data, dict):
                    raise ValueError
                self._json(200, {"results": app.recommend(data)})
            except RecommendationError as exc:
                self._json(400, {"error": str(exc)})
            except (ValueError, TypeError, AttributeError):
                self._json(400, {"error": "invalid request"})

    return Handler


def make_server(app: App, port: int = 8000, host: str = "127.0.0.1"):
    return ThreadingHTTPServer((host, port), make_handler(app))


def main():
    parser = argparse.ArgumentParser(description="MovieMatch web app")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    server = make_server(App(), args.port)
    url = f"http://127.0.0.1:{server.server_address[1]}"
    print(f"MovieMatch is running at {url}  (press Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
