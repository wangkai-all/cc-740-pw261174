"""HTTP API and static file serving, stdlib only.

Endpoints
---------
GET  /                       single-page web UI (web/index.html)
GET  /api/presets            fixed risk configurations
GET  /api/plans              saved plans
POST /api/plans              save a plan (validated)
GET  /api/batches            batch summaries
POST /api/batches            create a batch from a plan
GET  /api/batches/{id}       full replayed state
POST /api/batches/{id}/items record a round of items
POST /api/batches/{id}/round open the next round
POST /api/batches/{id}/withdraw  withdraw a mis-filed item
POST /api/batches/{id}/release   freeze and stamp release time
POST /api/batches/{id}/correction post-release correction note
GET  /api/batches/{id}/verify    replay-consistency check
POST /api/simulate           run a reproducible experiment grid
GET  /api/handcheck          hand-calculation cross-check report
GET  /api/audit              raw JSONL audit stream
"""

from __future__ import annotations

import json
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from . import handcheck, simulate, state, stats, storage

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


def _json_default(obj):
    raise TypeError(f"not JSON serializable: {obj!r}")


def make_handler(store: "storage.Store"):
    class Handler(BaseHTTPRequestHandler):
        server_version = "SequentialInspection/1.0"

        def log_message(self, fmt, *args):  # quieter console
            pass

        # -- helpers -------------------------------------------------------

        def _send(self, code, payload):
            body = json.dumps(payload, ensure_ascii=False,
                              allow_nan=False,
                              default=_json_default).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_file(self, path: Path, content_type: str):
            if not path.exists():
                self._send(404, {"error": "not found"})
                return
            body = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_body(self) -> dict:
            length = int(self.headers.get("Content-Length", 0))
            if length == 0:
                return {}
            raw = self.rfile.read(length)
            try:
                data = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON body: {exc}") from exc
            if not isinstance(data, dict):
                raise ValueError("request body must be a JSON object")
            return data

        def _error(self, exc: Exception):
            code = 400
            if isinstance(exc, KeyError):
                code = 404
            self._send(code, {"error": str(exc),
                              "type": exc.__class__.__name__})

        def _batch_or_404(self, batch_id):
            try:
                return store.load_batch(batch_id)
            except KeyError:
                self._send(404, {"error": f"no batch {batch_id!r}"})
                return None

        # -- GET -----------------------------------------------------------

        def do_GET(self):
            try:
                url = urlparse(self.path)
                path = url.path.rstrip("/") or "/"
                if path == "/":
                    self._send_file(WEB_DIR / "index.html",
                                    "text/html; charset=utf-8")
                    return
                if path.startswith("/web/"):
                    name = Path(path[4:]).name
                    self._send_file(
                        WEB_DIR / name,
                        "text/css" if name.endswith(".css")
                        else "application/javascript")
                    return
                if path == "/api/presets":
                    self._send(200, {"presets": storage.RISK_PRESETS})
                elif path == "/api/plans":
                    self._send(200, {"plans": store.list_plans()})
                elif path == "/api/batches":
                    self._send(200, {"batches": store.list_batches()})
                elif path == "/api/audit":
                    self._send(200, {"records": store.read_audit()})
                elif path == "/api/handcheck":
                    self._send(200, handcheck.full_report())
                elif path.startswith("/api/batches/"):
                    rest = path[len("/api/batches/"):].split("/")
                    batch = self._batch_or_404(rest[0])
                    if batch is None:
                        return
                    if len(rest) == 1:
                        self._send(200, state.rebuild(batch))
                    elif rest[1] == "verify":
                        self._send(200, self._verify(batch))
                    else:
                        self._send(404, {"error": "unknown route"})
                else:
                    self._send(404, {"error": "not found"})
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                self._error(exc)

        # -- POST ----------------------------------------------------------

        def do_POST(self):
            try:
                url = urlparse(self.path)
                path = url.path.rstrip("/")
                body = self._read_body()

                if path == "/api/plans":
                    plan = stats.make_plan(**body)
                    payload = plan.to_dict()
                    store.save_plan(payload)
                    self._send(201, {"plan": payload,
                                     "assumptions":
                                         stats.assumptions_text(plan)})
                elif path == "/api/batches":
                    plan_kwargs = body.get("plan", body)
                    plan = stats.make_plan(**plan_kwargs)
                    batch = state.create_batch(
                        plan, lot_note=body.get("lot_note", ""),
                        batch_id=body.get("batch_id"))
                    store.save_batch(batch, "batch_created",
                                     {"plan": plan.to_dict()})
                    self._send(201, state.rebuild(batch))
                elif path == "/api/simulate":
                    self._handle_simulate(body)
                elif path.startswith("/api/batches/"):
                    self._handle_batch_action(path, body)
                else:
                    self._send(404, {"error": "not found"})
            except (ValueError, stats.PlanError, state.StateError) as exc:
                self._error(exc)
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                self._error(exc)

        def _handle_batch_action(self, path, body):
            parts = path[len("/api/batches/"):].split("/")
            if len(parts) != 2:
                self._send(404, {"error": "unknown route"})
                return
            batch_id, action = parts
            batch = self._batch_or_404(batch_id)
            if batch is None:
                return

            if action == "items":
                entries = body.get("entries")
                if not isinstance(entries, list) or not entries:
                    raise ValueError("entries must be a non-empty list")
                batch, intake = state.record_items(batch, entries)
                store.save_batch(batch, "item_recorded", {"intake": intake})
                self._send(200, {"state": state.rebuild(batch),
                                 "intake": intake})
            elif action == "round":
                batch = state.open_round(batch)
                store.save_batch(batch, "round_opened")
                self._send(200, state.rebuild(batch))
            elif action == "withdraw":
                batch = state.withdraw_item(
                    batch, str(body["item_id"]), str(body["reason"]))
                store.save_batch(batch, "item_withdrawn",
                                 {"item_id": body["item_id"]})
                self._send(200, state.rebuild(batch))
            elif action == "release":
                batch = state.release_batch(batch, at=body.get("at"))
                store.save_batch(batch, "batch_released")
                self._send(200, state.rebuild(batch))
            elif action == "correction":
                batch = state.flag_correction(batch, str(body["note"]))
                store.save_batch(batch, "correction_flagged",
                                 {"note": body["note"]})
                self._send(200, state.rebuild(batch))
            else:
                self._send(404, {"error": "unknown action"})

        def _handle_simulate(self, body):
            plans = [stats.make_plan(**p) for p in body["plans"]]
            true_ps = [float(x) for x in body["true_ps"]]
            reps = int(body.get("replications", 200))
            round_size = int(body.get("round_size", 1))
            base_seed = int(body.get("base_seed", 20261124))
            cap = body.get("cap", 1500)
            cap = int(cap) if cap is not None else None
            grids = simulate.compare_plans(
                plans, true_ps, reps, base_seed, round_size, cap)
            self._send(200, {"grids": grids, "base_seed": base_seed,
                             "replications": reps, "cap": cap})

        @staticmethod
        def _verify(batch):
            before = json.dumps(batch, sort_keys=True, ensure_ascii=False)
            view = state.rebuild(batch)
            after = json.dumps(batch, sort_keys=True, ensure_ascii=False)
            return {
                "consistent": before == after,
                "event_count": view["event_count"],
                "status": view["status"],
                "n": view["current"]["n"],
                "k": view["current"]["k"],
                "round": view["round"],
                "released": view["released"],
                "released_at": view["released_at"],
            }

    return Handler


def serve(data_dir: str, host: str = "127.0.0.1", port: int = 8000):
    store = storage.Store(data_dir)
    for preset in storage.RISK_PRESETS:
        if store.get_plan(preset["name"]) is None:
            store.save_plan(preset)
    handler = make_handler(store)
    httpd = ThreadingHTTPServer((host, port), handler)
    print(f"序贯抽检应用运行于 http://{host}:{port}  数据目录 {data_dir}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sequential inspection app")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    serve(args.data_dir, args.host, args.port)
