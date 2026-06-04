from __future__ import annotations

import argparse
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger("relay")


def _json_dumps(payload: object) -> str:
    return json.dumps(payload, separators=(",", ":"), sort_keys=False)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return float(default)


@dataclass
class InMemoryRelayStore:
    lock: threading.Lock = field(default_factory=threading.Lock)
    results: list[Dict[str, Any]] = field(default_factory=list)
    events: list[Dict[str, Any]] = field(default_factory=list)
    blacklist_votes: Dict[str, int] = field(default_factory=dict)
    coldkeys_by_hotkey: Dict[str, str] = field(default_factory=dict)
    sota_threshold: float = 0.0
    next_event_id: int = 1

    def add_result(self, record: Dict[str, Any]) -> str:
        with self.lock:
            self.results.append(record)
        return str(record.get("id") or "")

    def list_results(self, *, limit: int, task_id: Optional[str]) -> list[Dict[str, Any]]:
        with self.lock:
            results = list(self.results)
        if task_id:
            results = [r for r in results if str(r.get("task_id") or "") == str(task_id)]
        results.sort(key=lambda r: float(r.get("created_at", 0.0) or 0.0), reverse=True)
        return results[: max(0, int(limit))]

    def verify_result(self, result_id: str) -> bool:
        with self.lock:
            for r in self.results:
                if str(r.get("id") or "") == str(result_id):
                    r["verified"] = True
                    return True
        return False

    def record_blacklist_vote(self, miner_hotkey: str) -> int:
        key = str(miner_hotkey or "")
        if not key:
            return 0
        with self.lock:
            self.blacklist_votes[key] = int(self.blacklist_votes.get(key, 0) or 0) + 1
            return int(self.blacklist_votes[key])

    def update_coldkey(self, *, hotkey: str, coldkey_address: str) -> None:
        hk = str(hotkey or "")
        ck = str(coldkey_address or "")
        if not hk or not ck:
            return
        with self.lock:
            self.coldkeys_by_hotkey[hk] = ck

    def add_sota_event(
        self,
        *,
        miner_hotkey: str,
        score: float,
        seen_block: int,
        result_id: Optional[str],
    ) -> Dict[str, Any]:
        with self.lock:
            event_id = int(self.next_event_id)
            self.next_event_id += 1
            self.sota_threshold = max(float(self.sota_threshold), float(score))

            event = {
                "id": int(event_id),
                "miner_hotkey": str(miner_hotkey),
                "score": float(score),
                "seen_block": int(seen_block),
                "start_block": int(seen_block),
                "end_block": int(seen_block),
                "result_id": str(result_id) if result_id is not None else None,
                "finalized_at": float(time.time()),
            }
            # Prepend newest first.
            self.events.insert(0, event)
            return dict(event)

    def list_sota_events(self, *, limit: int) -> list[Dict[str, Any]]:
        with self.lock:
            return list(self.events)[: max(0, int(limit))]


def _make_handler(store: InMemoryRelayStore, *, dev_mode: bool):
    class RelayHandler(BaseHTTPRequestHandler):
        server_version = "BitSotaLocalRelay/0.1"

        def log_message(self, fmt: str, *args):  # noqa: N802 - stdlib signature
            if dev_mode:
                logger.info("%s - %s", self.address_string(), fmt % args)

        def _send_json(self, status: int, payload: object) -> None:
            body = _json_dumps(payload).encode("utf-8")
            self.send_response(int(status))
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json_body(self) -> Optional[Dict[str, Any]]:
            try:
                length = int(self.headers.get("Content-Length", "0") or "0")
            except Exception:
                length = 0
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            try:
                decoded = raw.decode("utf-8")
            except Exception:
                decoded = ""
            if not decoded:
                return {}
            try:
                obj = json.loads(decoded)
            except Exception:
                return None
            return obj if isinstance(obj, dict) else None

        def do_GET(self):  # noqa: N802 - stdlib signature
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/")
            qs = parse_qs(parsed.query)

            if path in {"", "/health"}:
                return self._send_json(200, {"status": "ok"})

            if path == "/sota_threshold":
                return self._send_json(200, {"sota_threshold": float(store.sota_threshold)})

            if path == "/sota/events":
                try:
                    limit = int((qs.get("limit") or ["20"])[0])
                except Exception:
                    limit = 20
                return self._send_json(200, store.list_sota_events(limit=limit))

            if path == "/results":
                try:
                    limit = int((qs.get("limit") or ["256"])[0])
                except Exception:
                    limit = 256
                task_id = (qs.get("task_id") or [None])[0]
                return self._send_json(200, store.list_results(limit=limit, task_id=task_id))

            if path == "/invitation_code/linked":
                # GUI calls this unless test_mode is enabled.
                return self._send_json(200, {"status": "success", "data": {"linked": True}})

            if path in {"/version", "/version.json"}:
                return self._send_json(
                    200,
                    {
                        "version": "dev",
                        "versionCode": 0,
                        "desc": "Local relay (dev only)",
                        "mac": None,
                        "linux": None,
                        "windows": None,
                    },
                )

            return self._send_json(404, {"error": "not_found", "path": parsed.path})

        def do_POST(self):  # noqa: N802 - stdlib signature
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/")

            if path == "/submit_solution":
                body = self._read_json_body()
                if body is None:
                    return self._send_json(400, {"error": "invalid_json"})

                miner_hotkey = str(self.headers.get("X-Key") or "")
                timestamp_message = str(self.headers.get("X-Timestamp") or "")
                signature = str(self.headers.get("X-Signature") or "")

                algorithm_result = body.get("algorithm_result")
                if isinstance(algorithm_result, str):
                    algorithm_result_str = algorithm_result
                else:
                    algorithm_result_str = _json_dumps(algorithm_result or {})

                record = {
                    "id": str(uuid.uuid4()),
                    "task_id": str(body.get("task_id") or ""),
                    "miner_hotkey": miner_hotkey,
                    "score": _safe_float(body.get("score"), 0.0),
                    "timestamp_message": timestamp_message,
                    "signature": signature,
                    # Validator expects this to be a JSON string.
                    "algorithm_result": algorithm_result_str,
                    "created_at": float(time.time()),
                    "verified": False,
                }
                store.add_result(record)
                return self._send_json(200, {"status": "success", "id": record["id"]})

            if path.startswith("/verify/"):
                result_id = path.split("/", 2)[2] if "/" in path[1:] else ""
                if not result_id:
                    return self._send_json(400, {"error": "missing_result_id"})
                if store.verify_result(result_id):
                    return self._send_json(200, {"status": "verified"})
                return self._send_json(404, {"error": "not_found", "id": result_id})

            if path.startswith("/blacklist/"):
                miner_hotkey = path.split("/", 2)[2] if "/" in path[1:] else ""
                if not miner_hotkey:
                    return self._send_json(400, {"error": "missing_miner_hotkey"})
                votes = store.record_blacklist_vote(miner_hotkey)
                return self._send_json(
                    200,
                    {
                        "status": "blacklist vote recorded",
                        "miner_hotkey": str(miner_hotkey),
                        "votes": int(votes),
                    },
                )

            if path == "/coldkey_address/update":
                body = self._read_json_body()
                if body is None:
                    return self._send_json(400, {"error": "invalid_json"})
                hotkey = str(self.headers.get("X-Key") or "")
                coldkey_address = str(body.get("coldkey_address") or "")
                store.update_coldkey(hotkey=hotkey, coldkey_address=coldkey_address)
                return self._send_json(200, {"status": "success"})

            if path == "/sota/vote":
                body = self._read_json_body()
                if body is None:
                    return self._send_json(400, {"error": "invalid_json"})

                miner_hotkey = str(body.get("miner_hotkey") or "")
                score = _safe_float(body.get("score"), 0.0)
                try:
                    seen_block = int(body.get("seen_block") or 0)
                except Exception:
                    seen_block = 0
                result_id = body.get("result_id")
                result_id_s = str(result_id) if result_id is not None else None

                if not miner_hotkey:
                    return self._send_json(400, {"error": "missing_miner_hotkey"})

                finalized_event = store.add_sota_event(
                    miner_hotkey=miner_hotkey,
                    score=float(score),
                    seen_block=int(seen_block),
                    result_id=result_id_s,
                )

                return self._send_json(
                    200,
                    {
                        "status": "finalized",
                        "votes_for_candidate": 1,
                        "votes_needed": 1,
                        "finalized_event": finalized_event,
                    },
                )

            return self._send_json(404, {"error": "not_found", "path": parsed.path})

    return RelayHandler


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="BitSota local relay (dev/testing only).")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8002, help="Bind port (default: 8002)")
    parser.add_argument(
        "--sota-threshold",
        type=float,
        default=0.0,
        help="Initial SOTA threshold (default: 0.0)",
    )
    parser.add_argument("--dev-log", action="store_true", help="Log each request")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")

    store = InMemoryRelayStore(sota_threshold=float(args.sota_threshold))
    handler_cls = _make_handler(store, dev_mode=bool(args.dev_log))

    httpd = ThreadingHTTPServer((str(args.host), int(args.port)), handler_cls)
    logger.info("Local relay listening on http://%s:%d", str(args.host), int(args.port))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

