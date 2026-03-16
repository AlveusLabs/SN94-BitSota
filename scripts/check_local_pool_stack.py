#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from typing import Any


def _fetch_json(url: str, timeout_s: float) -> Any:
    with urllib.request.urlopen(url, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check that a local Pool API + sidecar stack is reachable and making progress."
    )
    parser.add_argument("--pool-url", default="http://127.0.0.1:8434")
    parser.add_argument("--sidecar-url", default="http://127.0.0.1:8123")
    parser.add_argument("--timeout-s", type=float, default=5.0)
    parser.add_argument("--require-progress", action="store_true")
    parser.add_argument("--min-successful-submissions", type=int, default=1)
    parser.add_argument("--min-completed-jobs", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pool_health_url = f"{str(args.pool_url).rstrip('/')}/health"
    sidecar_state_url = f"{str(args.sidecar_url).rstrip('/')}/state"

    try:
        pool = _fetch_json(pool_health_url, args.timeout_s)
        sidecar = _fetch_json(sidecar_state_url, args.timeout_s)
        _require(pool.get("status") == "healthy", f"Pool health is not healthy: {pool}")
        _require(sidecar.get("status") == "running", f"Sidecar is not running: {sidecar}")
        if args.require_progress:
            _require(
                int(sidecar.get("successful_submissions", 0)) >= int(args.min_successful_submissions),
                f"Sidecar successful_submissions is below {args.min_successful_submissions}: {sidecar}",
            )
            _require(
                int(sidecar.get("completed_jobs", 0)) >= int(args.min_completed_jobs),
                f"Sidecar completed_jobs is below {args.min_completed_jobs}: {sidecar}",
            )
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"FAIL: request error: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print(f"PASS pool health: {pool_health_url}")
    print(json.dumps(pool, indent=2, sort_keys=True))
    print(f"PASS sidecar state: {sidecar_state_url}")
    print(json.dumps(sidecar, indent=2, sort_keys=True))
    if args.require_progress:
        print("PASS progress thresholds met")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
