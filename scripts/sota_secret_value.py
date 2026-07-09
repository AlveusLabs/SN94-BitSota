#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Any


def _extract_field(raw: str, *, fields: list[str], source: str) -> str:
    text = str(raw or "").strip()
    if not text:
        raise RuntimeError(f"{source} is empty")
    try:
        payload: Any = json.loads(text)
    except json.JSONDecodeError:
        return text
    if not isinstance(payload, dict):
        return text
    for field in fields:
        value = str(payload.get(field) or "").strip()
        if value:
            return value
    raise RuntimeError(f"{source} JSON is missing required field: {' or '.join(fields)}")


def load_value(args: argparse.Namespace) -> str:
    fields = [str(field) for field in args.field if str(field)]
    if not fields:
        raise RuntimeError("at least one --field is required")

    env_value = os.environ.get(args.env_name, "").strip()
    if env_value:
        return _extract_field(env_value, fields=fields, source=args.env_name)

    command = [
        "aws",
        "secretsmanager",
        "get-secret-value",
        "--profile",
        args.aws_profile,
        "--region",
        args.aws_region,
        "--secret-id",
        args.secret_id,
        "--query",
        "SecretString",
        "--output",
        "text",
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=args.timeout)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or f"aws exited {result.returncode}").strip()
        raise RuntimeError(
            f"could not read AWS secret {args.secret_id!r} with profile {args.aws_profile!r}: {detail}"
        )
    return _extract_field(result.stdout, fields=fields, source=f"AWS secret {args.secret_id}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resolve a secret value from an env var or AWS Secrets Manager.")
    parser.add_argument("--env", dest="env_name", required=True, help="Environment variable override to read first.")
    parser.add_argument("--secret-id", required=True)
    parser.add_argument("--field", action="append", required=True, help="JSON field to read. Repeat for fallback fields.")
    parser.add_argument("--aws-profile", default=os.environ.get("AWS_PROFILE", "moonrocklab-frankfurt"))
    parser.add_argument("--aws-region", default=os.environ.get("AWS_REGION", "eu-central-1"))
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args(argv)
    try:
        print(load_value(args))
        return 0
    except Exception as exc:
        print(f"sota_secret_value: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
