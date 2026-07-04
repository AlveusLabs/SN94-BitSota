#!/usr/bin/env python3
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from typing import Any


REPOS = Path("/home/mekaneeky/repos")
DEFAULT_ARTIFACTS_DIR = REPOS / ".sota-base-testnet"
DEFAULT_SERVICE_PACK = DEFAULT_ARTIFACTS_DIR / "base-sota-testnet-service-pack.json"
DEFAULT_INPUT_DIR = DEFAULT_ARTIFACTS_DIR / "apprunner"
DEFAULT_OUT_DIR = DEFAULT_ARTIFACTS_DIR / "apprunner-source"
DEFAULT_REPORT_OUT = DEFAULT_ARTIFACTS_DIR / "base-sota-testnet-apprunner-source-pack.json"
DEFAULT_AWS_INVENTORY = DEFAULT_ARTIFACTS_DIR / "base-sota-testnet-aws-inventory.json"
DEFAULT_AWS_PROFILE = "moonrocklab-frankfurt"
DEFAULT_REGION = "eu-central-1"
DEFAULT_CONNECTION_NAME = "bitsota"
DEFAULT_INSTANCE_ROLE_ARN = "arn:aws:iam::924380800822:role/AppRunnerReadSecrets"
CONNECTION_ARN_ENV = "SOTA_APPRUNNER_CONNECTION_ARN"
INSTANCE_ROLE_ARN_ENV = "SOTA_APPRUNNER_INSTANCE_ROLE_ARN"
DEPLOYMENT_RELEVANT_PREFIXES = {
    "claims_ui": (
        "app/",
        "components/Claims/",
        "components/Layout/",
        "lib/constants.ts",
        "next.config.ts",
        "services/baseClaims/",
        "types/global.d.ts",
    ),
    "indexer_api": (
        "experiments/base_protocol_design/fixtures/",
        "experiments/base_protocol_design/sota_base_indexer/",
        "experiments/base_protocol_design/sota_base_snapshotter/",
        "tests/test_sota_base_",
        "Dockerfile.sota-base-indexer",
        ".dockerignore",
    ),
    "autoresearch_coordinator": (
        ".env.testnet.example",
        "ARCH.md",
        "README.md",
        "docs/sota-",
        "pyproject.toml",
        "scripts/sota_root_lifecycle_drill.py",
        "src/autoresearch_bittensor/",
        "tests/test_evm_submission_identity.py",
        "tests/test_sota_",
    ),
}
DEPLOYMENT_RELEVANT_IGNORES = (
    "__pycache__/",
    ".pytest_cache/",
)
DEPLOYMENT_RELEVANT_IGNORED_SUFFIXES = (
    ".pyc",
    ".pyo",
)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run(command: list[str], *, cwd: Path | None = None, timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, check=False, capture_output=True, text=True, timeout=timeout)


def _run_aws(args: list[str], *, profile: str, region: str, timeout: float) -> dict[str, Any]:
    command = ["aws", *args, "--output", "json"]
    if profile:
        command.extend(["--profile", profile])
    if region:
        command.extend(["--region", region])
    result = _run(command, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f"aws exited {result.returncode}")
    payload = json.loads(result.stdout or "{}")
    if not isinstance(payload, dict):
        raise RuntimeError("aws returned non-object JSON")
    return payload


def _cached_connection_arn(args: argparse.Namespace) -> tuple[str, str]:
    path = Path(getattr(args, "aws_inventory", DEFAULT_AWS_INVENTORY))
    if not path.exists():
        return _cached_rendered_connection_arn(args)
    try:
        payload = _load_json(path)
    except Exception as exc:
        rendered_arn, rendered_source = _cached_rendered_connection_arn(args)
        if rendered_arn:
            return rendered_arn, rendered_source
        return "", f"cached inventory {path} could not be read: {exc}"
    inventory = dict(payload.get("inventory") or {})
    matches = [
        dict(item)
        for item in inventory.get("app_runner_connections") or []
        if isinstance(item, dict) and item.get("name") == args.connection_name
    ]
    available = [item for item in matches if str(item.get("status") or "").upper() == "AVAILABLE"]
    selected = available[0] if available else (matches[0] if matches else {})
    arn = str(selected.get("arn") or "")
    if not arn:
        rendered_arn, rendered_source = _cached_rendered_connection_arn(args)
        if rendered_arn:
            return rendered_arn, rendered_source
        return "", f"cached inventory {path} does not contain connection {args.connection_name!r}"
    status = str(selected.get("status") or "UNKNOWN")
    if status.upper() != "AVAILABLE":
        rendered_arn, rendered_source = _cached_rendered_connection_arn(args)
        if rendered_arn:
            return rendered_arn, rendered_source
        return "", f"cached connection {args.connection_name!r} in {path} is {status}, not AVAILABLE"
    return arn, str(path)


def _cached_rendered_connection_arn(args: argparse.Namespace) -> tuple[str, str]:
    out_dir = Path(getattr(args, "out_dir", DEFAULT_OUT_DIR))
    if not out_dir.exists():
        return "", ""
    for path in sorted(out_dir.glob("*.json")):
        try:
            payload = _load_json(path)
        except Exception:
            continue
        source = dict(payload.get("SourceConfiguration") or {})
        auth = dict(source.get("AuthenticationConfiguration") or {})
        arn = str(auth.get("ConnectionArn") or "").strip()
        if arn and not arn.startswith("${"):
            return arn, f"rendered source input {path}"
    return "", ""


def _resolve_connection_arn(args: argparse.Namespace, checks: list[dict[str, str]]) -> str:
    if args.connection_arn:
        checks.append({"name": "connection_arn", "status": "green", "detail": "Using explicitly supplied App Runner GitHub connection ARN."})
        return str(args.connection_arn)
    env_arn = os.environ.get(CONNECTION_ARN_ENV, "").strip()
    if env_arn:
        checks.append({"name": "connection_arn", "status": "green", "detail": f"Using App Runner GitHub connection ARN from ${CONNECTION_ARN_ENV}."})
        return env_arn
    if args.no_resolve_connection_arn:
        checks.append(
            {
                "name": "connection_arn",
                "status": "yellow",
                "detail": "Connection ARN resolution was skipped.",
                "remediation": f"Set ${CONNECTION_ARN_ENV} or rerun without --no-resolve-connection-arn.",
            }
        )
        return ""
    try:
        payload = _run_aws(["apprunner", "list-connections"], profile=args.aws_profile, region=args.region, timeout=args.timeout)
    except Exception as exc:
        cached_arn, cached_source = _cached_connection_arn(args)
        if cached_arn:
            checks.append(
                {
                    "name": "connection_arn",
                    "status": "green",
                    "detail": (
                        f"Using cached App Runner GitHub connection ARN for "
                        f"{args.connection_name!r} from {cached_source}; live AWS lookup failed: {exc}"
                    ),
                }
            )
            return cached_arn
        fallback_detail = f" Cached fallback was unavailable: {cached_source}" if cached_source else ""
        checks.append(
            {
                "name": "connection_arn",
                "status": "yellow",
                "detail": f"Could not resolve App Runner connection {args.connection_name!r}: {exc}.{fallback_detail}",
                "remediation": f"Set ${CONNECTION_ARN_ENV} to the approved App Runner GitHub connection ARN.",
            }
        )
        return ""
    matches = [
        dict(item)
        for item in payload.get("ConnectionSummaryList") or []
        if isinstance(item, dict) and item.get("ConnectionName") == args.connection_name
    ]
    available = [item for item in matches if str(item.get("Status") or "").upper() == "AVAILABLE"]
    selected = available[0] if available else (matches[0] if matches else {})
    arn = str(selected.get("ConnectionArn") or "")
    if not arn:
        checks.append(
            {
                "name": "connection_arn",
                "status": "yellow",
                "detail": f"No App Runner GitHub connection named {args.connection_name!r} was found.",
                "remediation": "Create or select an approved App Runner GitHub connection before source-based service creation.",
            }
        )
        return ""
    status = str(selected.get("Status") or "UNKNOWN")
    checks.append(
        {
            "name": "connection_arn",
            "status": "green" if status.upper() == "AVAILABLE" else "yellow",
            "detail": f"Resolved App Runner GitHub connection {args.connection_name!r} with status {status}.",
            **({} if status.upper() == "AVAILABLE" else {"remediation": "Use an AVAILABLE App Runner GitHub connection."}),
        }
    )
    return arn


def _resolve_instance_role_arn(args: argparse.Namespace, checks: list[dict[str, str]]) -> str:
    if args.instance_role_arn:
        checks.append({"name": "instance_role_arn", "status": "green", "detail": "Using explicitly supplied App Runner runtime instance role ARN."})
        return str(args.instance_role_arn)
    env_arn = os.environ.get(INSTANCE_ROLE_ARN_ENV, "").strip()
    if env_arn:
        checks.append({"name": "instance_role_arn", "status": "green", "detail": f"Using App Runner runtime instance role ARN from ${INSTANCE_ROLE_ARN_ENV}."})
        return env_arn
    checks.append(
        {
            "name": "instance_role_arn",
            "status": "yellow",
            "detail": "No App Runner runtime instance role ARN was supplied.",
            "remediation": f"Set ${INSTANCE_ROLE_ARN_ENV} or pass --instance-role-arn so services can read Secrets Manager handles.",
        }
    )
    return ""


def _service_by_name(service_pack: dict[str, Any]) -> dict[str, dict[str, Any]]:
    services: dict[str, dict[str, Any]] = {}
    for item in service_pack.get("services") or []:
        if not isinstance(item, dict):
            continue
        recipe = dict(item.get("deployment_recipe") or {})
        service_name = str(recipe.get("service_name") or "")
        if service_name:
            services[service_name] = dict(item)
    return services


def _git_status(path: Path, *, timeout: float) -> str:
    result = _run(["git", "status", "--porcelain", "-uall"], cwd=path, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f"git status exited {result.returncode}")
    return result.stdout.rstrip("\n")


def _dirty_paths_from_status(status: str) -> list[str]:
    paths: list[str] = []
    for line in status.splitlines():
        if not line.strip():
            continue
        raw_path = line[3:] if len(line) > 3 else line.strip()
        path = raw_path.split(" -> ", 1)[-1].strip()
        if path:
            paths.append(path)
    return paths


def _service_key(service_name: str, service: dict[str, Any]) -> str:
    explicit = str(service.get("key") or "").strip()
    if explicit:
        return explicit
    if "claims-ui" in service_name:
        return "claims_ui"
    if "indexer" in service_name:
        return "indexer_api"
    if "autoresearch" in service_name or "coordinator" in service_name:
        return "autoresearch_coordinator"
    return service_name


def _deployment_relevant_paths(service_name: str, service: dict[str, Any], dirty_paths: list[str]) -> list[str]:
    prefixes = DEPLOYMENT_RELEVANT_PREFIXES.get(_service_key(service_name, service), ())
    paths: list[str] = []
    for path in dirty_paths:
        if any(part in path for part in DEPLOYMENT_RELEVANT_IGNORES):
            continue
        if path.endswith(DEPLOYMENT_RELEVANT_IGNORED_SUFFIXES):
            continue
        if any(path == prefix or path.startswith(prefix) for prefix in prefixes):
            paths.append(path)
    return paths


def _publication_record(service_name: str, service: dict[str, Any], *, timeout: float) -> dict[str, Any]:
    source = dict(service.get("source") or {})
    local_path = Path(str(source.get("path") or ""))
    record: dict[str, Any] = {
        "service_name": service_name,
        "service_key": _service_key(service_name, service),
        "local_path": str(local_path) if str(local_path) else "",
        "branch": str(source.get("branch") or ""),
        "remote_url": str(source.get("remote_url") or ""),
        "commit_sha": str(source.get("commit_sha") or ""),
        "dirty_paths": [],
        "deployment_relevant_dirty_paths": [],
        "dirty_count": 0,
        "deployment_relevant_dirty_count": 0,
        "status": "green",
    }
    if not local_path.exists():
        record["status"] = "yellow"
        record["message"] = f"Local source path does not exist: {local_path}"
        return record
    try:
        status = _git_status(local_path, timeout=timeout)
    except Exception as exc:
        record["status"] = "yellow"
        record["message"] = f"Could not inspect git status: {exc}"
        return record
    dirty_paths = _dirty_paths_from_status(status)
    relevant_paths = _deployment_relevant_paths(service_name, service, dirty_paths)
    record["dirty_paths"] = dirty_paths
    record["deployment_relevant_dirty_paths"] = relevant_paths
    record["dirty_count"] = len(dirty_paths)
    record["deployment_relevant_dirty_count"] = len(relevant_paths)
    if relevant_paths:
        record["status"] = "yellow"
        record["message"] = "Deployment-relevant local changes must be committed and pushed before App Runner service creation."
    elif dirty_paths:
        record["status"] = "yellow"
        record["message"] = "Local worktree has dirty paths, but none matched the source-pack deployment relevance prefixes."
    return record


def _git_remote_head(remote_url: str, branch: str, *, timeout: float) -> str:
    result = _run(["git", "ls-remote", remote_url, f"refs/heads/{branch}"], timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f"git ls-remote exited {result.returncode}")
    first = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    return first.split()[0] if first else ""


def _source_checks(service_name: str, service: dict[str, Any], app_input: dict[str, Any], *, timeout: float, skip_remote: bool) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []
    repo = dict(dict(app_input.get("SourceConfiguration") or {}).get("CodeRepository") or {})
    source = dict(service.get("source") or {})
    repo_url = str(repo.get("RepositoryUrl") or "").strip()
    branch = str(dict(repo.get("SourceCodeVersion") or {}).get("Value") or source.get("branch") or "").strip()
    local_path = Path(str(source.get("path") or ""))
    local_commit = str(source.get("commit_sha") or "").strip()

    if not repo_url or repo_url == "<GITHUB_REPOSITORY_URL>":
        checks.append(
            {
                "name": f"source_repo_{service_name}",
                "status": "red",
                "detail": f"{service_name} does not have a concrete GitHub RepositoryUrl.",
                "remediation": "Set the service source repository before creating the App Runner service.",
            }
        )
    else:
        checks.append({"name": f"source_repo_{service_name}", "status": "green", "detail": f"{service_name} uses {repo_url}."})

    if not branch:
        checks.append(
            {
                "name": f"source_branch_{service_name}",
                "status": "red",
                "detail": f"{service_name} does not have a source branch.",
                "remediation": "Set SourceCodeVersion.Value to the branch that contains the Base SOTA code.",
            }
        )
    else:
        checks.append({"name": f"source_branch_{service_name}", "status": "green", "detail": f"{service_name} uses branch {branch}."})

    if local_path.exists():
        try:
            dirty = _git_status(local_path, timeout=timeout)
        except Exception as exc:
            checks.append(
                {
                    "name": f"source_dirty_{service_name}",
                    "status": "yellow",
                    "detail": f"Could not inspect local git status for {service_name}: {exc}",
                    "remediation": "Verify the deployed branch contains the Base SOTA code before creating the service.",
                }
            )
        else:
            dirty_paths = _dirty_paths_from_status(dirty)
            relevant_paths = _deployment_relevant_paths(service_name, service, dirty_paths)
            if relevant_paths:
                checks.append(
                    {
                        "name": f"source_dirty_{service_name}",
                        "status": "yellow",
                        "detail": f"{service_name} has {len(relevant_paths)} deployment-relevant local changes; App Runner deploys the remote branch, not this worktree.",
                        "remediation": "Commit and push the Base SOTA service changes to the configured source branch before creating the service.",
                    }
                )
            elif dirty_paths:
                checks.append(
                    {
                        "name": f"source_dirty_{service_name}",
                        "status": "green",
                        "detail": f"{service_name} has local dirty paths, but none match the deployment-relevant service path set.",
                    }
                )
            else:
                checks.append({"name": f"source_dirty_{service_name}", "status": "green", "detail": f"{service_name} local worktree is clean."})
    else:
        checks.append(
            {
                "name": f"source_dirty_{service_name}",
                "status": "yellow",
                "detail": f"Local source path for {service_name} does not exist: {local_path}.",
                "remediation": "Verify the configured GitHub branch contains the Base SOTA code before creating the service.",
            }
        )

    if skip_remote or not repo_url or not branch:
        return checks
    try:
        remote_head = _git_remote_head(repo_url, branch, timeout=timeout)
    except Exception as exc:
        checks.append(
            {
                "name": f"source_remote_head_{service_name}",
                "status": "yellow",
                "detail": f"Could not inspect remote branch for {service_name}: {exc}",
                "remediation": "Verify the remote branch exists and contains the Base SOTA service code before creating the service.",
            }
        )
    else:
        if not remote_head:
            checks.append(
                {
                    "name": f"source_remote_head_{service_name}",
                    "status": "red",
                    "detail": f"Remote branch {branch} was not found for {repo_url}.",
                    "remediation": "Push the configured Base SOTA service branch before creating the service.",
                }
            )
        elif local_commit and remote_head != local_commit:
            checks.append(
                {
                    "name": f"source_remote_head_{service_name}",
                    "status": "yellow",
                    "detail": f"{service_name} remote branch head {remote_head[:12]} differs from local recorded commit {local_commit[:12]}.",
                    "remediation": "Push or retarget the service branch so App Runner builds the intended Base SOTA code.",
                }
            )
        else:
            checks.append({"name": f"source_remote_head_{service_name}", "status": "green", "detail": f"{service_name} remote branch contains the recorded commit."})
    return checks


def _status_rank(status: str) -> int:
    return {"green": 0, "yellow": 1, "red": 2}.get(status, 2)


def _worst(checks: list[dict[str, str]]) -> str:
    return max((check["status"] for check in checks), key=_status_rank, default="green")


def _next_actions(checks: list[dict[str, str]]) -> list[str]:
    actions: list[str] = []
    for check in checks:
        remediation = check.get("remediation")
        if check.get("status") != "green" and remediation and remediation not in actions:
            actions.append(remediation)
    return actions


def build_pack(args: argparse.Namespace) -> dict[str, Any]:
    service_pack = _load_json(args.service_pack)
    services = _service_by_name(service_pack)
    inputs = sorted(args.apprunner_input_dir.glob("*.json"))
    checks: list[dict[str, str]] = []
    if not inputs:
        checks.append(
            {
                "name": "apprunner_inputs",
                "status": "red",
                "detail": f"No App Runner source input JSON files found in {args.apprunner_input_dir}.",
                "remediation": "Regenerate the Base SOTA service pack before creating App Runner services.",
            }
        )
    connection_arn = _resolve_connection_arn(args, checks)
    instance_role_arn = _resolve_instance_role_arn(args, checks)
    rendered: list[dict[str, Any]] = []
    source_publication: list[dict[str, Any]] = []
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for input_path in inputs:
        payload = _load_json(input_path)
        service_name = str(payload.get("ServiceName") or input_path.stem)
        source_config = dict(payload.get("SourceConfiguration") or {})
        if "CodeRepository" not in source_config:
            checks.append(
                {
                    "name": f"source_mode_{service_name}",
                    "status": "red",
                    "detail": f"{input_path} is not a CodeRepository App Runner input.",
                    "remediation": "Use the source-based service pack App Runner files, not ECR image inputs, for this path.",
                }
            )
            continue
        output_payload = deepcopy(payload)
        auth = output_payload.setdefault("SourceConfiguration", {}).setdefault("AuthenticationConfiguration", {})
        if connection_arn:
            auth["ConnectionArn"] = connection_arn
            checks.append({"name": f"connection_applied_{service_name}", "status": "green", "detail": f"Resolved GitHub connection ARN written for {service_name}."})
        else:
            checks.append(
                {
                    "name": f"connection_applied_{service_name}",
                    "status": "yellow",
                    "detail": f"{service_name} still has an unresolved GitHub connection ARN placeholder.",
                    "remediation": f"Set ${CONNECTION_ARN_ENV} or allow AWS connection resolution before creating the service.",
                }
            )
        instance_config = output_payload.setdefault("InstanceConfiguration", {})
        current_instance_role = str(instance_config.get("InstanceRoleArn") or "")
        if current_instance_role.startswith("${"):
            if instance_role_arn:
                instance_config["InstanceRoleArn"] = instance_role_arn
                checks.append({"name": f"instance_role_applied_{service_name}", "status": "green", "detail": f"Resolved runtime instance role ARN written for {service_name}."})
            else:
                checks.append(
                    {
                        "name": f"instance_role_applied_{service_name}",
                        "status": "yellow",
                        "detail": f"{service_name} still has an unresolved runtime instance role ARN placeholder.",
                        "remediation": f"Set ${INSTANCE_ROLE_ARN_ENV} or pass --instance-role-arn before creating this service.",
                    }
                )
        elif current_instance_role:
            checks.append({"name": f"instance_role_applied_{service_name}", "status": "green", "detail": f"{service_name} already has a runtime instance role ARN."})
        service = services.get(service_name, {})
        source_publication.append(_publication_record(service_name, service, timeout=args.timeout))
        checks.extend(_source_checks(service_name, service, output_payload, timeout=args.timeout, skip_remote=args.skip_remote_check))
        out_path = args.out_dir / input_path.name
        _write_json(out_path, output_payload)
        rendered.append(
            {
                "service_name": service_name,
                "input": str(input_path),
                "rendered_input": str(out_path),
                "create_service_command": [
                    "aws",
                    "apprunner",
                    "create-service",
                    "--cli-input-json",
                    f"file://{out_path}",
                    "--profile",
                    args.aws_profile,
                    "--region",
                    args.region,
                ],
            }
        )
    status = _worst(checks)
    return {
        "schema": "sota-base-testnet-apprunner-source-pack/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ok": status == "green",
        "status": status,
        "service_pack": str(args.service_pack),
        "input_dir": str(args.apprunner_input_dir),
        "out_dir": str(args.out_dir),
        "aws": {
            "profile": args.aws_profile,
            "region": args.region,
            "connection_name": args.connection_name,
            "connection_resolved": bool(connection_arn),
            "instance_role_resolved": bool(instance_role_arn),
        },
        "rendered_services": rendered,
        "source_publication": source_publication,
        "checks": checks,
        "summary": {
            "green": sum(1 for check in checks if check["status"] == "green"),
            "yellow": sum(1 for check in checks if check["status"] == "yellow"),
            "red": sum(1 for check in checks if check["status"] == "red"),
        },
        "next_actions": _next_actions(checks),
        "does_not": ["create_services", "deploy", "touch_production_bittensor", "touch_base_mainnet", "print_secret_values"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render AWS-ready source-based App Runner inputs for Base SOTA Base Sepolia services.")
    parser.add_argument("--service-pack", type=Path, default=DEFAULT_SERVICE_PACK)
    parser.add_argument("--apprunner-input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_OUT)
    parser.add_argument("--aws-profile", default=DEFAULT_AWS_PROFILE)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--connection-name", default=DEFAULT_CONNECTION_NAME)
    parser.add_argument("--connection-arn", default="")
    parser.add_argument("--aws-inventory", type=Path, default=DEFAULT_AWS_INVENTORY)
    parser.add_argument("--instance-role-arn", default=DEFAULT_INSTANCE_ROLE_ARN)
    parser.add_argument("--no-resolve-connection-arn", action="store_true")
    parser.add_argument("--skip-remote-check", action="store_true")
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--allow-blocked", action="store_true")
    args = parser.parse_args(argv)

    report = build_pack(args)
    _write_json(args.report_out, report)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Base SOTA App Runner source pack: {report['status'].upper()}")
        print(f"Summary: {report['summary']['green']} green, {report['summary']['yellow']} yellow, {report['summary']['red']} red")
        print(f"Report: {args.report_out}")
        for service in report["rendered_services"]:
            print(f"- {service['service_name']}: {service['rendered_input']}")
        for action in report["next_actions"]:
            print(f"- next: {action}")
    return 0 if report["ok"] or args.allow_blocked else 1


if __name__ == "__main__":
    raise SystemExit(main())
