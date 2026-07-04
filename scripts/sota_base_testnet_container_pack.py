#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any


REPOS = Path("/home/mekaneeky/repos")
DEFAULT_ARTIFACTS_DIR = REPOS / ".sota-base-testnet"
DEFAULT_SERVICE_PACK = DEFAULT_ARTIFACTS_DIR / "base-sota-testnet-service-pack.json"
DEFAULT_OUT = DEFAULT_ARTIFACTS_DIR / "base-sota-testnet-container-pack.json"
DEFAULT_APPRUNNER_IMAGE_DIR = DEFAULT_ARTIFACTS_DIR / "apprunner-image"
DEFAULT_REGION = "eu-central-1"
DEFAULT_PROFILE = "moonrocklab-frankfurt"
DEFAULT_TAG = "test"

SERVICE_IMAGES = {
    "claims_ui": {
        "repository": "base-sota-claims-ui-test",
        "context": REPOS / "bitsota_website",
        "dockerfile": REPOS / "bitsota_website" / "Dockerfile",
        "port": "3000",
        "health_path": "/claims",
    },
    "indexer_api": {
        "repository": "base-sota-indexer-api-test",
        "context": REPOS / "94-agent-community",
        "dockerfile": REPOS / "94-agent-community" / "Dockerfile.sota-base-indexer",
        "port": "8010",
        "health_path": "/health",
    },
    "autoresearch_coordinator": {
        "repository": "base-sota-autoresearch-coordinator-test",
        "context": REPOS / "autoresearch-bittensor",
        "dockerfile": REPOS / "autoresearch-bittensor" / "Dockerfile",
        "port": "8000",
        "health_path": "/readyz",
    },
}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_aws(args: list[str], *, profile: str, region: str, timeout: float) -> dict[str, Any]:
    command = ["aws", *args, "--output", "json"]
    if profile:
        command.extend(["--profile", profile])
    if region:
        command.extend(["--region", region])
    result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f"aws exited {result.returncode}")
    return json.loads(result.stdout or "{}")


def _account_id(profile: str, timeout: float) -> str:
    payload = _run_aws(["sts", "get-caller-identity"], profile=profile, region="", timeout=timeout)
    account = str(payload.get("Account") or "").strip()
    if not account:
        raise RuntimeError("AWS STS did not return an account id")
    return account


def _ensure_ecr_repo(name: str, *, profile: str, region: str, timeout: float) -> dict[str, Any]:
    try:
        payload = _run_aws(["ecr", "describe-repositories", "--repository-names", name], profile=profile, region=region, timeout=timeout)
        repos = payload.get("repositories") or []
        if repos:
            return {"status": "green", "action": "exists", "repository_uri": str(repos[0].get("repositoryUri") or "")}
    except Exception as exc:
        if "RepositoryNotFoundException" not in str(exc):
            raise
    payload = _run_aws(
        [
            "ecr",
            "create-repository",
            "--repository-name",
            name,
            "--image-scanning-configuration",
            "scanOnPush=true",
            "--tags",
            "Key=project,Value=base-sota",
            "Key=environment,Value=base-sepolia",
        ],
        profile=profile,
        region=region,
        timeout=timeout,
    )
    repo = dict(payload.get("repository") or {})
    return {"status": "green", "action": "created", "repository_uri": str(repo.get("repositoryUri") or "")}


def _service_by_key(pack: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(service.get("key")): dict(service)
        for service in pack.get("services") or []
        if isinstance(service, dict) and service.get("key")
    }


def _docker_build_args(service: dict[str, Any]) -> list[str]:
    args: list[str] = []
    if service.get("key") != "claims_ui":
        return args
    for key, value in sorted(dict(service.get("env_public_values") or {}).items()):
        if key.startswith("NEXT_PUBLIC_"):
            args.extend(["--build-arg", f"{key}={value}"])
    return args


def _apprunner_image_input(
    *,
    service: dict[str, Any],
    image_uri: str,
    access_role_arn: str,
    instance_role_arn: str,
    port: str,
    health_path: str,
) -> dict[str, Any]:
    env_values = {str(k): str(v) for k, v in dict(service.get("env_public_values") or {}).items()}
    env_secrets = {str(k): str(v) for k, v in dict(service.get("env_secret_map") or {}).items()}
    source_configuration: dict[str, Any] = {
        "ImageRepository": {
            "ImageIdentifier": image_uri,
            "ImageRepositoryType": "ECR",
            "ImageConfiguration": {
                "Port": port,
                "RuntimeEnvironmentVariables": env_values,
                "RuntimeEnvironmentSecrets": env_secrets,
            },
        },
        "AutoDeploymentsEnabled": False,
    }
    if access_role_arn:
        source_configuration["AuthenticationConfiguration"] = {"AccessRoleArn": access_role_arn}
    payload: dict[str, Any] = {
        "ServiceName": f"base-sota-{str(service['key']).replace('_', '-')}-test",
        "SourceConfiguration": source_configuration,
        "InstanceConfiguration": {
            "Cpu": "1024",
            "Memory": "2048",
        },
        "HealthCheckConfiguration": {
            "Protocol": "HTTP",
            "Path": health_path,
            "Interval": 10,
            "Timeout": 5,
            "HealthyThreshold": 1,
            "UnhealthyThreshold": 5,
        },
        "Tags": [
            {"Key": "project", "Value": "base-sota"},
            {"Key": "environment", "Value": "base-sepolia"},
            {"Key": "component", "Value": str(service["key"])},
        ],
    }
    if instance_role_arn:
        payload["InstanceConfiguration"]["InstanceRoleArn"] = instance_role_arn
    return payload


def build_pack(args: argparse.Namespace) -> dict[str, Any]:
    service_pack = _load_json(args.service_pack)
    services = _service_by_key(service_pack)
    account = args.account_id or _account_id(args.aws_profile, args.timeout)
    registry = f"{account}.dkr.ecr.{args.region}.amazonaws.com"
    args.apprunner_out_dir.mkdir(parents=True, exist_ok=True)
    images: list[dict[str, Any]] = []
    checks: list[dict[str, str]] = []

    for key, spec in SERVICE_IMAGES.items():
        service = services.get(key)
        if not service:
            checks.append({"name": f"service_{key}", "status": "red", "detail": f"{key} is missing from {args.service_pack}."})
            continue
        context = Path(spec["context"])
        dockerfile = Path(spec["dockerfile"])
        repo_name = str(spec["repository"])
        image_uri = f"{registry}/{repo_name}:{args.tag}"
        ecr = {"status": "yellow", "action": "not_checked", "repository_uri": f"{registry}/{repo_name}"}
        if args.ensure_ecr:
            try:
                ecr = _ensure_ecr_repo(repo_name, profile=args.aws_profile, region=args.region, timeout=args.timeout)
            except Exception as exc:
                ecr = {"status": "red", "action": "ensure_failed", "detail": str(exc), "repository_uri": f"{registry}/{repo_name}"}
        build_cmd = [
            "docker",
            "build",
            "-f",
            str(dockerfile),
            "-t",
            image_uri,
            *_docker_build_args(service),
            str(context),
        ]
        push_cmd = ["docker", "push", image_uri]
        input_payload = _apprunner_image_input(
            service=service,
            image_uri=image_uri,
            access_role_arn=args.apprunner_ecr_access_role_arn,
            instance_role_arn=args.apprunner_instance_role_arn,
            port=str(spec["port"]),
            health_path=str(spec["health_path"]),
        )
        input_path = args.apprunner_out_dir / f"{input_payload['ServiceName']}.json"
        _write_json(input_path, input_payload)
        images.append(
            {
                "key": key,
                "repository_name": repo_name,
                "repository_uri": ecr.get("repository_uri"),
                "image_uri": image_uri,
                "context": str(context),
                "dockerfile": str(dockerfile),
                "ecr": ecr,
                "build_command": build_cmd,
                "push_command": push_cmd,
                "apprunner_input": str(input_path),
                "create_service_command": [
                    "aws",
                    "apprunner",
                    "create-service",
                    "--cli-input-json",
                    f"file://{input_path}",
                    "--profile",
                    args.aws_profile,
                    "--region",
                    args.region,
                ],
            }
        )
        checks.append(
            {
                "name": f"container_{key}",
                "status": "green" if dockerfile.exists() and context.exists() and ecr.get("status") != "red" else "red",
                "detail": f"{key} container path uses {dockerfile} -> {image_uri}.",
            }
        )

    if not args.apprunner_ecr_access_role_arn:
        checks.append(
            {
                "name": "apprunner_ecr_access_role",
                "status": "yellow",
                "detail": "No App Runner ECR access role ARN was supplied; image service JSON was written without AuthenticationConfiguration.",
                "remediation": "Set --apprunner-ecr-access-role-arn to a role App Runner can assume for private ECR pulls before creating services.",
            }
        )
    status = "red" if any(check["status"] == "red" for check in checks) else ("yellow" if any(check["status"] == "yellow" for check in checks) else "green")
    return {
        "schema": "sota-base-testnet-container-pack/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ok": status == "green",
        "status": status,
        "aws": {"account": account, "region": args.region, "profile": args.aws_profile},
        "service_pack": str(args.service_pack),
        "tag": args.tag,
        "images": images,
        "checks": checks,
        "summary": {
            "green": sum(1 for check in checks if check["status"] == "green"),
            "yellow": sum(1 for check in checks if check["status"] == "yellow"),
            "red": sum(1 for check in checks if check["status"] == "red"),
        },
        "does_not": ["touch_production_bittensor", "touch_base_mainnet", "print_secret_values"],
        "next_actions": [str(check.get("remediation")) for check in checks if check.get("status") != "green" and check.get("remediation")],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Base SOTA testnet ECR/App Runner image deployment artifacts.")
    parser.add_argument("--service-pack", type=Path, default=DEFAULT_SERVICE_PACK)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--apprunner-out-dir", type=Path, default=DEFAULT_APPRUNNER_IMAGE_DIR)
    parser.add_argument("--aws-profile", default=DEFAULT_PROFILE)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--account-id", default="")
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--ensure-ecr", action="store_true", help="Create missing test ECR repositories.")
    parser.add_argument("--apprunner-ecr-access-role-arn", default="")
    parser.add_argument("--apprunner-instance-role-arn", default="arn:aws:iam::924380800822:role/AppRunnerReadSecrets")
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--allow-blocked", action="store_true")
    args = parser.parse_args(argv)

    report = build_pack(args)
    _write_json(args.out, report)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Base SOTA container pack: {report['status'].upper()}")
        print(f"Summary: {report['summary']['green']} green, {report['summary']['yellow']} yellow, {report['summary']['red']} red")
        print(f"Report: {args.out}")
        for image in report["images"]:
            print(f"- {image['key']}: {image['image_uri']}")
        for action in report["next_actions"]:
            print(f"- next: {action}")
    return 0 if report["ok"] or args.allow_blocked else 1


if __name__ == "__main__":
    raise SystemExit(main())
