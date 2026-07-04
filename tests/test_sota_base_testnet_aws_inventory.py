from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "sota_base_testnet_aws_inventory.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("sota_base_testnet_aws_inventory", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _args(tmp_path: Path, **overrides):
    values = {
        "aws_profile": "moonrocklab-frankfurt",
        "region": "eu-central-1",
        "timeout": 1.0,
        "service_url": [],
        "external_dns_owner": "",
        "required_secret": list(
            (
                "base-sota/test/base-sepolia/rpc-url",
                "base-sota/test/base-sepolia/deployer",
                "base-sota/test/base-sepolia/root-publisher",
                "base-sota/test/base-sepolia/indexer-admin-token",
                "base-sota/test/base-sepolia/autoresearch-database-url",
                "base-sota/test/base-sepolia/autoresearch-admin-token",
            )
        ),
        "out": tmp_path / "inventory.json",
        "json": False,
        "allow_blocked": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_inventory_reports_red_without_base_specific_resources(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    args = _args(tmp_path)

    def fake_run_aws(cmd: list[str], *, profile: str, region: str, timeout: float):
        assert profile == "moonrocklab-frankfurt"
        if cmd[:2] == ["sts", "get-caller-identity"]:
            assert region == ""
            return {"Account": "924380800822", "Arn": "arn:aws:sts::924380800822:assumed-role/Frankfurt-PowerUser/test"}
        if cmd[:2] == ["apprunner", "list-services"]:
            assert region == "eu-central-1"
            return {
                "ServiceSummaryList": [
                    {"ServiceName": "bitsota-autoresearch-test", "Status": "RUNNING", "ServiceUrl": "example.awsapprunner.com"},
                    {"ServiceName": "bitsota-pool-test", "Status": "RUNNING", "ServiceUrl": "pool.awsapprunner.com"},
                ]
            }
        if cmd[:2] == ["route53", "list-hosted-zones"]:
            assert region == ""
            return {"HostedZones": [{"Name": "example.com.", "Id": "/hostedzone/Z1"}]}
        if cmd[:2] == ["ecr", "describe-repositories"]:
            return {"repositories": [{"repositoryName": "bitsota-autoresearch-test"}]}
        if cmd[:2] == ["secretsmanager", "list-secrets"]:
            return {"SecretList": [{"Name": "bitsota/test/autoresearch", "ARN": "arn:secret"}]}
        if cmd[:2] == ["apprunner", "list-connections"]:
            return {"ConnectionSummaryList": [{"ConnectionName": "bitsota", "Status": "AVAILABLE", "ProviderType": "GITHUB"}]}
        raise AssertionError(cmd)

    monkeypatch.setattr(module, "_run_aws", fake_run_aws)

    report = module.build_inventory(args)
    checks = {check["name"]: check for check in report["checks"]}

    assert report["schema"] == "sota-base-testnet-aws-inventory/v1"
    assert report["ok"] is False
    assert report["status"] == "red"
    assert checks["aws_identity"]["status"] == "green"
    assert checks["route53_bitsota_zone"]["status"] == "red"
    assert checks["base_sota_apprunner_services"]["status"] == "red"
    assert checks["apprunner_github_connection"]["status"] == "green"
    assert checks["base_sota_ecr_repos"]["status"] == "yellow"
    assert checks["base_sepolia_secret_handles"]["status"] == "red"
    assert report["read_only"] is True
    assert "read_secret_values" in report["does_not"]


def test_inventory_reports_green_with_base_sota_resources(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    args = _args(tmp_path)

    def fake_run_aws(cmd: list[str], *, profile: str, region: str, timeout: float):
        if cmd[:2] == ["sts", "get-caller-identity"]:
            return {"Account": "924380800822", "Arn": "arn:aws:sts::924380800822:assumed-role/Frankfurt-PowerUser/test"}
        if cmd[:2] == ["apprunner", "list-services"]:
            return {
                "ServiceSummaryList": [
                    {"ServiceName": "base-sota-claims-ui-test", "Status": "RUNNING"},
                    {"ServiceName": "base-sota-claims-api-test", "Status": "RUNNING"},
                    {"ServiceName": "base-sota-coordinator-test", "Status": "RUNNING"},
                    {"ServiceName": "base-sota-root-publisher-test", "Status": "RUNNING"},
                ]
            }
        if cmd[:2] == ["route53", "list-hosted-zones"]:
            return {"HostedZones": [{"Name": "bitsota.com.", "Id": "/hostedzone/ZBIT", "ResourceRecordSetCount": 12}]}
        if cmd[:2] == ["ecr", "describe-repositories"]:
            return {"repositories": [{"repositoryName": "base-sota-claims-api"}]}
        if cmd[:2] == ["secretsmanager", "list-secrets"]:
            return {
                "SecretList": [
                    {"Name": name, "ARN": f"arn:{index}"}
                    for index, name in enumerate(args.required_secret, start=1)
                ]
            }
        if cmd[:2] == ["apprunner", "list-connections"]:
            return {"ConnectionSummaryList": [{"ConnectionName": "bitsota", "Status": "AVAILABLE", "ProviderType": "GITHUB"}]}
        raise AssertionError(cmd)

    monkeypatch.setattr(module, "_run_aws", fake_run_aws)

    report = module.build_inventory(args)

    assert report["ok"] is True
    assert report["status"] == "green"
    assert report["summary"] == {"green": 6, "yellow": 0, "red": 0}
    assert all(check["status"] == "green" for check in report["checks"])


def test_inventory_accepts_configured_app_runner_service_urls(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    args = _args(
        tmp_path,
        service_url=[
            "claims_ui=https://ui.awsapprunner.com",
            "claims_api=https://api.awsapprunner.com",
            "coordinator=https://coordinator.awsapprunner.com",
            "root_publisher=https://root.awsapprunner.com",
        ],
    )

    def fake_run_aws(cmd: list[str], *, profile: str, region: str, timeout: float):
        if cmd[:2] == ["sts", "get-caller-identity"]:
            return {"Account": "924380800822", "Arn": "arn:aws:sts::924380800822:assumed-role/Frankfurt-PowerUser/test"}
        if cmd[:2] == ["apprunner", "list-services"]:
            return {
                "ServiceSummaryList": [
                    {"ServiceName": "bitsota-website-test", "Status": "RUNNING", "ServiceUrl": "ui.awsapprunner.com"},
                    {"ServiceName": "bitsota-pool-test", "Status": "RUNNING", "ServiceUrl": "api.awsapprunner.com"},
                    {"ServiceName": "bitsota-autoresearch-test", "Status": "RUNNING", "ServiceUrl": "coordinator.awsapprunner.com"},
                    {"ServiceName": "bitsota-root-test", "Status": "RUNNING", "ServiceUrl": "root.awsapprunner.com"},
                ]
            }
        if cmd[:2] == ["route53", "list-hosted-zones"]:
            return {"HostedZones": [{"Name": "bitsota.com.", "Id": "/hostedzone/ZBIT", "ResourceRecordSetCount": 12}]}
        if cmd[:2] == ["ecr", "describe-repositories"]:
            return {"repositories": [{"repositoryName": "base-sota-claims-api"}]}
        if cmd[:2] == ["secretsmanager", "list-secrets"]:
            return {
                "SecretList": [
                    {"Name": name, "ARN": f"arn:{index}"}
                    for index, name in enumerate(args.required_secret, start=1)
                ]
            }
        if cmd[:2] == ["apprunner", "list-connections"]:
            return {"ConnectionSummaryList": [{"ConnectionName": "bitsota", "Status": "AVAILABLE", "ProviderType": "GITHUB"}]}
        raise AssertionError(cmd)

    monkeypatch.setattr(module, "_run_aws", fake_run_aws)

    report = module.build_inventory(args)
    check = next(check for check in report["checks"] if check["name"] == "base_sota_apprunner_services")

    assert report["ok"] is True
    assert check["status"] == "green"
    assert "claims_ui=bitsota-website-test" in check["detail"]
    assert report["inventory"]["configured_service_urls"]["claims_api"] == "https://api.awsapprunner.com"
    assert report["inventory"]["app_runner_connections"][0]["name"] == "bitsota"
    assert report["inventory"]["required_secret_handles"] == args.required_secret


def test_inventory_accepts_direct_service_urls_without_route53_zone(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    args = _args(
        tmp_path,
        service_url=[
            "claims_ui=https://ui.awsapprunner.com",
            "claims_api=https://api.awsapprunner.com",
            "coordinator=https://coordinator.awsapprunner.com",
            "root_publisher=https://root.awsapprunner.com",
        ],
    )

    def fake_run_aws(cmd: list[str], *, profile: str, region: str, timeout: float):
        if cmd[:2] == ["sts", "get-caller-identity"]:
            return {"Account": "924380800822", "Arn": "arn:aws:sts::924380800822:assumed-role/Frankfurt-PowerUser/test"}
        if cmd[:2] == ["apprunner", "list-services"]:
            return {
                "ServiceSummaryList": [
                    {"ServiceName": "bitsota-website-test", "Status": "RUNNING", "ServiceUrl": "ui.awsapprunner.com"},
                    {"ServiceName": "bitsota-pool-test", "Status": "RUNNING", "ServiceUrl": "api.awsapprunner.com"},
                    {"ServiceName": "bitsota-autoresearch-test", "Status": "RUNNING", "ServiceUrl": "coordinator.awsapprunner.com"},
                    {"ServiceName": "bitsota-root-test", "Status": "RUNNING", "ServiceUrl": "root.awsapprunner.com"},
                ]
            }
        if cmd[:2] == ["route53", "list-hosted-zones"]:
            return {"HostedZones": []}
        if cmd[:2] == ["ecr", "describe-repositories"]:
            return {"repositories": [{"repositoryName": "base-sota-claims-api"}]}
        if cmd[:2] == ["secretsmanager", "list-secrets"]:
            return {
                "SecretList": [
                    {"Name": name, "ARN": f"arn:{index}"}
                    for index, name in enumerate(args.required_secret, start=1)
                ]
            }
        if cmd[:2] == ["apprunner", "list-connections"]:
            return {"ConnectionSummaryList": [{"ConnectionName": "bitsota", "Status": "AVAILABLE", "ProviderType": "GITHUB"}]}
        raise AssertionError(cmd)

    monkeypatch.setattr(module, "_run_aws", fake_run_aws)

    report = module.build_inventory(args)
    checks = {check["name"]: check for check in report["checks"]}

    assert report["ok"] is True
    assert report["status"] == "green"
    assert checks["route53_bitsota_zone"]["status"] == "green"
    assert "direct public service URLs" in checks["route53_bitsota_zone"]["detail"]


def test_inventory_accepts_external_dns_owner_without_route53_zone(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    args = _args(tmp_path, external_dns_owner="Cloudflare bitsota.com account")

    def fake_run_aws(cmd: list[str], *, profile: str, region: str, timeout: float):
        if cmd[:2] == ["sts", "get-caller-identity"]:
            return {"Account": "924380800822", "Arn": "arn:aws:sts::924380800822:assumed-role/Frankfurt-PowerUser/test"}
        if cmd[:2] == ["apprunner", "list-services"]:
            return {
                "ServiceSummaryList": [
                    {"ServiceName": "base-sota-claims-ui-test", "Status": "RUNNING"},
                    {"ServiceName": "base-sota-claims-api-test", "Status": "RUNNING"},
                    {"ServiceName": "base-sota-coordinator-test", "Status": "RUNNING"},
                    {"ServiceName": "base-sota-root-publisher-test", "Status": "RUNNING"},
                ]
            }
        if cmd[:2] == ["route53", "list-hosted-zones"]:
            return {"HostedZones": []}
        if cmd[:2] == ["ecr", "describe-repositories"]:
            return {"repositories": [{"repositoryName": "base-sota-claims-api"}]}
        if cmd[:2] == ["secretsmanager", "list-secrets"]:
            return {
                "SecretList": [
                    {"Name": name, "ARN": f"arn:{index}"}
                    for index, name in enumerate(args.required_secret, start=1)
                ]
            }
        if cmd[:2] == ["apprunner", "list-connections"]:
            return {"ConnectionSummaryList": [{"ConnectionName": "bitsota", "Status": "AVAILABLE", "ProviderType": "GITHUB"}]}
        raise AssertionError(cmd)

    monkeypatch.setattr(module, "_run_aws", fake_run_aws)

    report = module.build_inventory(args)
    checks = {check["name"]: check for check in report["checks"]}

    assert report["ok"] is True
    assert checks["route53_bitsota_zone"]["status"] == "green"
    assert "Cloudflare bitsota.com account" in checks["route53_bitsota_zone"]["detail"]
    assert report["inventory"]["external_dns_owner"] == "Cloudflare bitsota.com account"


def test_inventory_reports_unavailable_resources_when_aws_queries_fail(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    args = _args(tmp_path)

    def fake_run_aws(cmd: list[str], *, profile: str, region: str, timeout: float):
        raise RuntimeError("SSO expired")

    monkeypatch.setattr(module, "_run_aws", fake_run_aws)

    report = module.build_inventory(args)
    checks = {check["name"]: check for check in report["checks"]}

    assert report["ok"] is False
    assert report["status"] == "red"
    assert checks["aws_identity"]["status"] == "red"
    assert checks["route53_bitsota_zone"]["status"] == "red"
    assert "inventory unavailable" in checks["route53_bitsota_zone"]["detail"]
    assert "No Route53 hosted zone" not in checks["route53_bitsota_zone"]["detail"]
    assert checks["base_sota_apprunner_services"]["status"] == "red"
    assert "service inventory unavailable" in checks["base_sota_apprunner_services"]["detail"]
    assert "Missing Base SOTA-specific" not in checks["base_sota_apprunner_services"]["detail"]
    assert checks["apprunner_github_connection"]["status"] == "red"
    assert "connection inventory unavailable" in checks["apprunner_github_connection"]["detail"]
    assert checks["base_sota_ecr_repos"]["status"] == "yellow"
    assert "repository inventory unavailable" in checks["base_sota_ecr_repos"]["detail"]
    assert checks["base_sepolia_secret_handles"]["status"] == "red"
    assert "handle inventory unavailable" in checks["base_sepolia_secret_handles"]["detail"]
    assert "Missing required" not in checks["base_sepolia_secret_handles"]["detail"]
    assert checks["apprunner_list_services"]["status"] == "red"


def test_inventory_only_records_secret_handles_not_secret_values(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    args = _args(tmp_path)

    def fake_run_aws(cmd: list[str], *, profile: str, region: str, timeout: float):
        if cmd[:2] == ["sts", "get-caller-identity"]:
            return {"Account": "924380800822", "Arn": "arn:aws:sts::924380800822:assumed-role/Frankfurt-PowerUser/test"}
        if cmd[:2] == ["apprunner", "list-services"]:
            return {"ServiceSummaryList": []}
        if cmd[:2] == ["route53", "list-hosted-zones"]:
            return {"HostedZones": []}
        if cmd[:2] == ["ecr", "describe-repositories"]:
            return {"repositories": []}
        if cmd[:2] == ["secretsmanager", "list-secrets"]:
            return {
                "SecretList": [
                    {
                        "Name": "base-sota/test/deployer",
                        "ARN": "arn:secret",
                        "Description": "test handle",
                        "SecretString": "must-not-appear",
                        "SecretBinary": "must-not-appear",
                        "Tags": [{"Key": "owner", "Value": "ops"}],
                    }
                ]
            }
        if cmd[:2] == ["apprunner", "list-connections"]:
            return {"ConnectionSummaryList": []}
        raise AssertionError(cmd)

    monkeypatch.setattr(module, "_run_aws", fake_run_aws)

    report = module.build_inventory(args)
    secret_handles = report["inventory"]["secret_handles"]

    assert secret_handles == [{"name": "base-sota/test/deployer", "arn": "arn:secret", "description": "test handle"}]
    serialized = str(report)
    assert "must-not-appear" not in serialized
