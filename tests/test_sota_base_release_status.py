from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "sota_base_release_status.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("sota_base_release_status", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_report(path: Path, *, schema: str, ok: bool, status: str = "green") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": schema,
                "ok": ok,
                "status": status,
                "summary": {"green": 1 if ok else 0, "yellow": 0, "red": 0 if ok else 1},
                "message": "ok" if ok else "blocked",
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _args(tmp_path: Path, *, local_only: bool = False):
    return argparse.Namespace(
        local_report=tmp_path / "local" / "report.json",
        local_claim_proof=tmp_path / "local" / "claim-proof.json",
        testnet_artifacts_dir=tmp_path / "testnet",
        local_only=local_only,
    )


def test_release_status_local_only_green(tmp_path: Path) -> None:
    module = _load_module()
    args = _args(tmp_path, local_only=True)
    _write_report(args.local_report, schema="sota-local-claims-ui-smoke/v1", ok=True)
    _write_report(args.local_claim_proof, schema="sota-local-claim-proof/v1", ok=True)

    report = module.run_status(args)

    assert report["ok"] is True
    assert report["status"] == "green"
    assert report["local_ok"] is True
    assert report["testnet_ok"] is None
    assert report["blocked_gates"] == []
    assert [gate["name"] for gate in report["gates"]] == ["local_demo", "local_claim_proof"]


def test_release_status_local_only_requires_claim_proof(tmp_path: Path) -> None:
    module = _load_module()
    args = _args(tmp_path, local_only=True)
    _write_report(args.local_report, schema="sota-local-claims-ui-smoke/v1", ok=True)

    report = module.run_status(args)

    assert report["ok"] is False
    assert report["status"] == "red"
    assert report["local_ok"] is False
    assert [gate["name"] for gate in report["blocked_gates"]] == ["local_claim_proof"]


def test_release_status_full_requires_all_testnet_gates(tmp_path: Path) -> None:
    module = _load_module()
    args = _args(tmp_path)
    _write_report(args.local_report, schema="sota-local-claims-ui-smoke/v1", ok=True)
    _write_report(args.local_claim_proof, schema="sota-local-claim-proof/v1", ok=True)
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-operator-run.json", schema="sota-base-testnet-operator-run/v1", ok=False, status="red")
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-blockers.json", schema="sota-base-testnet-blockers/v1", ok=False, status="red")
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-aws-inventory.json", schema="sota-base-testnet-aws-inventory/v1", ok=False, status="red")
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-funding.json", schema="sota-base-testnet-funding/v1", ok=False, status="red")
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-secret-handles.json", schema="sota-base-testnet-secret-bootstrap/v1", ok=False, status="red")
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-apprunner-source-pack.json", schema="sota-base-testnet-apprunner-source-pack/v1", ok=False, status="yellow")
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-container-pack.json", schema="sota-base-testnet-container-pack/v1", ok=False, status="yellow")
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-browser-smoke.json", schema="sota-base-testnet-browser-smoke/v1", ok=False, status="red")
    _write_report(args.testnet_artifacts_dir / "base-sota-claim-tx-evidence.json", schema="sota-base-claim-tx-evidence/v1", ok=False, status="red")

    report = module.run_status(args)

    assert report["ok"] is False
    assert report["status"] == "red"
    assert report["local_ok"] is True
    assert report["testnet_ok"] is False
    assert {gate["name"] for gate in report["blocked_gates"]} == {
        "testnet_operator_run",
        "testnet_blockers",
        "testnet_aws_inventory",
        "testnet_funding",
        "testnet_secret_handles",
        "testnet_apprunner_source_pack",
        "testnet_browser_smoke",
        "claim_tx_evidence",
    }


def test_release_status_full_green_requires_operator_gate(tmp_path: Path) -> None:
    module = _load_module()
    args = _args(tmp_path)
    _write_report(args.local_report, schema="sota-local-claims-ui-smoke/v1", ok=True)
    _write_report(args.local_claim_proof, schema="sota-local-claim-proof/v1", ok=True)
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-operator-run.json", schema="sota-base-testnet-operator-run/v1", ok=True)
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-blockers.json", schema="sota-base-testnet-blockers/v1", ok=True)
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-aws-inventory.json", schema="sota-base-testnet-aws-inventory/v1", ok=True)
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-funding.json", schema="sota-base-testnet-funding/v1", ok=True)
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-secret-handles.json", schema="sota-base-testnet-secret-bootstrap/v1", ok=True)
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-apprunner-source-pack.json", schema="sota-base-testnet-apprunner-source-pack/v1", ok=True)
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-container-pack.json", schema="sota-base-testnet-container-pack/v1", ok=False, status="yellow")
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-browser-smoke.json", schema="sota-base-testnet-browser-smoke/v1", ok=True)
    _write_report(args.testnet_artifacts_dir / "base-sota-claim-tx-evidence.json", schema="sota-base-claim-tx-evidence/v1", ok=True)

    report = module.run_status(args)

    assert report["ok"] is True
    assert report["status"] == "green"
    assert report["testnet_ok"] is True
    assert [gate["name"] for gate in report["gates"]] == [
        "local_demo",
        "local_claim_proof",
        "testnet_operator_run",
        "testnet_blockers",
        "testnet_aws_inventory",
        "testnet_funding",
        "testnet_secret_handles",
        "testnet_apprunner_source_pack",
        "testnet_container_pack",
        "testnet_browser_smoke",
        "claim_tx_evidence",
    ]


def test_release_status_rejects_schema_mismatch(tmp_path: Path) -> None:
    module = _load_module()
    args = _args(tmp_path, local_only=True)
    _write_report(args.local_report, schema="wrong-schema", ok=True)
    _write_report(args.local_claim_proof, schema="sota-local-claim-proof/v1", ok=True)

    report = module.run_status(args)
    gate = report["gates"][0]

    assert report["ok"] is False
    assert gate["status"] == "red"
    assert gate["schema"] == "wrong-schema"
    assert gate["expected_schema"] == "sota-local-claims-ui-smoke/v1"


def test_release_status_missing_report_is_red(tmp_path: Path) -> None:
    module = _load_module()
    args = _args(tmp_path, local_only=True)

    report = module.run_status(args)
    gates = {gate["name"]: gate for gate in report["gates"]}

    assert report["ok"] is False
    assert report["local_ok"] is False
    assert gates["local_demo"]["status"] == "red"
    assert gates["local_demo"]["message"] == "Report is missing."
    assert gates["local_claim_proof"]["status"] == "red"
    assert gates["local_claim_proof"]["message"] == "Report is missing."


def test_release_status_json_without_report_out_only_prints(tmp_path: Path, capsys) -> None:
    module = _load_module()
    args = _args(tmp_path, local_only=True)
    _write_report(args.local_report, schema="sota-local-claims-ui-smoke/v1", ok=True)
    _write_report(args.local_claim_proof, schema="sota-local-claim-proof/v1", ok=True)

    exit_code = module.main(
        [
            "--local-only",
            "--local-report",
            str(args.local_report),
            "--local-claim-proof",
            str(args.local_claim_proof),
            "--json",
        ]
    )

    assert exit_code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["local_ok"] is True
    assert printed["ok"] is True
