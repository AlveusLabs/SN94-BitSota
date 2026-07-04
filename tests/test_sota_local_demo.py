from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "sota_local_demo.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("sota_local_demo", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_launch_claim_proof_runs_reset_after(monkeypatch) -> None:
    module = _load_module()
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="proof ok", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    module._run_local_claim_proof_reset()

    command, kwargs = calls[0]
    assert command[0] == sys.executable
    assert command[1].endswith("scripts/sota_local_claim_proof.py")
    assert "--reset-after" in command
    assert str(module.RUN_DIR / "claim-proof" / "latest.json") in command
    assert str(module.RUN_DIR / "claim-proof" / "local-claim-tx-evidence.json") in command
    assert kwargs["cwd"] == module.DOCS_REPO
    assert kwargs["timeout"] == 420
