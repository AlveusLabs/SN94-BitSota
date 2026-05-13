from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from neurons import research_agent_miner


class _FailingAgent:
    def __init__(self) -> None:
        self.mine_calls = 0

    def mine_once(self, **_kwargs):
        self.mine_calls += 1
        raise RuntimeError("boom")

    def peer_evaluate_once(self, **_kwargs):
        raise RuntimeError("peer-eval unavailable")


def test_loop_keeps_iterating_after_mining_failures(monkeypatch, capsys) -> None:
    agent = _FailingAgent()
    monkeypatch.setattr(research_agent_miner, "_build_agent", lambda _args: agent)
    monkeypatch.setattr(research_agent_miner.time, "sleep", lambda *_args, **_kwargs: None)

    args = Namespace(
        cycles=2,
        allow_peer_evaluation=False,
        interval_seconds=0,
        task_id=None,
        task_slug="cifar10-matrix-decomposition-frontier",
        participation_style="pool",
    )

    result = research_agent_miner._loop(args)

    output = capsys.readouterr().out
    assert result == 0
    assert agent.mine_calls == 2
    assert output.count("[research-agent] mining cycle failed: boom") == 2


def test_submit_workspace_delegates_to_helper(monkeypatch, capsys, tmp_path: Path) -> None:
    calls: list[dict] = []

    def _fake_submit_workspace(_args):
        calls.append(
            {
                "claim_id": _args.claim_id,
                "repo_dir": _args.repo_dir,
                "submission_file": _args.submission_file,
                "allowed_patch_path": _args.allowed_patch_path,
                "max_patch_bytes": _args.max_patch_bytes,
            }
        )
        print('{"id":"submission-1"}')
        return 0

    monkeypatch.setattr(research_agent_miner, "_submit_workspace", _fake_submit_workspace)
    result = research_agent_miner.main(
        [
            "submit-workspace",
            "--coordinator-url",
            "http://127.0.0.1:8000",
            "--claim-id",
            "claim-1",
            "--repo-dir",
            str(tmp_path / "repo"),
            "--submission-file",
            str(tmp_path / "submission.json"),
            "--allowed-patch-path",
            "train.py",
            "--max-patch-bytes",
            "1234",
        ]
    )

    output = capsys.readouterr().out
    assert result == 0
    assert calls[0]["claim_id"] == "claim-1"
    assert calls[0]["repo_dir"] == str(tmp_path / "repo")
    assert calls[0]["submission_file"] == str(tmp_path / "submission.json")
    assert calls[0]["allowed_patch_path"] == ["train.py"]
    assert calls[0]["max_patch_bytes"] == 1234
    assert '"id":"submission-1"' in output


def test_signed_request_uses_wallet_file(monkeypatch, capsys, tmp_path: Path) -> None:
    wallet_file = tmp_path / "Wallet mine.txt"
    wallet_file.write_text("Hotkey mnemonic: " + "abandon " * 11 + "about\n", encoding="utf-8")

    calls: list[dict] = []

    class _Response:
        text = ""

        def json(self) -> dict[str, str]:
            return {"status": "ok"}

    class _FakeCoordinatorClient:
        def __init__(self, *, base_url: str, wallet) -> None:
            calls.append(
                {
                    "base_url": base_url,
                    "wallet_hotkey": wallet.hotkey.ss58_address,
                }
            )

        def request(self, method: str, path: str, *, body=None, params=None, sign=False):
            calls.append(
                {
                    "method": method,
                    "path": path,
                    "body": body,
                    "params": params,
                    "sign": sign,
                }
            )
            return _Response()

    monkeypatch.setattr(research_agent_miner, "CoordinatorClient", _FakeCoordinatorClient)

    result = research_agent_miner.main(
        [
            "signed-request",
            "--coordinator-url",
            "http://127.0.0.1:8000",
            "--method",
            "POST",
            "--path",
            "/api/v1/tasks/task-1/claim",
            "--body-json",
            '{"claim_description":"test"}',
            "--wallet-file",
            str(wallet_file),
        ]
    )

    output = capsys.readouterr().out
    assert result == 0
    assert calls[0]["base_url"] == "http://127.0.0.1:8000"
    assert calls[1]["method"] == "POST"
    assert calls[1]["path"] == "/api/v1/tasks/task-1/claim"
    assert calls[1]["body"] == {"claim_description": "test"}
    assert calls[1]["sign"] is True
    assert '"status": "ok"' in output
