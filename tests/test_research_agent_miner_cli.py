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
        ]
    )

    output = capsys.readouterr().out
    assert result == 0
    assert calls[0]["claim_id"] == "claim-1"
    assert calls[0]["repo_dir"] == str(tmp_path / "repo")
    assert calls[0]["submission_file"] == str(tmp_path / "submission.json")
    assert '"id":"submission-1"' in output
