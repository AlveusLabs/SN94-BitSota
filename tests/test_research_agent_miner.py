from __future__ import annotations

import json
from pathlib import Path
import shlex
import subprocess
import sys

from substrateinterface import Keypair

from miner.research_agent import (
    AgentMinerConfig,
    OpenAICompatibleChatClient,
    ResearchAgentMiner,
    build_agent_intro_markdown,
    submit_claimed_workspace,
)
from miner.research_competitions import CompetitionMode, ParticipationStyle, list_builtin_research_competitions


class FakeCoordinator:
    def __init__(self) -> None:
        self.hotkey = Keypair.create_from_mnemonic(Keypair.generate_mnemonic()).ss58_address
        self.base_url = "http://127.0.0.1:8000"
        self.claimed_tasks: list[dict] = []
        self.claimed_work_items: list[dict] = []
        self.cancelled_claims: list[str] = []
        self.submissions: list[dict] = []
        self.peer_reviews: list[dict] = []
        self.selected = None
        self.onboard = "# onboard\n"
        self.task_submissions: list[dict] = []
        self.detail_payload: dict = {
            "peer_evaluations": [],
            "peer_consensus": None,
            "submission": {},
        }

    def select_task(self, *, task_id=None, task_slug=None, participation_style: str):
        return self.selected

    def get_onboard_markdown(self, task_id: str) -> str:
        return self.onboard

    def list_submissions(self, *, task_id=None, status=None):
        rows = list(self.task_submissions)
        if status is not None:
            rows = [row for row in rows if str(row.get("status")) == str(status)]
        return rows

    def claim_task(self, *, task_id: str, claim_description: str, base_submission_id=None):
        payload = {
            "id": "claim-direct-1",
            "task_id": task_id,
            "claim_description": claim_description,
            "base_submission_id": base_submission_id,
        }
        self.claimed_tasks.append(payload)
        return payload

    def claim_work_item(self, *, work_item_id: str, claim_description: str | None = None):
        payload = {
            "id": "claim-work-1",
            "work_item_id": work_item_id,
            "claim_description": claim_description,
        }
        self.claimed_work_items.append(payload)
        return payload

    def submit_submission(self, **kwargs):
        payload = {"id": f"submission-{len(self.submissions) + 1}", **kwargs}
        self.submissions.append(payload)
        return payload

    def cancel_claim(self, *, claim_id: str):
        self.cancelled_claims.append(str(claim_id))
        return {"id": str(claim_id), "status": "cancelled"}

    def get_submission_detail(self, submission_id: str):
        return dict(self.detail_payload)

    def peer_evaluate_submission(self, **kwargs):
        payload = {"id": f"peer-eval-{len(self.peer_reviews) + 1}", **kwargs}
        self.peer_reviews.append(payload)
        return payload

    def get_peer_consensus(self, submission_id: str):
        return {"submission_id": submission_id, "status": "accepted", "accepted_count": 2}


class FakeLLM:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)
        self.prompts: list[dict] = []

    def complete_json(self, *, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int):
        self.prompts.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        if not self.responses:
            raise RuntimeError("no fake LLM responses left")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return dict(response)


class FakeResponse:
    def __init__(self, status_code: int, payload: dict) -> None:
        self.status_code = status_code
        self._payload = dict(payload)
        self.text = str(payload)

    def json(self) -> dict:
        return dict(self._payload)


class RecordingSession:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = list(responses)
        self.requests: list[dict] = []

    def post(self, url: str, *, json=None, headers=None, timeout=None):
        self.requests.append(
            {
                "url": url,
                "json": dict(json or {}),
                "headers": dict(headers or {}),
                "timeout": timeout,
            }
        )
        if not self.responses:
            raise RuntimeError("no fake responses left")
        return self.responses.pop(0)


def _build_agent(
    coordinator: FakeCoordinator,
    llm: FakeLLM,
    *,
    execution_enabled: bool = False,
    execution_attempts: int = 2,
) -> ResearchAgentMiner:
    agent = ResearchAgentMiner(
        coordinator=coordinator,  # type: ignore[arg-type]
        llm=llm,  # type: ignore[arg-type]
        config=AgentMinerConfig(
            participation_style=ParticipationStyle.direct,
            workspace_root=Path(".tmp-agent-tests"),
            execution_enabled=execution_enabled,
            execution_max_attempts=execution_attempts,
        ),
    )
    agent._load_repository_context = lambda **_: [  # type: ignore[method-assign]
        {"path": "README.md", "content": "example context"}
    ]
    return agent


def _init_git_repo(tmp_path: Path, files: dict[str, str]) -> tuple[Path, str]:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    subprocess.run(["git", "init"], cwd=repo_dir, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_dir, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_dir, check=True, capture_output=True, text=True)
    for rel_path, content in files.items():
        target = repo_dir / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo_dir, check=True, capture_output=True, text=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=repo_dir, check=True, capture_output=True, text=True)
    base_ref = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return repo_dir, base_ref


def _build_patch(repo_dir: Path, rel_path: str, new_content: str) -> str:
    target = repo_dir / rel_path
    original = target.read_text(encoding="utf-8")
    target.write_text(new_content, encoding="utf-8")
    patch = subprocess.run(
        ["git", "diff", "--", rel_path],
        cwd=repo_dir,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    target.write_text(original, encoding="utf-8")
    return patch


def test_builtin_research_catalog_exposes_builtin_competitions() -> None:
    templates = list_builtin_research_competitions()
    assert len(templates) == 5
    assert templates[0].slug == "nanogpt-default"
    assert any(template.slug == "bitnet-cpu-ternary-kernel" for template in templates)
    bitnet = next(template for template in templates if template.slug == "bitnet-cpu-ternary-kernel")
    assert bitnet.metric_name == "tokens_per_second"
    assert bitnet.metric_direction.value == "maximize"


def test_build_agent_intro_markdown_separates_autonomous_and_gui_modes() -> None:
    task = {
        "id": "task-1",
        "slug": "nanogpt-default",
        "title": "nanoGPT",
        "base_ref": "main",
        "repository": "https://example.com/repo.git",
        "metric_name": "val_bpb",
        "metric_direction": "minimize",
        "competition_mode": CompetitionMode.standard.value,
        "benchmark_command": "python3 train.py",
    }
    claim = {"id": "claim-1"}
    work_item = {"id": "work-1", "title": "Baseline", "instructions": "Edit train.py."}

    gui_intro = build_agent_intro_markdown(
        task=task,
        work_item=work_item,
        claim=claim,
        onboard="# onboard\nEdit train.py and report val_bpb.\n",
        coordinator_url="http://127.0.0.1:8000",
        mode="gui_managed",
        repo_dir=Path("/tmp/work/repo"),
        workspace_dir=Path("/tmp/work"),
        submission_file=Path("/tmp/work/submission.json"),
        submission_result_file=Path("/tmp/work/submission_result.json"),
    )
    autonomous_intro = build_agent_intro_markdown(
        task=task,
        work_item=work_item,
        claim=claim,
        onboard="# onboard\nEdit train.py and report val_bpb.\n",
        coordinator_url="http://127.0.0.1:8000",
        mode="autonomous",
        repo_dir=Path("/tmp/work/repo"),
        workspace_dir=Path("/tmp/work"),
        submission_file=Path("/tmp/work/submission.json"),
        submission_result_file=Path("/tmp/work/submission_result.json"),
    )

    assert "Do not submit directly to the coordinator" in gui_intro
    assert "submission.json" in gui_intro
    assert "You are allowed to submit directly to the coordinator" in autonomous_intro
    assert "bitsota-research-agent submit-workspace" in autonomous_intro
    assert "claim-1" in autonomous_intro


def test_infer_file_candidates_supports_native_kernel_paths() -> None:
    agent = _build_agent(FakeCoordinator(), FakeLLM([]))

    candidates = agent._infer_file_candidates(
        """
        Inspect README.md, http://127.0.0.1:8000/api/v1/tasks/example/onboard.md,
        utils/generate-dummy-bitnet-model.py, utils/e2e_benchmark.py,
        src/ggml-bitnet-mad.cpp, src/ggml-bitnet-lut.cpp, include/bitnet.h, and CMakeLists.txt.
        """
    )

    assert candidates[0] == "README.md"
    assert "utils/generate-dummy-bitnet-model.py" in candidates
    assert "utils/e2e_benchmark.py" in candidates
    assert "src/ggml-bitnet-mad.cpp" in candidates
    assert "src/ggml-bitnet-lut.cpp" in candidates
    assert "include/bitnet.h" in candidates
    assert "CMakeLists.txt" in candidates
    assert "//github.c" not in candidates
    assert "8000/api/v1/tasks/example/onboard.md" not in candidates


def test_agent_mine_once_can_claim_pool_work_item_and_submit() -> None:
    coordinator = FakeCoordinator()
    coordinator.selected = type(
        "Selection",
        (),
        {
            "task": {
                "id": "task-standard",
                "slug": "nanogpt-default",
                "title": "nanoGPT",
                "base_ref": "main",
                "repository": "https://example.com/repo.git",
                "metric_name": "val_bpb",
                "metric_direction": "minimize",
                "competition_mode": CompetitionMode.standard.value,
            },
            "work_item": {
                "id": "work-1",
                "title": "Establish baseline",
                "instructions": "Run one baseline replay.",
            },
        },
    )()
    llm = FakeLLM(
        [
            {
                "claim_description": "Try a tighter baseline replay.",
                "summary": "Agentic baseline submission.",
                "patch": "diff --git a/train.py b/train.py\n+baseline\n",
                "claimed_metrics": {"val_bpb": 1.95},
            }
        ]
    )
    agent = _build_agent(coordinator, llm)

    result = agent.mine_once(participation_style=ParticipationStyle.pool)

    assert result["submission"]["claim_id"] == "claim-work-1"
    assert coordinator.claimed_work_items[0]["work_item_id"] == "work-1"
    assert coordinator.submissions[0]["claimed_metrics"]["val_bpb"] == 1.95


def test_agent_mine_once_centerless_auto_fills_prior_idea_reference() -> None:
    coordinator = FakeCoordinator()
    coordinator.selected = type(
        "Selection",
        (),
        {
            "task": {
                "id": "task-centerless",
                "slug": "nanogpt-default",
                "title": "nanoGPT",
                "base_ref": "main",
                "repository": "https://example.com/repo.git",
                "metric_name": "val_bpb",
                "metric_direction": "minimize",
                "competition_mode": CompetitionMode.centerless.value,
            },
            "work_item": None,
        },
    )()
    coordinator.task_submissions = [
        {
            "id": "prior-idea-1",
            "miner_hotkey": "someone-else",
            "status": "accepted",
            "summary": "Earlier idea",
            "proposed_idea": "Try lower depth first.",
        }
    ]
    llm = FakeLLM(
        [
            {
                "claim_description": "Implement another miner's idea.",
                "summary": "Centerless submission.",
                "patch": "diff --git a/train.py b/train.py\n+centerless\n",
                "claimed_metrics": {"val_bpb": 1.84},
                "proposed_idea": "Now try a slower warmup schedule.",
            }
        ]
    )
    agent = _build_agent(coordinator, llm)

    result = agent.mine_once(participation_style=ParticipationStyle.direct)

    assert coordinator.claimed_tasks[0]["task_id"] == "task-centerless"
    assert result["submission"]["implemented_submission_id"] == "prior-idea-1"
    assert result["submission"]["proposed_idea"] == "Now try a slower warmup schedule."


def test_agent_peer_evaluate_once_reviews_pending_submission() -> None:
    coordinator = FakeCoordinator()
    coordinator.selected = type(
        "Selection",
        (),
        {
            "task": {
                "id": "task-peer",
                "slug": "nanogpt-default",
                "title": "nanoGPT",
                "base_ref": "main",
                "repository": "https://example.com/repo.git",
                "metric_name": "val_bpb",
                "metric_direction": "minimize",
                "competition_mode": CompetitionMode.peer_evaluation.value,
                "min_peer_evaluations": 2,
            },
            "work_item": None,
        },
    )()
    coordinator.task_submissions = [
        {
            "id": "pending-submission-1",
            "miner_hotkey": "another-miner",
            "status": "pending_verification",
            "summary": "Peer-review this",
            "patch": "diff --git a/train.py b/train.py\n+peer\n",
            "claimed_metrics": {"val_bpb": 1.88},
        }
    ]
    coordinator.detail_payload = {
        "submission": coordinator.task_submissions[0],
        "peer_evaluations": [],
        "peer_consensus": None,
    }
    llm = FakeLLM(
        [
            {
                "status": "accepted",
                "observed_metrics": {"val_bpb": 1.87},
                "notes": "Looks like a valid improvement.",
            }
        ]
    )
    agent = _build_agent(coordinator, llm)

    result = agent.peer_evaluate_once()

    assert coordinator.peer_reviews[0]["submission_id"] == "pending-submission-1"
    assert coordinator.peer_reviews[0]["status"] == "accepted"
    assert result["consensus"]["status"] == "accepted"


def test_agent_mine_once_retries_with_reduced_prompt_after_length_failure() -> None:
    coordinator = FakeCoordinator()
    coordinator.selected = type(
        "Selection",
        (),
        {
            "task": {
                "id": "task-standard",
                "slug": "bitnet-cpu-ternary-kernel",
                "title": "BitNet CPU",
                "base_ref": "main",
                "repository": "https://example.com/repo.git",
                "metric_name": "tokens_per_second",
                "metric_direction": "maximize",
                "competition_mode": CompetitionMode.standard.value,
                "time_budget_seconds": 900,
            },
            "work_item": {
                "id": "work-1",
                "title": "Establish baseline",
                "instructions": "Run one baseline replay.",
            },
        },
    )()
    llm = FakeLLM(
        [
            RuntimeError("chat completion returned no JSON payload (finish_reason=length, content_len=0, reasoning_len=0)"),
            {
                "claim_description": "Try a smaller prompt plan.",
                "summary": "Recovered after prompt compaction.",
                "patch": "diff --git a/README.md b/README.md\n+compact\n",
                "claimed_metrics": {"tokens_per_second": 12.5},
            },
        ]
    )
    agent = _build_agent(coordinator, llm)
    agent._load_repository_context = lambda **_: [  # type: ignore[method-assign]
        {"path": "README.md", "content": "x" * 12000},
        {"path": "requirements.txt", "content": "y" * 4000},
        {"path": "utils/e2e_benchmark.py", "content": "z" * 5000},
    ]
    coordinator.onboard = "# onboard\n" + ("notes\n" * 2000)

    result = agent.mine_once(participation_style=ParticipationStyle.pool)

    assert result["submission"]["claim_id"] == "claim-work-1"
    assert len(llm.prompts) == 2
    assert len(llm.prompts[1]["user_prompt"]) < len(llm.prompts[0]["user_prompt"])


def test_agent_mine_once_uses_minimal_prompt_after_repeated_length_failures() -> None:
    coordinator = FakeCoordinator()
    coordinator.selected = type(
        "Selection",
        (),
        {
            "task": {
                "id": "task-standard",
                "slug": "cifar10-matrix-decomposition-frontier",
                "title": "CIFAR-10",
                "base_ref": "master",
                "repository": "https://example.com/repo.git",
                "metric_name": "cifar10_top1",
                "metric_direction": "maximize",
                "competition_mode": CompetitionMode.standard.value,
                "time_budget_seconds": 1800,
            },
            "work_item": None,
        },
    )()
    llm = FakeLLM(
        [
            RuntimeError("chat completion returned no JSON payload (finish_reason=length, content_len=0, reasoning_len=0)"),
            RuntimeError("chat completion returned no JSON payload (finish_reason=length, content_len=0, reasoning_len=0)"),
            RuntimeError("chat completion returned no JSON payload (finish_reason=length, content_len=0, reasoning_len=0)"),
            {
                "claim_description": "Try a minimal CIFAR decomposition.",
                "summary": "Recovered after the micro prompt fallback.",
                "patch": "diff --git a/main.py b/main.py\n+compact\n",
                "claimed_metrics": {"cifar10_top1": 88.1},
            },
        ]
    )
    agent = _build_agent(coordinator, llm)
    agent._load_repository_context = lambda **_: [  # type: ignore[method-assign]
        {"path": "README.md", "content": "x" * 9000},
        {"path": "main.py", "content": "print('main')\n" * 200},
        {"path": "models/dla_simple.py", "content": "class SimpleDLA:\n    pass\n" * 120},
        {"path": "utils.py", "content": "def helper():\n    return 1\n" * 150},
    ]
    coordinator.onboard = "\n".join(
        [
            "# onboard",
            "Run python main.py",
            "Only submit runs with compression_ratio >= 4.0",
            "Emit final compressed_checkpoint_mb=<value>, compression_ratio=<value>, and cifar10_top1=<value> lines.",
            "Artifact: checkpoint/compressed_ckpt.pth",
        ]
    )

    result = agent.mine_once(participation_style=ParticipationStyle.pool)

    assert result["submission"]["claim_id"] == "claim-direct-1"
    assert len(llm.prompts) == 4
    assert len(llm.prompts[3]["user_prompt"]) < len(llm.prompts[2]["user_prompt"])
    assert "File main.py:" in llm.prompts[3]["user_prompt"]
    assert "8000/api/v1/tasks" not in llm.prompts[3]["user_prompt"]


def test_agent_mine_once_retries_when_patch_is_empty() -> None:
    coordinator = FakeCoordinator()
    coordinator.selected = type(
        "Selection",
        (),
        {
            "task": {
                "id": "task-standard",
                "slug": "bitnet-cpu-ternary-kernel",
                "title": "BitNet CPU",
                "base_ref": "main",
                "repository": "https://example.com/repo.git",
                "metric_name": "tokens_per_second",
                "metric_direction": "maximize",
                "competition_mode": CompetitionMode.standard.value,
                "time_budget_seconds": 900,
            },
            "work_item": {
                "id": "work-1",
                "title": "Establish baseline",
                "instructions": "Run one baseline replay.",
            },
        },
    )()
    llm = FakeLLM(
        [
            {
                "claim_description": "Initial attempt",
                "summary": "But forgot the patch.",
                "patch": "",
                "claimed_metrics": {"tokens_per_second": 11.0},
            },
            {
                "claim_description": "Retry attempt",
                "summary": "Includes a patch now.",
                "patch": "diff --git a/README.md b/README.md\n+retry\n",
                "claimed_metrics": {"tokens_per_second": 12.0},
            },
        ]
    )
    agent = _build_agent(coordinator, llm)

    result = agent.mine_once(participation_style=ParticipationStyle.pool)

    assert result["submission"]["claim_id"] == "claim-work-1"
    assert result["submission"]["patch"].startswith("diff --git")
    assert len(llm.prompts) == 2


def test_agent_mine_once_executes_patch_and_submits_observed_metric(tmp_path: Path) -> None:
    repo_dir, base_ref = _init_git_repo(
        tmp_path,
        {
            "README.md": "demo\n",
            "train.py": "print('val_bpb: 2.5')\n",
        },
    )
    patch = _build_patch(repo_dir, "train.py", "print('val_bpb: 1.23')\n")
    coordinator = FakeCoordinator()
    coordinator.selected = type(
        "Selection",
        (),
        {
            "task": {
                "id": "task-standard",
                "slug": "nanogpt-default",
                "title": "nanoGPT",
                "base_ref": base_ref,
                "repository": str(repo_dir),
                "metric_name": "val_bpb",
                "metric_direction": "minimize",
                "competition_mode": CompetitionMode.standard.value,
                "benchmark_command": "python3 train.py",
                "time_budget_seconds": 30,
            },
            "work_item": {
                "id": "work-1",
                "title": "Run a baseline",
                "instructions": "Modify train.py and report val_bpb.",
            },
        },
    )()
    llm = FakeLLM(
        [
            {
                "claim_description": "Try a lower reported val_bpb.",
                "summary": "Switch the demo output.",
                "patch": patch,
                "claimed_metrics": {"val_bpb": 9.99},
            }
        ]
    )
    agent = ResearchAgentMiner(
        coordinator=coordinator,  # type: ignore[arg-type]
        llm=llm,  # type: ignore[arg-type]
        config=AgentMinerConfig(
            participation_style=ParticipationStyle.pool,
            workspace_root=tmp_path / "workspace",
            execution_enabled=True,
            execution_max_attempts=1,
        ),
    )
    agent._load_repository_context = lambda **_: [  # type: ignore[method-assign]
        {"path": "train.py", "content": "print('val_bpb: 2.5')\n"}
    ]

    result = agent.mine_once(participation_style=ParticipationStyle.pool)

    submission = coordinator.submissions[0]
    expected_base_commit = (
        subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_dir, check=True, capture_output=True, text=True)
        .stdout.strip()
    )
    assert submission["claim_id"] == "claim-work-1"
    assert submission["base_ref"] == expected_base_commit
    assert submission["claimed_metrics"]["val_bpb"] == 1.23
    assert "Observed locally: val_bpb=1.23." in submission["summary"]
    assert "benchmark_exit=0" in submission["execution_log"]
    assert Path(str(submission["artifact_uri"])).exists()
    assert result["execution"]["observed_metrics"]["val_bpb"] == 1.23


def test_submit_claimed_workspace_uses_submission_json_and_git_diff(tmp_path: Path) -> None:
    repo_dir, base_ref = _init_git_repo(
        tmp_path,
        {
            "README.md": "demo\n",
            "train.py": "print('val_bpb: 2.5')\n",
        },
    )
    (repo_dir / "train.py").write_text("print('val_bpb: 1.11')\n", encoding="utf-8")
    submission_file = tmp_path / "submission.json"
    submission_file.write_text(
        json.dumps(
            {
                "summary": "External agent changed the training script.",
                "claimed_metrics": {"val_bpb": 1.11},
                "notes": "managed by external cli",
            }
        ),
        encoding="utf-8",
    )
    coordinator = FakeCoordinator()

    result = submit_claimed_workspace(
        coordinator=coordinator,  # type: ignore[arg-type]
        claim_id="claim-external-1",
        repo_dir=repo_dir,
        submission_file=submission_file,
        default_base_ref=base_ref,
    )
    head_commit = (
        subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_dir, check=True, capture_output=True, text=True)
        .stdout.strip()
    )

    assert result["id"] == "submission-1"
    assert coordinator.submissions[0]["claim_id"] == "claim-external-1"
    assert coordinator.submissions[0]["base_ref"] == head_commit
    assert coordinator.submissions[0]["summary"] == "External agent changed the training script."
    assert coordinator.submissions[0]["claimed_metrics"]["val_bpb"] == 1.11
    assert coordinator.submissions[0]["patch"].startswith("diff --git a/train.py b/train.py")


def test_agent_mine_once_can_run_external_gui_managed_agent(tmp_path: Path) -> None:
    repo_dir, base_ref = _init_git_repo(
        tmp_path,
        {
            "README.md": "demo\n",
            "train.py": "print('val_bpb: 2.5')\n",
        },
    )
    coordinator = FakeCoordinator()
    coordinator.selected = type(
        "Selection",
        (),
        {
            "task": {
                "id": "task-standard",
                "slug": "nanogpt-default",
                "title": "nanoGPT",
                "base_ref": base_ref,
                "repository": str(repo_dir),
                "metric_name": "val_bpb",
                "metric_direction": "minimize",
                "competition_mode": CompetitionMode.standard.value,
                "benchmark_command": "python3 train.py",
                "time_budget_seconds": 30,
            },
            "work_item": {
                "id": "work-1",
                "title": "Run a baseline",
                "instructions": "Modify train.py and report val_bpb.",
            },
        },
    )()
    coordinator.onboard = "# onboard\nEdit train.py and report val_bpb.\n"
    runner = (
        "import json, os, pathlib, sys\n"
        "workspace = pathlib.Path(os.environ['BITSOTA_AGENT_WORKSPACE'])\n"
        "repo = pathlib.Path(os.environ['BITSOTA_AGENT_REPO_DIR'])\n"
        "intro = workspace / 'INTRO_GUI.md'\n"
        "assert 'Do not submit directly to the coordinator' in intro.read_text(encoding='utf-8')\n"
        "print('external agent started')\n"
        "print('external agent stderr', file=sys.stderr)\n"
        "(repo / 'train.py').write_text(\"print('val_bpb: 1.01')\\n\", encoding='utf-8')\n"
        "(workspace / 'submission.json').write_text(json.dumps({"
        "\"summary\": \"Codex-style external agent submission.\","
        "\"claimed_metrics\": {\"val_bpb\": 1.01}"
        "}), encoding='utf-8')\n"
    )
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(runner)}"
    agent = ResearchAgentMiner(
        coordinator=coordinator,  # type: ignore[arg-type]
        llm=None,  # type: ignore[arg-type]
        config=AgentMinerConfig(
            participation_style=ParticipationStyle.pool,
            workspace_root=tmp_path / "workspace",
            execution_enabled=False,
            external_agent_command=command,
            external_agent_mode="gui_managed",
        ),
    )

    result = agent.mine_once(participation_style=ParticipationStyle.pool)

    submission = coordinator.submissions[0]
    assert submission["claim_id"] == "claim-work-1"
    assert submission["summary"] == "Codex-style external agent submission."
    assert submission["claimed_metrics"]["val_bpb"] == 1.01
    assert submission["patch"].startswith("diff --git a/train.py b/train.py")
    workspace_dir = Path(result["execution"]["workspace_dir"])
    assert (workspace_dir / "INTRO_GUI.md").exists()
    assert (workspace_dir / "submission.json").exists()
    assert (workspace_dir / "agent.stdout.txt").read_text(encoding="utf-8") == "external agent started\n"
    assert (workspace_dir / "agent.stderr.txt").read_text(encoding="utf-8") == "external agent stderr\n"


def test_agent_mine_once_external_gui_managed_centerless_auto_fills_prior_idea_reference(
    tmp_path: Path,
) -> None:
    repo_dir, base_ref = _init_git_repo(
        tmp_path,
        {
            "README.md": "demo\n",
            "train.py": "print('val_bpb: 2.5')\n",
        },
    )
    coordinator = FakeCoordinator()
    coordinator.selected = type(
        "Selection",
        (),
        {
            "task": {
                "id": "task-centerless",
                "slug": "nanogpt-default",
                "title": "nanoGPT",
                "base_ref": base_ref,
                "repository": str(repo_dir),
                "metric_name": "val_bpb",
                "metric_direction": "minimize",
                "competition_mode": CompetitionMode.centerless.value,
                "benchmark_command": "python3 train.py",
                "time_budget_seconds": 30,
            },
            "work_item": {
                "id": "work-1",
                "title": "Implement the prior idea",
                "instructions": "Modify train.py and report val_bpb.",
            },
        },
    )()
    coordinator.task_submissions = [
        {
            "id": "prior-idea-1",
            "miner_hotkey": "another-miner",
            "status": "accepted",
            "summary": "Earlier idea",
            "proposed_idea": "Try the lower depth first.",
        }
    ]
    coordinator.onboard = "# onboard\nEdit train.py and report val_bpb.\n"
    runner = (
        "import json, os, pathlib\n"
        "workspace = pathlib.Path(os.environ['BITSOTA_AGENT_WORKSPACE'])\n"
        "repo = pathlib.Path(os.environ['BITSOTA_AGENT_REPO_DIR'])\n"
        "(repo / 'train.py').write_text(\"print('val_bpb: 1.01')\\n\", encoding='utf-8')\n"
        "(workspace / 'submission.json').write_text(json.dumps({"
        "\"summary\": \"Centerless external agent submission.\","
        "\"claimed_metrics\": {\"val_bpb\": 1.01},"
        "\"proposed_idea\": \"Try 0.99 next.\""
        "}), encoding='utf-8')\n"
    )
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(runner)}"
    agent = ResearchAgentMiner(
        coordinator=coordinator,  # type: ignore[arg-type]
        llm=None,  # type: ignore[arg-type]
        config=AgentMinerConfig(
            participation_style=ParticipationStyle.pool,
            workspace_root=tmp_path / "workspace",
            execution_enabled=False,
            external_agent_command=command,
            external_agent_mode="gui_managed",
        ),
    )

    result = agent.mine_once(participation_style=ParticipationStyle.pool)

    submission = coordinator.submissions[0]
    assert submission["claim_id"] == "claim-work-1"
    assert submission["implemented_submission_id"] == "prior-idea-1"
    assert submission["proposed_idea"] == "Try 0.99 next."
    assert result["submission"]["implemented_submission_id"] == "prior-idea-1"


def test_agent_mine_once_cancels_claim_after_local_execution_failure(tmp_path: Path) -> None:
    repo_dir, base_ref = _init_git_repo(
        tmp_path,
        {
            "README.md": "demo\n",
            "train.py": "print('val_bpb: 2.5')\n",
        },
    )
    coordinator = FakeCoordinator()
    coordinator.selected = type(
        "Selection",
        (),
        {
            "task": {
                "id": "task-standard",
                "slug": "nanogpt-default",
                "title": "nanoGPT",
                "base_ref": base_ref,
                "repository": str(repo_dir),
                "metric_name": "val_bpb",
                "metric_direction": "minimize",
                "competition_mode": CompetitionMode.standard.value,
                "benchmark_command": "python3 train.py",
                "time_budget_seconds": 30,
            },
            "work_item": {
                "id": "work-1",
                "title": "Run a baseline",
                "instructions": "Modify train.py and report val_bpb.",
            },
        },
    )()
    llm = FakeLLM(
        [
            {
                "claim_description": "Broken patch",
                "summary": "This should fail before benchmark.",
                "patch": "diff --git a/train.py b/train.py\n+broken\n",
                "claimed_metrics": {"val_bpb": 9.99},
            }
        ]
    )
    agent = ResearchAgentMiner(
        coordinator=coordinator,  # type: ignore[arg-type]
        llm=llm,  # type: ignore[arg-type]
        config=AgentMinerConfig(
            participation_style=ParticipationStyle.pool,
            workspace_root=tmp_path / "workspace",
            execution_enabled=True,
            execution_max_attempts=1,
        ),
    )
    agent._load_repository_context = lambda **_: [  # type: ignore[method-assign]
        {"path": "train.py", "content": "print('val_bpb: 2.5')\n"}
    ]

    try:
        agent.mine_once(participation_style=ParticipationStyle.pool)
        assert False, "expected mine_once to raise on local execution failure"
    except RuntimeError as exc:
        assert "patch apply failed" in str(exc)

    assert coordinator.cancelled_claims == ["claim-work-1"]
    assert coordinator.submissions == []


def test_openai_client_retries_reasoning_only_response_with_larger_budget() -> None:
    session = RecordingSession(
        [
            FakeResponse(
                200,
                {
                    "choices": [
                        {
                            "finish_reason": "length",
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "reasoning": "Used all tokens thinking.",
                            },
                        }
                    ]
                },
            ),
            FakeResponse(
                200,
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": '{"ok": true}',
                            },
                        }
                    ]
                },
            ),
        ]
    )
    client = OpenAICompatibleChatClient(
        base_url="https://openrouter.ai/api/v1",
        model="deepseek/deepseek-v3.2",
        api_key="test-key",
        session=session,  # type: ignore[arg-type]
    )

    result = client.complete_json(
        system_prompt="Return JSON only.",
        user_prompt='{"task":"demo"}',
        temperature=0.2,
        max_tokens=1800,
    )

    assert result == {"ok": True}
    assert len(session.requests) == 2
    assert session.requests[0]["json"]["reasoning"] == {"effort": "minimal", "exclude": True}
    assert session.requests[0]["json"]["response_format"] == {"type": "json_object"}
    assert session.requests[1]["json"]["max_tokens"] > session.requests[0]["json"]["max_tokens"]


def test_openai_client_uses_large_budget_floor_for_stepfun_openrouter() -> None:
    session = RecordingSession(
        [
            FakeResponse(
                200,
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": '{"ok": true}',
                            },
                        }
                    ]
                },
            )
        ]
    )
    client = OpenAICompatibleChatClient(
        base_url="https://openrouter.ai/api/v1",
        model="stepfun/step-3.5-flash:free",
        api_key="test-key",
        session=session,  # type: ignore[arg-type]
    )

    result = client.complete_json(
        system_prompt="Return JSON only.",
        user_prompt='{"task":"demo"}',
        temperature=0.2,
        max_tokens=1800,
    )

    assert result == {"ok": True}
    assert len(session.requests) == 1
    assert session.requests[0]["json"]["max_tokens"] == 8192
    assert session.requests[0]["json"]["reasoning"] == {"max_tokens": 1024, "exclude": True}


def test_openai_client_can_extract_json_from_reasoning_text() -> None:
    session = RecordingSession(
        [
            FakeResponse(
                200,
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "reasoning": 'Thinking...\n{"status": "accepted", "observed_metrics": {"val_bpb": 1.87}}',
                            },
                        }
                    ]
                },
            )
        ]
    )
    client = OpenAICompatibleChatClient(
        base_url="https://openrouter.ai/api/v1",
        model="stepfun/step-3.5-flash:free",
        api_key="test-key",
        session=session,  # type: ignore[arg-type]
    )

    result = client.complete_json(
        system_prompt="Return JSON only.",
        user_prompt='{"submission":"demo"}',
        temperature=0.2,
        max_tokens=1800,
    )

    assert result["status"] == "accepted"
    assert result["observed_metrics"]["val_bpb"] == 1.87
