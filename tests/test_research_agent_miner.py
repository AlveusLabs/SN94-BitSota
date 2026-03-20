from __future__ import annotations

from pathlib import Path

from substrateinterface import Keypair

from miner.research_agent import AgentMinerConfig, ResearchAgentMiner
from miner.research_competitions import CompetitionMode, ParticipationStyle, list_builtin_research_competitions


class FakeCoordinator:
    def __init__(self) -> None:
        self.hotkey = Keypair.create_from_mnemonic(Keypair.generate_mnemonic()).ss58_address
        self.claimed_tasks: list[dict] = []
        self.claimed_work_items: list[dict] = []
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
        return dict(self.responses.pop(0))


def _build_agent(coordinator: FakeCoordinator, llm: FakeLLM) -> ResearchAgentMiner:
    agent = ResearchAgentMiner(
        coordinator=coordinator,  # type: ignore[arg-type]
        llm=llm,  # type: ignore[arg-type]
        config=AgentMinerConfig(
            participation_style=ParticipationStyle.direct,
            workspace_root=Path(".tmp-agent-tests"),
        ),
    )
    agent._load_repository_context = lambda **_: [  # type: ignore[method-assign]
        {"path": "README.md", "content": "example context"}
    ]
    return agent


def test_builtin_research_catalog_exposes_five_competitions() -> None:
    templates = list_builtin_research_competitions()
    assert len(templates) == 5
    assert templates[0].slug == "nanogpt-default"


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
