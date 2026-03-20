from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any, Callable, Optional
from urllib.parse import urlparse

import requests

from miner.research_competitions import CompetitionMode, ParticipationStyle
from miner.research_coordinator_client import CoordinatorClient, CoordinatorSelection


_FILE_PATTERN = re.compile(r"([A-Za-z0-9_./-]+\.(?:md|txt|py|toml|json|ya?ml|sh))")


def _noop_log(_: str) -> None:
    return


def _coerce_claimed_metrics(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def _strip_code_fences(text: str) -> str:
    raw = str(text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", raw, count=1)
        raw = re.sub(r"\s*```$", "", raw, count=1)
    return raw.strip()


@dataclass(slots=True)
class AgentMinerConfig:
    participation_style: ParticipationStyle = ParticipationStyle.direct
    workspace_root: Path = Path(".bitsota_agent_workspace")
    max_files: int = 6
    max_file_chars: int = 12000
    llm_temperature: float = 0.2
    llm_max_tokens: int = 1800


class OpenAICompatibleChatClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_s: float = 60.0,
        session: requests.Session | None = None,
    ) -> None:
        self.api_url = self._normalize_api_url(base_url)
        self.model = str(model or "").strip()
        if not self.model:
            raise ValueError("model is required")
        self.api_key = str(api_key or "").strip()
        self.timeout_s = max(1.0, float(timeout_s))
        self.session = session or requests.Session()

    @staticmethod
    def _normalize_api_url(base_url: str) -> str:
        raw = str(base_url or "").strip().rstrip("/")
        if not raw:
            raise ValueError("base_url is required")
        parsed = urlparse(raw)
        path = parsed.path.rstrip("/")
        if path.endswith("/chat/completions"):
            return raw
        if not path:
            return f"{raw}/v1/chat/completions"
        return f"{raw}/chat/completions"

    @staticmethod
    def _message_text(message: Any) -> str:
        if isinstance(message, str):
            return message
        if isinstance(message, list):
            chunks: list[str] = []
            for item in message:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text" and item.get("text") is not None:
                    chunks.append(str(item.get("text")))
            return "\n".join(chunks).strip()
        return str(message or "")

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": str(system_prompt)},
                {"role": "user", "content": str(user_prompt)},
            ],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }
        response = self.session.post(self.api_url, json=payload, headers=headers, timeout=self.timeout_s)
        if response.status_code >= 400:
            raise RuntimeError(f"chat completion failed: HTTP {response.status_code} ({response.text})")
        data = response.json() or {}
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("chat completion returned no choices")
        message = choices[0].get("message") or {}
        content = self._message_text(message.get("content"))
        if not content:
            raise RuntimeError("chat completion returned empty content")
        cleaned = _strip_code_fences(content)
        try:
            return dict(json.loads(cleaned))
        except Exception as exc:
            raise RuntimeError(f"chat completion did not return valid JSON: {cleaned[:300]}") from exc


class ResearchAgentMiner:
    def __init__(
        self,
        *,
        coordinator: CoordinatorClient,
        llm: OpenAICompatibleChatClient,
        config: AgentMinerConfig | None = None,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.coordinator = coordinator
        self.llm = llm
        self.config = config or AgentMinerConfig()
        self.log = log or _noop_log

    @property
    def hotkey(self) -> str:
        return self.coordinator.hotkey

    def mine_once(
        self,
        *,
        task_id: str | None = None,
        task_slug: str | None = None,
        participation_style: ParticipationStyle | None = None,
    ) -> dict[str, Any]:
        style = participation_style or self.config.participation_style
        selection = self.coordinator.select_task(
            task_id=task_id,
            task_slug=task_slug,
            participation_style=str(style.value),
        )
        task = selection.task
        work_item = selection.work_item
        onboard = self.coordinator.get_onboard_markdown(str(task["id"]))
        submissions = self.coordinator.list_submissions(task_id=str(task["id"]))
        repo_context = self._load_repository_context(task=task, onboard=onboard)
        idea_candidates = self._idea_candidates(task=task, submissions=submissions)
        plan = self._generate_submission_plan(
            task=task,
            work_item=work_item,
            onboard=onboard,
            repo_context=repo_context,
            idea_candidates=idea_candidates,
            participation_style=style,
        )

        claim_description = str(plan.get("claim_description") or "").strip()
        if not claim_description:
            claim_description = (
                str(work_item.get("title"))
                if work_item is not None
                else f"Agentic research pass for {task.get('slug') or task.get('title')}"
            )

        if work_item is not None:
            claim = self.coordinator.claim_work_item(
                work_item_id=str(work_item["id"]),
                claim_description=claim_description,
            )
        else:
            claim = self.coordinator.claim_task(
                task_id=str(task["id"]),
                claim_description=claim_description,
            )

        implemented_submission_id = str(plan.get("implemented_submission_id") or "").strip() or None
        mode = str(task.get("competition_mode") or CompetitionMode.standard.value)
        if mode == CompetitionMode.centerless.value and idea_candidates and not implemented_submission_id:
            implemented_submission_id = str(idea_candidates[0]["id"])

        submission = self.coordinator.submit_submission(
            claim_id=str(claim["id"]),
            base_ref=str(plan.get("base_ref") or task.get("base_ref") or ""),
            patch=str(plan.get("patch") or ""),
            summary=str(plan.get("summary") or ""),
            claimed_metrics=_coerce_claimed_metrics(plan.get("claimed_metrics")),
            proposed_idea=(
                str(plan.get("proposed_idea")).strip()
                if plan.get("proposed_idea") is not None
                else None
            ),
            implemented_submission_id=implemented_submission_id,
        )
        self.log(
            f"[research-agent] submitted task={task.get('slug')} mode={mode} "
            f"style={style.value} submission_id={submission.get('id')}"
        )
        return {
            "task": task,
            "work_item": work_item,
            "claim": claim,
            "submission": submission,
            "plan": plan,
        }

    def peer_evaluate_once(
        self,
        *,
        task_id: str | None = None,
        task_slug: str | None = None,
    ) -> dict[str, Any]:
        selection = self.coordinator.select_task(
            task_id=task_id,
            task_slug=task_slug,
            participation_style=ParticipationStyle.direct.value,
        )
        task = selection.task
        if str(task.get("competition_mode")) != CompetitionMode.peer_evaluation.value:
            raise RuntimeError("peer_evaluate_once requires a peer_evaluation task")
        onboard = self.coordinator.get_onboard_markdown(str(task["id"]))
        submissions = self.coordinator.list_submissions(task_id=str(task["id"]), status="pending_verification")
        target = None
        for submission in submissions:
            if str(submission.get("miner_hotkey")) != self.hotkey:
                target = submission
                break
        if target is None:
            raise RuntimeError("no pending peer-evaluation submissions from other miners were found")
        detail = self.coordinator.get_submission_detail(str(target["id"]))
        review = self._generate_peer_evaluation(
            task=task,
            submission=target,
            detail=detail,
            onboard=onboard,
        )
        observed_metrics = _coerce_claimed_metrics(review.get("observed_metrics"))
        if str(review.get("status")) == "accepted" and task.get("metric_name") not in observed_metrics:
            claimed = target.get("claimed_metrics") or {}
            metric_name = str(task.get("metric_name") or "")
            if metric_name and metric_name in claimed:
                observed_metrics[metric_name] = claimed[metric_name]
        peer_evaluation = self.coordinator.peer_evaluate_submission(
            submission_id=str(target["id"]),
            status=str(review.get("status") or "rejected"),
            observed_metrics=observed_metrics,
            notes=str(review.get("notes") or ""),
        )
        consensus = self.coordinator.get_peer_consensus(str(target["id"]))
        self.log(
            f"[research-agent] peer-evaluated task={task.get('slug')} submission_id={target.get('id')} "
            f"status={peer_evaluation.get('status')} consensus={consensus.get('status')}"
        )
        return {
            "task": task,
            "submission": target,
            "detail": detail,
            "review": review,
            "peer_evaluation": peer_evaluation,
            "consensus": consensus,
        }

    def _generate_submission_plan(
        self,
        *,
        task: dict[str, Any],
        work_item: dict[str, Any] | None,
        onboard: str,
        repo_context: list[dict[str, str]],
        idea_candidates: list[dict[str, Any]],
        participation_style: ParticipationStyle,
    ) -> dict[str, Any]:
        mode = str(task.get("competition_mode") or CompetitionMode.standard.value)
        system_prompt = (
            "You are a BitSota research miner. Return JSON only. "
            "Propose one concrete submission for this coordinator task. "
            "The JSON must contain claim_description, summary, patch, claimed_metrics, and optional "
            "proposed_idea plus implemented_submission_id. "
            "patch must be a unified diff string. "
            "Do not wrap JSON in markdown."
        )
        prompt = {
            "participation_style": str(participation_style.value),
            "task": {
                "id": task.get("id"),
                "slug": task.get("slug"),
                "title": task.get("title"),
                "repository": task.get("repository"),
                "base_ref": task.get("base_ref"),
                "metric_name": task.get("metric_name"),
                "metric_direction": task.get("metric_direction"),
                "competition_mode": mode,
                "time_budget_seconds": task.get("time_budget_seconds"),
            },
            "work_item": (
                {
                    "id": work_item.get("id"),
                    "title": work_item.get("title"),
                    "instructions": work_item.get("instructions"),
                    "base_submission_id": work_item.get("base_submission_id"),
                }
                if work_item is not None
                else None
            ),
            "mode_rules": self._mode_rules(task=task, has_ideas=bool(idea_candidates)),
            "idea_candidates": idea_candidates[:5],
            "onboard_markdown": onboard,
            "repository_files": repo_context,
            "response_schema": {
                "claim_description": "string",
                "summary": "string",
                "patch": "string unified diff",
                "claimed_metrics": {"metric_name": "number"},
                "proposed_idea": "string or null",
                "implemented_submission_id": "string or null",
                "base_ref": "optional string",
            },
        }
        result = self.llm.complete_json(
            system_prompt=system_prompt,
            user_prompt=json.dumps(prompt, indent=2),
            temperature=self.config.llm_temperature,
            max_tokens=self.config.llm_max_tokens,
        )
        return dict(result)

    def _generate_peer_evaluation(
        self,
        *,
        task: dict[str, Any],
        submission: dict[str, Any],
        detail: dict[str, Any],
        onboard: str,
    ) -> dict[str, Any]:
        system_prompt = (
            "You are a BitSota peer evaluator for a research coordinator. "
            "Return JSON only with status, observed_metrics, and notes. "
            "status must be accepted, rejected, or error. "
            "If status is accepted, observed_metrics must include the task metric. "
            "Do not wrap JSON in markdown."
        )
        prompt = {
            "task": {
                "id": task.get("id"),
                "slug": task.get("slug"),
                "title": task.get("title"),
                "metric_name": task.get("metric_name"),
                "metric_direction": task.get("metric_direction"),
                "competition_mode": task.get("competition_mode"),
                "min_peer_evaluations": task.get("min_peer_evaluations"),
            },
            "submission": submission,
            "submission_detail": detail,
            "onboard_markdown": onboard,
            "response_schema": {
                "status": "accepted|rejected|error",
                "observed_metrics": {"metric_name": "number"},
                "notes": "string",
            },
        }
        result = self.llm.complete_json(
            system_prompt=system_prompt,
            user_prompt=json.dumps(prompt, indent=2),
            temperature=self.config.llm_temperature,
            max_tokens=self.config.llm_max_tokens,
        )
        return dict(result)

    def _mode_rules(self, *, task: dict[str, Any], has_ideas: bool) -> list[str]:
        mode = str(task.get("competition_mode") or CompetitionMode.standard.value)
        if mode == CompetitionMode.centerless.value:
            rules = [
                "Every submission must include proposed_idea.",
            ]
            if has_ideas:
                rules.append(
                    "Because other miners already proposed ideas, implemented_submission_id should reference "
                    "another miner's prior idea-bearing submission."
                )
            return rules
        if mode == CompetitionMode.peer_evaluation.value:
            return [
                "This task finalizes through peer evaluation rather than validator verification.",
                "A miner submission still needs patch, summary, and claimed metrics.",
            ]
        return ["Standard mode requires a normal submission with patch, summary, and claimed metrics."]

    def _idea_candidates(
        self,
        *,
        task: dict[str, Any],
        submissions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if str(task.get("competition_mode")) != CompetitionMode.centerless.value:
            return []
        candidates: list[dict[str, Any]] = []
        for submission in submissions:
            if str(submission.get("miner_hotkey")) == self.hotkey:
                continue
            idea = str(submission.get("proposed_idea") or "").strip()
            if not idea:
                continue
            candidates.append(
                {
                    "id": submission.get("id"),
                    "miner_hotkey": submission.get("miner_hotkey"),
                    "status": submission.get("status"),
                    "summary": submission.get("summary"),
                    "proposed_idea": idea,
                }
            )
        accepted = [row for row in candidates if str(row.get("status")) == "accepted"]
        pending = [row for row in candidates if str(row.get("status")) != "accepted"]
        return accepted + pending

    def _load_repository_context(
        self,
        *,
        task: dict[str, Any],
        onboard: str,
    ) -> list[dict[str, str]]:
        repo_url = str(task.get("repository") or "").strip()
        base_ref = str(task.get("base_ref") or "").strip()
        if not repo_url or not base_ref:
            return []

        workspace_root = Path(self.config.workspace_root).expanduser().resolve()
        workspace_root.mkdir(parents=True, exist_ok=True)
        temp_root = Path(tempfile.mkdtemp(prefix=f"research-agent-{task.get('slug', 'task')}-", dir=str(workspace_root)))
        repo_dir = temp_root / "repo"
        try:
            subprocess.run(
                ["git", "clone", "--quiet", repo_url, str(repo_dir)],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                ["git", "checkout", base_ref],
                cwd=str(repo_dir),
                check=True,
                capture_output=True,
                text=True,
            )
            file_candidates = self._infer_file_candidates(onboard)
            contexts: list[dict[str, str]] = []
            for rel_path in file_candidates:
                if len(contexts) >= int(self.config.max_files):
                    break
                candidate = (repo_dir / rel_path).resolve()
                try:
                    candidate.relative_to(repo_dir.resolve())
                except Exception:
                    continue
                if not candidate.exists() or not candidate.is_file():
                    continue
                try:
                    content = candidate.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
                contexts.append(
                    {
                        "path": rel_path,
                        "content": content[: int(self.config.max_file_chars)],
                    }
                )
            return contexts
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

    def _infer_file_candidates(self, onboard: str) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for match in _FILE_PATTERN.finditer(str(onboard or "")):
            candidate = str(match.group(1)).strip()
            key = candidate.lower()
            if key in seen:
                continue
            seen.add(key)
            ordered.append(candidate)
        if "readme.md" not in seen:
            ordered.insert(0, "README.md")
        return ordered
