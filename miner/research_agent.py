from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile
from typing import Any, Callable, Optional
from urllib.parse import urlparse

import requests

from miner.research_competitions import CompetitionMode, ParticipationStyle
from miner.research_coordinator_client import CoordinatorClient, CoordinatorSelection


_FILE_PATTERN = re.compile(
    r"([A-Za-z0-9_./-]+\.(?:md|txt|py|toml|json|ya?ml|sh|cpp|cxx|cc|cuh|cu|hpp|hxx|hh|h|c|rs|go))(?![A-Za-z0-9])"
)
_NUMBER_PATTERN = r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"


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


def parse_metric_from_text(text: str, metric_name: str) -> float | None:
    pattern = re.compile(rf"(?im)\b{re.escape(metric_name)}\b\s*[:=]\s*{_NUMBER_PATTERN}")
    match = pattern.search(str(text))
    if match is None:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _safe_name_fragment(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "").strip()).strip("-")
    return cleaned or "task"


def _compact_json_dumps(payload: Any) -> str:
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


def _normalized_submission_mode(value: str | None) -> str:
    raw = str(value or "").strip().lower()
    if raw == "autonomous":
        return "autonomous"
    return "gui_managed"


def build_agent_intro_markdown(
    *,
    task: dict[str, Any],
    work_item: dict[str, Any] | None,
    claim: dict[str, Any],
    onboard: str,
    coordinator_url: str,
    mode: str,
    repo_dir: Path,
    workspace_dir: Path,
    submission_file: Path,
    submission_result_file: Path,
) -> str:
    normalized_mode = _normalized_submission_mode(mode)
    intro_name = "INTRO.md" if normalized_mode == "autonomous" else "INTRO_GUI.md"
    lines = [
        f"# {intro_name}",
        "",
        f"Task: {task.get('title') or task.get('slug')}",
        f"Task slug: {task.get('slug')}",
        f"Task id: {task.get('id')}",
        f"Claim id: {claim.get('id')}",
        f"Coordinator URL: {coordinator_url}",
        f"Repository checkout: {repo_dir}",
        f"Workspace: {workspace_dir}",
        f"Submission sidecar path: {submission_file}",
        f"Submission result path: {submission_result_file}",
        "",
        "## Task Onboarding",
        "",
        str(onboard or "").strip() or "(no onboarding text provided)",
        "",
    ]
    if work_item is not None:
        lines.extend(
            [
                "## Work Item",
                "",
                f"- id: {work_item.get('id')}",
                f"- title: {work_item.get('title')}",
                f"- instructions: {work_item.get('instructions')}",
                "",
            ]
        )
    lines.extend(
        [
            "## Workspace Contract",
            "",
            "- Edit the checked-out repository files directly.",
            f"- Write a JSON sidecar to `{submission_file.name}` in the workspace root.",
            "- Required sidecar fields: `summary`, `claimed_metrics`.",
            "- Optional sidecar fields: `base_ref`, `proposed_idea`, `implemented_submission_id`, `artifact_uri`, `execution_log`, `notes`.",
            "- Do not put the patch in the sidecar. The launcher derives the patch from `git diff` in the repo checkout.",
            "",
        ]
    )
    if normalized_mode == "autonomous":
        lines.extend(
            [
                "## Submission Authority",
                "",
                "You are allowed to submit directly to the coordinator for this claim if wallet access is available.",
                "Use the local helper CLI once the repo changes and sidecar are ready:",
                "",
                "```bash",
                "bitsota-research-agent submit-workspace \\",
                f"  --coordinator-url {coordinator_url} \\",
                f"  --claim-id {claim.get('id')} \\",
                f"  --repo-dir {repo_dir} \\",
                f"  --submission-file {submission_file}",
                "```",
                "",
                "If you use the helper, it will write the coordinator response to `submission_result.json`.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Submission Authority",
                "",
                "Do not submit directly to the coordinator. The launcher owns signing and submission for this run.",
                "Your job is to edit the repo and write `submission.json` only.",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def load_submission_sidecar(
    *,
    submission_file: Path,
    metric_name: str = "",
) -> dict[str, Any]:
    try:
        payload = json.loads(submission_file.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"submission sidecar is missing: {submission_file}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"submission sidecar is not valid JSON: {submission_file}") from exc

    if not isinstance(payload, dict):
        raise RuntimeError("submission sidecar must be a JSON object")
    summary = str(payload.get("summary") or "").strip()
    claimed_metrics = _coerce_claimed_metrics(payload.get("claimed_metrics"))
    if not summary:
        raise RuntimeError("submission sidecar is missing required field: summary")
    if not claimed_metrics:
        raise RuntimeError("submission sidecar is missing required field: claimed_metrics")
    if metric_name and metric_name not in claimed_metrics:
        raise RuntimeError(f"submission sidecar is missing required metric: {metric_name}")
    result = dict(payload)
    result["summary"] = summary
    result["claimed_metrics"] = claimed_metrics
    return result


def compute_repo_patch(repo_dir: Path) -> str:
    subprocess.run(
        ["git", "add", "-N", "."],
        cwd=str(repo_dir),
        check=False,
        capture_output=True,
        text=True,
    )
    result = subprocess.run(
        ["git", "diff", "--binary", "--no-ext-diff"],
        cwd=str(repo_dir),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git diff failed with exit={result.returncode}: {result.stderr}")
    patch = str(result.stdout or "")
    if not patch.strip():
        raise RuntimeError("repo checkout has no patch to submit")
    return patch


def resolve_repo_head_commit(repo_dir: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo_dir),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git rev-parse HEAD failed with exit={result.returncode}: {result.stderr}")
    commit_sha = str(result.stdout or "").strip()
    if not re.fullmatch(r"[0-9a-fA-F]{40}", commit_sha):
        raise RuntimeError("repo checkout HEAD did not resolve to a full commit SHA")
    return commit_sha


def submit_claimed_workspace(
    *,
    coordinator: Any,
    claim_id: str,
    repo_dir: Path,
    submission_file: Path,
    default_base_ref: str,
    competition_mode: str | None = None,
    idea_candidates: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    payload = load_submission_sidecar(
        submission_file=Path(submission_file),
        metric_name="",
    )
    pinned_base_ref = resolve_repo_head_commit(Path(repo_dir))
    patch = compute_repo_patch(Path(repo_dir))
    summary = str(payload.get("summary") or "").strip()
    claimed_metrics = _coerce_claimed_metrics(payload.get("claimed_metrics"))
    implemented_submission_id = (
        str(payload.get("implemented_submission_id")).strip()
        if payload.get("implemented_submission_id") is not None
        else None
    )
    if (
        str(competition_mode or CompetitionMode.standard.value) == CompetitionMode.centerless.value
        and idea_candidates
        and not implemented_submission_id
    ):
        implemented_submission_id = str(idea_candidates[0]["id"])
    return dict(
        coordinator.submit_submission(
            claim_id=str(claim_id),
            base_ref=pinned_base_ref,
            patch=patch,
            summary=summary,
            claimed_metrics=claimed_metrics,
            proposed_idea=(
                str(payload.get("proposed_idea")).strip()
                if payload.get("proposed_idea") is not None
                else None
            ),
            implemented_submission_id=implemented_submission_id,
            artifact_uri=(
                str(payload.get("artifact_uri")).strip()
                if payload.get("artifact_uri") is not None
                else None
            ),
            execution_log=(
                str(payload.get("execution_log")).strip()
                if payload.get("execution_log") is not None
                else None
            ),
        )
        or {}
    )


@dataclass(slots=True)
class AgentMinerConfig:
    participation_style: ParticipationStyle = ParticipationStyle.direct
    workspace_root: Path = Path(".bitsota_agent_workspace")
    max_files: int = 6
    max_file_chars: int = 12000
    llm_temperature: float = 0.2
    llm_max_tokens: int = 1800
    execution_enabled: bool = True
    execution_max_attempts: int = 2
    keep_execution_checkout: bool = False
    max_execution_log_chars: int = 32000
    external_agent_command: str = ""
    external_agent_mode: str = "gui_managed"
    submission_filename: str = "submission.json"
    submission_result_filename: str = "submission_result.json"


@dataclass(slots=True)
class LocalExecutionResult:
    observed_metrics: dict[str, float]
    execution_log: str
    artifact_dir: Path
    benchmark_output: str
    base_commit_sha: str | None = None


class LocalExecutionError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        artifact_dir: Path,
        execution_log: str,
    ) -> None:
        super().__init__(message)
        self.artifact_dir = Path(artifact_dir)
        self.execution_log = str(execution_log or "")


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
        self.provider_host = (urlparse(self.api_url).hostname or "").strip().lower()
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
                if item.get("text") is not None:
                    chunks.append(str(item.get("text")))
            return "\n".join(chunks).strip()
        return str(message or "")

    @staticmethod
    def _extract_json_object(raw_text: str) -> dict[str, Any] | None:
        cleaned = _strip_code_fences(raw_text)
        if not cleaned:
            return None

        decoder = json.JSONDecoder()
        try:
            parsed = decoder.decode(cleaned)
            if isinstance(parsed, dict):
                return dict(parsed)
        except json.JSONDecodeError:
            pass

        start = cleaned.find("{")
        while start != -1:
            try:
                parsed, _ = decoder.raw_decode(cleaned[start:])
            except json.JSONDecodeError:
                start = cleaned.find("{", start + 1)
                continue
            if isinstance(parsed, dict):
                return dict(parsed)
            start = cleaned.find("{", start + 1)
        return None

    @staticmethod
    def _response_text_candidates(choice: Any) -> list[tuple[str, str]]:
        if not isinstance(choice, dict):
            return []
        message = choice.get("message") or {}
        candidates = [
            ("content", OpenAICompatibleChatClient._message_text(message.get("content"))),
            ("text", OpenAICompatibleChatClient._message_text(choice.get("text"))),
            ("reasoning", OpenAICompatibleChatClient._message_text(message.get("reasoning"))),
            (
                "reasoning_details",
                OpenAICompatibleChatClient._message_text(message.get("reasoning_details")),
            ),
        ]
        return [(label, text.strip()) for label, text in candidates if str(text or "").strip()]

    def _build_payload(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": str(system_prompt)},
                {"role": "user", "content": str(user_prompt)},
            ],
            "temperature": float(temperature),
            "max_tokens": max(1, int(max_tokens)),
        }
        # OpenRouter supports structured JSON output and provider-specific reasoning controls.
        # Some reasoning-heavy models otherwise exhaust the completion budget before emitting
        # their final JSON in `message.content`.
        if self.provider_host.endswith("openrouter.ai"):
            payload["response_format"] = {"type": "json_object"}
            payload["reasoning"] = self._openrouter_reasoning_config()
        return payload

    def _openrouter_reasoning_config(self) -> dict[str, Any]:
        model_name = self.model.lower()
        if model_name == "stepfun/step-3.5-flash:free":
            return {"max_tokens": 1024, "exclude": True}
        return {"effort": "minimal", "exclude": True}

    def _request_json(self, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        response = self.session.post(self.api_url, json=payload, headers=headers, timeout=self.timeout_s)
        if response.status_code >= 400:
            raise RuntimeError(f"chat completion failed: HTTP {response.status_code} ({response.text})")
        return dict(response.json() or {})

    @staticmethod
    def _parse_completion_payload(data: dict[str, Any]) -> dict[str, Any]:
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("chat completion returned no choices")

        choice = choices[0] if isinstance(choices[0], dict) else {}
        for _, text in OpenAICompatibleChatClient._response_text_candidates(choice):
            parsed = OpenAICompatibleChatClient._extract_json_object(text)
            if parsed is not None:
                return parsed

        message = choice.get("message") or {}
        content_text = OpenAICompatibleChatClient._message_text(message.get("content"))
        reasoning_text = OpenAICompatibleChatClient._message_text(message.get("reasoning"))
        finish_reason = (
            str(choice.get("finish_reason") or choice.get("native_finish_reason") or "").strip() or "unknown"
        )
        raise RuntimeError(
            "chat completion returned no JSON payload "
            f"(finish_reason={finish_reason}, content_len={len(content_text)}, reasoning_len={len(reasoning_text)})"
        )

    def _retry_max_tokens(self, max_tokens: int) -> int | None:
        current = max(1, int(max_tokens))
        grown = min(8192, max(3000, current * 2, current + 1024))
        if grown <= current:
            return None
        return grown

    def _initial_max_tokens(self, max_tokens: int) -> int:
        current = max(1, int(max_tokens))
        if self.provider_host.endswith("openrouter.ai") and self.model.lower() == "stepfun/step-3.5-flash:free":
            return max(current, 8192)
        return current

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
        budgets = [self._initial_max_tokens(max_tokens)]
        retry_budget = self._retry_max_tokens(budgets[0])
        if retry_budget is not None:
            budgets.append(retry_budget)

        last_error: RuntimeError | None = None
        for attempt, budget in enumerate(budgets, start=1):
            payload = self._build_payload(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=budget,
            )
            try:
                data = self._request_json(payload, headers)
                return self._parse_completion_payload(data)
            except RuntimeError as exc:
                last_error = exc
                if attempt >= len(budgets):
                    break

        if last_error is None:
            raise RuntimeError("chat completion failed for an unknown reason")
        if len(budgets) > 1:
            raise RuntimeError(
                f"{last_error}. Retried once with a larger completion budget ({budgets[-1]} tokens)."
            ) from last_error
        raise last_error


class ResearchAgentMiner:
    def __init__(
        self,
        *,
        coordinator: CoordinatorClient,
        llm: OpenAICompatibleChatClient | None,
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
        self.log(
            f"[research-agent] selected task slug={task.get('slug')} mode={task.get('competition_mode')} "
            f"style={style.value} work_item_id={work_item.get('id') if work_item else None}"
        )
        if style == ParticipationStyle.pool and work_item is None:
            self.log(
                "[research-agent] no open coordinator work items were available; "
                "falling back to a direct task claim for this run."
            )
        onboard = self.coordinator.get_onboard_markdown(str(task["id"]))
        submissions = self.coordinator.list_submissions(task_id=str(task["id"]))
        self.log(
            f"[research-agent] fetched coordinator context onboard_chars={len(onboard)} submissions={len(submissions)}"
        )
        if self._uses_external_agent():
            return self._mine_once_with_external_agent(
                task=task,
                work_item=work_item,
                onboard=onboard,
                submissions=submissions,
                participation_style=style,
            )
        repo_context = self._load_repository_context(task=task, onboard=onboard)
        idea_candidates = self._idea_candidates(task=task, submissions=submissions)
        self.log(
            f"[research-agent] repo context ready repo_files={len(repo_context)} "
            f"repo_chars={sum(len(row.get('content', '')) for row in repo_context)} "
            f"idea_candidates={len(idea_candidates)} execution_enabled={self.config.execution_enabled} "
            f"execution_attempts={max(1, int(self.config.execution_max_attempts))}"
        )
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
        self.log(
            f"[research-agent] plan ready claim_len={len(claim_description)} summary_len={len(str(plan.get('summary') or ''))} "
            f"patch_len={len(str(plan.get('patch') or ''))} metric_keys={sorted(_coerce_claimed_metrics(plan.get('claimed_metrics')).keys())}"
        )

        if work_item is not None:
            self.log(
                f"[research-agent] claiming work item id={work_item.get('id')} title={work_item.get('title')!r}"
            )
            claim = self.coordinator.claim_work_item(
                work_item_id=str(work_item["id"]),
                claim_description=claim_description,
            )
        else:
            self.log(f"[research-agent] claiming task id={task.get('id')} slug={task.get('slug')}")
            claim = self.coordinator.claim_task(
                task_id=str(task["id"]),
                claim_description=claim_description,
            )

        implemented_submission_id = str(plan.get("implemented_submission_id") or "").strip() or None
        mode = str(task.get("competition_mode") or CompetitionMode.standard.value)
        if mode == CompetitionMode.centerless.value and idea_candidates and not implemented_submission_id:
            implemented_submission_id = str(idea_candidates[0]["id"])
        execution_result: LocalExecutionResult | None = None
        current_plan = dict(plan)
        try:
            if self.config.execution_enabled:
                current_plan, execution_result = self._execute_submission_plan_with_retries(
                    task=task,
                    work_item=work_item,
                    initial_plan=current_plan,
                    onboard=onboard,
                    repo_context=repo_context,
                    idea_candidates=idea_candidates,
                    participation_style=style,
                )
            claimed_metrics = _coerce_claimed_metrics(current_plan.get("claimed_metrics"))
            summary = str(current_plan.get("summary") or "")
            artifact_uri: str | None = None
            execution_log: str | None = None
            if execution_result is not None:
                claimed_metrics.update(execution_result.observed_metrics)
                summary = self._append_execution_summary(summary, execution_result.observed_metrics)
                artifact_uri = str(execution_result.artifact_dir)
                execution_log = execution_result.execution_log
            current_plan["summary"] = summary
            current_plan["claimed_metrics"] = claimed_metrics
            if artifact_uri is not None:
                current_plan["artifact_uri"] = artifact_uri

            self.log(
                f"[research-agent] submitting claim_id={claim.get('id')} base_ref={current_plan.get('base_ref') or task.get('base_ref')} "
                f"implemented_submission_id={implemented_submission_id} artifact_uri={artifact_uri}"
            )
            submission = self.coordinator.submit_submission(
                claim_id=str(claim["id"]),
                base_ref=str(
                    (
                        execution_result.base_commit_sha
                        if execution_result is not None and execution_result.base_commit_sha
                        else current_plan.get("base_ref") or task.get("base_ref") or ""
                    )
                ),
                patch=str(current_plan.get("patch") or ""),
                summary=summary,
                claimed_metrics=claimed_metrics,
                proposed_idea=(
                    str(current_plan.get("proposed_idea")).strip()
                    if current_plan.get("proposed_idea") is not None
                    else None
                ),
                implemented_submission_id=implemented_submission_id,
                artifact_uri=artifact_uri,
                execution_log=execution_log,
            )
        except Exception as exc:
            self._cancel_claim_after_failure(claim_id=str(claim.get("id") or ""), reason=str(exc))
            raise
        self.log(
            f"[research-agent] submitted task={task.get('slug')} mode={mode} "
            f"style={style.value} submission_id={submission.get('id')}"
        )
        return {
            "task": task,
            "work_item": work_item,
            "claim": claim,
            "submission": submission,
            "plan": current_plan,
            "execution": (
                {
                    "artifact_uri": str(execution_result.artifact_dir),
                    "observed_metrics": execution_result.observed_metrics,
                }
                if execution_result is not None
                else None
            ),
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

    def _cancel_claim_after_failure(self, *, claim_id: str, reason: str) -> None:
        claim_id = str(claim_id or "").strip()
        if not claim_id:
            return
        try:
            self.coordinator.cancel_claim(claim_id=claim_id)
            self.log(f"[research-agent] cancelled claim id={claim_id} after failure: {reason}")
        except Exception as exc:
            self.log(f"[research-agent] failed to cancel claim id={claim_id}: {exc}")

    def _uses_external_agent(self) -> bool:
        return bool(str(self.config.external_agent_command or "").strip())

    def _mine_once_with_external_agent(
        self,
        *,
        task: dict[str, Any],
        work_item: dict[str, Any] | None,
        onboard: str,
        submissions: list[dict[str, Any]],
        participation_style: ParticipationStyle,
    ) -> dict[str, Any]:
        mode = str(task.get("competition_mode") or CompetitionMode.standard.value)
        idea_candidates = self._idea_candidates(task=task, submissions=submissions)
        claim_description = (
            str(work_item.get("title") or "").strip()
            if work_item is not None
            else f"External agent run for {task.get('slug') or task.get('title')}"
        )
        if work_item is not None:
            self.log(
                f"[research-agent] claiming work item id={work_item.get('id')} title={work_item.get('title')!r}"
            )
            claim = self.coordinator.claim_work_item(
                work_item_id=str(work_item["id"]),
                claim_description=claim_description,
            )
        else:
            self.log(f"[research-agent] claiming task id={task.get('id')} slug={task.get('slug')}")
            claim = self.coordinator.claim_task(
                task_id=str(task["id"]),
                claim_description=claim_description,
            )

        try:
            agent_result = self._run_external_agent(
                task=task,
                work_item=work_item,
                claim=claim,
                onboard=onboard,
                idea_candidates=idea_candidates,
            )
            if _normalized_submission_mode(self.config.external_agent_mode) == "autonomous":
                result_path = Path(agent_result["submission_result_file"])
                if result_path.exists():
                    submission = json.loads(result_path.read_text(encoding="utf-8"))
                    self.log(
                        f"[research-agent] autonomous agent reported submission_id={submission.get('id')} "
                        f"task={task.get('slug')}"
                    )
                    return {
                        "task": task,
                        "work_item": work_item,
                        "claim": claim,
                        "submission": submission,
                        "plan": None,
                        "execution": {
                            "workspace_dir": agent_result["workspace_dir"],
                            "repo_dir": agent_result["repo_dir"],
                        },
                    }
                raise RuntimeError(
                    "autonomous agent finished without writing submission_result.json; "
                    "claim was left to the agent to manage"
                )

            submission = submit_claimed_workspace(
                coordinator=self.coordinator,
                claim_id=str(claim["id"]),
                repo_dir=Path(agent_result["repo_dir"]),
                submission_file=Path(agent_result["submission_file"]),
                default_base_ref=str(task.get("base_ref") or ""),
                competition_mode=mode,
                idea_candidates=idea_candidates,
            )
        except Exception as exc:
            self._cancel_claim_after_failure(claim_id=str(claim.get("id") or ""), reason=str(exc))
            raise
        self.log(
            f"[research-agent] submitted task={task.get('slug')} mode={mode} "
            f"style={participation_style.value} submission_id={submission.get('id')}"
        )
        return {
            "task": task,
            "work_item": work_item,
            "claim": claim,
            "submission": submission,
            "plan": None,
            "execution": {
                "workspace_dir": agent_result["workspace_dir"],
                "repo_dir": agent_result["repo_dir"],
                "submission_file": agent_result["submission_file"],
            },
        }

    def _run_external_agent(
        self,
        *,
        task: dict[str, Any],
        work_item: dict[str, Any] | None,
        claim: dict[str, Any],
        onboard: str,
        idea_candidates: list[dict[str, Any]],
    ) -> dict[str, str]:
        repo_url = str(task.get("repository") or "").strip()
        base_ref = str(task.get("base_ref") or "").strip()
        if not repo_url or not base_ref:
            raise RuntimeError("task is missing repository or base_ref")

        workspace_root = self._workspace_root()
        workspace_dir = Path(
            tempfile.mkdtemp(
                prefix=f"external-agent-{_safe_name_fragment(str(task.get('slug') or task.get('title') or 'task'))}-",
                dir=str(workspace_root),
            )
        )
        repo_dir = workspace_dir / "repo"
        submission_file = workspace_dir / str(self.config.submission_filename or "submission.json")
        submission_result_file = workspace_dir / str(
            self.config.submission_result_filename or "submission_result.json"
        )
        intro_filename = "INTRO.md" if _normalized_submission_mode(self.config.external_agent_mode) == "autonomous" else "INTRO_GUI.md"
        intro_path = workspace_dir / intro_filename

        self.log(
            f"[research-agent] external agent cloning repo url={repo_url} ref={base_ref} workspace_dir={workspace_dir}"
        )
        clone_result = self._run_local_command(
            ["git", "clone", "--quiet", repo_url, str(repo_dir)],
            timeout_seconds=max(30, int(task.get("time_budget_seconds") or 300)),
        )
        if clone_result.returncode != 0:
            raise RuntimeError(f"external agent clone failed with exit={clone_result.returncode}")
        checkout_result = self._run_local_command(
            ["git", "checkout", base_ref],
            cwd=repo_dir,
            timeout_seconds=max(30, int(task.get("time_budget_seconds") or 300)),
        )
        if checkout_result.returncode != 0:
            raise RuntimeError(f"external agent checkout failed with exit={checkout_result.returncode}")

        intro_text = build_agent_intro_markdown(
            task=task,
            work_item=work_item,
            claim=claim,
            onboard=onboard,
            coordinator_url=str(self.coordinator.base_url),
            mode=self.config.external_agent_mode,
            repo_dir=repo_dir,
            workspace_dir=workspace_dir,
            submission_file=submission_file,
            submission_result_file=submission_result_file,
        )
        self._write_text(intro_path, intro_text)
        self._write_text(workspace_dir / "onboard.md", onboard)
        self._write_json(
            workspace_dir / "claim_context.json",
            {
                "task": task,
                "work_item": work_item,
                "claim": claim,
                "idea_candidates": idea_candidates,
            },
        )
        command = self._format_external_agent_command(
            command_template=str(self.config.external_agent_command or ""),
            workspace_dir=workspace_dir,
            repo_dir=repo_dir,
            intro_path=intro_path,
            submission_file=submission_file,
            submission_result_file=submission_result_file,
        )
        self.log(
            f"[research-agent] external agent launch mode={_normalized_submission_mode(self.config.external_agent_mode)} "
            f"command={command}"
        )
        env = os.environ.copy()
        env.update(
            {
                "BITSOTA_AGENT_WORKSPACE": str(workspace_dir),
                "BITSOTA_AGENT_REPO_DIR": str(repo_dir),
                "BITSOTA_AGENT_INTRO_PATH": str(intro_path),
                "BITSOTA_AGENT_SUBMISSION_PATH": str(submission_file),
                "BITSOTA_AGENT_SUBMISSION_RESULT_PATH": str(submission_result_file),
                "BITSOTA_RESEARCH_CLAIM_ID": str(claim.get("id") or ""),
                "BITSOTA_RESEARCH_TASK_ID": str(task.get("id") or ""),
                "BITSOTA_RESEARCH_TASK_SLUG": str(task.get("slug") or ""),
                "BITSOTA_RESEARCH_COORDINATOR_URL": str(self.coordinator.base_url),
            }
        )
        result = self._run_local_command(
            command,
            cwd=workspace_dir,
            timeout_seconds=max(60, int(task.get("time_budget_seconds") or 300)),
            shell=True,
            env=env,
        )
        (workspace_dir / "agent.stdout.txt").write_text(str(result.stdout or ""), encoding="utf-8")
        (workspace_dir / "agent.stderr.txt").write_text(str(result.stderr or ""), encoding="utf-8")
        if result.returncode != 0:
            raise RuntimeError(
                f"external agent command failed with exit={result.returncode}: "
                f"{self._truncate_text(str(result.stderr or result.stdout or ''), 600)}"
            )
        return {
            "workspace_dir": str(workspace_dir),
            "repo_dir": str(repo_dir),
            "submission_file": str(submission_file),
            "submission_result_file": str(submission_result_file),
        }

    @staticmethod
    def _format_external_agent_command(
        *,
        command_template: str,
        workspace_dir: Path,
        repo_dir: Path,
        intro_path: Path,
        submission_file: Path,
        submission_result_file: Path,
    ) -> str:
        template = str(command_template or "").strip()
        if not template:
            raise RuntimeError("external agent command is missing")
        mapping = {
            "workspace_dir": str(workspace_dir),
            "repo_dir": str(repo_dir),
            "intro_path": str(intro_path),
            "submission_path": str(submission_file),
            "submission_result_path": str(submission_result_file),
            "workspace_dir_quoted": shlex.quote(str(workspace_dir)),
            "repo_dir_quoted": shlex.quote(str(repo_dir)),
            "intro_path_quoted": shlex.quote(str(intro_path)),
            "submission_path_quoted": shlex.quote(str(submission_file)),
            "submission_result_path_quoted": shlex.quote(str(submission_result_file)),
        }
        try:
            return template.format(**mapping)
        except KeyError:
            return template

    def _execute_submission_plan_with_retries(
        self,
        *,
        task: dict[str, Any],
        work_item: dict[str, Any] | None,
        initial_plan: dict[str, Any],
        onboard: str,
        repo_context: list[dict[str, str]],
        idea_candidates: list[dict[str, Any]],
        participation_style: ParticipationStyle,
    ) -> tuple[dict[str, Any], LocalExecutionResult]:
        current_plan = dict(initial_plan)
        last_error: LocalExecutionError | None = None
        total_attempts = max(1, int(self.config.execution_max_attempts))
        for attempt in range(1, total_attempts + 1):
            try:
                result = self._execute_submission_plan_once(
                    task=task,
                    plan=current_plan,
                    attempt=attempt,
                )
                merged_metrics = _coerce_claimed_metrics(current_plan.get("claimed_metrics"))
                merged_metrics.update(result.observed_metrics)
                current_plan["claimed_metrics"] = merged_metrics
                return current_plan, result
            except LocalExecutionError as exc:
                last_error = exc
                self.log(
                    f"[research-agent] execution failed attempt={attempt}/{total_attempts} "
                    f"artifact_dir={exc.artifact_dir} error={exc}"
                )
                if attempt >= total_attempts:
                    break
                current_plan = self._generate_submission_plan(
                    task=task,
                    work_item=work_item,
                    onboard=onboard,
                    repo_context=repo_context,
                    idea_candidates=idea_candidates,
                    participation_style=participation_style,
                    previous_plan=current_plan,
                    execution_feedback=exc.execution_log,
                )
        if last_error is None:
            raise RuntimeError("local execution failed for an unknown reason")
        raise last_error

    def _execute_submission_plan_once(
        self,
        *,
        task: dict[str, Any],
        plan: dict[str, Any],
        attempt: int,
    ) -> LocalExecutionResult:
        repo_url = str(task.get("repository") or "").strip()
        base_ref = str(plan.get("base_ref") or task.get("base_ref") or "").strip()
        metric_name = str(task.get("metric_name") or "").strip()
        benchmark_command = str(task.get("benchmark_command") or "").strip()
        if not repo_url or not base_ref or not metric_name or not benchmark_command:
            raise RuntimeError("task is missing repository, base_ref, metric_name, or benchmark_command")

        runs_root = self._workspace_root() / "runs"
        runs_root.mkdir(parents=True, exist_ok=True)
        artifact_dir = Path(
            tempfile.mkdtemp(
                prefix=f"{_safe_name_fragment(str(task.get('slug') or task.get('title') or 'task'))}-attempt-{attempt}-",
                dir=str(runs_root),
            )
        )
        repo_dir = artifact_dir / "repo"
        patch_text = str(plan.get("patch") or "")
        self._write_text(artifact_dir / "submission.patch.diff", patch_text)
        self._write_json(artifact_dir / "plan.json", plan)

        log_lines = [
            f"task_slug={task.get('slug')}",
            f"repository={repo_url}",
            f"base_ref={base_ref}",
            f"metric_name={metric_name}",
            f"time_budget_seconds={task.get('time_budget_seconds')}",
            f"artifact_dir={artifact_dir}",
        ]
        metadata: dict[str, Any] = {
            "task_slug": task.get("slug"),
            "repository": repo_url,
            "base_ref": base_ref,
            "metric_name": metric_name,
            "time_budget_seconds": task.get("time_budget_seconds"),
            "artifact_dir": str(artifact_dir),
        }

        def _raise_execution_error(message: str) -> None:
            execution_log = self._truncate_execution_log("\n".join(part for part in log_lines if part))
            self._write_text(artifact_dir / "execution.log", execution_log)
            self._write_json(artifact_dir / "metadata.json", metadata)
            raise LocalExecutionError(
                message,
                artifact_dir=artifact_dir,
                execution_log=execution_log,
            )

        try:
            self.log(
                f"[research-agent] execution attempt={attempt} cloning repo url={repo_url} "
                f"ref={base_ref} artifact_dir={artifact_dir}"
            )
            clone_result = self._run_local_command(
                ["git", "clone", "--quiet", repo_url, str(repo_dir)],
                timeout_seconds=max(30, int(task.get("time_budget_seconds") or 300)),
            )
            self._write_completed_process_artifacts(artifact_dir, "clone", clone_result)
            log_lines.extend(self._format_completed_process("clone", clone_result))
            metadata["clone_exit"] = clone_result.returncode
            if clone_result.returncode != 0:
                _raise_execution_error(f"repository clone failed with exit={clone_result.returncode}")

            checkout_result = self._run_local_command(
                ["git", "checkout", base_ref],
                cwd=repo_dir,
                timeout_seconds=max(30, int(task.get("time_budget_seconds") or 300)),
            )
            self._write_completed_process_artifacts(artifact_dir, "checkout", checkout_result)
            log_lines.extend(self._format_completed_process("checkout", checkout_result))
            metadata["checkout_exit"] = checkout_result.returncode
            if checkout_result.returncode != 0:
                _raise_execution_error(f"repository checkout failed with exit={checkout_result.returncode}")
            base_commit_sha = resolve_repo_head_commit(repo_dir)
            metadata["base_commit_sha"] = base_commit_sha
            log_lines.append(f"base_commit_sha={base_commit_sha}")

            apply_result = self._run_local_command(
                ["git", "apply", "-"],
                cwd=repo_dir,
                timeout_seconds=max(30, int(task.get("time_budget_seconds") or 300)),
                input_text=patch_text,
            )
            self._write_completed_process_artifacts(artifact_dir, "patch_apply", apply_result)
            log_lines.extend(self._format_completed_process("patch_apply", apply_result))
            metadata["patch_apply_exit"] = apply_result.returncode
            if apply_result.returncode != 0:
                _raise_execution_error(f"patch apply failed with exit={apply_result.returncode}")

            setup_command = str(task.get("setup_command") or "").strip()
            if setup_command:
                setup_result = self._run_local_command(
                    setup_command,
                    cwd=repo_dir,
                    timeout_seconds=int(task.get("time_budget_seconds") or 300),
                    shell=True,
                )
                self._write_completed_process_artifacts(artifact_dir, "setup", setup_result)
                log_lines.extend(self._format_completed_process("setup", setup_result, command=setup_command))
                metadata["setup_exit"] = setup_result.returncode
                if setup_result.returncode != 0:
                    _raise_execution_error(f"setup command failed with exit={setup_result.returncode}")

            benchmark_result = self._run_local_command(
                benchmark_command,
                cwd=repo_dir,
                timeout_seconds=int(task.get("time_budget_seconds") or 300),
                shell=True,
            )
            self._write_completed_process_artifacts(artifact_dir, "benchmark", benchmark_result)
            log_lines.extend(self._format_completed_process("benchmark", benchmark_result, command=benchmark_command))
            metadata["benchmark_exit"] = benchmark_result.returncode

            combined_output = "\n".join(
                part for part in [benchmark_result.stdout.strip(), benchmark_result.stderr.strip()] if part
            )
            metric_value = parse_metric_from_text(combined_output, metric_name)
            if benchmark_result.returncode != 0:
                _raise_execution_error(f"benchmark command failed with exit={benchmark_result.returncode}")
            if metric_value is None:
                _raise_execution_error(f"benchmark output did not include metric {metric_name!r}")

            observed_metrics = {metric_name: float(metric_value)}
            metadata["observed_metrics"] = observed_metrics
            log_lines.append(f"observed_metrics={observed_metrics}")
            execution_log = self._truncate_execution_log("\n".join(part for part in log_lines if part))
            self._write_text(artifact_dir / "execution.log", execution_log)
            self._write_json(artifact_dir / "metadata.json", metadata)
            self.log(
                f"[research-agent] execution succeeded attempt={attempt} "
                f"{metric_name}={metric_value} artifact_dir={artifact_dir}"
            )
            return LocalExecutionResult(
                observed_metrics=observed_metrics,
                execution_log=execution_log,
                artifact_dir=artifact_dir,
                benchmark_output=combined_output,
                base_commit_sha=str(metadata.get("base_commit_sha") or "") or None,
            )
        except subprocess.TimeoutExpired as exc:
            log_lines.extend(
                [
                    f"timeout_command={exc.cmd}",
                    f"timeout_seconds={exc.timeout}",
                    f"timeout_stdout={str(exc.stdout or '')}",
                    f"timeout_stderr={str(exc.stderr or '')}",
                ]
            )
            metadata["timeout"] = {"command": str(exc.cmd), "seconds": exc.timeout}
            _raise_execution_error("local execution timed out")
        finally:
            if repo_dir.exists() and self.config.execution_enabled and self.config.keep_execution_checkout is False:
                execution_log_path = artifact_dir / "execution.log"
                if execution_log_path.exists():
                    log_text = execution_log_path.read_text(encoding="utf-8", errors="replace")
                    if "observed_metrics=" in log_text:
                        shutil.rmtree(repo_dir, ignore_errors=True)

    def _workspace_root(self) -> Path:
        root = Path(self.config.workspace_root).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        return root

    @staticmethod
    def _run_local_command(
        command: list[str] | str,
        *,
        cwd: Path | None = None,
        timeout_seconds: int,
        shell: bool = False,
        input_text: str | None = None,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command,
            cwd=str(cwd) if cwd is not None else None,
            shell=shell,
            check=False,
            capture_output=True,
            text=True,
            timeout=max(1, int(timeout_seconds)),
            input=input_text,
            env=env,
        )

    @staticmethod
    def _format_completed_process(
        name: str,
        result: subprocess.CompletedProcess[str],
        *,
        command: str | None = None,
    ) -> list[str]:
        lines = [f"{name}_exit={result.returncode}"]
        if command:
            lines.append(f"{name}_command={command}")
        if str(result.stdout or "").strip():
            lines.append(f"{name}_stdout:\n{result.stdout.strip()}")
        if str(result.stderr or "").strip():
            lines.append(f"{name}_stderr:\n{result.stderr.strip()}")
        return lines

    @staticmethod
    def _write_completed_process_artifacts(
        artifact_dir: Path,
        name: str,
        result: subprocess.CompletedProcess[str],
    ) -> None:
        (artifact_dir / f"{name}.stdout.txt").write_text(
            str(result.stdout or ""),
            encoding="utf-8",
        )
        (artifact_dir / f"{name}.stderr.txt").write_text(
            str(result.stderr or ""),
            encoding="utf-8",
        )

    @staticmethod
    def _write_text(path: Path, content: str) -> None:
        path.write_text(str(content or ""), encoding="utf-8")

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")

    def _truncate_execution_log(self, text: str) -> str:
        limit = max(2000, int(self.config.max_execution_log_chars))
        return self._truncate_text(text, limit)

    def _append_execution_summary(self, summary: str, observed_metrics: dict[str, Any]) -> str:
        if not observed_metrics:
            return str(summary or "")
        metrics_text = ", ".join(f"{key}={value:.10g}" for key, value in observed_metrics.items())
        suffix = f"Observed locally: {metrics_text}."
        combined = str(summary or "").strip()
        if combined:
            combined = f"{combined}\n\n{suffix}"
        else:
            combined = suffix
        return self._truncate_text(combined, 7900)

    def _generate_submission_plan(
        self,
        *,
        task: dict[str, Any],
        work_item: dict[str, Any] | None,
        onboard: str,
        repo_context: list[dict[str, str]],
        idea_candidates: list[dict[str, Any]],
        participation_style: ParticipationStyle,
        previous_plan: dict[str, Any] | None = None,
        execution_feedback: str | None = None,
    ) -> dict[str, Any]:
        mode = str(task.get("competition_mode") or CompetitionMode.standard.value)
        retry_note = (
            "Previous local execution failed. Return a corrected patch that should apply cleanly, "
            "run under the fixed benchmark, and report the benchmark metric accurately."
            if execution_feedback
            else None
        )
        system_prompt = (
            "You are a BitSota research miner. Return JSON only. "
            "Propose one concrete submission for this coordinator task. "
            "The JSON must contain claim_description, summary, patch, claimed_metrics, and optional "
            "proposed_idea plus implemented_submission_id. "
            "patch must be a unified diff string. "
            "Do not wrap JSON in markdown."
        )
        prompt_variants = [
            self._build_submission_prompt(
                task=task,
                work_item=work_item,
                onboard=onboard,
                repo_context=repo_context,
                idea_candidates=idea_candidates[:5],
                participation_style=participation_style,
                mode=mode,
                previous_plan=previous_plan,
                execution_feedback=execution_feedback,
                generation_note=retry_note,
            ),
            self._build_submission_prompt(
                task=task,
                work_item=work_item,
                onboard=self._truncate_text(onboard, 3200),
                repo_context=self._compact_repo_context(repo_context, max_files=3, total_chars=3500),
                idea_candidates=idea_candidates[:3],
                participation_style=participation_style,
                mode=mode,
                previous_plan=previous_plan,
                execution_feedback=execution_feedback,
                generation_note=self._join_notes(
                    retry_note,
                    "Keep the response compact and ensure patch, summary, and claimed_metrics are non-empty.",
                ),
            ),
            self._build_submission_prompt(
                task=task,
                work_item=work_item,
                onboard=self._truncate_text(onboard, 1800),
                repo_context=[],
                idea_candidates=idea_candidates[:2],
                participation_style=participation_style,
                mode=mode,
                previous_plan=previous_plan,
                execution_feedback=execution_feedback,
                generation_note=self._join_notes(
                    retry_note,
                    "Return the smallest valid plan possible. patch must be a non-empty unified diff string.",
                ),
            ),
            self._build_minimal_submission_prompt(
                task=task,
                work_item=work_item,
                onboard=onboard,
                repo_context=repo_context,
                participation_style=participation_style,
                mode=mode,
                previous_plan=previous_plan,
                execution_feedback=execution_feedback,
                generation_note=self._join_notes(
                    retry_note,
                    "Target one file and return the smallest valid JSON plan possible.",
                ),
            ),
        ]
        last_error: RuntimeError | None = None
        for index, prompt in enumerate(prompt_variants, start=1):
            try:
                repo_files = []
                if isinstance(prompt, dict):
                    repo_files = prompt.get("repository_files") or prompt.get("files") or []
                repo_chars = sum(len(str(row.get("content") or row.get("snippet") or "")) for row in repo_files if isinstance(row, dict))
                onboard_chars = 0
                idea_count = 0
                if isinstance(prompt, dict):
                    onboard_chars = len(str(prompt.get("onboard_markdown") or prompt.get("constraints_text") or ""))
                    idea_count = len(prompt.get("idea_candidates") or [])
                prompt_json = _compact_json_dumps(prompt) if isinstance(prompt, dict) else str(prompt)
                self.log(
                    f"[research-agent] planner attempt={index}/{len(prompt_variants)} "
                    f"prompt_chars={len(prompt_json)} onboard_chars={onboard_chars} "
                    f"repo_files={len(repo_files)} repo_chars={repo_chars} ideas={idea_count}"
                )
                result = self.llm.complete_json(
                    system_prompt=system_prompt,
                    user_prompt=prompt_json,
                    temperature=self.config.llm_temperature,
                    max_tokens=self.config.llm_max_tokens,
                )
                return self._validate_submission_plan(result, metric_name=str(task.get("metric_name") or ""))
            except RuntimeError as exc:
                last_error = exc
                if index >= len(prompt_variants) or not self._should_retry_with_smaller_prompt(exc):
                    break
                self.log(
                    f"[research-agent] planner retrying with reduced context after model response failure: {exc}"
                )
        if last_error is None:
            raise RuntimeError("submission plan generation failed for an unknown reason")
        raise last_error

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
        prompt_variants = [
            {
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
            },
            {
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
                "submission_detail": {
                    "submission": detail.get("submission"),
                    "peer_consensus": detail.get("peer_consensus"),
                    "peer_evaluations": (detail.get("peer_evaluations") or [])[:3],
                },
                "onboard_markdown": self._truncate_text(onboard, 1800),
                "response_schema": {
                    "status": "accepted|rejected|error",
                    "observed_metrics": {"metric_name": "number"},
                    "notes": "string",
                },
            },
        ]
        last_error: RuntimeError | None = None
        for index, prompt in enumerate(prompt_variants, start=1):
            try:
                prompt_json = json.dumps(prompt, indent=2)
                self.log(
                    f"[research-agent] peer-eval attempt={index}/{len(prompt_variants)} "
                    f"prompt_chars={len(prompt_json)} onboard_chars={len(str(prompt.get('onboard_markdown') or ''))}"
                )
                result = self.llm.complete_json(
                    system_prompt=system_prompt,
                    user_prompt=prompt_json,
                    temperature=self.config.llm_temperature,
                    max_tokens=self.config.llm_max_tokens,
                )
                return dict(result)
            except RuntimeError as exc:
                last_error = exc
                if index >= len(prompt_variants) or not self._should_retry_with_smaller_prompt(exc):
                    break
                self.log(
                    f"[research-agent] peer evaluation retrying with reduced context after model response failure: {exc}"
                )
        if last_error is None:
            raise RuntimeError("peer evaluation generation failed for an unknown reason")
        raise last_error

    def _build_submission_prompt(
        self,
        *,
        task: dict[str, Any],
        work_item: dict[str, Any] | None,
        onboard: str,
        repo_context: list[dict[str, str]],
        idea_candidates: list[dict[str, Any]],
        participation_style: ParticipationStyle,
        mode: str,
        previous_plan: dict[str, Any] | None = None,
        execution_feedback: str | None = None,
        generation_note: str | None = None,
    ) -> dict[str, Any]:
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
            "idea_candidates": idea_candidates,
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
        if previous_plan is not None:
            prompt["previous_attempt"] = {
                "claim_description": previous_plan.get("claim_description"),
                "summary": previous_plan.get("summary"),
                "claimed_metrics": previous_plan.get("claimed_metrics"),
                "patch": self._truncate_text(str(previous_plan.get("patch") or ""), 6000),
            }
        if execution_feedback:
            prompt["execution_feedback"] = self._truncate_execution_log(str(execution_feedback or ""))
        if generation_note:
            prompt["generation_note"] = str(generation_note)
        return prompt

    def _build_minimal_submission_prompt(
        self,
        *,
        task: dict[str, Any],
        work_item: dict[str, Any] | None,
        onboard: str,
        repo_context: list[dict[str, str]],
        participation_style: ParticipationStyle,
        mode: str,
        previous_plan: dict[str, Any] | None = None,
        execution_feedback: str | None = None,
        generation_note: str | None = None,
    ) -> str:
        metric_name = str(task.get("metric_name") or "metric").strip() or "metric"
        lines = [
            f"Task {task.get('slug')} @ {task.get('base_ref')}",
            f"Metric {metric_name} ({task.get('metric_direction')})",
            f"Rules: {self._extract_onboard_constraints(onboard, metric_name=metric_name)}",
        ]
        if work_item is not None:
            lines.append(
                "Work item: "
                + self._truncate_text(
                    f"{work_item.get('title')}: {work_item.get('instructions')}",
                    180,
                )
            )
        files = self._minimal_repo_context(repo_context, max_files=1, total_chars=320)
        for row in files:
            lines.append(f"File {row.get('path')}:\n{row.get('snippet')}")
        if previous_plan is not None:
            lines.append(
                "Previous attempt summary: "
                + self._truncate_text(str(previous_plan.get("summary") or ""), 120)
            )
        if execution_feedback:
            lines.append("Last error: " + self._truncate_text(self._truncate_execution_log(str(execution_feedback or "")), 220))
        if generation_note:
            lines.append("Note: " + self._truncate_text(str(generation_note), 120))
        lines.append(
            'Return JSON only: {"claim_description":"...","summary":"...","patch":"diff --git ...","claimed_metrics":{"'
            + metric_name
            + '":0}}'
        )
        return "\n".join(line for line in lines if str(line).strip())

    @staticmethod
    def _truncate_text(text: str, max_chars: int) -> str:
        raw = str(text or "").strip()
        if len(raw) <= int(max_chars):
            return raw
        return raw[: max(0, int(max_chars) - 3)].rstrip() + "..."

    @staticmethod
    def _join_notes(*notes: str | None) -> str | None:
        parts = [str(note).strip() for note in notes if str(note or "").strip()]
        if not parts:
            return None
        return " ".join(parts)

    def _compact_repo_context(
        self,
        repo_context: list[dict[str, str]],
        *,
        max_files: int,
        total_chars: int,
    ) -> list[dict[str, str]]:
        remaining = max(0, int(total_chars))
        compact: list[dict[str, str]] = []
        for item in repo_context:
            if remaining <= 0 or len(compact) >= int(max_files):
                break
            path = str(item.get("path") or "").strip()
            content = str(item.get("content") or "")
            if not path or not content:
                continue
            slice_chars = min(len(content), remaining, max(600, remaining // max(1, int(max_files) - len(compact))))
            compact.append({"path": path, "content": self._truncate_text(content, slice_chars)})
            remaining = max(0, remaining - len(compact[-1]["content"]))
        return compact

    def _minimal_repo_context(
        self,
        repo_context: list[dict[str, str]],
        *,
        max_files: int,
        total_chars: int,
    ) -> list[dict[str, str]]:
        preferred = [row for row in repo_context if not str(row.get("path") or "").lower().endswith((".md", ".txt"))]
        ordered = preferred + [row for row in repo_context if row not in preferred]
        remaining = max(0, int(total_chars))
        compact: list[dict[str, str]] = []
        for item in ordered:
            if remaining <= 0 or len(compact) >= int(max_files):
                break
            path = str(item.get("path") or "").strip()
            content = str(item.get("content") or "").strip()
            if not path or not content:
                continue
            slice_chars = min(len(content), remaining, max(400, remaining // max(1, int(max_files) - len(compact))))
            compact.append({"path": path, "snippet": self._truncate_text(content, slice_chars)})
            remaining = max(0, remaining - len(compact[-1]["snippet"]))
        return compact

    def _extract_onboard_constraints(self, onboard: str, *, metric_name: str) -> str:
        keywords = {
            str(metric_name or "").strip().lower(),
            "benchmark",
            "artifact",
            "checkpoint",
            "compression",
            "emit final",
            "only submit",
            "must",
            "run ",
            "python ",
            "uv ",
            "patch",
        }
        lines: list[str] = []
        seen: set[str] = set()
        for raw in str(onboard or "").splitlines():
            line = raw.strip().lstrip("-").strip()
            if not line or line.startswith("#"):
                continue
            lowered = line.lower()
            if not any(keyword and keyword in lowered for keyword in keywords):
                continue
            key = lowered
            if key in seen:
                continue
            seen.add(key)
            lines.append(line)
            if len(lines) >= 4:
                break
        text = " ".join(lines).strip()
        if not text:
            text = self._truncate_text(onboard, 420)
        return self._truncate_text(text, 420)

    @staticmethod
    def _should_retry_with_smaller_prompt(exc: RuntimeError) -> bool:
        text = str(exc or "").lower()
        return (
            "finish_reason=length" in text
            or "returned no json payload" in text
            or "returned empty content" in text
            or "did not return valid json" in text
            or "submission plan missing required fields" in text
        )

    @staticmethod
    def _validate_submission_plan(result: dict[str, Any], *, metric_name: str) -> dict[str, Any]:
        plan = dict(result or {})
        missing: list[str] = []
        for key in ("claim_description", "summary", "patch"):
            if not str(plan.get(key) or "").strip():
                missing.append(key)

        claimed_metrics = _coerce_claimed_metrics(plan.get("claimed_metrics"))
        if not claimed_metrics:
            missing.append("claimed_metrics")
        elif metric_name and metric_name not in claimed_metrics:
            missing.append(f"claimed_metrics.{metric_name}")

        if missing:
            raise RuntimeError(
                "submission plan missing required fields: " + ", ".join(missing)
            )

        plan["claim_description"] = str(plan.get("claim_description")).strip()
        plan["summary"] = str(plan.get("summary")).strip()
        plan["patch"] = str(plan.get("patch"))
        plan["claimed_metrics"] = claimed_metrics
        return plan

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
            self.log(f"[research-agent] cloning repo url={repo_url} ref={base_ref} into {repo_dir}")
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
            self.log(f"[research-agent] inferred repo file candidates={file_candidates}")
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
            self.log(
                f"[research-agent] loaded repo context paths={[row['path'] for row in contexts]} "
                f"chars={sum(len(row['content']) for row in contexts)}"
            )
            return contexts
        except subprocess.CalledProcessError as exc:
            stderr = str(getattr(exc, "stderr", "") or "").strip()
            stdout = str(getattr(exc, "stdout", "") or "").strip()
            self.log(
                f"[research-agent] repository context load failed returncode={exc.returncode} "
                f"stdout={stdout[:300]!r} stderr={stderr[:300]!r}"
            )
            raise
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

    def _infer_file_candidates(self, onboard: str) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for match in _FILE_PATTERN.finditer(str(onboard or "")):
            candidate = str(match.group(1)).strip()
            lowered = candidate.lower()
            if lowered.startswith(("http/", "https/", "api/")):
                continue
            if re.match(r"^\d+/", candidate):
                continue
            if "/api/" in lowered:
                continue
            key = candidate.lower()
            if key in seen:
                continue
            seen.add(key)
            ordered.append(candidate)
        if "readme.md" not in seen:
            ordered.insert(0, "README.md")
        return ordered
