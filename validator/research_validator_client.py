from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode, urljoin

import requests

from miner.research_auth import sign_hotkey_request


DEFAULT_RESEARCH_COORDINATOR_URL = "https://chvp2wytst.eu-central-1.awsapprunner.com"
DEFAULT_VALIDATOR_JOB_CLAIM_PATH = "/api/v1/validator/jobs/claim"


class PublicValidatorApiError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True, slots=True)
class ReplayJob:
    job_id: str | None
    submission_id: str
    submission: dict[str, Any]
    task: dict[str, Any]
    replay_spec: dict[str, Any]
    detail: dict[str, Any]
    source: str


class AutoresearchValidatorClient:
    def __init__(
        self,
        *,
        base_url: str = DEFAULT_RESEARCH_COORDINATOR_URL,
        wallet: Any,
        timeout_s: float = 30.0,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = str(base_url or "").rstrip("/")
        if not self.base_url:
            raise ValueError("base_url is required")
        self.wallet = wallet
        self.timeout_s = max(1.0, float(timeout_s))
        self.session = session or requests.Session()

    @property
    def hotkey(self) -> str:
        hotkey = getattr(getattr(self.wallet, "hotkey", None), "ss58_address", None)
        if not hotkey:
            raise PublicValidatorApiError("wallet hotkey is unavailable")
        return str(hotkey)

    def _signed_headers(
        self,
        *,
        method: str,
        path: str,
        body: Any = None,
        query: str = "",
    ) -> dict[str, str]:
        keypair = getattr(self.wallet, "hotkey", None)
        if keypair is None:
            raise PublicValidatorApiError("wallet hotkey signer is unavailable")
        headers = sign_hotkey_request(
            keypair=keypair,
            method=method,
            path=path,
            query=query,
            body=body,
        ).as_headers()
        if body is not None:
            headers["Content-Type"] = "application/json"
        return headers

    def _request(
        self,
        method: str,
        path: str,
        *,
        body: Any = None,
        params: dict[str, Any] | None = None,
        sign: bool = False,
    ) -> requests.Response:
        rel_path = str(path or "")
        if not rel_path.startswith("/"):
            rel_path = f"/{rel_path}"
        query = urlencode({k: v for k, v in (params or {}).items() if v is not None}, doseq=True)
        headers: dict[str, str] = {}
        if sign:
            headers.update(self._signed_headers(method=method, path=rel_path, body=body, query=query))
        request_kwargs: dict[str, Any] = {
            "method": str(method).upper(),
            "url": urljoin(f"{self.base_url}/", rel_path.lstrip("/")),
            "params": params,
            "headers": headers,
            "timeout": self.timeout_s,
        }
        if body is not None:
            request_kwargs["json"] = body
        response = self.session.request(**request_kwargs)
        if response.status_code >= 400:
            try:
                payload = response.json()
            except Exception:
                payload = response.text
            raise PublicValidatorApiError(
                f"{method.upper()} {rel_path} failed: HTTP {response.status_code} ({payload})",
                status_code=response.status_code,
            )
        return response

    def list_tasks(self) -> list[dict[str, Any]]:
        return list(self._request("GET", "/api/v1/tasks").json() or [])

    def list_pending_submissions(self, *, task_id: str | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"status": "pending_verification"}
        if task_id:
            params["task_id"] = str(task_id)
        return list(self._request("GET", "/api/v1/submissions", params=params).json() or [])

    def get_submission_detail(self, submission_id: str) -> dict[str, Any]:
        return dict(self._request("GET", f"/api/v1/submissions/{submission_id}/detail").json() or {})

    def claim_replay_job(
        self,
        *,
        task_id: str | None = None,
        task_slug: str | None = None,
        claim_path: str | None = DEFAULT_VALIDATOR_JOB_CLAIM_PATH,
    ) -> ReplayJob | None:
        resolved_task_id = self._resolve_task_id(task_id=task_id, task_slug=task_slug)
        if claim_path:
            body: dict[str, Any] = {}
            if resolved_task_id:
                body["task_id"] = resolved_task_id
            try:
                response = self._request("POST", claim_path, body=body or None, sign=True)
            except PublicValidatorApiError as exc:
                if exc.status_code == 404:
                    if claim_path == DEFAULT_VALIDATOR_JOB_CLAIM_PATH and "no pending validator jobs" not in str(exc):
                        raise
                    return None
                raise
            payload = response.json() if response.text else None
            if not payload:
                return None
            return self._job_from_payload(dict(payload), source=f"claim_endpoint:{claim_path}")
        return self._claim_from_pending_list(task_id=resolved_task_id)

    def submit_verification(
        self,
        *,
        submission_id: str,
        status: str,
        observed_metrics: dict[str, Any],
        notes: str | None = None,
        replay_log: str | None = None,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        if job_id:
            return self.submit_validator_job_result(
                job_id=job_id,
                submission_id=submission_id,
                status=status,
                observed_metrics=observed_metrics,
                notes=notes,
                replay_log=replay_log,
            )
        body: dict[str, Any] = {
            "status": str(status),
            "observed_metrics": dict(observed_metrics or {}),
        }
        if notes is not None:
            body["notes"] = str(notes)
        if replay_log is not None:
            body["replay_log"] = str(replay_log)
        return dict(
            self._request(
                "POST",
                f"/api/v1/submissions/{submission_id}/verify",
                body=body,
                sign=True,
            ).json()
            or {}
        )

    def submit_validator_job_result(
        self,
        *,
        job_id: str,
        submission_id: str,
        status: str,
        observed_metrics: dict[str, Any],
        notes: str | None = None,
        replay_log: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "submission_id": str(submission_id),
            "status": str(status),
            "observed_metrics": dict(observed_metrics or {}),
        }
        if notes is not None:
            body["notes"] = str(notes)
        if replay_log is not None:
            body["replay_log"] = str(replay_log)
        return dict(
            self._request(
                "POST",
                f"/api/v1/validator/jobs/{job_id}/result",
                body=body,
                sign=True,
            ).json()
            or {}
        )

    def _resolve_task_id(self, *, task_id: str | None, task_slug: str | None) -> str | None:
        wanted_id = str(task_id or "").strip()
        if wanted_id:
            return wanted_id
        wanted_slug = str(task_slug or "").strip().lower()
        if not wanted_slug:
            return None
        for task in self.list_tasks():
            if str(task.get("slug", "")).strip().lower() == wanted_slug:
                return str(task.get("id") or "")
        raise PublicValidatorApiError(f"no task found for slug: {task_slug}")

    def _claim_from_pending_list(self, *, task_id: str | None) -> ReplayJob | None:
        tasks_by_id = {str(row.get("id")): dict(row) for row in self.list_tasks()}
        candidates = sorted(
            self.list_pending_submissions(task_id=task_id),
            key=lambda row: str(row.get("created_at", "")),
        )
        for submission in candidates:
            submission_id = str(submission.get("id") or "").strip()
            if not submission_id:
                continue
            task = tasks_by_id.get(str(submission.get("task_id") or ""))
            if task is None:
                continue
            if str(task.get("competition_mode") or "").strip() == "peer_evaluation":
                continue
            detail = self.get_submission_detail(submission_id)
            return self._job_from_parts(
                submission=dict(detail.get("submission") or submission),
                task=task,
                detail=detail,
                replay_spec=dict(detail.get("replay_spec") or {}),
                source="pending_submissions",
            )
        return None

    def _job_from_payload(self, payload: dict[str, Any], *, source: str) -> ReplayJob | None:
        if str(payload.get("status") or "").strip().lower() in {"idle", "empty", "none"}:
            return None
        job = dict(payload.get("job") or payload)
        job_id = str(job.get("job_id") or job.get("id") or "").strip() or None
        detail = dict(job.get("detail") or {})
        submission = dict(job.get("submission") or detail.get("submission") or {})
        submission_id = str(job.get("submission_id") or submission.get("id") or "").strip()
        if not submission and submission_id:
            detail = self.get_submission_detail(submission_id)
            submission = dict(detail.get("submission") or {})
        if not submission_id:
            submission_id = str(submission.get("id") or "").strip()
        if not submission_id:
            raise PublicValidatorApiError("replay job payload is missing submission_id")

        replay_spec = dict(
            job.get("replay_spec")
            or detail.get("replay_spec")
            or submission.get("replay_spec")
            or {}
        )

        task = dict(job.get("task") or {})
        if not task and not replay_spec:
            task_id = str(submission.get("task_id") or job.get("task_id") or "").strip()
            if task_id:
                tasks_by_id = {str(row.get("id")): dict(row) for row in self.list_tasks()}
                task = tasks_by_id.get(task_id, {})

        return self._job_from_parts(
            submission=submission,
            task=task,
            detail=detail,
            replay_spec=replay_spec,
            source=source,
            job_id=job_id,
        )

    def _job_from_parts(
        self,
        *,
        submission: dict[str, Any],
        task: dict[str, Any],
        detail: dict[str, Any],
        replay_spec: dict[str, Any],
        source: str,
        job_id: str | None = None,
    ) -> ReplayJob:
        submission_id = str(submission.get("id") or "").strip()
        if not submission_id:
            raise PublicValidatorApiError("replay job is missing submission.id")
        if not replay_spec:
            replay_spec = self._task_replay_spec_fallback(submission=submission, task=task)
        return ReplayJob(
            job_id=job_id,
            submission_id=submission_id,
            submission=dict(submission),
            task=dict(task),
            replay_spec=dict(replay_spec),
            detail=dict(detail),
            source=source,
        )

    def _task_replay_spec_fallback(
        self,
        *,
        submission: dict[str, Any],
        task: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "repository": task.get("repository"),
            "base_ref": submission.get("base_ref") or task.get("base_ref"),
            "base_commit_sha": submission.get("base_commit_sha"),
            "setup_command": task.get("setup_command"),
            "benchmark_command": task.get("benchmark_command"),
            "result_path": task.get("result_path"),
            "allowed_patch_paths": list(task.get("allowed_patch_paths") or []),
            "metric_name": task.get("metric_name"),
            "metric_direction": task.get("metric_direction"),
            "ranking_mode": task.get("ranking_mode"),
            "secondary_metric_name": task.get("secondary_metric_name"),
            "secondary_metric_direction": task.get("secondary_metric_direction"),
            "validator_benchmark_env": dict(task.get("validator_benchmark_env") or {}),
            "time_budget_seconds": task.get("time_budget_seconds"),
            "_source": "task_response_fallback",
        }
