from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import urlencode, urljoin

import requests

from miner.research_auth import sign_hotkey_request


class CoordinatorApiError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class CoordinatorSelection:
    task: dict[str, Any]
    work_item: dict[str, Any] | None = None


class CoordinatorClient:
    def __init__(
        self,
        *,
        base_url: str,
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
            raise CoordinatorApiError("wallet hotkey is unavailable")
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
            raise CoordinatorApiError("wallet hotkey signer is unavailable")
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
            raise CoordinatorApiError(f"{method.upper()} {rel_path} failed: HTTP {response.status_code} ({payload})")
        return response

    def request(
        self,
        method: str,
        path: str,
        *,
        body: Any = None,
        params: dict[str, Any] | None = None,
        sign: bool = False,
    ) -> requests.Response:
        return self._request(
            method,
            path,
            body=body,
            params=params,
            sign=sign,
        )

    def list_tasks(self) -> list[dict[str, Any]]:
        return list(self._request("GET", "/api/v1/tasks").json() or [])

    def list_work_items(
        self,
        *,
        task_id: str | None = None,
        status: str | None = "open",
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {}
        if task_id:
            params["task_id"] = str(task_id)
        if status:
            params["status"] = str(status)
        return list(self._request("GET", "/api/v1/work-items", params=params).json() or [])

    def list_claims(
        self,
        *,
        task_id: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {}
        if task_id:
            params["task_id"] = str(task_id)
        if status:
            params["status"] = str(status)
        return list(self._request("GET", "/api/v1/claims", params=params).json() or [])

    def get_onboard_markdown(self, task_id: str) -> str:
        return str(self._request("GET", f"/api/v1/tasks/{task_id}/onboard.md").text or "")

    def list_submissions(
        self,
        *,
        task_id: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {}
        if task_id:
            params["task_id"] = str(task_id)
        if status:
            params["status"] = str(status)
        return list(self._request("GET", "/api/v1/submissions", params=params).json() or [])

    def get_submission_detail(self, submission_id: str) -> dict[str, Any]:
        return dict(self._request("GET", f"/api/v1/submissions/{submission_id}/detail").json() or {})

    def claim_task(
        self,
        *,
        task_id: str,
        claim_description: str,
        base_submission_id: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"claim_description": str(claim_description)}
        if base_submission_id:
            body["base_submission_id"] = str(base_submission_id)
        return dict(
            self._request(
                "POST",
                f"/api/v1/tasks/{task_id}/claim",
                body=body,
                sign=True,
            ).json()
            or {}
        )

    def claim_work_item(
        self,
        *,
        work_item_id: str,
        claim_description: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {}
        if claim_description:
            body["claim_description"] = str(claim_description)
        return dict(
            self._request(
                "POST",
                f"/api/v1/work-items/{work_item_id}/claim",
                body=body,
                sign=True,
            ).json()
            or {}
        )

    def cancel_claim(self, *, claim_id: str) -> dict[str, Any]:
        return dict(
            self._request(
                "POST",
                f"/api/v1/claims/{claim_id}/cancel",
                sign=True,
            ).json()
            or {}
        )

    def submit_submission(
        self,
        *,
        claim_id: str,
        base_ref: str,
        patch: str,
        summary: str,
        claimed_metrics: dict[str, Any],
        proposed_idea: str | None = None,
        implemented_submission_id: str | None = None,
        artifact_uri: str | None = None,
        artifact_sha256: str | None = None,
        artifact_size_bytes: int | None = None,
        execution_log: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "claim_id": str(claim_id),
            "base_ref": str(base_ref),
            "patch": str(patch),
            "summary": str(summary),
            "claimed_metrics": dict(claimed_metrics or {}),
        }
        if proposed_idea:
            body["proposed_idea"] = str(proposed_idea)
        if implemented_submission_id:
            body["implemented_submission_id"] = str(implemented_submission_id)
        if artifact_uri:
            body["artifact_uri"] = str(artifact_uri)
        if artifact_sha256:
            body["artifact_sha256"] = str(artifact_sha256).strip().lower()
        if artifact_size_bytes is not None:
            body["artifact_size_bytes"] = int(artifact_size_bytes)
        if execution_log:
            body["execution_log"] = str(execution_log)
        return dict(self._request("POST", "/api/v1/submissions", body=body, sign=True).json() or {})

    def cancel_claim(self, *, claim_id: str) -> dict[str, Any]:
        return dict(
            self._request(
                "POST",
                f"/api/v1/claims/{claim_id}/cancel",
                sign=True,
            ).json()
            or {}
        )

    def list_peer_evaluations(
        self,
        *,
        submission_id: str | None = None,
        evaluator_hotkey: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {}
        if submission_id:
            params["submission_id"] = str(submission_id)
        if evaluator_hotkey:
            params["evaluator_hotkey"] = str(evaluator_hotkey)
        if status:
            params["status"] = str(status)
        return list(self._request("GET", "/api/v1/peer-evaluations", params=params).json() or [])

    def get_peer_consensus(self, submission_id: str) -> dict[str, Any]:
        return dict(self._request("GET", f"/api/v1/submissions/{submission_id}/peer-consensus").json() or {})

    def peer_evaluate_submission(
        self,
        *,
        submission_id: str,
        status: str,
        observed_metrics: dict[str, Any],
        notes: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "status": str(status),
            "observed_metrics": dict(observed_metrics or {}),
        }
        if notes:
            body["notes"] = str(notes)
        return dict(
            self._request(
                "POST",
                f"/api/v1/submissions/{submission_id}/peer-evaluate",
                body=body,
                sign=True,
            ).json()
            or {}
        )

    def select_task(
        self,
        *,
        task_id: str | None = None,
        task_slug: str | None = None,
        participation_style: str,
    ) -> CoordinatorSelection:
        tasks = self.list_tasks()
        wanted_id = str(task_id or "").strip()
        wanted_slug = str(task_slug or "").strip().lower()
        if wanted_id:
            matches = [task for task in tasks if str(task.get("id")) == wanted_id]
        elif wanted_slug:
            matches = [task for task in tasks if str(task.get("slug", "")).strip().lower() == wanted_slug]
        else:
            matches = list(tasks)
        if not matches:
            raise CoordinatorApiError("no matching coordinator task found")
        task = dict(matches[0])
        if str(participation_style) != "pool":
            return CoordinatorSelection(task=task)
        claimed_items = self.list_work_items(task_id=str(task.get("id")), status="claimed")
        if claimed_items:
            active_claims = self.list_claims(task_id=str(task.get("id")), status="active")
            active_claims_by_id = {str(row.get("id")): dict(row) for row in active_claims}
            resumable_items = [
                dict(row)
                for row in claimed_items
                if str(active_claims_by_id.get(str(row.get("claim_id")) or "", {}).get("miner_hotkey") or "")
                == self.hotkey
            ]
            if resumable_items:
                ordered = sorted(
                    resumable_items,
                    key=lambda row: (-int(row.get("priority", 0) or 0), str(row.get("created_at", ""))),
                )
                return CoordinatorSelection(task=task, work_item=ordered[0])
        work_items = self.list_work_items(task_id=str(task.get("id")), status="open")
        if not work_items:
            raise CoordinatorApiError("no open work items available for the selected task")
        ordered = sorted(
            work_items,
            key=lambda row: (-int(row.get("priority", 0) or 0), str(row.get("created_at", ""))),
        )
        return CoordinatorSelection(task=task, work_item=dict(ordered[0]))
