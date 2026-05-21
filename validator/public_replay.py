from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
from typing import Any
from urllib.parse import urlparse
from urllib.request import url2pathname

import requests

from validator.replay_sandbox import (
    DockerSandboxConfig,
    SandboxCommandResult,
    SandboxError,
    run_replay_in_docker_sandbox,
)
from validator.research_validator_client import ReplayJob


_NUMBER_PATTERN = r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
_PATCH_PATH_PATTERN = re.compile(r"^diff --git a/(.+?) b/(.+?)$", re.MULTILINE)
_DEFAULT_COMMAND_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
_PYTHON_BYTECODE_SUFFIXES = (".pyc", ".pyo")
DEFAULT_MAX_PATCH_BYTES = 262_144
HELDOUT_SOURCES_JSON_ENV = "AUTORESEARCH_HELDOUT_SOURCES_JSON"
HELDOUT_DATASET_ENV = "AUTORESEARCH_HELDOUT_DATASET"
PRIVATE_HELDOUT_MANIFEST_ENV = "AUTORESEARCH_PRIVATE_HELDOUT_MANIFEST"
PRIVATE_HELDOUT_ROOT_ENV = "AUTORESEARCH_PRIVATE_HELDOUT_ROOT"
HELDOUT_SYNC_SLOT_ENV = "AUTORESEARCH_HELDOUT_SYNC_SLOT"
HELDOUT_SYNC_NUMBER_ENV = "AUTORESEARCH_HELDOUT_SYNC_NUMBER"
HF_TOKEN_ENV_KEYS = (
    "AUTORESEARCH_PRIVATE_HELDOUT_HF_TOKEN",
    "HF_TOKEN",
    "HUGGINGFACE_TOKEN",
)
_HELDOUT_MANIFEST_DIR = ".autoresearch-heldout"
_HELDOUT_MANIFEST_PATH = f"{_HELDOUT_MANIFEST_DIR}/manifest.json"


class PublicReplayError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ReplaySpec:
    repository: str
    base_ref: str
    base_commit_sha: str | None
    setup_command: str | None
    benchmark_command: str
    result_path: str | None
    allowed_patch_paths: list[str]
    metric_name: str
    secondary_metric_name: str | None
    validator_benchmark_env: dict[str, str]
    time_budget_seconds: int
    source: str
    max_patch_bytes: int

    @classmethod
    def from_job(cls, job: ReplayJob) -> "ReplaySpec":
        raw = dict(job.replay_spec or {})
        task = dict(job.task or {})
        submission = dict(job.submission or {})
        repository = str(raw.get("repository") or task.get("repository") or "").strip()
        base_ref = str(
            raw.get("base_ref")
            or raw.get("base_commit_sha")
            or submission.get("base_ref")
            or task.get("base_ref")
            or "HEAD"
        ).strip()
        benchmark_command = str(raw.get("benchmark_command") or task.get("benchmark_command") or "").strip()
        metric_name = str(raw.get("metric_name") or task.get("metric_name") or job.detail.get("metric_name") or "").strip()
        if not repository:
            raise PublicReplayError("replay spec is missing repository")
        if not benchmark_command:
            raise PublicReplayError("replay spec is missing benchmark_command")
        if not metric_name:
            raise PublicReplayError("replay spec is missing metric_name")

        raw_env = raw.get("validator_benchmark_env") or raw.get("benchmark_env") or task.get("validator_benchmark_env") or {}
        benchmark_env = {
            str(key): str(value)
            for key, value in dict(raw_env).items()
            if str(key).strip()
        }
        raw_allowed_paths = raw.get("allowed_patch_paths")
        allowed_patch_paths = raw_allowed_paths if raw_allowed_paths is not None else task.get("allowed_patch_paths")
        max_patch_bytes = raw.get("max_patch_bytes") or task.get("max_patch_bytes") or DEFAULT_MAX_PATCH_BYTES
        return cls(
            repository=repository,
            base_ref=base_ref,
            base_commit_sha=str(raw.get("base_commit_sha") or "").strip() or None,
            setup_command=str(raw.get("setup_command") or task.get("setup_command") or "").strip() or None,
            benchmark_command=benchmark_command,
            result_path=str(raw.get("result_path") or task.get("result_path") or "").strip() or None,
            allowed_patch_paths=[str(row) for row in list(allowed_patch_paths or [])],
            metric_name=metric_name,
            secondary_metric_name=str(raw.get("secondary_metric_name") or task.get("secondary_metric_name") or "").strip() or None,
            validator_benchmark_env=benchmark_env,
            time_budget_seconds=max(1, int(raw.get("time_budget_seconds") or task.get("time_budget_seconds") or 3600)),
            source=str(raw.get("_source") or job.source or "unknown"),
            max_patch_bytes=max(1, int(max_patch_bytes)),
        )


@dataclass(frozen=True, slots=True)
class ReplayResult:
    submission_id: str
    status: str
    observed_metrics: dict[str, float]
    notes: str
    replay_log: str


@dataclass(frozen=True, slots=True)
class _CommandResult:
    returncode: int
    stdout: str
    stderr: str


def parse_metric_from_text(text: str, metric_name: str) -> float | None:
    pattern = re.compile(rf"(?im)\b{re.escape(metric_name)}\b\s*[:=]\s*{_NUMBER_PATTERN}")
    matches = list(pattern.finditer(str(text)))
    if not matches:
        return None
    try:
        return float(matches[-1].group(1))
    except ValueError:
        return None


def load_metric_from_result_file(result_path: Path, metric_name: str) -> float:
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"result file not found: {result_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"result file is not valid JSON: {result_path}") from exc

    if isinstance(payload, dict):
        if payload.get("metric_name") == metric_name and "metric_value" in payload:
            return float(payload["metric_value"])
        metrics = payload.get("metrics")
        if isinstance(metrics, dict) and metric_name in metrics:
            return float(metrics[metric_name])
        if metric_name in payload:
            return float(payload[metric_name])
    raise ValueError(f"result file missing metric '{metric_name}': {result_path}")


def _seed_from_text(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _text_from_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "\n".join(part for part in (_text_from_value(item) for item in value) if part)
    if isinstance(value, dict):
        return "\n".join(
            part
            for part in (_text_from_value(value[key]) for key in sorted(value))
            if part
        )
    return str(value)


def _positive_int_value(value: Any, default: int, *, max_value: int | None = None) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    parsed = parsed if parsed > 0 else default
    if max_value is not None:
        parsed = min(parsed, max_value)
    return parsed


def _entry_hf_dataset(entry: dict[str, Any]) -> str:
    raw_dataset = entry.get("hf_dataset") or entry.get("dataset")
    if not isinstance(raw_dataset, str) or "/" not in raw_dataset:
        return ""
    return raw_dataset.strip()


def _entry_is_hf_rows(entry: Any) -> bool:
    if not isinstance(entry, dict):
        return False
    if str(entry.get("source") or "").strip().lower() in {"hf_rows", "huggingface_rows", "dataset_rows"}:
        return True
    return bool(
        _entry_hf_dataset(entry)
        and str(entry.get("split") or "").strip()
        and (entry.get("text_column") or entry.get("text_columns"))
        and not (entry.get("path") or entry.get("file") or entry.get("text"))
    )


def _hf_rows_offset(entry: dict[str, Any], *, limit: int, benchmark_env: dict[str, str]) -> int:
    base_offset = max(0, int(entry.get("offset") or 0))
    row_count = int(entry.get("row_count") or entry.get("num_rows_total") or 0)
    rotate = str(entry.get("rotate") or entry.get("offset_from_sync") or "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    sync_slot = str(benchmark_env.get(HELDOUT_SYNC_SLOT_ENV) or "").strip()
    sync_number = str(benchmark_env.get(HELDOUT_SYNC_NUMBER_ENV) or "").strip()
    if not rotate or row_count <= limit or not (sync_slot or sync_number):
        return base_offset
    max_offset = max(0, row_count - limit)
    seed = _seed_from_text(
        ":".join(
            [
                _entry_hf_dataset(entry),
                str(entry.get("config") or "default"),
                str(entry.get("split") or ""),
                str(base_offset),
                str(limit),
                sync_slot,
                sync_number,
            ]
        )
    )
    return seed % (max_offset + 1)


def _hf_token_from_env(benchmark_env: dict[str, str]) -> str:
    for key in HF_TOKEN_ENV_KEYS:
        token = str(benchmark_env.get(key) or "").strip()
        if token:
            return token
    return ""


def _fetch_hf_rows(
    *,
    dataset: str,
    config: str,
    split: str,
    offset: int,
    length: int,
    token: str,
) -> dict[str, Any]:
    params = {
        "dataset": dataset,
        "config": config,
        "split": split,
        "offset": str(max(0, int(offset))),
        "length": str(_positive_int_value(length, 1, max_value=100)),
    }
    headers = {"User-Agent": "bitsota-public-validator-heldout/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            response = requests.get(
                "https://datasets-server.huggingface.co/rows",
                params=params,
                headers=headers,
                timeout=60,
            )
            response.raise_for_status()
            return dict(response.json())
        except Exception as exc:
            last_exc = exc
            if attempt < 2:
                time.sleep(1.0 + attempt)
    raise RuntimeError(f"failed to fetch HF dataset rows for {dataset}/{config}/{split}") from last_exc


def _texts_from_hf_rows_entry(
    entry: dict[str, Any],
    *,
    benchmark_env: dict[str, str],
) -> list[str]:
    dataset = _entry_hf_dataset(entry)
    if not dataset:
        raise RuntimeError("HF rows heldout entry requires dataset or hf_dataset")
    config = str(entry.get("config") or entry.get("subset") or "default").strip() or "default"
    split = str(entry.get("split") or "").strip()
    if not split:
        raise RuntimeError("HF rows heldout entry requires split")
    raw_columns = entry.get("text_columns", entry.get("text_column"))
    columns = (
        [str(column).strip() for column in raw_columns if str(column).strip()]
        if isinstance(raw_columns, list)
        else [str(raw_columns or "").strip()]
    )
    columns = [column for column in columns if column]
    if not columns:
        raise RuntimeError("HF rows heldout entry requires text_column or text_columns")

    limit = _positive_int_value(entry.get("limit", entry.get("page_size", 100)), 100)
    page_size = _positive_int_value(entry.get("page_size", min(limit, 100)), min(limit, 100), max_value=100)
    offset = _hf_rows_offset(entry, limit=limit, benchmark_env=benchmark_env)
    min_chars = max(0, int(entry.get("min_chars") or 1))
    max_chars = max(1, int(entry.get("max_chars") or 6000))
    separator = str(entry.get("field_separator") or "\n\n")
    token = _hf_token_from_env(benchmark_env)

    texts: list[str] = []
    fetched = 0
    while fetched < limit:
        batch_size = min(page_size, limit - fetched)
        payload = _fetch_hf_rows(
            dataset=dataset,
            config=config,
            split=split,
            offset=offset + fetched,
            length=batch_size,
            token=token,
        )
        rows = payload.get("rows")
        if not isinstance(rows, list) or not rows:
            break
        for row_payload in rows:
            row = row_payload.get("row") if isinstance(row_payload, dict) else None
            if not isinstance(row, dict):
                continue
            parts = [_text_from_value(row.get(column)).strip() for column in columns]
            text = separator.join(part for part in parts if part).strip()
            if len(text) < min_chars:
                continue
            texts.append(text[:max_chars])
        fetched += len(rows)
        if len(rows) < batch_size:
            break
    if not texts:
        raise RuntimeError(f"HF rows heldout entry produced no text for {dataset}/{config}/{split}")
    return texts


def _normalize_patch_path(raw: str) -> str | None:
    path = str(raw or "").strip()
    if not path:
        return None
    path = path.split("\t", 1)[0].strip()
    if path in {"/dev/null", "dev/null"}:
        return None
    if path.startswith("a/") or path.startswith("b/"):
        path = path[2:]
    return path or None


def is_generated_python_cache_path(raw: str) -> bool:
    path = _normalize_patch_path(raw)
    if not path:
        return False
    normalized = path.replace("\\", "/").lstrip("./").lower()
    return normalized.endswith(_PYTHON_BYTECODE_SUFFIXES)


def extract_patch_paths(patch: str) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    for before_path, after_path in _PATCH_PATH_PATTERN.findall(str(patch or "")):
        for candidate in (before_path, after_path):
            path = _normalize_patch_path(candidate)
            if not path or path in seen:
                continue
            seen.add(path)
            paths.append(path)
    for raw_line in str(patch or "").splitlines():
        line = raw_line.rstrip()
        if not (line.startswith("--- ") or line.startswith("+++ ")):
            continue
        path = _normalize_patch_path(line[4:])
        if not path or path in seen:
            continue
        seen.add(path)
        paths.append(path)
    return paths


def find_disallowed_patch_paths(patch: str, allowed_patterns: list[str] | None) -> list[str]:
    changed_paths = extract_patch_paths(patch)
    if not changed_paths:
        return []
    if not allowed_patterns:
        return list(changed_paths)
    return [
        path
        for path in changed_paths
        if not any(fnmatch(path, pattern) for pattern in allowed_patterns)
    ]


class PublicReplayEngine:
    def __init__(
        self,
        *,
        workspace_root: Path,
        allow_unsafe_host_replay: bool = False,
        allow_local_artifacts: bool = False,
        max_replay_log_chars: int = 128_000,
        local_benchmark_env: dict[str, str] | None = None,
        replay_sandbox_mode: str = "host",
        docker_sandbox_config: DockerSandboxConfig | None = None,
    ) -> None:
        self.workspace_root = Path(workspace_root).expanduser().resolve()
        self.allow_unsafe_host_replay = bool(allow_unsafe_host_replay)
        self.allow_local_artifacts = bool(allow_local_artifacts)
        self.max_replay_log_chars = max(1024, int(max_replay_log_chars))
        self.local_benchmark_env = {
            str(key): str(value)
            for key, value in dict(local_benchmark_env or {}).items()
            if str(key).strip()
        }
        self.replay_sandbox_mode = str(replay_sandbox_mode or "host").strip().lower()
        self.docker_sandbox_config = docker_sandbox_config or DockerSandboxConfig()

    def run(self, job: ReplayJob) -> ReplayResult:
        if self.replay_sandbox_mode == "host" and not self.allow_unsafe_host_replay:
            raise PublicReplayError("host replay is disabled; pass --allow-unsafe-host-replay to run submitted code")
        if self.replay_sandbox_mode not in {"host", "docker"}:
            raise PublicReplayError(f"unsupported replay_sandbox_mode: {self.replay_sandbox_mode}")
        spec = ReplaySpec.from_job(job)
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        work_dir = Path(tempfile.mkdtemp(prefix=f"{job.submission_id}-", dir=str(self.workspace_root)))
        repo_dir = work_dir / "repo"
        logs: list[str] = [
            f"public_validator_job_source={job.source}",
            f"replay_spec_source={spec.source}",
        ]
        deadline = time.monotonic() + spec.time_budget_seconds
        try:
            return self._run_in_workspace(
                job=job,
                spec=spec,
                repo_dir=repo_dir,
                deadline=deadline,
                logs=logs,
            )
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    def _run_in_workspace(
        self,
        *,
        job: ReplayJob,
        spec: ReplaySpec,
        repo_dir: Path,
        deadline: float,
        logs: list[str],
    ) -> ReplayResult:
        submission = dict(job.submission or {})
        patch = str(submission.get("patch") or "")
        benchmark_env = {
            **dict(spec.validator_benchmark_env),
            **self.local_benchmark_env,
        }
        redactions = self._sensitive_redactions(benchmark_env)
        try:
            if len(patch.encode("utf-8")) > spec.max_patch_bytes:
                return self._result(
                    job,
                    status="rejected",
                    observed_metrics={},
                    notes=f"Patch exceeds maximum size of {spec.max_patch_bytes} bytes",
                    replay_log="\n".join(
                        [
                            *logs,
                            "validator_rejection=patch_too_large",
                            f"max_patch_bytes={spec.max_patch_bytes}",
                        ]
                    ),
                )

            changed_paths = extract_patch_paths(patch)
            if patch.strip() and not changed_paths:
                return self._result(
                    job,
                    status="rejected",
                    observed_metrics={},
                    notes="Patch surface could not be determined from the submitted diff",
                    replay_log="\n".join([*logs, "validator_rejection=unparseable_patch_surface"]),
                )

            generated_cache_paths = [
                path for path in changed_paths if is_generated_python_cache_path(path)
            ]
            if generated_cache_paths:
                return self._result(
                    job,
                    status="rejected",
                    observed_metrics={},
                    notes="Patch modified generated Python bytecode/cache files: "
                    + ", ".join(generated_cache_paths),
                    replay_log="\n".join(
                        [
                            *logs,
                            "validator_rejection=generated_python_cache_paths",
                            "changed_paths=" + ",".join(changed_paths),
                        ]
                    ),
                )

            disallowed_paths = find_disallowed_patch_paths(patch, spec.allowed_patch_paths)
            if disallowed_paths:
                return self._result(
                    job,
                    status="rejected",
                    observed_metrics={},
                    notes="Patch modified files outside the fixed competition surface: " + ", ".join(disallowed_paths),
                    replay_log="\n".join(
                        [
                            *logs,
                            "validator_rejection=disallowed_patch_paths",
                            "allowed_patch_paths=" + ",".join(spec.allowed_patch_paths),
                            "changed_paths=" + ",".join(changed_paths),
                        ]
                    ),
                )

            self._clone_repository(spec, dest=repo_dir)
            self._apply_patch(patch, cwd=repo_dir)
            self._materialize_artifact(
                submission,
                repo_dir=repo_dir,
                benchmark_env=benchmark_env,
                required=self._artifact_required(submission, benchmark_env),
                logs=logs,
            )
            self._materialize_backend_heldout(
                repo_dir=repo_dir,
                benchmark_env=benchmark_env,
                logs=logs,
            )
        except (subprocess.SubprocessError, OSError, RuntimeError) as exc:
            return self._result(
                job,
                status="error",
                observed_metrics={},
                notes=f"Replay preparation failed: {exc}",
                replay_log="\n".join(logs),
            )

        setup_result: _CommandResult | None = None
        result_file_path: Path | None = None
        replay_surface = "host replay"
        if self.replay_sandbox_mode == "docker":
            replay_surface = "docker sandbox"
            logs.extend(
                [
                    "replay_execution_surface=docker_sandbox",
                    f"docker_sandbox_image={self.docker_sandbox_config.image}",
                    f"docker_sandbox_gpus={self.docker_sandbox_config.gpus or 'none'}",
                ]
            )
            try:
                sandbox_result = run_replay_in_docker_sandbox(
                    repo_dir=repo_dir,
                    setup_command=spec.setup_command,
                    benchmark_command=spec.benchmark_command,
                    result_path=spec.result_path,
                    benchmark_env=benchmark_env,
                    work_dir=repo_dir.parent,
                    time_budget_seconds=self._remaining_time_seconds(deadline=deadline),
                    config=self.docker_sandbox_config,
                )
            except SandboxError as exc:
                return self._result(
                    job,
                    status="error",
                    observed_metrics={},
                    notes=f"Docker sandbox failed during {exc.stage}: {exc.message}",
                    replay_log="\n".join(logs),
                )
            setup_result = self._from_sandbox_command(sandbox_result.setup_result)
            benchmark_result = self._from_sandbox_command(sandbox_result.benchmark_result)
            result_file_path = sandbox_result.result_path
            if setup_result is not None:
                logs.extend(self._format_command_log("setup", setup_result, redactions=redactions))
                if setup_result.returncode != 0:
                    return self._result(
                        job,
                        status="error",
                        observed_metrics={},
                        notes=f"Setup failed in {replay_surface} | exit={setup_result.returncode}",
                        replay_log="\n".join(logs),
                    )
        else:
            logs.append("replay_execution_surface=host")
            if spec.setup_command:
                try:
                    setup_result = self._run_command(
                        spec.setup_command,
                        cwd=repo_dir,
                        timeout_seconds=self._remaining_time_seconds(deadline=deadline),
                        env=benchmark_env,
                    )
                except subprocess.TimeoutExpired:
                    return self._result(
                        job,
                        status="error",
                        observed_metrics={},
                        notes="Setup exceeded the fixed replay time budget",
                        replay_log="\n".join(logs),
                    )
                logs.extend(self._format_command_log("setup", setup_result, redactions=redactions))
                if setup_result.returncode != 0:
                    return self._result(
                        job,
                        status="error",
                        observed_metrics={},
                        notes=f"Setup failed in {replay_surface} | exit={setup_result.returncode}",
                        replay_log="\n".join(logs),
                    )

            try:
                benchmark_result = self._run_command(
                    spec.benchmark_command,
                    cwd=repo_dir,
                    timeout_seconds=self._remaining_time_seconds(deadline=deadline),
                    env=benchmark_env,
                )
            except subprocess.TimeoutExpired:
                return self._result(
                    job,
                    status="error",
                    observed_metrics={},
                    notes="Benchmark exceeded the fixed replay time budget",
                    replay_log="\n".join(logs),
                )

        logs.extend(self._format_command_log("benchmark", benchmark_result, redactions=redactions))
        combined_output = "\n".join(
            part for part in [benchmark_result.stdout.strip(), benchmark_result.stderr.strip()] if part
        )
        combined_output = self._redact_text(combined_output, redactions=redactions)
        replay_log = "\n".join(part for part in logs if part)

        metric = None
        secondary_metric = None
        if spec.result_path:
            result_path = result_file_path or repo_dir / spec.result_path
            try:
                metric = load_metric_from_result_file(result_path, spec.metric_name)
                if spec.secondary_metric_name:
                    secondary_metric = load_metric_from_result_file(result_path, spec.secondary_metric_name)
            except ValueError as exc:
                return self._result(
                    job,
                    status="error",
                    observed_metrics={},
                    notes=str(exc),
                    replay_log=replay_log,
                )
        if metric is None:
            metric = parse_metric_from_text(combined_output, spec.metric_name)
        if secondary_metric is None and spec.secondary_metric_name:
            secondary_metric = parse_metric_from_text(combined_output, spec.secondary_metric_name)

        if benchmark_result.returncode != 0 or metric is None:
            rejected_by_benchmark = "validator_rejection=" in combined_output
            status = "rejected" if rejected_by_benchmark else "error"
            notes_prefix = "Replay rejected" if rejected_by_benchmark else "Replay failed"
            return self._result(
                job,
                status=status,
                observed_metrics={},
                notes=f"{notes_prefix} in {replay_surface} | exit={benchmark_result.returncode}",
                replay_log=replay_log,
            )
        if spec.secondary_metric_name and secondary_metric is None:
            return self._result(
                job,
                status="error",
                observed_metrics={},
                notes=f"Replay missing secondary metric '{spec.secondary_metric_name}' in {replay_surface}",
                replay_log=replay_log,
            )

        observed_metrics = {spec.metric_name: float(metric)}
        if spec.secondary_metric_name and secondary_metric is not None:
            observed_metrics[spec.secondary_metric_name] = float(secondary_metric)
        return self._result(
            job,
            status="accepted",
            observed_metrics=observed_metrics,
            notes=f"Replay succeeded in {replay_surface} | exit={benchmark_result.returncode}",
            replay_log=replay_log,
        )

    def _from_sandbox_command(self, result: SandboxCommandResult | None) -> _CommandResult | None:
        if result is None:
            return None
        return _CommandResult(
            returncode=int(result.returncode),
            stdout=str(result.stdout or ""),
            stderr=str(result.stderr or ""),
        )

    def _clone_repository(self, spec: ReplaySpec, *, dest: Path) -> None:
        checkout_ref = spec.base_commit_sha or spec.base_ref
        subprocess.run(["git", "clone", "--quiet", spec.repository, str(dest)], check=True, capture_output=True, text=True)
        subprocess.run(["git", "checkout", checkout_ref], cwd=str(dest), check=True, capture_output=True, text=True)

    def _apply_patch(self, patch: str, *, cwd: Path) -> None:
        if not patch.strip():
            return
        subprocess.run(
            ["git", "apply", "-"],
            cwd=str(cwd),
            input=patch,
            check=True,
            capture_output=True,
            text=True,
        )

    def _artifact_required(self, submission: dict[str, Any], benchmark_env: dict[str, str]) -> bool:
        return (
            not str(submission.get("patch") or "").strip()
            or str(benchmark_env.get("AUTORESEARCH_REQUIRE_SUBMISSION_ARTIFACT") or "").strip().lower()
            in {"1", "true", "yes"}
        )

    def _materialize_artifact(
        self,
        submission: dict[str, Any],
        *,
        repo_dir: Path,
        benchmark_env: dict[str, str],
        required: bool,
        logs: list[str],
    ) -> None:
        artifact_uri = str(submission.get("artifact_uri") or "").strip()
        if not artifact_uri:
            if required:
                raise RuntimeError("submission artifact is required for this replay")
            return
        expected_sha256 = str(submission.get("artifact_sha256") or "").strip().lower() or None
        expected_size = submission.get("artifact_size_bytes")
        expected_size_bytes = int(expected_size) if expected_size is not None else None
        dest = repo_dir / ".autoresearch-submission" / "artifact"
        dest.parent.mkdir(parents=True, exist_ok=True)
        parsed = urlparse(artifact_uri)

        if parsed.scheme in {"http", "https"}:
            if parsed.scheme != "https":
                if required:
                    raise RuntimeError("required artifacts must use https")
                return
            if not expected_sha256 or expected_size_bytes is None:
                if required:
                    raise RuntimeError("remote submission artifact is missing artifact_sha256 or artifact_size_bytes")
                return
            artifact_bytes, digest = self._download_remote_artifact(
                artifact_uri,
                dest=dest,
                expected_size_bytes=expected_size_bytes,
            )
        elif parsed.scheme == "file" or not parsed.scheme:
            if not self.allow_local_artifacts:
                if required:
                    raise RuntimeError("local submission artifacts are disabled")
                return
            source = self._local_artifact_path(parsed_path=parsed.path, repo_dir=repo_dir)
            with source.open("rb") as handle:
                artifact_bytes, digest = self._copy_stream_with_hash(handle, dest)
        else:
            if required:
                raise RuntimeError(f"unsupported artifact_uri scheme: {parsed.scheme}")
            return

        if expected_size_bytes is not None and artifact_bytes != expected_size_bytes:
            dest.unlink(missing_ok=True)
            raise RuntimeError(f"artifact_size_bytes mismatch: expected {expected_size_bytes}, got {artifact_bytes}")
        if expected_sha256 is not None and digest != expected_sha256:
            dest.unlink(missing_ok=True)
            raise RuntimeError(f"artifact_sha256 mismatch: expected {expected_sha256}, got {digest}")

        artifact_path = dest.relative_to(repo_dir).as_posix()
        benchmark_env["AUTORESEARCH_SUBMISSION_ARTIFACT_PATH"] = artifact_path
        benchmark_env["AUTORESEARCH_SUBMISSION_ARTIFACT_URI"] = artifact_uri
        benchmark_env["AUTORESEARCH_SUBMISSION_ARTIFACT_BYTES"] = str(artifact_bytes)
        logs.extend(
            [
                f"submission_artifact_path={artifact_path}",
                f"submission_artifact_sha256={digest}",
                f"submission_artifact_bytes={artifact_bytes}",
            ]
        )

    def _materialize_backend_heldout(
        self,
        *,
        repo_dir: Path,
        benchmark_env: dict[str, str],
        logs: list[str],
    ) -> None:
        raw_sources = str(benchmark_env.get(HELDOUT_SOURCES_JSON_ENV) or "").strip()
        if not raw_sources:
            return
        try:
            payload = json.loads(raw_sources)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{HELDOUT_SOURCES_JSON_ENV} is invalid JSON") from exc
        entries = payload.get("documents") if isinstance(payload, dict) else payload
        if not isinstance(entries, list) or not entries:
            raise RuntimeError(f"{HELDOUT_SOURCES_JSON_ENV} must contain a non-empty documents list")

        documents: list[dict[str, str]] = []
        source_count = 0
        for entry in entries:
            if isinstance(entry, str):
                text = entry.strip()
                if text:
                    documents.append({"text": text})
                continue
            if not isinstance(entry, dict):
                continue
            category = str(entry.get("category") or entry.get("dataset") or "").strip()
            if _entry_is_hf_rows(entry):
                texts = _texts_from_hf_rows_entry(entry, benchmark_env=benchmark_env)
                source_count += 1
                for text in texts:
                    row: dict[str, str] = {"text": text}
                    if category:
                        row["category"] = category
                    documents.append(row)
                continue
            text = str(entry.get("text") or "").strip()
            if text:
                row = {"text": text}
                if category:
                    row["category"] = category
                documents.append(row)

        if not documents:
            raise RuntimeError(f"{HELDOUT_SOURCES_JSON_ENV} produced no heldout documents")

        heldout_dir = repo_dir / _HELDOUT_MANIFEST_DIR
        manifest_path = repo_dir / _HELDOUT_MANIFEST_PATH
        heldout_dir.mkdir(parents=True, exist_ok=True)
        manifest_payload = {
            "documents": documents,
            "metadata": {
                "source": "validator-prefetched-backend-heldout",
                "source_version": str(payload.get("version") or "") if isinstance(payload, dict) else "",
                "source_count": source_count,
            },
        }
        manifest_path.write_text(json.dumps(manifest_payload, separators=(",", ":")), encoding="utf-8")

        benchmark_env[PRIVATE_HELDOUT_MANIFEST_ENV] = _HELDOUT_MANIFEST_PATH
        benchmark_env[PRIVATE_HELDOUT_ROOT_ENV] = _HELDOUT_MANIFEST_DIR
        benchmark_env[HELDOUT_DATASET_ENV] = "validator-private-shard"
        benchmark_env.pop(HELDOUT_SOURCES_JSON_ENV, None)
        for key in HF_TOKEN_ENV_KEYS:
            benchmark_env.pop(key, None)

        digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        logs.extend(
            [
                "heldout_prefetch=ok",
                f"heldout_prefetch_sources={source_count}",
                f"heldout_prefetch_documents={len(documents)}",
                f"heldout_prefetch_manifest={_HELDOUT_MANIFEST_PATH}",
                f"heldout_prefetch_manifest_sha256={digest}",
            ]
        )

    def _download_remote_artifact(
        self,
        artifact_uri: str,
        *,
        dest: Path,
        expected_size_bytes: int,
    ) -> tuple[int, str]:
        response = requests.get(artifact_uri, stream=True, allow_redirects=False, timeout=60)
        if response.status_code >= 400:
            raise RuntimeError(f"submission artifact download failed: HTTP {response.status_code}")
        if response.status_code in {301, 302, 303, 307, 308}:
            raise RuntimeError("submission artifact redirects are not allowed")
        content_length = response.headers.get("Content-Length")
        if content_length:
            try:
                remote_size = int(content_length)
            except ValueError:
                remote_size = None
            if remote_size is not None and remote_size != expected_size_bytes:
                raise RuntimeError(f"artifact_size_bytes mismatch: expected {expected_size_bytes}, got {remote_size}")
        return self._copy_iter_with_hash(response.iter_content(chunk_size=1024 * 1024), dest)

    def _local_artifact_path(self, *, parsed_path: str, repo_dir: Path) -> Path:
        source = Path(url2pathname(parsed_path)).expanduser()
        if not source.is_absolute():
            source = repo_dir / source
        return source

    def _copy_stream_with_hash(self, source, dest: Path) -> tuple[int, str]:  # type: ignore[no-untyped-def]
        return self._copy_iter_with_hash(iter(lambda: source.read(1024 * 1024), b""), dest)

    def _copy_iter_with_hash(self, chunks, dest: Path) -> tuple[int, str]:  # type: ignore[no-untyped-def]
        total = 0
        digest = hashlib.sha256()
        with dest.open("wb") as handle:
            for chunk in chunks:
                if not chunk:
                    continue
                total += len(chunk)
                digest.update(chunk)
                handle.write(chunk)
        return total, digest.hexdigest()

    def _run_command(
        self,
        command: str,
        *,
        cwd: Path,
        timeout_seconds: int,
        env: dict[str, str] | None = None,
    ) -> _CommandResult:
        result = subprocess.run(
            command,
            cwd=str(cwd),
            shell=True,
            check=False,
            capture_output=True,
            text=True,
            env=self._command_env(cwd=cwd, env=env),
            timeout=max(1, int(timeout_seconds)),
        )
        return _CommandResult(result.returncode, result.stdout, result.stderr)

    def _command_env(self, *, cwd: Path, env: dict[str, str] | None) -> dict[str, str]:
        command_env = {
            "PATH": _DEFAULT_COMMAND_PATH,
            "HOME": str(cwd),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PYTHONNOUSERSITE": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        }
        for raw_key, raw_value in dict(env or {}).items():
            key = str(raw_key or "").strip()
            if key:
                command_env[key] = str(raw_value)
        return command_env

    def _remaining_time_seconds(self, *, deadline: float) -> int:
        remaining = math.ceil(deadline - time.monotonic())
        if remaining <= 0:
            raise subprocess.TimeoutExpired(cmd="replay", timeout=0)
        return remaining

    def _sensitive_redactions(self, env: dict[str, str]) -> list[str]:
        values: list[str] = []
        for value in dict(env or {}).values():
            text = str(value or "")
            if len(text) >= 4:
                values.append(text)
        return sorted(set(values), key=len, reverse=True)

    def _redact_text(self, text: str, *, redactions: list[str]) -> str:
        redacted = str(text or "")
        for value in redactions:
            redacted = redacted.replace(value, "[REDACTED]")
        return redacted

    def _format_command_log(
        self,
        stage: str,
        result: _CommandResult,
        *,
        redactions: list[str],
    ) -> list[str]:
        return [
            f"{stage} exit={result.returncode}",
            self._redact_text(result.stdout.strip(), redactions=redactions),
            self._redact_text(result.stderr.strip(), redactions=redactions),
        ]

    def _result(
        self,
        job: ReplayJob,
        *,
        status: str,
        observed_metrics: dict[str, float],
        notes: str,
        replay_log: str,
    ) -> ReplayResult:
        return ReplayResult(
            submission_id=job.submission_id,
            status=status,
            observed_metrics=dict(observed_metrics or {}),
            notes=str(notes or "")[: self.max_replay_log_chars],
            replay_log=str(replay_log or "")[: self.max_replay_log_chars],
        )
