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

from validator.research_validator_client import ReplayJob


_NUMBER_PATTERN = r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
_PATCH_PATH_PATTERN = re.compile(r"^diff --git a/(.+?) b/(.+?)$", re.MULTILINE)
_DEFAULT_COMMAND_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
_PYTHON_BYTECODE_SUFFIXES = (".pyc", ".pyo")


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


def _dedupe_paths(paths: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        ordered.append(path)
    return ordered


def is_ignored_generated_python_cache_path(raw: str) -> bool:
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


def strip_ignored_generated_python_cache_diffs(patch: str) -> tuple[str, list[str]]:
    text = str(patch or "")
    if not text.strip():
        return text, []

    lines = text.splitlines(keepends=True)
    section_starts = [index for index, line in enumerate(lines) if line.startswith("diff --git ")]
    if not section_starts:
        paths = extract_patch_paths(text)
        ignored_paths = [path for path in paths if is_ignored_generated_python_cache_path(path)]
        if paths and len(ignored_paths) == len(paths):
            return "", _dedupe_paths(ignored_paths)
        return text, []

    preamble = lines[: section_starts[0]]
    kept_sections: list[str] = []
    ignored_paths: list[str] = []
    for offset, start in enumerate(section_starts):
        end = section_starts[offset + 1] if offset + 1 < len(section_starts) else len(lines)
        section = "".join(lines[start:end])
        section_paths = extract_patch_paths(section)
        section_ignored_paths = [
            path for path in section_paths if is_ignored_generated_python_cache_path(path)
        ]
        if section_paths and len(section_ignored_paths) == len(section_paths):
            ignored_paths.extend(section_ignored_paths)
            continue
        kept_sections.append(section)

    if not kept_sections:
        return "", _dedupe_paths(ignored_paths)
    return "".join(preamble) + "".join(kept_sections), _dedupe_paths(ignored_paths)


def find_disallowed_patch_paths(patch: str, allowed_patterns: list[str] | None) -> list[str]:
    changed_paths = [
        path for path in extract_patch_paths(patch) if not is_ignored_generated_python_cache_path(path)
    ]
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
    ) -> None:
        self.workspace_root = Path(workspace_root).expanduser().resolve()
        self.allow_unsafe_host_replay = bool(allow_unsafe_host_replay)
        self.allow_local_artifacts = bool(allow_local_artifacts)
        self.max_replay_log_chars = max(1024, int(max_replay_log_chars))

    def run(self, job: ReplayJob) -> ReplayResult:
        if not self.allow_unsafe_host_replay:
            raise PublicReplayError("host replay is disabled; pass --allow-unsafe-host-replay to run submitted code")
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
        benchmark_env = dict(spec.validator_benchmark_env)
        redactions = self._sensitive_redactions(benchmark_env)
        try:
            effective_patch, ignored_generated_paths = strip_ignored_generated_python_cache_diffs(patch)
            changed_paths = extract_patch_paths(effective_patch)
            if patch.strip() and not changed_paths and not ignored_generated_paths:
                return self._result(
                    job,
                    status="rejected",
                    observed_metrics={},
                    notes="Patch surface could not be determined from the submitted diff",
                    replay_log="\n".join([*logs, "validator_rejection=unparseable_patch_surface"]),
                )
            if ignored_generated_paths:
                logs.append("ignored_generated_patch_paths=" + ",".join(ignored_generated_paths))

            disallowed_paths = find_disallowed_patch_paths(effective_patch, spec.allowed_patch_paths)
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
            self._apply_patch(effective_patch, cwd=repo_dir)
            effective_submission = dict(submission, patch=effective_patch)
            self._materialize_artifact(
                effective_submission,
                repo_dir=repo_dir,
                benchmark_env=benchmark_env,
                required=self._artifact_required(effective_submission, benchmark_env),
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
                    notes=f"Setup failed in host replay | exit={setup_result.returncode}",
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
            result_path = repo_dir / spec.result_path
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
                notes=f"{notes_prefix} in host replay | exit={benchmark_result.returncode}",
                replay_log=replay_log,
            )
        if spec.secondary_metric_name and secondary_metric is None:
            return self._result(
                job,
                status="error",
                observed_metrics={},
                notes=f"Replay missing secondary metric '{spec.secondary_metric_name}' in host replay",
                replay_log=replay_log,
            )

        observed_metrics = {spec.metric_name: float(metric)}
        if spec.secondary_metric_name and secondary_metric is not None:
            observed_metrics[spec.secondary_metric_name] = float(secondary_metric)
        return self._result(
            job,
            status="accepted",
            observed_metrics=observed_metrics,
            notes=f"Replay succeeded in host replay | exit={benchmark_result.returncode}",
            replay_log=replay_log,
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
