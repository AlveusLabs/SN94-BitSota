from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import subprocess
import time
import uuid


SANDBOX_WORKDIR = "/workspace"
_MODULE_PATH = Path(__file__).resolve()
_ENV_KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True, slots=True)
class DockerSandboxConfig:
    image: str = "bitsota-research-validator-cuda:local"
    dockerfile: str = "docker/research-validator-cuda.Dockerfile"
    gpus: str = ""
    setup_network_mode: str = "bridge"
    benchmark_network_mode: str = "bridge"
    memory_limit: str = "16g"
    pids_limit: int = 512
    cpus: float = 0.0
    workspace_size_bytes: int = 2_147_483_648
    result_max_bytes: int = 10_485_760


@dataclass(frozen=True, slots=True)
class SandboxCommandResult:
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True, slots=True)
class SandboxedReplayResult:
    setup_result: SandboxCommandResult | None
    benchmark_result: SandboxCommandResult
    result_path: Path | None
    image: str


class SandboxError(RuntimeError):
    def __init__(self, stage: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage
        self.message = message


class SandboxTimeoutError(SandboxError):
    pass


def _docker(*args: str, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _format_output(stdout: str, stderr: str) -> str:
    return "\n".join(part for part in [str(stdout or "").strip(), str(stderr or "").strip()] if part)


def _run_docker_checked(*args: str, stage: str, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    try:
        result = _docker(*args, timeout=timeout)
    except FileNotFoundError as exc:
        raise SandboxError(stage, "docker command not found") from exc
    except subprocess.TimeoutExpired as exc:
        raise SandboxTimeoutError(stage, f"{stage} timed out after {timeout} seconds") from exc
    if result.returncode != 0:
        details = _format_output(result.stdout, result.stderr)
        raise SandboxError(stage, f"{stage} failed with exit={result.returncode}\n{details}".strip())
    return result


def _remaining_timeout_seconds(*, deadline: float, stage: str) -> int:
    remaining = int(deadline - time.monotonic())
    if remaining <= 0:
        raise SandboxTimeoutError(stage, f"{stage} could not start because the fixed replay time budget was already exhausted")
    return remaining


def _resolve_repo_root(*, dockerfile: str) -> tuple[Path, Path]:
    dockerfile_value = str(dockerfile or "").strip()
    dockerfile_path = Path(dockerfile_value).expanduser()
    if dockerfile_path.is_absolute():
        resolved = dockerfile_path.resolve()
        if not resolved.exists():
            raise SandboxError("image", f"validator sandbox Dockerfile not found: {resolved}")
        return resolved.parent, resolved

    candidate_roots: list[Path] = []
    env_root = str(os.getenv("BITSOTA_REPO_ROOT", "")).strip()
    if env_root:
        candidate_roots.append(Path(env_root).expanduser())
    candidate_roots.append(Path.cwd())
    candidate_roots.extend(_MODULE_PATH.parents)

    seen: set[Path] = set()
    for root in candidate_roots:
        resolved_root = root.resolve()
        if resolved_root in seen:
            continue
        seen.add(resolved_root)
        candidate = (resolved_root / dockerfile_path).resolve()
        if candidate.exists():
            return resolved_root, candidate

    searched = ", ".join(str(path) for path in seen)
    raise SandboxError(
        "image",
        f"validator sandbox Dockerfile not found: {dockerfile_value} (searched roots: {searched})",
    )


def _ensure_validator_image(config: DockerSandboxConfig) -> str:
    image = str(config.image or "").strip()
    if not image:
        raise SandboxError("image", "validator sandbox image is not configured")
    inspect_result = _docker("image", "inspect", image)
    if inspect_result.returncode == 0:
        return image
    dockerfile = str(config.dockerfile or "").strip()
    if not dockerfile:
        raise SandboxError("image", "validator sandbox Dockerfile is not configured")
    repo_root, dockerfile_path = _resolve_repo_root(dockerfile=dockerfile)
    _run_docker_checked(
        "build",
        "--tag",
        image,
        "--file",
        str(dockerfile_path),
        str(repo_root),
        stage="docker build",
    )
    return image


def _container_name(prefix: str) -> str:
    return f"bitsota-{prefix}-{uuid.uuid4().hex[:12]}"


def _create_workspace_volume(*, config: DockerSandboxConfig, timeout: int) -> str:
    volume_name = _container_name("replay-workspace")
    mount_opts = [f"size={max(1, int(config.workspace_size_bytes))}"]
    if hasattr(os, "getuid"):
        mount_opts.append(f"uid={os.getuid()}")
    if hasattr(os, "getgid"):
        mount_opts.append(f"gid={os.getgid()}")
    _run_docker_checked(
        "volume",
        "create",
        "--driver",
        "local",
        "--opt",
        "type=tmpfs",
        "--opt",
        "device=tmpfs",
        "--opt",
        "o=" + ",".join(mount_opts),
        volume_name,
        stage="docker volume create",
        timeout=timeout,
    )
    return volume_name


def _create_container(
    *,
    config: DockerSandboxConfig,
    image: str,
    workspace_volume: str,
    network_mode: str,
    timeout: int,
) -> str:
    command = [
        "create",
        "--init",
        "--hostname",
        "bitsota-validator-replay",
        "--read-only",
        "--workdir",
        SANDBOX_WORKDIR,
        "--name",
        _container_name("validator-replay"),
        "--network",
        str(network_mode or "none").strip() or "none",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges:true",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,exec,size=536870912",
        "--tmpfs",
        "/run:rw,nosuid,nodev,size=67108864",
        "--pids-limit",
        str(max(64, int(config.pids_limit))),
        "--mount",
        f"type=volume,source={workspace_volume},target={SANDBOX_WORKDIR}",
    ]
    gpus = str(config.gpus or "").strip()
    if gpus and gpus.lower() not in {"none", "false", "0", "off"}:
        command.extend(["--gpus", gpus])
    if hasattr(os, "getuid") and hasattr(os, "getgid"):
        command.extend(["--user", f"{os.getuid()}:{os.getgid()}"])
    if str(config.memory_limit or "").strip():
        command.extend(["--memory", str(config.memory_limit).strip()])
    if float(config.cpus or 0.0) > 0.0:
        command.extend(["--cpus", str(float(config.cpus))])
    command.extend([image, "sleep", "infinity"])
    created = _run_docker_checked(*command, stage="docker create", timeout=timeout)
    return created.stdout.strip()


def _start_container(container_id: str, *, timeout: int) -> None:
    _run_docker_checked("start", container_id, stage="docker start", timeout=timeout)


def _copy_repo_to_container(container_id: str, *, repo_dir: Path, timeout: int) -> None:
    _run_docker_checked(
        "cp",
        f"{repo_dir}/.",
        f"{container_id}:{SANDBOX_WORKDIR}",
        stage="docker cp input",
        timeout=timeout,
    )


def _wrap_workspace_command(command: str) -> str:
    return "\n".join(
        [
            "mkdir -p /workspace/.tmp /workspace/.home",
            "export HOME=/workspace/.home",
            "export TMPDIR=/workspace/.tmp",
            "export PIP_CACHE_DIR=/workspace/.tmp/pip-cache",
            "export UV_CACHE_DIR=/workspace/.tmp/uv-cache",
            "if [ ! -x /workspace/.venv/bin/python ]; then python -m venv --system-site-packages /workspace/.venv; fi",
            "export VIRTUAL_ENV=/workspace/.venv",
            "export PATH=/workspace/.venv/bin:$PATH",
            command,
        ]
    )


def _exec_command_in_container(
    container_id: str,
    *,
    stage: str,
    command: str,
    env: dict[str, str] | None,
    timeout_seconds: int,
) -> SandboxCommandResult:
    env_args: list[str] = []
    for raw_key, raw_value in dict(env or {}).items():
        key = str(raw_key or "").strip()
        if not key or not _ENV_KEY_PATTERN.fullmatch(key):
            continue
        env_args.extend(["--env", f"{key}={raw_value}"])
    try:
        result = _docker(
            "exec",
            "--workdir",
            SANDBOX_WORKDIR,
            "--env",
            "HOME=/workspace/.home",
            *env_args,
            container_id,
            "bash",
            "-lc",
            _wrap_workspace_command(command),
            timeout=max(1, int(timeout_seconds)),
        )
    except FileNotFoundError as exc:
        raise SandboxError(stage, "docker command not found") from exc
    except subprocess.TimeoutExpired as exc:
        raise SandboxTimeoutError(stage, f"{stage} timed out after {timeout_seconds} seconds") from exc
    return SandboxCommandResult(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)


def _copy_result_from_container_if_present(
    container_id: str,
    *,
    result_path: str,
    dest: Path,
    max_bytes: int,
    timeout: int,
) -> tuple[Path | None, str]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    remote_path = result_path.lstrip("/")
    result = _docker(
        "cp",
        f"{container_id}:{SANDBOX_WORKDIR}/{remote_path}",
        str(dest),
        timeout=timeout,
    )
    if result.returncode == 0:
        if not dest.exists() or not dest.is_file():
            if dest.exists():
                if dest.is_dir():
                    for child in dest.iterdir():
                        if child.is_file():
                            child.unlink()
                else:
                    dest.unlink(missing_ok=True)
            return None, "validator_result_copy_error=result_path must resolve to a regular file"
        max_result_bytes = max(1, int(max_bytes))
        if dest.stat().st_size > max_result_bytes:
            dest.unlink(missing_ok=True)
            return None, f"validator_result_copy_error=result file exceeds {max_result_bytes} bytes"
        return dest, ""
    details = _format_output(result.stdout, result.stderr)
    return None, f"validator_result_copy_error={details}"


def _remove_container(container_id: str | None) -> None:
    if container_id:
        _docker("rm", "--force", container_id)


def _remove_volume(volume_name: str | None) -> None:
    if volume_name:
        _docker("volume", "rm", "--force", volume_name)


def run_replay_in_docker_sandbox(
    *,
    repo_dir: Path,
    setup_command: str | None,
    benchmark_command: str,
    result_path: str | None,
    benchmark_env: dict[str, str] | None,
    work_dir: Path,
    time_budget_seconds: int,
    config: DockerSandboxConfig,
) -> SandboxedReplayResult:
    image = _ensure_validator_image(config)
    workspace_volume: str | None = None
    setup_container: str | None = None
    benchmark_container: str | None = None
    deadline = time.monotonic() + max(1, int(time_budget_seconds))
    try:
        workspace_volume = _create_workspace_volume(
            config=config,
            timeout=_remaining_timeout_seconds(deadline=deadline, stage="docker volume create"),
        )
        setup_container = _create_container(
            config=config,
            image=image,
            workspace_volume=workspace_volume,
            network_mode=config.setup_network_mode,
            timeout=_remaining_timeout_seconds(deadline=deadline, stage="docker create"),
        )
        _start_container(setup_container, timeout=_remaining_timeout_seconds(deadline=deadline, stage="docker start"))
        _copy_repo_to_container(
            setup_container,
            repo_dir=repo_dir,
            timeout=_remaining_timeout_seconds(deadline=deadline, stage="docker cp input"),
        )

        setup_result: SandboxCommandResult | None = None
        if setup_command:
            setup_result = _exec_command_in_container(
                setup_container,
                stage="setup",
                command=setup_command,
                env=None,
                timeout_seconds=_remaining_timeout_seconds(deadline=deadline, stage="setup"),
            )
            if setup_result.returncode != 0:
                return SandboxedReplayResult(
                    setup_result=setup_result,
                    benchmark_result=SandboxCommandResult(returncode=1, stdout="", stderr="setup failed"),
                    result_path=None,
                    image=image,
                )

        benchmark_container = _create_container(
            config=config,
            image=image,
            workspace_volume=workspace_volume,
            network_mode=config.benchmark_network_mode,
            timeout=_remaining_timeout_seconds(deadline=deadline, stage="docker create"),
        )
        _start_container(benchmark_container, timeout=_remaining_timeout_seconds(deadline=deadline, stage="docker start"))
        benchmark_result = _exec_command_in_container(
            benchmark_container,
            stage="benchmark",
            command=benchmark_command,
            env=benchmark_env,
            timeout_seconds=_remaining_timeout_seconds(deadline=deadline, stage="benchmark"),
        )

        copied_result_path = None
        copy_diagnostic = ""
        if result_path:
            copied_result_path, copy_diagnostic = _copy_result_from_container_if_present(
                benchmark_container,
                result_path=result_path,
                dest=work_dir / "sandbox-results" / Path(result_path).name,
                max_bytes=int(config.result_max_bytes),
                timeout=_remaining_timeout_seconds(deadline=deadline, stage="docker cp result"),
            )
        if copy_diagnostic:
            benchmark_result = SandboxCommandResult(
                returncode=benchmark_result.returncode,
                stdout=benchmark_result.stdout,
                stderr=_format_output(benchmark_result.stderr, copy_diagnostic),
            )
        return SandboxedReplayResult(
            setup_result=setup_result,
            benchmark_result=benchmark_result,
            result_path=copied_result_path,
            image=image,
        )
    finally:
        _remove_container(benchmark_container)
        _remove_container(setup_container)
        _remove_volume(workspace_volume)
