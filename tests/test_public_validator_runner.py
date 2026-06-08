from __future__ import annotations

import hashlib
import json
from pathlib import Path
import py_compile
import subprocess
from types import SimpleNamespace

import requests
from substrateinterface import Keypair

from validator import public_replay as public_replay_module
from validator import replay_sandbox as replay_sandbox_module
from validator.public_replay import PublicReplayEngine, ReplayResult, load_numeric_metrics_from_result_file
from validator.replay_sandbox import (
    DockerSandboxConfig,
    SandboxCommandResult,
    SandboxedReplayResult,
)
from validator.research_validator_client import (
    AutoresearchValidatorClient,
    DEFAULT_VALIDATOR_JOB_CLAIM_PATH,
    DEFAULT_VALIDATOR_WORKLIST_PATH,
    ReplayJob,
)
from validator.research_validator_runner import (
    PublicValidatorRunOutcome,
    PublicValidatorRunner,
    PublicValidatorRunnerConfig,
    _build_parser,
    _config_from_args,
    _load_runner_config_file,
    _wallet_kwargs_from_args,
)


class _Response:
    def __init__(self, payload, *, status_code: int = 200, text: str | None = None) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = text if text is not None else "json"

    def json(self):
        return self._payload


class _Session:
    def __init__(self) -> None:
        self.requests: list[dict] = []

    def request(self, **kwargs):
        self.requests.append(kwargs)
        return _Response({"id": "verification-1", "status": "accepted"})


def _init_git_repo(repo_dir: Path) -> None:
    subprocess.run(["git", "init"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_dir, check=True, capture_output=True)


def _compile_python_cache(repo_dir: Path, relative_path: str) -> Path:
    source_path = repo_dir / relative_path
    py_compile.compile(str(source_path), doraise=True)
    pycache_dir = source_path.parent / "__pycache__"
    candidates = sorted(pycache_dir.glob(f"{source_path.stem}*.pyc"))
    if not candidates:
        raise AssertionError(f"expected pycache artifact for {relative_path}")
    return candidates[-1]


def test_result_file_metric_name_value_does_not_emit_metric_value_key(tmp_path: Path) -> None:
    result_path = tmp_path / "result.json"
    result_path.write_text('{"metric_name": "score", "metric_value": 2.5}', encoding="utf-8")

    assert load_numeric_metrics_from_result_file(result_path) == {"score": 2.5}


def test_validator_client_posts_signed_verification() -> None:
    keypair = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
    wallet = SimpleNamespace(hotkey=keypair)
    session = _Session()
    client = AutoresearchValidatorClient(
        base_url="http://127.0.0.1:8000",
        wallet=wallet,
        session=session,  # type: ignore[arg-type]
    )

    result = client.submit_verification(
        submission_id="submission-1",
        status="accepted",
        observed_metrics={"score": 2.5},
        notes="ok",
        replay_log="score=2.5",
    )

    assert result["status"] == "accepted"
    assert len(session.requests) == 1
    request = session.requests[0]
    assert request["method"] == "POST"
    assert request["url"] == "http://127.0.0.1:8000/api/v1/submissions/submission-1/verify"
    assert request["json"]["observed_metrics"] == {"score": 2.5}
    assert request["headers"]["X-Hotkey"] == keypair.ss58_address
    assert request["headers"]["X-Signature"]
    assert request["headers"]["Content-Type"] == "application/json"


def test_validator_client_claims_signed_backend_job() -> None:
    keypair = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
    wallet = SimpleNamespace(hotkey=keypair)

    class _JobSession:
        def __init__(self) -> None:
            self.requests: list[dict] = []

        def request(self, **kwargs):
            self.requests.append(kwargs)
            return _Response(
                {
                    "job_id": "job-1",
                    "submission": {
                        "id": "submission-1",
                        "task_id": "task-1",
                        "base_ref": "main",
                        "patch": "",
                    },
                    "replay_spec": {
                        "repository": "https://github.com/example/repo",
                        "base_ref": "main",
                        "benchmark_command": "python benchmark.py",
                        "metric_name": "score",
                        "time_budget_seconds": 60,
                    },
                }
            )

    session = _JobSession()
    client = AutoresearchValidatorClient(
        base_url="http://127.0.0.1:8000",
        wallet=wallet,
        session=session,  # type: ignore[arg-type]
    )

    job = client.claim_replay_job(task_id="task-1", claim_path=DEFAULT_VALIDATOR_JOB_CLAIM_PATH)

    assert job is not None
    assert job.job_id == "job-1"
    assert job.submission_id == "submission-1"
    assert job.replay_spec["metric_name"] == "score"
    assert len(session.requests) == 1
    request = session.requests[0]
    assert request["method"] == "POST"
    assert request["url"] == "http://127.0.0.1:8000/api/v1/validator/jobs/claim"
    assert request["json"] == {"task_id": "task-1"}
    assert request["headers"]["X-Hotkey"] == keypair.ss58_address
    assert request["headers"]["X-Signature"]


def test_validator_client_scans_signed_worklist() -> None:
    keypair = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
    wallet = SimpleNamespace(hotkey=keypair)

    class _WorklistSession:
        def __init__(self) -> None:
            self.requests: list[dict] = []

        def request(self, **kwargs):
            self.requests.append(kwargs)
            return _Response(
                {
                    "jobs": [
                        {
                            "job_id": "job-1",
                            "submission": {
                                "id": "submission-1",
                                "task_id": "task-1",
                                "base_ref": "main",
                                "patch": "",
                            },
                            "replay_spec": {
                                "repository": "https://github.com/example/repo",
                                "base_ref": "main",
                                "benchmark_command": "python benchmark.py",
                                "metric_name": "score",
                                "time_budget_seconds": 60,
                            },
                        },
                        {
                            "job_id": "job-2",
                            "submission": {
                                "id": "submission-2",
                                "task_id": "task-1",
                                "base_ref": "main",
                                "patch": "",
                            },
                            "replay_spec": {
                                "repository": "https://github.com/example/repo",
                                "base_ref": "main",
                                "benchmark_command": "python benchmark.py",
                                "metric_name": "score",
                                "time_budget_seconds": 60,
                            },
                        },
                    ]
                }
            )

    session = _WorklistSession()
    client = AutoresearchValidatorClient(
        base_url="http://127.0.0.1:8000",
        wallet=wallet,
        session=session,  # type: ignore[arg-type]
    )

    jobs = client.claim_replay_jobs(task_id="task-1")

    assert [job.job_id for job in jobs] == ["job-1", "job-2"]
    assert [job.submission_id for job in jobs] == ["submission-1", "submission-2"]
    assert len(session.requests) == 1
    request = session.requests[0]
    assert request["method"] == "POST"
    assert request["url"] == "http://127.0.0.1:8000/api/v1/validator/submissions/scan"
    assert request["json"] == {"task_id": "task-1"}
    assert request["headers"]["X-Hotkey"] == keypair.ss58_address
    assert request["headers"]["X-Signature"]


def test_public_validator_runner_loads_yaml_config(tmp_path: Path) -> None:
    config_path = tmp_path / "research-validator.yaml"
    workspace_root = tmp_path / "workspaces"
    wallet_path = tmp_path / "wallets"
    config_path.write_text(
        f"""
coordinator_url: "https://validator.example.test/"
task_slug: "qwen-task"
claim_path: "/api/v1/validator/submissions/scan"
workspace_root: "{workspace_root}"
once: true
interval_seconds: 12.5
timeout_s: 7
allow_unsafe_host_replay: true
allow_local_artifacts: true
max_replay_log_chars: 4096
replay_sandbox_mode: "docker"
replay_sandbox_image: "bitsota-test-validator:local"
replay_sandbox_dockerfile: "docker/test.Dockerfile"
replay_sandbox_gpus: "all"
replay_sandbox_setup_network_mode: "bridge"
replay_sandbox_benchmark_network_mode: "none"
replay_sandbox_memory_limit: "12g"
replay_sandbox_pids_limit: 256
replay_sandbox_cpus: 2.5
replay_sandbox_workspace_size_bytes: 123456
replay_sandbox_result_max_bytes: 654321
dry_run: true
local_benchmark_env:
  AUTORESEARCH_PRIVATE_HELDOUT_MANIFEST: "{tmp_path / 'heldout.json'}"
  AUTORESEARCH_PRIVATE_HELDOUT_ROOT: "{tmp_path}"
wallet_name: "validator-wallet"
wallet_hotkey: "validator-hotkey"
wallet_path: "{wallet_path}"
""".lstrip(),
        encoding="utf-8",
    )
    args = _build_parser().parse_args(["--config", str(config_path)])
    config_data = _load_runner_config_file(args.config)

    config = _config_from_args(args, config_data)
    wallet_kwargs = _wallet_kwargs_from_args(args, config_data)

    assert config.coordinator_url == "https://validator.example.test"
    assert config.task_slug == "qwen-task"
    assert config.claim_path == DEFAULT_VALIDATOR_WORKLIST_PATH
    assert config.workspace_root == workspace_root.resolve()
    assert config.cycles == 1
    assert config.interval_seconds == 12.5
    assert config.timeout_s == 7.0
    assert config.allow_unsafe_host_replay is True
    assert config.allow_local_artifacts is True
    assert config.max_replay_log_chars == 4096
    assert config.replay_sandbox_mode == "docker"
    assert config.replay_sandbox_image == "bitsota-test-validator:local"
    assert config.replay_sandbox_dockerfile == "docker/test.Dockerfile"
    assert config.replay_sandbox_gpus == "all"
    assert config.replay_sandbox_setup_network_mode == "bridge"
    assert config.replay_sandbox_benchmark_network_mode == "none"
    assert config.replay_sandbox_memory_limit == "12g"
    assert config.replay_sandbox_pids_limit == 256
    assert config.replay_sandbox_cpus == 2.5
    assert config.replay_sandbox_workspace_size_bytes == 123456
    assert config.replay_sandbox_result_max_bytes == 654321
    assert config.local_benchmark_env == {
        "AUTORESEARCH_PRIVATE_HELDOUT_MANIFEST": str(tmp_path / "heldout.json"),
        "AUTORESEARCH_PRIVATE_HELDOUT_ROOT": str(tmp_path),
    }
    assert config.dry_run is True
    assert wallet_kwargs == {
        "hotkey_mnemonic": "",
        "wallet_file": "",
        "wallet_name": "validator-wallet",
        "wallet_hotkey": "validator-hotkey",
        "wallet_path": str(wallet_path),
    }


def test_public_validator_runner_cli_overrides_yaml_config(tmp_path: Path) -> None:
    config_path = tmp_path / "research-validator.config"
    config_path.write_text(
        """
coordinator_url: "https://config.example.test"
cycles: 9
interval_seconds: 60
dry_run: true
allow_unsafe_host_replay: true
""".lstrip(),
        encoding="utf-8",
    )
    args = _build_parser().parse_args(
        [
            "--config",
            str(config_path),
            "--coordinator-url",
            "https://cli.example.test/",
            "--cycles",
            "2",
            "--interval-seconds",
            "5",
            "--no-dry-run",
            "--claim-path",
            DEFAULT_VALIDATOR_WORKLIST_PATH,
        ]
    )
    config = _config_from_args(args, _load_runner_config_file(args.config))

    assert config.coordinator_url == "https://cli.example.test"
    assert config.cycles == 2
    assert config.interval_seconds == 5.0
    assert config.dry_run is False
    assert config.claim_path == DEFAULT_VALIDATOR_WORKLIST_PATH


def test_public_validator_runner_defaults_to_signed_worklist() -> None:
    args = _build_parser().parse_args([])

    config = _config_from_args(args)

    assert config.coordinator_url == "https://autoresearch.bitsota.com"
    assert config.claim_path == DEFAULT_VALIDATOR_WORKLIST_PATH
    assert config.replay_sandbox_setup_network_mode == "none"
    assert config.replay_sandbox_benchmark_network_mode == "none"


def test_validator_client_posts_signed_job_result() -> None:
    keypair = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
    wallet = SimpleNamespace(hotkey=keypair)
    session = _Session()
    client = AutoresearchValidatorClient(
        base_url="http://127.0.0.1:8000",
        wallet=wallet,
        session=session,  # type: ignore[arg-type]
    )

    result = client.submit_verification(
        job_id="job-1",
        submission_id="submission-1",
        status="accepted",
        observed_metrics={"score": 2.5},
        notes="ok",
        replay_log="score=2.5",
    )

    assert result["status"] == "accepted"
    assert len(session.requests) == 1
    request = session.requests[0]
    assert request["method"] == "POST"
    assert request["url"] == "http://127.0.0.1:8000/api/v1/validator/jobs/job-1/result"
    assert request["json"]["submission_id"] == "submission-1"
    assert request["json"]["observed_metrics"] == {"score": 2.5}
    assert request["headers"]["X-Hotkey"] == keypair.ss58_address
    assert request["headers"]["X-Signature"]
    assert request["headers"]["Content-Type"] == "application/json"


def test_validator_client_retries_job_result_with_compact_log_after_connection_drop() -> None:
    keypair = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
    wallet = SimpleNamespace(hotkey=keypair)

    class _FlakyResultSession:
        def __init__(self) -> None:
            self.requests: list[dict] = []

        def request(self, **kwargs):
            self.requests.append(kwargs)
            if len(self.requests) == 1:
                raise requests.exceptions.ConnectionError("remote disconnected")
            if kwargs["method"] == "GET":
                return _Response([])
            return _Response({"id": "job-1", "status": "accepted"})

    session = _FlakyResultSession()
    client = AutoresearchValidatorClient(
        base_url="http://127.0.0.1:8000",
        wallet=wallet,
        session=session,  # type: ignore[arg-type]
    )
    full_log = "x" * 32_000

    result = client.submit_verification(
        job_id="job-1",
        submission_id="submission-1",
        status="accepted",
        observed_metrics={"score": 2.5},
        notes="ok",
        replay_log=full_log,
    )

    assert result["status"] == "accepted"
    assert len(session.requests) == 3
    assert session.requests[1]["method"] == "GET"
    retry = session.requests[2]
    assert retry["method"] == "POST"
    assert retry["url"] == "http://127.0.0.1:8000/api/v1/validator/jobs/job-1/result"
    assert retry["json"]["observed_metrics"] == {"score": 2.5}
    assert retry["json"]["notes"].startswith("ok | validator result submit retry")
    assert "compacted before retrying" in retry["json"]["replay_log"]
    assert len(retry["json"]["replay_log"]) < len(full_log)
    assert retry["headers"]["X-Hotkey"] == keypair.ss58_address
    assert retry["headers"]["X-Signature"]


def test_validator_client_reconciles_completed_job_after_connection_drop() -> None:
    keypair = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
    wallet = SimpleNamespace(hotkey=keypair)

    class _CommittedResultSession:
        def __init__(self) -> None:
            self.requests: list[dict] = []

        def request(self, **kwargs):
            self.requests.append(kwargs)
            if len(self.requests) == 1:
                raise requests.exceptions.ConnectionError("remote disconnected")
            assert kwargs["method"] == "GET"
            return _Response(
                [
                    {
                        "id": "job-1",
                        "submission_id": "submission-1",
                        "status": "accepted",
                        "observed_metrics": {"score": 2.5},
                    }
                ]
            )

    session = _CommittedResultSession()
    client = AutoresearchValidatorClient(
        base_url="http://127.0.0.1:8000",
        wallet=wallet,
        session=session,  # type: ignore[arg-type]
    )

    result = client.submit_verification(
        job_id="job-1",
        submission_id="submission-1",
        status="accepted",
        observed_metrics={"score": 2.5},
        notes="ok",
        replay_log="score=2.5",
    )

    assert result["status"] == "accepted"
    assert result["observed_metrics"] == {"score": 2.5}
    assert len(session.requests) == 2
    assert session.requests[0]["method"] == "POST"
    assert session.requests[1]["method"] == "GET"
    assert session.requests[1]["url"] == "http://127.0.0.1:8000/api/v1/verifications"
    assert session.requests[1]["params"] == {"submission_id": "submission-1"}


def test_public_replay_engine_accepts_local_patch(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOST_SECRET", "host-secret")
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    _init_git_repo(repo_dir)
    (repo_dir / "score.txt").write_text("1.0\n", encoding="utf-8")
    (repo_dir / "benchmark.py").write_text(
        "import os\n"
        "from pathlib import Path\n"
        "score = Path('score.txt').read_text(encoding='utf-8').strip()\n"
        "print('host_secret=' + os.environ.get('HOST_SECRET', ''))\n"
        "print('heldout=' + os.environ.get('AUTORESEARCH_HELDOUT_SPLIT', ''))\n"
        "print(f'score={score}')\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=repo_dir, check=True, capture_output=True)

    (repo_dir / "score.txt").write_text("2.5\n", encoding="utf-8")
    patch = subprocess.run(
        ["git", "diff", "--", "score.txt"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    job = ReplayJob(
        job_id=None,
        submission_id="submission-1",
        submission={
            "id": "submission-1",
            "task_id": "task-1",
            "base_ref": "HEAD",
            "patch": patch,
        },
        task={
            "id": "task-1",
            "repository": str(repo_dir),
            "base_ref": "HEAD",
            "benchmark_command": "python3 benchmark.py",
            "allowed_patch_paths": ["score.txt"],
            "metric_name": "score",
            "validator_benchmark_env": {"AUTORESEARCH_HELDOUT_SPLIT": "secret-split"},
            "time_budget_seconds": 60,
        },
        replay_spec={},
        detail={"metric_name": "score"},
        source="test",
    )
    engine = PublicReplayEngine(
        workspace_root=tmp_path / "workspaces",
        allow_unsafe_host_replay=True,
    )

    result = engine.run(job)

    assert result.status == "accepted"
    assert result.observed_metrics == {"score": 2.5}
    assert "score=2.5" in result.replay_log
    assert "host-secret" not in result.replay_log
    assert "secret-split" not in result.replay_log
    assert "heldout=[REDACTED]" in result.replay_log


def test_public_replay_engine_uses_docker_sandbox_without_host_replay(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    _init_git_repo(repo_dir)
    (repo_dir / "score.txt").write_text("1.0\n", encoding="utf-8")
    (repo_dir / "benchmark.py").write_text("print('score=1.0')\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=repo_dir, check=True, capture_output=True)

    (repo_dir / "score.txt").write_text("4.0\n", encoding="utf-8")
    patch = subprocess.run(
        ["git", "diff", "--", "score.txt"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    job = ReplayJob(
        job_id="job-1",
        submission={
            "id": "submission-docker",
            "task_id": "task-1",
            "base_ref": "HEAD",
            "patch": patch,
        },
        submission_id="submission-docker",
        task={
            "id": "task-1",
            "repository": str(repo_dir),
            "base_ref": "HEAD",
            "benchmark_command": "python3 benchmark.py",
            "result_path": "result.json",
            "allowed_patch_paths": ["score.txt"],
            "metric_name": "score",
            "time_budget_seconds": 60,
            "validator_benchmark_env": {"AUTORESEARCH_HELDOUT_SPLIT": "secret-split"},
        },
        replay_spec={},
        detail={"metric_name": "score"},
        source="test",
    )
    calls: list[dict] = []

    def fake_sandbox(**kwargs):
        calls.append(kwargs)
        result_path = kwargs["work_dir"] / "sandbox-results" / "result.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            '{"metrics": {"score": 4.0, "compressed_size_bytes": 12345}}',
            encoding="utf-8",
        )
        return SandboxedReplayResult(
            setup_result=None,
            benchmark_result=SandboxCommandResult(returncode=0, stdout="score=4.0\n", stderr=""),
            result_path=result_path,
            image=kwargs["config"].image,
        )

    monkeypatch.setattr(public_replay_module, "run_replay_in_docker_sandbox", fake_sandbox)
    engine = PublicReplayEngine(
        workspace_root=tmp_path / "workspaces",
        allow_unsafe_host_replay=False,
        replay_sandbox_mode="docker",
        docker_sandbox_config=DockerSandboxConfig(image="bitsota-test:local", gpus="all"),
    )

    result = engine.run(job)

    assert result.status == "accepted"
    assert result.observed_metrics == {"score": 4.0, "compressed_size_bytes": 12345.0}
    assert "Replay succeeded in docker sandbox" in result.notes
    assert "replay_execution_surface=docker_sandbox" in result.replay_log
    assert "secret-split" not in result.replay_log
    assert calls[0]["benchmark_env"]["AUTORESEARCH_HELDOUT_SPLIT"] == "secret-split"
    assert calls[0]["config"].gpus == "all"


def test_public_replay_engine_prefetches_backend_heldout_for_docker(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    _init_git_repo(repo_dir)
    (repo_dir / "benchmark.py").write_text("print('score=1.0')\n", encoding="utf-8")
    (repo_dir / "score.txt").write_text("1.0\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=repo_dir, check=True, capture_output=True)
    (repo_dir / "score.txt").write_text("2.0\n", encoding="utf-8")
    patch = subprocess.run(
        ["git", "diff", "--", "score.txt"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    sources_json = json.dumps(
        {
            "version": "qwen-frontier-test-hf-mix-v1",
            "documents": [
                {
                    "category": "general",
                    "source": "hf_rows",
                    "dataset": "HuggingFaceFW/fineweb-edu",
                    "config": "sample-10BT",
                    "split": "train",
                    "text_column": "text",
                    "offset": 7,
                    "limit": 2,
                    "page_size": 2,
                    "min_chars": 1,
                    "rotate": False,
                }
            ],
        }
    )
    job = ReplayJob(
        job_id="job-heldout",
        submission={"id": "submission-heldout", "task_id": "task-1", "base_ref": "HEAD", "patch": patch},
        submission_id="submission-heldout",
        task={
            "id": "task-1",
            "repository": str(repo_dir),
            "base_ref": "HEAD",
            "benchmark_command": "python3 benchmark.py",
            "result_path": "result.json",
            "allowed_patch_paths": ["score.txt"],
            "metric_name": "score",
            "time_budget_seconds": 60,
            "validator_benchmark_env": {
                "AUTORESEARCH_HELDOUT_DATASET": "validator-backend-heldout",
                "AUTORESEARCH_HELDOUT_DATASET_MIX": "general:1.0",
                "AUTORESEARCH_HELDOUT_SOURCES_JSON": sources_json,
                "AUTORESEARCH_HELDOUT_SYNC_SLOT": "123",
            },
        },
        replay_spec={},
        detail={"metric_name": "score"},
        source="test",
    )
    fetch_calls: list[dict] = []
    sandbox_calls: list[dict] = []
    manifest_payloads: list[dict] = []

    def fake_fetch_hf_rows(**kwargs):
        fetch_calls.append(kwargs)
        offset = int(kwargs["offset"])
        length = int(kwargs["length"])
        return {
            "rows": [
                {"row": {"text": f"heldout row {offset + index}"}}
                for index in range(length)
            ]
        }

    def fake_sandbox(**kwargs):
        sandbox_calls.append(kwargs)
        env = kwargs["benchmark_env"]
        manifest_path = kwargs["repo_dir"] / env["AUTORESEARCH_PRIVATE_HELDOUT_MANIFEST"]
        manifest_payloads.append(json.loads(manifest_path.read_text(encoding="utf-8")))
        result_path = kwargs["work_dir"] / "sandbox-results" / "result.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text('{"metrics": {"score": 1.0}}', encoding="utf-8")
        return SandboxedReplayResult(
            setup_result=None,
            benchmark_result=SandboxCommandResult(returncode=0, stdout="score=1.0\n", stderr=""),
            result_path=result_path,
            image=kwargs["config"].image,
        )

    monkeypatch.setattr(public_replay_module, "_fetch_hf_rows", fake_fetch_hf_rows)
    monkeypatch.setattr(public_replay_module, "run_replay_in_docker_sandbox", fake_sandbox)
    engine = PublicReplayEngine(
        workspace_root=tmp_path / "workspaces",
        replay_sandbox_mode="docker",
        local_benchmark_env={"HF_TOKEN": "secret-hf-token"},
        docker_sandbox_config=DockerSandboxConfig(image="bitsota-test:local"),
    )

    result = engine.run(job)

    assert result.status == "accepted", result.notes + "\n" + result.replay_log
    assert fetch_calls == [
        {
            "dataset": "HuggingFaceFW/fineweb-edu",
            "config": "sample-10BT",
            "split": "train",
            "offset": 7,
            "length": 2,
            "token": "secret-hf-token",
        }
    ]
    benchmark_env = sandbox_calls[0]["benchmark_env"]
    assert benchmark_env["AUTORESEARCH_HELDOUT_DATASET"] == "validator-private-shard"
    assert benchmark_env["AUTORESEARCH_PRIVATE_HELDOUT_MANIFEST"] == ".autoresearch-heldout/manifest.json"
    assert benchmark_env["AUTORESEARCH_PRIVATE_HELDOUT_ROOT"] == ".autoresearch-heldout"
    assert "AUTORESEARCH_HELDOUT_SOURCES_JSON" not in benchmark_env
    assert "HF_TOKEN" not in benchmark_env
    assert manifest_payloads[0]["documents"] == [
        {"text": "heldout row 7", "category": "general"},
        {"text": "heldout row 8", "category": "general"},
    ]
    assert "heldout_prefetch=ok" in result.replay_log
    assert "secret-hf-token" not in result.replay_log


def test_remote_artifact_accepts_gzip_transfer_content_length(
    tmp_path: Path,
    monkeypatch,
) -> None:
    artifact = b'{"artifact_version":1,"payload":"smoke"}'

    class _CompressedResponse:
        status_code = 200
        headers = {
            "Content-Length": "12",
            "Content-Encoding": "gzip",
        }

        def iter_content(self, chunk_size: int):  # type: ignore[no-untyped-def]
            assert chunk_size > 0
            yield artifact

    def fake_get(*args, **kwargs):  # type: ignore[no-untyped-def]
        assert kwargs["allow_redirects"] is False
        return _CompressedResponse()

    monkeypatch.setattr(public_replay_module.requests, "get", fake_get)
    engine = PublicReplayEngine(workspace_root=tmp_path / "workspaces")

    size, digest = engine._download_remote_artifact(
        "https://example.test/artifact.json",
        dest=tmp_path / "artifact.json",
        expected_size_bytes=len(artifact),
    )

    assert size == len(artifact)
    assert digest == hashlib.sha256(artifact).hexdigest()


def test_public_replay_engine_passes_local_validator_benchmark_env(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    _init_git_repo(repo_dir)
    (repo_dir / "score.txt").write_text("1.0\n", encoding="utf-8")
    (repo_dir / "benchmark.py").write_text(
        "import os\n"
        "from pathlib import Path\n"
        "print('manifest=' + os.environ.get('AUTORESEARCH_PRIVATE_HELDOUT_MANIFEST', ''))\n"
        "print('score=' + Path('score.txt').read_text(encoding='utf-8').strip())\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=repo_dir, check=True, capture_output=True)
    (repo_dir / "score.txt").write_text("2.0\n", encoding="utf-8")
    patch = subprocess.run(
        ["git", "diff", "--", "score.txt"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    manifest_path = tmp_path / "private-heldout.json"
    manifest_path.write_text("{}", encoding="utf-8")
    job = ReplayJob(
        job_id=None,
        submission_id="submission-1",
        submission={
            "id": "submission-1",
            "task_id": "task-1",
            "base_ref": "HEAD",
            "patch": patch,
        },
        task={
            "id": "task-1",
            "repository": str(repo_dir),
            "base_ref": "HEAD",
            "benchmark_command": "python3 benchmark.py",
            "allowed_patch_paths": ["score.txt"],
            "metric_name": "score",
            "time_budget_seconds": 60,
        },
        replay_spec={},
        detail={"metric_name": "score"},
        source="test",
    )
    engine = PublicReplayEngine(
        workspace_root=tmp_path / "workspaces",
        allow_unsafe_host_replay=True,
        local_benchmark_env={"AUTORESEARCH_PRIVATE_HELDOUT_MANIFEST": str(manifest_path)},
    )

    result = engine.run(job)

    assert result.status == "accepted"
    assert result.observed_metrics == {"score": 2.0}
    assert str(manifest_path) not in result.replay_log
    assert "manifest=[REDACTED]" in result.replay_log


def test_public_replay_engine_rejects_generated_python_cache_patch_paths(tmp_path: Path) -> None:
    repo_dir = tmp_path / "ignored-pycache-repo"
    repo_dir.mkdir()
    _init_git_repo(repo_dir)
    (repo_dir / "benchmark.py").write_text("print('score=1.0')\n", encoding="utf-8")
    pycache_path = _compile_python_cache(repo_dir, "benchmark.py")
    pycache_rel = pycache_path.relative_to(repo_dir).as_posix()
    subprocess.run(["git", "add", "."], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=repo_dir, check=True, capture_output=True)

    (repo_dir / "benchmark.py").write_text("print('score=2.0')\n# changed\n", encoding="utf-8")
    _compile_python_cache(repo_dir, "benchmark.py")
    patch = subprocess.run(
        ["git", "diff", "--binary", "--", "benchmark.py", pycache_rel],
        cwd=repo_dir,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "diff --git a/benchmark.py b/benchmark.py" in patch
    assert f"diff --git a/{pycache_rel} b/{pycache_rel}" in patch

    job = ReplayJob(
        job_id=None,
        submission_id="submission-pycache",
        submission={
            "id": "submission-pycache",
            "task_id": "task-1",
            "base_ref": "HEAD",
            "patch": patch,
        },
        task={
            "id": "task-1",
            "repository": str(repo_dir),
            "base_ref": "HEAD",
            "benchmark_command": "python3 benchmark.py",
            "allowed_patch_paths": ["benchmark.py"],
            "metric_name": "score",
            "time_budget_seconds": 60,
        },
        replay_spec={},
        detail={"metric_name": "score"},
        source="test",
    )
    engine = PublicReplayEngine(
        workspace_root=tmp_path / "workspaces",
        allow_unsafe_host_replay=True,
    )

    result = engine.run(job)

    assert result.status == "rejected"
    assert result.observed_metrics == {}
    assert "generated Python bytecode/cache" in result.notes
    assert pycache_rel in result.notes
    assert "validator_rejection=generated_python_cache_paths" in result.replay_log


def test_public_replay_engine_rejects_out_of_surface_source_patch_paths(
    tmp_path: Path,
) -> None:
    repo_dir = tmp_path / "source-plus-pycache-repo"
    repo_dir.mkdir()
    _init_git_repo(repo_dir)
    (repo_dir / "benchmark.py").write_text("print('score=1.0')\n", encoding="utf-8")
    (repo_dir / "support.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=repo_dir, check=True, capture_output=True)

    (repo_dir / "support.py").write_text("VALUE = 2\n", encoding="utf-8")
    patch = subprocess.run(
        ["git", "diff", "--binary", "--", "support.py"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "diff --git a/support.py b/support.py" in patch

    job = ReplayJob(
        job_id=None,
        submission_id="submission-source-plus-pycache",
        submission={
            "id": "submission-source-plus-pycache",
            "task_id": "task-1",
            "base_ref": "HEAD",
            "patch": patch,
        },
        task={
            "id": "task-1",
            "repository": str(repo_dir),
            "base_ref": "HEAD",
            "benchmark_command": "python3 benchmark.py",
            "allowed_patch_paths": ["benchmark.py"],
            "metric_name": "score",
            "time_budget_seconds": 60,
        },
        replay_spec={},
        detail={"metric_name": "score"},
        source="test",
    )
    engine = PublicReplayEngine(
        workspace_root=tmp_path / "workspaces",
        allow_unsafe_host_replay=True,
    )

    result = engine.run(job)

    assert result.status == "rejected"
    assert "support.py" in result.notes
    assert "validator_rejection=disallowed_patch_paths" in result.replay_log


def test_public_replay_engine_rejects_oversized_patches(tmp_path: Path) -> None:
    repo_dir = tmp_path / "oversized-patch-repo"
    repo_dir.mkdir()
    _init_git_repo(repo_dir)
    (repo_dir / "benchmark.py").write_text("print('score=1.0')\n", encoding="utf-8")
    (repo_dir / "score.txt").write_text("1.0\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=repo_dir, check=True, capture_output=True)

    (repo_dir / "score.txt").write_text(("2.0\n" * 40), encoding="utf-8")
    patch = subprocess.run(
        ["git", "diff", "--binary", "--", "score.txt"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    job = ReplayJob(
        job_id=None,
        submission_id="submission-oversized",
        submission={
            "id": "submission-oversized",
            "task_id": "task-1",
            "base_ref": "HEAD",
            "patch": patch,
        },
        task={
            "id": "task-1",
            "repository": str(repo_dir),
            "base_ref": "HEAD",
            "benchmark_command": "python3 benchmark.py",
            "allowed_patch_paths": ["score.txt"],
            "max_patch_bytes": 64,
            "metric_name": "score",
            "time_budget_seconds": 60,
        },
        replay_spec={},
        detail={"metric_name": "score"},
        source="test",
    )
    engine = PublicReplayEngine(
        workspace_root=tmp_path / "workspaces",
        allow_unsafe_host_replay=True,
    )

    result = engine.run(job)

    assert result.status == "rejected"
    assert "maximum size" in result.notes
    assert "validator_rejection=patch_too_large" in result.replay_log


def test_replay_sandbox_create_container_passes_gpu_and_limits(monkeypatch) -> None:
    calls: list[tuple[str, tuple[str, ...]]] = []

    def fake_run_checked(*args, stage, timeout=None):
        calls.append((stage, args))
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="container-id\n", stderr="")

    monkeypatch.setattr(replay_sandbox_module, "_run_docker_checked", fake_run_checked)

    container_id = replay_sandbox_module._create_container(
        config=DockerSandboxConfig(
            gpus="all",
            memory_limit="8g",
            pids_limit=128,
            cpus=2.0,
        ),
        image="bitsota-test:local",
        workspace_volume="bitsota-test-volume",
        network_mode="bridge",
        timeout=30,
    )

    assert container_id == "container-id"
    stage, args = calls[0]
    assert stage == "docker create"
    flat = list(args)
    assert "--gpus" in flat
    assert flat[flat.index("--gpus") + 1] == "all"
    assert "--network" in flat
    assert flat[flat.index("--network") + 1] == "bridge"
    assert "--memory" in flat
    assert flat[flat.index("--memory") + 1] == "8g"
    assert "--pids-limit" in flat
    assert flat[flat.index("--pids-limit") + 1] == "128"
    assert "--cpus" in flat
    assert flat[flat.index("--cpus") + 1] == "2.0"
    assert "--read-only" in flat
    assert "--security-opt" in flat
    assert "no-new-privileges:true" in flat


def test_replay_sandbox_workspace_command_uses_writable_venv() -> None:
    wrapped = replay_sandbox_module._wrap_workspace_command("python benchmark.py")

    assert "python -m venv --system-site-packages /workspace/.venv" in wrapped
    assert "export VIRTUAL_ENV=/workspace/.venv" in wrapped
    assert "export PATH=/workspace/.venv/bin:$PATH" in wrapped
    assert wrapped.endswith("python benchmark.py")


def test_public_validator_runner_submits_replay_result(tmp_path: Path) -> None:
    job = ReplayJob(
        job_id="job-1",
        submission_id="submission-1",
        submission={"id": "submission-1"},
        task={},
        replay_spec={},
        detail={},
        source="test",
    )
    calls: list[dict] = []

    class _Client:
        def claim_replay_job(self, **kwargs):
            calls.append({"claim": kwargs})
            return job

        def submit_verification(self, **kwargs):
            calls.append({"submit": kwargs})
            return {"id": "verification-1", "status": kwargs["status"]}

    class _Engine:
        def run(self, claimed_job):
            assert claimed_job is job
            return ReplayResult(
                submission_id="submission-1",
                status="accepted",
                observed_metrics={"score": 3.0},
                notes="ok",
                replay_log="score=3.0",
            )

    runner = PublicValidatorRunner(
        client=_Client(),  # type: ignore[arg-type]
        engine=_Engine(),  # type: ignore[arg-type]
        config=PublicValidatorRunnerConfig(
            workspace_root=tmp_path,
            allow_unsafe_host_replay=True,
        ),
    )

    outcome = runner.run_once()

    assert isinstance(outcome, PublicValidatorRunOutcome)
    assert outcome.job_id == "job-1"
    assert outcome.status == "accepted"
    assert calls[0]["claim"] == {
        "task_id": None,
        "task_slug": None,
        "claim_path": DEFAULT_VALIDATOR_WORKLIST_PATH,
    }
    assert calls[1]["submit"]["job_id"] == "job-1"
    assert calls[1]["submit"]["submission_id"] == "submission-1"
    assert calls[1]["submit"]["observed_metrics"] == {"score": 3.0}
