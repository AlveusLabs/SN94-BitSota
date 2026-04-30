from __future__ import annotations

from pathlib import Path
import subprocess
from types import SimpleNamespace

from substrateinterface import Keypair

from validator.public_replay import PublicReplayEngine, ReplayResult
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
dry_run: true
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
pending_submissions_fallback: true
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

    assert config.claim_path == DEFAULT_VALIDATOR_WORKLIST_PATH


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


def test_validator_client_claims_oldest_pending_non_peer_submission(monkeypatch) -> None:
    wallet = SimpleNamespace(hotkey=SimpleNamespace(ss58_address="validator-hotkey"))
    client = AutoresearchValidatorClient(base_url="http://127.0.0.1:8000", wallet=wallet)
    monkeypatch.setattr(
        client,
        "list_tasks",
        lambda: [
            {"id": "task-peer", "slug": "peer", "competition_mode": "peer_evaluation"},
            {
                "id": "task-standard",
                "slug": "standard",
                "competition_mode": "standard",
                "repository": "repo",
                "base_ref": "HEAD",
                "benchmark_command": "python benchmark.py",
                "metric_name": "score",
                "time_budget_seconds": 60,
            },
        ],
    )
    monkeypatch.setattr(
        client,
        "list_pending_submissions",
        lambda *, task_id=None: [
            {"id": "peer-submission", "task_id": "task-peer", "created_at": "2026-04-01T00:00:00"},
            {"id": "standard-submission", "task_id": "task-standard", "created_at": "2026-04-01T00:00:01"},
        ],
    )
    monkeypatch.setattr(
        client,
        "get_submission_detail",
        lambda submission_id: {
            "submission": {
                "id": submission_id,
                "task_id": "task-standard",
                "base_ref": "HEAD",
                "patch": "",
            },
            "metric_name": "score",
        },
    )

    job = client.claim_replay_job(claim_path=None)

    assert job is not None
    assert job.job_id is None
    assert job.submission_id == "standard-submission"
    assert job.task["id"] == "task-standard"
    assert job.replay_spec["benchmark_command"] == "python benchmark.py"
    assert job.replay_spec["_source"] == "task_response_fallback"


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
