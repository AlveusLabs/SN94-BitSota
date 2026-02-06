"""Local miner harness for experimenting with hyperparameters.

This script reuses the same DirectClient/evolution-engine stack that the GUI and
relay path use, but keeps everything local: tasks are generated on the fly from
offline datasets, SOTA verification runs through a deterministic validator-style
task suite for the selected task type, and submissions are logged instead of
hitting the relay.
"""

import argparse
import csv
import json
from collections import deque
import contextlib
import io
import importlib
import multiprocessing as mp
import os
import queue as queue_module
from pathlib import Path
import random
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from miner.client import DirectClient, DEFAULT_TASK_TYPE, TASK_REGISTRY
from miner.engines.ga_engine import BaselineEvolutionEngine
from miner.engines.archive_engine import ArchiveAwareBaselineEvolution
from miner.engines.base_engine import BaseEvolutionEngine

try:
    import bitsota_sdk.evaluations as evaluations
    from bitsota_sdk.dsl_parser import DSLParser
    from bitsota_sdk.tasks.mnist import MNISTBinaryTask
    from bitsota_sdk.tasks.scalar_linear import ScalarLinearRegressionTask
except ModuleNotFoundError:
    import core.evaluations as evaluations
    from core.dsl_parser import DSLParser
    from core.tasks.cifar10 import CIFAR10BinaryTask
    from core.tasks.mnist import MNISTBinaryTask
    from core.tasks.scalar_linear import ScalarLinearRegressionTask

    if not hasattr(evaluations, "_OFFLINE_BINARY_TASK_REGISTRY"):
        evaluations._OFFLINE_BINARY_TASK_REGISTRY = {
            "cifar10_binary": CIFAR10BinaryTask,
            "mnist_binary": MNISTBinaryTask,
            "scalar_linear": ScalarLinearRegressionTask,
        }

    if not callable(getattr(evaluations, "_get_offline_validator_task_specs", None)):
        def _compat_get_offline_validator_task_specs(task_type: str, input_dim: int):
            key = str(task_type).strip().lower()
            if key == "cifar10_binary":
                return evaluations._get_cifar_validator_task_specs(int(input_dim))
            if key == "mnist_binary":
                return evaluations._get_mnist_validator_task_specs(int(input_dim))
            if key == "scalar_linear":
                return evaluations._get_scalar_validator_task_specs(int(input_dim))
            return []

        evaluations._get_offline_validator_task_specs = _compat_get_offline_validator_task_specs

    if not hasattr(evaluations, "SUPPORTED_OFFLINE_BINARY_TASK_TYPES"):
        evaluations.SUPPORTED_OFFLINE_BINARY_TASK_TYPES = tuple(evaluations._OFFLINE_BINARY_TASK_REGISTRY.keys())

    if not hasattr(evaluations, "DEFAULT_TASK_TYPE"):
        evaluations.DEFAULT_TASK_TYPE = DEFAULT_TASK_TYPE


LOCAL_TASK_REGISTRY: Dict[str, Any] = dict(TASK_REGISTRY)
LOCAL_TASK_REGISTRY["mnist_binary"] = MNISTBinaryTask
LOCAL_TASK_REGISTRY["scalar_linear"] = ScalarLinearRegressionTask


class LocalTestingClient(DirectClient):
    """DirectClient variant that runs fully offline for experimentation."""

    def __init__(
        self,
        *,
        public_address: str,
        engine_type: str,
        miner_task_count: int,
        pop_size: int,
        tournament_size: int,
        mutation_prob: float,
        sota_threshold: float,
        verbose: bool = False,
        phase_max_sizes: Optional[Dict[str, int]] = None,
        scalar_count: Optional[int] = None,
        vector_count: Optional[int] = None,
        matrix_count: Optional[int] = None,
        vector_dim: Optional[int] = None,
        train_examples: Optional[int] = None,
        val_examples: Optional[int] = None,
        feature_dim: Optional[int] = None,
        task_type: str = DEFAULT_TASK_TYPE,
        task_registry: Optional[Dict[str, Any]] = None,
        fec_cache_size: int = 0,
        cifar_seed: Optional[int] = None,
    ):
        super().__init__(
            public_address=public_address,
            relay_endpoint="http://127.0.0.1:0",  # never contacted
            verbose=verbose,
            wallet=None,
            metrics_log_file=None,
            contract_manager=None,
            miner_task_count=miner_task_count,
            engine_type=engine_type,
        )
        self._engine_type = engine_type
        self._pop_size = pop_size
        self._tournament_size = tournament_size
        self._mutation_prob = mutation_prob
        self._sota_threshold = sota_threshold
        self._max_generations = None
        self._phase_max_sizes = phase_max_sizes
        self._scalar_count = scalar_count
        self._vector_count = vector_count
        self._matrix_count = matrix_count
        self._vector_dim = vector_dim
        if (train_examples is None) ^ (val_examples is None):
            raise ValueError("Provide both --train-examples and --val-examples to override dataset sizes.")
        self._train_examples = train_examples
        self._val_examples = val_examples
        self._feature_dim = feature_dim
        self._task_type = task_type
        self._task_registry = task_registry or LOCAL_TASK_REGISTRY
        self._fec_cache_size = max(0, int(fec_cache_size))
        self._cifar_seed = cifar_seed

    # ---------- hooks ----------
    def _auth_payload(self) -> Dict[str, Any]:
        # Local mode never hits relay, but DirectClient expects a payload.
        return {
            "public_address": self.public_address,
            "signature": "local",
            "message": str(time.time()),
        }

    def _fetch_sota_threshold(self) -> float:
        return self._sota_threshold

    def _get_engine(self, task_type: str, engine_type: str = "baseline") -> BaseEvolutionEngine:
        """Override to inject hyperparameters for each engine type."""

        key = (task_type, engine_type)
        if key in self._engine_cache:
            return self._engine_cache[key]

        task_cls = self._task_registry.get(task_type)
        if not task_cls:
            raise ValueError(f"Unknown task type: {task_type}")

        # When running with multiple workers, we want deterministic task bootstrapping across
        # processes (spawn) so one worker doesn't crash on a missing offline dataset file
        # while another happens to pick an existing task at random.
        sampler_seed = 0 if self._cifar_seed is None else int(self._cifar_seed)
        try:
            task = task_cls(sampler_seed=sampler_seed)
        except TypeError:
            task = task_cls()
        load_kwargs = {}
        if str(task_type).strip().lower() == "scalar_linear":
            if self._train_examples is not None and self._val_examples is not None:
                load_kwargs["train_samples"] = self._train_examples
                load_kwargs["val_samples"] = self._val_examples
            if self._feature_dim is not None:
                load_kwargs["n_features"] = self._feature_dim
        else:
            if self._train_examples is not None and self._val_examples is not None:
                train_count = int(self._train_examples)
                val_count = int(self._val_examples)
                if train_count <= 0 or val_count <= 0:
                    raise ValueError("--train-examples and --val-examples must both be >= 1")
                total = train_count + val_count
                # These offline tasks model dataset size as (n_samples, train_split). Use a
                # tiny epsilon so int(train_split * n_samples) reliably matches train_count.
                train_split = (train_count + 1e-6) / float(total)
                train_split = min(max(train_split, 1e-12), 1.0 - 1e-12)
                load_kwargs["n_samples"] = int(total)
                load_kwargs["train_split"] = float(train_split)
            if self._feature_dim is not None:
                load_kwargs["n_components"] = self._feature_dim
            # Canonical offline datasets: keep the initial load deterministic.
            # The evolution engine will re-load per-task specs as needed.
            if task_type in {"cifar10_binary", "mnist_binary"}:
                load_kwargs.setdefault("task_id", 0)
        if load_kwargs:
            task.load_data(**load_kwargs)
        else:
            task.load_data()

        if engine_type == "archive":
            engine = ArchiveAwareBaselineEvolution(
                task,
                pop_size=self._pop_size,
                verbose=self.verbose,
                miner_task_count=self.miner_task_count,
                phase_max_sizes=self._phase_max_sizes,
                scalar_count=self._scalar_count,
                vector_count=self._vector_count,
                matrix_count=self._matrix_count,
                vector_dim=self._vector_dim,
                fec_cache_size=self._fec_cache_size,
                cifar_seed=self._cifar_seed,
            )
        elif engine_type == "baseline":
            engine = BaselineEvolutionEngine(
                task,
                pop_size=self._pop_size,
                verbose=self.verbose,
                miner_task_count=self.miner_task_count,
                tournament_size=self._tournament_size,
                mutation_prob=self._mutation_prob,
                phase_max_sizes=self._phase_max_sizes,
                scalar_count=self._scalar_count,
                vector_count=self._vector_count,
                matrix_count=self._matrix_count,
                vector_dim=self._vector_dim,
                fec_cache_size=self._fec_cache_size,
                cifar_seed=self._cifar_seed,
            )
        else:
            raise ValueError(f"Unknown engine type: {engine_type}")

        self._engine_cache[key] = engine
        return engine

    def submit_solution(self, solution_data: Dict[str, Any]) -> Dict[str, Any]:
        """Skip relay submission but keep validator-style verification."""

        sota_threshold = self._fetch_sota_threshold()
        current_score = float(solution_data.get("eval_score", -np.inf))
        task_type = solution_data.get("task_type") or self._task_type
        if str(task_type).strip().lower() == "scalar_linear":
            input_dim = int(solution_data.get("input_dim") or self._feature_dim or 16)
            score = _score_scalar_linear_validator_suite(
                DSLParser.from_dsl(solution_data.get("algorithm_dsl", ""), input_dim),
                input_dim=input_dim,
                train_samples=self._train_examples,
                val_samples=self._val_examples,
                validator_task_count=getattr(evaluations, "VALIDATOR_TASK_COUNT", None),
            )
            is_valid = float(score) >= float(sota_threshold)
        else:
            is_valid, score = evaluations.verify_solution_quality_on_task_type(
                solution_data, task_type, sota_threshold
            )

        if not is_valid:
            return {
                "status": "not_submitted",
                "reason": f"Below SOTA threshold. Score {score} < {sota_threshold}",
            }

        payload = {
            "task_id": solution_data.get("task_id"),
            "score": current_score,
            "algorithm_result": solution_data,
            "validator_score": score,
        }
        print("[local submit]", payload)
        return {"status": "validated", "relay_response": payload}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local miner hyperparameter harness")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional JSON config file. Values become argparse defaults (CLI flags still override).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of independent worker processes to run (uses multiprocessing; avoids GIL).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base RNG seed. When --workers>1, each worker uses seed+worker_id.",
    )
    parser.add_argument("--engine", choices=["archive", "baseline"], default="baseline")
    parser.add_argument("--task-type", choices=list(LOCAL_TASK_REGISTRY.keys()), default=DEFAULT_TASK_TYPE, help="Task to train on")
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Total evolve_generation iterations to run per worker (not global).",
    )
    parser.add_argument("--pop-size", type=int, default=100, help="Population size / queue length")
    parser.add_argument("--miner-task-count", type=int, default=3, help="D tasks per fitness evaluation")
    parser.add_argument("--tournament-size", type=int, default=10, help="Tournament size T for baseline engine")
    parser.add_argument("--mutation-prob", type=float, default=0.9, help="Mutation probability U for baseline engine")
    parser.add_argument("--validator-task-count", type=int, default=None, help="Override validator task count D for verification")
    parser.add_argument(
        "--validate-every",
        type=int,
        default=10000,
        help="Run validator-style verification at most once every N iterations when the engine best improves (0 disables).",
    )
    parser.add_argument("--sota-threshold", type=float, default=0.0, help="Local SOTA threshold to beat")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--dsl-roundtrip-check",
        action="store_true",
        help=(
            "During validation, also score the in-memory AlgorithmArray on the same validator suite and "
            "compare it to the DSL->AlgorithmArray reconstruction score (helps detect DSL parsing drift)."
        ),
    )
    parser.add_argument(
        "--timing",
        action="store_true",
        help="Print per-interval timing breakdown for evolve/validate overhead (helps find bottlenecks).",
    )
    parser.add_argument(
        "--worker-logs",
        dest="worker_logs",
        action="store_true",
        default=True,
        help="Enable per-worker progress logs in meta-miner (default: enabled).",
    )
    parser.add_argument(
        "--no-worker-logs",
        dest="worker_logs",
        action="store_false",
        help="Disable per-worker progress logs in meta-miner.",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="Optional tag added to CSV filename and stored in CSV rows (useful for experiment tracking).",
    )
    parser.add_argument("--csv-dir", type=str, default=None, help="Optional directory to write per-iteration CSV logs")
    parser.add_argument(
        "--offline-data-dir",
        type=str,
        default=None,
        help=(
            "Root directory for offline binary datasets (contains dataset subfolders like mnist/ and cifar10/). "
            "Overrides BITSOTA_BINARY_DATA_DIR/BITSOTA_DATA_DIR/BINARY_DATASET_DIR."
        ),
    )
    parser.add_argument("--fec-cache-size", type=int, default=0, help="Enable LRU functional cache (0 disables)")
    parser.add_argument("--cifar-seed", type=int, default=None, help="Deterministic CIFAR task sampling seed")
    parser.add_argument("--setup-max-ops", type=int, default=None, help="Max instructions in setup phase")
    parser.add_argument("--predict-max-ops", type=int, default=None, help="Max instructions in predict phase")
    parser.add_argument("--learn-max-ops", type=int, default=None, help="Max instructions in learn phase")
    parser.add_argument("--scalar-regs", type=int, default=None, help="Number of scalar registers")
    parser.add_argument("--vector-regs", type=int, default=None, help="Number of vector registers")
    parser.add_argument("--matrix-regs", type=int, default=None, help="Number of matrix registers")
    parser.add_argument("--vector-dim", type=int, default=None, help="Vector register dimension (defaults to input_dim + 10)")
    parser.add_argument(
        "--feature-dim",
        type=int,
        default=None,
        help="Projection dimension per task (default 16); must match generated offline datasets.",
    )
    parser.add_argument(
        "--train-examples",
        type=int,
        default=None,
        help=(
            "Override training set size. For scalar_linear this sets train_samples; "
            "for offline MNIST/CIFAR tasks it sets (n_samples=train+val, train_split=train/(train+val))."
        ),
    )
    parser.add_argument(
        "--val-examples",
        type=int,
        default=None,
        help=(
            "Override validation set size. For scalar_linear this sets val_samples; "
            "for offline MNIST/CIFAR tasks it sets (n_samples=train+val, train_split=train/(train+val))."
        ),
    )
    parser.add_argument("--log-every", type=int, default=10, help="Print progress every N iterations (per worker).")
    parser.add_argument(
        "--migration-iters",
        type=int,
        default=0,
        help="Every N iterations, exchange half the population between workers (0 disables).",
    )
    parser.add_argument(
        "--initial-population-path",
        type=str,
        default=None,
        help="Optional JSON file with a serialized population snapshot to seed the engine (resume).",
    )
    parser.add_argument(
        "--population-snapshot-every",
        type=int,
        default=0,
        help="Every N iterations emit a full population snapshot event (0 disables).",
    )

    pre_args, _ = parser.parse_known_args()
    if getattr(pre_args, "config", None):
        try:
            cfg_path = Path(str(pre_args.config)).expanduser().resolve()
        except Exception:
            cfg_path = None
        if cfg_path and cfg_path.exists():
            try:
                raw = json.loads(cfg_path.read_text())
            except Exception:
                raw = None
            if isinstance(raw, dict):
                overrides: Dict[str, Any] = {}
                merged: Dict[str, Any] = dict(raw)
                if isinstance(raw.get("mining"), dict):
                    merged.update(raw["mining"])
                if isinstance(raw.get("args"), dict):
                    merged.update(raw["args"])

                allowed_dests = {action.dest for action in parser._actions if action.dest}
                for key, value in merged.items():
                    if key in {"mining", "args", "env"}:
                        continue
                    dest = str(key).strip().replace("-", "_")
                    if dest in allowed_dests:
                        overrides[dest] = value

                if overrides:
                    parser.set_defaults(**overrides)

    return parser.parse_args()


_OFFLINE_DATASET_BY_TASK_TYPE = {
    "cifar10_binary": "cifar10",
    "mnist_binary": "mnist",
}


def _configure_offline_data_dir(args: argparse.Namespace) -> None:
    """Set offline dataset root for local runs (if not already configured)."""

    env_keys = ("BITSOTA_BINARY_DATA_DIR", "BITSOTA_DATA_DIR", "BINARY_DATASET_DIR")
    if getattr(args, "offline_data_dir", None):
        os.environ["BITSOTA_BINARY_DATA_DIR"] = str(args.offline_data_dir)
        return

    if any(os.environ.get(k) for k in env_keys):
        return

    task_type = str(getattr(args, "task_type", "")).strip().lower()
    dataset = _OFFLINE_DATASET_BY_TASK_TYPE.get(task_type)
    if not dataset:
        return

    repo_root = Path(__file__).resolve().parent.parent
    candidate = repo_root / ".offline_data" / dataset
    if not candidate.is_dir():
        return
    try:
        has_npz = any(candidate.glob("*.npz"))
    except Exception:
        has_npz = False
    if not has_npz:
        return

    os.environ["BITSOTA_BINARY_DATA_DIR"] = str(repo_root / ".offline_data")
    if getattr(args, "verbose", False):
        print(f"[meta] Using offline dataset dir: {os.environ['BITSOTA_BINARY_DATA_DIR']}")


def _configure_validator_task_count(desired: Optional[int]) -> None:
    desired_validator_task_count: Optional[int] = desired
    if desired_validator_task_count is None and not os.environ.get("VALIDATOR_TASK_COUNT"):
        # Local harness default: keep verification lighter unless explicitly configured.
        desired_validator_task_count = 16
    if desired_validator_task_count is not None:
        try:
            desired_validator_task_count = max(1, int(desired_validator_task_count))
        except Exception:
            desired_validator_task_count = None

    if (
        desired_validator_task_count is not None
        and desired_validator_task_count != evaluations.VALIDATOR_TASK_COUNT
    ):
        os.environ["VALIDATOR_TASK_COUNT"] = str(desired_validator_task_count)
        importlib.reload(evaluations)


def _phase_overrides_from_args(args: argparse.Namespace) -> Optional[Dict[str, int]]:
    phase_overrides: Dict[str, int] = {}
    if args.setup_max_ops is not None:
        phase_overrides["setup"] = args.setup_max_ops
    if args.predict_max_ops is not None:
        phase_overrides["predict"] = args.predict_max_ops
    if args.learn_max_ops is not None:
        phase_overrides["learn"] = args.learn_max_ops
    return phase_overrides or None


def _build_engine(args: argparse.Namespace) -> BaseEvolutionEngine:
    client = LocalTestingClient(
        public_address="local-miner",
        engine_type=args.engine,
        miner_task_count=args.miner_task_count,
        pop_size=args.pop_size,
        tournament_size=args.tournament_size,
        mutation_prob=args.mutation_prob,
        sota_threshold=args.sota_threshold,
        verbose=args.verbose,
        phase_max_sizes=_phase_overrides_from_args(args),
        scalar_count=args.scalar_regs,
        vector_count=args.vector_regs,
        matrix_count=args.matrix_regs,
        vector_dim=args.vector_dim,
        train_examples=args.train_examples,
        val_examples=args.val_examples,
        feature_dim=args.feature_dim,
        task_type=args.task_type,
        task_registry=LOCAL_TASK_REGISTRY,
        fec_cache_size=args.fec_cache_size,
        cifar_seed=args.cifar_seed,
    )
    return client._get_engine(args.task_type, args.engine)


def _compute_population_stats(scores):
    valid_scores = [s for s in scores if s is not None and np.isfinite(s)]
    pop_best = max(valid_scores) if valid_scores else -np.inf
    pop_median = float(np.median(valid_scores)) if valid_scores else float("nan")
    if valid_scores:
        pop_q1 = float(np.percentile(valid_scores, 25))
        pop_q3 = float(np.percentile(valid_scores, 75))
    else:
        pop_q1 = float("nan")
        pop_q3 = float("nan")
    return pop_best, pop_median, pop_q1, pop_q3


def _roundtrip_validator_suite_scores(
    algo: object,
    algo_dsl: str,
    *,
    task_type: str,
    input_dim: int,
) -> Tuple[float, float, int, str, int, int, int]:
    """
    Return (native_score, reconstructed_score, parse_warning_count, parse_output,
    native_finite_count, reconstructed_finite_count, total_task_count).

    Scores are medians over the deterministic validator suite for the given task type,
    using the same task specs and eval seeds for both representations.
    """

    task_type = str(task_type).strip().lower()
    input_dim = int(input_dim)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        reconstructed = DSLParser.from_dsl(algo_dsl, input_dim)
    parse_out = buf.getvalue().strip()
    warn_count = 0
    if parse_out:
        warn_count = sum(
            1 for line in parse_out.splitlines() if "Warning: Could not parse line" in line
        )

    if task_type == "scalar_linear":
        # Scalar linear regression uses synthetic data; build a deterministic holdout suite
        # with the same feature dimension as the candidate.
        task = ScalarLinearRegressionTask(seed_offset=0)
        count = max(1, int(getattr(evaluations, "VALIDATOR_TASK_COUNT", 16)))
        default_seed = int(getattr(evaluations, "VALIDATOR_TASK_SEED", 1337))
        suite_seed = default_seed + 7919 * max(1, input_dim) + 15485863 * 2
        specs = ScalarLinearRegressionTask.sample_task_specs(
            num_tasks=count,
            replace=True,
            seed=suite_seed,
            n_features=input_dim,
        )
        native_scores: List[float] = []
        reconstructed_scores: List[float] = []
        for spec in specs:
            task.load_data(task_spec=spec)
            eval_seed = (
                int(default_seed)
                + 7919 * int(input_dim)
                + 15485863 * 2
                + 104729 * (int(getattr(spec, "param_seed", 0)) + 1)
                + 1000003 * (int(getattr(spec, "data_seed", 0)) + 2)
            ) % (2**31 - 1)
            if eval_seed <= 0:
                eval_seed = 1
            native_scores.append(float(task.evaluate_algorithm(algo, epochs=1, rng_seed=eval_seed)))
            reconstructed_scores.append(
                float(task.evaluate_algorithm(reconstructed, epochs=1, rng_seed=eval_seed))
            )

        finite_native = [s for s in native_scores if np.isfinite(s)]
        finite_reconstructed = [s for s in reconstructed_scores if np.isfinite(s)]
        native_median = float(np.median(finite_native)) if finite_native else -np.inf
        reconstructed_median = float(np.median(finite_reconstructed)) if finite_reconstructed else -np.inf
        return (
            native_median,
            reconstructed_median,
            warn_count,
            parse_out,
            len(finite_native),
            len(finite_reconstructed),
            len(specs),
        )

    registry = getattr(evaluations, "_OFFLINE_BINARY_TASK_REGISTRY", None) or {}
    task_cls = registry.get(task_type)
    if task_cls is None:
        raise ValueError(f"Unsupported task type for validator suite scoring: {task_type}")

    get_specs = getattr(evaluations, "_get_offline_validator_task_specs", None)
    if not callable(get_specs):
        raise RuntimeError("Validator task spec helper not available in core.evaluations")

    task = task_cls()
    task_specs = get_specs(task_type, input_dim)
    if not task_specs:
        return -np.inf, -np.inf, warn_count, parse_out, 0, 0, 0

    default_task_type = getattr(evaluations, "DEFAULT_TASK_TYPE", "cifar10_binary")
    task_salt = 0 if task_type == default_task_type else 1

    native_scores: List[float] = []
    reconstructed_scores: List[float] = []
    for spec in task_specs:
        task.load_data(task_spec=spec)
        spec_seed = int(getattr(spec, "seed", 0) or 0)
        spec_task_id = int(getattr(spec, "task_id", -1))
        eval_seed = (
            int(getattr(evaluations, "VALIDATOR_TASK_SEED", 1337))
            + 15485863 * int(task_salt)
            + 7919 * int(input_dim)
            + 104729 * (spec_seed + 1)
            + 1000003 * (spec_task_id + 2)
        ) % (2**31 - 1)
        if eval_seed <= 0:
            eval_seed = 1
        native_scores.append(float(task.evaluate_algorithm(algo, epochs=1, rng_seed=eval_seed)))
        reconstructed_scores.append(
            float(task.evaluate_algorithm(reconstructed, epochs=1, rng_seed=eval_seed))
        )

    finite_native = [s for s in native_scores if np.isfinite(s)]
    finite_reconstructed = [s for s in reconstructed_scores if np.isfinite(s)]
    native_median = float(np.median(finite_native)) if finite_native else -np.inf
    reconstructed_median = float(np.median(finite_reconstructed)) if finite_reconstructed else -np.inf
    return (
        native_median,
        reconstructed_median,
        warn_count,
        parse_out,
        len(finite_native),
        len(finite_reconstructed),
        len(task_specs),
    )


def _score_scalar_linear_validator_suite(
    algo: object,
    *,
    input_dim: int,
    train_samples: Optional[int] = None,
    val_samples: Optional[int] = None,
    validator_task_count: Optional[int] = None,
) -> float:
    """Score AlgorithmArray on a deterministic scalar-linear holdout suite (median over tasks)."""

    input_dim = int(input_dim)
    count = int(
        validator_task_count
        if validator_task_count is not None
        else getattr(evaluations, "VALIDATOR_TASK_COUNT", 16)
    )
    count = max(1, count)

    ts = int(train_samples) if train_samples is not None else None
    vs = int(val_samples) if val_samples is not None else None
    default_seed = int(getattr(evaluations, "VALIDATOR_TASK_SEED", 1337))
    suite_seed = default_seed + 7919 * max(1, input_dim) + 15485863 * 2

    specs = ScalarLinearRegressionTask.sample_task_specs(
        num_tasks=count,
        replace=True,
        seed=suite_seed,
        n_features=input_dim,
        train_samples=ts if ts is not None else ScalarLinearRegressionTask.get_task_spec(0).train_samples,
        val_samples=vs if vs is not None else ScalarLinearRegressionTask.get_task_spec(0).val_samples,
    )

    task = ScalarLinearRegressionTask(seed_offset=0)
    scores: List[float] = []
    for spec in specs:
        task.load_data(task_spec=spec)
        eval_seed = (
            int(default_seed)
            + 7919 * int(input_dim)
            + 15485863 * 2
            + 104729 * (int(getattr(spec, "param_seed", 0)) + 1)
            + 1000003 * (int(getattr(spec, "data_seed", 0)) + 2)
        ) % (2**31 - 1)
        if eval_seed <= 0:
            eval_seed = 1
        scores.append(float(task.evaluate_algorithm(algo, epochs=1, rng_seed=eval_seed)))

    finite_scores = [s for s in scores if np.isfinite(s)]
    if not finite_scores:
        return -np.inf
    return float(np.median(finite_scores))


def _sanitize_run_tag(tag: Optional[str]) -> Optional[str]:
    if tag is None:
        return None
    try:
        tag = str(tag).strip()
    except Exception:
        return None
    if not tag:
        return None
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    cleaned = []
    for ch in tag:
        if ch in allowed:
            cleaned.append(ch)
        elif ch.isspace():
            cleaned.append("_")
        else:
            cleaned.append("_")
    result = "".join(cleaned).strip("._-")
    if not result:
        return None
    return result[:64]


def _sample_indices(pop_size: int, count: int) -> List[int]:
    pop_size = int(pop_size)
    count = int(count)
    if pop_size <= 0 or count <= 0:
        return []
    if count >= pop_size:
        return list(range(pop_size))
    indices = np.random.choice(pop_size, size=count, replace=False)
    return [int(i) for i in np.atleast_1d(indices)]


def _migration_size_for_engine(engine: BaseEvolutionEngine) -> int:
    pop_size = int(getattr(engine, "pop_size", 0) or 0)
    return max(0, pop_size // 2)


def _serialize_migrant(engine: BaseEvolutionEngine, algo: object) -> Dict[str, object]:
    """Serialize an individual for cross-process migration.

    Prefer a DSL string to avoid pickling surprises and reduce payload size.
    """

    input_dim = int(
        getattr(algo, "input_dim", getattr(engine.task, "input_dim", 0) or 0) or 0
    )
    dsl = DSLParser.to_dsl(algo)  # type: ignore[arg-type]
    return {"format": "dsl", "dsl": dsl, "input_dim": input_dim}


def _deserialize_migrant(engine: BaseEvolutionEngine, payload: object) -> object:
    """Reconstruct a migrated individual inside the local worker process."""

    if isinstance(payload, dict):
        fmt = payload.get("format")
        if fmt == "dsl":
            dsl = str(payload.get("dsl") or "")
            input_dim = payload.get("input_dim")
            if input_dim is None:
                input_dim = getattr(engine.task, "input_dim", None)
            if input_dim is None:
                raise ValueError("Cannot deserialize migrant DSL without input_dim")
            return DSLParser.from_dsl(dsl, int(input_dim))
        return payload.get("algo")

    if isinstance(payload, str):
        input_dim = getattr(engine.task, "input_dim", None)
        if input_dim is None:
            raise ValueError("Cannot deserialize migrant DSL without input_dim")
        return DSLParser.from_dsl(payload, int(input_dim))

    return payload


def _load_population_payloads_from_json(
    raw: object,
    *,
    worker_id: int,
    task_type: str,
    engine_type: str,
) -> Optional[List[object]]:
    """
    Best-effort population loader.

    Supports shapes:
    - {"workers": {"0": {"population": [...]}, "1": [...]}, "task_type": "...", "engine": "..."}
    - {"population": [...], "task_type": "...", "engine": "..."}
    - [{"type": "population", "worker_id": 0, "population": [...], ...}, ...]  # sidecar items
    - [...]  # direct list of individuals
    """

    def _matches(meta: object, expected: str) -> bool:
        if meta is None:
            return True
        try:
            return str(meta).strip().lower() == str(expected).strip().lower()
        except Exception:
            return False

    if isinstance(raw, dict):
        if not _matches(raw.get("task_type"), task_type):
            return None
        if not _matches(raw.get("engine"), engine_type):
            return None

        workers = raw.get("workers")
        if isinstance(workers, dict):
            selected = None
            for key in (str(worker_id), int(worker_id), "default"):
                if key in workers:
                    selected = workers.get(key)
                    break
            if selected is None:
                return None
            if isinstance(selected, dict):
                pop = selected.get("population")
                if isinstance(pop, list):
                    return list(pop)
            if isinstance(selected, list):
                return list(selected)
            return None

        pop = raw.get("population")
        if isinstance(pop, list):
            return list(pop)
        return None

    if isinstance(raw, list):
        # Sidecar list of events (choose latest matching population entry).
        for item in reversed(raw):
            if not isinstance(item, dict):
                continue
            if str(item.get("type") or "population") != "population":
                continue
            if "worker_id" in item:
                try:
                    if int(item.get("worker_id")) != int(worker_id):
                        continue
                except Exception:
                    continue
            if not _matches(item.get("task_type"), task_type):
                continue
            if not _matches(item.get("engine"), engine_type):
                continue
            pop = item.get("population")
            if isinstance(pop, list):
                return list(pop)

        # Direct list of individuals.
        return list(raw)

    return None


def _apply_initial_population(engine: BaseEvolutionEngine, payloads: List[object]) -> bool:
    payloads = list(payloads or [])
    if not payloads:
        return False

    algos: List[object] = []
    fitnesses: List[float] = []

    for payload in payloads:
        algo = _deserialize_migrant(engine, payload)
        if algo is None:
            continue

        fitness = None
        if isinstance(payload, dict) and payload.get("fitness") is not None:
            try:
                fitness = float(payload.get("fitness"))
            except Exception:
                fitness = None

        if fitness is None:
            try:
                fitness = float(engine._evaluate_on_miner_tasks(algo))  # type: ignore[arg-type]
            except Exception:
                fitness = -np.inf

        algos.append(algo)
        fitnesses.append(float(fitness))

    if not algos:
        return False

    pop_size = int(getattr(engine, "pop_size", len(algos)) or len(algos))
    if pop_size <= 0:
        pop_size = len(algos)

    # Normalize length to engine.pop_size.
    if len(algos) > pop_size:
        algos = algos[:pop_size]
        fitnesses = fitnesses[:pop_size]
    elif len(algos) < pop_size:
        missing = pop_size - len(algos)
        for _ in range(missing):
            try:
                algo = engine.create_initial_algorithm()
                fit = float(engine._evaluate_on_miner_tasks(algo))
            except Exception:
                algo = engine.create_initial_algorithm()
                fit = -np.inf
            algos.append(algo)
            fitnesses.append(float(fit))

    # Baseline engine stores a FIFO queue; archive engine uses list population.
    if hasattr(engine, "_population_queue"):
        try:
            engine._population_queue = deque(  # type: ignore[attr-defined]
                [{"algo": a, "fitness": f} for a, f in zip(algos, fitnesses)],
                maxlen=pop_size,
            )
            engine.population = [entry["algo"] for entry in engine._population_queue]  # type: ignore[attr-defined]
            best_idx = max(range(len(fitnesses)), key=lambda i: float(fitnesses[i]))
            engine.best_algo = algos[best_idx]
            engine.best_fitness = float(fitnesses[best_idx])
            return True
        except Exception:
            pass

    try:
        engine.population = list(algos)
        best_idx = max(range(len(fitnesses)), key=lambda i: float(fitnesses[i]))
        engine.best_algo = algos[best_idx]
        engine.best_fitness = float(fitnesses[best_idx])
        return True
    except Exception:
        return False


def _maybe_load_initial_population(
    engine: BaseEvolutionEngine,
    args: argparse.Namespace,
    *,
    worker_id: int,
) -> None:
    path = getattr(args, "initial_population_path", None)
    if not path:
        return

    try:
        p = Path(str(path)).expanduser().resolve()
    except Exception:
        return
    if not p.exists():
        return

    try:
        raw = json.loads(p.read_text())
    except Exception:
        return

    payloads = _load_population_payloads_from_json(
        raw,
        worker_id=int(worker_id),
        task_type=str(getattr(args, "task_type", "") or ""),
        engine_type=str(getattr(args, "engine", "") or ""),
    )
    if not payloads:
        return
    if _apply_initial_population(engine, payloads) and getattr(args, "verbose", False):
        print(f"[meta] Loaded initial population for worker {int(worker_id)} from {p}")


def _export_migrants(
    engine: BaseEvolutionEngine, migrate_count: int
) -> Tuple[List[int], List[object]]:
    migrate_count = int(migrate_count)
    if migrate_count <= 0:
        return [], []

    if hasattr(engine, "_population_queue") and getattr(engine, "_population_queue", None) is not None:
        queue_list = list(getattr(engine, "_population_queue"))
        indices = _sample_indices(len(queue_list), migrate_count)
        migrants = [
            _serialize_migrant(engine, queue_list[idx]["algo"])
            for idx in indices
            if 0 <= idx < len(queue_list)
        ]
        return indices, migrants

    population = list(getattr(engine, "population", []) or [])
    indices = _sample_indices(len(population), migrate_count)
    migrants = [
        _serialize_migrant(engine, population[idx])
        for idx in indices
        if 0 <= idx < len(population)
    ]
    return indices, migrants


def _serialize_population_snapshot(
    engine: BaseEvolutionEngine,
    population: List[object],
    scores: List[object],
) -> List[Dict[str, object]]:
    snapshot: List[Dict[str, object]] = []
    for algo, score in zip(list(population or []), list(scores or [])):
        entry = dict(_serialize_migrant(engine, algo))
        if score is not None:
            try:
                entry["fitness"] = float(score)
            except Exception:
                pass
        snapshot.append(entry)
    return snapshot


def _apply_migrants(
    engine: BaseEvolutionEngine,
    replace_indices: List[int],
    incoming: List[object],
) -> None:
    if not replace_indices or not incoming:
        return

    if hasattr(engine, "_population_queue") and getattr(engine, "_population_queue", None) is not None:
        queue_list = list(getattr(engine, "_population_queue"))
        for idx, payload in zip(replace_indices, incoming):
            if idx < 0 or idx >= len(queue_list):
                continue
            algo = _deserialize_migrant(engine, payload)
            try:
                fitness = engine._evaluate_on_miner_tasks(algo)  # type: ignore[attr-defined]
            except Exception:
                fitness = -np.inf
            queue_list[idx] = {"algo": algo, "fitness": fitness}

        engine._population_queue = deque(queue_list, maxlen=int(engine.pop_size))  # type: ignore[attr-defined]
        engine.population = [entry["algo"] for entry in engine._population_queue]  # type: ignore[attr-defined]

        try:
            best_entry = max(queue_list, key=lambda entry: float(entry.get("fitness", -np.inf)))
        except Exception:
            best_entry = None
        if best_entry is not None:
            try:
                best_fit = float(best_entry.get("fitness", -np.inf))
            except Exception:
                best_fit = -np.inf
            if engine.best_algo is None or best_fit > float(engine.best_fitness):
                engine.best_fitness = best_fit
                engine.best_algo = best_entry.get("algo")
        return

    population = list(getattr(engine, "population", []) or [])
    if not population:
        return
    for idx, payload in zip(replace_indices, incoming):
        if idx < 0 or idx >= len(population):
            continue
        population[idx] = _deserialize_migrant(engine, payload)
    engine.population = population


def _warn_about_blas_threads(workers: int) -> None:
    if workers <= 1:
        return
    thread_envs = [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]
    for key in thread_envs:
        if os.environ.get(key) is None:
            os.environ[key] = "1"


def _seed_worker_rng(base_seed: Optional[int], worker_id: int) -> None:
    if base_seed is None:
        return
    worker_seed = int(base_seed) + int(worker_id)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _seed_coordinator_rng(base_seed: Optional[int]) -> None:
    if base_seed is None:
        return
    coord_seed = int(base_seed) + 1000003
    random.seed(coord_seed)
    np.random.seed(coord_seed)


def _run_single(args: argparse.Namespace) -> None:
    _seed_worker_rng(getattr(args, "seed", None), 0)
    engine = _build_engine(args)
    _maybe_load_initial_population(engine, args, worker_id=0)
    best_validated_score = -np.inf
    validate_every = max(0, int(args.validate_every))
    # "validate every N" should first validate at iteration N (not at 1).
    last_validation_iteration = 0

    csv_writer = None
    csv_file = None
    csv_path = None
    run_tag = _sanitize_run_tag(getattr(args, "run_tag", None))
    start_time = time.time()
    last_time = start_time
    last_iter = 0
    if args.csv_dir:
        os.makedirs(args.csv_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        val_count = evaluations.VALIDATOR_TASK_COUNT
        tag_prefix = f"{run_tag}_" if run_tag else ""
        filename = (
            f"{timestamp}_{tag_prefix}{args.engine}_pop{args.pop_size}_mt{args.miner_task_count}_"
            f"vt{val_count}_tor{args.tournament_size}_mut{args.mutation_prob:.2f}_"
            f"iter{args.iterations}.csv"
        )
        csv_path = os.path.join(args.csv_dir, filename)
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            [
                "timestamp",
                "run_tag",
                "iteration",
                "engine",
                "pop_size",
                "miner_task_count",
                "validator_task_count",
                "tournament_size",
                "mutation_prob",
                "sota_threshold",
                "engine_best",
                "pop_best",
                "pop_mean",
                "pop_q1",
                "pop_q3",
                "validator_score",
                "validation_status",
            ]
        )
        csv_file.flush()

    log_every = max(1, int(args.log_every))
    perf_now = time.perf_counter
    block_start = perf_now()
    block_iter_start = 0
    block_evolve_s = 0.0
    block_validate_s = 0.0
    block_roundtrip_s = 0.0
    try:
        for iteration in range(1, args.iterations + 1):
            iter_start = perf_now()
            evolve_start = iter_start
            best_algo, best_score, _, scores = engine.evolve_generation()
            evolve_end = perf_now()
            block_evolve_s += float(evolve_end - evolve_start)

            pop_best, pop_median, pop_q1, pop_q3 = _compute_population_stats(scores)

            if iteration % log_every == 0 or iteration == 1:
                now = time.time()
                delta_time = now - last_time
                rate = None
                if delta_time > 0:
                    rate = (iteration - last_iter) / delta_time
                last_time = now
                last_iter = iteration

                rate_txt = f" rate={rate:.2f} it/s" if rate is not None else ""
                print(
                    f"Iter {iteration:04d}: engine_best={best_score:.4f} "
                    f"pop_best={pop_best:.4f} pop_median={pop_median:.4f} "
                    f"pop_q1={pop_q1:.4f} pop_q3={pop_q3:.4f}{rate_txt}"
                )

            validation_status = None
            validator_score = None
            roundtrip_duration_s = 0.0
            should_validate = (
                validate_every > 0
                and (
                    args.task_type == "scalar_linear"
                    or args.task_type in getattr(
                        evaluations, "SUPPORTED_OFFLINE_BINARY_TASK_TYPES", (DEFAULT_TASK_TYPE,)
                    )
                )
                and best_algo is not None
                and best_score > best_validated_score
                and (iteration - last_validation_iteration) >= validate_every
            )
            if should_validate:
                algo_dsl = DSLParser.to_dsl(best_algo)
                solution = {
                    "task_type": args.task_type,
                    "algorithm_dsl": algo_dsl,
                    "input_dim": engine.task.input_dim,
                    "eval_score": best_score,
                }
                if getattr(args, "dsl_roundtrip_check", False):
                    rt_start = perf_now()
                    native, reconstructed, warn_count, parse_out, native_ok, dsl_ok, total_tasks = _roundtrip_validator_suite_scores(
                        best_algo,
                        algo_dsl,
                        task_type=args.task_type,
                        input_dim=int(engine.task.input_dim),
                    )
                    rt_end = perf_now()
                    roundtrip_duration_s = float(rt_end - rt_start)
                    if np.isfinite(native) and np.isfinite(reconstructed):
                        delta_txt = f"{(float(reconstructed) - float(native)):+.4f}"
                    else:
                        delta_txt = "n/a"
                    warn_txt = f" parse_warnings={int(warn_count)}" if warn_count else ""
                    finite_txt = (
                        f" finite={int(native_ok)}/{int(total_tasks)},{int(dsl_ok)}/{int(total_tasks)}"
                        if total_tasks
                        else ""
                    )
                    native_txt = f"{float(native):.4f}" if np.isfinite(native) else str(native)
                    dsl_txt = f"{float(reconstructed):.4f}" if np.isfinite(reconstructed) else str(reconstructed)
                    print(
                        f"[roundtrip] iter {iteration}: "
                        f"native={native_txt} dsl={dsl_txt} delta={delta_txt}{finite_txt}{warn_txt}"
                    )
                    if warn_count and getattr(args, "verbose", False):
                        excerpt = "\n".join(parse_out.splitlines()[:5])
                        print(f"[roundtrip] parse output (first lines):\n{excerpt}")
                    validator_score = float(reconstructed)
                    passed = float(validator_score) >= float(args.sota_threshold)
                else:
                    validate_start = perf_now()
                    if args.task_type == "scalar_linear":
                        train_samples = getattr(engine.task, "_train_samples", None)
                        val_samples = getattr(engine.task, "_val_samples", None)
                        validator_score = _score_scalar_linear_validator_suite(
                            best_algo,
                            input_dim=int(engine.task.input_dim),
                            train_samples=train_samples,
                            val_samples=val_samples,
                            validator_task_count=getattr(evaluations, "VALIDATOR_TASK_COUNT", None),
                        )
                        passed = float(validator_score) >= float(args.sota_threshold)
                    else:
                        passed, validator_score = evaluations.verify_solution_quality_on_task_type(
                            solution, args.task_type, args.sota_threshold
                        )
                    validate_end = perf_now()
                    block_validate_s += float(validate_end - validate_start)
                validation_status = "passed" if passed else "failed"
                best_validated_score = best_score
                last_validation_iteration = iteration
                block_roundtrip_s += float(roundtrip_duration_s)
                print(
                    f"   validation: status={validation_status} validator_score={validator_score:.4f}"
                )

            iter_end = perf_now()

            if getattr(args, "timing", False) and (iteration % log_every == 0 or iteration == 1):
                block_end = iter_end
                iters = max(1, int(iteration - block_iter_start))
                total_s = float(block_end - block_start)
                other_s = max(0.0, total_s - block_evolve_s - block_validate_s - block_roundtrip_s)
                print(
                    f"   timing: evolve={block_evolve_s/iters:.4f}s "
                    f"validate={block_validate_s/iters:.4f}s "
                    f"roundtrip={block_roundtrip_s/iters:.4f}s "
                    f"other={other_s/iters:.4f}s "
                    f"(avg/iter over last {iters})"
                )
                get_fec_stats = getattr(engine, "get_fec_stats", None)
                if callable(get_fec_stats):
                    try:
                        fec = get_fec_stats()
                    except Exception:
                        fec = None
                    if isinstance(fec, dict) and fec:
                        lookups = int(fec.get("lookups", 0) or 0)
                        hits = int(fec.get("hits", 0) or 0)
                        misses = int(fec.get("misses", 0) or 0)
                        size = int(fec.get("size", 0) or 0)
                        cap = int(fec.get("capacity", 0) or 0)
                        hit_rate = (hits / lookups) if lookups > 0 else 0.0
                        print(
                            f"   fec: lookups={lookups} hits={hits} misses={misses} "
                            f"hit_rate={hit_rate:.3f} size={size} cap={cap}"
                        )
                block_start = perf_now()
                block_iter_start = iteration
                block_evolve_s = 0.0
                block_validate_s = 0.0
                block_roundtrip_s = 0.0

            if csv_writer:
                csv_writer.writerow(
                    [
                        time.strftime("%Y-%m-%d %H:%M:%S"),
                        run_tag,
                        iteration,
                        args.engine,
                        args.pop_size,
                        args.miner_task_count,
                        evaluations.VALIDATOR_TASK_COUNT,
                        args.tournament_size,
                        args.mutation_prob,
                        args.sota_threshold,
                        best_score,
                        pop_best,
                        pop_median,
                        pop_q1,
                        pop_q3,
                        validator_score,
                        validation_status,
                    ]
                )
                csv_file.flush()
    finally:
        if csv_file:
            csv_file.close()
            print(f"CSV log written to {csv_path}")


def _worker_loop(
    args: argparse.Namespace,
    *,
    worker_id: int,
    out_queue: "mp.Queue",
    in_queue: "mp.Queue",
    stop_event: "mp.Event",
) -> None:
    _configure_offline_data_dir(args)
    _configure_validator_task_count(getattr(args, "validator_task_count", None))
    _seed_worker_rng(getattr(args, "seed", None), worker_id)

    engine = _build_engine(args)
    _maybe_load_initial_population(engine, args, worker_id=worker_id)
    validate_every = max(0, int(args.validate_every))
    snapshot_every = max(0, int(getattr(args, "population_snapshot_every", 0) or 0))
    last_validation_iteration = 0
    best_checked_score = -np.inf
    best_verified_score = -np.inf

    start_time = time.time()
    last_time = start_time
    last_iter = 0
    log_every = max(1, int(args.log_every))
    migration_iters = max(0, int(getattr(args, "migration_iters", 0) or 0))

    iteration = 0
    for iteration in range(1, args.iterations + 1):
        if stop_event.is_set():
            break

        best_algo, best_score, population, scores = engine.evolve_generation()

        if getattr(args, "worker_logs", True) and (iteration % log_every == 0 or iteration == 1):
            pop_best, pop_median, pop_q1, pop_q3 = _compute_population_stats(scores)
            now = time.time()
            delta_time = now - last_time
            rate = None
            if delta_time > 0:
                rate = (iteration - last_iter) / delta_time
            last_time = now
            last_iter = iteration
            out_queue.put(
                {
                    "type": "progress",
                    "worker_id": worker_id,
                    "iteration": iteration,
                    "engine_best": float(best_score),
                    "pop_best": float(pop_best),
                    "pop_median": float(pop_median),
                    "pop_q1": float(pop_q1),
                    "pop_q3": float(pop_q3),
                    "rate": float(rate) if rate is not None else None,
                }
            )

        if snapshot_every > 0 and iteration % snapshot_every == 0:
            try:
                pop_list = list(population or [])
            except Exception:
                pop_list = []
            snap = _serialize_population_snapshot(engine, pop_list, list(scores or []))
            out_queue.put(
                {
                    "type": "population",
                    "worker_id": worker_id,
                    "iteration": iteration,
                    "task_type": args.task_type,
                    "engine": args.engine,
                    "population": snap,
                }
            )

        should_validate = (
            validate_every > 0
            and (
                args.task_type == "scalar_linear"
                or args.task_type in getattr(
                    evaluations, "SUPPORTED_OFFLINE_BINARY_TASK_TYPES", (DEFAULT_TASK_TYPE,)
                )
            )
            and best_algo is not None
            and best_score > best_checked_score
            and (iteration - last_validation_iteration) >= validate_every
        )
        if should_validate:
            algo_dsl = DSLParser.to_dsl(best_algo)
            solution = {
                "task_type": args.task_type,
                "algorithm_dsl": algo_dsl,
                "input_dim": int(engine.task.input_dim),
                "eval_score": float(best_score),
            }
            if getattr(args, "dsl_roundtrip_check", False):
                native, reconstructed, warn_count, parse_out, native_ok, dsl_ok, total_tasks = _roundtrip_validator_suite_scores(
                    best_algo,
                    algo_dsl,
                    task_type=args.task_type,
                    input_dim=int(engine.task.input_dim),
                )
                if np.isfinite(native) and np.isfinite(reconstructed):
                    delta_txt = f"{(float(reconstructed) - float(native)):+.4f}"
                else:
                    delta_txt = "n/a"
                warn_txt = f" parse_warnings={int(warn_count)}" if warn_count else ""
                finite_txt = (
                    f" finite={int(native_ok)}/{int(total_tasks)},{int(dsl_ok)}/{int(total_tasks)}"
                    if total_tasks
                    else ""
                )
                native_txt = f"{float(native):.4f}" if np.isfinite(native) else str(native)
                dsl_txt = f"{float(reconstructed):.4f}" if np.isfinite(reconstructed) else str(reconstructed)
                print(
                    f"[roundtrip] w{worker_id} iter {iteration}: "
                    f"native={native_txt} dsl={dsl_txt} delta={delta_txt}{finite_txt}{warn_txt}"
                )
                if warn_count and getattr(args, "verbose", False):
                    excerpt = "\n".join(parse_out.splitlines()[:5])
                    print(f"[roundtrip] parse output (first lines):\n{excerpt}")
                validator_score = float(reconstructed)
                passed = float(validator_score) >= float(args.sota_threshold)
            else:
                if args.task_type == "scalar_linear":
                    train_samples = getattr(engine.task, "_train_samples", None)
                    val_samples = getattr(engine.task, "_val_samples", None)
                    validator_score = _score_scalar_linear_validator_suite(
                        best_algo,
                        input_dim=int(engine.task.input_dim),
                        train_samples=train_samples,
                        val_samples=val_samples,
                        validator_task_count=getattr(evaluations, "VALIDATOR_TASK_COUNT", None),
                    )
                    passed = float(validator_score) >= float(args.sota_threshold)
                else:
                    passed, validator_score = evaluations.verify_solution_quality_on_task_type(
                        solution, args.task_type, args.sota_threshold
                    )
            best_checked_score = best_score
            last_validation_iteration = iteration
            if passed and float(validator_score) > float(best_verified_score):
                best_verified_score = float(validator_score)
                out_queue.put(
                    {
                        "type": "candidate",
                        "worker_id": worker_id,
                        "iteration": iteration,
                        "eval_score": float(best_score),
                        "validator_score": float(validator_score),
                        "input_dim": int(engine.task.input_dim),
                        "algorithm_dsl": algo_dsl,
                    }
                )

        should_migrate = migration_iters > 0 and iteration % migration_iters == 0
        if should_migrate and not stop_event.is_set():
            migrate_count = _migration_size_for_engine(engine)
            replace_indices, migrants = _export_migrants(engine, migrate_count)
            if migrants:
                out_queue.put(
                    {
                        "type": "migration_request",
                        "worker_id": worker_id,
                        "iteration": iteration,
                        "count": len(migrants),
                        "migrants": migrants,
                    }
                )
                while not stop_event.is_set():
                    try:
                        resp = in_queue.get(timeout=0.5)
                    except queue_module.Empty:
                        continue
                    if (
                        isinstance(resp, dict)
                        and resp.get("type") == "migration_response"
                        and int(resp.get("iteration", -1)) == int(iteration)
                    ):
                        incoming = resp.get("incoming") or []
                        _apply_migrants(engine, replace_indices, list(incoming))
                        break

    out_queue.put({"type": "done", "worker_id": worker_id, "iteration": iteration})


def _worker_entrypoint(
    args: argparse.Namespace,
    worker_id: int,
    out_queue: "mp.Queue",
    in_queue: "mp.Queue",
    stop_event: "mp.Event",
) -> None:
    try:
        _worker_loop(
            args,
            worker_id=worker_id,
            out_queue=out_queue,
            in_queue=in_queue,
            stop_event=stop_event,
        )
    except KeyboardInterrupt:
        stop_event.set()
        out_queue.put({"type": "done", "worker_id": worker_id, "iteration": -1})
    except Exception:
        stop_event.set()
        out_queue.put(
            {
                "type": "error",
                "worker_id": worker_id,
                "traceback": traceback.format_exc(),
            }
        )


def _run_meta(args: argparse.Namespace) -> None:
    workers = max(1, int(args.workers))
    _warn_about_blas_threads(workers)
    _seed_coordinator_rng(getattr(args, "seed", None))

    ctx = mp.get_context("spawn")
    out_queue = ctx.Queue()
    stop_event = ctx.Event()

    processes: Dict[int, mp.Process] = {}
    in_queues: Dict[int, mp.Queue] = {}
    for worker_id in range(workers):
        in_queue = ctx.Queue()
        in_queues[worker_id] = in_queue
        proc = ctx.Process(
            target=_worker_entrypoint,
            args=(args, worker_id, out_queue, in_queue, stop_event),
            name=f"miner-worker-{worker_id}",
        )
        proc.start()
        processes[worker_id] = proc

    csv_writer = None
    csv_file = None
    csv_path = None
    run_tag = _sanitize_run_tag(getattr(args, "run_tag", None))
    if args.csv_dir:
        os.makedirs(args.csv_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        val_count = evaluations.VALIDATOR_TASK_COUNT
        tag_prefix = f"{run_tag}_" if run_tag else ""
        filename = (
            f"{timestamp}_{tag_prefix}meta_{args.engine}_w{workers}_pop{args.pop_size}_mt{args.miner_task_count}_"
            f"vt{val_count}_iter{args.iterations}.csv"
        )
        csv_path = os.path.join(args.csv_dir, filename)
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            [
                "timestamp",
                "run_tag",
                "worker_id",
                "iteration",
                "engine",
                "pop_size",
                "miner_task_count",
                "validator_task_count",
                "sota_threshold",
                "engine_best",
                "validator_score",
                "validation_status",
            ]
        )
        csv_file.flush()

    done_workers = set()
    best_seen = {
        "validator_score": -np.inf,
        "eval_score": -np.inf,
        "worker_id": None,
        "iteration": None,
        "algorithm_dsl": None,
    }
    migration_iteration = None
    pending_migrations: Dict[int, List[object]] = {}
    pending_migration_counts: Dict[int, int] = {}
    try:
        while len(done_workers) < workers:
            try:
                msg = out_queue.get(timeout=0.5)
            except queue_module.Empty:
                # Detect early worker exits.
                for worker_id, proc in processes.items():
                    if worker_id in done_workers:
                        continue
                    if proc.exitcode is None:
                        continue
                    if proc.exitcode != 0:
                        print(f"[coord] worker {worker_id} exited early (exitcode={proc.exitcode})")
                        stop_event.set()
                    done_workers.add(worker_id)
                continue

            msg_type = msg.get("type")
            if msg_type == "progress":
                if not getattr(args, "worker_logs", True):
                    continue
                rate = msg.get("rate")
                rate_txt = f" rate={rate:.2f} it/s" if rate is not None else ""
                print(
                    f"[w{msg['worker_id']}] Iter {int(msg['iteration']):04d}: "
                    f"engine_best={float(msg['engine_best']):.4f} "
                    f"pop_best={float(msg['pop_best']):.4f} pop_median={float(msg['pop_median']):.4f} "
                    f"pop_q1={float(msg['pop_q1']):.4f} pop_q3={float(msg['pop_q3']):.4f}{rate_txt}"
                )
                continue

            if msg_type == "candidate":
                worker_id = int(msg["worker_id"])
                iteration = int(msg["iteration"])
                eval_score = float(msg["eval_score"])
                validator_score = float(msg.get("validator_score", -np.inf))
                algo_dsl = msg.get("algorithm_dsl")
                status = "passed"
                print(
                    f"[coord] w{worker_id} iter {iteration}: local={eval_score:.4f} "
                    f"validator={validator_score:.4f} status={status}"
                )
                if csv_writer:
                    csv_writer.writerow(
                        [
                            time.strftime("%Y-%m-%d %H:%M:%S"),
                            run_tag,
                            worker_id,
                            iteration,
                            args.engine,
                            args.pop_size,
                            args.miner_task_count,
                            evaluations.VALIDATOR_TASK_COUNT,
                            args.sota_threshold,
                            eval_score,
                            float(validator_score),
                            status,
                        ]
                    )
                    csv_file.flush()
                if float(validator_score) > float(best_seen["validator_score"]):
                    best_seen.update(
                        {
                            "validator_score": float(validator_score),
                            "eval_score": eval_score,
                            "worker_id": worker_id,
                            "iteration": iteration,
                            "algorithm_dsl": algo_dsl,
                        }
                    )
                    print(
                        f"[coord] NEW BEST: validator={float(validator_score):.4f} "
                        f"(from w{worker_id} iter {iteration}, local={eval_score:.4f})"
                    )
                continue

            if msg_type == "migration_request":
                worker_id = int(msg["worker_id"])
                iteration = int(msg["iteration"])
                migrants = list(msg.get("migrants") or [])
                if migration_iteration is None:
                    migration_iteration = iteration
                if iteration != int(migration_iteration):
                    print(
                        f"[coord] migration desync: got iter {iteration} (expected {migration_iteration}) from w{worker_id}"
                    )
                    stop_event.set()
                    continue
                pending_migrations[worker_id] = migrants
                pending_migration_counts[worker_id] = len(migrants)
                if len(pending_migrations) >= workers:
                    pool: List[object] = []
                    for wid in range(workers):
                        pool.extend(pending_migrations.get(wid, []))
                    random.shuffle(pool)
                    if getattr(args, "verbose", False):
                        best_sota = best_seen.get("validator_score", -np.inf)
                        best_txt = (
                            f"{float(best_sota):.4f}"
                            if np.isfinite(best_sota)
                            else str(best_sota)
                        )
                        print(
                            f"[coord] migration iter {int(migration_iteration)}: mixing {len(pool)} individuals (best_sota={best_txt})"
                        )
                    cursor = 0
                    for target_worker in range(workers):
                        k = int(pending_migration_counts.get(target_worker, 0) or 0)
                        incoming = pool[cursor : cursor + k]
                        cursor += k
                        q = in_queues.get(target_worker)
                        if q is not None:
                            q.put(
                                {
                                    "type": "migration_response",
                                    "iteration": int(migration_iteration),
                                    "incoming": incoming,
                                }
                            )
                    pending_migrations.clear()
                    pending_migration_counts.clear()
                    migration_iteration = None
                continue

            if msg_type == "error":
                stop_event.set()
                worker_id = msg.get("worker_id")
                print(f"[coord] worker {worker_id} crashed:\n{msg.get('traceback')}")
                if worker_id is not None:
                    done_workers.add(int(worker_id))
                continue

            if msg_type == "done":
                done_workers.add(int(msg["worker_id"]))
                continue
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        stop_event.set()
        for proc in processes.values():
            proc.join(timeout=10)
        for proc in processes.values():
            if proc.is_alive():
                proc.terminate()
        for proc in processes.values():
            proc.join(timeout=5)
        if csv_file:
            csv_file.close()
            print(f"[meta] CSV log written to {csv_path}")

    if best_seen["algorithm_dsl"]:
        print(
            f"[meta] Best passing candidate: validator={float(best_seen['validator_score']):.4f} "
            f"local={float(best_seen['eval_score']):.4f} (w{best_seen['worker_id']} iter {best_seen['iteration']})"
        )


def main() -> None:
    args = parse_args()
    if getattr(args, "config", None):
        try:
            cfg_path = Path(str(args.config)).expanduser().resolve()
        except Exception:
            cfg_path = None
        if cfg_path and cfg_path.exists():
            try:
                raw = json.loads(cfg_path.read_text())
            except Exception:
                raw = None
            if isinstance(raw, dict) and isinstance(raw.get("env"), dict):
                for key, value in raw["env"].items():
                    name = str(key).strip()
                    if not name:
                        continue
                    os.environ[name] = str(value)

    _configure_offline_data_dir(args)
    _configure_validator_task_count(getattr(args, "validator_task_count", None))

    if int(getattr(args, "workers", 1) or 1) > 1:
        _run_meta(args)
    else:
        _run_single(args)


if __name__ == "__main__":
    main()
