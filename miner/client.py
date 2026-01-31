import abc
import logging
import os
import time
import uuid
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import requests

logger = logging.getLogger("miner")

# Default configuration constants
DEFAULT_RELAY_ENDPOINT = "https://relay.bitsota.com"
LOW_FITNESS_VALUE = -float('inf')

# New AlgorithmArray-based imports
from core.tasks.cifar10 import CIFAR10BinaryTask
from core.tasks.mnist import MNISTBinaryTask
from core.tasks.scalar_linear import ScalarLinearRegressionTask
from core.dsl_parser import DSLParser
from core.evaluations import verify_solution_quality
from core.hyperparams import get_miner_hyperparams

from .auth_mixins import BittensorAuthMixin
from .engines.ga_engine import BaselineEvolutionEngine
from .engines.archive_engine import ArchiveAwareBaselineEvolution
from .engines.base_engine import BaseEvolutionEngine, DEFAULT_MINER_TASK_COUNT
from .metrics_logger import MinerMetricsLogger
from .state_store import MinerStateStore

DEFAULT_TASK_TYPE = "cifar10_binary"

TASK_REGISTRY = {
    DEFAULT_TASK_TYPE: CIFAR10BinaryTask,
    "mnist_binary": MNISTBinaryTask,
    "scalar_linear": ScalarLinearRegressionTask,
}



class DirectClient:
    """
    Talks to validator HTTP endpoints directly.
    Expected to be used with an auth mixin that provides _auth_payload method.
    """

    def __init__(
        self,
        public_address: str,
        relay_endpoint: Optional[str] = None,
        verbose: bool = False,
        wallet: Optional[Any] = None,
        metrics_log_file: Optional[str] = "miner_metrics.log",
        contract_manager: Optional[Any] = None,
        miner_task_count: Optional[int] = None,
        validator_task_count: Optional[int] = None,
        fec_cache_size: Optional[int] = None,
        fec_train_examples: Optional[int] = None,
        fec_valid_examples: Optional[int] = None,
        fec_forget_every: Optional[int] = None,
        engine_type: str = "archive",
        validate_every_n_generations: Optional[int] = None,
        engine_params: Optional[Dict[str, Any]] = None,
        submit_only_if_improved: Optional[bool] = None,
        max_submission_attempts_per_generation: Optional[int] = None,
        worker_id: Optional[int] = None,
        persist_state: Optional[bool] = None,
        state_dir: Optional[str] = None,
    ):
        hp = get_miner_hyperparams()

        self.public_address = public_address
        self.relay_endpoint = relay_endpoint or DEFAULT_RELAY_ENDPOINT
        self.verbose = verbose
        self.wallet = wallet
        self.stop_signal = False
        self.total_submissions = 0
        self.total_sota_breaks = 0
        self.mining_start_time = None
        self.metrics_logger = MinerMetricsLogger(metrics_log_file) if metrics_log_file else None
        self.contract_manager = contract_manager
        if miner_task_count is None:
            miner_task_count = int(getattr(hp, "miner_task_count", DEFAULT_MINER_TASK_COUNT) or DEFAULT_MINER_TASK_COUNT)
        self.miner_task_count = max(1, int(miner_task_count))
        self.validator_task_count = None
        if validator_task_count is None:
            validator_task_count = getattr(hp, "validator_task_count", None)
        if validator_task_count is not None:
            try:
                self.validator_task_count = max(1, int(validator_task_count))
            except Exception:
                self.validator_task_count = None

        if fec_cache_size is None:
            fec_cache_size = getattr(hp, "fec_cache_size", None)
        if fec_train_examples is None:
            fec_train_examples = getattr(hp, "fec_train_examples", None)
        if fec_valid_examples is None:
            fec_valid_examples = getattr(hp, "fec_valid_examples", None)
        if fec_forget_every is None:
            fec_forget_every = getattr(hp, "fec_forget_every", None)

        self.fec_cache_size = fec_cache_size
        self.fec_train_examples = fec_train_examples
        self.fec_valid_examples = fec_valid_examples
        self.fec_forget_every = fec_forget_every
        self.default_engine_type = engine_type
        self._engine_cache: Dict[Tuple[str, str], BaseEvolutionEngine] = {}
        self.engine_params: Dict[str, Any] = dict(engine_params) if isinstance(engine_params, dict) else {}
        self._local_best_verified_score: Dict[str, float] = {}
        self._last_local_best_skip_log_key = None
        self._local_best_skip_suppressed = 0
        self._warned_info_level_suppressed = False
        self._last_submission_timestamp = 0.0
        try:
            self.submission_cooldown_seconds = max(0, int(getattr(hp, "submission_cooldown_seconds", 60)))
        except Exception:
            self.submission_cooldown_seconds = 60

        if submit_only_if_improved is None:
            submit_only_if_improved = bool(getattr(hp, "submit_only_if_improved", False))
        self.submit_only_if_improved = bool(submit_only_if_improved)
        if self.submit_only_if_improved:
            logger.info(
                "Miner submission gate enabled: only submit if verified score improves local best"
            )

        if max_submission_attempts_per_generation is None:
            max_submission_attempts_per_generation = getattr(hp, "max_submission_attempts_per_generation", None)
        if max_submission_attempts_per_generation is None:
            max_submission_attempts_per_generation = 3 if self.submit_only_if_improved else 1
        self.max_submission_attempts_per_generation = max(
            1, int(max_submission_attempts_per_generation)
        )

        try:
            default_validate_every = max(1, int(getattr(hp, "validate_every_n_generations", 1)))
        except Exception:
            default_validate_every = 1

        if validate_every_n_generations is not None:
            try:
                self.validate_every_n_generations = max(1, int(validate_every_n_generations))
            except Exception:
                self.validate_every_n_generations = int(default_validate_every)
        else:
            self.validate_every_n_generations = int(default_validate_every)
        if self.validate_every_n_generations > 1:
            logger.info(
                "Miner validation throttle enabled (validate_every_n_generations=%d)",
                int(self.validate_every_n_generations),
            )

        try:
            self.sota_cache_seconds = max(0.0, float(getattr(hp, "sota_cache_seconds", 30.0)))
        except Exception:
            self.sota_cache_seconds = 30.0
        try:
            self.sota_fetch_failure_backoff_seconds = max(
                0.0, float(getattr(hp, "sota_failure_backoff_seconds", 5.0))
            )
        except Exception:
            self.sota_fetch_failure_backoff_seconds = 5.0

        self._cached_sota_threshold: Optional[float] = None
        self._cached_sota_timestamp = 0.0
        self._sota_next_fetch_time = 0.0

        if worker_id is None:
            raw_worker_id = (
                os.getenv("MINER_WORKER_ID")
                or os.getenv("BITSOTA_WORKER_ID")
                or os.getenv("WORKER_ID")
            )
            if raw_worker_id:
                try:
                    worker_id = int(raw_worker_id)
                except Exception:
                    worker_id = 0
            else:
                worker_id = 0
        self.worker_id = int(worker_id)

        if persist_state is None:
            persist_state = bool(getattr(hp, "persist_state", True))
        self.persist_state = bool(persist_state)

        try:
            self.persist_every_n_generations = max(
                0, int(getattr(hp, "persist_every_n_generations", 5000))
            )
        except Exception:
            self.persist_every_n_generations = 5000

        try:
            self.gene_dump_every = max(1, int(getattr(hp, "gene_dump_every", 1000)))
        except Exception:
            self.gene_dump_every = 1000

        state_dir_path: Optional[Path] = None
        if state_dir:
            try:
                state_dir_path = Path(state_dir).expanduser().resolve()
            except Exception:
                state_dir_path = None

        self._state_store = MinerStateStore(
            public_address=self.public_address,
            worker_id=self.worker_id,
            state_dir=state_dir_path,
        )
        if self.persist_state:
            self._restore_persisted_client_state()

    def _maybe_persist_checkpoint(
        self,
        *,
        generation: int,
        task_type: str,
        engine_type: str,
        engine: BaseEvolutionEngine,
    ) -> None:
        if not self.persist_state:
            return

        every = int(getattr(self, "persist_every_n_generations", 0) or 0)
        if every <= 0:
            return
        if int(generation) <= 0 or (int(generation) % every) != 0:
            return

        try:
            path = self._state_store.save_engine_state(
                task_type=str(task_type),
                engine_type=str(engine_type),
                engine=engine,
            )
            self._persist_client_state()
            if path is not None and logger.isEnabledFor(logging.INFO):
                logger.info(
                    "Checkpointed miner state at gen=%d to %s",
                    int(generation),
                    str(path),
                )
        except Exception:
            # Persistence must never crash mining.
            return

    def _restore_persisted_client_state(self) -> None:
        payload = self._state_store.load_client_state()
        if not payload:
            return

        lbs = payload.get("local_best_verified_score")
        if isinstance(lbs, dict):
            restored: Dict[str, float] = {}
            for key, value in lbs.items():
                try:
                    restored[str(key)] = float(value)
                except Exception:
                    continue
            if restored:
                self._local_best_verified_score.update(restored)

        for field in ("total_submissions", "total_sota_breaks"):
            if field in payload:
                try:
                    setattr(self, field, int(payload[field]))
                except Exception:
                    pass

    def _persist_client_state(self) -> None:
        if not self.persist_state:
            return
        try:
            self._state_store.save_client_state(
                {
                    "local_best_verified_score": dict(self._local_best_verified_score),
                    "total_submissions": int(self.total_submissions),
                    "total_sota_breaks": int(self.total_sota_breaks),
                }
            )
        except Exception:
            # Persistence failures must never crash mining.
            return

    def _persist_all_engine_state(self) -> None:
        if not self.persist_state:
            return
        for (task_type, engine_type), engine in list(self._engine_cache.items()):
            try:
                self._state_store.save_engine_state(
                    task_type=str(task_type),
                    engine_type=str(engine_type),
                    engine=engine,
                )
            except Exception:
                continue

    def _submission_cooldown_remaining(self) -> float:
        if not self.submission_cooldown_seconds:
            return 0.0
        if not self._last_submission_timestamp:
            return 0.0
        elapsed = time.time() - float(self._last_submission_timestamp)
        remaining = float(self.submission_cooldown_seconds) - elapsed
        return remaining if remaining > 0 else 0.0

    def _log_local_best_skip(self, task_type: str, verified_score: float, best_verified: float):
        key = (
            task_type,
            round(float(verified_score), 6),
            round(float(best_verified), 6),
        )
        if self._last_local_best_skip_log_key == key:
            self._local_best_skip_suppressed += 1
            return
        if self._local_best_skip_suppressed:
            logger.info(
                "Suppressed %d duplicate local-best skips",
                self._local_best_skip_suppressed,
            )
            self._local_best_skip_suppressed = 0
        self._last_local_best_skip_log_key = key
        logger.info(
            "Skipping submission: verified_score %.6f <= local_best %.6f",
            float(verified_score),
            float(best_verified),
        )

    def _auth_payload(self) -> Dict[str, Any]:
        """
        Return {public_address, signature, message, …} for every request.
        This method should be implemented by auth mixins.
        """
        raise NotImplementedError("Auth payload method must be implemented by mixin")

    def __enter__(self):
        """Context manager entry - return self for use in with statement"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources"""
        self.stop_mining()

    def stop_mining(self):
        """Signal to stop mining gracefully"""
        self.stop_signal = True  # TODO: Does this actually stop the process?
        logger.info("Mining stop signal received")

    def get_local_best_verified_score(self, task_type: str = DEFAULT_TASK_TYPE) -> Optional[float]:
        """Return the best verified (validator-style) score seen locally for a task type."""
        try:
            return self._local_best_verified_score.get(task_type)
        except Exception:
            return None

    # ------------ public API ---------------------------------------------
    def register(self) -> Dict[str, str]:
        """No-op for direct mode."""
        # TODO: don't folks need to still register their wallets?
        return {"status": "registered", "mode": "direct"}

    def get_miner_info(self) -> Dict[str, str]:
        return {"address": self.public_address, "mode": "direct"}

    def get_balance(self) -> Dict[str, Any]:
        return {"balance": 0, "mode": "direct"}

    # ------------ task generation & submission ----------------------------
    def request_task(self, task_type: str) -> Dict[str, Any]:
        """
        Generate a task locally instead of pulling from a pool.
        """
        task_cls = TASK_REGISTRY.get(task_type)
        if not task_cls:
            raise ValueError(f"Unknown task type: {task_type}")

        task = task_cls()
        task.load_data()
        algo = task.create_initial_algorithm()

        return {
            "batch_id": str(uuid.uuid4()),
            "task_type": task_type,
            "functions": [{"id": "initial", "function": str(algo)}],
            "component_type": task_type,
            "algorithm": algo,
        }

    def _get_engine(self, task_type: str, engine_type: str = "archive") -> BaseEvolutionEngine:
        key = (task_type, engine_type)
        if key in self._engine_cache:
            return self._engine_cache[key]

        task_cls = TASK_REGISTRY.get(task_type)
        if not task_cls:
            raise ValueError(f"Unknown task type: {task_type}")

        task = task_cls()
        task.load_data()

        engine_kwargs: Dict[str, Any] = {}
        try:
            pop_size = self.engine_params.get("pop_size")
            if pop_size is not None:
                engine_kwargs["pop_size"] = max(1, int(pop_size))
        except Exception:
            pass

        if engine_type == "baseline":
            try:
                tournament_size = self.engine_params.get("tournament_size")
                if tournament_size is not None:
                    engine_kwargs["tournament_size"] = max(1, int(tournament_size))
            except Exception:
                pass
            try:
                mutation_prob = self.engine_params.get("mutation_prob")
                if mutation_prob is not None:
                    engine_kwargs["mutation_prob"] = float(mutation_prob)
            except Exception:
                pass

        if engine_type == "archive":
            try:
                archive_size = self.engine_params.get("archive_size")
                if archive_size is not None:
                    engine_kwargs["archive_size"] = max(1, int(archive_size))
            except Exception:
                pass

        phase_max_sizes = self.engine_params.get("phase_max_sizes")
        if isinstance(phase_max_sizes, dict):
            cleaned_phase_sizes: Dict[str, int] = {}
            for phase, size in phase_max_sizes.items():
                if not isinstance(phase, str):
                    continue
                try:
                    cleaned_phase_sizes[str(phase)] = max(1, int(size))
                except Exception:
                    continue
            if cleaned_phase_sizes:
                engine_kwargs["phase_max_sizes"] = cleaned_phase_sizes

        for key_name in ("scalar_count", "vector_count", "matrix_count", "vector_dim", "cifar_seed"):
            value = self.engine_params.get(key_name)
            if value is None:
                continue
            try:
                engine_kwargs[key_name] = int(value)
            except Exception:
                continue

        if engine_type == "archive":
            engine = ArchiveAwareBaselineEvolution(
                task,
                **engine_kwargs,
                verbose=self.verbose,
                miner_task_count=self.miner_task_count,
                fec_cache_size=self.fec_cache_size,
                fec_train_examples=self.fec_train_examples,
                fec_valid_examples=self.fec_valid_examples,
                fec_forget_every=self.fec_forget_every,
            )
        elif engine_type == "baseline":
            engine = BaselineEvolutionEngine(
                task,
                **engine_kwargs,
                verbose=self.verbose,
                miner_task_count=self.miner_task_count,
                fec_cache_size=self.fec_cache_size,
                fec_train_examples=self.fec_train_examples,
                fec_valid_examples=self.fec_valid_examples,
                fec_forget_every=self.fec_forget_every,
            )
        else:
            raise ValueError(f"Unknown engine type: {engine_type}")

        if self.persist_state:
            try:
                restored = self._state_store.load_engine_state(
                    task_type=str(task_type),
                    engine_type=str(engine_type),
                    engine=engine,
                )
                if restored:
                    logger.info(
                        "Resumed miner state (worker_id=%d task_type=%s engine_type=%s)",
                        int(self.worker_id),
                        str(task_type),
                        str(engine_type),
                    )
            except Exception:
                pass

        self._engine_cache[key] = engine
        return engine

    def submit_solution(
        self,
        solution_data: Dict[str, Any],
        *,
        prevalidated: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Submit solution to relay endpoint for validators to retrieve.
        """
        cooldown_remaining = self._submission_cooldown_remaining()
        if cooldown_remaining > 0:
            return {
                "status": "not_submitted",
                "reason": "submission_cooldown",
                "cooldown_remaining_seconds": float(cooldown_remaining),
            }

        # The relay expects 'score' but we use 'eval_score' internally
        current_score = solution_data.get("eval_score", LOW_FITNESS_VALUE)

        if prevalidated is None:
            sota_threshold = self._fetch_sota_threshold()
            is_valid, verified_score = verify_solution_quality(
                solution_data,
                sota_threshold,
                task_count=self.validator_task_count,
            )
        else:
            try:
                verified_score = float(prevalidated.get("verified_score", -np.inf))
            except Exception:
                verified_score = -np.inf
            if "sota_threshold" in prevalidated:
                try:
                    sota_threshold = float(prevalidated["sota_threshold"])
                except Exception:
                    sota_threshold = self._fetch_sota_threshold()
            else:
                sota_threshold = self._fetch_sota_threshold()
            is_valid = verified_score >= sota_threshold

        if not is_valid:
            if self.verbose:
                logger.info(
                    "Not submitting: verified_score %.6f < sota_threshold %.6f",
                    float(verified_score),
                    float(sota_threshold),
                )
            return {
                "status": "not_submitted",
                "reason": "below_sota_threshold",
                "verified_score": float(verified_score),
                "sota_threshold": float(sota_threshold),
            }

        task_type = solution_data.get("task_type", DEFAULT_TASK_TYPE)
        best_verified = self._local_best_verified_score.get(task_type)
        if (
            self.submit_only_if_improved
            and best_verified is not None
            and verified_score <= best_verified
        ):
            self._log_local_best_skip(task_type, verified_score, best_verified)
            return {
                "status": "not_submitted",
                "reason": "below_local_best",
                "verified_score": float(verified_score),
                "local_best_verified_score": float(best_verified),
            }

        auth = self._auth_payload()

        # --- Payload transformation for relay ---
        # The relay expects a flat structure defined by its `ResultSubmission` model.
        # We need to map our internal `solution_data` to that structure.

        # 1. The main algorithm description goes into `algorithm_result` as a JSON string.
        #    We exclude fields that the relay expects at the top level.
        algorithm_details = {
            k: v for k, v in solution_data.items() if k not in ["task_id", "eval_score"]
        }

        # 2. Construct the final payload for the body.
        submission_score = float(verified_score)
        payload = {
            "task_id": solution_data.get("task_id", str(uuid.uuid4())),
            "score": submission_score,
            "algorithm_result": algorithm_details,  # Send as a dict
        }

        # 3. Prepare headers for authentication.
        headers = {
            "X-Key": auth.get("public_address"),
            "X-Signature": auth.get("signature"),
            "X-Timestamp": auth.get("message"),
        }
        # --- End of transformation ---

        try:
            if self.verbose:
                logger.info(
                    "Submitting to relay %s task_id=%s score=%.6f (verified) eval_score=%.6f",
                    self.relay_endpoint.rstrip("/"),
                    payload["task_id"],
                    float(submission_score),
                    float(current_score),
                )
            response = requests.post(
                f"{self.relay_endpoint.rstrip('/')}/submit_solution",
                json=payload,
                headers=headers,
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()

            if self.verbose:
                logger.info(f"Solution submitted to relay: {result}")

            if best_verified is None or verified_score > best_verified:
                self._local_best_verified_score[task_type] = float(verified_score)

            self._last_submission_timestamp = time.time()
            self._persist_client_state()
            return {
                "status": "submitted",
                "relay_response": result,
                "verified_score": float(verified_score),
                "eval_score": float(current_score),
            }

        except Exception as e:
            logger.error(f"Failed to submit to relay {self.relay_endpoint}: {e}")
            return {"status": "failed", "error": str(e)}

    # ------------ task processing helpers --------------------------------
    def process_evolution_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process evolution task with early SOTA detection.
        Submits as soon as any algorithm beats SOTA threshold.
        """
        task_type = task_data["task_type"]
        engine_type = getattr(self, "default_engine_type", "archive")
        engine = self._get_engine(task_type, engine_type)
        task = engine.task

        sota_threshold = self._fetch_sota_threshold()
        logger.info(
            f"Starting evolution with {type(engine).__name__}. SOTA threshold: {sota_threshold}"
        )
        
        if self.metrics_logger:
            self.metrics_logger.log_session_start(task_type, type(engine).__name__)

        # Run evolution generation by generation
        max_generations = int(os.getenv("MAX_EVOLUTION_GENERATIONS", 15))

        for gen in range(max_generations):
            # Evolve one generation
            best_algo, best_score, population, scores = engine.evolve_generation()
            self._maybe_persist_checkpoint(
                generation=gen + 1,
                task_type=str(task_type),
                engine_type=str(engine_type),
                engine=engine,
            )

            best_candidate_algo = None
            best_candidate_score = -np.inf
            for algo, score in zip(population, scores):
                if score != -np.inf and score > sota_threshold and score > best_candidate_score:
                    best_candidate_algo = algo
                    best_candidate_score = float(score)

            if best_candidate_algo is not None:
                cooldown_remaining = self._submission_cooldown_remaining()
                if cooldown_remaining > 0:
                    logger.info(
                        "SOTA candidate found but in submission cooldown (%.1fs remaining). Continuing evolution.",
                        float(cooldown_remaining),
                    )
                    continue

                solution_data = {
                    "task_id": task_data["batch_id"],
                    "task_type": task_type,
                    "algorithm_dsl": DSLParser.to_dsl(best_candidate_algo),
                    "eval_score": float(best_candidate_score),
                    "input_dim": task.input_dim,
                    "generation": gen,
                    "total_algorithms_evaluated": (gen + 1) * engine.pop_size,
                    "candidate_rank": 1,
                    "metadata": {"log_all_task_scores": True},
                }
                is_valid, verified_score = verify_solution_quality(
                    solution_data,
                    sota_threshold,
                    task_count=self.validator_task_count,
                )
                if not is_valid:
                    logger.info(
                        "Best SOTA-breaking candidate failed validation (score=%.6f). Continuing evolution.",
                        float(best_candidate_score),
                    )
                    continue

                if self.metrics_logger:
                    self.metrics_logger.log_sota_breakthrough(
                        gen,
                        float(verified_score),
                        float(sota_threshold),
                        1,
                    )

                result = self.submit_solution(
                    solution_data,
                    prevalidated={
                        "verified_score": float(verified_score),
                        "sota_threshold": float(sota_threshold),
                    },
                )

                if self.metrics_logger:
                    self.metrics_logger.log_submission(result, best_candidate_score, gen)

                if result.get("status") == "submitted":
                    return result

                logger.info(
                    "Validated SOTA breaker not submitted (likely blocked by local best gate). Continuing evolution."
                )

            # Log progress even if no SOTA breaker
            valid_scores = [s for s in scores if s != -np.inf]
            if valid_scores:
                logger.info(
                    f"Generation {gen}: best={best_score:.4f}, "
                    f"pop_best={max(valid_scores):.4f}, "
                    f"pop_mean={np.mean(valid_scores):.4f}, "
                    f"distance_to_sota={sota_threshold - max(valid_scores):.4f}"
                )
                
                if self.metrics_logger:
                    self.metrics_logger.log_generation(
                        gen, best_score, scores, sota_threshold, 
                        (gen + 1) * engine.pop_size
                    )

        # If we've exhausted all generations without beating SOTA
        logger.info(
            f"Evolution completed {max_generations} generations. "
            f"Final best score: {engine.best_fitness:.4f}, "
            f"SOTA threshold: {sota_threshold}"
        )

        if engine.best_algo is not None and engine.best_fitness > sota_threshold:
            # Edge case: final best beats SOTA (shouldn't happen with above logic but safety check)
            return self.submit_solution(
                    {
                        "task_id": task_data["batch_id"],
                        "task_type": task_type,
                        "algorithm_dsl": DSLParser.to_dsl(engine.best_algo),
                        "eval_score": engine.best_fitness,
                        "input_dim": task.input_dim,
                        "generation": max_generations - 1,
                        "metadata": {"log_all_task_scores": True},
                    }
                )
        else:
            return {
                "status": "not_submitted",
                "reason": "Below SOTA threshold",
                "best_score": engine.best_fitness,
                "sota_threshold": sota_threshold,
                "generations_run": max_generations,
            }

    # ------------ continuous mining --------------------------------------
    def run_mining_cycle(self, task_type: str = DEFAULT_TASK_TYPE) -> Dict[str, Any]:
        task = self.request_task(task_type)
        return self.process_evolution_task(task)

    def _mine_until_sota(
        self, task_type: str, engine_type: str, checkpoint_generations: int
    ) -> Dict[str, Any]:
        """
        Mine continuously until SOTA is found, then submit.

        Returns:
            Dict with submission result
        """
        # Create task
        engine = self._get_engine(task_type, engine_type)
        task = engine.task

        # Get current SOTA threshold
        sota_threshold = self._fetch_sota_threshold()
        logger.info(f"Current SOTA threshold: {sota_threshold:.4f}")

        try:
            local_best_verified = float(self.get_local_best_verified_score(task_type) or -np.inf)
        except Exception:
            local_best_verified = -np.inf
        best_verified_local = max(float(sota_threshold), float(local_best_verified))
        logger.info(
            "Initial best_verified_local=%.6f (sota_threshold=%.6f local_best_verified=%.6f)",
            float(best_verified_local),
            float(sota_threshold),
            float(local_best_verified),
        )

        generation = 0
        best_ever_score = -np.inf
        generations_since_improvement = 0
        try:
            validate_every = max(1, int(getattr(self, "validate_every_n_generations", 1)))
        except Exception:
            validate_every = 1
        logger.info(f"Validating every {validate_every} generations")
        # Treat "validate every N generations" as: first validation happens at generation N
        # (not at generation 1). Using a negative sentinel causes validation on the first
        # generation for N>1, which is a large and surprising startup cost.
        last_validation_generation = 0
        pending_best_candidate = None
        pending_best_candidate_score = -np.inf
        pending_best_candidate_over_local_count = 0
        pending_best_candidate_from_cooldown = False
        pending_prevalidated = None
        last_submit_attempt_generation = 0

        throttled_mode = validate_every > 1
        pop_size = int(getattr(engine, "pop_size", 0) or 0)
        try:
            gene_dump_every = max(1, int(getattr(self, "gene_dump_every", 1000)))
        except Exception:
            gene_dump_every = 1000
        logger.info(
            "Mining loop start: task_type=%s engine_type=%s pop_size=%d throttled_mode=%s",
            str(task_type),
            str(engine_type),
            int(pop_size),
            bool(throttled_mode),
        )
        logger.info("Gene dump every %d generations", int(gene_dump_every))

        def _submitted_result(submission_result: Dict[str, Any], mining_score: float) -> Dict[str, Any]:
            verified_score = submission_result.get("verified_score")
            return {
                "status": "submitted",
                "score": float(verified_score) if verified_score is not None else float(mining_score),
                "verified_score": float(verified_score) if verified_score is not None else None,
                "mining_score": float(mining_score),
                "generation": generation,
                "submission_result": submission_result,
            }

        while not self.stop_signal:
            # Evolve one generation
            best_algo, best_score, population, scores = engine.evolve_generation()
            generation += 1
            self._maybe_persist_checkpoint(
                generation=generation,
                task_type=str(task_type),
                engine_type=str(engine_type),
                engine=engine,
            )

            # Check for improvement
            if best_score > best_ever_score:
                prev_best_ever = best_ever_score
                best_ever_score = best_score
                generations_since_improvement = 0
                logger.debug(
                    "New best_ever_score at gen=%d: %.6f (prev=%.6f)",
                    int(generation),
                    float(best_ever_score),
                    float(prev_best_ever),
                )
            else:
                generations_since_improvement += 1

            best_over_local_algo = None
            best_over_local_score = -np.inf
            over_local_count = 0
            for algo, score in zip(population, scores):
                if score != -np.inf and score > best_verified_local:
                    over_local_count += 1
                    if score > best_over_local_score:
                        best_over_local_algo = algo
                        best_over_local_score = score

            if best_over_local_algo is not None and float(best_over_local_score) > float(
                pending_best_candidate_score
            ):
                prev_pending = pending_best_candidate_score
                pending_best_candidate = best_over_local_algo
                pending_best_candidate_score = float(best_over_local_score)
                pending_best_candidate_over_local_count = int(over_local_count)
                logger.debug(
                    "New pending_best_candidate at gen=%d: score=%.6f (prev=%.6f) over_local_count=%d phase_sizes=%s",
                    int(generation),
                    float(pending_best_candidate_score),
                    float(prev_pending),
                    int(pending_best_candidate_over_local_count),
                    {
                        "setup": int(best_over_local_algo.get_phase_size("setup"))
                        if "setup" in best_over_local_algo.phase_arrays
                        else 0,
                        "predict": int(best_over_local_algo.get_phase_size("predict"))
                        if "predict" in best_over_local_algo.phase_arrays
                        else 0,
                        "learn": int(best_over_local_algo.get_phase_size("learn"))
                        if "learn" in best_over_local_algo.phase_arrays
                        else 0,
                    },
                )
            elif over_local_count > 0 and logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Candidates over local best at gen=%d: count=%d best_score=%.6f best_verified_local=%.6f",
                    int(generation),
                    int(over_local_count),
                    float(best_over_local_score),
                    float(best_verified_local),
                )

            if (
                pending_best_candidate is not None
                and float(pending_best_candidate_score) <= float(best_verified_local)
            ):
                logger.debug(
                    "Dropping pending_best_candidate at gen=%d: pending_score=%.6f <= best_verified_local=%.6f",
                    int(generation),
                    float(pending_best_candidate_score),
                    float(best_verified_local),
                )
                pending_best_candidate = None
                pending_best_candidate_score = -np.inf
                pending_best_candidate_over_local_count = 0
                pending_best_candidate_from_cooldown = False

            if throttled_mode and pending_prevalidated is not None:
                try:
                    if float(pending_prevalidated.get("verified_score", -np.inf)) <= float(
                        best_verified_local
                    ):
                        logger.debug(
                            "Clearing pending_prevalidated at gen=%d: verified_score=%.6f <= best_verified_local=%.6f",
                            int(generation),
                            float(pending_prevalidated.get("verified_score", -np.inf)),
                            float(best_verified_local),
                        )
                        pending_prevalidated = None
                except Exception:
                    logger.debug(
                        "Clearing pending_prevalidated at gen=%d: invalid verified_score",
                        int(generation),
                    )
                    pending_prevalidated = None

            cooldown_remaining = self._submission_cooldown_remaining()
            if cooldown_remaining > 0 and best_over_local_algo is not None:
                if not pending_best_candidate_from_cooldown:
                    logger.debug(
                        "In submission cooldown (%.1fs remaining); caching candidate opportunities",
                        float(cooldown_remaining),
                    )
                pending_best_candidate_from_cooldown = True

            if (
                throttled_mode
                and pending_prevalidated is not None
                and cooldown_remaining <= 0
                and (generation - last_submit_attempt_generation) >= validate_every
            ):
                last_submit_attempt_generation = generation
                logger.debug(
                    "Retrying submission for prevalidated candidate at gen=%d (verified_score=%.6f)",
                    int(generation),
                    float(pending_prevalidated.get("verified_score", -np.inf)),
                )
                submission_result = self.submit_solution(
                    pending_prevalidated["solution_data"],
                    prevalidated={
                        "verified_score": pending_prevalidated["verified_score"],
                        "sota_threshold": sota_threshold,
                    },
                )
                if submission_result.get("status") == "submitted":
                    mining_score = float(pending_prevalidated.get("candidate_score", -np.inf))
                    logger.info(
                        "Submission succeeded for prevalidated candidate at gen=%d mining_score=%.6f",
                        int(generation),
                        float(mining_score),
                    )
                    return _submitted_result(submission_result, mining_score)
                if submission_result.get("status") == "not_submitted" and submission_result.get(
                    "reason"
                ) in {"below_sota_threshold", "below_local_best"}:
                    logger.info(
                        "Prevalidated submission rejected at gen=%d (status=%s reason=%s); clearing pending_prevalidated",
                        int(generation),
                        submission_result.get("status"),
                        submission_result.get("reason"),
                    )
                    pending_prevalidated = None
                elif logger.isEnabledFor(logging.INFO):
                    logger.info(
                        "No SOTA-breaker submission sent (status=%s reason=%s). Continuing mining.",
                        submission_result.get("status"),
                        submission_result.get("reason"),
                    )
                elif not self._warned_info_level_suppressed:
                    self._warned_info_level_suppressed = True
                    logger.warning(
                        "SOTA-breaker submission not sent but INFO logs are suppressed; "
                        "set logging level to INFO for details (status=%s reason=%s).",
                        submission_result.get("status"),
                        submission_result.get("reason"),
                    )

            should_validate = (
                (not throttled_mode or pending_prevalidated is None)
                and pending_best_candidate is not None
                and cooldown_remaining <= 0
                and (generation - last_validation_generation) >= validate_every
            )
            if should_validate:
                last_validation_generation = generation
                logger.info(
                    "Validating candidate at gen=%d: score=%.6f best_verified_local=%.6f sota_threshold=%.6f cooldown_remaining=%.1f validate_every=%d delayed=%s",
                    int(generation),
                    float(pending_best_candidate_score),
                    float(best_verified_local),
                    float(sota_threshold),
                    float(cooldown_remaining),
                    int(validate_every),
                    bool(pending_best_candidate_from_cooldown),
                )

                metadata: Dict[str, Any] = {
                    "generation": generation,
                    "engine_type": engine_type,
                    "total_algorithms_evaluated": generation * pop_size,
                    "generations_since_improvement": generations_since_improvement,
                    "population_candidates_over_local_best": int(
                        pending_best_candidate_over_local_count
                    ),
                    "candidate_rank": 0 if pending_best_candidate_from_cooldown else 1,
                }
                metadata["log_all_task_scores"] = True
                if pending_best_candidate_from_cooldown:
                    metadata["delayed_submission"] = True
                if validate_every > 1:
                    metadata["validate_every_n_generations"] = validate_every

                solution_data = {
                    "task_id": f"sota-mine-{uuid.uuid4()}",
                    "task_type": task_type,
                    "algorithm_dsl": DSLParser.to_dsl(pending_best_candidate),
                    "eval_score": float(pending_best_candidate_score),
                    "input_dim": task.input_dim,
                    "metadata": metadata,
                }

                is_valid, verified_score = verify_solution_quality(
                    solution_data,
                    sota_threshold,
                    task_count=self.validator_task_count,
                )
                try:
                    verified_score_f = float(verified_score)
                except Exception:
                    verified_score_f = -np.inf
                logger.info(
                    "Validation result at gen=%d: is_valid=%s verified_score=%.6f (mining_score=%.6f sota_threshold=%.6f)",
                    int(generation),
                    bool(is_valid),
                    float(verified_score_f),
                    float(solution_data.get("eval_score", -np.inf)),
                    float(sota_threshold),
                )
                if verified_score_f > best_verified_local:
                    prev_best_verified_local = best_verified_local
                    best_verified_local = verified_score_f
                    logger.info(
                        "Updated best_verified_local at gen=%d: %.6f -> %.6f",
                        int(generation),
                        float(prev_best_verified_local),
                        float(best_verified_local),
                    )

                pending_best_candidate = None
                pending_best_candidate_score = -np.inf
                pending_best_candidate_over_local_count = 0
                pending_best_candidate_from_cooldown = False

                if not is_valid:
                    continue

                last_submit_attempt_generation = generation
                logger.info(
                    "Attempting submission at gen=%d with verified_score=%.6f",
                    int(generation),
                    float(verified_score_f),
                )
                submission_result = self.submit_solution(
                    solution_data,
                    prevalidated={
                        "verified_score": float(verified_score_f),
                        "sota_threshold": sota_threshold,
                    },
                )
                if submission_result.get("status") == "submitted":
                    logger.info(
                        "Submission succeeded at gen=%d (verified_score=%.6f mining_score=%.6f)",
                        int(generation),
                        float(verified_score_f),
                        float(solution_data.get("eval_score", -np.inf)),
                    )
                    return _submitted_result(submission_result, float(solution_data["eval_score"]))
                if throttled_mode:
                    pending_prevalidated = {
                        "solution_data": solution_data,
                        "verified_score": float(verified_score_f),
                        "candidate_score": float(solution_data.get("eval_score", -np.inf)),
                    }
                    if submission_result.get("status") == "not_submitted" and submission_result.get(
                        "reason"
                    ) in {"below_sota_threshold", "below_local_best"}:
                        logger.info(
                            "Submission rejected at gen=%d (status=%s reason=%s); clearing pending_prevalidated",
                            int(generation),
                            submission_result.get("status"),
                            submission_result.get("reason"),
                        )
                        pending_prevalidated = None
                elif logger.isEnabledFor(logging.INFO):
                    logger.info(
                        "No SOTA-breaker submission sent (status=%s reason=%s). Continuing mining.",
                        submission_result.get("status"),
                        submission_result.get("reason"),
                    )
                elif not self._warned_info_level_suppressed:
                    self._warned_info_level_suppressed = True
                    logger.warning(
                        "SOTA-breaker submission not sent but INFO logs are suppressed; "
                        "set logging level to INFO for details (status=%s reason=%s).",
                        submission_result.get("status"),
                        submission_result.get("reason"),
                    )

            if (
                gene_dump_every > 0
                and generation % gene_dump_every == 0
                and logger.isEnabledFor(logging.INFO)
                and best_algo is not None
            ):
                try:
                    from core.algorithm_array import OPCODES

                    opcode_name = {int(v): str(k) for k, v in OPCODES.items()}

                    dump_lines = [
                        f"=== GENE DUMP gen={generation} best_score={float(best_score):.6f} best_ever={float(best_ever_score):.6f} best_verified_local={float(best_verified_local):.6f} ===",
                        "DSL:",
                        DSLParser.to_dsl(best_algo),
                        "",
                        f"Memory: scalar_count={int(getattr(best_algo, 'scalar_count', 0))} vector_count={int(getattr(best_algo, 'vector_count', 0))} matrix_count={int(getattr(best_algo, 'matrix_count', 0))} vector_dim={int(getattr(best_algo, 'vector_dim', 0))}",
                    ]
                    for phase in best_algo.get_phases():
                        ops, arg1, arg2, dest, const1, const2 = best_algo.get_phase_ops(phase)
                        ops_names = [opcode_name.get(int(o), str(int(o))) for o in ops]
                        dump_lines.append(
                            f"\n[{phase}] size={len(ops)} ops={ops_names}"
                        )
                        dump_lines.append(f"[{phase}] arg1={np.array2string(arg1, threshold=100000, max_line_width=240)}")
                        dump_lines.append(f"[{phase}] arg2={np.array2string(arg2, threshold=100000, max_line_width=240)}")
                        dump_lines.append(f"[{phase}] dest={np.array2string(dest, threshold=100000, max_line_width=240)}")
                        dump_lines.append(f"[{phase}] const1={np.array2string(const1, threshold=100000, max_line_width=240)}")
                        dump_lines.append(f"[{phase}] const2={np.array2string(const2, threshold=100000, max_line_width=240)}")

                    logger.info("\n".join(dump_lines))
                except Exception as e:
                    logger.info("Failed to dump gene at gen=%d: %s", int(generation), str(e))

            # Progress logging
            if generation % checkpoint_generations == 0:
                valid_scores = [s for s in scores if s != -np.inf]
                if valid_scores:
                    pop_mean = np.mean(valid_scores)
                    pop_max = max(valid_scores)
                    distance_to_sota = sota_threshold - pop_max

                    logger.info(
                        f"Gen {generation}: best_ever={best_ever_score:.4f}, "
                        f"current_best={best_score:.4f}, pop_mean={pop_mean:.4f}, "
                        f"distance_to_sota={distance_to_sota:.4f}, "
                        f"stagnation={generations_since_improvement}"
                    )

                    # Adaptive restart if heavily stagnated
                    # if generations_since_improvement > 100:
                    #     logger.info("Heavy stagnation detected. Restarting with fresh population...")
                    #     engine.population = None  # Force fresh start
                    #     generations_since_improvement = 0

                    #     # Optionally increase population size
                    #     if hasattr(engine, 'pop_size') and engine.pop_size < 16:
                    #         engine.pop_size = min(engine.pop_size + 2, 16)
                    #         logger.info(f"Increased population size to {engine.pop_size}")

                # Check if we should refresh SOTA threshold (in case it changed)
                if generation % 5000 == 0:
                    new_sota = self._fetch_sota_threshold()
                    if new_sota != sota_threshold:
                        logger.info(
                            f"SOTA threshold updated: {sota_threshold:.4f} -> {new_sota:.4f}"
                        )
                        sota_threshold = new_sota
                        best_verified_local = max(float(best_verified_local), float(sota_threshold))
                        if (
                            pending_best_candidate is not None
                            and float(pending_best_candidate_score) <= float(best_verified_local)
                        ):
                            pending_best_candidate = None
                            pending_best_candidate_score = -np.inf
                            pending_best_candidate_over_local_count = 0
                            pending_best_candidate_from_cooldown = False

        # If we exit the loop due to stop_signal, return appropriate status
        logger.info(
            "Mining loop stopped at gen=%d best_ever_score=%.6f best_verified_local=%.6f sota_threshold=%.6f",
            int(generation),
            float(best_ever_score),
            float(best_verified_local),
            float(sota_threshold),
        )
        return {
            "status": "stopped",
            "reason": "Mining stopped by user or signal",
            "generations_run": generation,
            "best_score": best_ever_score,
            "sota_threshold": sota_threshold,
        }

    def run_continuous_mining(
        self,
        task_type: str = DEFAULT_TASK_TYPE,
        engine_type: str = "archive",  # "baseline" or "archive"
        checkpoint_generations: int = 10,  # Log progress every N generations
    ) -> Dict[str, Any]:
        """
        Run continuous mining until stopped or SOTA found.
        After finding SOTA, submits and continues mining.

        Args:
            task_type: Type of task to mine (from TASK_REGISTRY)
            engine_type: Evolution engine to use
            checkpoint_generations: Generations between progress logs

        Returns:
            Dict with final mining statistics
        """
        self.stop_signal = False
        logger.info(
            f"Starting continuous mining for {task_type} with {engine_type} engine"
        )
        self.mining_start_time = time.time()
        
        if self.metrics_logger:
            self.metrics_logger.log_session_start(task_type, engine_type)

        while not self.stop_signal:
            try:
                result = self._mine_until_sota(
                    task_type, engine_type, checkpoint_generations
                )

                if result["status"] == "submitted":
                    self.total_submissions += 1
                    self.total_sota_breaks += 1

                    logger.info(
                        f"SOTA submission #{self.total_submissions} successful!"
                    )
                    logger.info(
                        "Score: %.4f (verified), Mining Score: %.4f, Generation: %d",
                        float(result.get("verified_score", result.get("score", -np.inf))),
                        float(result.get("mining_score", -np.inf)),
                        int(result.get("generation", -1)),
                    )
                    logger.info(f"Total SOTA breaks: {self.total_sota_breaks}")

                    logger.info("Continuing mining for next SOTA...")
                else:
                    logger.warning(f"Submission failed: {result}")

            except KeyboardInterrupt:
                logger.info("Mining interrupted by user")
                break
            except Exception as e:
                logger.error(f"Mining error: {e}", exc_info=True)
                time.sleep(10)  # Pause on error before retry

        if self.persist_state:
            self._persist_all_engine_state()
            self._persist_client_state()

        # Final stats
        runtime = time.time() - self.mining_start_time
        logger.info(f"Mining stopped. Runtime: {runtime / 3600:.2f} hours")
        logger.info(f"Total SOTA submissions: {self.total_submissions}")

        if self.metrics_logger:
            self.metrics_logger.log_session_end(self.total_submissions, self.total_sota_breaks)

        return {
            "status": "stopped",
            "runtime_hours": runtime / 3600,
            "total_submissions": self.total_submissions,
            "total_sota_breaks": self.total_sota_breaks,
        }

    # ------------ internal helpers ---------------------------------------
    def _fetch_sota_threshold(self, *, force_refresh: bool = False) -> float:
        """
        Get current SOTA threshold from relay endpoint first (cached), fallback to contract then 0.0.
        """
        now = time.time()
        if not force_refresh:
            if (
                self._cached_sota_threshold is not None
                and self.sota_cache_seconds > 0
                and now < float(self._sota_next_fetch_time or 0.0)
            ):
                return float(self._cached_sota_threshold)
            if now < float(self._sota_next_fetch_time or 0.0):
                # Throttled due to a recent failure; fall back to cached value if available.
                return float(self._cached_sota_threshold or 0.0)

        try:
            response = requests.get(
                f"{self.relay_endpoint.rstrip('/')}/sota_threshold",
                timeout=5,
            )
            if response.status_code == 200:
                sota = float(response.json().get("sota_threshold", 0.0) or 0.0)
                prev = self._cached_sota_threshold
                self._cached_sota_threshold = sota
                self._cached_sota_timestamp = now
                self._sota_next_fetch_time = (
                    now + float(self.sota_cache_seconds or 0.0)
                    if self.sota_cache_seconds
                    else now
                )
                if self.verbose and (prev is None or abs(float(prev) - sota) > 1e-12):
                    logger.info(f"Fetched SOTA from relay: {sota}")
                return float(sota)
            if self.verbose:
                logger.debug(
                    "Unexpected status from relay /sota_threshold: %s", response.status_code
                )
            self._sota_next_fetch_time = now + float(
                self.sota_fetch_failure_backoff_seconds or 0.0
            )
        except Exception as e:
            if self.verbose:
                logger.debug(f"Failed to fetch SOTA from relay: {e}, trying contract")
            self._sota_next_fetch_time = now + float(
                self.sota_fetch_failure_backoff_seconds or 0.0
            )

        if self.contract_manager:
            try:
                sota = float(self.contract_manager.get_current_sota_threshold() or 0.0)
                prev = self._cached_sota_threshold
                self._cached_sota_threshold = sota
                self._cached_sota_timestamp = now
                self._sota_next_fetch_time = (
                    now + float(self.sota_cache_seconds or 0.0)
                    if self.sota_cache_seconds
                    else now
                )
                if self.verbose and (prev is None or abs(float(prev) - sota) > 1e-12):
                    logger.info(f"Fetched SOTA from contract: {sota}")
                return float(sota)
            except Exception as e:
                logger.warning(f"Failed to fetch SOTA from contract: {e}")
                self._sota_next_fetch_time = now + float(
                    self.sota_fetch_failure_backoff_seconds or 0.0
                )

        return 0.0

class PoolClient:
    """
    Classic pool mode – all communication via ONE pool URL.
    No validators, no direct submissions.
    """

    def __init__(
        self,
        public_address: str,
        base_url: str = "https://pool.hivetensor.com/",
    ):
        self.public_address = public_address
        self.base_url = base_url.rstrip("/")
        self.api_prefix = "/api/v1"

    def __enter__(self):
        """Context manager entry - return self for use in with statement"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources"""
        pass  # No specific cleanup needed for PoolClient

    # ------------ auth abstraction ---------------------------------------
    @abc.abstractmethod
    def _auth_payload(self) -> Dict[str, Any]:
        pass

    # ------------ pool API -----------------------------------------------
    def register(self) -> Dict[str, Any]:
        r = requests.post(
            f"{self.base_url}{self.api_prefix}/miners/register",
            json=self._auth_payload(),
            timeout=10,
        )
        r.raise_for_status()
        return r.json()

    def get_miner_info(self) -> Dict[str, Any]:
        r = requests.get(
            f"{self.base_url}{self.api_prefix}/miners/{self.public_address}",
            timeout=10,
        )
        r.raise_for_status()
        return r.json()

    def get_balance(self) -> Dict[str, Any]:
        r = requests.get(
            f"{self.base_url}{self.api_prefix}/miners/{self.public_address}/balance",
            timeout=10,
        )
        r.raise_for_status()
        return r.json()

    # ------------ task flow ----------------------------------------------
    def request_task(self, task_type: str, max_retries: int = 3) -> Dict[str, Any]:
        payload = {"task_type": task_type, **self._auth_payload()}
        for attempt in range(1, max_retries + 1):
            try:
                r = requests.post(
                    f"{self.base_url}{self.api_prefix}/tasks/{self.public_address}/request",
                    json=payload,
                    timeout=10,
                )
                r.raise_for_status()
                return r.json()
            except Exception as e:
                logger.warning(f"request_task attempt {attempt}: {e}")
                if attempt == max_retries:
                    raise
                time.sleep(1)

        raise RuntimeError("Unexpected end of retry loop")

    def submit_evolution(  # TODO: how come not used? if not needed we can delete to reduce confusion
        self,
        batch_id: str,
        evolved_function: str,
        parent_functions: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = {
            **self._auth_payload(),
            "batch_id": batch_id,
            "evolved_function": evolved_function,
            "parent_functions": parent_functions,
            "metadata": metadata or {},
        }
        r = requests.post(
            f"{self.base_url}{self.api_prefix}/evolution/submit",
            json=payload,
            timeout=30,
        )
        r.raise_for_status()
        return r.json()

    def submit_evaluation(
        self,
        batch_id: str,
        evaluations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        payload = {
            **self._auth_payload(),
            "batch_id": batch_id,
            "evaluations": evaluations,
        }
        r = requests.post(
            f"{self.base_url}{self.api_prefix}/evaluation/submit",
            json=payload,
            timeout=30,
        )
        r.raise_for_status()
        return r.json()

    # ------------ mining loops ------------------------------------------
    def run_mining_cycle(
        self, task_type: str = DEFAULT_TASK_TYPE
    ) -> Dict[str, Any]:  # TODO: needed for only testing or actually used?
        task = self.request_task(task_type)
        return task  # caller decides how to process

    def run_continuous_mining(
        self,
        cycles: int = 0,
        alternate: bool = True,
        delay: float = 5.0,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        count = 0
        while cycles == 0 or count < cycles:
            try:
                task_type = (
                    "evolve" if (count % 2 == 0 or not alternate) else "evaluate"
                )
                task = self.request_task(task_type, max_retries=max_retries)
                # TODO: Shouldn't task be running here?
                logger.info(f"Retrieved task {count + 1}: {task}")
                count += 1
                if delay > 0:
                    time.sleep(delay)
            except Exception as e:
                logger.error(f"Continuous mining error: {e}")
                if delay > 0:
                    time.sleep(delay)

        return {"status": "completed", "cycles_completed": count}

    def reset_active_tasks(self) -> Dict[str, Any]:
        r = requests.post(
            f"{self.base_url}{self.api_prefix}/tasks/{self.public_address}/reset",
            json=self._auth_payload(),
            timeout=10,
        )
        r.raise_for_status()
        return r.json()


class BittensorDirectClient(BittensorAuthMixin, DirectClient):
    pass


class BittensorPoolClient(BittensorAuthMixin, PoolClient):
    pass
