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
from core.dsl_parser import DSLParser
from core.evaluations import verify_solution_quality

from .auth_mixins import BittensorAuthMixin
from .engines.ga_engine import BaselineEvolutionEngine
from .engines.archive_engine import ArchiveAwareBaselineEvolution
from .engines.base_engine import BaseEvolutionEngine, DEFAULT_MINER_TASK_COUNT
from .metrics_logger import MinerMetricsLogger
from .state_store import (
    STATE_VERSION,
    default_state_path,
    read_state_file,
    score_from_json,
    score_to_json,
    write_state_file,
)

DEFAULT_TASK_TYPE = "cifar10_binary"

TASK_REGISTRY = {
    DEFAULT_TASK_TYPE: CIFAR10BinaryTask,
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
        engine_type: str = "archive",
        submit_only_if_improved: Optional[bool] = None,
        max_submission_attempts_per_generation: Optional[int] = None,
    ):
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
        self.miner_task_count = max(1, miner_task_count or DEFAULT_MINER_TASK_COUNT)
        self.default_engine_type = engine_type
        self._engine_cache: Dict[Tuple[str, str], BaseEvolutionEngine] = {}
        self._local_best_verified_score: Dict[str, float] = {}
        self._last_local_best_skip_log_key = None
        self._local_best_skip_suppressed = 0
        self._warned_info_level_suppressed = False
        self._last_submission_timestamp = 0.0
        try:
            self.submission_cooldown_seconds = max(
                0, int(os.getenv("MINER_SUBMISSION_COOLDOWN_SECONDS", "60"))
            )
        except Exception:
            self.submission_cooldown_seconds = 60

        if submit_only_if_improved is None:
            gate = os.getenv("MINER_SUBMIT_ONLY_IF_IMPROVED", "").strip().lower()
            submit_only_if_improved = gate in {"1", "true", "yes", "y", "on"}
        self.submit_only_if_improved = bool(submit_only_if_improved)
        if self.submit_only_if_improved:
            logger.info(
                "Miner submission gate enabled (MINER_SUBMIT_ONLY_IF_IMPROVED): only submit if verified score improves local best"
            )

        if max_submission_attempts_per_generation is None:
            raw = os.getenv("MINER_MAX_SUBMISSION_ATTEMPTS_PER_GENERATION", "").strip()
            if raw:
                try:
                    max_submission_attempts_per_generation = int(raw)
                except Exception:
                    max_submission_attempts_per_generation = None
        if max_submission_attempts_per_generation is None:
            max_submission_attempts_per_generation = 3 if self.submit_only_if_improved else 1
        self.max_submission_attempts_per_generation = max(
            1, int(max_submission_attempts_per_generation)
        )

        try:
            self.validate_every_n_generations = max(
                1, int(os.getenv("MINER_VALIDATE_EVERY_N_GENERATIONS", "1"))
            )
        except Exception:
            self.validate_every_n_generations = 1
        if self.validate_every_n_generations > 1:
            logger.info(
                "Miner validation throttle enabled (MINER_VALIDATE_EVERY_N_GENERATIONS=%d)",
                int(self.validate_every_n_generations),
            )

        try:
            self.sota_cache_seconds = max(
                0.0, float(os.getenv("MINER_SOTA_CACHE_SECONDS", "30"))
            )
        except Exception:
            self.sota_cache_seconds = 30.0
        try:
            self.sota_fetch_failure_backoff_seconds = max(
                0.0, float(os.getenv("MINER_SOTA_FAILURE_BACKOFF_SECONDS", "5"))
            )
        except Exception:
            self.sota_fetch_failure_backoff_seconds = 5.0

        self._cached_sota_threshold: Optional[float] = None
        self._cached_sota_timestamp = 0.0
        self._sota_next_fetch_time = 0.0

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

        if engine_type == "archive":
            engine = ArchiveAwareBaselineEvolution(
                task, verbose=self.verbose, miner_task_count=self.miner_task_count
            )
        elif engine_type == "baseline":
            engine = BaselineEvolutionEngine(
                task, verbose=self.verbose, miner_task_count=self.miner_task_count
            )
        else:
            raise ValueError(f"Unknown engine type: {engine_type}")

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
                solution_data, sota_threshold
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
                is_valid, verified_score = verify_solution_quality(solution_data, sota_threshold)
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
        self,
        task_type: str,
        engine_type: str,
        checkpoint_generations: int,
        *,
        state_path: Optional[str] = None,
        resume_from_state: bool = False,
    ) -> Dict[str, Any]:
        """
        Mine continuously until SOTA is found, then submit.

        Returns:
            Dict with submission result

        Notes:
            When state_path is set, population state is periodically persisted and can be resumed.
        """
        # Create task
        engine = self._get_engine(task_type, engine_type)
        task = engine.task

        state_path_resolved = self._resolve_state_path(state_path)
        resume_state = None
        if resume_from_state and state_path_resolved:
            resume_state = self._load_population_state(
                state_path=state_path_resolved,
                task_type=task_type,
                engine_type=engine_type,
                engine=engine,
            )

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
        if resume_state:
            try:
                generation = int(resume_state.get("generation", generation) or generation)
            except Exception:
                generation = 0
            if "best_ever_score" in resume_state:
                best_ever_score = score_from_json(
                    resume_state.get("best_ever_score"), default=best_ever_score
                )
            elif engine.best_fitness is not None:
                best_ever_score = float(engine.best_fitness)
            if "generations_since_improvement" in resume_state:
                try:
                    generations_since_improvement = int(
                        resume_state.get("generations_since_improvement", 0) or 0
                    )
                except Exception:
                    generations_since_improvement = 0
        try:
            validate_every = max(1, int(getattr(self, "validate_every_n_generations", 1)))
        except Exception:
            validate_every = 1
        logger.info(f"Validating every {validate_every} generations")
        last_validation_generation = -validate_every
        pending_best_candidate = None
        pending_best_candidate_score = -np.inf
        pending_best_candidate_over_local_count = 0
        pending_best_candidate_from_cooldown = False
        pending_prevalidated = None
        last_submit_attempt_generation = -validate_every

        throttled_mode = validate_every > 1
        pop_size = int(getattr(engine, "pop_size", 0) or 0)
        try:
            gene_dump_every = max(1, int(os.getenv("MINER_GENE_DUMP_EVERY", "1000")))
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

        def _maybe_save_state() -> None:
            if not state_path_resolved:
                return
            self._save_population_state(
                state_path=state_path_resolved,
                task_type=task_type,
                engine_type=engine_type,
                engine=engine,
                generation=generation,
                best_ever_score=best_ever_score,
                generations_since_improvement=generations_since_improvement,
            )

        while not self.stop_signal:
            # Evolve one generation
            best_algo, best_score, population, scores = engine.evolve_generation()
            generation += 1

            # Check for improvement
            if best_score > best_ever_score:
                prev_best_ever = best_ever_score
                best_ever_score = best_score
                generations_since_improvement = 0
                logger.info(
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
                logger.info(
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
            elif over_local_count > 0 and logger.isEnabledFor(logging.INFO):
                logger.info(
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
                logger.info(
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
                        logger.info(
                            "Clearing pending_prevalidated at gen=%d: verified_score=%.6f <= best_verified_local=%.6f",
                            int(generation),
                            float(pending_prevalidated.get("verified_score", -np.inf)),
                            float(best_verified_local),
                        )
                        pending_prevalidated = None
                except Exception:
                    logger.info(
                        "Clearing pending_prevalidated at gen=%d: invalid verified_score",
                        int(generation),
                    )
                    pending_prevalidated = None

            cooldown_remaining = self._submission_cooldown_remaining()
            if cooldown_remaining > 0 and best_over_local_algo is not None:
                if not pending_best_candidate_from_cooldown:
                    logger.info(
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
                logger.info(
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
                    _maybe_save_state()
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

                is_valid, verified_score = verify_solution_quality(solution_data, sota_threshold)
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
                    _maybe_save_state()
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
                if generation % 50 == 0:
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
                _maybe_save_state()

        # If we exit the loop due to stop_signal, return appropriate status
        logger.info(
            "Mining loop stopped at gen=%d best_ever_score=%.6f best_verified_local=%.6f sota_threshold=%.6f",
            int(generation),
            float(best_ever_score),
            float(best_verified_local),
            float(sota_threshold),
        )
        _maybe_save_state()
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
        state_path: Optional[str] = None,
        resume_from_state: bool = False,
    ) -> Dict[str, Any]:
        """
        Run continuous mining until stopped or SOTA found.
        After finding SOTA, submits and continues mining.

        Args:
            task_type: Type of task to mine (from TASK_REGISTRY)
            engine_type: Evolution engine to use
            checkpoint_generations: Generations between progress logs
            state_path: Optional path to persist population state
            resume_from_state: Whether to resume from existing state if available

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
                    task_type,
                    engine_type,
                    checkpoint_generations,
                    state_path=state_path,
                    resume_from_state=resume_from_state,
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

    def _resolve_state_path(self, state_path: Optional[str]) -> Optional[Path]:
        if state_path is None:
            resolved = default_state_path()
        else:
            if isinstance(state_path, str) and not state_path.strip():
                return None
            try:
                resolved = Path(state_path).expanduser().resolve()
            except Exception:
                return None
        try:
            resolved = resolved.expanduser().resolve()
        except Exception:
            return None
        if resolved.exists() and resolved.is_dir():
            logger.warning("State path %s is a directory; skipping state persistence", str(resolved))
            return None
        return resolved

    def _load_population_state(
        self,
        *,
        state_path: Path,
        task_type: str,
        engine_type: str,
        engine: BaseEvolutionEngine,
    ) -> Optional[Dict[str, Any]]:
        state = read_state_file(state_path)
        if not state:
            return None
        version = state.get("version")
        if version is not None:
            try:
                version_int = int(version)
            except Exception:
                version_int = None
            if version_int != int(STATE_VERSION):
                logger.warning(
                    "State file %s has unsupported version=%s; skipping resume",
                    str(state_path),
                    str(version),
                )
                return None
        if state.get("task_type") != task_type or state.get("engine_type") != engine_type:
            logger.warning(
                "State file %s does not match task_type=%s engine_type=%s; skipping resume",
                str(state_path),
                str(task_type),
                str(engine_type),
            )
            return None
        engine_state = state.get("engine_state")
        if not engine_state:
            logger.warning("State file %s missing engine_state; skipping resume", str(state_path))
            return None
        saved_pop_size = engine_state.get("pop_size")
        try:
            if saved_pop_size is not None and int(saved_pop_size) != int(engine.pop_size):
                logger.warning(
                    "State file pop_size=%s does not match engine pop_size=%s; skipping resume",
                    str(saved_pop_size),
                    str(engine.pop_size),
                )
                return None
        except Exception:
            pass
        if engine.population is not None:
            logger.info(
                "Engine already initialized for %s/%s; skipping state load",
                str(task_type),
                str(engine_type),
            )
            return None
        try:
            engine.load_state(engine_state)
        except Exception as exc:
            logger.warning("Failed to load engine state from %s: %s", str(state_path), str(exc))
            return None
        run_state = state.get("run_state") or {}
        if "local_best_verified_score" in run_state:
            self._local_best_verified_score[task_type] = score_from_json(
                run_state.get("local_best_verified_score")
            )
        if "last_submission_timestamp" in run_state:
            try:
                self._last_submission_timestamp = float(
                    run_state.get("last_submission_timestamp", 0.0) or 0.0
                )
            except Exception:
                pass
        logger.info(
            "Resumed mining state from %s (generation=%s)",
            str(state_path),
            str(run_state.get("generation", "?")),
        )
        return run_state

    def _save_population_state(
        self,
        *,
        state_path: Path,
        task_type: str,
        engine_type: str,
        engine: BaseEvolutionEngine,
        generation: int,
        best_ever_score: float,
        generations_since_improvement: int,
    ) -> None:
        run_state = {
            "generation": int(generation),
            "best_ever_score": score_to_json(best_ever_score),
            "generations_since_improvement": int(generations_since_improvement),
            "last_submission_timestamp": float(self._last_submission_timestamp or 0.0),
        }
        local_best = self._local_best_verified_score.get(task_type)
        if local_best is not None:
            run_state["local_best_verified_score"] = score_to_json(local_best)
        payload = {
            "version": int(STATE_VERSION),
            "saved_at": float(time.time()),
            "task_type": str(task_type),
            "engine_type": str(engine_type),
            "run_state": run_state,
            "engine_state": engine.get_state(),
        }
        try:
            write_state_file(state_path, payload)
        except Exception as exc:
            logger.warning(
                "Failed to save mining state to %s: %s",
                str(state_path),
                str(exc),
            )

    def clear_population_state(
        self,
        *,
        task_type: Optional[str] = None,
        engine_type: Optional[str] = None,
        state_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        path = self._resolve_state_path(state_path)
        cleared = False
        error = None
        if path and path.exists():
            try:
                path.unlink()
                cleared = True
            except Exception as exc:
                error = str(exc)

        if task_type and engine_type:
            self._engine_cache.pop((task_type, engine_type), None)
        else:
            self._engine_cache = {}

        if task_type:
            self._local_best_verified_score.pop(task_type, None)
        else:
            self._local_best_verified_score = {}

        if task_type or cleared:
            self._last_submission_timestamp = 0.0

        if error:
            return {"status": "error", "message": error, "path": str(path) if path else None}
        if cleared:
            return {"status": "cleared", "path": str(path) if path else None}
        return {"status": "not_found", "message": "No state file to clear", "path": str(path) if path else None}

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
