import abc
import logging
import os
import time
import uuid
import json
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import requests

logger = logging.getLogger(__name__)

# Default configuration constants
DEFAULT_RELAY_ENDPOINT = "https://relay.bitsota.com"
LOW_FITNESS_VALUE = -float('inf')

# New AlgorithmArray-based imports
from core.tasks.cifar10 import CIFAR10BinaryTask
from core.evaluations import verify_solution_quality

from .auth_mixins import BittensorAuthMixin
from .engines.ga_engine import BaselineEvolutionEngine
from .engines.archive_engine import ArchiveAwareBaselineEvolution
from .engines.base_engine import BaseEvolutionEngine, DEFAULT_MINER_TASK_COUNT
from .metrics_logger import MinerMetricsLogger

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

    def submit_solution(self, solution_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Submit solution to relay endpoint for validators to retrieve.
        """
        sota_threshold = self._fetch_sota_threshold()

        # The relay expects 'score' but we use 'eval_score' internally
        current_score = solution_data.get("eval_score", LOW_FITNESS_VALUE)

        # Pass the full solution_data dictionary for verification
        is_valid, score = verify_solution_quality(solution_data, sota_threshold)

        if not is_valid:
           return {
                "status": "not_submitted",
                "reason": f"Below SOTA threshold. Score {score} < {sota_threshold}",
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
        payload = {
            "task_id": solution_data.get("task_id", str(uuid.uuid4())),
            "score": current_score,
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
            print(f"Posting payload to relay: {payload}")
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

            return {"status": "submitted", "relay_response": result}

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

            # Check if any algorithm in the population beats SOTA
            sota_breakers = []
            for algo, score in zip(population, scores):
                if score > sota_threshold and score != -np.inf:
                    sota_breakers.append((algo, score))

            if sota_breakers:
                # Sort by score and take the best
                sota_breakers.sort(key=lambda x: x[1], reverse=True)
                winning_algo, winning_score = sota_breakers[0]

                logger.info(
                    f"🎉 SOTA broken in generation {gen}! Score: {winning_score:.4f} > {sota_threshold:.4f}"
                )
                logger.info(
                    f"Found {len(sota_breakers)} SOTA-breaking algorithms in this generation"
                )
                
                if self.metrics_logger:
                    self.metrics_logger.log_sota_breakthrough(
                        gen, winning_score, sota_threshold, len(sota_breakers)
                    )

                # Submit the best SOTA-breaking algorithm
                solution_data = {
                    "task_id": task_data["batch_id"],
                    "task_type": task_type,
                    "algorithm_dsl": str(winning_algo),
                    "eval_score": winning_score,
                    "input_dim": task.input_dim,
                    "generation": gen,
                    "total_algorithms_evaluated": (gen + 1) * engine.pop_size,
                }
                result = self.submit_solution(solution_data)
                
                if self.metrics_logger:
                    self.metrics_logger.log_submission(result, winning_score, gen)
                
                return result

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
                    "algorithm_dsl": str(engine.best_algo),
                    "eval_score": engine.best_fitness,
                    "input_dim": task.input_dim,
                    "generation": max_generations - 1,
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

        generation = 0
        best_ever_score = -np.inf
        generations_since_improvement = 0

        while not self.stop_signal:
            # Evolve one generation
            best_algo, best_score, population, scores = engine.evolve_generation()
            generation += 1

            # Check for improvement
            if best_score > best_ever_score:
                best_ever_score = best_score
                generations_since_improvement = 0
            else:
                generations_since_improvement += 1

            # Check entire population for SOTA breakers
            sota_breakers = []
            for algo, score in zip(population, scores):
                if score > sota_threshold and score != -np.inf:
                    sota_breakers.append((algo, score))

            if sota_breakers:
                # Found SOTA! Sort and take best
                sota_breakers.sort(key=lambda x: x[1], reverse=True)
                winning_algo, winning_score = sota_breakers[0]
                logger.info(
                    f"SOTA BROKEN! Generation {generation}, Score: {winning_score:.4f}"

                )

                # Submit to validators
                submission_result = self.submit_solution(
                    {
                        "task_id": f"sota-mine-{uuid.uuid4()}",
                        "task_type": task_type,
                        "algorithm_dsl": str(winning_algo),
                        "eval_score": winning_score,
                        "input_dim": task.input_dim,
                        "metadata": {
                            "generation": generation,
                            "engine_type": engine_type,
                            "total_algorithms_evaluated": generation * engine.pop_size,
                            "generations_since_improvement": generations_since_improvement,
                            "population_sota_breakers": len(sota_breakers),
                        },
                    }
                )

                return {
                    "status": "submitted",
                    "score": winning_score,
                    "generation": generation,
                    "submission_result": submission_result,
                }

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

        # If we exit the loop due to stop_signal, return appropriate status
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
                        f"Score: {result['score']:.4f}, Generation: {result['generation']}"
                    )
                    logger.info(f"Total SOTA breaks: {self.total_sota_breaks}")

                    time.sleep(5)

                    logger.info("Continuing mining for next SOTA...")
                else:
                    logger.warning(f"Submission failed: {result}")
                    time.sleep(5)

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

    # ------------ internal helpers ---------------------------------------
    def _fetch_sota_threshold(self) -> float:
        """
        Get current SOTA threshold from relay endpoint first, fallback to contract then 0.0.
        """
        try:
            response = requests.get(
                f"{self.relay_endpoint.rstrip('/')}/sota_threshold",
                timeout=5,
            )
            if response.status_code == 200:
                sota = response.json().get("sota_threshold", 0.0)
                if self.verbose:
                    logger.info(f"Fetched SOTA from relay: {sota}")
                return sota
        except Exception as e:
            if self.verbose:
                logger.debug(f"Failed to fetch SOTA from relay: {e}, trying contract")

        if self.contract_manager:
            try:
                sota = self.contract_manager.get_current_sota_threshold()
                if self.verbose:
                    logger.info(f"Fetched SOTA from contract: {sota}")
                return sota
            except Exception as e:
                logger.warning(f"Failed to fetch SOTA from contract: {e}")

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

