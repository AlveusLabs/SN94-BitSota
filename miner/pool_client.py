import requests
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.dsl_parser import DSLParser
from core.tasks.cifar10 import CIFAR10BinaryTask
from miner.engines.archive_engine import ArchiveAwareBaselineEvolution

logger = logging.getLogger(__name__)

DEFAULT_TASK_TYPE = "cifar10_binary"

TASK_REGISTRY = {
    DEFAULT_TASK_TYPE: CIFAR10BinaryTask,
}


class PoolClient:
    def __init__(self, pool_url: str, public_address: str, wallet=None):
        self.pool_url = pool_url.rstrip('/')
        self.public_address = public_address
        self.wallet = wallet
        self.registered = False
        self.task_cache = {}

    def _get_auth_payload(self) -> Dict:
        if not self.wallet:
            return {}
        msg = f"auth:{int(time.time())}"
        sig = self.wallet.hotkey.sign(msg).hex()
        return {
            "public_address": self.wallet.hotkey.ss58_address,
            "signature": sig,
            "message": msg,
        }

    def register(self) -> bool:
        try:
            response = requests.post(
                f"{self.pool_url}/api/v1/miners/register",
                json={"public_address": self.public_address},
                timeout=10
            )
            response.raise_for_status()
            self.registered = True
            logger.info(f"Registered with pool: {response.json()}")
            return True
        except requests.exceptions.ConnectionError:
            logger.error(f"Cannot connect to pool at {self.pool_url}. Pool server may be down.")
            return False
        except requests.exceptions.Timeout:
            logger.error(f"Pool registration timed out. Check your connection.")
            return False
        except Exception as e:
            logger.error(f"Pool registration failed: {e}")
            return False

    def request_task(self, task_type: str = "evolve") -> Optional[Dict]:
        if not self.registered and not self.register():
            logger.error("Cannot request task - not registered with pool")
            return None

        try:
            response = requests.post(
                f"{self.pool_url}/api/v1/tasks/{self.public_address}/request",
                json={"task_type": task_type},
                timeout=30
            )
            response.raise_for_status()
            task = response.json()

            if not task.get('algorithms') or len(task.get('algorithms', [])) == 0:
                logger.warning("Pool returned empty task - no algorithms available")
                return None

            logger.info(f"Received {task_type} task with {len(task.get('algorithms', []))} algorithms")
            return task
        except requests.exceptions.ConnectionError:
            logger.error(f"Cannot connect to pool at {self.pool_url}")
            return None
        except requests.exceptions.Timeout:
            logger.error("Task request timed out")
            return None
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.warning("No tasks available from pool at this time")
            else:
                logger.error(f"HTTP error requesting task: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to request task: {e}")
            return None

    @staticmethod
    def evolve_algorithm(algorithm_dsl: str, task_type: str, input_dim: int, generations: int = 3) -> Optional[str]:
        try:
            task_class = TASK_REGISTRY.get(task_type)
            if not task_class:
                logger.error(f"Unknown task type: {task_type}")
                return None

            task = task_class()
            if hasattr(task, "sample_miner_task_spec"):
                spec = task.sample_miner_task_spec(input_dim)
                task.load_data(task_spec=spec)
            else:
                task.load_data()

            engine = ArchiveAwareBaselineEvolution(task=task, pop_size=5, verbose=False)
            best_algo, best_fitness = engine.evolve(generations=generations)

            if best_algo is None:
                logger.error("Evolution produced no valid algorithms")
                return None

            evolved_dsl = DSLParser.to_dsl(best_algo)
            logger.info(f"Evolution complete. Score: {best_fitness:.4f}")
            return evolved_dsl

        except Exception as e:
            logger.error(f"Evolution failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def submit_evolution(self, batch_id: str, evolved_function: str, parent_ids: list = None) -> bool:
        try:
            payload = {
                "evolved_function": evolved_function,
                "parent_functions": parent_ids or []
            }
            payload.update(self._get_auth_payload())

            response = requests.post(
                f"{self.pool_url}/api/v1/tasks/{self.public_address}/{batch_id}/submit_evolution",
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            logger.info("Evolution submitted successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to submit evolution: {e}")
            return False

    @staticmethod
    def evaluate_algorithm(algorithm_dsl: str, task_type: str, input_dim: int) -> Optional[float]:
        try:
            task_class = TASK_REGISTRY.get(task_type)
            if not task_class:
                logger.error(f"Unknown task type: {task_type}")
                return None

            task = task_class()
            if hasattr(task, "sample_miner_task_spec"):
                spec = task.sample_miner_task_spec(input_dim)
                task.load_data(task_spec=spec)
            else:
                task.load_data()

            algorithm = DSLParser.from_dsl(algorithm_dsl, input_dim)
            score = task.evaluate_algorithm(algorithm, epochs=1)
            #TODO: Clarify with Ali on evaluation metrics
            logger.info(f"Evaluation complete. Score: {score}")
            return float(score)

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return None

    def submit_evaluation(self, batch_id: str, evaluations: list, evaluation_metrics: dict = None) -> bool:
        try:
            payload = {
                "evaluations": evaluations,
                "evaluation_metrics": evaluation_metrics
            }
            payload.update(self._get_auth_payload())

            response = requests.post(
                f"{self.pool_url}/api/v1/tasks/{self.public_address}/{batch_id}/submit_evaluation",
                json=payload,
            )
            response.raise_for_status()
            logger.info("Evaluation submitted successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to submit evaluation: {e}")
            return False

    def get_balance(self) -> Optional[Dict]:
        try:
            response = requests.get(
                f"{self.pool_url}/api/v1/rewards/balance/{self.public_address}"
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return None

    def request_withdrawal(self) -> Optional[Dict]:
        try:
            response = requests.post(
                f"{self.pool_url}/api/v1/rewards/withdraw/{self.public_address}"
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to request withdrawal: {e}")
            return None

    def get_reward_history(self) -> Optional[Dict]:
        try:
            response = requests.get(
                f"{self.pool_url}/api/v1/rewards/history/{self.public_address}"
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get reward history: {e}")
            return None

    def check_pending_evaluations(self) -> bool:
        try:
            response = requests.get(
                f"{self.pool_url}/api/v1/tasks/{self.public_address}/pending_evaluations"
            )
            response.raise_for_status()
            data = response.json()
            return data.get('has_pending_evaluations', False)
        except Exception as e:
            logger.error(f"Failed to check pending evaluations: {e}")
            return False
