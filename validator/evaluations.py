"""
Evaluation utilities for connecting algorithm evaluation to validator verification.
This module provides the bridge between the core evaluation functions and the validator.
"""

import os
from typing import Dict, Any, Optional

# Import from core
from core.evaluations import verify_solution_quality as core_verify_solution_quality


def verify_solution_quality(
    solution_data: Dict[str, Any],
    sota_threshold: float = None,
    *,
    epochs: Optional[int] = None,
    task_count: Optional[int] = None,
    task_seed: Optional[int] = None,
    n_samples: Optional[int] = None,
    train_split: Optional[float] = None,
) -> Any:
    """
    Verify that a submitted solution beats the global SOTA threshold.
    Delegates to core implementation.
    """
    os.environ["AUTOML_ZERO_SHARED_MEMORY"] = "1"
    return core_verify_solution_quality(
        solution_data,
        sota_threshold,
        epochs=epochs,
        task_count=task_count,
        task_seed=task_seed,
        n_samples=n_samples,
        train_split=train_split,
    )
