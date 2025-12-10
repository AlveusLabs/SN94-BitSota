"""
Evaluation utilities for connecting algorithm evaluation to validator verification.
This module provides the bridge between the core evaluation functions and the validator.
"""

from typing import Dict, Any

# Import from core
from core.evaluations import verify_solution_quality as core_verify_solution_quality


def verify_solution_quality(
    solution_data: Dict[str, Any], sota_threshold: float = None
) -> bool:
    """
    Verify that a submitted solution beats the global SOTA threshold.
    Delegates to core implementation.
    """
    return core_verify_solution_quality(solution_data, sota_threshold)
