"""
Validator package containing evaluation utilities and validation logic.
This package provides the bridge between core evaluation functions and validator operations.
"""

def verify_solution_quality(*args, **kwargs):
    from .evaluations import verify_solution_quality as _verify_solution_quality

    return _verify_solution_quality(*args, **kwargs)

__all__ = [
    "verify_solution_quality",
]
