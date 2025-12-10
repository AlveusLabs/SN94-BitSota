"""
Core substrate: algorithm DSL, memory layout, base task, and evaluator.
The abstract Task base is imported from tasks.base so both miners
and validators share a single source of truth.
"""

from .algorithm import Algorithm, create_initial_algorithm, create_ops_summary
from .memory import Memory
from .tasks.base import Task  # re-exported for convenience

__all__ = [
    "Task",
    "Algorithm",
    "Memory",
    "create_initial_algorithm",
    "create_ops_summary",
]
