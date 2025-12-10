"""
Auto-discovery registry for all concrete tasks.
Importing this module populates TASK_REGISTRY with every Task subclass found
in the current directory and sub-packages.
"""

import importlib
import inspect
import os
import pkgutil

from .base import Task

TASK_REGISTRY: dict[str, type[Task]] = {}


def _collect_tasks():
    """Walk the tasks package and register every concrete Task subclass."""
    pkg_path = os.path.dirname(__file__)
    for _, modname, is_pkg in pkgutil.iter_modules([pkg_path]):
        if modname.startswith("_"):
            continue
        module = importlib.import_module(f".{modname}", package=__name__)
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, Task) and obj is not Task:
                TASK_REGISTRY[obj.__name__] = obj
                globals()[obj.__name__] = obj  # re-export to top-level


_collect_tasks()
