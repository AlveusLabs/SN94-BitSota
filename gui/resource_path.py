import sys
import os
from pathlib import Path


def resource_path(relative_path: str) -> str:
    if getattr(sys, "frozen", False):
        base_path = Path(sys._MEIPASS)
    else:
        base_path = Path(__file__).parent.parent

    # Use POSIX separators so Qt/QSS url() parsing works on Windows too
    return str((base_path / relative_path).resolve().as_posix())
