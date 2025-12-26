from __future__ import annotations

import json
import math
import os
import platform
from pathlib import Path
from typing import Any, Dict, Optional

STATE_VERSION = 1


def default_state_path() -> Path:
    if platform.system().lower() == "windows":
        app_data = os.environ.get(
            "APPDATA", str(Path.home() / "AppData" / "Roaming")
        )
        base_dir = Path(app_data) / "BitSota"
    else:
        base_dir = Path.home() / ".bitsota"
    return base_dir / "mining_state.json"


def score_to_json(value: Any) -> Any:
    if value is None:
        return None
    try:
        value_f = float(value)
    except Exception:
        return None
    if math.isnan(value_f):
        return "nan"
    if math.isinf(value_f):
        return "inf" if value_f > 0 else "-inf"
    return value_f


def score_from_json(value: Any, default: float = -float("inf")) -> float:
    if value is None:
        return float(default)
    if isinstance(value, str):
        if value == "inf":
            return float("inf")
        if value == "-inf":
            return -float("inf")
        if value == "nan":
            return float("nan")
    try:
        return float(value)
    except Exception:
        return float(default)


def read_state_file(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except Exception:
        return None


def write_state_file(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
    else:
        tmp_path = Path(f"{path}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)
    os.replace(tmp_path, path)
