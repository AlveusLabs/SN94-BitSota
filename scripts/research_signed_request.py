#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from neurons import research_agent_miner


if __name__ == "__main__":
    raise SystemExit(research_agent_miner.main(["signed-request", *sys.argv[1:]]))
