#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path


def main() -> int:
    repo_dir = Path(os.environ["BITSOTA_AGENT_REPO_DIR"])
    submission_path = Path(os.environ["BITSOTA_AGENT_SUBMISSION_PATH"])
    miner_index = int(os.environ.get("BITSOTA_LOCAL_MINER_INDEX", "0"))
    metric_value = max(0.000001, 0.81 - (miner_index * 0.01))
    train_path = repo_dir / "train.py"
    train_path.write_text(
        f"score = {metric_value:.6f}\nprint({{'heldout_ppl': score}})\n",
        encoding="utf-8",
    )
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission_path.write_text(
        json.dumps(
            {
                "summary": (
                    "Deterministic local miner pass for the SOTA Base multi-miner swarm "
                    f"(miner {miner_index})."
                ),
                "claimed_metrics": {"heldout_ppl": metric_value},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
