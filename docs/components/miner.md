# Miner

Current public mining is coordinator-backed.

Miner tools in this repo:

- `bitsota-research-agent list-tasks`
- `bitsota-research-agent submit-workspace`
- `bitsota-research-agent signed-request`

## Paths

| Path | Use it when |
| --- | --- |
| [Mining Without an Agent](../mining.md) | You edit and submit the task repo yourself. |
| [Agent-Only Mining](../codex-only-mining.md) | You run a coding agent directly against the task repo and submit only after local checks pass. |

The legacy `neurons/miner.py` AutoML-Zero path is archived and is not the
current production mining path.
