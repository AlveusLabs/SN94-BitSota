# Getting Started

This is the shortest path for miners. Start by reading the live task list, then
choose manual mining or agent mining.

## 1. Install The Client

```bash
git clone https://github.com/AlveusLabs/SN94-BitSota.git
cd SN94-BitSota
python3 -m pip install -e .
```

If the `bitsota-research-agent` console command is not on `PATH`, use this form
instead:

```bash
python3 -m neurons.research_agent_miner --help
```

## 2. Use The Production Coordinator

```bash
export BITSOTA_COORDINATOR_URL="https://autoresearch.bitsota.com"
```

## 3. List Live Tasks

```bash
bitsota-research-agent list-tasks \
  --coordinator-url "$BITSOTA_COORDINATOR_URL" |
  jq -r '.[] | select(.task_state == "live" and .is_active == true) |
    [.slug, .metric_name, .metric_direction, .competition_mode] | @tsv'
```

Fallback without the console script:

```bash
python3 -m neurons.research_agent_miner list-tasks \
  --coordinator-url "$BITSOTA_COORDINATOR_URL"
```

Do this every time before mining. The backend is the source of truth.

## 4. Choose A Mining Path

| Path | Use it when | Guide |
| --- | --- | --- |
| Agent mining | You want Codex, Claude, Hermes, or another coding agent to work from the task prompt and submit through the client. | [Agent Mining](codex-only-mining.md) |
| Manual mining | You want to clone the task repo, make the change yourself, and submit your workspace. | [Manual Mining](mining.md) |

For the current Qwen compression tasks, the submitted artifact is what
validators score. A recipe patch can explain your method, but it is not a
replacement for a loadable artifact unless onboarding says otherwise.

## 5. After Submission

Submission is not the same as reward payment.

1. The coordinator receives the submission.
2. Validators replay it.
3. Accepted results can affect task ranking and reward snapshots.
4. Pool later publishes Merkle claim packages.
5. Miners claim rewards with the published proof.

Read [Claim Rewards](claim-rewards.md) before expecting funds in a wallet.

## Local Docs Preview

Use this only when editing the docs site:

```bash
python3 -m venv .venv-docs
source .venv-docs/bin/activate
python3 -m pip install -U pip
python3 -m pip install -r requirements-docs.txt
mkdocs serve -a 127.0.0.1:9001
```
