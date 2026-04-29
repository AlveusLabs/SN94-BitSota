You are mining the live autoresearch testnet through the direct agent path.

Use only public repos and the public coordinator/pool endpoints. Do not attempt to access private coordinator, pool, validator, deployment, or backend repos. Do not ask for private repo access. Treat coordinator and pool only as HTTP endpoints.

Live coordinator:
https://chvp2wytst.eu-central-1.awsapprunner.com

Live pool:
https://3fhi3ukpyw.eu-central-1.awsapprunner.com

Target competitions are exactly:

- qwen3-06b-binary-frontier
- qwen3-06b-ternary-frontier
- qwen3-06b-binary-kernel
- qwen3-06b-ternary-kernel

Do not work on legacy competitions.

## Goal

Mine one currently live Qwen3 competition end to end:

1. Discover live Qwen3 tasks from the coordinator.
2. Select one task suitable for direct mining.
3. Clone the task's public repo.
4. Run local bounded smoke validation.
5. Make one focused patch inside the task's allowed edit surface.
6. Re-run the benchmark locally.
7. Submit a direct claim and submission to the live coordinator through the public SN94 signing helper.
8. Poll for submission / verification / reward / pool publication state.
9. Report concrete results and earliest bad state if anything fails.

## Hard Constraints

- Do not download full FineWeb.
- Do not download full Qwen checkpoints.
- Do not modify coordinator or pool services directly.
- Do not attempt to access private repos.
- Do not replace bounded proxy/heldout handling with a huge public dataset.
- Only edit files listed in the live task metadata field `allowed_patch_paths`.
- Do not edit `benchmark.py`, shared validator helpers, manifests, generated result files, or anything outside the allowed patch surface.
- Do not submit generated files like `last_run.json`, `baseline_manifest.json`, `__pycache__`, local logs, or submission sidecars.
- Do not print wallet mnemonics, private keys, or secrets.
- Do not put wallet mnemonics into repo files, patches, logs, summaries, or shell history.
- Use the local wallet material only by passing it to the public SN94 helper scripts.

## Wallet / Signing

Use the local text file named `Wallet mine.txt` in the current workspace folder only as input to the SN94 public helper scripts.

- Read it locally.
- Do not parse or sign requests manually inside your own code.
- Do not print its contents.
- Do not copy it into any repo.
- Do not include it in submitted patch, summary, execution log, or final answer.
- If the wallet file is missing or malformed, stop and report that signing is blocked.

For signed coordinator mutations, use the public helper from the SN94 checkout:

- `python /home/mekaneeky/repos/SN94-BitSota/scripts/research_signed_request.py ...`
- installed equivalent if available: `bitsota-research-agent signed-request ...`

For workspace submission, prefer the higher-level helper:

- `bitsota-research-agent submit-workspace ...`
- fallback: `python -m neurons.research_agent_miner submit-workspace ...`

For Pool claim inspection and `claim_single`, use the public claim helper:

- `python /home/mekaneeky/repos/SN94-BitSota/scripts/claim_merkle_rewards.py ...`
- installed equivalent if available: `bitsota-claim-rewards ...`

Do not construct `X-Hotkey`, `X-Timestamp`, or `X-Signature` yourself.

## Current Qwen Competition Config Expectations

Use these values to validate live task metadata and reason about benchmark constraints. Do not hardcode these into benchmark code unless they already exist in the public pack contract.

Task-level expectations:

- `time_budget_seconds`: `21600` for all 4 Qwen tasks
- max high precision fraction: `<= 0.05`
- binary frontier quality floor: `0.76`
- ternary frontier quality floor: `0.78`
- binary kernel quality floor: `0.76`
- ternary kernel quality floor: `0.78`

Expected primary metric for all tasks:

- `heldout_quality`
- direction: `maximize`

Expected secondary metrics:

- frontier/compression tasks: `compression_ratio`
- kernel tasks: `speedup`

Expected ranking mode:

- `pareto`

Expected competition modes:

- binary tasks: `standard`
- ternary tasks: `centerless`

Expected validator heldout env keys:

- `AUTORESEARCH_HELDOUT_DATASET`
- `AUTORESEARCH_HELDOUT_SPLIT`
- `AUTORESEARCH_HELDOUT_ROTATION_KEY`

Current live validator values are expected to be:

- `AUTORESEARCH_HELDOUT_DATASET=fineweb`
- `AUTORESEARCH_HELDOUT_SPLIT=validator-rotating-heldout`
- `AUTORESEARCH_HELDOUT_ROTATION_KEY=2026-q2-rotation-a`

Treat these as bounded shard handles supplied by the backend. Do not download FineWeb locally.

For local miner iteration, use only bounded local/debug handles, for example:

- `AUTORESEARCH_HELDOUT_DATASET=pack-local-proxy`
- `AUTORESEARCH_HELDOUT_SPLIT=smoke`
- `AUTORESEARCH_HELDOUT_ROTATION_KEY=local-smoke`

Public debug datasets are allowed for local iteration only:

- `wikitext2`
- `minipile`

## Centerless Mode Rules

Do not treat `centerless` tasks as `standard`.

For `centerless` tasks:

- every submission must include `proposed_idea`
- if another miner has already proposed an idea on that task, new submissions must also include `implemented_submission_id` referencing a prior idea-bearing submission
- if there are no prior idea-bearing submissions, include a fresh `proposed_idea` and omit `implemented_submission_id`
- for `standard` tasks, do not invent centerless fields unless the coordinator/task explicitly requires them

Before submitting to a centerless task, query existing submissions for that task and decide whether `implemented_submission_id` is required.

## Reward / Scoring Expectations

Use this only for post-submit validation and economics debugging.

Coordinator-side miner score inputs are expected to behave like:

- accepted submission: `+1.0`
- current best bonus: `+1.0`
- idea proposer reward: `+0.5`
- idea implementer reward: `+0.5`
- score decay: 10-day EMA / exponential decay

Pool-side reward conversion:

- Pool consumes coordinator competition scores
- competition `weight` splits the total publishable budget across competitions
- miner `score` splits a competition slice among miners
- `automl_pool` should be effectively muted by default for the current autoresearch launch

If debugging economics, verify both:

- coordinator reward snapshot contents
- Pool allocation/publication behavior

## Setup

Create or reuse a local workspace directory. If repos already exist, inspect their current state and do not overwrite unrelated user changes.

Clone SN94-BitSota public branch if needed:

```bash
git clone https://github.com/AlveusLabs/SN94-BitSota.git
cd SN94-BitSota
git checkout testnet-net-gui-pool-agents
```
