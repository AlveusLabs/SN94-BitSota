# Autoresearch Testnet E2E

One end-to-end guide for testing the shared autoresearch testnet from either:

1. direct agent CLI execution, or
2. the GUI with a GUI-managed external agent such as Codex or Claude.

This guide also covers the missing piece that older docs glossed over: how submissions become accepted.

## Scope

The shared testnet flow spans three systems:

- coordinator: tasks, claims, submissions, verification state, best results
- SN94 GUI or agent miner: claim, run, and submit
- Pool claim service: epoch publication and Merkle claim packages

Important:

- `autoresearch-bittensor` itself does not do Merkle claims or chain publishing.
- For `standard` and `centerless` tasks, submission creation is not enough. A validator must verify the submission.
- For `peer_evaluation` tasks, miners judge miners through peer consensus instead of `/verify`.

## Live Endpoints

Current shared testnet endpoints:

```json
{
  "pool_endpoint": "https://3fhi3ukpyw.eu-central-1.awsapprunner.com",
  "research_coordinator_endpoint": "https://chvp2wytst.eu-central-1.awsapprunner.com",
  "merkle_claim_endpoint": "https://3fhi3ukpyw.eu-central-1.awsapprunner.com/claims",
  "onchain_ws_url": "wss://test.finney.opentensor.ai:443",
  "onchain_contract": "5GUw1gZVfUTXWLEbA7G6Xdp8QsUHAy2xpVmaj5fRc1gW1Xyy",
  "onchain_metadata_path": "/home/mekaneeky/repos/Pool/new_merkle/app/assets/merklepool.json"
}
```

Operational notes:

- Use task `slug`, not a hardcoded task ID.
- Claim publication is windowed, so Merkle packages appear after the next pool rollover, not immediately.
- Reward success is not measured by free balance on the miner hotkey.

## Validation Modes

There are two validation paths in the current implementation.

### `standard` and `centerless`

These require validator-backed acceptance.

The path is:

1. miner claims task or work item
2. miner submits patch plus `submission.json`
3. submission is stored as `pending_verification`
4. validator replay records `accepted`, `rejected`, or `error` through `POST /api/v1/submissions/{id}/verify`
5. accepted submission updates the task best result

Implemented validation paths:

- background validator worker via `autoresearch-validate`
- manual signed `POST /api/v1/submissions/{id}/verify` from an allowlisted validator hotkey

### `peer_evaluation`

These do not use `/verify`.

The path is:

1. miner submits work
2. other miners evaluate the pending submission
3. consensus is tracked through `POST /api/v1/submissions/{id}/peer-evaluate`
4. once the threshold is met, peer consensus marks the accepted result

Relevant CLI support already exists:

- `python -m neurons.research_agent_miner peer-evaluate-once`
- `python -m neurons.research_agent_miner loop --allow-peer-evaluation`

## Prereqs

- a funded test wallet or generated test mnemonic
- coordinator, claim service, and websocket reachable
- Python environment for `python -m neurons.research_agent_miner`
- GUI environment if testing the GUI path
- `codex` or `claude` installed if using a GUI-managed external agent
- Merkle metadata file available at the configured path

## Direct Agent E2E

List live tasks first:

```bash
cd /home/mekaneeky/repos/SN94-BitSota
PYTHONPATH=/home/mekaneeky/repos/SN94-BitSota \
python -m neurons.research_agent_miner list-tasks \
  --coordinator-url https://chvp2wytst.eu-central-1.awsapprunner.com
```

Run one direct claim -> run -> submit pass with an external agent:

```bash
cd /home/mekaneeky/repos/SN94-BitSota
PYTHONPATH=/home/mekaneeky/repos/SN94-BitSota \
python -m neurons.research_agent_miner mine-once \
  --coordinator-url https://chvp2wytst.eu-central-1.awsapprunner.com \
  --participation-style direct \
  --task-slug sn97-distil-mini-kl \
  --agent-command "codex exec --full-auto -C {repo_dir_quoted} --add-dir {workspace_dir_quoted} -o {submission_result_path_quoted} - < {intro_path_quoted}" \
  --agent-mode gui_managed \
  --hotkey-mnemonic "<test mnemonic>"
```

Alternative model-driven mode without an external agent command:

```bash
cd /home/mekaneeky/repos/SN94-BitSota
PYTHONPATH=/home/mekaneeky/repos/SN94-BitSota \
python -m neurons.research_agent_miner mine-once \
  --coordinator-url https://chvp2wytst.eu-central-1.awsapprunner.com \
  --participation-style direct \
  --task-slug sn97-distil-mini-kl \
  --llm-base-url http://127.0.0.1:11434/v1 \
  --llm-model prism-ml/Bonsai-8B-gguf \
  --hotkey-mnemonic "<test mnemonic>"
```

Expected workspace artifacts:

- `INTRO_GUI.md`
- `submission.json`
- `agent.stdout.txt`
- `agent.stderr.txt`
- repo checkout with a valid patch over the task base ref

## GUI E2E

Set GUI config for shared testnet plus the external agent command:

```json
{
  "pool_endpoint": "https://3fhi3ukpyw.eu-central-1.awsapprunner.com",
  "merkle_claim_endpoint": "https://3fhi3ukpyw.eu-central-1.awsapprunner.com/claims",
  "onchain_ws_url": "wss://test.finney.opentensor.ai:443",
  "onchain_contract": "5GUw1gZVfUTXWLEbA7G6Xdp8QsUHAy2xpVmaj5fRc1gW1Xyy",
  "onchain_metadata_path": "/home/mekaneeky/repos/Pool/new_merkle/app/assets/merklepool.json",
  "research_coordinator_endpoint": "https://chvp2wytst.eu-central-1.awsapprunner.com",
  "research_agent_mode": "gui_managed",
  "research_agent_command": "codex exec --full-auto -C {repo_dir_quoted} --add-dir {workspace_dir_quoted} -o {submission_result_path_quoted} - < {intro_path_quoted}"
}
```

For Claude, swap only the command:

```json
{
  "research_agent_mode": "gui_managed",
  "research_agent_command": "claude code --dangerously-skip-permissions -C {repo_dir_quoted} < {intro_path_quoted} > {submission_result_path_quoted}"
}
```

You can also launch a prepared live testnet GUI setup with:

```bash
cd /home/mekaneeky/repos/SN94-BitSota
scripts/run_live_testnet_research_guis.sh <wallet_name>:<hotkey_name>
```

What to verify in the GUI path:

1. the GUI loads real coordinator tasks, not fallback template cards
2. pool mining starts against a live task
3. the external agent writes `submission.json`, `agent.stdout.txt`, and `agent.stderr.txt`
4. the coordinator records a submission
5. validation accepts the submission
6. the claim service later exposes a Merkle package
7. the claim client submits successfully

## Validator Step For LLM-Based Autoresearch

This is the part older docs under-described.

For `standard` and `centerless` tasks, you still need validator replay after the agent produces a patch and `submission.json`.

### Option A: background validator worker

Run the coordinator-side validator worker from `autoresearch-bittensor`:

```bash
cd /home/mekaneeky/repos/autoresearch-bittensor
python -m venv .venv
source .venv/bin/activate
pip install -e .[test]
autoresearch-validate \
  --validator-hotkey <allowlisted_validator_hotkey> \
  --workspace-root ./data/validator-workspaces
```

For a single pass:

```bash
autoresearch-validate \
  --validator-hotkey <allowlisted_validator_hotkey> \
  --workspace-root ./data/validator-workspaces \
  --once
```

This polls pending submissions, replays them, and records `accepted`, `rejected`, or `error`.

### Option B: manual signed verify

If a validator worker is not running, an allowlisted validator hotkey can manually call:

- `POST /api/v1/submissions/{id}/verify`

Use this only with a current live submission ID and real observed metrics from replay.

### Peer-evaluation exception

Do not use `/verify` for `peer_evaluation` tasks.

Use:

- `python -m neurons.research_agent_miner peer-evaluate-once ...`
- or `python -m neurons.research_agent_miner loop --allow-peer-evaluation ...`

The coordinator will reject `/verify` for those tasks.

## Reward And Claim Step

After a submission is accepted:

1. the coordinator best result should update
2. the reward snapshot should include the accepted competition state
3. Pool should publish the next Merkle epoch on rollover
4. the claim API should serve a package for the miner hotkey
5. the GUI or local claim client should submit `claim_single`

Useful checks:

```bash
curl -fsS https://chvp2wytst.eu-central-1.awsapprunner.com/healthz
curl -fsS https://chvp2wytst.eu-central-1.awsapprunner.com/api/v1/tasks | jq
curl -fsS https://chvp2wytst.eu-central-1.awsapprunner.com/api/v1/submissions | jq
curl -fsS https://chvp2wytst.eu-central-1.awsapprunner.com/api/v1/verifications | jq
curl -fsS https://3fhi3ukpyw.eu-central-1.awsapprunner.com/claims/health
curl -fsS https://3fhi3ukpyw.eu-central-1.awsapprunner.com/claims/epochs | jq
curl -fsS https://3fhi3ukpyw.eu-central-1.awsapprunner.com/claims/epoch/<epoch>/claim/<hotkey> | jq
```

## Success Criteria

The flow is only truly end-to-end when all of these happen:

1. a live task is claimed
2. a valid submission is created
3. the submission becomes accepted through validator replay or peer consensus
4. the task best result updates
5. the next claim window publishes an epoch
6. a claim package appears for the miner hotkey
7. the claim is submitted successfully
8. no remaining claim packages exist for that hotkey

## Common Gotchas

- Do not treat submission creation as end-to-end success.
- Do not hardcode task IDs across reseeds.
- Do not use fallback template cards as proof that the live coordinator path works.
- Do not check miner hotkey free balance as the reward success signal.
- If `/api/v1/submissions/{id}/verify` returns `503`, validator allowlisting or validator deployment is wrong.
- If accepted submissions never show up in `/claims/epochs`, the reward publication side is broken even if coordinator validation is healthy.
