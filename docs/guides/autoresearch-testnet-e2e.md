# Autoresearch Testnet E2E

One end-to-end guide for testing the shared autoresearch testnet from either:

1. a direct independent agent path, or
2. the GUI with a GUI-managed external agent.

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

- `bitsota-research-agent peer-evaluate-once`
- `bitsota-research-agent loop --allow-peer-evaluation`

## Prereqs

- a funded test wallet or generated test mnemonic
- coordinator, claim service, and websocket reachable
- Python environment with `bitsota-research-agent` installed from this repo
- GUI environment if testing the GUI path
- `codex`, `claude`, or `hermes` installed if using an external agent
- Merkle metadata file available at the configured path

If the console entrypoint is not on `PATH`, the fallback is:

```bash
python -m neurons.research_agent_miner ...
```

## Direct Independent Agent Path

If you want to skip `bitsota-research-agent` entirely, you can drive the coordinator manually with an agent.

Use the master prompt in [autoresearch-agent-master-prompt.md](/home/mekaneeky/repos/SN94-BitSota/docs/guides/autoresearch-agent-master-prompt.md) and point the agent at the SN94 checkout only:

- `/home/mekaneeky/repos/SN94-BitSota`

The signing helper the agent should use is:

- `/home/mekaneeky/repos/SN94-BitSota/miner/research_auth.py`
- `/home/mekaneeky/repos/SN94-BitSota/miner/research_coordinator_client.py`

Tested direct Codex launch shape:

```bash
cat >/tmp/direct-autoresearch-prompt.txt <<'EOF'
<paste the contents of docs/guides/autoresearch-agent-master-prompt.md here>
EOF

codex exec --dangerously-bypass-approvals-and-sandbox \
  -C /home/mekaneeky/repos \
  --add-dir /home/mekaneeky/repos/SN94-BitSota \
  - < /tmp/direct-autoresearch-prompt.txt
```

Untested direct launch shapes using the same prompt:

Claude Code:

```bash
claude code --dangerously-skip-permissions \
  -C /home/mekaneeky/repos \
  < /tmp/direct-autoresearch-prompt.txt
```

Hermes:

```bash
hermes -C /home/mekaneeky/repos \
  < /tmp/direct-autoresearch-prompt.txt
```

The tested live Codex run on `2026-04-08` did this successfully:

- selected task `1bad46f1-f601-49b2-a312-830a44fbb01e` with slug `sn97-distil-mini-kl`
- created claim `4bbb4c6f-c6a1-4f24-95b9-35d3b33afeac`
- cloned the task repo at base ref `157e44ce7868d3240094dc6b8a482e819abd894d`
- inserted a no-op top-of-file marker in `competition_packs/distil_kl_mini/train.py`
- replayed `python3 competition_packs/distil_kl_mini/benchmark.py`
- submitted `submission_id=07aa6531-b588-40ab-b2d8-7202733fdb3b`
- received `status=pending_verification` from the live coordinator

What this path does not replace:

- validator replay for `standard` and `centerless`
- peer consensus for `peer_evaluation`
- Merkle publication and claim timing

## Agent-Managed E2E

List live tasks first:

```bash
cd /home/mekaneeky/repos/SN94-BitSota
bitsota-research-agent list-tasks \
  --coordinator-url https://chvp2wytst.eu-central-1.awsapprunner.com
```

The placeholders available to `--agent-command` are:

- `{workspace_dir_quoted}`
- `{repo_dir_quoted}`
- `{intro_path_quoted}`
- `{submission_path_quoted}`
- `{submission_result_path_quoted}`

Use the master agent prompt in [autoresearch-agent-master-prompt.md](/home/mekaneeky/repos/SN94-BitSota/docs/guides/autoresearch-agent-master-prompt.md). That prompt is written to be agent-agnostic and tells the model to handle live task discovery, signed claims, benchmark replay, `submission.json`, and signed coordinator submission.

Run one direct claim -> run -> submit pass with an external agent:

Codex:

```bash
cd /home/mekaneeky/repos/SN94-BitSota
bitsota-research-agent mine-once \
  --coordinator-url https://chvp2wytst.eu-central-1.awsapprunner.com \
  --participation-style direct \
  --task-slug sn97-distil-mini-kl \
  --agent-command "bash -lc 'cat {intro_path_quoted} /home/mekaneeky/repos/SN94-BitSota/docs/guides/autoresearch-agent-master-prompt.md | codex exec --skip-git-repo-check --full-auto -C {repo_dir_quoted} --add-dir {workspace_dir_quoted} -o {submission_result_path_quoted} -'" \
  --agent-mode gui_managed \
  --hotkey-mnemonic '<test mnemonic>'
```

Claude Code:

```bash
cd /home/mekaneeky/repos/SN94-BitSota
bitsota-research-agent mine-once \
  --coordinator-url https://chvp2wytst.eu-central-1.awsapprunner.com \
  --participation-style direct \
  --task-slug sn97-distil-mini-kl \
  --agent-command "bash -lc 'cat {intro_path_quoted} /home/mekaneeky/repos/SN94-BitSota/docs/guides/autoresearch-agent-master-prompt.md | claude code --dangerously-skip-permissions -C {repo_dir_quoted} > {submission_result_path_quoted}'" \
  --agent-mode gui_managed \
  --hotkey-mnemonic '<test mnemonic>'
```

Hermes:

```bash
cd /home/mekaneeky/repos/SN94-BitSota
bitsota-research-agent mine-once \
  --coordinator-url https://chvp2wytst.eu-central-1.awsapprunner.com \
  --participation-style direct \
  --task-slug sn97-distil-mini-kl \
  --agent-command "bash -lc 'cat {intro_path_quoted} /home/mekaneeky/repos/SN94-BitSota/docs/guides/autoresearch-agent-master-prompt.md | hermes -C {repo_dir_quoted} > {submission_result_path_quoted}'" \
  --agent-mode gui_managed \
  --hotkey-mnemonic '<test mnemonic>'
```

Notes:

- The Codex command above was tested on `2026-04-08`.
- The Claude Code and Hermes command shapes are untested examples that use the same master prompt.
- Pick the launch flags your local Claude or Hermes install expects if they differ from these examples.

Alternative model-driven mode without an external agent command:

```bash
cd /home/mekaneeky/repos/SN94-BitSota
bitsota-research-agent mine-once \
  --coordinator-url https://chvp2wytst.eu-central-1.awsapprunner.com \
  --participation-style direct \
  --task-slug sn97-distil-mini-kl \
  --llm-base-url http://127.0.0.1:11434/v1 \
  --llm-model prism-ml/Bonsai-8B-gguf \
  --hotkey-mnemonic '<test mnemonic>'
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
  "research_agent_command": "bash -lc 'cat {intro_path_quoted} /home/mekaneeky/repos/SN94-BitSota/docs/guides/autoresearch-agent-master-prompt.md | codex exec --skip-git-repo-check --full-auto -C {repo_dir_quoted} --add-dir {workspace_dir_quoted} -o {submission_result_path_quoted} -'"
}
```

Claude Code example:

```json
{
  "research_agent_mode": "gui_managed",
  "research_agent_command": "bash -lc 'cat {intro_path_quoted} /home/mekaneeky/repos/SN94-BitSota/docs/guides/autoresearch-agent-master-prompt.md | claude code --dangerously-skip-permissions -C {repo_dir_quoted} > {submission_result_path_quoted}'"
}
```

Hermes example:

```json
{
  "research_agent_mode": "gui_managed",
  "research_agent_command": "bash -lc 'cat {intro_path_quoted} /home/mekaneeky/repos/SN94-BitSota/docs/guides/autoresearch-agent-master-prompt.md | hermes -C {repo_dir_quoted} > {submission_result_path_quoted}'"
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

## Planner-Driven Pool Path

If you want shared task assignment instead of each miner choosing work directly, run the planner and mine in `pool` mode.

Planner-driven flow:

1. planner creates `work_items`
2. pool miners claim those `work_items`
3. miners run and submit against the claimed work item
4. validator accepts or rejects the submission for `standard` or `centerless`
5. accepted results flow through reward publication and Merkle claim

Recommended pairings:

- `standard` + direct claims + validator
- `centerless` + planner work items + validator
- `peer_evaluation` + direct or planner work items + no validator

### Run the planner

Deterministic planner:

```bash
cd /home/mekaneeky/repos/autoresearch-bittensor
python -m venv .venv
source .venv/bin/activate
pip install -e .[test]
autoresearch-plan --once
```

Looping deterministic planner:

```bash
autoresearch-plan --interval-seconds 30
```

LLM-based agentic planner:

```bash
cd /home/mekaneeky/repos/autoresearch-bittensor
export PLANNER_LLM_BASE_URL=http://127.0.0.1:11434/v1
export PLANNER_LLM_MODEL=planner-model
autoresearch-plan-agentic --once
```

Looping agentic planner:

```bash
autoresearch-plan-agentic --interval-seconds 30
```

You can also trigger one planner pass through the coordinator admin API:

Deterministic:

```bash
curl -X POST https://chvp2wytst.eu-central-1.awsapprunner.com/api/v1/planner/run \
  -H 'X-Admin-Token: <admin-token>'
```

Agentic:

```bash
curl -X POST https://chvp2wytst.eu-central-1.awsapprunner.com/api/v1/planner/run-agentic \
  -H 'X-Admin-Token: <admin-token>'
```

### Mine planner-created work items directly

Use `--participation-style pool` so the miner claims a shared work item instead of a direct task claim.

Codex:

```bash
cd /home/mekaneeky/repos/SN94-BitSota
bitsota-research-agent mine-once \
  --coordinator-url https://chvp2wytst.eu-central-1.awsapprunner.com \
  --participation-style pool \
  --task-slug sn97-distil-mini-kl \
  --agent-command "bash -lc 'cat {intro_path_quoted} /home/mekaneeky/repos/SN94-BitSota/docs/guides/autoresearch-agent-master-prompt.md | codex exec --skip-git-repo-check --full-auto -C {repo_dir_quoted} --add-dir {workspace_dir_quoted} -o {submission_result_path_quoted} -'" \
  --agent-mode gui_managed \
  --hotkey-mnemonic "<test mnemonic>"
```

What to verify in planner mode:

- `GET /api/v1/work-items` returns open work items for the task
- miner claims a work item instead of a direct task claim
- submission is linked to the claimed work item
- work item moves to completed on successful submission
- follow-up work items appear if the planner creates them

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

- `bitsota-research-agent peer-evaluate-once ...`
- or `bitsota-research-agent loop --allow-peer-evaluation ...`

The coordinator will reject `/verify` for those tasks.

If you want peer evaluation with an external agent command directly, use the same `--agent-command` pattern as `mine-once`, but switch the subcommand:

Codex:

```bash
cd /home/mekaneeky/repos/SN94-BitSota
bitsota-research-agent peer-evaluate-once \
  --coordinator-url https://chvp2wytst.eu-central-1.awsapprunner.com \
  --task-slug <peer_evaluation_task_slug> \
  --agent-command "bash -lc 'cat {intro_path_quoted} /home/mekaneeky/repos/SN94-BitSota/docs/guides/autoresearch-agent-master-prompt.md | codex exec --skip-git-repo-check --full-auto -C {repo_dir_quoted} --add-dir {workspace_dir_quoted} -o {submission_result_path_quoted} -'" \
  --agent-mode gui_managed \
  --hotkey-mnemonic "<test mnemonic>"
```

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
