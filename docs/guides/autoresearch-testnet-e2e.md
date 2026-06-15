# Autoresearch Testnet E2E

One end-to-end guide for testing the shared autoresearch testnet from either:

1. a direct independent agent path, or
2. the GUI with a GUI-managed external agent.

This guide also covers the missing piece that older docs glossed over: how submissions become accepted.
It also documents the recipient coldkey split so miners can tell local wallet state apart from published claim state.

## Scope

The shared testnet flow spans three systems:

- coordinator: tasks, claims, submissions, verification state, best results
- SN94 GUI or agent miner: claim, run, submit, and declare recipient coldkey
- Pool claim service: recipient mapping, epoch publication, and Merkle claim packages

Important:

- `autoresearch-bittensor` itself does not do Merkle claims or chain publishing.
- For `standard` and `centerless` tasks, submission creation is not enough. A validator must verify the submission.
- For `peer_evaluation` tasks, miners judge miners through peer consensus instead of `/verify`.
- The miner hotkey is how claim packages are discovered. The published `recipient_coldkey` is where the Pool claim path pays.
- A locally declared recipient coldkey is not the same thing as a published claim recipient unless Pool has stored and republished that mapping into the epoch.

## Live Endpoints

Current shared testnet endpoints:

```json
{
  "pool_endpoint": "https://3fhi3ukpyw.eu-central-1.awsapprunner.com",
  "research_coordinator_endpoint": "https://chvp2wytst.eu-central-1.awsapprunner.com",
  "merkle_claim_endpoint": "https://3fhi3ukpyw.eu-central-1.awsapprunner.com/claims",
  "onchain_ws_url": "wss://test.finney.opentensor.ai:443",
  "onchain_contract": "5G1fuA6RPVCUu7K5ep7SWJLzQaqzdJAwchQHppkfVKzEVv49",
  "onchain_metadata_path": "/home/mekaneeky/repos/Pool/new_merkle/app/assets/merklepool.json"
}
```

Operational notes:

- Use task `slug`, not a hardcoded task ID.
- Claim publication is windowed, so Merkle packages appear after the next pool rollover, not immediately.
- Reward success is not measured by free balance on the miner hotkey.
- For testnet claims, think in this order: GUI declaration -> Pool storage -> consensus publication -> claim package -> `claim_single`.
- Only the latest published epoch is claimable; the published `amount_units` are cumulative for that hotkey/recipient pair.

## Current Coordinator Catalog

On the current `autoresearch-bittensor:testing` branch, the built-in seeded task catalog is:

- `qwen3-06b-binary-frontier` -> `standard`
- `qwen3-06b-ternary-frontier` -> `centerless`
- `qwen3-06b-binary-kernel` -> `standard`
- `qwen3-06b-ternary-kernel` -> `centerless`

All 4 use `heldout_quality` as the primary metric, Pareto ranking, and a secondary metric of either `compression_ratio` or `speedup`.

## Validation Modes

There are two validation paths in the current implementation.

### `standard` and `centerless`

These require validator-backed acceptance.

The path is:

1. miner claims task or work item
2. miner submits patch plus `submission.json`
3. submission is stored as `pending_verification`
4. validator replay records `accepted`, `rejected`, or `error` through a signed validator job result
5. accepted submission updates the task best result

Implemented validation paths:

- public SN94 validator runner: signed `POST /api/v1/validator/submissions/scan`, local replay for every returned `replay_spec`, then signed `POST /api/v1/validator/jobs/{job_id}/result`
- legacy public signed `POST /api/v1/submissions/{id}/verify` from an allowlisted validator hotkey, using SN94-BitSota signing helpers
- backend-owned background validator worker via `autoresearch-validate`

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
python3 -m neurons.research_agent_miner ...
```

## Agent-Only Testnet Path

If you want a coding agent to work directly against a task repo, use the
agent-only public guide instead of the retired wrapper path.

Use the direct testnet prompt in
[autoresearch-testnet-direct-prompt.md](/home/mekaneeky/repos/SN94-BitSota/docs/guides/autoresearch-testnet-direct-prompt.md)
or the production prompt in
[autoresearch-agent-master-prompt.md](/home/mekaneeky/repos/SN94-BitSota/docs/guides/autoresearch-agent-master-prompt.md).
Point the agent at the SN94 checkout only:

- `/home/mekaneeky/repos/SN94-BitSota`

The public helpers available for manual signed calls are:

- `/home/mekaneeky/repos/SN94-BitSota/scripts/research_signed_request.py`
- `/home/mekaneeky/repos/SN94-BitSota/scripts/claim_merkle_rewards.py`
- `bitsota-research-agent submit-workspace`

Direct agent launch shape, using Codex as one example:

```bash
cat >/tmp/direct-autoresearch-prompt.txt <<'EOF'
<paste the contents of docs/guides/autoresearch-agent-master-prompt.md here>
EOF

codex exec --dangerously-bypass-approvals-and-sandbox \
  -C /home/mekaneeky/repos \
  --add-dir /home/mekaneeky/repos/SN94-BitSota \
  - < /tmp/direct-autoresearch-prompt.txt
```

Historical note:

- A direct Distil run reached `pending_verification` on `2026-04-08`, but that old Distil slug is no longer the default seeded catalog on `testing`.
- For the current codebase, validate against one of the live Qwen task slugs
  returned by the coordinator.

What this path does not replace:

- validator replay for `standard` and `centerless`
- peer consensus for `peer_evaluation`
- Merkle publication and claim timing

## Retired Wrapper Path

The local wrapper launcher is no longer part of public miner onboarding. Public
docs should direct miners to [Manual Mining](../mining.md) or
[Agent Mining](../codex-only-mining.md).

## GUI E2E

Set GUI config for shared testnet plus the selected manual or agent workflow.

If you are using the current guided setup flow, configure the same values through `Research Setup` and wallet setup instead of hand-editing JSON.

The recipient coldkey expectations are:

- wallet setup stores the miner's declared recipient coldkey locally
- Pool, not the coordinator, owns the published claim recipient mapping
- the claim table `Recipient` value comes from the published claim package for that epoch
- if the local declaration and the claim row differ, do not assume the GUI is wrong; check Pool publication
- the GUI should make the local split visible by showing both the connected hotkey and the declared recipient coldkey near the claims view

Example JSON for a manual source/dev run:

```json
{
  "pool_endpoint": "https://3fhi3ukpyw.eu-central-1.awsapprunner.com",
  "merkle_claim_endpoint": "https://3fhi3ukpyw.eu-central-1.awsapprunner.com/claims",
  "onchain_ws_url": "wss://test.finney.opentensor.ai:443",
  "onchain_contract": "5G1fuA6RPVCUu7K5ep7SWJLzQaqzdJAwchQHppkfVKzEVv49",
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
7. the claim package includes the expected `recipient_coldkey`
8. the claim client submits successfully against that published recipient

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

### Planner-created work items

Planner-created work items are an operator/testnet surface, not a current public
miner onboarding path. Public miners should start from the live competition
catalog and use manual mining or agent-only mining.

What to verify in planner mode:

- `GET /api/v1/work-items` returns open work items for the task
- miner claims a work item instead of a direct task claim
- submission is linked to the claimed work item
- work item moves to completed on successful submission
- follow-up work items appear if the planner creates them

## Validator Step For LLM-Based Autoresearch

This is the part older docs under-described.

For `standard` and `centerless` tasks, you still need validator replay after the agent produces a patch and `submission.json`.

### Public validator path

Public validators should use the SN94-BitSota checkout and talk to the live
coordinator over signed HTTP requests. They do not need `autoresearch-bittensor`
DB access, AWS credentials, `DATABASE_URL`, or `ADMIN_TOKEN`.

The validator hotkey must still be allowlisted by the live backend. The backend
checks `X-Hotkey`, `X-Timestamp`, and `X-Signature`; use the SN94 helper to build
those headers instead of hand-rolling them.

```bash
git clone https://github.com/AlveusLabs/SN94-BitSota.git
cd SN94-BitSota
git checkout testnet-net-gui-pool-agents
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Inspect pending submissions through the public API:

```bash
bitsota-research-agent signed-request \
  --coordinator-url https://chvp2wytst.eu-central-1.awsapprunner.com \
  --method GET \
  --path /api/v1/submissions \
  --params-json '{"status":"pending_verification"}' \
  --wallet-name <validator_wallet> \
  --wallet-hotkey <validator_hotkey>
```

Fetch the submission detail and task onboarding for the candidate you will
replay:

```bash
bitsota-research-agent signed-request \
  --coordinator-url https://chvp2wytst.eu-central-1.awsapprunner.com \
  --method GET \
  --path /api/v1/submissions/<submission_id>/detail \
  --wallet-name <validator_wallet> \
  --wallet-hotkey <validator_hotkey>

bitsota-research-agent signed-request \
  --coordinator-url https://chvp2wytst.eu-central-1.awsapprunner.com \
  --method GET \
  --path /api/v1/tasks/<task_id>/onboard.md \
  --wallet-name <validator_wallet> \
  --wallet-hotkey <validator_hotkey>
```

Preferred unattended public validator path:

```bash
cp research_validator_config.yaml.example research_validator_config.yaml
# Edit coordinator_url, wallet_name, and wallet_hotkey.
python -m validator.research_validator_runner --config research_validator_config.yaml --once
```

For manual fallback, replay the submission in a clean local workspace using the
task repository, base ref, allowed patch paths, and benchmark instructions from
the submission detail and task onboarding output. Then record the observed
result through the legacy `/verify` route:

```bash
cat >/tmp/verify-submission.json <<'JSON'
{
  "status": "accepted",
  "observed_metrics": {
    "heldout_quality": 0.0
  },
  "notes": "validator replay completed; replace this with task, commit, and benchmark summary",
  "replay_log": "replace this with bounded replay output or failure summary"
}
JSON

bitsota-research-agent signed-request \
  --coordinator-url https://chvp2wytst.eu-central-1.awsapprunner.com \
  --method POST \
  --path /api/v1/submissions/<submission_id>/verify \
  --body-file /tmp/verify-submission.json \
  --wallet-name <validator_wallet> \
  --wallet-hotkey <validator_hotkey>
```

Use `rejected` or `error` instead of `accepted` when replay fails. Do not send
`accepted` from claimed miner metrics; the canonical score is the validator's
observed replay metric.

Public runner notes:

- The SN94 public runner is the shared-testnet path for validators that should not hold backend database or AWS credentials.
- The legacy signed `/verify` path remains useful for manual replay and for testing an older backend.

### Backend-owned validator worker

Only backend operators with the `autoresearch-bittensor` checkout and database
credentials should run the coordinator-local worker:

```bash
cd /home/mekaneeky/repos/autoresearch-bittensor
autoresearch-validate \
  --wallet-name <validator_wallet> \
  --wallet-hotkey <validator_hotkey> \
  --workspace-root ./data/validator-workspaces
```

This worker is not the public validator runbook for shared testnet operators
because it reads and writes the coordinator database directly.

### Peer-evaluation exception

Do not use `/verify` for `peer_evaluation` tasks.

Use:

- `bitsota-research-agent peer-evaluate-once ...`
- or `bitsota-research-agent loop --allow-peer-evaluation ...`

The coordinator will reject `/verify` for those tasks.

Peer-evaluation task onboarding should be documented per live competition before
it is exposed as a public miner path.

## Reward And Claim Step

After a submission is accepted:

1. the coordinator best result should update
2. the reward snapshot should include the accepted competition state
3. Pool should resolve the miner hotkey to a stored recipient coldkey
4. Pool should publish the next Merkle epoch on rollover with that `recipient_coldkey` in the `claim_list`
5. the claim API should serve a package for the miner hotkey that includes the same `recipient_coldkey`
6. the GUI or local claim client should submit `claim_single` against the published recipient

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

When inspecting a claim package, check these fields explicitly:

- `hotkey`: miner identity used for lookup and reward attribution
- `recipient_coldkey`: payout recipient published into the Merkle leaf
- `proof` / `amount_units` / `index`: proof material for `claim_single`

## Success Criteria

The flow is only truly end-to-end when all of these happen:

1. a live task is claimed
2. a valid submission is created
3. the submission becomes accepted through validator replay or peer consensus
4. the task best result updates
5. the next claim window publishes an epoch
6. a claim package appears for the miner hotkey
7. the claim package recipient matches the intended declared payout path
8. the claim is submitted successfully
9. no remaining claim packages exist for that hotkey

## Common Gotchas

- Do not treat submission creation as end-to-end success.
- Do not hardcode task IDs across reseeds.
- Do not use fallback template cards as proof that the live coordinator path works.
- Do not check miner hotkey free balance as the reward success signal.
- Do not assume the GUI's locally declared coldkey automatically controls claim payout. Pool publication is the source of truth for each epoch.
- Do not assume `POST /coldkey_address/update` on the legacy relay path controls autoresearch Merkle claims.
- If `/api/v1/validator/submissions/scan` or `/api/v1/submissions/{id}/verify` returns `503`, validator allowlisting or validator deployment is wrong.
- If accepted submissions never show up in `/claims/epochs`, the reward publication side is broken even if coordinator validation is healthy.
