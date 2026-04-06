# Testnet Research Testing

This guide covers two live testnet paths:

1. Direct autoresearch execution with the agent miner.
2. GUI-driven research execution with the GUI-managed external agent flow.

## Endpoints

Use the current testnet endpoints:

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

## Prereqs

- A funded test wallet or a generated test wallet.
- Working access to the coordinator and pool endpoints.
- A Python environment that can run `python -m gui` and the miner CLI.
- `codex` installed if using GUI-managed external agents.
- Merkle metadata file present at the configured path.

## Direct Agent Path

This path bypasses the GUI and tests the coordinator directly.

Example:

```bash
PYTHONPATH=/home/mekaneeky/repos/SN94-BitSota \
python -m neurons.research_agent_miner mine-once \
  --wallet-name <wallet_name> \
  --wallet-hotkey <hotkey_name> \
  --coordinator-url https://chvp2wytst.eu-central-1.awsapprunner.com \
  --participation-style direct \
  --task-id <task_id_or_slug>
```

What to verify:

- A direct task claim is created.
- The workspace contains `submission.json`.
- `POST /api/v1/submissions` succeeds.
- Validator verification moves the submission to accepted.
- The task best-result endpoint updates.
- After the claim window, the Merkle claim service exposes a package for the miner hotkey.

Useful API checks:

```bash
curl -fsS https://chvp2wytst.eu-central-1.awsapprunner.com/api/v1/dashboard | jq
curl -fsS https://chvp2wytst.eu-central-1.awsapprunner.com/api/v1/submissions | jq
curl -fsS https://3fhi3ukpyw.eu-central-1.awsapprunner.com/claims/epochs | jq
curl -fsS https://3fhi3ukpyw.eu-central-1.awsapprunner.com/claims/epoch/<epoch>/claim/<hotkey> | jq
```

## GUI Path

The GUI path uses the coordinator-backed pool/research flow and launches an external Codex worker from the GUI.

You can either run the GUI manually with a config file or use the helper script:

```bash
cd /home/mekaneeky/repos/SN94-BitSota
scripts/run_live_testnet_research_guis.sh <wallet_name>:<hotkey_name>
```

The helper script writes a temporary GUI config with:

- pool endpoint
- coordinator endpoint
- merkle claim endpoint
- on-chain websocket and contract
- GUI-managed external-agent command
- research autostart enabled for a chosen task slug

What to verify in the GUI path:

- The GUI boots with the selected wallet.
- Pool mining mode selects the research task.
- The external agent starts and writes `agent.stdout.txt`, `agent.stderr.txt`, and `submission.json`.
- The coordinator accepts the submission.
- Validators verify it and the task best result updates.
- The Merkle claim package appears after the publication window.
- Claiming succeeds and the hotkey no longer has outstanding claim packages.

Common runtime artifacts:

- `.bitsota_agent_workspace/.../submission.json`
- `.bitsota_agent_workspace/.../agent.stdout.txt`
- `.bitsota_agent_workspace/.../agent.stderr.txt`
- `.bitsota_agent_logs/*.log`

## Claiming

The GUI claim client queries packages by miner hotkey and submits to the Merkle contract.

Important:

- Claim consumption is tracked locally in `~/.bitsota/merkle_claim_state.json`.
- The contract path uses stake transfer semantics, not normal free-balance payout to the hotkey account.
- If the signing account lacks fees or storage deposit, use a funded signer for the contract call.

## Failure Modes

- `POST /api/v1/submissions` returns `500`: coordinator backend bug.
- `/api/v1/submissions/<id>/verify` returns `503`: validator allowlist or verifier config bug.
- `/claims/epochs` stays empty after acceptance and the publication window: reward publication bug.
- Claim submission fails with insufficient funds: signer balance problem, not necessarily claim-package corruption.
- GUI external agent stalls: missing network, missing cached models, or missing dataset dependencies in the Codex runtime.

## Minimal End-to-End Checklist

1. Claim a task.
2. Produce `submission.json`.
3. Create a submission successfully.
4. Verify and accept the submission.
5. Wait for a Merkle epoch to publish.
6. Fetch the claim package for the miner hotkey.
7. Submit the claim successfully.
8. Confirm no remaining claim packages for that hotkey.
