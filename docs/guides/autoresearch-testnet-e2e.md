# Autoresearch Testnet E2E

Small end-to-end guide for humans or agent operators testing the current shared testnet flow.

As of `2026-04-06`, the deployed branch heads are:

- `SN94-BitSota:testnet-net-gui-pool-agents` at `8b44760`
- `AlveusLabs/autoresearch-bittensor:testing` at `157e44c`
- `AlveusLabs/Pool:testing` at `d274d63`

## Live Endpoints

- Research coordinator: `https://chvp2wytst.eu-central-1.awsapprunner.com`
- Pool API: `https://3fhi3ukpyw.eu-central-1.awsapprunner.com`
- Merkle claim API: `https://3fhi3ukpyw.eu-central-1.awsapprunner.com/claims`
- On-chain websocket: `wss://test.finney.opentensor.ai:443`
- On-chain contract: `5GUw1gZVfUTXWLEbA7G6Xdp8QsUHAy2xpVmaj5fRc1gW1Xyy`

Current live checks when this guide was written:

- coordinator `/healthz`: `200`
- coordinator `/api/v1/tasks`: `200`
- pool `/health`: `200`
- claims `/health`: `200`
- claims `/epochs`: non-empty
- pool window size: about `5` minutes

Important:

- Task IDs rotate. Use task `slug`, not a hardcoded task ID.
- Claim success does not show up as free balance on the miner hotkey. It lands as subnet alpha stake on the recipient coldkey.

## GUI Path

Use the GUI build from `testnet-net-gui-pool-agents`.

Set these GUI config values:

```json
{
  "research_coordinator_endpoint": "https://chvp2wytst.eu-central-1.awsapprunner.com",
  "pool_endpoint": "https://3fhi3ukpyw.eu-central-1.awsapprunner.com",
  "merkle_claim_endpoint": "https://3fhi3ukpyw.eu-central-1.awsapprunner.com/claims",
  "onchain_ws_url": "wss://test.finney.opentensor.ai:443",
  "onchain_contract": "5GUw1gZVfUTXWLEbA7G6Xdp8QsUHAy2xpVmaj5fRc1gW1Xyy",
  "onchain_metadata_path": "/absolute/path/to/Pool-testing/new_merkle/app/assets/merklepool.json"
}
```

Then:

1. Open the research/pool mining screen.
2. Let the GUI load live tasks from the coordinator.
3. Start mining on a live card, not a fallback template card.
4. Wait for the miner to claim, run, and submit.
5. If reward epochs are published, use the profile claim flow to fetch and submit claims.

What good looks like:

- the GUI sees live coordinator tasks
- mining starts against a live task ID
- a submission is created on the coordinator
- validator acceptance happens
- a Merkle epoch appears
- the claim client shows no remaining claimable packages after success

## Direct Agent Path

List live tasks first:

```bash
bitsota-research-agent list-tasks \
  --coordinator-url https://chvp2wytst.eu-central-1.awsapprunner.com
```

Run one direct pass with an external agent CLI:

```bash
bitsota-research-agent mine-once \
  --coordinator-url https://chvp2wytst.eu-central-1.awsapprunner.com \
  --agent-command 'codex exec {intro_path_quoted}' \
  --agent-mode gui_managed \
  --task-slug sn97-distil-mini-kl \
  --participation-style direct \
  --hotkey-mnemonic 'replace with test mnemonic'
```

Notes:

- `mine-once` is the easiest E2E probe because it does one claim -> run -> submit cycle.
- Use `sn97-distil-mini-kl` first unless you are testing another specific competition.
- The miner now submits a pinned commit SHA back to the coordinator, not a floating branch name.

Expected artifacts in the workspace:

- `INTRO_GUI.md`
- `submission.json`
- `agent.stdout.txt`
- a valid git patch over the task repo checkout

## Validator Step

Submission creation is not the end of the flow.

An accepted result requires validator verification. On the shared testnet, either:

- a background validator worker must be running against the same coordinator state, or
- an allowlisted validator hotkey must send signed `/api/v1/submissions/{id}/verify`

If you are manually verifying, verify a real current submission ID from the live coordinator, not an old ID from a prior reseed.

## Reward And Claim Step

After a submission is accepted:

1. the coordinator reward snapshot should include the competition
2. Pool publishes the next epoch on window rollover
3. the claim API serves the epoch package
4. the miner or GUI claim client submits `claim_single`

Because testnet windows are about `5` minutes, claim publication is not immediate. Wait for the next rollover.

## Quick Checklist

- Coordinator health is `200`
- Pool health is `200`
- Claims health is `200`
- Live task discovery works
- Submission is created
- Submission is accepted after validator replay
- `/api/v1/reward-snapshot` is non-empty
- `/claims/epochs` includes a new epoch
- claim submission succeeds

## Common Gotchas

- Do not hardcode task IDs.
- Do not use fallback template cards as proof that the coordinator task is live.
- Do not use miner hotkey free balance as the reward success signal.
- If `/verify` returns `503 validator allowlist is not configured`, the coordinator deployment is stale.
- If a claim package disappears locally, confirm the claim extrinsic receipt, not just local state.
