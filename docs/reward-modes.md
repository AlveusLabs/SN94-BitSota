# Reward Modes (Capacitor vs Capacitorless)

This repo supports multiple **validator reward modes** (configured via `reward_mode` in `validator_config.yaml`). These modes change *how validators reward miners* after validating relay submissions.

## Quick Comparison

| Mode | What pays the miner | When weights change | Relay required | EVM key required |
|------|----------------------|-------------------|----------------|------------------|
| `capacitor` | Capacitor EVM contract vote (`releaseReward`) | Not miner-targeted by default (optional `contract_bots`) | Optional (recommended for submissions/blacklist) | Yes |
| `capacitorless` + `sticky_burnsplit` (default) | **On-chain weights**: 90% burn + 10% to latest accepted SOTA winner | Only when a **new SOTA event** is finalized | Yes | No |
| `capacitorless` + `windowed` | **On-chain weights**: 100% to winner during an event window, else burn | On event boundaries (`alignment_mod`) and window end | Yes | No |

## `capacitor` (EVM Contract Voting)

In `reward_mode: capacitor`, validators:

1. Poll the relay for miner submissions.
2. Independently re-evaluate and pick the best valid candidate above SOTA.
3. Vote on the Capacitor contract by calling `releaseReward(minerColdkey, score)`.
4. When enough trustees vote for the same candidate, the contract transfers stake to the winning miner.

Notes:
- This mode requires an EVM key configured via `evm_key_path` and a working contract config (`contract.rpc_url`, `contract.address`, `contract.abi_file`).
- Weight setting (Bittensor emissions) is separate from the Capacitor bonus. In this codebase, `WeightManager` optionally weights configured `contract_bots` hotkeys; it does not automatically distribute weights across miners.

## `capacitorless` (Relay SOTA + On-Chain Weights)

In capacitorless modes, validators still validate relay submissions, but instead of voting via an EVM contract they:

- Vote for a new SOTA candidate on the relay (`POST /sota/vote`).
- Use a weight manager that sets **on-chain weights** so emissions flow to a burn hotkey and/or the SOTA winner.

### Sticky Burn-Split (Default)

Config:
- `reward_mode: capacitorless` (or `reward_mode: capacitorless_sticky`)
- `capacitorless.mode: sticky_burnsplit` (default if omitted)
- `capacitorless.burn_share: 0.9` (and optionally `capacitorless.winner_share: 0.1`)

Behavior:
- Always sets weights to `{burn_hotkey: 0.9, winner_hotkey: 0.1}`.
- “Winner” is the **most recently finalized SOTA event** from the relay.
- Weights only change when validators finalize a newer SOTA event.

Why you might want it:
- Simple, predictable, and doesn’t depend on time windows.

### Windowed (Timeboxed Rewards)

Config:
- `reward_mode: capacitorless`
- `capacitorless.mode: windowed`
- `capacitorless.alignment_mod`: interval size in blocks (e.g. 360)

Behavior:
- Default weights are `{burn_hotkey: 1.0}`.
- During an active relay SOTA event window `[start_block, end_block)`, sets weights to `{winner_hotkey: 1.0}`.
- Reverts to burn at window end; newer events cut off older ones at the next event start.

Why you might want it:
- Enforces a limited reward window per accepted SOTA, so emissions return to burn unless the network keeps improving.

## Block Number / Sync Requirements

- **All capacitorless modes** need a block number when submitting relay SOTA votes (`seen_block`).
- **Windowed mode** also uses the current chain block to decide whether an event is active and to apply changes around boundaries.
- **Sticky mode** does not use block timing for *when* to switch winners (it switches on “latest finalized event”), but it still interacts with the chain to submit weight updates, which may be rate-limited by Bittensor epoch rules.

## Minimal Config Examples

Sticky default:
```yaml
reward_mode: "capacitorless"
relay:
  url: "http://127.0.0.1:8002"
capacitorless:
  burn_hotkey: "5G..."
  burn_share: 0.9
  # mode omitted => sticky_burnsplit
```

Windowed:
```yaml
reward_mode: "capacitorless"
relay:
  url: "http://127.0.0.1:8002"
capacitorless:
  mode: "windowed"
  burn_hotkey: "5G..."
  alignment_mod: 360
```

