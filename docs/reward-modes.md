# Reward Modes

This repo supports multiple **validator reward modes** (configured via `reward_mode` in `validator_config.yaml`). These modes change *how validators reward miners* after validating relay submissions.

## Quick Comparison

| Mode | What pays the miner | When weights change | Relay required | Winner source |
|------|----------------------|-------------------|----------------|---------------|
| `capacitorless_sticky` (default) | **On-chain weights**: 90% burn + 10% to SOTA winner | When new SOTA event finalized (relay) or found (local) | Yes | `relay` or `local` |
| `capacitorless` + `windowed` | **On-chain weights**: 100% to winner during event window, else burn | On event boundaries (`alignment_mod`) and window end | Yes | `relay` or `local` |
| `capacitor` (deprecated) | Capacitor EVM contract vote (`releaseReward`) | Not miner-targeted by default | Optional | N/A (uses contract) |

## `capacitorless_sticky` (Default - Yuma Consensus)

In `reward_mode: capacitorless_sticky` or `reward_mode: capacitorless`, validators use on-chain weight setting to direct emissions.

**How it works:**
1. Poll relay for miner submissions
2. Independently re-evaluate and pick best valid candidate above SOTA
3. Choose winner source (`capacitorless.winner_source`):
   - `relay`: Vote on relay, wait for consensus, set weights when SOTA event finalized
   - `local`: Set weights immediately based on own evaluation
4. Set on-chain weights: 90% burn_hotkey, 10% winner_hotkey
5. Network emissions flow via Yuma consensus

**Configuration:**
- No EVM key required
- Requires `capacitorless.burn_hotkey` configured
- Relay required for submissions and SOTA tracking
- Winner source determines coordination level

## Winner Source Options

Both `capacitorless_sticky` and `windowed` modes support two winner selection methods:

### Relay Consensus (winner_source: "relay")

Config:
```yaml
capacitorless:
  winner_source: "relay"
  events_limit: 50
  event_refresh_interval_s: 60
```

Behavior:
- Validator votes on relay to accept new SOTA candidate
- Relay aggregates votes from multiple validators
- When consensus reached, relay finalizes SOTA event
- Validator fetches finalized events and updates weights
- All validators converge on same winner

Why use it:
- Coordinated subnet behavior
- All validators agree on same winner
- Recommended for production

### Local Mode (winner_source: "local")

Config:
```yaml
capacitorless:
  winner_source: "local"
  min_winner_improvement: 0.0
  apply_weights_inline: true
  submit_sota_votes: true  # still vote on relay for coordination
```

Behavior:
- Validator tracks best score it has evaluated
- When better candidate found, updates local winner immediately
- Sets weights without waiting for relay consensus
- Each validator may choose different winner temporarily
- Optionally still submits votes to relay

Why use it:
- Faster weight updates
- Lower latency response to breakthroughs
- Useful for independent operation

## Sticky Burn-Split Mode (Default)

Config:
- `reward_mode: capacitorless_sticky` or `reward_mode: capacitorless`
- `capacitorless.mode: sticky_burnsplit` (default if omitted)
- `capacitorless.burn_share: 0.9` (and optionally `capacitorless.winner_share: 0.1`)

Behavior:
- Always sets weights to `{burn_hotkey: 0.9, winner_hotkey: 0.1}`
- Winner determined by `winner_source` setting (relay or local)
- Weights only change when new winner identified
- No time windows or expiration

Why use it:
- Simple and predictable
- Continuous rewards for SOTA winner
- Deflationary mechanism (90% burn)

## Windowed Mode (Timeboxed Rewards)

Config:
- `reward_mode: capacitorless`
- `capacitorless.mode: windowed`
- `capacitorless.alignment_mod`: interval size in blocks (e.g. 360)
- `capacitorless.winner_source`: `relay` or `local`

Behavior:
- Default weights are `{burn_hotkey: 1.0}`
- During active SOTA event window `[start_block, end_block)`, sets weights to `{winner_hotkey: 1.0}`
- Winner determined by `winner_source` setting (relay event or local best)
- Reverts to burn at window end
- Newer events cut off older ones at next event start

Why use it:
- Limited reward window per SOTA
- Emissions return to burn unless network keeps improving
- Creates urgency for continuous innovation

## Block Number / Sync Requirements

- All capacitorless modes need a block number when submitting relay SOTA votes (`seen_block`)
- Windowed mode uses current chain block to decide whether event is active and apply changes around boundaries
- Sticky mode switches winners on "latest finalized event" (relay) or "local best" (local), but still uses Bittensor epoch rules for weight rate-limiting

## Minimal Config Examples

**Sticky mode with relay consensus (recommended):**
```yaml
reward_mode: "capacitorless_sticky"
relay:
  url: "https://relay.bitsota.com"
capacitorless:
  burn_hotkey: "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
  burn_share: 0.9
  winner_source: "relay"
  events_limit: 50
  event_refresh_interval_s: 60
```

**Sticky mode with local evaluation (faster):**
```yaml
reward_mode: "capacitorless_sticky"
relay:
  url: "https://relay.bitsota.com"
capacitorless:
  burn_hotkey: "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
  burn_share: 0.9
  winner_source: "local"
  min_winner_improvement: 0.0
  apply_weights_inline: true
  submit_sota_votes: true
```

**Windowed mode:**
```yaml
reward_mode: "capacitorless"
relay:
  url: "https://relay.bitsota.com"
capacitorless:
  mode: "windowed"
  burn_hotkey: "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
  winner_source: "relay"
  alignment_mod: 360
```

## Capacitor Mode (Deprecated)

The original EVM contract voting mode is still available but deprecated:

Config:
- `reward_mode: "capacitor"`
- Requires `evm_key_path` configured
- Requires `contract.rpc_url`, `contract.address`, `contract.abi_file`

Behavior:
- Validators vote on Capacitor contract by calling `releaseReward(minerColdkey, score)`
- When 2/3 trustees agree, contract transfers stake to winning miner
- Weight setting separate from contract voting

This mode is deprecated in favor of capacitorless modes using Yuma consensus.

