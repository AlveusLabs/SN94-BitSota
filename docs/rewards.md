# Rewards and Incentive Mechanisms

BitSota rewards currently flow through two production systems:

1. Bittensor subnet emissions, directed by validator `set_weights`.
2. Pool reputation rewards, converted to RAO at pool epoch boundaries.

The legacy relay SOTA voting, local validator winner mode, and Capacitor reward
path have been removed from the public validator flow. Production validator
weights come from the autoresearch backend reward snapshot and are applied by
`validator.backend_weight_setter`.

## Bittensor Network Emissions

Bittensor distributes subnet emissions continuously as blocks are produced.
Validators earn from stake and subnet performance. Miners receive emissions
according to validator-set weights.

Current SN94 production weight direction is backend controlled:

```text
Autoresearch backend reward snapshot -> validator.backend_weight_setter -> Bittensor set_weights
```

The backend policy should resolve to:

```text
90% UID 0
10% 5F7MJ2fAyxBG7ci4xP7kQPJanoMdNurk1QBP1AQuFT2Jmzg2
```

The contract hotkey must be resolved dynamically through the metagraph. Do not
hardcode its UID.

## Pool Mining Rewards

Pool miners earn through a reputation system that converts to RAO at epoch
boundaries.

### Reputation Accumulation

Evaluation tasks:

- Evaluator points are derived from agreement quality against strict consensus
  clusters.
- Agreement and disagreement are measured with configurable tolerance
  (`tolerance_ratio`) and threshold (`consensus_threshold`).
- Effective evaluator points follow
  `(agreements - disagreements) * evaluator_weight`.

Evolution tasks:

- Evolver points are based on verified improvement over a configured baseline:
  `sota`, `genealogy`, or `local_evolver`.
- Effective evolver points follow
  `(total_improvement * evolver_weight) - repetition_penalty`.
- Repeat penalties are optional and can be scoped by `miner`, `global`, or
  `both`.

Consensus gating:

- Positive rewards require `in_consensus == true`.
- Positive rewards also require minimum current-window activity
  (`min_reward_activity`).

### Epoch Conversion

At epoch end:

```text
Your RAO = (Your Reputation / Total Pool Reputation) x Epoch Budget x (1 - Pool Fee)
```

Example:

```text
RAO = (21 / 5000) x 1,000,000,000 x 0.95
    = 3,990,000 RAO
```

## Validator Operations

Validators should run:

- `validator.research_validator_runner` for backend-assigned replay validation.
- `validator.backend_weight_setter` for backend-directed chain weights.

Use [Public Autoresearch Validator Runner](public-validator-runner.md) for the
current production runbook.

## Related Guides

- [Direct Mining](mining.md)
- [Pool Mining](pool-mining.md)
- [Validation](validation.md)
- [Reward Direction](reward-modes.md)
