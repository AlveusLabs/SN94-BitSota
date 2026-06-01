# Validator Reward Direction

Production SN94 validators no longer choose between relay consensus, local
winner mode, windowed capacitorless mode, or the older Capacitor contract path.
Those validator reward modes were part of the legacy relay validator stack and
have been removed from the public validator path.

Current production weight direction is backend controlled:

```text
Autoresearch backend reward snapshot -> validator.backend_weight_setter -> Bittensor set_weights
```

The backend weight setter fetches:

```text
GET https://autoresearch.bitsota.com/api/v1/reward-snapshot
```

It reads `reward_policy.validator_weights` and applies only those targets. The
weight setter defaults to `https://autoresearch.bitsota.com`; pass
`--coordinator-url` only for an explicit test backend.

## Current Policy

The production policy should resolve to:

```text
90% UID 0
10% 5F7MJ2fAyxBG7ci4xP7kQPJanoMdNurk1QBP1AQuFT2Jmzg2
```

The contract hotkey must be resolved dynamically through the current metagraph.
Do not hardcode its UID.

## Validator Commands

Use the current runbook:

- [Public Autoresearch Validator Runner](public-validator-runner.md)
- [Validation Guide](validation.md)

Do not run old relay/local validator services or any separate local weight
setter. The only production weight-setting service should be
`validator.backend_weight_setter`.
