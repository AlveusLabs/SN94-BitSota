# Validator Guide

SN94 production validators now use the autoresearch backend path only:

1. `validator.research_validator_runner` signs backend worklist requests,
   replays assigned submissions in the Docker/CUDA sandbox, and posts observed
   metrics back to the backend.
2. `validator.backend_weight_setter` polls
   `https://autoresearch.bitsota.com/api/v1/reward-snapshot`, reads
   `reward_policy.validator_weights`, and submits Bittensor `set_weights`.

The legacy relay validator, relay SOTA voting, and local winner weight-setting
path have been removed from this repo. Do not run any old service or process
that calls `set_weights` from relay/local SOTA state.

## Production Setup

Use the public runbook for current production instructions:

- [Public Autoresearch Validator Runner](public-validator-runner.md)
- [Public Validator Quickstart](public-validator-quickstart.md)
- [SN94 System Structure](sn94-system-structure.md)

Minimal process model:

```text
Autoresearch backend -> replay validator -> backend result consensus
Autoresearch backend -> backend weight setter -> Bittensor SN94 weights
```

Production backend weight setting defaults to:

```text
https://autoresearch.bitsota.com
```

The current backend policy should target:

```text
90% UID 0
10% 5F7MJ2fAyxBG7ci4xP7kQPJanoMdNurk1QBP1AQuFT2Jmzg2
```

Always resolve the contract hotkey dynamically through the metagraph. Do not
hardcode its UID.

## Health Checks

```bash
systemctl status bitsota-replay-validator.service --no-pager
systemctl status bitsota-backend-weights.service --no-pager
journalctl -u bitsota-replay-validator.service -n 100 --no-pager
journalctl -u bitsota-backend-weights.service -n 100 --no-pager
```

Check the backend policy without submitting weights:

```bash
python -m validator.backend_weight_setter \
  --config validator_config.weights.yaml \
  --dry-run \
  --ignore-rate-limit
```

If the dry run does not show the contract hotkey above, do not start the live
weight service.
