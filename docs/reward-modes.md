# Backend Reward Snapshot

The current production validator path uses backend-directed reward snapshots.
The old `capacitorless`, relay SOTA, and local winner modes are archived under
[AutoML-Zero Archive](archive/automl-zero/index.md).

## Snapshot Endpoint

```text
GET https://autoresearch.bitsota.com/api/v1/reward-snapshot
```

The snapshot is the coordination point between:

- accepted autoresearch submissions;
- Pool reward publication;
- Merkle claim packages;
- validator weight setting.

## Validator Weight Policy

Validators should read:

```json
{
  "reward_policy": {
    "validator_weights": {
      "mode": "targets",
      "targets": []
    }
  }
}
```

Supported backend modes in the client are:

| Mode | Meaning |
| --- | --- |
| `local` | Do not override local behavior. Not the production target. |
| `burn_uid0` | Set all weight to UID `0`. |
| `targets` | Set explicit UID/hotkey targets from the backend. |

Production should use explicit targets and resolve hotkeys dynamically.

## Safety Rules

- Run only one weight-owning process per validator hotkey.
- Do not run legacy relay/local SOTA weight setters in production.
- Do not hardcode the contract UID; resolve the contract hotkey each run.
- Dry-run the backend weight setter before enabling live weight submission.

See [Validator Guide](validation.md) and
[Public Autoresearch Validator Runner](public-validator-runner.md).
