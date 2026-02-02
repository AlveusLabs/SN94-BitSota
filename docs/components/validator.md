# Validator

Validators:

- Poll the relay for submissions
- Re-evaluate candidates on a deterministic task suite
- Vote to finalize SOTA events and set on-chain weights

## Entrypoints

- `python neurons/validator_node.py` runs the main validator node
  - Optional: `--accept-test` enables UID0-only relay test-submission evaluation for logging (no weights)
- `python3 -m validator.local_validator` runs a relay-focused local validator used for testing

See [Validation](../validation.md) for setup and tuning, and [Configuration Reference](../configuration.md) for `validator_config.yaml`.
