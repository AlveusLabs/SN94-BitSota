# Validator

Validators:

- Request signed replay jobs from the autoresearch backend
- Re-evaluate assigned submissions in the Docker/CUDA public replay sandbox
- Post observed metrics back to the backend
- Run a separate backend weight setter that applies backend `validator_weights`

## Entrypoints

- `python -m validator.research_validator_runner` runs the replay validator
- `python -m validator.backend_weight_setter` runs backend-directed chain weights

The legacy relay validator and local winner weight-setting path have been
removed. See [Validation](../validation.md), [Public Autoresearch Validator
Runner](../public-validator-runner.md), and [Architecture Overview](../architecture.md).
