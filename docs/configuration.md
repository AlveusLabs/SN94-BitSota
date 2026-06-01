# Configuration Reference

- Miner configuration (`miner_config.yaml`)
- Validator configuration (`validator_config.yaml`)
- Relay configuration (environment variables + `python -m relay` flags)
- Desktop GUI dev overrides (`new_gui` JSON config + env vars)

If you’re looking for *how* rewards work, see `docs/reward-modes.md`. If you’re looking for *how to run* a miner/validator, see `docs/mining.md`, `docs/pool-mining.md`, and `docs/validation.md`.

---

## Miner (`miner_config.yaml`)

Used by:
- CLI miner: `neurons/miner.py`

### `wallet`

- `wallet.wallet_name` (string, required): Bittensor wallet name (directory under `wallet.path`).
- `wallet.hotkey_name` (string, required): Hotkey name within the wallet.

### `pool_url`

- `pool_url` (string or `null`):
  - `null` → **direct** mode (submit to relay/validators)
  - non-null URL → **pool** mode (talk to pool service)

### `validators`

- `validators` (list[string]): Relay base URLs. In direct mode, the CLI uses the **first** entry as the relay endpoint. If empty, the miner falls back to `http://127.0.0.1:8002`.

### `mining`

- `mining.mode` (string): UI/metadata hint (`direct` or `pool`). The current CLI decides direct vs pool strictly via `pool_url`.
- `mining.task_type` (string): Task identifier.
  - Default/subnet task is `cifar10_binary`. In `test_mode` (local testing), the GUI also exposes `mnist_binary` and `scalar_linear`.
- `mining.engine_type` (string): Evolution engine in direct mode (`archive` or `ga`).
- `mining.cycles` (int): Pool mode only; `0` means run forever.
- `mining.alternate_tasks` (bool): Pool mode only; alternate `evolve`/`evaluate` task requests.
- `mining.delay` (float seconds): Pool mode only; sleep between cycles.
- `mining.max_retries` (int): Pool mode only; retry count for pool task requests. (Supported by code; add it to your YAML if you need it. Default: `3`.)
- `mining.iterations` (int): Reserved/legacy (not used by `neurons/miner.py`).
- `mining.verbose` (bool): Reserved/legacy (the direct client verbosity is driven by `evolution.verbose` today).

### `evolution`

- `evolution.max_generations` (int): Direct mode only; number of generations per mining loop. The CLI sets `MAX_EVOLUTION_GENERATIONS` from this value.
- `evolution.verbose` (bool): Enables extra miner logging.
- `evolution.fec` (dict, optional): Functional Equivalence Cache (FEC) settings.
  - `evolution.fec.cache_size` (int): LRU cache size for probe-based FEC (default: `100000`, `0` disables).
  - `evolution.fec.num_train_examples` (int): Probe train subset size (default: `32`).
  - `evolution.fec.num_valid_examples` (int): Probe validation subset size (default: `32`).
  - `evolution.fec.forget_every` (int): Clear cache every N inserts (default: `0`, disabled).

### `logging`

- `logging.level` (string): Python log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`).
- `logging.file` (string or `null`): If set, logs are written to this file (directories are created automatically).

## Validator Weight Setter (`validator_config.weights.yaml`)

Used by: `validator.backend_weight_setter` plus `bittensor_network/*` helpers.
The legacy relay/local validator path has been removed from production
validation. Production validators should not copy relay, `reward_mode`, or
`capacitorless` settings into `validator_config.weights.yaml`; they should read
targets from the autoresearch backend reward snapshot.

### Bittensor network settings

- `netuid` (int): Subnet netuid.
- `wallet_name` (string): Wallet name.
- `wallet_hotkey` (string): Hotkey name.
- `path` (string): Wallet base path (e.g. `~/.bittensor/wallets/`).
- `network` (string): Subtensor network name (e.g. `test`, `finney`).
- `subtensor_chain_endpoint` (string or `null`): Override websocket endpoint (e.g. `wss://test.finney.opentensor.ai:443`). `null` lets bittensor choose.
- `epoch_length` (int): Used as a local *minimum blocks between weight updates* gate (and compared against chain rate limits).

### Backend reward snapshot

The backend weight setter fetches:

```text
GET https://autoresearch.bitsota.com/api/v1/reward-snapshot
```

It reads `reward_policy.validator_weights` and applies only those targets. The
service defaults to the production autoresearch backend; use `--coordinator-url`
only for an explicit test environment.

### Weights

- `weights.wait_for_inclusion` (bool): Passed to `subtensor.set_weights(...)` via `bittensor_network/_weights.py`.
- `weights.wait_for_finalization` (bool): Passed to `subtensor.set_weights(...)` via `bittensor_network/_weights.py`.
- `weights.check_interval` (int seconds): Reserved; intended to control how often a weight background loop runs.
- `weights.auto_restart` (bool): Reserved; intended to restart the weight loop if it crashes.

---

## Hyperparameters JSON (preferred)

The miner + validator evaluation defaults are now sourced from JSON files in the repo root:

- `miner_hyperparams.json` (direct miner defaults)
- `validator_hyperparams.json` (validator evaluation defaults)

These are meant to replace the “set a pile of env vars” workflow for routine tuning.

### Miner (`miner_hyperparams.json`)

Key fields (all optional; defaults shown in the file):
- `miner_task_count`, `miner_task_seed`
- `validator_task_count` (optional local pre-submit verification suite size)
- `fec_cache_size`, `fec_train_examples`, `fec_valid_examples`, `fec_forget_every`
- `submission_cooldown_seconds`, `submit_only_if_improved`, `max_submission_attempts_per_generation`
- `validate_every_n_generations`
- `sota_cache_seconds`, `sota_failure_backoff_seconds`
- `persist_state`, `persist_every_n_generations`
- `gene_dump_every`

### Validator evaluation (`validator_hyperparams.json`)

Key fields (all optional; defaults shown in the file):
- `epochs`, `task_count`, `task_seed`
- `default_task_type`, `default_input_dim`
- `log_task_scores` (only logs per-task scores when log level is `DEBUG`)
- `tasks.<task_type>.n_samples` / `train_split` (for `cifar10_binary`, `mnist_binary`)
- `tasks.scalar_linear.train_samples` / `val_samples`

### Override config file paths (optional)

- `BITSOTA_MINER_HYPERPARAMS_PATH` / `MINER_HYPERPARAMS_PATH`
- `BITSOTA_VALIDATOR_HYPERPARAMS_PATH` / `VALIDATOR_HYPERPARAMS_PATH`

## Runtime tuning (environment variable overrides)

Env vars are still supported for backward compatibility, and override the JSON defaults above.

### Miner

- `MAX_EVOLUTION_GENERATIONS` (default: `15`): Upper bound for direct mining loops (normally set from `evolution.max_generations`).
- `MINER_TASK_COUNT` (default: `32`): Number of deterministic subtasks scored per genome (CIFAR-10 only).
- `MINER_TASK_SEED` (default: `0`): Seed for deterministic miner task suite generation.
- `MINER_FEC_CACHE_SIZE` (default: `100000`): LRU cache size for functional equivalence caching (0 disables).
- `MINER_FEC_TRAIN_EXAMPLES` (default: `32`): Probe train subset size for FEC.
- `MINER_FEC_VALID_EXAMPLES` (default: `32`): Probe validation subset size for FEC.
- `MINER_FEC_FORGET_EVERY` (default: `0`): Clear the FEC cache every N inserts (0 disables).
- `MINER_SUBMISSION_COOLDOWN_SECONDS` (default: `60`): Minimum seconds between relay submissions.
- `MINER_SUBMIT_ONLY_IF_IMPROVED` (default: false): If enabled, only submit when verified score beats the miner’s local best.
- `MINER_MAX_SUBMISSION_ATTEMPTS_PER_GENERATION` (default: `1` or `3`): Per-generation submission attempts.
- `MINER_VALIDATE_EVERY_N_GENERATIONS` (default: `1`): Throttle local “validator-style” scoring.
- `MINER_SOTA_CACHE_SECONDS` (default: `30`): Cache duration for SOTA fetches.
- `MINER_SOTA_FAILURE_BACKOFF_SECONDS` (default: `5`): Backoff after failed SOTA fetch.
- `MINER_GENE_DUMP_EVERY` (default: `1000`): Frequency for debug gene dumps (when enabled in code).

### Validator evaluation

- `VALIDATOR_TASK_COUNT` (default: `128`): Number of deterministic tasks in the validator eval suite.
- `VALIDATOR_TASK_SEED` (default: `1337`): Seed for validator task suite generation.
- `LOG_VALIDATOR_TASK_SCORES` (default: `0`): If truthy *and* validator logging is `DEBUG`, logs per-task scores.

### CIFAR task caching

- `CIFAR_TASK_CACHE_MAXSIZE` (default: `512`): LRU cache size for prepared CIFAR projection tasks.

### Script key (operational tooling)

- `PRIVATE_KEY` (required for some scripts): Used by `scripts/common.py` when no `--keyfile` is provided.

---

## Desktop GUI dev overrides (`new_gui`)

Frozen desktop builds ignore overrides and use hardcoded defaults. Local/dev runs support endpoint overrides via JSON.

### `BITSOTA_GUI_CONFIG`

If set, `BITSOTA_GUI_CONFIG` points to a JSON file that overrides GUI endpoints:

- `relay_endpoint` (string)
- `update_manifest_url` (string)
- `pool_endpoint` (string)
- `test_mode` (bool)
- `test_invite_code` (string)
- `miner_task_count` (int)
- `validator_task_count` (int)
- `miner_validate_every_n_generations` (int)
- `problem_config_path` (string)

If `BITSOTA_GUI_CONFIG` is not set, the app looks for:
- `./bitsota_gui_config.json`
- `./gui_config.json`
- `~/.bitsota/gui_config.json`
