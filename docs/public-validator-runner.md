# Public Autoresearch Validator Runner

SN94 exposes a public validator runner so an operator can validate
autoresearch submissions without running the private coordinator database
worker. The normal path is a signed backend worklist scan:

1. The validator calls `POST /api/v1/validator/submissions/scan`.
2. It replays every returned submission with the returned `replay_spec`.
3. It posts observed metrics to `POST /api/v1/validator/jobs/{job_id}/result`.

The replay runner does not set Bittensor weights and does not talk to the
Pool/Merkle contract. Chain weights are handled by the separate backend weight
setter.

## Endpoints

Production:

```text
autoresearch backend: https://autoresearch.bitsota.com
subtensor network: finney
SN94 netuid: 94
production contract-hotkey target: 5F7MJ2fAyxBG7ci4xP7kQPJanoMdNurk1QBP1AQuFT2Jmzg2
```

Testing:

```text
autoresearch backend: https://autoresearch-test.bitsota.com
raw test backend: https://chvp2wytst.eu-central-1.awsapprunner.com
```

The example replay config defaults to the raw testing backend. Change
`coordinator_url` before running against production.

## Fresh Validator Host

Use an Ubuntu GPU host with NVIDIA drivers, Docker, and NVIDIA Container
Toolkit. The replay path expects CUDA to be available inside Docker.

Install baseline packages:

```bash
sudo apt update
sudo apt install -y git curl ca-certificates python3 python3-venv python3-pip build-essential docker.io
sudo usermod -aG docker "$USER"
```

Install NVIDIA Container Toolkit using NVIDIA's official Ubuntu instructions
for the host OS, then restart Docker. The current validator smoke requirement is
that this command succeeds:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

Log out and back in after adding the Docker group.

## Clone And Install

Until the public validator fixes are merged into the default branch, use the
active validator branch:

```bash
sudo mkdir -p /opt/bitsota
sudo chown "$USER:$USER" /opt/bitsota

git clone --branch testnet-net-gui-pool-agents https://github.com/AlveusLabs/SN94-BitSota.git /opt/bitsota/SN94-BitSota
cd /opt/bitsota/SN94-BitSota

python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

Required fix level:

```bash
git merge-base --is-ancestor 41e4375 HEAD
```

If that exits nonzero, the checkout is missing the Docker heldout prefetch fix.

## Wallets

Create or restore a Bittensor wallet whose hotkey is registered on SN94. The
backend must allowlist this validator hotkey before signed scans return private
work. The same hotkey can run replay validation and backend-directed weight
setting, but do not run multiple weight setters on the same hotkey.

```bash
btcli wallet new_coldkey --wallet.name validator_wallet
btcli wallet new_hotkey --wallet.name validator_wallet --wallet.hotkey validator_hotkey
btcli subnet register --netuid 94 --wallet.name validator_wallet --wallet.hotkey validator_hotkey --network finney
```

Existing validators can restore/import instead of creating new keys. Do not put
mnemonics in tracked config files.

## Replay Validator Config

Copy the example config:

```bash
cp research_validator_config.yaml.example research_validator_config.yaml
```

Production replay config:

```yaml
coordinator_url: "https://autoresearch.bitsota.com"
claim_path: "/api/v1/validator/submissions/scan"
pending_submissions_fallback: false

wallet_name: "validator_wallet"
wallet_hotkey: "validator_hotkey"
wallet_path: "~/.bittensor/wallets/"

workspace_root: "/srv/bitsota/public-validator-workspaces"
cycles: 0
interval_seconds: 30
timeout_s: 7200

replay_sandbox_mode: "docker"
replay_sandbox_image: "bitsota-research-validator-cuda:local"
replay_sandbox_dockerfile: "docker/research-validator-cuda.Dockerfile"
replay_sandbox_gpus: "all"
replay_sandbox_setup_network_mode: "none"
replay_sandbox_benchmark_network_mode: "none"
replay_sandbox_memory_limit: "32g"
replay_sandbox_pids_limit: 512
replay_sandbox_workspace_size_bytes: 17179869184

allow_unsafe_host_replay: false
allow_local_artifacts: false
dry_run: true
```

Dry-run one backend validation cycle:

```bash
source .venv/bin/activate
python -m validator.research_validator_runner --config research_validator_config.yaml --once --dry-run
```

Run one real cycle:

```bash
python -m validator.research_validator_runner --config research_validator_config.yaml --once --no-dry-run
```

Run continuously:

```bash
python -m validator.research_validator_runner --config research_validator_config.yaml --no-dry-run
```

Equivalent entrypoints:

```bash
python scripts/research_validator_runner.py --config research_validator_config.yaml
bitsota-research-validator --config research_validator_config.yaml
```

## Replay Options

- `--config`: read runner, wallet, and replay settings from a YAML, `.config`,
  or JSON file. CLI flags override config-file values.
- `--task-slug` or `--task-id`: restrict replay to one task.
- `--hotkey-mnemonic` or `--wallet-file`: use the same SN94 wallet input helpers
  as the research-agent miner.
- `--dry-run` / `--no-dry-run`: replay locally without posting results, or post
  results back to the backend.
- `--replay-sandbox-mode`: `docker` runs setup and benchmark commands inside
  the Docker/CUDA sandbox. `host` requires `--allow-unsafe-host-replay`.
- `--replay-sandbox-gpus`: Docker `--gpus` value. Use `all` on a CUDA validator
  host.
- `--claim-path`: override the signed backend worklist endpoint. Use
  `/api/v1/validator/jobs/claim` only for legacy single-job compatibility.
- `--pending-submissions-fallback`: use the older public pending-submissions
  scan only when testing an undeployed or old backend.
- `--allow-local-artifacts`: allow `file://` or relative artifact URIs during
  local testing only.

## Heldout Delivery

The config file intentionally does not include heldout dataset names,
percentages, or sync numbers. Those values come from the backend in the signed
worklist response after validator auth and on-chain checks pass.

When the backend sends `AUTORESEARCH_HELDOUT_SOURCES_JSON`, the runner fetches
the Hugging Face rows in the validator host process before starting Docker. It
then writes `.autoresearch-heldout/manifest.json` into the replay workspace and
rewrites benchmark env to:

```text
AUTORESEARCH_HELDOUT_DATASET=validator-private-shard
AUTORESEARCH_PRIVATE_HELDOUT_MANIFEST=.autoresearch-heldout/manifest.json
AUTORESEARCH_PRIVATE_HELDOUT_ROOT=.autoresearch-heldout
```

The runner strips `AUTORESEARCH_HELDOUT_SOURCES_JSON` and Hugging Face token
env vars before Docker starts. The benchmark container should not need network
access.

## Docker/CUDA Sandbox

Recommended replay uses Docker mode:

```yaml
replay_sandbox_mode: "docker"
allow_unsafe_host_replay: false
replay_sandbox_image: "bitsota-research-validator-cuda:local"
replay_sandbox_dockerfile: "docker/research-validator-cuda.Dockerfile"
replay_sandbox_gpus: "all"
replay_sandbox_setup_network_mode: "none"
replay_sandbox_benchmark_network_mode: "none"
replay_sandbox_memory_limit: "32g"
replay_sandbox_pids_limit: 512
replay_sandbox_workspace_size_bytes: 17179869184
```

The sandbox image is built automatically from
`docker/research-validator-cuda.Dockerfile` if the configured tag does not
already exist. The runner copies only the prepared replay workspace into a
tmpfs-backed Docker volume, runs setup and benchmark commands in read-only
containers, and copies only the configured result file back out. Wallets and
wallet files stay outside the sandbox.

Setup and benchmark networking default to `none`. The validator host downloads
submission artifacts and backend-directed heldout rows before Docker starts.
Reward-active task repos must have dependencies available through the sandbox
image/workspace, or the operator must explicitly opt into networked setup for
that task.

Host mode executes submitted setup and benchmark commands directly on the
validator machine and requires `allow_unsafe_host_replay: true`. Use host mode
only for local development or a disposable isolated machine.

## Backend Compatibility

- Current backends expose signed `POST /api/v1/validator/submissions/scan` and
  `POST /api/v1/validator/jobs/{job_id}/result`.
- The worklist response includes validator-only replay parameters, including
  hidden heldout handles and sync numbers. Validators do not need production DB,
  App Runner, or admin credentials.
- The fallback path is only for older backends or explicitly enabled legacy
  direct verification. It uses public task/submission APIs and cannot recover
  backend-private replay values.

Use the signed validator worklist endpoint for reward-active competitions.

## Patch Surface

Patch-surface enforcement happens before replay. The public runner rejects:

- any submitted patch path outside task `allowed_patch_paths`;
- generated Python bytecode/cache paths;
- patches larger than `max_patch_bytes` when provided by backend/task.

The default patch cap is `262144` bytes.

## Backend-Controlled Chain Weights

The public replay runner only evaluates submissions and posts results back to
the autoresearch backend. Chain weight setting is a separate process. Do not
run two processes that both call `set_weights` for the same validator hotkey
unless you intentionally want them to race.

The standalone backend weight setter reads `reward_policy.validator_weights`
from `GET /api/v1/reward-snapshot`, resolves backend UID/hotkey targets through
the SN94 metagraph, normalizes target weights to sum to `1.0`, and calls the
existing Bittensor `set_weights` path.

Dry-run the backend policy:

```bash
python scripts/autoresearch_weight_setter.py \
  --config validator_config.yaml \
  --coordinator-url https://autoresearch.bitsota.com \
  --dry-run
```

Run one real update:

```bash
python scripts/autoresearch_weight_setter.py \
  --config validator_config.yaml \
  --coordinator-url https://autoresearch.bitsota.com
```

Run continuously:

```bash
python scripts/autoresearch_weight_setter.py \
  --config validator_config.yaml \
  --coordinator-url https://autoresearch.bitsota.com \
  --loop \
  --interval-seconds 300
```

Installed package entrypoint:

```bash
bitsota-autoresearch-weights --config validator_config.yaml --coordinator-url https://autoresearch.bitsota.com
```

Supported backend modes:

- `local`: no-op; leave weights unchanged.
- `burn_uid0`: set `100%` to UID `0`.
- `targets`: use backend-provided `targets`, where each target has exactly one
  of `uid` or `hotkey` plus a positive `weight`.

For the production contract-hotkey path, the backend target set should include:

```text
5F7MJ2fAyxBG7ci4xP7kQPJanoMdNurk1QBP1AQuFT2Jmzg2
```

## Legacy Relay Or Capacitor Contract Node

`neurons/validator_node.py` can run legacy relay/capacitor validation:

- `reward_mode: "capacitor"` uses the old EVM Capacitor contract manager.
- `reward_mode: "capacitorless"` and `reward_mode: "capacitorless_sticky"` use
  relay votes plus Bittensor weight setting.

Use this deliberately and separately from the public autoresearch replay runner.
Do not describe the Pool/Merkle contract hotkey as an EVM Capacitor contract
address.

## Systemd Unit For Replay Validator

```ini
[Unit]
Description=BitSota autoresearch replay validator
After=docker.service network-online.target
Wants=network-online.target

[Service]
WorkingDirectory=/opt/bitsota/SN94-BitSota
Environment=PYTHONUNBUFFERED=1
ExecStart=/opt/bitsota/SN94-BitSota/.venv/bin/python -m validator.research_validator_runner --config /opt/bitsota/SN94-BitSota/research_validator_config.yaml --no-dry-run
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Systemd unit for backend-policy weights, only if that process is meant to own
chain weights for this validator hotkey:

```ini
[Unit]
Description=BitSota backend-policy validator weights
After=network-online.target
Wants=network-online.target

[Service]
WorkingDirectory=/opt/bitsota/SN94-BitSota
Environment=PYTHONUNBUFFERED=1
ExecStart=/opt/bitsota/SN94-BitSota/.venv/bin/python -m validator.backend_weight_setter --config /opt/bitsota/SN94-BitSota/validator_config.yaml --coordinator-url https://autoresearch.bitsota.com --loop --interval-seconds 300
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```
