# Public Validator Quickstart

This is the short path for operators who already have a clean Ubuntu GPU host
and want the minimum production validator setup. It skips the Pool/Merkle
verifier and most troubleshooting checks from the full runbook.

Use the full [Public Autoresearch Validator Runner](public-validator-runner.md)
if any command here fails or if you are migrating from an old relay/local
validator.

## Minimum Host

- Ubuntu 22.04 LTS or newer.
- One NVIDIA CUDA-capable GPU with at least 24 GB VRAM.
- Tested EC2 class: `g5.2xlarge` with one NVIDIA A10G, 24 GB VRAM, 8 vCPU,
  and 32 GiB RAM.
- At least 150 GB SSD/EBS disk; 200 GB recommended.
- Docker must be able to run CUDA containers.

Production validators run two services:

- `bitsota-replay-validator.service`
- `bitsota-backend-weights.service`

Do not run any legacy relay/local validator process or any other process that
calls `set_weights` for the same validator hotkey.

## 1. Prepare Host

```bash
sudo apt update
sudo apt install -y git curl ca-certificates python3.11 python3.11-venv python3.11-dev python3-pip build-essential docker.io
sudo systemctl enable --now docker

sudo useradd -m -s /bin/bash validator 2>/dev/null || true
sudo usermod -aG docker validator
sudo install -d -m 0750 -o validator -g validator /opt/bitsota
sudo install -d -m 0700 -o validator -g validator /srv/bitsota/public-validator-workspaces
```

Install NVIDIA Container Toolkit for your Ubuntu version, restart Docker, then
confirm CUDA works:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

Continue as the validator user:

```bash
sudo -iu validator
```

## 2. Install Code

```bash
git clone --branch main https://github.com/AlveusLabs/SN94-BitSota.git /opt/bitsota/SN94-BitSota
cd /opt/bitsota/SN94-BitSota

python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

If the repo already exists:

```bash
cd /opt/bitsota/SN94-BitSota
git checkout main
git pull --ff-only origin main
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## 3. Prepare Wallet

Use a validator hotkey registered on SN94 netuid 94 and allowlisted by the
autoresearch backend.

For a new wallet:

```bash
btcli wallet new_coldkey --wallet.name validator_wallet
btcli wallet new_hotkey --wallet.name validator_wallet --wallet.hotkey validator_hotkey
btcli subnet register --netuid 94 --wallet.name validator_wallet --wallet.hotkey validator_hotkey --network finney
```

Existing validators can restore/import their real validator wallet instead. Do
not put mnemonics in tracked files.

## 4. Configure Replay

```bash
cd /opt/bitsota/SN94-BitSota
cat > research_validator_config.yaml <<'EOF'
coordinator_url: "https://autoresearch.bitsota.com"
claim_path: "/api/v1/validator/submissions/scan"

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
dry_run: false
EOF
```

Replace `validator_wallet` and `validator_hotkey` with the local wallet names.

Run one replay pass:

```bash
source .venv/bin/activate
python -m validator.research_validator_runner --config research_validator_config.yaml --once --no-dry-run
```

`idle: no pending replay jobs` is acceptable. Allowlist or auth errors mean the
backend operator must allowlist your validator hotkey.

## 5. Configure Weights

```bash
cd /opt/bitsota/SN94-BitSota
cat > validator_config.weights.yaml <<'EOF'
netuid: 94
wallet_name: "validator_wallet"
wallet_hotkey: "validator_hotkey"
path: "~/.bittensor/wallets/"
network: "finney"
subtensor_chain_endpoint: "wss://entrypoint-finney.opentensor.ai:443"
epoch_length: 100

weights:
  wait_for_inclusion: true
  wait_for_finalization: false
EOF
```

Replace `validator_wallet` and `validator_hotkey` with the local wallet names.

Check the backend target before enabling the service:

```bash
source .venv/bin/activate
python -m validator.backend_weight_setter \
  --config validator_config.weights.yaml \
  --dry-run \
  --ignore-rate-limit
```

The dry run must show the production contract hotkey:

```text
5F7MJ2fAyxBG7ci4xP7kQPJanoMdNurk1QBP1AQuFT2Jmzg2
```

Stop old weight setters if this host was previously used:

```bash
sudo systemctl disable --now bitsota-prod-weight-setter.service 2>/dev/null || true
sudo systemctl disable --now bitsota-validator.service 2>/dev/null || true
sudo systemctl disable --now bitsota-capacitorless-weights.service 2>/dev/null || true
sudo systemctl disable --now bitsota-local-weights.service 2>/dev/null || true
pgrep -af 'prod_weight_setter|validator_node|local_validator|capacitorless|relay_client|set_weights' || true
```

## 6. Install Services

Exit the validator shell for the `sudo tee` commands:

```bash
exit
```

Create the replay service:

```bash
sudo tee /etc/systemd/system/bitsota-replay-validator.service >/dev/null <<'EOF'
[Unit]
Description=BitSota SN94 autoresearch replay validator
After=network-online.target docker.service
Wants=network-online.target docker.service

[Service]
User=validator
Group=validator
WorkingDirectory=/opt/bitsota/SN94-BitSota
Environment=HOME=/home/validator
Environment=PYTHONUNBUFFERED=1
ExecStart=/opt/bitsota/SN94-BitSota/.venv/bin/python -m validator.research_validator_runner --config /opt/bitsota/SN94-BitSota/research_validator_config.yaml --no-dry-run
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
```

Create the backend weight service:

```bash
sudo tee /etc/systemd/system/bitsota-backend-weights.service >/dev/null <<'EOF'
[Unit]
Description=BitSota backend-policy validator weights
After=network-online.target
Wants=network-online.target

[Service]
User=validator
Group=validator
WorkingDirectory=/opt/bitsota/SN94-BitSota
Environment=HOME=/home/validator
Environment=PYTHONUNBUFFERED=1
ExecStart=/opt/bitsota/SN94-BitSota/.venv/bin/python -m validator.backend_weight_setter --config /opt/bitsota/SN94-BitSota/validator_config.weights.yaml --loop --interval-seconds 300
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
```

Enable both services:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now bitsota-replay-validator.service
sudo systemctl enable --now bitsota-backend-weights.service
```

## 7. Check Status

```bash
systemctl status bitsota-replay-validator.service --no-pager
systemctl status bitsota-backend-weights.service --no-pager

journalctl -u bitsota-replay-validator.service -f
journalctl -u bitsota-backend-weights.service -f
```

Healthy weight logs should repeatedly show backend `targets` mode, the
validator hotkey, and the contract hotkey. `skipped_rate_limit` is normal
between permitted Bittensor weight updates.

If the quickstart fails, switch to the full
[Public Autoresearch Validator Runner](public-validator-runner.md).
