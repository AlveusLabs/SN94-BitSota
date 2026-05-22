# Public Autoresearch Validator Runner

This is the operator runbook for a public SN94 validator. A working production
validator host runs two SN94 processes plus a Pool/Merkle contract check:

1. Replay validator: asks the autoresearch backend for submissions, replays them
   in Docker/CUDA, and posts observed metrics back to the backend.
2. Backend weight setter: reads the backend reward policy and submits Bittensor
   `set_weights` from the validator hotkey.
3. Contract monitor: checks that Pool/Merkle publication, claim visibility, and
   the on-chain contract state are healthy.

If the service boundaries are unfamiliar, read
[SN94 System Structure](sn94-system-structure.md) first.

## Production Values

The install steps below use these folders:

```text
/opt/bitsota/SN94-BitSota/
  SN94 validator code, replay config, weight-setter config, and Pool/Merkle
  contract monitor.
  Put research_validator_config.yaml and validator_config.weights.yaml here.

/etc/systemd/system/
  Ubuntu background service files.
  Put bitsota-replay-validator.service, bitsota-backend-weights.service,
  and bitsota-contract-verifier.service here.
```

Use these exact values when the later steps ask you to edit config files or run
commands.

In `/opt/bitsota/SN94-BitSota/research_validator_config.yaml`:

```yaml
coordinator_url: "https://autoresearch.bitsota.com"
```

Example `/opt/bitsota/SN94-BitSota/validator_config.weights.yaml`:

```yaml
netuid: 94
network: "finney"
subtensor_chain_endpoint: "wss://entrypoint-finney.opentensor.ai:443"
```

When running the backend weight setter, including inside its systemd service:

```bash
--coordinator-url https://autoresearch.bitsota.com
```

When checking the live production Pool/Merkle service in step 5:

```bash
--pool-url https://pool.bitsota.com
```

Testing uses `https://autoresearch-test.bitsota.com`, but production validators
should use `https://autoresearch.bitsota.com`.

## 1. Prepare The Host

Use an Ubuntu GPU host with NVIDIA drivers, Docker, and NVIDIA Container
Toolkit. The validator replays untrusted miner work inside Docker with CUDA GPU
access.

Install baseline packages:

```bash
sudo apt update
sudo apt install -y git curl ca-certificates python3 python3-venv python3-pip build-essential docker.io
sudo usermod -aG docker "$USER"
```

Install NVIDIA Container Toolkit using NVIDIA's official Ubuntu instructions
for your Ubuntu version, then restart Docker and log out/back in.

These two commands must both work before continuing:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

## 2. Install SN94-BitSota

Clone `main`. This gives you the validator code. Step 3 sets the production
backend URL in the config.

```bash
sudo mkdir -p /opt/bitsota
sudo chown "$USER:$USER" /opt/bitsota

git clone --branch main https://github.com/AlveusLabs/SN94-BitSota.git /opt/bitsota/SN94-BitSota
cd /opt/bitsota/SN94-BitSota

python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

If the repo already exists on the host, update it:

```bash
cd /opt/bitsota/SN94-BitSota
git checkout main
git pull --ff-only origin main
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```


## 3. Configure Replay Validation

This host needs a Bittensor wallet whose hotkey is registered on SN94 and
allowlisted by the autoresearch backend. If the validator already has a wallet,
use its existing wallet name and hotkey name in the config below. If it does
not, create/import the wallet with `btcli` first and register the hotkey on
netuid `94` using network `finney`. Do not put mnemonics in tracked config
files.

Create the replay validator config:

```bash
cd /opt/bitsota/SN94-BitSota
cp research_validator_config.yaml.example research_validator_config.yaml
```

Edit `research_validator_config.yaml` so it contains these production settings.
Replace `validator_wallet` and `validator_hotkey` with the local wallet names on
this host:

```yaml
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
```

Run one validation cycle to confirm the wallet, backend allowlist, Docker, and
CUDA setup are correct:

```bash
cd /opt/bitsota/SN94-BitSota
source .venv/bin/activate
python -m validator.research_validator_runner --config research_validator_config.yaml --once --no-dry-run
```

If this fails with an allowlist or auth error, the backend operator needs to add
the validator hotkey to the backend validator allowlist.

## 4. Configure Chain Weight Setting

Create a weight-setter config:

```bash
cd /opt/bitsota/SN94-BitSota
cp validator_config.yaml.example validator_config.weights.yaml
```

Edit `validator_config.weights.yaml` so it uses the same validator wallet and
SN94 production chain:

```yaml
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
```

Run one weight update:

```bash
cd /opt/bitsota/SN94-BitSota
source .venv/bin/activate
python scripts/autoresearch_weight_setter.py \
  --config validator_config.weights.yaml \
  --coordinator-url https://autoresearch.bitsota.com
```

The backend policy should include the production contract-hotkey target:

```text
5F7MJ2fAyxBG7ci4xP7kQPJanoMdNurk1QBP1AQuFT2Jmzg2
```

Do not run any other process that also calls `set_weights` for the same
validator hotkey.

## 5. Check The Pool/Merkle Contract

This is the validator-facing Pool/Merkle contract monitor. It checks that Pool
is publishing normally, the claim API is visible, and the on-chain contract is
not locked by a veto. It does not recompute rewards or submit vetoes yet.

Run the SN94-side Pool/Merkle check:

```bash
cd /opt/bitsota/SN94-BitSota
source .venv/bin/activate
bitsota-pool-contract-verifier --pool-url https://pool.bitsota.com
```

That command checks:

- Pool `/status` is healthy;
- on-chain runtime is enabled;
- contract reads do not return `read_error`;
- veto is not active unless `--allow-active-veto` is passed;
- the Pool publisher process is running;
- `/claims/epochs` is readable.

Use JSON output if you want machine-readable health evidence:

```bash
bitsota-pool-contract-verifier --pool-url https://pool.bitsota.com --json
```

Use `--require-claimable-epoch` only after Pool has published a non-empty
Merkle root. It is normal for claim epochs to be empty during the first daily
window or when no rewards are publishable yet.

Reward recomputation and veto submission should also live in SN94 before
external vetoer operators are asked to run that part.

## 6. Keep The Validator Running

On Ubuntu, `systemd` is the standard background-process manager. A `systemd
unit` is just a config file that says: start this command on boot, restart it if
it crashes, and let me inspect logs with `journalctl`.

Install a background service for replay validation:

```bash
sudo tee /etc/systemd/system/bitsota-replay-validator.service >/dev/null <<'EOF'
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
EOF
```

Install a background service for backend-directed chain weights:

```bash
sudo tee /etc/systemd/system/bitsota-backend-weights.service >/dev/null <<'EOF'
[Unit]
Description=BitSota backend-policy validator weights
After=network-online.target
Wants=network-online.target

[Service]
WorkingDirectory=/opt/bitsota/SN94-BitSota
Environment=PYTHONUNBUFFERED=1
ExecStart=/opt/bitsota/SN94-BitSota/.venv/bin/python -m validator.backend_weight_setter --config /opt/bitsota/SN94-BitSota/validator_config.weights.yaml --coordinator-url https://autoresearch.bitsota.com --loop --interval-seconds 300
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
```

Install a background service for the Pool/Merkle contract monitor. This runs the
same health check every five minutes and restarts it if the process exits:

```bash
sudo tee /etc/systemd/system/bitsota-contract-verifier.service >/dev/null <<'EOF'
[Unit]
Description=BitSota Pool/Merkle contract monitor
After=network-online.target
Wants=network-online.target

[Service]
WorkingDirectory=/opt/bitsota/SN94-BitSota
Environment=PYTHONUNBUFFERED=1
ExecStart=/opt/bitsota/SN94-BitSota/.venv/bin/python -m validator.pool_contract_verifier --pool-url https://pool.bitsota.com --loop --interval-seconds 300
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
```

Start the services and enable them after reboot:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now bitsota-replay-validator.service
sudo systemctl enable --now bitsota-backend-weights.service
sudo systemctl enable --now bitsota-contract-verifier.service
```

Check that the services are running:

```bash
systemctl status bitsota-replay-validator.service --no-pager
systemctl status bitsota-backend-weights.service --no-pager
systemctl status bitsota-contract-verifier.service --no-pager
```

Watch live logs:

```bash
journalctl -u bitsota-replay-validator.service -f
journalctl -u bitsota-backend-weights.service -f
journalctl -u bitsota-contract-verifier.service -f
```

## What The Runner Does With Heldout Data

The backend sends validator-only heldout dataset instructions in the signed
worklist response. The validator host fetches those Hugging Face rows before
Docker starts, writes `.autoresearch-heldout/manifest.json` into the replay
workspace, and passes only local manifest paths into the benchmark container.

The benchmark container should not need internet access. Setup and benchmark
networking default to `none`.

## Patch-Surface Enforcement

Before replay, the validator rejects:

- submitted patch paths outside task `allowed_patch_paths`;
- generated Python bytecode/cache paths;
- patches larger than `max_patch_bytes` when provided by the backend or task.

The default patch cap is `262144` bytes.
