# Public Autoresearch Validator Runner

This is the operator runbook for a public SN94 validator. A working production
validator host runs two SN94 processes plus a Pool/Merkle contract check:

1. Replay validator: asks the autoresearch backend for submissions, replays them
   in Docker/CUDA, and posts observed metrics back to the backend.
2. Backend weight setter: reads the backend reward policy and submits Bittensor
   `set_weights` from the validator hotkey.
3. Contract monitor/verifier: checks that Pool/Merkle publication and the
   on-chain contract state are healthy. Vetoer operators can also run the
   challenge-capable Pool verifier with their own key.

Production validators must not run the legacy relay/local validator weight
path. Relay-based SOTA voting and local winner weight setting have been removed
from the public validator path; production weights come only from the
autoresearch backend reward snapshot.

If the service boundaries are unfamiliar, read
[SN94 System Structure](sn94-system-structure.md) first.

## Production Service Model

The normal production state is `systemd`-managed, not an interactive shell. The
foreground commands in steps 4 and 5 are smoke checks to prove the wallet,
backend allowlist, Docker replay, and backend weight policy are correct before
the services are enabled.

Required services on every production validator host:

- `bitsota-replay-validator.service` runs
  `validator.research_validator_runner`.
- `bitsota-backend-weights.service` runs
  `validator.backend_weight_setter`.

Optional service, only for operators assigned Pool/Merkle verifier inputs:

- `bitsota-contract-verifier.service` runs the Pool verifier.

Create and enable these units in [Keep The Validator Running](#keep-the-validator-running)
after the foreground checks pass. Do not leave the replay validator or backend
weight setter running manually in a terminal.

## Production Values

The install steps below use these folders:

```text
/home/validator/
  Dedicated Linux account for the validator processes and Bittensor wallet.

/opt/bitsota/SN94-BitSota/
  SN94 validator code, replay config, and weight-setter config.
  Put research_validator_config.yaml and validator_config.weights.yaml here.

/opt/bitsota/Pool/
  Pool verifier code and Merkle contract metadata.
  Put/read new_merkle/app/assets/merklepool.json here.
  The validator operator needs GitHub access to AlveusLabs/Pool before this
  clone step works.

/etc/bitsota/
  Local machine secrets and env files.
  Put pool-contract-verifier.env here.

/etc/systemd/system/
  Ubuntu background service files.
  Put bitsota-replay-validator.service, bitsota-backend-weights.service,
  and optional bitsota-contract-verifier.service here.
```

Use these exact values when the later steps ask you to edit config files or run
commands.

In `/opt/bitsota/SN94-BitSota/research_validator_config.yaml`:

```yaml
coordinator_url: "https://autoresearch.bitsota.com"
```

In `/opt/bitsota/SN94-BitSota/validator_config.weights.yaml`:

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

When registering the validator hotkey with `btcli`:

```bash
--netuid 94 --network finney
```

The backend weight setter defaults to `https://autoresearch.bitsota.com`.
Testing uses `https://autoresearch-test.bitsota.com`, but production validators
should not point weight setting at test or relay endpoints.

In `/etc/bitsota/pool-contract-verifier.env`, if this validator also runs the
Pool/Merkle contract verifier:

```env
ONCHAIN_WS_URL=wss://entrypoint-finney.opentensor.ai:443
ONCHAIN_CONTRACT=5CUo48Vuwidb4pTogCCqAeYyMRUwNieTjeEL8FyYvwmQ9XA5
ONCHAIN_STAKE_CONTRACT_HOTKEY=5F7MJ2fAyxBG7ci4xP7kQPJanoMdNurk1QBP1AQuFT2Jmzg2
ONCHAIN_STAKE_NETUID=94
ONCHAIN_VETO_EPOCH=first-published
AUTORESEARCH_REWARD_SNAPSHOT_URL=https://autoresearch.bitsota.com/api/v1/reward-snapshot
```

When checking the live production Pool/Merkle service in step 6:

```bash
POOL_STATUS_URL="https://pool.bitsota.com/health"
POOL_CLAIMS_URL="https://pool.bitsota.com/claims"
```

Testing uses `https://autoresearch-test.bitsota.com`, but production validators
should use `https://autoresearch.bitsota.com`.

## Host Requirements

Use a dedicated Ubuntu GPU host. CPU-only instances such as `t3.*` are not
production replay validators; replay validation needs Docker with CUDA GPU
access.

Known-good production baseline:

- OS: Ubuntu 22.04 LTS or newer.
- GPU: one NVIDIA CUDA-capable GPU with at least 24 GB VRAM.
- Tested EC2 class: `g5.2xlarge` with one NVIDIA A10G, 24 GB VRAM, 8 vCPU,
  and 32 GiB system RAM.
- CUDA: NVIDIA driver must run CUDA 12.4 containers. Both `nvidia-smi` and
  `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`
  must pass before validator setup continues.
- RAM: 32 GiB for the current replay sandbox config.
- Disk: use at least 150 GB of fast SSD/EBS storage; 200 GB is recommended.
  Keep at least 50 GB free for Docker images, Python wheels, cloned replay
  repos, and temporary replay workspaces.

The default replay config gives each replay container a `32g` memory limit and
a 16 GiB workspace cap. Larger future tasks may require raising the host class
and the replay limits together.

## 1. Prepare The Host

Use an Ubuntu GPU host with NVIDIA drivers, Docker, and NVIDIA Container
Toolkit. The validator replays untrusted miner work inside Docker with CUDA GPU
access.

Install baseline packages and create the dedicated `validator` Linux user:

```bash
sudo apt update
sudo apt install -y git curl ca-certificates python3.11 python3.11-venv python3.11-dev python3-pip build-essential

if ! command -v docker >/dev/null 2>&1; then
  sudo apt install -y docker.io || \
    sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
fi
sudo systemctl enable --now docker

sudo useradd -m -s /bin/bash validator 2>/dev/null || true
sudo usermod -aG docker validator
sudo install -d -m 0750 -o validator -g validator /opt/bitsota
sudo install -d -m 0700 -o validator -g validator /srv/bitsota/public-validator-workspaces
```

Install NVIDIA Container Toolkit using NVIDIA's official Ubuntu instructions
for your Ubuntu version, then restart Docker. Run the validator install and
foreground smoke commands as the `validator` user so the wallet path, repo
ownership, and systemd service user all match:

```bash
sudo -iu validator
```

These two commands must both work before continuing:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

## 2. Install SN94-BitSota

Clone `main`. This gives you the validator code. Step 4 sets the production
backend URL in the config.

```bash
git clone --branch main https://github.com/AlveusLabs/SN94-BitSota.git /opt/bitsota/SN94-BitSota
cd /opt/bitsota/SN94-BitSota

python3.11 -m venv .venv
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

## 3. Prepare The Validator Wallet

Use a Bittensor wallet whose hotkey is registered on SN94. The backend must
allowlist this validator hotkey before signed validator scans return work.

Create a new wallet and hotkey:

```bash
btcli wallet new_coldkey --wallet.name validator_wallet
btcli wallet new_hotkey --wallet.name validator_wallet --wallet.hotkey validator_hotkey
btcli subnet register --netuid 94 --wallet.name validator_wallet --wallet.hotkey validator_hotkey --network finney
```

Existing validators can restore/import their real validator wallet instead. Do
not put mnemonics in tracked config files.

## 4. Configure Replay Validation

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

After this passes, the replay validator should run as
`bitsota-replay-validator.service`; the unit file is created in
[Keep The Validator Running](#keep-the-validator-running). Do not leave the
replay validator running manually in a terminal.

## 5. Configure Chain Weight Setting

Create a backend-only weight-setter config. Replace `validator_wallet` and
`validator_hotkey` with the local wallet names on this host:

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

Confirm the backend policy resolves to the production contract hotkey before
submitting any chain transaction:

```bash
cd /opt/bitsota/SN94-BitSota
source .venv/bin/activate
python -m validator.backend_weight_setter \
  --config validator_config.weights.yaml \
  --dry-run \
  --ignore-rate-limit
```

The dry run must show the production contract-hotkey target:

```text
5F7MJ2fAyxBG7ci4xP7kQPJanoMdNurk1QBP1AQuFT2Jmzg2
```

It should also show the current on-chain UID for that hotkey. Do not hardcode
the UID permanently; UIDs can change when registration state changes.

Optional: run one live weight update only after the dry-run target is correct:

```bash
python -m validator.backend_weight_setter \
  --config validator_config.weights.yaml
```

Do not run any other process that also calls `set_weights` for the same
validator hotkey. In particular, stop any old relay/local validator services
before starting backend-directed weights:

```bash
sudo systemctl disable --now bitsota-validator.service 2>/dev/null || true
sudo systemctl disable --now bitsota-capacitorless-weights.service 2>/dev/null || true
sudo systemctl disable --now bitsota-local-weights.service 2>/dev/null || true
pgrep -af 'validator_node|local_validator|capacitorless|relay_client|set_weights' || true
```

After the dry run is correct and old weight setters are stopped, the backend
weight setter should run as `bitsota-backend-weights.service`; the unit file is
created in [Keep The Validator Running](#keep-the-validator-running). Do not
leave the weight setter running manually in a terminal.

## 6. Check The Pool/Merkle Contract

This is the contract-side check. It lives in the `Pool` repo because Pool owns
Merkle publication, proof serving, and contract challenge logic.

First check the live production Pool/contract state:

```bash
POOL_STATUS_URL="https://pool.bitsota.com/health"
POOL_CLAIMS_URL="https://pool.bitsota.com/claims"

curl -fsS "$POOL_STATUS_URL" | python3 -m json.tool
curl -fsS "$POOL_CLAIMS_URL/epochs" | python3 -m json.tool
```

In the status output:

- `onchain_runtime.enabled` should be `true`;
- `onchain_runtime.contract_status.read_error` should be `null`;
- `onchain_runtime.contract_status.is_veto_active` should normally be `false`;
- `onchain_runtime.processes` should show the Pool publisher running;
- `/claims/epochs` should list a claimable epoch after Pool publishes a
  non-empty Merkle root.

For challenge-capable verification, the validator also needs the Pool verifier.
Run this only if the operator has given you the required private inputs:

- GitHub access to `AlveusLabs/Pool`;
- a verifier/vetoer SURI whose SS58 address is allowlisted in the Merkle
  contract;
- a read-only Pool `DATABASE_URL` or an approved local replica of the reward
  input database;
- access to the same epoch artifact directory/feed that the Pool publisher uses.

Install the Pool verifier code:

```bash
git clone --branch production https://github.com/AlveusLabs/Pool.git /opt/bitsota/Pool
cd /opt/bitsota/Pool

python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Create the verifier secret file. Fill in only values the operator assigned to
this validator:

```bash
sudo install -d -m 0750 /etc/bitsota
sudo tee /etc/bitsota/pool-contract-verifier.env >/dev/null <<'EOF'
DATABASE_URL=postgresql://READ_ONLY_POOL_DB_URL
ONCHAIN_WS_URL=wss://entrypoint-finney.opentensor.ai:443
ONCHAIN_CONTRACT=5CUo48Vuwidb4pTogCCqAeYyMRUwNieTjeEL8FyYvwmQ9XA5
ONCHAIN_SURI=REPLACE_WITH_VALIDATOR_VETOER_SURI
ONCHAIN_METADATA=/opt/bitsota/Pool/new_merkle/app/assets/merklepool.json
ONCHAIN_STAKE_CONTRACT_HOTKEY=5F7MJ2fAyxBG7ci4xP7kQPJanoMdNurk1QBP1AQuFT2Jmzg2
ONCHAIN_STAKE_NETUID=94
ONCHAIN_VETO_EPOCH=first-published
AUTORESEARCH_REWARD_SNAPSHOT_URL=https://autoresearch.bitsota.com/api/v1/reward-snapshot
POOL_COMPETITION_WEIGHT=1.0
STAKE_GAIN_SOURCE=contract_reserve
VERIFY_BOOTSTRAP_MODE=history_then_latest_non_vetoed
EOF
sudo chmod 0600 /etc/bitsota/pool-contract-verifier.env
```

Run one foreground verifier pass:

```bash
cd /opt/bitsota/Pool
source .venv/bin/activate
set -a
source /etc/bitsota/pool-contract-verifier.env
set +a

python -u scripts/consensus_daemon.py \
  --mode verify \
  --node-id contract-verifier \
  --out-dir /srv/bitsota/pool-epochs \
  --poll-s 60 \
  --verify-bootstrap-mode history_then_latest_non_vetoed \
  --onchain-veto-epoch first-published
```

If `/srv/bitsota/pool-epochs` is empty and the operator has not provided an
epoch artifact sync/feed, the verifier has nothing to compare yet. Do not treat
an idle verifier as proof that the contract is checked.

`ONCHAIN_VETO_EPOCH=first-published` means the verifier recomputes the real
published epoch, but if it finds a blocking payout mismatch it vetoes the first
successful contract epoch as the global lock epoch. If no successful contract
epoch has been published yet, the verifier waits instead of sending an invalid
veto.

## 7. Keep The Validator Running

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

Install a background service for backend-directed chain weights:

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

If this validator is also running the challenge-capable Pool verifier, install
its background service:

```bash
sudo tee /etc/systemd/system/bitsota-contract-verifier.service >/dev/null <<'EOF'
[Unit]
Description=BitSota Pool/Merkle contract verifier
After=network-online.target
Wants=network-online.target

[Service]
User=validator
Group=validator
WorkingDirectory=/opt/bitsota/Pool
Environment=HOME=/home/validator
Environment=PYTHONUNBUFFERED=1
EnvironmentFile=/etc/bitsota/pool-contract-verifier.env
ExecStart=/opt/bitsota/Pool/.venv/bin/python -u scripts/consensus_daemon.py --mode verify --node-id contract-verifier --out-dir /srv/bitsota/pool-epochs --poll-s 60 --verify-bootstrap-mode history_then_latest_non_vetoed --onchain-veto-epoch first-published
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
# Only run this one after the Pool verifier prerequisites in step 6 are filled.
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
