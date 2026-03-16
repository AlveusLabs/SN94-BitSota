# Testnet Test Recipes

This guide adds Docker recipes for:

- testnet relay + validators
- GUI build packaging

## 0) Fresh clone layout (recommended)

`SN94-BitSota` does not vendor relay source. For local relay mode, clone both repos side-by-side:

```bash
mkdir -p ~/bitsota-dev
cd ~/bitsota-dev
git clone https://github.com/AlveusLabs/SN94-BitSota.git current-sn-2
git clone https://github.com/AlveusLabs/Relay.git Relay
git clone https://github.com/AlveusLabs/Pool.git Pool
# Needed for the C++ sidecar worker flow used by the new GUI and section 7 below.
git clone https://github.com/mekaneeky/automl_zero_cpp.git automl_zero_cpp
cd current-sn-2
git fetch origin
git switch testnet-new-gui-pool
git -C ../Pool fetch origin
git -C ../Pool switch testnet-pool-v1
test -f ../automl_zero_cpp/automl_zero/tools/baseline_sidecar_bridge.py
```

If you only use hosted relay mode, `SN94-BitSota` alone is enough.

## 1) Relay + Validators

Files:

- `docker-compose.testnet-relay-validators.yaml`
- `.env.relay-validators.example`
- `docker/validator-node.Dockerfile`
- `docker/run-validator.sh`
- `docker/relay.Dockerfile`

Hosted relay mode (default):

```bash
cp .env.relay-validators.example .env.relay-validators
# edit wallet names/hotkeys
docker compose --env-file .env.relay-validators -f docker-compose.testnet-relay-validators.yaml up -d --build validator_1 validator_2
```

Important:

- `NETUID` in `.env.relay-validators` must match the subnet where those validator hotkeys are registered (for this test, `NETUID=402`).
- `BURN_HOTKEY` must also be a hotkey registered on the same `NETUID` (you can use one validator hotkey for testing).

Optional local relay profile:

```bash
cp .env.relay-validators.example .env.relay-validators
# set wallet names/hotkeys, relay target, and local relay source path
echo "RELAY_URL=http://relay:8002" >> .env.relay-validators
echo "RELAY_SOURCE_DIR=../Relay" >> .env.relay-validators

# relay source must exist in a separate Relay checkout
test -f ../Relay/relay/main.py

# start relay first (separate command)
docker compose --env-file .env.relay-validators -f docker-compose.testnet-relay-validators.yaml --profile local-relay up -d --build relay

# then start validators
docker compose --env-file .env.relay-validators -f docker-compose.testnet-relay-validators.yaml up -d --build validator_1 validator_2
```

You can still start all three together with one command:

```bash
docker compose --env-file .env.relay-validators -f docker-compose.testnet-relay-validators.yaml --profile local-relay up -d --build
```

`local-relay` builds from `RELAY_SOURCE_DIR` (default `../Relay`).

If relay image build fails with:

```text
unable to prepare context: path "<...>" not found
```

set `RELAY_SOURCE_DIR` to the actual checkout path, or clone beside `current-sn-2`:

```bash
cd ..
git clone https://github.com/AlveusLabs/Relay.git Relay
cd current-sn-2
```

or run hosted relay mode (no local `relay` service build, no `BitSota` checkout required).

Stop:

```bash
docker compose -f docker-compose.testnet-relay-validators.yaml down
```

## 2) GUI Build

Files:

- `docker-compose.gui-build.yaml`
- `docker/gui-build.Dockerfile`

Build:

```bash
docker compose -f docker-compose.gui-build.yaml build
docker compose -f docker-compose.gui-build.yaml run --rm gui_builder
```

Artifacts:

- `dist/`
- `build/`

## 3) Local Subtensor Overrides

If you are running a local subtensor node on the same VM host (example endpoint `ws://127.0.0.1:9944` on host), use container-reachable host IP in `.env.relay-validators`:

```bash
SUBTENSOR_NETWORK=local
SUBTENSOR_CHAIN_ENDPOINT=ws://172.17.0.1:9944
```

Then start validators as usual (hosted relay or local relay profile):

```bash
docker compose --env-file .env.relay-validators -f docker-compose.testnet-relay-validators.yaml up -d --build validator_1 validator_2
```

## 4) Deploy `new_merkle` Contract (for on-chain publish/verify)

This deployment step is separate from validator/relay compose.

Build contract:

```bash
cd ../Pool/new_merkle
cargo contract build --release
cd ..
```

Deploy to testnet:

```bash
export NETUID=402
export CONTRACT_HOTKEY_SS58=5DhX66kX37LcACNbNPTwz93DMP9tbQs6xga3KUwjCPwcVVmX
export PUBLISHER_SURI="<publisher_suri>"
export ONCHAIN_GAS=50000000000
export ONCHAIN_PROOF_SIZE=200000

cargo contract instantiate \
  new_merkle/target/ink/merklepool.contract \
  --constructor new \
  --args "$NETUID" "$CONTRACT_HOTKEY_SS58" \
  --suri "$PUBLISHER_SURI" \
  --url wss://test.finney.opentensor.ai:443 \
  --gas "$ONCHAIN_GAS" \
  --proof-size "$ONCHAIN_PROOF_SIZE" \
  -x -y --skip-dry-run \
  --output-json
```

Deploy to local subtensor:

```bash
export NETUID=94
export CONTRACT_HOTKEY_SS58=<registered_hotkey_ss58>
export ONCHAIN_GAS=50000000000
export ONCHAIN_PROOF_SIZE=200000

cargo contract instantiate \
  new_merkle/target/ink/merklepool.contract \
  --constructor new \
  --args "$NETUID" "$CONTRACT_HOTKEY_SS58" \
  --suri //Alice \
  --url ws://127.0.0.1:9944 \
  --gas "$ONCHAIN_GAS" \
  --proof-size "$ONCHAIN_PROOF_SIZE" \
  -x -y --skip-dry-run \
  --output-json
```

Notes:

- Copy the instantiated contract address from command output into `ONCHAIN_CONTRACT`.
- Deployment itself does not require `register` / `burned_register` calls.

## 5) Start Pool Testnet Stack With Deployed Contract

From `../Pool`:

```bash
cp .env.testnet.example .env.testnet
```

Set at minimum in `.env.testnet`:

```bash
ONCHAIN_CONTRACT=<instantiated_contract_address>
ONCHAIN_PUBLISHER_SURI=<publisher_suri>
ONCHAIN_VERIFIER_1_SURI=<verifier_1_suri>
ONCHAIN_VERIFIER_2_SURI=<verifier_2_suri>
```

If using local subtensor for this stack, also set:

```bash
SUBTENSOR_NETWORK=local
SUBTENSOR_CHAIN_ENDPOINT=ws://172.17.0.1:9944
ONCHAIN_WS_URL=ws://172.17.0.1:9944
SUBMISSION_SUBTENSOR_NETWORK=local
SUBMISSION_SUBTENSOR_CHAIN_ENDPOINT=ws://172.17.0.1:9944
```

Run:

```bash
docker compose --env-file .env.testnet -f docker-compose.testnet.yaml up -d --build
```

## 6) Failure-Mode Dashboard Signals

The monitor now surfaces additional failure-mode checks in `http://127.0.0.1:9000` and `http://127.0.0.1:9000/metrics.json`:

- relay health + recent SOTA event freshness
- lease pipeline health (`issued_15m`, `completed_15m`, zero-eval ratio, overdue leases)
- submission backlog (`verified_unsubmitted_total`, oldest backlog age)
- submission node heartbeat/status file (`/data/submission_node_status.json`)
- synthesized alerts (`alerts.overall`, `alerts.counts`, `alerts.items`)

Optional monitor envs (in `Pool/.env.testnet`):

```bash
MONITOR_RELAY_URL=${RELAY_URL}
MONITOR_RELAY_ADMIN_TOKEN=<optional_admin_token>
MONITOR_SUBMISSION_STATUS_FILE=/data/submission_node_status.json
MONITOR_LOG_FULL_JSON=false
MONITOR_STALE_FINALIZED_S=3600
MONITOR_STALE_RELAY_EVENT_S=3600
MONITOR_SUBMISSION_BACKLOG_WARN_S=1200
```

## 7) C++-Only Lease Flow (Pool + Contract Payout Smoke)

This path uses the lease coordinator and C++ backend only.

It is also the same backend path used by the new GUI when you open `Pool Mining`, click `Join Pool` on `Lease Pool`, and then click `Start Mining` with `BITSOTA_MINER_BACKEND=cpp`.

Set a common workspace root first so every command below uses the same paths:

```bash
export BITSOTA_DEV_ROOT=~/bitsota-dev
```

If Docker is unavailable in your WSL or local dev environment, use this verified direct Pool API path first:

Terminal 0 (Pool API):

```bash
cd "${BITSOTA_DEV_ROOT}/Pool"
git switch testnet-pool-v1

export PYENV_VERSION=automl_pool
export POSTGRES_USER=pooler
export POSTGRES_PASSWORD=test
export POSTGRES_DB=mining_pool
export POSTGRES_HOST=127.0.0.1
export POSTGRES_PORT=5432
export ENVIRONMENT=development
export LEASE_MIN_ITERATIONS=0

python -m uvicorn app.main:app --host 127.0.0.1 --port 8434
```

Quick check:

```bash
curl -sS http://127.0.0.1:8434/health
```

Terminal A (sidecar):

```bash
cd "${BITSOTA_DEV_ROOT}/current-sn-2"
PYENV_VERSION=automl_pool python3 -m sidecar --host 127.0.0.1 --port 8123
```

Terminal B (C++ worker bridge, real mode):

```bash
cd "${BITSOTA_DEV_ROOT}/current-sn-2"
export AUTOML_ZERO_CPP_ROOT="${BITSOTA_DEV_ROOT}/automl_zero_cpp/automl_zero"
PYENV_VERSION=automl_pool python3 -m scripts.miner_cpp_sidecar \
  --cpp-mode lease \
  --mode real \
  --sidecar-url http://127.0.0.1:8123 \
  --run-id pool_smoke \
  --workers 1 \
  --lease-evolve-generations 40 \
  --automl-root "${AUTOML_ZERO_CPP_ROOT}" \
  --bitsota-root "$(pwd)"
```

Terminal C (lease coordinator driver, real pool leases):

```bash
cd "${BITSOTA_DEV_ROOT}/current-sn-2"
PYENV_VERSION=automl_pool BITSOTA_CPP_BACKEND=1 python3 -m scripts.pool_lease_sidecar_driver \
  --pool-url http://127.0.0.1:8434 \
  --sidecar-url http://127.0.0.1:8123 \
  --run-id pool_smoke \
  --duration-s 120
```

Expected signal in Terminal C:

- `submit_lease ok=True ...`
- `Enqueued lease ...`

If you see repeated register failures or the GUI looks idle, check the Pool API terminal first. The most common local issue is running the right Pool branch against the wrong old database schema.

Simple pass/fail check after Terminal C starts submitting work:

```bash
cd "${BITSOTA_DEV_ROOT}/current-sn-2"
PYENV_VERSION=automl_pool python3 -m scripts.check_local_pool_stack \
  --pool-url http://127.0.0.1:8434 \
  --sidecar-url http://127.0.0.1:8123 \
  --require-progress
```

Verify pool published a non-zero payout epoch:

```bash
docker logs --since=30m pool-testnet-pool-v1-consensus_publisher-1 2>&1 | rg "onchain publish epoch="
```

Look for at least one line like:

- `onchain publish epoch=<n> ok root=0x... total_rao=...`

Check claim proof package for a miner hotkey in that epoch:

```bash
curl -sS http://127.0.0.1:8844/epoch/<epoch_number> | jq
curl -sS http://127.0.0.1:8844/epoch/<epoch_number>/claim/<miner_hotkey> | jq
```
