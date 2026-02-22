# Testnet Test Recipes

This guide adds Docker recipes for:

- testnet relay + validators
- GUI build packaging

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
echo "RELAY_SOURCE_DIR=../BitSota" >> .env.relay-validators

# start relay first (separate command)
docker compose --env-file .env.relay-validators -f docker-compose.testnet-relay-validators.yaml --profile local-relay up -d --build relay

# then start validators
docker compose --env-file .env.relay-validators -f docker-compose.testnet-relay-validators.yaml up -d --build validator_1 validator_2
```

You can still start all three together with one command:

```bash
docker compose --env-file .env.relay-validators -f docker-compose.testnet-relay-validators.yaml --profile local-relay up -d --build
```

`local-relay` builds from `RELAY_SOURCE_DIR` (default `../BitSota`).

If relay image build fails with:

```text
unable to prepare context: path "<...>" not found
```

set `RELAY_SOURCE_DIR` to the actual checkout path, or clone beside `current-sn-2`:

```bash
cd ..
git clone https://github.com/AlveusLabs/BitSota.git
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
export PUBLISHER_SURI="say sorry flight era model roast income gap ramp aisle health lyrics"
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
