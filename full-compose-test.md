# Full local loop with Docker Compose relay + validators + local GUI miners

This setup runs:
- Relay (FastAPI) in Docker, in `--test` mode (built from a separate `BitSota` checkout)
- Local validators in Docker (`validator_1`, `validator_2`)
- GUI miners locally on your host (each spawns its own sidecar + local miner process)

## 0) Repo layout (required)

`SN94-BitSota` does not contain relay source. Keep both repos side-by-side:

```bash
mkdir -p ~/bitsota-dev
cd ~/bitsota-dev
git clone https://github.com/AlveusLabs/SN94-BitSota.git current-sn-2
git clone https://github.com/AlveusLabs/BitSota.git BitSota
git clone https://github.com/mekaneeky/automl_zero_cpp.git automl_zero_cpp
cd current-sn-2
test -f ../automl_zero_cpp/automl_zero/tools/baseline_sidecar_bridge.py
```

## 1) Prereqs

- Docker + Docker Compose working: `docker ps` and `docker compose version`
- Host Python env for GUI miners (see `docs/local-testing.md`)
- Validator wallet hotkeys on host at `${HOME}/.bittensor/wallets`
- If you want the C++ backend path in the GUI, export:

```bash
export BITSOTA_MINER_BACKEND=cpp
export AUTOML_ZERO_CPP_ROOT=~/bitsota-dev/automl_zero_cpp/automl_zero
```

## 2) Configure relay + validator compose env

```bash
cp .env.relay-validators.example .env.relay-validators
```

Set or confirm these in `.env.relay-validators`:

```bash
RELAY_URL=http://relay:8002
RELAY_SOURCE_DIR=../BitSota
VALIDATOR_1_WALLET_NAME=<wallet_name>
VALIDATOR_1_WALLET_HOTKEY=<hotkey_name>
VALIDATOR_2_WALLET_NAME=<wallet_name>
VALIDATOR_2_WALLET_HOTKEY=<hotkey_name>
NETUID=402
```

Sanity check relay source path:

```bash
test -f ../BitSota/relay/main.py
```

## 3) Start relay + validators

From `current-sn-2` repo root:

```bash
docker compose --env-file .env.relay-validators -f docker-compose.testnet-relay-validators.yaml --profile local-relay up -d --build relay validator_1 validator_2
docker compose --env-file .env.relay-validators -f docker-compose.testnet-relay-validators.yaml ps
```

Quick relay checks from host:

```bash
curl http://127.0.0.1:8002/health
curl http://127.0.0.1:8002/sota_threshold
curl "http://127.0.0.1:8002/sota-events?page=1&page_size=10"
```

## 4) Configure GUI to point at local relay

Create or update `gui_config.json` in `current-sn-2` root:

```json
{
  "relay_endpoint": "http://127.0.0.1:8002",
  "update_manifest_url": "http://127.0.0.1:8002/version.json",
  "test_mode": true,
  "test_invite_code": "TESTTEST1",
  "miner_validate_every_n_generations": 1000,
  "problem_config_path": "./problem_config.json"
}
```

Then ensure you have a problem config:

```bash
cp -n problem_config.json.example problem_config.json
```

## 5) Run 3 local GUI miners in test mode

Use a different sidecar port per GUI instance.

Terminal 1:

```bash
export BITSOTA_SIDECAR_PORT=8123
python3 -m gui
```

Terminal 2:

```bash
export BITSOTA_SIDECAR_PORT=8124
python3 -m gui
```

Terminal 3:

```bash
export BITSOTA_SIDECAR_PORT=8125
python3 -m gui
```

In each GUI window:
- Select a wallet hotkey for that miner
- For direct mining, stay on `Direct Mining` and click `Start Mining`.
- For pool lease testing, open `Pool Mining`, click `Join Pool` on `Lease Pool`, then click `Start Mining`.

## 6) Logs and monitoring

Relay logs:

```bash
docker compose --env-file .env.relay-validators -f docker-compose.testnet-relay-validators.yaml logs -f relay
```

Validator logs:

```bash
docker compose --env-file .env.relay-validators -f docker-compose.testnet-relay-validators.yaml logs -f validator_1
docker compose --env-file .env.relay-validators -f docker-compose.testnet-relay-validators.yaml logs -f validator_2
```

All logs:

```bash
docker compose --env-file .env.relay-validators -f docker-compose.testnet-relay-validators.yaml logs -f
```

## 7) Stop and cleanup

Stop containers:

```bash
docker compose --env-file .env.relay-validators -f docker-compose.testnet-relay-validators.yaml down
```

Also remove volumes (clears relay DB and validator data):

```bash
docker compose --env-file .env.relay-validators -f docker-compose.testnet-relay-validators.yaml down -v
```

## Troubleshooting

- `unable to prepare context` for relay build: `RELAY_SOURCE_DIR` is wrong or `../BitSota` is missing.
- Validator can't find wallet files: confirm `${HOME}/.bittensor/wallets` contains configured names/hotkeys.
- Multiple GUI miners fail to start: each GUI needs a unique `BITSOTA_SIDECAR_PORT` and `test_mode: true`.
