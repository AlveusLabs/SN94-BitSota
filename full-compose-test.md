# Full local loop with Docker Compose relay + validator + local GUI miners

This setup runs:
- Relay (FastAPI) in Docker, in `--test` mode
- Local validator (polls relay + votes) in Docker
- 3 GUI miners locally on your host (each spawns its own sidecar + local miner process)

## 0) Prereqs

- Docker + Docker Compose working: `docker ps` and `docker compose version`
- Host Python env for the GUI miners (see `docs/local-testing.md` for full setup)
- A validator wallet hotkey on the host (mounted into the validator container)

## 1) Create a validator wallet hotkey on the host

If you don’t already have one:

```bash
btcli wallet new_coldkey --wallet.name local_val
btcli wallet new_hotkey --wallet.name local_val --wallet.hotkey local_val_hot
```

The compose file mounts your host wallets from:
- `${HOME}/.bittensor/wallets` → `/wallets` in the validator container

Defaults used by the compose file:
- `VALIDATOR_WALLET_NAME=local_val`
- `VALIDATOR_WALLET_HOTKEY=local_val_hot`

Override these when starting compose if you want different names:

```bash
export VALIDATOR_WALLET_NAME=local_val
export VALIDATOR_WALLET_HOTKEY=local_val_hot
```

## 2) Start relay + validator with Docker Compose

From the repo root:

```bash
docker compose -f docker-compose.full-test.yaml up -d --build
docker compose -f docker-compose.full-test.yaml ps
```

Quick relay checks from the host:

```bash
curl http://127.0.0.1:8002/health
curl http://127.0.0.1:8002/sota_threshold
curl "http://127.0.0.1:8002/sota-events?page=1&page_size=10"
```

## 3) Configure the GUI to point at the local relay

Create (or update) `gui_config.json` in the repo root:

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

Important: `test_mode: true` disables the GUI single-instance lock, so you can run 3 GUI miners at once.

## 4) Run 3 local GUI miners in test mode

You must use a different sidecar port per GUI instance.

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
- Select a wallet hotkey for that miner (use different hotkeys if you want them to be distinct miners)
- Click “Start Mining”

## 5) Viewing logs and basic monitoring

### Docker logs

Relay:

```bash
docker compose -f docker-compose.full-test.yaml logs -f relay
```

Validator:

```bash
docker compose -f docker-compose.full-test.yaml logs -f validator
```

Both:

```bash
docker compose -f docker-compose.full-test.yaml logs -f
```

### Validator metrics file

The validator writes JSONL metrics to `/data/local_validator_metrics.log` inside its container.

Tail it:

```bash
docker compose -f docker-compose.full-test.yaml exec validator tail -f /data/local_validator_metrics.log
```

Disable metrics logging by setting:

```bash
export VALIDATOR_METRICS_LOG=""
docker compose -f docker-compose.full-test.yaml up -d --build --force-recreate
```

### GUI logs

Each GUI run writes a debug log under:
- `~/.bitsota/logs/`

## 6) Stopping and cleanup

Stop containers:

```bash
docker compose -f docker-compose.full-test.yaml down
```

Also remove the validator data volume (clears cached datasets and metrics log):

```bash
docker compose -f docker-compose.full-test.yaml down -v
```

## Troubleshooting

- Validator can’t find wallet files: confirm `${HOME}/.bittensor/wallets` exists on the host and contains the wallet name and hotkey you configured.
- Relay is up but validator errors on startup: check `docker compose -f docker-compose.full-test.yaml logs validator` for the exact exception.
- Multiple GUI miners fail to start: ensure each has a unique `BITSOTA_SIDECAR_PORT` and `test_mode: true` is set in the GUI config.
