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

Optional local relay profile:

```bash
export RELAY_URL=http://relay:8002
docker compose --env-file .env.relay-validators -f docker-compose.testnet-relay-validators.yaml --profile local-relay up -d --build
```

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
