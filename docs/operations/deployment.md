# Deployment

This repo is deployed in multiple shapes:

- Desktop GUI for miners
- Relay service
- Pool service
- Validator nodes

This page is a deployment checklist for each component.

## Relay

- Provide a writable `DATABASE_URL` (SQLite for dev, Postgres for production)
- Set `ADMIN_AUTH_TOKEN` for admin endpoints
- Set `RELAY_NETUID` and `RELAY_NETWORK` if you are using metagraph-based validator allowlisting
- Configure logging with `RELAY_LOG_LEVEL` and `RELAY_LOG_FILE`

Recommended production run pattern:

- run behind a reverse proxy
- terminate TLS at the proxy
- set explicit CORS origins
- use Postgres and backups

## Pool

- Provide Postgres connection settings in env
- Set `ENVIRONMENT=production`
- Set `MONITOR_TOKEN` if you expose monitor endpoints publicly

If you run the sim stack locally, use `Pool/docker-compose.sim.yaml`.

## Validator

- Ensure wallet files are present on disk and readable
- Configure `validator_config.yaml` and `validator_hyperparams.json`
- Plan for dataset caching and disk usage

## GUI

- Desktop builds are shipped as binaries; local dev runs use `python3 -m gui`
