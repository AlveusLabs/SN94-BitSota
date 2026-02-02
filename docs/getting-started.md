# Getting Started

This repo contains multiple services and roles. Pick the path that matches what you want to do.

## Default ports

- Relay: `http://127.0.0.1:8002`
- Sidecar: `http://127.0.0.1:8123`
- Pool API: `http://127.0.0.1:8434`
- Pool monitor: `http://127.0.0.1:9000`
- Docs website: `http://127.0.0.1:9001`

## I want to run the docs website

```bash
python3 -m venv .venv-docs
source .venv-docs/bin/activate
python3 -m pip install -U pip
python3 -m pip install -r requirements-docs.txt
mkdocs serve -a 127.0.0.1:9001
```

Open `http://127.0.0.1:9001`.

## I want to run a local end-to-end loop

Follow [Local Testing](local-testing.md).

## I want to run just the relay

See [Relay](components/relay.md).

## I want to run just the Pool

See [Pool](components/pool.md).
