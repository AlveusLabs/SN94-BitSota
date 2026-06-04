# Getting Started

This repo contains the public client layer for the current BitSota/SN94
autoresearch setup. Pick the path that matches what you want to do.

## Current Production Endpoint

```text
https://autoresearch.bitsota.com
```

## I want to run the docs website

```bash
python3 -m venv .venv-docs
source .venv-docs/bin/activate
python3 -m pip install -U pip
python3 -m pip install -r requirements-docs.txt
mkdocs serve -a 127.0.0.1:9001
```

Open `http://127.0.0.1:9001`.

## I want to understand the current architecture

Start with [Architecture Overview](architecture.md).

## I want to see current competitions

Open [Current Competitions](current-competitions.md).

## I want to mine without an agent

Follow [Mining Without an Agent](mining.md).

## I want to mine with Codex

Follow [Codex-Only Mining](codex-only-mining.md).

## I want to post a problem

Read [Problem Posting Requirements](problem-posting.md).

## I want to run a validator

Read [Validator Guide](validation.md) and
[Public Autoresearch Validator Runner](public-validator-runner.md).
