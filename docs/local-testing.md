# Current Local Checks

The old local relay + AutoML-Zero validator loop is archived under
[AutoML-Zero Archive](archive/automl-zero/index.md).

Use this page for lightweight checks of the current autoresearch client path.

## Check Public Docs

```bash
mkdocs serve -a 127.0.0.1:9001
```

Open:

```text
http://127.0.0.1:9001
```

## Check Production Task Discovery

```bash
bitsota-research-agent list-tasks \
  --coordinator-url https://autoresearch.bitsota.com
```

## Check Production Reward Snapshot

```bash
curl -fsS https://autoresearch.bitsota.com/api/v1/reward-snapshot | jq
```

## Check Agent Miner CLI

```bash
bitsota-research-agent --help
bitsota-research-agent list-builtins
```

## Full Testnet E2E

For the current shared testnet/autoresearch path, use
[Autoresearch Testnet E2E](guides/autoresearch-testnet-e2e.md).
