# Miner

There are two paths to run mining code:

- Desktop GUI mode via sidecar
- CLI mode via `neurons/miner.py`
- Experimental coordinator-backed research-agent mode via `bitsota-research-agent`

## CLI entrypoint

The CLI miner reads `miner_config.yaml` and can run in:

- direct mode: submit to the relay
- pool mode: talk to the Pool API

See [Configuration Reference](../configuration.md) for `miner_config.yaml` keys.

## Direct mining

See [Mining](../mining.md).

## Pool mining

See [Pool Mining](../pool-mining.md) and [Pool](pool.md).

## Research-agent mining

This is a separate, additive miner for coordinator-backed research tasks. It uses an OpenAI-compatible API and supports `direct` and `pool` participation styles across `standard`, `centerless`, and `peer_evaluation` task modes.

See [Research-Agent Mining](../research-agent-mining.md).
