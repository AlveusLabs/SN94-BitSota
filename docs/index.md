# BitSota documentation

BitSota is a decentralized research subnet on Bittensor. This documentation is organized around:

- Roles: miner, validator, pool operator
- Components: GUI, sidecar, relay, pool
- Workflows: direct mining, pool mining, local end-to-end testing

## Start here

- [Getting Started](getting-started.md)
- [Local Testing](local-testing.md)
- [Configuration Reference](configuration.md)

## System overview

```mermaid
flowchart TB
  subgraph Local[Your machine]
    GUI[Desktop GUI]
    Sidecar[Sidecar API]
    Miner[Local miner process]
    GUI -->|HTTP| Sidecar
    Sidecar -->|spawn| Miner
  end

  subgraph Services[Network services]
    Relay[Relay API]
    PoolAPI[Pool API]
    Chain[Bittensor chain]
  end

  GUI -->|direct submit| Relay
  GUI -->|pool tasks| PoolAPI
  Validator[Validator] -->|poll and vote| Relay
  Validator -->|set weights| Chain
  PoolAPI -->|optional relay submit| Relay
```

Default local ports used by the docs and scripts:
- Relay: `8002`
- Sidecar: `8123`
- Pool API: `8434`
- Pool monitor: `9000`
