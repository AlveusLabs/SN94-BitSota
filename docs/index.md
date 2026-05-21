# BitSota documentation

BitSota is a decentralized research subnet on Bittensor. This documentation is organized around:

- Roles: miner, validator, pool operator
- Components: GUI, sidecar, autoresearch backend, public validator runner, pool, Merkle claim contract
- Workflows: research-agent mining, backend validation, backend-directed weights, Pool/Merkle claims, local end-to-end testing

## Start here

- [Getting Started](getting-started.md)
- [SN94 System Structure](sn94-system-structure.md)
- [Local Testing](local-testing.md)
- [Configuration Reference](configuration.md)
- [Research-Agent Mining](research-agent-mining.md)

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
    Backend[Autoresearch backend]
    Validator[Public validator runner]
    WeightSetter[Backend weight setter]
    PoolAPI[Pool API]
    Contract[Merkle claim contract]
    Chain[Bittensor chain]
  end

  GUI -->|task discovery and submission| Backend
  Miner -->|signed submission| Backend
  Backend -->|signed worklist| Validator
  Validator -->|observed metrics| Backend
  Backend -->|reward snapshot| PoolAPI
  PoolAPI -->|Merkle root| Contract
  Backend -->|validator_weights| WeightSetter
  WeightSetter -->|set_weights| Chain
```

Default local ports used by the docs and scripts:
- Sidecar: `8123`
- Pool API: `8434`
- Pool monitor: `9000`
