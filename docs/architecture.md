# Architecture

This page maps the current autoresearch production path. For operator
responsibilities and key ownership, start with
[SN94 System Structure](sn94-system-structure.md).

## Research Mining Flow

```mermaid
sequenceDiagram
  participant GUI as Desktop GUI
  participant Sidecar as Sidecar API
  participant Miner as Research miner or agent
  participant Backend as Autoresearch backend

  GUI->>Sidecar: Start mining
  Sidecar->>Miner: Spawn miner process
  Miner->>Backend: Claim task and submit patch or artifact
  Backend-->>Miner: Submission status and task updates
  Miner->>Sidecar: Logs, local progress, submission id
  GUI->>Sidecar: Poll logs and state
```

## Validation And Rewards

```mermaid
sequenceDiagram
  participant Backend as Autoresearch backend
  participant Validator as Public validator runner
  participant WeightSetter as Backend weight setter
  participant Chain as Bittensor SN94
  participant Pool as Pool service
  participant Contract as Merkle claim contract
  participant Miner as Miner recipient

  Validator->>Backend: Signed scan request
  Backend-->>Validator: Replay worklist and private heldout handles
  Validator->>Validator: Replay in Docker/CUDA sandbox
  Validator->>Backend: Observed metrics
  Backend->>Pool: Reward/accounting output
  Pool->>Contract: Publish Merkle root
  Miner->>Contract: Claim with Merkle proof
  WeightSetter->>Backend: Read validator_weights
  WeightSetter->>Chain: set_weights
```

## Service Boundaries

```mermaid
flowchart TB
  subgraph Local[Operator machine]
    GUI[Desktop GUI]
    Sidecar[Sidecar API]
    Miner[Research miner or agent]
    Validator[Public validator runner]
    WeightSetter[Backend weight setter]
    GUI --> Sidecar
    Sidecar --> Miner
  end

  subgraph Remote[Network services]
    Backend[Autoresearch backend]
    PoolAPI[Pool API]
    Contract[Merkle claim contract]
    Chain[Bittensor chain]
  end

  Miner --> Backend
  Validator --> Backend
  Backend --> PoolAPI
  PoolAPI --> Contract
  WeightSetter --> Backend
  WeightSetter --> Chain
```
