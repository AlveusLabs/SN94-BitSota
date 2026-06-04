# Pool Claims Service

In the current autoresearch setup, Pool consumes accepted reward data and
publishes claimable Merkle epochs.

Pool does not decide validator replay results. The autoresearch backend records
accepted submissions and exposes the reward snapshot that Pool consumes.

## Current Boundary

| Component | Responsibility |
| --- | --- |
| Autoresearch backend | Accepted submissions, best results, reward snapshot. |
| Pool | Merkle leaves, root publication, claim proof API. |
| Claim contract | On-chain proof verification and claim settlement. |

See [Rewards And Claims](../rewards.md).
