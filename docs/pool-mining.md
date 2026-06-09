# Pool Claims

The old evolutionary Pool mining guide is archived under
[AutoML-Zero Archive](archive/automl-zero/index.md).

In the current autoresearch setup, Pool is primarily part of reward publication
and claim serving:

1. backend accepts and scores submissions;
2. backend exposes reward data through `/api/v1/reward-snapshot`;
3. Pool builds Merkle claim leaves;
4. Pool publishes a root to the claim contract;
5. miners fetch proofs and submit claims.

For current miner flow, start with:

- [Mining Without an Agent](mining.md)
- [Codex-Only Mining](codex-only-mining.md)

For reward flow, see [Rewards And Claims](rewards.md).

For the miner-facing claim steps, see [Claim Rewards](claim-rewards.md).
