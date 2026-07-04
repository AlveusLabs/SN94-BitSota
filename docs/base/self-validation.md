# Self-Validation

Self-validation is the gate between "a miner submitted work" and "a miner can
claim SOTA."

The local demo runs this path automatically. A seeded miner submits work, the
backend records validation evidence, and only accepted work becomes an emission
claim.

## Submission Gate

For self-validation tasks:

1. A miner submits with an EVM identity and artifact hash.
2. A configured or reproducible committee evaluates the submission.
3. The backend records committee evidence.
4. The emission coordinator includes only accepted submissions that pass the
   quorum gate.

## Claim-Root Accuracy Gate

Before a root is accepted, an independent verifier recomputes:

- claim list hash;
- contract-compatible claim leaves;
- Merkle proofs;
- Merkle root;
- total amount;
- root freshness;
- expected backend root.

If a reward hash, amount, account, leaf, proof, or root is tampered with, the
validator must challenge before the root becomes claimable.

## In The Local Demo

Run:

```bash
cd /home/mekaneeky/repos/SN94-BitSota-live-docs
./scripts/sota_local_demo.py launch
```

Then open the printed website URL and look for the seeded miner emission claim.
That claim should show validation evidence before the wallet can submit the
local claim transaction.

## Important Constraint

Scoring must not punish exceptional true performers just because they are far
from the mean. Validation policy should detect false-score collusion without
discarding frontier discoveries.

The local demo uses a deterministic task and snapshot input, but the validation
idea must stay general. A real competition needs rules that catch false scores
without rejecting genuine frontier improvements.
