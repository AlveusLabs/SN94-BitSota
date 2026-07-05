# Self-Validation

Self-validation is the gate between "a miner submitted work" and "a miner can
claim SOTA."

The local demo runs this path automatically. A seeded miner submits work, the
backend records validation evidence, and only accepted work becomes an emission
claim.

Base SOTA uses a `7,200 SOTA/day` global emission target. For R&D/frontier
tasks, that budget is capacitored: it accrues for the lane and is released only
when peer self-validation confirms the artifact is real and improves the current
frontier. Production policy values are editable; testnet uses a daily epoch and
the default frontier-capacitor policy.

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

The seeded local committee is Bob, Charlie, and Dave. A passing demo shows
`3/3 accepted` for Alice's mined emission before the emission claim is
available. This is local autoresearch evidence, not Bittensor validator
weighting or Yuma consensus.

To test more than the seeded Alice path, run the local multi-miner smoke:

```bash
cd /home/mekaneeky/repos/SN94-BitSota-live-docs
./scripts/sota_local_demo.py swarm-smoke --count 5
```

That command spawns five local miner processes, gives each one a distinct
hotkey plus EVM miner/reward address, records Bob/Charlie/Dave committee
acceptance, publishes the SOTA emission root, submits five local claim
transactions on Anvil, and writes the evidence report at
`/home/mekaneeky/repos/.sota-base-local/miner-swarm/latest.json`.

## Important Constraint

Scoring must not punish exceptional true performers just because they are far
from the mean. Validation policy should detect false-score collusion without
discarding frontier discoveries.

If a miner passes self-validation and a later miner improves the same frontier
before root publication, the accepted miner remains eligible for the emission
root. The evidence preserves both the original accepted gate and the current
frontier status so reviewers can see why the claim stayed eligible.

The local demo uses a deterministic task and snapshot input, but the validation
idea must stay general. A real competition needs rules that catch false scores
without rejecting genuine frontier improvements.
