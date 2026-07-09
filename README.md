# BitSota

BitSota is now organized around the Base SOTA fork path: SOTA claims settle on
Base-compatible EVM contracts, genesis allocation comes from the approved
TAO/alpha snapshot formula, and ongoing emissions are released only for accepted
self-validated research work.

The old SN94/Bittensor subnet code is still present where it is needed for
current production or historical reference, but it is not the product direction
for the Base SOTA fork. New work should prefer the Base SOTA docs, local demo,
EVM contracts, autoresearch coordinator flow, and lane-based emission language.

## Start Here

Run the complete local Base SOTA loop:

```bash
cd /home/mekaneeky/repos/SN94-BitSota-live-docs
./scripts/sota_local_demo.py launch
```

That one command starts:

- local Base-compatible EVM node;
- SOTA token, vault, root registry, lane registry, and claim distributor
  contracts;
- local claims indexer;
- autoresearch backend;
- seeded genesis claim;
- seeded miner submission with self-validation evidence;
- claims UI and local docs.

For a multi-miner local swarm against a running stack:

```bash
./scripts/sota_local_demo.py miner-swarm --count 5
```

For a fresh start-to-finish swarm smoke:

```bash
./scripts/sota_local_demo.py swarm-smoke --count 5
```

## Current Model

Base SOTA has two reward paths:

- Genesis: eligible legacy holders claim SOTA from the snapshot. Direct TAO
  converts 1:1, plus synthetic alpha value from the approved pro-rata pool
  formula.
- Ongoing emissions: miners submit research work with an EVM miner identity and
  optional reward address. Accepted self-validation evidence becomes a Merkle
  claim root. Rewards are paid in SOTA.

Base SOTA does not use Bittensor subnet registration, netuids, validator
weights, Yuma emissions, or protocol alpha tokens for ongoing rewards.

## Docs

Rendered docs:

```bash
python3 -m venv .venv-docs
source .venv-docs/bin/activate
python3 -m pip install -U pip
python3 -m pip install -r requirements-docs.txt
mkdocs serve -a 127.0.0.1:9001
```

Useful pages:

- [Base SOTA start](docs/base/index.md)
- [New user guide](docs/base/new-users.md)
- [Bittensor migration guide](docs/base/bittensor-migrants.md)
- [Local E2E demo](docs/base/local-e2e.md)
- [Self-validation](docs/base/self-validation.md)
- [Contracts](docs/base/contracts.md)

## Repo Boundaries

- `scripts/sota_local_demo.py` owns the local Base SOTA tester loop.
- `scripts/sota_base_testnet_*.py` owns Base Sepolia/testnet operator work.
- `miner/` and `neurons/` contain public miner/client helpers.
- `validator/` and `bittensor_network/` remain for current production SN94
  validator paths and should not be expanded for new Base SOTA work.

## Safety

The local demo uses deterministic local keys and chain ID `31337`. It does not
touch production Bittensor, production TAO, Base Sepolia, or Base mainnet.

Do not put real seed phrases, production private keys, or mainnet funds into the
local demo.
