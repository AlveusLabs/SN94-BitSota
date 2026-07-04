# Contracts

The Base SOTA contract set controls the token, claim roots, and claim
distributors. Users do not need to call these contracts directly during the
local demo. The website and indexer prepare the claim transaction for the
wallet.

The Solidity package lives at:

```text
/home/mekaneeky/repos/Pool/contracts/sota-base
```

## Contracts

| Contract | Role |
| --- | --- |
| `SOTAToken` | Capped ERC-20 SOTA token with explicit supply and emission authorities. |
| `SOTAVault` | Holds SOTA and rejects native asset custody. Authorized distributors release SOTA. |
| `SOTARootRegistry` | Publishes active genesis/emission roots, prevents replay, supports supersession. |
| `SOTALaneRegistry` | SOTA-native lane/category registry; stores IDs, budgets, active flags, and policy hashes. |
| `GenesisClaimDistributor` | Verifies fork genesis proofs and releases SOTA to the caller. |
| `EmissionClaimDistributor` | Verifies emission proofs, checks active lane/category budget, and releases SOTA to the caller. |

The contract package is for a SOTA fork on Base, not for Bittensor subnet 94.
Legacy snapshot data may contain Bittensor `netuid` and coldkey fields, but
runtime emissions are SOTA-native and Base-settled.

## Demo Flow

The one-command demo deploys the contract set to a local EVM and prints the
website URL:

```bash
cd /home/mekaneeky/repos/SN94-BitSota-live-docs
./scripts/sota_local_demo.py launch
```

In the demo, claimants do not need contract addresses. They select a seeded
demo user, review the proof, and submit a local claim transaction from the
website.

## Base Sepolia Deploy Path

Use the Pool helper for the contract deployment and write its compact output to
a durable artifact path:

```bash
cd /home/mekaneeky/repos/Pool
export BASE_RPC_URL=https://sepolia.base.org
# Load SOTA_DEPLOYER_PRIVATE_KEY from the approved secret store before running.
python3 scripts/deploy_sota_base.py \
  --rpc-url "$BASE_RPC_URL" \
  --environment base-sepolia \
  --output /secure/artifacts/base-sepolia-compact-deployment.json
```

The helper refuses Base mainnet unless `--allow-mainnet` is explicitly passed.
For this testnet work, do not pass that flag. After deployment, normalize the
compact output into the full review manifest:

```bash
cd /home/mekaneeky/repos/SN94-BitSota-live-docs
python3 scripts/sota_base_testnet_manifest.py \
  --template docs/base/manifests/base-sepolia-deployment-manifest.template.json \
  --deployment /secure/artifacts/base-sepolia-compact-deployment.json \
  --manifest-out /secure/artifacts/base-sepolia-deployment-manifest.json \
  --env-out /secure/artifacts/base-sota.env.testnet \
  --claims-ui-url https://claims-test.bitsota.com \
  --indexer-api-url https://claims-api-test.bitsota.com \
  --root-publisher-url https://root-publisher-test.bitsota.com \
  --attestation-builder-url https://attestation-test.bitsota.com \
  --monitoring-url https://monitoring-test.bitsota.com \
  --autoresearch-api-url https://coordinator-test.bitsota.com \
  --test-wallet-address "$SOTA_TEST_WALLET_ADDRESS"
```

Run the preflight against that generated manifest/env before any
browser-wallet smoke.

## Claim Leaf Formats

Genesis leaves:

```solidity
keccak256(abi.encode("SOTA_GENESIS_CLAIM", account, amount, allocationHash))
```

Emission leaves:

```solidity
keccak256(
    abi.encode(
        "SOTA_EMISSION_CLAIM",
        epoch,
        offchainLaneId,
        account,
        amount,
        rewardHash
    )
)
```

Merkle pairs are sorted before hashing:

```solidity
uint256(left) < uint256(right)
    ? keccak256(abi.encodePacked(left, right))
    : keccak256(abi.encodePacked(right, left))
```

## Launch Notes

Before public launch, operators still need:

- deployed Base/Base Sepolia addresses;
- source verification links;
- constructor args and role assignments;
- owner, publisher, releaser, pause guardian, and multisig/timelock records;
- final deployment manifest committed to the relevant ops repo;
- external audit scope and findings.
