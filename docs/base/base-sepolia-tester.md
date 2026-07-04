# Base Sepolia Tester

This page is for a nontechnical tester who has been given a seeded Base
Sepolia test wallet by the operator. It is not a Base mainnet claim page and it
does not move production TAO, production Bittensor assets, or production SOTA.

Use the generated tester handoff for the current wallet address, root IDs, claim
amounts, and links. The handoff is refreshed from live artifacts whenever a
fresh test wallet/root cycle is prepared.

## Before You Start

Confirm these are all true:

1. The operator gave you access to a seeded Base Sepolia test wallet out of
   band.
2. The wallet has Base Sepolia test ETH for gas.
3. The generated handoff says `Base Sepolia claim test ready: true`.
4. The selected MetaMask account exactly matches the test wallet address shown
   in the handoff.

Do not use Base mainnet. Do not use a production Bittensor wallet, production
TAO wallet, real seed phrase, or production private key for this dry run.

## MetaMask Network

| Field | Value |
| --- | --- |
| Network name | Base Sepolia |
| Chain ID | `84532` |
| Native gas token | Test ETH |
| Public RPC | `https://sepolia.base.org` |
| Explorer | `https://sepolia.basescan.org` |

If MetaMask shows Base mainnet or chain ID `8453`, stop and switch to Base
Sepolia before submitting any transaction.

## Test Steps

1. Open the Base Sepolia claims URL from the generated handoff.
2. Connect the seeded MetaMask wallet.
3. Confirm the connected wallet matches the handoff test wallet.
4. Confirm the page says Base Sepolia and shows the expected genesis and
   emission claim amounts.
5. Submit the genesis claim in MetaMask and wait for confirmation.
6. Copy the genesis claim transaction hash.
7. Submit the mined emission claim in MetaMask and wait for confirmation.
8. Copy the emission claim transaction hash.
9. Send the operator the two transaction hashes, connected wallet address,
   final SOTA balance screenshot, and any error text.

The operator runs the evidence verifier. The tester does not need to run shell
commands.

## What Success Looks Like

- The genesis claim transaction confirms on Base Sepolia.
- The emission claim transaction confirms on Base Sepolia.
- The SOTA balance shown by the claim page increases after the claims.
- The operator can verify both transaction hashes against the current root
  cycle.

## What To Send Back

| Field | Required? | Notes |
| --- | --- | --- |
| Connected wallet address | Yes | Copy from MetaMask or the claim page. |
| Genesis claim transaction hash | Yes | Base Sepolia transaction hash only. |
| Emission claim transaction hash | Yes | Base Sepolia transaction hash only. |
| Final SOTA balance screenshot | Yes | Hide unrelated wallet details. |
| Error text or screenshot | If anything fails | Never show seed phrases, private keys, mnemonics, or QR-code secret screens. |

## Troubleshooting

- Wrong network: switch to Base Sepolia, chain ID `84532`.
- Wrong account: select the exact seeded wallet from the handoff.
- No gas: ask the operator to top up the seeded wallet with Base Sepolia test
  ETH.
- Already claimed: ask the operator whether the current root cycle was already
  used by another tester.
- Claim button blocked: send the displayed status text and screenshot to the
  operator.
