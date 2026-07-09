# Base Sepolia Tester

This page is for a nontechnical tester using Base Sepolia. A tester can bring
their own Base Sepolia wallet and bind a Bittensor snapshot coldkey, or use an
operator-provided seeded wallet when the handoff says to do that. It is not a
Base mainnet claim page and it does not move production TAO, production
Bittensor assets, or production SOTA.

Use the generated tester handoff for the current wallet address, root IDs, claim
amounts, and links. The handoff is refreshed from live artifacts and says
whether real holder testing is deferred or an operator-seeded wallet should be
used.

## Before You Start

Confirm these are all true:

1. The operator gave you access to a seeded Base Sepolia test wallet out of
   band, or you are using your own Base Sepolia wallet plus your own Bittensor
   snapshot coldkey.
2. The wallet has Base Sepolia test ETH for gas.
3. The generated handoff says `Base Sepolia claim test ready: true`.
4. If the handoff lists a specific test wallet, the selected MetaMask account
   exactly matches it. If the handoff says real holder testing is deferred, use
   the Base wallet you want to receive SOTA.

Do not use Base mainnet. If you are testing the real holder path, you may sign
the binding with the matching snapshot coldkey, but the site must never receive
your seed phrase, mnemonic, private key, or production TAO transfer.

## If The Handoff Says Not Ready

Stop before opening MetaMask if the handoff says `Base Sepolia claim test
ready: false`.

The common blocker is genesis binding. That means a snapshot holder has not yet
proved ownership of the legacy Bittensor coldkey and bound it to the Base wallet
that will receive SOTA. A normal MetaMask tester cannot fix that by trying the
claim button.

The snapshot holder should use the Genesis binding panel in the claims UI:

1. Open the Base Sepolia claims URL from the handoff.
2. Select `Genesis`.
3. Enter the snapshot coldkey and the Base reward wallet.
4. Click `Create binding payload`.
5. Sign the payload with the matching Bittensor coldkey by using the browser
   coldkey extension or by pasting a signature produced by the local helper.
6. Click `Submit binding`.
7. Check binding status for the same coldkey and reward wallet.

The page must never ask for a seed phrase or private key. The binding only
proves coldkey control and chooses the Base wallet that will receive SOTA.

After the binding is accepted, the Base Sepolia batch publisher checks for
unbatched accepted bindings about every 10 minutes. When it finds one or more,
it builds one genesis Merkle root for the batch, publishes the root, imports the
claim artifact, and marks those bindings as included. Then the tester can
refresh the Genesis tab and claim.

Mined emission claims use a separate publisher. After autoresearch
self-validation accepts miner work, the emission publisher checks about every
10 minutes for the latest accepted emission root, publishes it if it is not
already indexed, imports the claim artifact, and the tester can refresh the
Emission tab.

If the timer is stopped, the fallback operator action is:

1. Run `scripts/run_sota_base_genesis_batch_publisher_once.sh`.
2. Run `scripts/run_sota_base_emission_batch_publisher_once.sh`.
3. Rerun browser smoke and refresh the handoff.

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
2. Connect the Base Sepolia MetaMask wallet.
3. If you are testing a real holder path, use the Genesis binding panel: enter
   the snapshot coldkey and connected Base reward wallet, create the binding
   payload, sign it with the matching coldkey, and submit it.
4. Wait for the handoff or binding status to show the binding is included in a
   published genesis root.
5. Confirm the page says Base Sepolia and shows the expected genesis and
   emission claim amounts.
6. Submit the genesis claim in MetaMask and wait for confirmation.
7. Copy the genesis claim transaction hash.
8. Submit the mined emission claim in MetaMask and wait for confirmation when
   an emission is available for your wallet.
9. Copy the emission claim transaction hash.
10. Send the operator the transaction hashes, connected wallet address,
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
