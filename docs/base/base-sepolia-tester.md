# Base Sepolia Tester

This page is for a nontechnical tester using Base Sepolia. A tester can either
bring their own Base Sepolia wallet and bind a Bittensor snapshot coldkey, or
use an operator-provided fresh emission wallet when the handoff says to do that.
It is not a Base mainnet claim page and it does not move production TAO,
production Bittensor assets, or production SOTA.

Use the generated tester handoff for the current wallet address, root IDs, claim
amounts, and links. The handoff is refreshed from live artifacts and says
whether real holder testing is deferred, whether a fresh emission wallet is
ready, and which transaction hashes the tester should send back.

## Before You Start

Confirm these are all true:

1. The operator gave you access to a fresh Base Sepolia emission test wallet
   out of band, or you are using your own Base Sepolia wallet plus your own
   Bittensor snapshot coldkey.
2. The wallet has Base Sepolia test ETH for gas.
3. The generated handoff says `Base Sepolia claim test ready: true`.
4. If the handoff says `Fresh Emission Tester Prep`, the selected MetaMask
   account exactly matches that reward wallet. If the handoff says real holder
   testing is deferred, use the Base wallet you want to receive SOTA for your
   future real-holder test.

Do not use Base mainnet. If you are testing the real holder path, you may sign
the binding with the matching snapshot coldkey, but the site must never receive
your seed phrase, mnemonic, private key, or production TAO transfer.

## Pick The Right Path

### Fresh Emission Tester

Use this path when the handoff shows `Fresh Emission Tester Prep` with status
`green`.

1. Import or select the operator-provided fresh emission wallet in MetaMask.
2. Open the Base Sepolia claims URL from the handoff.
3. Connect MetaMask and confirm the connected account matches the listed reward
   wallet.
4. Open the emission/mining claim view.
5. Load the mined emission claim.
6. Submit the claim in MetaMask and wait for confirmation.
7. Send the operator the emission transaction hash and a final SOTA balance
   screenshot.

This path does not require a Bittensor coldkey. It proves the current
self-validation result can become a Merkle claim and a MetaMask transaction.

### Real Holder Genesis

Use this path later when you are testing a real snapshot holder claim with your
own coldkey and Base Sepolia wallet.

1. Open the Base Sepolia claims URL from the handoff.
2. Select `Genesis`.
3. Enter the snapshot coldkey and the Base reward wallet.
4. Click `Create binding payload`.
5. Sign the payload with the matching Bittensor coldkey by using the browser
   coldkey extension or by pasting a signature produced by the local helper.
6. Click `Submit binding`.
7. Wait for the genesis batch publisher to include the accepted binding.
8. Refresh the Genesis tab, load the genesis claim, and submit it in MetaMask.

The page must never ask for a seed phrase or private key. The binding only
proves coldkey control and chooses the Base wallet that will receive SOTA.

## If The Handoff Says Not Ready

Stop before opening MetaMask if the handoff says `Base Sepolia claim test
ready: false`.

The common blocker is genesis binding. That means a snapshot holder has not yet
proved ownership of the legacy Bittensor coldkey and bound it to the Base wallet
that will receive SOTA. A normal MetaMask tester cannot fix that by trying the
claim button.

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
3. Confirm the page says Base Sepolia and the connected account is the expected
   wallet for your path.
4. For a fresh emission test, load the mined emission claim and submit it in
   MetaMask.
5. For a real holder test, complete the genesis binding, wait for inclusion,
   then load and submit the genesis claim in MetaMask.
6. Copy every transaction hash MetaMask shows for the path you tested.
7. Send the operator the transaction hash or hashes, connected wallet address,
   final SOTA balance screenshot, and any error text.

The operator runs the evidence verifier. The tester does not need to run shell
commands.

## What Success Looks Like

- The transaction for the selected path confirms on Base Sepolia.
- The SOTA balance shown by the claim page increases after the claims.
- The operator can verify both transaction hashes against the current root
  cycle when both paths were tested, or the emission hash against the fresh
  emission root for the current no-real-key path.

## What To Send Back

| Field | Required? | Notes |
| --- | --- | --- |
| Connected wallet address | Yes | Copy from MetaMask or the claim page. |
| Genesis claim transaction hash | Only for real-holder genesis tests | Base Sepolia transaction hash only. |
| Emission claim transaction hash | Only for fresh emission or mining tests | Base Sepolia transaction hash only. |
| Final SOTA balance screenshot | Yes | Hide unrelated wallet details. |
| Error text or screenshot | If anything fails | Never show seed phrases, private keys, mnemonics, or QR-code secret screens. |

## Troubleshooting

- Wrong network: switch to Base Sepolia, chain ID `84532`.
- Wrong account: select the exact fresh emission reward wallet from the
  handoff, or your own Base wallet for a real-holder test.
- No gas: ask the operator to top up the fresh test wallet with Base Sepolia
  test ETH, or faucet your own Base Sepolia wallet.
- Already claimed: ask the operator whether the current root cycle was already
  used by another tester.
- Claim button blocked: send the displayed status text and screenshot to the
  operator.
