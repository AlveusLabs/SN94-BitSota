# Support FAQ

This page is the public-safe support script for Base SOTA local and Base
Sepolia testnet claim questions. It is not a Base mainnet announcement and it
does not say that production SOTA claims are open.

Use this wording when helping a nontechnical user understand claim eligibility,
wallet setup, failed transactions, and what support can or cannot do.

## Safe Short Answer

Base Sepolia is the Base testnet. A Base Sepolia claim uses test ETH, test-only
contracts, and test-only roots. It does not move production TAO, production
Bittensor emissions, Base mainnet assets, or production SOTA.

SOTA claims are non-custodial wallet transactions. The app prepares claim data,
but the user reviews and signs the transaction from their own wallet. Support
can help explain the status and collect evidence, but support cannot claim for a
user, control a user's wallet, promise a payout date, or silently reverse a
confirmed chain transaction.

Never share a seed phrase, private key, mnemonic, wallet file, RPC token, or
admin token with support or with any website. The local demo may print a
deterministic Anvil private key; that key is imported only into a throwaway
MetaMask profile for the local chain and must not be pasted into the claims UI,
docs, support chat, or any public page.

## Claim Types

| Claim type | Plain-language meaning | Support wording |
| --- | --- | --- |
| Genesis claim | A one-time SOTA claim based on the approved legacy snapshot and the user's legacy coldkey binding to a Base wallet. | "Genesis is the starting allocation path. The legacy coldkey proves snapshot ownership, and the Base wallet receives the SOTA claim if the published manifest and root include that user." |
| Emission claim | An ongoing SOTA claim from accepted work after self-validation and root publication. | "Emissions are claim credits from validated competition work. A root must be valid and claimable before the wallet can submit a successful claim." |

The exact claim state comes from the published manifest, root, indexer/API, and
contract state. Do not estimate a user's amount from screenshots or private
spreadsheets.

Synthetic alpha credit is a SOTA accounting credit from the approved snapshot
and manifest rules. It is not a protocol alpha token, not an external ERC-20,
and not an exchange-created instrument.

## Wallet Setup

For Base Sepolia testnet support, say "Base Sepolia testnet" everywhere the
user is asked to switch network or submit a transaction.

| Field | Base Sepolia testnet value |
| --- | --- |
| Network name | Base Sepolia |
| Chain ID | `84532` |
| Native gas token | Test ETH |
| Public RPC | `https://sepolia.base.org` |
| Explorer | `https://sepolia.basescan.org` |

If the wallet shows Base mainnet or chain ID `8453`, stop and switch to Base
Sepolia before sending any testnet claim transaction.

If the public claim URL or API URL has not been announced for Base Sepolia yet,
say: "Base Sepolia testnet claims are not open to public users yet. Please wait
for the published testnet link and do not use mainnet settings for this dry
run."

## Common Questions

### Why does the app ask for my legacy coldkey?

Genesis eligibility comes from the legacy snapshot. The legacy coldkey proves
the user controlled the snapshot account. The Base wallet is where the SOTA
claim is received after the binding is accepted.

Do not ask the user for a seed phrase or private key. A signature challenge or
public address is enough for the claim flow.

### Why am I not eligible?

Check that the user entered the exact legacy coldkey from the snapshot. A
hotkey, a different coldkey, an exchange deposit address, or a wallet that did
not control the snapshot account is not equivalent.

Operator check:

```bash
cd /home/mekaneeky/repos/SN94-BitSota-live-docs
python3 scripts/sota_snapshot_wallet_check.py \
  --snapshot-dir /mnt/4tb/tao_fork_snapshot \
  --address 5UserProvidedAddress
```

`claimable_coldkey` can proceed to binding. `hotkey_with_staked_alpha` means
the user gave a hotkey; ask for the staking coldkey instead.

If a custodian or exchange controlled the snapshot coldkey, that controller is
the party that can mechanically claim. Any downstream customer distribution is
outside SOTA custody unless a later public integration says otherwise.

### Why is my emission claim pending?

Emission claims depend on accepted work, self-validation evidence, and a valid
claim root. If the root is still pending validation, inside a challenge window,
superseded, paused, or invalid, the user should not force the transaction.

### Why did my transaction fail?

First check the network. For this dry run it must be Base Sepolia testnet, chain
ID `84532`. Then check that the wallet has test ETH for gas, the root is
claimable, the account matches the claim destination, and the claim has not
already been spent.

Support should collect the transaction hash and error message before escalating.
Do not promise that a confirmed transaction can be reversed.

### What does already claimed mean?

Emission roots can be cumulative. If the account already claimed the cumulative
amount for the current root lineage, the unclaimed amount can be zero until a
later valid root increases the allocation.

Do not describe the full proof amount as a fresh incremental payout.

### I used the wrong address. What now?

If the transaction has not been submitted, restart the flow with the intended
Base Sepolia wallet address.

If the transaction is already confirmed, collect evidence and escalate. Do not
promise reversal, manual reassignment, or exchange distribution.

## Support Intake Fields

Ask for these fields before escalating a Base Sepolia testnet claim issue:

| Field | When to ask | Notes |
| --- | --- | --- |
| Wallet address | Always | The Base Sepolia wallet address shown in the claim UI or wallet. |
| Legacy coldkey | Genesis claims only, or when eligibility depends on legacy ownership | Public SS58 address only. Never request seed phrases, private keys, or wallet files. |
| Root ID | Claim, proof, already-claimed, pending-root, or amount-display issues | Copy from the UI or API response when visible. |
| Transaction hash | Submitted or failed transaction issues | Base Sepolia explorer hash only for testnet dry runs. |
| Chain ID | Wallet/network issues | Expected Base Sepolia chain ID is `84532`. |
| Error message | Any failed lookup, signature, wallet, API, or transaction path | Ask for exact text, not a summary. |
| Screenshot | UI confusion, proof display, wallet network, or error state | Hide seed phrases, private keys, mnemonics, QR codes, browser extension secret screens, and unrelated personal data. |

Helpful optional fields: claim type, public claim page URL, wallet app and
version, browser, approximate time of the attempt, and whether the account has
Base Sepolia test ETH for gas.

## Escalation Guide

| User report | First support check | Escalate when |
| --- | --- | --- |
| No allocation | Exact legacy coldkey, claim type, root ID, and current lookup result. | The published manifest/root includes the key but the UI or API says no allocation. |
| Failed signature | Correct coldkey controller, unedited challenge text, and wallet error text. | The correct controller cannot sign or repeated failures show the same error. |
| Wrong destination | Compare entered wallet, bound reward address, and pending transaction destination. | A transaction confirmed to an unexpected destination or the UI shows inconsistent addresses. |
| Failed transaction | Base Sepolia chain ID `84532`, test ETH, tx hash, revert/error text, root state, and already-claimed state. | The root is valid and the wallet is funded but the transaction still reverts. |
| Already claimed | Prior transaction hash, claimed amount, root lineage, and current unclaimed amount. | The UI/API disagrees with explorer or contract evidence. |
| Root pending, paused, invalid, or superseded | Root ID and displayed root status. | The UI allows a claim despite a blocked root state, or support needs a user-safe status update. |

Escalation boundaries:

- Product or website issue: claim UI, wallet switching, displayed labels, or
  transaction-builder behavior.
- Indexer/API issue: missing eligibility, proof, claim status, root status, or
  unsigned transaction data.
- Root validation issue: self-validation evidence, claim-list hash, challenge
  status, or root lifecycle.
- Contract/security issue: confirmed transaction behavior, pause/supersession,
  source verification, or custody/role questions.
- Public wording issue: unclear user-facing text, repeated confusion, or any
  question that might imply payout timing, exchange distribution, audit
  completion, or Base mainnet readiness.

## Do Not Say

Do not say or imply:

- a reward amount is guaranteed before the published manifest/root and contract
  state support it;
- a claim date or payout date is guaranteed;
- support can reverse a confirmed transaction;
- an exchange or custodian will distribute SOTA to its customers;
- Base mainnet claims are open from Base Sepolia testnet evidence;
- a contract audit is complete unless an approved public audit record says so;
- external alpha tokens, wrappers, points, or exchange-created instruments are
  protocol alpha tokens.

## Dry-Run Checklist

Before support signs off on Base Sepolia wording, verify that:

1. Every testnet instruction says "Base Sepolia" or "Base Sepolia testnet."
2. The wallet/network instructions show chain ID `84532`.
3. The FAQ explains genesis and emission claims without assuming architecture
   knowledge.
4. The safety warning tells users never to share seed phrases or private keys.
5. The support intake fields include wallet address, legacy coldkey when
   relevant, root ID, transaction hash, chain ID, error message, and screenshots
   with secrets hidden.
6. The copy does not promise reward amounts, claim dates, exchange
   distribution, audit completion, reversals, or Base mainnet readiness.
7. Any public URL points to the public docs or claim site, not internal notes,
   handoff logs, private issue trackers, or agent context files.
