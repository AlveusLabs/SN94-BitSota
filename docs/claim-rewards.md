# Claim Rewards

This page explains how BitSota reward claiming works from a miner's point of
view.

The short version:

1. Mine with a miner hotkey.
2. Set your payout recipient before Pool publishes rewards.
3. Wait for your submission to be accepted.
4. Wait for Pool to publish a claim epoch.
5. Look up the claim package by miner hotkey.
6. Submit the claim transaction.
7. The reward goes to the published recipient account.

## The Key Words

A key is an account.

It has two parts:

- Public address: safe to share. This is the address that starts with `5...`.
- Secret key, seed phrase, or mnemonic: never share this. It signs messages and
  transactions.

Signing means proving "this key approved this action" without revealing the
secret key.

Bittensor normally talks about two key roles:

- Coldkey: the owner account. It usually holds value and owns hotkeys.
- Hotkey: the worker identity. Miners and validators use hotkeys to do work.

BitSota reward claims are simpler than normal Bittensor ownership rules:

- Miner hotkey: the identity that earned the reward.
- Recipient coldkey: the account that receives the claimed alpha stake.
- Fee signer: the key that sends the claim transaction and pays the chain fee.

These can be different accounts.

Important:

- The miner hotkey is used to find your claim.
- The miner hotkey was fixed earlier when the task claim/submission was signed.
- The published `recipient_coldkey` is where the reward goes.
- The recipient coldkey must be set before Pool publishes that reward epoch.
- The fee signer only pays the transaction fee.
- The fee signer cannot change which miner hotkey earned the reward.
- The fee signer cannot change the recipient coldkey for an already published
  claim.
- The recipient does not need to own the miner hotkey.
- The recipient does not need to be subnet-registered.

## Before You Claim

You need:

- This repo installed with `pip install -e .`.
- The miner hotkey address that earned the reward.
- A local wallet key with enough free balance to pay the transaction fee.
- A claim package published by Pool.

You do not need:

- The Pool publisher key.
- The contract owner key.
- Validator keys.
- AWS or backend access.

## Step 1: List Your Claim Packages

Run this from the SN94 repo:

```bash
python scripts/claim_merkle_rewards.py list \
  --hotkey <MINER_HOTKEY>
```

Replace `<MINER_HOTKEY>` with the public address of the miner hotkey that earned
the reward.

Example shape:

```json
[
  {
    "epoch": 1234,
    "index": 0,
    "hotkey": "5MinerHotkey...",
    "recipient_coldkey": "5Recipient...",
    "amount_rao": 250000000,
    "proof_count": 8,
    "root": "0x..."
  }
]
```

What the fields mean:

- `hotkey`: the miner identity that earned the reward.
- `recipient_coldkey`: the account that will receive the reward.
- `amount_rao`: the cumulative reward amount in alpha base units.
- `proof_count` and `root`: proof data used by the contract.

If this prints `[]`, there is no claim package ready for that hotkey right now.
That can mean validation is not done, Pool has not published the next epoch, or
the hotkey has no claimable reward.

## Step 2: Check The Recipient

Before claiming, read the `recipient_coldkey`.

That is the payout account.

If it is not the account you expected, do not claim yet. Fix the recipient
mapping for future epochs or ask for help. The current claim package will still
pay the already published `recipient_coldkey`.

Claim time is too late to choose the recipient. Claiming only submits the
already published proof.

## Step 3: Submit The Claim

Use a local wallet key to sign the transaction:

```bash
python scripts/claim_merkle_rewards.py claim \
  --hotkey <MINER_HOTKEY> \
  --wallet-name <WALLET_NAME> \
  --wallet-hotkey <HOTKEY_NAME>
```

The wallet key in `--wallet-name` and `--wallet-hotkey` pays the transaction
fee. It does not decide where the reward goes.

The reward destination is still the published `recipient_coldkey`.

## Step 4: Check Success

A successful claim prints JSON with `ok: true` and usually an extrinsic hash:

```json
[
  {
    "ok": true,
    "epoch": 1234,
    "index": 0,
    "extrinsic_hash": "0x..."
  }
]
```

The reward is subnet alpha stake. Do not expect it to appear as free TAO in the
wallet balance.

## Common Mistakes

Wrong hotkey:

- You listed claims for the wrong miner hotkey.
- Use the hotkey that signed the mining submission.

No claim package:

- Your submission may not be accepted yet.
- Pool may not have published a new epoch yet.
- Your hotkey may not have earned a reward in the published epoch.

Wrong recipient:

- The claim pays the published `recipient_coldkey`.
- Your local wallet setting does not change an already published epoch.

Not enough fee balance:

- The signer needs enough free balance to pay transaction fees.
- The recipient can be different from the signer.

## Small Claim Warning

Some small claims cannot transfer yet.

The claim package amount is cumulative. That means the contract stores the total
earned amount for your miner hotkey. When you claim, the contract only transfers
the new part:

```text
new transfer = amount_rao - already_claimed_amount
```

The chain also has a live minimum transfer size for same-subnet alpha stake.

If your new transfer is below that live minimum, the claim can fail with a vague
contract error even if everything else is correct.

If a tiny claim fails with `ContractReverted`:

1. Do not keep retrying it.
2. Wait for more rewards to accumulate.
3. Try again after a later Pool epoch.
4. Ask an operator to check the live transfer floor if you are unsure.

## Safety Rules

- Never paste your mnemonic into chat.
- Never commit wallet files, seed phrases, or private keys.
- Prefer local wallet names over command-line mnemonics.
- Only use `--signer-mnemonic` for throwaway test wallets.
- Check the `recipient_coldkey` before claiming.
