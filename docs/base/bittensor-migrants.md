<section class="bitsota-hero compact">
  <p class="bitsota-kicker">Migration guide</p>
  <h1>Move from subnet thinking to a Base-settled SOTA fork.</h1>
  <p class="bitsota-lede">
    Base SOTA keeps research competitions, but it does not use Bittensor subnet
    registration, netuids, validator weights, or alpha emissions. Claims settle
    through EVM contracts and self-validation evidence.
  </p>
</section>

<div class="bitsota-quick-facts">
  <div class="bitsota-fact"><strong>Coldkey</strong><span>Used for legacy genesis eligibility, not as a Base custody account.</span></div>
  <div class="bitsota-fact"><strong>Hotkey</strong><span>Dropped from settlement. Mining uses an EVM miner key and optional reward address.</span></div>
  <div class="bitsota-fact"><strong>Alpha</strong><span>No protocol alpha token. Snapshot exposure becomes synthetic SOTA credit.</span></div>
  <div class="bitsota-fact"><strong>Emissions</strong><span>Paid in SOTA after self-validation, not Yuma weight emission.</span></div>
</div>

<div class="bitsota-command">
  <p class="title">Run the migration demo locally</p>
  <pre><code>cd /home/mekaneeky/repos/SN94-BitSota-live-docs
./scripts/sota_local_demo.py launch</code></pre>
</div>

The launcher prints the website, backend, indexer, local EVM, docs, and
`Tester handoff` URLs.

## Run Checklist For Migrants

1. Open the printed `Tester handoff` URL.
2. Import the printed local EVM key into a throwaway MetaMask profile.
3. Use the printed old coldkey only for the genesis lookup. Do not import a
   Bittensor wallet seed into MetaMask.
4. Add the local network with chain ID `31337`.
5. Claim genesis SOTA from the seeded TAO plus alpha accounting credit.
6. Load the mined emission, confirm `3/3 accepted` self-validation, then claim
   the emission.
7. Ignore hotkeys, netuid, UID, Yuma weights, and alpha emissions for this
   Base-settled flow.

## What Changes

| Bittensor concept | Base SOTA equivalent |
| --- | --- |
| Coldkey | Used once to prove legacy snapshot ownership for genesis binding. |
| Hotkey | Replaced by EVM miner identity for Base SOTA submissions. |
| Netuid | Replaced by SOTA-native lane or competition IDs. |
| UID/metagraph position | Not part of the Base settlement model. |
| Validator weights/Yuma emissions | Replaced by self-validation evidence and claim roots. |
| Substrate balance changes | Replaced by EVM claims against Base contracts. |
| TAO/alpha accounting | Used only for the genesis allocation formula, not ongoing emissions. |
{: .bitsota-compare}

## Migration Checklist

1. Treat Base SOTA as a fork claim, not a subnet registration.
2. Use the legacy coldkey only for genesis ownership proof.
3. Choose the Base wallet that should receive SOTA.
4. Mine with an EVM miner identity.
5. Optionally route rewards to a separate EVM reward address.
6. Expect ongoing SOTA only when a submission passes self-validation and appears in a published claim root.

<div class="bitsota-callout">
  <strong>Genesis is the compatibility layer.</strong> TAO transfers 1:1 into
  SOTA for eligible coldkeys, and alpha LP value is added from the approved
  liquidation formula. Ongoing emissions are SOTA-native and do not mint alpha
  tokens.
</div>

## Genesis Is A Bridge From Legacy State

Genesis starts from a legacy snapshot. A holder proves ownership with the
legacy Bittensor coldkey and binds the claim to a Base wallet.

After binding, the Base wallet is the account that claims SOTA. The coldkey does
not receive or custody Base SOTA.

## Mining Uses EVM Identity

A Base SOTA miner submission includes:

- EVM miner address;
- optional reward address;
- competition or lane ID;
- artifact hash;
- miner signature;
- optional reward-address delegation signature.

Accepted work can become an emission claim only after self-validation. The claim
settles through the emission distributor contract.

<div class="bitsota-status-grid">
  <div class="bitsota-status">
    <strong>Miner key</strong>
    <span>Signs the submission and identifies the work.</span>
  </div>
  <div class="bitsota-status">
    <strong>Reward address</strong>
    <span>Can be the same EVM address or a separate destination chosen by the miner.</span>
  </div>
  <div class="bitsota-status">
    <strong>Claim contract</strong>
    <span>Releases SOTA only after the accepted work appears in a valid root.</span>
  </div>
</div>

## What To Ignore From The Old Mental Model

Do not look for:

- subnet registration;
- validator weight setting;
- Yuma consensus;
- metagraph UID emissions;
- on-chain alpha emissions.

The local demo is the fastest way to see the new model end to end. It shows
legacy eligibility, EVM wallet binding, miner validation, claim roots, and SOTA
claims in one local flow.
