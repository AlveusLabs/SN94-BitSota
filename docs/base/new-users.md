<section class="bitsota-hero compact">
  <p class="bitsota-kicker">First-time guide</p>
  <h1>Try Base SOTA without knowing Bittensor.</h1>
  <p class="bitsota-lede">
    The local demo shows the product loop with test-only accounts: an eligible
    user claims SOTA, a miner submits work, three other local users
    self-validate it, and the miner reward becomes claimable.
  </p>
</section>

<div class="bitsota-quick-facts">
  <div class="bitsota-fact"><strong>You need</strong><span>A browser with MetaMask and the printed local-only test key.</span></div>
  <div class="bitsota-fact"><strong>You do not need</strong><span>TAO, a real Bittensor wallet, Base ETH, or a seed phrase.</span></div>
  <div class="bitsota-fact"><strong>You will claim</strong><span>1.5 local SOTA from genesis accounting, then 2 local SOTA from mining.</span></div>
  <div class="bitsota-fact"><strong>You will verify</strong><span>The mined reward shows 3/3 accepted peer-validation evidence.</span></div>
</div>

<div class="bitsota-command">
  <p class="title">Run this first</p>
  <pre><code>cd /home/mekaneeky/repos/SN94-BitSota-live-docs
./scripts/sota_local_demo.py launch</code></pre>
</div>

Open the `Tester handoff` URL printed by the launcher. It shows the local-only
wallet, the local network button, the claim UI link, and the exact amounts the
tester should see. If the handoff shows `localhost` share mode, use MetaMask on
the same computer that started the launcher. If someone is testing from another
computer on the same Tailscale, the handoff should show `tailscale-https` share
mode and an `https://...:8545` MetaMask RPC URL.

<div class="bitsota-path-grid">
  <section class="bitsota-panel">
    <div class="bitsota-panel-head">
      <h2>The Claim Path</h2>
    </div>
    <div class="bitsota-panel-body">
      <ol class="bitsota-step-list">
        <li><b>1</b><span>The app loads a seeded user who is eligible for a genesis SOTA claim.</span></li>
        <li><b>2</b><span>The app shows the proof and why the claim can be made.</span></li>
        <li><b>3</b><span>The tester claims SOTA with the printed local MetaMask account.</span></li>
      </ol>
    </div>
  </section>
  <section class="bitsota-panel">
    <div class="bitsota-panel-head">
      <h2>The Mining Path</h2>
    </div>
    <div class="bitsota-panel-body">
      <ol class="bitsota-step-list">
        <li><b>4</b><span>The local backend has already run a seeded miner submission.</span></li>
        <li><b>5</b><span>Three local peer validators accept the submission through self-validation.</span></li>
        <li><b>6</b><span>The tester claims the mined emission and sees the local SOTA balance increase again.</span></li>
      </ol>
    </div>
  </section>
</div>

## Terms In Plain English

| Term | Meaning |
| --- | --- |
| SOTA | The ERC-20 token claimed in the Base SOTA fork. |
| Base | The Ethereum layer-2 network where claims settle. The local demo uses a local EVM instead of the real network. |
| Claim | A wallet transaction that releases SOTA to an eligible user. |
| Genesis claim | A one-time claim for eligible legacy holders. |
| Emission claim | A miner reward claim from accepted competition work. |
| Proof | Data that lets the contract verify a claim belongs in a published root. |
| Self-validation | The check that decides whether miner work is accepted before it can receive SOTA. |

## What The Buttons Mean

| Button | Plain-English meaning |
| --- | --- |
| Add SOTA Local Base network | Adds the local Anvil chain to MetaMask. |
| Load genesis claim | Shows the SOTA created from seeded TAO plus alpha accounting credit. |
| Claim | Sends the local EVM transaction that releases SOTA to the connected wallet. |
| Load mined emission | Shows the reward from the seeded miner and its self-validation evidence. |
| Refresh balance | Reads the local SOTA ERC-20 balance after a claim confirms. |

## What You Should See

The demo website should show:

- a seeded user with a genesis claim;
- a seeded miner or reward wallet with an emission claim;
- `3/3 accepted` peer-validation evidence for the mined emission;
- proof details for each claim;
- a local claim transaction;
- a local SOTA balance change after the claim.

The demo does not move real money. It uses a deterministic local snapshot input
and local accounts so you can see the product flow safely.

<div class="bitsota-callout danger">
  <strong>Do not use real keys in the local demo.</strong> Import only the
  local-only private key printed by the launcher into a throwaway MetaMask
  profile.
</div>

## What To Read Next

- [One-Command Demo](local-e2e.md) for the command, expected output, and
  troubleshooting.
- [Claims And Allocation](claims-and-allocation.md) if you want to understand
  genesis and emission claims.
- [Self-Validation](self-validation.md) if you want to understand how miner
  rewards become claimable.
