<section class="bitsota-hero">
  <p class="bitsota-kicker">Base SOTA fork</p>
  <h1>Claim and mine SOTA on a local Base chain.</h1>
  <p class="bitsota-lede">
    Base SOTA is a Base-settled SOTA fork, not a Bittensor subnet. The local
    demo starts the EVM node, claim contracts, autoresearch backend, indexer,
    website, and docs so a tester can move through the full flow in one session.
  </p>
  <div class="bitsota-actions">
    <a class="bitsota-button" href="local-e2e/">Run the local demo</a>
    <a class="bitsota-button secondary" href="new-users/">New user guide</a>
    <a class="bitsota-button secondary" href="bittensor-migrants/">Migrating from Bittensor</a>
  </div>
</section>

<div class="bitsota-path-grid">
  <section class="bitsota-panel">
    <div class="bitsota-panel-head">
      <h2>For A Tester</h2>
    </div>
    <div class="bitsota-panel-body">
      <ol class="bitsota-step-list">
        <li><b>1</b><span>Start the local stack and open the printed Tester handoff URL.</span></li>
        <li><b>2</b><span>Add the local Base network in MetaMask and import the printed local-only key.</span></li>
        <li><b>3</b><span>Claim genesis SOTA, load the mined emission, inspect the 3/3 self-validation evidence, and claim again.</span></li>
      </ol>
    </div>
  </section>
  <section class="bitsota-panel">
    <div class="bitsota-panel-head">
      <h2>What Runs Locally</h2>
    </div>
    <div class="bitsota-panel-body">
      <div class="bitsota-mini-map">
        <span><b>Claim</b><small>seeded TAO + alpha accounting credit</small></span>
        <span><b>Mine</b><small>Alice submits a local frontier improvement</small></span>
        <span><b>Validate</b><small>Bob, Charlie, and Dave accept the work</small></span>
      </div>
    </div>
  </section>
</div>

<div class="bitsota-command">
  <p class="title">Start the complete local stack</p>
  <pre><code>cd /home/mekaneeky/repos/SN94-BitSota-live-docs
./scripts/sota_local_demo.py launch</code></pre>
</div>

<div class="bitsota-quick-facts">
  <div class="bitsota-fact"><strong>Local-only</strong><span>Chain ID 31337, deterministic Anvil accounts, no production keys.</span></div>
  <div class="bitsota-fact"><strong>No mocks</strong><span>Contracts are deployed, roots are published, claims are real local EVM txs.</span></div>
  <div class="bitsota-fact"><strong>Fork logic</strong><span>Genesis SOTA = TAO 1:1 plus synthetic alpha pool credit.</span></div>
  <div class="bitsota-fact"><strong>Ongoing rewards</strong><span>SOTA emissions only after accepted self-validation evidence.</span></div>
</div>

<div class="bitsota-status-grid">
  <div class="bitsota-status">
    <strong>Local EVM</strong>
    <span>Deploys SOTA token, vault, root registry, lane registry, and claim distributors.</span>
  </div>
  <div class="bitsota-status">
    <strong>Autoresearch backend</strong>
    <span>Seeds a miner submission and records accepted self-validation evidence from three local peer validators.</span>
  </div>
  <div class="bitsota-status">
    <strong>Claims UI</strong>
    <span>Lets a tester claim genesis SOTA and mined emission SOTA with MetaMask.</span>
  </div>
</div>

## Pick Your Path

<div class="bitsota-card-grid">
  <a class="bitsota-card" href="new-users/">
    <strong>New user</strong>
    <span>Learn what SOTA, Base, claims, proofs, and self-validation mean before clicking through the demo.</span>
  </a>
  <a class="bitsota-card" href="bittensor-migrants/">
    <strong>Migrating from Bittensor</strong>
    <span>Map coldkeys, hotkeys, netuids, Yuma emissions, and alpha accounting to the Base SOTA model.</span>
  </a>
  <a class="bitsota-card" href="local-e2e/">
    <strong>Local tester</strong>
    <span>Run the complete local stack, import the printed wallet, claim SOTA, and inspect validation evidence.</span>
  </a>
  <a class="bitsota-card" href="architecture/">
    <strong>Technical reviewer</strong>
    <span>Review the contracts, root lifecycle, indexer, attestation checks, and launch gates.</span>
  </a>
</div>

## What Base SOTA Does

Base SOTA turns research competition results into claimable SOTA on Base.

1. A legacy snapshot creates the one-time genesis allocation.
2. A claimant binds their legacy Bittensor coldkey to a Base wallet.
3. Miners submit work with an EVM identity.
4. Self-validation checks accepted work before emissions are published.
5. Claim roots are posted to the local or deployed Base contracts.
6. Users claim SOTA from their own wallet without giving custody to the app.

```mermaid
flowchart LR
  Snapshot[Legacy snapshot] --> Binding[Bind to Base wallet]
  Binding --> Genesis[Genesis claim root]
  Miner[Miner submission] --> Validation[Self-validation]
  Validation --> Emissions[Emission claim root]
  Genesis --> Registry[Root registry]
  Emissions --> Registry
  Registry --> Claims[Claim distributors]
  Claims --> Wallet[User wallet receives SOTA]
```

## Current Status

The local demo path is the supported way to experience the Base flow. It uses a
local dev chain and deterministic local accounts. It does not touch production
Bittensor, Base Sepolia, or Base mainnet.

<div class="bitsota-callout good">
  <strong>Local path:</strong> one launcher command deploys local contracts,
  seeds a snapshot claim, runs a miner/self-validation flow with three local
  peer validators, publishes a claim root, starts the UI, and prints the URLs a
  tester needs.
</div>

<div class="bitsota-callout">
  <strong>Public deployment:</strong> Base Sepolia and Base mainnet deployment
  details are not public launch instructions until the testnet preflight gate
  is green against real deployed contracts and public service URLs.
</div>

## Keep Reading

- [One-Command Demo](local-e2e.md) for the run command, expected output, and
  troubleshooting.
- [Claims And Allocation](claims-and-allocation.md) for genesis claims,
  ongoing emissions, and wallet binding.
- [Self-Validation](self-validation.md) for how miner work becomes eligible for
  a SOTA emission.
- [Support FAQ](support-faq.md) for Base Sepolia support wording, intake fields,
  and escalation boundaries.
- [Operator Readiness](operations.md) for the launch checklist and escalation
  path.
