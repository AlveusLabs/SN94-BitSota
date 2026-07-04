<section class="bitsota-hero compact">
  <p class="bitsota-kicker">Local test path</p>
  <h1>Run the whole Base SOTA loop on this machine.</h1>
  <p class="bitsota-lede">
    One command starts the local chain, contracts, indexer, autoresearch
    backend, website, and docs. It seeds a genesis claim and a miner emission
    claim backed by a three-user self-validation committee so a tester can
    claim both in MetaMask.
  </p>
</section>

<div class="bitsota-command">
  <p class="title">Start</p>
  <pre><code>cd /home/mekaneeky/repos/SN94-BitSota-live-docs
./scripts/sota_local_demo.py launch</code></pre>
</div>

`launch` runs the readiness checks, submits and verifies one automated pair of
local genesis/emission claims, resets the stack so the human tester still sees
unclaimed SOTA, serves the tester handoff, leaves the services running, and
returns to the shell. Developers who want the launcher to hold the terminal
open can use `./scripts/sota_local_demo.py start` instead.

Do not assemble the demo by running separate contract, backend, indexer, or UI
tests. Those checks are useful for developers, but they are not the product
walkthrough.

## What The Launcher Starts

The launcher starts a complete local stack:

<div class="bitsota-status-grid">
  <div class="bitsota-status"><strong>Local EVM</strong><span>Deploys the SOTA token, vault, root registry, lane registry, and claim distributors.</span></div>
  <div class="bitsota-status"><strong>Indexer/API</strong><span>Serves eligibility, proofs, claim status, and unsigned claim calldata.</span></div>
  <div class="bitsota-status"><strong>Autoresearch</strong><span>Seeds demo competitions, miner submissions, three peer validators, self-validation evidence, and claim roots.</span></div>
  <div class="bitsota-status"><strong>Claims UI</strong><span>Runs the wallet-facing claim experience against the local stack.</span></div>
  <div class="bitsota-status"><strong>Docs</strong><span>Serves this guide next to the running demo for another Tailscale computer.</span></div>
  <div class="bitsota-status"><strong>MetaMask path</strong><span>Uses a printed local-only private key and Anvil RPC URL.</span></div>
</div>

The demo uses a local dev chain and deterministic local accounts. It does not
touch production Bittensor, production TAO, Base Sepolia, Base mainnet, or
production SOTA. Inside the local stack, the claims are real: the launcher
deploys contracts, publishes roots, imports public claim artifacts, syncs local
claim events, and the wallet sends local EVM transactions to release local SOTA.

<div class="bitsota-callout good">
  <strong>The mining round is real local backend state.</strong> During startup,
  the autoresearch backend creates a local binary-frontier task, records Alice's
  EVM-signed miner submission, records accepted evaluations from Bob, Charlie,
  and Dave, builds the emission claim root, publishes it to the local contracts,
  and imports it into the indexer for the claims UI.
</div>

## Plain-English Walkthrough

Use this flow when a reviewer is not technical and only wants to know whether
the product story makes sense:

<ol class="bitsota-step-list">
  <li><b>1</b><span>Start the launcher and wait until it says the demo is ready.</span></li>
  <li><b>2</b><span>Open the printed <strong>Tester handoff</strong> URL.</span></li>
  <li><b>3</b><span>Use the handoff buttons to add the local network, copy the local-only key, and open the claims website.</span></li>
  <li><b>4</b><span>Confirm the <strong>Local readiness</strong> panel says the local stack is ready.</span></li>
  <li><b>5</b><span>Import the printed local-only MetaMask account into a throwaway browser profile.</span></li>
  <li><b>6</b><span>Add the printed Anvil RPC as the wallet network if the handoff button was not used.</span></li>
  <li><b>7</b><span>Click <strong>Load genesis claim</strong>.</span></li>
  <li><b>8</b><span>Confirm the page shows the TAO credit, synthetic alpha credit, proof root, and claimable state.</span></li>
  <li><b>9</b><span>Submit the local genesis claim and watch the wallet SOTA balance increase.</span></li>
  <li><b>10</b><span>Click <strong>Load mined emission</strong>.</span></li>
  <li><b>11</b><span>Confirm the page shows the mined task, 3/3 accepted peer-validation consensus, root, and emission amount.</span></li>
  <li><b>12</b><span>Submit the local emission claim and watch the wallet SOTA balance increase again.</span></li>
</ol>

The reviewer does not need to understand coldkeys, Merkle proofs, or validator
weights to pass this demo. They only need to see three things: the page explains
who can claim, the wallet asks for the claim transaction, and the balance
changes after the claim confirms.

<div class="bitsota-command">
  <p class="title">Stop</p>
  <pre><code>cd /home/mekaneeky/repos/SN94-BitSota-live-docs
./scripts/sota_local_demo.py stop</code></pre>
</div>

## Expected Output

A successful run prints a ready message and a URL block. Use the URLs from your
terminal. When Tailscale MagicDNS is available, the launcher publishes the
browser-facing services with Tailscale Serve HTTPS so another computer on the
same tailnet can open the handoff and MetaMask can use an HTTPS local RPC URL.

```text
SOTA Base local demo is ready.

Claims UI:              https://sota-host.example.ts.net:3000/claims
Autoresearch dashboard: https://sota-host.example.ts.net:8000/dashboard
Docs:                   https://sota-host.example.ts.net:9002/base/
Tester handoff:         https://sota-host.example.ts.net:9003/
Anvil RPC for MetaMask: https://sota-host.example.ts.net:8545
Share mode:             tailscale-https (green)

Import this local-only account in MetaMask:
Private key: 0x...
Address: 0x...
Old coldkey for genesis lookup: 5...
```

If the printed share mode is `http`, the demo still works on the machine that
started it. For another computer on Tailscale, rerun:

```bash
cd /home/mekaneeky/repos/SN94-BitSota-live-docs
./scripts/sota_local_demo.py launch --share-mode tailscale-https
```

## Success Checklist

You know the demo is working when you can do these steps:

1. Open the printed `Tester handoff` URL.
2. Confirm the `Local readiness` panel is green. It checks the claims API,
   indexer sync, local contract roles, autoresearch backend, and
   self-validation evidence through the same browser proxy routes.
3. Import the printed local-only private key into MetaMask.
4. Add the printed Anvil RPC as the wallet network. For another Tailscale
   computer, the RPC URL should be `https://...:8545`.
5. Use the printed old coldkey and address to look up the genesis claim.
6. Submit the genesis claim and see the local SOTA balance card update.
7. Load the mined emission for the same EVM address.
8. Confirm the mining and self-validation card shows `3/3 accepted` peer
   consensus from the seeded local validators.
9. Submit the emission claim created by the seeded miner and self-validation flow.
10. Open the autoresearch dashboard and see the seeded task, submission, and
   self-validation evidence.

The launcher runs this UI smoke automatically. If you changed code after launch,
rerun it against the already-running stack:

```bash
cd /home/mekaneeky/repos/SN94-BitSota-live-docs
./scripts/sota_local_demo.py ui-smoke
```

This checks the actual claims page, the local readiness proxy routes, the
website proxy to the claims indexer, the website proxy to autoresearch
self-validation evidence, the tester-facing wallet RPC URL, the Base docs pages
a tester follows, and unsigned transaction payload generation. It also writes a
report to
`/home/mekaneeky/repos/.sota-base-local/ui-smoke/report.json` and, when Firefox
is available, a screenshot to
`/home/mekaneeky/repos/.sota-base-local/ui-smoke/claims-page.png`.

Generate the tester handoff after the smoke passes:

```bash
cd /home/mekaneeky/repos/SN94-BitSota-live-docs
python3 scripts/sota_base_tester_handoff.py --environment local
```

The launcher also serves the handoff at the printed `Tester handoff` URL. The
handoff contains the live claims URL, docs URL, MetaMask RPC URL, share mode,
chain ID, local-only wallet, old coldkey, and plain-English steps. It is
generated from the current local state and smoke report so the URLs and
pass/fail status do not drift from the running demo. Running the handoff
generator without custom output paths refreshes that served copy automatically
when the handoff includes local content.

For a nontechnical reader, the important point is simple: the app shows who can
claim, why they can claim, and how the claim is verified before SOTA is
released.

To preserve evidence from a real MetaMask run, record the genesis and mined
emission transaction hashes shown by the claims UI, then verify them:

```bash
cd /home/mekaneeky/repos/SN94-BitSota-live-docs
python3 scripts/sota_base_claim_tx_evidence.py --environment local \
  --state /home/mekaneeky/repos/.sota-base-local/state.json \
  --genesis-tx "$LOCAL_GENESIS_TX_HASH" \
  --emission-tx "$LOCAL_EMISSION_TX_HASH" \
  --report-out /home/mekaneeky/repos/.sota-base-local/claim-proof/manual-claim-tx-evidence.json
```

This verifier is read-only. It checks the local chain ID, distributor
addresses, receipts, SOTA transfer events, and final balance evidence without
signing or broadcasting anything.

The launcher also runs the no-mock operator proof automatically before it prints
the final ready block. It fetches the same unsigned calldata that the claims UI
gives MetaMask, signs it with the printed local-only key, submits both local
claim transactions, runs the receipt verifier, writes evidence, and then resets
the stack so the tester still starts with unclaimed SOTA.

If you need to rerun only that proof against an already-running stack, use:

```bash
cd /home/mekaneeky/repos/SN94-BitSota-live-docs
./scripts/sota_local_claim_proof.py --reset-after
```

This command is state-changing on the local Anvil chain only. It refuses any
RPC that is not chain ID `31337`, does not touch Base Sepolia, Base mainnet, or
production Bittensor, and writes its latest reports under
`/home/mekaneeky/repos/.sota-base-local/claim-proof/`.
When run with `--reset-after`, the transaction receipts are archived proof from
the pre-reset local chain, and the current tester stack is restored with
unclaimed SOTA.

For a Bittensor migrant, the important point is the model change: claims settle
to an EVM wallet on Base instead of relying on Substrate emissions or Yuma
validator weights.

For a testnet operator, the important point is the boundary: a passing local
demo proves the product loop, local contract/indexer wiring, public claim
artifact ingestion, and self-validation evidence lookup. It does not prove Base
Sepolia deployment, public RPC configuration, contract source verification,
monitoring, or browser-wallet gas behavior on a public network.

## Troubleshooting

| Problem | What to do |
| --- | --- |
| `No such file or directory` | Make sure you are in `/home/mekaneeky/repos/SN94-BitSota-live-docs` and that the launcher branch is checked out. Do not replace the launcher with the old split-test flow. |
| A port is already in use | Run `./scripts/sota_local_demo.py stop`, close any old local demo terminals, then start again. |
| The website opens but claims are empty | Check that the terminal printed the old coldkey, local EVM address, and self-validation consensus block. Restart the launcher if the backend or indexer failed to seed. |
| The claim button fails | Confirm MetaMask is on the printed Anvil RPC and the imported account matches the printed address. |
| A service fails during startup | Save the first error block from the terminal. It usually names the missing prerequisite or service that failed. |

## Developer Smoke

The noninteractive smoke runs the same local deployment and seeding path, submits
both claim transactions to Anvil, verifies the local SOTA balance, and stops the
stack:

```bash
cd /home/mekaneeky/repos/SN94-BitSota-live-docs
./scripts/sota_local_demo.py smoke
```

Use these checks for local release readiness:

1. `./scripts/sota_local_demo.py smoke` proves the contract/indexer/backend
   transaction loop without a browser.
2. `./scripts/sota_local_demo.py launch` proves the running user-facing page,
   proxy routes, state-changing claim payloads, receipt evidence, reset path,
   docs, and tester handoff.
3. `./scripts/sota_local_demo.py ui-smoke --skip-screenshot` reruns the
   browser-facing readiness checks after code or config changes.

The release status report requires both the UI smoke report and the latest
local claim proof report before it marks `local_ok` as true.

## Escalation

If the command fails after the troubleshooting steps, send the launcher output
and the printed URL block to the Base demo owner.

Route by symptom:

| Symptom | Owner path |
| --- | --- |
| Website or claim UI problem | Full-stack product engineering |
| Backend seeding, self-validation, or claim-root problem | Autoresearch backend engineering |
| Contract deployment or local EVM problem | Engineering release owner |
| Public wording or docs problem | Technical docs / product release |
