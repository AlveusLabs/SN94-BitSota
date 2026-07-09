# Operator Readiness

This is the local and testnet readiness runbook for Base SOTA. It is not a
public launch announcement, not a Base mainnet runbook, and not production
Bittensor evidence.

Use plain environment names in every note:

| Environment | What it means | What it proves |
| --- | --- | --- |
| Local demo | A local EVM, local accounts, local indexer/API, local backend, local website, and local docs. | The product loop is understandable and works on one machine. |
| Base Sepolia testnet | Public Base testnet with test ETH, deployed test contracts, public test service config, and test-only roots. | The public-network wiring works without real claims or mainnet value. |
| Base mainnet | Real Base network and production SOTA claims. | Not covered by this page until the operator records mainnet evidence and approval. |

SN402/Bittensor test evidence is useful background, but it is not enough for
Base SOTA testnet readiness. Base Sepolia needs its own contracts, manifest,
API/indexer wiring, browser wallet smoke, and rollback evidence. Monitoring is
recommended observability, not a blocker for the nontechnical claims/mining
test path.

## Local Demo Gate

Before any public testnet or mainnet instructions are published, the local demo
must run from one command:

```bash
cd /home/mekaneeky/repos/SN94-BitSota-live-docs
./scripts/sota_local_demo.py launch
```

The launcher must start the local EVM, indexer, autoresearch backend, website,
and docs. It must seed demo users, run miners through self-validation, and
print the URLs a reviewer should open. Before printing the final ready block,
it must also submit and verify one automated local genesis claim and one
automated local emission claim, then reset the stack so the reviewer still
starts with unclaimed SOTA.

Success means a reviewer can:

1. Open the printed website URL.
2. Import the printed local-only wallet into a throwaway MetaMask profile.
3. Add the printed Anvil RPC as the wallet network.
4. View a genesis claim and understand why the account is eligible.
5. Submit a local genesis claim transaction.
6. View an emission claim created after self-validation.
7. Submit a local emission claim transaction.
8. Confirm the local SOTA balance changed after each claim.

Stop the demo when review is finished:

```bash
cd /home/mekaneeky/repos/SN94-BitSota-live-docs
./scripts/sota_local_demo.py stop
```

Save evidence that a nontechnical reviewer can read later:

- the ready URL block printed by the launcher;
- the local claim proof report under
  `/home/mekaneeky/repos/.sota-base-local/claim-proof/latest.json`;
- the local multi-miner swarm report under
  `/home/mekaneeky/repos/.sota-base-local/miner-swarm/latest.json`;
- the local wallet address and old coldkey shown by the launcher;
- screenshots or notes for before/after local SOTA balances;
- the autoresearch dashboard showing the seeded task, submission, and
  self-validation evidence;
- the command output for any failure.

## Readiness Colors

Use these labels when summarizing status:

| Label | Meaning |
| --- | --- |
| Green | A reviewer can repeat the step from the documented command and the expected result is visible. |
| Yellow | The step works only with developer help, fixture data, optional monitoring gaps, or a manual workaround. |
| Red | The step is missing, unsafe, unverifiable, or pointed at the wrong environment. |

Do not call testnet ready while any required Base Sepolia gate is red. Yellow
items need an owner, a deadline, and a written risk.

## One Status Command

Use the aggregate status command when you need the honest current answer across
local demo and Base Sepolia readiness:

```bash
cd /home/mekaneeky/repos/SN94-BitSota-live-docs
python3 scripts/sota_base_release_status.py \
  --snapshot-claim-bindings-url "$SOTA_CLAIMS_API_URL/api/v1/base/genesis/bindings" \
  --report-out /home/mekaneeky/repos/.sota-base-testnet/base-sota-release-status.json \
  --allow-blocked
```

This command reads the local UI smoke report, the local state-changing claim
proof report, the local multi-miner self-validation/claim report, Base Sepolia
operator report, blocker report, public browser-smoke report, and claim
transaction evidence report. It also requires the fresh emission tester report,
which proves a new funded testnet wallet has a real self-validated emission
claim and unsigned MetaMask calldata. By default it also checks the user-systemd
genesis and emission publisher timers directly; use
`--skip-publisher-timer-check` only when reviewing offline artifacts on a host
that is not responsible for publishing. It does not deploy contracts, sign
messages, broadcast transactions, or touch production Bittensor. Green means
the required gates are green. Red means at least one required gate still
blocks nontechnical testing.

If `--snapshot-claim-bindings-url` is supplied and
`SOTA_BASE_INDEXER_ADMIN_TOKEN` is set, the report also records the public
claims API's accepted signed snapshot binding count. The report stores only the
count, URL, token environment variable name, and whether an auth header was
used; it must not store the token value.

For a local-only check, run the local claim proof and swarm smoke first:

```bash
./scripts/sota_local_demo.py smoke
./scripts/sota_local_demo.py swarm-smoke --count 5
python3 scripts/sota_base_release_status.py --local-only
```

If the operator intentionally is not running a real holder claim yet, use the
deferred-holder mode. It keeps strict mode unchanged, but lets the status turn
green when the public binding-message route, invalid-signature rejection,
locked snapshot source, and scheduled genesis/emission publishers are ready:

```bash
python3 scripts/sota_base_release_status.py \
  --snapshot-claim-bindings-url "$SOTA_CLAIMS_API_URL/api/v1/base/genesis/bindings" \
  --defer-real-holder-test \
  --report-out /home/mekaneeky/repos/.sota-base-testnet/base-sota-release-status.json
```

The generated report includes `real_holder_test_deferred: true`; do not present
that as a completed real-holder claim.

Generate a nontechnical tester handoff from the current reports:

```bash
python3 scripts/sota_base_tester_handoff.py \
  --json-out /home/mekaneeky/repos/.sota-base-testnet/base-sota-tester-handoff.json \
  --markdown-out /home/mekaneeky/repos/.sota-base-testnet/base-sota-tester-handoff.md
```

The Markdown handoff includes the live local URLs, local-only MetaMask key,
Anvil RPC settings, old coldkey lookup, local claim proof status, Base Sepolia
gate status, and the next blocked action. Use `--environment local` when
preparing a local-only tester sheet. Use `--environment testnet` only after the
Base Sepolia gates are green; that mode omits the local-only private key.
When no custom output paths are supplied and the handoff includes local
content, the generator also refreshes the served local copy at
`/home/mekaneeky/repos/.sota-base-local/handoff/index.html`.

When the local launcher starts with website and docs enabled, it also serves
the generated local handoff at the printed `Tester handoff` URL on port `9003`.

## Base Sepolia Testnet Gates

Publish Base Sepolia instructions only after each gate has evidence:

| Gate | Required evidence |
| --- | --- |
| Local demo | One-command local run passes and the reviewer evidence above is saved. |
| Deployment manifest | One manifest names `base-sepolia`, chain ID, public RPC label, secret handles, source branches, commit SHAs, contract addresses, ABI bundle, service URLs, owners, and rollback owner. Use `docs/base/manifests/base-sepolia-deployment-manifest.template.json` until a deployed Base Sepolia manifest replaces it. |
| Contracts | SOTA token, vault, root registry, lane registry, genesis distributor, and emission distributor are deployed to Base Sepolia with constructor args and source verification links. |
| Funding | Deployer, root publisher, and the tester wallet have Base Sepolia ETH before deployment, root publication, or browser-wallet smoke. |
| Roles and custody | Owner, publisher, releaser, pause guardian, and multisig or timelock records are written using public addresses and secret handles only. |
| Snapshot genesis | The finalized Base Sepolia genesis claim artifact is built from `/mnt/4tb/tao_fork_snapshot`, includes the locked Bittensor block, and records `tao_credit_rao`, `alpha_synthetic_credit_rao`, and `alpha_credit_rao_by_netuid` per allocation. |
| Claim/root artifacts | Genesis uses signed coldkey snapshot bindings. Emissions use accepted autoresearch self-validation evidence. Both roots are finalized with emitted on-chain root IDs before indexer import. |
| Indexer/API | Testnet API reads from the manifest, catches up contract events, ingests public claim artifacts, serves genesis binding-message and signed-binding submit routes, and serves eligibility, proof, claim status, root status, and unsigned claim calldata routes. |
| Claims website | Website is configured for Base Sepolia, shows split genesis/emission claims, exposes the genesis binding payload and submit flow, and labels the network as testnet. |
| Browser wallet smoke | A funded test wallet can switch to Base Sepolia, submit a test claim, and produce an explorer transaction hash or a recorded revert. |
| Root lifecycle | Test-only genesis and emission roots can be built, validated, published, indexed, read back, and challenged or paused when appropriate. |
| Observability | Operators can see service health, RPC failures, index lag, failed claim transactions, API errors, signer actions, root publication, and pause state. |
| Recovery | Operators know how to pause, stop writers, preserve logs and transaction hashes, correct config, redeploy if needed, and verify recovery. |
| Support wording | Public wording explains testnet claims without promising payout timing, reversals, exchange distribution, audit completion, or mainnet launch. |
| QA gate | QA records pass/fail/blocker status for every gate and decides whether the final staged end-to-end test can run. |

Pass signed coldkey bindings into the operator with
`--snapshot-claim-binding`, or export accepted signed bindings from the claims
API with `--snapshot-claim-bindings-url "$SOTA_CLAIMS_API_URL/api/v1/base/genesis/bindings"`.
For API export, load the claims API admin token as
`SOTA_BASE_INDEXER_ADMIN_TOKEN`; the operator keeps a fallback for the older
`SOTA_INDEXER_ADMIN_TOKEN` local-script name.
For Base Sepolia, the preferred path is the scheduled genesis batch publisher:
`base-sota-genesis-publisher.timer` runs
`scripts/run_sota_base_genesis_batch_publisher_once.sh` roughly every 10 minutes.
It reads accepted unrooted bindings from the claims API, publishes one genesis
root for the batch, imports the finalized claim artifact, and marks those
bindings as included so they are not republished.

Ongoing autoresearch emissions use a separate scheduled publisher:
`base-sota-emission-publisher.timer` runs
`scripts/run_sota_base_emission_batch_publisher_once.sh` roughly every 10
minutes. It checks the coordinator for accepted self-validation emission roots,
skips the latest root if the claims API already indexes that Merkle root,
publishes a new emission root when needed, finalizes the claim artifact with
the emitted `root_id`, and imports it into the claims API. The wrapper loads
the indexer admin token, root-publisher key, and deployer key from AWS Secrets
Manager at runtime.
The aggregate release-status command fails `testnet_publisher_timers` if these
user-systemd timers are not active/enabled or if the last oneshot service result
is not success.

If Base Sepolia is still using the old seeded demo genesis artifact,
`sota_base_release_status.py` marks
`testnet_snapshot_genesis` red and reports the accepted signed binding count
from local exported files and, when configured, the public claims API. Ongoing
emissions still come from accepted self-validation evidence.

## Base Sepolia Blocker Gate

Before running deployment, ask the read-only blocker gate what still prevents a
nontechnical browser-wallet test. It checks AWS identity, signer/test-wallet
gas, service DNS, Base Sepolia RPC chain ID, the generated deployment
artifacts, the service deployment pack, the source-based App Runner pack, and
the public readiness file. It does not deploy contracts, sign messages,
broadcast transactions, read secret values, or touch production Bittensor.

```bash
cd /home/mekaneeky/repos/SN94-BitSota-live-docs
python3 scripts/sota_base_testnet_blockers.py \
  --aws-profile moonrocklab-frankfurt \
  --report-out /home/mekaneeky/repos/.sota-base-testnet/base-sota-testnet-blockers.json \
  --allow-blocked
```

Use the generated JSON report as the current blocker list. Green means the
operator can continue to the guarded rehearsal and browser-wallet smoke. Red
means do not invite a nontechnical tester yet. The default host checks are:

- `claims-test.bitsota.com`
- `claims-api-test.bitsota.com`
- `coordinator-test.bitsota.com`
- `attestation-test.bitsota.com`
- `root-publisher-test.bitsota.com`

The default gas checks are:

- `gas_deployer`: reads only the public `sota-address` tag from
  `base-sota/test/base-sepolia/deployer`, then checks that address has Base
  Sepolia ETH.
- `gas_root_publisher`: reads only the public `sota-address` tag from
  `base-sota/test/base-sepolia/root-publisher`, then checks that address has
  Base Sepolia ETH.
- `gas_test_wallet`: included by the end-to-end operator when a test wallet is
  configured, so browser-wallet claim smoke cannot start with a wallet that has
  no Base Sepolia gas.

If any of these are red, fund the listed public address with Base Sepolia test
ETH and rerun the blocker gate. Do not paste private keys into the blocker
command; it only needs public addresses and approved secret-handle tags.
If AWS SSO is temporarily expired, the blocker gate may reuse cached public
addresses from prior funding or blocker reports so the gas checks still name
the address to fund. The AWS identity check remains red until the approved
profile is authenticated again.

Override a host only when the public testnet URL plan changes:

```bash
python3 scripts/sota_base_testnet_blockers.py \
  --host claims_ui=https://new-claims-test.example.com \
  --host claims_api=https://new-claims-api-test.example.com \
  --allow-blocked
```

The end-to-end operator command forwards its configured service URLs into both
the blocker gate and AWS inventory. If `bitsota.com` DNS is not delegated yet,
use direct public HTTPS service URLs, including App Runner URLs. This only
removes the custom-DNS dependency; browser smoke still has to prove the
services are configured for Base Sepolia, expose real claim artifacts, and
return unsigned Base Sepolia claim transactions.

If `bitsota.com` is delegated outside the AWS account used for App Runner,
record that explicitly instead of requiring a Route53 hosted zone in this
account:

```bash
python3 scripts/sota_base_testnet_operator.py \
  --external-dns-owner "Cloudflare bitsota.com production account" \
  --allow-blocked
```

## Base Sepolia AWS Inventory

Run the read-only AWS inventory after authenticating with the approved testnet
profile. It checks whether the account has explicit Base SOTA App Runner
services, a `bitsota.com` Route53 hosted zone or an explicit non-Route53 DNS
plan, optional Base SOTA/claims ECR repositories, and Base Sepolia/Base SOTA
secret handles. It records secret names and ARNs only; it never reads secret
values.

```bash
cd /home/mekaneeky/repos/SN94-BitSota-live-docs
python3 scripts/sota_base_testnet_aws_inventory.py \
  --aws-profile moonrocklab-frankfurt \
  --external-dns-owner "Cloudflare bitsota.com production account" \
  --out /home/mekaneeky/repos/.sota-base-testnet/base-sota-testnet-aws-inventory.json \
  --allow-blocked
```

Green means the public AWS side is named and discoverable. Red means do not
invite a nontechnical tester to the Base Sepolia path yet. Yellow only appears
for optional ECR repository discovery when the deployment plan might use source
deploys instead of containers. Do not use `--external-dns-owner` to skip DNS
work; the blocker gate and browser smoke still verify that the actual public
URLs resolve and serve the Base Sepolia app.

## Base Sepolia Funding Gate

Run the read-only funding gate before deployment or browser-wallet smoke. It
reads only AWS identity, public `sota-address` tags on the approved signer
secret handles, and Base Sepolia native balances. It never reads private keys,
signs messages, broadcasts transactions, or touches Base mainnet.

```bash
cd /home/mekaneeky/repos/SN94-BitSota-live-docs
python3 scripts/sota_base_testnet_funding.py \
  --aws-profile moonrocklab-frankfurt \
  --report-out /home/mekaneeky/repos/.sota-base-testnet/base-sota-testnet-funding.json \
  --allow-blocked
```

Green means the deployer, root publisher, and tester wallet all meet the
minimum Base Sepolia ETH balance for their role. The defaults are `0.020 ETH`
for the deployer, `0.005 ETH` for the root publisher, and `0.005 ETH` for the
tester wallet. Red means fund the listed public address with the displayed
additional amount and rerun this gate. The report includes BaseScan address
links and the official Base network faucet documentation URL. Use native Base
Sepolia ETH only; never fund these testnet roles with Base mainnet ETH.
If AWS SSO is temporarily expired, the script may reuse cached public addresses
from existing funding or blocker reports so the funding checklist remains
actionable. The AWS identity check still remains red until the approved profile
is authenticated again.

Override the minimum only when the operator explicitly accepts a different gas
budget:

```bash
python3 scripts/sota_base_testnet_funding.py \
  --aws-profile moonrocklab-frankfurt \
  --min-balance deployer=0.030 \
  --min-balance root_publisher=0.008 \
  --min-balance test_wallet=0.006 \
  --report-out /home/mekaneeky/repos/.sota-base-testnet/base-sota-testnet-funding.json \
  --allow-blocked
```

## Source App Runner Pack

Use source-based App Runner first. It uses the existing App Runner GitHub
connection and does not require an App Runner ECR access role. The service pack
writes templates under `.sota-base-testnet/apprunner/`; render them into
AWS-ready inputs with:

```bash
cd /home/mekaneeky/repos/SN94-BitSota-live-docs
python3 scripts/sota_base_testnet_apprunner_source_pack.py \
  --aws-profile moonrocklab-frankfurt \
  --region eu-central-1 \
  --allow-blocked
```

This writes `.sota-base-testnet/base-sota-testnet-apprunner-source-pack.json`
and rendered create-service inputs under `.sota-base-testnet/apprunner-source/`.
The rendered files replace `SOTA_APPRUNNER_CONNECTION_ARN` with the approved
App Runner GitHub connection and set `AppRunnerReadSecrets` as the runtime
instance role for services that read Secrets Manager handles. Green means the
rendered files are ready for `aws apprunner create-service`. Yellow means the
current local service code has not been committed and pushed to the configured
GitHub branches yet. In that case, read the report's `source_publication`
section; it lists the dirty paths and the deployment-relevant subset for each
service so operators can publish only the Base SOTA service changes.

The ECR/container pack is optional. Use it only if an existing App Runner ECR
access role ARN is provided and `iam:PassRole` works for that role.

## Base Sepolia Fresh Tester Prep

Use this when the next nontechnical tester has a known MetaMask test wallet and
a signed Bittensor coldkey snapshot binding for that wallet. It seeds real
public self-validation evidence, tops the wallet up with Base Sepolia test ETH
from the local faucet wallet if needed, publishes/imports the snapshot genesis
root and emission root, runs browser smoke, and refreshes release status plus
the tester handoff. It also refreshes the website repo's public Base Sepolia
JSON artifacts locally so the static readiness and claim-artifact files match
the current wallet/root cycle.

```bash
cd /home/mekaneeky/repos/SN94-BitSota-live-docs
python3 scripts/sota_prepare_fresh_testnet_tester.py
```

The command now fails fast unless `--reward-key-file` and
`--snapshot-claim-binding` are supplied. The binding's `reward_address` must
match the wallet file. Build the binding message with
the Genesis binding panel in the claims UI, or
`scripts/sota_sign_snapshot_binding.py` if the holder signs from a local
Bittensor wallet. Then rerun prep with the signed binding. The command prints
only the reward address, claim URL, handoff path, and next operator action; it
does not print private keys.

After the command succeeds, commit and push the refreshed public JSON artifacts
in the website repo so the App Runner claims UI serves the current readiness,
seed, genesis claim, and emission claim files. Do not commit wallet key JSON or
private key material.

## Base Sepolia Self-Validation Seed

Before building emission claim artifacts for a fresh public tester, seed real
self-validation evidence on the Base Sepolia autoresearch coordinator. This
creates a test-only task, posts a signed miner submission with EVM reward
delegation to the tester reward wallet, records the three peer evaluations,
builds the next SOTA emission root, and writes the evidence bundle used by the
operator.

```bash
cd /home/mekaneeky/repos/SN94-BitSota-live-docs
python3 scripts/sota_seed_testnet_autoresearch.py \
  --reward-key-file /secure/artifacts/fresh-claim-wallet.json \
  --evidence-out /secure/artifacts/base-sota-testnet-emission-evidence-fresh-public.json \
  --report-out /secure/artifacts/base-sota-testnet-autoresearch-seed.json \
  --require-single-claim
```

The script reads the autoresearch admin token from the approved AWS Secrets
Manager handle unless `SOTA_AUTORESEARCH_ADMIN_TOKEN` is already set. It does
not print the reward wallet private key and does not touch production
Bittensor, Base mainnet, or production TAO.

The fresh-tester prep command above wraps this seed step and the guarded
operator run. Use this lower-level command only when debugging or reviewing a
specific seed artifact.

## Base Sepolia Fresh Emission Tester Prep

Use this when you need a new public testnet mining/emission claim without
running a real holder genesis claim. It creates or loads a testnet reward
wallet, seeds real autoresearch self-validation evidence, funds that wallet
with Base Sepolia ETH if needed, publishes/imports the exact emission epoch,
and verifies the claims API returns unsigned emission calldata for MetaMask.
It does not create a genesis allocation and does not test a real holder claim.

```bash
cd /home/mekaneeky/repos/SN94-BitSota-live-docs
python3 scripts/sota_prepare_fresh_testnet_emission_tester.py
```

The report is written to
`/home/mekaneeky/repos/.sota-base-testnet/base-sota-fresh-emission-tester.json`
and the tester handoff shows it under `Fresh Emission Tester Prep`. The command
does not print the testnet private key; give wallet access out of band only to
the person doing the MetaMask claim. Use
`scripts/sota_prepare_fresh_testnet_tester.py` instead when the tester also has
a signed snapshot coldkey binding for genesis.

## Base Sepolia Operator Run

Use the operator command when you want the whole public testnet path attempted
in order. It generates the service pack, source App Runner pack, funding
report, blocker report, AWS inventory, deployment manifest/env from a compact
deployment or fresh deploy, snapshot genesis artifacts from signed coldkey
bindings, emission artifacts from real autoresearch evidence, root-publish
requests/results, finalized claim artifacts, optional
indexer import, browser smoke, and release status.

Read-only/current-blocker run:

```bash
cd /home/mekaneeky/repos/SN94-BitSota-live-docs
python3 scripts/sota_base_testnet_operator.py \
  --aws-profile moonrocklab-frankfurt \
  --allow-blocked
```

Full operator run with an existing compact Base Sepolia contract deployment:

```bash
python3 scripts/sota_base_testnet_operator.py \
  --aws-profile moonrocklab-frankfurt \
  --deployment /secure/artifacts/base-sepolia-compact-deployment.json \
  --emission-evidence /secure/artifacts/base-sota-emission-evidence.json \
  --snapshot-dir /mnt/4tb/tao_fork_snapshot \
  --snapshot-claim-binding "$SOTA_SNAPSHOT_CLAIM_BINDING" \
  --test-wallet-address "$SOTA_TEST_WALLET_ADDRESS" \
  --test-old-coldkey "$SOTA_TEST_OLD_COLDKEY" \
  --default-lane-id "$NEXT_PUBLIC_SOTA_DEFAULT_LANE_ID" \
  --build-website \
  --allow-blocked
```

Full operator run that deploys contracts from an approved AWS Secrets Manager
handle:

```bash
python3 scripts/sota_base_testnet_operator.py \
  --aws-profile moonrocklab-frankfurt \
  --deploy \
  --private-key-secret-id "$SOTA_DEPLOYER_PRIVATE_KEY_SECRET_ID" \
  --emission-evidence /secure/artifacts/base-sota-emission-evidence.json \
  --snapshot-dir /mnt/4tb/tao_fork_snapshot \
  --snapshot-claim-binding "$SOTA_SNAPSHOT_CLAIM_BINDING" \
  --test-wallet-address "$SOTA_TEST_WALLET_ADDRESS" \
  --test-old-coldkey "$SOTA_TEST_OLD_COLDKEY" \
  --default-lane-id "$NEXT_PUBLIC_SOTA_DEFAULT_LANE_ID" \
  --build-website \
  --allow-blocked
```

The operator loads that secret into the child process as
`SOTA_DEPLOYER_PRIVATE_KEY` and redacts the child command in its report. The
secret may be a raw private key string or a JSON object with
`SOTA_DEPLOYER_PRIVATE_KEY`, `private_key`, `deployer_private_key`, or the key
named by `--private-key-secret-json-key`.

To execute the state-changing parts after review, add:

```bash
--broadcast-roots \
--root-publisher-private-key-secret-id "$SOTA_ROOT_PUBLISHER_PRIVATE_KEY_SECRET_ID" \
--import-artifacts
```

`--broadcast-roots` requires `SOTA_ROOT_PUBLISHER_PRIVATE_KEY` in the process
environment or `--root-publisher-private-key-secret-id` pointing at an approved
AWS Secrets Manager handle. `--import-artifacts` posts finalized genesis and
emission claim artifacts into the configured testnet claims API. The operator
report is written to `base-sota-testnet-operator-run.json`; it stays red or
yellow until every generated report is green, both roots have emitted on-chain
root IDs, the snapshot genesis artifact includes TAO/alpha rao credits, the
claim artifacts are imported, browser smoke is green, and
MetaMask claim transaction evidence is recorded.

`SOTA_TEST_WALLET_ADDRESS` must be the same Base wallet named in the signed
snapshot binding's `reward_address`. The operator verifies the binding
signature and claim amount; release status verifies the finalized artifact
came from the locked snapshot and contains the alpha-credit fields.

## Base Sepolia Service Pack

Generate the service pack before assigning public testnet infrastructure work.
It writes one JSON/Markdown/HTML bundle with the service list, repo/cwd,
build/run commands, DNS hosts, ports, health URLs, public env keys, secret
handle references, dependencies, and Base-Sepolia-only safeguards.

```bash
cd /home/mekaneeky/repos/SN94-BitSota-live-docs
python3 scripts/sota_base_testnet_service_pack.py \
  --json-out /home/mekaneeky/repos/.sota-base-testnet/base-sota-testnet-service-pack.json \
  --markdown-out /home/mekaneeky/repos/.sota-base-testnet/base-sota-testnet-service-pack.md \
  --html-out /home/mekaneeky/repos/.sota-base-testnet/base-sota-testnet-service-pack.html
```

When a filled deployment manifest exists, pass it explicitly:

```bash
python3 scripts/sota_base_testnet_service_pack.py \
  --manifest /secure/artifacts/base-sepolia-deployment-manifest.json \
  --env-file /secure/artifacts/base-sota.env.testnet \
  --claims-ui https://claims-test.bitsota.com \
  --claims-api https://claims-api-test.bitsota.com \
  --coordinator https://coordinator-test.bitsota.com \
  --attestation https://attestation-test.bitsota.com \
  --root-publisher https://root-publisher-test.bitsota.com \
  --claim-artifacts https://claims-test.bitsota.com/base-sota-testnet-seed-artifacts-finalized.json \
  --monitoring https://monitoring-test.bitsota.com \
  --readiness-url https://claims-test.bitsota.com/base-sota-testnet-readiness.json
```

The service pack is read-only. It does not deploy services, sign transactions,
publish roots, broadcast transactions, or touch production Bittensor/Base
mainnet. The blocker gate now requires
`base-sota-testnet-service-pack.json` and treats `deployment_ready=false` as a
red testnet blocker. Before the deployment manifest and env file exist, the
pack is expected to stay yellow and the blocker gate is expected to stay red.

## Base Sepolia Claim Artifact Building

Testnet claims have a two-phase artifact path. First build root artifacts and
pending claim templates from the deployed manifest plus real autoresearch
emission evidence:

```bash
cd /home/mekaneeky/repos/SN94-BitSota-live-docs
python3 scripts/sota_base_testnet_seed_artifacts.py build \
  --manifest /secure/artifacts/base-sepolia-deployment-manifest.json \
  --emission-evidence /secure/artifacts/base-sota-emission-evidence.json \
  --test-wallet-address "$SOTA_TEST_WALLET_ADDRESS" \
  --test-old-coldkey "$SOTA_TEST_OLD_COLDKEY" \
  --lane-id "$NEXT_PUBLIC_SOTA_DEFAULT_LANE_ID" \
  --out-dir /secure/artifacts/base-sepolia-rehearsal
```

The builder recomputes emission leaves, proofs, totals, and the Merkle root. It
refuses emission evidence unless each claim has accepted self-validation
consensus with the configured committee minimums. It does not sign, publish, or
broadcast.

For release readiness, do not publish the genesis artifact produced by this
seed builder. Use the operator with `--snapshot-claim-binding` so genesis is
rebuilt by `sota_snapshot_claim_bridge.py` from the locked TAO plus alpha
snapshot. The release-status gate rejects the old seeded genesis artifact.

Next publish the generated genesis and emission root artifacts using the root
publisher. The build report prints dry-run and broadcast commands. After both
broadcasts succeed, finalize the claim artifacts with the emitted on-chain root
IDs:

```bash
python3 scripts/sota_base_testnet_seed_artifacts.py finalize \
  --build-report /secure/artifacts/base-sepolia-rehearsal/base-sota-testnet-seed-artifacts.json \
  --genesis-publish-result /secure/artifacts/base-sepolia-rehearsal/base-sota-testnet-genesis-root-publish-result.json \
  --emission-publish-result /secure/artifacts/base-sepolia-rehearsal/base-sota-testnet-emission-root-publish-result.json \
  --out-dir /secure/artifacts/base-sepolia-rehearsal
```

The finalized report prints the exact `curl` commands for importing
`base-sota-testnet-genesis-claim-artifact.json` and
`base-sota-testnet-emission-claim-artifact.json` into the claims indexer. Do not
import the pending templates; they intentionally lack root IDs.

For normal Base Sepolia autoresearch emissions, prefer the scheduled publisher
over this manual seed-artifact path:

```bash
scripts/run_sota_base_emission_batch_publisher_once.sh
```

Dry-run checks that should not broadcast:

```bash
python3 scripts/sota_base_emission_batch_publisher.py --once --json
```

The publisher validates the coordinator evidence, recomputes the Merkle root,
requires accepted self-validation consensus for each claim, and skips roots the
claims API already indexes.

## Base Sepolia Root Publisher

Root publication must use the guarded wrapper, not an ad hoc Web3 shell. The
wrapper builds a dry-run publish transaction by default:

```bash
cd /home/mekaneeky/repos/SN94-BitSota-live-docs
python3 scripts/sota_base_publish_root.py \
  --manifest /secure/artifacts/base-sepolia-deployment-manifest.json \
  --root-artifact /secure/artifacts/base-sepolia-root-artifact.json \
  --kind emission \
  --nonce "$SOTA_ROOT_NONCE" \
  --out /secure/artifacts/base-sepolia-root-publish-request.json
```

It refuses Base mainnet chain ID `8453`, requires nonzero Merkle root, budget,
policy hash, attestation hash, and nonce, and writes calldata without signing.
To broadcast on Base Sepolia, load `SOTA_ROOT_PUBLISHER_PRIVATE_KEY` from the
approved secret store and pass `--broadcast`. Do not put that private key in
the manifest, env file, service pack, docs, Linear, or shell history. A
successful broadcast decodes the `RootPublished` event and records the emitted
`root_id`; use that result when finalizing claim artifacts.

## Base Sepolia Preflight

Run the read-only preflight before asking a reviewer to open the Base Sepolia
claims page. It checks the manifest shape, chain ID, RPC, deployed contract
addresses, service URLs, browser public env, and public test-wallet funding.
It does not broadcast transactions, sign messages, deploy contracts, or touch
production Bittensor.

The preferred operator path is the guarded rehearsal script. With an existing
compact deployment artifact:

```bash
cd /home/mekaneeky/repos/SN94-BitSota-live-docs
python3 scripts/sota_base_testnet_rehearsal.py \
  --deployment /secure/artifacts/base-sepolia-compact-deployment.json \
  --artifacts-dir /secure/artifacts/base-sepolia-rehearsal \
  --claims-ui-url https://claims-test.bitsota.com \
  --claims-ui-health-url https://claims-test.bitsota.com/health \
  --indexer-api-url https://claims-api-test.bitsota.com \
  --indexer-api-health-url https://claims-api-test.bitsota.com/health \
  --root-publisher-url https://root-publisher-test.bitsota.com \
  --root-publisher-health-url https://root-publisher-test.bitsota.com/health \
  --attestation-builder-url https://attestation-test.bitsota.com \
  --attestation-builder-health-url https://attestation-test.bitsota.com/health \
  --monitoring-url https://monitoring-test.bitsota.com \
  --autoresearch-api-url https://coordinator-test.bitsota.com \
  --test-wallet-address "$SOTA_TEST_WALLET_ADDRESS" \
  --test-old-coldkey "$SOTA_TEST_OLD_COLDKEY" \
  --build-website
```

To have the rehearsal script deploy the contracts first, add `--deploy` and
load `SOTA_DEPLOYER_PRIVATE_KEY` from the approved testnet secret store in the
process environment. The script refuses Base mainnet RPC chain ID `8453` and
does not print the private key.

```bash
cd /home/mekaneeky/repos/SN94-BitSota-live-docs
python3 scripts/sota_base_testnet_rehearsal.py \
  --deploy \
  --artifacts-dir /secure/artifacts/base-sepolia-rehearsal \
  --claims-ui-url https://claims-test.bitsota.com \
  --indexer-api-url https://claims-api-test.bitsota.com \
  --root-publisher-url https://root-publisher-test.bitsota.com \
  --attestation-builder-url https://attestation-test.bitsota.com \
  --monitoring-url https://monitoring-test.bitsota.com \
  --autoresearch-api-url https://coordinator-test.bitsota.com \
  --test-wallet-address "$SOTA_TEST_WALLET_ADDRESS" \
  --test-old-coldkey "$SOTA_TEST_OLD_COLDKEY"
```

The rehearsal writes:

- `base-sepolia-compact-deployment.json`, when `--deploy` is used;
- `base-sepolia-deployment-manifest.json`;
- `base-sota.env.testnet`;
- `base-sota-testnet-readiness.json`;
- an optional JSON report when `--report-out` is supplied.

Publish `base-sota-testnet-readiness.json` at the URL used by
`NEXT_PUBLIC_SOTA_READINESS_URL`. The claims website reads this public file and
shows the tester whether Base Sepolia is ready for browser-wallet smoke. The
file contains preflight status and public check details only; it must not
contain private keys, RPC tokens, admin tokens, database URLs, or raw secret
values.

After the contract deploy helper writes its compact deployment output, convert
that output into the full operations manifest and public/service env file:

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
  --test-wallet-address "$SOTA_TEST_WALLET_ADDRESS" \
  --test-old-coldkey "$SOTA_TEST_OLD_COLDKEY"
```

This adapter only writes public addresses, public URLs, and secret-handle
references inherited from the template. Do not put deployer private keys,
mnemonics, RPC tokens, admin tokens, or database URLs into the generated env
file.

```bash
cd /home/mekaneeky/repos/SN94-BitSota-live-docs
python3 scripts/sota_base_testnet_preflight.py \
  /secure/artifacts/base-sepolia-deployment-manifest.json \
  --env-file /secure/artifacts/base-sota.env.testnet
```

The template is expected to be red until real Base Sepolia contracts and
services are deployed. A reviewer-ready testnet run needs this command to be
green against the filled deployment manifest and real testnet env. Yellow means
the check was skipped or needs operator evidence. Red means do not invite a
nontechnical tester yet.

## Base Sepolia Browser Smoke

After the blocker report and preflight are green, run the read-only public
browser smoke before handing the testnet to a nontechnical MetaMask tester. It
loads the public claims page, requires Base Sepolia UI copy and readiness
wording, checks the public claims API/indexer, verifies the snapshot old
coldkey/test-wallet genesis claim, verifies the self-validated emission
claim, checks unsigned claim calldata for both claim types, and confirms
autoresearch self-validation evidence is public.

```bash
cd /home/mekaneeky/repos/SN94-BitSota-live-docs
python3 scripts/sota_base_testnet_browser_smoke.py \
  --report-out /secure/artifacts/base-sepolia-rehearsal/base-sota-testnet-browser-smoke.json
```

The script is read-only. It does not deploy contracts, sign messages, broadcast
transactions, or touch production Bittensor. A green report means the public UI
and APIs are ready for the human MetaMask step: connect the tester wallet,
submit the genesis claim, then submit the mined-emission claim. Red means do
not invite a nontechnical tester yet. The default timeout is long enough for a
fresh App Runner deploy to sync the public indexer before eligibility and
calldata checks run.

The generated testnet env must include:

- `SOTA_TEST_WALLET_ADDRESS`
- `SOTA_TEST_OLD_COLDKEY`
- `SOTA_TEST_EPOCH`
- `NEXT_PUBLIC_SOTA_DEFAULT_LANE_ID`

## Base Sepolia Claim Transaction Evidence

After the public browser-smoke report is green, ask the human tester to connect
the tester MetaMask wallet, switch to Base Sepolia, submit the genesis claim,
then submit the mined-emission claim. Record both transaction hashes.

Verify those hashes with the read-only evidence verifier:

```bash
cd /home/mekaneeky/repos/SN94-BitSota-live-docs
python3 scripts/sota_base_claim_tx_evidence.py \
  --environment testnet \
  --genesis-tx "$SOTA_TESTNET_GENESIS_CLAIM_TX" \
  --emission-tx "$SOTA_TESTNET_EMISSION_CLAIM_TX" \
  --report-out /secure/artifacts/base-sepolia-rehearsal/base-sota-claim-tx-evidence.json
```

The verifier reads the generated manifest/env, queries the configured Base
Sepolia RPC, and checks:

- both transactions succeeded;
- both transactions came from the tester wallet;
- genesis targeted `GenesisClaimDistributor`;
- emission targeted `EmissionClaimDistributor`;
- both transactions used the expected claim function selector;
- both distributors emitted the expected claim event and `ClaimRecorded`;
- the SOTA token emitted a positive `Transfer` to the tester wallet;
- the vault emitted `SOTAReleased` to the tester wallet;
- the tester wallet has a positive SOTA balance after the claims.

Release status also checks that genesis is a snapshot artifact with TAO and
alpha rao credits, and compares this report with the current finalized
artifacts. If a fresh root cycle changes the tester wallet or claim amounts,
old claim transaction evidence is marked stale and the `claim_tx_evidence` gate
returns red until the current wallet submits both claims.

The verifier does not deploy contracts, sign messages, broadcast transactions,
or touch production Bittensor. A green
`base-sota-claim-tx-evidence.json` is the machine-readable proof that the human
MetaMask claim flow completed.

For dry documentation review without network access:

```bash
python3 scripts/sota_base_testnet_preflight.py <manifest.json> \
  --env-file <base-sota.env.testnet> \
  --offline \
  --allow-blocked
```

## Evidence Record Template

Use this shape for every deployment or readiness note:

```text
Environment: local-demo or base-sepolia
Service or contract:
Owner:
Source repo:
Source branch:
Commit SHA:
Manifest path/version:
Public URL or explorer link:
Raw service URL, if approved for test evidence:
Chain ID:
Contract address, if applicable:
Secret handles used:
Health check or transaction hash:
Rollback or pause owner:
Known blockers:
Next update:
```

Never include private keys, seed phrases, RPC tokens, admin tokens, deployer
secret values, or unapproved production raw URLs in the evidence record.

## Do Not Publish Yet

Do not publish public instructions that claim any of the following until the
operator records evidence:

- Base Sepolia deployment is live;
- Base mainnet deployment is live;
- production SOTA claims are open;
- a contract audit is complete;
- a reward, payout, or claim date is guaranteed.

## Escalation

| Topic | Owner path |
| --- | --- |
| Local demo launcher, website, or URL printing | Full-stack product engineering |
| Backend seeding, validation evidence, or claim roots | Autoresearch backend engineering |
| Contract deployment, ownership, pausing, or audit scope | Smart contract security engineering and SRE |
| Indexer/API deployment, health checks, rollback, or optional monitoring | SRE / DevOps engineering |
| Public docs wording and release readiness | Technical docs / product release |
| Final QA evidence and staged end-to-end approval | QA / release engineering |
