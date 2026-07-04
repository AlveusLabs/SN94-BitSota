# Deployment Manifest

Base Sepolia deployment evidence must be reviewable from one manifest. Use this
template before writing public testnet instructions:

```text
docs/base/manifests/base-sepolia-deployment-manifest.template.json
```

The deployment owner should copy it to a dated or final manifest when Base
Sepolia contracts and services are actually deployed. Keep the template in the
docs repo so reviewers can see the required shape before any addresses exist.

The Pool deploy helper may emit useful deployment JSON, but that helper output
is not the canonical review manifest unless it is normalized into this template.
Use this template for the artifact reviewers inspect.

## Generate From Contract Deployment

Before deploying contracts, create or verify the Base Sepolia secret handles.
This bootstrap creates only values that can be safely generated for testnet:
the deployer key, root-publisher key, public RPC URL handle, and admin tokens.
It does not create placeholder autoresearch database URLs or monitoring keys.

```bash
cd /home/mekaneeky/repos/SN94-BitSota-live-docs
python3 scripts/sota_base_testnet_secrets.py create \
  --aws-profile moonrocklab-frankfurt \
  --region eu-central-1 \
  --allow-blocked
```

The claims indexer does not need a managed database secret for the testnet path:
it uses a local SQLite read model, imports finalized public claim artifacts from
`SOTA_BASE_CLAIM_ARTIFACT_URLS` on startup, and syncs on-chain claim events from
Base Sepolia RPC. The autoresearch coordinator still needs a real database URL
for public multi-user competitions.

Set `SOTA_BASE_CLAIM_ARTIFACT_REQUIRED=true` on the public indexer so it fails
closed if the finalized genesis/emission claim artifacts are not reachable. Set
`SOTA_BASE_INDEXER_ADMIN_TOKEN` from the approved secret handle; indexer
mutation endpoints for artifact import, manual events, and sync reject requests
without this token. The indexer includes embedded minimal event ABIs for the SOTA
contracts, so App Runner does not need a local Pool checkout just to sync Base
Sepolia claim/root/lane events.

The report is written to
`.sota-base-testnet/base-sota-testnet-secret-handles.json`. Fund the deployer
address shown in that report with Base Sepolia ETH before running the contract
deployment operator. Keep private keys inside AWS Secrets Manager; the report
contains handles and addresses only.

For an end-to-end operator run, prefer the guarded rehearsal script:

```bash
cd /home/mekaneeky/repos/SN94-BitSota-live-docs
python3 scripts/sota_base_testnet_rehearsal.py \
  --deployment /secure/artifacts/base-sepolia-compact-deployment.json \
  --artifacts-dir /secure/artifacts/base-sepolia-rehearsal \
  --claims-ui-url https://claims-test.bitsota.com \
  --indexer-api-url https://claims-api-test.bitsota.com \
  --root-publisher-url https://root-publisher-test.bitsota.com \
  --attestation-builder-url https://attestation-test.bitsota.com \
  --monitoring-url https://monitoring-test.bitsota.com \
  --autoresearch-api-url https://coordinator-test.bitsota.com \
  --test-wallet-address "$SOTA_TEST_WALLET_ADDRESS"
```

The rehearsal script validates the RPC chain ID, generates the full manifest
and env file, runs the preflight, and can optionally run the claims website
production build with `--build-website`. It also writes
`base-sota-testnet-readiness.json`, the public status file consumed by the
claims website through `NEXT_PUBLIC_SOTA_READINESS_URL`.

After `/home/mekaneeky/repos/Pool/scripts/deploy_sota_base.py` emits its compact
deployment JSON, normalize it into the full review manifest and matching
public/service env file:

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
  --test-wallet-address "$SOTA_TEST_WALLET_ADDRESS"
```

The adapter rejects non-Base-Sepolia deployments and zero contract addresses.
It does not deploy, sign, publish roots, or broadcast transactions. The
generated env file is for public/service configuration only; keep signer keys,
mnemonics, RPC tokens, admin tokens, and database credentials in secret stores.

## Generate The Service Pack

After the manifest/env exists, generate the infrastructure service pack:

```bash
cd /home/mekaneeky/repos/SN94-BitSota-live-docs
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

This writes `base-sota-testnet-service-pack.json`,
`base-sota-testnet-service-pack.md`, and
`base-sota-testnet-service-pack.html`. It also writes App Runner create-service
input templates for the public web/API services under
`.sota-base-testnet/apprunner/`. Render those templates into AWS-ready
source-based App Runner inputs with:

```bash
python3 scripts/sota_base_testnet_apprunner_source_pack.py --allow-blocked
```

The rendered inputs live under `.sota-base-testnet/apprunner-source/` and the
report is `.sota-base-testnet/base-sota-testnet-apprunner-source-pack.json`.
This is the preferred public-service path because it uses the existing App
Runner GitHub connection and does not require a private ECR access role.

The service pack is the handoff for infrastructure agents: service commands,
repo/cwd, health URLs, DNS hosts, dependencies, public env keys, secret-handle
references, and the split between public App Runner services, controlled
operator workers, and static readiness artifacts. It is not deployment evidence
by itself. The blocker gate requires the service pack and source App Runner
pack, and remains red while the pack says `deployment_ready=false`.

The claim-artifact service entry points at
`scripts/sota_base_testnet_seed_artifacts.py`. It builds root artifacts from the
deployed manifest and actual autoresearch evidence, then finalizes importable
claim artifacts only after root-publish results contain emitted on-chain
`root_id` values. The root-publisher service entry points at
`scripts/sota_base_publish_root.py`. That wrapper dry-runs by default, refuses
Base mainnet, records the emitted `RootPublished` root ID after broadcast, and
only broadcasts when `--broadcast` is passed with
`SOTA_ROOT_PUBLISHER_PRIVATE_KEY` loaded from the approved secret store.

For the full ordered run, use `scripts/sota_base_testnet_operator.py`. It calls
the service pack, source App Runner pack, blocker gate, AWS inventory,
manifest/env rehearsal, seed artifact builder, root publisher, claim artifact
finalizer/importer, browser smoke, and release-status checks in one place. The
operator command still refuses to mark testnet ready while any required
generated report is red or yellow.

The public App Runner services are the claims UI, claims indexer/API, and
autoresearch coordinator. The genesis/emission claim artifact builder,
attestation builder, root publisher, and readiness publisher remain controlled
operator jobs until they have explicit service wrappers. Monitoring is optional
observability for the testnet path. Operator jobs must load signer/admin
material from secret handles at runtime and write JSON evidence back to the
Base Sepolia artifacts directory.

## Required Shape

The manifest fixes these sections:

- `environment`: must be `base-sepolia`.
- `chain`: Base Sepolia chain ID, chain name, explorer, public browser RPC URL,
  and RPC secret handle reference.
- `source`: branch and commit SHA for docs, contracts, claims UI, indexer/API,
  and attestation tools.
- `abi_bundle`: ABI version, source path, hash, and source commit.
- `deployer`: deployer address plus signer secret handle reference.
- `roles`: owner, pending owner, supply authority, emission authority, root
  publisher, pause guardian, and vault releaser records.
- `contracts`: SOTA token, vault, root registry, lane registry, genesis
  distributor, and emission distributor.
- `services`: claims UI, indexer/API, claim artifact builder, root publisher,
  attestation builder, and optional monitoring.
- `browser_safe`: values allowed in website config and public review notes.
- `secret_handles`: handle names only, never raw secret values.
- `rollback`: owner, pause owner role, rollback plan URL, and minimum recovery
  actions.
- `evidence_links`: deployment, verification, health, smoke, optional
  monitoring, and rollback evidence.

## Syntax Check

Validate the template or a filled deployment manifest with:

```bash
cd /home/mekaneeky/repos/SN94-BitSota-live-docs
python3 scripts/validate_base_sota_manifest.py docs/base/manifests/base-sepolia-deployment-manifest.template.json
```

The validator checks that the manifest stays Base Sepolia-only, includes every
contract and service needed by the claims UI, indexer/API, root publisher,
claim artifact builder, and attestation builder, and keeps secret handles
separate from browser-safe values.

## Testnet Preflight

After the syntax check passes, run the read-only preflight against the filled
Base Sepolia manifest and the testnet env file:

```bash
cd /home/mekaneeky/repos/SN94-BitSota-live-docs
python3 scripts/sota_base_testnet_preflight.py <base-sepolia-manifest.json> \
  --env-file /home/mekaneeky/repos/bitsota_website/docs/operations/base-sota.env.testnet.example
```

This checks the RPC chain ID, nonzero deployed contract addresses, bytecode at
those addresses, service health URLs, browser-safe env values, and the public
test-wallet address. It exits nonzero while any red check remains. A green
preflight is required before the Base Sepolia browser-wallet smoke can be
treated as nontechnical-reviewer ready.

## Secret Boundary

Do not add these values to the manifest, docs, Linear, screenshots, or public
evidence:

- private keys;
- mnemonics or seed phrases;
- RPC tokens;
- admin token values;
- database URLs with embedded credentials;
- production SN94, Pool, Autoresearch, App Runner, Base mainnet, or Bittensor
  mainnet config.

Only secret handles belong in `secret_handles`. Public Base Sepolia contract
addresses, source verification URLs, transaction hashes, service URLs, and
health-check links can be recorded after deployment.
