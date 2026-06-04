<section class="bitsota-hero compact">
  <p class="bitsota-kicker">API ROUTES</p>
  <h1>Pool Backend Routes</h1>
  <p class="bitsota-lede">Pool app and Merkle claim route inventory from the local Pool backend code.</p>
</section>

Source of truth:

```text
Pool/main.py
Pool/app/api/v1/miners.py
Pool/app/api/v1/tasks.py
Pool/app/api/v1/results.py
Pool/app/api/v1/disclaimer.py
Pool/app/services/onchain_runtime.py
Pool/scripts/merkle_claim_server.py
```

This is a route inventory, not a statement that every route is production-active
for current autoresearch mining. The current autoresearch coordinator remains
the source of truth for task claims, submissions, validator replay, and reward
snapshots.

## Surfaces

| Surface | Default base | Source |
| --- | --- | --- |
| Pool app | `/api/v1/...` plus root compatibility routes | `Pool/main.py` routers |
| Mounted Merkle claim app | `/claims/...` by default | `MERKLE_CLAIM_PATH_PREFIX`, default `/claims` |
| Standalone claim server | `/...` | `scripts/merkle_claim_server.py` when run directly |

Environment examples set `API_PORT=8434` for testnet and
`MERKLE_CLAIM_PATH_PREFIX=/claims`.

## Auth Classes

| Class | Headers | Used for |
| --- | --- | --- |
| Public | none | Health, leaderboard, SOTA, disclaimer, public recipient lookup, claim proof reads. |
| Pool wallet signed | `X-Key`, `X-Timestamp`, `X-Signature` | Miner registration, stats, Pool task leases/submissions, result verification, recipient self-lookup. |
| Recipient update signed | `X-Key`, `X-Timestamp`, `X-Signature` | Recipient coldkey update; signed message binds timestamp and recipient. |

Pool wallet auth signs `auth:{timestamp}`. Recipient update auth signs:

```text
recipient_coldkey:update:{timestamp}:{recipient_coldkey}
```

## Root And Disclaimer

| Method | Path | Auth | Purpose |
| --- | --- | --- | --- |
| `GET` | `/health` | Public | Pool health, tempo/window status, current block, and on-chain runtime status. |
| `GET` | `/api/v1/disclaimer` | Public | JSON disclaimer payload. |
| `GET` | `/api/v1/disclaimer.md` | Public | Markdown disclaimer. |
| `GET` | `/disclaimer.md` | Public | Root compatibility markdown disclaimer. |

## Miners

These routes are mounted with `/api/v1` except the root compatibility route
called out below.

| Method | Path | Auth | Purpose |
| --- | --- | --- | --- |
| `POST` | `/api/v1/miners/register` | Pool wallet signed | Register or refresh miner record. |
| `GET` | `/api/v1/miners/stats` | Pool wallet signed | Miner stats and pool summary. |
| `GET` | `/api/v1/miners/leaderboard` | Public | Top miners by reputation. |
| `PUT` | `/api/v1/miners/recipient-coldkey` | Recipient update signed | Update miner payout recipient coldkey. |
| `POST` | `/api/v1/miners/recipient-coldkey` | Recipient update signed | Compatibility update route. |
| `POST` | `/api/v1/miners/coldkey_address/update` | Recipient update signed | Compatibility update route under `/api/v1/miners`. |
| `POST` | `/coldkey_address/update` | Recipient update signed | Root compatibility update route. |
| `GET` | `/api/v1/miners/recipient-coldkey` | Pool wallet signed | Get recipient coldkey for signed miner hotkey. |
| `GET` | `/api/v1/miners/recipient-coldkey/{miner_hotkey}` | Public | Public recipient mapping lookup by hotkey. |

Recipient update bodies accept `recipient_coldkey` and legacy
`coldkey_address` aliases.

## Pool Task / Lease Routes

These routes are defined in the Pool app. They are lease/evolution/evaluation
routes, not the current autoresearch coordinator task-submission routes.

| Method | Path | Auth | Purpose |
| --- | --- | --- | --- |
| `GET` | `/api/v1/tasks/pending_evaluations` | Pool wallet signed | Check pending evaluation assignments for the signed miner. |
| `POST` | `/api/v1/tasks/request` | Pool wallet signed | Request an `evolve` or `evaluate` batch. |
| `POST` | `/api/v1/tasks/lease` | Pool wallet signed | Request a lease batch with evaluation algorithms, seed algorithms, and gossip. |
| `POST` | `/api/v1/tasks/{lease_id}/submit_lease` | Pool wallet signed | Submit lease evaluations, evolutions, and optional gossip. |
| `POST` | `/api/v1/tasks/{batch_id}/submit_evolution` | Pool wallet signed | Submit an evolved function for an active evolution assignment. |
| `POST` | `/api/v1/tasks/{batch_id}/submit_evaluation` | Pool wallet signed | Submit evaluations for an active evaluation assignment. |

Important request models:

| Model | Fields |
| --- | --- |
| `TaskRequest` | optional `task_type`. |
| `LeaseRequest` | optional `task_type`, `eval_batch_size`, `seed_batch_size`, `gossip_limit`, `sim_window_number`. |
| `LeaseSubmission` | `evaluations`, `evolutions`, optional `gossip`, `iterations`. |
| `EvolutionSubmission` | `evolved_function`, `parent_functions`. |

## Pool Results

| Method | Path | Auth | Purpose |
| --- | --- | --- | --- |
| `GET` | `/api/v1/results/` | Pool wallet signed | List verified results; supports `limit`, `task_type`, and `min_score`. |
| `GET` | `/api/v1/results/sota` | Public | Current SOTA threshold for `task_type`; default `cifar10_binary`. |
| `POST` | `/api/v1/results/verify/{result_id}` | Pool wallet signed | Mark a completed result as verified. |

## Mounted Merkle Claim Routes

When `MERKLE_CLAIM_ENABLED=true`, the Pool app mounts the Merkle claim app at
`MERKLE_CLAIM_PATH_PREFIX`, default:

```text
/claims
```

The table below shows mounted paths using that default prefix.

| Method | Path | Auth | Purpose |
| --- | --- | --- | --- |
| `GET` | `/claims/health` | Public | Claim app health. |
| `GET` | `/claims/epochs` | Public | Claimable/default epoch list. |
| `GET` | `/claims/epoch` | Public | Default epoch metadata. |
| `GET` | `/claims/epoch/{epoch_id}` | Public | Epoch metadata by id. |
| `GET` | `/claims/epoch/proof/{hotkey}` | Public | Default epoch proof package by hotkey. |
| `GET` | `/claims/epoch/claim/{hotkey}` | Public | Alias for default proof package. |
| `GET` | `/claims/epoch/{epoch_id}/proof/{hotkey}` | Public | Proof package for epoch and hotkey. |
| `GET` | `/claims/epoch/{epoch_id}/claim/{hotkey}` | Public | Alias for epoch proof package. |
| `POST` | `/claims/epoch/claim` | Public when simulation enabled | Simulate or record a claim against default epoch. |
| `POST` | `/claims/epoch/{epoch_id}/claim` | Public when simulation enabled | Simulate or record a claim against selected epoch. |
| `GET` | `/claims/ledger` | Public when simulation enabled | Default epoch simulated ledger. |
| `GET` | `/claims/ledger/{epoch_id}` | Public when simulation enabled | Simulated ledger by epoch. |

Claim bodies use:

```text
index
hotkey
recipient_coldkey
amount_units
proof
```

If claim simulation is disabled, `GET` proof package routes still serve proofs,
but `POST` claim and ledger routes are not registered by the claim app.

## Standalone Merkle Claim Server

If `scripts/merkle_claim_server.py` is run directly, the same claim routes are
available without the `/claims` mount prefix:

```text
/health
/epochs
/epoch
/epoch/{epoch_id}
/epoch/proof/{hotkey}
/epoch/claim/{hotkey}
/epoch/{epoch_id}/proof/{hotkey}
/epoch/{epoch_id}/claim/{hotkey}
/epoch/claim
/epoch/{epoch_id}/claim
/ledger
/ledger/{epoch_id}
```

## Generated FastAPI Docs

The Pool app and claim app use FastAPI defaults, so OpenAPI documentation is
available when the deployment does not disable it:

```text
/openapi.json
/docs
/redoc
```
