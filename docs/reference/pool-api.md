# Pool API

The Pool API is a FastAPI service. When running locally, the OpenAPI UI is available at:

- `http://127.0.0.1:8434/docs`

## Auth headers

Most miner endpoints require:

- `X-Key` (miner hotkey)
- `X-Timestamp` (unix seconds)
- `X-Signature` (SR25519 signature of the message `auth:{X-Timestamp}`)

The timestamp must be within 5 minutes of server time.

## Endpoints

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/health` | none | Liveness and current window info |
| POST | `/api/v1/miners/register` | miner | Register or refresh a miner |
| GET | `/api/v1/miners/stats` | miner | Miner stats and pool summary |
| GET | `/api/v1/miners/leaderboard` | none | Top miners by reputation |
| GET | `/api/v1/tasks/pending_evaluations` | miner | Check pending evaluation assignments |
| POST | `/api/v1/tasks/request` | miner | Request a simple evolve or evaluate batch |
| POST | `/api/v1/tasks/lease` | miner | Request a lease with evaluate, seed, and gossip |
| POST | `/api/v1/tasks/{lease_id}/submit_lease` | miner | Submit a lease result bundle |
| POST | `/api/v1/tasks/{batch_id}/submit_evolution` | miner | Submit an evolved algorithm for a batch |
| POST | `/api/v1/tasks/{batch_id}/submit_evaluation` | miner | Submit evaluations for a batch |
| GET | `/api/v1/results` | miner | List verified results |
| GET | `/api/v1/results/sota` | none | Current pool SOTA for a task type |
| POST | `/api/v1/results/verify/{result_id}` | miner | Mark a completed result as verified |
| GET | `/api/v1/monitor/summary` | monitor | Aggregated monitor summary |

## Contract integration note

The Pool API itself does not expose direct endpoints for:
- `publish_epoch`
- `challenge_epoch`
- `claim`

Those are handled by Pool scripts:
- `scripts/consensus_daemon.py` for publish/verify and optional on-chain bridge
- `scripts/merkle_claim_server.py` for off-chain claim simulation/proof serving

## Examples

### Register

`POST /api/v1/miners/register` with auth headers only. Response includes current reputation and timestamps.

### Lease work

Lease a bundle:

`POST /api/v1/tasks/lease`

```json
{
  "task_type": "cifar10_binary",
  "eval_batch_size": 8,
  "seed_batch_size": 8,
  "gossip_limit": 20,
  "sim_window_number": 9000
}
```

Submit a bundle:

`POST /api/v1/tasks/{lease_id}/submit_lease`

```json
{
  "evaluations": [
    { "algorithm_id": 123, "score": 0.8123 }
  ],
  "evolutions": [
    {
      "parent_algorithm_ids": [123],
      "algorithm_dsl": "# your evolved DSL"
    }
  ],
  "gossip": {
    "task_type": "cifar10_binary",
    "algorithm_ids": [123, 456]
  }
}
```

## Lease semantics and constraints

- `eval_batch_size` and `seed_batch_size` must be `>= 2` when provided.
- `gossip_limit` is clamped by schema to `0..50`.
- `sim_window_number` is for local/dev synthetic windows and is disabled in production.
- A miner can hold only one active assignment at a time.
- In enforced-window mode, a miner can receive at most one lease per window.
- Lease response includes:
  - `evaluate_algorithms` (population to evaluate),
  - `seed_algorithms` (context for local evolution),
  - `evolve_budget` (currently `0` or `1`).
- Bootstrap behavior: if the global in-evaluation pool is still too small, the server can seed `evaluate_algorithms` from the seed set to avoid startup deadlock.
- `submit_lease` stores evaluations, increments candidate evaluation counters, and accepts at most `evolve_budget` evolution proposals.

## Source of truth

See `Pool/app/api/v1` for request handling and `Pool/app/schemas` for models.
