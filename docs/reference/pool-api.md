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
  "seed_batch_size": 2,
  "gossip_limit": 5
}
```

Submit a bundle:

`POST /api/v1/tasks/{lease_id}/submit_lease`

```json
{
  "evaluations": [],
  "evolutions": [],
  "gossip": null
}
```

## Source of truth

See `Pool/app/api/v1` for request handling and `Pool/app/schemas` for models.

