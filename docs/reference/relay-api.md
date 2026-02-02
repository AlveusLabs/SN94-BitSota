# Relay API

The relay is a FastAPI service. When running locally, the OpenAPI UI is available at:

- `http://127.0.0.1:8002/docs`

Every response includes an `X-Request-ID` header for log correlation.

## Endpoints

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/health` | none | Liveness check |
| GET | `/version.json` | none | GUI update manifest |
| GET | `/sota_threshold` | none | Current SOTA threshold |
| GET | `/sota-events` | none | Paginated public SOTA events |
| POST | `/submit_solution` | miner | Submit or overwrite latest miner solution |
| GET | `/results` | validator | Fetch recent miner submissions |
| GET | `/results/{miner_hotkey}` | validator | Fetch latest submission for a miner |
| POST | `/sota/vote` | validator | Vote to finalize a SOTA event |
| GET | `/sota/events` | validator | List detailed SOTA events |
| POST | `/blacklist/{miner_hotkey}` | validator | Vote to blacklist a miner |
| POST | `/invitation_code/generate/{count}` | admin | Generate invite codes |
| GET | `/invitation_code/list/{page}/{size}` | admin | List invite codes |
| GET | `/invitation_code/linked` | miner | Get invite linked to this miner |
| POST | `/invitation_code/link` | miner | Link invite to this miner |
| POST | `/coldkey_address/update` | miner | Associate coldkey with miner |
| POST | `/test/submit_solution` | validator uid0 | Submit a test solution for logging only |
| GET | `/test/submissions` | validator uid0 | Claim and fetch queued test submissions |
| GET | `/admin/dashboard` | admin | Password-protected HTML dashboard |
| GET | `/admin/status` | admin | JSON status + request-rate metrics |

Auth headers:

- Miner and validator: `X-Key`, `X-Timestamp`, `X-Signature`
- Admin: `X-Auth-Token`
- Admin dashboard: HTTP Basic (defaults to `admin` + `ADMIN_AUTH_TOKEN`) or `X-Auth-Token`

## Examples

### Submit a solution

`POST /submit_solution`

```json
{
  "task_id": "cifar10_binary",
  "score": 0.93,
  "algorithm_result": {
    "dsl": "setup:\\n  CONST 0.5 -> s0\\n...",
    "metadata": {
      "engine": "baseline",
      "seed": 123
    }
  }
}
```

Notes:
- In non-test mode, the miner must have a linked invitation code.
- The relay keeps at most one submission per miner and overwrites it on every accepted submission.

### Vote on a candidate

`POST /sota/vote`

```json
{
  "miner_hotkey": "5F...",
  "score": 0.931,
  "seen_block": 123456,
  "result_id": "abcd1234"
}
```

The response returns:
- `status` such as `vote_recorded`, `vote_updated`, or `finalized`
- vote counters and the current SOTA
- `finalized_event` only when consensus is reached

## Common errors

- `401` missing or invalid auth headers
- `403` miner not invited or miner blacklisted
- `400` score is missing, below SOTA, duplicate within the duplicate window, or invalid input
 
## Source of truth

See `relay/main.py` for the complete behavior and `relay/schemas.py` for request and response models.
