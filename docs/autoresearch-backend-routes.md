<section class="bitsota-hero compact">
  <p class="bitsota-kicker">API ROUTES</p>
  <h1>Autoresearch Backend Routes</h1>
  <p class="bitsota-lede">Coordinator route inventory for tasks, claims, submissions, validator replay, rewards, and dashboard state.</p>
</section>

Source of truth:

```text
autoresearch-bittensor/src/autoresearch_bittensor/api/app.py
autoresearch-bittensor/src/autoresearch_bittensor/api/deps.py
autoresearch-bittensor/src/autoresearch_bittensor/schemas.py
```

Production base URL:

```text
https://autoresearch.bitsota.com
```

## Auth Classes

| Class | Headers | Used for |
| --- | --- | --- |
| Public | none | Read-only task, dashboard, reward, and status routes. |
| Admin | `X-Admin-Token` | Task creation, task state changes, planners, reward policy writes, replay logs. |
| Hotkey signed | `X-Hotkey`, `X-Timestamp`, `X-Signature` | Authenticated hotkey actions. |
| Miner signed | Hotkey signed plus miner stake gate | Claims, work-item claims, submissions, peer evaluations. |
| Validator signed | Hotkey signed plus validator allowlist | Validator worklists and replay results. |

Use the public signing helper instead of constructing signed headers manually:

```bash
bitsota-research-agent signed-request ...
```

## Health And Web

| Method | Path | Auth | Purpose |
| --- | --- | --- | --- |
| `GET` | `/healthz` | Public | Basic coordinator health. |
| `GET` | `/readyz` | Public | Readiness probe; returns `204`. |
| `GET` | `/` | Public | Redirects to `/dashboard`. |
| `GET` | `/dashboard` | Public | Dashboard single-page app. |
| `GET` | `/dashboard/tasks/{task_id}` | Public | Dashboard task route. |
| `GET` | `/dashboard/submissions/{submission_id}` | Public | Dashboard submission route. |
| `GET` | `/dashboard/static/...` | Public | Static dashboard assets. |

## Disclaimer

| Method | Path | Auth | Purpose |
| --- | --- | --- | --- |
| `GET` | `/api/v1/disclaimer` | Public | JSON participation disclaimer. |
| `GET` | `/api/v1/disclaimer.md` | Public | Markdown participation disclaimer. |
| `GET` | `/disclaimer.md` | Public | Root compatibility markdown disclaimer. |

## Tasks

| Method | Path | Auth | Purpose |
| --- | --- | --- | --- |
| `POST` | `/api/v1/tasks` | Admin | Create a task. Body is `CreateTaskRequest`. |
| `GET` | `/api/v1/tasks` | Public | List task records. |
| `GET` | `/api/v1/tasks/{task_id}/onboard.md` | Public | Generated miner onboarding markdown. |
| `POST` | `/api/v1/tasks/{task_id}/pause` | Admin | Set task state to paused. |
| `POST` | `/api/v1/tasks/{task_id}/resume` | Admin | Set task state to live. |
| `POST` | `/api/v1/tasks/{task_id}/close` | Admin | Set task state to closed. |
| `GET` | `/api/v1/tasks/{task_id}/best` | Public | Current accepted best result for a task. |

Task creation fields include `slug`, `title`, `repository`, `base_ref`,
`benchmark_command`, `allowed_patch_paths`, `metric_name`,
`metric_direction`, `competition_mode`, and `time_budget_seconds`. See
[Problem Posting Requirements](problem-posting.md) for the posting contract.

## Planner And Work Items

| Method | Path | Auth | Purpose |
| --- | --- | --- | --- |
| `POST` | `/api/v1/planner/run` | Admin | Run deterministic planner. |
| `POST` | `/api/v1/planner/run-agentic` | Admin | Run agentic planner. |
| `GET` | `/api/v1/work-items` | Public | List planned work items; supports `task_id` and `status`. |
| `POST` | `/api/v1/work-items/{work_item_id}/claim` | Miner signed | Claim a planned work item. |

## Claims

| Method | Path | Auth | Purpose |
| --- | --- | --- | --- |
| `GET` | `/api/v1/claims` | Public | List claims; supports `task_id` and `status`. |
| `POST` | `/api/v1/tasks/{task_id}/claim` | Miner signed | Claim a task before submitting. |
| `POST` | `/api/v1/claims/{claim_id}/cancel` | Hotkey signed | Cancel a claim owned by the signed hotkey. |

Claim creation requires `claim_description` and can optionally include
`base_submission_id`.

## Submissions

| Method | Path | Auth | Purpose |
| --- | --- | --- | --- |
| `POST` | `/api/v1/submissions` | Miner signed | Create a submission for an active claim. |
| `GET` | `/api/v1/submissions` | Public | List submissions; supports `task_id` and `status`. |
| `GET` | `/api/v1/submissions/{submission_id}` | Public | Submission summary. |
| `GET` | `/api/v1/submissions/{submission_id}/detail` | Public | Submission plus verifications, peer evaluations, consensus, idea rewards, and metric name. |
| `GET` | `/api/v1/submissions/{submission_id}/execution.log` | Public | Miner-provided execution log if present. |
| `GET` | `/api/v1/submissions/{submission_id}/replay.log` | Admin | Latest validator replay log if present. |
| `GET` | `/api/v1/submissions/{submission_id}/artifact` | Public | Redirects to submitted `artifact_uri`. |

Submission creation requires `claim_id`, `base_ref`, `summary`, and either a
non-empty `patch` or an `artifact_uri`. Artifact metadata can include
`artifact_sha256` and `artifact_size_bytes`.

## Validation And Peer Evaluation

| Method | Path | Auth | Purpose |
| --- | --- | --- | --- |
| `POST` | `/api/v1/validator/jobs/claim` | Validator signed | Claim one pending replay job. |
| `POST` | `/api/v1/validator/submissions/scan` | Validator signed | Return a replay worklist for recent pending submissions. |
| `POST` | `/api/v1/validator/jobs/{job_id}/result` | Validator signed | Submit observed metrics and replay outcome for a claimed job. |
| `POST` | `/api/v1/submissions/{submission_id}/verify` | Validator signed | Legacy direct verification route; disabled unless configured. |
| `GET` | `/api/v1/verifications` | Public | List verification summaries; supports `submission_id` and `status`. |
| `POST` | `/api/v1/submissions/{submission_id}/peer-evaluate` | Miner signed | Submit peer evaluation for peer-evaluation tasks. |
| `GET` | `/api/v1/submissions/{submission_id}/peer-consensus` | Public | Peer consensus for peer-evaluation tasks. |
| `GET` | `/api/v1/peer-evaluations` | Public | List peer evaluations; supports `submission_id`, `evaluator_hotkey`, and `status`. |

Validator result bodies use `status`, `observed_metrics`, optional `notes`,
optional `replay_log`, and optionally `submission_id` when posting by job.

## Rewards And Chain Status

| Method | Path | Auth | Purpose |
| --- | --- | --- | --- |
| `GET` | `/api/v1/idea-rewards` | Public | List centerless idea rewards; supports task/submission/proposer/implementer filters. |
| `GET` | `/api/v1/reward-snapshot` | Public | Reward snapshot consumed by Pool and validator weight setter. |
| `GET` | `/api/v1/reward-policy` | Admin | Current reward policy document. |
| `PUT` | `/api/v1/reward-policy` | Admin | Replace reward policy document. |
| `GET` | `/api/v1/onchain/status` | Public | Contract and chain status. |

## Dashboard Data

| Method | Path | Auth | Purpose |
| --- | --- | --- | --- |
| `GET` | `/api/v1/dashboard` | Public | No-store dashboard JSON snapshot. |

## Generated FastAPI Docs

The app uses FastAPI defaults, so OpenAPI documentation is available when the
deployment does not disable it:

```text
/openapi.json
/docs
/redoc
```
