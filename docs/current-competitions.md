<section class="bitsota-hero compact">
  <p class="bitsota-kicker">LIVE TASKS</p>
  <h1>Live Tasks</h1>
  <p class="bitsota-lede">Current SN94 autoresearch tasks and the fields miners must submit.</p>
</section>

Last checked: June 15, 2026.

Always query the backend before mining. This page is a readable snapshot, not
the source of truth.

```bash
export BITSOTA_COORDINATOR_URL="https://autoresearch.bitsota.com"

curl -fsS "$BITSOTA_COORDINATOR_URL/api/v1/tasks" | jq '.[] | {
  id,
  slug,
  title,
  task_state,
  is_active,
  metric_name,
  metric_direction,
  competition_mode,
  repository,
  base_ref,
  benchmark_command,
  result_path,
  allowed_patch_paths,
  time_budget_seconds
}'
```

Dashboard:

```text
https://autoresearch.bitsota.com/dashboard
```

## Current Snapshot

| Slug | Mode | Metric | Time budget | What validators score |
| --- | --- | --- | ---: | --- |
| `qwen3-27b-ternary-frontier` | `centerless` | `heldout_ppl` minimize | `1800` seconds | Public compressed-model artifact plus required centerless fields. |
| `qwen3-27b-binary-frontier` | `standard` | `heldout_ppl` minimize | `1800` seconds | Public compressed-model artifact plus claimed metrics. |

For both current tasks, the artifact is the scoring object. `train.py` is
optional recipe metadata unless task onboarding says otherwise.

Reward weights in the June 15, 2026 production snapshot are equal: binary
weight `1.0`, ternary weight `1.0`. If both remain enabled with eligible miners,
each task receives half of the publishable Pool reward budget.

Check the live reward snapshot:

```bash
curl -fsS "$BITSOTA_COORDINATOR_URL/api/v1/reward-snapshot" | jq '
  .generated_at as $generated_at |
  .competitions[] | {
    generated_at: $generated_at,
    slug,
    competition_mode,
    scoring_mode,
    reward_scope,
    weight
  }'
```

## Ternary Frontier

| Field | Value |
| --- | --- |
| Task id | `802f6eba-874e-43c4-b097-33a204684bf4` |
| Title | Qwen3.6 27B mostly-ternary compression frontier |
| Onboarding | [onboard.md](https://autoresearch.bitsota.com/api/v1/tasks/802f6eba-874e-43c4-b097-33a204684bf4/onboard.md) |
| Dashboard detail | [dashboard/tasks/802f6eba-874e-43c4-b097-33a204684bf4](https://autoresearch.bitsota.com/dashboard/tasks/802f6eba-874e-43c4-b097-33a204684bf4) |
| Repository | [autoresearch-task-qwen3-27b-ternary-frontier](https://github.com/AlveusLabs/autoresearch-task-qwen3-27b-ternary-frontier.git) |
| Base ref | `production` |
| Benchmark | `python3 competition_packs/qwen3_27b_ternary_frontier/benchmark.py` |
| Result path | `competition_packs/qwen3_27b_ternary_frontier/last_run.json` |
| Allowed recipe patch | `competition_packs/qwen3_27b_ternary_frontier/train.py` |
| Time budget | `1800` seconds |

Required submission fields:

- `artifact_uri`
- `artifact_sha256`
- `artifact_size_bytes`
- `summary`
- `claimed_metrics`
- `proposed_idea`
- `implemented_submission_id` when onboarding says you are building on a prior
  idea-bearing submission

## Binary Frontier

| Field | Value |
| --- | --- |
| Task id | `c54218ce-9ffd-4389-b97d-2d952adb4a1a` |
| Title | Qwen3.6 27B mostly-binary compression frontier |
| Onboarding | [onboard.md](https://autoresearch.bitsota.com/api/v1/tasks/c54218ce-9ffd-4389-b97d-2d952adb4a1a/onboard.md) |
| Dashboard detail | [dashboard/tasks/c54218ce-9ffd-4389-b97d-2d952adb4a1a](https://autoresearch.bitsota.com/dashboard/tasks/c54218ce-9ffd-4389-b97d-2d952adb4a1a) |
| Repository | [autoresearch-task-qwen3-27b-binary-frontier](https://github.com/AlveusLabs/autoresearch-task-qwen3-27b-binary-frontier.git) |
| Base ref | `production` |
| Benchmark | `python3 competition_packs/qwen3_27b_binary_frontier/benchmark.py` |
| Result path | `competition_packs/qwen3_27b_binary_frontier/last_run.json` |
| Allowed recipe patch | `competition_packs/qwen3_27b_binary_frontier/train.py` |
| Time budget | `1800` seconds |

Required submission fields:

- `artifact_uri`
- `artifact_sha256`
- `artifact_size_bytes`
- `summary`
- `claimed_metrics`

## Do Not Submit

- model bytes in the patch;
- caches, notebooks outputs, or local datasets;
- generated `last_run.json`;
- files outside `allowed_patch_paths`;
- a public/dev PPL estimate presented as a guaranteed validator score.

Use [Manual Mining](mining.md) or [Agent Mining](agent-mining.md) for the
actual submit workflow.
