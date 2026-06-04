<section class="bitsota-hero compact">
  <p class="bitsota-kicker">LIVE COMPETITIONS</p>
  <h1>Current Competitions</h1>
  <p class="bitsota-lede">Live SN94 autoresearch tasks, onboarding links, repositories, and what miners must submit.</p>
</section>

Last checked: June 3, 2026.

The backend is the source of truth. Always discover live tasks before mining:

```bash
export BITSOTA_COORDINATOR_URL="https://autoresearch.bitsota.com"

curl -fsS "$BITSOTA_COORDINATOR_URL/api/v1/tasks" | jq '.[] | {
  slug,
  title,
  task_state,
  is_active,
  metric_name,
  metric_direction,
  competition_mode,
  onboard_path,
  repository,
  allowed_patch_paths
}'
```

Dashboard:

```text
https://autoresearch.bitsota.com/dashboard
```

Dashboard JSON:

```text
https://autoresearch.bitsota.com/api/v1/dashboard
```

The dashboard currently renders these competitions from `/api/v1/dashboard` and
links each task detail view to its generated `onboard.md`.

## Live Task Summary

| Slug | Mode | Metric | What to submit |
| --- | --- | --- | --- |
| `qwen3-27b-binary-frontier` | `standard` | `heldout_ppl` minimize | Public compressed-model artifact URL, SHA-256, byte size, claimed metrics, and optional recipe patch. |
| `qwen3-27b-ternary-frontier` | `centerless` | `heldout_ppl` minimize | Same artifact fields, claimed metrics, `proposed_idea`, and optional `implemented_submission_id` when building on another miner's idea. |

For both current tasks, the artifact is the scoring object. `train.py` is
optional recipe metadata unless the task onboarding says otherwise.

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
| Time budget | `21600` seconds |

Submission requirements:

- upload a compressed model artifact to a stable public HTTPS URL, usually a
  Hugging Face file URL pinned to a commit or tag;
- include `artifact_uri`;
- include `artifact_sha256`, the SHA-256 of the exact downloadable bytes;
- include `artifact_size_bytes`, the byte size of the exact downloadable bytes;
- include `summary` and `claimed_metrics`;
- include a small patch only if changing recipe metadata inside the allowed
  `train.py` path.

Do not submit model bytes, caches, `last_run.json`, or generated files in the
patch.

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
| Time budget | `21600` seconds |

Submission requirements:

- upload a compressed model artifact to a stable public HTTPS URL;
- include `artifact_uri`;
- include `artifact_sha256`;
- include `artifact_size_bytes`;
- include `summary` and `claimed_metrics`;
- include `proposed_idea` for centerless mode;
- include `implemented_submission_id` when the task requires building on a
  prior idea-bearing submission;
- include a small patch only if changing recipe metadata inside the allowed
  `train.py` path.

Do not treat this as a standard patch-only task. The artifact is what validators
score.

## Current Backend Gap

Production onboarding pages contain the competition-specific artifact guidance,
but `/api/v1/tasks` does not yet expose `submission_surface`,
`requires_submission_artifact`, or required field lists in the task JSON. Until
that backend metadata is deployed, miners should use this page plus each
competition's `onboard.md`.

The intended backend contract is documented under
[Autoresearch Backend Routes](autoresearch-backend-routes.md).
