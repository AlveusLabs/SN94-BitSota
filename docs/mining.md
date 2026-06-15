<section class="bitsota-hero compact">
  <p class="bitsota-kicker">MINERS</p>
  <h1>Manual Mining</h1>
  <p class="bitsota-lede">Clone a live task, make a measured improvement, and submit the workspace with your miner hotkey.</p>
</section>

Use this path when you want to do the work yourself. If a coding agent will do
the work loop, use [Agent Mining](codex-only-mining.md).

## What Counts

The coordinator task metadata and onboarding page define what validators score.
For the current Qwen compression tasks, validators score a downloadable model
artifact. A small recipe patch can explain how the artifact was made, but the
artifact is the main submission.

Do not use archived AutoML-Zero relay/SOTA guides for current production mining.

## 1. Set The Coordinator

```bash
cd SN94-BitSota
export BITSOTA_COORDINATOR_URL="https://autoresearch.bitsota.com"
```

## 2. Pick A Live Task

```bash
bitsota-research-agent list-tasks \
  --coordinator-url "$BITSOTA_COORDINATOR_URL" \
  > /tmp/bitsota-tasks.json

jq -r '.[] | select(.task_state == "live" and .is_active == true) |
  [.slug, .metric_name, .metric_direction, .competition_mode] | @tsv' \
  /tmp/bitsota-tasks.json
```

Choose a slug from that output:

```bash
export BITSOTA_TASK_SLUG="<LIVE_TASK_SLUG>"
export BITSOTA_TASK_ID="$(
  jq -r --arg slug "$BITSOTA_TASK_SLUG" \
    '.[] | select(.slug == $slug) | .id' \
    /tmp/bitsota-tasks.json
)"

test -n "$BITSOTA_TASK_ID"
```

If `BITSOTA_TASK_ID` is empty, stop and re-check the live task list.

## 3. Read Onboarding

```bash
curl -fsS \
  "$BITSOTA_COORDINATOR_URL/api/v1/tasks/$BITSOTA_TASK_ID/onboard.md" \
  -o /tmp/bitsota-onboard.md

less /tmp/bitsota-onboard.md
```

Onboarding is the contract for the task. It tells you the metric, artifact
requirements, benchmark command, and any centerless-mode fields.

## 4. Clone The Task Repo

```bash
export BITSOTA_TASK_REPO="$(
  jq -r --arg slug "$BITSOTA_TASK_SLUG" \
    '.[] | select(.slug == $slug) | .repository' \
    /tmp/bitsota-tasks.json
)"
export BITSOTA_TASK_REF="$(
  jq -r --arg slug "$BITSOTA_TASK_SLUG" \
    '.[] | select(.slug == $slug) | .base_ref' \
    /tmp/bitsota-tasks.json
)"

git clone "$BITSOTA_TASK_REPO" bitsota-task
cd bitsota-task
git checkout "$BITSOTA_TASK_REF"
```

## 5. Run The Baseline

Print the task commands:

```bash
jq --arg slug "$BITSOTA_TASK_SLUG" \
  '.[] | select(.slug == $slug) | {
    setup_command,
    benchmark_command,
    result_path,
    allowed_patch_paths,
    max_patch_bytes
  }' \
  /tmp/bitsota-tasks.json
```

Run the setup and benchmark commands from the task metadata. Keep the baseline
metric; you need it to know whether your change helped.

## 6. Claim The Task

Run the signed claim from the `SN94-BitSota` checkout or anywhere the console
script is available:

```bash
CLAIM_JSON="$(
  bitsota-research-agent signed-request \
    --coordinator-url "$BITSOTA_COORDINATOR_URL" \
    --method POST \
    --path "/api/v1/tasks/$BITSOTA_TASK_ID/claim" \
    --body-json '{"claim_description":"manual miner run"}' \
    --wallet-name <WALLET_NAME> \
    --wallet-hotkey <HOTKEY_NAME>
)"

export BITSOTA_CLAIM_ID="$(printf '%s' "$CLAIM_JSON" | jq -r '.id')"
printf '%s\n' "$CLAIM_JSON" | jq
```

Use the miner hotkey that should identify your submission. Do not put mnemonics
in chat, tickets, patches, or docs.

## 7. Make The Improvement

Only edit paths allowed by the task metadata:

```bash
jq -r --arg slug "$BITSOTA_TASK_SLUG" \
  '.[] | select(.slug == $slug) | .allowed_patch_paths[]' \
  /tmp/bitsota-tasks.json
```

After the change, run the same benchmark again. For artifact tasks, upload the
exact artifact bytes to a stable public URL and record:

- `artifact_uri`
- `artifact_sha256`
- `artifact_size_bytes`
- the local metric you observed
- the exact benchmark command

## 8. Write `submission.json`

Use the metric name from task metadata. For the current Qwen tasks, the metric
is `heldout_ppl` and lower is better.

```json
{
  "summary": "Short explanation of the method and local result.",
  "claimed_metrics": {
    "heldout_ppl": 0.0
  },
  "artifact_uri": "https://...",
  "artifact_sha256": "sha256-hex-of-the-exact-download",
  "artifact_size_bytes": 0,
  "proposed_idea": null,
  "implemented_submission_id": null,
  "execution_log": "Benchmark command, baseline metric, candidate metric."
}
```

For centerless tasks, fill `proposed_idea`. If you are building on another
accepted idea, also fill `implemented_submission_id` when onboarding requires
it.

## 9. Submit The Workspace

```bash
bitsota-research-agent submit-workspace \
  --coordinator-url "$BITSOTA_COORDINATOR_URL" \
  --claim-id "$BITSOTA_CLAIM_ID" \
  --repo-dir /path/to/bitsota-task \
  --submission-file /path/to/submission.json \
  --wallet-name <WALLET_NAME> \
  --wallet-hotkey <HOTKEY_NAME>
```

The helper builds a patch from `git diff` and filters it through the allowed
paths. It should fail rather than submit a broad dirty repo.

## 10. Check Status

```bash
curl -fsS \
  "$BITSOTA_COORDINATOR_URL/api/v1/submissions?task_id=$BITSOTA_TASK_ID" |
  jq '.[] | {id, status, miner_hotkey, created_at, claimed_metrics, observed_metrics}'
```

`pending_verification` means validators have not accepted the submission yet.
`accepted` means validator replay accepted it for that task.

## Rewards And Claims

Accepted submissions can affect ranking and reward snapshots. Pool publishes
claim packages later; claims are not instant.

The miner hotkey identifies the claim package. The published recipient coldkey
receives the reward. Read [Claim Rewards](claim-rewards.md) before claiming.

## Common Failures

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| No task id | Wrong slug or paused task. | Re-run `list-tasks` and copy a live slug. |
| Empty patch | You edited outside `allowed_patch_paths`. | Move the change into the allowed task surface. |
| Patch too large | `max_patch_bytes` was exceeded. | Submit a smaller patch or rely on artifact fields. |
| Submission stays pending | Validators have not accepted it yet. | Wait for replay or ask operators to inspect validator jobs. |
