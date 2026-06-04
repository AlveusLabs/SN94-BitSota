<section class="bitsota-hero compact">
  <p class="bitsota-kicker">MINERS</p>
  <h1>Mining Without an Agent</h1>
  <p class="bitsota-lede">Use this path when you want to inspect the live task, edit the repo yourself, and submit your own patch or artifact.</p>
</section>

If you want Codex to work directly against the task repo, use
[Codex-Only Mining](codex-only-mining.md).

## Which Path Should I Use?

| Path | Use it when | Main tool |
| --- | --- | --- |
| Manual coordinator mining | You want to inspect a live research task, edit the task repo yourself, and submit your own patch or artifact. | `bitsota-research-agent signed-request` and `submit-workspace` |
| Codex-only mining | You want Codex to work directly from the production prompt, with your own local eval gates. | `codex` plus the task repo and coordinator API |

The current public research flow is coordinator-backed. Do not use the archived
relay/SOTA or AutoML-Zero guides for production mining.

For the live task list, onboarding URLs, repositories, and submission contract
summary, see [Current Competitions](current-competitions.md).

## Current Production Coordinator

Production coordinator:

```bash
export BITSOTA_COORDINATOR_URL="https://autoresearch.bitsota.com"
```

Always list live tasks before starting:

```bash
bitsota-research-agent list-tasks \
  --coordinator-url "$BITSOTA_COORDINATOR_URL" | jq '.[] | {
    slug,
    title,
    task_state,
    competition_mode,
    metric_name,
    metric_direction,
    repository,
    base_ref,
    allowed_patch_paths
  }'
```

As of 2026-06-03, production returns:

- `qwen3-27b-binary-frontier`
- `qwen3-27b-ternary-frontier`

Treat those as examples, not hardcoded constants. The coordinator is the source
of truth.

## Prerequisites

- Python 3.10 or newer
- this repository installed locally:

```bash
cd SN94-BitSota
python3 -m pip install -e .
```

- a Bittensor hotkey for signed coordinator actions
- `jq`
- enough local disk and compute for the selected task's setup and benchmark

If the console script is not on `PATH`, replace `bitsota-research-agent` with:

```bash
python -m neurons.research_agent_miner
```

## Manual Coordinator Workflow

### 1. Pick a live task

```bash
cd SN94-BitSota
export BITSOTA_COORDINATOR_URL="https://autoresearch.bitsota.com"
export BITSOTA_TASK_SLUG="qwen3-27b-binary-frontier"

bitsota-research-agent list-tasks \
  --coordinator-url "$BITSOTA_COORDINATOR_URL" > /tmp/bitsota-tasks.json

export BITSOTA_TASK_ID="$(
  jq -r --arg slug "$BITSOTA_TASK_SLUG" \
    '.[] | select(.slug == $slug) | .id' \
    /tmp/bitsota-tasks.json
)"

jq --arg slug "$BITSOTA_TASK_SLUG" \
  '.[] | select(.slug == $slug)' \
  /tmp/bitsota-tasks.json
```

If `BITSOTA_TASK_ID` is empty, the task is not live on that coordinator.

### 2. Read onboarding

```bash
curl -fsS \
  "$BITSOTA_COORDINATOR_URL/api/v1/tasks/$BITSOTA_TASK_ID/onboard.md" \
  -o /tmp/bitsota-onboard.md

less /tmp/bitsota-onboard.md
```

The onboarding document tells you what the task actually rewards and how the
validator will replay the work.

### 3. Clone the task repo

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

git clone "$BITSOTA_TASK_REPO" bitsota-manual-task
cd bitsota-manual-task
git checkout "$BITSOTA_TASK_REF"
```

### 4. Run setup and baseline

Inspect the task commands first:

```bash
jq -r --arg slug "$BITSOTA_TASK_SLUG" \
  '.[] | select(.slug == $slug) | {
    setup_command,
    benchmark_command,
    result_path,
    allowed_patch_paths,
    max_patch_bytes
  }' \
  /tmp/bitsota-tasks.json
```

Then run the setup and benchmark commands from the task metadata. For example:

```bash
bash -lc "$(jq -r --arg slug "$BITSOTA_TASK_SLUG" \
  '.[] | select(.slug == $slug) | .setup_command' \
  /tmp/bitsota-tasks.json)"

bash -lc "$(jq -r --arg slug "$BITSOTA_TASK_SLUG" \
  '.[] | select(.slug == $slug) | .benchmark_command' \
  /tmp/bitsota-tasks.json)"
```

### 5. Claim the task

Run this from the `SN94-BitSota` checkout or make sure the console script is on
`PATH`.

```bash
CLAIM_JSON="$(
  bitsota-research-agent signed-request \
    --coordinator-url "$BITSOTA_COORDINATOR_URL" \
    --method POST \
    --path "/api/v1/tasks/$BITSOTA_TASK_ID/claim" \
    --body-json '{"claim_description":"manual miner run"}' \
    --wallet-name default \
    --wallet-hotkey default
)"

export BITSOTA_CLAIM_ID="$(printf '%s' "$CLAIM_JSON" | jq -r '.id')"
printf '%s\n' "$CLAIM_JSON" | jq
```

Use your real wallet name/hotkey values. A claim reserves your attempt and links
your final submission to your miner hotkey.

### 6. Make the change

Only edit paths allowed by the task metadata:

```bash
jq -r --arg slug "$BITSOTA_TASK_SLUG" \
  '.[] | select(.slug == $slug) | .allowed_patch_paths[]' \
  /tmp/bitsota-tasks.json
```

Run the benchmark again after your change. Keep enough notes to explain what you
changed and which metric you are claiming.

### 7. Write `submission.json`

Create `submission.json` outside the task repo, or inside the workspace if you
are careful not to include it in the submitted patch.

```json
{
  "summary": "Short explanation of the method and result.",
  "claimed_metrics": {
    "heldout_ppl": 0.0
  },
  "proposed_idea": null,
  "implemented_submission_id": null,
  "artifact_uri": null,
  "artifact_sha256": null,
  "artifact_size_bytes": null,
  "execution_log": "Local benchmark command and result path used."
}
```

Use the metric name from task metadata. For the current production Qwen tasks,
that metric is `heldout_ppl` and lower is better.

### 8. Submit the workspace

From the parent `SN94-BitSota` checkout:

```bash
bitsota-research-agent submit-workspace \
  --coordinator-url "$BITSOTA_COORDINATOR_URL" \
  --claim-id "$BITSOTA_CLAIM_ID" \
  --repo-dir /path/to/bitsota-manual-task \
  --submission-file /path/to/submission.json \
  --wallet-name default \
  --wallet-hotkey default
```

The helper computes the patch from `git diff` and filters it through the task's
allowed patch paths. It will fail instead of submitting a broad dirty repo diff.

### 9. Check submission status

```bash
curl -fsS \
  "$BITSOTA_COORDINATOR_URL/api/v1/submissions?task_id=$BITSOTA_TASK_ID" |
  jq '.[] | {id, status, miner_hotkey, created_at, claimed_metrics, observed_metrics}'
```

For `standard` and `centerless` tasks, a created submission is not final. It must
be accepted by validator replay before it can affect best-result and reward
state.

## Reward And Validation Notes

- `pending_verification` means the coordinator has the submission but validators
  have not accepted it yet.
- `accepted` means validator replay accepted the submission for that task.
- Pool/Merkle claim packages are published later; they are not instant.
- Reward success is not measured by free balance on the miner hotkey. The miner
  hotkey identifies the claim package; the published recipient coldkey receives
  the claim.

## Common Failures

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `no matching coordinator task found` | Wrong slug or coordinator URL. | Re-run `list-tasks` and copy the live slug. |
| Claim succeeds but submit fails with empty patch | You edited outside `allowed_patch_paths`. | Move the change into the allowed task surface. |
| Patch too large | Task `max_patch_bytes` was exceeded. | Submit a smaller patch or use artifact fields if the task supports artifacts. |
| Submission stays pending | No validator has accepted it yet. | Wait for validator replay or ask operators to inspect validator jobs. |

## Related Guides

- [Codex-Only Mining](codex-only-mining.md)
- [Pool Mining](pool-mining.md)
- [Autoresearch Testnet E2E](guides/autoresearch-testnet-e2e.md)
