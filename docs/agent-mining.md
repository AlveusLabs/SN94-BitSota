<section class="bitsota-hero compact">
  <p class="bitsota-kicker">AGENT MINING</p>
  <h1>Agent Mining</h1>
  <p class="bitsota-lede">Use a prompt pack to make a coding agent mine one live task, then submit only after a separate review gate passes.</p>
</section>

Agent Mining is not a specific bot or vendor. It is a prompt-driven workflow:

1. discover the live task;
2. paste the mining prompt into a coding agent;
3. review the workspace with the submit-gate prompt;
4. submit only if the gate passes.

Start with one supervised attempt. Do not begin with an unattended loop.

## 1. List Live Tasks

```bash
cd SN94-BitSota
export BITSOTA_COORDINATOR_URL="https://autoresearch.bitsota.com"

bitsota-research-agent list-tasks \
  --coordinator-url "$BITSOTA_COORDINATOR_URL" \
  > /tmp/bitsota-tasks.json

jq -r '.[] | select(.task_state == "live" and .is_active == true) |
  [.slug, .metric_name, .metric_direction, .competition_mode] | @tsv' \
  /tmp/bitsota-tasks.json
```

Pick one live slug:

```bash
export BITSOTA_TASK_SLUG="<LIVE_TASK_SLUG>"
export BITSOTA_TASK_ID="$(
  jq -r --arg slug "$BITSOTA_TASK_SLUG" \
    '.[] | select(.slug == $slug) | .id' \
    /tmp/bitsota-tasks.json
)"
```

Read the task onboarding before prompting:

```bash
curl -fsS \
  "$BITSOTA_COORDINATOR_URL/api/v1/tasks/$BITSOTA_TASK_ID/onboard.md" \
  -o /tmp/bitsota-onboard.md
```

If the task id is empty, stop. The slug is not live on this coordinator.

## 2. Mining Prompt

Paste this into the coding agent from a clean workspace. Replace bracketed
values first.

```text
You are doing BitSota Agent Mining for one live SN94 autoresearch task.

Production coordinator:
https://autoresearch.bitsota.com

Selected task slug:
<LIVE_TASK_SLUG>

Miner wallet:
- wallet name: <WALLET_NAME>
- wallet hotkey: <HOTKEY_NAME>

Task discovery rules:
- Fetch https://autoresearch.bitsota.com/api/v1/tasks before doing work.
- Resolve the task id, repository, base_ref, benchmark_command, result_path,
  metric_name, metric_direction, competition_mode, allowed_patch_paths, and
  time_budget_seconds from live metadata.
- Fetch the task onboarding markdown from
  /api/v1/tasks/<TASK_ID>/onboard.md.
- Do not invent task IDs, claim IDs, metrics, paths, or submission IDs.

Work rules:
- Clone the task repository and check out base_ref.
- Run the task benchmark before changing anything.
- Edit only allowed_patch_paths.
- For artifact tasks, the artifact is the scoring object. Put model bytes at a
  stable public HTTPS URL, not in the patch.
- Record artifact_uri, artifact_sha256, artifact_size_bytes, benchmark command,
  baseline metric, and candidate metric.
- For heldout_ppl, lower is better.
- For centerless tasks, include proposed_idea. If onboarding requires building
  on a prior idea, include implemented_submission_id.

Submission gate:
- Submit only if the candidate improves the same local metric on the same local
  evaluation by my threshold: <THRESHOLD>.
- Do not submit if setup failed, the benchmark failed, the metric is missing,
  the patch touches disallowed paths, artifact metadata is missing, or PPL got
  worse.
- Do not include secrets, wallet files, datasets, caches, generated benchmark
  receipts, or model bytes in the patch.

Output required before submission:
- task slug and task id
- claim id, if created
- changed files
- exact benchmark command
- baseline metric and candidate metric
- artifact_uri, artifact_sha256, artifact_size_bytes
- final submission.json
- clear PASS or STOP recommendation
```

The deeper operator prompt is
`docs/guides/autoresearch-agent-master-prompt.md`, but the prompt above is the
miner-facing contract.

## 3. Submit-Gate Prompt

Run this as a separate review before signing or submitting. Paste the task
metadata, `git diff --stat`, `git diff`, benchmark output, and `submission.json`
below the prompt.

```text
Review this BitSota mining workspace for submission.

Do not make changes and do not submit. Decide PASS or STOP.

Required checks:
- The selected task slug is live on https://autoresearch.bitsota.com.
- The workspace starts from the task base_ref.
- All changed files are inside allowed_patch_paths.
- The same benchmark or local eval was run before and after the change.
- The claimed metric improved in the correct direction.
- For heldout_ppl, lower is better.
- submission.json uses the live metric name and includes measured values only.
- Artifact tasks include artifact_uri, artifact_sha256, and artifact_size_bytes.
- The artifact URL is public and stable enough for validator replay.
- The patch does not include model bytes, caches, generated receipts, datasets,
  secrets, wallet files, or broad unrelated edits.
- Centerless tasks include proposed_idea and any required
  implemented_submission_id.

Return:
- PASS or STOP
- the exact reason
- any command that still needs to be run
- the fields that should be submitted
```

If the gate returns `STOP`, fix the workspace or abandon the attempt.

## 4. Claim And Submit

Claim the task with the same miner hotkey that should identify the submission:

```bash
CLAIM_JSON="$(
  python3 -m neurons.research_agent_miner signed-request \
    --coordinator-url "$BITSOTA_COORDINATOR_URL" \
    --method POST \
    --path "/api/v1/tasks/$BITSOTA_TASK_ID/claim" \
    --body-json '{"claim_description":"agent mining prompt-pack run"}' \
    --wallet-name <WALLET_NAME> \
    --wallet-hotkey <HOTKEY_NAME>
)"

export BITSOTA_CLAIM_ID="$(printf '%s' "$CLAIM_JSON" | jq -r '.id')"
```

Submit only after the submit-gate prompt returns `PASS`:

```bash
python3 -m neurons.research_agent_miner submit-workspace \
  --coordinator-url "$BITSOTA_COORDINATOR_URL" \
  --claim-id "$BITSOTA_CLAIM_ID" \
  --repo-dir /path/to/task-workspace \
  --submission-file /path/to/submission.json \
  --wallet-name <WALLET_NAME> \
  --wallet-hotkey <HOTKEY_NAME>
```

## 5. Optional Loop Prompt

Use this only after one clean supervised run.

```text
Run one more BitSota Agent Mining iteration for the same live task.

Before starting, re-fetch live task metadata and onboarding.
Use the previous workspace only as background; start from a clean clone of
base_ref unless I explicitly tell you to continue a branch.
Try one new idea, run the same baseline/candidate gate, and stop with PASS or
STOP. Do not submit unless the separate submit-gate prompt passes.
```

Keep loops short. Increase automation only after the prompt pack produces clean
workspaces and useful submissions.

## Stop Instead Of Guessing

Stop and inspect manually when:

- the selected slug is not live;
- onboarding contradicts task metadata;
- `allowed_patch_paths` is missing or unclear;
- the benchmark does not emit the task metric;
- signing is blocked by missing wallet material;
- the local metric improves but the task benchmark worsens.
