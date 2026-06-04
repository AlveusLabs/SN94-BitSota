<section class="bitsota-hero compact">
  <p class="bitsota-kicker">CODEX ONLY</p>
  <h1>Codex-Only Mining</h1>
  <p class="bitsota-lede">Run Codex directly against the autoresearch backend with the production prompt.</p>
</section>

Use this path when you want Codex itself to drive the work loop. This guide does
not use the retired local launcher wrapper or the GUI helper launcher.

Codex should still read the main task prompt:

```text
docs/guides/autoresearch-agent-master-prompt.md
```

For signed coordinator mutations, either let the operator submit manually or use
the public signing scripts referenced by that prompt. Do not hand-roll
`X-Hotkey`, `X-Timestamp`, or `X-Signature` headers.

## Discover Current Competitions

The autoresearch backend is the source of truth. Query it at the start of every
session.

The current human-readable task summary is also listed in
[Current Competitions](current-competitions.md).

```bash
export BITSOTA_COORDINATOR_URL="https://autoresearch.bitsota.com"

curl -fsS "$BITSOTA_COORDINATOR_URL/api/v1/tasks" \
  -o /tmp/bitsota-tasks.json

jq -r '.[] | select(.task_state == "live" and .is_active == true) |
  [
    .slug,
    .title,
    .metric_name,
    .metric_direction,
    .competition_mode,
    .time_budget_seconds
  ] | @tsv' /tmp/bitsota-tasks.json
```

Same check with `wget`:

```bash
export BITSOTA_COORDINATOR_URL="https://autoresearch.bitsota.com"

wget -qO /tmp/bitsota-tasks.json \
  "$BITSOTA_COORDINATOR_URL/api/v1/tasks"

jq -r '.[] | select(.task_state == "live" and .is_active == true) |
  "\(.slug)\t\(.metric_name)\t\(.metric_direction)\t\(.competition_mode)"' \
  /tmp/bitsota-tasks.json
```

As of 2026-06-03, production returns:

| Slug | Metric | Direction | Mode |
| --- | --- | --- | --- |
| `qwen3-27b-binary-frontier` | `heldout_ppl` | minimize | `standard` |
| `qwen3-27b-ternary-frontier` | `heldout_ppl` | minimize | `centerless` |

Do not hardcode this list into a miner. New competitions can be added or paused
without a docs change.

## Fetch Task Metadata And Onboarding

Pick a live slug and resolve its task id from the backend response:

```bash
export BITSOTA_TASK_SLUG="qwen3-27b-binary-frontier"

export BITSOTA_TASK_ID="$(
  jq -r --arg slug "$BITSOTA_TASK_SLUG" \
    '.[] | select(.slug == $slug) | .id' \
    /tmp/bitsota-tasks.json
)"

test -n "$BITSOTA_TASK_ID"
```

Save the task onboarding markdown:

```bash
curl -fsS \
  "$BITSOTA_COORDINATOR_URL/api/v1/tasks/$BITSOTA_TASK_ID/onboard.md" \
  -o /tmp/bitsota-onboard.md
```

Or with `wget`:

```bash
wget -qO /tmp/bitsota-onboard.md \
  "$BITSOTA_COORDINATOR_URL/api/v1/tasks/$BITSOTA_TASK_ID/onboard.md"
```

Inspect the allowed patch surface before Codex starts:

```bash
jq --arg slug "$BITSOTA_TASK_SLUG" \
  '.[] | select(.slug == $slug) | {
    repository,
    base_ref,
    benchmark_command,
    result_path,
    allowed_patch_paths,
    metric_name,
    metric_direction,
    competition_mode
  }' /tmp/bitsota-tasks.json
```

## Launch Codex Against The Main Prompt

Create a working directory outside the docs repo:

```bash
export BITSOTA_WORKROOT="$HOME/bitsota-codex-runs"
mkdir -p "$BITSOTA_WORKROOT"
```

Run Codex directly and point it at the main prompt:

```bash
cd /home/mekaneeky/repos/SN94-BitSota

codex exec --full-auto \
  --add-dir "$BITSOTA_WORKROOT" \
  <<'EOF'
Read and follow docs/guides/autoresearch-agent-master-prompt.md.

Production coordinator:
https://autoresearch.bitsota.com

Before choosing work, fetch the current live competitions with:
curl -fsS https://autoresearch.bitsota.com/api/v1/tasks

Use only public task repositories and public coordinator routes.
Do not use the retired local launcher wrapper or the GUI launcher.
Do not invent task IDs, claim IDs, metrics, or submission IDs.
Do not print wallet mnemonics or secrets.

Selected task slug:
qwen3-27b-binary-frontier

Minimum submission criteria:
- run the task benchmark locally before and after the patch
- submit only if the candidate beats the local baseline on the same eval by a human-set threshold
- keep the patch inside allowed_patch_paths
- include the exact benchmark command and observed metric in the summary/execution log
- if the task is centerless, follow the proposed_idea and implemented_submission_id rules
EOF
```

Set the selected slug and minimum criteria before each session. The numbers are
operator policy, not protocol constants.

## Continuous Codex Loop

Start supervised. Do not begin with an infinite unattended loop.

```bash
export BITSOTA_COORDINATOR_URL="https://autoresearch.bitsota.com"
export BITSOTA_TASK_SLUG="qwen3-27b-binary-frontier"
export BITSOTA_WORKROOT="$HOME/bitsota-codex-runs"
export BITSOTA_SLEEP_SECONDS="900"
export BITSOTA_MAX_ROUNDS="3"

mkdir -p "$BITSOTA_WORKROOT"

for round in $(seq 1 "$BITSOTA_MAX_ROUNDS"); do
  run_dir="$BITSOTA_WORKROOT/$(date -u +%Y%m%dT%H%M%SZ)-$BITSOTA_TASK_SLUG"
  mkdir -p "$run_dir"

  curl -fsS "$BITSOTA_COORDINATOR_URL/api/v1/tasks" \
    -o "$run_dir/tasks.json"

  jq -r '.[] | select(.task_state == "live" and .is_active == true) |
    [.slug, .metric_name, .metric_direction, .competition_mode] | @tsv' \
    "$run_dir/tasks.json" | tee "$run_dir/live-tasks.tsv"

  codex exec --full-auto --add-dir "$run_dir" <<EOF
Read and follow /home/mekaneeky/repos/SN94-BitSota/docs/guides/autoresearch-agent-master-prompt.md.

Production coordinator:
$BITSOTA_COORDINATOR_URL

Selected task slug:
$BITSOTA_TASK_SLUG

Runtime directory:
$run_dir

Do not use the retired local launcher wrapper or the GUI launcher.
Use curl or wget to discover the live task metadata.
Use only public repos and public coordinator routes.
Keep wallet secrets out of logs, patches, summaries, and shell history.

Minimum submission criteria:
- establish a local baseline first
- run the same local evaluation after the patch
- submit only if local heldout/proxy PPL improves by the operator-set threshold
- reject changes that only improve speed while worsening PPL unless the task ranking explicitly says that is acceptable
- keep edits inside allowed_patch_paths
- record benchmark command, local baseline metric, candidate metric, and files changed
EOF

  sleep "$BITSOTA_SLEEP_SECONDS"
done
```

When this is stable, increase `BITSOTA_MAX_ROUNDS`. Use an infinite loop only
after several clean supervised rounds.

## Minimum Submission Criteria

Give Codex a concrete gate before every run. Examples:

| Gate | Why it matters |
| --- | --- |
| Local benchmark exits `0` before and after the patch | Separates mining from broken setup. |
| Candidate improves local PPL by a fixed threshold | Avoids spam submissions from noise. |
| Same eval set for baseline and candidate | Makes deltas comparable. |
| Patch only touches `allowed_patch_paths` | Validator rejects out-of-surface patches. |
| No generated files in the patch | Keeps submissions reviewable and replayable. |
| Summary includes exact command and metric | Makes failed validator replay easier to debug. |

Do not tell Codex “submit anything that runs.” Tell it exactly what delta is
worth submitting.

## Local PPL Eval Set Tips

For PPL-style competitions, keep a local proxy eval set for iteration. This is
not the validator heldout set and should not be presented as the validator
score.

Practical rules:

- keep the eval set outside the task repo or in an ignored local-only directory;
- use the same examples for baseline and candidate;
- include enough variety to catch obvious overfitting;
- keep it small enough that Codex can run it every loop;
- never commit the eval set or tune the public benchmark to it;
- report local proxy PPL separately from backend `heldout_ppl`.

If the public task repo supports a custom eval corpus flag, use that. If it does
not, ask Codex to write a local-only evaluation script outside the submitted
patch surface.

Example instruction to include in the Codex prompt:

```text
Before submitting, evaluate baseline and candidate on my local proxy PPL set at
$HOME/bitsota-local-evals/qwen-ppl/*.txt. Treat this only as a local gate.
Submit only if the candidate improves local proxy PPL by at least <threshold>
and the task benchmark still passes. Do not add this eval set to the patch.
```

## Failure Handling

Tell Codex to stop and report instead of guessing when:

- the selected slug is not live;
- `allowed_patch_paths` is empty or unclear;
- onboarding contradicts task metadata;
- benchmark output does not expose the configured metric;
- signing is blocked by missing wallet material;
- the local proxy eval improves but the task benchmark worsens.

## Related Pages

- [Autoresearch Backend Routes](autoresearch-backend-routes.md)
- [Mining Without an Agent](mining.md)
