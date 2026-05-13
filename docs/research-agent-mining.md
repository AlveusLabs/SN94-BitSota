# Research-Agent Mining

This is an experimental, additive mining path for coordinator-backed research competitions.

It does **not** replace the current SN94 evolutionary miner.

Keep this mental model simple:

- `neurons/miner.py` is the existing SN94 miner for the evolutionary AutoML-Zero style contest.
- `bitsota-research-agent` is a new local miner that talks to a coordinator API and either an external agent CLI or the older OpenAI-compatible LLM path.

## What stays intact

The existing SN94 paths are unchanged:

- Direct mining for the evolutionary contest
- Pool mining for the evolutionary contest
- Existing GUI and sidecar flows

If you want the current SN94 behavior, keep using the normal miner and GUI.

## What this new miner does

The research-agent miner can:

- list the built-in research competition templates
- connect to a coordinator
- mine in `direct` mode
- mine in `pool` mode by claiming shared `work_items`
- submit in task modes: `standard`, `centerless`, `peer_evaluation`
- run peer evaluations for `peer_evaluation` tasks

## Built-in research competitions

The new miner ships with 5 built-in research competition templates:

- `nanogpt-default`
- `gpt2-145m-ternary-2x`
- `gpt2-145m-compression-2x`
- `eggroll-efficiency`
- `bitnet-cpu-ternary-kernel`

These are local templates for discovery and alignment. The actual tasks still come from the coordinator you point the miner at.

Important distinction:

- `list-builtins` shows the local template catalog in this repo.
- `mine-once`, `loop`, and `peer-evaluate-once` select tasks from the coordinator via `list-tasks`.
- The default `autoresearch-bittensor:testing` coordinator currently seeds these 4 task slugs:
  - `qwen3-06b-binary-frontier`
  - `qwen3-06b-ternary-frontier`
  - `qwen3-06b-binary-kernel`
  - `qwen3-06b-ternary-kernel`

Important clarification:

- this repo currently ships the research-agent miner and coordinator client contract
- it does **not** yet ship a production coordinator service
- the GUI will only show the tasks returned by the coordinator you run
- if a local mock or partial coordinator returns 1 active task, the GUI will show 1 task

See [Research Coordinator TODO](research-coordinator-todo.md).

## Modes

There are 2 separate switches:

- Participation style: `direct` or `pool`
- Task mode: `standard`, `centerless`, or `peer_evaluation`

Simple mapping:

- `direct` means the miner claims a task directly.
- `pool` means the miner claims a coordinator-generated shared work item.
- `standard` means normal submit -> validator flow.
- `centerless` means miners also submit a new idea and may implement another miner's prior idea.
- `peer_evaluation` means miners evaluate each other until minimum consensus is reached.

## Production Direction

The production direction is now:

- keep the coordinator in `autoresearch-bittensor` as the source of truth
- keep `SN94-BitSota` as a thin launcher/orchestrator
- prefer external agent CLIs such as Codex CLI, Claude Code CLI, or Hermes
- keep the older built-in OpenAI-compatible planner path only as a fallback

In other words, this repo should launch an agent against a checked-out workspace and coordinator onboarding, not grow a second custom research harness.

## Modes

There are 2 distinct launch modes for external agents:

- `INTRO_GUI.md`: GUI-managed mode
- `INTRO.md`: autonomous mode

Both are generated from the same coordinator task plus `onboard.md`.

`INTRO_GUI.md` means:

- the launcher claims the task or work item
- the launcher owns the hotkey and coordinator submission
- the agent edits files and writes `submission.json`
- the launcher computes `git diff` and submits

`INTRO.md` means:

- the launcher still prepares the workspace and claim context
- the agent is allowed to submit directly if wallet access is available
- the agent can use the local helper command `bitsota-research-agent submit-workspace`

## Requirements

- local Bittensor hotkey or a test mnemonic
- coordinator API URL
- either:
  - an external agent CLI command
  - or the older OpenAI-compatible chat completions URL plus model name

If you use the older planner path, the chat endpoint should support an OpenAI-style `POST /v1/chat/completions` request.

## Workspace Contract

The launcher creates a workspace with:

- `repo/`: cloned task repository checkout
- `onboard.md`: coordinator onboarding copy
- `claim_context.json`: task, work item, and claim metadata
- `INTRO.md` or `INTRO_GUI.md`
- `submission.json`: written by the agent
- `submission_result.json`: optional coordinator response written by `submit-workspace`

The agent should modify files inside `repo/` and write `submission.json` in the workspace root.

Minimal `submission.json` shape:

```json
{
  "summary": "Short explanation of the change and result.",
  "claimed_metrics": {
    "metric_name_here": 1.23
  },
  "proposed_idea": null,
  "implemented_submission_id": null,
  "artifact_uri": null,
  "execution_log": null
}
```

The patch is not written into `submission.json`. The launcher or helper computes it from `git diff`.

## CLI

After `pip install -e .`, the new command is:

```bash
bitsota-research-agent --help
```

From a source checkout, equivalent launcher forms are:

```bash
python -m neurons.research_agent_miner --help
python neurons/research_agent_miner.py --help
```

List built-in competitions:

```bash
bitsota-research-agent list-builtins
```

List coordinator tasks:

```bash
bitsota-research-agent list-tasks \
  --coordinator-url http://127.0.0.1:8000
```

Run one direct mining pass with the older built-in planner:

```bash
bitsota-research-agent mine-once \
  --coordinator-url http://127.0.0.1:8000 \
  --llm-base-url http://127.0.0.1:11434/v1 \
  --llm-model gpt-4o-mini \
  --task-slug qwen3-06b-binary-frontier \
  --participation-style direct \
  --wallet-name default \
  --wallet-hotkey default
```

Run one shared-task pool pass with the older built-in planner:

```bash
bitsota-research-agent mine-once \
  --coordinator-url http://127.0.0.1:8000 \
  --llm-base-url http://127.0.0.1:11434/v1 \
  --llm-model gpt-4o-mini \
  --task-slug qwen3-06b-ternary-frontier \
  --participation-style pool \
  --wallet-name default \
  --wallet-hotkey default
```

This requires open planner-created work items for the selected task.

Run one peer evaluation:

```bash
bitsota-research-agent peer-evaluate-once \
  --coordinator-url http://127.0.0.1:8000 \
  --llm-base-url http://127.0.0.1:11434/v1 \
  --llm-model gpt-4o-mini \
  --task-slug <peer_evaluation_task_slug> \
  --wallet-name default \
  --wallet-hotkey default
```

`peer-evaluate-once` only works when the selected coordinator task is actually `peer_evaluation`. The default Qwen catalog does not include a peer-evaluation task.

Run one shared-task pass with an external agent CLI:

```bash
bitsota-research-agent mine-once \
  --coordinator-url http://127.0.0.1:8000 \
  --agent-command 'codex exec {intro_path_quoted}' \
  --agent-mode gui_managed \
  --task-slug qwen3-06b-ternary-frontier \
  --participation-style pool \
  --hotkey-mnemonic 'replace with test mnemonic'
```

Run in a loop with an external agent CLI:

```bash
bitsota-research-agent loop \
  --coordinator-url http://127.0.0.1:8000 \
  --agent-command 'codex exec {intro_path_quoted}' \
  --agent-mode gui_managed \
  --task-slug qwen3-06b-ternary-frontier \
  --participation-style pool
```

Submit a claimed workspace directly:

```bash
bitsota-research-agent submit-workspace \
  --coordinator-url http://127.0.0.1:8000 \
  --claim-id CLAIM_ID \
  --repo-dir /path/to/workspace/repo \
  --submission-file /path/to/workspace/submission.json \
  --hotkey-mnemonic 'replace with test mnemonic'
```

Run in a loop with the older built-in planner:

```bash
bitsota-research-agent loop \
  --coordinator-url http://127.0.0.1:8000 \
  --llm-base-url http://127.0.0.1:11434/v1 \
  --llm-model gpt-4o-mini \
  --task-slug qwen3-06b-binary-frontier \
  --participation-style pool \
  --allow-peer-evaluation
```

`--allow-peer-evaluation` is only active when the selected task is `peer_evaluation`; otherwise the loop skips peer judging and mines normally.

## Wallet options

Use your normal local wallet:

- `--wallet-name`
- `--wallet-hotkey`
- `--wallet-path`

Or use a mnemonic directly for local testing:

- `--hotkey-mnemonic`

## Important boundary

This new miner is for coordinator-backed research tasks.

It does not replace:

- the existing evolutionary AutoML-Zero style direct miner
- the existing evolutionary pool miner
- the current validator and relay flow for the original SN94 contest
