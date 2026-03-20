# Research-Agent Mining

This is an experimental, additive mining path for coordinator-backed research competitions.

It does **not** replace the current SN94 evolutionary miner.

Keep this mental model simple:

- `neurons/miner.py` is the existing SN94 miner for the evolutionary AutoML-Zero style contest.
- `bitsota-research-agent` is a new local miner that talks to a coordinator API and an OpenAI-compatible LLM API.

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

## Requirements

- local Bittensor hotkey or a test mnemonic
- coordinator API URL
- OpenAI-compatible chat completions URL
- model name

The chat endpoint should support an OpenAI-style `POST /v1/chat/completions` request.

## CLI

After `pip install -e .`, the new command is:

```bash
bitsota-research-agent --help
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

Run one direct mining pass:

```bash
bitsota-research-agent mine-once \
  --coordinator-url http://127.0.0.1:8000 \
  --llm-base-url http://127.0.0.1:11434/v1 \
  --llm-model gpt-4o-mini \
  --task-slug nanogpt-default \
  --participation-style direct \
  --wallet-name default \
  --wallet-hotkey default
```

Run one shared-task pool pass:

```bash
bitsota-research-agent mine-once \
  --coordinator-url http://127.0.0.1:8000 \
  --llm-base-url http://127.0.0.1:11434/v1 \
  --llm-model gpt-4o-mini \
  --task-slug nanogpt-default \
  --participation-style pool \
  --wallet-name default \
  --wallet-hotkey default
```

Run one peer evaluation:

```bash
bitsota-research-agent peer-evaluate-once \
  --coordinator-url http://127.0.0.1:8000 \
  --llm-base-url http://127.0.0.1:11434/v1 \
  --llm-model gpt-4o-mini \
  --task-slug nanogpt-default \
  --wallet-name default \
  --wallet-hotkey default
```

Run in a loop:

```bash
bitsota-research-agent loop \
  --coordinator-url http://127.0.0.1:8000 \
  --llm-base-url http://127.0.0.1:11434/v1 \
  --llm-model gpt-4o-mini \
  --task-slug nanogpt-default \
  --participation-style pool \
  --allow-peer-evaluation
```

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
