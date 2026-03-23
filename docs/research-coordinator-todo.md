# Research Coordinator TODO

The production coordinator is the separate [`autoresearch-bittensor`](https://github.com/AlveusLabs/autoresearch-bittensor) service.

`current-sn-2` should not reimplement coordinator logic. It should integrate with that coordinator cleanly.

## Core Direction

- coordinator owns tasks, onboarding, work items, submissions, verification, and planner state
- external agent CLIs are first-class
- GUI is optional
- headless agentic flow is the primary path

## Launcher Responsibilities

`current-sn-2` should only:

- select a coordinator task or work item
- claim it
- clone the target repo
- write `INTRO.md` or `INTRO_GUI.md`
- launch an external agent CLI
- read `submission.json`
- compute `git diff`
- submit through the coordinator

## Modes

### GUI-managed mode

- generated file: `INTRO_GUI.md`
- launcher owns claim and submission
- agent edits files and writes `submission.json`
- launcher submits

### Autonomous mode

- generated file: `INTRO.md`
- agent is allowed to submit directly if wallet access is available
- helper command: `bitsota-research-agent submit-workspace`
- useful for Codex CLI / Claude Code CLI / Hermes style self-initiated loops

## Immediate Work

1. Ship thin external-agent support in `bitsota-research-agent`.
2. Keep the older OpenAI-compatible planner only as fallback.
3. Add GUI launch support for the same external-agent path.
4. Add presets for:
   - Codex CLI
   - Claude Code CLI
   - Hermes
5. Keep the task contract markdown-first via coordinator `onboard.md`.

## Non-Goals

- do not build a second research harness in `current-sn-2`
- do not duplicate planner logic already present in `autoresearch-bittensor`
- do not make GUI the only way to participate
