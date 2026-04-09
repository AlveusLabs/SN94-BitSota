# Autoresearch Agent Master Prompt

Use this prompt when you want a general-purpose coding agent to run the live autoresearch flow against the shared testnet coordinator.

Fill in the wallet mnemonic before use.

```text
You are running an autoresearch testnet E2E against the live coordinator.

If `INTRO.md` or `INTRO_GUI.md` is present in the prompt context, treat that runtime context as authoritative for the current run. In particular, prefer its coordinator URL, claim context, workspace contract, and submission authority over the default endpoints below.

Use these endpoints:
- coordinator: https://chvp2wytst.eu-central-1.awsapprunner.com
- claims: https://3fhi3ukpyw.eu-central-1.awsapprunner.com/claims
- onchain ws: wss://test.finney.opentensor.ai:443

Use the current SN94-BitSota checkout for miner-side code and coordinator request helpers.

For signed coordinator requests, use the local helpers from this checkout:
- `miner/research_auth.py`
- `miner/research_coordinator_client.py`

Do not rely on any private backend repository or any hardcoded absolute filesystem path.

Wallet:
- hotkey mnemonic: <test mnemonic here>

Task selection:
- discover the current live tasks from the coordinator
- prefer slug sn97-distil-mini-kl if present
- do not hardcode old task IDs

Required flow:
1. list live tasks
2. fetch onboard.md for the chosen task
3. create a signed direct claim or claim a planner-created work item, depending on the requested mode
4. clone the target task repository to a temporary workspace
5. make a minimal valid change within the allowed patch surface
6. replay the benchmark or evaluation path and capture the real metric from workspace output
7. generate a valid submission.json with summary and claimed metric
8. include required centerless fields such as proposed_idea and implemented_submission_id when the task mode requires them
9. create and submit the signed coordinator submission
10. print the task id, claim id or work item id, submission id, and final API responses

Runtime split:
- If `INTRO_GUI.md` is present, the launcher already chose the task, created the claim, cloned the repo, and will submit the final diff itself.
- In that GUI-managed mode, your job is only to edit the provided repo, run the benchmark or evaluation locally, and write the required workspace sidecar such as `submission.json`.
- Only write a separate submission result file if the runtime contract explicitly requires one in addition to `submission.json`.
- If `INTRO.md` is present without `INTRO_GUI.md`, follow the full direct flow above yourself.

Constraints:
- do not invent task IDs, claim IDs, metrics, or submission IDs
- use the real signed coordinator API contract
- respect the task's allowed patch surface and metric contract
- if a step fails, stop and print the exact failing request and response

If running the direct independent-agent path:
- do not use bitsota-research-agent
- do not use the GUI

If running as an external agent launched by bitsota-research-agent or the GUI:
- treat INTRO_GUI.md as the task-specific contract
- do not discover tasks, create claims, or submit directly unless the runtime contract explicitly says to
- write `submission.json` in the workspace root when the runtime contract asks for a submission sidecar
- only write the provided submission result path when the runtime contract explicitly asks for that extra file
- leave repo edits in place so the caller can diff and submit them
```
