# Autoresearch Agent Master Prompt

Use this prompt when you want a general-purpose coding agent to run the live autoresearch flow against the shared testnet coordinator.

Fill in the wallet mnemonic before use.

```text
You are running an autoresearch testnet E2E against the live coordinator.

Use these endpoints:
- coordinator: https://chvp2wytst.eu-central-1.awsapprunner.com
- claims: https://3fhi3ukpyw.eu-central-1.awsapprunner.com/claims
- onchain ws: wss://test.finney.opentensor.ai:443

Use this repo for miner-side code and examples:
- /home/mekaneeky/repos/SN94-BitSota

Use this repo for signed coordinator requests:
- /home/mekaneeky/repos/autoresearch-bittensor
- import and use autoresearch_bittensor.auth.hotkey.sign_request

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
- write the final structured result to the provided submission result path
- leave repo edits in place so the caller can diff and submit them
```
