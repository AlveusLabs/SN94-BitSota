# Public Autoresearch Validator Runner

SN94 exposes a public validator client in this repo so an operator can replay
autoresearch submissions without running the private coordinator database worker.
The default path is a signed backend lease: the validator claims
`POST /api/v1/validator/jobs/claim`, replays the returned submission and
`replay_spec`, then submits `POST /api/v1/validator/jobs/{job_id}/result`.

Default coordinator:

```text
https://chvp2wytst.eu-central-1.awsapprunner.com
```

Run one replay:

```bash
cd /home/mekaneeky/repos/SN94-BitSota
python -m validator.research_validator_runner \
  --once \
  --coordinator-url https://chvp2wytst.eu-central-1.awsapprunner.com \
  --wallet-name <validator-wallet> \
  --wallet-hotkey <validator-hotkey> \
  --allow-unsafe-host-replay
```

Script wrapper:

```bash
cd /home/mekaneeky/repos/SN94-BitSota
python scripts/research_validator_runner.py \
  --once \
  --coordinator-url https://chvp2wytst.eu-central-1.awsapprunner.com \
  --wallet-name <validator-wallet> \
  --wallet-hotkey <validator-hotkey> \
  --allow-unsafe-host-replay
```

Installed console script:

```bash
bitsota-research-validator \
  --once \
  --coordinator-url https://chvp2wytst.eu-central-1.awsapprunner.com \
  --wallet-name <validator-wallet> \
  --wallet-hotkey <validator-hotkey> \
  --allow-unsafe-host-replay
```

Loop:

```bash
python -m validator.research_validator_runner \
  --coordinator-url https://chvp2wytst.eu-central-1.awsapprunner.com \
  --wallet-name <validator-wallet> \
  --wallet-hotkey <validator-hotkey> \
  --interval-seconds 30 \
  --allow-unsafe-host-replay
```

Useful options:

- `--task-slug` or `--task-id`: restrict replay to one task.
- `--hotkey-mnemonic` or `--wallet-file`: use the same SN94 wallet input helpers as the research-agent miner.
- `--dry-run`: claim and replay locally but do not post the job result.
- `--claim-path`: override the signed backend replay-job claim endpoint.
- `--pending-submissions-fallback`: use the older public pending-submissions scan when testing an undeployed backend.
- `--allow-local-artifacts`: allow `file://` or relative artifact URIs during local testing.

Current backend compatibility:

- The matching backend exposes signed `POST /api/v1/validator/jobs/claim`
  and `POST /api/v1/validator/jobs/{job_id}/result`.
- The lease response includes `submission`, `replay_spec`, lease timing, and warnings.
  Validators do not need prod database, App Runner, or admin credentials.
- The fallback path is only for older backends or backends with legacy direct
  verification explicitly enabled. It uses `GET /api/v1/submissions?status=pending_verification`,
  `GET /api/v1/tasks`, `GET /api/v1/submissions/{id}/detail`, and signed
  `POST /api/v1/submissions/{id}/verify`.
- The pending-submissions fallback can only rebuild a replay spec from public task
  metadata. If a competition depends on a backend-pinned `replay_spec` or hidden
  validator environment that is not exposed by the public API, use a backend lease
  endpoint that returns those fields.

Security note:

The incremental runner uses host execution for setup and benchmark commands and
therefore requires `--allow-unsafe-host-replay`. Run it only on an isolated
validator machine or container. Benchmark commands receive a scrubbed environment
containing only minimal process variables and backend-supplied benchmark env.
The script signs `/verify` requests with the validator hotkey in fallback mode
and signs validator job requests in the default mode. The backend still enforces
validator allowlisting.
