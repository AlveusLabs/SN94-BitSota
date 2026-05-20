# Public Autoresearch Validator Runner

SN94 exposes a public validator client in this repo so an operator can replay
autoresearch submissions without running the private coordinator database worker.
The default path is a signed backend worklist scan: the validator calls
`POST /api/v1/validator/submissions/scan`, replays every returned submission and
`replay_spec`, then submits each score to `POST /api/v1/validator/jobs/{job_id}/result`.

Default coordinator:

```text
https://chvp2wytst.eu-central-1.awsapprunner.com
```

Config-file run:

```bash
cd /home/mekaneeky/repos/SN94-BitSota
cp research_validator_config.yaml.example research_validator_config.yaml
# Edit coordinator_url, wallet_name, and wallet_hotkey.
python -m validator.research_validator_runner --config research_validator_config.yaml
```

Script wrapper:

```bash
cd /home/mekaneeky/repos/SN94-BitSota
python scripts/research_validator_runner.py --config research_validator_config.yaml
```

Installed console script:

```bash
bitsota-research-validator --config research_validator_config.yaml
```

Run one replay by overriding the config:

```bash
python -m validator.research_validator_runner \
  --config research_validator_config.yaml \
  --once
```

Useful options:

- `--config`: read runner, wallet, and replay settings from a YAML, `.config`,
  or JSON file. CLI flags override config-file values.
- `--task-slug` or `--task-id`: restrict replay to one task.
- `--hotkey-mnemonic` or `--wallet-file`: use the same SN94 wallet input helpers as the research-agent miner.
- `--dry-run`: claim and replay locally but do not post the job result.
- `--replay-sandbox-mode`: `docker` runs submitted setup and benchmark commands
  in the Docker/CUDA replay sandbox. `host` is a fallback and requires
  `--allow-unsafe-host-replay`.
- `--replay-sandbox-gpus`: Docker `--gpus` value. Use `all` on a CUDA validator
  host.
- `--claim-path`: override the signed backend worklist endpoint. Use
  `/api/v1/validator/jobs/claim` only for legacy single-job compatibility.
- `--pending-submissions-fallback`: use the older public pending-submissions scan when testing an undeployed backend.
- `--allow-local-artifacts`: allow `file://` or relative artifact URIs during local testing.

The config file intentionally does not include holdout dataset names,
percentages, or sync numbers. Those values come from the backend in the signed
worklist response after validator auth and on-chain checks pass.

Current backend compatibility:

- The matching backend exposes signed `POST /api/v1/validator/submissions/scan`
  and `POST /api/v1/validator/jobs/{job_id}/result`.
- The worklist response includes all recent unseen `submission`/`replay_spec`
  jobs for the validator, plus validator-only replay parameters including
  hidden holdout handles and sync numbers.
  Validators do not need prod database, App Runner, or admin credentials.
- The fallback path is only for older backends or backends with legacy direct
  verification explicitly enabled. It uses `GET /api/v1/submissions?status=pending_verification`,
  `GET /api/v1/tasks`, `GET /api/v1/submissions/{id}/detail`, and signed
  `POST /api/v1/submissions/{id}/verify`.
- The pending-submissions fallback can only rebuild a replay spec from public task
  metadata. If a competition depends on a backend-pinned `replay_spec` or hidden
  validator replay parameter that is not exposed by the public API, use the
  signed validator worklist endpoint that returns those fields.

Docker/CUDA sandbox:

Recommended production validator replay uses Docker mode:

```yaml
replay_sandbox_mode: "docker"
allow_unsafe_host_replay: false
replay_sandbox_image: "bitsota-research-validator-cuda:local"
replay_sandbox_dockerfile: "docker/research-validator-cuda.Dockerfile"
replay_sandbox_gpus: "all"
replay_sandbox_setup_network_mode: "bridge"
replay_sandbox_benchmark_network_mode: "bridge"
replay_sandbox_memory_limit: "16g"
replay_sandbox_pids_limit: 512
replay_sandbox_workspace_size_bytes: 2147483648
```

Validator host prerequisites:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

The sandbox image is built automatically from
`docker/research-validator-cuda.Dockerfile` if the configured tag does not
already exist. The runner copies only the prepared replay workspace into a
tmpfs-backed Docker volume, runs the optional setup command and benchmark command
in read-only containers, and copies only the configured result file back out.
The validator wallet and wallet files stay outside the sandbox.

`bridge` networking is the default for setup and benchmark so task repos can
install dependencies and validator replay can fetch backend-directed Hugging
Face heldout handles. Operators can set the benchmark network to `none` only
when every dependency, model, artifact, and heldout byte is already local or
prefetched.

Host mode:

Host mode executes submitted setup and benchmark commands directly on the
validator machine and therefore requires `--allow-unsafe-host-replay` or
`allow_unsafe_host_replay: true`. Use it only for local development or a
throwaway isolated validator host. Benchmark commands receive a scrubbed
environment containing only minimal process variables and backend-supplied
replay parameters. The script signs `/verify` requests with the validator hotkey
in fallback mode and signs validator job requests in the default mode. The
backend still enforces validator allowlisting.

Patch-surface enforcement is strict before replay. The public runner rejects any
submitted patch path that is not in the task `allowed_patch_paths`, rejects
generated Python bytecode/cache paths, and rejects patches larger than
`max_patch_bytes` when the backend or task provides it. The default cap is
262,144 bytes.
