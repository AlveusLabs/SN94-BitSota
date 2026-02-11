# Monitoring

## Relay

- `GET /health`
- `GET /sota_threshold`
- `GET /sota-events`
- `GET /admin/status` (admin auth required) returns JSON health + request-rate metrics
- `GET /admin/dashboard` (admin auth required) shows a live HTML dashboard
- `GET /docs` for interactive OpenAPI (local)

Logs:
- Set `RELAY_LOG_LEVEL` and optionally `RELAY_LOG_FILE`
- Use the `X-Request-ID` response header to correlate request logs

## Pool

- `GET /health`
- `GET /api/v1/monitor/summary` (optional `X-Monitor-Token`)
- `GET /docs` for interactive OpenAPI (local)

When using `Pool/docker-compose.sim.yaml`:
- Monitor UI is published on `http://127.0.0.1:9000`

Pool-specific observability checks:
- `GET http://127.0.0.1:9000/metrics.json` for stack summary.
- `docker compose -f Pool/docker-compose.sim.yaml logs -f consensus_publisher`
- `docker compose -f Pool/docker-compose.sim.yaml logs -f consensus_verifier_1`
- Inspect epoch artifacts in `Pool/.local_sim/epochs`:
  - `epoch_<n>.json`
  - `verify_<epoch>_<node>.json`
  - `onchain_publish_<epoch>.json` (when on-chain bridge is enabled)
  - `onchain_challenge_<epoch>_<node>.json` (when challenges are submitted)

Dashboard and JSON metrics highlights:
- Multi-source comparison UI supports toggling multiple simulation URLs on the same reward-distribution chart.
- Distribution JSON includes total/eval/evolve curves:
  - `distribution.curve_total_reward`
  - `distribution.curve_eval_reward`
  - `distribution.curve_evolve_reward`
- Role-segmented reward quality is exposed for debugging:
  - `distribution.evaluator_good_reward_share`, `distribution.evaluator_bad_reward_share`
  - `distribution.evolver_good_reward_share`, `distribution.evolver_bad_reward_share`
  - plus rewarded miner counts for each group.

On-chain bridge behavior:
- If `ONCHAIN_WS_URL`, `ONCHAIN_CONTRACT`, and signer vars are set, `consensus_daemon.py` attempts chain calls.
- If unset, flow remains local/off-chain only.

## Validator

- `validator.local_validator` writes JSONL metrics to `local_validator_metrics.log` by default
- Reduce HTTP poll log noise with `--relay-client-log-level WARNING`
