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

## Validator

- `validator.local_validator` writes JSONL metrics to `local_validator_metrics.log` by default
- Reduce HTTP poll log noise with `--relay-client-log-level WARNING`
