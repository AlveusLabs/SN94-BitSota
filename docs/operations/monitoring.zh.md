# 监控

## Relay

- `GET /health`
- `GET /sota_threshold`
- `GET /sota-events`
- `GET /admin/status` 需要管理员认证 返回 JSON 健康状态与请求速率指标
- `GET /admin/dashboard` 需要管理员认证 展示实时 HTML 管理面板
- `GET /docs` 交互式 OpenAPI  本地

日志：
- 设置 `RELAY_LOG_LEVEL`，可选设置 `RELAY_LOG_FILE`
- 使用响应头 `X-Request-ID` 关联请求日志

## Pool

- `GET /health`
- `GET /api/v1/monitor/summary`  可选 `X-Monitor-Token`
- `GET /docs` 交互式 OpenAPI  本地

使用 `Pool/docker-compose.sim.yaml` 时：
- Monitor UI 发布在 `http://127.0.0.1:9000`

## Validator

- `validator.local_validator` 默认把 JSONL 指标写入 `local_validator_metrics.log`
- 用 `--relay-client-log-level WARNING` 降低 HTTP 轮询日志噪声
