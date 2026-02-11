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

Pool 侧可观测性检查：
- `GET http://127.0.0.1:9000/metrics.json` 查看整栈摘要。
- `docker compose -f Pool/docker-compose.sim.yaml logs -f consensus_publisher`
- `docker compose -f Pool/docker-compose.sim.yaml logs -f consensus_verifier_1`
- 查看 `Pool/.local_sim/epochs` 中的产物：
  - `epoch_<n>.json`
  - `verify_<epoch>_<node>.json`
  - `onchain_publish_<epoch>.json`（启用链上桥接时）
  - `onchain_challenge_<epoch>_<node>.json`（提交挑战时）

链上桥接行为：
- 若配置了 `ONCHAIN_WS_URL`、`ONCHAIN_CONTRACT` 及签名账户变量，`consensus_daemon.py` 会尝试链上调用。
- 若未配置，则保持本地/链下流程。

## Validator

- `validator.local_validator` 默认把 JSONL 指标写入 `local_validator_metrics.log`
- 用 `--relay-client-log-level WARNING` 降低 HTTP 轮询日志噪声
