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

Dashboard 与 JSON 指标重点：
- 多来源对比 UI 支持把多个 simulation URL 叠加到同一奖励分布图并按来源开关显示。
- 分布数据包含总奖励/评估奖励/演化奖励三条曲线：
  - `distribution.curve_total_reward`
  - `distribution.curve_eval_reward`
  - `distribution.curve_evolve_reward`
- 为排查好坏矿工奖励问题，提供按角色拆分指标：
  - `distribution.evaluator_good_reward_share`、`distribution.evaluator_bad_reward_share`
  - `distribution.evolver_good_reward_share`、`distribution.evolver_bad_reward_share`
  - 以及各组被奖励矿工数量。

链上桥接行为：
- 若配置了 `ONCHAIN_WS_URL`、`ONCHAIN_CONTRACT` 及签名账户变量，`consensus_daemon.py` 会尝试链上调用。
- 若未配置，则保持本地/链下流程。

## Validator

- `validator.local_validator` 默认把 JSONL 指标写入 `local_validator_metrics.log`
- 用 `--relay-client-log-level WARNING` 降低 HTTP 轮询日志噪声
