# 部署

本仓库的部署形态包括：

- 面向矿工的桌面 GUI
- Relay 服务
- Pool 服务
- 验证者节点

本页是各组件的部署检查清单。

## Relay

- 提供可写的 `DATABASE_URL`  开发用 SQLite，生产用 Postgres
- 为管理接口设置 `ADMIN_AUTH_TOKEN`
- 如果使用基于 metagraph 的验证者 allowlist，请设置 `RELAY_NETUID` 与 `RELAY_NETWORK`
- 用 `RELAY_LOG_LEVEL` 与 `RELAY_LOG_FILE` 配置日志

推荐的生产运行方式：

- 置于反向代理之后
- 在代理处终止 TLS
- 显式配置 CORS origins
- 使用 Postgres 并做好备份

## Pool

- 在环境变量中提供 Postgres 连接设置
- 设置 `ENVIRONMENT=production`
- 若公开监控端点，请设置 `MONITOR_TOKEN`

如果你在本地运行 sim 栈，请使用 `Pool/docker-compose.sim.yaml`。

## Validator

- 确保钱包文件在磁盘上存在且可读
- 配置 `validator_config.yaml` 与 `validator_hyperparams.json`
- 规划数据集缓存与磁盘占用

## GUI

- 桌面构建以二进制形式发布，本地开发可用 `python3 -m gui`
