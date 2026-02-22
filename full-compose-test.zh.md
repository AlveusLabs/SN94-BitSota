# 使用 Docker Compose 跑通完整本地闭环：relay + validator + 本地 GUI miners

该方案会运行：
- Relay  FastAPI  在 Docker 中运行，使用 `--test` 模式
- 本地 validator  轮询 relay 并投票  在 Docker 中运行
- 3 个 GUI miner 在宿主机本地运行  每个都会启动自己的 sidecar 与本地 miner 进程

## 0  前置条件

- Docker 与 Docker Compose 可用：`docker ps` 与 `docker compose version`
- 宿主机上的 GUI miners Python 环境  完整步骤见 `docs/local-testing.md`
- 宿主机上有一个 validator 钱包 hotkey  将挂载到 validator 容器中

## 1  在宿主机创建 validator 钱包 hotkey

如果你还没有：

```bash
btcli wallet new_coldkey --wallet.name local_val
btcli wallet new_hotkey --wallet.name local_val --wallet.hotkey local_val_hot
```

compose 文件会把宿主机钱包目录挂载到容器：
- `${HOME}/.bittensor/wallets` → validator 容器中的 `/wallets`

compose 文件使用的默认值：
- `VALIDATOR_WALLET_NAME=local_val`
- `VALIDATOR_WALLET_HOTKEY=local_val_hot`

如果想改成别的名字，请在启动 compose 前覆盖：

```bash
export VALIDATOR_WALLET_NAME=local_val
export VALIDATOR_WALLET_HOTKEY=local_val_hot
```

## 2  使用 Docker Compose 启动 relay 与 validator

在仓库根目录执行：

```bash
docker compose -f docker-compose.full-test.yaml up -d --build
docker compose -f docker-compose.full-test.yaml ps
```

在宿主机上对 relay 做快速检查：

```bash
curl http://127.0.0.1:8002/health
curl http://127.0.0.1:8002/sota_threshold
curl "http://127.0.0.1:8002/sota-events?page=1&page_size=10"
```

## 3  配置 GUI 指向本地 relay

在仓库根目录创建或更新 `gui_config.json`：

```json
{
  "relay_endpoint": "http://127.0.0.1:8002",
  "update_manifest_url": "http://127.0.0.1:8002/version.json",
  "test_mode": true,
  "test_invite_code": "TESTTEST1",
  "miner_validate_every_n_generations": 1000,
  "problem_config_path": "./problem_config.json"
}
```

确保你有一个 problem config：

```bash
cp -n problem_config.json.example problem_config.json
```

重要：`test_mode: true` 会禁用 GUI 单实例锁，因此你可以同时运行 3 个 GUI miner。

## 4  以测试模式运行 3 个本地 GUI miner

每个 GUI 实例必须使用不同的 sidecar 端口。

终端 1：

```bash
export BITSOTA_SIDECAR_PORT=8123
python3 -m gui
```

终端 2：

```bash
export BITSOTA_SIDECAR_PORT=8124
python3 -m gui
```

终端 3：

```bash
export BITSOTA_SIDECAR_PORT=8125
python3 -m gui
```

在每个 GUI 窗口中：
- 为该 miner 选择一个钱包 hotkey  若希望它们是不同矿工，请使用不同 hotkey
- 点击 “Start Mining”

## 5  查看日志与基础监控

### Docker 日志

Relay：

```bash
docker compose -f docker-compose.full-test.yaml logs -f relay
```

Validator：

```bash
docker compose -f docker-compose.full-test.yaml logs -f validator
```

两者一起：

```bash
docker compose -f docker-compose.full-test.yaml logs -f
```

### Validator 指标文件

validator 会把 JSONL 指标写入容器内的 `/data/local_validator_metrics.log`。

查看：

```bash
docker compose -f docker-compose.full-test.yaml exec validator tail -f /data/local_validator_metrics.log
```

如需关闭指标日志，请设置：

```bash
export VALIDATOR_METRICS_LOG=""
docker compose -f docker-compose.full-test.yaml up -d --build --force-recreate
```

### GUI 日志

每次 GUI 运行都会把调试日志写到：
- `~/.bitsota/logs/`

## 6  停止与清理

停止容器：

```bash
docker compose -f docker-compose.full-test.yaml down
```

同时移除 validator 数据卷  会清理缓存的数据集与指标日志：

```bash
docker compose -f docker-compose.full-test.yaml down -v
```

## 故障排查

- validator 找不到钱包文件：确认宿主机上 `${HOME}/.bittensor/wallets` 存在，并包含你配置的钱包名与 hotkey。
- relay 起来了但 validator 启动时报错：查看 `docker compose -f docker-compose.full-test.yaml logs validator` 获取具体异常。
- 多个 GUI miner 无法启动：确认每个实例使用唯一的 `BITSOTA_SIDECAR_PORT`，并且 GUI 配置中设置了 `test_mode: true`。
