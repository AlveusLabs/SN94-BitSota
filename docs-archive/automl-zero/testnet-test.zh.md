# 测试网测试配方

本指南新增两个 Docker 配方：

- 测试网 relay + validators
- GUI 打包构建

## 1) Relay + Validators

相关文件：

- `docker-compose.testnet-relay-validators.yaml`
- `.env.relay-validators.example`
- `docker/validator-node.Dockerfile`
- `docker/run-validator.sh`
- `docker/relay.Dockerfile`

默认使用线上 relay：

```bash
cp .env.relay-validators.example .env.relay-validators
# 修改钱包名称与 hotkey
docker compose --env-file .env.relay-validators -f docker-compose.testnet-relay-validators.yaml up -d --build validator_1 validator_2
```

可选本地 relay profile：

```bash
export RELAY_URL=http://relay:8002
docker compose --env-file .env.relay-validators -f docker-compose.testnet-relay-validators.yaml --profile local-relay up -d --build
```

停止：

```bash
docker compose -f docker-compose.testnet-relay-validators.yaml down
```

## 2) GUI 打包构建

相关文件：

- `docker-compose.gui-build.yaml`
- `docker/gui-build.Dockerfile`

构建：

```bash
docker compose -f docker-compose.gui-build.yaml build
docker compose -f docker-compose.gui-build.yaml run --rm gui_builder
```

产物目录：

- `dist/`
- `build/`
