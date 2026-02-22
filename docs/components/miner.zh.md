# Miner

运行挖矿代码主要有两条路径：

- 通过 sidecar 的桌面 GUI 模式
- 通过 `neurons/miner.py` 的 CLI 模式

## CLI 入口

CLI miner 读取 `miner_config.yaml`，并可运行在：

- 直接模式：提交到 relay
- 矿池模式：与 Pool API 交互

关于 `miner_config.yaml` 的键，请参考 [配置参考](../configuration.md)。

## 直接挖矿

参考 [直接挖矿](../mining.md)。

## 矿池挖矿

参考 [矿池挖矿](../pool-mining.md) 与 [Pool](pool.md)。
