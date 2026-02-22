# Validator

验证者会：

- 轮询 relay 拉取矿工提交
- 在确定性的任务套件上重新评估候选
- 通过投票最终确定 SOTA 事件，并设置链上权重

## 入口

- `python neurons/validator_node.py` 运行主验证者节点
  - 可选：`--accept-test` 启用仅 UID0 的 relay 测试提交评估，用于日志，不会设置权重
- `python3 -m validator.local_validator` 运行用于测试的本地验证者，主要面向 relay

配置与调参请参考 [验证](../validation.md) 与 [配置参考](../configuration.md) 中的 `validator_config.yaml`。
