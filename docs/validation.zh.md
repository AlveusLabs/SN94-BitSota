# 验证者指南

SN94 生产验证者现在只使用 autoresearch backend 路径：

1. `validator.research_validator_runner` 会签名请求 backend worklist，
   在 Docker/CUDA sandbox 中 replay 分配到的提交，并把观测指标提交回 backend。
2. `validator.backend_weight_setter` 会轮询
   `https://autoresearch.bitsota.com/api/v1/reward-snapshot`，读取
   `reward_policy.validator_weights`，并提交 Bittensor `set_weights`。

旧的 relay validator、relay SOTA 投票与本地 winner 权重设置路径已经从
仓库中移除。不要运行任何会根据 relay/local SOTA state 调用 `set_weights`
的旧服务或进程。

## 生产设置

当前生产说明请使用 public runbook：

- [Public Autoresearch Validator Runner](public-validator-runner.md)
- [SN94 System Structure](sn94-system-structure.md)

最小进程模型：

```text
Autoresearch backend -> replay validator -> backend result consensus
Autoresearch backend -> backend weight setter -> Bittensor SN94 weights
```

生产 backend weight setting 默认使用：

```text
https://autoresearch.bitsota.com
```

当前 backend policy 应该指向：

```text
90% UID 0
10% 5F7MJ2fAyxBG7ci4xP7kQPJanoMdNurk1QBP1AQuFT2Jmzg2
```

必须通过当前 metagraph 动态解析 contract hotkey。不要硬编码它的 UID。

## 健康检查

```bash
systemctl status bitsota-replay-validator.service --no-pager
systemctl status bitsota-backend-weights.service --no-pager
journalctl -u bitsota-replay-validator.service -n 100 --no-pager
journalctl -u bitsota-backend-weights.service -n 100 --no-pager
```

不提交权重，只检查 backend policy：

```bash
python -m validator.backend_weight_setter \
  --config validator_config.weights.yaml \
  --dry-run \
  --ignore-rate-limit
```

如果 dry run 没有显示上面的 contract hotkey，不要启动 live weight service。
