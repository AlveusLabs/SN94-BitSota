# 验证者奖励方向

生产 SN94 验证者不再选择 relay consensus、本地 winner、windowed
capacitorless 或旧的 Capacitor contract 路径。这些验证者奖励模式属于旧
relay validator stack，已经从 public validator 路径中移除。

当前生产权重方向由 backend 控制：

```text
Autoresearch backend reward snapshot -> validator.backend_weight_setter -> Bittensor set_weights
```

backend weight setter 会获取：

```text
GET https://autoresearch.bitsota.com/api/v1/reward-snapshot
```

它读取 `reward_policy.validator_weights`，并且只应用这些 targets。weight
setter 默认使用 `https://autoresearch.bitsota.com`；只有明确测试 backend
时才传 `--coordinator-url`。

## 当前策略

生产策略应该解析为：

```text
90% UID 0
10% 5F7MJ2fAyxBG7ci4xP7kQPJanoMdNurk1QBP1AQuFT2Jmzg2
```

必须通过当前 metagraph 动态解析 contract hotkey。不要硬编码它的 UID。

## 验证者命令

使用当前 runbook：

- [Public Autoresearch Validator Runner](public-validator-runner.md)
- [验证者指南](validation.md)

不要运行旧 relay/local validator service，也不要运行单独的本地 weight
setter。唯一的生产权重设置服务应该是 `validator.backend_weight_setter`。
