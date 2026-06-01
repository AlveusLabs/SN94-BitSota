# 奖励与激励机制

BitSota 当前生产奖励主要通过两套系统流动：

1. Bittensor 子网 emissions，由验证者 `set_weights` 指向。
2. Pool reputation rewards，在 pool epoch 边界换算为 RAO。

旧的 relay SOTA 投票、本地 validator winner 模式与 Capacitor reward 路径
已经从 public validator 流程中移除。生产 validator 权重来自 autoresearch
backend reward snapshot，并由 `validator.backend_weight_setter` 应用。

## Bittensor 网络 Emissions

Bittensor 会随区块持续分配子网 emissions。验证者基于质押和子网表现获得
收益。矿工根据验证者设置的权重获得 emissions。

当前 SN94 生产权重方向由 backend 控制：

```text
Autoresearch backend reward snapshot -> validator.backend_weight_setter -> Bittensor set_weights
```

backend policy 应该解析为：

```text
90% UID 0
10% 5F7MJ2fAyxBG7ci4xP7kQPJanoMdNurk1QBP1AQuFT2Jmzg2
```

必须通过 metagraph 动态解析 contract hotkey。不要硬编码它的 UID。

## 矿池挖矿奖励

Pool miner 通过 reputation system 获得奖励，reputation 会在 epoch 边界换算
为 RAO。

### Reputation 累计

评估任务：

- Evaluator points 来自严格 consensus cluster 上的一致性质量。
- 一致与不一致由可配置的 `tolerance_ratio` 和 `consensus_threshold` 衡量。
- 有效 evaluator points 近似为
  `(agreements - disagreements) * evaluator_weight`。

演化任务：

- Evolver points 基于相对配置 baseline 的已验证改进，baseline 可为
  `sota`、`genealogy` 或 `local_evolver`。
- 有效 evolver points 近似为
  `(total_improvement * evolver_weight) - repetition_penalty`。
- 重复惩罚是可选项，范围可配置为 `miner`、`global` 或 `both`。

Consensus gating：

- 正向奖励要求 `in_consensus == true`。
- 正向奖励也要求当前窗口活动量达到 `min_reward_activity`。

### Epoch 换算

epoch 结束时：

```text
Your RAO = (Your Reputation / Total Pool Reputation) x Epoch Budget x (1 - Pool Fee)
```

示例：

```text
RAO = (21 / 5000) x 1,000,000,000 x 0.95
    = 3,990,000 RAO
```

## Validator 运行

验证者应该运行：

- `validator.research_validator_runner`：执行 backend 分配的 replay validation。
- `validator.backend_weight_setter`：执行 backend 指定的链上权重设置。

当前生产 runbook 请使用 [Public Autoresearch Validator Runner](public-validator-runner.md)。

## 相关指南

- [直接挖矿](mining.md)
- [矿池挖矿](pool-mining.md)
- [验证者指南](validation.md)
- [奖励方向](reward-modes.md)
