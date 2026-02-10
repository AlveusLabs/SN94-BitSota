# 奖励模式

本仓库支持多种 **验证者奖励模式**，通过 `validator_config.yaml` 中的 `reward_mode` 配置。不同模式会改变验证者在验证 relay 提交后**如何奖励矿工**。

## 快速对比

| 模式 | 谁来支付矿工 | 权重何时变化 | 是否依赖 relay | Winner 来源 |
|------|--------------|--------------|----------------|-------------|
| `capacitorless_sticky` 默认 | **链上权重**：90% burn + 10% 给 SOTA winner | 新 SOTA 事件最终确定后  relay  或本地发现后 | 是 | `relay` 或 `local` |
| `capacitorless` + `windowed` | **链上权重**：事件窗口内 100% 给 winner，否则 burn | 事件边界 `alignment_mod` 与窗口结束 | 是 | `relay` 或 `local` |
| `capacitor` 已弃用 | Capacitor EVM 合约投票 `releaseReward` | 默认不以矿工为目标 | 可选 | 不适用  合约模式 |

## `capacitorless_sticky` 默认  Yuma 共识

在 `reward_mode: capacitorless_sticky` 或 `reward_mode: capacitorless` 中，验证者通过设置链上权重来引导 emissions。

**工作方式：**
1. 轮询 relay 获取矿工提交
2. 独立重新评估，选择超过 SOTA 的最佳有效候选
3. 选择 winner 来源 `capacitorless.winner_source`：
   - `relay`：在 relay 上投票，等待共识，SOTA 事件最终确定后设置权重
   - `local`：基于自身评估立即设置权重
4. 设置链上权重：90% burn_hotkey，10% winner_hotkey
5. 网络 emissions 通过 Yuma 共识按权重流动

**配置要点：**
- 不需要 EVM key
- 需要配置 `capacitorless.burn_hotkey`
- 提交与 SOTA 跟踪需要 relay
- winner 来源决定协同程度

## Winner 来源选项

`capacitorless_sticky` 与 `windowed` 模式都支持两种 winner 选择方式：

### Relay 共识  `winner_source: "relay"`

配置：
```yaml
capacitorless:
  winner_source: "relay"
  events_limit: 50
  event_refresh_interval_s: 60
```

行为：
- 验证者在 relay 上投票接受新的 SOTA 候选
- relay 聚合多个验证者投票
- 达成共识后，relay 最终确定 SOTA 事件
- 验证者拉取最终事件并更新权重
- 所有验证者收敛到同一 winner

适用场景：
- 子网协同行为更一致
- 所有验证者对 winner 达成一致
- 生产环境推荐

### 本地模式  `winner_source: "local"`

配置：
```yaml
capacitorless:
  winner_source: "local"
  min_winner_improvement: 0.0
  apply_weights_inline: true
  submit_sota_votes: true  # 仍向 relay 投票以便协调
```

行为：
- 验证者跟踪自身评估过的最佳分数
- 当发现更优候选时立即更新本地 winner
- 不等待 relay 共识即可设置权重
- 验证者可能暂时选择不同 winner
- 可选：仍向 relay 提交投票

适用场景：
- 更快的权重更新
- 对突破的响应更低延迟
- 适合独立运行或测试

## Sticky burn split 模式  默认

配置：
- `reward_mode: capacitorless_sticky` 或 `reward_mode: capacitorless`
- `capacitorless.mode: sticky_burnsplit` 省略时默认
- `capacitorless.burn_share: 0.9` 以及可选的 `capacitorless.winner_share: 0.1`

行为：
- 始终设置权重为 `{burn_hotkey: 0.9, winner_hotkey: 0.1}`
- winner 由 `winner_source` 决定  relay 或本地
- 仅当识别出新 winner 时权重才会变化
- 无时间窗口或过期概念

适用场景：
- 简单且可预测
- SOTA winner 持续获得奖励
- 具备通缩属性  90% burn

## Windowed 模式  限时奖励

配置：
- `reward_mode: capacitorless`
- `capacitorless.mode: windowed`
- `capacitorless.alignment_mod`：区块间隔，例如 360
- `capacitorless.winner_source`：`relay` 或 `local`

行为：
- 默认权重为 `{burn_hotkey: 1.0}`
- 在活动 SOTA 事件窗口 `[start_block, end_block)` 内，将权重设为 `{winner_hotkey: 1.0}`
- winner 由 `winner_source` 决定  relay 事件或本地最优
- 窗口结束后回退为 burn
- 新事件会在下一事件开始时切断旧事件

适用场景：
- 每次 SOTA 仅有有限奖励窗口
- 若网络没有持续改进，emissions 会回到 burn
- 对持续创新形成驱动

## 区块号与同步要求

- 所有 capacitorless 模式在提交 relay SOTA 投票时都需要区块号 `seen_block`
- windowed 模式使用当前链上区块来判断事件是否处于活动状态，并在边界处应用变化
- sticky 模式在 relay 模式下切换为最新最终事件的 winner，在本地模式下切换为本地最优，但仍需遵循 Bittensor 的 epoch 规则与权重限速

## 最小配置示例

**Sticky 模式  relay 共识  推荐：**
```yaml
reward_mode: "capacitorless_sticky"
relay:
  url: "https://relay.bitsota.com"
capacitorless:
  burn_hotkey: "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
  burn_share: 0.9
  winner_source: "relay"
  events_limit: 50
  event_refresh_interval_s: 60
```

**Sticky 模式  本地评估  更快：**
```yaml
reward_mode: "capacitorless_sticky"
relay:
  url: "https://relay.bitsota.com"
capacitorless:
  burn_hotkey: "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
  burn_share: 0.9
  winner_source: "local"
  min_winner_improvement: 0.0
  apply_weights_inline: true
  submit_sota_votes: true
```

**Windowed 模式：**
```yaml
reward_mode: "capacitorless"
relay:
  url: "https://relay.bitsota.com"
capacitorless:
  mode: "windowed"
  burn_hotkey: "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
  winner_source: "relay"
  alignment_mod: 360
```

## Capacitor 模式  已弃用

原始的 EVM 合约投票模式仍可用，但已弃用：

配置：
- `reward_mode: "capacitor"`
- 需要配置 `evm_key_path`
- 需要配置 `contract.rpc_url`、`contract.address`、`contract.abi_file`

行为：
- 验证者调用 `releaseReward(minerColdkey, score)` 对 Capacitor 合约投票
- 当 2/3 trustees 达成一致时，合约将 stake 转移给获胜矿工
- 权重设置与合约投票相互独立

该模式已被基于 Yuma 共识的 capacitorless 系列模式替代。
