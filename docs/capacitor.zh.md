# UltraSimplifiedMultiTrusteeDistributor — 测试操作手册
---

## ✅ 前置条件

* RPC 访问：`https://test.chain.opentensor.ai`
* 已安装 Foundry  `cast` 与 `btcli`
* 部署者账户已注资，合约已部署
* 合约的 hotkey、coldkey 与接收方 coldkey 的 bytes32
* 可选：用于 `btcli` 的 ss58 版本

---

## 🔧 快速环境变量设置

```bash
export RPC_URL="https://test.chain.opentensor.ai"
export NETUID=94
export NEURON=0x0000000000000000000000000000000000000804
export STAKING=0x0000000000000000000000000000000000000805
export CONTRACT=0xYourContractAddress
export HOTKEY32=0x<contract_hotkey_bytes32>
export COLDKEY32=0x<contract_coldkey_bytes32>
export RECIPIENT_COLDKEY32=0x<recipient_coldkey_bytes32>
export CONTRACT_HOTKEY_SS58=<contract_hotkey_ss58>
export RECIPIENT_WALLET_NAME=<recipient_wallet_name>
export DEPLOYER_PK=0x<hex>
export TRUSTEE1_PK=0x<hex>
export TRUSTEE2_PK=0x<hex>
export TRUSTEE3_PK=0x<hex>
```

---

## 🧭 Foundry 流程

### 0  健康检查

```bash
cast call $CONTRACT "getDebugIdentity()(bytes32,bytes32,uint16)" --rpc-url $RPC_URL
cast call $CONTRACT "getOwnedStake()(uint256)" --rpc-url $RPC_URL
cast call $CONTRACT "getTotalHotkeyStake()(uint256)" --rpc-url $RPC_URL
```

---

### 1  给合约注资  可选

```bash
cast send $CONTRACT --value 10000000000wei --private-key $DEPLOYER_PK --rpc-url $RPC_URL
```

---

### 1  可选  注册 hotkey

```bash
cast send $CONTRACT "registerHotkey(uint16,bytes32)" $NETUID $HOTKEY32 \
  --value 0wei \
  --private-key $DEPLOYER_PK \
  --rpc-url $RPC_URL
```

---

### 1  bis  设置合约 coldkey

```bash
cast send $CONTRACT "setContractColdkey(bytes32)" $COLDKEY32 \
  --private-key $DEPLOYER_PK --rpc-url $RPC_URL
```

---

### 2  向合约 hotkey 质押

```bash
btcli st add --subtensor.network test \
  -n $NETUID \
  -in $CONTRACT_HOTKEY_SS58 \
  --amount 100000000 \
  --tolerance 0.5 \
  --allow-partial-stake
```

---

### 3  把 stake 转移到合约 coldkey

```bash
btcli st transfer --subtensor.network test
```

验证：

```bash
cast call $STAKING "getStake(bytes32,bytes32,uint256)(uint256)" $HOTKEY32 $COLDKEY32 $NETUID --rpc-url $RPC_URL
cast call $CONTRACT "getOwnedStake()(uint256)" --rpc-url $RPC_URL
```

---

### 4  两个 trustee 释放奖励

```bash
cast send $CONTRACT "releaseReward(bytes32,uint256)" $RECIPIENT_COLDKEY32 12345 \
  --private-key $TRUSTEE1_PK --rpc-url $RPC_URL
cast send $CONTRACT "releaseReward(bytes32,uint256)" $RECIPIENT_COLDKEY32 12345 \
  --private-key $TRUSTEE2_PK --rpc-url $RPC_URL
```

---

### 5  验证 stake

```bash
cast call $CONTRACT "getOwnedStake()(uint256)" --rpc-url $RPC_URL
cast call $STAKING "getStake(bytes32,bytes32,uint256)(uint256)" $HOTKEY32 $RECIPIENT_COLDKEY32 $NETUID --rpc-url $RPC_URL
```

---

### 6  接收方证明所有权

```bash
btcli st remove --subtensor.network test
```

---

## 🧩 Remix 测试流程  RAO 单位

### 0  连接网络

使用注入式 Provider  MetaMask  → Subtensor EVM  `https://test.chain.opentensor.ai`

---

### 1  给合约注资

* 低级交互
* 粘贴合约地址
* Value = `10000000000 wei`
* 点击 Transact

---

### 2  注册 hotkey

* 函数：`registerHotkey(uint16,bytes32)`
* netuid = `94`
* hotkey = `bytes32`
* Value = `0 wei`  若已注资
* Transact

---

### 3  设置合约 coldkey

* 函数：`setContractColdkey(bytes32)`
* coldkey = `bytes32`
* Transact

---

### 4  添加 stake

```bash
btcli st add --subtensor.network test -n 94 -in <contract_hotkey_ss58> --amount 100000000 --tolerance 0.5 --allow-partial-stake
```

---

### 5  转移 stake

```bash
btcli st transfer --subtensor.network test
```

检查：

```solidity
getOwnedStake()
```

---

### 6  两个 trustee 释放奖励

每个 trustee：

* 函数：`releaseReward(bytes32,uint256)`
* recipientColdkey = `bytes32`
* newScore = `12345`
* Value = `0 wei`
* 分别从 trustee1 与 trustee2 发送两次交易

---

### 7  验证 stake

`getOwnedStake()` → 0
`getTotalHotkeyStake()` → updated

---

### 8  接收方证明所有权

```bash
btcli st remove --subtensor.network test
```

---

### Remix 总结

| Step | Action           | In Remix? | Units              |
| ---- | ---------------- | --------- | ------------------ |
| 0    | Connect network  | ✅         | —                  |
| 1    | Fund contract    | ✅         | 10⁹ RAO = 1 τ      |
| 2    | Register hotkey  | ✅         | 0 wei ok if funded |
| 3    | Set coldkey      | ✅         | bytes32            |
| 4    | Stake            | ❌         | CLI only           |
| 5    | Transfer stake   | ❌         | CLI only           |
| 6    | Release reward   | ✅         | 0 wei ok           |
| 7    | Verify ownership | ✅         | RAO                |
| 8    | Remove stake     | ❌         | CLI                |

---
