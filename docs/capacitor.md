# UltraSimplifiedMultiTrusteeDistributor — Test Playbook
---

## ✅ Prerequisites

* RPC access: `https://test.chain.opentensor.ai`
* Foundry (`cast`) and `btcli` installed
* Three trustees funded
* Deployer funded and contract deployed
* Contract’s hotkey, coldkey, and recipient coldkey in bytes32
* Optionally: ss58 versions for `btcli`

---

## 🔧 Quick Env Setup

```bash
export RPC_URL="https://test.chain.opentensor.ai"
export NETUID=402
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

## 🧭 Foundry Flow

### 0) Sanity Checks

```bash
cast call $CONTRACT "getDebugIdentity()(bytes32,bytes32,uint16)" --rpc-url $RPC_URL
cast call $CONTRACT "getOwnedStake()(uint256)" --rpc-url $RPC_URL
cast call $CONTRACT "getTotalHotkeyStake()(uint256)" --rpc-url $RPC_URL
```

---

### 1) Fund the Contract (optional)

```bash
cast send $CONTRACT --value 10000000000wei --private-key $DEPLOYER_PK --rpc-url $RPC_URL
```

---

### 1-optional) Register Hotkey

```bash
cast send $CONTRACT "registerHotkey(uint16,bytes32)" $NETUID $HOTKEY32 \
  --value 0wei \
  --private-key $DEPLOYER_PK \
  --rpc-url $RPC_URL
```

---

### 1-bis) Set Contract Coldkey

```bash
cast send $CONTRACT "setContractColdkey(bytes32)" $COLDKEY32 \
  --private-key $DEPLOYER_PK --rpc-url $RPC_URL
```

---

### 2) Stake to Contract Hotkey

```bash
btcli st add --subtensor.network test \
  -n $NETUID \
  -in $CONTRACT_HOTKEY_SS58 \
  --amount 100000000 \
  --tolerance 0.5 \
  --allow-partial-stake
```

---

### 3) Transfer Stake to Contract Coldkey

```bash
btcli st transfer --subtensor.network test
```

Verify:

```bash
cast call $STAKING "getStake(bytes32,bytes32,uint256)(uint256)" $HOTKEY32 $COLDKEY32 $NETUID --rpc-url $RPC_URL
cast call $CONTRACT "getOwnedStake()(uint256)" --rpc-url $RPC_URL
```

---

### 4) Two Trustees Release Reward

```bash
cast send $CONTRACT "releaseReward(bytes32,uint256)" $RECIPIENT_COLDKEY32 12345 \
  --private-key $TRUSTEE1_PK --rpc-url $RPC_URL
cast send $CONTRACT "releaseReward(bytes32,uint256)" $RECIPIENT_COLDKEY32 12345 \
  --private-key $TRUSTEE2_PK --rpc-url $RPC_URL
```

---

### 5) Verify Stake

```bash
cast call $CONTRACT "getOwnedStake()(uint256)" --rpc-url $RPC_URL
cast call $STAKING "getStake(bytes32,bytes32,uint256)(uint256)" $HOTKEY32 $RECIPIENT_COLDKEY32 $NETUID --rpc-url $RPC_URL
```

---

### 6) Recipient Proves Ownership

```bash
btcli st remove --subtensor.network test
```

---

## 🧩 Remix Testing Flow (RAO Units)

### 0) Connect Network

Use Injected Provider (MetaMask) → Subtensor EVM (`https://test.chain.opentensor.ai`)

---

### 1) Fund Contract

* “Low-level interactions”
* Paste contract address
* Value = `10000000000 wei`
* Click Transact

---

### 2) Register Hotkey

* Function: `registerHotkey(uint16,bytes32)`
* netuid = `402`
* hotkey = `bytes32`
* Value = `0 wei` (if funded)
* Transact

---

### 3) Set Contract Coldkey

* Function: `setContractColdkey(bytes32)`
* coldkey = `bytes32`
* Transact

---

### 4) Add Stake

```bash
btcli st add --subtensor.network test -n 402 -in <contract_hotkey_ss58> --amount 100000000 --tolerance 0.5 --allow-partial-stake
```

---

### 5) Transfer Stake

```bash
btcli st transfer --subtensor.network test
```

Check:

```solidity
getOwnedStake()
```

---

### 6) Two Trustees Release Reward

Each trustee:

* Function: `releaseReward(bytes32,uint256)`
* recipientColdkey = `bytes32`
* newScore = `12345`
* Value = `0 wei`
* Transact twice from trustee1 and trustee2

---

### 7) Verify Stake

`getOwnedStake()` → 0
`getTotalHotkeyStake()` → updated

---

### 8) Recipient Proves Ownership

```bash
btcli st remove --subtensor.network test
```

---

### Remix Summary

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
