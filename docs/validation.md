# Validator Guide

Validators in BitSota evaluate miner submissions, verify algorithm quality, and vote on rewards through a smart contract voting system.

## What is a Validator?

Validators receive algorithm submissions from miners through the relay server, independently re-evaluate each submission on standardized datasets, and vote through the Capacitor smart contract to distribute rewards to miners who discover algorithms that beat the current performance threshold.

## How It Works

```
Relay Server → Validator Polls → Re-evaluates Submission → Verifies Score → Votes via Contract → Miner Gets Reward
```

Validators currently fix all evaluations to the CIFAR-10 binary (airplane vs automobile) dataset. Every relay submission is re-run on that benchmark regardless of the task type claimed by a miner, so miners should target this dataset when evolving algorithms.

1. RelayPoller background thread fetches new submissions every 60 seconds
2. Validator verifies miner's cryptographic signature to prevent forgery
3. Validator runs the algorithm independently using the same test datasets
4. Validator compares their score against the miner's claimed score
5. If score delta exceeds 10%, validator votes to blacklist the miner
6. If validation passes, validator selects the best submission from the batch
7. Validator calls Capacitor contract's releaseReward function with miner's coldkey and score
8. When 2 out of 3 validator trustees vote for the same submission, contract transfers ALPHA stake

## Requirements

**Hardware:**
- CPU: 8+ cores recommended
- RAM: 16GB minimum
- Storage: 10GB free space
- Network: Stable, low-latency connection

**Software:**
- Python 3.10 or higher
- Bittensor wallet registered as validator on subnet
- EVM wallet for smart contract interactions

**Stake Requirements:**
Validators must have sufficient ALPHA stake to be recognized by the Metagraph. Check current requirements with the subnet owner.

## Setup

**1. Install Dependencies**
```bash
git clone https://github.com/AlveusLabs/BitSota.git
cd BitSota
pip install -r requirements.txt
```

**2. Create Bittensor Wallet**
```bash
btcli wallet new_coldkey --wallet.name validator_wallet
btcli wallet new_hotkey --wallet.name validator_wallet --wallet.hotkey validator_hotkey
```

**3. Register as Validator**
```bash
btcli subnet register --netuid 94 --wallet.name validator_wallet --wallet.hotkey validator_hotkey
```

**4. Setup EVM Key**
Validators need an Ethereum-compatible key to interact with the Capacitor smart contract.

Option A: Generate new key
```bash
python scripts/generate_evm_key.py --output ~/.bittensor/evm_keys/validator.json
```

Option B: Use existing key by setting path in config

**5. Configure Validator**
Copy and edit the config:
```bash
cp validator_config.yaml.example validator_config.yaml
```

Edit `validator_config.yaml`:
```yaml
netuid: 94
wallet_name: "validator_wallet"
wallet_hotkey: "validator_hotkey"
evm_key_path: "~/.bittensor/evm_keys/validator.json"
network: "test"

contract:
  rpc_url: "https://test.chain.opentensor.ai"
  address: "0xYourCapacitorContractAddress"

relay:
  url: "https://relay.bitsota.com"
  poll_interval_seconds: 60

# Optional contract submission schedule
submission_schedule:
  mode: "immediate"        # "immediate", "interval", or "utc_times"
  interval_seconds: 0      # used when mode == "interval"
  utc_times: []            # used when mode == "utc_times"

submission_threshold:
  mode: "sota_only"        # "local_best" keeps local SOTA as an additional floor

blacklist:
  cutoff_percentage: 0.1

weights:
  check_interval: 300
```

When `submission_schedule.mode` is set to `interval` or `utc_times`, the validator caches the best validated result if the schedule blocks an immediate vote. The cached entry is retried automatically as soon as the next window opens, and any newer submission with a higher validator score replaces the pending one.

Set `submission_threshold.mode` to `local_best` if you want the validator to require every vote to beat both the current on-chain/relay SOTA and the best score it has validated locally during the current session. This prevents regressing to weaker submissions even when the global SOTA temporarily drops.

## Running the Validator

```bash
python neurons/validator_node.py
```

The validator will start three background services:
1. RelayPoller: Fetches miner submissions every 60 seconds
2. WeightManager: Updates on-chain weights every 5 minutes
3. ContractManager: Handles smart contract interactions

## Understanding Validator Output

```
Processing 5 results from relay...
Miner abc12345: Miner Score = 0.9300, Validator Score = 0.9250, SOTA = 0.9200
Miner def67890: Miner Score = 0.9400, Validator Score = 0.8100, SOTA = 0.9200
Blacklisting miner def67890 for score delta. Validator score: 0.8100, Miner score: 0.9400
Best result is from abc12345 with validated score 0.9250
Submitted relay solution from abc12345 with validated score 0.9250... tx: 0xabcdef123...
```

**What this means:**
- First miner's submission is legitimate (0.93 claimed, 0.925 validated, within 10% tolerance)
- Second miner attempted to cheat (0.94 claimed, 0.81 actual, 14% difference)
- Validator votes to blacklist the cheater
- Validator submits vote for legitimate miner through smart contract

## Validation Process

**Step 1: Signature Verification**
```python
ValidatorAuth.verify_miner_signature(miner_hotkey, timestamp_message, signature)
```
Ensures the submission actually came from the claimed miner's wallet.

**Step 2: Algorithm Re-evaluation**
```python
is_valid, validator_score = verify_solution_quality(algorithm_result, sota_score)
```
Validators now run every candidate on a deterministic slice of the CIFAR-10 task space.
Each "task" corresponds to a class pair, projection matrix, and dataset split. The
validator samples `VALIDATOR_TASK_COUNT` (default 5) tasks once using
`VALIDATOR_TASK_SEED` and reuses that set for every submission, providing a repeatable
score while still covering multiple projections.

**Step 3: SOTA Check**
If validator's score is below the current State-of-the-Art threshold, submission is dropped regardless of miner's claimed score.

**Step 4: Score Delta Check**
```python
if abs(validator_score - miner_score) > 0.1:  # 10% tolerance
    blacklist_miner(miner_hotkey)
```
Large discrepancies indicate dishonesty or bugs.

**Step 5: Contract Voting**
```python
contract_manager.submit_contract_entry(recipient_ss58_address=miner_hotkey, new_score=validator_score)
```
Validator submits vote to Capacitor contract using the validator's own evaluated score, not the miner's claim.

## Smart Contract Voting

The Capacitor contract uses a multi-trustee voting system:

- 3 validators are designated as trustees
- 2 votes required to trigger reward distribution
- Votes must agree on both recipient and score
- If votes mismatch, voting resets and starts over
- When consensus reached, contract transfers ALL its ALPHA stake to the winning miner

**Checking Contract Status:**
```bash
python scripts/check_contract_status.py --config validator_config.yaml
```

Shows:
- Current pending vote (if any)
- Vote count
- Which trustees have voted
- Contract's available stake balance

[//]: # (## Funding the Contract)

[//]: # ()
[//]: # ([//]: # &#40;The Capacitor contract needs ALPHA stake to distribute rewards. Validators fund it by running the burn script:&#41;)
[//]: # ()
[//]: # (```bash)

[//]: # (python scripts/burn_script.py --wallet validator_wallet --amount 100000)

[//]: # (```)

[//]: # ()
[//]: # (This script:)

[//]: # (1. Takes specified amount of your ALPHA stake)

[//]: # (2. Burns 2/3 &#40;destroyed forever as a cost&#41;)

[//]: # (3. Unstakes 1/3)

[//]: # (4. Transfers that 1/3 to the Capacitor contract)

[//]: # ()
[//]: # (The burn mechanism ensures validators only fund the contract when they believe the subnet produces valuable results.)

[//]: # ()
[//]: # (**Warning:** Only run this script when you intend to fund rewards. The burned stake is permanently destroyed.)

## Weight Management

Validators set on-chain weights to indicate miner performance. This affects future network emissions.

The WeightManager runs automatically in the background and:
- Checks every 5 minutes if weights need updating
- Discovers active bots on the subnet
- Sets weights based on recent miner performance
- Uses exponential backoff on failures

Weights are separate from the immediate Capacitor rewards. Weights affect long-term emissions, while Capacitor provides instant bonuses for exceptional submissions.

## Blacklisting

When validators detect cheating (score delta > 10%), they vote to blacklist miners through the relay server:

```python
relay_client.blacklist_miner(miner_hotkey)
```

After enough validators vote (consensus-based), the relay server rejects all future submissions from that miner. This is a protection mechanism against:
- Miners falsifying evaluation scores
- Buggy miner implementations
- Deliberate gaming attempts

## Monitoring

**Metrics Logging:**
All validators log detailed metrics to `validator_metrics.log`:
- Evaluation batch timings
- Individual miner results (passed/failed/blacklisted)
- Contract submission successes
- Periodic summaries every 10 minutes

**Health Checks:**
The validator automatically restarts background threads if they crash:
```
Weight manager thread died. Restarting...
Relay poller thread died. Restarting...
```

**Manual Checks:**
```bash
# View recent logs
tail -f validator_metrics.log

# Check if validator is running
ps aux | grep validator_node

# Monitor contract votes
python scripts/monitor_contract.py --config validator_config.yaml
```

## Troubleshooting

**"Failed to get results from relay":**
- Check relay URL in config is correct
- Verify internet connection
- Relay server might be down (check Discord)

**"Could not get SOTA score, cannot process results":**
- Contract RPC endpoint is unreachable
- Check `contract.rpc_url` in config
- Try alternative RPC endpoint

**"Already voted for miner with this score":**
Normal behavior. Validator tracks recent votes to avoid duplicate contract transactions.

**"No results passed validation and SOTA checks":**
All submissions in this batch either:
- Had invalid signatures
- Scored below SOTA
- Had score deltas indicating cheating
This is normal if miner quality is low.

**EVM key issues:**
Make sure your EVM key file exists at the path specified in `evm_key_path`. The key must have a small amount of ETH for gas fees on Bittensor EVM.

## Advanced Configuration

**Adjust polling interval:**
```yaml
relay:
  poll_interval_seconds: 30  # Poll more frequently
```
Lower values = more responsive but higher bandwidth usage.

**Adjust blacklist threshold:**
```yaml
blacklist:
  cutoff_percentage: 0.05  # Stricter (5% tolerance)
```
Lower values = stricter validation but might blacklist honest miners with slight evaluation differences.

**Change weight update frequency:**
```yaml
weights:
  check_interval: 600  # Check every 10 minutes instead of 5
```

## Economics

Validators earn TAO/ALPHA emissions through the standard Bittensor mechanism based on:
- Validator stake amount
- Subnet performance
- Network-wide emission schedule

The Capacitor contract voting is additional work that helps the subnet function but doesn't directly earn validators more. Validators participate because:
1. Healthy subnet = higher subnet emissions = more validator earnings
2. Validator operators may also run miners
3. Subnet success attracts more stake and participants

## Security Best Practices

**Protect your keys:**
- Never share wallet seeds or private keys
- Use separate wallets for different functions (cold storage for main funds)
- Encrypt EVM key files

**Monitor for attacks:**
- Watch for unusual submission patterns
- Check for coordinated score manipulation
- Report suspicious activity to subnet operators

**Keep software updated:**
```bash
git pull origin main
pip install -r requirements.txt --upgrade
```

## Next Steps

- Join [Hivetensor Discord](https://discord.gg/hivetensor) for validator coordination
- Monitor your validator's performance and vote success rate
- Consider funding the Capacitor contract when stake is low
- Review the [Rewards Guide](rewards.md) to understand the full incentive mechanism

For understanding the reward distribution system, see [Rewards Guide](rewards.md).
