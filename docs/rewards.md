# Rewards and Incentive Mechanisms

BitSota uses multiple reward systems that work together. Understanding how they interact helps you maximize earnings and participate effectively.

## Overview of Reward Systems

**1. Bittensor Network Emissions**
ALPHA emissions distributed by the Bittensor protocol based on validator-set weights.

**2. SOTA Weight Distribution**
Validators set on-chain weights to direct emissions: 90% to burn address, 10% to current SOTA winner.

**3. Pool Reputation System**
For pool miners, reputation converts to RAO rewards at epoch boundaries.

Each system serves a different purpose and rewards different behaviors.

## Bittensor Network Emissions

This is the foundation of subnet economics. The Bittensor blockchain automatically distributes TAO/ALPHA tokens based on:

**For Miners:**
- Validators set weights reflecting your performance
- Higher weights = larger share of subnet emissions
- Emissions happen continuously as blocks are produced
- No claiming required, tokens appear in your account

**For Validators:**
- Validators earn based on their stake amount
- Subnet performance affects validator earnings
- Better subnet = more network attention = higher emissions

**Emission Schedule:**
Bittensor has a fixed emission schedule. Each subnet receives a portion based on subnet performance. Your share depends on your weights relative to other participants.

**Weight Setting:**
Validators run weight managers that set on-chain weights when SOTA events are finalized. This determines how emissions flow to miners.

## SOTA Weight Distribution

Validators direct emissions to SOTA winners through on-chain weight setting using Yuma consensus. Two modes are available:

### Relay Consensus Mode (winner_source: "relay")

When a miner submits an algorithm that beats SOTA:
1. Validators independently verify the algorithm
2. Each validator votes on relay to accept new SOTA candidate
3. Relay tracks votes and reaches consensus
4. When enough validators agree, relay finalizes SOTA event
5. SOTA event includes: miner_hotkey, score, start_block, end_block

Weight Setting:
1. Validators fetch finalized SOTA events from relay
2. Weight manager identifies current SOTA winner (most recent event)
3. Validator sets on-chain weights: 90% burn_hotkey, 10% winner_hotkey
4. Network emissions flow according to these weights
5. Weights update only when new SOTA event is finalized

This mode ensures all validators converge on the same winner before setting weights.

### Local Mode (winner_source: "local")

When a miner submits an algorithm that beats SOTA:
1. Validator independently verifies the algorithm
2. If score beats current local best, validator updates local winner
3. Validator sets on-chain weights immediately: 90% burn, 10% local winner
4. Optionally still votes on relay for coordination
5. Weights update whenever validator finds better local candidate

Weight Setting:
1. Weight manager tracks best score validated by this validator
2. When new best found, weights update immediately
3. No waiting for relay consensus
4. Faster response but validators may choose different winners
5. `min_winner_improvement` prevents frequent weight changes

This mode is faster but validators may temporarily diverge on winner choice.

**Key Points (Both Modes):**
- Rewards flow continuously via network emissions
- 90% of emissions go to burn (deflationary mechanism)
- 10% of emissions go to current SOTA winner
- Winner changes when new SOTA is accepted (relay) or found (local)
- No claiming required, emissions automatic

### SOTA Threshold

SOTA (State-of-the-Art) is the minimum score required for acceptance. It increases over time as better algorithms are discovered.

**Current SOTA:** Check with `curl https://relay.bitsota.com/sota_threshold`

**Progressive Improvement:**
When someone beats SOTA with score 0.92, the new SOTA becomes 0.92. Next submission must beat 0.92. This ensures continuous improvement.

**Score Verification:**
Validators don't trust miner-reported scores. They re-run algorithms and use their own evaluated scores for voting. This prevents cheating.

**Blacklisting:**
If your claimed score differs from validator's score by more than 10%, validators vote to blacklist you. After multiple blacklist votes, the relay rejects your submissions.

### Economic Implications

**For Miners:**
Discovering SOTA-breaking algorithms makes you the current winner, earning you 10% of network emissions until someone beats your score. The longer your algorithm remains SOTA, the more you earn.

**For Validators:**
Validators earn based on stake. They distribute emissions to miners through weight setting:
- 90% to burn reduces circulating supply (deflationary)
- 10% to winner rewards innovation
- Healthy subnet attracts more stake and participants

## Pool Mining Rewards

Pool miners earn through a reputation system that converts to RAO at epoch boundaries.

### Reputation Accumulation

**Evaluation Tasks:**
- Base: 1.0 reputation per accurate evaluation
- "Accurate" means within 10% of median consensus
- Example: Evaluate 10 algorithms accurately = 10.0 reputation

**Evolution Tasks:**
- Base: 2.0 reputation points
- Multiplied by consensus score if >= 0.7
- Example: Your algorithm scores 0.85 in consensus = 2.0 × 0.85 = 1.7 reputation
- Example: Your algorithm scores 0.65 = 0 reputation (below threshold)

### Epoch Conversion

At epoch end (typically every hour):

**Total epoch budget:** e.g., 1,000,000,000 RAO

**Distribution formula:**
```
Your RAO = (Your Reputation / Total Pool Reputation) × Epoch Budget × (1 - Pool Fee)
```

**Example:**
- Epoch budget: 1,000,000,000 RAO
- Total pool reputation: 5,000 points
- Your reputation: 50 points
- Pool fee: 5%

```
Your RAO = (50 / 5000) × 1,000,000,000 × 0.95
         = 0.01 × 1,000,000,000 × 0.95
         = 9,500,000 RAO
```
### Per-Miner Cap

Pools often cap individual rewards at 5% of epoch budget to ensure fair distribution. If your reputation would earn you more than 5%, excess is redistributed to other miners.

## TAO vs ALPHA

Bittensor recently launched Dynamic TAO which introduced subnet-specific ALPHA tokens.

**TAO:**
- Main Bittensor token
- Used for registration fees
- Staking for validators
- Network governance

**ALPHA:**
- Subnet-specific token (each subnet has its own)
- Subnet 94's ALPHA represents value created by this subnet
- Used for staking within the subnet
- Can be converted to/from TAO through liquidity pools

**Transition Period:**
Bittensor is transitioning from TAO-only to ALPHA-weighted rewards over ~100 days. Eventually subnet rewards will be primarily in ALPHA.

**What This Means:**
Your rewards (both emissions and Capacitor) are in ALPHA stake. ALPHA stake can be:
- Held for validator registration
- Converted to TAO through exchanges
- Used within subnet ecosystem

## Reward Calculation Examples

### Example 1: Direct Miner

**Setup:**
- You discover SOTA-breaking algorithm
- Validators vote and relay finalizes your SOTA event
- You become current winner with 10% weight
- Subnet receives 1000 ALPHA/day in emissions
- You remain SOTA winner for 7 days

### Example 2: Pool Miner

**Setup:**
- You run pool mining 24/7
- You complete ~20 evaluation tasks per hour
- You complete ~2 evolution tasks per hour
- 90% evaluation accuracy
- Average evolution score: 0.75

**Hourly reputation:**
- Evaluations: 20 × 0.9 (accuracy) × 1.0 = 18 reputation
- Evolutions: 2 × 2.0 × 0.75 = 3 reputation
- Total: 21 reputation/hour

**Hourly earnings (assuming 5000 total pool reputation):**
```
RAO = (21 / 5000) × 1,000,000,000 × 0.95
    = 3,990,000 RAO
```

**Daily earnings:**
- 24 hours × 3,990,000 = 95,760,000 RAO = ~0.096 ALPHA

### Example 3: Validator

**Setup:**
- You have 1000 ALPHA staked as validator
- Subnet total validator stake: 10,000 ALPHA
- Subnet emissions: 1000 ALPHA/day

**Daily earnings:**
```
Your share = (1000 / 10000) × 1000 = 100 ALPHA/day
```
## Maximizing Rewards

### For Direct Miners

**Optimize Evolution:**
- Use archive engine for better exploration
- Run longer generation counts (150+)
- Focus on the CIFAR-10 binary benchmark used by validators
- Monitor SOTA threshold before starting runs

**Hardware:**
Better CPUs = more generations/minute = higher probability of finding SOTA-breaking algorithms.

**Timing:**
Submit when SOTA threshold is low (early subnet stages or after SOTA hasn't updated in a while).

### For Pool Miners

**Maintain High Accuracy:**
One incorrect evaluation doesn't hurt much, but consistent inaccuracy reduces earnings by 10-20%.

**Balance Task Types:**
- Evaluations: Fast reputation
- Evolutions: Higher reputation per task if your algorithms score well

**Run Continuously:**
Pool mining rewards consistency. 24/7 operation maximizes reputation accumulation.

**Choose Right Pool:**
Monitor pool population. Overpopulated pools mean smaller shares. Consider switching pools or upgrading to direct mining.

### For Validators

**Set Accurate Weights:**
WeightManager does this automatically, but monitoring miner quality helps subnet reputation which increases your emissions.


## Understanding Reward Timing

**Network Emissions:** Continuous, per-block distribution

**SOTA Winner Rewards:** Begin flowing after validators set weights for finalized SOTA event

**Pool Rewards:** Hourly epoch conversions, withdrawals processed within 24 hours

**Weight Updates:** Occur when new SOTA events are finalized, affects immediate emissions


## Common Questions

**Q: Can I earn from both direct mining and pool mining?**
A: Yes, run separate miners with different wallets. Don't use same wallet for both or pool may reject you.


**Q: What happens to the 90% burned?**
A: Tokens sent to burn address are permanently removed from circulation, making the remaining supply more scarce.

**Q: How is SOTA threshold initially set?**
A: First submission sets baseline. Threshold increases from there.

**Q: Do pool miners get network emissions?**
A: No. Pool operator is registered on subnet and receives emissions, then distributes to pool participants through reputation system.

**Q: Can I lose rewards?**
A: Blacklisting blocks future rewards but doesn't take away earned rewards. If you're SOTA winner when blacklisted, you continue earning until someone beats your score.

**Q: What's the difference between relay mode and local mode?**
A: Relay mode waits for multiple validators to agree on winner before setting weights (slower, more coordinated). Local mode sets weights immediately based on validator's own evaluation (faster, less coordinated). Both modes distribute 90% to burn and 10% to winner.

## Next Steps

- Decide which role suits your resources: direct miner, pool miner, or validator
- Review role-specific guides for detailed setup
- Monitor your earnings and optimize strategy
- Join Discord for reward discussions and subnet economics

**Related Guides:**
- [Direct Mining](mining.md)
- [Pool Mining](pool-mining.md)
- [Validation](validation.md)
