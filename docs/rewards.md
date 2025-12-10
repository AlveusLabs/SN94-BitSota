# Rewards and Incentive Mechanisms

BitSota uses multiple reward systems that work together. Understanding how they interact helps you maximize earnings and participate effectively.

## Overview of Reward Systems

**1. Bittensor Network Emissions**
Standard TAO/ALPHA emissions distributed by the Bittensor protocol based on validator weights.

**2. Capacitor Smart Contract**
Bonus rewards for exceptional algorithm discoveries.

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
Validators run WeightManager which evaluates miners and sets on-chain weights every 5 minutes. This affects your long-term emission income.

## Capacitor Smart Contract Rewards

The Capacitor contract provides immediate bonuses for discovering breakthrough algorithms.

### How It Works
**Funding:**
The contract is funded through emissions meant for miner rewards. These are sent to the capacitor contract. 

**Voting:**
When a miner submits an algorithm that beats SOTA:
1. Validators independently verify the algorithm
2. Each validator calls `releaseReward(minerColdkey, score)`
3. Contract tracks votes on-chain
4. When 2 out of 3 trustees agree on the same (miner, score), contract executes transfer
5. Contract transfers ALL its stake to the winning miner's coldkey

**Key Points:**
- Rewards are immediate (within minutes of consensus)
- One winner per voting round
- Winner gets all available contract stake
- No claiming required, stake transfers automatically

### SOTA Threshold

SOTA (State-of-the-Art) is the minimum score required for rewards. It increases over time as better algorithms are discovered.

**Current SOTA:** Check with `contract_manager.get_current_sota_threshold()`

**Progressive Improvement:**
When someone beats SOTA with score 0.92, the new SOTA becomes 0.92. Next submission must beat 0.92. This ensures continuous improvement.

**Score Verification:**
Validators don't trust miner-reported scores. They re-run algorithms and use their own evaluated scores for voting. This prevents cheating.

**Blacklisting:**
If your claimed score differs from validator's score by more than 10%, validators vote to blacklist you. After multiple blacklist votes, the relay rejects your submissions.

### Economic Implications

**For Miners:**
Discovering SOTA-breaking algorithms can yield large instant rewards (entire contract balance). But this is competitive - only the best submission wins.

**For Validators:**
- Subnet is producing valuable algorithms
- Higher subnet reputation attracts more participants
- Long-term emissions outweigh short-term costs

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
- You run direct mining 24/7
- You discover 1 SOTA-breaking algorithm per week
- Your weights earn you 10 ALPHA/day from emissions

**Weekly earnings:**
- Network emissions: 10 × 7 = 70 ALPHA
- Capacitor bonus: 50 ALPHA (from winning one round)
- Total: 120 ALPHA/week

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

**Capacitor Rewards:** Immediate upon reaching 2/3 consensus

**Pool Rewards:** Hourly epoch conversions, withdrawals processed within 24 hours

**Weight Updates:** Every 5 minutes, affects future emissions

## Tax and Accounting Considerations

Note: Not financial advice. Consult a tax professional.

**Emissions:** Likely taxable as income when received

**Staking Rewards:** Tax treatment varies by jurisdiction

**Track:**
- All incoming rewards (date, amount, source)
- Wallet addresses and transaction hashes

Many jurisdictions require reporting crypto income even if not converted to fiat.

## Future: L2Pool Scaling

When L2Pool launches, reward distribution will change:

**Current (Capacitor):**
- One winner per round
- Validators vote on-chain
- Winner gets all contract stake

**Future (L2Pool):**
- Thousands of winners per epoch
- Validators sign off-chain
- Winners claim their share with Merkle proofs
- More gas-efficient at scale

This will enable:
- More miners receiving rewards per epoch
- Lower validator operational costs
- Fairer distribution across performance tiers

Monitor Discord for L2Pool launch announcements.

## Common Questions
**Q: Can I earn from both direct mining and pool mining?**
A: Yes, run separate miners with different wallets. Don't use same wallet for both or pool may reject you.

**Q: What happens if Capacitor contract runs out of stake?**
A: No bonuses until a  it's funded again. Network emissions continue normally.

**Q: How is SOTA threshold initially set?**
A: First submission sets baseline. Threshold increases from there.

**Q: Do pool miners get network emissions?**
A: No. Pool operator is registered on subnet and receives emissions, then distributes to pool participants through reputation system.

**Q: Can I lose rewards?**
A: Blacklisting blocks future rewards but doesn't take away earned rewards. Validators can't lose emissions unless they unstake.

## Next Steps

- Decide which role suits your resources: direct miner, pool miner, or validator
- Review role-specific guides for detailed setup
- Monitor your earnings and optimize strategy
- Join Discord for reward discussions and subnet economics

**Related Guides:**
- [Direct Mining](mining.md)
- [Pool Mining](pool-mining.md)
- [Validation](validation.md)
