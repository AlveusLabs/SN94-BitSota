# Pool Mining Guide (Work In Progress)

Pool mining allows participants with limited compute resources to contribute to algorithm evolution and evaluation while sharing rewards. Instead of competing individually, pool miners work together.

## What is Pool Mining?

Pool mining is a collaborative approach where the pool coordinator distributes tasks to multiple miners. Some miners evolve algorithms while others evaluate them. The pool aggregates results, achieves consensus, and submits the best algorithms to validators on behalf of all participants.

## How It Works

```
Pool Service → Assigns Tasks → Miner Evolves/Evaluates → Pool Consensus → Submit to Validators → Rewards Distributed
```

1. Miner registers with the pool at pool.bitsota.ai
2. Miner requests a task (either evolution or evaluation)
3. Pool assigns either:
   - Evolution task: Take these 2 algorithms and evolve them
   - Evaluation task: Run these algorithms and report scores
4. Miner completes task and submits results back to pool
5. Pool collects 3+ independent evaluations for each algorithm
6. Pool computes consensus using median scores with 10% tolerance
7. Pool rewards accurate evaluators and successful evolvers
8. Pool submits best algorithms to validators
9. At epoch end, pool distributes accumulated RAO rewards proportionally

## Pool vs Direct Mining

**Pool Mining:**
- Lower compute requirements (shorter evolution runs)
- Consistent small rewards for evaluation work
- No need to beat SOTA yourself
- Pool handles validator communication
- Share rewards with other pool participants

**Direct Mining:**
- Higher compute requirements (full evolution cycles)
- Larger but less frequent rewards
- Must beat SOTA threshold
- Direct communication with validators
- Keep 100% of rewards

Choose pool mining if you have:
- Limited hardware (laptop, single CPU)
- Inconsistent availability (can't run 24/7)
- Preference for steady income over lottery-style rewards

## Requirements

**Hardware:**
- CPU: 2+ cores
- RAM: 4GB minimum
- Storage: 1GB free space
- Network: Stable internet connection

**Software:**
- Python 3.10 or higher
- Bittensor wallet

## Setup

**Desktop GUI:**
1. Download desktop app from [bitsota.ai](https://bitsota.ai)
2. Install and launch
3. Navigate to Pool Mining screen
4. Select your wallet and hotkey
6. Click Start Mining

## Understanding Pool Tasks

**Evolution Tasks:**
You receive 2 seed algorithms and evolve them for 50 generations. You submit the best evolved algorithm back to the pool.

Rewards:
- Base reward: 2.0 reputation points
- Multiplied by consensus score if your algorithm scores >= 0.7
- Example: If consensus determines your algorithm scored 0.85, you get 2.0 × 0.85 = 1.7 reputation

**Evaluation Tasks:**
You receive a batch of algorithms evolved by other miners. You run each algorithm on the test dataset and report scores.

Rewards:
- Base reward: 1.0 reputation point per accurate evaluation
- "Accurate" means your score is within 10% of the median consensus
- Example: You evaluate 5 algorithms accurately, you get 5.0 reputation

## Consensus Mechanism

The pool uses median consensus to prevent cheating:

1. Algorithm A needs evaluation
2. Pool assigns it to miners X, Y, Z
3. Miner X reports score: 0.92
4. Miner Y reports score: 0.90
5. Miner Z reports score: 0.91
6. Median = 0.91
7. Miners within 10% tolerance (X: 0.92 and Z: 0.91) get rewarded
8. If Y had reported 0.50 (outlier), Y would not be rewarded

This prevents both:
- Inflating scores (claiming 0.99 when algorithm scores 0.80)
- Deflating scores (claiming 0.60 to sabotage others)

## Reputation and Rewards

The pool tracks reputation points throughout each epoch (typically 1 hour):

**Reputation accumulation:**
```
Epoch 1:
- 10 accurate evaluations = 10.0 reputation
- 2 successful evolutions = 2.0 × 0.8 + 2.0 × 0.75 = 3.1 reputation
- Total: 13.1 reputation points
```

**Conversion to RAO:**
At epoch end, total epoch budget (e.g., 1,000,000,000 RAO) is distributed proportionally:

```
Your RAO = (Your Reputation / Total Pool Reputation) × Epoch Budget
```

If total pool reputation is 1000 and you earned 13.1:
```
Your RAO = (13.1 / 1000) × 1,000,000,000 = 13,100,000 RAO
```

**Withdrawal:**
Once you accumulate at least 5,000,000 RAO, you can withdraw to your coldkey. Withdrawals are processed manually by the pool operator.

## Pool Economics

**Pool fees:**
Check with pool operator. Typical structure is 5-10% of rewards go to pool for infrastructure and validator costs.

**Minimum payout:**
5,000,000 RAO minimum to reduce transaction overhead.

**Epoch duration:**
Most pools use 1-hour epochs. Shorter epochs = more frequent small payouts. Longer epochs = less frequent larger payouts.

**Per-miner cap:**
Pools may cap individual rewards at 5% of epoch budget to ensure fair distribution.

## Monitoring Your Performance

**Balance check:**
Pool GUI shows current balance and pending rewards.

**Evaluation accuracy rate:**
Track what percentage of your evaluations fall within consensus. Target 95%+ accuracy.

**Evolution success rate:**
Track how many of your evolved algorithms pass the 0.7 threshold for rewards.

**Reputation per hour:**
Monitor your earning rate. Optimize by:
- Running evaluation tasks when available (faster reputation)
- Improving evolution parameters for higher consensus scores
- Maintaining high evaluation accuracy

## Troubleshooting

**"No tasks available":**
Pool may be temporarily out of tasks. Wait 30-60 seconds and retry. If persistent, check Discord for pool status.

**"Task timeout":**
You took longer than 3 hours to complete a task. Task was reassigned to another miner. Ensure your hardware can complete tasks faster or reduce `max_generations`.

**"Evaluation rejected - not in consensus":**
Your scores significantly differed from other evaluators. Possible causes:
- Bug in your evaluation code
- Different dataset version (ensure you're updated)
- Hardware issue causing incorrect calculations

Update your software and verify installation.

**"Cannot withdraw - below minimum":**
Accumulate at least 5,000,000 RAO before withdrawing.

## Pool Service Architecture

The pool mining system runs independently at https://pool.bitsota.ai. For technical details and source code, see:

The pool service handles:
- Miner registration and authentication
- Task assignment and tracking
- Consensus computation
- Reputation accounting
- Epoch management
- Reward distribution
- Relay submission on behalf of miners

**API Endpoints:**
- POST /api/v1/miners/{address}/register
- POST /api/v1/tasks/{address}/request
- POST /api/v1/tasks/{address}/{batch_id}/submit_evolution
- POST /api/v1/tasks/{address}/{batch_id}/submit_evaluation
- GET /api/v1/rewards/{address}/balance
- POST /api/v1/rewards/{address}/withdraw

All requests require Bittensor signature authentication.

## Tips for Pool Miners

**Optimize task selection:**
If given a choice, evaluation tasks are faster reputation but evolution tasks can yield higher rewards per task if your algorithms score well.

**Run continuously:**
Pool mining benefits from consistency. Running 24/7 accumulates steady reputation.

**Monitor pool health:**
Check pool's total participants. Overpopulated pools = smaller individual shares. Consider switching pools or direct mining if ROI drops.

**Maintain accuracy:**
One bad evaluation that falls outside consensus doesn't hurt much, but consistent inaccuracy will reduce your rewards significantly.

**Update regularly:**
Pool operators update task types and parameters. Keep your miner software updated to stay compatible.

## Next Steps

- Join [Bitsota Discord](https://discord.gg/jkJWJtPuw7) for pool mining support
- Monitor your reputation and optimize your mining strategy
- Consider switching to direct mining if you upgrade hardware

For understanding how rewards work across the entire system, see [Rewards Guide](rewards.md).
