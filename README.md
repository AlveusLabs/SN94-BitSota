# BitSota

**Decentralized Research Network on Bittensor**

BitSota is a decentralized research network that evolves machine learning algorithms through competitive optimization. We're a problem agnostic platform and enable the optimization of different categories of problems, with a focus on self-improving and self-generating AI.
Currently, Miners develop ML algorithms using genetic programming, while validators evaluate performance and distribute rewards through smart contract voting on the Bittensor network.

## Overview

Bitsota is a platform that allows for decentralized open research problems to continuously evolve machine learning algorithms, state of the art results and self-improving AI. The system supports multiple participation modes to accommodate different hardware capabilities and preferences. BitSota is running without a miner cap and can support a theoretically infinite number of miners.

### Core Components

**Miners:** Evolve ML algorithms using genetic programming and self-improving methods on our fixed CIFAR-10 binary evaluation pipeline, research benchmarks or hidden evaluation criteria. Can operate in direct or pool mining modes.

**Validators:** Evaluate miner submissions independently, verify algorithm performance, and vote on rewards through smart contracts.

## Inspiration

BitSota builds on concepts from Google's [AutoML-Zero](https://research.google/blog/automl-zero-evolving-code-that-learns/) research, Schmidhuber's [Goedel Machines] (https://people.idsia.ch/~juergen/goedelmachine.html) and Jeff Clune's [AI generating Algorithms](https://arxiv.org/abs/1905.10985). We build on this approach by:

- Decentralizing the evolution process across a distributed network of miners
- Using blockchain incentives to drive continuous algorithm improvement
- Implementing competitive markets where miners evolve algorithms in parallel
- Applying cryptoeconomic mechanisms to ensure honest evaluation and quality control

Where AutoML-Zero showed that algorithm evolution is possible in a research setting, Bitsota explores whether market incentives and distributed competition can sustainably produce novel ML algorithms and solve research problems at scale.

## Participation Modes

### Direct Mining

Individual miners evolve algorithms locally and submit breakthroughs to validators. Requires higher compute but offers larger individual rewards.

**Best for:** Experienced miners with dedicated hardware

**[→ Direct Mining Guide](docs/mining.md)**

### Pool Mining

Collaborative mining where participants handle smaller evolution and evaluation tasks. Pool aggregates results and submits to validators on behalf of all participants.

**Best for:** New miners or those with limited compute resources

**[→ Pool Mining Guide](docs/pool-mining.md)**

### Validation

Validators evaluate algorithm submissions, verify performance claims, and vote on rewards through multi-signature smart contracts.

**[→ Validation Guide](docs/validation.md)**

## Architecture

### Direct Mining Flow

```
Miner → Evolve Locally → Beat SOTA → Submit to Relay → Validators Verify → Contract Votes → Rewards
```

1. Miner runs genetic programming engine for up to 150 generations
2. When algorithm beats State-of-the-Art threshold, submits to relay
3. Validators download submission and independently re-evaluate
4. Validators vote through Capacitor smart contract
5. When 2/3 trustees agree, contract transfers ALPHA stake to miner

### Pool Mining Flow

```
Pool → Assigns Tasks → Miners Execute → Pool Consensus → Submit to Validators → Epoch Rewards
```

1. Pool distributes evolution and evaluation tasks to participants
2. Multiple miners evaluate each algorithm (3+ required)
3. Pool computes median consensus with 10% tolerance
4. Rewards distributed based on reputation at epoch boundaries
5. Pool submits best algorithms to validators on behalf of participants

**[→ Detailed Rewards Guide](docs/rewards.md)**

## Quick Start

### For Miners

**Desktop GUI (Recommended):**
1. Download from [bitsota.ai](https://bitsota.ai)
2. Install for your platform
3. Import your Bittensor hotkey
4. Choose mining mode (Direct or Pool)
5. Start mining

See detailed setup guides:
- [Direct Mining Setup](docs/mining.md#setup)
- [Pool Mining Setup](docs/pool-mining.md#setup)

### For Validators

```bash
git clone https://github.com/AlveusLabs/BitSota.git
cd BitSota
pip install -r requirements.txt
pip install -e .

Edit validator_config.yaml

python scripts/generate_evm.py --wallet-name your_wallet --hotkey your_hotkey # or use an existing EVM wallet
python neurons/validator_node.py
```

**[→ Full Validator Setup](docs/validation.md#setup)**

## Requirements

**Minimum:**
- Python 3.10+
- 4GB RAM
- 2GB storage
- Stable internet connection

**For Validation:**
- 16GB RAM
- 8+ CPU cores

## Documentation

- **[Mining Guide](docs/mining.md)** - Direct mining setup and strategies
- **[Pool Mining Guide](docs/pool-mining.md)** - Collaborative mining details
- **[Validation Guide](docs/validation.md)** - Running a validator node
- **[Rewards Guide](docs/rewards.md)** - Understanding incentive mechanisms

## Links

- **Website:** [bitsota.ai](https://bitsota.ai)
- **Discord:** [discord.gg/bitsota](https://discord.gg/jkJWJtPuw7)

## Security Best Practices

**Key Management:**
- Never share private keys or seed phrases
- Keep coldkeys offline
- Backup both Bittensor and EVM keys securely
- Use environment variables for private keys in production

## Contributing

This subnet is under active development. Contributions welcome through:
- Bug reports and feature requests via GitHub issues
- Code contributions via pull requests
- Community discussion on Discord

## License

See LICENSE file for details.

---

**Disclaimer:** This is experimental software. Use at your own risk. Always backup your keys and start with small amounts when testing. Cryptocurrency rewards involve financial risk.
