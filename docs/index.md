<section class="bitsota-hero">
  <p class="bitsota-kicker">BITSOTA DOCS</p>
  <h1>Autoresearch on SN94</h1>
  <p class="bitsota-lede">Post, train, validate, and pay only for confirmed progress.</p>
  <div class="bitsota-strip">
    <span>Problem owners define tasks</span>
    <span>Miners submit improvements</span>
    <span>Validators replay results</span>
    <span>Pool publishes claims</span>
  </div>
</section>

BitSota is a decentralized research subnet on Bittensor. This documentation now
focuses on the current autoresearch backend path.

The old AutoML-Zero relay/SOTA docs are preserved in
[AutoML-Zero Archive](archive/automl-zero/index.md).

## Start here

- [Getting Started](getting-started.md)
- [Architecture Overview](architecture.md)
- [Current Competitions](current-competitions.md)
- [Mining Without an Agent](mining.md)
- [Codex-Only Mining](codex-only-mining.md)
- [Problem Posting Requirements](problem-posting.md)
- [Future Roadmap](roadmap.md)

## System overview

```mermaid
flowchart TB
  Owner[Problem owner] --> Backend[Autoresearch backend]
  Miner[Miner or Codex] --> Backend
  Validator[Validator runner] --> Backend
  Backend --> Pool[Pool claims]
  Pool --> Contract[Merkle contract]
  Backend --> Weights[Validator weight policy]
  Weights --> Chain[Bittensor SN94]
```
