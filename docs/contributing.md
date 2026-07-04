# Contributing

## Repo layout

- `miner/` and `neurons/` contain public miner and research-agent entrypoints.
- `validator/` contains the public replay validator and backend weight setter.
- `scripts/` contains signing, claim, and operational helpers.
- `docs/` contains the public MkDocs site.
- `docs/base/` contains the living SOTA Base project docs. Update it whenever a
  Base contract, root, claim, indexer, UI, or ops behavior changes.
- `docs/archive/automl-zero/` contains historical relay/AutoML-Zero docs.

## Documentation changes

The docs website is built with MkDocs Material.

```bash
python3 -m pip install -r requirements-docs.txt
mkdocs serve -a 127.0.0.1:9001
```

For Base SOTA work, update the relevant `docs/base/` page and append a dated
entry to `docs/base/project-log.md` when the change affects local E2E,
deployment readiness, or operator behavior.
