# Contributing

## Repo layout

- `miner/` and `neurons/` contain public miner and research-agent entrypoints.
- `validator/` contains the public replay validator and backend weight setter.
- `scripts/` contains signing, claim, and operational helpers.
- `docs/` contains the public MkDocs site.
- `docs/archive/automl-zero/` contains historical relay/AutoML-Zero docs.

## Documentation changes

The docs website is built with MkDocs Material.

```bash
python3 -m pip install -r requirements-docs.txt
mkdocs serve -a 127.0.0.1:9001
```
