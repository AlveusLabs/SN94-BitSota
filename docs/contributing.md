# Contributing

## Repo layout

- `relay/` is the relay service
- `Pool/` is the pool service
- `sidecar/` is the local API used by the GUI
- `miner/` and `neurons/` contain miner entrypoints
- `validator/` and `neurons/validator_node.py` contain validator entrypoints

## Documentation changes

The docs website is built with MkDocs Material.

```bash
python3 -m pip install -r requirements-docs.txt
mkdocs serve -a 127.0.0.1:9001
```
