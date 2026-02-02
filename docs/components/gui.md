# Desktop GUI

The GUI:

- Spawns a local sidecar and miner process when you click Start Mining
- Reads logs and candidate state from the sidecar
- Submits solutions to the relay

## Config overrides

Local/dev runs support endpoint overrides via JSON. The GUI reads (first match):

- `BITSOTA_GUI_CONFIG` (path to a JSON file)
- `./bitsota_gui_config.json`
- `./gui_config.json`
- `~/.bitsota/gui_config.json`

Common keys:

- `relay_endpoint`
- `update_manifest_url`
- `pool_endpoint`
- `test_mode` and `test_invite_code`
- `problem_config_path`

For end-to-end local runs, start with [Local Testing](../local-testing.md).
