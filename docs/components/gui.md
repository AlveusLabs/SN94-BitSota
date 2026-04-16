# Desktop GUI

The GUI:

- Spawns a local sidecar and miner process when you click Start Mining
- Reads logs and candidate state from the sidecar
- Submits solutions to the relay

For the autoresearch + Pool claim path, the GUI also owns the miner-facing recipient coldkey declaration. That declared payout address is separate from the connected hotkey:

- `hotkey`: used for mining, coordinator auth, and claim package lookup
- `declared recipient coldkey`: the payout address the miner wants Pool to publish into future Merkle epochs

The claim screen should be read with that split in mind:

- the local wallet/setup flow stores the declared recipient coldkey
- the Pool backend is the source of truth for published claim recipients
- the `Recipient` column in claim rows reflects the published claim package, not a blind echo of local GUI state

If the declared recipient and the published recipient differ, the mismatch is on the backend publication path and should be investigated there before claiming.

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

## Recipient coldkey flow

For the testnet/autoresearch claim path, the intended ownership is:

1. the GUI captures the miner's declared recipient coldkey
2. Pool persists the hotkey -> recipient coldkey mapping
3. Pool consensus publication writes that recipient into the Merkle `claim_list`
4. the claim package returned to the GUI includes that same `recipient_coldkey`
5. the GUI submits `claim_single` against the published recipient

Important:

- the GUI must not invent or override the recipient during claim submission
- the published Merkle leaf is the claim-time source of truth
- legacy relay-only coldkey association is not the authority for autoresearch Merkle claims
