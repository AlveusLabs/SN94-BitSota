# Local Testing (Relay + Validator + GUI)

This repo supports a simple **local relay** for development plus the normal validator + GUI.

The goal is to run:
- `relay`: local HTTP server (stores submissions in memory)
- `validator`: polls the relay, verifies submissions, optionally submits SOTA votes
- `gui`: runs the offline AutoML search and submits SOTA breakers to the relay

> The local relay is **dev-only** (no auth enforcement, no persistence). Do not expose it publicly.

---

## 0) Install

```bash
cd SN94-BitSota
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Optional (keeps datasets + cache inside the repo):

```bash
export SCIKIT_LEARN_DATA="$PWD/.data/sklearn"
```

---

## 1) Run the local relay

Default port `8002` matches `validator_config_test.yaml.example`.

```bash
python -m relay --host 127.0.0.1 --port 8002 --dev-log
```

Smoke-check:

```bash
curl -s http://127.0.0.1:8002/health
curl -s http://127.0.0.1:8002/sota_threshold
```

---

## 2) Run the GUI against the local relay

Create a dev config override (the GUI reads this when **not** frozen):

`bitsota_gui_config.json` (in repo root):

```json
{
  "relay_endpoint": "http://127.0.0.1:8002",
  "update_manifest_url": "http://127.0.0.1:8002/version.json",
  "test_mode": true,
  "miner_workers": 2,
  "miner_validate_every_n_generations": 50
}
```

Run:

```bash
BITSOTA_TEST_MODE=1 python -m gui
```

Notes:
- `test_mode` skips invite-code gating and unlocks extra tasks in the GUI task dropdown.
- The GUI still uses a wallet to sign relay requests; use the GUI wallet screen to create/import one.

---

## 3) Run the validator against the local relay

Start from the test example:

```bash
cp validator_config_test.yaml.example validator_config_local.yaml
```

Edit `validator_config_local.yaml`:
- Set `wallet_name`, `wallet_hotkey`, `path`
- Set `netuid`, `network`, `subtensor_chain_endpoint` appropriately
- Ensure:
  - `relay.url: "http://127.0.0.1:8002"`

Run:

```bash
python neurons/validator_node.py --config validator_config_local.yaml
```

Important:
- The validator connects to Bittensor and expects the hotkey to be registered on `netuid`.

---

## 4) End-to-end check

1. Start relay (`python -m relay ...`)
2. Start validator (`python neurons/validator_node.py ...`)
3. Start GUI (`python -m gui`)
4. In the GUI:
   - Load/create a wallet
   - Start mining (Direct Mining)

You should see:
- GUI logs emitting periodic stats + verified-score updates
- Relay logs for `/submit_solution` and (if capacitorless voting enabled) `/sota/vote`
- Validator logs polling `/results` and evaluating submissions

