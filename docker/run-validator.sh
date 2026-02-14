#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${VALIDATOR_WALLET_NAME:-}" ]]; then
  echo "VALIDATOR_WALLET_NAME is required" >&2
  exit 1
fi

if [[ -z "${VALIDATOR_WALLET_HOTKEY:-}" ]]; then
  echo "VALIDATOR_WALLET_HOTKEY is required" >&2
  exit 1
fi

CONFIG_PATH="${VALIDATOR_CONFIG_PATH:-/tmp/validator_config.yaml}"
LOG_FILE="${VALIDATOR_LOG_FILE:-/data/validator.log}"

mkdir -p "$(dirname "${LOG_FILE}")"

cat >"${CONFIG_PATH}" <<EOF
reward_mode: "capacitorless"

netuid: ${NETUID:-94}
wallet_name: "${VALIDATOR_WALLET_NAME}"
wallet_hotkey: "${VALIDATOR_WALLET_HOTKEY}"
path: "${VALIDATOR_WALLET_PATH:-/wallets}"
network: "${SUBTENSOR_NETWORK:-test}"
epoch_length: ${EPOCH_LENGTH:-360}
subtensor_chain_endpoint: "${SUBTENSOR_CHAIN_ENDPOINT:-wss://test.finney.opentensor.ai:443}"

capacitorless:
  mode: "sticky_burnsplit"
  winner_source: "${WINNER_SOURCE:-relay}"
  min_winner_improvement: ${MIN_WINNER_IMPROVEMENT:-0.0}
  submit_sota_votes: ${SUBMIT_SOTA_VOTES:-true}
  apply_weights_inline: ${APPLY_WEIGHTS_INLINE:-true}
  burn_hotkey: "${BURN_HOTKEY:-5Ef5EsPQoMVmJ8rYectQ26BEvscvATEGm365bcQjo1Y6bxGr}"
  burn_share: ${BURN_SHARE:-0.9}
  alignment_mod: ${ALIGNMENT_MOD:-360}
  events_limit: ${EVENTS_LIMIT:-50}
  event_refresh_interval_s: ${EVENT_REFRESH_INTERVAL_S:-60}
  metagraph_refresh_interval_s: ${METAGRAPH_REFRESH_INTERVAL_S:-600}
  poll_interval_s: ${WEIGHT_POLL_INTERVAL_S:-6.0}
  retry_interval_s: ${WEIGHT_RETRY_INTERVAL_S:-5.0}

relay:
  url: "${RELAY_URL:-https://relay.bitsota.com}"
  poll_interval_seconds: ${RELAY_POLL_INTERVAL_SECONDS:-60}

submission_schedule:
  mode: "${SUBMISSION_SCHEDULE_MODE:-interval}"
  interval_seconds: ${SUBMISSION_INTERVAL_SECONDS:-600}
  utc_times: []

submission_threshold:
  mode: "${SUBMISSION_THRESHOLD_MODE:-local_best}"

blacklist:
  cutoff_percentage: ${BLACKLIST_CUTOFF_PERCENTAGE:-0.1}

logging:
  level: "${VALIDATOR_LOG_LEVEL:-INFO}"
  file: "${LOG_FILE}"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
EOF

echo "Starting validator wallet=${VALIDATOR_WALLET_NAME} hotkey=${VALIDATOR_WALLET_HOTKEY}"
echo "Relay URL: ${RELAY_URL:-https://relay.bitsota.com}"

exec python -u neurons/validator_node.py --config "${CONFIG_PATH}"
