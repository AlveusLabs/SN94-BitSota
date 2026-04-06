#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ORIGINAL_HOME="${HOME:-}"
RUNTIME_DIR="${RUNTIME_DIR:-${ROOT_DIR}/.gui-live-testnet}"
CONFIG_PATH="${RUNTIME_DIR}/gui_live_testnet_config.json"
POOL_ENDPOINT="${POOL_ENDPOINT:-https://3fhi3ukpyw.eu-central-1.awsapprunner.com}"
CLAIM_ENDPOINT="${CLAIM_ENDPOINT:-${POOL_ENDPOINT}/claims}"
COORDINATOR_ENDPOINT="${COORDINATOR_ENDPOINT:-https://chvp2wytst.eu-central-1.awsapprunner.com}"
ONCHAIN_WS_URL="${ONCHAIN_WS_URL:-wss://test.finney.opentensor.ai:443}"
ONCHAIN_CONTRACT="${ONCHAIN_CONTRACT:-5GUw1gZVfUTXWLEbA7G6Xdp8QsUHAy2xpVmaj5fRc1gW1Xyy}"
ONCHAIN_METADATA_PATH="${ONCHAIN_METADATA_PATH:-/home/mekaneeky/repos/Pool/new_merkle/app/assets/merklepool.json}"
RESEARCH_TASK_SLUG="${RESEARCH_TASK_SLUG:-bitnet-cpu-ternary-kernel}"
RESEARCH_RESTART_DELAY_SECONDS="${RESEARCH_RESTART_DELAY_SECONDS:-30}"
PYTHON_BIN="${PYTHON_BIN:-${ORIGINAL_HOME}/.pyenv/versions/bitsota_live/bin/python}"
CODEX_BIN="${CODEX_BIN:-$(command -v codex || true)}"
DISPLAY_VALUE="${DISPLAY:-}"
WAYLAND_VALUE="${WAYLAND_DISPLAY:-}"
XDG_RUNTIME_VALUE="${XDG_RUNTIME_DIR:-}"
XAUTH_FILE="${XAUTHORITY:-${ORIGINAL_HOME}/.Xauthority}"
SHARED_CACHE_DIR="${SHARED_CACHE_DIR:-${ORIGINAL_HOME}/.cache}"
SHARED_XDG_DATA_HOME="${SHARED_XDG_DATA_HOME:-${ORIGINAL_HOME}/.local/share}"
SHARED_SCIKIT_LEARN_DATA="${SHARED_SCIKIT_LEARN_DATA:-${SHARED_CACHE_DIR}/scikit_learn}"
SHARED_TFDS_DATA_DIR="${SHARED_TFDS_DATA_DIR:-${SHARED_XDG_DATA_HOME}/tensorflow_datasets}"
SHARED_CODEX_HOME="${SHARED_CODEX_HOME:-${CODEX_HOME:-${ORIGINAL_HOME}/.codex}}"
WALLET_SPECS=("$@")

if [[ ${#WALLET_SPECS[@]} -eq 0 ]]; then
  WALLET_SPECS=("bot:bot_hot")
fi

if [[ -z "${CODEX_BIN}" || ! -x "${CODEX_BIN}" ]]; then
  echo "Could not resolve codex binary. Set CODEX_BIN explicitly." >&2
  exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Could not resolve python at ${PYTHON_BIN}. Set PYTHON_BIN explicitly." >&2
  exit 1
fi

if [[ ! -f "${ONCHAIN_METADATA_PATH}" ]]; then
  echo "Missing Merkle metadata at ${ONCHAIN_METADATA_PATH}" >&2
  exit 1
fi

mkdir -p "${RUNTIME_DIR}"
mkdir -p "${SHARED_CACHE_DIR}" "${SHARED_XDG_DATA_HOME}" "${SHARED_SCIKIT_LEARN_DATA}" "${SHARED_TFDS_DATA_DIR}"

PY_PREFIX="$(cd "$(dirname "${PYTHON_BIN}")/.." && pwd)"
QT_LIB_DIR="${QT_LIB_DIR:-${PY_PREFIX}/lib/python3.11/site-packages/PySide6/Qt/lib}"

cat > "${CONFIG_PATH}" <<JSON
{
  "pool_endpoint": "${POOL_ENDPOINT}",
  "merkle_claim_endpoint": "${CLAIM_ENDPOINT}",
  "onchain_ws_url": "${ONCHAIN_WS_URL}",
  "onchain_contract": "${ONCHAIN_CONTRACT}",
  "onchain_metadata_path": "${ONCHAIN_METADATA_PATH}",
  "research_coordinator_endpoint": "${COORDINATOR_ENDPOINT}",
  "research_agent_command": "${CODEX_BIN} exec --full-auto -C {repo_dir_quoted} --add-dir {workspace_dir_quoted} -o {submission_result_path_quoted} - < {intro_path_quoted}",
  "research_agent_mode": "gui_managed",
  "test_mode": true,
  "research_test_autostart": true,
  "research_test_auto_restart": true,
  "research_test_restart_delay_seconds": ${RESEARCH_RESTART_DELAY_SECONDS},
  "research_test_task_slug": "${RESEARCH_TASK_SLUG}"
}
JSON

launch_one() {
  local idx="$1"
  local wallet_name="$2"
  local hotkey_name="$3"
  local run_dir="${RUNTIME_DIR}/gui-${idx}"
  local home_dir="${run_dir}/home"
  local log_file="${run_dir}/gui.log"
  local pid_file="${run_dir}/gui.pid"
  local coldkey_ss58

  coldkey_ss58="$("${PYTHON_BIN}" - <<'PY' "$wallet_name" "$hotkey_name"
from bittensor_network.wallet import Wallet
import sys

wallet = Wallet(name=sys.argv[1], hotkey=sys.argv[2], path="~/.bittensor/wallets/")
coldkey = getattr(wallet, "coldkeypub", None)
if coldkey is None or not getattr(coldkey, "ss58_address", ""):
    raise SystemExit(1)
print(coldkey.ss58_address)
PY
)"

  mkdir -p "${run_dir}" "${home_dir}/.bitsota" "${home_dir}/.bittensor"
  rm -rf "${home_dir}/.cache"
  ln -s "${SHARED_CACHE_DIR}" "${home_dir}/.cache"
  mkdir -p "${home_dir}/.local"
  rm -rf "${home_dir}/.local/share"
  ln -s "${SHARED_XDG_DATA_HOME}" "${home_dir}/.local/share"
  rm -rf "${home_dir}/.bittensor/wallets"
  ln -s "${ORIGINAL_HOME}/.bittensor/wallets" "${home_dir}/.bittensor/wallets"

  cat > "${home_dir}/.bitsota/wallet_settings.json" <<JSON
{
  "last_wallet": "${wallet_name}",
  "last_hotkey": "${hotkey_name}",
  "coldkey_address": "${coldkey_ss58}"
}
JSON

  (
    cd "${ROOT_DIR}"
    local -a launch_env
    launch_env=(
      "HOME=${home_dir}"
      "BITSOTA_GUI_CONFIG=${CONFIG_PATH}"
      "BITSOTA_RESEARCH_GUI_AUTOSTART=1"
      "BITSOTA_RESEARCH_GUI_AUTO_RESTART=1"
      "BITSOTA_RESEARCH_GUI_RESTART_DELAY_SECONDS=${RESEARCH_RESTART_DELAY_SECONDS}"
      "BITSOTA_RESEARCH_GUI_TASK_SLUG=${RESEARCH_TASK_SLUG}"
      "XDG_CACHE_HOME=${SHARED_CACHE_DIR}"
      "SCIKIT_LEARN_DATA=${SHARED_SCIKIT_LEARN_DATA}"
      "XDG_DATA_HOME=${SHARED_XDG_DATA_HOME}"
      "TFDS_DATA_DIR=${SHARED_TFDS_DATA_DIR}"
      "LD_LIBRARY_PATH=${QT_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    )
    if [[ -n "${SHARED_CODEX_HOME}" ]]; then
      launch_env+=("CODEX_HOME=${SHARED_CODEX_HOME}")
    fi
    if [[ -n "${DISPLAY_VALUE}" ]]; then
      launch_env+=("DISPLAY=${DISPLAY_VALUE}")
    fi
    if [[ -n "${WAYLAND_VALUE}" ]]; then
      launch_env+=("WAYLAND_DISPLAY=${WAYLAND_VALUE}")
    fi
    if [[ -n "${XDG_RUNTIME_VALUE}" ]]; then
      launch_env+=("XDG_RUNTIME_DIR=${XDG_RUNTIME_VALUE}")
    fi
    if [[ -n "${XAUTH_FILE}" && -f "${XAUTH_FILE}" ]]; then
      launch_env+=("XAUTHORITY=${XAUTH_FILE}")
    fi
    setsid env \
      "${launch_env[@]}" \
      "${PYTHON_BIN}" -m gui \
      > "${log_file}" 2>&1 < /dev/null &
    echo "$!" > "${pid_file}"
  )

  local pid
  pid="$(cat "${pid_file}")"
  sleep 2
  if ! kill -0 "${pid}" >/dev/null 2>&1; then
    echo "Failed to launch GUI #${idx} for ${wallet_name}/${hotkey_name}. Check ${log_file}" >&2
    return 1
  fi

  echo "GUI #${idx}: pid=${pid} wallet=${wallet_name}/${hotkey_name}"
  echo "         coldkey=${coldkey_ss58}"
  echo "         log=${log_file}"
}

idx=0
for spec in "${WALLET_SPECS[@]}"; do
  idx=$((idx + 1))
  wallet_name="${spec%%:*}"
  hotkey_name="${spec#*:}"
  if [[ -z "${wallet_name}" || -z "${hotkey_name}" || "${wallet_name}" == "${hotkey_name}" ]]; then
    echo "Invalid wallet spec '${spec}'. Use wallet_name:hotkey_name." >&2
    exit 1
  fi
  launch_one "${idx}" "${wallet_name}" "${hotkey_name}"
done

echo
echo "Launched ${idx} live testnet research GUI instance(s)."
echo "Python: ${PYTHON_BIN}"
echo "Codex: ${CODEX_BIN}"
echo "Config: ${CONFIG_PATH}"
echo "Runtime dir: ${RUNTIME_DIR}"
echo "Stop all launched GUIs:"
echo "  for f in ${RUNTIME_DIR}/gui-*/gui.pid; do kill \"\$(cat \"\$f\")\"; done"
