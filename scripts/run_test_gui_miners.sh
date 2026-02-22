#!/usr/bin/env bash
set -euo pipefail

COUNT="${1:-3}"
START_PORT="${START_PORT:-8123}"
RELAY_ENDPOINT="${RELAY_ENDPOINT:-http://127.0.0.1:8002}"
POOL_ENDPOINT="${POOL_ENDPOINT:-http://127.0.0.1:8434}"
LEASE_EVAL_BATCH_SIZE="${LEASE_EVAL_BATCH_SIZE:-6}"
LEASE_SEED_BATCH_SIZE="${LEASE_SEED_BATCH_SIZE:-6}"
LEASE_EVOLVE_GENERATIONS="${LEASE_EVOLVE_GENERATIONS:-100000}"
LEASE_SUBMIT_BUFFER_S="${LEASE_SUBMIT_BUFFER_S:-180}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="${RUNTIME_DIR:-${ROOT_DIR}/.gui-test-runs}"
CONFIG_PATH="${RUNTIME_DIR}/gui_test_config.json"
ORIGINAL_HOME="${HOME:-}"
if [[ -n "${XAUTHORITY:-}" ]]; then
  XAUTH_FILE="${XAUTHORITY}"
elif [[ -n "${ORIGINAL_HOME}" ]]; then
  XAUTH_FILE="${ORIGINAL_HOME}/.Xauthority"
else
  XAUTH_FILE=""
fi
PYENV_NAME="${PYENV_NAME:-automl}"
PYTHON_BIN="${PYTHON_BIN:-}"

if [[ -z "${PYTHON_BIN}" ]]; then
  if ! command -v pyenv >/dev/null 2>&1; then
    echo "pyenv is required when PYTHON_BIN is not set." >&2
    exit 1
  fi
  PYTHON_BIN="$(PYENV_VERSION="${PYENV_NAME}" pyenv which python3 2>/dev/null || true)"
fi

if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
  echo "Could not resolve python3 in pyenv env '${PYENV_NAME}'. Set PYTHON_BIN explicitly." >&2
  exit 1
fi

PY_PREFIX="$(cd "$(dirname "${PYTHON_BIN}")/.." && pwd)"
QT_LIB_DIR="${QT_LIB_DIR:-${PY_PREFIX}/lib/python3.11/site-packages/PySide6/Qt/lib}"
if [[ ! -d "${QT_LIB_DIR}" ]]; then
  echo "Warning: Qt lib directory not found at ${QT_LIB_DIR}" >&2
fi

if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
  echo "No DISPLAY/WAYLAND_DISPLAY detected. GUI windows may not appear." >&2
fi

mkdir -p "${RUNTIME_DIR}"

cat > "${CONFIG_PATH}" <<JSON
{
  "relay_endpoint": "${RELAY_ENDPOINT}",
  "pool_endpoint": "${POOL_ENDPOINT}",
  "update_manifest_url": "${RELAY_ENDPOINT}/version.json",
  "test_mode": true,
  "test_invite_code": "TESTTEST1",
  "problem_config_path": "${ROOT_DIR}/problem_config.json",
  "pool_lease_evolve_generations": ${LEASE_EVOLVE_GENERATIONS}
}
JSON

launch_one() {
  local idx="$1"
  local port=$((START_PORT + idx - 1))
  local run_dir="${RUNTIME_DIR}/gui-${idx}"
  local home_dir="${run_dir}/home"
  local wallet_info="${run_dir}/wallet.json"
  local log_file="${run_dir}/gui.log"
  local pid_file="${run_dir}/gui.pid"

  mkdir -p "${run_dir}" "${home_dir}"

  HOME="${home_dir}" "${PYTHON_BIN}" - <<'PY' > "${wallet_info}"
import importlib.util
import json
import uuid
from pathlib import Path

from substrateinterface import Keypair, KeypairType

repo_root = Path.cwd()
keyfile_path = repo_root / "bittensor_network" / "keyfile.py"
spec = importlib.util.spec_from_file_location("bitsota_keyfile", str(keyfile_path))
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
keyfile_cls = module.keyfile

wallet_name = f"testrun_{uuid.uuid4().hex[:8]}"
hotkey_name = f"hk_{uuid.uuid4().hex[:8]}"

wallet_root = Path.home() / ".bitsota" / "wallets" / wallet_name
hotkey_path = wallet_root / "hotkeys" / hotkey_name
coldkey_path = wallet_root / "coldkey"
coldkeypub_path = wallet_root / "coldkeypub.txt"

hotkey = Keypair.create_from_mnemonic(
    Keypair.generate_mnemonic(),
    crypto_type=KeypairType.SR25519,
)
coldkey = Keypair.create_from_mnemonic(
    Keypair.generate_mnemonic(),
    crypto_type=KeypairType.SR25519,
)
coldkeypub = Keypair(ss58_address=coldkey.ss58_address, crypto_type=KeypairType.SR25519)

keyfile_cls(str(hotkey_path)).set_keypair(hotkey, encrypt=False, overwrite=True)
keyfile_cls(str(coldkey_path)).set_keypair(coldkey, encrypt=False, overwrite=True)
keyfile_cls(str(coldkeypub_path)).set_keypair(coldkeypub, encrypt=False, overwrite=True)

settings_path = Path.home() / ".bitsota" / "wallet_settings.json"
settings_path.parent.mkdir(parents=True, exist_ok=True)
settings = {
    "last_wallet": wallet_name,
    "last_hotkey": hotkey_name,
    "coldkey_address": coldkey.ss58_address,
}
settings_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")

print(
    json.dumps(
        {
            "wallet_name": wallet_name,
            "hotkey_name": hotkey_name,
            "hotkey_ss58": hotkey.ss58_address,
            "coldkey_ss58": coldkey.ss58_address,
        }
    )
)
PY

  (
    cd "${ROOT_DIR}"
    local -a launch_env
    launch_env=(
      "HOME=${home_dir}"
      "BITSOTA_GUI_CONFIG=${CONFIG_PATH}"
      "BITSOTA_TEST_MODE=1"
      "BITSOTA_AUTO_START_MINING=1"
      "BITSOTA_AUTO_TASK_TYPE=pool_lease"
      "BITSOTA_SIDECAR_PORT=${port}"
      "BITSOTA_POOL_LEASE_EVAL_BATCH_SIZE=${LEASE_EVAL_BATCH_SIZE}"
      "BITSOTA_POOL_LEASE_SEED_BATCH_SIZE=${LEASE_SEED_BATCH_SIZE}"
      "BITSOTA_POOL_LEASE_SUBMIT_BUFFER_S=${LEASE_SUBMIT_BUFFER_S}"
      "LD_LIBRARY_PATH=${QT_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    )
    if [[ -n "${XAUTH_FILE}" ]]; then
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
  sleep 1

  if ! kill -0 "${pid}" >/dev/null 2>&1; then
    echo "Failed to launch GUI #${idx}. Check ${log_file}" >&2
    return 1
  fi

  local wallet_name
  local hotkey_name
  local hotkey_ss58
  local coldkey_ss58
  wallet_name="$("${PYTHON_BIN}" - <<'PY' "${wallet_info}"
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    d = json.load(f)
print(d.get("wallet_name", ""))
PY
)"
  hotkey_name="$("${PYTHON_BIN}" - <<'PY' "${wallet_info}"
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    d = json.load(f)
print(d.get("hotkey_name", ""))
PY
)"
  hotkey_ss58="$("${PYTHON_BIN}" - <<'PY' "${wallet_info}"
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    d = json.load(f)
print(d.get("hotkey_ss58", ""))
PY
)"
  coldkey_ss58="$("${PYTHON_BIN}" - <<'PY' "${wallet_info}"
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    d = json.load(f)
print(d.get("coldkey_ss58", ""))
PY
)"

  echo "GUI #${idx}: pid=${pid} sidecar_port=${port} wallet=${wallet_name}/${hotkey_name}"
  echo "         hotkey=${hotkey_ss58}"
  echo "         coldkey=${coldkey_ss58}"
  echo "         log=${log_file}"
}

for i in $(seq 1 "${COUNT}"); do
  launch_one "${i}"
done

echo
echo "All GUI launch requests submitted."
echo "Python: ${PYTHON_BIN}"
echo "Qt libs: ${QT_LIB_DIR}"
echo "Runtime dir: ${RUNTIME_DIR}"
echo "Stop all launched GUIs:"
echo "  for f in ${RUNTIME_DIR}/gui-*/gui.pid; do kill \"\$(cat \"\$f\")\"; done"
