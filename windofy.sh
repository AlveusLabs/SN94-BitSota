#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

MAIN_REF="${MAIN_REF:-main}"
WINDOWS_REF="${WINDOWS_REF:-origin/windows}"

log() {
  printf '[windofy] %s\n' "$*"
}

has_ref() {
  git rev-parse --verify --quiet "$1^{commit}" >/dev/null 2>&1
}

remove_line_literal() {
  local file="$1"
  local literal="$2"
  [[ -f "$file" ]] || return 0
  if grep -Fqx "$literal" "$file"; then
    LITERAL="$literal" perl -0pi -e 'my $x = $ENV{LITERAL}; s/^\Q$x\E\s*\n//mg' "$file"
    log "removed line in $file: $literal"
  fi
}

replace_literal() {
  local file="$1"
  local old="$2"
  local new="$3"
  [[ -f "$file" ]] || return 0
  local tmp
  tmp="$(mktemp)"
  OLD="$old" NEW="$new" perl -0pe 'my ($o, $n) = @ENV{qw(OLD NEW)}; s/\Q$o\E/$n/g' "$file" > "$tmp"
  if ! cmp -s "$file" "$tmp"; then
    mv "$tmp" "$file"
    log "patched $file"
  else
    rm -f "$tmp"
  fi
}

insert_after_literal_if_missing() {
  local file="$1"
  local marker="$2"
  local to_insert="$3"
  [[ -f "$file" ]] || return 0
  if grep -Fq "$marker" "$file" && ! grep -Fq "$to_insert" "$file"; then
    MARKER="$marker" INSERT="$to_insert" perl -0pi -e 'my ($m, $i) = @ENV{qw(MARKER INSERT)}; s/\Q$m\E\n/$m\n$i\n/' "$file"
    log "inserted line in $file after marker: $marker"
  fi
}

print_compare_context() {
  if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    log "not a git repository, skipping comparison context"
    return
  fi

  if ! has_ref "$MAIN_REF"; then
    log "main ref not found: $MAIN_REF"
    return
  fi

  if ! has_ref "$WINDOWS_REF"; then
    if has_ref "windows"; then
      WINDOWS_REF="windows"
    else
      log "windows ref not found: $WINDOWS_REF (and no local windows branch)"
      return
    fi
  fi

  local merge_base
  merge_base="$(git merge-base "$MAIN_REF" "$WINDOWS_REF")"
  log "main ref: $MAIN_REF"
  log "windows ref: $WINDOWS_REF"
  log "equivalent main history commit (merge-base): $merge_base"

  echo
  log "windows-only commits:"
  git log --oneline "${merge_base}..${WINDOWS_REF}" || true

  echo
  log "files changed on windows since merge-base:"
  git diff --name-status "${merge_base}..${WINDOWS_REF}" || true
  echo
}

apply_windows_fixes() {
  log "searching for known Windows-hostile imports"
  if command -v rg >/dev/null 2>&1; then
    rg -n '^(import bittensor as bt|import bittensor\.utils\.networking as net)$' \
      bittensor_network validator 2>/dev/null || true
  fi

  local import_targets=(
    "bittensor_network/_state.py"
    "bittensor_network/_weights.py"
    "bittensor_network/bittensor_config.py"
    "validator/bittensor_validation.py"
  )

  local file
  for file in "${import_targets[@]}"; do
    remove_line_literal "$file" "import bittensor as bt"
  done
  remove_line_literal "validator/bittensor_validation.py" "import bittensor.utils.networking as net"

  if [[ -f requirements.txt ]]; then
    perl -0pi -e 's/^bittensor[^\n]*\n//mg' requirements.txt
    replace_literal "requirements.txt" "setuptools==70.0.0" "setuptools"
    replace_literal "requirements.txt" "setuptools==44.1.1" "setuptools"
    log "updated requirements.txt (removed bittensor pin and normalized setuptools)"
  fi

  replace_literal ".github/workflows/build.yml" "new_gui/images/main_icon.svg" "gui/images/main_icon.svg"
  replace_literal "create_icon.sh" "new_gui/images/main_icon.svg" "gui/images/main_icon.svg"

  replace_literal "gui/screens/mining_screen.py" \
    "self.start_mining_btn.update_icon(\"gui/images/stop.svg\")" \
    "self.start_mining_btn.update_icon(resource_path(\"gui/images/stop.svg\"))"
  replace_literal "gui/screens/pool_mining_screen.py" \
    "self.join_pool_btn.update_icon(\"gui/images/stop.svg\")" \
    "self.join_pool_btn.update_icon(resource_path(\"gui/images/stop.svg\"))"

  replace_literal "gui/resource_path.py" \
    "return str(base_path / relative_path)" \
    "return str((base_path / relative_path).resolve().as_posix())"

  # PyInstaller: include all PySide6 resources/plugins needed on Windows.
  replace_literal "bitsota.spec" \
    "from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs" \
    $'from PyInstaller.utils.hooks import (\n    collect_all,\n    collect_data_files,\n    collect_dynamic_libs,\n)'
  insert_after_literal_if_missing "bitsota.spec" \
    "numpy_data = collect_data_files('numpy')" \
    "pyside_datas, pyside_binaries, pyside_hidden = collect_all('PySide6')"
  insert_after_literal_if_missing "bitsota.spec" \
    "datas.extend(numpy_data)" \
    "datas.extend(pyside_datas)"
  replace_literal "bitsota.spec" "binaries=torch_binaries," "binaries=torch_binaries + pyside_binaries,"
  if [[ -f bitsota.spec ]] && ! grep -Fq "] + pyside_hidden," bitsota.spec; then
    perl -0pi -e "s/'multiprocessing\\.queues'\\n(\\s*)\\],/'multiprocessing.queues'\\n\$1] + pyside_hidden,/s" bitsota.spec
    if grep -Fq "] + pyside_hidden," bitsota.spec; then
      log "patched bitsota.spec"
    fi
  fi

  insert_after_literal_if_missing "gui/__main__.py" "import sys" "import os"

  local old_tee_block
  local new_tee_block
  old_tee_block="$(cat <<'EOF'
    class Tee:
        def __init__(self, file):
            self.file = file
            self.terminal = sys.stdout

        def write(self, message):
            self.terminal.write(message)
            self.file.write(message)
            self.file.flush()

        def flush(self):
            self.terminal.flush()
            self.file.flush()

    log_handle = open(log_file, 'w', buffering=1)
    sys.stdout = Tee(log_handle)
    sys.stderr = sys.stdout
EOF
)"
  new_tee_block="$(cat <<'EOF'
    class Tee:
        def __init__(self, file, terminal):
            self.file = file
            # In a windowed app sys.stdout can be None; fall back to a devnull handle
            self.terminal = terminal

        def write(self, message):
            if self.terminal:
                try:
                    self.terminal.write(message)
                    self.terminal.flush()
                except Exception:
                    # If the terminal stream is not writable, ignore and keep logging to file
                    pass
            self.file.write(message)
            self.file.flush()

        def flush(self):
            if self.terminal:
                try:
                    self.terminal.flush()
                except Exception:
                    pass
            self.file.flush()

    log_handle = open(log_file, 'w', buffering=1)
    terminal = sys.stdout if sys.stdout is not None else open(os.devnull, 'w')
    sys.stdout = Tee(log_handle, terminal)
    sys.stderr = sys.stdout
EOF
)"
  replace_literal "gui/__main__.py" "$old_tee_block" "$new_tee_block"
}

print_compare_context
apply_windows_fixes

log "done. review with: git diff -- ."
