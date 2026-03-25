#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# install-memmachine-plugin.sh
#
# Installs the MemMachine OpenClaw plugin into a NemoClaw/OpenShell sandbox by:
#   1. finding a ready sandbox
#   2. packing the npm package outside the sandbox
#   3. uploading/extracting it into ~/.openclaw/extensions/openclaw-memmachine
#   4. optionally updating sandbox policy to allow api.memmachine.ai:443
#   5. configuring OpenClaw to use the plugin as the memory backend
#
# Assumptions:
# - Linux host OR macOS with a Docker container running NemoClaw tooling
# - openshell is available (host on Linux, container on macOS)
# - npm is available outside the sandbox
#
# Env vars you can pre-set:
#   MEMMACHINE_API_KEY
#   MEMMACHINE_BASE_URL         (default: https://api.memmachine.ai/v2)
#   MEMMACHINE_USER_ID          (default: openclaw)
#   MEMMACHINE_ORG_ID           (default: openclaw)
#   MEMMACHINE_PROJECT_ID       (default: openclaw)
#   MEMMACHINE_AUTO_CAPTURE     (default: true)
#   MEMMACHINE_AUTO_RECALL      (default: true)
#   MEMMACHINE_SEARCH_THRESHOLD (default: 0.5)
#   MEMMACHINE_TOP_K            (default: 5)
#   NEMOCLAW_CONTAINER_NAME     (default on macOS: nemoclaw-dev)
#   SANDBOX_NAME                (optional: use a specific sandbox)
#   SKIP_POLICY_UPDATE          (default: false)
###############################################################################

PLUGIN_PACKAGE="@memmachine/openclaw-memmachine"
PLUGIN_ID="openclaw-memmachine"
PLUGIN_DIR_NAME="openclaw-memmachine"

MEMMACHINE_BASE_URL="${MEMMACHINE_BASE_URL:-https://api.memmachine.ai/v2}"
MEMMACHINE_USER_ID="${MEMMACHINE_USER_ID:-openclaw}"
MEMMACHINE_ORG_ID="${MEMMACHINE_ORG_ID:-openclaw}"
MEMMACHINE_PROJECT_ID="${MEMMACHINE_PROJECT_ID:-openclaw}"
MEMMACHINE_AUTO_CAPTURE="${MEMMACHINE_AUTO_CAPTURE:-true}"
MEMMACHINE_AUTO_RECALL="${MEMMACHINE_AUTO_RECALL:-true}"
MEMMACHINE_SEARCH_THRESHOLD="${MEMMACHINE_SEARCH_THRESHOLD:-0.5}"
MEMMACHINE_TOP_K="${MEMMACHINE_TOP_K:-5}"
SKIP_POLICY_UPDATE="${SKIP_POLICY_UPDATE:-false}"

OS="$(uname -s)"
NEMOCLAW_CONTAINER_NAME="${NEMOCLAW_CONTAINER_NAME:-nemoclaw-dev}"

RED="$(printf '\033[31m')"
YELLOW="$(printf '\033[33m')"
GREEN="$(printf '\033[32m')"
BLUE="$(printf '\033[34m')"
BOLD="$(printf '\033[1m')"
RESET="$(printf '\033[0m')"

# Log to stderr so stdout stays clean for $(command) captures (e.g. build_plugin_bundle path).
info()    { printf "%b[INFO]%b %s\n" "$BLUE" "$RESET" "$*" >&2; }
warn()    { printf "%b[WARN]%b %s\n" "$YELLOW" "$RESET" "$*" >&2; }
success() { printf "%b[ OK ]%b %s\n" "$GREEN" "$RESET" "$*" >&2; }
die()     { printf "%b[ERR ]%b %s\n" "$RED" "$RESET" "$*" >&2; exit 1; }

cleanup() {
  if [[ -n "${TMP_DIR:-}" && -d "${TMP_DIR:-}" ]]; then
    rm -rf "${TMP_DIR}"
  fi
}
trap cleanup EXIT

run_cmd() {
  if [[ "$OS" == "Darwin" ]]; then
    docker exec -i "${NEMOCLAW_CONTAINER_NAME}" bash -lc "$*"
  else
    bash -lc "$*"
  fi
}

require_host_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

check_prereqs() {
  require_host_cmd npm
  require_host_cmd tar

  if [[ "$OS" == "Darwin" ]]; then
    require_host_cmd docker
    docker ps --format '{{.Names}}' | grep -qx "${NEMOCLAW_CONTAINER_NAME}" \
      || die "Docker container '${NEMOCLAW_CONTAINER_NAME}' is not running"
    run_cmd "command -v openshell >/dev/null 2>&1" \
      || die "'openshell' not found inside container '${NEMOCLAW_CONTAINER_NAME}'"
  else
    command -v openshell >/dev/null 2>&1 \
      || die "'openshell' not found on host"
  fi
}

find_sandbox() {
  if [[ -n "${SANDBOX_NAME:-}" ]]; then
    echo "${SANDBOX_NAME}"
    return
  fi

  # Try to grab the first listed sandbox name.
  local name
  name="$(run_cmd "openshell sandbox list 2>/dev/null | awk 'NR>1 && NF {print \$1; exit}'" || true)"

  [[ -n "${name}" ]] || die "No sandbox found. Create/start a sandbox first."
  echo "${name}"
}

wait_for_sandbox_ready() {
  local sandbox="$1"
  local tries=24
  local phase=""

  info "Waiting for sandbox '${sandbox}' to become Ready..."

  for ((i=1; i<=tries; i++)); do
    # openshell colors output; strip ANSI so we compare to plain "Ready".
    phase="$(run_cmd "openshell sandbox get '${sandbox}' 2>/dev/null | sed 's/\\x1b\\[[0-9;]*m//g' | grep -iE 'Phase:' | head -n1 | sed 's/.*Phase:[[:space:]]*//'" || true)"

    if [[ "${phase}" == "Ready" ]]; then
      success "Sandbox '${sandbox}' is Ready"
      return
    fi

    sleep 5
  done

  die "Sandbox '${sandbox}' did not become Ready in time"
}

sandbox_exec() {
  local sandbox="$1"
  shift
  local script="$*"

  run_cmd "cat <<'EOF' | openshell sandbox connect '${sandbox}'
stty -echo 2>/dev/null || true
${script}
stty echo 2>/dev/null || true
exit 0
EOF"
}

sandbox_upload() {
  local sandbox="$1"
  local src="$2"
  local dest="$3"

  run_cmd "openshell sandbox upload '${sandbox}' '${src}' '${dest}'"
}

sandbox_download() {
  local sandbox="$1"
  local src="$2"
  local dest="$3"

  run_cmd "openshell sandbox download '${sandbox}' '${src}' '${dest}'"
}

plugin_exists_in_sandbox() {
  local sandbox="$1"

  sandbox_exec "${sandbox}" "
set -e
test -f /sandbox/.openclaw/extensions/${PLUGIN_DIR_NAME}/package.json
" >/dev/null 2>&1
}

build_plugin_bundle() {
  TMP_DIR="$(mktemp -d)"
  local pack_dir="${TMP_DIR}/pack"
  local extract_dir="${TMP_DIR}/extract"
  mkdir -p "${pack_dir}" "${extract_dir}"

  info "Packing ${PLUGIN_PACKAGE} outside the sandbox..."
  (
    cd "${pack_dir}"
    npm pack "${PLUGIN_PACKAGE}" >/dev/null
  )

  local tgz
  tgz="$(find "${pack_dir}" -maxdepth 1 -name '*.tgz' | head -n1)"
  [[ -n "${tgz}" ]] || die "npm pack did not produce a tarball"

  info "Extracting package..."
  tar -xzf "${tgz}" -C "${extract_dir}"

  [[ -d "${extract_dir}/package" ]] || die "Expected extracted 'package/' directory not found"

  info "Installing production dependencies..."
  (
    cd "${extract_dir}/package"
    npm install --omit=dev 1>&2
  )

  local final_tgz="${TMP_DIR}/${PLUGIN_DIR_NAME}.tar.gz"
  (
    cd "${extract_dir}"
    tar -czf "${final_tgz}" package
  )

  echo "${final_tgz}"
}

install_plugin_into_sandbox() {
  local sandbox="$1"
  local bundle_tgz="$2"
  # Unique name avoids collisions with a mistaken directory from an older upload, and
  # avoids a pre-upload `sandbox connect` (connect is interactive; piping a lone `rm`
  # often hangs waiting for a TTY).
  local unique remote_bundle local_upload
  unique="$(date +%s)-$$-${RANDOM}"
  remote_bundle="/tmp/${PLUGIN_DIR_NAME}-${unique}.tar.gz"
  local_upload="$(dirname "${bundle_tgz}")/$(basename "${remote_bundle}")"
  cp -f "${bundle_tgz}" "${local_upload}"

  info "Uploading plugin bundle into sandbox..."
  # openshell expects DEST to be a directory; upload basename becomes the remote file.
  sandbox_upload "${sandbox}" "${local_upload}" "/tmp/"

  info "Extracting plugin into sandbox extension directory..."
  sandbox_exec "${sandbox}" "
set -e
mkdir -p /sandbox/.openclaw/extensions
rm -rf /sandbox/.openclaw/extensions/${PLUGIN_DIR_NAME}
mkdir -p /sandbox/.openclaw/extensions/${PLUGIN_DIR_NAME}

cd /sandbox/.openclaw/extensions
tar -xzf '${remote_bundle}'

if [ -d /sandbox/.openclaw/extensions/package ]; then
  rm -rf /sandbox/.openclaw/extensions/${PLUGIN_DIR_NAME}
  mv /sandbox/.openclaw/extensions/package /sandbox/.openclaw/extensions/${PLUGIN_DIR_NAME}
fi

test -f /sandbox/.openclaw/extensions/${PLUGIN_DIR_NAME}/package.json
"
}

update_policy_for_memmachine() {
  local sandbox="$1"

  if [[ "${SKIP_POLICY_UPDATE}" == "true" ]]; then
    warn "Skipping policy update because SKIP_POLICY_UPDATE=true"
    return
  fi

  info "Attempting to update sandbox policy to allow api.memmachine.ai:443..."

  local tmp_policy
  tmp_policy="$(mktemp -t nemoclaw-memmachine-policy.XXXXXX.yaml)"

  # Pull current policy if available, otherwise fall back to a basic path assumption.
  # Must use --full: without it, only metadata is printed (no YAML), so merging never works.
  if ! run_cmd "openshell policy get '${sandbox}' --full > '${tmp_policy}'" >/dev/null 2>&1; then
    warn "Could not fetch current policy with 'openshell policy get --full'."
    warn "Skipping automatic policy update. You may need to allow api.memmachine.ai:443 manually."
    return
  fi

  POLICY_TMP="${tmp_policy}" python3 <<'PY'
import os
from pathlib import Path

import yaml

p = Path(os.environ["POLICY_TMP"])
raw = p.read_text()

# "openshell policy get --full" prints metadata then "---" then the YAML document.
# "openshell policy set" accepts only the YAML document (not the Version/Hash header).
if "---" in raw:
    yaml_text = raw.split("---", 1)[1].lstrip("\n")
else:
    yaml_text = raw

BINARIES = [
    {"path": "/usr/bin/curl"},
    {"path": "/usr/local/bin/openclaw"},
    {"path": "/usr/local/bin/node"},
    {"path": "/usr/bin/node"},
]

data = yaml.safe_load(yaml_text)
np = data.setdefault("network_policies", {})

if "memmachine_api" not in np:
    np["memmachine_api"] = {
        "name": "memmachine_api",
        "endpoints": [
            {
                "host": "api.memmachine.ai",
                "port": 443,
                "protocol": "rest",
                "tls": "terminate",
                "enforcement": "enforce",
                "rules": [
                    {"allow": {"method": "GET", "path": "/**"}},
                    {"allow": {"method": "POST", "path": "/**"}},
                    {"allow": {"method": "PUT", "path": "/**"}},
                    {"allow": {"method": "DELETE", "path": "/**"}},
                ],
            }
        ],
        "binaries": BINARIES,
    }
else:
    mm = np["memmachine_api"]
    if not mm.get("binaries"):
        mm["binaries"] = BINARIES

p.write_text(
    yaml.dump(
        data,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
        width=120,
    )
)
PY

  if run_cmd "openshell policy set '${sandbox}' --policy '${tmp_policy}' --wait" >/dev/null 2>&1; then
    success "Policy updated"
  else
    warn "Automatic policy update failed. You may need to allow api.memmachine.ai:443 manually."
  fi
}

prompt_for_config() {
  if [[ -z "${MEMMACHINE_API_KEY:-}" ]]; then
    printf "%bEnter MemMachine API key%b: " "$BOLD" "$RESET" >&2
    read -r MEMMACHINE_API_KEY
  fi

  [[ -n "${MEMMACHINE_API_KEY:-}" ]] || die "MemMachine API key is required"
}

configure_openclaw_plugin() {
  local sandbox="$1"

  info "Configuring OpenClaw to use ${PLUGIN_ID} as memory plugin..."

  local key_tmp key_file key_basename
  key_tmp="$(mktemp -u)"
  key_basename="$(basename "${key_tmp}")"
  key_file="${key_tmp}"
  printf "%s" "${MEMMACHINE_API_KEY}" > "${key_file}"
  chmod 600 "${key_file}"

  # Upload API key content without embedding it in the remote command string.
  sandbox_upload "${sandbox}" "${key_file}" "/tmp/"

  # shellcheck disable=SC2064
  rm -f "${key_file}" || true

  local patch_py patch_basename
  patch_py="$(mktemp /tmp/memmachine-openclaw-patch.XXXXXX.py)"
  cat > "${patch_py}" <<'EOPY'
import json
import os
from pathlib import Path

def env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name, str(default)).strip().lower()
    return v in ("1", "true", "yes", "y", "on")

cfg_path = Path("/sandbox/.openclaw-data/openclaw.json")
data = json.loads(cfg_path.read_text())

plugin_id = os.environ["PLUGIN_ID"]
api_key = Path(os.environ["KEY_FILE"]).read_text().strip()

plugins = data.setdefault("plugins", {})
plugins["allow"] = [plugin_id]
plugins.setdefault("slots", {})["memory"] = plugin_id

entry = plugins.setdefault("entries", {}).setdefault(plugin_id, {})
entry["enabled"] = True
entry["config"] = {
    "apiKey": api_key,
    "baseUrl": os.environ.get("BASE_URL", ""),
    "userId": os.environ.get("USER_ID", ""),
    "orgId": os.environ.get("ORG_ID", ""),
    "projectId": os.environ.get("PROJECT_ID", ""),
    "autoCapture": env_bool("AUTO_CAPTURE", True),
    "autoRecall": env_bool("AUTO_RECALL", True),
    "searchThreshold": float(os.environ.get("SEARCH_THRESHOLD", "0.5")),
    "topK": int(os.environ.get("TOP_K", "5")),
}

cfg_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
EOPY
  patch_basename="$(basename "${patch_py}")"
  sandbox_upload "${sandbox}" "${patch_py}" "/tmp/"
  rm -f "${patch_py}"

  sandbox_exec "${sandbox}" "
set -e

mkdir -p /sandbox/.openclaw-data
# Memory indexer uses SQLite under OPENCLAW_STATE_DIR/memory/ (e.g. main.sqlite).
mkdir -p /sandbox/.openclaw-data/memory
# Agent workspace memory roots (MEMORY.md / memory/**) live under workspace.
mkdir -p /sandbox/.openclaw-data/workspace/memory 2>/dev/null || true

trap 'rm -f /tmp/${key_basename} /tmp/${patch_basename} >/dev/null 2>&1 || true' EXIT

# Writable config copy (default ~/.openclaw/openclaw.json is often root-owned).
cp -f /sandbox/.openclaw/openclaw.json /sandbox/.openclaw-data/openclaw.json
export OPENCLAW_STATE_DIR=/sandbox/.openclaw-data
export OPENCLAW_CONFIG_PATH=/sandbox/.openclaw-data/openclaw.json

export PLUGIN_ID='${PLUGIN_ID}'
export KEY_FILE=/tmp/${key_basename}
export BASE_URL='${MEMMACHINE_BASE_URL}'
export USER_ID='${MEMMACHINE_USER_ID}'
export ORG_ID='${MEMMACHINE_ORG_ID}'
export PROJECT_ID='${MEMMACHINE_PROJECT_ID}'
export AUTO_CAPTURE='${MEMMACHINE_AUTO_CAPTURE}'
export AUTO_RECALL='${MEMMACHINE_AUTO_RECALL}'
export SEARCH_THRESHOLD='${MEMMACHINE_SEARCH_THRESHOLD}'
export TOP_K='${MEMMACHINE_TOP_K}'

python3 /tmp/${patch_basename}

openclaw config validate >/dev/null
echo 'OpenClaw config validated (memory slot updated)'

openclaw gateway restart >/dev/null 2>&1 || true

# Persist env vars so every sandbox shell session uses the writable config.
grep -q OPENCLAW_STATE_DIR ~/.bashrc 2>/dev/null || \
  echo 'export OPENCLAW_STATE_DIR=/sandbox/.openclaw-data' >> ~/.bashrc
grep -q OPENCLAW_CONFIG_PATH ~/.bashrc 2>/dev/null || \
  echo 'export OPENCLAW_CONFIG_PATH=/sandbox/.openclaw-data/openclaw.json' >> ~/.bashrc

"
}

patch_axios_for_proxy() {
  local sandbox="$1"

  info "Patching @memmachine/client axios to use fetch adapter (proxy compatibility)..."

  local patch_py
  patch_py="$(mktemp /tmp/memmachine-axios-patch.XXXXXX.py)"
  cat > "${patch_py}" <<'EOPY'
import os

BASE = "/sandbox/.openclaw-data/extensions/openclaw-memmachine/node_modules/@memmachine/client/dist"

old = "timeout: timeout != null ? timeout : 6e4\n    });"
new = "timeout: timeout != null ? timeout : 6e4,\n      proxy: false,\n      adapter: 'fetch'\n    });"

patched = 0
for f in ("index.mjs", "index.js"):
    path = os.path.join(BASE, f)
    if not os.path.exists(path):
        continue
    code = open(path).read()
    if "adapter: 'fetch'" in code:
        patched += 1
    elif old in code:
        open(path, "w").write(code.replace(old, new, 1))
        patched += 1

print(f"axios patched ({patched} file(s))")
EOPY
  local patch_basename
  patch_basename="$(basename "${patch_py}")"
  sandbox_upload "${sandbox}" "${patch_py}" "/tmp/"
  rm -f "${patch_py}"

  sandbox_exec "${sandbox}" "
python3 /tmp/${patch_basename}
rm -f /tmp/${patch_basename}
"
}

main() {
  check_prereqs

  local sandbox
  sandbox="$(find_sandbox)"
  info "Using sandbox: ${sandbox}"

  wait_for_sandbox_ready "${sandbox}"

  if plugin_exists_in_sandbox "${sandbox}"; then
    warn "Plugin already exists in sandbox. Skipping upload/install."
  else
    local bundle_tgz
    bundle_tgz="$(build_plugin_bundle)"
    install_plugin_into_sandbox "${sandbox}" "${bundle_tgz}"
    success "Plugin files installed into sandbox"
  fi

  update_policy_for_memmachine "${sandbox}"
  prompt_for_config
  configure_openclaw_plugin "${sandbox}"
  patch_axios_for_proxy "${sandbox}"

  success "MemMachine plugin setup complete"
  printf "\n"
  printf "Sandbox      : %s\n" "${sandbox}"
  printf "Plugin       : %s\n" "${PLUGIN_ID}"
  printf "Base URL     : %s\n" "${MEMMACHINE_BASE_URL}"
  printf "User ID      : %s\n" "${MEMMACHINE_USER_ID}"
  printf "Org ID       : %s\n" "${MEMMACHINE_ORG_ID}"
  printf "Project ID   : %s\n" "${MEMMACHINE_PROJECT_ID}"
  printf "Auto Capture : %s\n" "${MEMMACHINE_AUTO_CAPTURE}"
  printf "Auto Recall  : %s\n" "${MEMMACHINE_AUTO_RECALL}"
  printf "Threshold    : %s\n" "${MEMMACHINE_SEARCH_THRESHOLD}"
  printf "Top K        : %s\n" "${MEMMACHINE_TOP_K}"
  printf "\n"
  printf "Next step (inside the sandbox): export OPENCLAW_STATE_DIR=/sandbox/.openclaw-data OPENCLAW_CONFIG_PATH=/sandbox/.openclaw-data/openclaw.json, then run 'openclaw plugins list' and 'openclaw memory status'.\n"
}

main "$@"