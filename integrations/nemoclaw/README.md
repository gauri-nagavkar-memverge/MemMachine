# MemMachine + NemoClaw Setup Guide

Add persistent, cloud-backed memory to your NemoClaw sandbox using the
[MemMachine](https://memmachine.ai) OpenClaw plugin. Conversations are
automatically captured to and recalled from MemMachine so your agent remembers
across sessions.

---

## Prerequisites

| Requirement | How to check |
|-------------|--------------|
| NemoClaw installed and working | `nemoclaw --version` |
| A running sandbox | `nemoclaw <sandbox-name> status` shows **Ready** |
| npm available on the **host** | `npm --version` |
| Python 3 + PyYAML on the **host** | `python3 -c "import yaml"` |
| A MemMachine API key | [console.memmachine.ai](https://console.memmachine.ai) |

> **Don't have PyYAML?** Install it: `pip install pyyaml` or
> `sudo apt install python3-yaml`.

---

## Quick start (one command)

```bash
MEMMACHINE_API_KEY="mm-sk-..." bash memmachine-nemoclaw.sh
```

The script handles everything: packing the plugin, uploading it into the
sandbox, updating the network policy, configuring OpenClaw, and patching the
HTTP client for proxy compatibility.

When it finishes you'll see:

```
[ OK ] MemMachine plugin setup complete

Sandbox      : my-assistant
Plugin       : openclaw-memmachine
Base URL     : https://api.memmachine.ai/v2
...
```

---

## Step-by-step walkthrough

### 1. Get a MemMachine API key

Sign up or log in at [console.memmachine.ai](https://console.memmachine.ai)
and create an API key. It looks like `mm-sk-...`.

### 2. Run the installer script

From your **host** (not inside the sandbox):

```bash
# If you have one sandbox:
MEMMACHINE_API_KEY="mm-sk-..." bash memmachine-nemoclaw.sh

# If you have multiple sandboxes, specify which one:
SANDBOX_NAME=my-assistant MEMMACHINE_API_KEY="mm-sk-..." bash memmachine-nemoclaw.sh
```

The script will:

1. **Find your sandbox** and wait for it to be Ready.
2. **Pack the plugin** (`@memmachine/openclaw-memmachine`) from npm, install
   its production dependencies, and bundle them into a tarball.
3. **Upload and extract** the plugin into the sandbox at
   `~/.openclaw/extensions/openclaw-memmachine/`.
4. **Update the network policy** to allow outbound HTTPS to
   `api.memmachine.ai:443` (with the correct binary allowlist and HTTP method
   rules for OpenShell's TLS-terminating proxy).
5. **Configure OpenClaw** — sets the plugin as the memory backend
   (`plugins.slots.memory`), adds it to `plugins.allow`, and writes the API
   key and settings into a writable copy of `openclaw.json`.
6. **Patch the HTTP client** — the MemMachine SDK uses axios, which has a
   known incompatibility with OpenShell's HTTPS proxy. The script patches it
   to use Node's native `fetch()` adapter instead.
7. **Persist env vars** — adds `OPENCLAW_STATE_DIR` and
   `OPENCLAW_CONFIG_PATH` to `~/.bashrc` inside the sandbox so every session
   uses the correct config.

### 3. Connect to the sandbox

```bash
nemoclaw my-assistant connect
```

### 4. Verify the plugin is loaded

```bash
openclaw plugins list
```

You should see **MemMachine** with status **loaded**:

```
│ MemMac │ ope │ loaded │ global:openclaw-  │ 0.3.2 │
│ hine   │ ncl │        │ memmachine/dist/  │       │
│        │ aw- │        │ index.mjs         │       │
│        │ mem │        │                   │       │
│        │ mac │        │                   │       │
│        │ hin │        │                   │       │
│        │ e   │        │                   │       │
```

### 5. Verify the config

```bash
openclaw config get plugins.slots.memory
# → openclaw-memmachine

openclaw config get plugins.allow
# → ["openclaw-memmachine"]
```

### 6. Test memory capture and recall

Send a message to your agent:

```bash
openclaw agent --agent main --local -m "Remember: my favorite color is blue"
```

Look for this in the output:

```
[plugins] openclaw-memmachine: auto-captured 1 memories
```

Then test recall:

```bash
openclaw agent --agent main --local -m "What is my favorite color?"
```

The agent should answer **blue**, recalled from MemMachine.

### 7. Verify directly via the API (optional)

```bash
curl -sS -X POST \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{"user_id":"openclaw","org_id":"openclaw","project_id":"openclaw","page_size":5,"page_num":0,"type":"episodic"}' \
  https://api.memmachine.ai/v2/memories/list
```

A non-empty `episodic_memory` array confirms memories are stored in
MemMachine.

---

## Configuration options

All options can be set as environment variables before running the script:

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMMACHINE_API_KEY` | *(required)* | Your MemMachine API key |
| `MEMMACHINE_BASE_URL` | `https://api.memmachine.ai/v2` | MemMachine API base URL |
| `MEMMACHINE_USER_ID` | `openclaw` | User ID for memory scoping |
| `MEMMACHINE_ORG_ID` | `openclaw` | Organization ID |
| `MEMMACHINE_PROJECT_ID` | `openclaw` | Project ID |
| `MEMMACHINE_AUTO_CAPTURE` | `true` | Automatically capture conversation memories |
| `MEMMACHINE_AUTO_RECALL` | `true` | Automatically recall relevant memories |
| `MEMMACHINE_SEARCH_THRESHOLD` | `0.5` | Minimum similarity score for recall |
| `MEMMACHINE_TOP_K` | `5` | Max number of memories to recall per query |
| `SANDBOX_NAME` | *(auto-detected)* | Target sandbox name |
| `SKIP_POLICY_UPDATE` | `false` | Skip the network policy update step |
| `NEMOCLAW_CONTAINER_NAME` | `nemoclaw-dev` | Docker container name (macOS only) |

---

## Troubleshooting

### `plugins.allow is empty` warning

The sandbox session isn't using the patched config. Make sure the env vars are
set:

```bash
export OPENCLAW_STATE_DIR=/sandbox/.openclaw-data
export OPENCLAW_CONFIG_PATH=/sandbox/.openclaw-data/openclaw.json
```

The script adds these to `~/.bashrc` automatically. If they're missing, run
`source ~/.bashrc` or start a new session.

### `CONNECT tunnel failed, response 403`

The network policy doesn't include `api.memmachine.ai`. Re-run the script or
manually update the policy on the host:

```bash
openshell policy get my-assistant --full > /tmp/policy.yaml
# Edit /tmp/policy.yaml to add memmachine_api under network_policies
openshell policy set my-assistant --policy /tmp/policy.yaml --wait
```

The `memmachine_api` block needs both `endpoints` (host, port, rules) **and**
`binaries` (curl, openclaw, node). Without `binaries`, the proxy denies the
connection.

### `stream has been aborted`

The MemMachine SDK uses axios, which has a bug with OpenShell's
TLS-terminating HTTPS proxy. The script patches the SDK to use Node's native
`fetch()` adapter. If you see this error, re-run the script or manually patch:

```bash
# Inside the sandbox:
python3 -c "
import os
BASE = '/sandbox/.openclaw-data/extensions/openclaw-memmachine/node_modules/@memmachine/client/dist'
old = \"timeout: timeout != null ? timeout : 6e4\n    });\"
new = \"timeout: timeout != null ? timeout : 6e4,\n      proxy: false,\n      adapter: 'fetch'\n    });\"
for f in ('index.mjs', 'index.js'):
    p = os.path.join(BASE, f)
    if not os.path.exists(p): continue
    c = open(p).read()
    if old in c:
        open(p, 'w').write(c.replace(old, new, 1))
        print(f'Patched {f}')
"
```

### `No reply from agent`

This usually means the LLM provider returned an empty response or timed out.
Check your inference configuration (`nemoclaw <sandbox> status`) and API key.

### `openclaw memory status` shows `Provider: none`

This refers to OpenClaw's **local** embedding provider (for vector search of
`.md` files), not the MemMachine plugin. The MemMachine plugin handles memory
through its own API. This status is expected if you haven't configured a
separate embedding API key (OpenAI, Google, etc.).

---

## How it works

```
┌─────────────────────────────────────────────────────┐
│  Host                                               │
│                                                     │
│  memmachine-nemoclaw.sh                             │
│    ├─ npm pack + bundle plugin                      │
│    ├─ openshell policy set (network rules)          │
│    ├─ openshell sandbox upload (plugin + config)    │
│    └─ openshell sandbox connect (configure inside)  │
│                                                     │
└────────────────┬────────────────────────────────────┘
                 │ upload / connect
                 ▼
┌─────────────────────────────────────────────────────┐
│  Sandbox (my-assistant)                             │
│                                                     │
│  ~/.openclaw/extensions/openclaw-memmachine/        │
│    └─ Plugin code + patched @memmachine/client      │
│                                                     │
│  ~/.openclaw-data/openclaw.json                     │
│    └─ plugins.slots.memory = openclaw-memmachine    │
│    └─ plugins.allow = [openclaw-memmachine]         │
│    └─ plugins.entries.openclaw-memmachine.config    │
│         └─ apiKey, baseUrl, autoCapture, etc.       │
│                                                     │
│  OpenClaw agent ──► MemMachine plugin               │
│    user message  ──► auto-recall (search memories)  │
│    agent reply   ──► auto-capture (store memories)  │
│                        │                            │
└────────────────────────┼────────────────────────────┘
                         │ HTTPS via proxy
                         ▼
              ┌──────────────────────┐
              │  api.memmachine.ai   │
              │  /v2/memories/search │
              │  /v2/memories        │
              │  /v2/memories/list   │
              └──────────────────────┘
```

---

## Uninstalling

Inside the sandbox:

```bash
rm -rf ~/.openclaw/extensions/openclaw-memmachine
rm -rf ~/.openclaw-data/extensions/openclaw-memmachine
```

Then remove the plugin from the config:

```bash
openclaw config set plugins.slots.memory ""
openclaw config set plugins.allow "[]"
```

Optionally remove the env var exports from `~/.bashrc`.
