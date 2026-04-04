# OpenClaw Multi-Agent on SBCs

Run multiple [OpenClaw](https://github.com/openclaw/openclaw) AI agents on a single-board computer with shared local models. This guide covers LXC container isolation, shared embedding/reranking/query-expansion via [rkllama](https://github.com/NotPunchnox/rkllama) on the RK3588 NPU, and hardware acceleration on ARM SBCs.

## Why

Running multiple AI agents on one SBC means each agent loads its own copy of the embedding model (~892MB), reranker (~892MB), and query expansion model (~2.3GB) into RAM. On a 16GB device running 3 agents, that's ~12GB of duplicated model weights.

This guide shows how to:
- Isolate each agent in its own LXC container
- Share models across all agents via a single `rkllama` instance on the NPU
- Configure OpenClaw to use QMD's hybrid search (BM25 + vector + reranking)
- Run all three models on the RK3588 NPU for dramatically faster inference

## Hardware Tested

| Device | SoC | RAM | GPU | NPU | Status |
|--------|-----|-----|-----|-----|--------|
| Orange Pi 5 Plus | RK3588 | 16GB | Mali-G610 (Vulkan) | 6 TOPS | Working |

Contributions for other SBCs (Raspberry Pi 5, Rock 5B, Khadas Edge2, etc.) welcome.

## Architecture

```
Orange Pi 5 Plus (Host)
├── rkllama (port 8080)                 ← NPU inference, loads models ONCE
│   ├── qwen3-embedding-0.6b  (NPU)     ← 892MB, 1024 dims
│   ├── qwen3-reranker-0.6b   (NPU)     ← 892MB
│   └── qmd-query-expansion   (NPU)     ← 2.3GB, fine-tuned Qwen3-1.7B
│
├── qmd serve (port 7832)               ← Routes requests to rkllama
│
├── LXC: agent-1                        ← Agent 1
│   └── openclaw-gateway → QMD_SERVER=http://host:7832
│
├── LXC: agent-2                        ← Agent 2
│   └── openclaw-gateway → QMD_SERVER=http://host:7832
│
└── LXC: agent-3                        ← Agent 3
    └── openclaw-gateway → QMD_SERVER=http://host:7832
```

## Quick Start

### 1. Set up rkllama for NPU inference

rkllama provides an Ollama-compatible API for the RK3588 NPU. Install it on the host:

```sh
git clone https://github.com/NotPunchnox/rkllama.git ~/rkllama
cd ~/rkllama && go build -o rkllama .
sudo cp rkllama /usr/local/bin/
```

Convert your models to `.rkllm` format **on an x86 machine** using RKLLM toolkit v1.2.3 (w8a8 quantization). See [docs/rk3588-acceleration.md](docs/rk3588-acceleration.md) for the full conversion process.

> **Important:** Pre-converted `.rkllm` models from HuggingFace may not work if they were built with a different toolkit version. You must convert with the matching toolkit version (v1.2.3 for RKLLM runtime v1.2.3).

Place converted models and Modelfiles in `~/.rkllama/models/`. See `configs/Modelfile.*` for templates.

Start rkllama:

```sh
rkllama serve --port 8080 --bind 0.0.0.0
```

### 2. Install QMD with remote model support

The upstream QMD doesn't support remote model serving yet. Use our fork:

```sh
# On the host (model server)
git clone -b feat/remote-llm-provider https://github.com/jaylfc/qmd.git ~/qmd-server
cd ~/qmd-server && npm install && npm run build

# Start serving models (routes to rkllama)
node dist/cli/qmd.js serve --port 7832 --bind 0.0.0.0
```

### 3. Create an LXC container for each agent

```sh
# Create a privileged container with macvlan networking
sudo incus launch images:debian/trixie/arm64 myagent \
  -c security.privileged=true \
  -c security.nesting=true

# Add network
sudo incus config device add myagent eth0 nic network=lan-macvlan

# Install Node.js 22 + OpenClaw inside
sudo incus exec myagent -- bash -c '
  curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
  apt-get install -y nodejs git curl build-essential
  npm install -g openclaw@latest
'

# Create agent user
sudo incus exec myagent -- useradd -m -s /bin/bash -G sudo,users myagent
```

### 4. Install QMD fork in the container

```sh
sudo incus exec myagent -- su - myagent -c '
  git clone -b feat/remote-llm-provider https://github.com/jaylfc/qmd.git ~/qmd
  cd ~/qmd && npm install && npm run build
  sudo ln -sf ~/qmd/dist/cli/qmd.js /usr/local/bin/qmd
  echo "export QMD_SERVER=http://HOST_TAILSCALE_IP:7832" >> ~/.bashrc
'
```

### 5. Configure OpenClaw to use QMD memory

In the agent's `~/.openclaw/openclaw.json`:

```json
{
  "memory": {
    "backend": "qmd",
    "qmd": {
      "command": "/usr/local/bin/qmd",
      "searchMode": "query",
      "includeDefaultMemory": true,
      "update": { "interval": "5m", "debounceMs": 15000, "onBoot": true },
      "limits": { "maxResults": 6, "maxSnippetChars": 700, "timeoutMs": 8000 }
    }
  }
}
```

### 6. Index and embed workspace

```sh
export QMD_SERVER=http://HOST_IP:7832

# Create QMD collection for workspace
mkdir -p ~/.config/qmd
cat > ~/.config/qmd/index.yml << 'YAML'
collections:
  workspace:
    path: /home/myagent/.openclaw/workspace
    pattern: '**/*.md'
    ignore: ['node_modules/**', '.git/**']
    includeByDefault: true
YAML

# Index files and generate embeddings
qmd update
qmd embed
```

> **Note on performance:** With NPU acceleration, embedding is ~1.25s per chunk (down from 3-5s on CPU). Initial indexing of large workspaces is much faster. Running parallel `qmd embed` from multiple agents simultaneously is still not recommended — stagger them or let incremental updates handle it.

## Networking Notes

### macvlan and Host Connectivity

LXC containers using macvlan networking **cannot reach the host's LAN IP** directly. This is a Linux networking limitation, not a bug.

**Solution:** Use Tailscale. Install Tailscale on both the host and each container, then use the host's Tailscale IP for `QMD_SERVER`.

```sh
# In each container
curl -fsSL https://tailscale.com/install.sh | sh
tailscale up --auth-key=YOUR_KEY --hostname=agent-name
```

## Systemd Services

### rkllama NPU Server (on host)

```ini
# /etc/systemd/system/rkllama.service
```

See `configs/rkllama.service` for the full unit file.

```sh
sudo systemctl enable --now rkllama
```

### QMD Model Server (on host)

```ini
# ~/.config/systemd/user/qmd-serve.service
[Unit]
Description=QMD Model Server (shared embeddings/reranking/expansion)
After=network.target rkllama.service

[Service]
Type=simple
ExecStart=/usr/bin/node /home/USER/qmd-server/dist/cli/qmd.js serve --port 7832 --bind 0.0.0.0
Restart=on-failure
RestartSec=5
Environment=NODE_OPTIONS=--max-old-space-size=4096
Environment=RKLLAMA_URL=http://localhost:8080

[Install]
WantedBy=default.target
```

```sh
systemctl --user enable --now qmd-serve
loginctl enable-linger $USER  # persist after logout
```

### OpenClaw Gateway (in each container)

```sh
# As the agent user inside the container
openclaw gateway install  # creates systemd user service
openclaw gateway start
```

Add `QMD_SERVER` to the gateway service:
```sh
sed -i '/\[Service\]/a Environment=QMD_SERVER=http://HOST_TAILSCALE_IP:7832' \
  ~/.config/systemd/user/openclaw-gateway.service
systemctl --user daemon-reload
openclaw gateway restart
```

## Forks and Patches

| Component | Issue | Fix | Status |
|-----------|-------|-----|--------|
| QMD remote model serving | [#489](https://github.com/tobi/qmd/issues/489), [#490](https://github.com/tobi/qmd/issues/490) | [jaylfc/qmd@feat/remote-llm-provider](https://github.com/jaylfc/qmd/tree/feat/remote-llm-provider) | Working |
| OpenClaw Vulkan GPU for local embeddings | — | [openclaw/openclaw#60347](https://github.com/openclaw/openclaw/pull/60347) | PR open |
| OpenClaw models.json provider fallback | — | [openclaw/openclaw#60369](https://github.com/openclaw/openclaw/pull/60369) | PR open |

## RK3588 Hardware Acceleration

All three models now run on the RK3588 NPU via rkllama:

| Model | Backend | Size | Performance |
|-------|---------|------|-------------|
| qwen3-embedding-0.6b | NPU | 892MB | ~1.25s/chunk (was 3-5s CPU) |
| qwen3-reranker-0.6b | NPU | 892MB | ~2.2s/query (was 10-15s CPU) |
| qmd-query-expansion | NPU | 2.3GB | ~3.4s/query (was 60+s CPU) |

Models are converted to `.rkllm` format using RKLLM toolkit v1.2.3 with w8a8 quantization on an x86 machine. See [docs/rk3588-acceleration.md](docs/rk3588-acceleration.md) for full setup details.

## Resources

- [OpenClaw Documentation](https://docs.openclaw.ai)
- [QMD Repository](https://github.com/tobi/qmd)
- [OpenClaw Memory System](https://docs.openclaw.ai/concepts/memory)
- [RK3588 NPU - rknn-llm](https://github.com/airockchip/rknn-llm)
- [rkllama - Ollama alternative for Rockchip NPU](https://github.com/NotPunchnox/rkllama)

## License

MIT
