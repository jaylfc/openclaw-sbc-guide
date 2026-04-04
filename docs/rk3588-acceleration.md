# RK3588 Hardware Acceleration for QMD Models

## Overview

The Rockchip RK3588 SoC (found in Orange Pi 5 Plus, Rock 5B, Khadas Edge2, etc.) has three compute paths:

| Backend | Hardware | Compute | RAM Access |
|---------|----------|---------|------------|
| ARM CPU | 4x A76 + 4x A55 | NEON SIMD | Shared 16GB |
| Mali-G610 GPU | 5 shader cores | Vulkan 1.1 | Shared 16GB (UMA) |
| NPU | 3 cores | INT8/INT16 | 6 TOPS, shared RAM |

## Current Status: All Models on NPU

All three models now run on the RK3588 NPU via rkllama:

- **qwen3-embedding-0.6b** (892MB, 1024 dims) — embedding model
- **qwen3-reranker-0.6b** (892MB) — cross-encoder reranker
- **qmd-query-expansion** (2.3GB, fine-tuned from Qwen3-1.7B) — query expansion LLM

Runtime: RKLLM v1.2.3, RKNPU driver v0.9.8, NPU device `/dev/dri/renderD129`

## Benchmark Results

| Model | CPU (8-core) | NPU | Speedup |
|-------|-------------|-----|---------|
| qwen3-embedding-0.6b | 3-5s / chunk | ~1.25s / chunk | ~3x |
| qwen3-reranker-0.6b | 10-15s / query | ~2.2s / query | ~6x |
| qmd-query-expansion | 60+s / query | ~3.4s / query | ~18x |

Query expansion sees the biggest gain because the CPU version was bottlenecked on a 2.3GB model with slow context prefill. The NPU handles this well with w8a8 quantization.

## Model Conversion

### Requirements

- **x86 machine** (RKLLM toolkit does not run on ARM)
- Python 3.10+ with pip
- RKLLM toolkit v1.2.3: `pip install rkllm-toolkit==1.2.3`

> **Version pinning is critical.** Pre-converted `.rkllm` models from HuggingFace may have been built with a different toolkit version and will fail to load or produce garbage output. Always convert with the toolkit version that matches your runtime (v1.2.3 for RKLLM runtime v1.2.3).

### Conversion Script

Run this on your x86 machine (tested on Fedora):

```python
from rkllm.api import RKLLM

# Example: convert qwen3-embedding-0.6b
llm = RKLLM()
llm.load_huggingface(model='Qwen/Qwen3-Embedding-0.6B')
llm.build(
    do_quantization=True,
    quantized_dtype='w8a8',
    target_platform='rk3588',
)
llm.export_rkllm('./qwen3-embedding-0.6b.rkllm')
```

Repeat for each model, substituting the HuggingFace model ID or a local path for fine-tuned models.

### Quantization

Only **w8a8** (8-bit weights, 8-bit activations) is supported for NPU inference. This is applied automatically when `quantized_dtype='w8a8'` is set. The resulting `.rkllm` files are roughly half the size of the original bf16 weights.

### Transferring to the SBC

```sh
scp qwen3-embedding-0.6b.rkllm orangepi:~/.rkllama/models/
scp qwen3-reranker-0.6b.rkllm orangepi:~/.rkllama/models/
scp qmd-query-expansion.rkllm orangepi:~/.rkllama/models/
```

## rkllama Setup

### Installation

```sh
git clone https://github.com/NotPunchnox/rkllama.git ~/rkllama
cd ~/rkllama && go build -o rkllama .
sudo cp rkllama /usr/local/bin/
```

### Model Directory Layout

rkllama expects each model to have a `Modelfile` alongside the `.rkllm` binary:

```
~/.rkllama/models/
├── qwen3-embedding-0.6b/
│   ├── qwen3-embedding-0.6b.rkllm
│   └── Modelfile
├── qwen3-reranker-0.6b/
│   ├── qwen3-reranker-0.6b.rkllm
│   └── Modelfile
└── qmd-query-expansion/
    ├── qmd-query-expansion.rkllm
    └── Modelfile
```

See `configs/Modelfile.embedding`, `configs/Modelfile.reranker`, and `configs/Modelfile.query-expansion` for templates.

### Modelfile Format

The Modelfile tells rkllama how to load and run the model:

```
# Minimal Modelfile example
FROM ./model-name.rkllm

PARAMETER num_npu_core 3
PARAMETER max_context_len 4096
PARAMETER max_new_tokens 512
```

Key parameters:
- `num_npu_core` — use all 3 NPU cores for maximum throughput
- `max_context_len` — set to match the model's training context length
- `max_new_tokens` — cap generation length (embedding/reranker don't generate, set low)

### Serving

```sh
# Start rkllama (Ollama-compatible API on port 8080)
rkllama serve --port 8080 --bind 0.0.0.0
```

Load a model:

```sh
curl http://localhost:8080/api/pull -d '{"name": "qwen3-embedding-0.6b"}'
```

Test embedding:

```sh
curl http://localhost:8080/api/embeddings -d '{
  "model": "qwen3-embedding-0.6b",
  "prompt": "Hello world"
}'
```

### Systemd Service

See `configs/rkllama.service`. Install as a system service (requires access to `/dev/dri/renderD129`):

```sh
sudo cp configs/rkllama.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now rkllama
```

## Hardware Notes

### NPU Device

The NPU is exposed at `/dev/dri/renderD129`. The rkllama process needs read/write access:

```sh
# Add your user to the render group, or run rkllama as root/system service
sudo usermod -aG render $USER
```

### CMA (Contiguous Memory Allocator)

The CMA pool was increased to 2GB in `/boot/armbianEnv.txt` for Vulkan fallback:

```
extraargs=cma=2048M
```

The NPU itself does not require large CMA allocations — it uses shared DRAM directly. The 2GB setting is only needed if you also want Vulkan (Mali-G610) to load a model without `ErrorOutOfDeviceMemory`.

### RKLLM Runtime Version

RKLLM runtime v1.2.3 with RKNPU driver v0.9.8. Check your installed version:

```sh
# Check driver version
cat /sys/kernel/debug/rknpu/version 2>/dev/null || dmesg | grep rknpu
```

If your driver version doesn't match, you may need to update the kernel or BSP.

## Vulkan Path (Not Used)

The Mali-G610 Vulkan driver (panvk/Mesa) has known limitations that make it unsuitable for running multiple models:

- No `VK_EXT_memory_budget` extension — llama.cpp can't query available VRAM
- Contiguous allocation limits — even with CMA bumped to 2GB, loading multiple models causes `ErrorOutOfDeviceMemory`
- Only a single small model (e.g., the 892MB embedding model) can load reliably

Since all three models now fit on the NPU and perform well there, Vulkan is not used.

## Resources

- [rkllama](https://github.com/NotPunchnox/rkllama)
- [airockchip/rknn-llm](https://github.com/airockchip/rknn-llm)
- [RKLLM toolkit docs](https://github.com/airockchip/rknn-llm/tree/main/rkllm-toolkit)
- [panvk Mesa driver](https://gitlab.freedesktop.org/mesa/mesa/-/tree/main/src/panfrost/vulkan)
- [RK3588 NPU benchmarks](https://tinycomputers.io/posts/rockchip-rk3588-npu-benchmarks.html)
