# RK3588 Hardware Acceleration for QMD Models

## Overview

The Rockchip RK3588 SoC (found in Orange Pi 5 Plus, Rock 5B, Khadas Edge2, etc.) has three compute paths that could each run a different model:

| Backend | Hardware | Compute | RAM Access |
|---------|----------|---------|------------|
| ARM CPU | 4x A76 + 4x A55 | NEON SIMD | Shared 16GB |
| Mali-G610 GPU | 5 shader cores | Vulkan 1.1 | Shared 16GB (UMA) |
| NPU | 3 cores | INT8/INT16 | 6 TOPS, shared RAM |

## Current Status (CPU only)

All three QMD models currently run on CPU via node-llama-cpp:
- **embeddinggemma-300M** (~314MB) - embedding model
- **qwen3-reranker-0.6B** (~610MB) - cross-encoder reranker
- **qmd-query-expansion-1.7B** (~1.2GB) - query expansion LLM

Total RAM: ~2.1GB. Performance is functional but slow for batch operations.

## Planned: Multi-Backend Split

The idea is to spread models across all available hardware:

```
NPU (rkllama)     → embeddinggemma-300M  (most frequent, smallest)
Vulkan (Mali-G610) → qwen3-reranker-0.6B  (medium size, fits in single alloc)
CPU (ARM NEON)     → qmd-query-expansion-1.7B  (largest, least frequent)
```

### Why This Split

- **Embeddings** are called most frequently (every document chunk during indexing, every query). NPU INT8 would give the biggest throughput improvement.
- **Reranker** is called once per search with a small batch of candidates. Vulkan can handle a single ~610MB model.
- **Query expansion** is called once per search. It's the largest model but least latency-sensitive.

## Investigation: NPU Path

### rkllama (Ollama-compatible)

[rkllama](https://github.com/NotPunchnox/rkllama) provides an Ollama-compatible API for RK3588 NPU:

```sh
# Install
git clone https://github.com/NotPunchnox/rkllama.git
cd rkllama && go build -o rkllama

# Serve a model
./rkllama serve --model path/to/model.rkllm --port 11434
```

**Challenge:** Models must be converted to `.rkllm` format via the RKLLM-Toolkit. Only w8a8 quantisation is supported. The toolkit only converts generative LLMs - embedding models may not be supported.

### RKNN-LLM (official Rockchip)

[airockchip/rknn-llm](https://github.com/airockchip/rknn-llm) is the official runtime. The `rkllm_embed_input` API exists but it's for feeding token embeddings INTO a model, not for generating sentence embeddings FROM a model.

### Custom RKNN Pipeline

The most viable path for NPU embeddings may be:
1. Export embeddinggemma-300M to ONNX
2. Convert ONNX to RKNN via `rknn-toolkit2`
3. Write a small inference server that runs the RKNN model and exposes an HTTP endpoint
4. Modify `qmd serve` to use this endpoint for embeddings

This is non-trivial but would give genuine hardware acceleration.

## Investigation: Vulkan Path

### Current Issue

The Mali-G610 uses the `panvk` Vulkan driver (open-source, part of Mesa). Known limitations:
- No `VK_EXT_memory_budget` extension - llama.cpp can't query available VRAM
- Contiguous allocation limits - CMA pool defaults to 256MB
- Loading multiple models causes `ErrorOutOfDeviceMemory`

### Potential Fixes

1. **Increase CMA pool**: Add `cma=1024M` to `/boot/armbianEnv.txt` kernel args
2. **Single model per Vulkan context**: Load only the reranker on Vulkan, keep others on CPU
3. **Wait for Mesa improvements**: panvk is under active development

### Testing Single Model on Vulkan

```sh
# In qmd serve, only the reranker would use Vulkan
# This requires changes to the serve.ts code to support per-model backends
```

## Benchmarks (TODO)

| Model | CPU (8-core) | Vulkan (Mali-G610) | NPU | Notes |
|-------|-------------|-------------------|-----|-------|
| embed (300M, Q8) | ? ms/chunk | ? ms/chunk | N/A | |
| rerank (0.6B, Q8) | ? ms/query | ? ms/query | N/A | |
| expand (1.7B, Q4) | ? ms/query | N/A | N/A | |

## Resources

- [panvk Mesa driver](https://gitlab.freedesktop.org/mesa/mesa/-/tree/main/src/panfrost/vulkan)
- [RK3588 NPU benchmarks](https://tinycomputers.io/posts/rockchip-rk3588-npu-benchmarks.html)
- [node-llama-cpp Vulkan guide](https://node-llama-cpp.withcat.ai/guide/Vulkan)
- [rk-llama.cpp (GGML NPU backend)](https://github.com/invisiofficial/rk-llama.cpp)
