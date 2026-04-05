# Benchmark Results: NPU vs Standard QMD

## Test Setup

- **Fedora Baseline**: x86_64, 77GB RAM, NVIDIA 3060 12GB, upstream QMD with embeddinggemma-300M (768d), qwen3-reranker-0.6B, qmd-query-expansion-1.7B via node-llama-cpp
- **Orange Pi NPU**: RK3588, 16GB RAM, NPU 6 TOPS, Qwen3-Embedding-0.6B (1024d), Qwen3-Reranker-0.6B, qmd-query-expansion-1.7B via rkllama
- **Test corpus**: 8 markdown documents covering 3D printing, Kubernetes, weather, Orange Pi, ML, and cooking

## BM25 Search (Keyword)

BM25 scores are identical since both use the same SQLite FTS5 engine:

| Query | Top Result | Score |
|-------|-----------|-------|
| "3d printing filament" | 3d-printing-materials.md | 0.83 |
| "kubernetes networking" | kubernetes-networking.md | 0.74 |
| "edge AI single board" | orange-pi-5-plus.md | 0.83 |
| "neural network" | machine-learning-basics.md | 0.63 |
| "fresh pasta" | italian-cooking.md | 0.73 |

## Vector Search (Semantic)

| Query | Fedora (768d) | NPU (1024d) | Ranking Match |
|-------|--------------|-------------|---------------|
| "best material for first 3d printer" | 0.80 3d-printing | 0.79 3d-printing | Yes |
| "how do containers communicate in k8s" | 0.68 kubernetes | 0.82 kubernetes | Yes |
| "AI hardware for edge computing" | 0.62 orange-pi | 0.76 orange-pi* | Yes |

*NPU embedding model (Qwen3) produces higher absolute scores than embeddinggemma but ranking order is correct.

### Speed

| Operation | Fedora (x86 + 3060) | Orange Pi NPU | Notes |
|-----------|--------------------:|---------------:|-------|
| Vector search | 4-5s | 34-51s (before KV fix) → ~5-10s (after) | Expand is the bottleneck |
| Hybrid query (no rerank) | ~5s | ~39s | Mostly expand time |
| Hybrid query (with rerank) | 6-7s | ~336s (8 chunks) | ~20s per doc reranking |
| Single embedding | <0.1s | 0.3-0.4s | After KV cache fix |

## Hybrid Query (with Reranking)

| Query | Fedora Score | Fedora Time | NPU Time |
|-------|-------------|-------------|----------|
| "beginner 3d printing materials" | 0.93 | 7s | ~336s |
| "container networking k8s" | 0.93 | 6s | TBD |

Reranking is the primary speed bottleneck on NPU. Each document requires a full forward pass through the reranker model (~20s per doc on NPU vs <1s on x86).

## Key Findings

1. **BM25 is identical** - same engine, same scores
2. **Vector search ranking is correct** - NPU produces correct document ordering despite using a different embedding model (Qwen3 1024d vs embeddinggemma 768d)
3. **Query expansion works on NPU** - produces structured lex/vec/hyde output matching the fine-tuned model's training
4. **Reranking is slow but accurate** - logit-based NPU scoring produces proper relevance scores (0.86 for relevant, 0.02 for irrelevant)
5. **KV cache clearing is critical** - without it, embedding speed degrades from 0.3s to 26s+ over hours
6. **Score ranges differ** - NPU embeddings produce higher absolute cosine similarities (0.75-0.82 range) vs Fedora (0.53-0.62). Ranking order is preserved.

## Recommendations for SBC Deployment

- **Use `--no-rerank` for real-time search** where <5s response time is needed
- **Enable reranking for batch/background operations** (agent memory search) where latency is acceptable
- **Restart rkllama every 2 hours** (RuntimeMaxSec=7200) to prevent KV cache degradation
- **Run `qmd embed` multiple times** for large workspaces - the retry logic handles intermittent NPU failures
