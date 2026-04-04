# Troubleshooting

## Embedding Issues

### `qmd embed` hangs or crashes silently in LXC containers

**Cause:** When `QMD_SERVER` is set, QMD should use the remote server for embeddings. But older versions of the fork fall through to `getDefaultLlamaCpp()` which tries to build the Vulkan backend locally and crashes in containers without GPU access.

**Fix:** Update to the latest fork version which auto-detects `QMD_SERVER` in `getDefaultLLM()`.

### "Session expired — skipping remaining chunks"

**Cause:** The default LLM session timeout was 10 minutes. Large workspaces (100+ documents) take longer to embed on NPU.

**Fix:** Updated to 60-minute session timeout. If you still hit this, run `qmd embed` again — it retries only the remaining pending chunks.

### "Error rate too high — aborting embedding"

**Cause:** The error threshold was set to abort if >80% of chunks failed. NPU backends can have intermittent failures during model hot-swapping.

**Fix:** Updated threshold to 95% with a doubled minimum sample size. Individual chunks now retry up to 3 times with backoff.

### Partial embedding (some chunks still "Pending" after `qmd embed`)

**Cause:** Very large documents may exceed the per-request timeout (300s). The chunks are left as "pending" for the next run.

**Fix:** Run `qmd embed` again — it only processes pending chunks. OpenClaw's QMD integration also auto-retries on its 5-minute update cycle.

## Reranking Issues

### Reranker returns flat scores (all 0.2)

**Cause:** The `qmd serve` reranker was using text generation to simulate reranking instead of the proper logit-based endpoint.

**Fix:** Updated `RKLlamaBackend.rerank()` to use rkllama's native `/api/rerank` endpoint which extracts logit probabilities for "yes"/"no" tokens via softmax.

### Rerank times out

**Cause:** Each document requires a full forward pass through the reranker model. With 10+ documents, this can take several minutes on NPU.

**Fix:** The reranker only processes the top candidates from the initial BM25+vector search (typically 5-10 docs, not hundreds). Ensure the search pipeline is filtering results before reranking.

## rkllama Issues

### "invalid rkllm model!"

**Cause:** The `.rkllm` model file was converted with a different RKLLM toolkit version than the runtime. Pre-converted models from HuggingFace often have this problem.

**Fix:** Convert models yourself using the RKLLM toolkit version that matches your runtime. Check runtime version: `dmesg | grep rkllm` or look at rkllama startup logs. Convert with the matching toolkit: `pip install rkllm-toolkit==1.2.3`.

### "can't request region for resource [mem 0xfdab0000-0xfdabffff]"

**Cause:** This appears in `dmesg` during boot. It's a non-fatal warning about NPU memory regions being claimed by another driver (often Scrypted's RKNN plugin).

**Fix:** No fix needed — the NPU still works. The DRM device at `/dev/dri/renderD129` is functional despite these warnings.

### rkllama blocks all requests while one model is processing

**Cause:** The upstream rkllama uses a single global lock for all model operations.

**Fix:** Use the fork at `jaylfc/rkllama` branch `feat/rerank-logprobs-endpoint` which implements per-model locking. Requests to different models run in parallel.

## Container Networking

### LXC containers can't reach the host's LAN IP

**Cause:** macvlan networking isolates host-to-container traffic on the same interface. The host IP is unreachable from inside the container even though they're on the same network.

**Fix:** Use Tailscale. Install it on both the host and each container. Use the host's Tailscale IP (e.g., `100.78.225.80`) for `QMD_SERVER` instead of the LAN IP.

### Container loses Tailscale connection after host reboot

**Cause:** Tailscale in the container may not auto-start, or the container itself may not auto-start.

**Fix:** Enable Tailscale service in the container: `sudo systemctl enable tailscaled`. Set containers to auto-start in Incus: `sudo incus config set CONTAINER boot.autostart=true`.

## Performance

### Embedding is slow (>5s per chunk)

**Expected:** ~1.25s per chunk on RK3588 NPU with Qwen3-Embedding-0.6B.

**Check:**
1. Is rkllama running? `curl http://localhost:8080/api/tags`
2. Is the embedding model preloaded? Use `--preload` flag
3. Is the NPU driver loaded? `dmesg | grep rknpu`
4. Is another model hogging the NPU? Check with `systemctl --user status rkllama`

### Query expansion takes >10 seconds

**Expected:** ~3.4s on NPU for the 1.7B model.

**Check:** The query expansion model (2.3GB) may need to be loaded into NPU memory. First call is slow (~5-10s for model load), subsequent calls should be ~3.4s.
