# Examples vLLM + LMCache w. CacheBlend
LMCache should be able to reduce the generation time of the second and following calls (even though the reused KV cache is not a prefix).

## Some ad-hoc changes needed in vLLM
- In `vllm/vllm/v1/worker/gpu_worker.py`, comment out `ensure_kv_transfer_initialized(vllm_config)` in function `def init_worker_distributed_environment`.
- In the same file, add 
```
from lmcache.v1.compute.models.utils import VLLMModelTracker
from lmcache.integration.vllm.utils import ENGINE_NAME
        
VLLMModelTracker.register_model(ENGINE_NAME, self.model_runner.model)
ensure_kv_transfer_initialized(self.vllm_config)
```
at the end of the function `def load_model`.

## CPU offloading
- `python blend.py` - CacheBlend with CPU as backend
## Disk offloading
- `python blend.py --use-disk` - CachBlend with local disk as backend

## DocCache paper follow-up experiments

Run the commands below from the LMCache repository root in the original CUDA
experiment environment. Set `DOCCACHE_MODEL` to the same model checkpoint used for
the saved answers. Add `--limit 20` for a smoke run and write smoke results to
different paths.

```bash
export PYTHONHASHSEED=0
DOCCACHE_MODEL=/path/to/Mistral-7B-Instruct-v0.2
DOCCACHE_TRACE_DIR=paper/results/traces_t1_new

python -u examples/blend_kv_v1/run_cold_post_quality.py \
  --trace "$DOCCACHE_TRACE_DIR/trace.json" --model "$DOCCACHE_MODEL" \
  --cached-result "$DOCCACHE_TRACE_DIR/results_proactive.json" \
  --cached-result "$DOCCACHE_TRACE_DIR/results_proactive_full.json" \
  --out followup/cold_post_quality.json
```

The cold control uses the updated trace at `post/0` with both vLLM prefix reuse
and LMCache disabled. It pairs answers by question ID, checks stored prompt lengths
when available and scores all arms against the trace's updated reference. Verify
the original model, trace and generation settings for historical JSON files that
lack those fields. Use `--cold-result followup/cold_post_quality.json` with the
same arguments to rescore without GPU inference.

```bash
(
  for arm in incremental full; do
    python -u examples/blend_kv_v1/run_refresh_cost_comparison.py run \
      --strategy "$arm" --trace "$DOCCACHE_TRACE_DIR/trace.json" \
      --model "$DOCCACHE_MODEL" --out "followup/cost_${arm}.json" || exit 1
  done

  python examples/blend_kv_v1/run_refresh_cost_comparison.py compare \
    --incremental followup/cost_incremental.json --full followup/cost_full.json \
    --out followup/refresh_cost_comparison.json
)
```

Run each cost arm in a fresh process on the same GPU with equal settings and equal
limits. Use a single-document `--schedule block` trace. Each arm warms the old
document using the pre requests, times refresh from the update handler through
GPU synchronization, and measures the first updated request separately using
`max_tokens=1` as a TTFT proxy. Both arms use the served no-BOS document tokens.
Submitted token counts describe encoder inputs. Use measured refresh wall time
and request latency to evaluate the cost of each policy.
