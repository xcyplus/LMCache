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

## On-demand document KV on SSD

Use the CacheBlend worker setup above with a single eager Llama worker and
standard, unscaled RoPE. Run from this example directory:

```sh
python blend_asyn_example.py --model /path/to/llama --document-kv-dir /mnt/ssd/document-kv
```

For hosted models, use `--revision` to pin the model weights. Use a new cache
directory when replacing a local checkpoint in place.

The example sends two requests with reversed document order and an uncached
separator. The worker logs whether each document was built or loaded. Run the
example again with the same model and directory to exercise persistence across
restarts. Only independent document KV is written to these SSD files. The
regular CPU backend stores non-document chunks from the request. Setting
`LMCACHE_DOCUMENT_KV_DIR` enables the worker feature for other applications
using the same document span registration IPC.
