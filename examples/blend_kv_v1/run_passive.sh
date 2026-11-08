#!/bin/bash
export PATH=/root/miniconda3/bin:$PATH
source /root/autodl-tmp/mfo/.venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
cd /root/autodl-tmp/mfo/LMCache/examples/blend_kv_v1
pkill -9 -f "run_trace_experiment.py" 2>/dev/null; sleep 3
echo "### passive START $(date) ###"
python -u run_trace_experiment.py --policy passive --trace traces_dbg/trace.json \
  --model /root/autodl-tmp/mfo/models/Mistral-7B-Instruct-v0.2 --out-dir traces_dbg/
echo "### passive DONE rc=$? $(date) ###"
