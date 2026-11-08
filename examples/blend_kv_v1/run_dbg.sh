#!/bin/bash
export PATH=/root/miniconda3/bin:$PATH
source /root/autodl-tmp/mfo/.venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
cd /root/autodl-tmp/mfo/LMCache/examples/blend_kv_v1
MODEL=/root/autodl-tmp/mfo/models/Mistral-7B-Instruct-v0.2
echo "### DBG START $(date) VLLM_MP=$VLLM_ENABLE_V1_MULTIPROCESSING ###"
for p in norefresh passive proactive proactive_full; do
  FREE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  echo "### pre-$p gpu_used=${FREE}MiB $(date) ###"
  pkill -9 -f "run_trace_experiment.py" 2>/dev/null; pkill -9 -f "EngineCore" 2>/dev/null
  sleep 3
  echo "### $p START $(date) ###"
  python -u run_trace_experiment.py --policy $p --trace traces_dbg/trace.json --model $MODEL --out-dir traces_dbg/
  echo "### $p DONE rc=$? $(date) ###"
  sleep 5
done
echo "### SUMMARIZE $(date) ###"
python -u run_trace_experiment.py --summarize traces_dbg/
echo "### ALL DONE $(date) ###"
