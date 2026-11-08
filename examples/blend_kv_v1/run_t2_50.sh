#!/bin/bash
export PATH=/root/miniconda3/bin:$PATH
source /root/autodl-tmp/mfo/.venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
cd /root/autodl-tmp/mfo/LMCache/examples/blend_kv_v1
MODEL=/root/autodl-tmp/mfo/models/Mistral-7B-Instruct-v0.2
echo "### T2_50 START $(date) ###"
for order in fifo hot oracle; do
  pkill -9 -f run_trace_experiment.py 2>/dev/null; sleep 3
  echo "### $order START $(date) ###"
  python -u run_trace_experiment.py --policy proactive --order $order --bg-budget 1 \
    --no-drain-on-update --trace traces_t2_50/trace.json --model $MODEL --out-dir traces_t2_50/
  echo "### $order DONE rc=$? $(date) ###"
  sleep 3
done
python -u run_trace_experiment.py --summarize traces_t2_50/
echo "### T2_50 ALL DONE $(date) ###"
