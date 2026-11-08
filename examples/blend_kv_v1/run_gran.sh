#!/bin/bash
export PATH=/root/miniconda3/bin:$PATH
source /root/autodl-tmp/mfo/.venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
cd /root/autodl-tmp/mfo/LMCache/examples/blend_kv_v1
MODEL=/root/autodl-tmp/mfo/models/Mistral-7B-Instruct-v0.2
echo "### GRAN START $(date) ###"
echo "--- question N=50 ---"
python -u trace_builder.py --num-questions 50 --granularity question --out-dir traces_gq/ 2>&1 | tail -2
pkill -9 -f run_trace_experiment.py 2>/dev/null; sleep 3
python -u run_trace_experiment.py --policy proactive --trace traces_gq/trace.json --model $MODEL --out-dir traces_gq/ 2>&1 | tail -5
echo "### question DONE rc=$? $(date) ###"
echo "--- paragraph N=50 ---"
python -u trace_builder.py --num-questions 50 --granularity paragraph --out-dir traces_gp/ 2>&1 | tail -2
pkill -9 -f run_trace_experiment.py 2>/dev/null; sleep 3
python -u run_trace_experiment.py --policy proactive --trace traces_gp/trace.json --model $MODEL --out-dir traces_gp/ 2>&1 | tail -5
echo "### paragraph DONE rc=$? $(date) ###"
echo "### GRAN ALL DONE $(date) ###"
