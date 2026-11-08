#!/bin/bash
export PATH=/root/miniconda3/bin:$PATH
source /root/autodl-tmp/mfo/.venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
cd /root/autodl-tmp/mfo/LMCache/examples/blend_kv_v1
MODEL=/root/autodl-tmp/mfo/models/Mistral-7B-Instruct-v0.2

cleanup() { pkill -9 -f run_trace_experiment.py 2>/dev/null; sleep 3; }
run1() { cleanup; echo "### $1 $2 START $(date) ###"; python -u run_trace_experiment.py --policy $1 --trace $2/trace.json --model $MODEL --out-dir $2/ $3; echo "### $1 $2 DONE rc=$? $(date) ###"; }

echo "########## TABLE1 四策略 N=200 block ##########"
echo "### TRACE BUILD $(date) ###"
python -u trace_builder.py --num-questions 200 --schedule block --out-dir traces_t1_new/
echo "### TRACES DONE $(date) ###"
run1 norefresh traces_t1_new
run1 passive traces_t1_new
run1 proactive traces_t1_new
run1 proactive_full traces_t1_new
python -u run_trace_experiment.py --summarize traces_t1_new/
echo "########## TABLE1 DONE $(date) ##########"

echo "########## TABLE2 三排序 N=50 phase --no-drain-on-update bg-budget=1 ##########"
python -u trace_builder.py --num-questions 50 --request-skew 1.0 --schedule phase --out-dir traces_t2_new/
for order in fifo hot oracle; do
  run1 proactive traces_t2_new "--order $order --bg-budget 1 --no-drain-on-update"
done
python -u run_trace_experiment.py --summarize traces_t2_new/
echo "########## TABLE2 DONE $(date) ##########"

echo "########## MOTIVATION passive 臂 ##########"
cleanup
echo "### motivation passive START $(date) ###"
python -u refresh_experiment.py --model $MODEL --arm passive --out refresh_passive.json
echo "### motivation passive DONE rc=$? $(date) ###"
echo "########## ALL SUPPLEMENT DONE $(date) ##########"
