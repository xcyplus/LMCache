#!/bin/bash
export PATH=/root/miniconda3/bin:$PATH
source /root/autodl-tmp/mfo/.venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
cd /root/autodl-tmp/mfo/LMCache/examples/blend_kv_v1
MODEL=/root/autodl-tmp/mfo/models/Mistral-7B-Instruct-v0.2
N=200

cleanup() { pkill -9 -f "run_trace_experiment.py" 2>/dev/null; pkill -9 -f "EngineCore" 2>/dev/null; sleep 5; }
run1() { cleanup; echo "### $1 $2 START $(date) ###"; python -u run_trace_experiment.py --policy $1 --trace $2/trace.json --model $MODEL --out-dir $2/ $3; echo "### $1 $2 DONE rc=$? $(date) ###"; }

echo "########## TRACE BUILD $(date) ##########"
python -u trace_builder.py --num-questions $N --schedule block --out-dir traces_t1/
python -u trace_builder.py --num-questions $N --request-skew 1.0 --schedule phase --out-dir traces_t2/
python -u trace_builder.py --num-questions $N --edit-mode tail --model $MODEL --out-dir traces_t3_tail/
python -u trace_builder.py --num-questions $N --edit-mode mid_iso --model $MODEL --out-dir traces_t3_iso/
python -u trace_builder.py --num-questions $N --edit-mode mid_len --model $MODEL --out-dir traces_t3_len/
echo "########## TRACES DONE $(date) ##########"

echo "########## TABLE1 (norefresh/proactive/proactive_full) $(date) ##########"
run1 norefresh traces_t1
run1 proactive traces_t1
run1 proactive_full traces_t1
python -u run_trace_experiment.py --summarize traces_t1/
echo "########## TABLE1 DONE $(date) ##########"

echo "########## TABLE2 (fifo/hot/oracle, phase, bg-budget=1) $(date) ##########"
run1 proactive traces_t2 "--order fifo --bg-budget 1"
run1 proactive traces_t2 "--order hot --bg-budget 1"
run1 proactive traces_t2 "--order oracle --bg-budget 1"
python -u run_trace_experiment.py --summarize traces_t2/
echo "########## TABLE2 DONE $(date) ##########"

echo "########## TABLE3 (tail/mid_iso/mid_len 编辑消融) $(date) ##########"
run1 proactive traces_t3_tail
run1 proactive traces_t3_iso
run1 proactive traces_t3_len
echo "########## TABLE3 DONE $(date) ##########"

echo "########## ALL DONE $(date) ##########"
