#!/bin/bash
# 跑 fifo + oracle（hot 已完成），budget=5, K=10 小批量排布
source /root/autodl-tmp/mfo/.venv/bin/activate
cd /root/autodl-tmp/mfo/LMCache/examples/blend_kv_v1

LOG=/tmp/t2fifo_oracle.log
exec > "$LOG" 2>&1

for order in fifo oracle; do
  echo ""
  echo "=== RUN order=$order budget=5 (K=10 batch) ==="
  VLLM_ENABLE_V1_MULTIPROCESSING=0 PYTHONHASHSEED=0 \
  HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  python -u run_trace_experiment.py \
    --policy proactive --order $order \
    --trace traces_t2_concurrent/trace.json \
    --model /root/autodl-tmp/mfo/models/Mistral-7B-Instruct-v0.2 \
    --out-dir traces_t2_concurrent \
    --bg-budget 5 --no-drain-on-update
  echo "=== order=$order DONE ==="
done

echo ""
echo "=== SUMMARIZE ==="
python run_trace_experiment.py --summarize traces_t2_concurrent/

echo ""
echo "ALL DONE rc=$?"
