#!/bin/bash
# 最小可行并发验证：跑三排序（trace 已就绪），budget=32
source /root/autodl-tmp/mfo/.venv/bin/activate
cd /root/autodl-tmp/mfo/LMCache/examples/blend_kv_v1

LOG=/tmp/t2concurrent.log
exec > "$LOG" 2>&1

echo "=== TRACE INFO ==="
python3 -c "
import json
d = json.load(open('traces_t2_concurrent/trace.json'))
events = d['events']
from collections import Counter
phases = Counter(e['phase'] for e in events if e['type']=='request')
updates = sum(1 for e in events if e['type']=='update')
pre_counts = {}
for e in events:
    if e['type']=='request' and e['phase']=='pre':
        for u in e['urls']:
            pre_counts[u] = pre_counts.get(u, 0) + 1
counts = sorted(pre_counts.values(), reverse=True)
print(f'requests by phase: {dict(phases)}')
print(f'updates: {updates}, docs: {len(pre_counts)}')
print(f'hotness top 5: {counts[:5]}, bottom 5: {counts[-5:]}')
"

for order in hot fifo oracle; do
  echo ""
  echo "=== RUN order=$order budget=32 ==="
  VLLM_ENABLE_V1_MULTIPROCESSING=0 PYTHONHASHSEED=0 \
  HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  python -u run_trace_experiment.py \
    --policy proactive --order $order \
    --trace traces_t2_concurrent/trace.json \
    --model /root/autodl-tmp/mfo/models/Mistral-7B-Instruct-v0.2 \
    --out-dir traces_t2_concurrent \
    --bg-budget 32 --no-drain-on-update
  echo "=== order=$order DONE ==="
done

echo ""
echo "=== SUMMARIZE ==="
python run_trace_experiment.py --summarize traces_t2_concurrent/

echo ""
echo "ALL DONE rc=$?"
