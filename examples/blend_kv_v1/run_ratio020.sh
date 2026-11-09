#!/bin/bash
# blend ratio=0.20 消融：proactive 策略，对比 ratio=0.15 基线（traces_t1_new/results_proactive.json）
#
# 环境：autodl RTX4090 容器
#   - venv:  /root/autodl-tmp/mfo/.venv
#   - model: /root/autodl-tmp/mfo/models/Mistral-7B-Instruct-v0.2
#   - repo:  /root/autodl-tmp/mfo/LMCache
# 前置：traces_t1_new/{trace.json,rag_db.json} 已存在（由 run_full.sh 生成）
# 产出：traces_t1_ratio020/results_proactive.json
# 论文：§5.5b (b'') ratio 敏感性消融
source /root/autodl-tmp/mfo/.venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
# 关键：blend 加载期重算比例 0.15 -> 0.20（blender.py 通过此环境变量读取）
export LMCACHE_BLEND_RECOMPUTE_RATIO=0.2

cd /root/autodl-tmp/mfo/LMCache/examples/blend_kv_v1
MODEL=/root/autodl-tmp/mfo/models/Mistral-7B-Instruct-v0.2

mkdir -p traces_t1_ratio020
cp traces_t1_new/trace.json traces_t1_ratio020/
cp traces_t1_new/rag_db.json traces_t1_ratio020/

pkill -9 -f "run_trace_experiment.py" 2>/dev/null
pkill -9 -f "EngineCore" 2>/dev/null
sleep 5

echo "=== RATIO=0.20 PROACTIVE START $(date) ==="
echo "LMCACHE_BLEND_RECOMPUTE_RATIO=$LMCACHE_BLEND_RECOMPUTE_RATIO"
python -u run_trace_experiment.py --policy proactive \
  --trace traces_t1_ratio020/trace.json \
  --model $MODEL --out-dir traces_t1_ratio020/
echo "=== DONE rc=$? $(date) ==="
echo "=== SUMMARY ==="
python -u run_trace_experiment.py --summarize traces_t1_ratio020/
echo "=== RATIO020 ALL DONE ==="
