# SPDX-License-Identifier: Apache-2.0
"""单策略 trace 回放实验：一个进程跑一个 policy（进程隔离避免跨策略缓存污染）。

回放 trace.json 的三相位事件流（pre 请求 → update → post 请求），
四个策略对比：

    norefresh       被动 chunk cache：更新后不做任何处理
    passive         只做 ② 差集失效（refresh_article）
    proactive       ② + ③ 空闲窗口后台盲编码新增 chunk（完整方案）
    proactive_full  无增量基线：清空整篇 + 后台全量重算

每请求指标（请求发出前计量缓存状态 + prefill 墙钟）：
    prompt_len, doc_total_chunks, doc_hit_chunks, doc_miss_tokens
    (phase, seq) 分组：pre/0 = 冷填充, pre/1 = 全命中基线,
                      post/0 = 更新后首请求（headline）, post/1 = 后续请求
质量指标：pre 阶段 F1/EM（期望 gold）；post 阶段 flip 率（期望新实体）

用法（每策略独立进程，最后汇总）：
    python run_trace_experiment.py --policy proactive --trace traces/trace.json
    python run_trace_experiment.py --policy passive  --trace traces/trace.json
    ...
    python run_trace_experiment.py --summarize traces/      # 打印四策略对比表
"""

# Standard
import argparse
import glob
import json
import logging
import os
import sys
import time

logger = logging.getLogger(__name__)

# First Party
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Third Party
from transformers import AutoTokenizer
from vllm import SamplingParams

# First Party
from blend_mfo import build_llm_with_lmcache, setup_environment_variables
from eval_utils import exact_match, f1_score, flip_ok
from lmcache.integration.vllm.utils import ENGINE_NAME
from lmcache.v1.cache_engine import LMCacheEngineBuilder
from lmcache.v1.proactive_refresh import ProactiveRefreshManager
from lmcache.v1.token_database import UrlSegmentTokenDatabase

POLICIES = ["norefresh", "passive", "proactive", "proactive_full"]
# 与 trace_builder.FAKE_ENTITY 保持一致（运行环境可能没有 datasets，
# 不 import trace_builder，避免把构建期依赖带进回放期）
FAKE_ENTITY = "Zyxoria"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=POLICIES, default=None)
    parser.add_argument("--trace", default=None, help="trace.json 路径")
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--out-dir", default="traces")
    parser.add_argument("--bg-budget", type=int, default=4)
    parser.add_argument(
        "--order", default="hot", choices=["hot", "fifo", "oracle"],
        help="刷新排序策略（论文表 2）: hot=热度贪心(默认), fifo=先来先做, "
             "oracle=离线知晓轨迹的上界",
    )
    parser.add_argument(
        "--no-drain-on-update", action="store_true",
        help="phase 模式专用：update 事件后不立即刷新（让积压累积），"
             "改为在首个 post 请求前一次性按 budget 刷新——让排序策略有区分度",
    )
    parser.add_argument(
        "--summarize", default=None,
        help="汇总模式：给出结果目录，打印所有 results_*.json 的对比表",
    )
    return parser.parse_args()


def get_engine(llm):
    """获取 LMCache engine；若 connector 尚未构建，先跑一次微型请求触发。"""
    engine = LMCacheEngineBuilder.get(ENGINE_NAME)
    if engine is None:
        llm.generate(prompt_token_ids=[1], sampling_params=SamplingParams(max_tokens=1))
        engine = LMCacheEngineBuilder.get(ENGINE_NAME)
    assert engine is not None, "LMCache engine not found"
    assert isinstance(engine.token_database, UrlSegmentTokenDatabase), (
        "本实验需要 UrlSegmentTokenDatabase（--trace 指向 trace_builder 的输出目录）"
    )
    return engine


def chunk_in_storage(engine, chunk_hash):
    """检查一个 chunk 的 KV 是否已在存储中（layerwise 需逐层检查）。"""
    key = engine.token_database._make_key_by_hash(chunk_hash)
    if engine.use_layerwise:
        return all(engine.storage_manager.contains(k) for k in key.split_layers(engine.num_layers))
    return engine.storage_manager.contains(key)


def build_prompt(tokenizer, sys_prompt, urls, current_docs, question):
    """[BOS+sys][去BOS 文档们][去BOS 问题]，与盲编码 token 形态严格对齐。"""
    ids = tokenizer.encode(sys_prompt)
    for u in urls:
        ids = ids + tokenizer.encode(current_docs[u])[1:]
    ids = ids + tokenizer.encode(question)[1:]
    return ids


def run_policy(policy, trace, llm, tokenizer, bg_budget, order="hot", drain_on_update=True):
    """回放整条事件流，返回结果 dict。"""
    sys_prompt = trace["sys_prompt"]
    current_docs = dict(trace["docs"])

    engine = get_engine(llm)
    sp_warm = SamplingParams(temperature=0, max_tokens=1)

    def encode_fn(token_ids):
        llm.generate(prompt_token_ids=token_ids, sampling_params=sp_warm)

    manager = ProactiveRefreshManager(
        engine, encode_fn, budget_per_round=bg_budget, order_policy=order
    )
    if order == "oracle":
        # oracle 排序：按每篇文档首个 post 请求在轨迹中的位置升序
        #（越早被请求越先刷新）——"完美预测"的可达上界（表 2）
        ranking = {}
        for i, ev in enumerate(trace["events"]):
            if ev["type"] == "request" and ev["phase"] == "post":
                for u in ev["urls"]:
                    ranking.setdefault(u, i)
        manager.set_oracle_ranking(ranking)

    rows = []
    bg = {"tokens": 0, "chunks": 0, "wall_s": 0.0}

    def drain_background():
        """proactive 策略：事件间隙（空闲窗口）执行排队的后台刷新。"""
        if policy != "proactive":
            return
        for job in manager.refresh_pending(force=False):
            bg["tokens"] += job.recomputed_tokens
            bg["chunks"] += job.n_chunks
            bg["wall_s"] += job.wall_time

    seq_in_phase = {}
    prev_phase = None
    for ev in trace["events"]:
        if ev["type"] == "request":
            prompt = build_prompt(
                tokenizer, sys_prompt, ev["urls"], current_docs, ev["question"]
            )
            # 请求前计量缓存状态
            total_chunks = hit_chunks = 0
            miss_tokens = 0
            for u in ev["urls"]:
                for chunk_tokens, h in engine.token_database.compute_refresh_chunks(
                    current_docs[u], include_bos_variant=False
                ):
                    total_chunks += 1
                    if chunk_in_storage(engine, h):
                        hit_chunks += 1
                    else:
                        miss_tokens += len(chunk_tokens)
            seq = seq_in_phase.get((ev["qid"], ev["phase"]), 0)
            seq_in_phase[(ev["qid"], ev["phase"])] = seq + 1

            # phase 模式：首个 post 请求前一次性 drain budget 篇文档。
            # 三排序在有限 budget 下决定先刷哪些文档→被刷的 post/0 命中、
            # 没刷的 miss→产生可测区分度（表 2 最小可行并发验证）。
            # 用整篇 generate（非逐块）——诊断实验证实逐块 generate 的 KV
            # 不被请求路径 vLLM retrieve 认（两口径独立），整篇 generate 的 KV 被认。
            if not drain_on_update and ev["phase"] == "post" and prev_phase != "post":
                target = bg_budget
                with manager._lock:
                    if not manager._pending:
                        candidates = []
                    elif manager.order_policy == "fifo":
                        candidates = sorted(manager._pending.values(),
                            key=lambda j: j.enqueued_at)[:target]
                    elif manager.order_policy == "oracle":
                        candidates = sorted(manager._pending.values(),
                            key=lambda j: manager._oracle_ranking.get(
                                j.url, float("inf")))[:target]
                    else:  # hot
                        candidates = sorted(manager._pending.values(),
                            key=lambda j: (-j.hotness, j.enqueued_at))[:target]
                refreshed = 0
                for job in candidates:
                    doc_ids = tokenizer.encode(job.new_text)[1:]
                    t0_bg = time.time()
                    encode_fn(doc_ids)
                    bg["tokens"] += len(doc_ids)
                    bg["chunks"] += 1
                    bg["wall_s"] += time.time() - t0_bg
                    refreshed += 1
                    with manager._lock:
                        manager._pending.pop(job.url, None)
                        manager.completed.append(job)
                logger.info(
                    "Post-phase drain (full-doc): refreshed %d/%d docs, "
                    "pending=%d, order=%s",
                    refreshed, target, manager.pending_count(), order,
                )

            sp = SamplingParams(temperature=0, max_tokens=1)
            t0 = time.time()
            llm.generate(prompt_token_ids=prompt, sampling_params=sp)
            prefill_s = time.time() - t0

            row = {
                "qid": ev["qid"], "phase": ev["phase"], "seq": seq,
                "prompt_len": len(prompt),
                "doc_total_chunks": total_chunks, "doc_hit_chunks": hit_chunks,
                "doc_miss_tokens": miss_tokens, "prefill_s": round(prefill_s, 4),
            }

            if ev["gen_answer"] or ev["phase"] == "pre":
                sp_ans = SamplingParams(temperature=0, max_tokens=64)
                t0 = time.time()
                outs = llm.generate(prompt_token_ids=prompt, sampling_params=sp_ans)
                row["answer_s"] = round(time.time() - t0, 4)
                answer = outs[0].outputs[0].text.strip()
                row["answer"] = answer
                if ev["expect"] == "gold":
                    row["em"] = exact_match(answer, ev["gold"])
                    row["f1"] = f1_score(answer, ev["gold"])
                else:  # post 阶段：期望翻转到注入实体
                    row["flip_ok"] = flip_ok(
                        answer, ev.get("fake_entity", FAKE_ENTITY)
                    )
            rows.append(row)
            # --no-drain-on-update 模式下 post 阶段内不再 drain：
            # budget 已在阶段开始时一次性消耗（只刷 budget 篇文档），
            # 三排序决定哪些文档被刷→被刷的 post/0 命中、没刷的 miss。
            if drain_on_update or ev["phase"] != "post":
                drain_background()
            prev_phase = ev["phase"]

        elif ev["type"] == "update":
            url, new_text = ev["url"], ev["new_text"]
            current_docs[url] = new_text  # 网页内容对所有策略都变了
            if policy == "norefresh":
                pass
            elif policy == "passive":
                # 只更新元数据（_url_variants），不删 storage 的 stale chunk。
                # stale 旧 KV 仍在，post/0 blend 命中没变部分（旧 KV），
                # blend_recompute_ratio 在线重算编辑差异；编辑部分 miss 由 vllm 算。
                # 隔离的是"后台预计算"的价值（proactive vs passive）。
                engine.token_database.refresh_article(url, new_text)
            elif policy == "proactive":
                manager.submit_update(url, new_text)
            elif policy == "proactive_full":
                engine.remove_article(url)
                engine.refresh_article(url, new_text)
                # 整篇盲编码（非逐块）。诊断实验证实：逐块 generate
                # （compute_refresh_chunks）虽能让 chunk_in_storage 计量 miss=0，
                # 但请求路径 vLLM retrieve 不认逐块存的 KV（post/0 升至 0.225s 重算）；
                # 整篇 generate 的 KV 能被请求路径命中（post/0=0.095s 全命中），
                # 但 chunk_in_storage 逐层查口径不一致（miss=1529.7 计量偏差）。
                # 两口径独立，故保持整篇 generate（请求路径命中优先），miss 标为计量偏差。
                doc_ids = tokenizer.encode(new_text)[1:]
                t0 = time.time()
                encode_fn(doc_ids)
                bg["tokens"] += len(doc_ids)
                bg["chunks"] += 1
                bg["wall_s"] += time.time() - t0
            if drain_on_update:
                drain_background()
            prev_phase = "update"

    # ---------------- 汇总 ----------------
    def group_mean(phase, seq, field):
        vals = [r[field] for r in rows if r["phase"] == phase and r["seq"] == seq]
        return round(sum(vals) / len(vals), 4) if vals else None

    pre_f1 = [r["f1"] for r in rows if r.get("f1") is not None]
    post_flip = [r["flip_ok"] for r in rows if "flip_ok" in r]
    return {
        "policy": policy,
        "order_policy": order,
        "bg_budget": bg_budget,
        "bg": bg,
        "n_requests": len(rows),
        "summary": {
            "prefill_pre_0_s": group_mean("pre", 0, "prefill_s"),
            "prefill_pre_1_s": group_mean("pre", 1, "prefill_s"),
            "prefill_post_0_s": group_mean("post", 0, "prefill_s"),
            "prefill_post_1_s": group_mean("post", 1, "prefill_s"),
            "miss_tokens_post_0": group_mean("post", 0, "doc_miss_tokens"),
            "hit_ratio_pre_1": group_mean("pre", 1, "doc_hit_chunks"),
            "pre_f1": round(sum(pre_f1) / len(pre_f1), 4) if pre_f1 else None,
            "post_flip_rate": round(sum(post_flip) / len(post_flip), 4) if post_flip else None,
            # post/0 全命中率：被后台刷新到的文档比例（三排序区分度的核心指标）
            "post_0_full_hit_rate": round(
                sum(1 for r in rows if r["phase"] == "post" and r["seq"] == 0
                    and r["doc_total_chunks"] > 0
                    and r["doc_hit_chunks"] == r["doc_total_chunks"])
                / max(1, sum(1 for r in rows if r["phase"] == "post" and r["seq"] == 0)), 4
            ),
        },
        "requests": rows,
    }


def summarize(out_dir):
    """读取目录下所有 results_*.json，打印跨策略对比表。"""
    files = sorted(glob.glob(os.path.join(out_dir, "results_*.json")))
    assert files, f"no results_*.json under {out_dir}"
    print(f"{'policy/order':<24}{'bg_tok':>10}{'bg_s':>7}{'pre/0':>7}{'pre/1':>7}"
          f"{'post/0':>7}{'miss@0':>9}{'hit%':>6}{'F1':>6}{'flip':>6}")
    print("-" * 96)
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            r = json.load(f)
        s = r["summary"]
        name = f"{r['policy']}/{r.get('order_policy', 'hot')}"
        hit_rate = s.get("post_0_full_hit_rate", 0)
        print(
            f"{name:<24}{r['bg']['tokens']:>10}{r['bg']['wall_s']:>7.2f}"
            f"{fmt(s['prefill_pre_0_s']):>7}{fmt(s['prefill_pre_1_s']):>7}"
            f"{fmt(s['prefill_post_0_s']):>7}"
            f"{fmt(s['miss_tokens_post_0']):>9}{hit_rate*100:>5.0f}%"
            f"{fmt(s['pre_f1']):>6}{fmt(s['post_flip_rate']):>6}"
        )
    print("-" * 96)
    print("解读: post/0 列 = 更新后首请求延迟（proactive 应最快，norefresh 最慢）")
    print("      miss@post/0 = 挪到请求路径的重算 token（proactive 应为 0）")
    print("      bg_tokens: proactive vs proactive_full 之差 = ② 增量收益")
    print("      flip: post 请求答出更新实体的比例（新鲜度）")


def fmt(v, suffix=""):
    return f"{v:.2f}{suffix}" if isinstance(v, (int, float)) else "-"


def main():
    args = parse_args()
    if args.summarize:
        summarize(args.summarize)
        return

    assert args.policy and args.trace, "回放模式需要 --policy 和 --trace"

    with open(args.trace, "r", encoding="utf-8") as f:
        trace = json.load(f)

    trace_dir = os.path.dirname(os.path.abspath(args.trace))
    rag_db_path = os.path.join(trace_dir, "rag_db.json")
    setup_environment_variables(use_disk=False, rag_db_path=rag_db_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    with build_llm_with_lmcache("LMCacheConnectorV1", args.model) as llm:
        result = run_policy(
            args.policy, trace, llm, tokenizer, args.bg_budget, args.order,
            drain_on_update=not args.no_drain_on_update,
        )

    os.makedirs(args.out_dir, exist_ok=True)
    suffix = "" if args.order == "hot" else f"_{args.order}"
    out_path = os.path.join(args.out_dir, f"results_{args.policy}{suffix}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    s = result["summary"]
    print(f"\n===== {args.policy} =====")
    print(f"bg: {result['bg']['tokens']} tokens / {result['bg']['wall_s']:.2f}s")
    print(f"prefill pre/0={fmt(s['prefill_pre_0_s'])}s pre/1={fmt(s['prefill_pre_1_s'])}s "
          f"post/0={fmt(s['prefill_post_0_s'])}s post/1={fmt(s['prefill_post_1_s'])}s")
    print(f"miss_tokens@post/0={s['miss_tokens_post_0']} pre_F1={s['pre_f1']} "
          f"flip={s['post_flip_rate']}")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
