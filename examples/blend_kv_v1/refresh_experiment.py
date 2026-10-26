# SPDX-License-Identifier: Apache-2.0
"""③ Proactive Refresh 消融实验：文档更新事件 × 请求流，四臂对比。

实验设计（论文 motivation 表的数据来源）：
  所有臂都经历同一件事——目标文档在 T 时刻被编辑（中部替换一个事实句，
  长度变化，会触发真实的级联失效），随后一个引用该文档的请求到达。
  四臂只差在"更新后、请求前"系统做了什么：

    norefresh       什么都不做。旧文本整段无法匹配 → 首请求全量重算该文档
                    （TTFT 最差，后台 0 成本）——被动 chunk cache 的现状。
    passive         只做 ② 差集失效（engine.refresh_article）。未变 chunk
                    保留命中，首请求只重算新增 chunk（TTFT 中等，后台 0）。
    proactive       ② + ③：差集失效后，在空闲窗口后台盲编码新增 chunk
                    （首请求全命中，后台只花 fresh chunk 的算力）。
    proactive_full  无增量基线：清空该文档后后台全量重算
                    （首请求全命中，后台花整篇文档的算力）。
                    proactive 与它的后台成本差 = ② 增量更新的收益。

指标：
    ttft_probe_s    更新后首个请求的墙钟时间（max_tokens=1，prefill 主导）
    answer_time_s   答案请求时间（max_tokens=64，此时缓存已收敛，各臂应接近）
    bg_tokens / bg_time_s   该臂后台重算的 token 数与墙钟时间
    fresh/stale     refresh_article 差集结果（chunk 数）
    flip_ok         答案是否包含更新后的事实（新鲜度 sanity check：
                    内容寻址保证"新内容→新 hash→miss"， staleness 表现为
                    时间代价而非错误答案，四臂都应为 True）

运行：
    cd examples/blend_kv_v1
    python refresh_experiment.py --model mistralai/Mistral-7B-Instruct-v0.2
    # 单臂调试： --arm proactive
"""

# Standard
import argparse
import json
import os
import sys
import time

# First Party
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Third Party
from transformers import AutoTokenizer
from vllm import SamplingParams

# First Party
from blend_mfo import build_llm_with_lmcache, setup_environment_variables
from lmcache.integration.vllm.utils import ENGINE_NAME
from lmcache.v1.cache_engine import LMCacheEngineBuilder
from lmcache.v1.proactive_refresh import ProactiveRefreshManager
from lmcache.v1.token_database import UrlSegmentTokenDatabase

TARGET_URL = "www.beijing.com"
# 中部替换式编辑（长度变化 → 编辑点之后的 chunk 边界滑移，级联失效是诚实的）
OLD_FACT = "截至2025年末常住人口2180.0万人。"
NEW_FACT = "根据2026年10月最新统计，全市常住人口已增长至2210.8万人，稳居全国城市第三位。"
QUESTION = "北京市现在的常住人口是多少？请直接给出数字。"
ARMS = ["norefresh", "passive", "proactive", "proactive_full"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="Path or name of the model to use",
    )
    parser.add_argument(
        "--rag-db",
        default=None,
        help="Path to the RAG database JSON (default: rag_db.json next to this script)",
    )
    parser.add_argument(
        "--arm",
        default="all",
        choices=ARMS + ["all"],
        help="Run a single arm or all four (default: all)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output JSON path (default: refresh_results.json next to this script)",
    )
    return parser.parse_args()


def timed_generate(llm, prompt, max_tokens):
    """一次 generate 的墙钟时间 + 文本（temperature=0 保证可复现）。"""
    sp = SamplingParams(temperature=0, max_tokens=max_tokens)
    t0 = time.time()
    outs = llm.generate(prompt_token_ids=prompt, sampling_params=sp)
    return time.time() - t0, outs[0].outputs[0].text


def reset_doc(engine, encode_fn, tokenizer, text):
    """把目标文档恢复为 text 并全量入缓存，得到干净的臂起点。

    三步 purge 保证跨臂零污染：
      1) refresh_article(url, 另一代文本) —— 差集失效驱逐当前代所有 chunk
      2) remove_article(url)              —— 驱逐可能在 storage 中的另一代残留
      3) refresh_article(url, text) + 盲编码 —— 目标文本全量入缓存
    """
    engine.refresh_article(TARGET_URL, text)
    engine.remove_article(TARGET_URL)
    engine.refresh_article(TARGET_URL, text)
    encode_fn(tokenizer.encode(text)[1:])


def main():
    args = parse_args()
    rag_db_path = args.rag_db or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "rag_db.json"
    )
    out_path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "refresh_results.json"
    )

    setup_environment_variables(use_disk=False, rag_db_path=rag_db_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    with open(rag_db_path, "r", encoding="utf-8") as f:
        rag = json.load(f)
    original_text = rag[TARGET_URL]
    assert OLD_FACT in original_text, "rag_db 中找不到待编辑的事实句"
    updated_text = original_text.replace(OLD_FACT, NEW_FACT)

    arms_to_run = ARMS if args.arm == "all" else [args.arm]

    with build_llm_with_lmcache("LMCacheConnectorV1", args.model) as llm:
        # 盲编码函数：standalone prefill，KV 经 connector store 路径按内容 hash 入库
        sp_warm = SamplingParams(temperature=0, max_tokens=1)

        def encode_fn(token_ids):
            llm.generate(prompt_token_ids=token_ids, sampling_params=sp_warm)

        # ---------------- 预热：初始语料全量盲编码 ----------------
        # 与请求时的 token 形态对齐：sys 带 BOS（首段），文档/问题去 BOS
        sys_ids = tokenizer.encode("You are a very helpful assistant.")
        question_ids = tokenizer.encode(QUESTION)[1:]
        doc_urls = [u for u in rag if not u.startswith(("sys://", "query://"))]

        t0 = time.time()
        for u in doc_urls:
            encode_fn(tokenizer.encode(rag[u])[1:])
        encode_fn(sys_ids)
        encode_fn(question_ids)
        warmup_time = time.time() - t0
        doc_len = len(tokenizer.encode(original_text)[1:])
        print(f"[warmup] {len(doc_urls)} docs blind-encoded in {warmup_time:.2f}s, "
              f"target doc = {doc_len} tokens")

        # 预热完成后 connector 必然已构建好 engine，此时获取
        engine = LMCacheEngineBuilder.get(ENGINE_NAME)
        assert engine is not None, "LMCache engine not found after warmup"
        assert isinstance(engine.token_database, UrlSegmentTokenDatabase), (
            "本实验需要 UrlSegmentTokenDatabase（--rag-db 指向 rag_db.json）"
        )
        manager = ProactiveRefreshManager(engine, encode_fn)

        # 探测 prompt：[sys][更新后的北京文本][问题]
        # 四臂的请求完全一致（网页已更新，用户看到的是新内容），差别只在缓存状态
        probe_prompt = (
            sys_ids + tokenizer.encode(updated_text)[1:] + question_ids
        )

        # 基线 sanity：旧文本全命中
        base_prompt = sys_ids + tokenizer.encode(original_text)[1:] + question_ids
        base_t, _ = timed_generate(llm, base_prompt, 1)
        print(f"[baseline] all-hit probe (original text): {base_t:.2f}s\n")

        # ---------------- 四臂主实验 ----------------
        results = {"warmup_time_s": warmup_time, "doc_tokens": doc_len, "arms": {}}
        for arm in arms_to_run:
            print(f"===== arm: {arm} =====")
            row = {"arm": arm}

            # 1) 干净起点：目标文档 = 旧文本，全量命中
            reset_doc(engine, encode_fn, tokenizer, original_text)

            # 2) 更新事件 + 各臂策略
            bg_t0 = time.time()
            if arm == "norefresh":
                # 什么都不做：旧缓存原样保留（chunk cache 的被动现状）
                pass
            elif arm == "passive":
                # 只做 ②：差集失效，未变 chunk 保留，新增 chunk 等请求 lazy 重算
                fresh, stale = engine.refresh_article(TARGET_URL, updated_text)
                row["fresh_chunks"], row["stale_chunks"] = len(fresh), len(stale)
            elif arm == "proactive":
                # ② + ③：差集失效 + 空闲窗口后台盲编码新增 chunk
                fresh, stale = manager.submit_update(TARGET_URL, updated_text)
                row["fresh_chunks"], row["stale_chunks"] = len(fresh), len(stale)
                # 演示空闲窗口策略：忙时跳过，空闲时执行
                manager.notify_request_begin()
                skipped = manager.refresh_pending(force=False)
                assert not skipped, "busy window should skip background refresh"
                manager.notify_request_end()
                jobs = manager.refresh_pending(force=False)  # 现在空闲，执行
                row["bg_tokens"] = sum(j.recomputed_tokens for j in jobs)
                row["bg_chunks"] = sum(j.n_chunks for j in jobs)
            elif arm == "proactive_full":
                # 无增量基线：整篇文档后台全量重算
                engine.remove_article(TARGET_URL)
                engine.refresh_article(TARGET_URL, updated_text)
                updated_ids = tokenizer.encode(updated_text)[1:]
                encode_fn(updated_ids)
                row["bg_tokens"] = len(updated_ids)
                row["bg_chunks"] = None
            row["bg_time_s"] = time.time() - bg_t0

            # 3) 更新后的首请求（TTFT 探测，max_tokens=1）
            ttft, _ = timed_generate(llm, probe_prompt, 1)
            row["ttft_probe_s"] = round(ttft, 3)

            # 4) 答案 + 新鲜度检查（此时各臂缓存已收敛）
            ans_t, answer = timed_generate(llm, probe_prompt, 64)
            row["answer_time_s"] = round(ans_t, 3)
            row["answer"] = answer.strip()
            row["flip_ok"] = "2210" in answer

            results["arms"][arm] = row
            print(
                f"  fresh={row.get('fresh_chunks')} stale={row.get('stale_chunks')} "
                f"bg_tokens={row.get('bg_tokens', 0)} bg={row['bg_time_s']:.2f}s "
                f"| first-request {ttft:.2f}s | flip_ok={row['flip_ok']}\n"
            )

        manager.stop()

    # ---------------- 汇总 ----------------
    print("=" * 78)
    print(f"{'arm':<16}{'bg_tokens':>10}{'bg_time':>9}{'fresh':>7}"
          f"{'stale':>7}{'ttft':>9}{'answer':>9}{'flip':>7}")
    print("-" * 78)
    for arm in arms_to_run:
        r = results["arms"][arm]
        print(
            f"{arm:<16}{str(r.get('bg_tokens', 0)):>10}{r['bg_time_s']:>8.2f}s"
            f"{str(r.get('fresh_chunks', '-')):>7}{str(r.get('stale_chunks', '-')):>7}"
            f"{r['ttft_probe_s']:>8.2f}s{r['answer_time_s']:>8.2f}s"
            f"{str(r['flip_ok']):>7}"
        )
    print("-" * 78)
    print("解读: proactive vs norefresh  = ③ 的请求路径收益（全命中 vs 全量重算）")
    print("      proactive vs passive    = ③ 把重算从请求路径挪到空闲后台的收益")
    print("      proactive vs proactive_full = ② 增量差集省下的后台算力")
    print("      四臂 flip_ok 应全为 True：内容寻址保证 staleness 只损失时间不答错")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nresults saved to {out_path}")


if __name__ == "__main__":
    main()
