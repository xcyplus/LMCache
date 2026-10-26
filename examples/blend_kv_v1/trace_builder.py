# SPDX-License-Identifier: Apache-2.0
"""从 HotpotQA 构建刷新实验的 trace：文档库 + 请求流 + 更新流。

输出（--out-dir 下）：
    trace.json     回放事件流（docs / sys_prompt / events）
    rag_db.json    url -> 文本（供 UrlSegmentTokenDatabase 通过
                   LMCACHE_RAG_DB_PATH 加载）

granularity 两种模式：
    question（默认）: 每题 10 个维基段落拼成一篇长文档（~500-2000 token，
        跨多个 256-token chunk），url = hotpot://{qid}。编辑发生在文档
        中部 → 触发级联失效，是 ② 增量收益的用武之地。
    paragraph: 每段一篇短文档，跨题目共享，url = hotpot://{qid}/p{i}。

合成编辑（Budgeted Repair 论文的 probe 方法论）：
    把 gold answer 的字符串替换为虚构实体 "Zyxoria"，
    pre 阶段请求应答出 gold，post 阶段请求应翻转答出 Zyxoria——
    自动构造 direct probe，无需 LLM 生成问题。

事件流（三相位）：
    phase "pre"    : 每题 2 个请求（第 1 个生成答案，期望 gold）
    phase "update" : 每题 1 个更新事件
    phase "post"   : 每题 2 个请求（第 1 个生成答案，期望翻转）

用法：
    python trace_builder.py --num-questions 40 --out-dir traces/
    # 论文规模: --num-questions 300+
"""

# Standard
import argparse
import json
import os
import re

# Third Party
from datasets import load_dataset

FAKE_ENTITY = "Zyxoria"
SYS_PROMPT = "Answer the question based on the provided documents."


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-questions", type=int, default=40, help="采样的题目数（默认 40 调试）"
    )
    parser.add_argument(
        "--split", default="validation", help="hotpot_qa split（默认 validation）"
    )
    parser.add_argument(
        "--granularity", default="question", choices=["question", "paragraph"]
    )
    parser.add_argument(
        "--model", default="mistralai/Mistral-7B-Instruct-v0.2",
        help="tokenizer 来源（--edit-mode mid_iso 需要）",
    )
    parser.add_argument(
        "--edit-mode", default="mid_len",
        choices=["mid_len", "mid_iso", "tail"],
        help="编辑类型（论文表 3 消融）: mid_len=变长中部替换(级联), "
             "mid_iso=等长中部替换(无级联, 需 tokenizer), tail=尾部编辑(仅末尾块)",
    )
    parser.add_argument(
        "--request-skew", type=float, default=0.0,
        help="Zipf 倾斜指数（论文表 2）: >0 时按 Zipf(s) 给头部题目更多 pre 请求，"
             "产生热度倾斜; 0 = 均匀（每题 2 次）",
    )
    parser.add_argument(
        "--schedule", default="block", choices=["block", "phase"],
        help="事件排布: block=每题自成一组(表 1 用); phase=全部 pre→全部 update→"
             "全部 post 三相位(表 2 用, 批量更新下预算内排序才有区分度)",
    )
    parser.add_argument("--out-dir", default="traces")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def make_question_doc(example):
    """question 粒度：10 段拼一篇长文档。返回 (doc_text, None)。"""
    titles = example["context"]["title"]
    sentences = example["context"]["sentences"]
    paras = [f"{t}: {' '.join(s)}" for t, s in zip(titles, sentences)]
    return "\n".join(paras), None


def make_paragraph_docs(example, qid):
    """paragraph 粒度：每段一篇短文档。返回 (doc_text, [urls])。"""
    titles = example["context"]["title"]
    sentences = example["context"]["sentences"]
    urls, texts = [], []
    for i, (t, s) in enumerate(zip(titles, sentences)):
        url = f"hotpot://{qid}/p{i}"
        urls.append(url)
        texts.append((url, f"{t}: {' '.join(s)}"))
    # 拼接文本仅用于编辑定位；请求用 urls 列表按序取各段文本
    return texts, urls


def apply_edit(doc_text, gold):
    """把 gold answer 替换为虚构实体。返回 (new_text, n_replaced)。"""
    if gold in doc_text:
        return doc_text.replace(gold, FAKE_ENTITY), doc_text.count(gold)
    pattern = re.compile(re.escape(gold), re.IGNORECASE)
    if pattern.search(doc_text):
        return pattern.sub(FAKE_ENTITY, doc_text), len(pattern.findall(doc_text))
    return None, 0


def make_edit(doc_text, gold, mode, tokenizer):
    """按编辑模式构造修订文本（论文表 3 的失效范围消融）。

    返回 (new_text, 注入实体串, n_replaced)；无法构造时返回 (None, ..., 0)。

      mid_len  变长中部替换（默认）：替换全部 gold 出现处，编辑点后边界
               滑移 → 级联失效
      mid_iso  等长中部替换：只替换首处，用中性填充把 token 数补齐到与
               gold 相同 → 编辑点后边界不动（隔离级联效应）
      tail     尾部编辑：只替换最后一次出现（尾部无出现时追加覆盖句）
               → 失效仅限末尾 chunk
    """
    if mode == "mid_len":
        new_text, n_replaced = apply_edit(doc_text, gold)
        return new_text, FAKE_ENTITY, n_replaced or 0
    if mode == "mid_iso":
        idx = doc_text.find(gold)
        if idx < 0:
            return None, FAKE_ENTITY, 0
        n_gold = len(tokenizer.encode(gold))
        inject = FAKE_ENTITY
        if len(tokenizer.encode(inject)) > n_gold:
            # 注入实体比 gold 的 token 数还长，无法等长对齐，弃题
            return None, FAKE_ENTITY, 0
        inject += " " + " ".join(
            ["x"] * (n_gold - len(tokenizer.encode(inject)))
        )
        new_text = doc_text[:idx] + inject + doc_text[idx + len(gold):]
        return new_text, FAKE_ENTITY, 1
    if mode == "tail":
        idx = doc_text.rfind(gold)
        if idx >= 0:
            new_text = doc_text[:idx] + FAKE_ENTITY + doc_text[idx + len(gold):]
            return new_text, FAKE_ENTITY, 1
        new_text = doc_text + f"\nNote: the current answer is {FAKE_ENTITY}."
        return new_text, FAKE_ENTITY, 1
    raise ValueError(f"unknown edit mode: {mode}")


def zipf_pre_counts(n_questions, skew):
    """Zipf(s) 分配每题的 pre 请求重复数，制造热度倾斜（论文表 2）。

    s=0 均匀（每题 2 次）；s>0 时头部题多、尾部题少，url_access_counts
    随之倾斜，热度排序策略与 FIFO/Oracle 产生可测差异。
    """
    if not skew:
        return [2] * n_questions
    return [max(1, int(round(6.0 / ((i + 1) ** skew)))) for i in range(n_questions)]


def pick_editable(examples, num_questions):
    """筛选可自动编辑的题目：非 yes/no、答案短、答案字符串在文中出现。"""
    picked = []
    for ex in examples:
        gold = ex["answer"].strip()
        if not gold or gold.lower() in ("yes", "no"):
            continue
        if len(gold.split()) > 8:
            continue
        picked.append(ex)
        if len(picked) >= num_questions:
            break
    return picked


def build_question_granularity(examples, edit_mode, tokenizer, skew, schedule):
    """返回 (docs: {url: text}, events: [...])。

    edit_mode 控制编辑类型（表 3），skew 控制 pre 请求的 Zipf 倾斜（表 2），
    schedule=phase 时把事件重排为 pre→update→post 三相位（表 2）。
    """
    docs, events = {}, []
    pre_counts = zipf_pre_counts(len(examples), skew)
    for ex, n_pre in zip(examples, pre_counts):
        qid = ex["id"]
        gold = ex["answer"].strip()
        url = f"hotpot://{qid}"
        doc_text, _ = make_question_doc(ex)
        new_text, inject, n_replaced = make_edit(doc_text, gold, edit_mode, tokenizer)
        if new_text is None or n_replaced == 0:
            continue
        docs[url] = doc_text
        question = ex["question"].strip()
        for r in range(n_pre):
            events.append({"type": "request", "qid": qid, "urls": [url],
                           "question": question, "gold": gold, "phase": "pre",
                           "gen_answer": r == 0, "expect": "gold",
                           "fake_entity": inject})
        events.append({"type": "update", "qid": qid, "url": url,
                       "new_text": new_text, "fake_entity": inject,
                       "gold": gold})
        for r in range(2):
            events.append({"type": "request", "qid": qid, "urls": [url],
                           "question": question, "gold": gold, "phase": "post",
                           "gen_answer": r == 0, "expect": "fake",
                           "fake_entity": inject})
    if schedule == "phase":
        # 三相位重排：批量更新 + 有限预算下，刷新排序策略才有区分度（表 2）
        order = {"pre": 0, "update": 1, "post": 2}
        events.sort(key=lambda e: order[e["phase"] if e["type"] == "request"
                                         else e["type"]])
    return docs, events


def build_paragraph_granularity(examples):
    """paragraph 粒度：编辑落在 supporting fact 所在段。"""
    docs, events = {}, []
    for ex in examples:
        qid = ex["id"]
        gold = ex["answer"].strip()
        titles = ex["context"]["title"]
        sup_title = ex["supporting_facts"]["title"][0]
        if sup_title not in titles:
            continue
        pidx = list(titles).index(sup_title)
        # 编辑目标段
        para_url = f"hotpot://{qid}/p{pidx}"
        para_text = f"{sup_title}: {' '.join(ex['context']['sentences'][pidx])}"
        new_para, n_replaced = apply_edit(para_text, gold)
        if new_para is None or n_replaced == 0:
            continue
        urls = []
        for i, t in enumerate(titles):
            url = f"hotpot://{qid}/p{i}"
            urls.append(url)
            if url not in docs:
                docs[url] = f"{t}: {' '.join(ex['context']['sentences'][i])}"
        docs[para_url] = para_text
        question = ex["question"].strip()
        events.append({"type": "request", "qid": qid, "urls": urls,
                       "question": question, "gold": gold, "phase": "pre",
                       "gen_answer": True, "expect": "gold"})
        events.append({"type": "request", "qid": qid, "urls": urls,
                       "question": question, "gold": gold, "phase": "pre",
                       "gen_answer": False, "expect": "gold"})
        events.append({"type": "update", "qid": qid, "url": para_url,
                       "new_text": new_para, "fake_entity": FAKE_ENTITY,
                       "gold": gold})
        events.append({"type": "request", "qid": qid, "urls": urls,
                       "question": question, "gold": gold, "phase": "post",
                       "gen_answer": True, "expect": "fake"})
        events.append({"type": "request", "qid": qid, "urls": urls,
                       "question": question, "gold": gold, "phase": "post",
                       "gen_answer": False, "expect": "fake"})
    return docs, events


def main():
    args = parse_args()
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=args.split)
    ds = ds.shuffle(seed=args.seed)
    examples = pick_editable(ds, args.num_questions)

    # 等长编辑模式需要 tokenizer 做 token 数对齐
    tokenizer = None
    if args.edit_mode == "mid_iso":
        # Third Party
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model)

    if args.granularity == "question":
        docs, events = build_question_granularity(
            examples, args.edit_mode, tokenizer, args.request_skew, args.schedule
        )
    else:
        docs, events = build_paragraph_granularity(examples)

    os.makedirs(args.out_dir, exist_ok=True)
    trace = {"sys_prompt": SYS_PROMPT, "docs": docs, "events": events}
    trace_path = os.path.join(args.out_dir, "trace.json")
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump(trace, f, ensure_ascii=False, indent=2)

    rag_db_path = os.path.join(args.out_dir, "rag_db.json")
    with open(rag_db_path, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

    n_req = sum(1 for e in events if e["type"] == "request")
    n_upd = sum(1 for e in events if e["type"] == "update")
    print(f"trace: {len(docs)} docs, {n_req} requests, {n_upd} updates")
    print(f"  -> {trace_path}")
    print(f"  -> {rag_db_path}")


if __name__ == "__main__":
    main()
