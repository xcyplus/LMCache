#!/usr/bin/env python
"""McNemar 配对显著性检验 + 翻转率 Wilson 95% 置信区间。

用 results JSON 的 requests[].answer + Zyxoria 实体重算每题翻转，做配对检验。
论文 §5.5b (b') 配对显著性检验。

用法（从 examples/blend_kv_v1/ 目录或任意目录运行均可）：
    python mcnemar_test.py
    python mcnemar_test.py --base /path/to/traces_t1_new

输入：traces_t1_new/results_{norefresh,passive,proactive,proactive_full}.json
输出：各策略翻转率 + 95%CI + 5 对 McNemar 精确检验
"""
import json
import os
import re
import string
import math
import argparse

DEFAULT_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "traces_t1_new")
POLICIES = ["norefresh", "passive", "proactive", "proactive_full"]
ENTITY = "Zyxoria"


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    return " ".join(remove_articles(remove_punc(s.lower())).split())


def flip_ok(answer, entity=ENTITY):
    """答案归一化后是否包含目标实体（翻转成功）。"""
    return int(normalize_answer(entity) in normalize_answer(answer))


def load_post0_flip(path):
    """从 results JSON 提取 post/0 阶段每题的翻转标志 {qid: 0/1}。"""
    d = json.load(open(path))
    reqs = d.get("requests", [])
    out = {}
    for r in reqs:
        if r.get("phase") == "post" and r.get("seq") == 0:
            qid = r.get("qid")
            ans = r.get("answer", "")
            if qid is None or not ans:
                continue
            out[str(qid)] = flip_ok(ans)
    return out


def mcnemar(fa, fb):
    """McNemar 配对检验：b=A翻转B不翻转, c=A不翻转B翻转。
    返回 (b, c, chi2_approx, p_exact_二项双侧, n)。
    """
    common = set(fa) & set(fb)
    b = sum(1 for k in common if fa[k] == 1 and fb[k] == 0)
    c = sum(1 for k in common if fa[k] == 0 and fb[k] == 1)
    n = len(common)
    disc = b + c
    if disc == 0:
        return b, c, 0.0, 1.0, n
    chi2 = (abs(b - c) - 1) ** 2 / disc
    # 二项精确双侧
    from math import comb
    k = min(b, c)
    tail = 2 * sum(comb(disc, i) for i in range(0, k + 1)) / (2 ** disc)
    if b == c:
        tail = 1.0
    p_exact = min(tail, 1.0)
    return b, c, chi2, p_exact, n


def wilson(rate, n, z=1.96):
    """Wilson 分数区间 95% CI。"""
    if n == 0:
        return (0, 0)
    denom = 1 + z * z / n
    center = (rate + z * z / (2 * n)) / denom
    half = z * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / denom
    return (max(0, center - half), min(1, center + half))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE, help="results 目录")
    args = ap.parse_args()

    flips = {}
    for p in POLICIES:
        path = os.path.join(args.base, f"results_{p}.json")
        if not os.path.exists(path):
            print(f"[WARN] {path} not found")
            continue
        flips[p] = load_post0_flip(path)
        n = len(flips[p])
        rate = sum(flips[p].values()) / n if n else 0
        lo, hi = wilson(rate, n)
        print(f"{p:>16}: n={n}  flip_rate={rate:.3f}  95%CI=[{lo:.3f},{hi:.3f}]")

    print("\n=== McNemar 配对检验 (post/0 翻转, entity=Zyxoria) ===")
    pairs = [
        ("proactive", "norefresh",       "增量 vs 无缓存基线"),
        ("proactive", "proactive_full",  "增量 vs 全量后台 [M2核心]"),
        ("proactive", "passive",         "增量 vs 仅差集失效"),
        ("proactive_full", "norefresh",  "全量后台 vs 无缓存"),
        ("passive", "norefresh",         "差集失效 vs 无缓存"),
    ]
    print(f"{'pair':<38} {'b':>4} {'c':>4} {'chi2':>7} {'p_exact':>9} {'n':>4}  sig")
    for a, bname, note in pairs:
        if a not in flips or bname not in flips:
            continue
        b, c, chi2, p, n = mcnemar(flips[a], flips[bname])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"{a+' vs '+bname:<38} {b:>4} {c:>4} {chi2:>7.2f} {p:>9.4f} {n:>4}  {sig}  {note}")

    print("\n b=A翻转B不翻转, c=A不翻转B翻转, p_exact=二项精确双侧")
    print(" *** p<0.001, ** p<0.01, * p<0.05, ns 不显著")


if __name__ == "__main__":
    main()
