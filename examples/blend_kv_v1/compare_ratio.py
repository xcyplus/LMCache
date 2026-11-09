#!/usr/bin/env python
"""对比 ratio=0.15 (基线) vs ratio=0.20 (消融) 的 summary 关键字段。
论文 §5.5b (b'') ratio 敏感性消融的配套查看工具。

用法：
    python compare_ratio.py
    python compare_ratio.py --base traces_t1_new --exp traces_t1_ratio020
"""
import json
import os
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
KEYS = ["prefill_post_0_s", "prefill_pre_1_s", "prefill_pre_0_s",
        "miss_tokens_post_0", "pre_f1", "post_flip_rate",
        "post_0_full_hit_rate", "bg_tokens_total"]


def show(label, path):
    if not os.path.exists(path):
        print(f"[WARN] {path} not found")
        return None
    s = json.load(open(path))["summary"]
    print(f"--- {label} ({os.path.basename(os.path.dirname(path))}) ---")
    for k in KEYS:
        if k in s:
            print(f"  {k}: {s[k]}")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.join(HERE, "traces_t1_new"),
                    help="ratio=0.15 基线目录")
    ap.add_argument("--exp", default=os.path.join(HERE, "traces_t1_ratio020"),
                    help="ratio=0.20 消融目录")
    args = ap.parse_args()

    sb = show("baseline ratio=0.15", os.path.join(args.base, "results_proactive.json"))
    se = show("experiment ratio=0.20", os.path.join(args.exp, "results_proactive.json"))

    if sb and se:
        print("\n=== delta (0.20 - 0.15) ===")
        for k in KEYS:
            if k in sb and k in se:
                try:
                    d = se[k] - sb[k]
                    print(f"  {k}: {sb[k]} -> {se[k]}  (delta={d:+.4f})")
                except TypeError:
                    print(f"  {k}: {sb[k]} -> {se[k]}")


if __name__ == "__main__":
    main()
