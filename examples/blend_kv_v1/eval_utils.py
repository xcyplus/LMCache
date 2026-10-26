# SPDX-License-Identifier: Apache-2.0
"""QA 质量指标：SQuAD 式 normalize / EM / token-F1，以及实体翻转检查。

F1/EM 与 HotpotQA 官方评测一致，保证和已有工作可比；
flip 检查沿用 Budgeted Repair 论文的 probe 思路：编辑后模型答案
是否翻转到注入的新实体。
"""

# Standard
import collections
import re
import string


def normalize_answer(s: str) -> str:
    """SQuAD 官方归一化：小写、去标点/冠词/多余空白。"""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def f1_score(prediction: str, gold: str) -> float:
    """Token 级 F1（SQuAD 式）。"""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = collections.Counter(pred_tokens) & collections.Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def flip_ok(answer: str, entity: str) -> bool:
    """编辑后的答案是否包含注入的新实体（新鲜度检查）。"""
    return normalize_answer(entity) in normalize_answer(answer)
