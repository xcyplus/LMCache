# SPDX-License-Identifier: Apache-2.0
"""Run a cache-free post/0 quality control and score saved strategy answers."""

# Standard
from pathlib import Path
from statistics import fmean
import argparse
import os
import sys

# First Party
from eval_utils import exact_match, f1_score, flip_ok, normalize_answer
from trace_experiment_utils import (
    DEFAULT_MODEL,
    MAX_MODEL_LEN,
    build_prompt,
    load_trace,
    read_json,
    write_json,
)

MAX_ANSWER_TOKENS = 64


def select_post_cases(trace: dict, limit: int | None) -> dict:
    """Snapshot the documents visible at each question's first post request."""
    docs = dict(trace["docs"])
    cases = {}
    for event in trace["events"]:
        if event["type"] == "update":
            docs[event["url"]] = event["new_text"]
            continue
        if event["type"] != "request" or event["phase"] != "post":
            continue
        qid = event["qid"]
        if qid in cases:
            continue
        reference = event.get("fake_entity")
        if not isinstance(reference, str) or not normalize_answer(reference):
            raise ValueError(f"{qid}: missing or empty post-update reference")
        cases[qid] = {
            "event": event,
            "documents": {url: docs[url] for url in event["urls"]},
            "reference": reference,
        }
        if len(cases) == limit:
            break
    if not cases:
        raise ValueError("Trace has no post/0 requests")
    return cases


def select_saved_answers(
    result: dict, cases: dict, trace_hash: str, source: Path
) -> dict:
    if result.get("trace_sha256", trace_hash) != trace_hash:
        raise ValueError(f"{source}: result was produced from a different trace")
    answers = {}
    for row in result["requests"]:
        qid = row["qid"]
        if qid not in cases or row["phase"] != "post" or row["seq"] != 0:
            continue
        if qid in answers:
            raise ValueError(f"{source}: duplicate post/0 answer for {qid}")
        if not isinstance(row.get("answer"), str):
            raise ValueError(f"{source}: missing post/0 answer for {qid}")
        answers[qid] = row
    missing = cases.keys() - answers.keys()
    if missing:
        example = min(missing)
        raise ValueError(
            f"{source}: missing {len(missing)} post/0 rows (e.g. {example})"
        )
    return answers


def score(answer: str, reference: str) -> dict:
    return {
        "new_entity_hit": flip_ok(answer, reference),
        "new_reference_em": exact_match(answer, reference),
        "new_reference_f1": f1_score(answer, reference),
    }


def score_summary(scores: list[dict]) -> dict:
    hits = sum(row["new_entity_hit"] for row in scores)
    return {
        "n": len(scores),
        "new_entity_hits": hits,
        "new_entity_hit_rate": hits / len(scores),
        "new_reference_em": fmean(row["new_reference_em"] for row in scores),
        "new_reference_f1": fmean(row["new_reference_f1"] for row in scores),
    }


def compare_answers(cases: dict, cold: dict, cached: dict) -> dict:
    pairs = []
    for qid, case in cases.items():
        cold_row = cold[qid]
        cached_row = cached[qid]
        if (
            cold_row.get("prompt_len") is not None
            and cached_row.get("prompt_len") is not None
            and cold_row["prompt_len"] != cached_row["prompt_len"]
        ):
            raise ValueError(f"{qid}: cold and cached prompt lengths differ")
        reference = case["reference"]
        pairs.append(
            {
                "qid": qid,
                "reference": reference,
                "cold": score(cold_row["answer"], reference),
                "cached": score(cached_row["answer"], reference),
            }
        )
    cold_summary = score_summary([pair["cold"] for pair in pairs])
    cached_summary = score_summary([pair["cached"] for pair in pairs])
    return {
        "cold": cold_summary,
        "cached": cached_summary,
        "cached_minus_cold_hit_rate": cached_summary["new_entity_hit_rate"]
        - cold_summary["new_entity_hit_rate"],
        "cold_hit_cached_miss": sum(
            pair["cold"]["new_entity_hit"] and not pair["cached"]["new_entity_hit"]
            for pair in pairs
        ),
        "cold_miss_cached_hit": sum(
            not pair["cold"]["new_entity_hit"] and pair["cached"]["new_entity_hit"]
            for pair in pairs
        ),
        "pairs": pairs,
    }


def generate_cold(trace: dict, cases: dict, cached_results: list, model: str) -> dict:
    # Import GPU dependencies only when generating answers.
    os.environ["VLLM_USE_V1"] = "1"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    # Third Party
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(model)
    prompts = {}
    for qid, case in cases.items():
        event = case["event"]
        prompt = build_prompt(
            tokenizer,
            trace["sys_prompt"],
            event["urls"],
            case["documents"],
            event["question"],
        )
        if len(prompt) + MAX_ANSWER_TOKENS > MAX_MODEL_LEN:
            raise ValueError(f"{qid}: prompt and answer budget exceed {MAX_MODEL_LEN}")
        for source, _, answers in cached_results:
            saved_len = answers[qid].get("prompt_len")
            if saved_len is not None and saved_len != len(prompt):
                raise ValueError(f"{source}: prompt length differs for {qid}")
        prompts[qid] = prompt

    # No KVTransferConfig means no LMCache connector. Disable vLLM's own prefix
    # cache too, so one request cannot warm the KV cache for another.
    llm = LLM(
        model=model,
        kv_transfer_config=None,
        enable_prefix_caching=False,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=0.8,
    )
    params = SamplingParams(temperature=0, max_tokens=MAX_ANSWER_TOKENS)
    rows = {}
    for i, (qid, prompt) in enumerate(prompts.items(), start=1):
        (result,) = llm.generate(
            {"prompt_token_ids": prompt},
            sampling_params=params,
            use_tqdm=False,
        )
        if getattr(result, "num_cached_tokens", 0):
            raise RuntimeError(f"{qid}: cache hit in the cold control")
        answer = result.outputs[0].text.strip()
        reference = cases[qid]["reference"]
        rows[qid] = {
            "qid": qid,
            "phase": "post",
            "seq": 0,
            "prompt_len": len(prompt),
            "answer": answer,
            "reference": reference,
            **score(answer, reference),
        }
        if i % 20 == 0 or i == len(prompts):
            print(f"Cold post/0: {i}/{len(prompts)}", file=sys.stderr, flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, help="Use the first N post/0 questions")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cached-result", type=Path, action="append", default=[])
    parser.add_argument("--cold-result", type=Path, help="Rescore saved cold answers")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    source_paths = [args.trace, *args.cached_result]
    if args.out.resolve() in {path.resolve() for path in source_paths}:
        parser.error("--out must differ from the trace and cached result paths")

    trace, trace_hash = load_trace(args.trace)
    cases = select_post_cases(trace, args.limit)
    cached_results = []
    for source in args.cached_result:
        result = read_json(source)
        if result.get("model", args.model) != args.model:
            parser.error(f"{source}: model differs from --model")
        if "trace_sha256" not in result:
            print(
                f"{source}: legacy result has no trace fingerprint. Confirm the "
                "original trace, model and generation settings before comparison.",
                file=sys.stderr,
            )
        cached_results.append(
            (
                source,
                result["policy"],
                select_saved_answers(result, cases, trace_hash, source),
            )
        )

    if args.cold_result:
        result = read_json(args.cold_result)
        if (
            result.get("policy") != "cold_post"
            or result.get("trace_sha256") != trace_hash
            or result.get("model") != args.model
            or result.get("max_answer_tokens") != MAX_ANSWER_TOKENS
        ):
            parser.error("--cold-result must match the trace, model and answer budget")
        cold = select_saved_answers(result, cases, trace_hash, args.cold_result)
    else:
        cold = generate_cold(trace, cases, cached_results, args.model)

    requests = []
    for qid, case in cases.items():
        reference = case["reference"]
        row = cold[qid]
        requests.append(
            {**row, "reference": reference, **score(row["answer"], reference)}
        )
    comparisons = [
        {"file": str(source), "policy": policy, **compare_answers(cases, cold, answers)}
        for source, policy, answers in cached_results
    ]
    summary = score_summary(requests)
    write_json(
        args.out,
        {
            "policy": "cold_post",
            "trace_sha256": trace_hash,
            "model": args.model,
            "max_answer_tokens": MAX_ANSWER_TOKENS,
            "prefix_cache": False,
            "kv_connector": None,
            "summary": summary,
            "requests": requests,
            "comparisons": comparisons,
        },
    )
    print(f"cold_post: {summary['new_entity_hits']}/{summary['n']} new-entity hits")
    for result in comparisons:
        cached_hits = result["cached"]["new_entity_hits"]
        print(
            f"{result['policy']}: {cached_hits}/{summary['n']} new-entity hits | "
            f"cold hit -> cached miss: {result['cold_hit_cached_miss']}; "
            f"cold miss -> cached hit: {result['cold_miss_cached_hit']}"
        )
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
