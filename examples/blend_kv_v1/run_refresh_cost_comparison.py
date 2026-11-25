# SPDX-License-Identifier: Apache-2.0
"""Measure incremental and whole-document refresh on the same block trace."""

# Standard
from dataclasses import dataclass
from importlib.metadata import version
from itertools import groupby, islice
from math import ceil
from pathlib import Path
from statistics import fmean
from tempfile import TemporaryDirectory
from time import perf_counter
import argparse
import os
import sys

# First Party
from trace_experiment_utils import (
    DEFAULT_MODEL,
    build_prompt,
    load_trace,
    read_json,
    write_json,
)

PROTOCOL = "serial_block_no_bos_v1"


@dataclass(frozen=True)
class Block:
    qid: str
    url: str
    pre: tuple[dict, ...]
    new_text: str
    post: dict


def select_blocks(trace: dict, limit: int | None) -> list[Block]:
    """Select independent single-document updates with a warm pre request."""
    blocks = []
    seen_qids = set()
    seen_urls = set()
    groups = groupby(trace["events"], key=lambda event: event["qid"])
    for qid, events in islice(groups, limit):
        pre, updates, post = [], [], []
        for event in events:
            kind = event["type"]
            if kind == "update" and pre and not updates:
                updates.append(event)
            elif kind == "request" and event["phase"] == "pre" and not updates:
                pre.append(event)
            elif kind == "request" and event["phase"] == "post" and updates:
                post.append(event)
            else:
                raise ValueError(f"{qid}: expected pre requests, one update, then post")
        if not pre or len(updates) != 1 or not post:
            raise ValueError(f"{qid}: incomplete block; use a --schedule block trace")
        update = updates[0]
        url = update["url"]
        if (
            qid in seen_qids
            or url in seen_urls
            or url not in trace["docs"]
            or any(event["urls"] != [url] for event in pre + post)
        ):
            raise ValueError(
                f"{qid}: each block must reference its own single document"
            )
        blocks.append(Block(qid, url, tuple(pre), update["new_text"], post[0]))
        seen_qids.add(qid)
        seen_urls.add(url)
    if not blocks:
        raise ValueError("Trace has no complete blocks")
    return blocks


def incremental_inputs(
    database, document_ids: list[int], fresh: list[int]
) -> list[list[int]]:
    """Select new chunks from the same no-BOS sequence submitted by the full arm."""
    inputs = []
    pending = set(fresh)
    for start in range(0, len(document_ids), database.chunk_size):
        chunk = document_ids[start : start + database.chunk_size]
        # Use the engine's hash domain. compute_refresh_chunks(..., False)
        # keeps the BOS variant, whereas this experiment serves no-BOS documents.
        chunk_hash = database._hash_tokens(chunk)
        if chunk_hash in pending:
            inputs.append(chunk)
            pending.remove(chunk_hash)
    return inputs


def measure_block(
    strategy: str,
    block: Block,
    trace: dict,
    engine,
    tokenizer,
    generate_one,
    synchronize,
) -> dict:
    def prompt(event, text):
        return build_prompt(
            tokenizer,
            trace["sys_prompt"],
            event["urls"],
            {block.url: text},
            event["question"],
        )

    for event in block.pre:
        generate_one(prompt(event, trace["docs"][block.url]))
    synchronize()
    if engine.get_article(block.url) is None:
        raise RuntimeError(f"{block.qid}: warm-up did not register {block.url}")

    # Both arms begin at the update handler and end after the last encoding
    # call and CUDA synchronization. The preceding warm-up is outside the timer.
    start = perf_counter()
    if strategy == "full":
        engine.remove_article(block.url)
    fresh, _ = engine.refresh_article(block.url, block.new_text)
    document_ids = tokenizer.encode(block.new_text)[1:]
    if not document_ids:
        raise ValueError(f"{block.qid}: updated document has no tokens")
    if strategy == "incremental":
        inputs = incremental_inputs(engine.token_database, document_ids, fresh)
    else:
        inputs = [document_ids]
    for ids in inputs:
        generate_one(ids)
    synchronize()
    refresh_s = perf_counter() - start

    request_ids = prompt(block.post, block.new_text)
    start = perf_counter()
    generate_one(request_ids)
    synchronize()
    post0_s = perf_counter() - start
    return {
        "qid": block.qid,
        "url": block.url,
        "updated_document_tokens": len(document_ids),
        "submitted_input_tokens": sum(len(ids) for ids in inputs),
        "encode_calls": len(inputs),
        "refresh_wall_s": refresh_s,
        "post0_wall_s": post0_s,
        "post0_prompt_tokens": len(request_ids),
    }


def summarize(rows: list[dict]) -> dict:
    post0 = sorted(row["post0_wall_s"] for row in rows)
    return {
        "n": len(rows),
        "refresh_total_s": sum(row["refresh_wall_s"] for row in rows),
        "refresh_mean_s": fmean(row["refresh_wall_s"] for row in rows),
        "post0_mean_s": fmean(post0),
        "post0_p95_s": post0[ceil(0.95 * len(post0)) - 1],
        "submitted_input_tokens": sum(row["submitted_input_tokens"] for row in rows),
        "encode_calls": sum(row["encode_calls"] for row in rows),
    }


def pair_results(incremental: dict, full: dict) -> dict:
    if (
        incremental.get("strategy") != "incremental"
        or full.get("strategy") != "full"
        or incremental.get("protocol") != PROTOCOL
        or full.get("protocol") != PROTOCOL
        or incremental["trace_sha256"] != full["trace_sha256"]
        or incremental["settings"] != full["settings"]
    ):
        raise ValueError(
            "Expected incremental/full results from the same trace and setup"
        )
    pairs = []
    for inc, whole in zip(incremental["rows"], full["rows"], strict=True):
        for key in ("qid", "url", "updated_document_tokens", "post0_prompt_tokens"):
            if inc[key] != whole[key]:
                raise ValueError(f"Paired workload mismatch for {key}: {inc['qid']}")
        pairs.append(
            {
                "qid": inc["qid"],
                "refresh_incremental_minus_full_s": (
                    inc["refresh_wall_s"] - whole["refresh_wall_s"]
                ),
                "post0_incremental_minus_full_s": (
                    inc["post0_wall_s"] - whole["post0_wall_s"]
                ),
            }
        )
    if not pairs:
        raise ValueError("No matched measurements")
    return {
        "protocol": PROTOCOL,
        "trace_sha256": incremental["trace_sha256"],
        "settings": incremental["settings"],
        "incremental": summarize(incremental["rows"]),
        "full": summarize(full["rows"]),
        "pairs": pairs,
    }


def run_arm(
    strategy: str, trace: dict, blocks: list[Block], model: str
) -> tuple[list[dict], dict]:
    # Keep offline comparison usable without vLLM, CUDA or LMCache imports.
    os.environ["VLLM_USE_V1"] = "1"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    if os.getenv("LMCACHE_CONFIG_FILE"):
        raise ValueError("Unset LMCACHE_CONFIG_FILE to use the local CPU KV backend")

    # Third Party
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    import torch

    # First Party
    from blend_mfo import build_llm_with_lmcache, setup_environment_variables
    from lmcache.v1.config import LMCacheEngineConfig
    from run_trace_experiment import get_engine

    if not torch.cuda.is_available():
        raise RuntimeError("The refresh cost benchmark needs a CUDA GPU")
    tokenizer = AutoTokenizer.from_pretrained(model)
    with TemporaryDirectory(prefix="lmcache-refresh-cost-") as directory:
        # Historical result directories may contain only trace.json.
        rag_db = Path(directory) / "rag_db.json"
        write_json(rag_db, trace["docs"])
        setup_environment_variables(use_disk=False, rag_db_path=str(rag_db))
        config = LMCacheEngineConfig.from_env()
        if (
            config.local_disk is not None
            or config.remote_url is not None
            or config.enable_controller
            or config.enable_p2p
            or config.enable_nixl
            or config.enable_xpyd
        ):
            raise ValueError("Use the local CPU KV backend for both arms")
        with build_llm_with_lmcache("LMCacheConnectorV1", model) as llm:
            engine = get_engine(llm)
            params = SamplingParams(temperature=0, max_tokens=1)

            def generate_one(ids):
                llm.generate(
                    {"prompt_token_ids": ids},
                    sampling_params=params,
                    use_tqdm=False,
                )

            device = torch.cuda.get_device_properties(torch.cuda.current_device())
            settings = {
                "model": model,
                "dtype": str(llm.llm_engine.model_config.dtype),
                "gpu": device.name,
                "gpu_memory_bytes": device.total_memory,
                "torch": str(torch.__version__),
                "cuda": torch.version.cuda,
                "vllm": version("vllm"),
                "lmcache": version("lmcache"),
                "chunk_size": config.chunk_size,
                "cpu_cache_gb": config.max_local_cpu_size,
                "blend_recompute_ratio": config.blend_recompute_ratio,
                "blend_min_tokens": config.blend_min_tokens,
                "hash_algorithm": config.pre_caching_hash_algorithm,
                "python_hash_seed": os.getenv("PYTHONHASHSEED"),
            }
            rows = []
            for i, block in enumerate(blocks, 1):
                rows.append(
                    measure_block(
                        strategy,
                        block,
                        trace,
                        engine,
                        tokenizer,
                        generate_one,
                        torch.cuda.synchronize,
                    )
                )
                if i % 20 == 0 or i == len(blocks):
                    print(
                        f"{strategy}: {i}/{len(blocks)} blocks",
                        file=sys.stderr,
                        flush=True,
                    )
    return rows, settings


def show(strategy: str, summary: dict) -> None:
    print(
        f"{strategy}: n={summary['n']}, refresh={summary['refresh_total_s']:.3f}s "
        f"(mean {summary['refresh_mean_s'] * 1000:.1f}ms), "
        f"post/0 mean={summary['post0_mean_s'] * 1000:.1f}ms, "
        f"p95={summary['post0_p95_s'] * 1000:.1f}ms, "
        f"submitted tokens={summary['submitted_input_tokens']}, "
        f"encoding calls={summary['encode_calls']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    run = modes.add_parser("run", help="Run one arm in a fresh process")
    run.add_argument("--strategy", choices=("incremental", "full"), required=True)
    run.add_argument("--trace", type=Path, required=True)
    run.add_argument("--model", default=DEFAULT_MODEL)
    run.add_argument("--limit", type=int, help="Use the first N blocks")
    run.add_argument("--out", type=Path, required=True)
    compare = modes.add_parser("compare", help="Pair and summarize two saved arms")
    compare.add_argument("--incremental", type=Path, required=True)
    compare.add_argument("--full", type=Path, required=True)
    compare.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "compare":
        if args.out.resolve() in {args.incremental.resolve(), args.full.resolve()}:
            parser.error("--out must differ from both input result paths")
        result = pair_results(read_json(args.incremental), read_json(args.full))
        for strategy in ("incremental", "full"):
            show(strategy, result[strategy])
    else:
        if args.limit is not None and args.limit < 1:
            parser.error("--limit must be positive")
        if args.out.resolve() == args.trace.resolve():
            parser.error("--out must differ from --trace")
        trace, trace_hash = load_trace(args.trace)
        blocks = select_blocks(trace, args.limit)
        rows, settings = run_arm(args.strategy, trace, blocks, args.model)
        result = {
            "protocol": PROTOCOL,
            "strategy": args.strategy,
            "trace_sha256": trace_hash,
            "settings": settings,
            "summary": summarize(rows),
            "rows": rows,
        }
        show(args.strategy, result["summary"])
    write_json(args.out, result)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
