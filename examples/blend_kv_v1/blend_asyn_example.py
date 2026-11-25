# SPDX-License-Identifier: Apache-2.0
"""Async RAG example with explicit document token spans registered over IPC."""
# Standard
from typing import List, Optional
import argparse
import asyncio
import os
import tempfile
import time
from uuid import uuid4

# Third Party
from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

# First Party
from docManager import DocCacheManager, RAG_Database
from lmcache.v1.lookup_client.lmcache_lookup_client import DOCUMENT_SPANS_ENDPOINT_ENV
from lmcache.v1.storage_backend.document_kv import DOCUMENT_KV_DIR_ENV


def setup_environment_variables(rag_db_path: Optional[str] = None):
    # LMCache-related environment variables

    # LMCache is set to use 256 tokens per chunk
    os.environ["LMCACHE_CHUNK_SIZE"] = "256"

    # Blending related config
    os.environ["LMCACHE_ENABLE_BLENDING"] = "True"
    # 选择支持注册文档区间的 token database。
    if rag_db_path is not None:
        # 名称必须与 config.from_env 的 get_env_name("rag_db_path") 一致
        os.environ["LMCACHE_RAG_DB_PATH"] = os.path.abspath(rag_db_path)
    os.environ["LMCACHE_USE_LAYERWISE"] = "True"

    # Enable local CPU backend in LMCache
    os.environ["LMCACHE_LOCAL_CPU"] = "True"
    os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "5"


def build_async_engine(model: str, revision: Optional[str]) -> AsyncLLMEngine:
    ktc = KVTransferConfig(
        kv_connector="LMCacheConnectorV1",
        kv_role="kv_both",
    )

    engine_args = AsyncEngineArgs(
        model=model,
        revision=revision,
        kv_transfer_config=ktc,
        max_model_len=8000,
        max_num_batched_tokens=8000,
        max_num_seqs=1,
        gpu_memory_utilization=0.8,
        enable_prefix_caching=False,
        enforce_eager=True,
    )

    return AsyncLLMEngine.from_engine_args(engine_args)


async def print_output(
    engine: AsyncLLMEngine,
    prompt: List[int],
    sampling_params: SamplingParams,
    req_str: str,
):

    start = time.time()
    generated_text = None
    async for output in engine.generate(
        prompt={"prompt_token_ids": prompt},
        sampling_params=sampling_params,
        request_id=req_str,
    ):
        generated_text = output.outputs[0].text

    print("-" * 50)
    print(f"Generated text: {generated_text!r}")
    print(f"Generation took {time.time() - start:.2f} seconds, {req_str} request done.")
    print("-" * 50)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-2-7b-chat-hf",
        help="Path or name of the model to use",
    )
    parser.add_argument(
        "--rag-db",
        default=None,
        help="Path to the RAG database (url -> text) JSON file "
        "(default: rag_db.json next to this script)",
    )
    parser.add_argument("--revision", help="Pinned model revision for persistent KV")
    parser.add_argument("--document-kv-dir", required=True, help="SSD cache directory")
    return parser.parse_args()


async def run(
    model: str, rag_db_path: Optional[str], document_kv_dir: str,
    revision: Optional[str],
) -> None:
    rag_db_path = rag_db_path or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "rag_db.json"
    )

    setup_environment_variables(rag_db_path)

    os.environ[DOCUMENT_KV_DIR_ENV] = os.path.abspath(document_kv_dir)
    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
    rag_data = RAG_Database(rag_db_path)
    system_prompt = "You are a very helpful assistant."
    question = "请用一句话总结上述城市的特点。"
    beijing_url = "www.beijing.com"
    chongqing_url = "www.chongqing.com"
    requests = [
        ("第一次请求（北京、重庆）", [beijing_url, chongqing_url]),
        ("第二次请求（重庆、北京）", [chongqing_url, beijing_url]),
    ]

    system_tokens = tokenizer.encode(system_prompt)
    question_tokens = tokenizer.encode(question, add_special_tokens=False)
    separator_tokens = tokenizer.encode(
        "\n请参考下一段材料。\n", add_special_tokens=False
    )
    document_tokens = {
        url: tokenizer.encode(rag_data.get_text(url), add_special_tokens=False)
        for url in (beijing_url, chongqing_url)
    }
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=200)

    with tempfile.TemporaryDirectory(prefix="lmcache-spans-") as ipc_dir:
        endpoint = f"ipc://{os.path.join(ipc_dir, 'register')}"
        os.environ[DOCUMENT_SPANS_ENDPOINT_ENV] = endpoint
        doc_cache_manager = DocCacheManager(rag_data, ipc_endpoint=endpoint)
        engine = build_async_engine(model, revision)

        for request_label, ordered_urls in requests:
            prompt = list(system_tokens)
            spans = []
            for url in ordered_urls:
                if spans:
                    prompt.extend(separator_tokens)
                start = len(prompt)
                prompt.extend(document_tokens[url])
                spans.append((start, len(prompt), url))
            prompt.extend(question_tokens)

            request_id = f"rag-{uuid4().hex}"
            print(f"{request_label}，问题：{question}")
            await doc_cache_manager.register_request(request_id, spans)
            print(f"Registered {request_id}: {spans}")
            try:
                await print_output(engine, prompt, sampling_params, request_id)
            finally:
                await doc_cache_manager.unregister_request(request_id)


def main():
    args = parse_args()
    asyncio.run(run(args.model, args.rag_db, args.document_kv_dir, args.revision))


if __name__ == "__main__":
    main()
