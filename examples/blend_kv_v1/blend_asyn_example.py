# SPDX-License-Identifier: Apache-2.0
"""AsyncLLMEngine 版本的 RAG 缓存复用示例（演示引导匹配）。

相比 blend_mfo.py 用高层 API ``LLM.generate``（拿不到 request_id），本示例
改用底层的 :class:`AsyncLLMEngine`，从而能够：

1. 上层自己生成 request_id（``rag_req_1`` / ``rag_req_2``）；
2. 在提交请求前调用 ``LMCacheEngine.set_request_urls(request_id, urls)``，
   把"该请求由哪些 url 拼接而成"告诉 LMCache，启用**引导匹配**——只在这
   几个 url 里定位 token 区间（O(K)），而不是全库扫描（O(N)）。

数据流：
    set_request_urls(request_id, urls)  → 存入 _request_input_urls
    add_request(request_id=request_id)  → vLLM 用同一个 request_id
    vLLM 内部 lookup → process_tokens(request_id) → 查到 urls → 引导匹配
"""
# Standard
from typing import List, Optional
import argparse
import asyncio
import os
import time

# Third Party
from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

# First Party
from docManager import DocCacheManager, RAG_Database
from lmcache.integration.vllm.utils import ENGINE_NAME
from lmcache.v1.cache_engine import LMCacheEngineBuilder


def setup_environment_variables(rag_db_path: Optional[str] = None):
    # LMCache-related environment variables

    # LMCache is set to use 256 tokens per chunk
    os.environ["LMCACHE_CHUNK_SIZE"] = "256"

    # Blending related config
    os.environ["LMCACHE_ENABLE_BLENDING"] = "True"
    # 用 url 元数据定位文本区间，prompt 中不再插入分隔符
    if rag_db_path is not None:
        # 名称必须与 config.from_env 的 get_env_name("rag_db_path") 一致
        os.environ["LMCACHE_RAG_DB_PATH"] = os.path.abspath(rag_db_path)
    os.environ["LMCACHE_USE_LAYERWISE"] = "True"

    # Enable local CPU backend in LMCache
    os.environ["LMCACHE_LOCAL_CPU"] = "True"
    os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "5"


def build_async_engine(model: str) -> AsyncLLMEngine:
    """创建带 LMCache connector 的 AsyncLLMEngine。

    ``AsyncLLMEngine.from_engine_args`` 返回时，vLLM 内部已经初始化了
    KV connector，即 ``LMCacheEngine`` 已注册到 ``LMCacheEngineBuilder``。
    """
    ktc = KVTransferConfig(
        kv_connector="LMCacheConnectorV1",
        kv_role="kv_both",
    )

    engine_args = AsyncEngineArgs(
        model=model,
        kv_transfer_config=ktc,
        max_model_len=8000,
        gpu_memory_utilization=0.8,
        enable_prefix_caching=False,
    )

    return AsyncLLMEngine.from_engine_args(engine_args)


def get_lmcache_engine():
    """拿到 LMCacheEngine 实例（connector 初始化时已注册）。"""
    return LMCacheEngineBuilder.get(ENGINE_NAME)


async def print_output(
    engine: AsyncLLMEngine,
    prompt: List[int],
    sampling_params: SamplingParams,
    req_str: str,
):

    start = time.time()

    # 引导匹配由子进程 LMCacheLookupServer 触发：它解析 req_str 里编码的
    # url 索引并调用 engine.set_request_urls（主进程拿不到 LMCacheEngine）。
    print(f"[encode] -> {req_str!r}")

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
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="Path or name of the model to use",
    )
    parser.add_argument(
        "--rag-db",
        default=None,
        help="Path to the RAG database (url -> text) JSON file "
        "(default: rag_db.json next to this script)",
    )
    return parser.parse_args()


async def run():
    args = parse_args()

    model = args.model
    rag_db_path = args.rag_db or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "rag_db.json"
    )

    setup_environment_variables(rag_db_path)

    tokenizer = AutoTokenizer.from_pretrained(model)
    rag_data = RAG_Database(rag_db_path)
    doc_manager = DocCacheManager(rag_data)

    engine = build_async_engine(model)

    try:
        # 文本区间由 url 元数据（rag_db.json）定位，prompt 中不插入分隔符。
        sys_prompt = tokenizer.encode("You are a very helpful assistant.")
        query_prompt = tokenizer.encode("请用一句话总结上述城市的特点。")[1:]

        beijing_text = rag_data.get_text("www.beijing.com")
        chongqing_text = rag_data.get_text("www.chongqing.com")
        beijing_prompt = tokenizer.encode(beijing_text)[1:]
        chongqing_prompt = tokenizer.encode(chongqing_text)[1:]

        # First request: [sys][beijing][chongqing][question]
        first_prompt = sys_prompt + beijing_prompt + chongqing_prompt + query_prompt
        # Second request: [sys][chongqing][beijing][question]
        # 两个文档顺序对调——引导匹配 + 内容寻址下仍应命中缓存。
        second_prompt = sys_prompt + chongqing_prompt + beijing_prompt + query_prompt

        sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=200)

        # 该请求由哪几篇文档拼接而成（上游 RAG 网关本来就知道）
        # 注意：urls 必须与 prompt 里文档的实际顺序一致（引导匹配是顺序敏感）
        urls_first = ["www.beijing.com", "www.chongqing.com"]
        urls_second = ["www.chongqing.com", "www.beijing.com"]  # 换序，与 second_prompt 一致

        # 用 DocCacheManager 生成带 url 索引编码的 request_id
        first_req_id = doc_manager.generate_request_id(urls_first)
        second_req_id = doc_manager.generate_request_id(urls_second)

        # 第一个请求：填充 L2 缓存（走引导匹配）
        await print_output(engine, first_prompt, sampling_params, first_req_id)

        time.sleep(1)

        # 第二个请求：文档顺序对调，应命中缓存
        await print_output(engine, second_prompt, sampling_params, second_req_id)
    finally:
        # 清理 LMCache backend
        LMCacheEngineBuilder.destroy(ENGINE_NAME)


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()
