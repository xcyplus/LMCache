# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import asdict
from typing import Optional
import argparse
import contextlib
import os
import time

# Third Party
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import EngineArgs

# First Party
from lmcache.integration.vllm.utils import ENGINE_NAME
from lmcache.v1.cache_engine import LMCacheEngineBuilder


def setup_environment_variables(
    use_disk: bool = False, rag_db_path: Optional[str] = None
):
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

    if use_disk:
        # Disable local CPU backend in LMCache
        os.environ["LMCACHE_LOCAL_CPU"] = "False"

        # Set the maximum size of the local CPU buffer size to 5GB
        os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "5"

        # Enable local disk backend in LMCache
        os.environ["LMCACHE_LOCAL_DISK"] = "file://local_disk/"

        # Set the maximum size of the local disk size to 10GB
        os.environ["LMCACHE_MAX_LOCAL_DISK_SIZE"] = "10"
    else:
        # Enable local CPU backend in LMCache
        os.environ["LMCACHE_LOCAL_CPU"] = "True"

        # Set the maximum size of the local CPU size to 5GB
        os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "5"


@contextlib.contextmanager
def build_llm_with_lmcache(lmcache_connector: str, model: str):
    ktc = KVTransferConfig(
        kv_connector=lmcache_connector,
        kv_role="kv_both",
    )

    llm_args = EngineArgs(
        model=model,
        kv_transfer_config=ktc,
        max_model_len=8000,
        gpu_memory_utilization=0.8,
        enable_prefix_caching=False,
    )

    llm = LLM(**asdict(llm_args))
    try:
        yield llm
    finally:
        # Clean up lmcache backend
        LMCacheEngineBuilder.destroy(ENGINE_NAME)


def print_output(
    llm: LLM,
    prompt: list[int],
    sampling_params: SamplingParams,
    req_str: str,
):
    start = time.time()
    outputs = llm.generate(prompt_token_ids=prompt, sampling_params=sampling_params)
    print("-" * 50)
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Generated text: {generated_text!r}")
    print(f"Generation took {time.time() - start:.2f} seconds, {req_str} request done.")
    print("-" * 50)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--use-disk",
        action="store_true",
        help="Specify whether to use disk as backend (default: False)",
    )

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


class RAG_Database:
    def __init__(self, json_path: str = "rag_db.json") -> None:
        """从 JSON 文件加载 RAG 数据库。

        JSON 文件格式：
            {
                "http://example.com/doc1": "文档1的文本内容...",
                "http://example.com/doc2": "文档2的文本内容..."
            }

        :param json_path: JSON 文件路径，默认 rag_db.json
        """
        import json

        self.json_path = json_path
        self.url_to_text: dict[str, str] = {}

        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                self.url_to_text = json.load(f)
        else:
            print(f"[RAG_Database] Warning: '{json_path}' not found, starting empty.")

    def get_text(self, url: str) -> Optional[str]:
        """根据 url 查询对应的文本内容。

        :param url: 要查询的文档 url
        :return: 对应的文本内容；url 不存在时返回 None
        """
        return self.url_to_text.get(url, None)


def main():
    args = parse_args()

    lmcache_connector = "LMCacheConnectorV1"
    model = args.model

    rag_db_path = args.rag_db or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "rag_db.json"
    )

    setup_environment_variables(args.use_disk, rag_db_path)

    tokenizer = AutoTokenizer.from_pretrained(model)
    rag_data_geter = RAG_Database(rag_db_path)

    with build_llm_with_lmcache(lmcache_connector, model) as llm:
        # This example script runs two requests that share non-prefix chunks
        # to test CacheBlend's content-addressing (order-independent matching).
        # 文本区间由 url 元数据（rag_db.json）定位，prompt 中不插入分隔符。
        sys_prompt = tokenizer.encode("You are a very helpful assistant.")
        query_prompt = tokenizer.encode("请用一句话总结上述城市的特点。")[1:]

        # Load two documents from RAG database
        beijing_text = rag_data_geter.get_text("www.beijing.com")
        chongqing_text = rag_data_geter.get_text("www.chongqing.com")

        beijing_prompt = tokenizer.encode(beijing_text)[1:]
        chongqing_prompt = tokenizer.encode(chongqing_text)[1:]

        # First request: [sys][beijing][chongqing][question]
        first_prompt = sys_prompt + beijing_prompt + chongqing_prompt + query_prompt

        # Second request: [sys][chongqing][beijing][question]
        # The two document chunks are swapped — CacheBlend should still hit them
        # via content-addressing rather than prefix matching.
        second_prompt = sys_prompt + chongqing_prompt + beijing_prompt + query_prompt

        sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=200)

        # Print the first output (populates L2 cache)
        print_output(llm, first_prompt, sampling_params, "first")

        time.sleep(1)

        # Print the second output (should hit CacheBlend despite swapped order)
        print_output(llm, second_prompt, sampling_params, "second")


if __name__ == "__main__":
    main()
