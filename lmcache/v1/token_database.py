# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Iterable, List, Optional, Tuple, Union
import abc
import threading
import time

# Third Party
from transformers import AutoTokenizer
import torch

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey, _lmcache_nvtx_annotate
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.doc_index import (
    ArticleCacheMetadata,
    ArticleRegistry,
)

logger = init_logger(__name__)

# NOTE: For centralized cache sharing, ensure PYTHONHASHSEED is
# set consistently across all processes (e.g., export PYTHONHASHSEED=0).
try:
    # Third Party
    from vllm.v1.core.kv_cache_utils import NONE_HASH
except ImportError:
    # Fallback to a default value if vllm is not available
    NONE_HASH = 0


class TokenDatabase(metaclass=abc.ABCMeta):
    """TokenDatabase is used to convert input tokens into list of
    cache engine keys. There are multiple ways to implement this:

    - ChunkedTokenDatabase: It processes tokens into chunks and convert
    each chunk into a cache engine key using prefix hash.

    - SegmentTokenDatabase: It processes tokens into segments based on
    special separators and convert each segment into a cache engine key.
    """

    @abc.abstractmethod
    def __init__(
        self,
        config: Optional[LMCacheEngineConfig] = None,
        metadata: Optional[LMCacheEngineMetadata] = None,
    ):
        vllm_is_available = True
        try:
            # Third Party
            from vllm.utils import sha256, sha256_cbor_64bit
        except ImportError:
            # sha256, sha256_cbor_64bit are available through vLLM only
            vllm_is_available = False

        hash_algorithm: str
        if config is not None:
            hash_algorithm = config.pre_caching_hash_algorithm
        else:  # Default value
            hash_algorithm = "builtin"  # fallback to builtin hash

        # Need to support vLLM hashing functions at a minimum
        self.hash_func = (
            sha256_cbor_64bit
            if hash_algorithm == "sha256_cbor_64bit" and vllm_is_available
            else sha256
            if hash_algorithm == "sha256" and vllm_is_available
            else hash
        )

        self.metadata = metadata

    @abc.abstractmethod
    def process_tokens(
        self,
        tokens: Optional[Union[torch.Tensor, List[int]]] = None,
        hashes: Optional[List[int]] = None,
        offsets: Optional[List[int]] = None,
        mask: Optional[torch.Tensor] = None,
        make_key: bool = True,
        request_id: Optional[str] = None,
    ) -> Iterable[Tuple[int, int, Union[CacheEngineKey, int]]]:
        """Process the tokens and return the corresponding cache engine keys.

        :param Optional[Union[torch.Tensor, List[int]]] tokens: The tokens to process.

        :param Optional[List[int]] hashes: The hashes to process. If provided,
            it will be used instead of tokens to generate cache engine keys.

        :param Optional[List[int]] offsets: The number of tokens in each chunk.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched,
            and the Falses will ALWAYS be at the PREFIX of the tensor.

        :returns: A iterable of tuples with three elements. The first element
            is the start index of the tokens for the key. The second element
            is the end index of the tokens for the key. The third element is
            the cache engine key (or hash) for the tokens.
        """

        raise NotImplementedError

    def _make_key_by_hash(self, chunk_hash: int):
        assert self.metadata is not None
        return CacheEngineKey(
            self.metadata.fmt,
            self.metadata.model_name,
            self.metadata.world_size,
            self.metadata.worker_id,
            chunk_hash,
        )

    def _hash_tokens(
        self, tokens: Union[torch.Tensor, List[int]], prefix_hash: Optional[int] = None
    ) -> int:
        if isinstance(tokens, torch.Tensor):
            tokens_tuple = tuple(tokens.cpu().tolist())
        elif isinstance(tokens, list):
            tokens_tuple = tuple(tokens)
        else:
            raise ValueError(f"Unsupported tokens type: {type(tokens)}")

        if prefix_hash is not None:
            return self.hash_func((prefix_hash, tokens_tuple))
        return self.hash_func(tokens_tuple)


class ChunkedTokenDatabase(TokenDatabase):
    def __init__(
        self,
        config: Optional[LMCacheEngineConfig] = None,
        metadata: Optional[LMCacheEngineMetadata] = None,
    ):
        super(ChunkedTokenDatabase, self).__init__(config, metadata)

        if config is not None:
            self.chunk_size = config.chunk_size
            self.save_unfull_chunk = config.save_unfull_chunk

            # Check for cross-process cache sharing setup
            # Standard
            import os

            if os.getenv("PYTHONHASHSEED") is None:
                if config.remote_url is not None:
                    logger.warning(
                        "Centralized cache sharing detected "
                        "but PYTHONHASHSEED not set. "
                        "For consistent caching, set: export PYTHONHASHSEED=0 "
                        "before the engine starts."
                    )
                if config.enable_nixl:
                    logger.error(
                        "P/D Disaggregation detected "
                        "but PYTHONHASHSEED not set. "
                        "For consistent caching, set: export PYTHONHASHSEED=0 "
                        "before the engine starts. "
                        "This will cause incorrect KV cache transfer."
                    )
        else:  # Default values
            self.chunk_size = 256
            self.save_unfull_chunk = True

    def _get_init_hash(self) -> int:
        return NONE_HASH

    def _chunk_tokens(
        self,
        tokens: Union[torch.Tensor, List[int]],
    ) -> Iterable[Union[torch.Tensor, List[int]]]:
        """
        Chunk the tokens into chunks of size self.chunk_size.

        :param tokens: the input tokens, with shape [seq_len]
            device: the target device after chunking

        :return: a generator of chunks of tokens, each with
                shape [chunk_size]
        """
        end = (
            len(tokens)
            if self.save_unfull_chunk
            else (len(tokens) - len(tokens) % self.chunk_size)
        )
        for i in range(0, end, self.chunk_size):
            yield tokens[i : i + self.chunk_size]

    def _prefix_hash(
        self,
        token_chunks: Iterable[Union[torch.Tensor, List[int]]],
    ) -> Iterable[int]:
        prefix_hash = self._get_init_hash()
        for token_chunk in token_chunks:
            prefix_hash = self._hash_tokens(token_chunk, prefix_hash)
            yield prefix_hash

    @_lmcache_nvtx_annotate
    def process_tokens(
        self,
        tokens: Optional[Union[torch.Tensor, List[int]]] = None,
        hashes: Optional[List[int]] = None,
        offsets: Optional[List[int]] = None,
        mask: Optional[torch.Tensor] = None,
        make_key: bool = True,
        request_id: Optional[str] = None,
    ) -> Iterable[Tuple[int, int, Union[CacheEngineKey, int]]]:
        """Process the tokens/hashes and return the corresponding cache engine keys.

        :param Optional[Union[torch.Tensor, List[int]]] tokens: The tokens to process.

        :param Optional[List[int]] hashes: The hashes to process. If provided,
            it will be used instead of tokens to generate cache engine keys.

        :param Optional[List[int]] offsets: The number of tokens in each chunk.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched,
            and the Falses will ALWAYS be at the PREFIX of the tensor.

        :param bool make_key: Whether to make the cache engine key or not.
            If False, the hash value will be returned instead.

        :returns: A iterable of tuples with three elements. The first element
            is the start index of the tokens for the key. The second element
            is the end index of the tokens for the key. The third element is
            the cache engine key (or hash) for the tokens.

        :raises: ValueError if the number of Falses in the mask is not a
            multiple of the chunk size.
        """
        if mask is not None:
            num_falses = mask.numel() - mask.long().sum().item()
        else:
            num_falses = 0

        if num_falses % self.chunk_size != 0:
            raise ValueError(
                "The number of Falses in the mask is not a multiple of the chunk size."
            )

        if tokens is not None:
            total_len = len(tokens)
            token_chunks = self._chunk_tokens(tokens)
            prefix_hashes = self._prefix_hash(token_chunks)
            for chunk_id, hash_val in enumerate(prefix_hashes):
                start_idx = chunk_id * self.chunk_size
                end_idx = min(start_idx + self.chunk_size, total_len)
                if start_idx < num_falses:
                    continue
                else:
                    if make_key:
                        yield start_idx, end_idx, self._make_key_by_hash(hash_val)
                    else:
                        yield start_idx, end_idx, hash_val
        elif hashes is not None:
            assert offsets is not None, (
                "If hashes are provided, offsets must also be provided."
            )
            start_idx = 0
            for hash_val, offset in zip(hashes, offsets, strict=False):
                end_idx = start_idx + offset
                if make_key:
                    yield start_idx, end_idx, self._make_key_by_hash(hash_val)
                else:
                    yield start_idx, end_idx, hash_val
                start_idx = end_idx
        else:
            raise ValueError("Either tokens or hashes must be provided.")


class SegmentTokenDatabase(TokenDatabase):
    """
    Currently, we still use special separators to identify chunks.
    In the future, we might need to implement a fast substring match.
    """

    def __init__(self, config: LMCacheEngineConfig, metadata: LMCacheEngineMetadata):
        super(SegmentTokenDatabase, self).__init__(config, metadata)

        self.tokenizer = AutoTokenizer.from_pretrained(metadata.model_name)

        # TODO (Jiayi): figure out how to decide when
        # to use `1:` (whether there's a special starting token
        # in the beginning)
        self.sep_tokens = self.tokenizer.encode(config.blend_special_str)[1:]
        self.sep_tokens = torch.tensor(self.sep_tokens, device="cpu")
        self.sep_len = len(self.sep_tokens)

    def _fast_split_by_subtensor(self, tokens: torch.Tensor) -> Iterable[torch.Tensor]:
        """Match the `sep_tokens` with sliding windows"""

        if self.sep_len == 0 or len(tokens) < self.sep_len:
            yield tokens

        # Unfold into sliding windows
        # shape: (num_tokens-sep_len+1, sep_len)
        windows = tokens.unfold(0, self.sep_len, 1)

        # Compare each window with sep_tokens
        matches = (
            (windows == self.sep_tokens).all(dim=1).nonzero(as_tuple=True)[0].tolist()
        )

        # Split based on matches
        start = 0
        for idx in matches:
            yield tokens[start:idx]
            start = idx + self.sep_len
        # yield last chunk
        yield tokens[start:]

    def process_tokens(
        self,
        tokens: Optional[Union[torch.Tensor, List[int]]] = None,
        hashes: Optional[List[int]] = None,
        offsets: Optional[List[int]] = None,
        mask: Optional[torch.Tensor] = None,
        make_key: bool = True,
        request_id: Optional[str] = None,
    ) -> Iterable[Tuple[int, int, Union[CacheEngineKey, int]]]:
        """Process the tokens and return the corresponding cache engine keys.

        :param Union[torch.Tensor, List[int]] tokens: The tokens to process.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched,
            and the Falses will ALWAYS be at the PREFIX of the tensor.

        :returns: A iterable of tuples with three elements. The first element
            is the start index of the tokens for the key. The second element
            is the end index of the tokens for the key. The third element is
            the cache engine key for the tokens.

        """

        assert isinstance(tokens, torch.Tensor), (
            "Only tokens in tensor format are supported for now."
        )
        if mask is not None:
            num_falses = mask.numel() - mask.long().sum().item()
        else:
            num_falses = 0
        assert num_falses < len(tokens), (
            "The number of Falses in the mask shouldn't "
            "be less than the length of tokens."
        )

        token_chunks = self._fast_split_by_subtensor(tokens)
        start_idx = 0
        for idx, token_chunk in enumerate(token_chunks):
            token_chunk_len = len(token_chunk)
            end_idx = start_idx + token_chunk_len
            if idx > 0:
                start_idx += self.sep_len
                end_idx += self.sep_len
                # end_idx = min(end_idx, len(tokens))
            if start_idx >= num_falses:
                if make_key:
                    yield (
                        start_idx,
                        end_idx,
                        self._make_key_by_hash(self._hash_tokens(token_chunk)),
                    )
                else:
                    yield start_idx, end_idx, self._hash_tokens(token_chunk)
            start_idx = end_idx


class UrlSegmentTokenDatabase(TokenDatabase):
    """Chunk prompts using document token spans registered by the application.

    Each span and each gap is split independently into content-hashed chunks.
    The article registry records URL ownership for incremental refresh.
    """

    def __init__(
        self,
        config: Optional[LMCacheEngineConfig] = None,
        metadata: Optional[LMCacheEngineMetadata] = None,
    ):
        super(UrlSegmentTokenDatabase, self).__init__(config, metadata)

        # chunk_size 不放在基类，这里从 config 取（与 ChunkedTokenDatabase 一致）
        self.chunk_size = config.chunk_size if config is not None else 256

        assert metadata is not None, "metadata is required for UrlSegmentTokenDatabase"
        self.tokenizer = AutoTokenizer.from_pretrained(metadata.model_name)

        self.registry = ArticleRegistry()

        # request_id -> 通过 IPC 注册的文档 token 区间
        self._request_document_spans: dict[str, tuple[tuple[int, int, str], ...]] = {}
        # url -> 累计命中次数（process_tokens 每次匹配到该 url 时 +1），
        # 供 ③ Proactive Refresh 按热度挑选后台刷新候选
        self.url_access_counts: dict[str, int] = {}
        # url -> 最近一次命中时间戳（热度 tie-break / 新鲜度参考）
        self.url_last_access: dict[str, float] = {}
        # 保护 请求文档区间 / 热度计数的并发写
        self._lock = threading.Lock()

    @_lmcache_nvtx_annotate
    def process_tokens(
        self,
        tokens: Optional[Union[torch.Tensor, List[int]]] = None,
        hashes: Optional[List[int]] = None,
        offsets: Optional[List[int]] = None,
        mask: Optional[torch.Tensor] = None,
        make_key: bool = True,
        request_id: Optional[str] = None,
    ) -> Iterable[Tuple[int, int, Union[CacheEngineKey, int]]]:
        """根据 url 元数据把 token 流切分成区间，并返回每个区间的 key。

        每个区间的 key 由区间内容单独 hash，不含前缀信息，所以位置无关。
        """
        masked_prefix_length = 0
        if mask is not None:
            masked_prefix_length = mask.numel() - mask.long().sum().item()

        if tokens is None:
            # 兼容 hashes/offsets 输入（无法定位 url，只能透传已有 hash）
            assert hashes is not None and offsets is not None, (
                "If tokens are not provided, hashes and offsets must be provided."
            )
            chunk_start = 0
            for chunk_hash, chunk_length in zip(hashes, offsets, strict=False):
                chunk_end = chunk_start + chunk_length
                if chunk_start >= masked_prefix_length:
                    if make_key:
                        yield chunk_start, chunk_end, self._make_key_by_hash(chunk_hash)
                    else:
                        yield chunk_start, chunk_end, chunk_hash
                chunk_start = chunk_end
            return

        if isinstance(tokens, torch.Tensor):
            prompt_tokens = tokens.cpu().tolist()
        else:
            prompt_tokens = list(tokens)

        # 用已匹配的文档区间和无 URL 的区间覆盖整个 prompt。
        prompt_spans: List[Tuple[int, int, Optional[str]]] = []
        previous_span_end = 0
        matched_urls: set[str] = set()
        registered_spans = (
            self._request_document_spans.get(request_id, ())
            if request_id is not None
            else ()
        )
        for span_start, registered_end, url in registered_spans:
            if span_start >= len(prompt_tokens):
                break
            span_end = min(registered_end, len(prompt_tokens))
            if span_start > previous_span_end:
                # 补上当前文档前的无 URL 文本，如系统提示或文档间的说明。
                prompt_spans.append((previous_span_end, span_start, None))
            # 加入当前文档的 token 区间，切块时按 URL 登记归属。
            prompt_spans.append((span_start, span_end, url))
            if url is not None:
                matched_urls.add(url)
            previous_span_end = span_end
        if previous_span_end < len(prompt_tokens):
            # 补上最后一篇文档后的文字；没有文档时，这段覆盖整个 prompt。
            prompt_spans.append((previous_span_end, len(prompt_tokens), None))

        # 先把所有子块算完并登记文章元数据，再开始 yield。这样即使调用方
        # 中途 break（lookup / retrieve 遇到 miss 就会），文章元数据依然完整。
        hashed_chunks: List[Tuple[int, int, int]] = []
        for span_start, span_end, url in prompt_spans:
            # 每个段（文章或空隙）内部再按 chunk_size 切分成子块，
            # 每个子块单独内容 hash，支持增量刷新与流水线并行。
            for chunk_start in range(span_start, span_end, self.chunk_size):
                chunk_end = min(chunk_start + self.chunk_size, span_end)
                # mask 前缀部分的 token 由 vLLM 自己算过，这里跳过
                if chunk_start < masked_prefix_length:
                    continue
                chunk_hash = self._hash_tokens(prompt_tokens[chunk_start:chunk_end])
                hashed_chunks.append((chunk_start, chunk_end, chunk_hash))
                if url is not None:
                    self.registry.register_chunk(url, chunk_hash)

        # 记录该请求对各 url 的命中（热度），供 ③ Proactive Refresh 调度。
        if request_id is not None and matched_urls:
            with self._lock:
                access_time = time.time()
                for url in matched_urls:
                    self.url_access_counts[url] = self.url_access_counts.get(url, 0) + 1
                    self.url_last_access[url] = access_time

        for chunk_start, chunk_end, chunk_hash in hashed_chunks:
            if make_key:
                yield chunk_start, chunk_end, self._make_key_by_hash(chunk_hash)
            else:
                yield chunk_start, chunk_end, chunk_hash

    # ------------------------------------------------------------------
    # 文章级元数据管理（仅内存）
    # ------------------------------------------------------------------

    def get_article(self, url: str) -> Optional[ArticleCacheMetadata]:
        """查询某篇文章的元数据。"""
        return self.registry.get(url)

    def get_url_by_chunk_hash(self, chunk_hash: int) -> Optional[str]:
        """查询某段 KV cache 对应的来源 url"""
        return self.registry.get_url_by_chunk_hash(chunk_hash)

    def get_url_access_count(self, url: str) -> int:
        """查询某 url 的累计命中次数（热度，供 ③ Proactive Refresh 调度）。"""
        with self._lock:
            return self.url_access_counts.get(url, 0)

    def get_hot_urls(self, top_k: Optional[int] = None) -> List[Tuple[str, int]]:
        """按热度降序返回 [(url, count)]，仅含命中过的 url。

        供主动刷新决定"哪些文档值得后台刷新"时使用：Web RAG 的 url
        访问天然近似 Zipf，热度排序即贪心调度的优先级。
        """
        with self._lock:
            items = [
                (url, cnt) for url, cnt in self.url_access_counts.items() if cnt > 0
            ]
        items.sort(key=lambda item: (-item[1], item[0]))
        if top_k is not None:
            items = items[:top_k]
        return items

    def compute_refresh_chunks(
        self, text: str, include_bos_variant: bool = True
    ) -> List[Tuple[List[int], int]]:
        """按与 refresh_article 完全一致的规则切块，返回 [(token_ids, hash)]。

        后台盲编码执行器用它定位"哪些 chunk 的 KV 需要重算"：只对 hash
        落在 fresh 集合中的 chunk 独立 prefill，未变 chunk 保持不动，
        这就是 ② Incremental Update 在后台路径上的增量红利。
        """
        ids = self.tokenizer.encode(text)
        variants: List[List[int]] = [list(ids)]
        if len(ids) > 1 and include_bos_variant:
            variants.append(list(ids[1:]))
        out: List[Tuple[List[int], int]] = []
        seen: set[int] = set()
        for v in variants:
            for i in range(0, len(v), self.chunk_size):
                chunk = v[i : i + self.chunk_size]
                h = self._hash_tokens(chunk)
                if h not in seen:
                    seen.add(h)
                    out.append((list(chunk), h))
        return out

    def refresh_article(
        self, url: str, new_text: str
    ) -> Tuple[List[int], List[int]]:
        """刷新某篇文章的内容，返回 (新增的 hash 列表, 失效的 hash 列表)。

        调用方拿到失效 hash 列表后，应主动调用 storage_manager.remove 驱逐
        对应的 KV cache。

        新增的 hash 此时尚无 KV（本函数只做失效与登记，不重算）：
        被动模式下，下次请求 lookup 这些 hash 会 miss，由请求路径重算存储；
        若配合 ③ ProactiveRefresh 在空闲窗口后台盲编码，则下次请求直接命中。
        """
        new_ids = self.tokenizer.encode(new_text)
        # 同时考虑带 BOS 与去 BOS 两种形式
        variants = [list(new_ids)]
        if len(new_ids) > 1:
            variants.append(list(new_ids[1:]))
        # 每种形式都按 chunk_size 切块，取所有子块 hash 的并集
        new_hashes = set()
        for v in variants:
            for i in range(0, len(v), self.chunk_size):
                chunk = v[i : i + self.chunk_size]
                new_hashes.add(self._hash_tokens(chunk))

        return self.registry.refresh(url, new_hashes)

    def remove_article(self, url: str) -> Optional[List[int]]:
        """删除某篇文章的元数据，返回它包含的 chunk hash 列表（供调用方驱逐 KV）。"""
        return self.registry.remove(url)

    def set_request_document_spans(
        self, request_id: str, spans: list[tuple[int, int, str]]
    ) -> None:
        previous_end = 0
        for start, end, url in spans:
            if (
                type(start) is not int
                or type(end) is not int
                or start < previous_end
                or end <= start
                or not isinstance(url, str)
                or not url
            ):
                raise ValueError("Expected ordered, non-overlapping token spans with URLs")
            previous_end = end
        with self._lock:
            self._request_document_spans[request_id] = tuple(spans)

    def remove_request_document_spans(self, request_id: str) -> None:
        with self._lock:
            self._request_document_spans.pop(request_id, None)
