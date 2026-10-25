# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Union
import abc
import json
import os
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


@dataclass
class ArticleCacheMetadata:
    """一篇文章在 LMCache 中的元数据。

    仅保存在进程内存中，与 CPU 内存 KV cache（LocalCPUBackend）同生命周期：
    进程重启后两者一起丢失，保持一致。如果 KV cache 落盘，才需要额外持久化
    本结构。
    """
    # 文章唯一标识
    url: str
    # 该文章所有 chunk 的 hash（按 chunk_size 切块后的内容 hash 列表）
    chunk_hashes: List[int]
    # 创建 / 最后一次刷新的时间戳（秒）
    created_at: int
    # 当前引用计数：有多少请求正在使用这篇文章的 KV cache。
    # 驱逐时跳过 ref_cnt > 0 的文章。
    ref_cnt: int = 0


class UrlSegmentTokenDatabase(TokenDatabase):
    """用 RAG 元数据 (url -> 文本) 定位 token 区间，不再依赖分隔符。

    工作方式：
    1. 初始化时从 `config.rag_db_path` 指向的 JSON 读入 url -> 文本，并用
       tokenizer 把每条文本编码成 token id 序列（同时保留带 BOS 与去掉
       BOS 两种形式，兼容 prompt 首段与后续段的不同处理方式）。
    2. `process_tokens` 在 token 流里做最长匹配，得到每个文本对应的
       [start, end) 区间；没有匹配上的部分整体作为一个区间。每个区间内部
       再按 `chunk_size` 切成子块，保证整条 prompt 都被覆盖，避免出现未
       加载的 KV 空洞。
    3. 每个子块的 key 只由该子块内容 hash 得到（与位置、前缀无关），因此
       同样的文档在 prompt 中换顺序后仍然能命中；文档内容更新时只有变化
       的子块失效，其余复用。
    4. 维护文章级元数据 `self.articles`（url -> ArticleCacheMetadata）和
       反向索引 `self.hash_to_url`，仅存内存，与 CPU 内存 KV cache 同
       生命周期。支持 `refresh_article` 增量刷新、`remove_article` 驱逐、
       `inc_ref/dec_ref` 引用计数。
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

        # (url, token_ids) 列表，按 token 数从长到短排序，保证最长匹配优先
        self.url_entries: List[Tuple[str, List[int]]] = []
        # chunk hash -> url，记录每段 cache 的来源文本
        self.hash_to_url: dict[int, str] = {}
        # url -> 文章元数据，仅内存，与 CPU 内存 KV cache 同生命周期
        self.articles: dict[str, ArticleCacheMetadata] = {}
        # request_id -> {url: 引用次数}（用于 inc_ref/dec_ref 成对计数）
        self._request_urls: dict[str, dict[str, int]] = {}
        # request_id -> 上层传入的有序 url 列表（引导匹配，避免全库扫描）
        self._request_input_urls: dict[str, List[str]] = {}
        # url -> 该 url 的 token 变体列表（带 BOS / 去 BOS），用于引导匹配
        self._url_variants: dict[str, List[List[int]]] = {}
        # 按首 token建索引：匹配时只查首 token 相同的条目。
        # 必须在读取 rag_db 之前初始化，否则文件缺失时 _full_scan_match 会
        # 访问不存在的属性。
        self._first_token_index: dict[int, List[Tuple[str, List[int]]]] = defaultdict(
            list
        )
        # 保护 articles / hash_to_url / _request_urls 的并发写
        self._lock = threading.Lock()

        db_path = config.rag_db_path if config is not None else None
        if db_path is None or not os.path.exists(db_path):
            logger.warning(
                "UrlSegmentTokenDatabase: rag_db_path '%s' 不存在，"
                "所有区间都会退化为按内容 hash 处理",
                db_path,
            )
            return

        with open(db_path, "r", encoding="utf-8") as f:
            url_to_text = json.load(f)

        for url, text in url_to_text.items():
            if not text:
                continue
            token_ids = self.tokenizer.encode(text)
            # NOTE: 与示例脚本保持一致：首段保留 BOS，其余段去掉 BOS
            variants: List[List[int]] = [list(token_ids)]
            if len(token_ids) > 1:
                variants.append(list(token_ids[1:]))
            self._url_variants[url] = variants
        self._rebuild_url_index()

        logger.info(
            "Loaded %d url entries from %s for UrlSegmentTokenDatabase",
            len(self.url_entries),
            db_path,
        )

    def _rebuild_url_index(self) -> None:
        """由 `_url_variants` 重建 url_entries 与首 token 索引。

        初始化阶段可不加锁；refresh_article / remove_article 调用时需持有
        `_lock`。这样文章刷新/删除后，匹配用的索引与内容保持一致。
        """
        url_entries: List[Tuple[str, List[int]]] = []
        for url, variants in self._url_variants.items():
            for variant in variants:
                if variant:
                    url_entries.append((url, list(variant)))
        url_entries.sort(key=lambda entry: len(entry[1]), reverse=True)

        # 带 BOS 的条目首 token 相同（全是 BOS），但它们只在 pos=0 参与匹配，
        # 中间位置都是去 BOS 的真实文本首 token，分布较散，过滤效果好。
        first_token_index: dict[int, List[Tuple[str, List[int]]]] = defaultdict(list)
        for url, entry_ids in url_entries:
            first_token_index[entry_ids[0]].append((url, entry_ids))

        self.url_entries = url_entries
        self._first_token_index = first_token_index

    def _match_segments(
        self, token_list: List[int], request_id: Optional[str] = None
    ) -> List[Tuple[int, int, str]]:
        """在 token 流中匹配 url 对应的区间，返回 [(start, end, url)]（按位置升序）。

        若 request_id 已通过 set_request_urls 注册了有序 url 列表，则只在这些
        url 中做引导匹配（O(K)，K 为本请求 url 数）；否则退化为全库扫描。
        """
        if request_id is not None:
            urls = self._request_input_urls.get(request_id)
            if urls is not None:
                return self._guided_match(token_list, urls)
        return self._full_scan_match(token_list)

    def _guided_match(
        self, token_list: List[int], urls: List[str]
    ) -> List[Tuple[int, int, str]]:
        """已知 url 有序列表时的引导匹配：只在这些 url 中找边界。

        由于 prompt 是各 url 文本的拼接，整体 tokenize 时边界处可能发生
        token 合并，因此不能简单按各 url 的 token 长度累加切分，而需要在
        当前位置附近的小窗口内搜索该 url 的 token 序列。
        """
        num_tokens = len(token_list)
        matches: List[Tuple[int, int, str]] = []
        pos = 0
        # 边界合并导致的偏移通常不超过几个 token，给 16 的搜索窗口足够
        search_window = 16

        for url in urls:
            variants = self._url_variants.get(url)
            if not variants:
                continue
            found = False
            window_end = min(pos + search_window, num_tokens)
            for search_pos in range(pos, window_end):
                for variant in variants:
                    vlen = len(variant)
                    if search_pos + vlen > num_tokens:
                        continue
                    if token_list[search_pos : search_pos + vlen] == variant:
                        matches.append((search_pos, search_pos + vlen, url))
                        pos = search_pos + vlen
                        found = True
                        break
                if found:
                    break
        return matches

    def _full_scan_match(
        self, token_list: List[int]
    ) -> List[Tuple[int, int, str]]:
        """全库扫描匹配（无 url 列表时的兜底）。"""
        num_tokens = len(token_list)
        matches: List[Tuple[int, int, str]] = []

        pos = 0
        while pos < num_tokens:
            # 只查首 token 相同的条目，大幅减少比较次数
            first_token = token_list[pos]
            candidates = self._first_token_index.get(first_token)
            if not candidates:
                pos += 1
                continue
            for url, entry_ids in candidates:
                entry_len = len(entry_ids)
                if pos + entry_len > num_tokens:
                    continue
                if token_list[pos : pos + entry_len] == entry_ids:
                    matches.append((pos, pos + entry_len, url))
                    pos += entry_len
                    break
            else:
                # 当前位置没有匹配到任何文本，向后移动一个 token
                pos += 1
        return matches

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
        num_falses = 0
        if mask is not None:
            num_falses = mask.numel() - mask.long().sum().item()

        if tokens is None:
            # 兼容 hashes/offsets 输入（无法定位 url，只能透传已有 hash）
            assert hashes is not None and offsets is not None, (
                "If tokens are not provided, hashes and offsets must be provided."
            )
            start_idx = 0
            for hash_val, offset in zip(hashes, offsets, strict=False):
                end_idx = start_idx + offset
                if start_idx >= num_falses:
                    if make_key:
                        yield start_idx, end_idx, self._make_key_by_hash(hash_val)
                    else:
                        yield start_idx, end_idx, hash_val
                start_idx = end_idx
            return

        if isinstance(tokens, torch.Tensor):
            token_list = tokens.cpu().tolist()
        else:
            token_list = list(tokens)

        # 把命中的文本区间和未命中的空隙拼成完整覆盖
        segments: List[Tuple[int, int, Optional[str]]] = []
        prev_end = 0
        matched_urls: set[str] = set()
        for start, end, url in self._match_segments(token_list, request_id):
            if start > prev_end:
                segments.append((prev_end, start, None))
            segments.append((start, end, url))
            if url is not None:
                matched_urls.add(url)
            prev_end = end
        if prev_end < len(token_list):
            segments.append((prev_end, len(token_list), None))

        # 先把所有子块算完并登记文章元数据，再开始 yield。这样即使调用方
        # 中途 break（lookup / retrieve 遇到 miss 就会），文章元数据依然完整。
        chunks: List[Tuple[int, int, Optional[str], int]] = []
        for start, end, url in segments:
            # 每个段（文章或空隙）内部再按 chunk_size 切分成子块，
            # 每个子块单独内容 hash，支持增量刷新与流水线并行。
            for sub_start in range(start, end, self.chunk_size):
                sub_end = min(sub_start + self.chunk_size, end)
                # mask 前缀部分的 token 由 vLLM 自己算过，这里跳过
                if sub_start < num_falses:
                    continue
                chunk_hash = self._hash_tokens(token_list[sub_start:sub_end])
                chunks.append((sub_start, sub_end, url, chunk_hash))
                if url is not None:
                    self._register_article_chunk(url, chunk_hash)

        # 记录该请求对各 url 的引用次数（每次 process_tokens 记 +1），并
        # 立刻增加文章引用计数。dec_ref 在请求结束时按次数成对扣减，避免
        # 同一 request 多次 lookup 造成计数泄漏。
        if request_id is not None and matched_urls:
            with self._lock:
                ref_counts = self._request_urls.setdefault(request_id, {})
                for url in matched_urls:
                    ref_counts[url] = ref_counts.get(url, 0) + 1
                    article = self.articles.get(url)
                    if article is not None:
                        article.ref_cnt += 1

        for sub_start, sub_end, _url, chunk_hash in chunks:
            if make_key:
                yield sub_start, sub_end, self._make_key_by_hash(chunk_hash)
            else:
                yield sub_start, sub_end, chunk_hash

    # ------------------------------------------------------------------
    # 文章级元数据管理（仅内存）
    # ------------------------------------------------------------------

    def _register_article_chunk(self, url: str, chunk_hash: int) -> None:
        """登记一个 chunk 属于某篇文章，幂等。"""
        with self._lock:
            article = self.articles.get(url)
            if article is None:
                article = ArticleCacheMetadata(
                    url=url,
                    chunk_hashes=[],
                    created_at=int(time.time()),
                )
                self.articles[url] = article
                logger.info(
                    "Registered article metadata: %s (first chunk hash %d)",
                    url,
                    chunk_hash,
                )
            if chunk_hash not in article.chunk_hashes:
                article.chunk_hashes.append(chunk_hash)
            self.hash_to_url[chunk_hash] = url

    def get_article(self, url: str) -> Optional[ArticleCacheMetadata]:
        """查询某篇文章的元数据。"""
        with self._lock:
            article = self.articles.get(url)
            if article is None:
                return None
            # 返回副本，避免外部修改内部状态
            return ArticleCacheMetadata(
                url=article.url,
                chunk_hashes=list(article.chunk_hashes),
                created_at=article.created_at,
                ref_cnt=article.ref_cnt,
            )

    def get_url_by_chunk_hash(self, chunk_hash: int) -> Optional[str]:
        """查询某段 KV cache 对应的来源 url"""
        with self._lock:
            return self.hash_to_url.get(chunk_hash)

    def refresh_article(
        self, url: str, new_text: str
    ) -> Tuple[List[int], List[int]]:
        """刷新某篇文章的内容，返回 (新增的 hash 列表, 失效的 hash 列表)。

        调用方拿到失效 hash 列表后，应主动调用 storage_manager.remove 驱逐
        对应的 KV cache；新增的 hash 在下次请求命中时自然 miss 并重算存储。
        """
        new_ids = self.tokenizer.encode(new_text)
        # 同时考虑带 BOS 与去 BOS 两种形式（与 __init__ 加载逻辑一致）
        variants = [list(new_ids)]
        if len(new_ids) > 1:
            variants.append(list(new_ids[1:]))
        # 每种形式都按 chunk_size 切块，取所有子块 hash 的并集
        new_hashes = set()
        for v in variants:
            for i in range(0, len(v), self.chunk_size):
                chunk = v[i : i + self.chunk_size]
                new_hashes.add(self._hash_tokens(chunk))

        with self._lock:
            # 先更新匹配索引，否则刷新后新文本匹配不到、旧文本仍会被命中
            self._url_variants[url] = variants
            self._rebuild_url_index()

            article = self.articles.get(url)
            if article is None:
                for h in new_hashes:
                    self.hash_to_url[h] = url
                self.articles[url] = ArticleCacheMetadata(
                    url=url,
                    chunk_hashes=list(new_hashes),
                    created_at=int(time.time()),
                )
                logger.info(
                    "Refresh article %s: not found, registered %d new hashes",
                    url,
                    len(new_hashes),
                )
                return list(new_hashes), []

            old_hashes = set(article.chunk_hashes)
            stale = list(old_hashes - new_hashes)
            fresh = list(new_hashes - old_hashes)

            # 清理失效 hash 的反向索引
            for h in stale:
                self.hash_to_url.pop(h, None)
            # 登记新增 hash
            for h in fresh:
                self.hash_to_url[h] = url

            article.chunk_hashes = list(new_hashes)
            article.created_at = int(time.time())

            logger.info(
                "Refresh article %s: %d fresh, %d stale",
                url,
                len(fresh),
                len(stale),
            )
            return fresh, stale

    def remove_article(self, url: str) -> Optional[List[int]]:
        """删除某篇文章的元数据，返回它包含的 chunk hash 列表（供调用方驱逐 KV）。"""
        with self._lock:
            # 同步移除匹配索引，否则被删除的文章仍能被后续 prompt 命中
            self._url_variants.pop(url, None)
            self._rebuild_url_index()

            article = self.articles.pop(url, None)
            if article is None:
                return None
            for h in article.chunk_hashes:
                self.hash_to_url.pop(h, None)
            logger.info(
                "Removed article metadata: %s (%d chunks)",
                url,
                len(article.chunk_hashes),
            )
            return list(article.chunk_hashes)

    def inc_ref(self, url: str) -> None:
        """文章被新请求引用时 +1。"""
        with self._lock:
            article = self.articles.get(url)
            if article is not None:
                article.ref_cnt += 1

    def dec_ref(self, url: str, count: int = 1) -> None:
        """请求结束不再引用时按引用次数扣减。"""
        with self._lock:
            article = self.articles.get(url)
            if article is not None:
                article.ref_cnt = max(0, article.ref_cnt - count)

    def get_request_urls(self, request_id: str) -> set[str]:
        """返回某请求命中的所有 url（副本）。"""
        with self._lock:
            return set(self._request_urls.get(request_id, {}))

    def set_request_urls(self, request_id: str, urls: List[str]) -> None:
        """注册某请求的有序 url 列表，启用引导匹配（避免全库扫描）。"""
        with self._lock:
            self._request_input_urls[request_id] = list(urls)

    def clear_request(self, request_id: str) -> dict[str, int]:
        """清除某请求的记录，返回 {url: 引用次数}（供 dec_ref 使用）。"""
        with self._lock:
            self._request_input_urls.pop(request_id, None)
            return self._request_urls.pop(request_id, {})
