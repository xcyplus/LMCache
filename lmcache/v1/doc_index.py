# SPDX-License-Identifier: Apache-2.0
"""文档级 KV 缓存的索引与账本构件。

从 ``token_database.UrlSegmentTokenDatabase`` 拆出，按职责收敛为两个内聚类：

- :class:`UrlSegmentIndex` —— 匹配层。管理 ``url -> token 变体`` 并派生
  加速索引（按长度排序的条目表 + 首 token 倒排索引），在 prompt 的 token
  流里认领出每个 url 对应的区间。
- :class:`ArticleRegistry` —— 账本层。管理 ``url -> chunk hash`` 正向清单与
  ``chunk hash -> url`` 反向索引，支撑差集式增量更新。

两者都只保存在进程内存中，与 CPU 内存 KV cache 同生命周期：进程重启后
一起丢失，保持一致。若 KV cache 落盘，才需要额外持久化这些结构。
"""
# Standard
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import threading
import time

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)


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


class UrlSegmentIndex:
    """url -> token 变体的匹配索引，负责把 token 流认领回 url。

    三个字段的关系：

    - ``_variants``：事实源，url -> token 变体列表（带 BOS / 去 BOS）。
    - ``_entries``：由 ``_variants`` 展开、按 token 数降序的条目表，
      保证最长匹配优先。
    - ``_first_token_index``：首 token 倒排索引，匹配时只查首 token
      相同的条目。

    写入（``set`` / ``remove`` / ``load``）在锁内整体重建索引；读取
    （``match``）无锁——重建是"整体替换引用"，读者要么看到旧索引、要么
    看到新索引，都是一致的。
    """

    def __init__(self):
        self._variants: Dict[str, List[List[int]]] = {}
        self._entries: List[Tuple[str, List[int]]] = []
        self._first_token_index: Dict[int, List[Tuple[str, List[int]]]] = defaultdict(
            list
        )
        self._lock = threading.Lock()

    def load(self, variants_map: Dict[str, List[List[int]]]) -> None:
        """一次性载入全部 url 变体（初始化用），只重建一次索引。"""
        with self._lock:
            self._variants = {
                url: [list(v) for v in variants]
                for url, variants in variants_map.items()
            }
            self._rebuild()

    def set(self, url: str, variants: List[List[int]]) -> None:
        """更新（或新增）某 url 的变体并重建索引。"""
        with self._lock:
            self._variants[url] = [list(v) for v in variants]
            self._rebuild()

    def remove(self, url: str) -> None:
        """移除某 url 的变体并重建索引。"""
        with self._lock:
            self._variants.pop(url, None)
            self._rebuild()

    def _rebuild(self) -> None:
        """由 ``_variants`` 重建 ``_entries`` 与首 token 索引（调用方需持锁）。"""
        entries: List[Tuple[str, List[int]]] = []
        for url, variants in self._variants.items():
            for variant in variants:
                if variant:
                    entries.append((url, list(variant)))
        entries.sort(key=lambda entry: len(entry[1]), reverse=True)

        # 带 BOS 的条目首 token 相同（全是 BOS），但它们只在 pos=0 参与匹配，
        # 中间位置都是去 BOS 的真实文本首 token，分布较散，过滤效果好。
        first_token_index: Dict[int, List[Tuple[str, List[int]]]] = defaultdict(list)
        for url, entry_ids in entries:
            first_token_index[entry_ids[0]].append((url, entry_ids))

        self._entries = entries
        self._first_token_index = first_token_index

    def match(
        self, token_list: List[int], urls: Optional[List[str]] = None
    ) -> List[Tuple[int, int, str]]:
        """在 token 流中匹配 url 区间，返回 [(start, end, url)]（按位置升序）。

        ``urls`` 非空时走引导匹配（O(K)，K 为 url 数）；否则走全库扫描。
        """
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
            variants = self._variants.get(url)
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


class ArticleRegistry:
    """文章级账本：url -> chunk hash 正向清单 + chunk hash -> url 反向索引。

    这是差集式增量更新的数据基础——"旧文本哪些块失效、新文本哪些块待重算"
    都由这里的两套索引做集合减法得到。账本只记账，KV 的驱逐由调用方
    （LMCacheEngine）执行。
    """

    def __init__(self):
        self._articles: Dict[str, ArticleCacheMetadata] = {}
        self._hash_to_url: Dict[int, str] = {}
        self._lock = threading.Lock()

    def register_chunk(self, url: str, chunk_hash: int) -> None:
        """登记一个 chunk 属于某篇文章，幂等。"""
        with self._lock:
            article = self._articles.get(url)
            if article is None:
                article = ArticleCacheMetadata(
                    url=url,
                    chunk_hashes=[],
                    created_at=int(time.time()),
                )
                self._articles[url] = article
                logger.info(
                    "Registered article metadata: %s (first chunk hash %d)",
                    url,
                    chunk_hash,
                )
            if chunk_hash not in article.chunk_hashes:
                article.chunk_hashes.append(chunk_hash)
            self._hash_to_url[chunk_hash] = url

    def get(self, url: str) -> Optional[ArticleCacheMetadata]:
        """查询某篇文章的元数据（返回副本，避免外部修改内部状态）。"""
        with self._lock:
            article = self._articles.get(url)
            if article is None:
                return None
            return ArticleCacheMetadata(
                url=article.url,
                chunk_hashes=list(article.chunk_hashes),
                created_at=article.created_at,
            )

    def get_url_by_chunk_hash(self, chunk_hash: int) -> Optional[str]:
        """查询某段 KV cache 对应的来源 url。"""
        with self._lock:
            return self._hash_to_url.get(chunk_hash)

    def refresh(
        self, url: str, new_hashes: set[int]
    ) -> Tuple[List[int], List[int]]:
        """用新内容的 chunk hash 集更新账本，返回 (fresh, stale)。

        差集：``stale = 旧 − 新``（失效，需驱逐），``fresh = 新 − 旧``
        （待重算，本函数不产生 KV）。只更新账本与反向索引。
        """
        with self._lock:
            article = self._articles.get(url)
            if article is None:
                for h in new_hashes:
                    self._hash_to_url[h] = url
                self._articles[url] = ArticleCacheMetadata(
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
                self._hash_to_url.pop(h, None)
            # 登记新增 hash
            for h in fresh:
                self._hash_to_url[h] = url

            article.chunk_hashes = list(new_hashes)
            article.created_at = int(time.time())

            logger.info(
                "Refresh article %s: %d fresh, %d stale",
                url,
                len(fresh),
                len(stale),
            )
            return fresh, stale

    def remove(self, url: str) -> Optional[List[int]]:
        """删除某篇文章的元数据，返回其 chunk hash 列表（供调用方驱逐 KV）。"""
        with self._lock:
            article = self._articles.pop(url, None)
            if article is None:
                return None
            for h in article.chunk_hashes:
                self._hash_to_url.pop(h, None)
            logger.info(
                "Removed article metadata: %s (%d chunks)",
                url,
                len(article.chunk_hashes),
            )
            return list(article.chunk_hashes)
