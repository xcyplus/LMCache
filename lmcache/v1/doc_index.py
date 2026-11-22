# SPDX-License-Identifier: Apache-2.0
"""In-memory document metadata for incremental KV refresh."""
# Standard
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
