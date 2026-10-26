# SPDX-License-Identifier: Apache-2.0
"""③ Proactive Refresh：空闲窗口内对热文档做后台盲编码刷新。

与 ② Incremental Update（LMCacheEngine.refresh_article 的差集失效）配合，
构成完整的文档级 KV cache 生命周期管理：

    文档变化事件 ──► submit_update ──► ② 同步差集失效（驱逐旧 chunk）
                                    └► ③ 排队后台重算（新增 chunk）
    空闲窗口出现 ──► refresh_pending ──► 按热度降序，逐 chunk 盲编码

盲编码：每个 chunk 的 token 从位置 0 独立 prefill，KV 经 store 路径按
内容 hash 写入。由于 chunk hash 与位置/前缀无关，请求时 lookup 的 hash
与后台写入的 hash 天然对齐——这是"主动刷新"成立的根基：后台算好的 KV，
任何后续请求都能直接命中，请求路径零重算。

调度三要素（论文中的在线调度问题的 greedy 基线）：
    what  —— 热度（url_access_counts，Web RAG 访问近似 Zipf）
    when  —— 空闲窗口（在途请求计数 == 0）
    how much —— budget_per_round（每轮最多刷新的文档数）
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from lmcache.v1.token_database import UrlSegmentTokenDatabase

logger = logging.getLogger(__name__)

# 盲编码函数：对给定 token id 做一次 standalone prefill，KV 经 LMCache
# connector 的 store 路径按内容 hash 写入（vLLM 模式下即
# llm.generate(prompt_token_ids=ids, SamplingParams(max_tokens=1))）。
EncodeFn = Callable[[List[int]], None]


@dataclass
class RefreshJob:
    """一次文档更新对应的待后台重算任务。"""

    url: str
    new_text: str
    # refresh_article 返回的新增 chunk hash 集合（需要后台盲编码的部分）
    fresh_hashes: set
    # 提交时刻该 url 的累计命中次数（调度优先级）
    hotness: int
    enqueued_at: float
    # ---- 完成后填写，供实验统计后台成本 ----
    recomputed_tokens: int = 0
    n_chunks: int = 0
    wall_time: float = 0.0


class ProactiveRefreshManager:
    """文档级 KV cache 的主动刷新管理器（③）。

    用法（vLLM 宿主）::

        def encode_fn(token_ids):
            llm.generate(prompt_token_ids=token_ids,
                         sampling_params=SamplingParams(temperature=0, max_tokens=1))

        manager = ProactiveRefreshManager(engine, encode_fn)
        manager.start()                       # 后台线程：空闲即刷新

        # 爬虫/上游系统发现网页变化时：
        manager.submit_update(url, new_text)  # ② 失效 + ③ 排队
    """

    def __init__(
        self,
        engine,
        encode_fn: EncodeFn,
        *,
        budget_per_round: int = 4,
        poll_interval: float = 1.0,
        include_bos_variant: bool = True,
        order_policy: str = "hot",
    ):
        from lmcache.v1.cache_engine import LMCacheEngine

        assert isinstance(engine, LMCacheEngine), "engine must be LMCacheEngine"
        assert isinstance(
            engine.token_database, UrlSegmentTokenDatabase
        ), "ProactiveRefreshManager requires UrlSegmentTokenDatabase"
        self.engine = engine
        self.token_database: UrlSegmentTokenDatabase = engine.token_database
        self.encode_fn = encode_fn
        # 每个空闲窗口最多刷新的文档数（后台预算）
        self.budget_per_round = budget_per_round
        # 后台线程的轮询间隔（秒）
        self.poll_interval = poll_interval
        # 是否同时重算带 BOS 变体的 chunk（文档作为 prompt 首段时的形态）
        self.include_bos_variant = include_bos_variant
        # 待刷新任务的排序策略（论文表 2 的三策略对比）：
        #   "fifo"   先来先做，忽略热度（基线）
        #   "hot"    热度降序（默认；Web RAG 访问近似 Zipf）
        #   "oracle" 按 set_oracle_ranking 注入的优先级升序（离线上界，仅实验用）
        assert order_policy in ("fifo", "hot", "oracle")
        self.order_policy = order_policy
        # url -> 优先级（数值越小越先刷新），oracle 模式由宿主注入
        self._oracle_ranking: Dict[str, float] = {}

        self._pending: Dict[str, RefreshJob] = {}
        self._active_requests = 0
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # 已完成任务（供 stats / 实验统计）
        self.completed: List[RefreshJob] = []

    # ------------------------------------------------------------------
    # 在途请求跟踪（空闲窗口检测）
    # ------------------------------------------------------------------
    def notify_request_begin(self) -> None:
        """请求进入系统时调用（vLLM 宿主可在 scheduler 回调里挂接）。"""
        with self._lock:
            self._active_requests += 1

    def notify_request_end(self) -> None:
        """请求离开系统时调用。"""
        with self._lock:
            self._active_requests = max(0, self._active_requests - 1)

    @property
    def is_idle(self) -> bool:
        """空闲窗口 = 没有在途请求（后台刷新不会与在线推理争抢算力）。"""
        return self._active_requests == 0

    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    def set_oracle_ranking(self, ranking: Dict[str, float]) -> None:
        """注入 oracle 优先级（url -> 数值越小越先刷新）。

        仅 order_policy="oracle" 时生效。宿主从未来请求轨迹离线构造
        （例如"该文档下一个 post 请求出现的位置"），用于测量完美预测
        的可达上界——论文表 2 中热度贪心与它的差距即学习型预测器的
        收益空间。
        """
        self._oracle_ranking = dict(ranking)

    # ------------------------------------------------------------------
    # 文档更新入口：② 差集失效（同步）+ ③ 排队后台重算
    # ------------------------------------------------------------------
    def submit_update(self, url: str, new_text: str) -> Tuple[List[int], List[int]]:
        """网页/文档内容变化事件。

        1) 同步调用 engine.refresh_article：更新匹配索引、计算新旧 chunk
           hash 差集、驱逐失效 chunk（② Incremental Update）；
        2) 若存在新增 chunk，把重算任务排入后台队列，等空闲窗口按热度
           执行（③ Proactive Refresh）。

        返回 (fresh_hashes, stale_hashes)。
        """
        fresh, stale = self.engine.refresh_article(url, new_text)
        if fresh:
            hotness = self.token_database.get_url_access_count(url)
            with self._lock:
                # 同一 url 再次更新时覆盖旧任务（差集以最新内容为准）
                self._pending[url] = RefreshJob(
                    url=url,
                    new_text=new_text,
                    fresh_hashes=set(fresh),
                    hotness=hotness,
                    enqueued_at=time.time(),
                )
            logger.info(
                "ProactiveRefresh: enqueued %s (%d fresh chunks, hotness=%d)",
                url,
                len(fresh),
                hotness,
            )
        else:
            # 内容没有产生新 chunk（例如更新已被覆盖），撤下旧任务
            with self._lock:
                self._pending.pop(url, None)
        return fresh, stale

    # ------------------------------------------------------------------
    # 空闲窗口执行：按热度选任务，只盲编码 fresh chunk
    # ------------------------------------------------------------------
    def refresh_pending(self, force: bool = False) -> List[RefreshJob]:
        """执行一轮后台刷新。

        非 force 模式下仅当空闲（无在途请求）时执行；force=True 跳过
        空闲检查（实验中用于精确控制时机）。

        返回本轮完成的任务列表（含 recomputed_tokens / wall_time）。
        """
        if not force and not self.is_idle:
            logger.debug("ProactiveRefresh: busy window, skip this round")
            return []

        with self._lock:
            if not self._pending:
                return []
            # 排序策略：表 2 的三策略对比（oracle 差距 = 学习型预测器的收益上界）
            if self.order_policy == "fifo":
                key_fn = lambda job: job.enqueued_at
            elif self.order_policy == "oracle":
                key_fn = lambda job: self._oracle_ranking.get(job.url, float("inf"))
            else:  # hot：热度降序 + 先到先做
                key_fn = lambda job: (-job.hotness, job.enqueued_at)
            candidates = sorted(self._pending.values(), key=key_fn)[
                : self.budget_per_round
            ]

        reports: List[RefreshJob] = []
        for job in candidates:
            t0 = time.time()
            # 切块计划与 refresh_article 的 hash 规则完全一致；
            # 只对 hash 落在 fresh 集合中的 chunk 做盲编码（增量：未变 chunk 不动）
            chunk_plan = self.token_database.compute_refresh_chunks(
                job.new_text, self.include_bos_variant
            )
            n_tokens = 0
            n_chunks = 0
            for chunk_tokens, chunk_hash in chunk_plan:
                if chunk_hash in job.fresh_hashes:
                    self.encode_fn(chunk_tokens)
                    n_tokens += len(chunk_tokens)
                    n_chunks += 1
            job.recomputed_tokens = n_tokens
            job.n_chunks = n_chunks
            job.wall_time = time.time() - t0

            # 若执行期间同一 url 又被更新，保留新任务，不误删
            with self._lock:
                if self._pending.get(job.url) is job:
                    self._pending.pop(job.url, None)
                self.completed.append(job)
            reports.append(job)
            logger.info(
                "ProactiveRefresh: refreshed %s (%d chunks / %d tokens in %.2fs)",
                job.url,
                n_chunks,
                n_tokens,
                job.wall_time,
            )
        return reports

    # ------------------------------------------------------------------
    # 后台线程模式：轮询空闲窗口并自动刷新
    # ------------------------------------------------------------------
    def start(self) -> None:
        """启动守护线程：每隔 poll_interval 检查一次，空闲且有积压即刷新。"""
        if self._thread is not None and self._thread.is_alive():
            return

        def _loop():
            while not self._stop_event.wait(self.poll_interval):
                if self._pending and self.is_idle:
                    self.refresh_pending(force=True)

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=_loop, name="proactive-refresh", daemon=True
        )
        self._thread.start()
        logger.info("ProactiveRefresh: background thread started")

    def stop(self) -> None:
        """停止后台线程（进程退出前调用；守护线程下可省略）。"""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    # ------------------------------------------------------------------
    # 统计（监控 / 实验记录）
    # ------------------------------------------------------------------
    def stats(self) -> dict:
        """汇总统计：在途请求数、积压任务、累计后台重算成本。"""
        with self._lock:
            pending = {
                url: {"fresh_chunks": len(job.fresh_hashes), "hotness": job.hotness}
                for url, job in self._pending.items()
            }
            active = self._active_requests
        return {
            "active_requests": active,
            "pending": pending,
            "completed_jobs": len(self.completed),
            "total_recomputed_tokens": sum(j.recomputed_tokens for j in self.completed),
            "total_refresh_wall_time": sum(j.wall_time for j in self.completed),
        }
