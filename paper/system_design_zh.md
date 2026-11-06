# DocCache 详设：存储、增量更新与主动刷新（含实验设计）

> 这份文档是 `paper_zh.md` 的施工图。论文讲"为什么这么设计"，这里讲"代码具体怎么做、实验具体怎么跑"。
> 全文用同一个例子贯穿：一篇关于北京的词条（约 2048 token），某天其中一句人口数据被官方更新了。
>
> 对应代码：
> - ① 存储：`lmcache/v1/token_database.py` 的 `UrlSegmentTokenDatabase`
> - ② 增量更新：同文件的 `refresh_article`，外加 `lmcache/v1/cache_engine.py` 的同名入口
> - ③ 主动刷新：`lmcache/v1/proactive_refresh.py` 的 `ProactiveRefreshManager`
> - 实验脚本：`examples/blend_kv_v1/` 下四个文件

---

## 0. 系统要解决什么问题

RAG 服务的用户请求长这样：

```
prompt = [系统提示] + [几篇网页正文] + [用户问题]
```

同一批网页会被无数请求反复引用，所以把它们算出的 KV 缓存下来、下次直接复用，不用重新 prefill。CacheBlend 这类技术已经解决了"缓存怎么跨请求、跨位置复用"，本系统直接站在它上面。

但它们都默认一件事：**网页永远不变**。现实里北京词条今天写着"常住人口 2180 万"，明天改成"2210.8 万"。这时候缓存里那份旧 KV 就成了三不管的东西：

- 没人知道缓存里哪几块 KV 是北京词条的（缓存里只有一串匿名 hash）；
- 不敢删也不知道删哪几块，要么全扔（浪费），要么留着（过期）；
- 就算知道要重算，什么时候算？用户请求来了才算，高峰期就得排队。

三个模块各管一件事：

| 模块 | 管什么 | 一句话 |
|---|---|---|
| ① URL-aware Storage | KV 怎么组织 | 每块缓存登记"我来自哪篇网页" |
| ② Incremental Update | 网页变了怎么办 | 只废掉变了的块，没变的不动 |
| ③ Proactive Refresh | 什么时候补算 | 趁系统空闲，后台把新块的 KV 提前算好 |

---

## 1. 一条铁律：缓存的 key 只看内容，不看位置

后面所有设计都建立在这一条上，先把它讲透。

一个 chunk（256 个 token 的一段）的 key，就是**这 256 个 token 本身的 hash**。不带位置，不带前缀，不带任何别的东西。

```
北京词条的第一块:  [北京, 是, 中华, 人民, ...]     → key = hash(这256个token)
它出现在哪个 prompt 里、排第几段，都不影响这个 key
```

对比一下：vLLM 自带的前缀缓存，key 是"从头到这里所有内容的 hash 链"——同一个文档换个位置，hash 就变了。所以那种缓存只能复用前缀，我们这种可以任意换序。

这条铁律带来三个直接后果：

1. 同一篇文档不管拼在 prompt 的开头、中间还是结尾，key 都一样 → 换序照样命中；
2. 后台可以单独把一个 chunk 从位置 0 算一遍存起来，将来任何请求来查，hash 对得上 → 这就是 ③ 能"提前算"的全部原理；
3. 网页改了之后，改过的地方 token 变了 → hash 必然变 → 新内容永远不会错误地命中旧缓存。过期最坏也就是"慢一下"（miss 了重算），绝不会"答错"。论文里叫**构造性新鲜保证**。

代价也有：每个 chunk 是从位置 0 独立算的，它没见过自己前面的内容，RoPE 位置也对不上真实位置。这个误差由 CacheBlend 的加载期选择性重算（重算边界附近 ~15% 的 token）来修补，本文不重复造这个轮子。

---

## 2. ① 存储：每块缓存都要有"户口"

### 2.1 语料库长什么样

一个 JSON 文件（`rag_db.json`），启动时通过 `LMCACHE_RAG_DB_PATH` 加载：

```json
{
  "www.beijing.com": "北京是中华人民共和国首都……常住人口2180.0万人……",
  "www.shanghai.com": "上海是中国最大的经济中心城市……",
  "sys://": "You are a very helpful assistant.",
  "query://population": "北京市现在的常住人口是多少？"
}
```

`UrlSegmentTokenDatabase.__init__` 把每条文本 encode 成 token 序列，存进 `_url_variants`。注意名字里的 "variants"（复数）——**每篇文档存两套 token 序列**，原因见 2.4。

### 2.2 请求来了，怎么知道哪段 token 是哪篇文档

用户 prompt 是各段文本直接拼起来的（不加分隔符，加分隔符会扰动模型输入分布）。请求到达后 vLLM 调 `engine.lookup`，最终走到 `process_tokens`，第一步是把 token 流"认领"回各篇文档——`_match_segments`。

它有两种走法：

**走法 A：全库扫描**（默认，实验脚本目前用的就是这条）。逐位置扫 token 流，在 `_first_token_index`（首 token 倒排索引）里查当前位置的 token 有没有文档以它开头，有就逐个比对全文。`url_entries` 按文档长度降序排，所以命中时是**最长匹配优先**。

**走法 B：引导匹配**（生产推荐）。上游检索器本来就知道这次请求用了哪几篇文档，提前调 `set_request_urls(request_id, [url1, url2])` 注册，`process_tokens` 拿到 request_id 就走 `_guided_match`：只在这几个 url 里找边界，O(K)，K 通常 2-5。

引导匹配里有个不起眼但关键的细节：**token 边界会合并**。两段文本分开 encode 各得 100 个 token，直接拼起来再 encode 可能只有 198 个（衔接处两个 token 被合并成一个）。所以不能按"各文档长度累加"来切分，代码里在预期位置附近开了个 **16-token 的搜索窗**，逐位比对找到真实边界。

匹配完，token 流被切成两类区间：认领到 url 的（如 `[9, 521) = www.beijing.com`），和谁也没认领的空隙（如系统提示 `[0, 9)`、问题 `[521, 542)`）。**两类都会继续切块**，空隙也一样 hash 一样缓存，保证整个 prompt 无死角覆盖。

### 2.3 切块与记账

每个区间内部按 `chunk_size = 256` 切：

```
北京词条 512 token → [9, 265) 和 [265, 521) 两块
每块独立算 hash → key → 逐 key 查存储
```

在产出 key 之前，`process_tokens` 还顺手做了三件记账的事（数据结构都在内存里，和 CPU 上的 KV 缓存同生共死）：

- `_register_article_chunk`：往 `articles[url]` 里追加这个 hash，往 `hash_to_url[hash]` 里登记反向索引。**这就是给缓存块上户口**——② 的差集全靠它；
- `url_access_counts[url] += 1`：这篇文档被引用了一次。**这就是 ③ 排队用的热度**；
- 文章的 `ref_cnt += 1`：当前有请求在用这篇文档。请求结束时 `lookup_unpin → clear_request` 会按次数成对扣减，有引用的文档在驱逐时被跳过。

有个实现细节值得知道：这些记账发生在**所有 key 算完之后、逐个 yield 出去之前**。因为调用方（lookup）遇到 miss 会提前 break，如果边算边记账，break 之后的块就丢账了。所以代码是"先算完全部、登记好，再开始吐 key"。

### 2.4 为什么每篇文档存两套 token（BOS 变体）

tokenizer 单独编码一段文本时会在开头加一个特殊 token（BOS）。于是同一个文档有两种形态：

```
单独编码:      [BOS, 北京, 是, ...]        ← 文档作为 prompt 第一段时
拼在中间:      [系统提示..., 北京, 是, ...]  ← BOS 没了
```

第一种形态的第一块内容是 `[BOS, 北京, 是, ...]`，第二种是 `[北京, 是, ...]`——**内容不同，hash 必然不同**。所以初始化和 `refresh_article` 都同时登记两种形态的 chunk hash，`include_bos_variant` 参数贯穿 ①②③，任何一处规则不一致就会出现"一边存了另一边查不到"的静默 miss。

---

## 3. ② 增量更新：网页变了，只重算变的部分

### 3.1 refresh_article 走一遍

网页被编辑，上游调 `engine.refresh_article(url, new_text)`。它做四件事（token_database 里算账，cache_engine 里动手）：

**第一步，算新账本。** 把新文本 encode、按老规矩切块（两种 BOS 形态都算），得到新 hash 全集。

**第二步，先换索引再算差集（顺序不能反）。** 持锁更新 `_url_variants[url]` 并 `_rebuild_url_index()`。这一步做完，新文本立刻可以被请求匹配到，旧文本从此匹配不上——如果先算差集再换索引，中间有个窗口：请求既能匹配旧文本、旧 KV 又可能已经被删了。

**第三步，做集合减法。**

```
旧账本 articles[url].chunk_hashes = {H1, H2, H3, H4, H5, H6, H7, H8}   ← 原文 2048 token，8 块
新账本                              = {H1, H2, H3, H4', H5', ..., H9'} ← 改完 2048+，9 块

stale = 旧的 − 新的 = {H4, H5, H6, H7, H8}   ← 失效，删
fresh = 新的 − 旧的 = {H4', ..., H9'}        ← 待算，先不算
没变的 H1, H2, H3 ∈ 交集                     ← 原样保留，下次照样命中
```

**第四步，动手删。** cache_engine 层拿到 stale 列表，立刻逐个 `storage_manager.remove`（layerwise 模式下把 key 按层拆开删）。删除是同步的，update 事件返回时存储里已经没有失效块了。

注意 `fresh` **只是个待办清单**，这一步不算 KV。什么时候算、怎么算，是 ③ 的事。② 只管账本，不管施工。差集本身是两个 set 相减，2048 token 的文档也就 8-10 个 hash，微秒级，忽略不计。

### 3.2 一个反直觉的现象：级联失效

上面例子里旧 8 块废了 5 块，但实际编辑可能只改了一句话（几十个 token）——为什么废了这么多？

因为编辑是**变长**的。在文档中部插入/替换了一小段，插入点之后所有 token 的位置都往后挪了。固定切块的边界是"每数够 256 切一刀"，位置一挪，刀就切歪了：

```
原文:    [a b c d][e f g h][i j k l][m n o p]      4 块，key 分别是 hash(abcd) ...
中部把 c 改成长句:
新文:    [a b ? ? ? ? ?][? h i j][k l m n][o p ...] 每块内容全变了
```

第 2 块往后，内容全变了 → hash 全变了 → 全进 fresh。**编辑 50 个 token，废掉 1000+**，这就是级联失效。我们没打算在论文里假装它不存在（表 3 会诚实量化它），但准备了一个预案把它收紧到编辑邻域——只有验收标准 2 不过线时才启用。

### 3.3 预案：CDC（内容定义切块）

想法来自 rsync/LBFS：**别按"数够 256"切，按内容切**。对每个位置算一个 8-token 滑窗的滚动 hash，hash 值满足条件（比如 `hash % 192 == 0`）的地方就切一刀——这种位置叫**锚点**。

```
原文:    [a b c d][e f g h][i j k l][m n o p]     d、h、l、p 恰好是锚点
中部插入 XY 后:
固定切:  边界整体滑移，插入点之后全废            ← 现状
CDC 切:  [a b ?][? ?][i j k l][m n o p]          l、p 还是锚点（内容没变，
                                                   hash 就不变），边界自动对齐
         只有插入点附近的 1-2 块是新的
```

关键在于锚点是**内容的函数，不是位置的函数**：插入搅乱的只是插入点附近那约 100 个滑窗，远处的滑窗内容原封不动，锚点纹丝不动。改造点只有一处——把 ② 的 `refresh_article` 和 `compute_refresh_chunks` 里"每 256 切一刀"换成滚动 hash 切刀，key 结构、存储、③ 的接口全部不动（hash 仍然是内容的纯函数，铁律没破）。

参数是三件套：`min=64`（两个锚点太近就不切，防碎块）、`target=192`（锚点密度，块平均长度）、`max=256`（到顶强制切，单块太长则盲编码的 O(L²) 注意力和拷贝 buffer 都撑不住）。

上线注意：切块规则一变，所有旧 hash 全部作废，必须清空缓存目录冷启动。

---

## 4. ③ 主动刷新：趁没人的时候把账算了

### 4.1 一次完整的后台刷新

`ProactiveRefreshManager` 是个带后台线程的小调度器。完整流程以北京词条为例：

```
14:00  爬虫发现北京词条更新了
       manager.submit_update("www.beijing.com", 新文本)
         ├─ 内部先调 ② 的 refresh_article → 差集算好，stale 已删
         └─ fresh 非空 → 包成一个 RefreshJob 存进 _pending 队列
              （同一个 url 再次更新会覆盖旧任务，账本以最新为准）

14:05  最后一个用户请求结束，_active_requests 归零（空闲窗口）
       后台线程（或实验里手动调 refresh_pending）开始干活：
         1. 从队列按当前策略挑出最多 budget_per_round 个任务
         2. 对每个任务：compute_refresh_chunks(新文本)
              → 得到 [(token序列, hash), ...]（和 ② 完全同一套切块规则）
         3. 对 hash 落在 fresh 集合里的块：
              encode_fn(token序列)  →  llm.generate(prompt_token_ids=这256个token,
                                                     max_tokens=1)
              这次 prefill 算出的 KV 经 connector 的 store 路径，
              按 hash 写进存储 —— 和请求路径用的是同一个 store

14:06  完工。job 移出队列，记录重算了几个块、多少 token、花了多久
```

为什么后台算的，用户请求就能直接命中？回到 §1 的铁律：后台把 `[北京, 是, 中华, ...]` 这 256 个 token 从位置 0 算了一遍，key = hash(这 256 个 token)。将来某个用户的 prompt 里恰好包含这 256 个 token（不管在 prompt 的第几段），`process_tokens` 切出来的块内容一模一样，hash 一模一样，直接命中。**后台不用预知谁会来请求、请求长什么样**。

### 4.2 调度器要回答的三个问题

- **算哪些（what）**：`url_access_counts` 里的累计命中数当热度。RAG 的网页访问天然近似 Zipf（少数文档被反复引用），热度排序就是个不错的贪心；
- **什么时候算（when）**：`_active_requests == 0`，即空闲窗口。请求进出时宿主调 `notify_request_begin/end` 维护这个计数。不空闲就跳过本轮，绝不跟在线流量抢卡；
- **一轮算多少（how much）**：`budget_per_round`（默认 4 篇/轮），防止后台一口气把积压全算完、把下一波请求堵在门外。

排序策略是个可换的插头（`order_policy`），论文表 2 对比三种：

| policy | 排序依据 | 意义 |
|---|---|---|
| `fifo` | 先来先做 | 基线：完全不看热度 |
| `hot` | 热度降序 | 默认贪心 |
| `oracle` | 宿主注入的"未来顺序" | 作弊上界：离线翻轨迹，按每篇文档下一个请求出现的位置排。hot 和它差多少，就说明"更好的预测模型"还有多少肉 |

### 4.3 竞态怎么处理

- **算到一半，同一 url 又更新了？** `submit_update` 会用新 job 覆盖 `_pending[url]`。执行完出队时，代码先检查 `_pending.get(url) is job`——不是自己（说明被新任务覆盖了）就不删，新任务完好保留；
- **记账冲突？** token_database 内部一把锁管 `articles / hash_to_url / _request_urls`，匹配索引是写时整体重建、读侧不加锁；
- **和请求撞车？** 空闲窗口检测本身就是互斥：不 idle 一轮都不跑。

### 4.4 预案：并发调度实验

当前实验是**串行回放**：事件一个接一个，事件之间在途请求永远是 0，"空闲窗口"从来没有真正工作过——调度器没被考验过，审稿人会问"你的'何时重算'是不是形同虚设"。

如果表 1 里 proactive 和 passive 的 TTFT 差距淹没在噪声里（验收标准 1 失败），就启用这个预案：实验脚本把逐个 `llm.generate` 改成批量提交、更新事件穿插在请求之间、启用 `manager.start()` 守护线程——后台刷新和在线流量**真实抢卡**。这时两个被掩盖的效果才显形：

- passive 的重算发生在请求路径上，高峰期要跟别的请求排队，TTFT 劣化被并发放大；proactive 在波谷已经算完了；
- 空闲窗口检测、窗口选择、SLO 干扰都变成真问题，"何时算"从演示变成测量。

改动全在实验脚本层，核心代码零改动（守护线程模式早就写好了，只是实验没用过）。

---

## 5. 端到端：三个模块串起来的完整时间线

```
T0  服务启动，rag_db.json 加载，_url_variants / 首token索引建好
T1  预热：对每篇文档调 encode_fn 盲编码，全部文档的 KV 入库，
    articles 账本随之登记（此刻没有用户请求）
T2  用户 A 问"北京人口多少"：
    prompt = [sys][北京词条][问题]
    process_tokens 匹配 → 切块 → 全命中 → TTFT 短
    （miss 的部分照常 prefill，算出的 KV 顺路写回缓存）
T3  爬虫报告北京词条更新：
    submit_update → ② 差集：stale(5块) 立刻删，fresh(6块) 排队
T4  用户 B 恰好这时问同一个问题：
    - norefresh 策略：旧缓存还在但根本匹配不上 → 整篇重算，TTFT 差
    - passive 策略：只剩 3 块命中，fresh 的 6 块在请求路径上补算
    - proactive 策略（若后台已完成）：新文本的全部块已在库 → 全命中
T5  空闲窗口出现，后台把 fresh 算完
T6  之后所有引用北京词条的请求：全命中，直到下一次更新
```

用户 B 看到的差异就是论文的 headline：同一时刻、同一个请求，三种策略下 TTFT 差多少。

---

## 6. 实验设计

### 6.1 整体结构

```
HotpotQA（distractor, validation）
   │
   │ trace_builder.py          ← 离线构造，不需要 GPU
   ▼
trace.json（文档库 + 请求/更新事件流） + rag_db.json
   │
   │ run_trace_experiment.py   ← 回放 + 计量，一个策略一个进程
   ▼
results_{policy}[_{order}].json  →  --summarize 汇总  → 论文表 1/2/3

refresh_experiment.py           ← 独立的单文档消融（论文 motivation 数据）
```

为什么**一个策略一个进程**：存储是进程内状态，跑完策略 A 再跑策略 B，B 的冷启动指标全被 A 污染。进程隔离是唯一干净的做法。

为什么每个请求都要**提前盘点缓存状态**（`run_trace_experiment` 在请求发出前调 `compute_refresh_chunks` 逐块查 `chunk_in_storage`）：命中率和 miss token 数是论文的核心论据，必须是对存储层的直接核实，不能从 vLLM 日志倒推。

### 6.2 轨迹怎么造（trace_builder.py）

**语料**：HotpotQA 每道题有 10 个维基段落，全拼成一篇长文档（约 500-2000 token，跨 2-8 个 chunk），url 形如 `hotpot://{qid}`。筛题规则（`pick_editable`）：非 yes/no、答案不超过 8 个词、答案字符串在文中真实出现。

**怎么模拟"网页更新"**：直接把文档里的 gold answer 字符串替换成一个生造的词 **Zyxoria**（方法论承自 Budgeted Repair 的配对探针）。这样不用 LLM 造题，判定也全自动——

```
pre 阶段（更新前）的请求：答案应该是 gold       → 算 F1/EM
post 阶段（更新后）的请求：答案应该翻转成 Zyxoria → 翻转率
```

**每题的事件流**：

```
pre × 2（第 1 个生成答案、第 2 个是全命中基线）
→ update × 1（gold → Zyxoria）
→ post × 2（第 1 个生成答案 = headline 指标，第 2 个看缓存收敛后）
```

四个旋钮，各服务一张论文表：

| 旋钮 | 取值 | 控制 | 给谁用 |
|---|---|---|---|
| `--granularity` | question / paragraph | 文档长短（长文档级联效应放大，短文档收益收窄） | §5.5 消融 |
| `--edit-mode` | mid_len / mid_iso / tail | 编辑类型 | 表 3 |
| `--request-skew` | 0 / >0 | pre 请求按 Zipf 倾斜（头部题更多请求 → 热度分化） | 表 2 |
| `--schedule` | block / phase | 事件排布 | 表 1 用 block，表 2 用 phase |

### 6.3 实验 1：四策略主对比（表 1）

**问题**：② 和 ③ 各自值多少钱？

四个策略经历完全相同的事件流，唯一的区别是**更新事件之后、下一个请求之前，系统做了什么**：

| 策略 | 更新后做了什么 | 预期 post/0 首请求 |
|---|---|---|
| norefresh | 什么都不做 | 最差：旧缓存匹配不上，整篇重算 |
| passive | 只做 ②（差集+驱逐） | 中等：只补算 fresh 的部分 |
| proactive | ② + ③（后台补齐） | 最好：全命中，≈ pre/1 基线 |
| proactive_full | 整篇删掉 + 后台全量重算 | 最好（但后台花整篇的算力） |

两组对比各回答一个问题：

- **proactive vs norefresh / passive**：③ 把重算从请求路径挪到空闲后台，用户拿到多少 TTFT 收益；
- **proactive vs proactive_full**：都做到了全命中，但后台一个只算 fresh（几块）、一个算整篇（8-10 块）——**差值就是 ② 增量差集省下的算力**。

回填数据时按预期排序核对，异常先查实验：post/0 未命中 token 应为 norefresh≈整篇、passive≈|fresh|、proactive=0；四种策略翻转率都应接近 100%（差了就说明有 KV 污染 bug，这是铁律的自动报警）。

### 6.4 实验 2：调度策略对比（表 2）

**问题**：③ 的"何时算、先算谁"是真本事还是摆设？

直接用表 1 的跑法会测不出排序差异，有两个必须的设计决策：

1. **`--schedule phase`**：事件重排成"全部 pre → 全部 update → 全部 post"。默认的 block 模式下每题自成一组，更新之后马上就有 post 请求把积压消化掉了，排序来不及起作用；phase 模式下 N 篇文档**同时**进入待刷新队列，"先刷谁"才开始有意义；
2. **`--bg-budget 1`**：每轮只准刷一篇。预算太宽裕就人人有份，排序没有区分度。

配上 `--request-skew 1.0`（pre 请求按 Zipf 倾斜，热度计数自然分化），三种排序策略在同一个轨迹上各跑一个进程：

```
--order fifo    先来先做（不看热度）
--order hot     热度贪心（默认）
--order oracle  离线翻轨迹，按"每篇文档下一个 post 请求的位置"排（完美预测上界）
```

预期：负载越倾斜，hot 越接近 oracle（热度信号在强 Zipf 下足够好），fifo 停在高 miss。hot 与 oracle 的差距 = "上学习型预测器"还能再赚多少——这是贡献二的讨论素材。另外补一组预算敏感性（B ∈ {1,2,4,8}）。

### 6.5 实验 3：编辑类型消融（表 3）

**问题**：② 的差集到底省了多少？级联失效（§3.2）有多痛？

三种编辑模式，各自的构造方式和要隔离的变量：

| edit-mode | 怎么改 | 隔离什么 | 预期失效范围 |
|---|---|---|---|
| `tail` | 只把 gold 的最后一次出现换掉（没有就文末补一句） | 无滑移的编辑 | 只有最后一块 |
| `mid_iso` | 只换第一处，用 `x x x` 填充把 **token 数补齐**到和 gold 一样长 | 内容变了但边界不滑移 | 编辑点所在的一块 |
| `mid_len` | 把所有 gold 出现处全换掉（长度变化） | 真实编辑：边界滑移 | 编辑点之后的全部（级联） |

三者的 fresh 块数依次拉开：tail 是下界，mid_iso 是"没有级联时的本征失效"，**mid_len 减 mid_iso 就是级联效应的单独贡献**。这张表直接支撑论文 §5.4 对级联的诚实量化，也是 §3.3 CDC 预案的动机数据。

### 6.6 实验 4：单文档消融（refresh_experiment.py，论文 motivation）

**问题**：给 §2.2 的病理故事配上真实数字。

脱离 HotpotQA，手工构造一个语料：目标文档 `www.beijing.com`，把其中一句人口事实换成更长的官方新表述（变长中部替换，触发真实级联）。四臂和表 1 相同，额外做了三件事：

- **每臂干净起点**：`reset_doc` 三步清场——先 `refresh_article`（差集把上一代驱逐干净）→ 再 `remove_article`（兜底清存储残留）→ 重新注册并全量盲编码目标文本；
- **空闲窗口的显式演示**：`notify_request_begin()` 后调 `refresh_pending()` 断言它跳过（忙时不干活）→ `notify_request_end()` 后再调才真正执行。这段是给读者/审稿人看的机制演示，不是测量；
- **新鲜度自动判定**：`flip_ok` 检查答案里有没有"2210"（新数字），四臂都应为 True。

产出字段：`ttft_probe_s`（更新后首请求）、`bg_tokens / bg_time_s`（后台成本）、`fresh / stale chunks`（差集规模——级联的直接证据）、`answer_time_s`（缓存收敛后应各臂接近）。

### 6.7 指标口径汇总

| 指标 | 是什么 | 在哪测、怎么测 |
|---|---|---|
| `prefill_s` | 请求的 TTFT 代理 | `max_tokens=1` 的 generate 墙钟，temperature=0 |
| `doc_hit_chunks / doc_total_chunks` | 命中块数/总块数 | 请求**发出前**，`chunk_in_storage` 直查存储（layerwise 逐层查） |
| `doc_miss_tokens` | 挪到请求路径的重算量 | 未命中块的 token 求和 |
| `bg_tokens / bg_time_s` | 后台成本 | RefreshJob 完成后累计 `recomputed_tokens / wall_time` |
| `fresh / stale chunks` | 差集规模（级联量化） | `refresh_article` 返回值 |
| `f1 / em` | 答案质量（pre 阶段对 gold） | eval_utils，标准化分词后计算 |
| `flip_ok / 翻转率` | 新鲜度（post 阶段） | 答案是否含注入实体 Zyxoria，按策略聚合 |

相位分组记号（表 1 的列名来源）：`pre/0` 冷启动、`pre/1` 全命中基线、`post/0` 更新后首请求（headline）、`post/1` 后续请求。

### 6.8 运行手册

```bash
cd examples/blend_kv_v1

# ── 实验 1（表 1）───────────────────────────────────────────
python trace_builder.py --num-questions 300 --out-dir traces/
python run_trace_experiment.py --policy norefresh      --trace traces/trace.json
python run_trace_experiment.py --policy passive        --trace traces/trace.json
python run_trace_experiment.py --policy proactive      --trace traces/trace.json
python run_trace_experiment.py --policy proactive_full --trace traces/trace.json
python run_trace_experiment.py --summarize traces/

# ── 实验 2（表 2）───────────────────────────────────────────
python trace_builder.py --num-questions 300 --request-skew 1.0 --schedule phase --out-dir traces_t2/
python run_trace_experiment.py --policy proactive --order fifo   --bg-budget 1 --trace traces_t2/trace.json
python run_trace_experiment.py --policy proactive --order hot    --bg-budget 1 --trace traces_t2/trace.json
python run_trace_experiment.py --policy proactive --order oracle --bg-budget 1 --trace traces_t2/trace.json
python run_trace_experiment.py --summarize traces_t2/
# 倾斜度扫描：--request-skew {0.0, 0.5, 1.0}；预算扫描：--bg-budget {1,2,4,8}

# ── 实验 3（表 3）───────────────────────────────────────────
python trace_builder.py --num-questions 300 --edit-mode tail    --out-dir traces_tail/
python trace_builder.py --num-questions 300 --edit-mode mid_iso --out-dir traces_iso/
python trace_builder.py --num-questions 300 --edit-mode mid_len --out-dir traces_len/
# 三条轨迹各跑 --policy proactive，对比 bg_tokens / fresh_chunks

# ── 实验 4（motivation）──────────────────────────────────────
python refresh_experiment.py --model mistralai/Mistral-7B-Instruct-v0.2
# 单臂调试加 --arm proactive
```

调试用 N=40，论文规模 N=300。所有回放每策略独立进程，`--summarize` 只读结果不做回放。环境：Mistral-7B-Instruct-v0.2 + vLLM/LMCache connector + CPU 卸载 KV 后端，全程 temperature=0，trace 构造 seed=42，可复现。

### 6.9 当前实验的已知局限（和预案的接口）

现在的回放是**串行**的：事件之间在途请求永远是 0，空闲窗口退化成"事件间隙"（`drain_background` 在事件之间跑），调度器没有被真实考验。这有两个出口：

- 表 1 的 proactive vs passive 差距够大 → 局限写进论文 §7，不做预案；
- 差距淹没在噪声里（验收标准 1 失败）→ 启用 §4.4 并发预案，把"何时算"放到真实争抢下测量。

同理，表 3 的 mid_len 级联如果大到让 proactive 的后台开销逼近 proactive_full（验收标准 2 失败）→ 启用 §3.3 CDC。

---

## 7. 验收标准对照表

| # | 验收标准 | 看哪个数 | 实验 | 不过线时的预案 |
|---|---|---|---|---|
| 1 | proactive vs passive 的 TTFT 差距可测 | 表 1 的 post/0 和 miss 列 | 实验 1 | §4.4 并发调度 |
| 2 | 增量后台开销远小于全量 | 表 1/3 的 bg_tokens：proactive vs proactive_full | 实验 3 | §3.3 CDC |
| 3 | 新鲜度：翻转率 ≈ 100% | 表 1 flip 列 | 实验 1/4 | 铁律保证，不过线即 bug |
| 4 | 质量：pre F1 不劣于冷启动 | pre/0 的 F1 配对检验 | 实验 1 | 查 blending 配置 |

四条全过 → 论文按 C 线投；哪条失败就上对应预案（各约 1-2 周工作量），反而可能把体量推到 B 线。

---

## 8. 附：参数与文件对照

**关键参数**

| 参数 | 在哪 | 默认 | 一句话 |
|---|---|---|---|
| `chunk_size` | LMCacheEngineConfig | 256 | 块大小（CDC 启用时变成 max） |
| `LMCACHE_RAG_DB_PATH` | 配置 | — | url→文本 JSON，① 的数据源 |
| `LMCACHE_ENABLE_BLENDING` | 环境变量 | False | **必须开**，否则命中归命中、质量会掉 |
| `budget_per_round` | ProactiveRefreshManager | 4 | 每个空闲窗口最多刷几篇 |
| `poll_interval` | 同上 | 1s | 守护线程轮询间隔 |
| `order_policy` | 同上 | hot | fifo / hot / oracle |
| `include_bos_variant` | 同上 | True | ②③ 是否同时处理去 BOS 形态 |
| `PYTHONHASHSEED` | 环境 | — | 多进程共享缓存时必须一致，否则 hash 对不上 |

**文件对照**

| 模块 | 文件 | 关键函数 |
|---|---|---|
| ① 存储 | `lmcache/v1/token_database.py` | `process_tokens`, `_guided_match`, `_full_scan_match`, `set_request_urls` |
| ② 差集 | 同上 + `lmcache/v1/cache_engine.py` | `refresh_article`, `compute_refresh_chunks`, `_remove_chunk` |
| ③ 调度 | `lmcache/v1/proactive_refresh.py` | `submit_update`, `refresh_pending`, `set_oracle_ranking`, `start` |
| 实验宿主 | `examples/blend_kv_v1/blend_mfo.py` | `build_llm_with_lmcache`, `setup_environment_variables` |
| 造轨迹 | `examples/blend_kv_v1/trace_builder.py` | `make_edit`, `zipf_pre_counts` |
| 回放计量 | `examples/blend_kv_v1/run_trace_experiment.py` | `run_policy`, `chunk_in_storage`, `summarize` |
| 单文档消融 | `examples/blend_kv_v1/refresh_experiment.py` | `reset_doc`, `timed_generate` |
| 质量指标 | `examples/blend_kv_v1/eval_utils.py` | `exact_match`, `f1_score`, `flip_ok` |
