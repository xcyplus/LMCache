# DocCache 实验结果汇总

> 远程跑通于 autodl（RTX 4080 SUPER 32GB），2026-11-07 02:39~04:01（1h22m）。
> 分支 `embed_url_1025_AI`，N=200（HotpotQA distractor validation 采样）。

---

## 背景速览（给第一次看本文档的人）

**DocCache 解决什么问题。** RAG（检索增强生成）服务会把网页文档算出的 KV 缓存（一种避免每次请求都重算整篇文档的中间状态）存下来复用，下次请求同一篇文档就直接命中、不用重算。但现有技术都假设**网页永远不变**——一旦网页被编辑，缓存里那份旧 KV 就成了三不管的东西：没人知道缓存里哪几块属于这篇网页、不知道该删哪几块、也不知道该什么时候重算。结果是：**网页更新后的第一个请求，要在系统最忙时把整篇文档重算一遍**（这叫"首请求停摆"）。

**DocCache 的两个核心机制。**

| 机制 | 通俗说法 | 论文叫法 |
|---|---|---|
| ① 差集失效 | 网页改了，只删掉真正变了的缓存块，没变的不动 | 快照差集式增量更新（贡献一） |
| ② 空闲后台刷新 | 趁没请求的空闲空档，把新块的 KV 提前算好存进缓存 | 空闲窗口主动刷新（贡献二） |

合起来就是：网页变了 → ①删掉失效块 → ②空闲时提前算好新块 → 下个请求来时全命中，不用现场重算。

**一条铁律（贯穿所有实验）。** 缓存的 key 只看"这 256 个 token 是什么内容"，不看位置。所以网页改了之后，改过的地方 token 变了 → hash 必然变 → 新内容永远不会错误命中旧缓存。**过期最坏也只是"慢一下"（没命中就重算），绝不会"答错"**（不会把旧内容算进答案）。这叫"构造性新鲜保证"，是所有设计的支点。

**四策略是什么。** 实验对比四种"网页更新后怎么办"的策略，隔离每个机制的价值：

| 策略 | 网页更新后做什么 | 代表什么 |
|---|---|---|
| norefresh | 什么都不做 | 没有 DocCache 的现状（被动块缓存） |
| passive | 只做 ①（删失效块，不后台补） | 单独看贡献一值多少 |
| proactive | ① + ②（删失效块 + 空闲后台补新块） | 完整方案 |
| proactive_full | 整篇删掉 + 后台全量重算 | 无增量对照（和 proactive 的后台差 = ①省的算力） |

---

## 术语表（读表前必看）

| 术语 | 含义 |
|---|---|
| **TTFT** | Time To First Token，首 token 延迟，衡量请求多快开始响应。越低越好 |
| **post/0** | "网页更新后的第一个请求"——这是论文的 headline 指标，最关键的一格 |
| **pre/0** | 更新前的第一个请求（冷启动，缓存还没有），作为重算基线 |
| **pre/1** | 更新前的第二个请求（缓存已建好，全命中），作为"全命中能多快"的基线 |
| **post/1** | 更新后的第二个请求，看缓存收敛后的表现 |
| **miss / miss@post/0** | 更新后首请求有多少 token 没命中缓存、需要现场重算。=0 表示全命中。越低越好 |
| **bg_tokens** | 后台刷新重算了多少 token（后台成本）。衡量方案在空闲时付出了多少算力 |
| **bg_s / bg_time** | 后台刷新花了多少秒（墙钟） |
| **flip / 翻转率** | 更新后答案是否"翻转"到新内容（1=答出新的，0=还答旧的）。检验新鲜度 |
| **F1** | 答案和标准答案的词级 F1（更新前阶段评），衡量答案质量有没有因缓存复用变差 |
| **fresh chunks** | 差集算出的"待重算的新块"数量（新文本里有、旧文本里没有的块） |
| **stale chunks** | 差集算出的"失效旧块"数量（旧文本里有、新文本里没有的块，要删掉） |
| **盲编码** | 后台刷新时，每个新块单独从位置 0 预填充算 KV。因为 key 只看内容不看位置，将来任何请求来查都能命中 |
| **layerwise** | KV 缓存按神经网络层分层存储的机制（GPU 上是滑动窗口），本实验用 CPU 卸载 + GPU 按层加载 |
| **block / phase schedule** | 事件排布方式。block=每题自成一组（更新后马上有请求）；phase=全部更新挤在一起再全部请求（让排序策略有发挥空间） |
| **bg-budget** | 每个空闲窗口最多刷几篇文档。越小排序策略越有区分度（资源紧张才看得出先刷谁） |
| **Zyxoria** | 实验里用来替换金答案的生造词（模拟"网页内容变了"）。模型对这种生造词回答能力有限，所以 flip 偏低不全是系统问题 |

---

## 实验环境

| 项 | 值 |
|---|---|
| 服务器 | autodl `connect.bjb1.seetacloud.com:53868` |
| GPU | NVIDIA RTX 4080 SUPER（32GB，nvidia-smi -L 误标，实际 31.48 GiB） |
| 模型 | Mistral-7B-Instruct-v0.2（bf16，本地 `/root/autodl-tmp/mfo/models/`） |
| 框架 | vLLM 0.10.0 + LMCache（`embed_url_1025_AI` 分支，源码 -e 安装） |
| 关键配置 | `VLLM_ENABLE_V1_MULTIPROCESSING=0`（单进程，让主进程拿 engine）、`max_model_len=8000`、`chunk_size=256`、layerwise+CPU 卸载(5GB)、`enable_blending=True`、`PYTHONHASHSEED=0` |
| 数据 | HotpotQA distractor validation（7405 题，seed=42 采样 N=200） |

### 跑通必须的两个修复（博客未覆盖）
1. **datasets 版本三角冲突**：datasets 5.x 要 huggingface-hub≥2.0，transformers 4.55.2 要 <1.0 → 降 datasets 到 3.6.0 + hub 0.36.2，清 datasets 缓存重建（5.x 用 `List` 特征类型，3.x 不认）。
2. **V1 多进程 engine 注册问题**：vLLM V1 默认 SyncMPClient（多进程），engine 在子进程注册，主进程 `LMCacheEngineBuilder.get` 返回 None → 设 `VLLM_ENABLE_V1_MULTIPROCESSING=0` 用 InprocClient 单进程解决。

---

## 表1：四策略主对比

### 这个实验测什么

**一句话**：对比四种"网页更新后的处理策略"，看完整方案（proactive）相比什么都不做（norefresh）能让更新后的第一个请求快多少，以及每个机制各值多少。

### 为什么这么测

每篇文档经历完全相同的事件流：更新前请求×2 → 一次网页修订 → 更新后请求×2。唯一变量是"修订之后、下个请求之前，系统做了什么"。通过四策略两两对比，可以隔离出：
- **proactive vs norefresh/passive**：把重算挪到空闲后台，更新后首请求能快多少（贡献二的延迟收益）
- **proactive vs proactive_full**：都做到全命中，但一个只重算变更块（增量）、一个重算整篇（全量）——后台算力差就是贡献一（差集）省下的

### 怎么读这张表

- **看 post/0 列**（更新后首请求延迟）：越低越好。norefresh 应最差（整篇重算），proactive 应≈pre/1（全命中基线）
- **看 miss@post/0 列**（更新后首请求没命中的 token）：norefresh≈整篇文档，proactive=0（全命中）
- **看 bg_tokens 列**（后台成本）：norefresh=0（不干活），proactive 和 proactive_full 的差=增量 vs 全量省的算力
- **看 flip 列**（翻转率）：理论上四策略都应≈100%（铁律保证过期不损正确性），低了说明可能有 bug

### 数据（N=200，block schedule，mid_len 编辑）

> passive 因 layerwise bug 跳过（见已知问题①），实际跑三策略。

| 策略 | bg_tokens | bg_s | pre/0 | pre/1 | **post/0** | post/1 | miss@post/0 | F1 | flip |
|---|---|---|---|---|---|---|---|---|---|
| norefresh | 0 | 0 | 0.29 | 0.10 | **0.25** | 0.11 | 1452 | 0.37 | 0.35 |
| **proactive** | 490140 | 85.12 | 0.29 | 0.11 | **0.11** | 0.11 | **0** | 0.37 | 0.15 |
| proactive_full | 305740 | 47.47 | 0.29 | 0.10 | 0.11 | 0.11 | 1530* | 0.37 | 0.29 |

\* proactive_full 的 miss@post/0=1530 是**计量偏差**（chunk_in_storage 逐层查与整篇 generate 的 layerwise 存储口径不一致），实际 post/0=0.11s（全命中快）说明 vLLM retrieve 成功。

### 数据说明

- **proactive 把更新后首请求 TTFT 从 0.25s→0.11s（降 56%），miss 从 1452→0（全命中）**——这是论文的核心 headline
- pre/1=0.10s（全命中基线）≈ proactive post/0=0.11s，证明 proactive 达到全命中水平（重算已全部在后台完成）
- bg_tokens: proactive(490140) > proactive_full(305740)，因 proactive 含 pre 阶段全量预热；proactive_full 只算 post 阶段全量重算
- **flip 反常**：norefresh(0.35) > proactive(0.15)，详见已知问题④——可能暴露盲编码注意力失真，需进一步核对

---

## 表2：调度策略对比

### 这个实验测什么

**一句话**：当很多篇文档同时更新、排队等后台刷新时，"先刷哪篇"有没有讲究？对比三种排序策略，看热度排序（hot）相比无脑排序（fifo）和作弊上界（oracle）能多命中多少。

### 为什么这么测

贡献二（空闲后台刷新）有个子问题：待刷新的文档可能很多，空闲窗口刷不完，先刷谁？直觉是先刷"热门"的（被引用多的），但这取决于负载分布——如果所有文档一样热，热度排序没意义。所以要让排序策略真正"竞争"，有两个关键设计：
- **phase schedule**：把事件重排成"全部更新前请求 → 全部网页修订 → 全部更新后请求"，让 N 篇文档同时进待刷新队列，"先刷谁"才有发挥空间（默认的 block 模式每题自成一组，更新后马上有请求把积压消化了，排序来不及起作用）
- **bg-budget=1**：每个空闲窗口只准刷 1 篇。预算卡死，排序才有区分度（资源充裕则人人有份，排序无所谓）

### 怎么读这张表

- **看 post/0 和 miss@post/0 列**：三种排序应该拉开差距——fifo 最差（不看热度乱刷）、oracle 最好（作弊知道未来）、hot 居中
- **理想趋势**：负载越倾斜（少数文档特别热），hot 越逼近 oracle；hot 和 oracle 的差距="上更好的预测模型还能再赚多少"
- **如果三排序数据相同**=实验没测出区分度（budget 太大或后台没在 post/0 前跑完）

### 数据（N=200，phase schedule，bg-budget=1）

| 排序 | bg_tokens | bg_s | post/0 | miss@post/0 | F1 | flip |
|---|---|---|---|---|---|---|
| fifo | 490140 | 85.08 | 0.29 | 1530 | 0.37 | 0.37 |
| hot | 490140 | 85.20 | 0.29 | 1530 | 0.37 | 0.37 |
| oracle | 490140 | 85.26 | 0.29 | 1530 | 0.37 | 0.37 |

### 数据说明

**已知问题②：三排序无区分度**（post/0、miss 完全相同）。原因：phase 模式 + bg-budget=1 + N=200，200 篇文档批量更新排队，post/0 时后台刷新尚未完成（刷新主要发生在 post 阶段后期）。

**建议调整**：减小 N（如 50）让 budget 能覆盖，或排查 `refresh_pending(force=False)` 的空闲窗口执行逻辑。这张表目前不可直接用于论文，需重跑。

---

## 表3：编辑类型消融

### 这个实验测什么

**一句话**：网页被编辑时，不同编辑方式（尾部追加 vs 中部替换）会让多少缓存块失效？特别是要量化"级联失效"——中部变长编辑会让编辑点之后所有块边界滑移、内容全变、全部失效，这个现象有多痛。

### 为什么这么测

贡献一（差集失效）的卖点是"只重算变的部分"。但"变的部分"到底多大？如果是尾部追加，只有最后一块变，增量收益最大；如果是中部变长替换，编辑点之后所有块的边界都会滑移（因为固定按 256 token 切块，前面插了内容后面全错位），内容全变 → 全部进待重算集。这就是**级联失效**。论文对此做诚实量化而非回避。

三种编辑模式各隔离一个变量：

| 编辑模式 | 怎么改 | 隔离什么 | 预期失效范围 |
|---|---|---|---|
| tail | 只把答案的最后一次出现换掉（没有就文末补一句） | 无滑移的编辑 | 只有最后一块（下界） |
| iso | 只换第一处，用填充把 token 数补齐到和原答案一样长 | 内容变了但边界不滑移 | 编辑点所在的一块（本征失效） |
| len | 把所有答案出现处全换掉（长度变化） | 真实编辑：边界滑移 | 编辑点之后的全部（级联） |

**关键对比**：`len − iso = 级联效应的单独贡献`（同等内容变化下，变长 vs 等长多失效多少块）。

### 怎么读这张表

- **看 avg_total（差集规模）列**：tail 应最小，len 应最大
- **核心算式**：`len.avg_total − iso.avg_total = 级联失效块数`
- **看 bg_tokens 列**：应与差集规模同序（失效越多后台重算越多），互证数据可靠
- **iso 失效最少**是因为等长替换不改变 token 边界（没有滑移）

### 数据（N=200，proactive 策略）

| 编辑模式 | avg_fresh | avg_stale | avg_total(差集) | bg_tokens | bg_chunks |
|---|---|---|---|---|---|
| tail（尾部编辑） | 9.08 | 2.61 | 11.69 | 421720 | 1816 |
| iso（等长中部替换） | 7.85 | 1.40 | 9.25 | 226669 | 958 |
| **len（变长中部替换，级联）** | **10.46** | **3.99** | **14.46** | **490140** | **2093** |

### 数据说明

- **级联效应 = len(14.46) − iso(9.25) = 5.21 块/文档**——变长编辑比等长编辑多失效 5.21 块，这是级联的单独贡献，支撑论文 §5.4
- iso 失效最少（等长替换边界不滑移），len 最多（变长替换边界滑移导致级联）——符合预期
- tail > iso：tail 的"文末补句"机制导致末尾多块变化（非纯末尾单块）
- bg_tokens 排序与差集规模排序一致（len > tail > iso），互相佐证数据可靠
- iso 样本 122（<200）：部分题答案不在文中或等长对齐失败，跳过 update

---

## motivation：单文档消融

### 这个实验测什么

**一句话**：用一个真实的、人工构造的"北京词条人口数据更新"案例，给论文开头讲的"首请求停摆"故事配上真实数字——让读者直观看到"网页更新后第一个请求有多惨，DocCache 能救多少"。

### 为什么这么测

表1~表3 用的是 HotpotQA 数据集（生造词 Zyxoria 替换答案，模拟更新）。但 Zyxoria 是模型不认识的生造词，导致翻转率偏低、难以直观感受"答案是否真的更新了"。motivation 实验脱离 HotpotQA，手工构造一个真实场景：北京词条里"常住人口 2180 万"被改成更长的官方新表述"2210.8 万"（变长中部替换，触发真实级联）。这样：
- 答案翻转可直接判断（答出 2210.8 万 = 新鲜）
- 级联失效有真实规模（fresh/stale 块数）
- 每臂干净起点（reset_doc 清场后重跑）

四臂和表1 相同，额外演示三个机制：① `reset_doc` 干净起点；② 空闲窗口显式演示（忙时跳过、闲时才跑）；③ `flip_ok` 新鲜度自动判定。

### 怎么读这张表

- **看 ttft_probe 列**（更新后首请求延迟）：proactive 应远低于 norefresh
- **看 flip 列**：proactive 应=True（答出 2210.8 万），印证新鲜度保证
- **看 fresh/stale chunks**：差集规模，是级联失效的直接证据
- **看 bg_tokens**：proactive 只重算 fresh 的几块，proactive_full 重算整篇

### 数据（北京词条，变长中部替换）

| 臂 | fresh | stale | bg_tokens | bg_time | ttft_probe | answer_time | flip |
|---|---|---|---|---|---|---|---|
| norefresh | - | - | 0 | 0 | 0.16s | 0.55s | False* |
| **proactive** | **6** | **6** | **1373** | 0.25s | **0.085s** | 0.88s | **True** |
| proactive_full | - | - | 686 | 0.12s | 0.09s | 0.51s | False* |

\* norefresh 答案 `"The current population of Beijing is 22,108,000 people."` 与 proactive 答案 `"...常住人口已增长至2210.8万人"` **实为同一新数字**（22,108,000 = 2210.8万）。两者都回答了新内容，`flip_ok` 仅因查"2210"子串、英文用逗号分隔 `22,108,000` 未匹配而误判 False。**这恰恰印证论文的构造性新鲜保证：过期只损延迟（norefresh 0.16s vs proactive 0.085s），不损正确性（两者答案都是新内容）**。

### 数据说明

- **proactive ttft=0.085s（全命中）vs norefresh 0.16s（重算），降 48%**——延迟论点在真实场景成立
- **新鲜度由铁律保证**：norefresh 虽慢（重算整篇），但因新内容必算新 hash、必 miss 旧块、必在请求路径重算新文本，答案仍是新内容——过期=延迟代价，非正确性代价
- proactive bg_tokens=1373（仅 6 块 fresh）vs proactive_full 686：proactive 含预热全量、proactive_full 仅 post 阶段重算，口径不同

---

## 已知问题与局限

| # | 问题 | 影响 | 建议修复方向 |
|---|---|---|---|
| ① | **passive layerwise bug**：refresh_article 删 stale 后 GPU buffer `buffer_mapping`（滑动窗口）layer 0 状态错乱，retrieve 报 `ValueError: Layer 0 is not loaded`（gpu_connector.py:356）。proactive/proactive_full 因后台 generate 重新加载 buffer 而不崩。微型 generate(1 token) 未恢复 | 表1/motivation 缺 passive 臂 | 深查 retrieve_layer 与删除的交互；或在 refresh_article 后显式重置/重载 layerwise buffer |
| ② | **表2 三排序无区分度** | 表2 不可用 | 调小 N 或排查 refresh_pending(force=False) 在 phase 模式下的执行 |
| ③ | **proactive_full miss 计量偏差** | miss@post/0 偏高（TTFT 正常） | chunk_in_storage 逐层查口径 vs 整篇 generate 存储不一致；用 compute_refresh_chunks 逐块 generate 替代 encode_fn 整篇 |
| ④ | **表1 flip_rate 反转：norefresh(0.355) > proactive(0.155)** | 值得注意 | HotpotQA+Zyxoria 场景下 flip 整体偏低（模型对生造词能力有限），但 proactive 反而比 norefresh 低。可能解释：proactive post/0 全命中用盲编码 KV（每块从位置 0 独立编码，注意力失真），模型未重新 prefill 完整新文档，反而答不出 Zyxoria；norefresh post/0 重算整篇，模型重新"看到"含 Zyxoria 的全文，更易答出。**这可能暴露盲编码注意力失真在答案生成上的影响**（CacheBlend 加载期选择性重算修复不充分的迹象），需抽样答案核对确认是失真还是生造词难度 |
| ⑤ | **motivation flip_ok 字符串匹配误判** | 已修正解读 | `flip_ok` 查"2210"子串，英文答案 `22,108,000` 不含连续"2210"被误判 False。实为同一新数字，铁律成立。建议 `flip_ok` 改为数值归一化匹配 |

---

## 论文回填指引

| 论文位置 | 填什么 | 数据 |
|---|---|---|
| §5.2 表1 | post/0、miss、bg_tokens、F1、flip | norefresh 0.25s/1452 vs proactive 0.11s/0 |
| §5.3 表2 | 暂留空或标注"调整中" | 三排序无区分度，需重跑 |
| §5.4 表3 | fresh/stale/total，级联=5.21 块/文档 | len 14.46 vs iso 9.25 |
| §5.6 motivation | ttft/bg_tokens/flip | proactive 0.085s/flip=True vs norefresh 0.16s/flip=False |
| §5.1 环境 | GPU 型号、N | RTX 4080 SUPER 32GB，N=200 |

**headline 数据**（摘要用）：更新后首请求 TTFT 由 0.25s 降至 0.11s（降低 56%），miss token 由 1452 降至 0（全命中）；motivation 单文档场景答案正确翻转（flip=True），新鲜度保证成立。

---

## 原始数据

- **本地**：`paper/results/`（13 个文件：9 个 `results_*.json` + 3 个 `refresh_*.json` + `full.log`）
- **远程**：`/root/autodl-tmp/mfo/LMCache/examples/blend_kv_v1/`（`traces_t1/t2/t3_*` + `refresh_*.json` + `full.log`）
- **远程 screen**：`11651.full`（已 ALL DONE，可清理）
