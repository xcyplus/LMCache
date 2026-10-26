# 从被动块缓存到文档级 KV 缓存：面向可变 Web 语料的 RAG 服务增量更新与主动刷新

> 状态：完整初稿（正文完稿，所有实验数据留 `【待填】` 占位，占位处注明应填的指标与预期排序关系，供实验完成后直接回填）。
> 目标会议：ICPADS / HPCC / NAS（CCF-C）；若调度对比实验（§5.3）完成度高，可冲 Middleware（CCF-B）。

---

## 摘要

检索增强生成（RAG）服务系统依赖 KV 缓存复用来掩盖检索文档的预填充（prefill）延迟。以 CacheBlend、EPIC、KVLink 为代表的位置无关复用技术已使文档 KV 能够跨请求、跨位置重组复用，但其隐含假设是语料库**静态**。真实 Web 语料持续被修订，而已有的内容寻址缓存既不记录缓存块的来源（provenance），也无从回答修订后"失效哪些、何时重算"的问题，由此产生一种我们称为**首请求停摆**（first-request stall）的病理：引用已更新文档的第一个请求在请求路径上重算整篇文档，恰逢系统高峰。

本文提出 DocCache，一种面向 Web RAG 服务的**文档级** KV 缓存，在既有位置无关复用技术之上补全语料变更的完整生命周期。其核心贡献有二：**(1) 快照差集式增量更新**——将每次文档修订建模为新旧子块哈希集的差集，仅驱逐失效子块、保留未变子块的驻留命中，将更新后的重算范围从整篇文档缩小到变更区域（含边界滑移级联，本文诚实量化）；**(2) 空闲窗口主动刷新**——将"重算何时发生"形式化为在线调度问题，按 URL 热度排序待刷新任务、仅在无在途请求的空闲窗口内按预算执行，执行器采用逐块盲编码，其产出哈希与请求期查找哈希按构造精确对齐。由此，更新后的第一个请求命中一份完全预热的最新缓存，重算成本全部落在波谷。

系统具备**构造性新鲜保证**：key 是内容的纯函数，新内容映射新哈希，过期块不可能被服务出去——过期性从正确性风险退化为纯延迟风险。我们在 LMCache/vLLM 上实现 DocCache，基于 HotpotQA 构造可变语料库回放基准，对比四种策略。实验表明：文档更新后的首请求预填充延迟由【待填】降至【待填】（降低【待填】倍），主动增量的后台开销仅为全量刷新的【待填】%，答案质量与冷启动全量预填充无可测差异（F1 【待填】 vs 【待填】），全部缓存感知策略对更新事实的翻转率为 100%。

**关键词**：KV 缓存；检索增强生成；缓存失效；在线调度；LLM 服务系统

---

## 1 引言

大语言模型（LLM）服务通过 KV 缓存复用分摊预填充成本。RAG 服务中，同一批检索文档被大量请求反复引用，催生了两代缓存技术：前缀缓存 [vLLM, SGLang] 只复用公共前缀；**位置无关复用**技术 [CacheBlend, EPIC, KVLink, BlockAttention] 则通过内容寻址、注意力重组或可组合位置编码，使文档 KV 可挂载到 prompt 中的任意位置，配合加载期少量选择性重算修复跨注意力失真。至此，"KV 如何跨请求复用"在技术上已被充分解决。

但这一整条工作线共享一个沉默的假设：**语料库是静态的**。Web 语料库违背它：知识库被编辑，百科条目被修订，新闻页面被整体替换。考虑如下场景（§2.2 实测）：一篇 2048 token、已缓存为 KV 的百科文档在 10:00 被编辑 200 token；14:00 迎来 100 个引用它的请求。在现有块缓存下：

- **定位失败**：内容寻址缓存不存出处。没有任何机制能回答"哪些缓存块属于这篇刚被编辑的文档"——缓存里只有匿名哈希。
- **失效两难**：不知道出处，唯一安全的动作是全部丢弃；而保留过期块则冒着日后被读出的风险。朴素策略下，爆发期的每个请求都支付整篇重算。
- **时机无主**：即便知道重算什么，也没有人为"何时算"负责。请求路径上算是高峰付费；双版本并存内存不可行（7B 模型一篇 2048 token 文档的 KV 占约 0.5–1 GB）。

我们称这一现象为**首请求停摆**：更新后的第一个请求（在朴素失效下是全部请求）为语料变更买单，TTFT 在系统最忙时恶化最重。

社区已从相邻角度逼近此问题：预算修复研究单次请求内如何修补过期 KV [BudgetedRepair]；预测式缓存管理面向**未来请求**做预取与驱逐 [CacheScout, AgentWorkflow]；答案层缓存维护可变文档的依赖一致性 [FinCacheServe]。缺失的是 **KV 层本身的存储级答案**：谁追踪出处、谁做差集失效、谁调度重算时机。

本文提出 **DocCache**，把块缓存升级为文档级缓存，管理缓存文档状态的完整生命周期。本文**不**声称在位置无关复用上有所创新——那是 CacheBlend/EPIC/KVLink 已解决的substrate；DocCache 的贡献在变更生命周期：

1. **URL 感知的文档级存储组织（§3.2，基础设施）**：以文档（URL）为一等公民组织 KV——无分隔符切段、256-token 内容寻址子块、子块哈希到 URL 的反向索引、引用计数与热度统计。该层继承既有复用技术，并为其补上"出处"，是后两项机制的使能底座。
2. **快照差集式增量更新（§3.3，贡献一）**：文档修订 = 新旧子块哈希集的差集运算。未变子块原样驻留，仅失效变更块。我们对变长中部编辑导致的**边界滑移级联**做诚实量化而非回避（§5.4），并说明内容定义切块是兼容的后续收紧方向。
3. **空闲窗口主动刷新（§3.4，贡献二）**：把"何时重算"形式化为在线调度问题（最大化期望命中收益 × 后台成本约束下的空闲/预算/SLO），实例化热度贪心策略，并与离线 oracle 上界、FIFO 基线对比（§5.3）——使"时机"从工程规则升格为可评测的调度问题。执行器为**逐块盲编码**：每个失效子块从位置 0 独立预填充并按内容哈希写入，与请求期查找哈希按构造对齐，产出可被所有后续请求复用。
4. **可变语料库评测方法学（§5）**：基于 HotpotQA 的文档库 + 合成修订流 + 请求流三相位交织回放，四策略进程隔离对比，配 SQuAD F1 与事实翻转率双质量指标，与既有工作可对话。

本文的关键性质——**构造性新鲜保证**——值得单独陈述：因为 key 是内容的纯函数，携带新内容的请求必然算出新哈希、必然未命中过期块，过期性**只可能表现为一次 miss（延迟代价），绝不可能表现为把旧内容算进答案（正确性代价）**。这一性质把缓存过期问题从"一致性协议"域搬到"延迟调度"域，是本文全部设计的支点。

## 2 背景与动机

### 2.1 位置无关复用：已解决的substrate

自回归预填充的 KV 代价随上下文长度增长。前缀缓存 [vLLM, SGLang] 受共享前缀约束；CacheBlend [CacheBlend] 以内容寻址子块 + 加载期约 15% 选择性重算解除该约束；EPIC [EPIC] 通过打破注意力的位置绑定（broken attention + 可组合旋转）使复用成为模型级一等能力；KVLink [KVLink] 将独立编码的文档 KV 以注意力重组衔接；Block Attention [BlockAttention] 在架构级支持块间独立预填充；TurboRAG [TurboRAG] 证明 RAG 文档离线盲编码的质量损失可接受；CacheSlide [CacheSlide] 与 CacheClip [CacheClip] 进一步扩展跨位置复用的适用范围。评测工作 [KVReuseEval, DontBreak] 系统量化了复用的精度边界。

这些系统共同构成本文的substrate。DocCache 直接复用其复用语义（本文实现挂接 CacheBlend 的加载期选择性重算），本文的全部增量发生在它们未覆盖的维度：**语料变更**。

### 2.2 可变语料库病理

【图 1 占位：文档 10:00 更新、14:00 爆发 100 请求的四策略 TTFT 柱状对比 + 后台 token 右轴。数据源：§5.2 主表】

动机实验（方法见 §5.1）中，被动块缓存下更新后首请求的预填充时延为【待填】秒，而全命中基线仅【待填】秒；文档越长差距越大（2048 token 文档约【待填】倍）。若采用"更新即整篇驱逐 + 后台全量重算"，后台开销为整篇【待填】token；而快照差集只需重算【待填】token（含级联），这是贡献一的价值下界。

### 2.3 内容一致性与注意力一致性

一个由 [BudgetedRepair] 实证、并由 [KVReuseEval, DontBreak] 独立佐证的事实：内容哈希检测"什么变了"，检测不到"什么在注意力上依赖这个变更"——因果自注意力下，局部编辑改变其后未变 token 的 KV 状态。因此哈希差集是**存储层**机制；注意力失真需由**修复层**处理：① 重算的新鲜子块消除主要失真，② 加载期选择性重算（CacheBlend 机制）掩盖残余失配，③ 预算修复策略 [BudgetedRepair] 可作为执行器的可插拔选项。DocCache 的定位是存储层与调度层，不与修复策略竞争。

## 3 系统设计

【图 2 占位：架构图。语料库/修订源 → URL 感知 token 数据库 → 文章元数据（哈希集/引用计数/热度）→ 差集引擎 →（实线）请求期查找、（虚线）刷新调度器 → 盲编码执行器 → KV 存储 → vLLM】

### 3.1 总体架构

DocCache 由四个组件构成：(a) **URL 感知 token 数据库**：请求期把 prompt 切成带出处标注的子块序列；(b) **文章元数据层**：每篇文档的哈希集、子块→URL 反向索引、引用计数、热度计数；(c) **差集引擎**：修订事件到达时计算并执行失效；(d) **刷新调度器与盲编码执行器**：空闲窗口内按策略重算失效子块。请求路径（实线）完全不走新增组件的慢路径——调度器对在线流量零侵入。

### 3.2 URL 感知文档级存储（基础设施）

**无分隔符段定位。** 语料库为 URL→文本索引 U→t。请求期以最长匹配在 token 流上定位各文档区间：检索器给出本请求 URL 列表时走引导式窗口匹配（每文档 O(K)，K 为请求文档数）；否则退化为首 token 倒排索引的全库扫描。prompt 为文档文本纯拼接，不注入分隔符，避免分隔符方案对模型分布的扰动（自言自语退化）。

**内容寻址与双形态。** 每段切成固定 256-token 子块，key = hash(子块 token)，不含位置与前缀。由于整句拼接的 token 化在边界处可能合并，每篇文档同时登记带 BOS 与去 BOS 两种 token 形态的子块哈希，分别对应文档作为 prompt 首段与中间段的形态。

**文档级元数据。** 每篇文档携带：哈希集、子块哈希→URL 反向索引、引用计数（驱逐保护：在途请求引用的子块不失效）、累计命中计数（热度，调度信号）、时间戳。元数据仅驻进程内存、与 CPU KV 后端同生命周期，无需持久化协议——KV 在，元数据在；KV 亡，元数据亡。

**构造性新鲜保证（性质 1）。** key 是内容的纯函数 ⇒ 携带新文本的请求必产生新哈希、必 miss 旧块。过期性仅表现为延迟，不表现为错误答案。这消除了版本一致性协议的必要性：系统**不需要**判断"缓存是否还新鲜"，只需要决定"何时为新内容付费"。

### 3.3 快照差集式增量更新（贡献一）

修订事件 (u, t_new) 同步处理：

```
H_new = chunk_hashes(t_new)            # 双形态全集
Δ⁻ = H_old \ H_new                     # 失效集
Δ⁺ = H_new \ H_old                     # 待重算集
evict(Δ⁻);  pending[u] = Δ⁺           # 驱逐 + 排队
更新匹配索引（URL 变体 → t_new）        # 后续请求定位到新文本
```

未变子块 H_old ∩ H_new 原样驻留并保留全部命中统计。三点说明：

1. **为什么差集是"增量"而非"局部"**：变长中部编辑使编辑点之后的子块边界整体滑移，其后子块虽语义未变但哈希已变，全部落入 Δ⁺。我们对级联做诚实量化（§5.4：中部替换在【待填】子块文档上平均失效【待填】个子块，占【待填】%），并指出内容定义切块（滚动哈希锚定边界，rsync/LBFS 思想）可将失效收紧到编辑邻域——由于 key 仍是子块内容纯函数，该扩展与现有哈希方案**兼容**，留作后续工作。
2. **同步性与安全**：差集计算与失效在修订事件线程内同步完成，避免"新文本已入索引、旧 KV 未驱逐"的窗口；在途请求经引用计数保护。
3. **成本**：差集为集合运算，复杂度 O(|H_old| + |H_new|)，对 2048 token 文档为微秒级，可忽略。

### 3.4 空闲窗口主动刷新（贡献二）

#### 3.4.1 问题形式化

修订后待办为 Δ⁺（一组需重算 KV 的子块）。刷新调度为在线优化问题：

```
maximize   Σ_j p_j · s_j  −  λ · Σ_j c_j
subject to 重算仅在空闲窗口执行（在途请求数 = 0）
           每轮预算 B、内存容量、后台不延迟在线预填充（SLO）
```

其中 p_j 为文档 j 近期被引用的概率（URL 热度是其 Zipf 一致代理），s_j 为命中节省的 TTFT，c_j 为后台重算成本。该形式化给出三个可评测的策略维度：**候选排序**（p_j 的估计）、**时机**（空闲窗口定义）、**预算**（B 与 λ）。

#### 3.4.2 策略与上界

本文实例化三个策略用于对比（§5.3）：(a) **FIFO**：先来先刷，忽略热度；(b) **热度贪心**（默认）：按累计命中数降序、同热度先到先做，空闲即按预算执行；(c) **Oracle**：离线读取回放轨迹的未来请求时间，作可达性上界。热度贪心与 Oracle 的差距即"更好预测模型"的可行收益空间——我们预期该差距随负载 Zipf 倾斜度增大而收窄（【待填】，§5.3 验证）。

#### 3.4.3 盲编码执行器

执行器将每个待重算子块**从位置 0 独立预填充**，按其内容哈希写入 KV 存储：

- **正确性**：key 与位置无关 ⇒ 未来请求查的哈希 = 执行器刚写的哈希，无论子块届时出现在 prompt 何处；
- **位置失真矫正**：从位置 0 编码引入的旋转相位偏差由加载期对 K 状态重旋转（re-RoPE）矫正——复用 CacheBlend 流水线的既有机制，不新增模型改动；
- **成本本地化**：逐块执行只触碰 Δ⁺，从不触碰整篇文档——这是与 [TurboRAG, KVLink] 离线全量编码的关键差异：它们假设静态语料一次性算完，我们为**增量**变更高频重算设计；
- **可组合性**：执行器的"选哪些子块、按什么窗口"可插拔——编辑局部窗口 [BudgetedRepair] 是一种实例化，留作扩展。

#### 3.4.4 与在途请求的交互

失效经引用计数跳过在途请求；刷新执行限制在空闲窗口；内存压力由既有 LRU 治理，文章元数据作为一等驱逐域（ref_cnt>0 的文章整体跳过）。

## 4 实现

DocCache 实现于 LMCache v1（以 KV connector 集成于 vLLM）：

| 组件 | 位置 | 规模 |
|---|---|---|
| URL 感知 token 数据库 | `lmcache/v1/token_database.py`（UrlSegmentTokenDatabase） | 【待填】行 |
| 文章元数据 + 差集引擎 | 同上（ArticleCacheMetadata / refresh_article） | 【待填】行 |
| 引擎级文章 API | `lmcache/v1/cache_engine.py` | 【待填】行 |
| 调度器 + 盲编码执行器 | `lmcache/v1/proactive_refresh.py`（ProactiveRefreshManager） | 【待填】行 |
| 评测框架 | `examples/blend_kv_v1/`（trace_builder / run_trace_experiment / eval_utils） | 【待填】行 |

语料索引经配置（`LMCACHE_RAG_DB_PATH`）从 JSON 加载；执行器以闭包注入 vLLM 生成调用，调度器与运行时解耦。chunk size = 256，全程贪心解码。

## 5 实验评估

### 5.1 方法

**可变语料库基准。** 由 HotpotQA（distractor）构造回放轨迹：每题 10 条检索段落拼为一篇长文档（【待填】–【待填】 token，跨多子块），以合成 URL 索引；按 [BudgetedRepair] 的配对探针方式，将金答案字符串替换为稀有虚构实体（Zyxoria）模拟一次内容修订。每篇文档交织三相位事件流：两个更新前请求（第二个为全命中基线）→ 一个修订事件 → 两个更新后请求。更新前答案对原金答案评 SQuAD F1/EM；更新后答案检查是否**翻转**到注入实体。

**策略。** 四策略进程隔离回放同一轨迹（杜绝跨策略缓存污染）：

| 策略 | 修订后行为 | 隔离出的机制 |
|---|---|---|
| norefresh | 无任何处理 | 被动块缓存现状 |
| passive | 仅差集失效（§3.3） | 贡献一 |
| proactive | 失效 + 空闲增量刷新（§3.4） | 贡献一 + 贡献二（完整方案） |
| proactive_full | 整篇驱逐 + 后台全量重算 | 无增量对照（与 proactive 的后台差 = 贡献一收益） |

**指标。** 每请求预填充墙钟（max_tokens=1 的 TTFT 代理）；文档子块命中率与请求路径未命中 token（对存储层直接核实，非日志推断）；后台重算 token 数与墙钟；答案 F1/EM；更新后翻转率。

**环境。** Mistral-7B-Instruct-v0.2，vLLM + LMCache connector，CPU 卸载 KV 后端，【待填：GPU 型号】，N=【待填】题（调试验证 N=40）。

### 5.2 端到端结果

**表 1：HotpotQA 可变语料库回放主结果（N=【待填】，均值 ± 标准差；post/0 = 修订后首请求）**

| 策略 | 后台 token | 后台耗时(s) | pre/0 TTFT(s) | pre/1 TTFT(s) | **post/0 TTFT(s)** | post/1 TTFT(s) | post/0 未命中 token | 翻转率 |
|---|---|---|---|---|---|---|---|---|
| norefresh | 0 | 0 | 【待填】 | 【待填】 | 【待填】 | 【待填】 | 【待填】 | 【待填】 |
| passive | 0 | 0 | 【待填】 | 【待填】 | 【待填】 | 【待填】 | 【待填】 | 【待填】 |
| proactive | 【待填】 | 【待填】 | 【待填】 | 【待填】 | 【待填】 | 【待填】 | 【待填】 | 【待填】 |
| proactive_full | 【待填】 | 【待填】 | 【待填】 | 【待填】 | 【待填】 | 【待填】 | 【待填】 | 【待填】 |

**回填校验清单**（预期排序，异常即回查实验）：
- post/0 TTFT：norefresh > passive > proactive ≈ proactive_full，且 proactive 的 post/0 ≈ pre/1（全命中）
- post/0 未命中 token：norefresh ≈ 整篇文档 token；passive ≈ |Δ⁺|（含级联）；proactive = proactive_full = 0
- 后台 token：proactive（=|Δ⁺|）≪ proactive_full（=整篇）；两者之比即贡献一的增量收益
- 翻转率：四策略均应 ≈ 100%（性质 1：过期只损延迟不损正确性——若 norefresh 翻转率显著低，说明存在 KV 污染 bug）
- pre/0 与 pre/1 的 F1：各策略应与冷启动 oracle 无显著差异（复用与盲编码的质量代价）

### 5.3 调度策略对比（贡献二的研究性）

**表 2：排序策略 × 负载倾斜度下的 post/0 命中率/TTFT（§3.4.2 三策略）**

| 倾斜度(Zipf s) | FIFO post/0 未命中 | 热度贪心 post/0 未命中 | Oracle post/0 未命中 |
|---|---|---|---|
| 0（均匀） | 【待填】 | 【待填】 | 【待填】 |
| 【待填】 | 【待填】 | 【待填】 | 【待填】 |
| 【待填】（强 Zipf） | 【待填】 | 【待填】 | 【待填】 |

预期：随倾斜度增大，热度贪心逼近 Oracle，FIFO 停留高位——热度信号的充分性由负载分布决定。另报：预算 B 的敏感性扫描（B ∈ {1, 2, 4, 8}）与刷新墙钟对在线 TTFT 的干扰（应为 0，因调度仅入空闲窗口）。

### 5.4 增量收益与级联分析

按编辑类型分解后台开销（隔离贡献一）：

**表 3：编辑类型 × 失效范围**

| 编辑类型 | 失效子块数 / 总子块数 | 后台 token | 相对全量刷新节省 |
|---|---|---|---|
| 尾部追加 | 【待填】 | 【待填】 | 【待填】% |
| 等长中部替换 | 【待填】 | 【待填】 | 【待填】% |
| 变长中部替换 | 【待填】（含级联） | 【待填】 | 【待填】% |

变长编辑的失效子块数应显著超过编辑邻域（级联效应），我们如实报告，并给出内容定义切块的理论收紧上界（【待填】，按锚点密度估算）作为讨论。

### 5.5 质量评估

- 更新前 F1/EM：四策略 vs 冷启动 oracle（表 1 附行）：预期差 < 【待填】且配对检验不显著（t-test p=【待填】）
- 翻转率 100% 的解释性验证：抽取【待填】个翻转样例人工核对答案确实来自新文本
- 消融：granularity（按段短文档 vs 拼接长文档）对增量收益的调制（短文档收益收窄，长文档放大）

## 6 相关工作

**位置无关 KV 复用（substrate，非本文贡献）。** 前缀缓存 [vLLM, SGLang]；CacheBlend [CacheBlend] 内容寻址 + 加载期选择性重算；EPIC [EPIC] 打破注意力位置绑定实现模型级位置无关复用；KVLink [KVLink] 独立编码文档 KV 并以注意力重组衔接；Block Attention [BlockAttention] 块级独立预填充；TurboRAG [TurboRAG] 离线盲编码证明质量损失可接受；CacheSlide [CacheSlide] 跨位置复用；CacheClip [CacheClip] RAG 场景 KV 复用；FlashBack [FlashBack] 位置无关处理检索块。**上述工作全部假设静态语料**；DocCache 在其上补全变更生命周期，复用其修复机制（加载期重算/re-RoPE），不与之竞争复用技术本身。

**请求内选择性重算与修复。** ProphetKV [ProphetKV] 以用户查询驱动选择性重算以保精度；Budgeted Repair [BudgetedRepair] 研究编辑后预算约束的原地修复，发现连续编辑局部窗口优于重要性选择。二者均在**请求路径**上、为**准确率**服务；DocCache 把重算移出请求路径、为**时效性**调度，且可将其策略作为执行器插件。

**预测式缓存管理。** CacheScout [CacheScout] 学习智能体转移概率预取/驱逐；面向动态智能体工作流的预测式 KV 管理 [AgentWorkflow] 同属此类。它们前馈预测**未来请求**；DocCache 后馈响应**内容变更**。正交可组合：预测模型可替换本文的热度信号。

**答案层与 KV 基础设施。** FinCacheServe [FinCacheServe] 为可变企业文档维护答案缓存的依赖一致性（答案层，非 KV 层）；Mooncake [Mooncake] KV 中心化分离式架构、Lethe [Lethe] 分层时间自适应剪枝等构成基础设施正交域。DocCache 的存储组织可部署于 Mooncake 类底座之上。

**复用风险评测。** KV 复用精度评测 [KVReuseEval] 与 prompt 缓存敏感性评测 [DontBreak] 系统量化了复用的破坏条件，支撑本文 §2.3 的"内容一致 ≠ 注意力一致"论点。

**合成修订探针方法学**沿用 [BudgetedRepair] 的实体替换配对探针思想，并扩展为三相位交织回放轨迹。

## 7 讨论与局限

1. **级联失效**：变长编辑滑移下游边界，§5.4 如实量化；内容定义切块是兼容的收紧方向（key 仍为内容纯函数），留作后续工作。
2. **注意力过期**：哈希差集只标记存储过期；编辑下游失真由加载期重算掩盖、可选预算修复插件处理。我们改进的是失效定位与时机，不改进修复算法本身。
3. **顺序回放**：评测为单 GPU 串行回放，空闲窗口退化为事件间隙；并发到达下的调度真实增益需在线服务评测，是下一步。
4. **热度即历史**：贪心策略用历史命中代替未来概率；§5.3 的 Oracle 差距给出学习型预测器的收益上界。

## 8 结论

Web 语料库在变，KV 缓存不应假装它不变。DocCache 把块缓存升级为文档级缓存：以出处追踪回答"谁的 KV"，以快照差集回答"失效哪些"，以空闲窗口调度回答"何时重算"，并凭借构造性新鲜保证把过期性从一致性风险转化为可调度的延迟问题。更新后的第一个请求，命中一份新鲜且完全预热的缓存。机制轻量、与既有复用技术正交组合。

**工件**：源码与评测框架位于 【待填：仓库链接】。

---

## 参考文献

1. Kwon W, Li Z, Zhuang S, et al. Efficient Memory Management for Large Language Model Serving with PagedAttention. SOSP 2023.
2. Zheng L, et al. SGLang: Efficient Execution of Structured Language Model Programs. arXiv:2312.07104, 2023.
3. LMCache: High-Performance KV Cache Layer for LLM Serving. 项目仓库.【核对正式引用】
4. Luo Y, et al. CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion. EuroSys 2026.
5. Mao M, Mackey W, Lin X. Contiguity, Not Importance: Budgeted Repair of Stale KV Caches After Document Edits. arXiv:2609.17983, 2026.
6. EPIC: Efficient Position-Independent Caching for Serving Large Language Models.【核对作者/发表信息】
7. KVLink: Accelerating Large Language Models via Efficient KV Cache Reuse. NeurIPS 2025.【核对作者】
8. Block Attention for Efficient Prefilling. ICLR 2025.【核对作者】
9. TurboRAG: Accelerating Retrieval-Augmented Generation with Precomputed KV Caches. arXiv:2410.07555.【核对编号】
10. ProphetKV: User-Query-Driven Selective Recomputation for Efficient KV Cache Reuse in Retrieval-Augmented Generation.【核对出处】
11. CacheClip: Accelerating RAG with Effective KV Cache Reuse.【核对出处】
12. CacheSlide: Unlocking Cross Position-Aware KV Cache Reuse.【核对出处】
13. CacheScout: Learning Agent Execution for KV-Cache Management in Agentic Serving. arXiv:2608.14624, 2026.【核对作者】
14. FinCacheServe: Dependency-Consistent Answer Reuse for RAG Serving over Mutable Enterprise Documents. arXiv:2607.26076, 2026.【核对作者】
15. Efficient Serving for Dynamic Agent Workflows with Prediction-based KV-Cache Management. 2026.【核对出处】
16. Mooncake: A KVCache-centric Disaggregated Architecture for Serving LLM. FAST 2024.
17. Lethe: Layer- and Time-Adaptive KV Cache Pruning for Reasoning-Intensive Serving.【核对出处】
18. Evaluating the Accuracy of KV Cache Reuse.【核对出处】
19. Don't Break the Cache: An Evaluation of Prompt Caching.【核对出处】
20. Yang Z, et al. HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering. EMNLP 2018.
21. Jiang A Q, et al. Mistral 7B. arXiv:2310.06825, 2023.
22. FlashBack: Efficient Retrieval-Augmented Language Modeling for Fast Inference. ACL 2025.【核对作者】
23. Ratner A, et al. Parallel Context Windows for Large Language Models. NeurIPS 2023.
24. StreamingQA: A Benchmark for Adaptation to Long-Term Memory Changes.【核对出处】

---

## 附：回填操作指引（实验完成后）

1. **表 1**：运行 `trace_builder.py --num-questions 300` 后依次以四策略跑 `run_trace_experiment.py`，最后 `--summarize traces/` 导出，逐格填入
2. **表 2**（代码已就绪）：`trace_builder.py --request-skew 1.0 --schedule phase` 生成三相位倾斜轨迹；`run_trace_experiment.py --policy proactive --order {hot,fifo,oracle} --bg-budget 1` 各跑一次——**预算设 1** 才能让排序策略真正竞争（预算过大则全部任务都被即时消化，排序无区分度）
3. **表 3**（代码已就绪）：`trace_builder.py --edit-mode {tail,mid_iso,mid_len}` 生成三条轨迹，各跑 proactive 策略对比后台 token；mid_iso 模式会用 tokenizer 做 token 等长对齐以隔离级联效应
4. **图 1/图 2**：图 1 用表 1 数据出柱状图；图 2 用 draw.io/TikZ 画架构
5. 全部 `【待填】` 处按就近的"预期排序校验清单"核对趋势是否成立，异常先查实验再动笔解释
