# 项目设计：HotpotQA Agent 的 KV 生命周期管理（Memory-Aware Agent Edge Serving）

> 一句话定位：在 KV 预算受限的边缘推理机器上，多个 HotpotQA 多跳问答 agent 并发运行。每个 agent 在两次 LLM 调用之间会去调用工具，这段时间里它的 KV cache 是**留在显存、换出到内存、丢掉以后重算、还是只留下有用的部分**？我们要让 serving 系统按代价模型做出这个决定，在不降低答题质量的前提下，提高单位时间内完成（且答对）的任务数。

---

## 0. 为什么这样定题

老师的要求可以归结为三点：

1. **只做调度只能拿普通分。** 所以调度（admission、排队）在本项目里只是实验框架和 baseline。核心贡献放在 **KV/上下文管理**。
2. **"Implementing many policies is less valuable than understanding one well."** 所以只做**一个机制**：工具调用边界上的 KV 生命周期决策。selective KV 作为这个机制的第四种动作加入，不另开一条线。
3. **"Any cache-aware extension must use cache controls and measurements that the runtime actually exposes."** 所以每种动作都对应 llama.cpp 里真实存在的接口（见 §6）。唯一需要改 runtime 的部分（按 token 区间删 KV）放在 Phase 3，改动量小，并且有不改 runtime 的退路。

HotpotQA 的一个优势是其他数据集少有的：**它标注了 supporting facts（真正用来答题的句子）**。因此我们能直接回答"KV 里有多少 token 是有用的"，selective KV 也有了 oracle 上界。这是本项目在问题动机上最有说服力的一点，报告里要突出。

---

## 1. 场景

**故事**：一台边缘机器（小 GPU 或 CPU 实例）上部署一个本地模型，为很多用户提供"研究助手"服务。每个请求是一个多跳问题（HotpotQA），由 ReAct agent 求解：

```
[System prompt + few-shot 示例]   ← 所有 agent 共享，约 1–2k tokens
[Question]
Thought 1 → Action 1: search[Entity A] → (工具延迟) → Observation 1: 段落…
Thought 2 → Action 2: search[Entity B] → (工具延迟) → Observation 2: 段落…
Thought 3 → Action 3: finish[answer]
```

这个场景同时具备课程要求的全部负载特征：

| 课程要求 | HotpotQA 场景中的对应 |
|---|---|
| 多步、相互依赖的请求 | ReAct 每步依赖上一步的 Observation；bridge 题通常 2–4 步，comparison 题 2–3 步 |
| 步与步之间的工具调用 | `search` / `lookup`，延迟可控（离线索引 + 注入延迟分布） |
| 上下文跨步保留、不断增长 | 每个 Observation 追加一个或多个段落，上下文单调增长 |
| 短/长上下文、短/长输出混合 | 通过每次检索返回的段落数、是否带 few-shot、输出风格来控制 |
| 可复用前缀 | 三层复用：跨任务共享的 system prompt；同一任务跨步的历史；跨任务共享的文档（见 §2 的 `doc_overlap`） |

### 三类 KV，三种复用机会

这是整个项目的概念骨架，Phase 1 的测量也按这三类组织：

1. **共享前缀 KV**（system + few-shot）：所有 agent 相同，应常驻，只存一份。
2. **Agent 私有历史 KV**（question + 之前的 thought/obs）：下一步还会用到，但工具调用期间处于闲置状态。这是主要矛盾。
3. **文档 KV**（Observation 中的段落）：体积最大，但大部分是干扰段落；多个 agent 可能检索到同一篇文档。

---

## 2. 负载设计（Workload）

所有旋钮都写在 YAML 配置里。下表是建议的取值范围：

| 旋钮 | 取值 | 影响 |
|---|---|---|
| `question_type` | bridge / comparison / 混合 | 步数分布 |
| `level` | easy / medium / hard | 步数和失败率 |
| `obs_size` | 每次 search 返回 1 段 / 3 段 / 全文前 N 句 | **短/长上下文**，决定 KV 增长速度 |
| `few_shot` | 0 / 2 / 6 个示例 | 共享前缀长度 |
| `tool_delay` | none / fixed / lognormal(中位数 0.3s–3s) / 按工具区分 | 闲置时长，决定 retain/drop/offload 的取舍 |
| `arrival` | Poisson(λ) / bursty (MMPP) / closed-loop N 个用户 | 负载压力 |
| `doc_overlap` | 0 / 0.3 / 0.7 | 并发任务共享文档的比例（按共同实体给问题分组），用来研究跨 agent 复用 |
| `memory` | full_history / sliding_window(k) | agent 侧的上下文管理（作为对照：agent 自己裁剪 vs 系统裁剪） |

**工具实现**：用 HotpotQA（distractor 或 fullwiki）里的 Wikipedia 段落建离线索引（BM25 或实体名精确匹配），从而保证工具返回内容确定、可复现。延迟单独注入，并在日志中记录。

**dev / test 划分**：从 validation 集中固定抽取 dev（调参用，比如选出最佳 fixed-K）和 test（只用于最终报告），两者不重叠。这是"tuned simple baselines"要求的前提。

---

## 3. 研究问题

- **RQ1（Phase 1，刻画）**：在 HotpotQA agent 负载下，KV 预算花在了哪里？瓶颈主要是 (a) 活跃请求的 KV 容量，(b) 工具调用期间闲置 agent 占着的 KV，还是 (c) 被驱逐后重新 prefill 的计算量？三者如何随并发度、上下文大小、工具延迟变化？
- **RQ2（Phase 2，核心机制）**：在工具调用边界上，按代价模型为每个 agent 选择 retain / offload / drop，能否胜过固定策略（llama.cpp 默认、always-retain、always-drop、LRU）和调优过的 fixed-K FIFO？
- **RQ3（Phase 3，beyond）**：能否通过 selective KV 缩小单个 agent 的 KV 占用，即只保留有用的 observation 段落？KV 级区间删除（不重算，近似）与文本级压缩（重算，精确）各自的质量和吞吐代价是多少？

---

## 4. 系统架构

```
                 ┌──────────────────────────────────────────┐
 Workload ──►    │  Admission controller                    │  按预测 KV 需求准入
 (arrivals,      │  （KV budget 会计）                        │
  HotpotQA)      ├──────────────────────────────────────────┤
                 │  Step scheduler  (FIFO / ready-first)    │  只是框架，不是贡献
                 ├──────────────────────────────────────────┤
 Agent runtime   │  ★ KV Lifecycle Manager ★                │  本项目核心
 (ReAct loop,    │   在每个 tool-call 边界做决策：             │
  tools, delays) │   RETAIN | OFFLOAD | DROP | TRIM(selective)│
     ▲  │        │   代价模型 + 在线估计（工具延迟、prefill   │
     │  ▼        │   速率、swap 带宽、内存压力）               │
  tool calls     ├──────────────────────────────────────────┤
                 │  Backend adapter: llama-server            │
                 │  /completion(id_slot, cache_prompt)       │
                 │  /slots save|restore|erase, /metrics      │
                 │  (+ Phase 3: 小 patch 支持 KV 区间删除)     │
                 └──────────────────────────────────────────┘
 Instrumentation: events.jsonl（每个请求/工具/任务的时间戳、prompt/cached/生成 token 数、
                  TTFT、prefill/decode 时间）+ KV 占用时间线 + 进程/GPU 内存采样
```

**关键设计点：agent 固定在 slot 上。** llama-server 默认按 prompt 相似度挑选 slot，并会自己清理闲置 slot。它会替我们做 KV 决策，这样实验就不可控。所以由 scheduler 显式指定 `id_slot`，把 slot 当作"该 agent 的 KV 容器"来管理。llama.cpp 自带的 slot 行为本身作为 **fixed cache policy baseline**（课程要求有这个对照）。

---

## 5. Phase 1：Baseline 与刻画（约 2 周）

### 5.1 实现
- FIFO 队列，固定并发 K（`--parallel K`），总 KV 预算 = `-c`（ctx_size）。
- 每个 LLM 请求和每个任务都插桩（见 §4 的事件日志）。

### 5.2 必须产出的实验与图

| 实验 | 扫描 | 产出的图 / 结论 |
|---|---|---|
| E1 并发 sweep | K ∈ {1,2,4,8,16} × λ | 吞吐-延迟曲线，找到 best fixed-K（在 dev 上） |
| E2 上下文增长 | obs_size × KV budget | 失败（context overflow）悬崖出现的位置 |
| E3 工具延迟 | tool_delay 中位数 0 → 3s | 闲置 KV 占比随延迟的变化 |
| E4 前缀复用 | cache_prompt on/off × few_shot 长度 | 前缀缓存是否显著影响该 runtime；命中率与 TTFT |
| E5 KV 组成分析 | 离线统计 | 按 token 拆分 KV：共享前缀 / 问题 / thought / observation(supporting) / observation(非 supporting) |

**E5 是 Phase 1 的亮点**：画一张"KV 占用时间线"堆叠面积图，分为 active / idle-during-tool / wasted（非 supporting 的 observation token）三类。这张图要直接回答 RQ1，同时为 Phase 2/3 提供依据。

### 5.3 预期假设（需要用数据验证或推翻）
- H1：并发高时瓶颈是**闲置 agent 占用的 KV**，而不是算力。工具延迟越长，这个现象越明显。
- H2：observation 占 KV 的大头（预计超过 60%），其中只有一小部分是 supporting facts。
- H3：KV 被挤掉后，重新 prefill 的开销随上下文长度线性增长。长上下文 agent 被驱逐的代价远高于短上下文 agent，所以"驱逐谁"很重要。

**Phase 1 交付**：可复现的 baseline、workload 定义、性能曲线、对主导瓶颈的解释。（与课程要求一一对应。）

---

## 6. Phase 2：KV Lifecycle Manager（核心，约 2 周）

### 6.1 动作空间（全部用 llama.cpp 已暴露的接口实现）

| 动作 | 含义 | llama.cpp 实现 | 代价 |
|---|---|---|---|
| RETAIN | KV 留在 slot 中 | 什么都不做（slot 固定） | 占用预算 × 工具时长 |
| OFFLOAD | 存到主机内存或磁盘，回来时恢复 | `/slots/{id}?action=save` → `erase` → `restore`（需要 `--slot-save-path`） | 2 × KV 字节数 / 带宽 |
| DROP | 丢弃，回来时重算 | `action=erase`，之后重新 prefill | 上下文长度 / prefill 速率 |
| TRIM（Phase 3） | 只删掉无用段落 | 见 §7 | 质量风险 |

另有一个共享的机制：**共享前缀固定常驻**（system + few-shot），任何 agent 的驱逐都不会碰到它。

### 6.2 代价模型（可以参考 InferCept 的 "min-waste" 思路，见 §11）

在 agent 发出工具调用时，对它的 KV（大小为 S tokens）估计：

```
waste(RETAIN)  = S × E[tool_delay]                          # 占着不用的 token·秒
waste(OFFLOAD) = S × (t_save + t_restore)                    # 换入换出期间的占用 + 带宽
waste(DROP)    = S_active_others × t_prefill(S)              # 重算时挤占别人的 token·秒
```

选 waste 最小的动作。只在内存压力超过阈值时才触发 OFFLOAD/DROP；压力低时一律 RETAIN，避免无谓开销。所需的参数全部**在线估计**：
- `E[tool_delay]`：按工具类型统计的移动平均（或分位数）
- `t_prefill(S)`：从 llama-server 返回的 `timings` 中拟合
- `t_save/restore`：实测（Phase 1 E4 顺便测出 swap 带宽）

### 6.3 Admission
按"当前活跃 KV + 新任务预计峰值 KV"准入。峰值 KV 用 obs_size × 预测步数估计，按问题类型分别学习。准入失败的任务排队，而不是让在途请求一起 overflow。之前的实测显示，在 llama.cpp 中活跃 KV 溢出会让所有在途请求返回 HTTP 500（需要重新验证）。

### 6.4 对照组（课程明确要求）
1. FIFO + 固定 K（多个 K）
2. **Best fixed-K**（在 dev 上调出来）
3. llama.cpp 默认 slot/缓存行为（fixed cache policy）
4. always-RETAIN、always-DROP、always-OFFLOAD、LRU 驱逐
5. 本方法（代价模型）
6. 本方法 + 消融：去掉 admission / 去掉 offload / 用固定的而不是在线的延迟估计

### 6.5 必须展示"没有收益的情况"
- 工具延迟接近 0 时，RETAIN 总是最优，本方法应该退化为 always-RETAIN。
- KV 预算充足时，所有策略表现相同。
- 全是短上下文时，DROP 很便宜，offload 没有意义。

把这几个点在图上标出来，报告会更有说服力。

---

## 7. Phase 3：Selective KV（beyond，约 1.5 周）

**问题**：agent 回到推理之前，能否只保留有用的 observation？

### 7.1 选哪些段落（scorer）
| scorer | 说明 |
|---|---|
| Oracle | 用 HotpotQA supporting facts 标注，作为上界 |
| Recency | 只保留最近 k 个 observation（相当于 sliding window） |
| Relevance | 段落与 question + 最近 thought 的 BM25 或 embedding 相似度 |
| Agent-referenced | 段落里的实体是否出现在后续 thought 中（几乎零开销的启发式） |

### 7.2 怎么删（两种实现，这是要比较的核心）
| 实现 | 做法 | 优点 | 缺点 |
|---|---|---|---|
| **文本级压缩** | 从 prompt 中删掉段落，从删除点开始重新 prefill | 精确；不改 runtime | 删除点之后的 KV 全部失效，需要重算 |
| **KV 级区间删除** | 直接从 slot 的 KV 中删掉该段 token，后续 token 做位置平移（RoPE shift） | 不用重算 | 近似：后续 token 的 KV 是在"看过"被删段落的条件下算出来的 |

KV 级删除用到 llama.cpp 内部已有的 `llama_memory_seq_rm` / `llama_memory_seq_add`（llama-server 的 context shift 功能就是基于它们实现的）。我们只需给 llama-server 加一个小 endpoint，比如 `POST /slots/{id}?action=drop_range&p0=&p1=`，预计改动约 100 行 C++。**这正好是一次深入 inference runtime 内部的学习机会。**

退路：如果 patch 做不出来，就只做文本级压缩加 scorer 对比。仍然能回答 RQ3 的一半问题，而且完全符合"只用 runtime 暴露的控制"这一要求。

### 7.3 评估
- 质量：EM / F1；**supporting-fact recall**（删掉的段落里有没有 supporting fact）。
- 系统：每个 agent 的峰值 KV、同一预算下可容纳的并发 agent 数、重算 token 数、goodput。
- 关键图：质量-内存 Pareto 曲线（x = 平均 KV 占用，y = F1），每个 scorer × 实现组合画一个点。

---

## 8. 评估指标（统一）

- 任务级：completed tasks、完成延迟 p50/p95/p99、排队时间、吞吐（tasks/min）、**goodput（答对的 tasks/min）**
- 请求级：TTFT、prefill/decode token 数、cached token 数（prefix 命中率）、重算 token 数
- 内存：峰值 KV 占用、KV 时间线、进程/GPU 峰值内存、swap 字节数
- 可靠性：失败数、超时数、context overflow 数
- 质量：EM、F1、supporting-fact recall；可选：用 API 大模型做 LLM-judge
- 公平性：长任务与短任务的 slowdown 分布（课程要求检查长任务是否被不公平地延迟）
- 开销：每次策略决策的耗时（µs），以及 save/restore 的耗时

**正确性检查（课程明确要求）**：性能提升不能来自跳过任务、缩短输出或降低质量。所以每张性能表都要同时报告 completed 数、平均输出长度和 F1。

---

## 9. 运行环境与 AWS / API 用法

### 9.1 模型与 runtime（主实验中固定不变）
- Runtime：**llama.cpp `llama-server`**（课程给的起点，slot save/restore 接口齐全）。所有实验都记录 build 号。
- 模型：建议 **Qwen2.5-7B-Instruct Q4_K_M**，或同级别的 7–8B 模型。1.5B 模型在 HotpotQA ReAct 上的 F1 太低（之前实测约 0.2–0.4），质量信号噪声大。先在 dev 上跑 quality check，确认 F1 足够高后再锁定。
- KV 占用估算公式（报告里要写）：`bytes/token = 2 × n_layers × n_kv_heads × head_dim × bytes_per_elem`。例如 Qwen2.5-7B（28 层、4 个 KV head、head_dim 128、fp16）约为 56 KiB/token，16k tokens 约 0.9 GB。

### 9.2 AWS
- 主实验机：单卡 GPU 实例（如 g5.xlarge / A10G 24GB，或 g4dn.xlarge / T4 16GB）。**通过 `-c` 人为限制 KV 预算**（比如 8k–32k tokens）来模拟边缘设备，这符合课程"limit large ones usable memory"的要求。
- 可选：一台 CPU 实例（如 c7i）复现关键实验，展示"真·边缘"场景下 prefill 更贵、DROP 更不划算这一不同的取舍。
- 实践：用 spot 实例跑 sweep；结果写入 S3 或直接 rsync；实验脚本支持断点续跑。**不用的时候一定关机。**

### 9.3 老师提供的模型 API 怎么用（不用于被测模型）
被测模型必须是本地模型，否则无法控制 KV。API 适合做以下几件事：
1. **LLM-judge**：EM 对改写过的答案判错较多，用大模型判定语义正确性，作为补充质量指标。
2. **质量上界**：用 API 模型跑同一批 ReAct，作为"小模型答题质量是否合理"的参照。
3. **真实网络延迟的工具（可选）**：加一个 `ask_expert` 工具调用 API，它的延迟分布是真实的而不是注入的，可以检验代价模型对延迟预测误差的鲁棒性。

---

## 10. 时间线与分工（今天 10/28，按约 6 周规划）

| 周 | 内容 | 里程碑 |
|---|---|---|
| W1（10/28–11/4） | llama-server、HotpotQA ReAct agent、事件日志；锁定模型；Phase 1 E1–E5 | baseline 曲线 + KV 组成分析图；**Phase 1 截止 11/4** |
| W2（11/5–11/10） | 补 Phase 1 实验；开始 Phase 2 机制 | 向老师确认方向 |
| W3（11/11–11/17） | Phase 2：slot 固定、OFFLOAD/DROP、代价模型 | 机制跑通 + 单元测试 |
| W4（11/18–11/24） | Phase 2 全量对比 + 消融；开始做 Phase 3 的 llama.cpp patch | Phase 2 主图 |
| W5（11/25–12/1） | Phase 3 selective KV + 质量评估 | Pareto 图 |
| W6（12/2–12/8） | 补实验（no-benefit 情形、公平性、开销）、写报告 | 报告 + 可复现脚本 |

建议分工（按 3–4 人调整）：
- **Runtime & 基础设施**：llama-server 配置、AWS、slot API 封装、Phase 3 的 C++ patch
- **Agent & Workload**：HotpotQA 工具/索引、ReAct prompt、负载生成器、质量评估
- **Scheduler & KV Manager**：admission、代价模型、各 baseline 策略
- **测量 & 分析**：插桩、监控、画图脚本、报告主笔

---

## 11. 阅读清单（按概念对应到本项目）

| 概念 | 文献 / 系统 | 在本项目中的位置 |
|---|---|---|
| Paged KV、KV 内存管理 | vLLM / PagedAttention (SOSP'23) | 理解 KV 为什么是瓶颈 |
| 前缀共享 | SGLang / RadixAttention | 共享前缀常驻 |
| 工具调用期间的 KV 处理 | **InferCept (ICML'24)**：preserve / discard / swap 的 min-waste 代价模型 | **Phase 2 最直接的参考**；我们的不同：边缘场景 + llama.cpp + selective 动作 + supporting facts 分析 |
| Agent 级调度 | Autellix（program-level LAS）、Parrot | admission 和公平性 |
| 工具调用期间的 KV 保留时长 / 预取 | Continuum（KV TTL）、KVFlow（建议核实后再引用） | Phase 2 related work |
| KV 卸载 / 分层存储 | LMCache | OFFLOAD 动作 |
| 非前缀 KV 复用 | CacheBlend | `doc_overlap` 场景的进一步方向 |
| Selective KV / KV 压缩 | StreamingLLM、H2O、SnapKV | Phase 3 scorer 与"近似删除"的理论背景 |

这个对照表也可以直接作为报告 Related Work 的骨架。

---

## 12. 风险与应对

| 风险 | 应对 |
|---|---|
| 小模型 HotpotQA 质量太差，掩盖了质量变化 | 换 7–8B 模型；同时报告 supporting-fact recall 这一与模型无关的指标 |
| llama.cpp 版本更新导致接口或行为变化 | 锁定 build 号；Phase 1 开头重新验证关键行为（slot save/restore、overflow 行为、`--cache-ram` 默认值） |
| Phase 3 的 C++ patch 做不出来 | 退回文本级压缩，结论仍然完整 |
| 代价模型收益不明显 | 这本身就是结论：用 Phase 1 的数据解释为什么（比如工具延迟太短）。课程要求展示 little-benefit 情形 |
| AWS 费用 | 先在笔记本上用小模型调通全部流程，再上云跑 sweep；使用 spot 实例 |

---

## 13. 与之前框架（已 revert 的 ef37620）的关系

那个提交实现了一套分层框架（backend / scheduler / agent / workload / evaluation），以及很多调度策略（RR、SRPT、MLFQ、LAS、mem_aware、多种 retention）。按本设计的思路：
- **可以复用的**：llama.cpp backend（含 slot save/erase/restore）、HotpotQA ReAct agent 与离线工具、事件日志与指标、fake backend 测试。
- **应当收窄的**：RR/SRPT/MLFQ/LAS 等抢占式调度不再作为贡献，最多保留 FIFO 和 ready-first 作为框架；retention 策略的思路并入 §6 的代价模型。
- 它的 README 里有几条实测结论（KV 池在启动时预分配、活跃 KV 溢出会导致在途请求全部 500、swap 优于 erase），可以作为 Phase 1 的假设来源，但需要在锁定的 build 和模型上重新测量。
