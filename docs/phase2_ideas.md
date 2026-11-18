# Phase 2 方向（2026-11-18 确定）

## 0. 目标和指标

**在显存固定的边缘 GPU 上，让同样大小的 KV pool 每分钟完成更多任务，延迟更低，答题质量不变或只略降。**
课程要求贡献在 KV cache 的管理上，每个 cache 操作都必须用 llama.cpp 实际提供的控制（或我们公开说明的补丁）。

每次实验都按同一张表报告：

| 层级 | 指标 |
|---|---|
| 主指标 | goodput（每分钟 EM 答对数）、任务延迟 p95 / p99 |
| 请求级 | TTFT p50 / p95（排队 + prefill）、TPOT p50 / p95（每个输出 token 的 decode 时间） |
| 约束 | 完成数、失败数（overflow）、EM / F1（F1 变化在 ±0.03 内算持平；同一配置两次重复之间就会差这么多） |
| 解释 | KV 占用（平均 / 峰值）、闲置 KV 比例、重算 token 数、前缀缓存命中率 |

这些指标在 `summary.json`、`analysis/aggregate.py` 和 Grafana 看板（`tools/observability/`）里都有。

## 1. 确定的方向和顺序

**底座 → selective KV → 想法 B**，一步一步做：每一步做完、汇报、确认后再进下一步。

### 1.1 底座：按 KV 预算放任务，绝不超出 pool

- **解决什么**（Phase 1 E2）：上下文在任务中增长 2.3–4.2 倍，超出 pool 时所有在跑的请求一起失败，
  所以 fixed K 只能按最坏情况少放任务（8k / 16k pool 只有大 pool 吞吐的 48% / 80%）。
- **做法**（`edgeserve/kvm.py`、`edgeserve/policies.py`）：
  - 准入：按每类任务预测的峰值 KV 预留，总和 ≤ ρ × pool 才放进来；峰值先用 Phase 1 数据，运行中在线更新。
  - 发请求前检查：估算这次请求后 slot 的 token 数，放不下就先处理其他 slot（已结束任务的 KV → 等工具的
    agent 的 KV），仍放不下就让请求等待。绝不发出会导致 overflow 的请求。
  - 由管理器选择 slot，优先复用同类任务留下的 slot（保留 system prompt 的前缀缓存）。
  - system prompt 只存一份：我们给 llama-server 加的 fork 接口（`tools/llama_patch/fork_slot.diff`，
    底层是 `seq_cp`，只给 cell 加一个序列标记，不复制数据），X3 验证过输出一致。
- **状态**：已实现（策略 `base`、`base_fork`），16k pool 冒烟测试 0 次 overflow。11/17 用当时的组合配置
  （底座 + fork + 压缩 + 已删除的"清理闲置 KV"策略，ρ=2.0）测得 goodput 28.6，比 fixed K=4 的 21.1 高 36%；
  这个配置在 11/18 清理代码时拆掉了，正式数字要在底座这一步用 `base_fork` 重新测。
  还缺：按正式场景调参和重复实验。
- **新颖性**：低（InferCept、Continuum 做过 tool call 时的 KV 保留 / 丢弃），作为基础设施，不算主要贡献。
- **selective KV 依赖它**：显存压力的判断、"先裁剪、再整份丢弃"的顺序都在这里实现。

### 1.2 selective KV：按段选择保留哪些 KV，在 KV 里直接删除，不重算（主要贡献）

**要回答的问题**：一个 agent 的 KV 里，哪些该留、哪些可以删，怎么删最便宜。

**依据**：
- Phase 1：`react_long` 检索回来的 token 只有 4.5% 是 supporting facts；KV 增长几乎全部来自检索结果。
- 阶段 1（见第 3 节）：一次检索平均增加约 860 个 token；去掉相关段落后 F1 不变，峰值 KV 减半，
  131k pool 下 goodput 从 23 涨到 45 /min，TTFT p95 2.2 s → 0.14 s，TPOT p50 38 → 20 ms。
  说明检索内容是 KV 浪费的大头，而且大部分删掉不影响答题。

**两个层面**：

| 层面 | 内容 |
|---|---|
| 执行机制（原 K2） | 给 llama-server 加一个"删除 slot 中指定范围的 KV"的接口：`seq_rm` 删除这段 cell，`seq_add` 把后面的 token 位置往前挪（重新施加 RoPE），再更新 slot 记录的 token 序列。llama.cpp 的 context shift 就是这样实现的（`tools/server/server-context.cpp:3858–3871`），我们只是把固定的"删 n_keep 之后一段"改成任意范围。补丁规模预计和 fork 相当。 |
| 选择策略 | 决定删哪些内容（见下）。 |

**设计空间**：
1. **粒度**：整条检索结果 / 段落 / 句子。不做逐 token（需要 attention 分数，llama.cpp 不对外提供，前人已做）。
2. **打分器**（判断内容还有没有用）：
   - oracle：HotpotQA 标注的 supporting facts，只作为效果上界；
   - BM25：句子和"问题 + 已写的思考"的相关度；
   - 被引用的实体：句子里的实体在后续思考中出现过就保留；
   - 时间先后：越早的检索结果越可能已经用完。
   - `edgeserve/compact.py` 已实现这四种打分器的文本版本（entity：句子里的大写词在后续思考中被提到过的个数），
     可直接复用来决定删哪些句子。
3. **触发时机**（memory-aware）：
   - agent 写完下一步思考之后（已经"读过"这段内容）；
   - 只在显存紧张时：和底座结合，显存不够时先裁剪等工具的 agent 的旧检索结果（不用重算），
     还不够再整份丢弃 KV（要重算）；显存宽松时不裁剪，质量不受影响。
4. **对照：改文本 + 重算**：`compact.py` 现在的做法（也是前人"截断旧工具输出"的做法）是改写 prompt 后重新 prefill。
   在 KV 里直接删 vs 改文本重算，省多少时间、质量差多少，是本方向的核心对比之一。

**新颖性**：
- 前人的 KV 裁剪（H2O、SnapKV、StreamingLLM）针对单个请求，按 attention 分数逐 token 删，不知道 agent 的结构；
- 前人的 agent 上下文管理（Complexity Trap、PACE 等）改文本，需要重算；
- 我们利用 agent 结构，在 KV 层面按段删除，不需要重算，按显存压力触发，服务多个 agent。没有找到这样的组合。

**预期提升的指标**：每个 agent 占的 KV 变小 → 紧张 pool 下放更多任务（goodput）；pool 里 KV 总量变小 →
decode 每步读的数据更少（TPOT）；和整份丢弃相比不用重新 prefill（TTFT）。

**风险**：
1. 质量：后面 token 的 KV 是在"看过被删内容"的情况下算出来的，删掉后有不一致；位置平移会重新施加 RoPE。需要实测。
2. 实现：删除后，客户端下次发的 prompt 必须和服务器缓存的 token 序列完全对得上，否则前缀缓存失效、整段重算。
   被删内容的起止要精确落在 token 边界上，这是主要的工程难点。
3. 和 fork 的交互：共享的 system prompt 不能被删（只删 slot 自己的检索结果）。

**步骤**：
1. **可行性**（基本不用 GPU）：写补丁，解决客户端 token 对齐，验证删除后缓存命中正常、输出正常。
2. **质量实验**（pool 给足）：打分器 × 保留比例 → F1 和节省的 KV；同时和"改文本 + 重算"对比。
3. **serving 实验**（紧张 pool）：底座 vs 底座 + 按压力触发的 selective KV，按第 0 节的指标表报告。

### 1.3 想法 B：分组 KV pool，按 agent 的增长来分配和迁移

#### Phase 1 的依据
llama.cpp 只提供两个极端，各有一个 Phase 1 测出的缺点：

| 模式 | 优点 | 缺点（Phase 1 测量） |
|---|---|---|
| 共享 pool（`--kv-unified`） | 容量灵活，能吸收上下文增长 | 每步 decode 要读到 pool 里最高的已用 cell；K=32 时每 token 耗时 254 ms，吞吐下降（E1） |
| 每 slot 独立 KV | 每个 slot 只读自己的 KV；K=16 时快 27% | 每个 slot 容量固定为 `-c / np`，上下文增长 2–4 倍就放不下（E2） |

X1 微基准（`docs/phase2_notes.md` §6）：decode 每步 ≈ 13.0 ms + 0.127 ms × (KV 跨度 / 1k cell)，开销来自显存带宽。

#### 做法
1. **llama.cpp 补丁**：把 KV 分成 G 组，组内共享，组间独立。
   - 源码依据：`src/llama-kv-cache.cpp` 里的 `seq_to_stream` 就是"序列 → stream"的映射表。目前只允许 `n_stream == 1`（全部共享）或 `n_stream == n_seq_max`（每序列一个），见第 134 行的 assert 和第 147–152 行。
   - 改成 G 组，例如 `seq_to_stream[s] = s * G / n_seq`，看起来改动不大。
   - 多 stream 时，CUDA 的 mask scan 会按每个 stream 自己的末尾裁剪（`fattn-common.cuh`，条件是 `Q->ne[3] > 1`），所以每组只读自己的跨度。
2. **调度器**（扩展 `kvm.py`）：
   - 按每个 agent 的预测峰值做装箱，决定它进哪个组（哪些 slot id）；
   - 某组快满、其他组还有空间时，用 save → restore 把 agent **迁移**到别的组，代替丢弃 KV 后重新 prefill；
   - 用 X1 的代价模型选 G：G 越大，读的跨度越短；但每组容量越小，越容易需要迁移。
3. G=1 就是共享 pool，G=K 就是每 slot 独立 KV。我们要找中间最好的点，并证明按增长装箱 + 迁移比两个极端都好。

#### 为什么新
vLLM 这类分页系统没有这个问题，所以没人研究。但 llama.cpp 是边缘部署最常用的运行时，这是边缘场景专属的设计点，和课题里的 "edge" 直接对应。原方向 1（闲置 KV 拖慢 decode）并入这里，X1 的测量作为动机。

#### 要对比的东西
- 共享 pool + 最优 fixed K（Phase 1 E1：K=8 时 45.2 tasks/min）；
- 每 slot 独立 KV（Phase 1 E1b：K=8/16/32 时 43.0 / 54.8 / 64.5 tasks/min）；
- 分组但不迁移（只装箱）；
- 分组 + 装箱 + 迁移。
- 场景：充足 pool（131k，E1 场景）和紧张 pool（16k/32k，E2 场景）都要测。B 的卖点就是两种场景下都不差。

#### 风险
- **补丁可行性未验证**：多 stream 时 ubatch 怎么切分（`split_equal`）、每组的 KV 大小怎么算、server 是否假设 stream 数只能是 1 或 np，都需要看源码确认。
- 共享 system prompt（fork）只能在组内用，所以每组要存一份 prompt。
- 迁移用 save/restore，会阻塞所有 decode（Phase 1：3k token 约 119 ms）。

#### 实现成本
中到高。需要先花约 1 天看源码、做原型，确认补丁可行，再决定做不做。

---

## 2. 探索过、没有采用的想法（记录，报告里可作为发现或否定结论）

| 想法 | 结论 | 依据 |
|---|---|---|
| 想法 A：按显存压力减少检索内容 | 不采用为方法；结果保留为"KV 浪费在哪里"的证据 | 阶段 1：减内容不掉质量，所以一直减就行，"按压力自适应"没有收益空间（第 3 节） |
| 跨 agent 共享同一段落的 KV | 放弃 | 阶段 1 的 L0 里 2,575 个不同段落只有 65 个被多个任务检索到，重复文本只占 2.3%；L2 为 0 |
| C：KV 优先给可能答对的 agent | 暂不做，可作为以后的扩展 | 被强制作答的任务占 12%，却占 30% 的 KV·时间、3% 的答对数；第 5 次调用后强制作答可使服务时间 p95 −17%，最多损失 3.8% 答对数 |
| K1：KV 量化（q8_0 / q4_0） | 暂不做 | 现成参数 `-ctk/-ctv`；方法已有，新颖性低 |
| 投机解码（ngram / 1.5B draft） | 不做（不是 KV 方案） | decode 占服务器时间 88%，输出 54% 的词延续上下文中的 3 词片段，加速潜力大，但和 KV 管理无关 |
| 原方向 1：闲置 KV 拖慢 decode | 并入想法 B | X1 已测出机制 |
| 原方向 2：压缩旧检索结果后重算 | 变成 selective KV 的对照 | 前人已做（Complexity Trap 等） |
| 原方向 3：fork 共享 system prompt | 并入底座 | 补丁已完成 |

## 3. 阶段 1 结果（2026-11-18）：检索档位的 KV 成本、性能和答题质量

设置：`react_long` 的四个检索档位，dev 200 题（每档相同题目），pool 131k（无显存压力），K=8，
fixed_k，每档 2 次重复。配置 `configs/sweeps/p2a_stage1.yaml`，分析 `analysis/obs_levels.py`，
结果 `results/p2a_stage1/levels.json`。ΔF1 为逐题配对（两次重复取平均）相对 L0 的差，bootstrap 95% CI。

**性能**（rep0）：

| 档位 | goodput | 吞吐 | 任务服务时间 p50 / p95 | TTFT p50 / p95 | TPOT p50 / p95 |
|---|---|---|---|---|---|
| L0（段落 + 9 个相关段落） | 23.0 /min | 44.1 /min | 9.6 / 21.7 s | 0.36 / 2.20 s | 38 / 90 ms |
| L2（只有段落） | 44.6 /min | 86.6 /min | 4.9 / 11.7 s | 0.07 / 0.14 s | 20 / 25 ms |

**KV 和质量**（两次重复合并）：

| 档位 | 检索返回 | F1（两次） | ΔF1 vs L0 [95% CI] | SF 覆盖 | LLM 调用 | 搜索次数 | 走到步数上限 | 一次检索 token p50 / p95 | 峰值上下文 mean / p95 | prefill / 任务 | 吞吐（131k, K=8） |
|---|---|---|---|---|---|---|---|---|---|---|---|
| L0 | 段落 + 9 个相关段落 | 0.623 / 0.640 | – | 0.83 | 3.58 | 1.96 | 12% | 858 / 1,655 | 3,428 / 5,215 | 1,925 | 44–46 /min |
| L1 | 段落 + 4 个 | 0.642 / 0.613 | −0.003 [−0.037, +0.030] | 0.83 | 3.54 | 1.89 | 12% | 577 / 844 | 2,712 / 4,154 | 1,210 | 62–64 /min |
| L2 | 只有段落 | 0.660 / 0.674 | +0.036 [−0.010, +0.084] | 0.74 | 3.78 | 2.27 | 13% | 131 / 244 | 1,926 / 2,518 | 418 | 86–87 /min |
| L3 | 段落前 3 句 | 0.616 / 0.636 | −0.005 [−0.060, +0.051] | 0.69 | 3.96 | 2.37 | 17% | 121 / 195 | 1,907 / 2,491 | 392 | 81–82 /min |

噪声：同一档位两次重复之间，9–15% 的题 F1 不同，平均 F1 相差最多 0.03（比 Phase 1 估计的 ±0.012 大）。
L1 的 2 个失败是同一道题两次都格式解析失败（agent 问题，不是系统故障）。

结论：
1. **检索内容是 KV 和延迟的大头，而且大部分可以去掉。** L0→L2 峰值 KV 降 44%（p95 降 52%），prefill 降 78%，
   goodput 和吞吐翻倍，TTFT 和 TPOT 大幅下降，F1 不变（L2 略高，但 CI 包含 0）。
   相关段落是 BM25 找来的干扰项，和 Phase 1 的"只有 4.5% 有用"一致。
2. **降到 L3 开始出现行为代价：** agent 多搜（1.96→2.37 次）、走到步数上限的比例 12%→17%，SF 覆盖 0.83→0.69；
   L3 不比 L2 省多少 KV，所以不划算。
3. **对方向的影响：** 想法 A 的"按压力自适应"没有收益空间；但"检索内容大多可删"直接支持 selective KV：
   与其一开始就少给内容，不如让 agent 完整读一遍，用完后在 KV 里删掉。

## 4. 参考
- InferCept: Abhyankar et al., ICML 2024
- Continuum: https://arxiv.org/abs/2511.02230
- The Complexity Trap: https://arxiv.org/html/2508.21433v3
- BACM: https://arxiv.org/html/2604.01664v1
- PACE (ACL 2026): https://aclanthology.org/2026.acl-long.1252.pdf
- KAIROS: https://arxiv.org/pdf/2604.16682
- Aries: https://arxiv.org/pdf/2607.29069
- H2O (NeurIPS 2023), SnapKV (2024), StreamingLLM (ICLR 2024): token-level KV eviction for single requests
- SGLang RadixAttention, vLLM automatic prefix caching: prefix sharing
