# AI 使用记录（AI Tracking）

本文件记录本项目开发中对 AI 助手的使用情况：做了什么、人做了哪些决定、怎样验证、哪些结果需要谨慎看待。

- 工具：Claude Code（模型 Claude Opus 5.5）
- 时间：2026-10（单次长会话，中途有一次上下文压缩）
- 范围：本仓库全部代码、配置、脚本、测试和 README 均由 AI 在用户指导下生成；实验由 AI 在本机（Apple M4 / 24 GB）上运行
- 状态：所有文件尚未提交到 git

## 1. 交互时间线

| # | 用户请求（原意） | AI 的工作 | 用户的决定或纠正 |
|---|---|---|---|
| 1 | 在 HotpotQA 上用小模型和 llama.cpp 搭一个 FCFS 基线，形成可评估调度的体系；说明如何申请 AWS GPU | 写出分层计划（L0 模型 → L5 评测），选定 Qwen2.5-1.5B-Instruct Q4_K_M、llama-server、ReAct agent；写 `docs/aws_gpu.md` | — |
| 2 | 要注意扩展性：模型、推理框架、任务、评测框架以后都要能换 | 改成插件式架构：`@register(kind, name)` + YAML `type:`，层间只通过 `core/types.py` 交互 | 用户提出的架构约束 |
| 3 | Agent 也要按编排、工具、记忆三个维度解耦 | 拆成 `agents/orchestration`、`agents/tools`（含延迟模型和执行器）、`agents/memory` 三个独立目录 | 用户提出的架构约束 |
| 4 | 直白解释 agent 使用场景、测试指标、是否涉及 KV cache | 文字说明（无代码改动） | — |
| 5 | 暂不用 AWS，想实现抢占式调度，列出可选方案 | 提出约 10 种调度方式 | — |
| 6 | 实现 4、5、6、7、8、10 | 实现 RR、SRPT、MLFQ、Agent 级 LAS、内存感知准入/抢占、KV 保留策略（lru/largest/tool_aware × erase/swap）；在真实模型上跑实验 | 用户选定要实现的子集 |
| 7 | 中文总结 | 中文总结报告 | — |
| 8 | 不用再跑实验，写 AI_Tracking.md | 本文件 | 用户决定暂不跑完整 sweep |

## 2. AI 生成的主要内容

- **框架**：`agentserve/core`（类型、注册表、事件日志、配置合并与 sweep 展开）、`runner.py`、`experiments/run.py`
- **后端**：`backends/llamacpp.py`（启停 llama-server、流式 `/completion`、slot save/erase/restore）；`backends/fake.py`（模拟 llama.cpp 实测行为，用于无模型测试）
- **调度**：`schedulers/` 下的 FCFS 基线、分段抢占机制、slot 固定与 KV 记账、在线估计器、5 种抢占策略、4 种保留策略
- **任务与 agent**：HotpotQA 数据加载、离线 search/lookup 工具、ReAct 与 single-shot 编排、full_history 与 sliding_window 记忆、工具延迟模型
- **负载、评测、监控**：到达过程、任务混合、EM/F1、系统指标与公平性指标、进程内存与后端探针、画图脚本
- **配置、脚本、文档**：`configs/`、`scripts/`、`README.md`、`docs/aws_gpu.md`
- **测试**：`tests/test_layers.py`、`tests/test_policies.py`，共 31 个，全部通过

## 3. 人工决策点

- 架构上的松耦合分层，以及 agent 三维解耦，由用户提出。
- 暂不使用 AWS，先在本机实现抢占式调度，由用户决定。
- 从 AI 列出的方案中选择实现 #4、#5、#6、#7、#8、#10，由用户决定。
- 暂不运行完整的 `phase2_policies` / `phase2_retention` sweep，由用户决定。

## 4. 验证方式

- **单元/行为测试**：用 fake 后端验证层间契约、每种策略的排序行为（如短请求能抢占长请求）、分段输出与一次性输出一致、跨段停止符、溢出断崖与 mem_aware 的对比、erase/swap 模式、tool_aware 选择被驱逐者。
- **对 llama.cpp 行为的实测**（build 11429），而不是凭文档假设：
  - `/v1/chat/completions` 不能续写 assistant 前缀，改用 `/apply-template` + `/completion`
  - `id_slot` 可固定 slot；slot save/erase/restore 可用，存盘约 28 KB/token；restore 需要空闲 KV 单元
  - 运行中请求的 KV 溢出时，所有在跑请求一起返回 HTTP 500
  - `/metrics` 没有 KV 用量指标
- **真实模型实验**：Phase 1 冒烟（8 任务，K=1/4）、调度策略冒烟（7 种策略各 8 任务）、`phase2_memory`（30 任务，KV 池 6144 token）。

## 5. AI 出错并被修正的地方

| 问题 | 发现方式 | 修正 |
|---|---|---|
| 配置继承时，被覆盖的 section 和旧 section 合并，换了 `type` 后旧参数残留 | 运行时报错/结果不对 | 带 `ref:` 或换了 `type` 的子配置改为整体替换 |
| Agent EM 为 0：模型照抄工具说明里的占位符 `search[entity]` | 质量检查实验 | 工具说明改为具体示例，增加两跳推理提示；EM 0.25，F1 0.34 |
| FCFS 任务准入存在竞态 | 测试失败 | 增加预留计数 |
| fake 后端没有为运行中请求预留 KV，模拟不出溢出 | 测试未触发预期行为 | 增加 `reserved` 记账 |
| mem_aware 多个实例共享状态 | 代码审查 | 改为 mixin 组合 |
| sweep 点名称重复，结果目录互相覆盖 | 检查结果目录 | 名称包含 K/模式/保留策略，重复时加 `#i`，加唯一性测试 |
| swap 模式一半任务失败：先恢复后腾空间 | 真实模型实验（15/30 失败） | 先腾空间再恢复，恢复失败退回重算；保留旧 slot 以复用共享前缀 |
| `--set` 传入带 `ref:` 的值报 KeyError | 运行实验 | 增加 `config.resolve` |

## 6. 结果的可信度与局限

- 所有真实模型结果都只有 1 个 seed，每组最多 30 个任务，只能看趋势。
- FCFS 的延迟只统计完成的任务，和 mem_aware 不能直接比较。
- `phase2_memory` 中 mem_aware + LAS 那一行是在修复 swap/restore bug 之前跑的，需要重跑。
- erase 模式下调度器的 KV 估计有一次超过池子约 10%（prompt 长度是估计值）。
- 轻负载下各策略差别很小，策略间的真正对比需要完整 sweep 和多个 seed，目前尚未运行。
- HotpotQA 质量（EM/F1）受 1.5B 小模型限制，只用于确认调度没有破坏输出，不代表 agent 能力上限。

## 7. 复现

```bash
uv sync
scripts/setup_llamacpp.sh && scripts/download_model.sh && scripts/download_hotpotqa.sh
uv run pytest                                                          # 31 passed
uv run python experiments/run.py configs/experiments/phase2_memory.yaml
uv run python experiments/run.py configs/experiments/phase2_policies.yaml    # 尚未运行
uv run python experiments/run.py configs/experiments/phase2_retention.yaml   # 尚未运行
```
