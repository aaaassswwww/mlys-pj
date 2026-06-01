# Phase 3 Report

## 概述

本阶段的目标是实现一个**提示词驱动的 LLM runtime 迭代优化 agent**。  
agent 在项目根目录直接生成 `engine.py`，并在多轮评测中依据 correctness 与 throughput 数据持续修订 runtime 实现，最终将评测结果写入根目录 `output3.json`。

与前一阶段不同，这一阶段的重点不再是单个 CUDA kernel 的搜索，而是围绕以下 serving runtime 能力进行持续优化：

- `create_engine(model_config, weight_dir, device="cuda")`
- `prefill(request_ids, input_ids)`
- `decode(request_ids, token_ids)`
- `remove(request_ids)`

整个流程强调：

- correctness-first
- 数据驱动的 runtime 修订
- 以提示词为核心的候选生成与迭代
- 根目录交付契约（`engine.py`、`output3.json`）

## 已完成的核心工作

### 1. 新建 Phase 3 agent 工作流

在 `profiler_agent/phase3/` 下实现了完整的 Phase 3 优化链路，包括：

- `workflow.py`
- `optimizer.py`
- `generator.py`
- `prompts.py`
- `evaluator.py`
- `candidate_store.py`
- `models.py`

这条链路负责：

- 生成候选 runtime
- 运行 correctness 检查
- 运行 prefill / decode / mixed throughput benchmark
- 根据反馈继续迭代修订
- 将 best candidate 写回根目录 `engine.py`

### 2. 建立根目录交付契约

按照当前项目需求，最终产物不再放在 `workspace/` 下，而是直接写到项目根目录：

- `engine.py`
- `output3.json`

`run.sh` 也已调整为以 root-level 产物为中心，执行顺序为：

1. 运行 Phase 3 agent
2. 生成根目录 `engine.py`
3. 执行本地 selfcheck
4. 记录 benchmark 输出
5. 汇总写入根目录 `output3.json`

### 3. 迁移并接入 runtime 基础组件

从已有 runtime 项目中迁入并整理了 Phase 3 所需的支撑模块，包括：

- request state
- KV cache
- model loader
- rope / layers
- scheduler

这些模块统一放在根目录 `runtime/` 下，作为 `engine.py` 的本地依赖。

## 提示词驱动的 runtime 迭代优化 agent

本阶段最关键的工作，是把生成链从“模板搜索”推进为**提示词驱动的 runtime 修订 agent**。

### 1. LLM 优先的候选生成路径

当前 `Phase3CandidateGenerator` 的主路径是：

1. 构造 system prompt
2. 构造包含评测反馈的 user prompt
3. 调用 LLM 生成新的 `engine.py`
4. 校验返回结果是否满足 runtime 契约
5. 若有效则进入评测

也就是说，runtime 候选的主生成方式已经是：

- **基于上下文提示词的源码修订**

而不是简单的固定模板切换。

### 2. 提示词中注入 runtime 约束

prompt 中明确编码了以下约束：

- 只能生成单文件 `engine.py`
- 必须位于项目根目录
- 不允许创建或依赖 `workspace/`
- 必须暴露 `create_engine(...)`
- 必须支持 `prefill/decode/remove`
- 必须维护 request-local state
- 必须优先保证 correctness
- 正确候选存在后，应以 throughput 优化为主

这样 LLM 在每轮生成时，拿到的不是泛化任务，而是一个**被严格 runtime 契约约束的源码修订问题**。

### 3. 反馈驱动的提示词修订

每轮 prompt 不只包含“上一轮失败了/成功了”这种粗粒度信息，而是包含结构化运行反馈，例如：

- 上一轮 candidate 源码
- 上一轮 correctness 结果
- prefill / decode / mixed 三类 benchmark
- baseline benchmark
- 分项 speedup breakdown
- 当前 best candidate
- 当前 best correct candidate
- 最近几轮候选表现

因此 agent 在修订时具备：

- **基于性能观测结果的定向改写能力**

而不是盲目重写 runtime。

### 4. 单目标迭代优化机制

为了减少 LLM 每轮“同时优化所有指标”的发散行为，本阶段进一步将 prompt 收紧成**单目标优化模式**。

每轮都会从上一轮 benchmark 中自动识别最弱项，例如：

- `prefill_speedup`
- `decode_speedup`
- `mixed_speedup`

然后把它作为本轮唯一主目标写入 prompt，包括：

- `primary_target_metric`
- 该指标对应的 focus lenses
- guardrails
- “只做局部修订，不要同时大范围改所有阶段”的明确要求

这使得 agent 的行为更接近真实优化循环：

- 先定位短板
- 再围绕短板做局部修补
- 保持其余路径尽量稳定

### 5. deterministic 机制的定位

本项目中仍保留了一个很小的 deterministic fallback，但它的定位已经被弱化为：

- LLM 不可用时的兜底
- LLM 返回非法源码时的安全回退

它不再是 Phase 3 的主优化逻辑，也不是本阶段报告的重点。  
本阶段真正想强调的是：

- **LLM 根据 runtime 约束与 benchmark 数据主动修订 `engine.py`**

## 评测与优化闭环

当前 agent 已形成较完整的闭环：

1. 生成 candidate `engine.py`
2. correctness 检查
3. prefill / decode / mixed benchmark
4. 生成结构化反馈
5. 将反馈注入下一轮 prompt
6. 继续修订 runtime

其中 evaluator 重点检查：

- logits correctness
- request remove 语义
- KV cache 路径是否工作
- 预填充与增量 decode 的吞吐

这一闭环使得 runtime 优化不再依赖人工逐次修改，而是能够由 agent 在本地完成多轮自我修订。

## 当前结果与意义

从实现角度看，Phase 3 已经完成了以下转变：

- 从单阶段代码生成，转为多轮 runtime 迭代优化
- 从静态模板输出，转为提示词驱动的源码修订
- 从单指标优化，转为带 correctness 与 throughput 反馈的闭环优化
- 从 `workspace/engine.py` 契约，转为根目录 `engine.py` / `output3.json` 契约

也就是说，当前仓库已经具备一个可运行的：

- **prompt-driven runtime optimization agent**

它能围绕 LLM inference runtime 的真实 serving 目标，持续生成、评测、修订并写回最优版本。

## 后续可继续加强的方向

虽然当前主链已经建立，但后续仍可继续提升：

- 更细粒度的 decode 路径分析与专项 prompt
- 更强的 best-correct anchored patch mode
- 更细的 mixed workload 调度策略反馈
- 更贴近最终评测 trace 的本地 benchmark
- 更强的 prompt-to-diff 约束，进一步减少无关改动

总体来说，本阶段最重要的成果不是某一个固定 runtime 版本，而是：

- **一个能够自己围绕 runtime 数据持续迭代优化的 agent 框架**
