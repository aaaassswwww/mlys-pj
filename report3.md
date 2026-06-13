# MLSys Course Project Final Report

## 1. Basic Information

- Name: `肖羽平`
- Student ID: `23302010020`
- Report date: `2026-06-13`

## 2. Project Overview

这个课程项目一共分为三个阶段，我最终实现的是一个逐步演化的 agent 系统：

- **Phase 1**：自动化 GPU profiling agent目标是根据 profiling target 自动选择探测方式、调用 profiling 工具或生成 synthetic microbenchmark，并产出可解释的分析结果。
- **Phase 2**：LoRA operator optimization agent目标是围绕 `optimized_lora.cu` 生成候选实现，自动编译、做 correctness 检查、做 benchmark，并迭代优化。
- **Phase 3**：LLM inference runtime optimization agent
  目标是自动生成并迭代修订根目录 `engine.py`，支持 `prefill(...)`、`decode(...)`、`remove(...)`，并围绕 serving workload 做 runtime 优化。

我在三个阶段中的总体策略是：

1. 先把**最小可运行闭环**搭起来。
2. 再把闭环中的每个步骤结构化：生成、执行、校验、记录、反馈。
3. 在 correctness 站稳后，再把优化从“宽搜”逐渐改成“带反馈的局部修订”。

从自动化程度上看：

- Phase 1 的 profiling/pipeline/analyzer 已经比较自动化。
- Phase 2 中候选生成、编译、correctness、benchmark、报告记录基本自动化，但中间仍然需要我手工判断一些数值问题和 evaluator 契约问题。
- Phase 3 中 runtime 生成和 benchmark 反馈已经接入提示词驱动链路，但仍然有一些系统判断需要人工把关，比如性能短板来自 scheduler、cache 还是 benchmark 契约偏差。

## 3. Phase 1: GPU Profiling Agent

### 3.1 我构建了什么

Phase 1 的主线是一个自动化 profiling agent。它的核心目标不是单纯跑一次 profiling，而是：

- 根据 target 自动判断该用哪类工具
- 如果真实 workload 不够稳定，生成 synthetic intrinsic probe 或 synthetic counter probe
- 对结果做 plausibility 检查
- 最终汇总成可解释的 analysis / evidence

仓库中这一阶段的核心实现主要在：

- `profiler_agent/orchestrator/pipeline.py`
- `profiler_agent/analyzer/service.py`
- `profiler_agent/report_summary.py`
- `profiler_agent/multi_agent/*`

### 3.2 Agent 试图识别什么

Phase 1 主要围绕 GPU 硬件/性能特征进行探测，例如：

- DRAM latency / bandwidth 相关 target
- intrinsic probe target
- synthetic counter target
- 是否需要 ncu/nsys 或 microbenchmark

我不是直接把每个 target 都映射成固定命令，而是让 agent 根据：

- target 语义
- command availability
- workload 是否存在
- profiling 工具是否可用

来决定是：

- 跑原始命令
- 跑 synthetic benchmark
- 还是降级到更保守的探测路径

### 3.3 使用了哪些测量与工具

这一阶段主要用到的方式有：

- `ncu`
- `nsys`
- synthetic microbenchmark
- proxy counters
- analyzer 里的 plausibility checks

其中一个重要设计是：
**不是把测量结果原样相信，而是通过 analyzer 再做一层“这个值是不是语义上合理”的判断。**

### 3.4 Agent 如何决定运行哪个 profiling command

在 Phase 1 里，我逐渐把逻辑收成几层：

1. 先判断目标属于哪类 target
2. 再探测本地工具是否可用
3. 如果 workload 路径不完整或工具不可用，则转 synthetic probe
4. 最后由 analyzer 标注结果的可信度和限制

这一步的经验是：
**自动化 profiling 不是“会调命令”就够了，而是要学会在工具缺失、数据缺失、命令失败时给出退化路径。**

### 3.5 我如何验证结果是否 plausible

我在 Phase 1 学到的一个核心点是：

- profiling 的第一份结果常常是错的，或者至少是误导性的。

因此我引入了几种验证方式：

- 结果是否来自真实 workload 还是 synthetic probe
- 工具是否真实执行成功
- target 语义和测量方式是否匹配
- 多个证据之间是否互相矛盾

### 3.6 一次错误测量的例子

一个很典型的问题是：

- 某些 target 在工具不可用时退化成 synthetic counter probe
- 如果只看数字本身，会误以为它是 workload 的真实性能

我一开始的错误假设是：

- “只要有一个数值输出，这个 target 就算测到了”

后来的证据表明这是不对的，因为：

- synthetic counter 只是 proxy，不是 workload 的原生性能事实
- analyzer 中必须把它明确标注为非 workload observation

最后的修正方式是：

- 在 evidence / analysis 中加入 `synthetic_counter_probe_report`
- 明确说明它是 proxy measurement
- 在报告里区分“接受了这个探测”与“它是否等价于真实 workload 结果”

### 3.7 一个小型证据表

| 类型            | 例子                                              | 说明                             |
| --------------- | ------------------------------------------------- | -------------------------------- |
| Profiling tool  | `ncu` / `nsys`                                | 工具可用时优先走真实 profiling   |
| Synthetic probe | intrinsic / counter probe                         | 工具缺失或 workload 不稳定时兜底 |
| Analyzer output | intrinsic probe report / synthetic counter report | 用于说明结果可信度与局限性       |

### 3.8 Phase 1 收获

Phase 1 让我第一次比较系统地意识到：

- profiling 不是“跑命令”，而是“管理不确定性”
- 一个自动化 agent 必须知道何时信工具、何时怀疑工具、何时明确说明退化路径

## 4. Phase 2: LoRA Operator Optimization Agent

### 4.1 我构建了什么

Phase 2 的目标是自动优化 LoRA operator，对应的核心公式是：

\[
Y = W X + A(B^T X)
\]

这一阶段我构建了完整的优化 workflow，包括：

- candidate generation
- compile / load
- correctness check
- benchmark
- state / report persistence
- LLM-guided revision

相关实现主要在：

- `profiler_agent/phase2/generator.py`
- `profiler_agent/phase2/evaluator.py`
- `profiler_agent/phase2/optimizer.py`
- `profiler_agent/phase2/workflow.py`

### 4.2 Agent 生成了哪些候选实现

Phase 2 里我尝试过多类候选：

- 纯手写 CUDA 路线
- raw cuBLAS 路线
- 混合路线：部分大 GEMM 交给库，部分低秩路径自定义
- 后期转向更安全的 ATen / PyTorch extension 路线

这些候选的共同点是：

- 都会自动生成 `optimized_lora.cu`
- 都会进入统一 evaluator
- correctness 不过就不给 benchmark credit

### 4.3 Agent 如何编译、测试、benchmark、比较候选

这一阶段的闭环是：

1. 生成候选源码
2. 编译 / 加载候选
3. 跑 correctness
4. correctness 通过后才跑 benchmark
5. 把结果写到：
   - `.agent_artifacts/phase2_state.json`
   - `.agent_artifacts/phase2_report.json`

这一步让我意识到，优化系统最重要的不只是找到一个快的版本，而是：

- **能不能稳定记录每一次失败和成功**

因为很多关键洞察其实都来自失败样本。

### 4.4 遇到的最重要性能瓶颈

Phase 2 最关键的瓶颈其实不是单纯“算得慢”，而是：

- **数值路径与 evaluator 契约都很脆弱**

最开始我以为：

- 只要把大矩阵乘交给 cuBLAS，correctness 和 speedup 都会自然变好

后来发现并不完全如此，因为：

1. 部分 raw cuBLAS 路线在本地 evaluator 可用，但在最终 PyTorch extension 加载契约下并不安全
2. 即使数学上等价，数值路径不同也可能导致相对 reference 的误差不通过阈值

### 4.5 一个有用的优化

真正有用的方向是：

- 从 raw cuBLAS 直调，逐渐转向更契合最终契约的 **ATen / PyTorch extension backed 路线**

这一步的好处有两个：

1. 更接近最终 evaluator 的真实加载方式
2. correctness 行为更接近 PyTorch reference

### 4.6 一个看起来很有希望但失败的优化

失败得最明显的一类是假设：

- “只要继续扩大 cuBLAS family 搜索，就能持续提高速度”

实际上遇到了两个问题：

1. 隐藏评测使用的是 `torch.utils.cpp_extension.load(...)`
2. 某些 raw cuBLAS 符号在该加载方式下会出现 `undefined symbol`

也就是说：

- 本地某些“看起来很强”的候选，并不是真正可提交的候选

### 4.7 一个 correctness bug

最有代表性的 bug 之一是：

- 某个候选在本地 report 里显示 correctness 已通过
- 但最终 `optimized_lora.cu` 并不能按真实 extension 契约成功加载

最初错误假设是：

- “phase2_report 里 pass 了，就等于提交契约也 pass 了”

后来证据表明：

- 本地 `nvcc + ctypes` 路线和最终 PyTorch extension toolchain 不一致
- 因此 correctness pass 并不自动等于提交 pass

修正方式是：

- 把 evaluator 尽量向 `torch cpp_extension` 的加载链靠拢
- 并将 raw cuBLAS family 从主线降级

### 4.8 一个错误的性能假设

另一个错误假设是：

- “只要更多轮次搜索，就一定能继续提高 speedup”

但实际 report 显示：

- 很多 family 在若干轮后已经进入明显 plateau
- 后续继续搜索只是重复试错，甚至重新引入 correctness 风险

因此我后来加了：

- best-correct anchor
- speedup-only prompt
- plateau / stop-buffer 风格的收缩逻辑

### 4.9 候选版本示例表

| Version | Main idea             | Correct?  | Runtime / speedup | What I learned                     |
| ------- | --------------------- | --------- | ----------------- | ---------------------------------- |
| v1      | 朴素 baseline         | 否/不稳定 | 很低              | 先过 correctness 比盲目提速更重要  |
| v2      | raw cuBLAS 主线       | 局部可过  | 本地看起来快      | 本地契约与最终契约可能不一致       |
| v3      | ATen / extension 路线 | 是        | 更稳              | evaluator 契约对齐比单点优化更关键 |

### 4.10 Phase 2 收获

Phase 2 让我真正理解到：

- kernel optimization 不是只看 FLOPs
- correctness、toolchain、ABI、加载方式、数值语义都会决定一个优化是否“真实有效”

## 5. Phase 3: Inference Runtime Agent

### 5.1 我构建了什么

Phase 3 的目标是实现一个**提示词驱动的 inference runtime optimization agent**。这一阶段不再是优化单个 operator，而是要生成并迭代一个完整 runtime：

- 根目录 `engine.py`
- 根目录 `output3.json`

核心接口是：

- `create_engine(model_config, weight_dir, device="cuda")`
- `prefill(...)`
- `decode(...)`
- `remove(...)`

相关实现主要在：

- `profiler_agent/phase3/*`
- `runtime/*`
- `run.sh`

### 5.2 Agent 如何生成 `engine.py`

Phase 3 最重要的变化是：

- 候选生成主路径从模板切换为 **LLM prompt-driven revision**

具体过程是：

1. 组织 system prompt
2. 组织包含 benchmark / correctness / previous candidate 的 user prompt
3. 让 LLM 返回新的 `engine.py`
4. 本地校验该源码是否满足 runtime 契约
5. 进入 correctness 与 throughput 评测

也就是说，这一阶段的重点不是“用 LLM 写一段代码”，而是：

- **让 LLM 基于上一轮数据持续修订 runtime**

### 5.3 Runtime 如何加载模型配置与权重

我在 root `runtime/` 下整理了支持组件，包括：

- loader
- model
- request state
- KV cache
- scheduler

`engine.py` 自己保持为单文件入口，但会从本地 `runtime/` 包读取：

- model config
- weight dir
- request/cache 支撑逻辑

这样做的原因是：

- 最终交付需要单入口 `engine.py`
- 但 runtime 内部仍然需要结构化模块，便于调试与后续优化

### 5.4 `prefill(...)`、`decode(...)`、`remove(...)` 如何实现

当前实现思路是：

- `prefill(...)`：对 prompt 做一次前向，建立 request state 与 KV cache
- `decode(...)`：基于 request id 找到已有 state，只对增量 token 做 decode
- `remove(...)`：删除 request 对应的 state 与 cache

我没有把这三者写成完全独立的黑盒，而是尽量用统一的 request state table 管理：

- request id
- token history
- kv cache
- seq len

### 5.5 是否实现了 KV cache

实现了，而且这是 Phase 3 能跑起来的关键。

如果没有 KV cache：

- 每次 decode 都要重算整段序列
- decode-heavy 场景吞吐会非常差

这也是我在 Phase 3 中最明确的一个系统结论：

- **runtime 是否真的在做增量 decode，是 inference 系统性能的根分界线**

### 5.6 我主要优化了什么

Phase 3 的优化重点不是单纯追求一个总吞吐数字，而是围绕三类 workload：

- prefill
- decode
- mixed serving

尤其后期我把 prompt 收紧成：

- **每轮只盯住最弱项做单目标优化**

例如：

- 如果 `decode_speedup` 最弱
- 那这一轮 prompt 就明确要求只围绕 decode path 做局部修订

### 5.7 一个 correctness test 结果

本地 evaluator 中，我对 runtime 做了至少三类 correctness 检查：

- prefill correctness
- decode correctness
- remove 后的 request state correctness

这一点很重要，因为 Phase 3 的 bug 往往不是“函数能不能跑”，而是：

- request state 是否在多请求下仍然正确

### 5.8 一个 throughput benchmark 结果

Phase 3 的 benchmark 会分别记录：

- `prefill_tokens_per_s`
- `decode_tokens_per_s`
- `mixed_tokens_per_s`
- `aggregate_tokens_per_s`

我后期的 prompt 也直接使用这组 breakdown，而不是只看一个总 speedup。

### 5.9 一个 cache / request-state 相关 bug

一个典型问题是：

- 某些 runtime 候选在 prefill correctness 正常
- 但 decode 后 request state 更新不一致
- 或 remove 之后再次 decode 语义不对

最初错误假设是：

- “只要 prefill 和 decode 单独都能跑，就说明 request-state 系统没问题”

后来证据表明：

- 真正的问题出在多请求状态迁移和 cache 生命周期

修复方式是：

- 用更明确的 request state table 统一管理
- 把 remove 路径也纳入 correctness evaluator

### 5.10 一个用于处理 hidden cases 的设计决策

我做的一个关键设计是：

- 让 prompt 不只看到 raw benchmark，而是看到 **baseline 对比 + 最弱项 + focus hints**

这样 agent 在面对 hidden cases 时，不必盲目地“整体提速”，而是更可能：

- 找到拖后腿的阶段
- 做局部修补

## 6. Cross-Phase Reflection

这是我觉得整个项目里最有价值的部分。

### 6.1 我对 MLSys 的理解如何变化

一开始我把 MLSys 理解成：

- 跑 profiling
- 写 CUDA
- 追求速度

但做完三个阶段后，我更倾向于把它理解成：

- **在不完整信息、复杂约束和真实系统边界下做正确的工程决策**

### 6.2 Phase 1 教会了我什么

Phase 1 教会我的主要不是“怎么调工具”，而是：

- profiling 结果必须有上下文
- synthetic probe 与真实 workload 结果不能混为一谈
- agent 要能处理缺失工具、缺失命令、缺失数据

### 6.3 Phase 2 教会了我什么

Phase 2 最让我震撼的是：

- 同样的数学公式，在不同 toolchain / evaluator / numeric path 下，完全可能是不同的问题

我学到的不是某个 kernel trick，而是：

- correctness、ABI、library linkage、evaluator 契约本身也是优化问题的一部分

### 6.4 Phase 3 教会了我什么

Phase 3 让我第一次真正把“runtime”当成系统而不是代码片段来看待。

真正重要的问题变成：

- request state 怎么管理
- KV cache 生命周期怎么设计
- prefill / decode / mixed 负载如何权衡
- 如何让 agent 不是盲改，而是依据 benchmark 反馈修订

### 6.5 我的 agent workflow 如何进化

三个阶段里，agent workflow 的进化大致是：

1. **Phase 1**：命令 / 工具驱动
2. **Phase 2**：candidate + evaluator 驱动
3. **Phase 3**：prompt + structured feedback 驱动

换句话说，我的自动化重心从：

- “怎么跑”

逐渐变成：

- “怎么根据结果改下一轮”

### 6.6 最难自动化的部分

最难自动化的是：

- **系统层面的判断**

例如：

- 某个结果差，到底是 benchmark 问题、cache 问题、调度问题、还是 evaluator 契约问题？

这类判断目前很难完全交给 LLM 自动完成。

### 6.7 LLM 最容易帮上忙的部分

LLM 最擅长的部分是：

- 带上下文的小范围代码修订
- 根据上一轮错误信息生成下一轮候选
- 帮助组织结构化实现与试错

### 6.8 仍然需要我自己做系统判断的部分

仍然需要人工系统判断的部分包括：

- 某条优化路线是否真的契合最终 evaluator
- 某个 correctness pass 是否等于可提交 pass
- benchmark breakdown 背后真正的系统瓶颈是什么

## 7. Mistakes, Pitfalls, and Debugging Stories

### 7.1 问题汇总表

| Problem                                         | Symptom                                 | Wrong hypothesis               | Actual cause                                    | Fix                                                   | Lesson                                     |
| ----------------------------------------------- | --------------------------------------- | ------------------------------ | ----------------------------------------------- | ----------------------------------------------------- | ------------------------------------------ |
| Phase 1 synthetic result 被误当成 workload 结果 | 数值看起来合理，但语义不对              | “有输出就算测到了”           | synthetic probe 只是 proxy                      | 在 analyzer/report 中显式标注 synthetic probe 语义    | profiling 结果必须带来源语义               |
| Phase 2 本地 pass 但最终契约不安全              | 本地 correctness 通过，但隐藏评测不通过 | “本地 evaluator pass 就够了” | 本地加载契约与最终 PyTorch extension 契约不一致 | 调整 evaluator，弃用部分 raw cuBLAS 主线              | toolchain 本身是问题的一部分               |
| Phase 2 盲目增加轮次无收益                      | 迭代数增加但 speedup 不再提高           | “多跑就会更好”               | 搜索进入 plateau，后续只是重复试错              | 加强 best-correct anchor 和 stop buffer               | 搜索空间需要收缩，不是越大越好             |
| Phase 3 request state bug                       | prefill 正常但 decode/remove 场景不稳定 | “函数单测通过就够了”         | 多请求状态迁移与 cache 生命周期处理不完整       | 把 remove 和 request-state 纳入 correctness evaluator | runtime bug 往往出在状态迁移，不在单点函数 |

### 7.2 Debug 故事 1：Phase 2 的 evaluator 契约偏差

症状：

- 本地 `phase2_report` 显示某条候选 correctness 已通过
- 但提交环境下却无法正确加载

最初假设：

- 这是某个候选本身的偶发 compile 问题

后来证据：

- 报错集中在 extension load / symbol resolution
- 同一条数学路线在不同构建链下行为不同

真实原因：

- 本地 evaluator 与最终 PyTorch extension toolchain 不一致

修复：

- 调整本地 evaluator 方向
- 把更契合最终契约的 ATen / extension 路线提升为主线

教训：

- 不能只优化代码，不优化 evaluator 契约理解

### 7.3 Debug 故事 2：Phase 3 的“最弱项”识别

症状：

- 总吞吐看起来还可以
- 但某些 case 下 decode 吞吐明显差

最初假设：

- 既然 aggregate 不低，说明 runtime 整体问题不大

后来证据：

- breakdown 里 `decode_tokens_per_second` 很差
- prefill 吞吐掩盖了 decode 路径的短板

真实原因：

- 不能只看 aggregate，需要看 stage-specific throughput

修复：

- 在 feedback 中加入 baseline 对比
- 自动计算 weakest metric
- prompt 改成单目标优化

教训：

- inference runtime 的优化不能只看一个总分

## 8. What I Would Do Differently

如果从 Phase 1 重新开始，我会做这些调整：

### 8.1 更早统一 evaluator 契约

我会更早把：

- 本地评测方式
- 目标 evaluator 契约
- 提交产物契约

统一起来。
这在 Phase 2 尤其重要，因为很多时间其实花在“本地能跑但不等于最终能交”上。

### 8.2 更早写状态与反馈日志

我会更早把这些作为一等公民：

- state.json
- report.json
- benchmark breakdown
- candidate lineage
- prompt debug logs

因为后期真正能帮我定位问题的，往往不是最终代码，而是这些中间记录。

### 8.3 更早做单目标优化而不是宽搜

一开始我有些阶段还是偏宽搜，尤其在性能优化时容易同时试很多方向。如果重来，我会更早采用：

- one-bottleneck-per-iteration

这种策略。

### 8.4 我会避免做的事

我会避免：

- 过早迷信某条“看起来很快”的库路线
- 只看 aggregate speedup
- 在 correctness 不稳时继续扩大搜索空间

### 8.5 我会投入更多时间的部分

如果有更多时间，我会投入到：

- Phase 3 的 request scheduler
- decode path profiling
- 更真实的 serving trace benchmark
- prompt-to-diff 约束

## 9. Conclusion

这个课程项目给我的最大技术收获是：

- **性能优化从来不是一个孤立的代码问题，而是 profiling、correctness、toolchain、runtime state 和 evaluator 契约共同决定的系统问题。**

最大的工程收获是：

- 只要把生成、执行、校验、记录、反馈做成闭环，LLM agent 就能在很多复杂问题上提供真实价值；但它最有效的前提是，人先把系统边界和反馈结构组织好。

做完三个阶段后，我仍然保留的一个开放问题是：

- 如果要进一步提高自动化程度，怎样才能让 agent 自己区分“代码问题”和“评测契约问题”？

我觉得这也是 MLSys 里一个很真实的问题：
系统优化并不只是“更快”，而是“在复杂约束下持续做出正确判断”。

## Appendix

### A. 可引用的仓库材料

- Phase 1 主线：
  - `profiler_agent/orchestrator/pipeline.py`
  - `profiler_agent/analyzer/service.py`
  - `profiler_agent/report_summary.py`
- Phase 2 主线：
  - `profiler_agent/phase2/*`
  - `.agent_artifacts/phase2_state.json`
  - `.agent_artifacts/phase2_report.json`
- Phase 3 主线：
  - `profiler_agent/phase3/*`
  - `runtime/*`
  - `run.sh`
  - root `engine.py`
  - root `output3.json`
