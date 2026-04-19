# Report

## 项目 Agent 简介

本项目是一个面向 GPU profiling 的多代理系统，目标是在有无真实 workload 的不同条件下，尽可能稳定地产出 `results.json`、`evidence.json` 和 `analysis.json`。

## 主要 Agent

- `router_agent`
  负责识别请求意图，把任务路由到合适的 profiling 流程。

- `planner_agent`
  负责制定执行计划，选择需要使用的工具，并决定本轮关注哪些 target。

- `executor_agent`
  负责真正执行工具链，包括 workload 运行、device attribute 查询、microbenchmark probe、`ncu` profiling 等。

- `interpreter_agent`
  负责读取执行结果，生成摘要、分析后续动作，并决定是否需要进入下一轮 refinement。

- `coordinator`
  负责串联整个多代理流程，维护轮次状态、持久化 agent state，并控制何时继续迭代或终止。

## 当前能力特点

- 对 `device_attribute` 类指标，支持直接查询。
- 对 `intrinsic_probe` 类指标，支持自动生成 CUDA probe 并迭代修正。
- 对 `workload_counter` 类指标，在没有真实 `run` 时支持 `synthetic_counter_probe` 路线，并明确标注其为 proxy signal，而非真实 workload observation。

## 总结

这个项目中的 agent 不是简单对话机器人，而是一套分工明确的执行代理系统：前端负责规划和解释，后端负责测量和验证，最终共同生成可供 evaluator 使用的结构化 profiling 结果。
