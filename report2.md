# Phase 2 简要报告

## 一、Phase 2 做了什么

Phase 2 的目标是：

1. 生成 LoRA CUDA 候选实现
2. 自动做 correctness 检查
3. correctness 通过后继续做 speedup 优化
4. 在时间预算内持续迭代，并保存中间状态与最终最优结果

## 二、这阶段最重要的改动

### 1. 评测契约改对了

一开始本地评测使用的是：

- `nvcc -shared`
- `ctypes` 加载 `.so`

这条链路过于宽松，会错误接受一些最终提交环境其实不能稳定工作的候选，尤其是直接调用 raw cuBLAS API 的实现。

后来改成了更接近真实提交环境的方式：

- `torch.utils.cpp_extension.load(...)`
- 直接按 PyTorch extension 的方式编译和加载 `optimized_lora.cu`

这一步是整个 Phase 2 最关键的修正。

### 2. 候选实现从 raw cuBLAS 路线切到 ATen 中路线

之前尝试过很多直接调用：

- `cublasGemmEx`
- `cublasSetMathMode`

的候选。

这些候选本地可能正确，但在最终评测里容易因为链接方式不同而失败。

现在主线改成了：

- `torch::matmul`
- `torch::addmm`
- `PYBIND11_MODULE(...)`
- `forward(W, X, A, B)`

也就是让 `optimized_lora.cu` 本身就是一个可被 PyTorch extension 直接加载的单文件实现。

### 3. 增加了更清晰的运行状态记录

现在 report 里新增了：

- `last_completed_iteration`

这样可以区分：

- `iterations_run`：已经开始到第几轮
- `last_completed_iteration`：真正完整跑完并记录到 state 的最后一轮

这解决了之前“看起来像只跑了 8 轮，但其实第 8 轮只是中途被截断”的歧义。

## 三、当前搜索策略

现在 Phase 2 的搜索策略分成两段：

### 1. correctness-first

在还没有正确候选之前，优先生成更稳的 ATen 候选，先把：

- `torch.allclose(..., rtol=1e-4, atol=1e-4)`

过掉。

### 2. speedup-after-correctness

一旦出现 `current_best_correct_candidate_id`，后续就不再乱开新 family，而是围绕当前 best correct candidate 做局部优化。

目前已经验证出更稳的主线是：

- `aten_*_addmm_bt_view-*`

其中较强的一支是：

- `aten_inplace_addmm_bt_view-*`

## 四、已经做过的性能收缩

为了减少浪费预算的尝试，已经做了这些收缩：

1. `bt_view` 一旦通过 correctness，就不再优先回去试 `bt_contiguous`
2. 降低 `functional addmm` 的优先级
3. speedup 阶段优先围绕 `best correct family`
4. 精简模板里的额外拷贝和无条件 `contiguous()`
5. 去掉一部分热路径上不必要的检查

## 五、为什么现在轮数比以前少

不是因为 API 坏了，而是因为：

1. 现在每轮评测更真实
2. 很多候选会真的走完：
   - `cpp_extension.load(...)`
   - correctness 检查
   - benchmark

以前很多候选更早失败，所以单轮很便宜，看起来能跑很多轮。  
现在轮数变少，反而说明每轮评测更接近真实提交环境。

## 六、当前成果

目前 Phase 2 已经做到：

1. 本地评测方式更接近最终提交契约
2. 可以找到 correctness 通过的 ATen 候选
3. 可以在 correctness 通过后继续做 speedup 搜索
4. report 与 state 更容易解释，不再只靠猜

## 七、当前仍然存在的限制

1. 本地 `best_speedup` 和外部最终 case 的 speedup 数值不一定完全一致
2. 外层 `timeout` 仍可能导致 report 留下中间态快照
3. 现在已经进入“小优化”为主的阶段，收益会比前期慢很多

## 八、结论

Phase 2 目前已经从“评测契约不对、容易误判正确候选”的阶段，进入了“围绕正确主线做真实性能微调”的阶段。

最重要的收获不是多写了几个候选，而是：

1. 把评测链路改对了
2. 把候选契约改对了
3. 把搜索主线收敛到了更接近最终提交环境的 ATen 中路线

现在这套 Phase 2 已经比最早那版稳定得多，也更适合继续做后续的小幅性能优化。
