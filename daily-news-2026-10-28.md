# Codex 技术情报每日动态（2026-10-28）

调研窗口：北京时间 (2026-10-27 05:51:58, 2026-10-28 06:00:19]。

覆盖方向：PyTorch、LLVM/MLIR、Triton & TileLang、RISC-V，以及 AI 模型、研究、芯片和推理基础设施。

信息口径：以项目官方公告、技术博客、正式发行、原始 RFC/讨论和仓库记录为依据；区分已发布、已合并和提案状态，性能数字保留具体测试条件。

## 今日要闻

- [Genie 4 公布研究预览，将原生音频和交互对话纳入实时世界模拟，现向可信测试者开放。](https://deepmind.google/blog/genie-4-a-new-generation-of-world-simulation/)
- [Triton 3.9.0 正式发布，新增原子操作能力，同时移除 block pointer 路径、Python 3.10 与 AMD gfx10 支持。](https://github.com/triton-lang/triton/releases/tag/v3.9.0)
- [CUDA 13.4 的显存边界经 GPU Fractioning Operator 与 KAI Scheduler 连接到 Kubernetes GPU 份额调度。](https://developer.nvidia.com/blog/memory-enforced-gpu-sharing-in-kubernetes-with-nvidia-cuda-13-4/)
- [Liquid AI 发布 LFM3 六模型家族，以共享约 2B 级骨干覆盖文本、视觉、音频与决策任务。](https://huggingface.co/blog/LiquidAI/lfm3)
- [RISC-V KVM 新 RFC 提议在安全的 vCPU 切换间保留 guest Vector 数据寄存器，减少重复恢复；默认关闭，硬件性能待测。](https://lists.infradead.org/pipermail/linux-riscv/2026-October/102208.html)
- [ClangIR 默认构建讨论新增 ABI classifier 接入说明：可先独立验证 LLVM ABI 分类，再接入 CIR calling-convention lowering。](https://discourse.llvm.org/t/rfc-enable-clangir-build-by-default/91730/116)

## 今日索引

- **PyTorch**：delegate 复用、SmolVLA 部署、Vulkan 共享 KV cache、变长上下文并行、DistMuon 与 Arm 稀疏导出。
- **LLVM/MLIR**：ClangIR ABI、sanitizer/对齐/外部测试 RFC、IREE RVV 成本模型、Buddy VIR Python 绑定与 torch-mlir PT2E 迁移。
- **Triton & TileLang**：Triton 3.9.0、Volta/Turing 社区补丁，以及 Ascend 条件任务、线程同步与动态流水线修正。
- **RISC-V**：KVM Vector/A210 新 RFC、RVA23 CRD 1.0、CVA6 配置、点积与 Hypervisor 参数、香山和视觉部署链。
- **AI 业界**：Genie 4、LFM3、GPU 显存共享、DGX Spark MoE、研究新颖性评估和 OCI Vera/Rubin。

## 一、PyTorch 生态核心动态

### 1.1 模型 & 技术｜[让后端在对同一下沉模块的多次调用之间共享一个 delegate](https://github.com/pytorch/executorch/pull/24127)

北京时间：2026-10-27 23:59:15｜来源类型：GitHub PR｜事件状态：已合并

**事实**：后端可通过 share_handle_across_calls 显式选择句柄共享：同一方法反复调用同一下沉模块时只初始化一次 delegate。Jetson 上一个调用十次 TensorRT 步骤的案例，加载时间由 435 ms 降至 74 ms，内存由 1,060 MiB 降至 284 MiB，输出与运行速度相同。

**重要性**：直接减少重复模型分区调用的初始化与驻留成本，对循环式推理部署有价值。

**风险与限制**：默认关闭；带 scratch 内存的 delegate 不共享，不同分区也不共享。serde schema 升至 5.6，PTE 文件格式未变；案例数据不能代表所有后端。

### 1.2 模型 & 技术｜[fix(conversion)：处理 slice_scatter 边界情况，并将连续的 slice_scatter 写成拼接](https://github.com/pytorch/TensorRT/pull/4898)

北京时间：2026-10-28 05:14:18｜来源类型：GitHub PR｜事件状态：已合并

**事实**：修正不同 dtype 写入、负 select_scatter 索引和 KV cache 原地更新边界；静态长度且步长为 1 的局部写入改用 cat，其他情况保留 scatter。 SmolVLA 经 ExecuTorch 部署的测量由 H100 的 14.9 ms 降至 13.5 ms、Jetson AGX Thor 的 28.7 ms 降至 27.0 ms、Orin Nano 的 130.5 ms 降至 125.3 ms。

**重要性**：同时处理模型转换正确性和端到端部署开销，覆盖视觉语言动作模型的实际路径。

**风险与限制**：作者还报告某些孤立 rotary 写入链微基准变慢；数据取决于模型图、硬件与 TensorRT/PyTorch 版本，不能把个别收益外推。

### 1.3 模型 & 技术｜[\[ET-VK\]\[SDPA\] 保留共享 KV cache 语义](https://github.com/pytorch/executorch/pull/23412)

北京时间：2026-10-27 07:52:15｜来源类型：GitHub PR｜事件状态：已合并

**事实**：Vulkan SDPA 传递显式 attention scale；只有当缓存更新属于唯一 donor attention、顺序和用户集合满足约束时才融合。共享或有歧义的 KV cache 保留显式更新，避免 YOCO consumer 读到陈旧缓存。

**重要性**：为 Gemma 4 的归一化 attention 与共享缓存保留语义，是移动 GPU 模型部署的实质正确性变化。

**风险与限制**：这项合并解决特定 scale 与缓存所有权问题；不能据此声称整套 Gemma 4 已完成性能或全面部署验证。

### 1.4 模型 & 技术｜[\[CP\] 添加对计算图安全的变长 K/V all-gather](https://github.com/pytorch/torchtitan/pull/5071)

北京时间：2026-10-27 06:02:00｜来源类型：GitHub PR｜事件状态：已合并

**事实**：为 causal self-attention 增加 sharded Q、all-gather K/V 的变长后端，使用固定 query chunks 支持 CUDA Graph replay；支持连续分片和 head-tail 分片，按文档与滑动窗口限制 K/V 切片。

**重要性**：把变长上下文并行与静态图重放衔接起来，可减少动态打包对图捕获的限制。

**风险与限制**：不支持 PTRR、cross-attention 和非 causal 窗口；固定 metadata 容量仍是重放边界，未提供端到端吞吐提升。

### 1.5 模型 & 技术｜[\[DistMuon\] Newton-Schulz 可在所选设备上暂存执行](https://github.com/pytorch/torchtitan/pull/5352)

北京时间：2026-10-27 06:39:55｜来源类型：GitHub PR｜事件状态：已合并

**事实**：新增 DistMuon(ns_device=None | "cuda" | "cpu")，可只把 Newton-Schulz 计算视图移到所选设备，动量、参数与重分布 buffer 留在原处。作者的 CPU offload 实验中 optimizer step 从约 15 秒降至约 1 秒。

**重要性**：让计算密集的优化器子步骤与参数存储位置解耦，为显存受限训练提供具体路径。

**风险与限制**：默认行为不变；约 15 倍来自作者单项实验，未给出可普遍外推的模型与硬件吞吐对照。

### 1.6 模型 & 技术｜[Arm 后端：为 QAT 示例添加 2:4 稀疏支持](https://github.com/pytorch/executorch/pull/24180)

北京时间：2026-10-27 18:49:23｜来源类型：GitHub PR｜事件状态：已合并

**事实**：QAT 示例新增 --prune-2-4，按模块名选择 Conv2d，每个输出通道和卷积核位置独立保留四个输入通道权重中幅值最大的两个；固定 mask 在优化器更新后恢复，直接导出剪枝权重。

**重要性**：将结构化稀疏准备接入 eager、PTQ 和 QAT 流程，避免在部署图中新增运行时 mask。

**风险与限制**：这是示例与权重准备能力；没有给出稀疏硬件加速或精度改善保证，所选卷积的每组输入通道必须满足四元素分组要求。

### 1.7 模型 & 技术｜[向 `CpuBackend` 添加 torchvision 导出工作流（#23934）](https://github.com/pytorch/executorch/pull/23934)

北京时间：2026-10-28 03:48:52｜来源类型：GitHub PR｜事件状态：已合并

**事实**：export_torchvision 通过已注册的 CPU 与 XNNPACK FP32 recipes 导出分类模型；除 model.pte 外保存 PT2 捕获、确定性输入、eager 与 edge logits，以及记录哈希和预处理来源的 export.json。

**重要性**：把模型导入、产物溯源和部署前数值比对放进同一工作流，与 PyTorch→编译后端的模型落地链路直接相关。

**风险与限制**：当前描述聚焦 torchvision 分类模型和 FP32 recipes；产物可用于验证，不代表所有模型或量化路径已验证。

## 二、LLVM/MLIR 最新进展

### 2.1 模型 & 技术｜[RFC：默认启用 ClangIR 构建-](https://discourse.llvm.org/t/rfc-enable-clangir-build-by-default/91730/116)

北京时间：2026-10-27 12:17:09｜来源类型：项目论坛／原始讨论｜事件状态：讨论新进展

**事实**：原提案是默认构建 ClangIR 并纳入 CIR 测试，带来 MLIR 构建依赖；默认代码生成仍沿用原路径，使用 CIR 仍需 -fclangir。 讨论新进展：andykaylor 说明新目标可先在 LLVM ABI 库实现分类器，并用实验性 ABI lowering 与传统 Clang 路径自动对照，再接到 CIR。

**重要性**：把新后端的 ABI 支持拆成可独立验证的阶段，对 MLIR 编译链和 RISC-V 目标接入有直接参考价值。

**风险与限制**：仍为默认构建方案讨论；另一条新回复报告 OpenMP task/barrier/critical 等支持不完整，不能理解为 CIR 已全面替代 Clang。

相关原文：[论坛 回复 #117](https://discourse.llvm.org/t/rfc-enable-clangir-build-by-default/91730/117)。

### 2.2 模型 & 技术｜[\[RFC\] Sanitizer 插桩区域：在 noinstr 函数内部重新启用插桩](https://discourse.llvm.org/t/rfc-sanitizer-instrumentation-regions-re-enabling-instrumentation-inside-noinstr-functions/92623/1)

北京时间：2026-10-27 22:31:27｜来源类型：项目论坛／原始讨论｜事件状态：RFC 提出

**事实**：提议增加 llvm.sanitizer.instrumentation.begin/end 与 Clang builtin，在整体禁止插桩的函数内显式恢复局部 sanitizer 插桩，让编译器识别 Linux kernel 已有的 instrumentation_begin/end 区域。 目标是修正 noinstr caller 调用插桩 callee 时 KMSAN 参数 shadow 未写入造成的误报，并扩展 KASAN、KCOV、KCSAN 可见范围。

**重要性**：把内核入口代码的插桩边界变成编译器可见语义，涉及 sanitizer ABI 与优化移动规则。

**风险与限制**：尚未形成一致设计；维护者指出普通 memory-effects 屏障不能约束非逃逸内存或 restrict 访问，区域语义仍有争议。

相关原文：[论坛 回复 #2](https://discourse.llvm.org/t/rfc-sanitizer-instrumentation-regions-re-enabling-instrumentation-inside-noinstr-functions/92623/2)。

### 2.3 模型 & 技术｜[\[RFC\] 在 flang 中设置变量对齐的指令](https://discourse.llvm.org/t/rfc-a-directive-to-set-the-alignment-of-variables-in-flang/92619/1)

北京时间：2026-10-27 09:55:13｜来源类型：项目论坛／原始讨论｜事件状态：RFC 提出

**事实**：提议支持 !DIR$ ALIGN(n) :: name-list 和兼容 Intel 的 ATTRIBUTES ALIGN 拼写，以 max(请求值、自然 ABI 对齐、现有默认值) 设置存储的最小字节对齐。首阶段面向局部和模块变量，ALLOCATABLE/POINTER 对象后续处理。

**重要性**：补足普通局部/自动数组不能直接请求过对齐的缺口，关联 HPC 数学库与向量化调用约束。

**风险与限制**：这是提案而非已实现特性；临时对象、数据共享副本与平台分配器有明确限制，ASSUME_ALIGNED 仍不能替代真实存储对齐。

### 2.4 模型 & 技术｜[\[RFC\] 面向 LLVM PR 的厂商自有预合并测试](https://discourse.llvm.org/t/rfc-vendor-owned-pre-merge-testing-for-llvm-prs/92624/1)

北京时间：2026-10-28 01:33:50｜来源类型：项目论坛／原始讨论｜事件状态：RFC 提出

**事实**：提议由贡献者显式触发、厂商维护的外部设施执行预合并测试，并将结果作为不阻断合并的状态返回原 LLVM PR；首个试点是 AMD ROCm，设计面向其他厂商开放。 方案通过 repository_dispatch 与分离的 GitHub App 权限连接两侧；只允许有写权限的提交者触发。

**重要性**：新增硬件/配置的测试接入机制，可减少把 PR 手动移植到私有 fork 的成本，属于基础设施设计提案。

**风险与限制**：触发方式、治理及失效检查的移除机制尚待讨论；维护者同时质疑为何不直接复用 runner/buildbot，以及 opt-in 如何保持构建健康。

### 2.5 深度洞见｜[RFC：流敏感的可空性](https://discourse.llvm.org/t/rfc-flow-sensitive-nullability/89042/11)

北京时间：2026-10-28 02:51:29｜来源类型：项目论坛／原始讨论｜事件状态：讨论新进展

**事实**：该 RFC 希望通过数据流分析对未被空值检查收窄的 _Nullable 指针解引用等行为给出警告，默认不开启。 讨论新进展：作者放弃默认 _Nullable，改用 unspecified 与渐进注解；其最坏配置测试中 sqlite、redis、Clang Sema 的分析开销分别约为正常 -O2 构建 CPU 时间的 3%、4%、11%。

**重要性**：同时提供了可复查的诊断策略和编译成本数据；随后维护者提出应改在 CIR 上实现，而非继续扩张 CFG 诊断。

**风险与限制**：仍是 draft 实现与架构讨论；关联条件可能误报，模板注解仍会丢失，静态分析器对照的采样次数也不同。

相关原文：[论坛 回复 #12](https://discourse.llvm.org/t/rfc-flow-sensitive-nullability/89042/12)。

### 2.6 模型 & 技术｜[\[Codegen\]\[CPU\] 使用 mmt4d 成本模型选择 inner_tiled 分布 tile](https://github.com/iree-org/iree/pull/25130)

北京时间：2026-10-27 19:41:18｜来源类型：GitHub PR｜事件状态：方案提出，未合并

**事实**：提出将 mmt4d 的成本模型移植到 inner_tiled：在静态 M/N 迭代边界的约数中选择每个 workgroup 的 tile 数，以 LHS+RHS+ACC 字节预算筛选，并考虑遍历成本与线程利用率。 540×516×332 的 f32 RVV 案例由每组一个 tile 的 1,530 个 workgroup 改为配置 [18, 1, 0]、85 个 workgroup；动态边界或非 MMA 等情况回退原策略。

**重要性**：这是 IREE CPU/RVV 矩阵 lowering 的具体设计变化，与团队的 RISC-V 编译部署链路相关。

**风险与限制**：PR 尚未合并，workgroup 数变化不是吞吐基准；作者明确硬件性能测量待完成，不能写成已实现加速。

### 2.7 模型 & 技术｜[\[midend\]\[VIR\] Python 绑定](https://github.com/buddy-compiler/buddy-mlir/pull/1009)

北京时间：2026-10-27 22:42:09｜来源类型：GitHub PR｜事件状态：已合并

**事实**：Buddy 增加 buddy_mlir.dialects.vir，为 VIR ops、动态向量类型和 ScalingFactorAttr 提供 Python/C API/nanobind 入口；Python PassManager 可调用 lower-vir-to-vector 和 lower-linalg-to-vir。 同时修复 Python 临时字符串触发的 ScalingFactorAttr 存储生命周期问题；新增测试覆盖固定与 scalable vector lowering。

**重要性**：把 VIR 图构造与 lowering 接到 Python 模型编译工作流，直接服务 Buddy Compiler 的模型导入与 RVV 编译链。

**风险与限制**：本次不改变 lowering 算法或生成代码，未给出性能提升；现有类型文档与打印拼写的不一致仍保留。

### 2.8 模型 & 技术｜[\[TorchOnnxToTorch\] 将 QLinearAdd、QLinearMul、QLinearLeakyRelu、QLinearSigmoid、QLinearConcat、QLinearGlobalAveragePoo、QLinearAveragePool 迁移到 PT2E](https://github.com/llvm/torch-mlir/pull/4833)

北京时间：2026-10-27 21:55:37｜来源类型：GitHub PR｜事件状态：已合并

**事实**：将 com.microsoft 域这组 QLinear 算子的 helper 从旧 make_per_tensor_quantized_tensor 表示迁移到 PT2E quantized_decomposed.quantize_per_tensor/dequantize_per_tensor，并加入 ONNX 原生端到端测试。

**重要性**：这是 PT2E RFC Milestone 6 的一环，使 ONNX 量化导入沿着同一 QDQ 表示衔接后续 lowering。

**风险与限制**：本次只覆盖标题所列的算子集合；DynamicQuantizeLinear 的后续迁移仍是另一个未合并 PR，不能合并表述为全部 ONNX 量化已迁移。

### 2.9 模型 & 技术｜[\[TMTensor\] 在读取 `ins` 操作数的 op 处将其 bufferize](https://github.com/llvm/torch-mlir/pull/4820)

北京时间：2026-10-27 22:33:04｜来源类型：GitHub PR｜事件状态：已合并

**事实**：将 tensor→buffer 读取显式放在实际消费的 TMTensor op 附近，使 One-Shot Bufferize 能看到读写冲突，避免共享 tensor.empty 的两个写入被错误折叠到同一 buffer。 原有 include_self=false 的 scatter_reduce 可出现 sum 为预期两倍、amax 返回 -inf 等错误；参考后端的 copy-before-write 与融合流程曾掩盖该问题。

**重要性**：修正 PyTorch→TMTensor→memref 模型导入链的静默数值错误，比仅在测试中绕过冲突更关键。

**风险与限制**：问题依赖具体 bufferization 形状与 pass 顺序；不能据此推断所有参考后端结果都曾错误。

### 2.10 模型 & 技术｜[\[Build\]\[AMDGPU\] 使用官方 ROCM 发行版中的 bitcode](https://github.com/iree-org/iree/pull/25133)

北京时间：2026-10-28 01:42:44｜来源类型：GitHub PR｜事件状态：已合并

**事实**：旧 shark-infra/amdgpu-device-libs 产物停止供应后，IREE 改从官方 ROCm 发行版取得 LLVM bitcode，版本固定在 rocm_package.json。作者指出下载量增至约 146 Mb，而实际使用不足 1 Mb。

**重要性**：改变 AMDGPU 后端构建产物的维护来源和供应链，是部署可持续性上的具体变化。

**风险与限制**：主要收益是来源维护与构建可用性；下载体积明显增加，没有声称运行时性能提升。

## 三、Triton & TileLang 技术动态

### 3.1 重磅｜[Triton 3.9.0 发行说明](https://github.com/triton-lang/triton/releases/tag/v3.9.0)

北京时间：2026-10-28 02:56:21｜来源类型：正式 Release｜事件状态：正式发布

**事实**：Triton 3.9.0 正式发布，新增 atomic_load/store、atomic_poll 等前端能力；发行说明同时列出 Gluon、AMD/NVIDIA 后端与 profiling 的变化。

**重要性**：这是 kernel DSL 与后端的完整发行事件，兼容性变化会直接影响下游编译和部署。

**风险与限制**：block pointer 路径被移除，Python 最低升至 3.11，AMD gfx10 支持移除；部分类型提升、扩展加载和自定义 pass pipeline 的接口也有破坏性变化。

### 3.2 模型 & 技术｜[Volta（sm_70）和 Turing（sm_75）上的 tensor-core 点积：Triton 3.6 上的一个补丁系列](https://github.com/triton-lang/triton/discussions/12517)

北京时间：2026-10-27 21:27:57｜来源类型：项目论坛／原始讨论｜事件状态：社区补丁系列公开

**事实**：作者公开基于 Triton 3.6.0 的 23 个提交，在自有 fork 中恢复 sm_75 MMAv2 与 sm_70 MMAv1 tensor-core dot，并处理旧架构的寄存器流水和布局。 其 fp16、M=4096、K=N=4096 实验中，RTX 8000 为 69.4 TFLOPS、原版 FMA 路径为 7.5；V100 分别为 76.4 与 8.6 TFLOPS。

**重要性**：为仍使用 Volta/Turing 的推理服务提供可检查的后端实现和基准线索。

**风险与限制**：作者明确没有请求改变上游支持政策或提交上游提案；未测试 sm_80+、lit 和 C++ 单测。结果来自作者设备与分支，不是官方支持恢复。

### 3.3 模型 & 技术｜[\[Ascend\] 允许 PerCoreTask 候选中存在内部条件分支](https://github.com/tile-ai/tilelang/pull/3686)

北京时间：2026-10-27 15:24:45｜来源类型：GitHub PR｜事件状态：已合并

**事实**：取消 T.PerCoreTask 内 T.Task 对 if/else 的全面禁止，改为统一校验完整 task；同步边界包住完整候选，各分支仍检查 pipe/core 兼容、嵌套任务及不支持的跨核同步。

**重要性**：让条件 L1 padding 等内部工作可完成 Ascend 源码 lowering，扩展了调度表达能力。

**风险与限制**：作者完成编译与 113 项相关检查，但未进行设备运行；不等于所有内部控制流均获支持。

### 3.4 模型 & 技术｜[\[BugFix\] 在每个线程上运行 `T.ws(-1)` 区域](https://github.com/tile-ai/tilelang/pull/3354)

北京时间：2026-10-27 12:53:00｜来源类型：GitHub PR｜事件状态：已合并

**事实**：把 -1 解释为不绑定 threadIdx.x，避免此前生成空线程范围并静默丢弃区域内全部 store。示例中原来 256 个线程均未写入，修正后所有线程执行；实际 warp-group 编号仍限制线程组。

**重要性**：修正 warp-specialization 控制语义，避免内核输出静默错误。

**风险与限制**：仓库内尚无 ws(-1) 的使用点；这是特定 API 语义修正，不是通用性能提升，相关检查仍有既有环境限制。

### 3.5 模型 & 技术｜[\[BugFix\] 在 ThreadSync 中将 address_of 指针访问视为写入](https://github.com/tile-ai/tilelang/pull/3346)

北京时间：2026-10-27 13:49:44｜来源类型：GitHub PR｜事件状态：已合并

**事实**：对未知消费者的 address_of(BufferLoad) 同时记录读与写，避免通过不透明调用写共享内存后遗漏 barrier；保留读依赖也防止循环内必要同步被错误外提。

**重要性**：修正线程同步规划中的数据竞争来源，覆盖外部调用与共享 buffer 交互的正确性。

**风险与限制**：更保守的访问模型可能增加 barrier；具体设备覆盖和测试范围有边界，不能据此声称所有同步竞态已消除。

### 3.6 模型 & 技术｜[\[BugFix\]\[Transform\] 展开运行时迭代次数流水线的尾声](https://github.com/tile-ai/tilelang/pull/3356)

北京时间：2026-10-27 15:02:28｜来源类型：GitHub PR｜事件状态：已合并

**事实**：先简化流水线 epilogue 的范围，使 (n+s)-n 在 n 为运行时值时仍被识别为常量并展开，保证 ptx_wait_group 的 wait count 是立即数。 作者在 sm_89/CUDA 13.0 对照中确认，下游 TileOPs 中 22 个原先因 Downcast 错误失败的 paged GQA/MHA decode 检查转为通过。

**重要性**：恢复 KV 长度动态决定循环次数的 paged decode 编译链路。

**风险与限制**：覆盖的是软件流水路径；TMA/warp-specialization 路径不同，相关大套件仍有双方都存在的失败项，未报告吞吐改进。

## 四、RISC-V 核心新闻

### 4.1 模型 & 技术｜[\[RFC PATCH 0/3\] RISC-V：KVM：保留驻留的 guest Vector 状态](https://lists.infradead.org/pipermail/linux-riscv/2026-October/102208.html)

北京时间：2026-10-27 21:13:06｜来源类型：官方邮件列表／RFC｜事件状态：RFC 提出

**事实**：提议在安全的同一 hart vCPU put/load 间保留 guest Vector 数据寄存器，命中时只恢复控制 CSR；dirty put 仍保存 guest 状态，以内存为迁移和用户态访问的权威副本。 通过带 allocation cookie、generation 的 per-hart owner 和多处失效钩子约束生命周期；默认关闭。嵌套 QEMU/RVV smoke 记录 784 次 CSR-only restore 与 8 次保守回退。

**重要性**：为 RVV 虚拟化提出减少重复寄存器恢复的具体接口与验证路径。

**风险与限制**：目前是 RFC；QEMU 结果仅证明功能与事件计数，真实硬件性能测量尚未完成，还依赖独立构建修复。

### 4.2 模型 & 技术｜[\[PATCH RFC 00/18\] riscv：使用 SCMI 时钟与复位添加 Zhihe A210 初始支持](https://lists.infradead.org/pipermail/linux-riscv/2026-October/102222.html)

北京时间：2026-10-27 23:36:51｜来源类型：官方邮件列表／RFC｜事件状态：RFC 提出

**事实**：新 RFC 为 A210 SoC 与 Melon Pi V1/V2 提供初始 Linux 支持；A210 含四个 C908、四个 C920 和 E902 管理核。 相比更早的寄存器直驱方案，这次把电源、PMIC、CPU DVFS 等顺序留给 E902 固件，Linux 使用 SCMI 时钟和复位协议，并加入 AP↔E902 mailbox 及相关设备树。

**重要性**：把 RISC-V 应用核与管理固件职责显式分离，是国产 SoC 上游部署链路的架构变化。

**风险与限制**：尚未合并；选择 SCMI 的理由与固件当前能力相关，不能把这份初始支持提案写成主线内核已支持整板外设。

### 4.3 模型 & 技术｜[将 RVA23 CRD 版本升至 1.0](https://github.com/riscv/riscv-arch-test/pull/3020)

北京时间：2026-10-28 02:42:19｜来源类型：GitHub PR｜事件状态：已合并

**事实**：RVA23 Certification Requirements Document 升为 1.0，修订记录注明 2026 年 10 月 27 日经 Certification Steering Committee 表决批准；版本改用 revnumber 属性显示。

**重要性**：从草案版本推进到获批准的认证要求文档，是 RVA23 兼容性与验证范围的重要依据。

**风险与限制**：文档版本获批不等于某款处理器已通过认证；同日自动构建的认证文档标签不能另算一条产品认证新闻。

### 4.4 模型 & 技术｜[新的目标配置：产品、FPGA 镜像、RISC-V Profile 和选项交叉组合](https://github.com/openhwfoundation/cva6/pull/3675)

北京时间：2026-10-28 04:27:00｜来源类型：GitHub PR｜事件状态：已合并

**事实**：CVA6 将原 52 个目标替换为 35 个配置，分为产品、FPGA、Profile 与结构选项交叉四类，并为目标标注维护者、角色及组合范围。 Profile 文档明确列出距离对应规范的缺项；FPGA 镜像配置与产品配置分离，结构组合通过 21 个 summit 目标组织。

**重要性**：使 CPU 配置与软件验证的覆盖关系可检查，超出普通 CI 名称整理。

**风险与限制**：接近某 Profile 不等于符合该 Profile；例如文中明确 RVA23S64 仍缺 V 等扩展，选项交叉配置允许继续变化。

### 4.5 模型 & 技术｜[feat(Zvdot4a8i)：添加扩展和指令数据](https://github.com/riscv/riscv-unified-db/pull/2015)

北京时间：2026-10-28 04:26:17｜来源类型：GitHub PR｜事件状态：已合并

**事实**：Unified DB 加入 Zvdot4a8i 的 YAML 扩展及指令数据，描述四个 INT8 元素点积并累加到 32 位目标的向量操作。

**重要性**：把面向 AI/ML、图像与 DSP 的整数点积扩展纳入机器可读描述层，利于后续工具识别。

**风险与限制**：此次记录版本是 development 0.1；作者明确当前规范已到 0.7，版本更新和 IDL 语义函数留待后续。不能写成规范批准或完整执行模型。

### 4.6 模型 & 技术｜[fix(param)：当父模式能够支持 32 位时，要求 VSXLEN 和 VUXLEN 也支持 32 位](https://github.com/riscv/riscv-unified-db/pull/2255)

北京时间：2026-10-28 04:55:14｜来源类型：GitHub PR｜事件状态：已合并

**事实**：补全虚拟化模式位宽约束：父模式能够运行 RV32 时，相应子模式必须也允许 32 位，避免 VSXLEN=[32,64]、VUXLEN=[64] 等与特权规范不一致的 hart 配置被接受。

**重要性**：将 Hypervisor 的模式依赖表达为可执行参数约束，有助于配置生成与架构验证保持一致。

**风险与限制**：现有 cfgs 中没有配置因此改变判定；这是规范模型正确性修正，不是新增 Hypervisor 硬件支持。

### 4.7 模型 & 技术｜[perf(LSQ)：修复 `LSQ` 中的几个性能问题](https://github.com/OpenXiangShan/XiangShan/pull/6767)

北京时间：2026-10-27 11:25:30｜来源类型：GitHub PR｜事件状态：已合并

**事实**：香山 LSQ 修改 replay/StoreQueue 的等待索引与跨 16 字节 Store 发送条件；后者由等待整个数据队列为空改为检查所需两个端口就绪。 同一 StoreQueue 修正系列还以 Sbuffer credits 保持完成前缀出队，并使 PA 候选匹配独立于 VA tag，避免物理重叠候选被提前排除。

**重要性**：直接涉及乱序访存、Store forwarding 和提交边界，属于团队关注的处理器软件验证底层语义。

**风险与限制**：未提供 IPC 或应用级性能测量；各项修正的适用场景由其具体匹配与完成条件限定。

相关原文：[OpenXiangShan/XiangShan PR #6768](https://github.com/OpenXiangShan/XiangShan/pull/6768)；[OpenXiangShan/XiangShan PR #6769](https://github.com/OpenXiangShan/XiangShan/pull/6769)。

### 4.8 模型 & 技术｜[feat(ame)：支持以 FP32 累加的 FP8 MMA](https://github.com/OpenXiangShan/NEMU/pull/1266)

北京时间：2026-10-27 16:23:37｜来源类型：GitHub PR｜事件状态：已合并

**事实**：NEMU 为 AME MMA 增加 E4M3×E4M3 与 E5M2×E5M2 输入、FP32 累加，统一浮点 MMA 向量化路径，并增加独立矩阵执行的 riscv64-matrix-xs-lite_defconfig。

**重要性**：扩展矩阵指令的可执行参考模型，为 FP8 编译输出与硬件差分验证提供新数值格式。

**风险与限制**：这是 NEMU 模型与配置能力，不代表相应硬件的性能、软件栈端到端精度或规范状态已获验证。

### 4.9 模型 & 技术｜[feat(yolo26_seg)：添加 YOLO26 实例分割](https://github.com/spacemit-com/model-zoo-vision/pull/112)

北京时间：2026-10-27 15:16:45｜来源类型：GitHub PR｜事件状态：已合并

**事实**：进迭时空视觉 model zoo 加入 YOLO26 实例分割 deploy segmentor、C++/Python 示例和下载脚本，并接入 download_all_models。

**重要性**：把新视觉任务纳入厂商部署样例链，提供从模型获取到调用的具体接口。

**风险与限制**：本次没有给出设备性能、精度或全型号适配结论；示例可用性与特定 SDK/模型产物仍有关。

## 五、AI 业界重磅

### 5.1 重磅｜[Genie 4：新一代世界模拟](https://deepmind.google/blog/genie-4-a-new-generation-of-world-simulation/)

北京时间：2026-10-28 00:13:00｜来源类型：官方技术博客｜事件状态：研究预览公布

**事实**：Google DeepMind 公布 Genie 4，将世界模拟与原生音频、环境声和交互对话联合生成，并利用 Gemini 系列增强场景外观、行为和连续性。 官方称已向可信测试者推出 research preview；后续计划从 Project Genie 的 Google AI Ultra 用户及早期 API/AI Studio 合作伙伴逐步开放。

**重要性**：世界模型从可导航视觉场景推进到多模态、可实时引导的交互模拟。

**风险与限制**：现阶段访问受限，消费者及开发者普遍开放仍是后续计划；展示不能替代对长期物理一致性与机器人迁移效果的独立测评。

### 5.2 模型 & 技术｜[使用 LFM3 构建专门的多模态模型](https://huggingface.co/blog/LiquidAI/lfm3)

北京时间：2026-10-28 01:16:52｜来源类型：官方技术博客｜事件状态：模型家族发布

**事实**：Liquid AI 发布共享 LFM3-1.7B backbone 的六个约 2B 级文本、视觉、音频、omni 与决策模型；文本模型支持 256K context，视觉版本加入视频输入。 24 层 backbone 采用 13 个 gated short-convolution、5 个 sliding-window attention 和 6 个 global attention 的 tri-hybrid 结构，目标是单 GPU 微调与端侧运行。

**重要性**：用共享骨干承载多模态专用模型，提供参数规模较小的模型导入和端侧部署对象。

**风险与限制**：能力与设备比较来自发布团队；不同模态有各自依赖和推理入口，约 1 GB 的文本 4-bit 权重大小不等于完整运行内存。

### 5.3 模型 & 技术｜[在 Kubernetes 中使用 NVIDIA CUDA 13.4 实施显存边界的 GPU 共享](https://developer.nvidia.com/blog/memory-enforced-gpu-sharing-in-kubernetes-with-nvidia-cuda-13-4/)

北京时间：2026-10-28 04:50:39｜来源类型：官方技术博客｜事件状态：基础设施能力公布

**事实**：CUDA 13.4 提供对应用透明的 GPU 显存边界，可设置精确额度或保证最小值加可突发上限，并选择 time-sharing 或 sm-sharing。 新的开源 GPU Fractioning Operator 在节点上配置 MPS、执行额度并按 fraction 度量；KAI Scheduler v0.18.0 通过 NvFractions 与每容器 Pod annotations 对接。

**重要性**：把调度侧的 GPU 份额声明连接到节点执行层的实际显存约束，面向共享 AI 集群。

**风险与限制**：超额分配会失败；调度器仍需选择有足够空闲内存的 GPU，同一 GPU 不能混用 compute 模式，不能将其等同于 MIG 硬件隔离。

### 5.4 模型 & 技术｜[在 NVIDIA DGX Spark 上提升 MoE 推理的 3 种方法](https://developer.nvidia.com/blog/3-ways-to-boost-moe-inference-on-nvidia-dgx-spark/)

北京时间：2026-10-27 23:00:00｜来源类型：官方技术博客｜事件状态：技术文章发布

**事实**：官方分析以 Ling-3.0-flash、DGX Spark、batch=1 为例，组合 Humming MXFP4 MoE kernel、在线 FP8 LM head 和 NEXTN 多 token 预测。 完整 GSM8K 测试中，三项组合使 decode throughput 增加 42.5%、平均延迟下降 24.5%，没有检测到统计显著的准确率变化。

**重要性**：把权重带宽与多 token 验证放在实际服务路径中分析，给出从单 kernel 到任务测量的对照。

**风险与限制**：合成负载中的 185.1% 吞吐增益不能替代 GSM8K 的 42.5%；设置与基线精度路径有关，不能推广为其他模型或批量的普遍收益。

### 5.5 深度洞见｜[我们能相信 LLM 判断研究想法是否新颖吗？](https://huggingface.co/blog/allenai/novelty-bench)

北京时间：2026-10-28 00:12:42｜来源类型：官方技术博客｜事件状态：研究与基准发布

**事实**：Ai2 发布 Old Ideas, Novel Problems 研究、benchmark、数据和代码，检查 LLM 对研究想法新颖性的判断稳定性。 在同一 Human+Generated idea pairs 上，仅调整提示中的几个词，GPT-5.4 判别准确率由 93.5% 降至 40.9%；检索与更多推理没有一致改善结果。

**重要性**：直接测量 AI 科研工作流中“生成后由另一个模型判新颖”的薄弱环节。

**风险与限制**：Human+Generated 使用弱标签假设，人工复查规模有限；提示敏感性有证据，但不能把这一个基准推广为全部科研审稿能力。

### 5.6 深度洞见｜[微调 NVIDIA Nemotron，使其了解本地规则，而不只是本地语言](https://developer.nvidia.com/blog/fine-tuning-nvidia-nemotron-to-know-local-rules-not-just-local-languages/)

北京时间：2026-10-28 00:00:00｜来源类型：官方技术博客｜事件状态：技术文章发布

**事实**：官方用法国公共服务场景串联 Nemotron-Personas 合成对话、官方程序文本、LoRA 微调、合并与 vLLM 服务；数据按 persona 来源拆分，避免同一 persona 进入训练和验证两侧。

**重要性**：把地域语言能力与实际办事规则分开验证，提供从数据生成到冻结测试集比较的完整部署示例。

**风险与限制**：这是工作流和示例回答，未给出证明普遍准确率提升的完整评测；规则事实仍依赖官方文本，示例不能替代真实业务评估。

### 5.7 工具 & 产品｜[NVIDIA 与 Oracle 将 NVIDIA Vera Rubin 扩展至数吉瓦规模，并将 NVIDIA Vera CPU 引入 Oracle Cloud Infrastructure](https://nvidianews.nvidia.com/news/nvidia-and-oracle-expand-nvidia-vera-rubin-to-multi-gigawatt-scale-and-bring-nvidia-vera-cpu-to-oracle-cloud-infrastructure)

北京时间：2026-10-28 00:00:00｜来源类型：官方公告｜事件状态：部分能力已普遍可用，扩容仍为计划

**事实**：NVIDIA 与 Oracle 宣布 Vera Rubin NVL72 及 Vera CPU 已在 OCI 普遍可用，同时计划把相关 Superclusters 扩至数吉瓦，并引入 x86 配置的 HGX Rubin NVL8。 官方将 Vera CPU 用于 agent 的工具调用、代码执行和检索等工作，并允许 CPU/GPU 资源独立扩展。

**重要性**：体现 AI 基础设施同时扩展模型计算与 agent 沙箱工作负载的资源路径。

**风险与限制**：数吉瓦扩容与 HGX Rubin NVL8 是计划；已可用的具体地域、客户容量与普遍吞吐不由这份公告保证。

## 六、总结与趋势观察

- **部署资源的优化跨越加载、图转换与调度层。** [ExecuTorch delegate 复用](https://github.com/pytorch/executorch/pull/24127)减少重复初始化与驻留，[Torch-TensorRT 的 slice_scatter 路径](https://github.com/pytorch/TensorRT/pull/4898)给出实际模型延迟对照，[Kubernetes GPU 显存约束](https://developer.nvidia.com/blog/memory-enforced-gpu-sharing-in-kubernetes-with-nvidia-cuda-13-4/)则把资源承诺落实到节点执行。三者作用层次不同，收益不能相互替代。
- **编译链正同时完善可调用接口和中间表示边界。** [Buddy VIR Python 绑定](https://github.com/buddy-compiler/buddy-mlir/pull/1009)让图构造与 lowering 进入 Python 工作流，[torch-mlir QLinear 的 PT2E 迁移](https://github.com/llvm/torch-mlir/pull/4833)统一量化表示，[ClangIR ABI 讨论](https://discourse.llvm.org/t/rfc-enable-clangir-build-by-default/91730/116)进一步说明新目标分类器的验证入口。接口接通、语义正确和完整后端覆盖仍是不同阶段。
- **RISC-V 的虚拟化与平台接入继续在状态所有权边界上细化。** [KVM Vector RFC](https://lists.infradead.org/pipermail/linux-riscv/2026-October/102208.html)约束寄存器驻留与失效，[A210 SCMI RFC](https://lists.infradead.org/pipermail/linux-riscv/2026-October/102222.html)划分应用核与管理固件职责，[Unified DB 的 Hypervisor 位宽约束](https://github.com/riscv/riscv-unified-db/pull/2255)补齐模式之间的合法配置关系。前两项仍为提案，不能等同于主线已交付能力。

## 附录：信源说明

本期主要来源为 PyTorch/ExecuTorch、LLVM/MLIR、IREE、Buddy Compiler、Triton/TileLang、linux-riscv 邮件列表、RISC-V 规范与验证项目，以及 Google DeepMind、Liquid AI、Ai2 和 NVIDIA 的官方原文。GitHub 条目区分正式发行、已合并实现与未合并提案；论坛内容区分主题方案和讨论新进展。

性能与资源数据均来自原作者所述环境；编译检查、仿真计数、单项微基准和端到端任务结果各有适用边界。研究预览、社区 fork、尚未合并的 RFC 与局部后端修正，不代表全面可用或普遍性能保证。
