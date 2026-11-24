# Codex 技术情报每日动态（2026-11-25）

**调研窗口**：北京时间（2026-11-24 16:48:40，2026-11-25 06:00:27]。

**覆盖方向**：PyTorch、LLVM/MLIR、Triton & TileLang、RISC-V 软件与硬件生态，以及 AI 模型研究、基础设施与部署。

**信息口径**：采用原始公告、正式技术讨论、发布记录与代码变更；性能数字限于原文给定的硬件、参数和测试方法，提案与计划按其当前状态表述。

## 今日要闻

- [ExecuTorch–TensorRT 公布不依赖 libtorch 的机器人部署链路，3.6B 策略可在 8 GB Orin Nano 上运行。](https://pytorch.org/blog/executorch-tensorrt-pytorch-robot-policies-at-nvidia-tensorrt-speed/)
- [OpenAI 公布 GPT‑6 蛋白质设计预印本及五个目标的实验结合证据，候选筛选仍借助专用模型。](https://openai.com/index/gpt-6-models-demonstrate-intrinsic-protein-design-capabilities)
- [Google 发布 Humanity’s First Exam，以 544 个基础图文问题评估模型重复作答的可靠性。](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/humanitys-first-exam/)
- [Google Workspace 的 MCP servers 与 CLI 宣布 GA，并公布智能体接口迁移及后续计费计划。](https://workspaceupdates.googleblog.com/2026/11/mcp-servers-cli-standardized-usage-tiers-now-available.html)

## 今日索引

- **一、PyTorch 生态核心动态**：机器人策略轻量部署、LSE 梯度、MPS 内存、ROCm 图捕获及端侧运行库。
- **二、LLVM/MLIR 最新进展**：INLINEASM_BR 提案、线性类型与字节码兼容讨论、Buddy VIR 调度、模型导入正确性。
- **三、Triton & TileLang 技术动态**：GPU 设备库内联、TMA 竞争检测、Intel scale 流水化、TMEM 布局与 Ascend wheel 验证。
- **四、RISC-V 核心新闻**：认证规则映射、Worlds 固件、K3 UFS 与推理正确性、香山缓存、UDB 和 CVA6。
- **五、AI 业界重磅**：蛋白质设计、可靠性基准、天文搜索、Workspace 智能体接口与本地机器人/世界模型部署。

## 一、PyTorch 生态核心动态

### 1.1 工具 & 产品｜[ExecuTorch-TensorRT: PyTorch Robot Policies at NVIDIA TensorRT Speed（ExecuTorch-TensorRT：以 NVIDIA TensorRT 的速度运行 PyTorch 机器人策略）](https://pytorch.org/blog/executorch-tensorrt-pytorch-robot-policies-at-nvidia-tensorrt-speed/)

北京时间：2026-11-25 04:31:02｜来源类型：官方博客｜事件状态：已发布。

**事实**：Torch-TensorRT 2.15 与 ExecuTorch 1.6 将 torch.export 捕获的策略编译成 ExecuTorch 程序，运行端无需加载 libtorch；TensorRT 无法承接的部分可交由 CUDA 后端，在同一 CUDA stream 上执行。 官方在 Jetson Orin Nano 8 GB 上测得 pi0.5（3.6B）每动作块 467 ms、进程内存 5538 MiB，ONNX–TensorRT 分别为 452 ms、5429 MiB；该板上 PyTorch eager 无法装下同一策略。

**重要性**：提供从 PyTorch 导出到机器人本机运行的完整链路，减少运行时与内存开销。

**风险与限制**：这是指定权重、功耗和 CUDA graph 条件下的官方测量；大策略仍需分段构建，green context 也不能隔离显存带宽。

### 1.2 模型 & 技术｜[\[varlen\] Propagate gradients through the returned lse (#206399)（\[varlen\] 通过返回的 lse 传播梯度 (#206399)）](https://github.com/pytorch/pytorch/commit/19db9aeb332fb4ad1207b3f6341b9837f3e73a74)

北京时间：2026-11-25 05:38:32｜来源类型：官方主干 commit｜事件状态：已进入主干。

**事实**：varlen_attn 请求返回 LSE 时，LSE 现在可微，其梯度传给 _flash_attention_backward 新增的 grad_logsumexp 输入，修复注意力 sink 重缩放损失中的 dQ/dK 静默错误。 当前只有 FA4 消费该梯度；不支持的 FA2、cuDNN 等路径报错，编译区域仅用于日志的 LSE 需要 detach。

**重要性**：修正训练损失依赖 LSE 时的自动微分语义，也改变部分 torch.compile 调用的兼容行为。

**风险与限制**：作者在 GB300 验证；FA4 在 SM100、head_dim=256 等组合仍有限制，不等于全部后端已支持。

相关原文：[\[varlen\] Propagate gradients through the returned lse](https://github.com/pytorch/pytorch/pull/206399)；[\[sdpa\] Add grad_logsumexp input to _flash_attention_backward](https://github.com/pytorch/pytorch/pull/206398)。

### 1.3 模型 & 技术｜[\[MPS\] Add mps_large_alloc_threshold_mb allocator setting (#188804)（\[MPS\] 添加 mps_large_alloc_threshold_mb 分配器设置 (#188804)）](https://github.com/pytorch/pytorch/commit/6171a705557735fbbfd4bcc274ff6a7cc7d2ea9a)

北京时间：2026-11-25 03:59:50｜来源类型：官方主干 commit｜事件状态：已进入主干。

**事实**：新增 mps_large_alloc_threshold_mb，使达到阈值的分配使用按需求大小配置的堆；默认值 0 保持原行为。 作者展示阈值 64 MB 时，100 MiB 单次分配引起的驱动内存增长从 1024 MiB 降至 100 MiB，200 MiB 分配从 1024 MiB 降至 200 MiB。

**重要性**：为 Apple MPS 中等大小张量避免首次占用整块 1 GiB XLARGE 堆提供控制项。

**风险与限制**：这是单次分配结果，不能外推整模型内存收益；运行时 allocator setter 在 MPS 上对部分非 MPS 配置的报错行为有变化。

相关原文：[\[MPS\] Add mps_large_alloc_threshold_mb allocator setting](https://github.com/pytorch/pytorch/pull/188804)。

### 1.4 模型 & 技术｜[\[c10d\]\[nccl2\] Fix deadlock between enqueueWork and graphCleanupCallback (#206476)（\[c10d\]\[nccl2\] 修复 enqueueWork 与 graphCleanupCallback 之间的死锁 (#206476)）](https://github.com/pytorch/pytorch/commit/2d924eca88cdf33f233b607b01a68c36327773e5)

北京时间：2026-11-25 03:28:45｜来源类型：官方主干 commit｜事件状态：已进入主干。

**事实**：将图捕获用户对象注册移出 graph_capture_work_mutex_ 临界区，解除 HIP 内部锁与图清理回调之间的锁顺序环。 作者在 MI300X 上的 source build 对照为未修复 200 轮出现 4 次挂起，修复后两个集体通信测试配置各 550 轮均未挂起。

**重要性**：面向 ROCm 图捕获中的间歇性 300 秒超时，影响分布式训练稳定性。

**风险与限制**：未在 CUDA 上验证，也未声称有限轮数能保证所有场景无死锁。

相关原文：[\[c10d\]\[nccl2\] Fix deadlock between enqueueWork and graphCleanupCallback](https://github.com/pytorch/pytorch/pull/206476)。

### 1.5 模型 & 技术｜[Arm backend: Add IO planning to Ethos-U delegate（Arm 后端：为 Ethos-U delegate 添加 I/O 规划）](https://github.com/pytorch/executorch/pull/25671)

北京时间：2026-11-24 22:51:17｜来源类型：官方 GitHub PR｜事件状态：已合并。

**事实**：Ethos-U delegate 根据 Vela 区域信息和运行时张量地址生成 I/O plan，并使用已规划的 scratch 决定输入输出直接访问或复制；共用执行路径连接 CoreDriver 与 LinuxDriver。

**重要性**：把 NPU 内存放置与执行传参显式连接，属于端侧部署链路变化。

**风险与限制**：是否需要复制仍取决于 Vela 区域和实际张量地址；该变更没有提供可普遍外推的时延或零拷贝收益。

### 1.6 工具 & 产品｜[Support multiple PTE files in CMake operator selection（在 CMake 算子选择中支持多个 PTE 文件）](https://github.com/pytorch/executorch/pull/24576)

北京时间：2026-11-25 00:08:29｜来源类型：官方 GitHub PR｜事件状态：已合并。

**事实**：EXECUTORCH_SELECT_OPS_MODEL 接受分号分隔的 PTE 列表，分别生成算子 YAML，再以 gen_all_oplist 合并；模型和中间文件均加入构建依赖。

**重要性**：多个模型共用裁剪后的 ExecuTorch 运行库时，无需手工维护单模型算子集合。

**风险与限制**：三个算子选择入口仍互斥；合并的是所需算子集合，未承诺运行时内存或性能改善。


## 二、LLVM/MLIR 最新进展

### 2.1 模型 & 技术｜[\[RFC\] Make INLINEASM_BR a terminator again（\[RFC\] 再次将 INLINEASM_BR 设为终结指令）](https://discourse.llvm.org/t/92967/1)

北京时间：2026-11-25 04:47:02｜来源类型：官方论坛｜事件状态：RFC 提案。

**事实**：提案把 asm goto 带输出的 INLINEASM_BR 恢复为 MIR 终结指令，拆分关键边并在每个后继入口读取输出，让溢出与复制放置符合真实控制流。

**重要性**：试图从表示层消除寄存器分配、MachineCSE 等逐 pass 修补仍反复出现的误编译根因。

**风险与限制**：仍是设计提案，需要同步调整指令选择、寄存器分配与验证器；并非已经完成修复。

### 2.2 模型 & 技术｜[\[RFC\] Linear Type System Support in Upstream MLIR（\[RFC\] 在上游 MLIR 中支持线性类型系统）](https://discourse.llvm.org/t/92947/7)

北京时间：2026-11-24 21:23:33｜来源类型：官方论坛｜事件状态：讨论新进展。

**事实**：完整提案是为必须恰好使用一次的 SSA 资源增加 LinearType trait 和消费—产生关系接口，以复用量子编译器的不可复制资源分析。 本期作者澄清线性约束可由独立 analysis pass 检查，并探索类似 BufferizableOpInterface 的外部接口模型，以避免 IR verifier 跨 use-def 链。

**重要性**：为上游资源所有权分析明确实现边界，关系到 MLIR 方言可复用性。

**风险与限制**：上游用户范围、维护成本及不依赖接口的方言如何接入仍有分歧，未形成采纳结论。

### 2.3 深度洞见｜[Bytecode compatibility guarantees for the Func dialect?（Func 方言的字节码兼容性保证？）](https://discourse.llvm.org/t/92814/4)

北京时间：2026-11-24 21:25:12｜来源类型：官方论坛｜事件状态：讨论新进展。

**事实**：Arm 希望 TOSA 工件中的 Func 操作也具备字节码兼容保证；新回复提出新 reader 接受历史工件、新 writer 在可表达时面向旧 reader，破坏变化需版本递增及兼容逻辑。 新增属性、operand 或顺序变化都可能触发版本处理；无法用新操作表达旧语义时，需要保留 legacy 表示，可先从 func.func 与 func.return 开始。

**重要性**：涉及模型编译产物的长期存储与部署兼容，不能只靠 TOSA 自身版本保护。

**风险与限制**：目前是作者拟议的兼容政策，尚非 Func 已承诺的稳定 ABI。

### 2.4 深度洞见｜[\[RFC\]\[LTO\] Make hosts responsible for loading plugins（\[RFC\]\[LTO\] 让宿主负责加载插件）](https://discourse.llvm.org/t/92892/3)

北京时间：2026-11-25 01:24:09｜来源类型：官方论坛｜事件状态：讨论新进展。

**事实**：提案让宿主向 LTO 交付已加载、已配置的插件；新回复说明反馈已转向移除旧接口，而非仅弃用，并把 ParseArguments 的调用从 LTO::run 移到 lld driver。

**重要性**：澄清嵌入 LTO 与插件加载器之间的责任，降低全局命令行处理对不同宿主的耦合。

**风险与限制**：插件配置放在全局、编译 context 还是每次编译对象仍是单独的 API 设计问题，不能视为本 RFC 已统一。

### 2.5 深度洞见｜[\[RFC\] Universal CFI（\[RFC\] 通用 CFI）](https://discourse.llvm.org/t/92955/12)

北京时间：2026-11-25 04:48:44｜来源类型：官方论坛｜事件状态：讨论新进展。

**事实**：uCFI 主线基于 kCFI 哈希检查，探索无需 LTO 的用户态 C/C++ 加固，并用 LLD trampoline 接入未插桩外部函数。 本期回复具体说明 LLVM 发出取址函数的 __kcfi_typeid_*，lld 在较早阶段创建 trampoline，仅改写需要它的对象文件引用；跨 DSO 传递各自 trampoline 地址会破坏函数指针等价。

**重要性**：为部署方明确无 LTO 加固的链接与 ABI 代价。

**风险与限制**：这是新回复披露的边界；不能写成已经兼容所有未插桩库或所有函数指针来源。

### 2.6 模型 & 技术｜[\[midend\]\[VIR\] Add -vir-schedule: order the ops of vir.set_vl regions for less register pressure（\[midend\]\[VIR\] 添加 -vir-schedule：对 vir.set_vl 区域的操作排序以降低寄存器压力）](https://github.com/buddy-compiler/buddy-mlir/pull/1079)

北京时间：2026-11-25 00:20:05｜来源类型：官方 GitHub PR｜事件状态：已合并。

**事实**：新增调度 pass 以依赖后序和寄存器需求排序，仅在 MaxLive 严格下降时重排；相关 #1082 对齐 RVV 标量广播计数，#1083 将它接到模型 kernel 与 YOLOv8 的 VIR lowering 前。 MoE top-8 combine 在 VLEN 256 时 Req_Group 从 9 降为 3、LMUL 2 升为 8；作者在 K3 X100 测得热缓存 22.5→13.2 µs，冷缓存 82.0→31.6 µs。

**重要性**：直接关联 Buddy 的 Linalg→VIR→Vector→RVV 链路，使加载顺序造成的保守向量宽度可以被优化。

**风险与限制**：收益属于单内核微基准；#1083 对现有 YOLOv8 和 decode attention 检查仍为字节或 IR 相同，未证明整模型提速；允许 reassoc 的 reduction 可能随宽度改变浮点分组。

相关原文：[\[midend\]\[VIR\] -vir-schedule: count the vector registers as on RVV with rvv=true](https://github.com/buddy-compiler/buddy-mlir/pull/1082)；[\[tools\]\[models\] Schedule the VIR regions before -lower-vir-to-vector](https://github.com/buddy-compiler/buddy-mlir/pull/1083)。

### 2.7 模型 & 技术｜[\[TorchToLinalg\] Add bounds checks to the index_select lowering（\[TorchToLinalg\] 为 index_select lowering 添加边界检查）](https://github.com/llvm/torch-mlir/pull/4912)

北京时间：2026-11-25 00:22:58｜来源类型：官方 GitHub PR｜事件状态：已合并。

**事实**：TorchToLinalg 的 index_select 在 tensor.extract 前复用 gather/embedding 检查，断言 index≥0 且 index<size；ONNX Gather 归一化后仍越界的索引也被拦截。

**重要性**：把模型导入后原本未检查的越界读取恢复为显式错误，关系到 PyTorch/ONNX 到 Linalg 的语义一致性。

**风险与限制**：本次只改这一路 lowering；动态 IndexSelect 在部分后端仍有既存失败，不能据此宣称所有导入路径已一致。

### 2.8 模型 & 技术｜[\[Torch\] Fix instance_norm with dynamic dims, no affine and running stats（\[Torch\] 修复动态维度、无 affine 和运行统计量情况下的 instance_norm）](https://github.com/llvm/torch-mlir/pull/4885)

北京时间：2026-11-24 17:29:14｜来源类型：官方 GitHub PR｜事件状态：已合并。

**事实**：方差改由 aten.mean.dim 计算，避免动态空间维度以 -1 相乘造成静默错误或 NaN；跳过 None 的 weight/bias，并在 use_input_stats=false 时走推理模式 batch_norm。

**重要性**：修正默认无 affine 参数与 eval 模式模型的常见导入语义。

**风险与限制**：匹配仍要求已知 rank、浮点 dtype 和常量布尔 use_input_stats，不能推广为任意动态形状均已支持。


## 三、Triton & TileLang 技术动态

### 3.1 模型 & 技术｜[\[AMD\] Apply target CPU and features to bundled device libraries before linking（\[AMD\] 在链接前向随附的设备库应用目标 CPU 和特性）](https://github.com/triton-lang/triton/pull/12928)

北京时间：2026-11-25 04:49:48｜来源类型：官方 GitHub PR｜事件状态：已合并。

**事实**：为随 Triton 附带的 OCML/OCKL 应用 kernel 的目标 CPU 与 feature，恢复 gfx950 禁用 packed-fp32-ops 时的内联兼容；用户库和显式库属性仍保留自己的设置。 MI350X 的 FP8 MQA kernel 中位时延由 2065.6 降至 1580.4 µs，寄存器 spill 从 8 降至 0。

**重要性**：修复后端与设备库特性不一致带来的显著性能退化。

**风险与限制**：只覆盖指定 kernel、设备和 feature 组合；不改变 LLVM 的内联安全规则。

### 3.2 模型 & 技术｜[\[ConSan\] Track the latest bulk or TMA load until each CTA acquires it（\[ConSan\] 跟踪最新的 bulk 或 TMA 加载，直到每个 CTA 获取它）](https://github.com/triton-lang/triton/pull/12871)

北京时间：2026-11-25 03:22:43｜来源类型：官方 GitHub PR｜事件状态：已合并。

**事实**：ConSan 为 cluster 中每个共享内存区域保存最新 bulk/TMA 写入，发起加载后清除各观察者可见性，等待完成 barrier 后再授权读取；相关 #12872 将每份 multicast replica 绑定到它实际对应的完成 barrier。

**重要性**：补上 multicast 重用缓冲区时可能漏检的写前读竞争，避免等待本地 barrier 错误授权覆盖其他 CTA 数据。

**风险与限制**：这是检测器覆盖与可见性建模变化，不是证明所有 kernel 无数据竞争。

相关原文：[\[ConSan\] Match each multicast replica to its completion barrier](https://github.com/triton-lang/triton/pull/12872)。

### 3.3 模型 & 技术｜[\[GSan\] Track memory reads from TMA im2col loads（\[GSan\] 跟踪 TMA im2col 加载的内存读取）](https://github.com/triton-lang/triton/pull/12874)

北京时间：2026-11-25 02:32:11｜来源类型：官方 GitHub PR｜事件状态：已合并。

**事实**：GSan 开始为此前跳过的 TMA im2col message 插桩，解码 tensor map，记录 predicated、multicast 和 FP4 padding 情况下实际读取的字节。

**重要性**：扩展竞争检测和分配分析对卷积类 TMA 访问的观测范围。

**风险与限制**：作者报告 GB200 的 25 个 GPU 用例通过；这不是卷积吞吐提升或跨硬件通用验证。

### 3.4 模型 & 技术｜[\[Intel\] Pipeline tt.dot_scaled K loops and load their scales in the DPAS scale layout（\[Intel\] 对 tt.dot_scaled K 循环进行流水化，并以 DPAS scale 布局加载 scale）](https://github.com/intel/intel-xpu-backend-for-triton/pull/8958)

北京时间：2026-11-24 23:11:11｜来源类型：官方 GitHub PR｜事件状态：已合并。

**事实**：流水化识别 tt.dot_scaled 的 A/B 操作数，并把 scale 的单消费者加载重新物化为目标 scale 布局，消除循环内 SLM 布局转换与 barrier。 CRI、M=N=8192、K=4096 的一组 512 GRF 配置从 210 提升到 298 TFLOPS，达到同构建 plain dot 的 94%。

**重要性**：改善低精度 GEMM 的预取与 scale 数据路径。

**风险与限制**：测量向容器注入了设备 capability；256 GRF 下部分配置 spill 增加，IGC 指令生成与字节 gather 瓶颈仍在。

### 3.5 模型 & 技术｜[\[NVIDIA\] Check TMEM row compatibility for tcgen05 MMA（\[NVIDIA\] 检查 tcgen05 MMA 的 TMEM 行兼容性）](https://github.com/triton-lang/triton/pull/12939)

北京时间：2026-11-25 01:04:15｜来源类型：官方 GitHub PR｜事件状态：已合并。

**事实**：LHS promotion 与 MMA verifier 共用 TMEM 行布局兼容检查，阻止 A 与 accumulator 独立切换半区；两 CTA、blockM=64 的 A 分配还检查 K 向量复制。

**重要性**：修复部分 64×512 accumulator 分块导致 tcgen05.mma 地址不对齐 fault 的路径。

**风险与限制**：不兼容 A 保留在 shared memory；不代表所有 TMEM 布局都获得支持或性能提升。

### 3.6 工具 & 产品｜[Add Ascend NPU wheel test to the Dist pipeline（向 Dist 流水线添加 Ascend NPU wheel 测试）](https://github.com/tile-ai/tilelang/pull/4419)

北京时间：2026-11-24 22:59:26｜来源类型：官方 GitHub PR｜事件状态：已合并。

**事实**：发布流水线新增真实 Ascend runner 作业，安装构建出的 Linux x86_64 fat wheel，移开源码包后运行 examples/ascend 和 testing/ascend 的非 PTO 用例；当前使用 CANN 9.3 weekly 与 Python 3.13。

**重要性**：首次把 wheel 中的 Ascend 部分纳入真实 NPU 执行检查，补上分发产物而非源码可用性的验证路径。

**风险与限制**：此作业刻意不在 publish-pypi 的 needs 链，runner 离线不会阻断发布；不能写成 wheel 必须通过 Ascend 才可发布。


## 四、RISC-V 核心新闻

### 4.1 模型 & 技术｜[CTP: fix normative-rule coverpoint mappings（CTP：修正规范规则的覆盖点映射）](https://github.com/riscv/riscv-arch-test/pull/3298)

北京时间：2026-11-25 02:16:46｜来源类型：官方 GitHub PR｜事件状态：已合并。

**事实**：认证测试计划为 33 个文件中的 125 条规范规则新增或修正映射，让规则只指向确实检查其行为的 coverpoint；缺口明确标为 Needs test、Needs infrastructure 或 No independent assertion。 例如 vnclip/vnclipu 的 vxsat、Za128rs reservation set、跨页取指 stval 和 M-mode 的 HLV/HSV pointer masking 仍有缺口；相关 #3299 统一解释前缀，#3293 更新 Sail→RVVI 覆盖说明。

**重要性**：使 ISA/Hypervisor/向量验证结论更可审计，避免用宽泛匹配误称完整覆盖。

**风险与限制**：这是认证覆盖映射与策略说明修正，没有把尚缺的测试变成已实现能力。

相关原文：[CTP: standardize normative-rule explanations and cite UDB checks](https://github.com/riscv/riscv-arch-test/pull/3299)；[CTP: update test descriptions and coverage guidance](https://github.com/riscv/riscv-arch-test/pull/3293)。

### 4.2 模型 & 技术｜[lib: sbi_hart_worlds: Implement WID hart protection backend（lib: sbi_hart_worlds: 实现 WID hart 保护后端）](https://github.com/riscv-software-src/opensbi/commit/cf5eca3b61115e6d1d4c6bdc43fd810aa3f2a157)

北京时间：2026-11-24 18:00:24｜来源类型：官方主干 commit｜事件状态：已进入主干。

**事实**：OpenSBI 增加基于 World ID 的 hart protection backend，在 domain 切换时校验并配置 mlwid/mwiddeleg，显式按顺序保存和恢复 slwid。 同系列在 hart reinit 锁定此前 RoT 设置的 mwid，补充 domain/root 的 wid 与 widdeleg 设备树属性。

**重要性**：将 RISC-V Worlds 的隔离标识接入固件 domain 生命周期，涉及可信启动后的权限边界。

**风险与限制**：依赖相应 Worlds ISA 扩展和平台允许的 WID 集合；OpenSBI 不负责替 RoT 生成 mwid，也不等于现有板卡新增硬件隔离能力。

相关原文：[相关原文](https://github.com/riscv-software-src/opensbi/commit/9a66d5d2313cbb72352ddf3e8bbf3b54c553b0a4)；[相关原文](https://github.com/riscv-software-src/opensbi/commit/d0139edf3168bb63df9b2a499e9031a1a8edd65e)。

### 4.3 模型 & 技术｜[K3UfsHcDxe: Stage PWM-to-FAST transitions through FASTAUTO（K3UfsHcDxe：通过 FASTAUTO 分阶段完成 PWM 到 FAST 的切换）](https://github.com/openRuyi-Project/edk2-ruyi/commit/b89f76f2889d0ad9b58188afd9ad6a9b57771afe)

北京时间：2026-11-24 18:08:22｜来源类型：官方主干 commit｜事件状态：已进入主干。

**事实**：K3 UFS 驱动在收发两端均为 PWM 时，先用协商的 gear、lane 和 HS series 切到 FASTAUTO，等待完成及 PLL lock，再释放 analog gear override 并请求 FAST。 该序列针对 YMTC YMUS8B5TH1A1C1 直接切换后无法回答 descriptor 查询的问题，随后被 ruyifirmware 的 K3 子模块更新采用。

**重要性**：关系到 K3 存储枚举和固件启动可靠性，是 Linux 驱动经验进入 UEFI 的适配。

**风险与限制**：上游已有对应 Linux 序列；本期新增是 UEFI 集成，不能写成新 UFS 规格或所有设备通用提速。

相关原文：[相关原文](https://github.com/openRuyi-Project/ruyifirmware/commit/ddd6b88afb)。

### 4.4 模型 & 技术｜[ggml : fix SpacemiT flash attention null mask, ALiBi and softcap（ggml : 修复 SpacemiT flash attention 的空 mask、ALiBi 和 softcap）](https://github.com/ggml-org/llama.cpp/pull/32182)

北京时间：2026-11-25 02:47:36｜来源类型：官方 GitHub PR｜事件状态：已合并。

**事实**：修复 K3 A100 核 VLEN=1024 flash attention 的三处问题：空 mask 解引用、跨 head 分组复用错误 ALiBi slope、先 padding 再 softcap 导致填充值重新参与 softmax。 作者在启用 GGML_CPU_RISCV64_SPACEMIT 的实机测试中，从 4 处失败并段错误恢复为 5721/5721 用例通过。

**重要性**：直接改善 K3 上大模型推理的正确性与可运行性。

**风险与限制**：范围限于相关 VLEN=1024 kernel；A100 在此指 K3 CPU 核，不是 NVIDIA GPU，测试计数也不是模型端到端评测。

### 4.5 模型 & 技术｜[fix(DCache): keep full client visibility on one memory channel（fix(DCache): 在单内存通道上保持完整的客户端可见性）](https://github.com/OpenXiangShan/XiangShan/pull/7035)

北京时间：2026-11-24 18:27:03｜来源类型：官方 GitHub PR｜事件状态：已合并。

**事实**：只有存在第二个 memory channel 时才分割客户端地址可见性，避免单客户端在 L1–L2 crossbar 前排除奇数 cache line。

**重要性**：修正香山单通道配置的缓存访问范围，已在 kunminghu-v3 产品分支核验。

**风险与限制**：这是指定缓存拓扑的正确性修复，不能推导所有芯片配置受影响或实测吞吐变化。

### 4.6 模型 & 技术｜[fix(ext): remove Ssube, Ssu32xl and Sstvecv（fix(ext): 移除 Ssube、Ssu32xl 和 Sstvecv）](https://github.com/riscv/riscv-unified-db/pull/2825)

北京时间：2026-11-25 04:42:21｜来源类型：官方 GitHub PR｜事件状态：已合并。

**事实**：UDB 移除误标为 ratified 的三个扩展名称；它们来自早期 profile 草稿与过期表格，未由正式 RISC-V 规范定义，相关行为仍由 U_MODE_ENDIANNESS、UXLEN、STVEC_MODES 参数表达。

**重要性**：纠正机器可读规范元数据，防止验证与支持矩阵继承不存在的已批准扩展。

**风险与限制**：删除名称不代表撤回现有硬件能力或改变正式 ISA。

### 4.7 模型 & 技术｜[Fix/issue 3463 decoder: enforce fixed-value fields for HLV/HLVX/HSV (rs2/rd) and CBO (rd)（Fix/issue 3463 解码器：强制 HLV/HLVX/HSV（rs2/rd）和 CBO（rd）的固定值字段）](https://github.com/openhwfoundation/cva6/pull/3756)

北京时间：2026-11-24 22:14:19｜来源类型：官方 GitHub PR｜事件状态：已合并。

**事实**：解码器对 HLV/HLVX 的保留 rs2、HLV.D 的固定 rs2，以及 HSV/CBO 固定为零的 rd 加入非法指令检查，避免保留编码被接受或产生 GPR 写回。

**重要性**：使 Hypervisor 和 cache-block 指令的非法编码行为与 ISA 表及参考模型对齐。

**风险与限制**：这次变更不是新增 Hypervisor 支持；仍需要按配置验证定向 trap 行为。


## 五、AI 业界重磅

### 5.1 模型 & 技术｜[GPT‑6 models demonstrate intrinsic protein-design capabilities（GPT‑6 模型展示出内在蛋白质设计能力）](https://openai.com/index/gpt-6-models-demonstrate-intrinsic-protein-design-capabilities)

北京时间：2026-11-24 21:00:00｜来源类型：官方博客｜事件状态：已发布。

**事实**：OpenAI 预印本报告 GPT‑6 系列可用推理和通用代码生成蛋白结构与序列；与 Onava 的实验中，基于 GPT‑6.2 结构设计的蛋白在五个目标上均有至少一个显示结合证据。 实验流水线仍使用 SolubleMPNN 生成候选序列，并以专用结构预测模型筛选；不能把内在设计能力解读为脱离专用模型的端到端零样本成功。

**重要性**：为通用模型参与分子设计提供了计算结果之外的湿实验检验。

**风险与限制**：预印本尚需独立复核；结合证据不是治疗效果，完整结构预测仍落后专用模型，部分基准可能受记忆影响。

### 5.2 模型 & 技术｜[We’re launching Humanity’s First Exam and a Kaggle hackathon to test LLM reliability.（我们正在推出 Humanity’s First Exam 和一场 Kaggle 黑客松，以测试 LLM 的可靠性。）](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/humanitys-first-exam/)

北京时间：2026-11-25 03:30:00｜来源类型：官方博客｜事件状态：已发布。

**事实**：Google 在 Kaggle 发布 Humanity’s First Exam，以 544 个基础图文 prompt 覆盖 20 项能力，关注简单任务及轻微变体能否被持续正确解决，而非单次能力上限。 同时开启截至 12 月 15 日的社区竞赛，征集当前模型仍会答错的日常问题。

**重要性**：把模型评估重点从偶尔解出难题补充到可重复完成基础任务。

**风险与限制**：这是特定数据集上的可靠性测量，不能替代真实工作流、长期运行或安全评测。

### 5.3 模型 & 技术｜[Searching for distant planets with Astra（使用 Astra 寻找遥远的行星）](https://openai.com/index/searching-for-distant-planets-with-astra)

北京时间：2026-11-24 19:00:00｜来源类型：官方博客｜事件状态：已发布。

**事实**：两个独立的业余天文搜索项目利用 Codex/Astra 分析公开望远镜数据；其中 Kepler 搜索给出的两个信号被 SETI Institute 天文学家独立找到，另有 TESS 候选等待进一步分析。

**重要性**：展示代码智能体如何把公开科学档案转化为可供专业人员复查的线索。

**风险与限制**：这些是潜在系外行星信号，尚非已确认发现；重复凌星、径向速度等进一步验证仍可能耗时数月或数年。

### 5.4 工具 & 产品｜[Simplifying agent access while keeping organizations in control: MCP servers, CLI, standardized usage tiers, and admin policies（在让组织保持控制的同时简化智能体访问：MCP 服务器、CLI、标准化用量层级和管理员策略）](https://workspaceupdates.googleblog.com/2026/11/mcp-servers-cli-standardized-usage-tiers-now-available.html)

北京时间：2026-11-25 01:02:59｜来源类型：官方博客｜事件状态：已发布。

**事实**：Google Workspace 宣布 MCP servers 与官方 CLI GA，覆盖 Gmail、Drive、Docs、Sheets、Slides、Calendar、Chat、Tasks，并配套管理员访问、人工确认和审计日志控制；GA 功能按两周节奏推出。 公告规定智能体访问 Workspace 数据须经 MCP，技术执行计划自 2027 年 2 月 1 日开始；Scaled tier 超过免费额度的计费计划自 4 月 1 日开始。

**重要性**：同时改变企业智能体的集成接口、权限治理和规模化使用成本。

**风险与限制**：未来执行与计费日期不是已经生效；具体额度、组织策略和逐项能力应按各产品文档确认，不能只据 GA 推断所有账户立即可用。

### 5.5 工具 & 产品｜[How to Run NVIDIA Cosmos 3 on Local AI Devices Using Hugging Face Diffusers（如何使用 Hugging Face Diffusers 在本地 AI 设备上运行 NVIDIA Cosmos 3）](https://developer.nvidia.com/blog/how-to-run-nvidia-cosmos-3-on-local-ai-devices-using-hugging-face-diffusers/)

北京时间：2026-11-25 02:44:05｜来源类型：官方博客｜事件状态：已发布。

**事实**：NVIDIA 给出 Cosmos 3 Nano 的 Diffusers/weight streaming 集成路径，模型权重留在系统内存、计算在 GPU，涵盖视频、声音、动作与控制条件生成。 12 GB RTX 5070、约 128 GB 主机内存的 720p 文生视频单次 BF16 测量为 2624.9 秒、峰值设备内存 9.94 GiB；原生混合 FP8 为 2235.7 秒、9.16 GiB。

**重要性**：公开可复现的低显存部署路径和完整等待时间，便于衡量本地物理 AI 模型的实际成本。

**风险与限制**：教程发布不等于 Cosmos 3 模型本日首发；结果为单次指定参数，Transfer 部分行关闭 guardrails，不能混作同等安全配置的 GPU 排名。

### 5.6 工具 & 产品｜[Microduck Obstacle Avoidance: A Sim2Real Workflow with Isaac Lab and Newton（Microduck 障碍物规避：使用 Isaac Lab 和 Newton 的 Sim2Real 工作流）](https://huggingface.co/blog/nvidia/microduck-obstacle-avoidance)

北京时间：2026-11-25 00:34:13｜来源类型：官方博客｜事件状态：已发布。

**事实**：NVIDIA 展示冻结原有步行策略、单独训练避障控制器的方案：以 Newton 模拟 8×8 ToF、15 Hz 深度采样，控制器以 50 Hz 调整速度指令，之后导出 ONNX 并连到真实 Microduck。

**重要性**：给出从仿真传感器随机化到真实机器人监督测试的部署链路。

**风险与限制**：导航模型在笔记本执行后把指令发给机器人；不能说已全量在机器人板端运行，也不是经过普遍场景验证的自主导航产品。

## 六、总结与趋势观察

- **部署路径正把模型格式、内存和硬件后端一起处理。** [ExecuTorch–TensorRT](https://pytorch.org/blog/executorch-tensorrt-pytorch-robot-policies-at-nvidia-tensorrt-speed/) 将导出与轻量运行时接通，[Cosmos 3 本地部署](https://developer.nvidia.com/blog/how-to-run-nvidia-cosmos-3-on-local-ai-devices-using-hugging-face-diffusers/) 用系统内存换取较低显存占用；能运行与能实时运行仍是两项不同指标。

- **编译器正确性开始更明确地覆盖模型语义与硬件边界。** [torch-mlir 索引检查](https://github.com/llvm/torch-mlir/pull/4912) 补上越界行为，[K3 flash attention](https://github.com/ggml-org/llama.cpp/pull/32182) 修正 mask 和数值处理，[认证规则映射](https://github.com/riscv/riscv-arch-test/pull/3298) 则明确测试仍未覆盖的规范条款。

- **局部性能优化需要与模型级收益分开评估。** [Buddy VIR 调度](https://github.com/buddy-compiler/buddy-mlir/pull/1079) 在 MoE combine 微基准改善寄存器压力，[Intel dot_scaled](https://github.com/intel/intel-xpu-backend-for-triton/pull/8958) 改善预取和 scale 布局；两者都披露了尚未消除的限制。

## 附录：信源说明

主要来源为 PyTorch、Google、OpenAI、NVIDIA 官方博客，LLVM Discourse，以及 Buddy Compiler、torch-mlir、Triton、TileLang、OpenSBI、RISC-V 验证与国内软件栈的上游记录。GitHub 变更表示对应主干或分支状态，不等于已经进入所有稳定发行版；论坛内容按提案或具体回复的状态解释。厂商性能和实验结果均沿用原文的条件与适用边界。
