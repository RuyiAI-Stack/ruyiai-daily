# Codex 技术情报每日动态（2026-11-24）

调研窗口：北京时间（2026-11-23 04:35:30.649，2026-11-24 17:28:23]。

覆盖方向：PyTorch、LLVM/MLIR、Triton & TileLang、RISC-V，以及 AI 模型研究、软件基础设施、芯片与产业。

信息口径：依据原始公告、正式工件、研究报告、具体技术回复和项目变更，区分设计提案、beta、主干实现与正式交付；测试和性能结论保留原始条件。

## 今日要闻

- [OpenAI 与 Cisco 开源 Shirudo，在永久补丁部署前提供经批准的 Linux 临时保护；受控案例阻止利用且无需重启。](https://openai.com/index/introducing-shirudo)
- [RISE 公开索引已列出 Triton-RISC-V 的 triton 3.9.0+gitbbe38e57 riscv64 wheel，与 Release 摘要一致。](https://github.com/riseproject-dev/python-wheels/releases/tag/triton-v3.9.0%2Bgitbbe38e57-20261124082151)
- [Google 提出 Universal CFI，探索让无需 LTO 的大型 C/C++ 软件更容易启用控制流加固。](https://discourse.llvm.org/t/92955/1)
- [NVIDIA 董事会恢复黄仁勋的总裁、CEO 和董事职务，独立调查与合规治理工作继续。](https://nvidianews.nvidia.com/news/nvidia-board-reinstates-jensen-huang-as-president-ceo-and-board-director)
- [Ai2 发布 RoboTrack 基准、仿真轨迹数据与模型，检验现有跟踪器在真实机器人操作中的表现。](https://huggingface.co/blog/allenai/robotrack)

## 今日索引

- **一、PyTorch 生态核心动态**：FSDP2 布局契约、ExecuTorch 严格 FP32 线性层、导出与 MPS 正确性、FP8 类型及主机 TMA。
- **二、LLVM/MLIR 最新进展**：通用 CFI、x86 自动分派、线性资源类型、浏览器编译与浮点语义，以及 Buddy 发行和模型导入。
- **三、Triton & TileLang 技术动态**：RISE wheel 分发、TritonNext 征集、跨 CTA scan、进程并行编译、gfx1200 布局与 RISC-V 线性代数。
- **四、RISC-V 核心新闻**：Zvfc 草案、K1 时间戳、VF2 系统启动、NEMU 发行、固件测试、UEFI 变量、机器人 IMU 和浮点验证。
- **五、AI 业界重磅**：Shirudo、RoboTrack、智能体评测、Sierra 融资、Claude Ship、Gemini 文件切片与 NVIDIA 人事。

## 一、PyTorch 生态核心动态

### 1.1 模型 & 技术｜[\[FSDP2\] Add an all-gather layout contract with FSDP-owned buffers (#199726)（\[FSDP2\] 添加采用由 FSDP 持有的缓冲区的 all-gather 布局契约 (#199726)）](https://github.com/pytorch/pytorch/commit/25acc7329a9d9266fc9dfcb2dcf218fdf9ad7cab)

北京时间：2026-11-24 06:51:39｜来源类型：官方主干提交｜事件状态：已进入主干

**事实**：FSDP2 引入 AllGatherLayout 的 prepare、copy_in 与 finalize_outputs 等接口，由 FSDP 持有并在 reshard 时释放输出缓冲区，支持自定义打包和输出布局。 私有 AllGatherResult 和 collective 签名发生兼容变化，旧 _all_gather_output_fn hook 被 set_all_gather_layout 替代；后端自行持有输出及零拷贝并未在这一层实现。

**重要性**：为 MoE、分片训练和自定义通信后端提供明确的存储生命周期边界。

**风险与限制**：作者说明最新通信 setter 调整尚未重跑全部 GPU 测试；现有私有 hook 用户需要迁移。

相关原文：[\[FSDP2\] Add an all-gather layout contract with FSDP-owned buffers](https://github.com/pytorch/pytorch/pull/199726)。

### 1.2 模型 & 技术｜[\[executorch\]\[cuda\] Add triton::linear, a strict FP32 linear op, and plan the GEMM forms for it（\[executorch\]\[cuda\] 添加严格 FP32 线性算子 triton::linear，并为其规划 GEMM 形式）](https://github.com/pytorch/executorch/pull/25610)

北京时间：2026-11-24 03:59:17｜来源类型：官方 GitHub PR｜事件状态：已合并

**事实**：ExecuTorch CUDA 增加读取原始 [N,K] 权重、按记录步幅写输出的 triton::linear，并识别 linear、matmul、mm 和 addmm 图形。关联变更加入少行 GEMV、tf32x3 和按固定次序归并的 split-K 调度。 编译时可测量候选与 Inductor 的表现；不适用的数据类型、计算生成的权重、批量乘法和布局继续使用常规 lowering。

**重要性**：把严格 FP32 线性层接入导出与编译期算子选择路径，覆盖解码和较长 K 的不同形状。

**风险与限制**：tf32x3 需要 sm_80 或更新硬件；这是精度门槛内的调度族，不能表述为所有形状均加速。

相关原文：[\[executorch\]\[cuda\] Add triton::linear, a strict FP32 linear op, and plan the GEMM forms for it](https://github.com/pytorch/executorch/pull/25384)；[\[executorch\]\[cuda\] Add a matrix-vector schedule to triton::linear and route GEMMs by measurement](https://github.com/pytorch/executorch/pull/25385)；[\[executorch\]\[cuda\] Add tf32x3 and deterministic split-K tile schedules to triton::linear](https://github.com/pytorch/executorch/pull/25386)；[\[executorch\]\[cuda\] Add a matrix-vector schedule to triton::linear and route GEMMs by measurement](https://github.com/pytorch/executorch/pull/25611)；[\[executorch\]\[cuda\] Add tf32x3 and deterministic split-K tile schedules to triton::linear](https://github.com/pytorch/executorch/pull/25612)；[\[executorch\]\[cuda\] Route strict FP32 GEMMs of 5 to 64 rows to triton::linear without measurement](https://github.com/pytorch/executorch/pull/25614)。

### 1.3 模型 & 技术｜[\[export\] Save and load a bool or symbolic result of a custom op next to a tensor (#205628)（\[export\] 保存并加载自定义算子中与张量并列的布尔或符号结果 (#205628)）](https://github.com/pytorch/pytorch/commit/de46fbb8532d712ea79ad3826b19715e8e494892)

北京时间：2026-11-24 09:35:00｜来源类型：官方主干提交｜事件状态：已进入主干

**事实**：torch.export 的保存/加载补齐自定义算子 (Tensor,bool)、(Tensor,SymInt)、(Tensor,SymBool) 和 (Tensor,Scalar) 等多结果路径，并覆盖动态形状区域产生的符号布尔值。 修正反序列化的 as_sym_bool 处理和序列化时 torch 符号类型与同名 schema 类型混淆；动态序列长度下 Flash Attention 的符号整数结果也受益。

**重要性**：减少模型导出到后续编译器或部署系统时，图可生成但工件无法往返加载的障碍。

**风险与限制**：验证范围为所列 CPU 序列化测试；不意味着所有自定义算子和下游后端都受支持。

相关原文：[\[export\] Save and load a bool or symbolic result of a custom op next to a tensor](https://github.com/pytorch/pytorch/pull/205628)。

### 1.4 模型 & 技术｜[\[MPS\] Fix silently wrong depthwise conv2d output for large filters (#206266)（\[MPS\] 修复大滤波器下静默错误的 depthwise conv2d 输出 (#206266)）](https://github.com/pytorch/pytorch/commit/a4742da251b6429f96b5a016fdfc893cbb1deb0f)

北京时间：2026-11-24 16:48:40｜来源类型：官方主干提交｜事件状态：已进入主干

**事实**：MPSGraph 的 depthwise conv2d 在线程组输入块超过 64 KiB 时可能静默产生错误输出，现改走已有 Metal conv2d 内核；M1/M2 在恰好 64 KiB 时也纳入回退。 作者在 M1 Pro、M2、M3 Ultra、M4 和 M5 Pro 上与 CPU 对照验证，并说明此前观察到的错误梯度源于前向输出。

**重要性**：修复会传播到模型结果和梯度的静默正确性问题。

**风险与限制**：触发条件与过滤器、步幅、数据类型和设备代际有关；主干修复不等于已进入已发布稳定版。

相关原文：[\[MPS\] Fix silently wrong depthwise conv2d output for large filters](https://github.com/pytorch/pytorch/pull/206266)。

### 1.5 模型 & 技术｜[\[MPS\]\[Inductor\] Support float8_e4m3fn and float8_e8m0fnu (#206056)（\[MPS\]\[Inductor\] 支持 float8_e4m3fn 和 float8_e8m0fnu (#206056)）](https://github.com/pytorch/pytorch/commit/990e1ce56708ce777f2294c04f77e4c28c7ea900)

北京时间：2026-11-24 08:15:11｜来源类型：官方主干提交｜事件状态：已进入主干

**事实**：Inductor 的 Metal 类型映射加入 float8_e4m3fn 和 float8_e8m0fnu，使用现有 c10::metal FP8 结构存储并以 float32 计算，消除触及 FP8 时的 KeyError。 常量地址空间加载和 bitcast 分别通过专用转换/编码路径处理；测试仍跳过 eager MPS 不支持的 e5m2 和 _scaled_mm v1。

**重要性**：补齐块缩放和 FP8 图在 Apple GPU 编译路径上的类型基础。

**风险与限制**：这是类型和代码生成支持，不是所有 FP8 运算或量化模型均可运行的承诺。

相关原文：[\[MPS\]\[Inductor\] Support float8_e4m3fn and float8_e8m0fnu](https://github.com/pytorch/pytorch/pull/206056)。

### 1.6 模型 & 技术｜[\[Inductor\]\[Host-side TMA\] Build Triton GEMM epilogue descriptors on host (#194414)（\[Inductor\]\[Host-side TMA\] 在主机端构建 Triton GEMM epilogue 描述符 (#194414)）](https://github.com/pytorch/pytorch/commit/69197b1962aaa286767d9ab30cafedd4ff810d82)

北京时间：2026-11-24 05:55:27｜来源类型：官方主干提交｜事件状态：已进入主干

**事实**：Inductor 将可提升的 GEMM epilogue 载入和模板输出的 TMA 描述符移到主机端构建，复用 prologue_cache 中已有的 tile 表达式。

**重要性**：让融合尾处理与主机描述符路径衔接，改善 Triton GEMM 的编译集成方式。

**风险与限制**：动态尺寸的模板 epilogue 仍保留内核端描述符；同一缓冲区需要两种形状描述符时也回退，没有公开端到端收益数字。

相关原文：[\[Inductor\]\[Host-side TMA\] Build Triton GEMM epilogue descriptors on host](https://github.com/pytorch/pytorch/pull/194414)。


## 二、LLVM/MLIR 最新进展

### 2.1 模型 & 技术｜[\[RFC\] Universal CFI（\[RFC\] 通用 CFI）](https://discourse.llvm.org/t/92955/1)

北京时间：2026-11-24 10:23:13｜来源类型：项目社区／RFC｜事件状态：RFC 已公开

**事实**：Google 提议以 kCFI 的哈希校验模型构建暂名 uCFI 的通用控制流完整性模式，目标是在无需 LTO、并与未插桩共享库混合的 C/C++ 软件中更容易部署前向边保护。 方案覆盖 C++ 虚函数/vtable、缩小合法目标集合的注解、检查优化，并探索缺少 PAC/CET 时的后向边保护；本期回复进一步讨论类型哈希和成员函数兼容边界。

**重要性**：把编译器加固重点推进到现有大型软件的部署成本与混合二进制兼容性。

**风险与限制**：仍是 RFC；不替代内存安全，也不改变现有 Linux kCFI 不变量，GNU 特定成员函数扩展等仍有取舍。

相关原文：[\[RFC\] Universal CFI](https://discourse.llvm.org/t/92955/4)；[\[RFC\] Universal CFI](https://discourse.llvm.org/t/92955/6)。

### 2.2 模型 & 技术｜[\[RFC\] Compiler-driven CPU dispatch for x86: -fauto-target-clones=（\[RFC\] 由编译器驱动的 x86 CPU 分派：-fauto-target-clones=）](https://discourse.llvm.org/t/92960/1)

北京时间：2026-11-24 16:07:04｜来源类型：项目社区／RFC｜事件状态：RFC／原型

**事实**：提案让 Clang 自动选择值得特化的函数，为基线与请求的 ISA/CPU 生成多个版本并加入运行时分派，目标是在维持旧 CPU 可运行性的同时利用新指令。 作者原型在 Nova Lake 上测量九个应用，平均性能距 -march=novalake 构建约 0.7%；zstd 压缩相对 x86-64 基线提升 12.6%，主要代价是二进制变大。

**重要性**：为发行版与通用软件提供不逐函数标注的多版本优化方向。

**风险与限制**：结果限于作者的原型、应用和设备；尚未形成上游可用选项。

### 2.3 模型 & 技术｜[\[RFC\] Linear Type System Support in Upstream MLIR（\[RFC\] 在上游 MLIR 中支持线性类型系统）](https://discourse.llvm.org/t/92947/1)

北京时间：2026-11-23 21:13:28｜来源类型：项目社区／RFC｜事件状态：RFC 已公开

**事实**：提案为 MLIR 增加标记线性资源的 trait，以及描述输入消耗和输出产生关系的接口，减少量子编译器对 move-only 资源语义的重复实现。 新讨论指出简单 SSA 使用次数不足以处理控制流；作者进一步澄清资源应显式移动、禁止隐式捕获，MVP 优先提供可供 dialect 构建的最小机制。

**重要性**：使资源所有权与控制流验证成为可复用 IR 基础设施的讨论对象。

**风险与限制**：复合类型、循环携带值、函数返回和验证器边界仍待确定，并未建立通用上游线性类型系统。

相关原文：[\[RFC\] Linear Type System Support in Upstream MLIR](https://discourse.llvm.org/t/92947/4)；[\[RFC\] Linear Type System Support in Upstream MLIR](https://discourse.llvm.org/t/92947/5)。

### 2.4 模型 & 技术｜[\[RFC\] Bringing interactive MLIR compilation and execution to WebAssembly（\[RFC\] 将交互式 MLIR 编译和执行带到 WebAssembly）](https://discourse.llvm.org/t/91566/5)

北京时间：2026-11-24 03:50:22｜来源类型：项目社区／RFC｜事件状态：讨论新进展

**事实**：该 RFC 的主线是让 MLIR、LLVM/LLD 与 Python/C/C++ 内核在 JupyterLite 浏览器环境中完成构建、lowering、链接和执行。 讨论新进展：作者公布 wasm64 演示，将 PyTorch、JAX、XLA 与 IREE 的编译、训练和预训练模型工作流带进浏览器，CPU 走 WebAssembly，GPU 走 WebGPU；另提出复用 emscripten-forge 的浏览器测试基础设施。

**重要性**：与 torch-mlir、IREE、Turbine 和模型导入链路直接相关，提供可交互检查编译阶段的新部署形态。

**风险与限制**：当前是打包栈和演示，不代表相关执行补丁均已上游合入；wasm64 的配置避开了此前特定 wasm32 MemRef 描述符问题。

相关原文：[\[RFC\] Bringing interactive MLIR compilation and execution to WebAssembly](https://discourse.llvm.org/t/91566/6)。

### 2.5 模型 & 技术｜[\[RFC\]\[mlir\]\[Linalg\]\[Vector\] Fast-math flags on Linalg and Vector ops（\[RFC\]\[mlir\]\[Linalg\]\[Vector\] Linalg 和 Vector 操作上的 fast-math 标志）](https://discourse.llvm.org/t/92764/20)

北京时间：2026-11-24 02:56:51｜来源类型：项目社区／RFC｜事件状态：讨论新进展

**事实**：原提案试图统一 Linalg/Vector 的浮点标志、重写传播规则以及归约重排和 FMA 收缩语义。 讨论新进展：本期把问题推进到 generic 与 named op 的特化/泛化边界；回复主张缺少 contract/reassoc 的 generic 不应自动特化为默认允许这些行为的 named op，并明确向量化后的标志传播责任。

**重要性**：影响 PyTorch/JAX 图经 Linalg 到 Vector/LLVM 的数值契约，值得编译链路保持关注。

**风险与限制**：参与者仍有设计分歧，尚不能把这一回复写成已经达成的方言规范。

### 2.6 工具 & 产品｜[BUDDY MLIR 0.1.8（BUDDY MLIR 0.1.8）](https://github.com/buddy-compiler/buddy-mlir/releases/tag/release/v0.1.8)

北京时间：2026-11-24 10:58:10｜来源类型：官方 Release｜事件状态：正式发行

**事实**：Buddy 正式发布 0.1.8，提供 x86_64 wheel、riscv64 CLI，以及 x86_64/riscv64 的 LLVM 工具链压缩包等工件。 该版本所含 K3 A100 的 Qwen3-0.6B FP16 优化复用 IME GEMM/GEMV、Decode 权重打包和 RVV/SiLU 路径。作者在八个物理核、固定输入下报告 Prefill 86.01 token/s、Decode 13.45 token/s；同配置 llama.cpp 为 71.64 和 14.38。

**重要性**：同时交付工具链工件和可复现的模型执行路径，直接关联 Buddy 与 RISC-V 部署。

**风险与限制**：测量为 128 位置 Prefill、从位置96开始的32步 teacher-forced Decode，排除加载；两框架线程组织不同，Decode 未超过比较对象，完整 check-buddy 受 Python 构建目标限制。

相关原文：[\[midend\]\[models\] Optimize Qwen3-0.6B FP16 Prefill and Decode on K3 A100](https://github.com/buddy-compiler/buddy-mlir/pull/1001)。

### 2.7 模型 & 技术｜[\[midend\] Approximate math ops in vector code and shorten VIR loop tails（\[midend\] 对向量代码中的数学操作使用近似，并缩短 VIR 循环尾部）](https://github.com/buddy-compiler/buddy-mlir/pull/1068)

北京时间：2026-11-24 11:42:13｜来源类型：官方 GitHub PR｜事件状态：已合并

**事实**：Buddy 新增可选多项式数学近似 pass，使 RVV 路径中的 exp 等操作避免逐 lane 调 libm；同时修正 log/log2、sin/cos、atan2 和 cbrt 的若干边界行为。 VIR 对能装入一个寄存器组的常量范围采用单个固定向量，并提供默认关闭的 active-length 尾部循环，保留固定主循环，避免全循环 AVL 方案的回退。

**重要性**：直接改善 Triton-RISC-V 经 Buddy 降到 RVV 的算术和短内核路径。

**风险与限制**：近似并非逐位等价；部分上游 exp/erfc 次正规结果仍可能冲零。性能数据是 VLEN-256、单核内核时间，不能外推整模型或所有 VLEN。

### 2.8 模型 & 技术｜[\[Torch\]\[TorchOnnxToTorch\]\[TorchToLinalg\] Fix and extend the RotaryEmbedding lowering（\[Torch\]\[TorchOnnxToTorch\]\[TorchToLinalg\] 修复并扩展 RotaryEmbedding lowering）](https://github.com/llvm/torch-mlir/pull/4884)

北京时间：2026-11-23 15:53:11｜来源类型：官方 GitHub PR｜事件状态：已合并

**事实**：修正三维 RotaryEmbedding 输入的维度解释，支持 interleaved、部分 rotary_embedding_dim 和单个 position_ids 偏移，并让 GroupQueryAttention 传递旋转维度与头数。

**重要性**：扩大 ONNX 模型经 Torch dialect 到 Linalg 的正确导入范围，避免注意力位置编码被错误展开。

**风险与限制**：奇数旋转维度、缓存宽度不匹配及非法 interleaved 仍显式拒绝；scale 行为与 ONNX Runtime 对齐，不是新增通用缩放语义。

### 2.9 模型 & 技术｜[\[TorchOnnxToTorch\] Detect GroupQueryAttention packed QKV by absent key/value（\[TorchOnnxToTorch\] 根据缺省的 key/value 检测 GroupQueryAttention 打包 QKV）](https://github.com/llvm/torch-mlir/pull/4895)

北京时间：2026-11-23 20:48:34｜来源类型：官方 GitHub PR｜事件状态：已合并

**事实**：GQA lowering 改为按 key/value 是否同时缺省识别 packed QKV，不再按操作数数量猜测，覆盖 onnxruntime-genai model builder 的12输入布局和尾随 none。 不支持的非空尾随输入、causal=0 与量化 KV 缓存改为拒绝匹配，避免此前静默按 causal 浮点注意力处理。

**重要性**：为模型导入增加明确兼容边界，并修复实际导出器格式变化导致的识别问题。

**风险与限制**：这些拒绝条件不表示量化 KV 或非 causal GQA 已支持；TOSA 路径仍有零维张量与 SDPA legalization 限制。


## 三、Triton & TileLang 技术动态

### 3.1 工具 & 产品｜[triton-v3.9.0+gitbbe38e57（triton-v3.9.0+gitbbe38e57）](https://github.com/riseproject-dev/python-wheels/releases/tag/triton-v3.9.0%2Bgitbbe38e57-20261124082151)

北京时间：2026-11-24 16:22:14｜来源类型：官方 Release｜事件状态：正式发行

**事实**：RISE Python 包索引已经列出 triton-3.9.0+gitbbe38e57-cp312-abi3-manylinux_2_39_riscv64.whl；对应官方 Release 资产的 SHA-256 与索引一致。 包元数据指向 RuyiAI-Stack/triton-riscv：通过 triton-shared 的 TTIR→Linalg、Buddy 的 Linalg→LLVM-MLIR，再交给 LLVM llc；wheel 随附所用工具链。

**重要性**：Triton-RISC-V 经 RISE 分发已有公开可下载工件和索引记录，部署链路从计划推进到可获取的包。

**风险与限制**：标签为 CPython 3.12 ABI3、manylinux_2_39_riscv64；包索引与工件可获取不等于目标硬件安装和性能已验证，也不能据此断言上游 Triton 原生支持 RISC-V。

相关原文：[Links for triton](https://pypi.riseproject.dev/simple/triton/)；[docs: update triton](https://github.com/riseproject-dev/python-wheels/commit/0ea62fb1cf76f5a84fc2fb529ebab73f657f5f4f)。

### 3.2 行业 & 人事｜[TritonNext 2027 (FlagOS Community): call for talks, posters and workshop proposals — Beijing, Jan 22–24（TritonNext 2027（FlagOS Community）：征集演讲、海报和工作坊提案——北京，1月22–24日）](https://github.com/triton-lang/triton/discussions/12919)

北京时间：2026-11-23 17:10:56｜来源类型：项目社区／RFC｜事件状态：征集已公布

**事实**：FlagOS Community 发布 TritonNext 2027 征集，面向 AI 编程、内核语言、编译器和跨架构优化；会议计划于2027年1月22–24日在北京举行。 演讲/海报提交截止11月30日23:59（北京时间）；工作坊与实践教程组织提案截止12月6日，会议方提供场地、设备和计算资源。

**重要性**：为 Triton、MLIR 和异构内核团队提供公开交流与贡献入口。

**风险与限制**：这是征集公告；不代表演讲已录用或研究已发表，公告说明接受不意味着归档出版。

### 3.3 模型 & 技术｜[\[GPU\] Route scan carries by logical segment and support scans across CTAs（\[GPU\] 按逻辑分段路由 scan 进位，并支持跨 CTA 的 scan）](https://github.com/triton-lang/triton/pull/12924)

北京时间：2026-11-24 06:08:08｜来源类型：官方 GitHub PR｜事件状态：已合并

**事实**：Triton scan lowering 按逻辑 segment 顺序选择前缀进位，处理 CTA rank 与扫描次序不同的布局，并接入跨 CTA scan。 分段总量经 distributed shared memory 交换，scratch 访问被标记为跨 CTA，使复用存储前执行正确同步；同时缩短寄存器进位的生命周期。

**重要性**：扩展并行前缀计算的布局和多 CTA 表达能力。

**风险与限制**：依赖后端对 CTA 集群、分布式共享内存与同步的支持，没有证明所有目标或扫描形状均有性能收益。

### 3.4 模型 & 技术｜[\[JIT\] Add process-based parallel compilation（\[JIT\] 添加基于进程的并行编译）](https://github.com/tile-ai/tilelang/pull/3903)

北京时间：2026-11-23 11:57:26｜来源类型：官方 GitHub PR｜事件状态：已合并

**事实**：TileLang 的 par_compile 增加可选 executor="process"，由父进程处理命中与去重，spawn worker 编译唯一缓存缺失项，并从已发布磁盘缓存直接载入结果。

**重要性**：为 Python 工作占比较高的批量内核编译提供绕开 GIL 的路径。

**风险与限制**：线程模式仍是默认；进程模式要求开启缓存、受保护的脚本入口，且拒绝不支持磁盘持久化的后端。作者在 Ascend 的64进程、128种配置上验证，未给出普遍加速比。

### 3.5 模型 & 技术｜[\[ROCm\] Support linear shared layouts for gfx1200 FP16/BF16 WMMA（\[ROCm\] 为 gfx1200 FP16/BF16 WMMA 支持线性共享布局）](https://github.com/tile-ai/tilelang/pull/3729)

北京时间：2026-11-24 11:45:09｜来源类型：官方 GitHub PR｜事件状态：已合并

**事实**：gfx1200 的 FP16/BF16 T.gemm 允许 A/B 保留 canonical linear shared-memory 布局；兼容检查获得操作、缓冲区、实际布局与 target 上下文，复用现有 WMMA 发射路径。

**重要性**：减少 ROCm 内核显式布局与 GEMM 推断冲突，为共享内存布局复用提供后端明确契约。

**风险与限制**：限 NN shared/shared、同类FP16或BF16输入、FP32累加、k_pack=1和满足对齐的连续2D存储；不代表其他架构或转置组合已放宽。

### 3.6 模型 & 技术｜[Add FlagGems linear algebra operators for RISC-V.（为 RISC-V 添加 FlagGems 线性代数算子。）](https://github.com/RuyiAI-Stack/triton-riscv/commit/8e3c8aa4eb50d0ea100caa10b0af4628720b8102)

北京时间：2026-11-24 11:12:09｜来源类型：官方主干提交｜事件状态：已进入主干

**事实**：Triton-RISC-V 的 FlagGems 示例加入 Cholesky、LU、QR、行列式、求逆与线性求解等算子及包装；测试使用矩阵乘法恒等式重建结果，避免依赖 LAPACK。

**重要性**：扩展 RISC-V CPU 后端可以演练和验证的线性代数范围，与 Buddy/MLIR lowering 链路直接相关。

**风险与限制**：实现仍在 examples 层；部分路径使用 float32 工作副本或稠密化，不能视为完整 LAPACK 替代或可伸缩高性能库。


## 四、RISC-V 核心新闻

### 4.1 模型 & 技术｜[Add Zvfc (Vector Floating-Point Complex Arithmetic) extension proposal（添加 Zvfc（向量浮点复数算术）扩展提案）](https://github.com/XUANTIE-RV/zvfx-proposal/commit/d864aa41aa83656c58e0d69518b72aa84fa7c6f0)

北京时间：2026-11-23 14:06:05｜来源类型：官方主干提交｜事件状态：扩展草案 v0.1.0

**事实**：玄铁公开 Zvfc v0.1.0 development 提案的机器可读扩展与指令描述，复数以相邻实部/虚部元素表示，提供向量向量和复数标量广播形式。 草案明确 vl 计数标量元素、odd vstart 保留、掩码与尾部规则，以及复数乘累加分量的融合舍入语义。

**重要性**：为复数数值计算提供可审阅的 RVV 指令语义和验证素材。

**风险与限制**：这是发展中的扩展提案，不是已批准 ISA，也不是宣布现有芯片已实现。

### 4.2 模型 & 技术｜[Running ptp4l with hardware timestamping on Orange Pi RV2 K1 SOC?（在 Orange Pi RV2 K1 SoC 上使用硬件时间戳运行 ptp4l？）](https://forum.spacemit.com/t/1793/3)

北京时间：2026-11-24 04:02:07｜来源类型：项目社区／RFC｜事件状态：讨论新进展

**事实**：讨论新进展：用户确认让主时钟发送 PTP 2.0（--ptp_minor_version 0）后，RV2 的接收硬件时间戳和同步可以正常工作；此前建议指出部分 linuxptp 默认的 PTP 2.1 版本字节不被 K1 识别。 后续回复补充 RV2 作为主时钟时，测试中可工作的最小 tx_timestamp_timeout 为50，而原帖使用100。

**重要性**：把“网卡声称支持时间戳但 PTP 无法同步”的问题收敛到可复现的协议版本兼容条件。

**风险与限制**：这是用户在特定 Armbian/Orange Pi 环境和网络上的验证，不能外推所有 K1 驱动或交换网络；原建议早于窗口，本期新增的是成功验证。

相关原文：[Running ptp4l with hardware timestamping on Orange Pi RV2 K1 SOC?](https://forum.spacemit.com/t/1793/4)。

### 4.3 工具 & 产品｜[Running LiberNT successfully on VF2（在 VF2 上成功运行 LiberNT）](https://forum.rvspace.org/t/6575/1)

北京时间：2026-11-23 11:15:55｜来源类型：项目社区／RFC｜事件状态：社区实机启动报告

**事实**：用户在 VisionFive 2 上用串口辅助、通过 GRUB chainload bootriscv64.efi 启动 LiberNT bootCD，并展示内核载入和 HDMI 图形输出。 同日补充预安装镜像的 MBR 分区表不被其 Debian GRUB 识别，改为 GPT 后可手动进入图形界面。

**重要性**：提供 RISC-V 实机运行 NT 类系统的具体启动路径与兼容问题记录。

**风险与限制**：仅为用户实机启动报告；LiberNT 以 Windows 11 兼容为目标，不意味着已达到应用或驱动兼容。

相关原文：[Running LiberNT successfully on VF2](https://forum.rvspace.org/t/6575/2)。

### 4.4 工具 & 产品｜[Release v2026.11.r10（发行版 v2026.11.r10）](https://github.com/OpenXiangShan/NEMU/releases/tag/v2026.11.r10)

北京时间：2026-11-23 21:54:16｜来源类型：官方 Release｜事件状态：正式发行

**事实**：NEMU 发布11月第10个正式版本，提供 xs、xs-dual、xs-kmhv2 与 xs-kmhv2-dual 的 ref/ref-debug 共享对象；变更修正浮点 MMA 的 AMU 控制事件所报告的 mfrm，并把 checkpoint 的 Zstandard 压缩级别调到9。

**重要性**：为香山差分测试和矩阵扩展相关验证提供明确版本工件。

**风险与限制**：发布工件的编译平台是 Ubuntu 20.04（声明兼容 CentOS 7）；压缩级别改变不代表模拟速度提升，也不能据模拟器行为推断硬件性能。

相关原文：[fix(ame): report mfrm for floating MMA AMU control events](https://github.com/OpenXiangShan/NEMU/pull/1311)。

### 4.5 工具 & 产品｜[docs: add agent-assisted RAM boot testing guide and skill（docs: 添加 agent 辅助 RAM 启动测试指南和 skill）](https://github.com/YuzukiHD/YuzukiNeko/commit/c432c0d7c12941eb1da375224a35135772d11abf)

北京时间：2026-11-23 18:55:15｜来源类型：官方主干提交｜事件状态：技术指南已公开

**事实**：YuzukiNeko 新公开 agent 辅助固件测试流程：构建后检查 ELF 入口、按 F101S3/S4 选择 PSRAM helper，经 FEL 写入 RAM、回读核对 SHA-256，再从 UART 收集运行证据。

**重要性**：为 Zephyr、RT-Thread 和 ESP-IDF 提供统一的可重复集成测试路径，减少固件迭代依赖人工刷写。

**风险与限制**：新增的是使用与自动化指南；流程仅适用于 RAM 镜像，不写 SPI NOR，也不代表发布了新硬件能力。

### 4.6 模型 & 技术｜[sg2042: Use SBI for persistent runtime variables（sg2042: 通过 SBI 使用持久化运行时变量）](https://github.com/openRuyi-Project/ruyifirmware/commit/9018f194346371fa3cc83f6bb40c3849cd89bd8d)

北京时间：2026-11-23 22:19:03｜来源类型：官方主干提交｜事件状态：已进入主干

**事实**：Ruyi Firmware 将 SG2042 的 SPI1 变量分区服务与 UEFI 客户端配对：M-mode 对限定变量分区执行读、编程和擦除，通过寄存器传数据，避免运行时依赖 S-mode 的高地址控制器映射。

**重要性**：改善 RISC-V UEFI 运行时变量存储与操作系统地址映射之间的衔接。

**风险与限制**：这是实验性 SBI 扩展和非安全变量存储，不等于 Secure Boot 或认证变量保障；仍需保持对应闪存时钟和分区布局。

相关原文：[platform: sophgo: Access the SG2042 variable partition from M-mode](https://github.com/openRuyi-Project/opensbi/commit/49a815995d64f90f1b7108f206b24ce4cdd6d93e)；[SophgoPlatformPkg: Keep SG2042 runtime variables independent of high MMIO](https://github.com/openRuyi-Project/edk2-ruyi/compare/c7d1542a87a5c910d2b7ce551ffc68e0a9f105da...d5b59a99e86b89c4494fb0cf2acee9164608de0d)。

### 4.7 模型 & 技术｜[feat(humanoid): preprocess camera IMU for G1 RGB-D SLAM（feat(humanoid): 为 G1 RGB-D SLAM 预处理相机 IMU）](https://github.com/spacemit-robotics/ros2_humanoid/commit/85ea3b49d4692c8a5c0e51b7ba4e28a5d749c60a)

北京时间：2026-11-24 15:21:20｜来源类型：官方主干提交｜事件状态：已进入主干

**事实**：G1 的 RGB-D SLAM launch 增加带自适应增益与陀螺零偏估计的互补滤波，将相机原始 IMU 转为姿态后交给 RTAB-Map，并匹配 RealSense 的 best-effort QoS。 use_imu=false 同时关闭滤波和重力约束；SLAM 模式启动会清空指定数据库，定位模式则保留并加载。

**重要性**：补齐 RISC-V 机器人部署中传感器、姿态约束与建图之间的集成环节。

**风险与限制**：仍需外部里程计、有效 TF 与对齐的深度图；这一 launch 不提供完整自主导航系统，数据库启动行为也需要部署侧留意。

### 4.8 模型 & 技术｜[Run TestFloat and cover-float vectors on Verilator; check every vector (#2103)（在 Verilator 上运行 TestFloat 和 cover-float 向量；检查每个向量 (#2103)）](https://github.com/openhwfoundation/cvw/commit/90a467cb10bf68624df77595e84524228088e388)

北京时间：2026-11-24 02:12:21｜来源类型：官方主干提交｜事件状态：已进入主干

**事实**：Wally 的浮点 testbench 重写向量驱动时序，使 TestFloat 能在 Verilator 上运行，并修复旧 checker 因 OpCtrl 编码重叠而跳过部分转换和比较向量的问题。 新增 cover-float 数据通路、按配置复用编译，以及缺文件/不支持测试/结果不匹配时的失败退出；覆盖还扩展到 Zfa 的相关舍入与安静比较。

**重要性**：新增实际验证能力与错误检出范围，降低仿真报告零错误却没有检查目标向量的风险。

**风险与限制**：这是 RTL 仿真与测试基础设施能力，不是芯片认证完成；向量格式和单元支持仍决定具体可运行集合。


## 五、AI 业界重磅

### 5.1 重磅｜[OpenAI and Cisco introduce Shirudo（OpenAI 和 Cisco 推出 Shirudo）](https://openai.com/index/introducing-shirudo)

北京时间：2026-11-24 05:20:00｜来源类型：官方博客／公告｜事件状态：已公开

**事实**：OpenAI 与 Cisco 开源基于 eBPF 的 Linux 运行时 Shirudo，允许在永久补丁准备期间部署经操作员批准的临时 shield；签名校验、eBPF verifier 与 TPM 记录用于部署可信性。 受控测试中，Codex 生成的输入检查阻止了针对运行中 Elasticsearch 的 Java4Shell 利用，服务无需重启。

**重要性**：把 AI 辅助防御推进到运行系统上的临时保护与验证环节。

**风险与限制**：这是受控案例和持续开发的运行时，临时 shield 不替代永久修复，也不等于已证明覆盖任意漏洞或生产负载。

### 5.2 模型 & 技术｜[RoboTrack: Point tracking for real-world robotic manipulation（RoboTrack：面向真实世界机器人操作的点跟踪）](https://huggingface.co/blog/allenai/robotrack)

北京时间：2026-11-24 01:20:37｜来源类型：官方博客／公告｜事件状态：已公开

**事实**：Ai2 发布 RoboTrack-Bench（307段视频、2,522条人工核验轨迹）、含1,500万点轨迹的仿真训练数据 RoboTrack-15M，以及 RoboTracker/RoboTracker-Warp 模型。 作者发现现有跟踪器在新基准上比 RoboTAP 低3.0—6.6 AJ；湿实验移液演示中，点跟踪在20种布局完成17次，物体掩码和光流分别为8次与5次。

**重要性**：把真实操作、遮挡和多视角纳入感知验证，并展示仿真数据如何服务机器人部署。

**风险与限制**：结果来自指定基准与20次任务设置；RoboTracker-Warp 权重限非商业使用，不能外推所有机器人任务或独立复现结果。

### 5.3 深度洞见｜[How to build and maintain an agentic evaluation（如何构建并维护智能体评测）](https://huggingface.co/blog/agentic-evals)

北京时间：2026-11-23 08:00:00｜来源类型：官方博客／公告｜事件状态：已公开

**事实**：Hugging Face 公布 agentic evaluation 方法与 Raman 光谱拟合实例：八个开发任务中，gpt-oss-120b 的 pass@3 为7/8，而三次都成功的 pass^3 为1/8。 作者用跳过拟合、直接读取峰高的捷径击穿了只检查比值的 grader，随后加入拟合质量检查；并要求固定 harness、provider、环境与轨迹，以便分清模型能力和评测设计。

**重要性**：为无人值守智能体区分“重试一次能做对”和“每次稳定做对”，给出可审阅的验证实例。

**风险与限制**：这是方法文章和八题开发集，不能据其给出跨领域模型排名；部分比较模型已全部通过，说明需要更难任务。

### 5.4 融资 & 商业｜[Building the digital front door（构建数字入口）](https://sierra.ai/blog/series-f-30b)

北京时间：2026-11-23 21:45:00｜来源类型：官方博客／公告｜事件状态：已公开

**事实**：Sierra 宣布以300亿美元估值融资17.5亿美元，由 Greenoaks 和 Lightspeed Venture Partners 领投，原有主要投资方继续参与。 公司同时披露 ARR 已超过4亿美元，并强调按问题解决等结果收费的企业智能体平台模式。

**重要性**：反映企业智能体从模型调用向可计费业务结果的平台竞争继续扩张。

**风险与限制**：融资、ARR 和业务成效均为公司披露；ARR 不是审计收入，也不能直接证明客户长期回报。

### 5.5 工具 & 产品｜[Introducing Claude Ship（推出 Claude Ship）](https://claude.com/resources/articles/introducing-claude-ship)

北京时间：2026-11-23 16:00至11-24 15:59（原文仅日期）｜来源类型：官方博客／公告｜事件状态：Code Review 正式可用；新增代理 beta

**事实**：Anthropic 发布 Claude Ship：已正式可用的 Code Review，加上 beta 阶段的 PR Steward 和 Claude Approvals，分别覆盖审阅、修复和按仓库策略判断批准。 面向 GitHub 仓库的 Team/Enterprise 用户；团队保留分支保护与必须人工处理的范围，PR Steward 本身不批准或合并，/ship 从 Claude Code 发起工作流。

**重要性**：把编码智能体能力延伸至审阅后的修复与批准链路。

**风险与限制**：按 token 计费；不适用于 Zero Data Retention、HIPAA 配置或客户自管密钥组织。AI 仍可能漏检缺陷，批准不等于绕过已有权限。

### 5.6 工具 & 产品｜[Introducing file slicing controls in the Gemini API（在 Gemini API 中推出文件切片控制）](https://blog.google/innovation-and-ai/technology/developers-tools/file-slicing-controls/)

北京时间：2026-11-24 00:12:00｜来源类型：官方博客／公告｜事件状态：已公开

**事实**：Gemini API 新增服务端文件切片和原生 Office/OpenDocument 上传支持，可按文档页、表格区域、音视频片段及内容层读取，按实际检查部分计 token。 文件上传一次后可跨轮次复用 URI，也可在工具函数结果中指定文件与切片，减少客户端为解析和渲染整份文件维护额外沙箱。

**重要性**：把多模态 agent 的细粒度上下文读取下沉到 API，减少全文件反复输入。

**风险与限制**：适用于 Gemini 4 系列，上传文件仍受原有大小和页数上限约束；没有证明任意文件都能降低总延迟或账单。

### 5.7 行业 & 人事｜[NVIDIA Board Reinstates Jensen Huang as President, CEO and Board Director（NVIDIA 董事会恢复黄仁勋的总裁、首席执行官和董事职务）](https://nvidianews.nvidia.com/news/nvidia-board-reinstates-jensen-huang-as-president-ceo-and-board-director)

北京时间：2026-11-23 21:55:00｜来源类型：官方博客／公告｜事件状态：已公开

**事实**：NVIDIA 董事会一致决定即刻恢复黄仁勋的总裁、CEO 和董事职务；Harvey Jones 继续任董事长，Colette Kress 继续任 CFO。 公司表示独立董事委员会调查仍将继续，并启动 Chief Compliance Officer 招聘，由其向提名与公司治理委员会汇报。

**重要性**：AI 计算平台供应商的领导与合规治理安排发生明确变化。

**风险与限制**：恢复任职不等于调查结束；公告不能证明对具体出口许可或产品交付的影响。

## 六、总结与趋势观察

- **部署链路正获得更明确的工件和兼容契约。** RISE 的 Triton wheel 与 Buddy 0.1.8 的工具链工件提供分发载体，torch-mlir 的 GQA 输入识别和 RotaryEmbedding 修正则收紧模型导入边界。工件存在、模型可导入和目标机实测通过仍是不同阶段。参见：[RISE wheel](https://github.com/riseproject-dev/python-wheels/releases/tag/triton-v3.9.0%2Bgitbbe38e57-20261124082151)、[Buddy 0.1.8](https://github.com/buddy-compiler/buddy-mlir/releases/tag/release/v0.1.8)、[GQA](https://github.com/llvm/torch-mlir/pull/4895)。

- **安全与稳定性需要跨层验证。** Universal CFI 从编译期控制流约束入手，Shirudo 在运行时提供临时保护；RoboTrack 和智能体评测实例进一步显示，单一基准分数或一次成功不足以证明可靠部署。参见：[Universal CFI](https://discourse.llvm.org/t/92955/1)、[Shirudo](https://openai.com/index/introducing-shirudo)、[RoboTrack](https://huggingface.co/blog/allenai/robotrack)、[智能体评测](https://huggingface.co/blog/agentic-evals)。

- **数值与设备集成边界继续显式化。** Linalg/Vector 讨论关注浮点权限传播，Buddy 的向量近似保留误差与适用范围；K1 PTP 验证和 G1 IMU 接入说明，软件接口可用仍取决于协议版本、传感器输入与运行条件。参见：[fast-math](https://discourse.llvm.org/t/92764/20)、[Buddy 数学近似](https://github.com/buddy-compiler/buddy-mlir/pull/1068)、[K1 PTP](https://forum.spacemit.com/t/1793/3)。

## 附录：信源说明

主要信源包括项目官方博客与公告、正式 Release 及工件索引、PyTorch/LLVM 与 RISC-V 技术社区，以及上游 GitHub API 的提交、PR 和差异。覆盖 PyTorch/ExecuTorch、LLVM/MLIR、Buddy、torch-mlir、Triton/TileLang、RISE 和 RISC-V 软件与验证项目。研究测量、社区实机报告和厂商披露均保留其设备、任务、版本与许可边界；原型、设计讨论与已交付能力分别标明。
