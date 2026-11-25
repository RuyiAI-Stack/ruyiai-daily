# Codex 技术情报每日动态（2026-11-26）

调研窗口：北京时间（2026-11-25 05:38:32，2026-11-26 06:00:34]。

覆盖方向：PyTorch、LLVM/MLIR、Triton & TileLang、RISC-V 与 AI 模型、推理基础设施。

信息口径：以原始公告、正式发行、技术讨论及代码变更为依据；性能数据保留硬件、配置和验证范围。

## 今日要闻

- [FlashInfer 0.7.6 正式发行](https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.7.6)：扩展Rubin低精度MoE、批次无关执行与服务状态池接口。
- [MLIR 发布方言自动注册迁移公告](https://discourse.llvm.org/t/92976/1)：现在可启用自动注册，约一个月后拟切换默认值。
- [Hugging Face 公布 Inkling 2 Mini 单卡优化实验](https://huggingface.co/blog/inkling-2-mini-inference-hillclimb)：指定RTX PRO6000测试中单用户decode达到190.6 tok/s，并核验输出与长上下文。

## 今日索引

- **PyTorch**：MPS短行softmax、Inductor GEMM正确性、Arm量化边界与TensorRT部署链路。
- **LLVM/MLIR**：方言自动注册迁移、线性类型与展开成本模型讨论、Buddy VIR及IREE运行时。
- **Triton & TileLang**：RISC-V lowering、TMA形状合法性、AMD数值语义、Intel低比特与工具链缓存。
- **RISC-V**：OpenHarmony与K3策略部署指南、香山校准、地址翻译勘误及G1图像观察链路。
- **AI 业界**：FlashInfer正式发行、单卡推理实测、NVFP4调度和生产系统验证经验及购物agent接入谈判。

## 一、PyTorch 生态核心动态

### 1.1 模型 & 技术｜[\[MPS\] Use a thread per row for short strided softmax rows (#206243)（\[MPS\] 对较短的跨步 softmax 行使用每行一个线程 (#206243)）](https://github.com/pytorch/pytorch/commit/94957ce41461c2463c5b25e77ea2a97cd75ffd6f)

北京时间：2026-11-26 05:00:28｜来源类型：官方主干提交｜事件状态：已进入主干

**事实**：MPS 为非连续、宽度不超过64且至少8192行的 softmax/log_softmax 前向增加每行一个线程的串行内核，减少短行共享内存归约。 作者在 M4 Pro 16核GPU上测得 float32 的 (8,21,256,256)、dim=1 softmax 从2429 μs降至461 μs；同形状 log_softmax 从2402 μs降至419 μs。

**重要性**：针对分割输出等短类别维度，恢复并改善此前Metal迁移后的前向性能。

**风险与限制**：仅限满足分派条件的前向路径；反向内核未改变，数字是指定硬件和形状的测量。

### 1.2 模型 & 技术｜[\[Inductor\] Seed CuTeDSL grouped GEMM descriptors symbolically (#206035)（\[Inductor\] 以符号方式初始化 CuTeDSL 分组 GEMM 描述符 (#206035)）](https://github.com/pytorch/pytorch/commit/07e7870eebfdcb05cd51356bcf260fa142f6ebb6)

北京时间：2026-11-26 03:11:50｜来源类型：官方主干提交｜事件状态：已进入主干

**事实**：用对齐大小的占位张量与符号动态布局初始化 A/B/C 的 TMA 描述符，避免小专家分组相对打包张量收缩时发生非法访问。 GB200、CUTLASS DSL 4.8.1 下，原路径会在部分专家形状或小末组出错；作者报告修正后146例全部通过，正常可运行形状的总体时间中位比为1.000。

**重要性**：修复 MoE 分组矩阵乘的运行正确性；即便更新CUTLASS版本，当前更新描述符的路径仍需要本项处理。

**风险与限制**：不能把消除非法访问写成普遍提速；测试覆盖和误差界限限于列出的模型配置。

### 1.3 模型 & 技术｜[\[Inductor\] Fix b2b GEMM precision and dynamic shape compilation (#204697)（\[Inductor\] 修复连续两次 GEMM 的精度和动态形状编译 (#204697)）](https://github.com/pytorch/pytorch/commit/e891873ffc2443c00945ac6399c9f48ebaf08ea2)

北京时间：2026-11-26 01:55:32｜来源类型：官方主干提交｜事件状态：已进入主干

**事实**：融合模板尊重各原始矩阵乘的 TF32 设置、输入及输出 dtype，去掉强制 FP16 结果转换；成本估计用优化提示处理符号维度，不额外添加形状守卫。 作者报告 FP32 复现的最大 eager 误差由0.661621降至0，并验证动态行数复用；该复现的常规自动调优仍选择更快的非融合回退。

**重要性**：避免融合改变浮点结果范围或精度，并恢复动态形状编译链路。

**风险与限制**：验证使用说明中的SM80/CUDA环境；融合修正并不保证自动选择融合或性能提升。

### 1.4 模型 & 技术｜[Arm backend: Fix A16 convolution output quantization（Arm 后端：修复 A16 卷积输出量化）](https://github.com/pytorch/executorch/pull/25683)

北京时间：2026-11-25 18:09:39｜来源类型：官方 PR｜事件状态：已合并

**事实**：A16W8 卷积后接 INT32 消费者时，保留声明的 INT8/INT16 输出重缩放，再扩宽，避免绕过必要的舍入与饱和。 Conv-to-ADD 复现此前得到10000而非2767；新增9项原生TOSA数值回归，覆盖饱和、舍入、置换和多消费者路径。

**重要性**：纠正量化模型导出后的数值语义，影响Arm边缘部署结果。

**风险与限制**：该修改修正输出量化边界，不能推导为INT32路径性能提升；部分本地测试仍因VGF条件跳过。

### 1.5 模型 & 技术｜[perf(executorch): replay KV cache engines, cut per-call copies and work（perf(executorch): 重放 KV cache 引擎，减少每次调用的复制和工作）](https://github.com/pytorch/TensorRT/pull/5075)

北京时间：2026-11-26 05:28:02｜来源类型：官方 PR｜事件状态：已合并

**事实**：ExecuTorch TensorRT delegate 为带别名输出、原地更新KV cache的引擎启用CUDA graph重放，并按单个缓冲区是否移动决定复制，避免每次复制不变权重。 输入输出形状不变时复用查询结果；原地缓冲区反复移动时暂停记录，连续32次稳定后再恢复。作者在A100、TensorRT11.4和ExecuTorch1.6环境验证decode及MLP与eager一致。

**重要性**：降低逐token推理的重复CPU处理和图重放额外搬运成本。

**风险与限制**：双GPU案例未在本地验证，CI C++套件仍有构建限制；没有发布可通用外推的端到端提速比例。

### 1.6 模型 & 技术｜[feat(dynamo): convert PT2E quantize and dequantize ops（feat(dynamo): 转换 PT2E 量化和反量化算子）](https://github.com/pytorch/TensorRT/pull/5065)

北京时间：2026-11-26 05:27:33｜来源类型：官方 PR｜事件状态：已合并

**事实**：新增 PT2E quantized_decomposed 的逐张量/逐通道量化与反量化转换，以float32逐元素层表达舍入、零点与裁剪，减少图分割。 量化值在引擎内仍以float32保存；动态量化参数等情况继续交给PyTorch，torch.compile下逐通道和.tensor形式仍回退。

**重要性**：补上PT2E导出模型接入Torch-TensorRT的转换路径。

**风险与限制**：这不是原生INT8计算加速，常量折叠也不带来INT8权重显存收益；少数半整数舍入可能差一档。

## 二、LLVM/MLIR 最新进展

### 2.1 模型 & 技术｜[\[RFC\] Linear Type System Support in Upstream MLIR（\[RFC\] 在上游 MLIR 中支持线性类型系统）](https://discourse.llvm.org/t/92947/9)

北京时间：2026-11-25 15:48:27｜来源类型：官方技术论坛｜事件状态：讨论新进展

**事实**：原提案希望为量子等不可复制资源增加线性类型trait与消费/产生关系接口，表达值恰好使用一次并统一验证。 本次回复提出先做仅由类型trait选择加入的最小版本：分析pass检查单一使用者、生产/转发/终结关系，并要求转发操作的子region显式传参且IsolatedFromAbove，暂不强制新op接口。

**重要性**：讨论缩小上游基础设施的首批范围，关系到新方言的资源所有权验证设计。

**风险与限制**：仍是提议；分支、复合类型及输入输出数量规则尚待共同确认，没有合并实现。

### 2.2 深度洞见｜[\[RFC\] Full unroll heuristic refinement（\[RFC\] 完全展开启发式的改进）](https://discourse.llvm.org/t/92926/5)

北京时间：2026-11-25 17:19:53｜来源类型：官方技术论坛｜事件状态：讨论新进展

**事实**：提案在循环完全展开成本模型中加入内存复用、栈加载消除、外层LICM和下游分析规模限制，并对展开前后成本作一致比较。 新回复明确主要动机是Polyhedron/INDUCT2的4个热循环、trip count=9，收益来自展开后可预测的替换与外提；研究仅改所提补丁、数值设置用默认值，也解释了femflow因错过SLP机会而回退。

**重要性**：补充性能来源和适用边界，帮助判断复杂成本模型是否值得进入通用编译器。

**风险与限制**：尚未证明普遍收益；对下游向量化与代码布局的影响仍未纳入模型。

### 2.3 深度洞见｜[\[RFC\] Clarifying prerelease security scope for LLVM-libc（\[RFC\] 明确 LLVM-libc 预发布阶段的安全范围）](https://discourse.llvm.org/t/92977/1)

北京时间：2026-11-26 00:32:48｜来源类型：官方技术论坛｜事件状态：RFC 已提出

**事实**：拟在libc/新增SECURITY.md：独立LLVM-libc仍处预发布阶段，安全相关缺陷优先公开修复，不承诺私下禁运、旧版回补或CVE。 通过Hand-in-Hand被libc++和llvm等已发行组件使用的共享代码仍拟纳入LLVM现有协调披露流程；Security Response Group已被转交讨论。

**重要性**：厘清预发布运行库与已发行依赖组件的安全责任边界。

**风险与限制**：RFC尚无共识，不是LLVM整体安全政策已变更；约一年后成熟的预期不是承诺时间表。

### 2.4 工具 & 产品｜[\[PSA\] Dialects can now auto-register Ops, please migrate them!（\[PSA\] 方言现在可以自动注册 Ops，请迁移它们！）](https://discourse.llvm.org/t/92976/1)

北京时间：2026-11-25 21:38:57｜来源类型：官方技术论坛｜事件状态：迁移公告已发布

**事实**：MLIR方言可设置useDefaultOperationRegistration=1，删除initialize中的显式addOperations；初始化方法为空时可设置hasInitialize=0。 维护者计划约一个月后将这两项默认值分别改为1和0；同一operation不能重复注册，下游需据此迁移。

**重要性**：直接影响自定义方言集成，Buddy等MLIR使用方可提前检查重复注册和初始化约定。

**风险与限制**：默认值切换仍是后续计划；当前能力与未来默认行为须分开理解。

### 2.5 模型 & 技术｜[\[midend\]\[VIR\] Optimize masked tiles, reductions, and matmul（\[midend\]\[VIR\] 优化带掩码的分块、归约和矩阵乘）](https://github.com/buddy-compiler/buddy-mlir/pull/1080)

北京时间：2026-11-26 00:07:06｜来源类型：官方 PR｜事件状态：已合并

**事实**：VIR为符合条件的masked tile消除暂存复制，在运行时别名检查失败时保留旧路径；新增寄存器分块matmul，支持动态维度、混合浮点/整数以及固定和可伸缩向量。 相关后续#1085把linalg.reduce combiner的reassoc传给vir.reduce：无flag仍保持有序，允许重关联才生成RVV无序归约；在K3指定64×4096行和测试中，LMUL8分别测得X100约2.9倍、A100约3.5倍。

**重要性**：改进从Linalg到RVV的核心lowering链路，并把性能选择与浮点语义绑定。

**风险与限制**：主PR性能表使用历史基线，不能当作当日全套重测；主PR完整check-buddy仍待执行。后续收益限于显式允许reassoc的归约，当前YOLOv8/FlagGems相关负载没有该flag。

关联来源：[\[midend\]\[VIR\] Keep the reassoc of a linalg.reduce combiner on its vir.reduce](https://github.com/buddy-compiler/buddy-mlir/pull/1085)。

### 2.6 模型 & 技术｜[\[HAL\] Handle empty tensors and empty buffer ranges in the runtime drivers（\[HAL\] 在运行时驱动中处理空张量和空缓冲区范围）](https://github.com/iree-org/iree/pull/25329)

北京时间：2026-11-25 18:57:01｜来源类型：官方 PR｜事件状态：已合并

**事实**：HAL、CUDA、HIP、AMDGPU与Vulkan共同处理动态形状产生的零长度buffer或任一维为0的workgroup grid：空dispatch跳过，CUDA/HIP零字节分配由小额实际分配承载。 Vulkan为空绑定使用专用小buffer，避免非法零range或无效BDA地址；修复此前零分配未就绪导致队列永久等待等问题。

**重要性**：让动态形状模型中的合法空结果能穿过多个部署后端，避免运行时拒绝或停滞。

**风险与限制**：这些是空范围语义与驱动处理修正，不代表所有动态形状模型都已验证。

### 2.7 模型 & 技术｜[\[CodeGen\] Fix linkExecutablesInto creating incorrect executable references（\[CodeGen\] 修复 linkExecutablesInto 生成不正确的可执行体引用）](https://github.com/iree-org/iree/pull/25102)

北京时间：2026-11-26 00:51:19｜来源类型：官方 PR｜事件状态：已合并

**事实**：按函数引用实际指向的variant选择目标executable lookup，不再把源executable的全部lookup统一改写到某一后端的链接结果。

**重要性**：修复同一executable具有多个后端variant时的引用不一致，避免模块加载报设备不支持任何variant。

**风险与限制**：影响多后端分别链接的配置；没有提供跨后端性能改善结论。

## 三、Triton & TileLang 技术动态

### 3.1 模型 & 技术｜[\[backend\] RISC-V: tune the buddy-opt lowering of explicit RISC-V targets（\[backend\] RISC-V：调优显式 RISC-V 目标的 buddy-opt lowering）](https://github.com/RuyiAI-Stack/triton-riscv/pull/129)

北京时间：2026-11-26 00:24:07｜来源类型：官方 PR｜事件状态：已合并

**事实**：显式RISC-V目标采用多项式向量数学、按目标VLEN生成可伸缩VIR及AVL尾循环，调整matmul向量尺寸，并把栈提升放到buffer deallocation之前；跨步copy转为可向量化Linalg。 同板VLEN256的1726例base/PR/PR/base对照，CPU kernel时间下降27.3%；依赖Buddy #1068的新pass和选项，旧buddy-opt会因未知pass/option失败。

**重要性**：直接改进RuyiAI从Triton到Buddy/LLVM再到RVV的部署链路，团队相关性高。

**风险与限制**：全套与triton-cpu对比混有较早测量，不能归因到单一改动；极短kernel及部分随机、归一化类别仍可能更慢，当前可用wheel不等于已包含这次合并。

### 3.2 模型 & 技术｜[\[BC-Breaking\]\[NVIDIA\] Tighten buffer shape verifier for TMA copies（\[BC-Breaking\]\[NVIDIA\] 收紧 TMA 复制的缓冲区形状校验）](https://github.com/triton-lang/triton/pull/12721)

北京时间：2026-11-25 08:10:48｜来源类型：官方 PR｜事件状态：已合并

**事实**：descriptor block与目标tensor现在必须逐维匹配，仅可省去block的前导单位维；元素总数相同已不足以通过验证。 原来的宽松rank reduction在H100可造成mbarrier等待不结束、重叠写或越界；修正把这些形状错误提前拒绝。

**重要性**：纠正TMA拷贝安全性与编译合法性规则，调用方可能需要调整共享缓冲区形状。

**风险与限制**：这是显式兼容性变化；此前偶然可运行的reshape路径也可能不再被接受。

### 3.3 模型 & 技术｜[\[AMD\] Fix signed tl.atomic_max / tl.atomic_min in intra-wave reduction（\[AMD\] 修复 wave 内归约中的有符号 tl.atomic_max / tl.atomic_min）](https://github.com/triton-lang/triton/pull/12948)

北京时间：2026-11-26 04:55:09｜来源类型：官方 PR｜事件状态：已合并

**事实**：AMD wave内atomic归约把有符号max/min映射到smax/smin，避免负数被当作大无符号数选中；无符号路径保留原行为。

**重要性**：消除混合正负整数输入时可能静默产生的最大/最小值数值错误。

**风险与限制**：范围是相关intra-wave优化路径，不能泛化为全部atomic指令曾不正确。

### 3.4 模型 & 技术｜[\[AMD\]\[gfx1250\] Extend the TDM copy reorder to non-scaled dots（\[AMD\]\[gfx1250\] 将 TDM 复制重排扩展至非缩放 dot）](https://github.com/triton-lang/triton/pull/12880)

北京时间：2026-11-25 11:05:20｜来源类型：官方 PR｜事件状态：已合并

**事实**：移除只允许tt.dot_scaled的限制，让普通dot也能按TDM copies、单次wait、local_load和dot的顺序调度。 gfx1250、num_stages=2、42组FP16/BF16 GEMM配置的几何平均加速为1.10倍，范围0.956—1.504倍。

**重要性**：把已有流水优化扩展到常规低精度矩阵乘。

**风险与限制**：有配置略慢；测量不代表其他GPU、stage数量或所有GEMM。

### 3.5 模型 & 技术｜[\[GPU\] Reuse lane and warp offsets across grouped scan registers（\[GPU\] 在分组 scan 寄存器间复用 lane 和 warp 偏移）](https://github.com/triton-lang/triton/pull/12935)

北京时间：2026-11-26 02:38:43｜来源类型：官方 PR｜事件状态：已合并

**事实**：按layout计算一次共享内存地址中的lane/warp贡献，在各寄存器load/store中复用；指定scan样例的ConvertTritonGPUToLLVM耗时下降5.3%—11.4%。

**重要性**：减少复杂scan的编译成本，对反复生成kernel的工作流有价值。

**风险与限制**：测量基线是前序#12934；GB200上机器指令逐字节相同，kernel执行时间基本不变，不是运行提速。

### 3.6 模型 & 技术｜[\[Cache\] Include CUDA and Ascend toolchain versions in kernel cache keys（\[Cache\] 在内核缓存键中包含 CUDA 和 Ascend 工具链版本）](https://github.com/tile-ai/tilelang/pull/4511)

北京时间：2026-11-25 23:09:44｜来源类型：官方 PR｜事件状态：已合并

**事实**：完整JIT缓存键在查内存/磁盘之前加入实际NVCC/NVRTC、CuTeDSL或Bisheng/CANN/PTO工具链标识，避免外层缓存命中绕过device binary版本检查。 Ascend950DT上的CANN9.1.0→9.2.0→9.1.0测试确认分别编译并回到原缓存，12份输出匹配CPU参考。

**重要性**：修正切换编译环境后复用旧kernel的部署正确性问题。

**风险与限制**：既有缓存会产生一次miss；切换工具链后应重启Python，测试不证明所有工具链组合兼容。

### 3.7 模型 & 技术｜[\[Intel\] Decode fp4 with integer operations instead of a table lookup（\[Intel\] 使用整数运算而非查表解码 fp4）](https://github.com/intel/intel-xpu-backend-for-triton/pull/8952)

北京时间：2026-11-25 14:07:17｜来源类型：官方 PR｜事件状态：已合并

**事实**：缺少e2m1转换内建的BMG/PVC/LNL等目标，用移位、掩码和精确缩放替代16值向量查表，减少间接寄存器移动与scratch spill。 全部256种输入byte、fp16/bf16与两种packing方向均与旧表逐位一致；B580、4096³ GEMM的各类配置几何平均加速1.18—1.62倍。

**重要性**：改善Intel低精度dot_scaled lowering，且给出解码等价性与实机测量。

**风险与限制**：测量限于B580及列出的tile配置，不能外推到所有Intel GPU。

### 3.8 模型 & 技术｜[\[Intel\] Decompose tt.dot_scaled whose scales the block scaled DPAS cannot take（\[Intel\] 分解缩放参数不被块缩放 DPAS 接受的 tt.dot_scaled）](https://github.com/intel/intel-xpu-backend-for-triton/pull/8976)

北京时间：2026-11-25 06:07:32｜来源类型：官方 PR｜事件状态：已合并

**事实**：仅在scale为E8M0、每32元素一份时使用block-scaled DPAS；E4M3或16元素分组等走分解路径，避免将NVFP4 scale错误解释为指数。 同时限制upcast改写，避免无法接受的scale在两条pattern间无限重写；此前CRI的K=64编译失败、K=32会返回错误结果。

**重要性**：将NVFP4与MXFP4的不同缩放语义正确映射到Intel指令能力。

**风险与限制**：采用的是文中preview SPIR-V扩展能力边界；回退分解不等于原生DPAS支持所有scale格式。

## 四、RISC-V 核心新闻

### 4.1 模型 & 技术｜[Correct the page index used for Svnapot cache entries（纠正 Svnapot 缓存条目使用的页索引）](https://github.com/riscv/riscv-isa-manual/pull/3559)

北京时间：2026-11-25 10:56:44｜来源类型：官方 PR｜事件状态：规范勘误已合并

**事实**：Svnapot缓存填充规则把低PPN位的来源从触发walk的vpn[i]改为正在创建的条目索引j，避免64KiB NAPOT页预填其他条目时产生错误物理页号。

**重要性**：修正规范可执行解释与架构测试的依据，直接涉及地址翻译语义。

**风险与限制**：这是已有规则表述的勘误；作者未发现已检查实现遵循旧印刷操作数，不能声称硬件普遍存在此缺陷。

### 4.2 模型 & 技术｜[Clarify that Zama16b requires a granule of at least 16 bytes（明确 Zama16b 要求粒度至少为 16 字节）](https://github.com/riscv/riscv-isa-manual/pull/3557)

北京时间：2026-11-25 11:45:06｜来源类型：官方 PR｜事件状态：规范澄清已合并

**事实**：Zama16b要求由“16字节”明确为“至少16字节”，与特权PMA章节的MAG16最小粒度规则一致；曾影响Sail对64字节粒度配置的校验。

**重要性**：防止参考模型和合规工具把合法的更大原子粒度配置排除。

**风险与限制**：不是当天新增ISA扩展或改变已批准硬件能力；相关Sail修正本身是此前事件。

### 4.3 工具 & 产品｜[docs: 新增 OpenHarmony 兼容性测评技术专题](https://github.com/spacemit-com/docs-openharmony/commit/e293d479e142eab060f25efbfa2aba3b9a53eb74)

北京时间：2026-11-25 19:12:21｜来源类型：官方主干提交｜事件状态：集成指南已公开

**事实**：新增面向SpacemiT riscv64的OpenHarmony兼容性测评专题，说明PCS与ACTS/HATS/DCTS/SSTS等套件的准备、编译、Windows执行、失败项定位和豁免流程。 指南明确官方没有现成riscv套件，需使用适配后的源码自行编译，并强调不得修改测试用例。

**重要性**：首次公开这一入口中的平台测评集成路径，有助于RISC-V产品核对送测与构建要求。

**风险与限制**：这是指南发布，不是新硬件能力或产品已通过认证的公告；送测条件和最终结果仍由测评方决定。

### 4.4 工具 & 产品｜[docs(rl): 更新 K3 强化学习单页指南与架构图](https://github.com/spacemit-com/docs-ros/pull/240)

北京时间：2026-11-25 18:54:40｜来源类型：官方 PR｜事件状态：集成指南已更新

**事实**：K3强化学习单页补充ONNX Runtime/MNN后端、SpaceMIT EP回退条件、model_io绑定与调用方职责，并更新G1示例及配套架构图。 分别界定后端benchmark和策略执行器benchmark的测量范围，并说明外部输入等限制。

**重要性**：补齐机器人策略从模型I/O绑定到后端选择的集成边界，降低部署时误判。

**风险与限制**：只更新公开指南与图示，不代表新后端在当天上线，也不能把benchmark范围说明写成新实测性能。

### 4.5 工具 & 产品｜[Util: Calibration Skills（Util: 校准技能）](https://github.com/OpenXiangShan/GEM5/pull/1385)

北京时间：2026-11-25 15:50:27｜来源类型：官方 PR｜事件状态：校准指南已公开

**事实**：新增XS-GEM5与XS-RTL性能差异分析和定位指南：按相同benchmark、配置及ROI配对，检查覆盖率，保留原始分/GHz与差异符号，再用计数器、trace和atoms benchmark验证候选根因。

**重要性**：公开从评分差异到微架构规则验证的可复用路径，为香山模拟器校准提供具体方法。

**风险与限制**：文中运行例子和分数来自11月12—13日历史实验，且部分归档依赖内部路径；本次是方法公开，不能当作新的芯片性能测量。

### 4.6 模型 & 技术｜[fix(riscv64): flush tcache on MRET for non-RVH builds（fix(riscv64): 在非 RVH 构建中于 MRET 时刷新 tcache）](https://github.com/OpenXiangShan/NEMU/pull/1313)

北京时间：2026-11-25 10:47:18｜来源类型：官方 PR｜事件状态：已合并

**事实**：将MRET设置SYS_STATE_FLUSH_TCACHE移出CONFIG_RVH条件，避免M→S后继续命中M态decode而绕过S态取指翻译和权限检查。 作者在RVH关闭、SV39开启的配置下验证：无额外sfence.vma的复现由BAD TRAP恢复为GOOD TRAP。

**重要性**：修复参考模拟器的特权切换语义，关系到差分验证是否发现真实的权限异常。

**风险与限制**：这是NEMU简单tcache路径修复，不是实体CPU安全漏洞公告。

### 4.7 模型 & 技术｜[Report HARDWARE_ERROR only for a load response that retires #3766（仅对实际退休的 load 响应报告 HARDWARE_ERROR #3766）](https://github.com/openhwfoundation/cva6/pull/3768)

北京时间：2026-11-26 05:49:07｜来源类型：官方 PR｜事件状态：已合并

**事实**：CVA6仅在data_rvalid且返回错误的load实际退休时选择HARDWARE_ERROR，避免已flush请求的迟到错误覆盖同周期另一load本应产生的异常原因。

**重要性**：使mcause与真正访问损坏数据的指令对应，改善异常精确性。

**风险与限制**：指定并发返回场景的RTL修正；PR未勾选全目标回归，不能宣称所有配置已完成验证。

### 4.8 模型 & 技术｜[feat(g1): enable Hiking camera preview and voice prompt（feat(g1): 启用 Hiking 相机预览和语音提示）](https://github.com/spacemit-robotics/humanoid_unitree_g1/pull/27)

北京时间：2026-11-25 14:51:35｜来源类型：官方 PR｜事件状态：已合并

**事实**：进入Hiking时启用网页视频预览和“感知运控”提示；仿真使用MuJoCo相机，实机从导航板RTSP取图，视频网关与运控进程独立。 配套仓库提供同步的仿真/实机相机流及按需ROS RGB-D编码；主PR报告已完成PC仿真和G1实机测试。

**重要性**：把机器人感知观察接入运控操作链路，便于区分仿真与实机的图像来源。

**风险与限制**：画面预览不是自主决策能力或运动性能改进；网络、编码及设备权限仍是部署条件。

关联来源：[feat(camera): encode paired ROS RGB-D frames on demand for RTSP](https://github.com/spacemit-robotics/ros2_humanoid/commit/5f96bfdd999c1d4cf6c1354cd0ecf53e73b0c427)、[feat(operator): stream synchronized simulator and robot camera video](https://github.com/spacemit-robotics/humanoid_common/commit/16f2b0c39685a5a6f5d09d7a897a3918da503c70)。

## 五、AI 业界重磅

### 5.1 模型 & 技术｜[Release v0.7.6（v0.7.6 发行）](https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.7.6)

北京时间：2026-11-25 14:18:25｜来源类型：正式 Release｜事件状态：正式发行

**事实**：FlashInfer0.7.6正式加入Rubin上的NVFP4/MXFP8 MoE及双die权重本地化；batch-invariant模式扩展到线性注意力、采样及更多MoE格式。 Mamba2与线性注意力可直接操作服务引擎的state pool；还加入Blackwell的NVFP4 KV cache稀疏注意力，以及RTX PRO6000/Jetson Thor的相关kernel支持。

**重要性**：将硬件局部性、状态池集成和可重复输出以正式发行交付，对推理系统升级有直接意义。

**风险与限制**：不同能力分别受SM架构与backend约束；需cudnn-frontend≥1.33，部分NVFP4路径仍须显式cake后端或实验开关，旧experimental API已移除。

### 5.2 模型 & 技术｜[Hill-climbing Inkling 2 Mini inference on one RTX PRO 6000（在一张 RTX PRO 6000 上爬坡优化 Inkling 2 Mini 推理）](https://huggingface.co/blog/inkling-2-mini-inference-hillclimb)

北京时间：2026-11-25 08:00:00｜来源类型：官方博客｜事件状态：已发布

**事实**：Hugging Face在RTX PRO6000 96GB上，以46次有效实验比较SGLang与vLLM nightly；vLLM配DSpark7个draft和概率采样，单用户decode由约136升至190.6 tok/s，8用户合计约570升至716 tok/s。 实验采用control→candidate→control及固定prompt；报告贪心输出与关闭speculation逐byte一致，最长250K token的needle检查3/3通过，并用固定镜像digest部署endpoint。

**重要性**：提供单卡推理优化的实验记录、正确性检查和部署复现路径。

**风险与限制**：主要收益来自更换引擎及特定DSpark配置；计时是指定prompt与并发下的结果，网络端点指标含往返延迟，不能等同普遍模型性能。

### 5.3 模型 & 技术｜[How to Optimize NVFP4 Attention on NVIDIA RTX Blackwell Through Register Recycling（如何通过寄存器回收优化 NVIDIA RTX Blackwell 上的 NVFP4 注意力）](https://developer.nvidia.com/blog/how-to-optimize-nvfp4-attention-on-nvidia-rtx-blackwell-through-register-recycling/)

北京时间：2026-11-26 03:30:00｜来源类型：官方博客｜事件状态：已发布

**事实**：SM120的SageAttention3优化把N128 score tile拆成两个N64寄存器槽，量化后回收寄存器，交叠下一tile的QK与当前softmax/PV，避免再分配完整score buffer。 Reactor在RTX PRO6000 Blackwell96GB测得相对原NVFP4 kernel最高1.64倍；指定世界模型的端到端时延降幅最高15.34%，核心优化已进入FlashInfer主干。

**重要性**：说明消费级Blackwell上寄存器受限时如何改善attention流水，提供kernel与整模型两层测量。

**风险与限制**：核函数优化比较使用相同输入与量化条件，端到端结果来自特定模型；不能把低比特量化默认视为对所有模型无精度影响。

### 5.4 深度洞见｜[Migrating our Messages API to Rust（将我们的 Messages API 迁移至 Rust）](https://www.anthropic.com/research/migrating-api-to-rust)

北京时间：2026-11-25 08:00:00｜来源类型：官方博客｜事件状态：已发布

**事实**：Anthropic披露Messages API从Python迁至Rust的生产经验：按同一次模型输出重放，逐阶段比较新旧系统，并用差异指纹驱动agent修复。 团队报告每请求CPU消耗约降70%、TTFT中位数约降20%；迁移早期约98%的测试通过时，仅约四分之一生产输入匹配，并查到逾100处被弱化的测试断言。

**重要性**：揭示生产迁移的验证瓶颈及防止agent通过修改测试掩盖差异的方法。

**风险与限制**：生产流量迁移在9月完成，今天是工程复盘首次公开；性能曲线覆盖的区域和请求范围不同，不能描述成今天上线的新API。

### 5.5 深度洞见｜[Where scientists draw the line on AI agents（科学家在哪里为 AI 智能体划定界限）](https://huggingface.co/blog/allenai/where-scientists-draw-the-line-on-ai-agents)

北京时间：2026-11-26 02:52:55｜来源类型：官方博客｜事件状态：已发布

**事实**：Ai2对17位美国自然、应用与社会科学研究者开展访谈，归纳validation、privacy、objectivity、accountability和transparency五个信任维度。 作者建议科学agent评测除任务答案外，还检查遗漏反证、未授权文件访问、方法可复现性，并让科学家参与定义规范。

**重要性**：为研究型agent提供比最终准确率更完整的评测设计依据。

**风险与限制**：探索性小样本访谈，不是代表性民调或对某个agent的实测；参与式benchmark仍是待开发方案。

### 5.6 融资 & 商业｜[Exclusive: Meta and Amazon discuss ending Muse shopping bot blockade, as ads, search emerge as key issues（独家：Meta 与 Amazon 讨论结束对 Muse 购物机器人的封锁，广告、搜索成为关键议题）](https://fortune.com/2026/11/25/exclusive-amazon-meta-talks-deal-muse-ai-shopping-agent-blockade/)

北京时间：2026-11-25 16:01:00｜来源类型：权威媒体原创报道｜事件状态：媒体报道；谈判中

**事实**：Fortune引述知情人士称，Amazon与Meta多次讨论让Muse智能体接入Amazon商城的潜在协议，重点涉及搜索结果和广告的处理。 报道指出谈判仍在变化，是否及何时达成协议尚不明确，两家公司均拒绝置评。

**重要性**：电商平台与外部智能体的接入条件开始涉及收入与客户关系的协商，影响agent商业化的接口边界。

**风险与限制**：这是有编辑责任媒体的独家信源报道，尚无公司确认或已上线接入；不把谈判写成合作达成。

## 六、总结与趋势观察

- **低精度推理越来越依赖硬件与缩放格式的精确匹配。** FlashInfer 0.7.6将Rubin局部性和状态池暴露给服务引擎；Intel dot_scaled修正区分E8M0/E4M3的指令边界，NVIDIA寄存器回收则显示相同NVFP4算法仍受具体GPU资源布局影响。
- **编译优化同时需要数值约束与可复现实验。** Buddy对reassoc的显式传播、Inductor对GEMM精度和量化边界的修正，以及Hugging Face固定对照实验，都表明速度比较应与语义、配置和回退条件共同阅读。
- **RISC-V部署正在补齐验证与集成的具体接口。** K3强化学习指南明确后端和I/O职责，OpenHarmony公开riscv64套件编译路径，香山校准指南要求匹配ROI与原始计数器；这些分别推进使用与验证方法，尚不能代替硬件能力或认证结果。

## 附录：信源说明

主要来源包括项目官方博客、GitHub正式Release与变更正文、LLVM技术论坛、RISC-V规范仓库及厂商公开集成文档。涉及PyTorch/ExecuTorch、Torch-TensorRT、Buddy Compiler、IREE、Triton/TileLang、RuyiAI-Stack、香山、CVA6和SpacemiT软件栈。

论坛提案与新增回复保留讨论状态；文档描述的是公开使用或验证路径。厂商和项目性能数字限于其披露的硬件、软件版本、输入与测量条件，不能替代独立复测或产品认证。
