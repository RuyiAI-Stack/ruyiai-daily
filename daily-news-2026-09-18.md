# Codex 技术情报每日动态（2026-09-18）

**调研窗口**：2026-09-17 04:37:07—2026-09-18 06:40:24（Asia/Shanghai，左开右闭）。
**覆盖方向**：PyTorch、LLVM/MLIR、Triton & TileLang、RISC-V 与 AI 业界模型/基础设施/芯片/应用。
**信息口径**：正文只采用已回到官方博客、公告、论坛/RFC、Release、官方 API 或有编辑责任媒体的 verified evidence；聚合器、Google News、Hacker News 与 Daily Papers 仅作候选发现。

## 今日要闻

- [推出 Astra for Law](https://openai.com/index/astra-for-law/)：OpenAI 将 GPT-6 Astra 与法律检索、工具和治理控制结合，并公布 200 道法律研究题的对照结果。
- [Claude 如何提升生物分子建模](https://www.anthropic.com/research/claude-uplifts-biomolecular-modeling)：Anthropic 报告对 30 多个开源生物分子模型的优化，并开源代码。
- [小板子运行大模型：deepin 25 RVA23 版本适配进迭时空芯片 SpacemiT K3](https://www.deepin.org/zh/deepin-25-rva23-spacemit-k3/)：deepin 公布从 RVA23 工具链到本地推理运行时的 K3 体验路径。
- [如何使用 AI 智能体为仿真准备 3D 场景](https://developer.nvidia.com/blog/how-to-use-ai-agents-to-prepare-3d-scenes-for-simulation/)：NVIDIA 将 Blender、OpenUSD、Omniverse Libraries 与 SimReady 验证串为仿真准备流程。

## 今日索引

PyTorch：Python 版本基线、量化融合和 ExecuTorch 符号形状与 Cortex-M 部署。
LLVM/MLIR：match 方言、CIR 边界、缩放收缩与成本模型监控。
Triton & TileLang：块缩放 GEMM、原子降低正确性、BMM 性能和 AMD TDM 边界。
RISC-V：K3/RVA23 本地 AI、RVV 推理、Hypervisor 测试门控与 ISA 语义。
AI 业界重磅：专业法律模型、科学推理优化、Claude Code 项目并行与仿真智能体。

## 一、PyTorch 生态核心动态

### 1.1 模型 & 技术｜[Cortex-M：将 SAME 填充融合进量化卷积](https://github.com/pytorch/executorch/pull/22861)

**北京时间**：2026-09-17 05:51；**来源类型**：official_github_pr；**事件状态**：已合并。

将支持的零填充 SAME_UPPER 融合进 Cortex-M 普通卷积与深度卷积，覆盖两种布局管线；融合在 dialect 转换后、scratch 大小规划前执行，并保留共享 pad 输出。 **重要性**：减少端侧量化卷积前单独的 int8 pad 操作，改变导出到运行时的实际部署路径。 **风险/限制**：不支持的 MVE 1×N 情形仍保留显式 padding；不能外推未覆盖布局和填充值。

### 1.2 模型 & 技术｜[支持用于共享输入线性层融合的 MXTensor cat](https://github.com/pytorch/ao/pull/4882)

**北京时间**：2026-09-17 21:06；**来源类型**：official_github_pr；**事件状态**：已合并。

为 MXTensor 增加 aten.cat，使共享输入的 Q/K/V、gate/up 线性层可以拼接权重后执行一次矩阵乘法；直接拼接量化数据，避免反量化再量化。Llama 3.1 8B 验证展示 Q/K/V 从三次线性计算变成一次矩阵乘法加拆分。 **重要性**：减少重复激活量化和 GEMM 启动，为低精度模型图优化增加可用路径。 **风险/限制**：仅支持二维权重沿输出特征维拼接，要求量化元数据匹配；不能据此推断所有模型的端到端加速。

### 1.3 模型 & 技术｜[并行化低比特预填充激活打包](https://github.com/pytorch/ao/pull/4854)

**北京时间**：2026-09-18 00:30；**来源类型**：official_github_pr；**事件状态**：已合并。

大批量低比特 prefill 的激活量化与打包改为并行处理，并按最多 1024 行的分块处理 token 维；Arm NEON GEMM 路径复用共享解包辅助函数，覆盖 1 至 8 位权重。 **重要性**：把低比特推理中 GEMM 前的数据准备纳入并行优化。 **风险/限制**：作者的 W3 直接 A/B 微基准性能不变，不能把并行化等同于所有位宽和形状都加速。

### 1.4 模型 & 技术｜[Arm 后端：放宽 SymbolicShapeSupport 检查](https://github.com/pytorch/executorch/pull/22834)

**北京时间**：2026-09-18 05:25；**来源类型**：official_github_pr；**事件状态**：已合并。

TOSA 支持检查区分符号张量元数据与形状值实体化：没有 TOSA shape 扩展时，仍允许不依赖实体化形状的算子携带符号输入输出形状；要求静态形状的目标在分区器层另行拒绝未解析形状。 **重要性**：扩大 Arm 模型导出与分区可接受的符号形状范围。 **风险/限制**：符号 SymInt 参数及明确依赖形状实体化的边界情形仍受限制。

### 1.5 工具 & 产品｜[通知：PyTorch 2.15 正在移除 Python 3.10 支持（2.14 是最后提供 3.10 wheels 的版本）](https://dev-discuss.pytorch.org/t/notice-python-3-10-support-is-being-removed-from-pytorch-2-15-2-14-is-the-last-release-with-3-10-wheels/3440)

**北京时间**：2026-09-17 06:36；**来源类型**：official_announcement；**事件状态**：支持调整已公告。

官方通知：2.14 是最后提供 cp310 wheels 的版本，2.15 最低支持 Python 3.11；nightly CD 计划在 9 月 21 日所在周移除 3.10，CI 于 9 月 28 日移除。已发布的 2.14 二进制继续保留。 **重要性**：影响 PyTorch 及下游模型导入、编译和部署环境的 Python 版本基线。 **风险/限制**：这是已公布的支持计划；2.15 发布前使用 nightly 的环境会先受影响。

## 二、LLVM/MLIR 最新进展

### 2.1 模型 & 技术｜[[RFC] match：位于 pdl 与 pdl_interp 之间的新方言](https://discourse.llvm.org/t/rfc-match-a-new-dialect-in-between-pdl-and-pdl-interp/91847)

**北京时间**：2026-09-18 00:00；**来源类型**：high_value_discussion_or_rfc；**事件状态**：提案讨论中。

提案新增 match 方言，显式表示 PDL 约束和匹配器调度，将调度与 pdl_interp 生成解耦；结构化控制流还可用于生成 C++，探索合并多模式、多根模式的 AOT 重写。帖子提供浏览器 playground 与示例。 **重要性**：给 MLIR 模式匹配优化、等式饱和及重写验证提供新的中间表示，贴近 Buddy Compiler 所关注的变换基础设施。 **风险/限制**：仍是设计提案；论文、实验工具和未来 Lean 验证方向不等于上游已交付能力。

### 2.2 模型 & 技术｜[[LLVMCPU] 添加 vscale_range 目标字段，并要求可伸缩向量使用它](https://github.com/iree-org/iree/pull/24848)

**北京时间**：2026-09-18 00:02；**来源类型**：official_github_pr；**事件状态**：已合并。

在 executable target 和 dispatch 上记录用户可指定的 vscale_range；启用 scalable vectorization 时必须提供该范围。目标具有 +zvl*b 特征时，有效下界取硬件特征与用户下界中的较大值。 **重要性**：为 IREE 的可伸缩向量分块和 RISC-V 向量后端提供显式范围契约。 **风险/限制**：PR 说明相关 lowering 后续才使用这一信息；不是完整的新优化管线已经上线。

### 2.3 模型 & 技术｜[将 Attention 算子从 onnx 降低到 onnx](https://github.com/onnx/onnx-mlir/pull/3637)

**北京时间**：2026-09-18 02:28；**来源类型**：official_github_pr；**事件状态**：已合并。

在 ONNXToKrnl 转换中将 onnx.Attention 分解成基础 ONNX 算子，再降低到 Krnl；同一模式用于 onnx-to-zhigh。作者报告两条编译路径的最大绝对误差小于 1e-8、最大相对误差小于 1e-3，并通过 granite-embedding r2 模型。 **重要性**：补足 Attention 算子的基础导入与降低链路，对 ONNX 到 MLIR 部署具有直接意义。 **风险/限制**：数值结果来自所列测试，不能推断全部 Attention 变体和模型已覆盖。

### 2.4 深度洞见｜[[RFC] LLVM 成本模型监控工具/基础设施](https://discourse.llvm.org/t/rfc-llvm-cost-model-monitoring-tool-infra/91840)

**北京时间**：2026-09-17 14:44；**来源类型**：high_value_discussion_or_rfc；**事件状态**：提案讨论中。

提案为 AArch64 自动生成指令和 intrinsic 的 IR 片段，运行 opt、llc，对照 TTI 成本与 LLVM MCA 调度成本，并每日追踪变化。作者给出窄加载从三条 NEON 指令变为两条 SVE 指令、TTI 成本未同步变化的实例。 **重要性**：把向量化收益估计与真实代码生成的偏差变成可持续监控的编译器工程问题。 **风险/限制**：TTI 与 MCA 衡量对象不同，数值不一致是调查线索，并非自动判定为缺陷。

### 2.5 深度洞见｜[[RFC] 用于 LLVM 和工具的替代 CLI 库](https://discourse.llvm.org/t/rfc-replacement-cli-library-for-llvm-and-tools/91841)

**北京时间**：2026-09-17 14:54；**来源类型**：high_value_discussion_or_rfc；**事件状态**：提案讨论中。

提案以可组合的选项 registry 和独立解析状态替代旧 cl:: 接口，并探索 TableGen 生成支持；讨论澄清目标是替换旧库，社区要求论证与现有 LLVMOption/OptTable 的关系及迁移行为兼容性。 **重要性**：回应 LLVM 嵌入、可重入工具调用和选项状态管理的架构需求。 **风险/限制**：仍有是否复用现有库的设计分歧；不能把原型迁移视为上游完成替换。

### 2.6 深度洞见｜[[RFC] 向量缩放收缩](https://discourse.llvm.org/t/rfc-vector-scaled-contraction/91822/5)

**北京时间**：2026-09-17 17:49；**来源类型**：high_value_discussion_or_rfc；**事件状态**：讨论新进展。

提案增加 vector.scaled_contract，作为 linalg.scaled_contract 的寄存器级对应形式，表示额外缩放因子和混合低精度输入，并面向 MX 硬件指令降低。新增回复反对为固定 d floordiv B 的缩放映射引入半仿射映射，建议将 B 显式编码在算子中。 **重要性**：把块缩放语义如何跨 Linalg、Vector 和硬件后端传递的问题具体化。 **风险/限制**：缩放索引的表达方式仍有分歧，尚不能作为已稳定的 MLIR 算子接口使用。

### 2.7 深度洞见｜[[RFC][ClangIR] 将 CIR 管线边界变为一等驱动程序产物](https://discourse.llvm.org/t/rfc-clangir-making-cir-pipeline-boundaries-first-class-driver-artifacts/90998/10)

**北京时间**：2026-09-18 01:41；**来源类型**：high_value_discussion_or_rfc；**事件状态**：讨论新进展。

提案主线是在现有 ABI-aware CIR 输出之外，将 target/ABI lowering 前的 ABI-free CIR 暴露为驱动程序产物，并要求序列化结果摆脱 ASTContext 依赖。新增回复结合 GSoC 结果说明，pre-lowered CIR 有望让分析跨目标复用，并在更早阶段进入 GPU 方言。 **重要性**：明确 ClangIR 高层表示的工具接口与分析边界，为多目标 MLIR 管线提供设计依据。 **风险/限制**：回复表达技术方向，未宣布接口稳定或 GPU lowering 已完成；两种 CIR 形式的自包含改造进度不同。

## 三、Triton & TileLang 技术动态

### 3.1 模型 & 技术｜[[TRITONGPU] 通过小幅更新显著改善 BMM 性能](https://github.com/triton-lang/triton/pull/11822)

**北京时间**：2026-09-17 09:24；**来源类型**：official_github_pr；**事件状态**：已合并。

避免在 batch 维度重复进行 warp 计算；作者给出复现脚本，并报告该案例可能获得最高 40 倍性能改善。 **重要性**：说明线程分布造成的重复计算可成为批量矩阵乘法的主要瓶颈。 **风险/限制**：40 倍来自作者提供的特定复现案例，不是通用 BMM 或模型吞吐承诺。

### 3.2 模型 & 技术｜[[不要审阅][Op][Language][CUDA] 添加通用块缩放 GEMM 语义和后端分派](https://github.com/tile-ai/tilelang/pull/3237)

**北京时间**：2026-09-17 17:42；**来源类型**：official_github_pr；**事件状态**：已合并。

新增通用 GemmBlockScaledNode 和独立后端选择器，显式保存 SFA/SFB 与 K 偏移，语义为 C (+)= (A×SFA)@(B×SFB)。SM100 与 SM120 分派到各自指令路径；CPU、ROCm、Metal 尚无实现时拒绝该操作，避免当作 dense GEMM 静默丢失缩放因子。 **重要性**：修正低精度块缩放运算的跨后端语义边界，并区分同步公共 API 与显式异步指令 API。 **风险/限制**：本地验证在 sm_103 上执行；作者明确未实机验证 SM120。标题保留主 PR 原有的“不要审阅”限定，API 状态为已合并。

### 3.3 模型 & 技术｜[[AMD][BACEND] 修复多 CTA 场景下 TDM gather/scatter 的越界处理](https://github.com/triton-lang/triton/pull/11833)

**北京时间**：2026-09-17 23:42；**来源类型**：official_github_pr；**事件状态**：已合并。

恢复多 CTA TDM gather/scatter 的源列偏移和依赖 CTA 偏移的越界钳制，修正可能加载错误列的数据路径；作者说明这类 CTA 相关边界不能由主机侧描述符预先处理。 **重要性**：保证 AMD 多 CTA 数据搬运的索引与边界语义。 **风险/限制**：影响限定于所述 TDM 多 CTA 回归；已有修改后的测试覆盖两处失败，不应解读为所有 TDM 场景均已验证。

### 3.4 模型 & 技术｜[[Backend] 使用内联 ptx 降低 atomic_{load,store}](https://github.com/triton-lang/triton/pull/11821)

**北京时间**：2026-09-18 05:05；**来源类型**：official_github_pr；**事件状态**：已合并。

原有分支加 LLVM load/store 降低会触发误编译并使 kernel 挂起；改为内联 PTX，同时支持完整 128 位向量化原子加载与存储。 **重要性**：同时影响原子操作正确性与生成指令能力。 **风险/限制**：该修复针对所述降低路径，不代表所有并发同步或上游 LLVM 误编译问题均已解决。

## 四、RISC-V 核心新闻

### 4.1 模型 & 技术｜[重构 Hypervisor 测试套件，更新扩展支持门控控制](https://github.com/XUANTIE-RV/damo-rv-priv-ats/pull/31)

**北京时间**：2026-09-17 18:06；**来源类型**：official_github_pr；**事件状态**：已合并。

Hypervisor 测试改用由 UDB 配置导出的 EXT_AVAILABLE 能力宏，代替运行时探测 DUT 是否支持扩展；同步调整 Spike ISA 配置、Whisper 版本和测试结果。 **重要性**：让架构测试的能力声明与实际执行目标保持一致，涉及虚拟化、状态使能及向量等测试范围。 **风险/限制**：测试能力依赖 UDB 配置正确性；不意味着所有 DUT 均已通过，亦非新增 ISA 扩展已获批准。

### 4.2 模型 & 技术｜[feat(preprocess)：添加可选 V2D 后端](https://github.com/spacemit-com/model-zoo-vision/pull/82)

**北京时间**：2026-09-17 19:41；**来源类型**：official_github_pr；**事件状态**：已合并。

预处理分派扩展为 CPU、OpenCL 和可选 V2D 后端，增加 CPU/RVV tensor packing 与缓存 padding，并提供可配置回退及提交前安全失败处理。 **重要性**：覆盖视觉模型在进迭时空平台上的预处理与执行衔接，减少部署链路中只优化推理核的盲区。 **风险/限制**：V2D 是可选后端；实际可用性依赖设备与运行配置，正文未给出可推广的端到端加速数字。

### 4.3 模型 & 技术｜[[MLAS] Riscv64 rvv 内核优化](https://github.com/microsoft/onnxruntime/pull/32540)

**北京时间**：2026-09-17 22:40；**来源类型**：official_github_pr；**事件状态**：已合并。

优化 MLAS RVV 的 softmax、NCHWc 卷积、深度卷积和激活函数。作者在单核单线程、固定频率条件下报告：ResNet50 在 C920V2 上从 1398.1 ms 降至 567.3 ms，在 K3/X100 上从 816.3 ms 降至 271.6 ms，分别为 2.46 倍和 3.01 倍加速。 **重要性**：为 RISC-V CPU 上的 ONNX FP32 推理提供直接的算子与模型级优化证据。 **风险/限制**：性能测试早于最终深度卷积文件整合及最新 rebase；结果限于文中构建、模型和单核设置。

### 4.4 模型 & 技术｜[允许配置 mcountinhibit CSR。](https://github.com/riscv/sail-riscv/pull/1935)

**北京时间**：2026-09-17 23:42；**来源类型**：official_github_pr；**事件状态**：已合并。

允许单独配置 mcountinhibit 是否实现及哪些位可写，与 base.writable_hpm_counters 解耦；模型现在可表达对未实现计数器的 inhibition 配置。 **重要性**：增加 RISC-V 形式模型对实现差异的表达能力，便于架构验证对齐真实 CSR 行为。 **风险/限制**：模型配置能力扩展不等于改变所有硬件的 CSR 行为；具体平台仍须提供正确配置。

### 4.5 模型 & 技术｜[priv：澄清 WFI 对 RNMI 的行为符合预期](https://github.com/riscv/riscv-isa-manual/pull/3401)

**北京时间**：2026-09-18 02:13；**来源类型**：official_github_pr；**事件状态**：已合并。

规范明确 RNMI 始终视为局部使能：NMIE=0 时若 RNMI pending，WFI 必须恢复执行；NMIE=1 时则进入中断。NMIE 是全局使能，不改变哪些中断在局部使能。 **重要性**：为 RISC-V 中断、低功耗等待和仿真一致性提供明确的规范语义。 **风险/限制**：这是规范澄清合并，不代表已有硬件或固件实现同步更新。

### 4.6 深度洞见｜[FENCE.TIME 指令实现设计](https://github.com/OpenXiangShan/XiangShanLab/commit/1c99a7f8f8de49e3a3288bbcae1438a4162f8d17)

**北京时间**：2026-09-17 15:39；**来源类型**：official_commit；**事件状态**：设计文档已提交。

香山课程仓库新增 BPU 上下文切换刷新设计文档，定义编译期 HasBpuFlush 开关、sticky 总使能、逐预测器刷新 mask，以及 contextFlush/bpuFlushing/resetDone 握手。PHR 与 CommonHR 不占 mask 位，但每次接受的刷新事务都参与。 **重要性**：把时序隔离相关的预测器刷新范围、硬件裁剪和完成条件写成可评审的设计契约。 **风险/限制**：属于课程仓库设计材料，文档仍列出待闭环时序问题；不是香山主线已实现或 RISC-V 标准已批准该指令。

### 4.7 深度洞见｜[五问 Openchip 的 Cesc Guim 与 Marc Fernández](https://riscv.org/blog/v-questions-with-openchips-cesc-guim-and-marc-fernandez/)

**北京时间**：2026-09-18 05:59；**来源类型**：official_blog；**事件状态**：官方访谈已发表。

RISC-V International 发布 Openchip 访谈。受访者介绍 BER10 采用经验证的 IP、验证与仿真流程，首次硅片无需 respin，并在收到芯片约三周后启动操作系统；芯片面向可运行 Linux 的 64 位应用处理器场景。 **重要性**：提供欧洲 RISC-V 芯片从 IP、验证到软件 bring-up 的一手工程经验。 **风险/限制**：访谈回顾的是此前完成的硬件里程碑，不是本日芯片首发；相关进度为受访者陈述，未附独立性能测试。

### 4.8 工具 & 产品｜[小板子运行大模型：deepin 25 RVA23 版本适配进迭时空芯片 SpacemiT K3](https://www.deepin.org/zh/deepin-25-rva23-spacemit-k3/)

**北京时间**：2026-09-17 09:48；**来源类型**：official_blog；**事件状态**：已公布。

deepin 公布面向 K3 的 deepin 25 体验镜像及 Next（RVA23）仓库适配，覆盖内核、图形驱动、工具链和本地模型运行时。小U同学通过 deepin-modelhub 使用本地模型，仓库中的 GGML/llama.cpp 路径启用 RVV 与 SpacemiT 扩展支持。 **重要性**：把 RVA23 系统基线、编译工具链与桌面本地推理连成可使用的部署路径。 **风险/限制**：这是体验镜像；Next 基础库和工具链仍在持续构建与验证，不能据此认定所有 RISC-V 硬件都兼容。

## 五、AI 业界重磅

### 5.1 模型 & 技术｜[Claude 如何提升生物分子建模](https://www.anthropic.com/research/claude-uplifts-biomolecular-modeling)

**北京时间**：2026-09-17（原文仅提供日期）；**来源类型**：official_blog；**事件状态**：研究与代码已公布。

Anthropic 公布在不到四周内优化 30 多个开源生物分子模型的研究，称轻微降低精度时平均约 4 倍加速、输出相同时接近 2 倍；优化代码开源，低内存模式可在单个 GPU 节点上处理超过一万 token 的准确结构预测。 **重要性**：把专用 kernel、缓存复用与模型级优化结合，展示科学推理软件工程的可复用成果。 **风险/限制**：结果由团队报告；超过七万 token 的演示仅说明可运行推理，原文明确结构预测不准确，不能混同为有效科学结果。

### 5.2 工具 & 产品｜[如何使用 AI 智能体为仿真准备 3D 场景](https://developer.nvidia.com/blog/how-to-use-ai-agents-to-prepare-3d-scenes-for-simulation/)

**北京时间**：2026-09-17 07:20；**来源类型**：official_blog；**事件状态**：已公布。

NVIDIA 给出从 Blender 场景到可交付 OpenUSD 仿真世界的工作流：经 MCP 检查对象，补充语义标签、物理属性与传感器，使用 Omniverse Libraries 渲染检查，并通过 SimReady 验证后交给 Isaac Sim 或 Isaac Lab。 **重要性**：把机器人仿真前的数据准备转化为带工具调用和验证步骤的智能体工作流。 **风险/限制**：文章是参考流程，不提供通用成功率或自动消除仿真误差的保证；关键物理与传感器配置仍需开发者审查。

### 5.3 工具 & 产品｜[Projects 重新设计：从文件夹到对话](https://claude.com/blog/projects-redesigned)

**北京时间**：2026-09-17（原文仅提供日期）；**来源类型**：official_blog；**事件状态**：beta 分批开放。

Claude Code 的 Projects 新版以协调者管理多个线程；每个线程使用独立云端会话、分支和仓库副本，共享项目记忆。beta 首批面向使用云端会话且尚无现有网页或桌面项目的部分 Pro、Max 用户。 **重要性**：把跨仓库任务拆分、执行、审查和共享上下文整合到长期项目入口。 **风险/限制**：并发线程可能更快触及用量限制，同一代码的重叠修改仍须解决合并冲突；本地线程运行尚未上线。

### 5.4 工具 & 产品｜[推出 Astra for Law](https://openai.com/index/astra-for-law/)

**北京时间**：2026-09-17 08:00；**来源类型**：official_announcement；**事件状态**：限定准入；API 尚未开放。

OpenAI 公布 Astra for Law，将 GPT-6 Astra 与法律检索索引、专业指令和上下文结合。官方称在 200 道美国法律研究题上，最高推理强度的整体正确率为 54.0%，对照通用 Astra 加网页搜索为 38.7%。 **重要性**：展示领域检索与工作流配置对专业任务的增益，而不只是替换基础模型。 **风险/限制**：首批通过 Trusted Access 面向部分律所提供；API 仍是即将推出。该基准不足以证明法律结论可免除专业复核。

### 5.5 工具 & 产品｜[扩展 Model Builder 以直接导出 Qwen 3.8 DFlash2 包](https://github.com/microsoft/onnxruntime-genai/pull/2585)

**北京时间**：2026-09-17 15:56；**来源类型**：official_github_pr；**事件状态**：已合并。

Model Builder 可从原始 Hugging Face checkpoints 直接导出 Qwen 3.8 27B INT4 target 与 INT4 DFlash2 包，移除先导出 ONNX 再执行模型专用 graph surgery 的独立流程。可选 gate/up 融合发生在量化前，并复用原有 drafter、共享 initializer 与配置序列化链路。 **重要性**：把投机解码模型组装纳入统一导出器，减少中间图与外部权重文件的重复改写。 **风险/限制**：需要重新导出并按具体任务验证性能、精度和 KV-cache 校准；不能将示例参数视为所有设备的推荐配置。

## 六、总结与趋势观察

- **编译器与推理后端正在把“语义边界”显式化。** PyTorch 的符号形状检查、MLIR 的 match/CIR 边界、TileLang 的块缩放 GEMM 与 ONNX Attention lowering 都把原先隐含在 lowering 或运行时的约束写成可检查的 IR、分派或导出契约。
- **RISC-V 软件栈从架构规范走向可验证部署链路。** deepin K3/RVA23、ONNX Runtime RVV 优化、XUANTIE Hypervisor 能力门控和 RISC-V ISA 的 RNMI/WFI 澄清分别覆盖系统、推理、测试与规范层。
- **智能体的价值开始由“会调用工具”转向“可验证地完成长流程”。** NVIDIA 的仿真场景流程、Claude Code Projects 的并行线程与共享记忆、Anthropic 的生物分子 kernel 优化都把中间产物、验证门槛或资源限制放到工作流核心。

## 附录：信源说明

本期使用官方博客、官方公告、官方论坛/RFC、GitHub 官方 API、RISC-V/深度科技社区官方页面及有限的权威媒体发现。主要项目包括 PyTorch/ExecuTorch、LLVM/MLIR、IREE、ONNX-MLIR、Triton、TileLang、RISC-V ISA/形式模型、deepin、ONNX Runtime、Anthropic、OpenAI、Claude Code 与 NVIDIA。分领域预扫和聚合文件只用于发现，bounded scan 不代表全源穷尽；日期字段缺少时区的原文只保留原始日期，不补造具体时刻。
