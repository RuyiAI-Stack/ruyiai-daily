# Codex 技术情报每日动态（2026-11-23）

调研窗口：北京时间（2026-11-22 05:46:39.933，2026-11-23 06:00:29]。

覆盖方向：PyTorch、LLVM/MLIR、Triton & TileLang、RISC-V，以及 AI 模型研究、基础设施与产品动态。

信息口径：时间均为北京时间；区分正式发行、预发行、主干实现、RFC、补丁审阅与作者实测。性能数字保留原文的硬件、精度、形状及测试限制。

## 今日要闻

- [RISC-V 维护者提交面向 Linux 7.4-rc4 的修复集，涉及上下文切换权限、向量/浮点状态及启动内存保护。](https://lore-kernel.gnuweeb.org/linux-riscv/1a74c335-ead7-bf62-b35b-d2afd06599ca@kernel.org/)
- [OpenAI 公开内部预训练交叉熵指标，提出将困惑度用于共享评测与安全协作。](https://openai.com/index/perplexity-and-intelligence)
- [Ruyi Firmware 20261122 预发行调整 SG2042 RTC 与 UEFI 变量存储；旧 SPI0 设置升级后需重新录入，并须配套内核修复。](https://github.com/openRuyi-Project/ruyifirmware/releases/tag/20261122)

## 今日索引

- **PyTorch**：MPS 块量化矩阵乘与池化反向；大 batch 卷积、Gloo 同步及 U55 构建修复。
- **LLVM/MLIR**：uClibc-ng、向量数学库与可嵌入驱动设计；Buddy K3 模型优化、StableHLO 和向量化边界。
- **Triton & TileLang**：Intel 后端的转置折叠、广播访存计费与 BMG 缓存写策略。
- **RISC-V**：Linux 修复集、K3 eFuse 与影子栈提案；EIC7702X 实机报告、Ruyi 固件及仿真接口。
- **AI 业界**：OpenAI 困惑度研究立场；腾讯 Chatterfly Android 版内测。

## 一、PyTorch 生态核心动态

### 1.1 模型 & 技术｜[\[MPS\] Use MPP and a decode kernel for MXFP8 and NVFP4 _scaled_mm_v2 (#205759)（\[MPS\] 为 MXFP8 和 NVFP4 _scaled_mm_v2 使用 MPP 和解码内核 (#205759)）](https://github.com/pytorch/pytorch/commit/c1aec0bbeb84ca90ac5739c6452771787bee4959)

北京时间：2026-11-22 17:45:48｜来源类型：官方主干提交｜事件状态：已进入主干

**事实**：MPS 为块缩放 _scaled_mm_v2 增加少行解码、MPP 分块以及 Metal 4.1 原生 MXFP8 路径；同日基础实现加入 MXFP8/NVFP4 配方和未重排块缩放布局。 作者在 M5 Pro、bf16 输出的 M=512,N=1024,K=3072 场景报告 MXFP8 从 2.19 升至 15.1 TFLOP/s、NVFP4 从 2.44 升至 16.7；M=1,N=1024,K=3072 的两者分别为 0.66、0.78 TFLOP/s。

**重要性**：扩展 Apple GPU 运行块量化权重的算子路径，并将解码与 Prefill 分开优化。

**风险与限制**：原生 MXFP8 路径要求 Metal 4.1、稠密及对齐条件；较大 M 时仍只有文中 bf16 路径约一半吞吐，不能推成整模型加速。

相关原文：[\[MPS\] Use MPP and a decode kernel for MXFP8 and NVFP4 _scaled_mm_v2](https://github.com/pytorch/pytorch/pull/205759)；[\[MPS\] Add MXFP8 and NVFP4 recipes to _scaled_mm_v2](https://github.com/pytorch/pytorch/pull/205758)。

### 1.2 模型 & 技术｜[\[MPS\] Move avg_pool2d_backward to Metal (#202942)（\[MPS\] 将 avg_pool2d_backward 移至 Metal (#202942)）](https://github.com/pytorch/pytorch/commit/f7d1358a1e02d5689527ad18acedfe5a2de49d9b)

北京时间：2026-11-23 00:21:39｜来源类型：官方主干提交｜事件状态：已进入主干

**事实**：avg_pool2d_backward 从 MPSGraph 移到 Metal gather 内核，每线程计算四列 grad_input，并提供大偏移的 64 位索引变体。 作者列出的 38 个算子用例加速比几何平均为 2.34 倍，范围 0.90—22.55 倍；int32、int16、int8、uint8 和 bool 输入现抛出 NotImplementedError，与 CPU/CUDA 行为对齐。

**重要性**：改善 MPS 平均池化反向路径，并带来需要注意的输入类型兼容变化。

**风险与限制**：结果是列举形状与精度的算子测试，部分用例退化；不能等同训练端到端收益。

相关原文：[\[MPS\] Move avg_pool2d_backward to Metal](https://github.com/pytorch/pytorch/pull/202942)。

### 1.3 模型 & 技术｜[\[MPS\] Fix conv2d and conv_transpose2d for batch sizes above 65535 (#206032)（\[MPS\] 修复 batch 大小超过 65535 时的 conv2d 和 conv_transpose2d (#206032)）](https://github.com/pytorch/pytorch/commit/ee0eed2fcd35650303548025d2378bbe2ca44a91)

北京时间：2026-11-22 20:56:10｜来源类型：官方主干提交｜事件状态：已进入主干

**事实**：MPSGraph 的二维卷积在 batch 超过 65535 时会得到错误结果；主干将前向和输入梯度路径按 65535 分块，每个前向块直接写入输出。

**重要性**：修正大 batch 场景的静默数值错误，涉及 conv2d 和转置卷积。

**风险与限制**：修复限定于相关 MPSGraph 路径，不能据此宣称所有大张量算子均已支持。

相关原文：[\[MPS\] Fix conv2d and conv_transpose2d for batch sizes above 65535](https://github.com/pytorch/pytorch/pull/206032)。

### 1.4 模型 & 技术｜[\[c10d\] Fix Gloo runLoop deadlock: release finished works outside workMutex_ (#206050) (#206050)（\[c10d\] 修复 Gloo runLoop 死锁：在 workMutex_ 外释放已完成工作 (#206050) (#206050)）](https://github.com/pytorch/pytorch/commit/60a7fa585a70d2daefe9f3ab7889fbcc89f1cee2)

北京时间：2026-11-22 12:48:02｜来源类型：官方主干提交｜事件状态：已进入主干

**事实**：ProcessGroupGloo 完成异步工作时，在 workMutex_ 内析构对象可能等待 GIL；Python dispatcher 路径又可能持有 GIL 等待该互斥量，形成死锁。 变更在锁内移出引用、解锁后释放，再重新加锁；原文给出生产 PT2 编译挂起栈及无需 GPU 的双 rank 复现。

**重要性**：消除分布式编译同步中可导致作业永久挂起的锁顺序问题。

**风险与限制**：原文针对特定析构与 Python dispatcher 交错，不代表解决所有 Gloo 超时或分布式训练挂起。

相关原文：[\[c10d\] Fix Gloo runLoop deadlock: release finished works outside workMutex_ (#206050)](https://github.com/pytorch/pytorch/pull/206050)。

### 1.5 工具 & 产品｜[Use short diagnostic function names in Arm GNU builds（在 Arm GNU 构建中使用简短诊断函数名）](https://github.com/pytorch/executorch/pull/25423)

北京时间：2026-11-22 15:13:22｜来源类型：官方实现 PR｜事件状态：已合入

**事实**：GCC 15.2 生成的实例化函数名使 U55 portable runner 只读数据增加 38,600 字节，耗尽 DTCM 并使堆栈区域重叠，导致链接失败。 Arm GNU 工具链设置现有 ET_USE_BUILTIN_FUNCTION_NAME=0，改用简短 __FUNCTION__；诊断消息、文件行号和运行时检查保留。

**重要性**：恢复受内存约束的 Ethos-U55 部署构建链路。

**风险与限制**：这是指定工具链与 runner 的体积修复，没有性能提升或新硬件支持主张。


## 二、LLVM/MLIR 最新进展

### 2.1 模型 & 技术｜[\[RFC\] uClibc-ng support（\[RFC\] uClibc-ng 支持）](https://discourse.llvm.org/t/92940/1)

北京时间：2026-11-22 19:08:40｜来源类型：官方技术论坛｜事件状态：RFC 已提出

**事实**：uClibc-ng 维护者提出 LLVM 上游支持 RFC：目前用未修改 LLVM 和 glibc triple 构建，已能构建 aarch64、arm、mips、riscv64、x86_64，并通过 libunwind/compiler-rt 提供不依赖 GCC/libgcc 的 NPTL。 关联 sanitizer 提案为 uClibc-ng 引入独立平台判断、信号结构与 getauxval 等兼容处理；作者报告 ARMv7-A 实机 AddressSanitizer 检出堆越界，要求 uClibc-ng 至少 1.0.58。

**重要性**：为嵌入式 Linux 和 RISC-V 的 LLVM 工具链形成明确上游协作入口。

**风险与限制**：RFC 在征询范围与下一步，关联实现 PR 尚未合入；实机验证是 ARM 平台，不能扩大为 RISC-V sanitizer 已完成实机验证。

### 2.2 模型 & 技术｜[\[RFC\]\[MATHVEC\] Introduce the LLVM's mathvec library as yet another veclib functions provider（\[RFC\]\[MATHVEC\] 将 LLVM 的 mathvec 库引入为又一种 veclib 函数提供者）](https://discourse.llvm.org/t/92591/10)

北京时间：2026-11-23 04:35:30｜来源类型：官方技术论坛｜事件状态：讨论新进展

**事实**：提案希望让 Clang 显式选择 LLVM libc 的向量数学库，而不必借用 libmvec 的映射。11 月 22 日作者将选项名从 -fveclib=mathvec 改为 -fveclib=llvm，并移除 sqrt 的函数映射。

**重要性**：缩小向量数学库接入接口与映射范围，对工具链配置具有直接意义。

**风险与限制**：这是提案修订，尚非正式发行；不推导向量函数性能或平台覆盖。

相关原文：[同主题回复](https://discourse.llvm.org/t/92591/9)。

### 2.3 深度洞见｜[\[RFC\] Embeddable LLVM tool drivers for long-lived hosts（\[RFC\] 面向长生命周期宿主的可嵌入 LLVM 工具驱动）](https://discourse.llvm.org/t/91754/28)

北京时间：2026-11-23 01:29:00｜来源类型：官方技术论坛｜事件状态：讨论新进展

**事实**：原提案希望把 llc 等命令行驱动变成可供浏览器、JupyterLite 等长生命周期宿主调用的库，并由宿主管理 LLVM 生命周期。 新增回复提出撤回尚未随正式版本发行的 LLVMLlcDriver：同时链接 LLVMOptDriver 会因重复注册选项 o 在启动时中止，llcMain 的错误处理仍会 exit(1)，也缺少头文件和扩展点。回复建议直接使用 TargetMachine 代码生成接口或 runCodeGenPipeline。

**重要性**：明确可嵌入编译服务不能直接依赖测试工具的进程生命周期及命令行全局状态。

**风险与限制**：撤回仍是提议，讨论未形成共识；不能写成稳定库已正式撤销。

### 2.4 深度洞见｜[\[RFC\] Communicating Array Extents to LLVM Using llvm.assume.chained（\[RFC\] 使用 llvm.assume.chained 向 LLVM 传递数组维度大小）](https://discourse.llvm.org/t/92820/12)

北京时间：2026-11-23 00:39:14｜来源类型：官方技术论坛｜事件状态：讨论新进展

**事实**：RFC 提议用返回原指针的 llvm.assume.chained，将 Flang 数组维度与步长交给去线性化和依赖分析，并讨论其相对既有 assume 与元数据的编译成本。 本日回复用 Z3 检查动机示例，认为 galgel 所需底层数组大小已在 IR 中，AIR 的 MXPx<=150 可由既有 llvm.assume 表达，建议先探索现有分析或元数据加循环版本化。

**重要性**：为是否新增 IR 语义提供具体的替代设计与验证问题。

**风险与限制**：回复仅给出理论可证明性，尚未完成 LLVM 分析实现或新的运行性能比较；不将其写成推翻提案的结论。

### 2.5 深度洞见｜[\[RFC\]\[LTO\] Make hosts responsible for loading plugins（\[RFC\]\[LTO\] 让宿主负责加载插件）](https://discourse.llvm.org/t/92892/2)

北京时间：2026-11-22 08:33:03｜来源类型：官方技术论坛｜事件状态：讨论新进展

**事实**：RFC 提议让宿主提前加载、配置并维持插件生命周期，通过 LoadedPassPlugins 交给 LTO，替代由 LTO 按文件名自行加载。 新增回复支持由宿主接管 ParseArguments；建议直接移除 PassPluginFilenames，避免弃用属性连带改变 lto::Config 的复制能力。回复还明确 PassPluginArgs 应只参与缓存键。

**重要性**：澄清编译服务与 ThinLTO 集成的接口迁移和兼容影响。

**风险与限制**：仍属讨论建议，不能写成接口已删除；下游需要关注最终合入的 API 形态。

### 2.6 模型 & 技术｜[\[midend\]\[models\] Optimize DeepSeek-R1-Distill-Qwen-1.5B FP16 Prefill and Decode on K3 A100（\[midend\]\[models\] 在 K3 A100 上优化 DeepSeek-R1-Distill-Qwen-1.5B FP16 Prefill 和 Decode）](https://github.com/buddy-compiler/buddy-mlir/pull/997)

北京时间：2026-11-23 00:25:46｜来源类型：官方实现 PR｜事件状态：已合入

**事实**：Buddy 的 K3 配置将 FP32 累加器保留在寄存器中，Prefill 使用 8×32 IME 块，Decode 按 32 输出通道组织，并为选定投影加入 N32/K8 权重打包、VLEN=1024 向量化及元素融合。 作者在 K3 A100 核 8—15、八线程下报告：Buddy Prefill 从 4.53 提升到 43.42 token/s，Decode 从 5.88 到 6.06；相同输入的 llama.cpp 配置为 27.95 和 3.90 token/s。

**重要性**：直接推进 Buddy Compiler 的 RISC-V FP16 模型编译与部署链路。

**风险与限制**：Prefill 测 128 个位置，Decode 为位置 64 起的 32 步 teacher-forced 测量，排除加载、每引擎四样本并交替测试；不能当作交互式生成或其他模型收益。

### 2.7 模型 & 技术｜[Validate element counts and dense buffer sizes in StablehloAggressiveFolder.（在 StablehloAggressiveFolder 中验证元素数量和稠密缓冲区大小。）](https://github.com/openxla/stablehlo/pull/3079)

北京时间：2026-11-22 06:14:03｜来源类型：官方实现 PR｜事件状态：已合入

**事实**：StableHLO 激进常量折叠改用受检查的元素数计算，拒绝静态形状乘积溢出；非 splat 稠密常量在索引前验证原始缓冲区字节数，零元素 slice/transpose 提前返回。

**重要性**：强化模型导入与常量折叠对异常形状和缓冲区的正确性边界。

**风险与限制**：是编译时防护，没有公布 CVE、攻击可利用性或推理速度提升。

### 2.8 模型 & 技术｜[\[mlir\]\[linalg\]\[vector\] Mark transfers in-bounds when no mask is needed（\[mlir\]\[linalg\]\[vector\] 无需掩码时将 transfer 标记为界内）](https://github.com/llvm/llvm-project/pull/244179)

北京时间：2026-11-22 23:06:15｜来源类型：官方实现 PR｜事件状态：已合入

**事实**：Linalg 向量化在确认无需 mask 时，为 vector.transfer_read/write 的所有 transfer 维度设置 in_bounds=true，而不再只依赖动态维度与向量大小相同的特殊分支。

**重要性**：使后续 lowering 能使用已有界内证明，避免重新生成无必要的边界掩码；与 MLIR/Linalg 编译链路直接相关。

**风险与限制**：前提是向量化器已确定无需掩码；该修改不是放宽越界访问规则，也没有端到端性能数据。

### 2.9 模型 & 技术｜[PHIElim: Trim PHI source subranges without a full shrinkToUses（PHIElim：无需完整 shrinkToUses 即可修剪 PHI 源子区间）](https://github.com/llvm/llvm-project/pull/228488)

北京时间：2026-11-22 08:16:34｜来源类型：官方实现 PR｜事件状态：已合入

**事实**：PHI 消除改为追踪逐 lane 插入的副本，在最后读取点修剪对应 live-range 片段，并限制在前驱块边界，替代对每个子区间全部使用点的重扫。 作者报告压力测试中该 pass 单独快 2—4 倍，640 个使用点的示例快 7.2 倍；补充测试覆盖前驱间存活信息与非索引顺序插入。

**重要性**：降低编译器寄存器活跃区间维护成本，同时保留细粒度正确性约束。

**风险与限制**：数字来自粗略压力测试与 pass 隔离测量，不能换算为整个编译流程或应用运行速度。


## 三、Triton & TileLang 技术动态

### 3.1 模型 & 技术｜[\[Intel\] Fold tl.trans of a computed tile into its producers（\[Intel\] 将已计算 tile 的 tl.trans 折叠进其生产者）](https://github.com/intel/intel-xpu-backend-for-triton/pull/8786)

北京时间：2026-11-22 20:18:59｜来源类型：官方实现 PR｜事件状态：已合入

**事实**：Intel 后端在 DPAS 布局选择之前重写计算 tile 的转置，将转置传播到 dot 与元素运算的生产者，使注意力反向的部分中间值留在寄存器中。 作者对 tutorial 06 的 dK/dV 循环测 96 组配置：无掩码/有掩码的几何平均加速分别为 B580 1.47/1.50 倍、B70 1.66/1.71 倍、PVC 1.53/1.49 倍。

**重要性**：减少注意力反向的共享内存搬运和布局转换。

**风险与限制**：仅对满足 DPAS 与 2D block IO 条件的区域生效；B580 某一最佳配置因寄存器溢出反而为 0.90/0.93 倍，不是普遍提升。

### 3.2 模型 & 技术｜[\[RemoveLayoutConversions\] Cost broadcast loads by distinct addresses（\[RemoveLayoutConversions\] 按不同地址数估算广播加载成本）](https://github.com/intel/intel-xpu-backend-for-triton/pull/8888)

北京时间：2026-11-22 22:14:57｜来源类型：官方实现 PR｜事件状态：已合入

**事实**：后端不再把 GEMM 偏置广播后的逻辑 tile 元素数直接当作访存成本，而在可证明部分维度均匀、结果接入 DPAS 时按不同地址数计费，使相关 load/store 与累加布局合并。 文中两组 FP16 addmm 形状采用 64×128×64 tile、16 warps：CRI 原始 epilogue 快 2.2%，三款 BMG 的 CI 驱动环境快 1.10—1.19 倍。

**重要性**：修正成本模型，可消除两次经共享内存进行的转换及屏障。

**风险与限制**：保留现有阈值；数字为进程间交错测量的设备时间，使用 256 MiB cache flush，不等同整模型吞吐。

### 3.3 模型 & 技术｜[\[BlockIO\] Write narrow-row 2D block stores back through L1 on BMG（\[BlockIO\] 在 BMG 上通过 L1 回写窄行二维块存储）](https://github.com/intel/intel-xpu-backend-for-triton/pull/8660)

北京时间：2026-11-22 16:22:48｜来源类型：官方实现 PR｜事件状态：已合入

**事实**：BMG 对符合条件的窄行 2D block store 启用 L1 write-back，让同一工作组写入的部分 cache line 先合并，减少直接写入 L2 的开销。 Arc B570 上 FP16 输出、4096² 且 K=64 的 GEMM 报告 1.95 倍；torch.compile 固定配置的同形状 mm 为 FP16 1.71 倍、bf16 1.74 倍。

**重要性**：把缓存写策略与编译器对模块内跨工作组通信的判断结合。

**风险与限制**：仅启用 BMG；含原子、volatile/.cv 读取、内联汇编或非纯外部调用时保留默认策略，以避免 L1 数据对其他工作组不可见。


## 四、RISC-V 核心新闻

### 4.1 重磅｜[\[GIT PULL\] RISC-V updates for v7.4-rc4（\[GIT PULL\] 面向 v7.4-rc4 的 RISC-V 更新）](https://lore-kernel.gnuweeb.org/linux-riscv/1a74c335-ead7-bf62-b35b-d2afd06599ca@kernel.org/)

北京时间：2026-11-23 04:11:58｜来源类型：上游技术邮件｜事件状态：维护者拉取请求已发出

**事实**：维护者提交面向 Linux v7.4-rc4 的 RISC-V 修复拉取请求，包含上下文切换的 SR_SUM 权限泄漏、向量寄存器上下文处理中 regs->status 读改写保护，以及早期启动 DTB 预留修复。 同时修复只读内核页映射定义，并从默认配置去掉 COMPAT_BRK 以启用堆地址随机化。

**重要性**：涵盖权限隔离、浮点状态与启动内存正确性，对 RISC-V 系统软件有直接影响。

**风险与限制**：原信是维护者向主线提交的拉取请求，不能据此认定 v7.4-rc4 已发行或所有下游内核已包含修复。

### 4.2 模型 & 技术｜[\[PATCH v3 0/3\] Add read-only SpacemiT K3 eFuse support（\[PATCH v3 0/3\] 添加只读 SpacemiT K3 eFuse 支持）](https://lore-kernel.gnuweeb.org/linux-riscv/20261122173810.2409146-1-duge1256@gmail.com/)

北京时间：2026-11-23 01:38:07｜来源类型：上游技术邮件｜事件状态：补丁已提交审阅

**事实**：v3 继续为 K3 eFuse bank 0 与 bank 7 提供只读 NVMEM 接入，本轮改用 512 字节地址空间及其他 bank 的 keepout，并采用独占托管复位、在卸载或后续 probe 失败时拉起复位。 作者在 K3 COM260 Kit 上重复读取并执行五次 unbind/rebind；配合独立 thermal 系列，七个温度通道在每次重新绑定后恢复上报。

**重要性**：明确温度校准数据的系统接入与复位生命周期约束。

**风险与限制**：是修订后的上游补丁，非新芯片能力；不支持烧写 eFuse，温度结果依赖另一组 thermal 补丁。

### 4.3 模型 & 技术｜[\[PATCH 1/1\] riscv: usercfi: bind signal shadow-stack restore to the active shadow stack（\[PATCH 1/1\] riscv: usercfi: 将信号影子栈恢复绑定至活动影子栈）](https://lore-kernel.gnuweeb.org/linux-riscv/tencent_7D0AC8F176A26AB26D61F6A8@qq.com/)

北京时间：2026-11-23 02:17:59｜来源类型：上游技术邮件｜事件状态：补丁已提交审阅

**事实**：作者报告 RISC-V Zicfiss 信号恢复只验证用户帧所指 token 的值关系，未与内核记录的活动影子栈位置绑定；提出增加位置比较，失配返回 -EINVAL。 作者在加固测试目标上报告旧 token 重放：未修补时完成侦察的 67/67 轮成功，补丁后完成侦察的 52/52 轮被阻止，同时检查正常与嵌套信号等路径。

**重要性**：为信号恢复与影子栈完整性提出具体的验证条件和修复方案。

**风险与限制**：这是作者公开报告与补丁，尚无维护者确认或正式发行；攻击前提已有可破坏信号帧的内存破坏能力，样本数字不是独立复测。

### 4.4 深度洞见｜[clk: eswin: eic7700: EIC7702X hangs when disabling unused D2D/PVT gates（clk: eswin: eic7700: 禁用未使用的 D2D/PVT 门控时 EIC7702X 挂起）](https://lore-kernel.gnuweeb.org/linux-riscv/20261122191830.2183513-1-mervynpretorius.oss@gmail.com/)

北京时间：2026-11-23 03:18:30｜来源类型：上游技术邮件｜事件状态：实机故障报告与局部验证已公开

**事实**：开发者在 DeepComputing FML13V03、EIC7702X 双 die、32 GiB 板上报告，未使用时钟关闭阶段禁用 D2D 门控会阻止启动；局部将两个门控标记 CLK_IS_CRITICAL 后能进入 initramfs。 原信给出 Linux v7.4-rc3 加 EIC7700 v5 系列的环境，并报告读取和写回 27×1 GiB tmpfs 校验一致；PVT 门控的故障仍有间歇性。

**重要性**：为 EIC7702X 主线上板指出 D2D 链路和时钟所有权问题，提供具体复现条件。

**风险与限制**：局部修正仍在征询正确建模方式；未测单 die、休眠关机、第二 die 的 CPU 核或完整系统工作负载。

### 4.5 工具 & 产品｜[Ruyi Firmware 20261122（Ruyi 固件 20261122）](https://github.com/openRuyi-Project/ruyifirmware/releases/tag/20261122)

北京时间：2026-11-22 23:25:57｜来源类型：官方预发行 Release｜事件状态：预发行已发布

**事实**：SG2042 的 UEFI 在启动时读取一次硬件 RTC，随后提供虚拟时钟；持久 UEFI 变量移到 SPI1，把 SPI0 留给 Linux。 发行说明要求旧设置保存在 SPI0 的用户升级前记录启动顺序和 Setup 设置，升级后重新录入；持久变量还要求单独更新到包含 EFI runtime MMIO 与 SG2042 时钟保护修复的内核。

**重要性**：改变 Pioneer 与 SG2042 EVB 的固件状态持久化与操作系统交接，升级有明确兼容要求。

**风险与限制**：仅覆盖支持的板卡；时钟更新需用 Linux RTC 驱动，K3 主线支持仍为 best effort，固件升级不会自动更新操作系统内核。

### 4.6 工具 & 产品｜[gsim 2026-11-22 adfff02（gsim 2026-11-22 adfff02）](https://github.com/OpenXiangShan/gsim/releases/tag/master-2026-11-22-adfff02)

北京时间：2026-11-22 10:16:28｜来源类型：正式 Release｜事件状态：正式发行

**事实**：gsim 发行包含顶层向量 IO setter/getter 修正：数组输入按数组复制并触发后续节点，数组输出返回匹配的指针或多维数组类型，标量路径保持单值接口。 原 PR 给出 FIRRTL 的 1/2/8/16 位数组与二维数组示例及生成的 C++ 接口。

**重要性**：修正香山仿真工具的顶层向量接口语义，影响测试平台与生成模型的连接。

**风险与限制**：示例验证范围是列出的 IO 类型；同日另一个自动发行只增加 aliasAnalysis 注释，未据此宣称模拟器新增架构能力。

相关原文：[Fix: Correct top-level vector IO setter/getter](https://github.com/OpenXiangShan/gsim/pull/162)。

### 4.7 工具 & 产品｜[feat(sun252i-f101): select 16 or 32 MiB PSRAM for S3 and S4（feat(sun252i-f101): 为 S3 和 S4 选择 16 或 32 MiB PSRAM）](https://github.com/YuzukiHD/esp-idf/commit/44b407bfa5bd8aade9c181708ae96b558f428b9d)

北京时间：2026-11-22 23:32:54｜来源类型：官方产品分支提交｜事件状态：产品分支已更新

**事实**：Yuzuki 的 ESP-IDF sun252i-f101 分支增加 CONFIG_ESP_F101_PSRAM_MB，统一控制链接内存区域和运行时堆区域，匹配 F101S3 的 16 MiB 与 F101S4 的 32 MiB PSRAM。

**重要性**：为不同容量 F101 板卡提供明确的镜像与内存布局适配，属于部署配置能力变化。

**风险与限制**：配置必须匹配实际芯片及 SyterKit 初始化结果；32 MiB 镜像用于 16 MiB 板卡会越过物理内存边界。这是厂商分支，不能写成 Espressif 上游支持全志芯片。


## 五、AI 业界重磅

### 5.1 深度洞见｜[Perplexity and Intelligence（困惑度与智能）](https://openai.com/index/perplexity-and-intelligence)

北京时间：2026-11-22 08:00:00｜来源类型：官方博客｜事件状态：研究立场文章已发布

**事实**：OpenAI 首席科学家 Jakub Pachocki 公开内部预训练交叉熵指标，并倡议开发者将困惑度纳入共享评测、安全透明度与研发节奏讨论。 文中把 GPT-3.5、GPT-4 在内部固定代码数据快照上的 log-perplexity 分别列为约 0.99、0.91，并讨论低于 0.8 后出现的新能力。

**重要性**：提供观察预训练进展和跨实验室安全协作的官方研究主张。

**风险与限制**：内部数据集不可直接横向比较，后训练能力也未被该指标覆盖；对未来能力与阈值的判断是作者观点，尚非共同标准。

### 5.2 工具 & 产品｜[腾讯 Chatterfly 输入法安卓版开启内测，支持语音转写、自动翻译等 AI 功能](https://www.ithome.com/1/030/576.htm)

北京时间：2026-11-22 14:32:45｜来源类型：权威媒体原始报道｜事件状态：Android 版内测已开启

**事实**：IT之家原始报道确认腾讯 Chatterfly 输入法 Android 版开启内测，支持语音转写、翻译、按场景润色与语音下达文本任务；iOS 页面仍标为即将发布。

**重要性**：将 AI 语音和表达辅助整合到跨应用输入入口。

**风险与限制**：目前是内测，未提供独立识别准确率或延迟对比；iOS 尚未发布，官网功能说明也不等同所有用户可用。

相关原文：[官方产品页面](https://chatterfly.tencent.com/)。

## 六、总结与趋势观察

- **低精度与端侧优化正在分解为更具体的执行路径。** [Buddy 的 K3 FP16 Prefill/Decode 优化](https://github.com/buddy-compiler/buddy-mlir/pull/997)与 [MPS 块量化矩阵乘](https://github.com/pytorch/pytorch/commit/c1aec0bbeb84ca90ac5739c6452771787bee4959)都按阶段、形状和数据布局选择内核；收益需要结合各自精度、硬件和测量边界理解。

- **编译器接口和运行边界仍影响部署可靠性。** [LLVM 可嵌入驱动讨论](https://discourse.llvm.org/t/91754/28)揭示命令行全局状态与宿主生命周期冲突，[Gloo 死锁修复](https://github.com/pytorch/pytorch/commit/60a7fa585a70d2daefe9f3ab7889fbcc89f1cee2)则处理 Python GIL 与原生互斥锁的交错；两者都要求在组件边界明确资源所有权。

- **RISC-V 上板正在把持久状态与资源生命周期显式化。** [SG2042 固件的 RTC/SPI1 交接](https://github.com/openRuyi-Project/ruyifirmware/releases/tag/20261122)和 [K3 eFuse v3 的复位管理](https://lore-kernel.gnuweeb.org/linux-riscv/20261122173810.2409146-1-duge1256@gmail.com/)分别约束跨启动持久状态与驱动重绑定；前者已发布预发行版本，后者仍在审阅。

## 附录：信源说明

本期主要依据项目官方博客、上游邮件与论坛、正式 Release 与预发行、GitHub 官方 API 的 PR/主干提交，以及有编辑责任的媒体原始报道。主要项目包括 PyTorch/ExecuTorch、LLVM/MLIR、Buddy Compiler、StableHLO、Intel Triton 后端、Linux RISC-V、openRuyi 与香山工具链。

邮件与 RFC 表示提议或讨论状态，Release 的正式发行与预发行状态分别标明；Ruyi Firmware 20261122 为已发布的预发行版本。主干与产品分支变化不等于稳定版本全面可用。作者性能与故障复现数据只适用于原文所述环境；硬件、发行版和下游集成状态需分别确认。
