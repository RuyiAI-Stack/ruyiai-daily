# Codex 技术情报每日动态（2026-09-30）

调研窗口：北京时间（2026-09-29 05:50:31.673，2026-09-30 06:00:22]。

覆盖方向：PyTorch 生态、LLVM/MLIR、Triton 与 TileLang、RISC-V，以及 AI 模型、推理基础设施与产业动态。

信息口径：采用项目官方公告、博客、原始设计讨论与实现记录；性能数据保留实验条件，提案、预览能力和正式发布分别表述。

## 今日要闻

- [OpenAI 发布 GPT-6.1 Sol，面向编码、专业任务与计算机使用；标准 API 输入和输出价格分别为每百万 token 2 美元和 10 美元。](https://openai.com/index/introducing-gpt-6-1-sol/)

- [MLIR 提出多结果操作的部分折叠 API，计划显式表达结果替换与原地修改，减少下游迁移中的静默语义变化。](https://discourse.llvm.org/t/91954/1)

- [NVIDIA 开放 Kumo Tabular 表格基础模型及推理库，以预训练和上下文学习支持分类与回归。](https://huggingface.co/blog/nvidia/kumo-tabular)

- [BorrowSanitizer 的 Rust 侧 MCP 已提出，计划把 LLVM 插桩与运行库作为 nightly 实验组件分发。](https://discourse.llvm.org/t/90938/23)

- [OpenAI 开始推出拥有云电脑的 dots 持续任务智能体，后台自主研究使用受限只读工具。](https://openai.com/index/introducing-dots/)

- [OpenClaw Enterprise 开放企业智能体控制平台，提供多租户、隔离和审计能力，目前定位内部试点。](https://openclaw.ai/blog/openclaw-enterprise)

## 今日索引

- **PyTorch**：64 位资源偏移、DSP 掩码性能、QNN Android 发行构建，以及 TensorRT 的 KV 缓存与显存复用。

- **LLVM/MLIR**：部分折叠、Flang 循环版本化、Arc/Wasm 设计讨论、ONNX lowering、BOSC AME 后端与 BorrowSanitizer。

- **Triton & TileLang**：AMD 编译标志性能回退、CDNA4 布局边界、异步共享内存正确性与 Intel GRF 重编译策略。

- **RISC-V**：MAG 与影子栈语义、Hypervisor 中断模型、跨边界访存验证、陷阱重定位和 SG2044 固件升级包。

- **AI 业界**：GPT-6.1 Sol、dots、Kumo Tabular、VSS 3.3、TensorRT Model Connect 工程实践、OpenClaw Enterprise 与 MTP 性能研究。

## 一、PyTorch 生态核心动态

### 1.1 模型 & 技术｜[为 DataLoader 提供 64 位源偏移量 (#23168)](https://github.com/pytorch/executorch/pull/23168)

北京时间：2026-09-30 04:45:57｜来源类型：官方 GitHub PR｜事件状态：已合并

ExecuTorch 为 DataLoader 增加 64 位源偏移 API，并用于 FlatTensor/PTD 读取，解决 32 位目标上超过 4 GiB 的文件偏移被 size_t 截断的问题。

**重要性**：让受限地址空间设备可以读取大模型资源文件中位置靠后的独立资源，补齐部署数据通路。

**限制与风险**：缓冲区大小仍为 size_t；这不支持单个资源大于设备地址空间。

### 1.2 模型 & 技术｜[将数据指针移出 sdpa\_bitwise\_mask\_gen 打包循环 (#23129)](https://github.com/pytorch/executorch/pull/23129)

北京时间：2026-09-30 04:00:43｜来源类型：官方 GitHub PR｜事件状态：已合并

Cadence 通用 SDPA 掩码打包内核将数据指针访问移到内层循环之外，保持位布局、极性与输入校验。提交者报告在 Jazz DSP 的 ISS 上，真实内核对 bool 和 float 输入分别加速 1.66 倍和 1.41 倍。

**重要性**：减少注意力掩码准备阶段的重复访问开销，对 DSP 推理路径具有可测量意义。

**限制与风险**：早期独立循环代理模型的 7.95 倍结果不是实际内核收益；功能虚拟平台的耗时也不能作为周期精确测量。

### 1.3 模型 & 技术｜[feat(executorch): 让 TensorRT 引擎原地写入调用方的 KV 缓存](https://github.com/pytorch/TensorRT/pull/4740)

北京时间：2026-09-30 01:29:38｜来源类型：官方 GitHub PR｜事件状态：已合并

Torch-TensorRT 增加 zero_copy_kv 选项，在图分区前重连缓存写入，并在 lowering 时移除 staging，使引擎直接更新调用方缓存；save 同时执行两步并检查最终程序。

**重要性**：避免逐层解码缓存往返复制，连接模型导出、别名分析与 ExecuTorch 运行时。

**限制与风险**：选项需要主动开启；手动拆分导出与 lowering 时必须同时完成两步，否则可能丢失缓存更新。验证仅覆盖提交列出的 GPU 环境。

### 1.4 模型 & 技术｜[perf(executorch): 可选的按设备共享激活临时缓冲池](https://github.com/pytorch/TensorRT/pull/4739)

北京时间：2026-09-30 01:27:06｜来源类型：官方 GitHub PR｜事件状态：已合并

ExecuTorch TensorRT delegate 新增默认关闭的共享 activation scratch 池，每个设备保留一个按需求增长的缓冲区，供顺序提交的引擎复用。

**重要性**：缓解逐层引擎各自长期持有临时显存造成的内存增长。

**限制与风险**：同设备调用会排序，扩容可能等待在途 GPU 工作；启用时不支持 CUDA Graph 捕获，Windows 和 wheel 构建尚未验证。

### 1.5 模型 & 技术｜[修复 QNN Android Maven 发行构建](https://github.com/pytorch/executorch/pull/23234)

北京时间：2026-09-29 23:04:58｜来源类型：官方 GitHub PR｜事件状态：已合并

ExecuTorch 修复 1.5.1 QNN Maven 发布在 AAR 组装前失败的问题，去掉与新 PyTorch 头文件冲突、且不属于 Android 产物的主机工具构建；QNN SDK 安装与 AAR 中运行时构建保留。

**重要性**：解除 Android QNN 发行链路阻断，已有版本可重新执行发布流程。

**限制与风险**：合并不等于 Maven 产物已经重新上传；提交中的验证是语法与差异检查。

## 二、LLVM/MLIR 最新进展

### 2.1 模型 & 技术｜[\[RFC\]\[flang\] 当运行时范围等于声明范围时，为数组切片循环生成多版本](https://discourse.llvm.org/t/91956/1)

北京时间：2026-09-30 04:20:52｜来源类型：项目官方讨论区｜事件状态：提案已发布

提案扩展 fir::LoopVersioning：当运行时切片范围等于编译期声明范围时，生成常量迭代次数的快速分支，否则保留原循环。作者的 SPEC CPU2017 527.cam4_r 原型采样总数由 20,425 降至 19,253。

**重要性**：利用 FIR 尚保留的形状信息，让小循环更容易完全展开，减少短 memset 和循环控制开销。

**限制与风险**：采样数减少 5.7% 不等于统一条件下端到端加速 5.7%；默认范围上限 32 尚待进一步测量，代码体积可能增加。

### 2.2 模型 & 技术｜[\[RFC\] 多结果操作的部分折叠](https://discourse.llvm.org/t/91954/1)

北京时间：2026-09-30 01:41:27｜来源类型：项目官方讨论区｜事件状态：提案已发布并有讨论进展

MLIR 提案引入 OpFoldResults，以逐结果槽位和原地修改标志表达部分折叠；保持单结果 API，计划分阶段迁移 dialect、驱动与调用方。新讨论集中于明确表达“原地修改并替换部分结果”，避免用旧返回值编码新语义造成静默兼容性问题。

**重要性**：涉及 MLIR 通用变换契约，对 Buddy Compiler 等下游 dialect 的折叠实现与迁移有直接参考价值。

**限制与风险**：仍是 RFC 和草案实现；四周默认切换及后续移除旧接口是提议节奏，不是已经生效的期限。

相关原文：[同主题回复 2](https://discourse.llvm.org/t/91954/2)；[同主题回复 3](https://discourse.llvm.org/t/91954/3)。

### 2.3 模型 & 技术｜[添加 col2im 的 onnx-to-krnl lowering](https://github.com/onnx/onnx-mlir/pull/3633)

北京时间：2026-09-29 17:44:50｜来源类型：官方 GitHub PR｜事件状态：已合并

ONNX-MLIR 实现 opset 18 Col2Im 到 Krnl 的转换，通过 image_shape、block_shape、dilations、pads 与 strides 计算输出位置，并增加空间维度和步长验证。

**重要性**：让含 Col2Im 的模型进入端到端编译路径，补充模型导入后的算子覆盖。

**限制与风险**：这是指定操作的 lowering；动态形状信息不足时仍需后续形状推导，不代表所有模型都已可编译。

### 2.4 模型 & 技术｜[BOSC: bosc-ame-instruction-update](https://github.com/RuyiAI-Stack/llvm-project/pull/29)

北京时间：2026-09-29 14:07:03｜来源类型：官方 GitHub PR｜事件状态：已合并

RuyiAI 的 LLVM 分支扩充 BOSC AME 矩阵指令：增加类型、步距和填充值配置 intrinsic、向量及 unfold/fold 访存形式，并扩展矩阵乘累加编码与累加寄存器类型映射。

**重要性**：这是团队维护分支的原创后端实现，对 Buddy Compiler 到 BOSC 矩阵硬件的代码生成接口有直接意义。

**限制与风险**：仅表示 RuyiAI 分支已合并；通用 Clang/OpenMP 回归结果不能代替每条厂商指令的实硬件正确性验证，也不代表上游 LLVM 已接纳。

### 2.5 模型 & 技术｜[\[NNPA\] 将“Reshape -> 4D-Add -> Reshape”重写为 3D-Add](https://github.com/onnx/onnx-mlir/pull/3661)

北京时间：2026-09-29 13:43:09｜来源类型：官方 GitHub PR｜事件状态：已合并

新增非广播 Add 的重写：两侧 3D 输入升为 4D、相加后再降回 3D 的链路直接变为 3D Add，与已有 MatMul、SoftMax 的三维转换配合。

**重要性**：使注意力链路中的 Add 能继续在 NNPA 上执行，减少因表示维度产生的后端覆盖缺口。

**限制与风险**：仅在不发生广播且满足形状关系时适用，不是任意四维加法的降维。

### 2.6 深度洞见｜[\[RFC\] BorrowSanitizer 的实验性支持](https://discourse.llvm.org/t/90938/23)

北京时间：2026-09-30 04:46:33｜来源类型：项目官方讨论区｜事件状态：讨论新进展

BorrowSanitizer 通过 LLVM 插桩与运行库检测 Rust Tree Borrows 别名违规，并覆盖与 C/C++ 交互的路径。本轮作者公布 Rust 侧 MCP，提议将 LLVM pass、包装运行库和 Rust 核心运行库作为 nightly 的实验组件分发。

**重要性**：将 LLVM 侧 sanitizer 设计推进到 Rust 工具链分发提案，降低跨语言内存错误检测工具的接入门槛。

**限制与风险**：MCP 尚未批准；工具不检测数据竞争、未初始化读取等全部未定义行为，不能替代 Miri。

相关原文：[Upstreaming BorrowSanitizer Experimentally in Nightly Rust](https://github.com/rust-lang/compiler-team/issues/1041)。

### 2.7 深度洞见｜[RFC：Clang 和 LLVM 中的 WebAssembly \_\_externref\_t 指针](https://discourse.llvm.org/t/91914/16)

北京时间：2026-09-30 03:34:48｜来源类型：项目官方讨论区｜事件状态：讨论新进展

提案通过链接器合成的 externref 表，为宿主引用提供指针、全局与数组表示，load/store 映射到 table.get/set。本轮作者回应 memcpy 类型混用风险，计划以自定义地址空间区分 externref 指针；共享表与宿主对象生命周期控制仍保留。

**重要性**：影响 Wasm 与 JavaScript、Rust 运行时之间的互操作和跨库指针约定。

**限制与风险**：地址空间方案仍需实现验证；直接将 externref 放入结构体或 lambda 捕获不属于当前目标，不能视为完整 C/C++ 值语义支持。

### 2.8 深度洞见｜[Arc 仿真工作负载的阶段感知结构分析：初步结果与问题](https://discourse.llvm.org/t/91912/2)

北京时间：2026-09-30 02:50:22｜来源类型：项目官方讨论区｜事件状态：讨论新进展

原研究比较 HW、Arc 和状态 lowering 后的结构，并在修改后的 PicoRV32 上交叉检查 Arcilator 与 Verilator。窗口内维护者说明：LowerState 负责保持即时内存读写顺序；状态传递分析更适合在 LowerState 前进行，之后的读改写表示会丢失部分依赖信息。

**重要性**：明确了 CIRCT 仿真分析的有效阶段，以及图指标与执行语义的区别。

**限制与风险**：维护者提出合并小状态的优化方向，尚非已实现优化；原实验只覆盖特定处理器配置与工作负载。

## 三、Triton & TileLang 技术动态

### 3.1 模型 & 技术｜[\[AMD\] 不要在不存在的行上交错排列 CDNA4 填充布局](https://github.com/triton-lang/triton/pull/11890)

北京时间：2026-09-29 18:08:49｜来源类型：官方 GitHub PR｜事件状态：已合并

修正 CDNA4 异步复制布局中连续行宽超过一个 warp 时的整数除法边界，避免为不存在的行生成基并触发布局 verifier 中止；作者报告下游 70 组形状和类型扫描中的中止从 8 次降为 0。

**重要性**：恢复 gfx950 上一类宽行 GEMM 的编译能力，属于明确的硬件后端路径问题。

**限制与风险**：修复针对填充布局构造；此前成功组合保持位相同代码，并不表示新增普遍性能收益。

### 3.2 模型 & 技术｜[回退“\[AMD\]\[GFX9\] 启用 amdgpu-use-amdgpu-trackers LLVM 标志”](https://github.com/triton-lang/triton/pull/11763)

北京时间：2026-09-29 12:11:50｜来源类型：官方 GitHub PR｜事件状态：已合并

Triton 回退对全部 gfx942/gfx950 编译启用 amdgpu trackers 的设置。提交者在 MI350X 上固定 4096³ FP4 GEMM 配置对比，启用标志后出现主循环 scratch 读写，吞吐约下降 48%。

**重要性**：用定位到具体编译标志的实验恢复该性能路径，说明寄存器压力会抵消调度策略收益。

**限制与风险**：数据针对给定 aiter 内核与配置；按 waves_per_eu 条件启用的替代方案仍是后续工作。

### 3.3 模型 & 技术｜[\[XPU\] 当溢出达到每硬件线程 1024 B 时，以 large GRF 重新构建](https://github.com/intel/intel-xpu-backend-for-triton/pull/8147)

北京时间：2026-09-29 09:21:02｜来源类型：官方 GitHub PR｜事件状态：已合并

Intel Triton 后端将寄存器溢出重编译门槛改为固定每硬件线程 1024 B，避免按 lane 槽位计数使 SIMD32 的有效预算翻倍。在 1,506 个内核离线回放中，三个 IGC 版本分别触发 47、45、45 次重编译。

**重要性**：让 Inductor 有机会比较 large GRF 生成的候选，修复先前门槛漏掉的优化区间。

**限制与风险**：增加编译工作，不能由重编译次数直接推导运行速度；LTS 分支仍按任意溢出重编译。

### 3.4 模型 & 技术｜[\[TRITONGPU\] 在提升共享内存分配时保留异步读取完成约束](https://github.com/triton-lang/triton/pull/11975)

北京时间：2026-09-29 06:45:16｜来源类型：官方 GitHub PR｜事件状态：已合并

指令重排不再把带初始化写入的共享内存分配移过等待先前异步读取完成的操作；新增 Blackwell warp-specialized 矩阵乘及辅助输出用例。

**重要性**：防止复用共享内存时过早覆盖仍被异步 MMA 使用的数据，维护并发执行正确性。

**限制与风险**：限制的是不安全的重排，不是新的同步 API；回归用例具有架构和布局条件。

## 四、RISC-V 核心新闻

### 4.1 模型 & 技术｜[防止陷阱向量重定位破坏测试镜像](https://github.com/riscv/riscv-arch-test/pull/2636)

北京时间：2026-09-30 04:19:48｜来源类型：官方 GitHub PR｜事件状态：已合并

架构测试拒绝与完整测试镜像重叠的陷阱向量搬移，要求链接脚本预留独立空间；记录已保存字节数，处理部分复制和提前中止，各特权模式使用自己的保存区。

**重要性**：防止测试基础设施覆盖自身代码或数据而产生误导性失败，保护特权测试结果的可信度。

**限制与风险**：地址零仅在镜像之外才允许；仍需要平台链接布局提供可写的陷阱处理空间。

### 4.2 模型 & 技术｜[SG2044：添加 OpenBMC 固件升级 tar 包](https://github.com/sophgo/bootloader-riscv/commit/5b04ff5b519d08e0703814fb43cdfd5342fdfc59)

北京时间：2026-09-29 17:04:08｜来源类型：官方默认分支提交｜事件状态：已进入默认分支

算能 bootloader-riscv 将 OpenBMC 固件包生成从 mango 分支扩展到其他平台，以大写 PLAT 生成 machine 标识，打包 image-bmc 为 obmc-bios.tar.gz。

**重要性**：为 SG2044 补齐经 BMC 分发 BIOS 固件的构建产物链路，是服务器部署与维护能力的具体增量。

**限制与风险**：提交证明的是打包流程，不包含固件升级在所有板卡上的实测结果；平台标识和签名配置仍需匹配目标。

### 4.3 模型 & 技术｜[Misalign：跨越 64 字节边界、测试两种符号位、覆盖全部 8 个偏移量](https://github.com/riscv/riscv-arch-test/pull/2611)

北京时间：2026-09-29 16:21:39｜来源类型：官方 GitHub PR｜事件状态：已合并

非对齐访存测试新增跨 scratch+64 的窗口与覆盖点；加载同时使用原始和取反模式，区分符号扩展与零扩展，并在 RV32/RV64 都覆盖全部八个字节偏移。

**重要性**：此前用例未跨越较宽总线边界，且正值数据难以暴露 lh/lhu 混淆；此次新增实际架构验证能力。

**限制与风险**：覆盖给定边界和数据模式不等于证明所有缓存线、总线宽度或访存实现。

### 4.4 模型 & 技术｜[当 GEILEN 为 0 时，将 hie.SGEIE 和 mie.SGEIE 硬连线为 0](https://github.com/riscv/sail-riscv/pull/1978)

北京时间：2026-09-29 11:47:14｜来源类型：官方 GitHub PR｜事件状态：已合并

Sail 在未实现 guest external interrupt 的配置中，将 hie.SGEIE 与 mie.SGEIE 固定为零，补齐原先只约束 mideleg.SGEI 的模型行为。

**重要性**：使 Hypervisor 中断 CSR 的可写行为与实现参数一致，避免参考模型接受不应存在的中断使能状态。

**限制与风险**：仅涉及 GEILEN=0 配置，不能扩写为新增 Hypervisor 功能。

### 4.5 模型 & 技术｜[澄清 MAG 不适用于 SSAMOSWAP](https://github.com/riscv/riscv-isa-manual/pull/3426)

北京时间：2026-09-29 11:24:35｜来源类型：官方 GitHub PR｜事件状态：已合并

RISC-V 规范将 misaligned atomicity granule PMA 的 AMO 适用范围明确限定为 Zaamo、Zabha 和 Zacas；同日 Sail 将 ShadowStack 原子操作从 MAG 适用集合移除。

**重要性**：使规范文字与参考模型对影子栈原子交换的对齐约束一致，影响处理器与模型验证。

**限制与风险**：这是既有语义的澄清，不是新扩展获批；不能把普通 AMO 的非对齐原子性规则套用于 SSAMOSWAP。

相关原文：[Fix alignment check for `ssamoswap`.](https://github.com/riscv/sail-riscv/pull/1980)。

## 五、AI 业界重磅

### 5.1 重磅｜[GPT-6.1 Sol](https://openai.com/index/introducing-gpt-6-1-sol/)

北京时间：2026-09-29 18:00:00｜来源类型：官方博客｜事件状态：已发布

OpenAI 发布 GPT-6.1 Sol，在智能体编码、专业任务和计算机使用上提升能力；API 标识为 gpt-6.1-sol，每百万输入、缓存输入和输出 token 的标准价格分别为 2、0.10 和 10 美元。

**重要性**：为较长智能体工作流提供新的能力与成本选择。

**限制与风险**：评估依赖指定推理强度与工具环境；当前在 ChatGPT Work、Codex 和 API 提供，尚未进入 Chat，Ultrafast 是后续推出计划。

### 5.2 模型 & 技术｜[使用 NVIDIA VSS Blueprint 3.3 降低构建和运行视觉 AI 智能体的成本](https://developer.nvidia.com/blog/lower-the-cost-of-building-and-running-visual-ai-agents-with-nvidia-vss-blueprint-3-3/)

北京时间：2026-09-30 02:35:09｜来源类型：官方博客｜事件状态：已发布

VSS Blueprint 3.3 增加组合视频工作流的 Build Vision Agent skill，并在实时 VLM 微服务中按 patch 和帧自适应裁剪冗余视觉 token。官方案例报告，60 分钟摘要的 VLM 输入 token 减少 80%，同 GPU 并发流增加 46%。

**重要性**：同时处理视频智能体的服务组合成本与推理负载。

**限制与风险**：案例收益取决于视频变化率和裁剪设置；需要分别衡量准确率、吞吐与延迟，不能视为所有视频的固定收益。

### 5.3 模型 & 技术｜[NVIDIA Kumo Tabular 为表格预测确立新的准确率—效率前沿](https://huggingface.co/blog/nvidia/kumo-tabular)

北京时间：2026-09-29 23:30:38｜来源类型：官方博客｜事件状态：已发布

NVIDIA 发布开放表格基础模型 Kumo Tabular，利用列、行和上下文注意力，在单次前向计算中完成分类或回归；同时提供权重及 structured-data-models 库。训练采用合成表格，查询行可复用上下文 KV 缓存。

**重要性**：将预训练与上下文学习用于企业常见结构化数据任务，减少每张表从头训练的需求。

**限制与风险**：原生输入是数值与类别列；超出训练范围或查询分布变化可能降低准确性，训练配方与生成器仍待发布。

### 5.4 深度洞见｜[设计即 AI 原生：构建 NVIDIA TensorRT Model Connect 的经验](https://developer.nvidia.com/blog/ai-native-by-design-lessons-learned-from-building-nvidia-tensorrt-model-connect/)

北京时间：2026-09-30 03:10:51｜来源类型：官方博客｜事件状态：已发布

NVIDIA 公开 TensorRT Model Connect 的工程经验：按模型家族隔离 builder、运行管线与验证，将检查点转为版本化 bundle 和原生 C++ 任务接口，以可复现实验、独立 QA 与人工复核筛选智能体生成的改动。

**重要性**：为模型导入与推理部署项目提供可审查的模块边界和质量控制实践。

**限制与风险**：项目仍为 public preview；共享基础设施故障仍可波及多个家族，大多数长任务仍由人发起。

### 5.5 深度洞见｜[调查：为何双 token MTP 在没有更快主导内核的情况下超过自回归解码](https://dev-discuss.pytorch.org/t/3451/1)

北京时间：2026-09-29 19:02:53｜来源类型：项目官方讨论区｜事件状态：研究讨论已发布

作者在受控 vLLM 部署中比较自回归解码与双 token MTP，报告输出吞吐提高 1.91—2.19 倍；主导 MTP GEMM 并未比 AR GEMV 更快，差异来自每个生成 token 所需的重复 GPU 执行减少。

**重要性**：提示推理性能分析应同时观察 token 进展与调度频率，不能只比较最耗时内核。

**限制与风险**：这是作者的特定部署调查，正在征求对运行时层次与 CUDA Graph 执行关系的反馈，不能推广为所有模型的收益。

### 5.6 工具 & 产品｜[OpenClaw Enterprise - 开放智能体平台](https://openclaw.ai/blog/openclaw-enterprise)

北京时间：2026-09-29 08:00:00｜来源类型：官方博客｜事件状态：已发布

OpenClaw 宣布 Enterprise 平台，提供多租户、隔离边界、权限及审计控制，允许替换 harness、模型与沙箱；目前可用 Docker Compose 本地运行或在内部 Kubernetes 部署。

**重要性**：让持久运行智能体具备独立于模型供应商的企业控制层。

**限制与风险**：项目仍在 1.0 前开发阶段，定位内部试点；参考安全架构将在后续公布，不能把试点视为完整生产保证。

### 5.7 工具 & 产品｜[推出 dots](https://openai.com/index/introducing-dots/)

北京时间：2026-09-29 08:00:00｜来源类型：官方博客｜事件状态：逐步推出

OpenAI 推出由 GPT-6 Astra 驱动、拥有独立云电脑的 dots，可通过连接的应用处理持续任务；后台 proactive research 使用受限只读工具，可能影响账户或分享信息的行动由 auto-review 检查。

**重要性**：持续任务、云端运行环境和权限控制被组合为面向用户的智能体产品。

**限制与风险**：在合资格市场逐步推出；Enterprise 需管理员启用 beta，specialist dots 处于企业试点，重要工作仍需人工复核。

## 六、总结与趋势观察

**推理部署优化继续深入数据与内存边界。** [KV 缓存原地写入](https://github.com/pytorch/TensorRT/pull/4740)与[共享激活临时缓冲池](https://github.com/pytorch/TensorRT/pull/4739)分别减少复制与长期显存占用；[Col2Im lowering](https://github.com/onnx/onnx-mlir/pull/3633)则补齐模型到可执行程序的算子路径。这些改进面向具体链路，不能直接外推为所有模型的加速。

**编译基础设施更强调显式语义。** [MLIR 部分折叠提案](https://discourse.llvm.org/t/91954/1)要求明确表达每个结果的替换状态，[Wasm externref 讨论](https://discourse.llvm.org/t/91914/16)要求区分宿主引用指针与普通内存指针；两者都把接口契约和下游兼容性作为核心设计问题。

**持久智能体的发布同时包含运行与治理设计。** [dots](https://openai.com/index/introducing-dots/)将持续任务与行动审查结合，[OpenClaw Enterprise](https://openclaw.ai/blog/openclaw-enterprise)将多租户与审计纳入控制层，[TensorRT Model Connect](https://developer.nvidia.com/blog/ai-native-by-design-lessons-learned-from-building-nvidia-tensorrt-model-connect/)强调模型家族隔离和独立质量复核。它们的成熟度不同，不能把产品推出或开源等同于安全性已经得到完整证明。

**RISC-V 软件栈正同时完善语义验证和部署产物。** [MAG 适用范围澄清](https://github.com/riscv/riscv-isa-manual/pull/3426)与[非对齐访存测试扩展](https://github.com/riscv/riscv-arch-test/pull/2611)提高规范与验证的一致性，[SG2044 OpenBMC 固件包](https://github.com/sophgo/bootloader-riscv/commit/5b04ff5b519d08e0703814fb43cdfd5342fdfc59)补齐系统维护链路。这些工作解决不同层次的问题，各自仍需目标平台验证。

## 附录：信源说明

本期主要采用 PyTorch/ExecuTorch、Torch-TensorRT、LLVM/MLIR、ONNX-MLIR、RuyiAI、Triton、Intel XPU、RISC-V 与算能的官方实现记录和原始讨论；AI 产业部分采用 OpenAI、NVIDIA、OpenClaw 的官方公告及作者在项目讨论区发布的研究说明。

PR 合并和默认分支提交不等于稳定版本已经发行；RFC、MCP 与维护者回复不等于方案已获批准。性能数字仅适用于原文硬件、软件与测试方法，预览和试点能力保留其适用边界。
