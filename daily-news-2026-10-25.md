# Codex 技术情报每日动态（2026-10-25）

**调研窗口：** `(2026-10-24 04:32:32, 2026-10-25 06:00:20]`，北京时间（Asia/Shanghai）。

**覆盖方向：** PyTorch 生态、LLVM/MLIR、Triton & TileLang、RISC-V 软件栈与 AI 业界。

**信息口径：** 以官方公告、规范、开发者讨论及有编辑责任的原始报道为依据；提案、讨论进展与已合并实现分别标注，实验数字保留原始适用条件。

## 今日要闻

- 中国信通院发布词元服务年度监测报告，结合吞吐、首词时延、稳定性与价格比较模型服务表现。 [原文](https://aihub.caict.ac.cn/docs/G9Ti6B1CNd3C)
- 链接脚本 LTO 草案加入 ThinLTO；作者报告了嵌入式应用内部测试进展。 [原文](https://discourse.llvm.org/t/91089/30)
- StableHLO 收录已批准的 f8E5M3FNU RFC，明确 block-scaling 类型及待解决的算术、版本化边界。 [原文](https://github.com/openxla/stablehlo/blob/92340885e714ecd7d0d7d1b656d4e933db34b133/rfcs/20261009-f8E5M3FNU.md)
- 36氪独家报道称小云雀并入即梦，独立入口与产品团队保留。 [原文](https://www.36kr.com/p/4039677577122946)
- SSAF 新 RFC 提议以 make 构建图编排全项目摘要、分析与源码变换。 [原文](https://discourse.llvm.org/t/92595)
- LLVM 新 RFC 提议用 TableGen 声明结构化优化备注，并通过 !remark 元数据将其关联至代码生成。 [原文](https://discourse.llvm.org/t/92594)

## 今日索引

- **PyTorch：** TorchTitan 的 Qwen3 MoE 强化学习配方，以及 ExecuTorch 的 Cadence 算子、Arm lowering 与 CUDA wheel 部署变更。
- **LLVM/MLIR：** SSAF 全项目编排、结构化优化备注、StableHLO 缩放类型 RFC、链接脚本 LTO 与数值语义讨论。
- **Triton & TileLang：** Triton 隐式浮点融合控制、RDNA4 IEEE min/max，以及 ROCm 发布线 trackers 回退。
- **RISC-V：** 控制流保护下的跳转表检查、P 扩展 packed slide，以及 Sscofpmf/Sdtrig 架构测试语义。
- **AI 业界：** 词元服务监测报告、Today AI 创始团队访谈，以及小云雀与即梦的资源整合报道。

## 一、PyTorch 生态核心动态

### 1.1 模型 & 技术｜[\[rl\] 添加 Qwen3-30B-A3B DAPO 数学配置](https://github.com/pytorch/torchtitan/pull/5598)

北京时间：2026-10-24 21:04:29｜来源类型：官方 GitHub PR（pytorch/torchtitan）｜事件状态：已合并。

**事实：** 加入 BF16 与 MXFP8 两套 Qwen3-30B-A3B DAPO math 配置；MXFP8 使用 mxfp8_mega_fused_inference。 作者在 4 张 B200 训练卡加 4 张 B200 生成卡、8K 回复、50 步的设置中报告：第 3—50 步平均步时由 24.3 秒降至 22.7 秒，单请求解码时间由 20.2 秒降至 17.8 秒。

**重要性：** 这给 MoE 强化学习提供了可对照的低精度训练与 rollout 配方，能区分训练计算和生成等待的耗时。

**风险与限制：** 数字来自作者的指定硬件与短程运行；两组 reward 与验证起点不同，不能据此断言收敛等价或任意规模都有同样收益。

### 1.2 模型 & 技术｜[直接以打包位生成环形掩码 (#24089)](https://github.com/pytorch/executorch/pull/24089)

北京时间：2026-10-24 09:33:56｜来源类型：官方 GitHub PR（pytorch/executorch）｜事件状态：已合并。

**事实：** 新增 sdpa_bitwise_ring_mask_gen 的 schema、meta/参考实现和 Cadence C++ out kernel，按 token 位置、滑动窗口和缓存容量直接写出 uint8 打包位。 实现跳过先建立稠密环形掩码再打包的中间路径，并覆盖环回、位序、空输入与大整数位置。

**重要性：** 直接生成注意力掩码减少了一次中间表示和打包步骤，属于端侧注意力执行链路的新算子能力。

**风险与限制：** 此 PR 的 Jarvis 注册仍在另一变更中；原文没有给出端到端吞吐或峰值内存实测。

### 1.3 模型 & 技术｜[在 vision quantized_layer_norm 中支持 int16 (#24094)](https://github.com/pytorch/executorch/pull/24094)

北京时间：2026-10-24 09:26:54｜来源类型：官方 GitHub PR（pytorch/executorch）｜事件状态：已合并。

**事实：** Cadence Vision 的两种量化 LayerNorm 重载加入 int16 输入支持，此前该类型会触发检查中止。 16 位路径使用 int64 累加并在乘法前扩宽，采用整数形式 N*sum(v²)-sum(v)² 计算方差分子，避免原 int32 溢出及大均值下的消减问题。

**重要性：** 这补齐 Cadence 量化模型部署中的类型缺口，同时改变了容易出错的数值计算路径。

**风险与限制：** 支持范围是相应 Cadence kernel；不能据此推断所有后端或所有张量规模的数值误差已消除。

### 1.4 模型 & 技术｜[Arm 后端：支持常量负填充值](https://github.com/pytorch/executorch/pull/24108)

北京时间：2026-10-25 01:34:36｜来源类型：官方 GitHub PR（pytorch/executorch）｜事件状态：已合并。

**事实：** Arm 后端先把负 padding 降低为 slice，再执行非负 TOSA padding，支持两侧裁剪、裁剪与填充混合以及多维裁剪。 原文报告浮点与量化 TOSA/VGF 流水线共 148 项 padding 测试通过，另有 3 项预期硬件 runner xfail。

**重要性：** 模型导入后的负 padding 语义现在有明确 lowering 路径，可减少端侧模型转换中的算子适配缺口。

**风险与限制：** 测试声明不等于全部硬件验证；该支持针对常量 padding，不能外推为动态裁剪普遍可用。

### 1.5 工具 & 产品｜[CUDA wheel：检查每个依赖，并恢复 torch_cpu_library 目标的 torch 编译设置](https://github.com/pytorch/executorch/pull/24087)

北京时间：2026-10-24 05:59:02｜来源类型：官方 GitHub PR（pytorch/executorch）｜事件状态：已合并。

**事实：** CUDA wheel 的依赖检查扩展到每个被打包库记录的依赖，允许范围覆盖 wheel 内库、CPU PyTorch 共有库、manylinux 系统库及安装必需包。 find_package_torch 为 torch_cpu_library 恢复接口 include、编译选项与编译特性，同时避免复制 CUDA 专用设置；aten_bridge 改为 STATIC。

**重要性：** 这是 CUDA wheel 与 CPU 版 PyTorch 并存部署链路的加固，防止遗漏 CUDA 动态库依赖和丢失 TORCH_CXX_FLAGS。

**风险与限制：** 仍须由目标安装环境满足允许的依赖集合；没有 torch_cpu_library 的静态 libtorch 情况不应用该复制逻辑。

## 二、LLVM/MLIR 最新进展

### 2.1 模型 & 技术｜[\[RFC\] 由 TableGen 声明的优化备注，以 !remark 元数据传递至代码生成](https://discourse.llvm.org/t/92594)

北京时间：2026-10-24 07:02:26｜来源类型：官方 RFC／规范（LLVM Discourse）｜事件状态：RFC 提案；未合并。

**事实：** 提议用 TableGen 集中声明优化备注的名称、结果、消息和参数，并为每类生成有类型的 builder。 第二部分提议把备注作为 !remark 附在 LLVM IR 指令上，经中端和指令选择传至 MachineInstr，让后端把备注写在代码旁；作者明确只有原型，尚未提出落地。

**重要性：** 结构化语义和指令关联可供性能分析工具、DSL 编译器及自动化优化工具消费，也涉及 MLIR/LLVM 交界处的诊断信息传递。

**风险与限制：** 设计仍在早期 RFC；元数据经过变换的保留与合并语义需要评审，不能当作已可使用的接口。

### 2.2 模型 & 技术｜[\[RFC\] 在整个项目上运行 SSAF：clang-ssaf-orchestrator-make 和 clang-ssaf-driver](https://discourse.llvm.org/t/92595)

北京时间：2026-10-24 10:43:06｜来源类型：官方 RFC／规范（LLVM Discourse）｜事件状态：RFC 提案；未合并。

**事实：** 提议新增 clang-ssaf-orchestrator-make，沿 make 构建图对编译和链接步骤执行摘要提取、实体链接与全程序分析。 clang-ssaf-driver 面向用户隐藏编排细节；变换以摘要提取、分析、源码变换的组合定义，源码编辑需要第二轮构建，并合并编辑、诊断和冲突报告。

**重要性：** 它把 SSAF 从单个翻译单元或链接单元扩展为整项目使用流程，为跨模块分析和 C++ 安全迁移提供工具设计。

**风险与限制：** 目前是 make 集成提案；其他构建系统、增量执行与冲突处理不能据此视为已经交付。

### 2.3 模型 & 技术｜[\[RFC\] 添加 f8E5M3FNU 类型](https://github.com/openxla/stablehlo/blob/92340885e714ecd7d0d7d1b656d4e933db34b133/rfcs/20261009-f8E5M3FNU.md)

北京时间：2026-10-24 07:53:21｜来源类型：官方 RFC／规范（StableHLO）｜事件状态：RFC 批准文档已入库；类型实现待后续。

**事实：** StableHLO 默认分支收录状态为 Approved 的 f8E5M3FNU RFC：这是 5 位指数、3 位尾数且无符号的 8 位格式，用于 block scaling，对应 PTX 的 ue5m3；文档描述目标为 PTX ISA 9.4 与 sm_107f。 方案希望让该格式可用于 XLA/JAX；要求增加 StableHLO minor version、VHLO 类型及 bytecode type code。数值过大转换为 NaN、不饱和；负值转换仍在另一个 PR 讨论，除转换外的算术未规定。

**重要性：** 这是低精度缩放格式进入稳定 IR 契约的设计进展，和 MLIR、XLA 及模型部署中类型与序列化的一致性直接相关。

**风险与限制：** 批准并入库的是 RFC 文档，不等于类型实现已全部落地；负值与算术语义仍有明确未决边界。

### 2.4 深度洞见｜[\[RFC\] \[LTO\] \[lld\] 带链接脚本的 LTO，已实现](https://discourse.llvm.org/t/91089/30)

北京时间：2026-10-25 04:35:54｜来源类型：官方开发者论坛（LLVM Discourse）｜事件状态：讨论新进展。

**事实：** 主提案使 LTO 下的链接脚本更接近普通链接行为：识别 bitcode 文件名、阻止跨输出节内联等变换，并使 KEEP 对 bitcode 符号生效；clang/lld 使用各自的显式开关。 窗口内 efriedma-quic 的第 30 楼称端到端草案已更新并加入 ThinLTO；内部测试表明当前方法可用于实际嵌入式应用，同时列出让 MachineOutliner 和 ConstantMerge 感知输出节的后续补丁。

**重要性：** 链接脚本是嵌入式工具链的关键约束；这条设计主线对 RISC-V 固件和定制内存布局的 LTO 具有直接参考价值。

**风险与限制：** 仍为草案，作者对索引中输出节表示尚不满意；内部测试结论属于作者陈述，不能等同上游合并或通用兼容保证。

### 2.5 深度洞见｜[\[RFC\] 将 llvm-ub-aware-interpreter 引入上游](https://discourse.llvm.org/t/89645/16)

北京时间：2026-10-24 16:26:25｜来源类型：官方开发者论坛（LLVM Discourse）｜事件状态：讨论新进展。

**事实：** 主提案讨论将能理解 LLVM IR 未定义行为的解释器 llvm-ub-aware-interpreter（llubi）引入上游，用于解释 IR 执行及诊断相关语义。 窗口内 dtcxzyw 的第 16 楼发布浏览器 playground：把 llubi nightly 构建为 WASM 并部署在静态网站；解释其另行建设工具是因为现有 Compiler Explorer executor 列表不含 lli。

**重要性：** 浏览器执行入口降低了复现 LLVM IR 语义问题的门槛，可帮助编译器开发者交换最小用例。

**风险与限制：** 作者明确当前工具没有类似 Compiler Explorer、Alive2 的永久可分享短链接；发布 playground 不代表上游接纳该 RFC。

### 2.6 深度洞见｜[KnownFPClass 推断对于 `afn` 函数是否仍然成立？](https://discourse.llvm.org/t/92463/8)

北京时间：2026-10-25 05:32:33｜来源类型：官方开发者论坛（LLVM Discourse）｜事件状态：讨论新进展。

**事实：** 主讨论质疑 KnownFPClass 对数学库结果范围、零和无穷的假设：afn 仅允许调用返回近似值，后续优化仍可能错误利用过强推断。 窗口内 zerico2005 的第 7—8 楼分析向负无穷舍入时 sinh/tanh 的正负零差异，并讨论为 sin/cos 等结果夹取精确区间的方案及其边界。

**重要性：** 这是数学库契约与编译优化之间的数值语义争议，对依赖 fast-math 或近似数学函数的 AI kernel 正确性有参考意义。

**风险与限制：** 回复提供分析和可能方案，没有形成 LLVM 已改变推断规则的决定；不能将某一数学实现的行为推广至所有库。

### 2.7 深度洞见｜[\[RFC\] 移除 GUI 模式](https://discourse.llvm.org/t/92570/8)

北京时间：2026-10-24 08:11:10｜来源类型：官方开发者论坛（LLVM Discourse）｜事件状态：讨论新进展。

**事实：** 主提案建议移除 LLDB 内置 GUI/TUI command，理由包括缺少活跃维护、崩溃和难以稳定测试，且可在 SB API 上独立实现界面。 窗口内原作者 clayborg 的第 8 楼支持移除，说明已有通过 Python LLDB command 和 curses API 接管屏幕的实验代码，可作为外置替代路径。

**重要性：** 这为调试器界面的核心与扩展边界提供了维护者技术解释，影响依赖 LLDB 内置 curses 界面的工具选型。

**风险与限制：** 支持意见不等于正式移除决定；替代实验也没有被描述为可直接替换的成熟产品。

### 2.8 模型 & 技术｜[\[mlir\]\[scf\] 为尾部剥离添加无符号循环支持](https://github.com/llvm/llvm-project/pull/234435)

北京时间：2026-10-24 08:51:19｜来源类型：官方 GitHub PR（llvm/llvm-project）｜事件状态：已合并。

**事实：** SCF loop back peeling 加入 unsigned 循环路径，并修复 signed 路径的减法溢出问题。 无符号路径用 minui/subi/remui 计算剥离边界，并在需要 affine.min/max 的边界简化处提前返回，避免把有符号推理直接用于无符号循环。

**重要性：** 这是 MLIR 循环变换支持范围与语义处理的实质变化，对使用 SCF 的张量编译流水线有直接价值。

**风险与限制：** 支持尾部剥离不等于所有 unsigned 循环优化都已可用；原文没有提供端到端性能增益。

## 三、Triton & TileLang 技术动态

### 3.1 模型 & 技术｜[遵循 enable_fp_fusion 前须显式启用](https://github.com/triton-lang/triton/pull/12479)

北京时间：2026-10-24 07:59:42｜来源类型：官方 GitHub PR（triton-lang/triton）｜事件状态：已合并。

**事实：** 以 TRITON_ALLOW_FP_FUSION 替换 TRITON_FORCE_DISABLE_FP_FUSION。 默认不启用隐式浮点融合，即使 launch option enable_fp_fusion 请求融合；设置 TRITON_ALLOW_FP_FUSION=1 才恢复既有 launch option 与 TRITON_DEFAULT_FP_FUSION 回退行为，显式 tl.fma 保持原语义。

**重要性：** 这改变了隐式融合的默认控制契约，关系到依赖分步舍入的 kernel 数值行为和性能配置。

**风险与限制：** 没有给出普遍性能影响；此处的隐式融合控制不能与显式 fused 运算混同。

### 3.2 模型 & 技术｜[\[AMD\] 为 RDNA4 启用 IEEE maximum/minimum](https://github.com/triton-lang/triton/pull/12480)

北京时间：2026-10-24 23:55:54｜来源类型：官方 GitHub PR（triton-lang/triton）｜事件状态：已合并。

**事实：** AMD TargetFeatures 的 supportMaximumMinimum 现在对 RDNA4 gfx120x 返回 true。 原文依据是 gfx12 具有 IEEE-754 2019 maximum/minimum 指令；对应转换测试随之更新。

**重要性：** RDNA4 的 min/max 降低可使用其原生 IEEE 指令能力，补齐目标特性声明与硬件能力之间的缺口。

**风险与限制：** 变化针对 gfx120x 和相关操作；原文未提供实测加速比例，不能推广成全部 AMD 目标支持。

### 3.3 模型 & 技术｜[回退“\[AMD\]\[GFX9\] 启用 amdgpu-use-amdgpu-trackers LLVM 标志 (#112…](https://github.com/ROCm/triton/pull/980)

北京时间：2026-10-24 06:27:57｜来源类型：官方 GitHub PR（ROCm/triton）｜事件状态：已合并。

**事实：** ROCm Triton 3.8 release stream 撤销为 gfx942/gfx950 启用 amdgpu-use-amdgpu-trackers 的变更。 PR 说明上游因访问未分配设备内存的风险已回退 trackers，此次让 3.8 发布线同样保持关闭；差异移除相应 compiler flag。

**重要性：** 它把潜在设备内存访问风险的防护同步到发布分支，是编译后端正确性相关的回退。

**风险与限制：** 原文说明的是风险与禁用策略，没有公布可利用漏洞或完整影响清单，不能改写为已确认安全漏洞。

## 四、RISC-V 核心新闻

### 4.1 模型 & 技术｜[\[SelectionDAG\]\[RISCV\]\[X86\] 在启用 cf-protection-branch 时保留跳转表范围检查](https://github.com/llvm/llvm-project/pull/234733)

北京时间：2026-10-25 00:11:21｜来源类型：官方 GitHub PR（llvm/llvm-project）｜事件状态：已合并。

**事实：** 当 switch 的 default 不可达时，SelectionDAG 现在也为启用 cf-protection-branch 的目标保留跳转表范围检查，覆盖 RISC-V Zicfilp 和 X86 IBT。 模块标志值为 0 不视为启用；RISC-V 与 X86 用例验证了有无标志时的不同代码生成。

**重要性：** 这把既有 Arm 分支目标保护下的检查扩展到其他控制流保护目标，降低无边界跳转表成为 JOP gadget 的风险。

**风险与限制：** 生效依赖对应模块标志及代码生成路径；这不是完整 CFI 方案，也没有提供漏洞编号。

### 4.2 模型 & 技术｜[\[RISCV\]\[P-ext\] 支持 Packed Slide](https://github.com/llvm/llvm-project/pull/234629)

北京时间：2026-10-24 13:18:00｜来源类型：官方 GitHub PR（llvm/llvm-project）｜事件状态：已合并。

**事实：** 为 RISC-V P 扩展加入 packed slide intrinsics 的 funnel-shift lowering，并添加 RV32/RV64 代码生成覆盖。 作者指出 RV32 的 64 位 funnel shift 仍会展开为包含分支的 select 路径，计划后续改进。

**重要性：** 它扩展了打包 SIMD 操作在 LLVM RISC-V 后端中的可表达与降低能力。

**风险与限制：** 当前合并不意味着 RV32 已实现最优无分支代码，也不能据此宣称 ISA 已新增批准的标准扩展。

### 4.3 模型 & 技术｜[Sscofpmf：在 QEMU 上计数真实事件，并在抑制计数时预设计数器](https://github.com/riscv/riscv-arch-test/pull/2869)

北京时间：2026-10-24 09:46:59｜来源类型：官方 GitHub PR（riscv/riscv-arch-test）｜事件状态：已合并。

**事实：** QEMU max 配置把 Sail 的 EV_FENCE 映射至 selector 0x2，令 DUT 与参考模型都实际计数，而非让溢出检查面对不计数的参考。 生成器在所有 inhibit 位设置后预设计数器，再启用目标模式计数，处理 RV32 写高低半部期间提前回绕的问题；stock QEMU 的 rv32/rv64 inhibit_lo 组因此移除排除项。

**重要性：** 这是 Sscofpmf 架构验证中可执行覆盖和参考模型语义的修正，避免验证结果因事件映射或预设顺序而失真。

**风险与限制：** 完整溢出、中断及 RV32 Sm_access 组仍受未解决的 QEMU 问题限制；原文的全通过结果使用了带补丁的 QEMU。

### 4.4 模型 & 技术｜[Sdtrig：当 mstateen0.CONTEXT 为只读 0 时，通过 T-SBI 设置 scontext](https://github.com/riscv/riscv-arch-test/pull/2894)

北京时间：2026-10-24 05:29:07｜来源类型：官方 GitHub PR（riscv/riscv-arch-test）｜事件状态：已合并。

**事实：** 当 MSTATEEN_CONTEXT_TYPE 为 read-only-0 时，S 模式测试改由每个 hart 上的 T-SBI 设置 scontext，避免在 HS 直接写入引发非法指令。 作者报告 Sail rv32/rv64-max 的 SdtrigS/SdtrigSm/SdtrigU 各 105/105 通过；只读 0 派生配置的 S 模式写入已生效。

**重要性：** 它让调试触发器上下文测试覆盖受 mstateen 权限限制的实现，关系到特权软件和参考模型验证的适用范围。

**风险与限制：** 只读 0 派生配置的 SdtrigU 仍失败；hcontext 与 H 扩展的依赖仍交给 UDB，不能宣称所有上下文组合已经验证。

## 五、AI 业界重磅

### 5.1 深度洞见｜[中国信通院人工智能所联合发布《大模型词元服务性能监测报告（2026年）》](https://aihub.caict.ac.cn/docs/G9Ti6B1CNd3C)

北京时间：2026-10-24 20:01:41｜来源类型：机构官方公告（中国信通院·鲸智社区）｜事件状态：监测报告已发布。

**事实：** 中国信通院人工智能所等联合发布年度监测报告，采用 2026 年 1—8 月数据，比较原厂平台以及 DeepSeek、Qwen、GLM、Kimi 在第三方平台的词元服务。 报告给出的 8 月平均 TPS 为国内 78.23 个/秒、国外典型模型 112.82 个/秒；国内平均 TTFT 为 1.26 秒。原文同时指出模型参数、上下文及调用量变化会影响这些指标，同一开源模型在不同平台的服务表现存在显著差异。

**重要性：** 它提供了吞吐、首词时延、稳定性和价格等服务层面的联合观测，为理解推理部署性能提供比单一模型分数更具体的参照。

**风险与限制：** 数据覆盖 1—8 月和报告所选样本，并非 10 月实时榜单；不同平台与模型的均值不能作为同条件硬件对比。

### 5.2 深度洞见｜[拿到2亿美金融资，Today AI回答了Personal AI的十个核心问题](https://www.36kr.com/p/4039749830120583)

北京时间：2026-10-24 18:50:23｜来源类型：权威媒体原始报道（36氪·智能涌现）｜事件状态：创始团队访谈已发布。

**事实：** 36氪发布对 Today AI 创始团队的深入交流：团队选择持续单一对话窗口和独立客户端，强调长期上下文、记忆、设备权限及跨系统服务连接。 团队称不只做应用，还训练交互模型以改善个性化及单位智能成本；原文把留存、主动提醒与打扰、隐私授权和生态准入列为仍在验证的问题。融资公告于 10 月 21 日发布。

**重要性：** 这给 Personal AI 的产品与工程路径提供了来自创始团队的具体解释：价值验证从一次任务扩展到长期使用。

**风险与限制：** 这是团队自述；访谈没有证明产品已达到成熟 PMF，也不能把目标能力视作独立实测结果。

### 5.3 行业 & 人事｜[独家｜字节AI应用再整合：小云雀并入即梦](https://www.36kr.com/p/4039677577122946)

北京时间：2026-10-24 17:38:23｜来源类型：权威媒体原始报道（36氪·智能涌现）｜事件状态：媒体独家报道；未见官方公告。

**事实：** 36氪·智能涌现独家报道称，小云雀并入即梦，仍保留独立产品入口；即梦已经接入小云雀的短剧能力。 报道指此次调整主要为资源整合，运营由即梦运营和市场负责人陈欣然负责，产品团队没有变化。

**重要性：** 该调整显示字节的视频生成应用在模型与创作工作流之间重新组织资源，涉及短剧能力和智能创作入口的关系。

**风险与限制：** 这是媒体基于知情人士的报道，具体内部执行时间未公开；不能扩大为产品关停或已获字节官方确认。

## 六、总结与趋势观察

**数值语义仍在约束编译优化的边界。** Triton 改变隐式浮点融合的默认控制方式，KnownFPClass 讨论继续拆解近似数学函数的推断条件，SCF 的 unsigned 剥离则单独处理有符号推理不适用的边界。这些变化分别涉及执行契约、分析假设和循环变换，不能用一个 fast-math 开关概括。 [依据 1](https://github.com/triton-lang/triton/pull/12479)、[依据 2](https://discourse.llvm.org/t/92463/8)、[依据 3](https://github.com/llvm/llvm-project/pull/234435)。

**编译工具正在把分析与诊断组织成可组合的接口。** TableGen 备注 RFC 希望统一机器可读语义和指令关联，SSAF 编排 RFC 希望沿真实构建图连接摘要、分析及源码编辑。两者都处于提案阶段，体现的是设计方向，尚不能视作已经形成统一工具链。 [依据 1](https://discourse.llvm.org/t/92594)、[依据 2](https://discourse.llvm.org/t/92595)。

**端侧适配与架构验证都在补齐具体运行条件。** ExecuTorch 补充负 padding 和 int16 LayerNorm 路径，RISC-V 架构测试修正计数事件及受限特权访问。共同点是把输入类型、数值范围与特权状态写入可执行路径；各自的硬件测试及未解决项仍决定适用边界。 [依据 1](https://github.com/pytorch/executorch/pull/24108)、[依据 2](https://github.com/pytorch/executorch/pull/24094)、[依据 3](https://github.com/riscv/riscv-arch-test/pull/2869)、[依据 4](https://github.com/riscv/riscv-arch-test/pull/2894)。

## 附录：信源说明

本期技术事实来自 PyTorch/TorchTitan/ExecuTorch、LLVM/MLIR、StableHLO、Triton、ROCm Triton 与 RISC-V 架构测试项目的官方仓库和开发者论坛；行业内容来自中国信通院鲸智社区，以及 36氪·智能涌现的原始采访与报道。

论坛条目引用具体技术回复，并区分主提案与新增进展；RFC 批准文档与类型实现是不同状态。作者提交的测试和性能数字按其硬件、模型、运行次数与配置理解；中国信通院报告的数据期为 2026 年 1—8 月。媒体独家内容按报道口径呈现。
