# Codex 技术情报每日动态（2026-11-22）

调研窗口：北京时间（2026-11-21 05:47:14，2026-11-22 06:00:37]。

覆盖方向：PyTorch、LLVM/MLIR、Triton & TileLang、RISC-V、AI 模型与研究、推理部署和产业动态。

信息口径：依据原始公告、正式发行与预发行、技术讨论与项目变更，区分提案、已合入能力、工作流示例和作者测量。

## 今日要闻

- **Ruyi Firmware 20261121 预发行版新增 SG2042 多板卡支持。** 为 Milk-V Pioneer 与 SOPHGO EVB v1/v2提供SD启动和SPI安装映像；Pioneer的UEFI RTC要求先升级兼容内核。[官方预发行](https://github.com/openRuyi-Project/ruyifirmware/releases/tag/20261121)
- **Buddy MLIR 0.1.7 发布跨架构工具资产。** 提供x86_64、riscv64的wheel、CLI与LLVM工具包；本日发行配置针对Triton下游缺失MLIR `dump()`符号的链接问题作出调整。[正式发行](https://github.com/buddy-compiler/buddy-mlir/releases/tag/release/v0.1.7)

## 今日索引

- **一、PyTorch 生态核心动态**：ESP-NN集成指南、高维分块、动态形状编译规模、传输通知、CUDA权重共享、MLX草稿验证与MPS梯度修复。
- **二、LLVM/MLIR 最新进展**：Buddy工具发行、LLVM libc常量折叠、clang-tidy宏诊断、`ub.assume` poison语义与AMD SPIR-V运行库。
- **三、Triton & TileLang 技术动态**：单CTA FP4 K96、FP8注意力签名、隐式cluster同步与环形缓冲索引分析。
- **四、RISC-V 核心新闻**：Ruyi固件、K3显示栈、KVM steal-time、SRET规范、向量密码模型、虚拟中断测试与自托管CI部署。
- **五、AI 业界重磅**：腾讯混元团队解释架构取舍与小模型规划；Figma展示Specs到编码智能体的交接工作流。

## 一、PyTorch 生态核心动态

### 1.1 工具 & 产品｜[Add Espressif ESP-NN backend documentation（添加 Espressif ESP-NN 后端文档）](https://github.com/pytorch/executorch/pull/25205)

北京时间：2026-11-21 06:39:38｜来源类型：官方 GitHub PR｜事件状态：集成指南已发布。

**事实**：新指南公开EspNN量化、分区、XTS3图格式、ESP-IDF组件接入和端到端导出/上板流程。作者报告ESP32-S3小CNN为13.6毫秒；MobileNetV3-Small的122个算子完全委托，默认/建议缓存配置分别为866/649毫秒。 关联的新仿真配置将ESP32-P4与ESP32-S31纳入优化ESP-NN内核验证，并从ESP-IDF生成的工具链配置继承RISC-V的march/mabi，解决运行时与固件ABI不匹配。

**重要性**：为端侧模型部署补齐具体集成路径和验证条件。

**风险与限制**：数据来自作者指定板卡与模型；文档发布不等于首次发布ESP-NN硬件能力，operator-library路线仍标注实验性。

关联证据：[原始说明 1](https://github.com/pytorch/executorch/pull/25287)。

### 1.2 模型 & 技术｜[\[Inductor\] Support ND pointwise and reduction tiling (#204324) (#204324)（\[Inductor\] 支持 ND 逐点和归约分块 (#204324) (#204324)）](https://github.com/pytorch/pytorch/commit/e716fffc7eded0e8ee852fbc2fa8a80ced15cf57)

北京时间：2026-11-21 11:29:45｜来源类型：官方主干 commit｜事件状态：已进入主干。

**事实**：Inductor 的逐点与归约分块扩展到更多独立轴，允许显式设置 triton.max_tiles > 3，并去除 prefer_nd_tiling 路径的联合五维上限。

**重要性**：为高维模型图提供更灵活的生成内核布局，相关导出和后端适配可直接关注该编译路径。

**风险与限制**：默认分块行为不变，prefer_nd_tiling 默认关闭；MPS 对超过三轴的配置会明确报错，不能推导为所有设备通用支持。

关联证据：[原始说明 1](https://github.com/pytorch/pytorch/pull/204324)。

### 1.3 模型 & 技术｜[\[inductor\] Cap the number of dynamic-shape sub-kernels per combo kernel (#205934)（\[inductor\] 限制每个组合内核中动态形状子内核的数量 (#205934)）](https://github.com/pytorch/pytorch/commit/cb41f4ef2382a6c77b67706485bf54ae613b8d76)

北京时间：2026-11-21 14:32:14｜来源类型：官方主干 commit｜事件状态：已进入主干。

**事实**：当前实现以16为上限，将每个组合内核中的符号分块子内核限制为16。约230输入的生产推荐图此前生成605参数、约120万字符内核，默认编译进程触发 RecursionError；作者在H100上测得加上限后302秒完成。

**重要性**：解决动态 cat 横向融合造成的编译规模失控，给大图部署提供明确的资源边界。

**风险与限制**：当前差异以类常量MAX_DYNAMIC_SUBKERNELS=16实现上限，静态形状foreach内核不受影响。测量限于作者的H100配置；串行编译为2177秒降至549秒，不能与302秒跨配置混比。

关联证据：[原始说明 1](https://github.com/pytorch/pytorch/pull/205934)。

### 1.4 模型 & 技术｜[\[Transport\] Add peer notifications to NIXL transports (#204249)（\[Transport\] 为 NIXL 传输添加对端通知 (#204249)）](https://github.com/pytorch/pytorch/commit/f57674a18136dc8869f0883861874788b3b48212)

北京时间：2026-11-21 09:16:52｜来源类型：官方主干 commit｜事件状态：已进入主干。

**事实**：NIXL transport 接入独立通知、随读写携带通知及轮询/等待接口，payload限制为4 KiB。关联 Mooncake 实现以单独的零长度传输携带通知，绕过0.3.14非空RDMA传输丢通知的问题。

**重要性**：为分布式推理与训练的传输完成协同提供可消费的控制信息。

**风险与限制**：NIXL 1.5.2及以前的UCX路径仍依赖发起端检查完成状态，因此实现使用轮询线程；不代表所有传输插件都支持通知。

关联证据：[原始说明 1](https://github.com/pytorch/pytorch/pull/204249)；[原始说明 2](https://github.com/pytorch/pytorch/pull/204136)；[原始说明 3](https://github.com/pytorch/pytorch/commit/f5fcbc1a25eb9d6ddbeaf0845e1121a0c29c6eab)。

### 1.5 模型 & 技术｜[CUDA backend: add a share_weights option, and stop low-memory mode from freeing GPU weights（CUDA 后端：添加 share_weights 选项，并阻止低内存模式释放 GPU 权重）](https://github.com/pytorch/executorch/pull/24937)

北京时间：2026-11-21 11:38:30｜来源类型：官方 GitHub PR｜事件状态：已合并。

**事实**：新增默认关闭的 share_weights compile spec，允许CUDA lowering共享调用者已有GPU权重，避免默认深复制带来的近双份权重显存。释放和autotune临时卸载仅作用于编译过程自己创建的CUDA张量。

**重要性**：让能够在单卡运行、却因导出复制而装不下的模型获得更可控的导出路径。

**风险与限制**：共享权重与低内存模式的组合有明确约束；编译期autotune关闭时，被图写入的真实buffer仍需复制，不能宣称完全零拷贝。

### 1.6 模型 & 技术｜[\[MLX\] Enable Muse Glimmer DFlash2 off-graph batching（\[MLX\] 启用 Muse Glimmer DFlash2 图外批处理）](https://github.com/pytorch/executorch/pull/25336)

北京时间：2026-11-21 20:43:23｜来源类型：官方 GitHub PR｜事件状态：已合并。

**事实**：载入DFlash2 GGUF卷积和候选选择器权重，导出条件候选格并在MLX图外批处理；草稿走贪心路径，再由目标采样器验证。 同系列修正每个DFlash query的滑动注意力窗口，按绝对query/key距离屏蔽越窗键及未填充环形槽位。

**重要性**：把新草稿结构的模型导入、导出与运行时验证连成可用路径。

**风险与限制**：CUDA DFlash2仍属计划，当前不支持的导出组合会提前拒绝；不能由MLX合入推断CUDA已经上线。

关联证据：[原始说明 1](https://github.com/pytorch/executorch/pull/25335)。

### 1.7 模型 & 技术｜[\[MPS\] Fix conv weight grad for batch >= 2^16 (#193577)（\[MPS\] 修复 batch >= 2^16 时的卷积权重梯度 (#193577)）](https://github.com/pytorch/pytorch/commit/4f275b1bbf5510c1d5fd5a6b7c2334c7f4ac0034)

北京时间：2026-11-22 04:45:14｜来源类型：官方主干 commit｜事件状态：已进入主干。

**事实**：MPSGraph卷积权重梯度在batch达到65536时可能给出错误值；现在将batch切成最多65535的块再累加，覆盖conv1d、conv2d和conv3d。

**重要性**：修复Apple设备大batch训练的明确数值正确性边界。

**风险与限制**：仅针对卷积权重梯度路径；不代表全部MPS大batch算子已验证。

关联证据：[原始说明 1](https://github.com/pytorch/pytorch/pull/193577)。

## 二、LLVM/MLIR 最新进展

### 2.1 深度洞见｜[\[RFC\] Use LLVM libc for constant folding of math library calls（\[RFC\] 使用 LLVM libc 对数学库调用进行常量折叠）](https://discourse.llvm.org/t/92936/1)

北京时间：2026-11-21 21:58:23｜来源类型：官方技术论坛｜事件状态：RFC 已提出。

**事实**：提案希望先用LLVM libc计算已有sin/sinf常量折叠，减少相同IR与目标因编译宿主libm不同而生成不同常量的问题。 维护者建议经APFloat接口调用，而不是让Analysis直接依赖libc；讨论确认静态舍入实现正在推进，作者计划等待exp的APFloat概念验证后更新草稿。

**重要性**：给跨宿主、跨目标编译的数值可复现性提出具体架构路径。

**风险与限制**：仍为RFC和草稿实现；需保留目标语义、浮点异常与舍入条件，不等于所有数学函数已切换。

### 2.2 深度洞见｜[RFC: Add ExcludeMacroDefinitionRegex to clang-tidy（RFC：为 clang-tidy 添加 ExcludeMacroDefinitionRegex）](https://discourse.llvm.org/t/92939/1)

北京时间：2026-11-22 05:46:39｜来源类型：官方技术论坛｜事件状态：RFC 已提出。

**事实**：提议按宏定义的spelling location文件名正则过滤诊断，以抑制GoogleTest/GoogleMock等第三方宏体噪声，同时保留宏实参中用户代码的诊断。

**重要性**：提供比展开位置和头文件过滤更精确的诊断控制设计。

**风险与限制**：尚未落地；嵌套宏、字符串化与token拼接产生的scratch-space位置仍需特殊处理。

### 2.3 深度洞见｜[A New `ub.assume` Operation（新的 `ub.assume` 操作）](https://discourse.llvm.org/t/92834/19)

北京时间：2026-11-22 03:42:19｜来源类型：官方技术论坛｜事件状态：讨论新进展。

**事实**：原提案以返回值的纯ub.assume向SSA值附加范围等信息，谓词不成立时产生poison；它不能直接降成具有立即UB语义的llvm.assume。 本轮回复给出nuw加法反例：范围缩窄仍可能为poison，建议在range lattice旁跟踪may-be-poison位，并由显式断言建立非poison根。

**重要性**：把范围推理与可投机执行的前提分开，直接影响MLIR分析和降级正确性。

**风险与限制**：这是讨论新进展，尚无统一设计决定；不是宣称新操作已经合入。

### 2.4 深度洞见｜[\[RFC\]\[Offload\] Introducing AMDGCN-flavored SPIR-V support for OpenMP（\[RFC\]\[Offload\] 为 OpenMP 引入 AMDGCN 风格的 SPIR-V 支持）](https://discourse.llvm.org/t/92547/5)

北京时间：2026-11-21 20:42:20｜来源类型：官方技术论坛｜事件状态：讨论新进展。

**事实**：原提案通过Comgr在运行时把单份AMDGPU风格SPIR-V设备镜像编译成本机代码，覆盖Clang驱动、offload runtime、DeviceRTL等路径。 新回复解释：可通过语言扩展适应目标的runtime库适合保留SPIR-V通用实例，再由JIT决定目标细节；仍可并存专用目标库。

**重要性**：补充多架构单镜像部署的运行库边界，减少为每个gfxip重复携带通用库的设计压力。

**风险与限制**：属于讨论新进展和设计理由，未证明实现完成，也不包括本RFC范围外的Fortran支持。

### 2.5 工具 & 产品｜[BUDDY MLIR 0.1.7（BUDDY MLIR 0.1.7）](https://github.com/buddy-compiler/buddy-mlir/releases/tag/release/v0.1.7)

北京时间：2026-11-22 01:12:02｜来源类型：官方 Release｜事件状态：正式发布。

**事实**：Buddy MLIR 0.1.7发布x86_64和riscv64的Python wheel、归档、buddy-cli及LLVM 24.0.0git工具包；同日0.1.6也发布对应资产。 本日发行工作流开启LLVM_ENABLE_DUMP，修复Release工具包缺少MLIR dump()符号、导致Triton下游调试辅助代码链接失败的问题；仍保持Release构建。

**重要性**：为Buddy与Triton-RISC-V下游提供可取得的构建产物，并面向实际链接阻断调整发行配置。

**风险与限制**：LLVM安装缓存键不含构建开关，采用新开关时需清理缓存；工具包发布不代表所有下游链接和模型部署均已通过验证。

关联证据：[原始说明 1](https://github.com/buddy-compiler/buddy-mlir/pull/1074)；[原始说明 2](https://github.com/buddy-compiler/buddy-mlir/releases/tag/release/v0.1.6)。

## 三、Triton & TileLang 技术动态

### 3.1 模型 & 技术｜[\[NVIDIA\] Use K96 automatically for eligible single-CTA scaled FP4 MMA（\[NVIDIA\] 为符合条件的单 CTA 缩放 FP4 MMA 自动使用 K96）](https://github.com/triton-lang/triton/pull/12875)

北京时间：2026-11-21 06:34:34｜来源类型：官方 GitHub PR｜事件状态：已合并。

**事实**：SM103后端对符合描述符与scale布局条件的单CTA缩放FP4 MMA自动启用K96。K=256可分为96+96+64，窄N=64布局需要四列scale stride，不满足时继续使用K64。

**重要性**：将更合适的矩阵指令分块交给编译器判断，减少内核作者手工选择。

**风险与限制**：重结合可能改变硬件浮点舍入；GB200强制K96曾触发非法指令，不能泛化到所有Blackwell设备。

### 3.2 模型 & 技术｜[\[FPSAN\] compact_exp2 and probability_cast for shard-invariant attention（\[FPSAN\] 用于分片不变注意力的 compact_exp2 和 probability_cast）](https://github.com/triton-lang/triton/pull/12881)

北京时间：2026-11-21 10:30:42｜来源类型：官方 GitHub PR｜事件状态：已合并。

**事实**：新增可选FPSan紧凑指数编码与概率cast，使online softmax的校正因子在FP8转换前后保持可比较的签名；支持通过SMEM/TMEM读到的真实字节检测拷贝、别名和中间写入。

**重要性**：为不同attention分片策略的语义比对提供专门的验证能力。

**风险与限制**：这是sanitizer编码，原生执行仍用常规exp2、FP8和MMA；分片间重新计算block scale可能改变比较对象，不能当作数值精度保证。

### 3.3 模型 & 技术｜[\[ConSan\] Model implicit cluster barriers during shared-memory reuse（\[ConSan\] 在共享内存复用期间建模隐式 cluster 屏障）](https://github.com/triton-lang/triton/pull/12868)

北京时间：2026-11-21 06:10:52｜来源类型：官方 GitHub PR｜事件状态：已合并。

**事实**：ConSan开始采用Membar记录的屏障阶段，识别原子、归约和布局转换生成的隐式cluster同步；私有helper经Gluon内联后按实际控制流记录可见性。

**重要性**：避免共享内存已同步复用却被错误报告为仍有未完成访问，提高并发验证与后端一致性。

**风险与限制**：覆盖指定cluster同步路径，不代表消除全部并发错误或误报。

### 3.4 模型 & 技术｜[\[Analysis\] Prove ring cursors with matching steps disjoint（\[Analysis\] 证明具有相同步长的环形游标互不相交）](https://github.com/triton-lang/triton/pull/12866)

北京时间：2026-11-21 05:54:36｜来源类型：官方 GitHub PR｜事件状态：已合并。

**事实**：对同一块中起始余数不同、各回边以相同步长推进的共享内存环形游标证明互不相交，允许移除本轮read与refill之间多余屏障，同时保留跨轮顺序。

**重要性**：让流水线缓冲索引分析识别不同SSA基值背后的稳定分离关系。

**风险与限制**：证明范围较窄，假定索引及游标的有符号i32加法不溢出；未知算术仍回退保守别名判断。

## 四、RISC-V 核心新闻

### 4.1 模型 & 技术｜[\[PATCH v3 00/17\] drm/spacemit: add SpacemiT K3 display support（\[PATCH v3 00/17\] drm/spacemit：添加 SpacemiT K3 显示支持）](https://lore-kernel.gnuweeb.org/linux-riscv/20261121-k3-display-v3-0-ec419e8e96b8@gmail.com)

北京时间：2026-11-21 15:40:39｜来源类型：Linux 原始邮件（公开归档）｜事件状态：补丁提案 v3。

**事实**：v3提交Saturn DPU→Innosilicon DP/eDP控制器→DP PHY显示栈；移除内部callback/tracepoint和command-list分组，将page-flip接入vblank，并用runtime PM供电。 作者在Pico-ITX和CoM260-IFX复测内建/模块配置、卸载、page-flip与DPMS；v2的既有Tested-by未沿用，因为DRM驱动改动较大。

**重要性**：推动K3图形栈进入上游审阅，并明确显示事件与电源管理的实现边界。

**风险与限制**：仍为v3补丁提案。仅单primary plane和linear RGB，无缩放、旋转、YUV、DSI、DSC或DP音频；链路上限HBR2，依赖带显示支持的U-Boot。

### 4.2 模型 & 技术｜[\[PATCH v2 0/3\] RISC-V: KVM: Lessen steal time update overhead（\[PATCH v2 0/3\] RISC-V：KVM：降低 steal time 更新开销）](https://lore-kernel.gnuweeb.org/linux-riscv/20261121173526.892344-1-den@valinux.co.jp)

北京时间：2026-11-22 01:35:23｜来源类型：Linux 原始邮件（公开归档）｜事件状态：补丁提案 v2。

**事实**：v2为SBI STA共享结构补上写入顺序，同时批量进行用户内存访问、跳过run_delay未变时的steal-time更新。 作者在SpacemiT K3以KVM selftest来回进出guest，5轮各100万次、每轮预热1万次；非竞争/竞争情况下vCPU线程CPU时间分别从2900.01/2876.42 ns降至2669.59/2670.01 ns。

**重要性**：减少虚拟机频繁退出场景的统计更新成本，并处理共享结构的正确发布顺序。

**风险与限制**：仅为补丁v2和微基准，约7%—8%的改善不能外推为整个虚拟机工作负载收益。

### 4.3 模型 & 技术｜[`sret` must trap when `mstatus.TSR` is set（当 `mstatus.TSR` 置位时，`sret` 必须陷入）](https://github.com/riscv/riscv-isa-manual/pull/3555)

北京时间：2026-11-21 09:02:35｜来源类型：官方 GitHub PR｜事件状态：规范澄清已合并。

**事实**：特权规范明确：在S-mode且mstatus.TSR=1时执行SRET必须触发非法指令异常，将该处should改为must，并与TSR控制位既有要求对齐。

**重要性**：为实现、形式模型和架构测试消除规范措辞歧义。

**风险与限制**：这是既有要求的澄清，不是新ISA扩展或新的硬件能力。

### 4.4 工具 & 产品｜[Ruyi Firmware 20261121（Ruyi Firmware 20261121）](https://github.com/openRuyi-Project/ruyifirmware/releases/tag/20261121)

北京时间：2026-11-21 20:53:27｜来源类型：官方 Release｜事件状态：已发布预发行版。

**事实**：新增Milk-V Pioneer及SOPHGO EVB v1/v2的SG2042固件，按board ID选择设备树，提供SD启动和SPI安装映像；Pioneer接入存储、双RTL8125B网口及UEFI RTC。 修复Megrez/StarPro64保存DDR ECC设置后无法启动时的USB恢复，并更新Sv39系统中256 GiB以上PCI窗口的UEFI驱动仿真处理。

**重要性**：将多板卡UEFI部署路径和恢复条件收敛到公开可下载的预发行版。

**风险与限制**：Pioneer升级前需先启动ruyi/v7.3.5-3或同时含EFI MMIO/保留设备时钟修复的内核，否则读取RTC可能panic；Pisces不在支持board ID内。

### 4.5 模型 & 技术｜[Fix missed LMUL and vstart checks for Zvknhab.（修复 Zvknhab 遗漏的 LMUL 和 vstart 检查。）](https://github.com/riscv/sail-riscv/pull/2143)

北京时间：2026-11-21 12:18:57｜来源类型：官方 GitHub PR｜事件状态：已合并。

**事实**：Sail向量SHA-2模型按SEW选择128/256位element-group约束，补齐Zvknha/Zvknhb的LMUL与vstart有效性检查。

**重要性**：修正作为验证参考的向量密码模型，避免参考模型接受不合规配置。

**风险与限制**：是模型检查修复，不能据此推断实际芯片密码性能或安全性已验证。

### 4.6 模型 & 技术｜[Sstc: check hvip.VSTIP writes while the vstimecmp comparison is true（Sstc：在 vstimecmp 比较为真时检查 hvip.VSTIP 写入）](https://github.com/riscv/riscv-arch-test/pull/3163)

北京时间：2026-11-21 10:31:57｜来源类型：官方 GitHub PR｜事件状态：已合并。

**事实**：新增8个hvip.VSTIP pending覆盖bin和24个mip.VSTIP只读别名bin；先保持vstimecmp比较成立，再结束比较以检查实际写入状态。 原有只读hip.VSTIP检查会被持续成立的比较遮蔽；新用例可暴露QEMU丢弃HS-mode hvip写入及mip别名写入的问题。

**重要性**：增强Sstc/Hypervisor中断状态验证，属于新增可发现缺陷的架构测试能力。

**风险与限制**：相关QEMU测试组仍被排除并注明问题链接，不能把新增测试写成模拟器缺陷已修复。

### 4.7 模型 & 技术｜[Add linux-riscv64 to the runner build（为 runner 构建添加 linux-riscv64）](https://github.com/openRuyi-Project/github-runner-riscv64/commit/80b939b0b6e6715072aa8cc55e3ec336488cde05)

北京时间：2026-11-22 00:22:54｜来源类型：官方主干 commit｜事件状态：已进入主干。

**事实**：基于dkurt与RevyOS既有RISC-V移植，为GitHub Actions runner加入linux-riscv64交叉发布和manifest核对；关联打包流程组合.NET 10、Node.js 20与24，README公开自托管配置及手动版本工作流。

**重要性**：为RISC-V主机运行CI任务提供具体部署链路。

**风险与限制**：构建与打包实现已进入仓库；正式可下载版本和RISC-V主机上的运行兼容性尚待确认，README中的下载示例不能作为版本交付证明。

关联证据：[原始说明 1](https://github.com/openRuyi-Project/github-runner-riscv64/commit/770ace9126d94b5a5bffba6c863b1ca79fb3a2bc)；[原始说明 2](https://github.com/openRuyi-Project/github-runner-riscv64/commit/58c51d3c974d98593a7323ced9e892f9234b3402)。

## 五、AI 业界重磅

### 5.1 深度洞见｜[腾讯姚顺雨团队深夜回应一切：算力不足、模型刷榜、雷霆大思考](https://zhidx.com/p/612973.html)

北京时间：2026-11-21 08:30:07｜来源类型：原始专业媒体报道｜事件状态：已发布。

**事实**：智东西参与腾讯混元团队Reddit AMA并发布问答整理：团队解释DSA降低了MLA解码计算强度，从而改变GQA/MLA架构选择；Hy4 flash的混合SWA、LatentMoE与Engram来自模型和推理系统协同设计。 团队确认Hy4-mini、Flash-Lite及面向24GB显存设备的模型属于路线图，并表示端侧模型会按边缘GPU特性采用不同设计。

**重要性**：提供国产模型架构、KV Cache与部署硬件协同取舍的直接团队解释。

**风险与限制**：这是原始采访/问答整理；规划不等于版本已发布。有关成本与架构收益是团队陈述，缺少独立复现实验；Hy4当前不支持音频理解或生成。

### 5.2 工具 & 产品｜[Workflow lab: Turning context into a plan with Specs and the Figma agent（工作流实验室：用 Specs 和 Figma agent 将上下文转化为计划）](https://www.figma.com/blog/workflow-lab-specs-and-the-figma-agent/)

北京时间：2026-11-21 06:17:00｜来源类型：官方博客｜事件状态：已发布。

**事实**：官方示例把PRD、Hex数据、FigJam流程、标注设计和组件说明放入Specs，再由Figma agent生成实现计划；经人工评审后通过Figma MCP将计划及关联上下文交给Codex。 文章注明Specs对所有套餐开放beta，可用于网页、桌面和移动端。

**重要性**：给设计到编码智能体的交接提供可复用的上下文组织方法。

**风险与限制**：这是工作流教程，不能把演示当作新的模型能力或无需人工审核的交付保证；beta状态保留。

## 六、总结与趋势观察

- **部署链路更重视产物与目标环境的匹配。** Buddy发布跨架构工具包，Ruyi固件把SG2042板卡识别及内核兼容条件写入公开预发行版；两者都说明“可取得的产物”还需要对应的运行前提。[Buddy](https://github.com/buddy-compiler/buddy-mlir/releases/tag/release/v0.1.7)、[Ruyi Firmware](https://github.com/openRuyi-Project/ruyifirmware/releases/tag/20261121)
- **数值可复现性需要同时约束计算与验证语义。** LLVM libc常量折叠提案关注宿主数学库差异，FPSan概率编码关注attention分片后的FP8签名；二者仍分别受舍入、异常与scale一致性限制。[LLVM RFC](https://discourse.llvm.org/t/92936/1)、[FPSan](https://github.com/triton-lang/triton/pull/12881)
- **虚拟化正确性与性能都落到共享状态细节。** KVM steal-time v2调整SBI共享结构写入顺序和更新频率，Sstc新测试则避免持续成立的比较掩盖虚拟中断位写入错误；前者仍在审阅，后者并不代表QEMU问题已修复。[KVM](https://lore-kernel.gnuweeb.org/linux-riscv/20261121173526.892344-1-den@valinux.co.jp)、[Sstc](https://github.com/riscv/riscv-arch-test/pull/3163)

## 附录：信源说明

本期主要使用PyTorch/ExecuTorch、Buddy Compiler、LLVM/MLIR与Triton项目原始记录，LLVM技术论坛、Linux RISC-V原始邮件的公开归档，RISC-V规范与Sail/架构测试仓库、openRuyi固件及部署仓库，以及Figma官方文章和智东西原始采访报道。

正式Release、预发行版、主干合入、补丁提案和讨论回复分别反映不同交付阶段。测试及性能数字限于原文列出的模型、硬件、负载与配置；作者或厂商测量不等于独立复测。教程示例、规划中的模型及发行配置调整不代表所有部署环境均已验证。
