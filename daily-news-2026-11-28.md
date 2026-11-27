# Codex 技术情报每日动态（2026-11-28）

调研窗口：北京时间（2026-11-27 05:44:21，2026-11-28 06:00:28]。

覆盖方向：PyTorch 生态、LLVM/MLIR、Triton & TileLang、RISC-V、AI 模型与研究、软件基础设施及芯片部署。

信息口径：依据原始公告、正式发行、技术讨论、实现记录和有编辑责任的原创报道；区分提案、实现落地与正式发行，性能及测试结论均限定具体条件。

## 今日要闻

- [SUICE V0.1 发布，遥感专家模块接入通用语言模型；辰光一号01星试验平台披露在轨推理验证。](https://www3.xinhuanet.com/tech/20261127/3bf5b749829a488b941255e83b38132d/c.html)
- [TileLang v0.1.16 正式发行，加入昇腾 950 PTO 后端，并扩展 CuTeDSL、多根缓存和并行编译。](https://github.com/tile-ai/tilelang/releases/tag/v0.1.16)
- [腾讯 Hy3D 3.6 更新高模及 PBR 材质生成，原创报道确认 Token Hub API 与三类生产工具接入。](https://pchome.tech/article/news/tencent-hunyuan-hy3d-3d-model-pbr)

## 今日索引

- **PyTorch**：MPS 的 NVFP4 分组矩阵乘与短三角 LU 优化；XPU 主机内存注册；Arm VGF 状态和 Vela 载荷所有权。
- **LLVM/MLIR**：协程帧空间复用、并行控制流建模、MLIR 文本兼容边界；无符号算子和 RCCL 通信正确性。
- **Triton & TileLang**：TileLang v0.1.16；AMD 描述符原子操作与缩放转换；Intel 块 I/O、BF16 降低和 SLM 对齐。
- **RISC-V**：ESP32-P4 UART 流控 RFC；进迭张量类型库与机器人模型部署；CVA6 异常语义及香山页表预取模拟。
- **AI 业界**：Hy3D 3.6、SUICE V0.1，以及 LTX-2.5 在 Trainium3 上的公开部署实验。

## 一、PyTorch 生态核心动态

### 1.1 模型 & 技术｜[\[MPS\] Use MPP for NVFP4 2d x 2d _scaled_grouped_mm_v2 (#206519)（\[MPS\] 对 NVFP4 2d x 2d _scaled_grouped_mm_v2 使用 MPP (#206519)）](https://github.com/pytorch/pytorch/commit/fe7ad294b196bac0155d1757e8a6d26bcf17c269)

北京时间：2026-11-28 00:23:25｜来源类型：官方主干提交｜事件状态：主干实现已落地。

**事实**：MPS 的块缩放读取器改用 8 字节半块加载 NVFP4，允许 K 分组从奇数块开始，并解除二维×二维 grouped_mm 使用 MPP 的限制。 提交者在 M5 Pro、M=N=1024、K=8192、8 组配置上报告吞吐从 2.6 提高到 18.6 TFLOPS；奇数块起点为 17.2 TFLOPS。

**重要性**：低精度分组矩阵乘可利用 MPP 路径，且修正分组边界对齐带来的后端限制。

**风险与限制**：数字针对给定矩阵和设备；二维×三维等路径基本持平，不能外推整模型加速或所有 Apple GPU。

### 1.2 模型 & 技术｜[\[MPS\] Speed up LU solves for short triangular tiles (#206893)（\[MPS\] 加速短三角分块的 LU 求解 (#206893)）](https://github.com/pytorch/pytorch/commit/20c4942dfd8366b60a256d7151542e9d5eb850be)

北京时间：2026-11-28 01:48:43｜来源类型：官方主干提交｜事件状态：主干实现已落地。

**事实**：为至多 16 行、少于 32 个右端列的尾块编译直接代入变体，绕过通用 32×32 三角块的装载与填充开销。 提交者的 FP32 3×3、批量 65,536、1 个右端列案例从 8.868 降至 2.381 ms/次；较大矩阵多为接近持平。

**重要性**：对单个矩阵很小、总批量很大的求解任务提供具体可复核的优化。

**风险与限制**：这是提交内测量；收益取决于形状，缺少统一硬件信息与独立复测，不应当作全尺寸 LU 普遍提升。

### 1.3 模型 & 技术｜[\[xpu\] Introduce pin_memory&unpin_memory on XPU (#195819)（\[xpu\] 在 XPU 上引入 pin_memory&unpin_memory (#195819)）](https://github.com/pytorch/pytorch/commit/5d781579b3d8ab5a612c7ce9d9b2055e0a26f6f0)

北京时间：2026-11-27 23:19:55｜来源类型：官方主干提交｜事件状态：主干实现已落地。

**事实**：新增 XPU 主机内存注册/注销及 Python 工具，使已有 CPU 内存区域可以注册为 USM host 内存，无需先换成另一块张量存储。 实现要求 SYCL_COMPILER_VERSION≥20260200、设备支持 ext_oneapi_register_host_memory；当前指针和长度均须按系统页大小对齐。

**重要性**：为已有缓冲区接入 XPU 数据搬运路径提供基础能力，影响外部数据与推理流水线集成。

**风险与限制**：当前接口有页对齐和驱动能力限制；提交中关于后续解除限制的计划尚不等于已实现。

### 1.4 模型 & 技术｜[Support selective local INT8 and FP32 VGF mutable state（支持选择性的本地 INT8 和 FP32 VGF 可变状态）](https://github.com/pytorch/executorch/pull/25788)

北京时间：2026-11-28 00:43:39｜来源类型：官方 GitHub PR｜事件状态：已合并。

**事实**：Arm VGF 可变状态扩展到零初始化 FP32，与 INT8 缓存并存；mutable_buffer_names 可精确选择内部持久化的缓冲区，其余仍由程序拥有。 导入同时验证输入/变更输出的形状、类型和归一化步长；状态的读取和变更须留在同一委托分区。

**重要性**：使混合精度持久状态部署具备明确的所有权和布局边界，便于 KV 缓存等连续调用链路。

**风险与限制**：别名仅在同一 delegate 内生效，不是跨方法、跨会话的共享资源能力。

### 1.5 模型 & 技术｜[Arm backend: Fix use-after-free of external Vela payloads（Arm 后端：修复外部 Vela 载荷的释放后使用）](https://github.com/pytorch/executorch/commit/640d45f847142dd4ac909a2d9b9e92b367bc46c6)

北京时间：2026-11-28 03:34:33｜来源类型：官方主干提交｜事件状态：主干实现已落地。

**事实**：外部 Vela 块若解析到拥有资源的 FreeableBuffer，旧解析器只保存裸指针，临时对象析构后后端可能继续使用已释放载荷。 现在将缓冲区所有者随 VelaHandles 移入执行句柄，并处理替换、临时标量及解析失败时的释放。

**重要性**：修正 Ethos-U 模型载荷在初始化与执行间的内存生命周期，关系到外部模型数据的可靠部署。

**风险与限制**：这是具体的所有权正确性修复；未有证据确认已被利用或影响所有 Arm 后端，不能扩大为通用安全漏洞。


## 二、LLVM/MLIR 最新进展

### 2.1 模型 & 技术｜[\[RFC\] Re-use Space for SSA Values in the Coroutine Frame（\[RFC\] 在协程帧中复用 SSA 值的空间）](https://discourse.llvm.org/t/92992/1)

北京时间：2026-11-27 21:44:23｜来源类型：官方技术论坛｜事件状态：新技术提案，尚在讨论。

**事实**：提案对跨恢复点存活的 SSA 值做基本块级生命周期分析，让存活区间不相交的值共享协程帧空间；通过布局回调判断重叠，不主动移动 load/store。 作者报告约 300 个翻译单元中 continuation 状态大小的几何平均降低 16.7%；单个运行时案例整体改善约 1%，编译指令数子集平均增加 0.08%。

**重要性**：把协程帧布局从消除对齐空洞推进到时间上的空间复用，可减少保存状态和缓存行占用。

**风险与限制**：仍为 RFC；本轮不覆盖 alloca 的联合优化，样本测量与单例运行收益不能代表所有协程。

### 2.2 模型 & 技术｜[Control flow interfaces and parallelism（控制流接口与并行性）](https://discourse.llvm.org/t/92989/1)

北京时间：2026-11-27 18:43:27｜来源类型：官方技术论坛｜事件状态：新技术提案，尚在讨论。

**事实**：新讨论指出 scf.parallel 与 scf.forall 对 region 控制流边的描述不一致：并发迭代没有传统环，但省略环又削弱 may-happen-before 推理，影响依赖冲突和竞态分析。 讨论同时核对 RegionBranchOpInterface 只能列出允许分支、可保守多报边的语义，并提出 early-exit、unreachable 和类似 monad 多分支的表示边界。

**重要性**：这是 MLIR 并行 IR 的基础建模问题，直接影响后端调度与合法性分析的推理前提。

**风险与限制**：尚未决定接口修改方案或合并实现，不能将问题讨论写成已修复的并行分析能力。

### 2.3 深度洞见｜[Stable MLIR text for library and kernel authors（面向库与内核作者的稳定 MLIR 文本）](https://discourse.llvm.org/t/92987/1)

北京时间：2026-11-27 08:29:33｜来源类型：官方技术论坛｜事件状态：新技术提案，尚在讨论。

**事实**：Vx 作者希望直接在语言的 mlir! 宏内嵌入 MLIR，以便优化器看见库内核且跨硬件降低；其标准库已有百余段手写 MLIR，因版本演进而遇到兼容性问题。 维护者回应 MLIR 有意不提供可移植性保证；generic IR 语法相对稳定，但不涵盖操作/方言验证规则。语言前端可自行把选定语法转换到当前 IR。

**重要性**：为 Buddy、torch-mlir 等链路依赖的内核文本与方言版本管理明确实际边界。

**风险与限制**：讨论没有建立新的兼容承诺；generic IR 相对稳定不能替代方言语义或文本语法的版本协议。

### 2.4 深度洞见｜[\[RFC\] Universal CFI（\[RFC\] 通用 CFI）](https://discourse.llvm.org/t/92955/19)

北京时间：2026-11-27 21:26:03｜来源类型：官方技术论坛｜事件状态：讨论新进展。

**事实**：uCFI 主线希望基于 kCFI 的类型哈希机制，将前向控制流保护推广至普通 C/C++、虚函数和 vtable，降低对 LTO 的依赖；硬件后向保护仍优先。 本次新回复补充 RISC-V Zicfilp：lpad 使用 20 位标签。发帖者在一次 Linux 内核试构建中，对约 50,000 种函数签名观察到约 1,100 次标签碰撞；函数签名标签方案与实现仍为 PR。

**重要性**：给出了 RISC-V 硬件 CFI 标签位宽与签名哈希冲突的具体实验边界，便于比较软件哈希与硬件标签策略。

**风险与限制**：这是一位参与者的特定试构建结果，未代表部署验证或安全失效率；方案尚未进入正式 psABI。

### 2.5 模型 & 技术｜[\[TorchToTMTensor/TorchToLinalg\] Fix unsigned sort, scatter_reduce, max/min and clamp（\[TorchToTMTensor/TorchToLinalg\] 修复无符号 sort、scatter_reduce、max/min 和 clamp）](https://github.com/llvm/torch-mlir/pull/4924)

北京时间：2026-11-28 00:10:22｜来源类型：官方 GitHub PR｜事件状态：已合并。

**事实**：修复把 Torch 张量转为 signless builtin 张量后丢失源 dtype 符号性的路径；uint8 的 128—255 不再按负数比较。 sort/kthvalue 改用无符号比较，scatter_reduce 的界限及除法使用无符号语义，混合 uint8/int16 的 max/min/clamp 按源类型零扩展；旧路径可能把 200 变为 -56。

**重要性**：保障 PyTorch 模型经 TMTensor/Linalg 导入后的整数语义，对量化数据和索引结果至关重要。

**风险与限制**：本补丁针对指定 lowering；差异仍列出 StableHLO 与 ONNX 路径的相关 xfail，不能宣称所有导出后端均已修复。

### 2.6 模型 & 技术｜[\[HIP\] Fix the RCCL all-gather send pointer（\[HIP\] 修复 RCCL all-gather 的发送指针）](https://github.com/iree-org/iree/pull/25302)

北京时间：2026-11-27 18:25:51｜来源类型：官方 GitHub PR｜事件状态：已合并。

**事实**：HIP all-gather 先前向 ncclAllGather 传入主机侧 iree_hal_buffer_t 地址；现在先解析实际设备指针，再叠加 buffer/binding 偏移。 作者在两张 MI350X、ROCm 7.2.2 上复现旧实现非法内存访问，修改后双 rank all-gather 与既有 StableHLO all-reduce 通过。

**重要性**：修复多卡部署的实际通信崩溃路径，直接关联 IREE 分布式推理的输入地址正确性。

**风险与限制**：两项 shard.grid 用例仍因 nightly 缺少 shard 方言无法编译，验证没有覆盖全部分片链路。


## 三、Triton & TileLang 技术动态

### 3.1 模型 & 技术｜[v0.1.16（v0.1.16）](https://github.com/tile-ai/tilelang/releases/tag/v0.1.16)

北京时间：2026-11-27 17:25:35｜来源类型：官方 Release｜事件状态：正式发行。

**事实**：TileLang v0.1.16 正式发行，加入昇腾 950 的 target="pto" 后端、T.vmi、SimdVF 内自动向量化、独立 AIV 执行和片上缓冲区精确放置。 同一版本扩展 CuTeDSL 的 cluster TMA、SM120/SM121 块缩放 MMA，并纳入进程式 par_compile、多根缓存和 tvm_ffi 上的 NVRTC 编译。

**重要性**：将新硬件后端与并行编译、缓存和设备编译器支持汇集到可追踪的正式版本，便于团队安排集成验证。

**风险与限制**：不同后端的支持范围并不相同；发行说明列出的能力不能替代各硬件、驱动和内核组合的实测。

### 3.2 模型 & 技术｜[\[AMD\] Accept buffer descriptors and soffset in buffer atomic RMW/CAS ops（\[AMD\] 在缓冲区原子 RMW/CAS 操作中接受缓冲区描述符和 soffset）](https://github.com/triton-lang/triton/pull/12964)

北京时间：2026-11-28 02:06:35｜来源类型：官方 GitHub PR｜事件状态：已合并。

**事实**：amdg.buffer_atomic_rmw 与 buffer_atomic_cas 现在可直接接受 buffer descriptor 及可选 soffset，并使用调用方描述符的边界信息。 原有标量指针路径保持；面向 amd.cdna5.buffer 等的描述符 atomic_* 前端 API 留待后续 PR。

**重要性**：补齐 AMD 显式描述符内存路径的原子操作表示，便于内核在统一边界语义下使用加载与原子访问。

**风险与限制**：本次是 IR/lowering 支持，不能提前宣称高层描述符原子 API 已发布。

### 3.3 模型 & 技术｜[\[AMD\]\[Gluon\] Fix scaled_downcast for CDNA4 verification and repeated registers（\[AMD\]\[Gluon\] 修复 scaled_downcast 的 CDNA4 验证和重复寄存器）](https://github.com/triton-lang/triton/pull/12691)

北京时间：2026-11-28 00:09:20｜来源类型：官方 GitHub PR｜事件状态：已合并。

**事实**：scaled_downcast 按目标检查每线程唯一值的缩放组：CDNA4 与 CDNA5 使用各自分组规则，避免以 CDNA5 八值规则拒绝合法 CDNA4 块。 降低阶段移除寄存器维的零基，修正重复寄存器对应错误 scale 的问题；同时去掉完整写入前多余的 FP4 清零。

**重要性**：修正低精度转换的验证与数值映射边界，有助于跨 AMD 代际复用 Gluon 量化内核。

**风险与限制**：影响限于相应目标、scale 布局与转换路径；本条没有整模型性能提升数据。

### 3.4 模型 & 技术｜[\[Intel\] Lower bf16 -> f32 to fpext when bf16 arithmetic is native（\[Intel\] 在原生支持 bf16 运算时将 bf16 -> f32 降低为 fpext）](https://github.com/intel/intel-xpu-backend-for-triton/pull/8945)

北京时间：2026-11-27 23:23:00｜来源类型：官方 GitHub PR｜事件状态：已合并。

**事实**：在原生 BF16 且 IGC≥2.43 的路径上，以 llvm.fpext 代替不透明转换调用，使 LLVM 可以把转换移过 select 并正确缩窄谓词加载。 作者在 B580、IGC 2.43.9 的 BF16 pyhpc_isoneutral_mixing 三次均值中，将默认路径从 2.866 降到 2.559 ms；主要内核从 53.7 降至 29.0 μs。

**重要性**：以具体驱动门槛恢复谓词内存优化，说明数值转换的 IR 形态也会改变内核访存。

**风险与限制**：旧版、未知或 LTS 驱动保留原 builtin；测试只覆盖给定工作负载，FP32→BF16 路径没有改变。

### 3.5 模型 & 技术｜[\[BLOCKIO\] Add order-independent tile size calculation（\[BLOCKIO\] 添加与顺序无关的分块大小计算）](https://github.com/intel/intel-xpu-backend-for-triton/pull/8122)

北京时间：2026-11-27 20:17:05｜来源类型：官方 GitHub PR｜事件状态：已合并。

**事实**：2D block I/O 按寄存器基的维度和步长查找几何形状，避免仅因寄存器基排列不同而缩小合法 tile；描述符仍限制在最内侧两维。 原 DPAS A、f16 64×32 反向寄存器基案例会把 tile 高度由 32 降至 2，产生约 16 倍加载消息；补丁对 2,013,888 种配置做了形状及可用性比较。

**重要性**：让等价数据布局生成更一致的块访问，减少布局转换给后端带来的额外访存指令。

**风险与限制**：配置枚举反映消息形状与块 I/O 资格，并不等同端到端吞吐提升；秩二无掩码路径不变。

### 3.6 模型 & 技术｜[\[Intel\] Keep sub-group transpose SLM rows dword-aligned for 8/16-bit types（\[Intel\] 对 8/16 位类型保持子组转置 SLM 行的双字对齐）](https://github.com/intel/intel-xpu-backend-for-triton/pull/8911)

北京时间：2026-11-27 06:09:22｜来源类型：官方 GitHub PR｜事件状态：已合并。

**事实**：8/16 位子组转置使用原 17/34 字节行距时可能与向量加载对齐要求冲突，导致部分 lane 数据错位；现在使用 20/36 字节行距，并在分配与降低间共用行距计算。 作者在 CRI 上验证独立转置、原始 packed-nibble dot 重现及新测试，报告对应 int8/fp8 组合 27/27 通过。

**重要性**：修复 FP8、INT8 和部分 16 位布局的实际错误结果，属于低精度矩阵内核的关键正确性边界。

**风险与限制**：特定硬件与 tile 配置触发；i16 scratch 占用随行距增加，需要其他目标继续覆盖，不能扩大为全部 Intel GPU 已验证。


## 四、RISC-V 核心新闻

### 4.1 模型 & 技术｜[\[PATCH RFC 1/2\] serial: esp32: Enable hardware RTS/CTS flow control（\[PATCH RFC 1/2\] serial: esp32: 启用硬件 RTS/CTS 流控制）](https://lore-kernel.gnuweeb.org/linux-riscv/20261127192514.174468-1-vitorhg20080@gmail.com/)

北京时间：2026-11-28 03:25:10｜来源类型：内核邮件 RFC｜事件状态：RFC 新补丁与回环验证已公开，尚未合并。

**事实**：在 ESP32-P4 初始 Linux 支持 RFC 的 UART 主线上，新补丁为 CRTSCTS 配置 64 字节 RX FIFO 阈值和 TX CTS 门控，并让 serial core 知道 UART 自动处理 CTS。 作者在 Waveshare ESP32-P4-Module-DEV-KIT 的 UART1 内部回环中验证：撤销 RTS 后传输停止，重新使能后缓存字节完整到达；配套第二补丁修正低有效 CTS 的逻辑状态报告。
关联原文：[[PATCH RFC 2/2] serial: esp32: Report active-low CTS correctly on ESP32-S3 and ESP32-P4](https://lore-kernel.gnuweeb.org/linux-riscv/20261127192514.174468-2-vitorhg20080@gmail.com/)。

**重要性**：补齐串口在持续数据流中的背压与状态语义，为 RISC-V ESP32-P4 的 Linux 外设和控制台集成提供可审阅的方案。

**风险与限制**：仍是未合并 RFC，依赖既有 UART 修订；验证限内部回环，ESP32-S3 实机测试仍待完成，不代表全部芯片串口支持已上线。

### 4.2 模型 & 技术｜[feat: initial header-only tensor layout & block-scale type system（feat: 初始的仅头文件张量布局与块缩放类型系统）](https://github.com/spacemit-com/spine-tensor-types/commit/9e9e192905ad5b06a12f8238f9578324bb292169)

北京时间：2026-11-27 17:55:35｜来源类型：官方主干提交｜事件状态：主干实现已落地。

**事实**：进迭公开 C++17 header-only 张量布局与 block-scale 类型系统，提供带逻辑行/列轴的 TileLayout、编译期布局合并与等价转换，以及对称、非对称、两级量化缩放的 SoA/AoS 表达。 初始差异包含 FP16/BF16、FP8、打包 INT4 类型、GEMM 样例和 RISC-V rv64gc_zfh 交叉编译配置；对齐默认使用安全的类型对齐，显式高对齐另行声明。

**重要性**：把量化数据布局、块缩放和内核接口的约束放进类型系统，与 RISC-V 推理后端及 MLIR/SPEIR 集成密切相关。

**风险与限制**：这是新公开的库和样例，不是新芯片指令能力或性能纪录；实际零开销、精度和硬件集成仍需下游验证。

### 4.3 模型 & 技术｜[feat(policy_adapter): support split SuperMimic models with a shared reference encoder (#44)（feat(policy_adapter): 支持带共享参考编码器的分离式 SuperMimic 模型 (#44)）](https://github.com/spacemit-robotics/humanoid_common/commit/c826e6259aab0c4047ca5f8731da354433cb2df5)

北京时间：2026-11-28 01:13:53｜来源类型：官方主干提交｜事件状态：主干实现已落地。

**事实**：人形机器人公共控制层支持 SuperMimic encoder/decoder 分离：encoder 将 goal 编为 z，decoder 保持 obs_hist 并从 supermimic_latent 接收潜变量；固定站姿只编码一次。 新 ReferenceEncoder 与 SONIC 复用参考编码基础设施，检查输入宽度、latent 维度和有限输出；项目已有 x86_64 与 K3 riscv64 构建路径。
关联原文：[humanoid_common — 人形机器人通用控制层](https://github.com/spacemit-robotics/humanoid_common/blob/c826e6259aab0c4047ca5f8731da354433cb2df5/README.md)。

**重要性**：扩展机器人动作模型的部署形态，避免分离模型必须重新拼成单个网络才能进入控制层。

**风险与限制**：代码与配置支持不等于所有机型实机验证；本次未提供新的控制时延或动作质量基准。

### 4.4 模型 & 技术｜[Report a PMP fault on a page table walk with the cause of the original access（按原始访问的原因报告页表遍历中的 PMP 故障）](https://github.com/openhwfoundation/cva6/pull/3700)

北京时间：2026-11-27 21:17:21｜来源类型：官方 GitHub PR｜事件状态：已合并。

**事实**：CVA6 统一 MMU 移除只对非 H/Sv39 才按 store 分类的条件；PTE 读取被 PMP 拒绝时，现在按最初 load/store 报告对应 access fault。 新增 Sv32/Sv39 测试：fpga_linux_cv32a6 从 61/63 到 63/63，RVH=1 的 profile_rva23s64 新例也通过；保留 H 相关异常附加字段。

**重要性**：对操作系统和虚拟化异常处理提供符合特权规范的故障分类，避免把 store 故障错误交给 load 路径。

**风险与限制**：这是 RTL 行为修正与指定配置验证，不是新安全认证或已流片芯片的现场修复。

### 4.5 模型 & 技术｜[Fix trap CSR update on ebreak entry to Debug Mode（修复通过 ebreak 进入调试模式时的陷阱 CSR 更新）](https://github.com/openhwfoundation/cva6/pull/3712)

北京时间：2026-11-27 18:46:14｜来源类型：官方 GitHub PR｜事件状态：已合并。

**事实**：当 dcsr.ebreak* 要求 ebreak 进入 Debug Mode 时，不再同时按普通 breakpoint trap 改写异常 CSR 和 mstatus。 提交者用 summit_monterosa 的 Verilator/JTAG/OpenOCD 路径验证：调试入口保持预设 mepc、mcause 与 MIE；关闭 ebreakm 时仍产生正常断点异常。

**重要性**：避免调试入口破坏被调试程序的陷阱上下文，对固件与内核调试的可重复性有实际影响。

**风险与限制**：验证为指定配置的仿真调试链路；并未宣称覆盖所有特权级与硬件实现。

### 4.6 模型 & 技术｜[fix(arch-riscv): Skip cached PTW next-line prefetches（fix(arch-riscv): 跳过已缓存的 PTW 下一行预取）](https://github.com/OpenXiangShan/GEM5/pull/1393)

北京时间：2026-11-27 15:09:11｜来源类型：官方 GitHub PR｜事件状态：已合并。

**事实**：香山 GEM5 在为 L0 PTE 行分配下一行读取前探测共享 L2 TLB；命中时跳过读取，且不更新替换状态或预取使用计数。 差异沿用已有 timing、nextline 和来源条件，并参照昆明湖 RTL 在 page cache 命中时避免发起 walk 的主线。

**重要性**：减少模拟器中不必要的页表访存，使相关预取行为更接近 RTL，影响性能评估的流量解释。

**风险与限制**：未建模 RTL 的超页检查、仲裁和在途去重；也未提供量化整机性能结果，不能称完成周期精确对齐。

### 4.7 模型 & 技术｜[Allow switching out of Go2 dance mode（允许切换出 Go2 舞蹈模式）](https://github.com/spacemit-robotics/unitree-go2/pull/8)

北京时间：2026-11-27 13:57:07｜来源类型：官方 GitHub PR｜事件状态：已合并。

**事实**：选择其他模式时立即停止 dance policy，经 FixStand 进入目标模式；请求 Passive 时继续走已有的保护性蹲伏流程。 作者用本地控制器复现旧行为，并检查舞蹈中切换到 Stand、Crouch、Walk 与 Passive，CTest 和 C++ lint 通过。

**重要性**：改变机器人运行中的模式退出路径，消除只能等待舞蹈完成才响应切换的操作限制。

**风险与限制**：本地控制器检查没有替代实机运动安全验证；过渡姿态、执行器与现场条件仍需集成测试。


## 五、AI 业界重磅

### 5.1 模型 & 技术｜[腾讯混元发布3D大模型Hy3D 3.6：重点优化高模和PBR材质贴图生成能力](https://pchome.tech/article/news/tencent-hunyuan-hy3d-3d-model-pbr)

北京时间：2026-11-27 12:10:12｜来源类型：原创媒体报道｜事件状态：新版本已提供 API（原创报道）。

**事实**：PChome 原创报道腾讯发布 Hy3D 3.6，重点更新高模几何与 PBR 材质贴图生成；报道确认已通过腾讯云 Token Hub 提供 API，Miora、LAP 和 WorkRally 已接入。 报道介绍原生 3D 路线以 VAE 统一稀疏潜表示、DiT 生成匹配参考图的几何与纹理，并在后训练中使用 SFT 和 DPO。

**重要性**：三维生成的可用入口和生产工具接入有具体变化，为游戏、影视资产工作流提供新的模型版本。

**风险与限制**：几何和纹理优势来自腾讯组织的内部盲测，报道未给公开评测集或独立复测；不能据此推定所有资产可直接生产使用。

### 5.2 模型 & 技术｜[我国发布地球大模型引擎SUICE V0.1版](https://www3.xinhuanet.com/tech/20261127/3bf5b749829a488b941255e83b38132d/c.html)

北京时间：2026-11-27 17:55:47｜来源类型：原创媒体报道｜事件状态：V0.1 已发布，在轨试验平台上线。

**事实**：新华社原创报道 SUICE V0.1 发布，以旁路式模块专家向通用语言模型接入遥感理解，支持地物分割、目标识别和变化分析。 同日上线的辰光一号01星在轨技术试验平台，报道称已完成多款大模型在轨推理验证，并披露 270 伏母线、25,000 TOPS 国产抗辐照算力与柔性散热技术。

**重要性**：把遥感专用认知与通用语言接口、天基算力试验关联起来，是有实际发布及在轨验证信息的行业动态。

**风险与限制**：性能与任务领先表述来自发布方，未见完整公开基准；试验平台验证不等于大规模太空数据中心已建成运营。

### 5.3 模型 & 技术｜[Taking Trainium3 to the Track: Generating Videos with LTX-2.5（将 Trainium3 带上赛道：使用 LTX-2.5 生成视频）](https://huggingface.co/blog/video-trainium3)

北京时间：2026-11-27 08:00:00｜来源类型：官方技术博客｜事件状态：部署实验与代码已公开。

**事实**：Hugging Face 作者公开 LTX-2.5 在 trn3.48xlarge 上的视频与音频生成实验：TP-4×SP-8 使用 32 个 NeuronCore，模型加载后生成 10 秒、2048×1024 片段耗时 22—23 秒。 新公开的工程路径包括分块编译、FP32 GELU、参数打包、主机线程隔离与设备视频解码；初始 4 核、960×544 运行约 5 分 20 秒，优化后分辨率与核数同时变化。

**重要性**：提供了从框架图划分、主机瓶颈到专用芯片内核的完整部署实验，能帮助评估非 CUDA 视频生成路径。

**风险与限制**：冷启动模型加载约 3 分钟、首片约 80 秒，部分 NaN 和图加载问题未解释；14 倍时间变化包含核数、分辨率和管线变化，不能当同配置硬件比较。

## 六、总结与趋势观察

- **低精度部署更依赖精确的数据语义。** [torch-mlir 的无符号算子修复](https://github.com/llvm/torch-mlir/pull/4924)与[Gluon 的 scale/寄存器映射修复](https://github.com/triton-lang/triton/pull/12691)分别处理符号信息和缩放组边界；支持某种数据类型仍不足以保证整条转换链正确。

- **系统集成需要验证异常与流控边界。** [CVA6 的 PMP 故障分类](https://github.com/openhwfoundation/cva6/pull/3700)与[ESP32-P4 的 UART 流控 RFC](https://lore-kernel.gnuweeb.org/linux-riscv/20261127192514.174468-1-vitorhg20080@gmail.com/)把检查推进到异常原因、传输背压和逻辑状态，后者仍需等待上游评审与更多硬件测试。

- **跨硬件编译正在同时推进能力与版本约束。** [TileLang 新版](https://github.com/tile-ai/tilelang/releases/tag/v0.1.16)扩展后端和编译入口，[Intel BF16 降低](https://github.com/intel/intel-xpu-backend-for-triton/pull/8945)则明确依赖 IGC 版本。

## 附录：信源说明

主要采用 PyTorch/ExecuTorch、LLVM/torch-mlir/IREE、Triton/TileLang、CVA6、香山与进迭项目的官方技术记录，Linux RISC-V 原始邮件及其公开镜像，以及 Hugging Face 技术博客、新华社和 PChome 原创报道。

RFC 与论坛内容表示提案、讨论或作者实验；主干实现记录不等于正式版本已向所有用户交付。性能数字来自原作者的指定设备与工作负载，未经统一条件复测；媒体转述的厂商测试保留其声明边界。
