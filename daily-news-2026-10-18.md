# Codex 技术情报每日动态（2026-10-18）

**调研窗口**：北京时间 (2026-10-17 05:29:13, 2026-10-18 06:00:24]。

**覆盖方向**：PyTorch 模型与部署、LLVM/MLIR 编译基础设施、Triton/TileLang 内核编译、RISC-V 软件与验证、AI 模型及产业。

**信息口径**：依据官方项目原文、技术提案、上游实现和媒体原创报道；源码合并、讨论提案与产品发布分别表述，作者自测与业务效果数据保留原有条件。

## 今日要闻

- DeepSeek 发布 V4.1-Flash-1017，并开放按成功请求计费的 Search/Fetch 接口。 [原文](https://www.ithome.com/1/014/328.htm)
- RISC-V 内核补丁提案修复仅执行页在回写或 THP fork 后变为可读的问题，并附复现条件。 [原文](https://lists.infradead.org/pipermail/linux-riscv/2026-October/101698.html)
- LLVM 新 RFC 提议禁用或重做 PPCDoubleDouble 通用浮点分析，集中处理该格式与 IEEE 类假设的冲突。 [原文](https://discourse.llvm.org/t/92540/1)
- 新华社报道山东垂域模型在冶金、矿山和港航等场景的部署，并披露企业提供的业务效果数据。 [原文](https://www.news.cn/local/20261017/83d164a5f7854854bbbffc81073a3fc3/c.html)

## 今日索引

- **PyTorch**：ExecuTorch 的 DLPack 互换、实验性 diffusion 扩展、浏览器 WebGPU 与 Cortex-M 量化 attention。
- **LLVM/MLIR**：PPCDoubleDouble 分析 RFC、MLIR 到 RTL 原型、错编与栈防护修复、上下文选项及 GPU 方言构建。
- **Triton & TileLang**：triton-riscv 适配 LLVM 24 与 Buddy MLIR v0.1.0、Intel FP8 转换优化、生产规模 MX4 解码基准。
- **RISC-V**：仅执行页权限、TLB 请求追踪、K1 启动设备树、KVM 测试与 IMSIC 锁设计、CSR 数据库及 RoPE 仿真。
- **AI 业界**：DeepSeek 模型与搜索接口更新，以及山东垂域模型产业应用。

## 一、PyTorch 生态核心动态

### 1.1 模型 & 技术｜[\[PyBindings\] 添加零拷贝 DLPack 互换](https://github.com/pytorch/executorch/pull/23090)

北京时间：2026-10-18 04:25:03｜来源类型：GitHub 实现 PR（pytorch/executorch）｜事件状态：已合并

**动态**：ExecuTorch Python 绑定新增稠密 CPU/CUDA DLPack 输入，并通过 `__dlpack__` 与 `__dlpack_device__` 输出结果；保留指针、形状、步长、偏移和设备信息。CUDA 结果保持在设备上，克隆使用设备到设备拷贝；可在不导入 Torch 的模式下返回 ExecuTorchResult。

**影响**：把 Python 模型输入与轻量运行时连接起来，减少 ATen 依赖和跨框架张量复制，对模型导入及部署接口有直接价值。

**风险与限制**：当前 CUDA 边界仍通过同步建立生产者、delegate 与消费者就绪关系；零拷贝不能理解为全程异步。作者本地未执行 GPU 和 Windows 验证。

相关原文：[\[PyBindings\] Preserve compatibility after removing ATen dependency](https://github.com/pytorch/executorch/pull/23939)。

### 1.2 模型 & 技术｜[添加 diffusion 扩展](https://github.com/pytorch/executorch/pull/23661)

北京时间：2026-10-17 07:21:01｜来源类型：GitHub 实现 PR（pytorch/executorch）｜事件状态：已合并

**动态**：新增实验性 diffusion 扩展，用于导出并运行 Stable Diffusion 1.5 与 SDXL，覆盖 Hugging Face pipeline 发现、组件导出、MLX lowering、动态分辨率和组件级精度配置。

**影响**：从单个算子支持延伸到扩散模型组件的导出及运行契约，提供新的端侧图像生成部署路径。

**风险与限制**：接口明确为实验性；本变更未给出跨设备端到端速度或图像质量对比，不能推导普遍加速。

### 1.3 模型 & 技术｜[在浏览器 Wasm 中运行 WebGPU 模型 (#23938)](https://github.com/pytorch/executorch/pull/23938)

北京时间：2026-10-17 09:23:14｜来源类型：GitHub 实现 PR（pytorch/executorch）｜事件状态：已合并

**动态**：通过默认关闭的 EXECUTORCH_BUILD_WASM_WEBGPU 把 WebGPU 后端接入 Wasm 构建，新增 initWebGPU()，启用对应 Asyncify、异常和最高 4 GB 可增长堆；同时扩展批量 aten.linear 与大尺寸 aten.where 的调度。

**影响**：让 WebGPU delegate 模型具备浏览器端执行通路，扩大无需本地原生安装的部署形态。

**风险与限制**：CI 为构建检查，实际模型执行仍需支持 WebGPU 的浏览器；不能把构建成功当作所有浏览器运行通过。

### 1.4 模型 & 技术｜[Cortex-M：修复量化 attention 的 pass 顺序](https://github.com/pytorch/executorch/pull/23923)

北京时间：2026-10-17 07:35:50｜来源类型：GitHub 实现 PR（pytorch/executorch）｜事件状态：已合并

**动态**：把标量参数转换安排在量化折叠前，将量化节点替换移到 ATen→Cortex-M kernel lowering 之后，避免在 int8/int32 矩阵类型不兼容的半降低 attention 图上重追踪。

**影响**：解除 DeiT Tiny 等模型 attention 模式进入默认 Cortex-M lowering 路径的阻碍；新增 M55/M85 数值和 PTE 序列化覆盖。

**风险与限制**：这是特定量化图及 pass 顺序的修复；没有整模型吞吐、功耗或所有 attention 变体的验证结论。

## 二、LLVM/MLIR 最新进展

### 2.1 模型 & 技术｜[\[展示\] 将 MLIR linalg.matmul 映射到 RTL 脉动阵列](https://discourse.llvm.org/t/92544/1)

北京时间：2026-10-17 22:19:22｜来源类型：项目论坛（LLVM Discourse）｜事件状态：原型展示

**动态**：hw-mlir-lab 展示从 linalg.matmul 经 tiling、padding、bufferization、自定义 MLIR 操作及 C 接口 lowering，连接 Verilator RTL 脉动阵列仿真的端到端流程；作者提供 Docker 环境与 make demo 入口。

**影响**：给 MLIR 到专用矩阵硬件的编译、运行接口与仿真验证提供可复现的教学和原型链路。

**风险与限制**：来源为作者项目展示，不代表 MLIR 上游已纳入该后端，也没有量产硬件吞吐、面积或功耗结论。

### 2.2 深度洞见｜[RFC：重做 ppc_fp128 (PPCDoubleDouble) 的浮点分析](https://discourse.llvm.org/t/92540/1)

北京时间：2026-10-17 06:09:55｜来源类型：项目论坛（LLVM Discourse）｜事件状态：提案讨论

**动态**：RFC 主张移除或禁用 PPCDoubleDouble 的通用浮点分析，将其视为不透明对象；并讨论独立 KnownFPClass 分析、展开为高低两个 double 或限制优化等替代方案。动因包括 normal/subnormal 范围重叠、非规范值、符号位以及 libgcc 与 compiler-rt 行为差异，帖子举出错误常量折叠实例。

**影响**：把浮点格式语义与通用 ValueTracking 假设的冲突明确化，对维护自定义数值类型和编译正确性边界有参考价值。

**风险与限制**：这仍是 RFC，尚未形成默认禁用决定；性能影响与兼容策略有待社区评估，不能写成 LLVM 已改变行为。

### 2.3 重磅｜[\[X86\]\[StackClash\] 修复使用 -fstack-clash-protection 时跳过 guard page 的问题](https://github.com/llvm/llvm-project/pull/230951)

北京时间：2026-10-17 16:41:16｜来源类型：GitHub 实现 PR（llvm/llvm-project）｜事件状态：已合并

**动态**：修复栈分配恰为 StackProbeSize 整数倍、AlignOffset 为零时的边界：循环剩余一整页，而尾部分配按小于一页处理，可能未插入必要探测而跳过 guard page。

**影响**：涉及栈冲突防护本身的完整性，属于编译器安全边界修复。

**风险与限制**：已合并不等于所有发行工具链都已回移；原文没有提供 CVE 编号或可利用性结论。

### 2.4 重磅｜[回移 5.1.2 第 3 和第 4 批](https://github.com/onnx/onnx-mlir/pull/3698)

北京时间：2026-10-17 07:32:47｜来源类型：GitHub 实现 PR（onnx/onnx-mlir）｜事件状态：已合并

**动态**：ONNX-MLIR 把 main 上第 3、4 批安全与稳定性改动回移到 0.5.1.2 分支，差异覆盖运行时外部常量边界、执行会话、ONNX shape helper、NNPA 运行时及 Python 后端等路径。

**影响**：将模型编译和执行链中的多类防护带回稳定分支，对部署侧比只更新开发主干更直接。

**风险与限制**：这是回移 PR，不能等同于新的正式 Release；各后端影响范围仍需对照所用分支及模型格式。

### 2.5 重磅｜[\[mlir\]\[linalg\] 修复降秩 insert_slice 向量化写入错误维度的问题](https://github.com/llvm/llvm-project/pull/226490)

北京时间：2026-10-17 17:30:36｜来源类型：GitHub 实现 PR（llvm/llvm-project）｜事件状态：已合并

**动态**：为降秩 tensor.insert_slice 显式构造非丢弃结果维度的 permutation map，并让 masked write 使用对应维度计算掩码。原先把向量写回尾部维度，可能丢失列元素，使用自定义向量尺寸时还会越过行边界；新增向量化与未向量化结果一致的集成检查。

**影响**：直接影响 Linalg 向量化结果正确性及内存访问边界，是模型编译链需要关注的错编修复。

**风险与限制**：修复覆盖所述降秩与偏移场景；它没有证明所有非平凡 permutation 都无运行开销，也没有表明 masking 始终更便宜。

**讨论新进展**：FedericoBruzzone 于北京时间 10-17 21:48:36 的[新回复](https://discourse.llvm.org/t/92542/2)指出，permutation_map 是仿射属性，在重写时构建不等于运行时动态计算；最终代价取决于数据重排、访存 lowering 与目标硬件。

相关原文：[Question on permutation maps vs masking in rank-reducing vectorization (Ref: PR #226490)](https://discourse.llvm.org/t/92542/2)。

### 2.6 模型 & 技术｜[\[Option\] 将 Passes、Scalar、Instrumentation 和 CGData 选项改为按上下文设置](https://github.com/llvm/llvm-project/pull/232599)

北京时间：2026-10-17 10:31:13｜来源类型：GitHub 实现 PR（llvm/llvm-project）｜事件状态：已合并

**动态**：把四组选项从全局切到 LLVMContext；IR pass 在运行时读取上下文配置，PipelineTuningOptions 向 PassBuilder 传递库选项。相关修复分别补齐 lld/llvm-lto2 的 PTO 初始化及 mlir-runner 执行引擎的选项传递。

**影响**：使嵌入式编译器和多上下文编译的优化配置更可控；对通过 MLIR ExecutionEngine 调用 LLVM 的工具链尤其相关。

**风险与限制**：旧全局读法切换时会出现参数被忽略的兼容问题；不能假设所有工具都已完成迁移。

相关原文：[\[LTO,lld\] Initialize lto::Config::PTO with LibraryOptions](https://github.com/llvm/llvm-project/pull/233069)；[\[ExecutionEngine\] Set mlir-runner's library options in the LLVMContext](https://github.com/llvm/llvm-project/pull/233131)。

### 2.7 模型 & 技术｜[\[AMDGPU\] 停止在 gfx1250 的 clause 中合并 load 和 store](https://github.com/llvm/llvm-project/pull/231269)

北京时间：2026-10-17 09:08:00｜来源类型：GitHub 实现 PR（llvm/llvm-project）｜事件状态：已合并

**动态**：区分允许混合访存类型的目标与 CDNA5/gfx1250：后者 clause 中 load、store 类型冲突，原先沿用 GFX10 的合并规则可能触发未定义行为。差异新增目标特征并调整 hard-clause 插入。

**影响**：让 AMDGPU 后端按具体代际规则生成访存序列，避免把相近架构的调度假设直接套用到新硬件。

**风险与限制**：没有提供整模型性能变化；这是目标特定正确性变化，不应外推到所有 AMD GPU。

### 2.8 模型 & 技术｜[\[mlir\]\[LLVMIR\] 预编译 NVVM 和 ROCDL 操作声明 (NFC)](https://github.com/llvm/llvm-project/pull/233106)

北京时间：2026-10-18 00:09:45｜来源类型：GitHub 实现 PR（llvm/llvm-project）｜事件状态：已合并

**动态**：为 NVVM、ROCDL 方言各自建立包含操作头的局部 PCH，并禁止下游复用，减少多个 source shard 重复解析。作者报告受影响对象的 CPU 编译时间由 103.12 秒降至 87.48 秒，减少 15.17%。

**影响**：缩短 MLIR GPU 方言的重复构建工作，对频繁迭代后端的开发环境有实用价值。

**风险与限制**：数据属于特定对象及作者测量环境，不是完整 LLVM 构建或运行时推理加速。

相关原文：[\[mlir\]\[LLVMIR\] Precompile NVVM and ROCDL operation declarations](https://github.com/llvm/llvm-project/pull/232976)。

## 三、Triton & TileLang 技术动态

### 3.1 模型 & 技术｜[\[build\] 支持 LLVM 24 并固定 Buddy MLIR v0.1.0](https://github.com/RuyiAI-Stack/triton-riscv/pull/87)

北京时间：2026-10-17 22:41:17｜来源类型：GitHub 实现 PR（RuyiAI-Stack/triton-riscv）｜事件状态：已合并

**动态**：triton-riscv 适配 LLVM 24 的 strict properties、浮点 contract 标志及 pass-plugin API，固定 Buddy MLIR release/v0.1.0；sanitizer 从全局 -sanitizer-type 改成 sanitizer-attributes<asan/tsan> 的 pass 参数，避免 opt 在加载插件前解析未知标志。

**影响**：直接贯通 Buddy Compiler、Triton 与 RISC-V 后端的版本兼容和 ASan/TSan 对象生成链路。作者报告 check-buddy 618 项通过、triton-shared 290 项通过及 9 项预期失败。

**风险与限制**：FlagGems 本地覆盖的 3829 项中仍有 10 次 native crash；部分示例需要原生 RVV。上述结果不能解释为 RISC-V 端全模型兼容或运行验证全部通过。

### 3.2 模型 & 技术｜[\[Intel\] 在 FP8E4M3 -> FP16 上转换中降低 NaN 检测成本](https://github.com/intel/intel-xpu-backend-for-triton/pull/8456)

北京时间：2026-10-17 13:40:26｜来源类型：GitHub 实现 PR（intel/intel-xpu-backend-for-triton）｜事件状态：已合并

**动态**：Intel Triton 后端把 FP8E4M3 软件上转换改成精确乘 256、幅值比较和 NaN 选择；254 个有限编码保持位级一致。作者在 B580、M=128/N=K=4096 的三轮交错中位数测试中，td_kn 从 572.2 降到 324.2 微秒、td_nk 从 583.4 降到 324.6 微秒。

**影响**：说明 tensor descriptor 路径的低精度性能瓶颈可来自 dot layout 重复转换，并给出可量化的后端优化。

**风险与限制**：fast-math 模式下保留编码结果仍未规定；LTS 驱动与 PVC 未覆盖，NaN 统一为 0x7E00 的行为也需保留。

### 3.3 模型 & 技术｜[mx4_to_fp32：添加生产环境 MX4 通信形状的解码侧 (#1309)](https://github.com/meta-pytorch/tritonbench/pull/1309)

北京时间：2026-10-17 06:55:40｜来源类型：GitHub 实现 PR（meta-pytorch/tritonbench）｜事件状态：已合并

**动态**：为 MX4 解码基准加入生产 quantize_comm/All2All_Pooled 消息形状，覆盖 group_size=8/16 和最高 2,421,580,288 个逻辑元素；直接构造 packed MX4 输入，避免先生成大型 FP32 中间结果导致 H100 开源 CI 内存不足。

**影响**：补足低比特通信编码与解码评测不对称，使生产规模和实际分组参数进入基准覆盖。

**风险与限制**：新增的是生产规模基准能力，没有披露解码 kernel 提速，也不代表通信系统吞吐已提升。

## 四、RISC-V 核心新闻

### 4.1 重磅｜[\[PATCH v1\] riscv：mm：不要在 pte_wrprotect() 中把仅执行 PTE 变为可读](https://lists.infradead.org/pipermail/linux-riscv/2026-October/101698.html)

北京时间：2026-10-18 00:22:28｜来源类型：官方邮件列表（linux-riscv 官方邮件列表归档）｜事件状态：补丁提案

**动态**：补丁报告指出，shadow stack 写保护逻辑无条件补上 _PAGE_READ，可能让 PROT_EXEC 的仅执行页在文件回写或匿名 THP fork 后变为可读。方案仅在原 PTE 可写时设置该位，并让 ptep_set_wrprotect() 复用同一 helper；作者附 QEMU、ext4 与 THP 复现。

**影响**：涉及用户态执行专用内存的权限边界，对 RISC-V 内核内存管理和 shadow stack 兼容尤为直接。

**风险与限制**：当前为 v1 提案，未证明已合入或回移；与另一份恢复 PTE 原子性的补丁存在修改重叠，需要协调应用顺序。

### 4.2 模型 & 技术｜[\[PATCH v5\] riscv：mm：追踪 TLB 刷新请求](https://lists.infradead.org/pipermail/linux-riscv/2026-October/101690.html)

北京时间：2026-10-17 21:49:45｜来源类型：官方邮件列表（linux-riscv 官方邮件列表归档）｜事件状态：补丁提案

**动态**：方案新增 riscv_tlb_flush_request，记录失效区间、stride、ASID、目标 CPU bitmap 及作用域，用于关联 MM、调度、IPI 与固件活动。v5 根据评审把事件定义移到 arch/riscv，按已记录 bitmap 计算 mask weight，并要求仅计数访问也满足 perf tracepoint 权限。

**影响**：为分析 RISC-V 地址空间刷新开销提供具体请求上下文，同时处理 CPU mask 并发一致性和内核地址信息暴露问题。

**风险与限制**：仍是补丁提案；事件描述分发前的请求，不能据此识别每次调用实际使用 SBI RFENCE 还是 Linux 跨 CPU 机制。作者的 QEMU 测试不等于全硬件覆盖。

相关原文：[\[PATCH v2\] riscv: mm: Trace TLB flush path selection](https://lists.infradead.org/pipermail/linux-riscv/2026-October/101691.html)。

### 4.3 模型 & 技术｜[\[PATCH v2 0/2\] riscv：dts：spacemit：将 K1 启动节点标记为 bootph-pre-ram](https://lists.infradead.org/pipermail/linux-riscv/2026-October/101703.html)

北京时间：2026-10-18 02:01:39｜来源类型：官方邮件列表（linux-riscv 官方邮件列表归档）｜事件状态：补丁提案

**动态**：提案把 K1 SPL 在 DRAM 初始化前需要的时钟、UART、启动介质、EEPROM、PMIC 电源等节点标记上移到 Linux DTS，覆盖 BPI、Milk-V、MusePi 和 OrangePi 多块板；v2 将通用 PMIC I²C8 标记集中到 k1.dtsi，并重基到 spacemit dt-for-next。

**影响**：让 Linux 与 U-Boot 共用启动阶段的设备树描述，减少各板私有 overlay，贴近 RISC-V 板级启动与部署链路。

**风险与限制**：作者完成 DTB 构建及 dtbs_check，并在 Milk-V Jupiter 的 Linux 环境检查设备响应；明确没有在硬件上运行由这些树构建的新 SPL。

相关原文：[\[PATCH v2 1/2\] riscv: dts: spacemit: k1: mark boot-critical SoC nodes bootph-pre-ram](https://lists.infradead.org/pipermail/linux-riscv/2026-October/101704.html)；[\[PATCH v2 2/2\] riscv: dts: spacemit: mark K1 board boot devices bootph-pre-ram](https://lists.infradead.org/pipermail/linux-riscv/2026-October/101705.html)。

### 4.4 模型 & 技术｜[\[PATCH 5/5\] KVM：selftests：测试 ucall MMIO 区域的保护](https://lists.infradead.org/pipermail/linux-riscv/2026-October/101688.html)

北京时间：2026-10-17 19:02:13｜来源类型：官方邮件列表（linux-riscv 官方邮件列表归档）｜事件状态：补丁提案

**动态**：系列方案为测试库预留 ucall MMIO 所用的 memslot 1，并新增保护检查，验证冲突映射被拒绝且 ucall 在尝试前后仍正常。讨论新进展中，作者自查承认首版测试误用 slot 2：实际验证了 GPA 重叠的 EEXIST，而非覆盖 slot 1 应触发的 EINVAL，并计划在 v2 修正。

**影响**：这是新增 Hypervisor 测试能力及其实际覆盖边界的技术讨论，揭示了“测试通过”与真正保护目标之间的差别。

**风险与限制**：首版尚未验证其原定 slot 1 覆盖保护，且会跳过不支持 MMIO ucall 或只读内存的配置；不能把该提案当成已生效的内核防护。

相关原文：[\[PATCH 4/5\] KVM: selftests: Reserve memslot 1 for the ucall MMIO address](https://lists.infradead.org/pipermail/linux-riscv/2026-October/101687.html)；[\[PATCH 5/5\] KVM: selftests: Test protection of the ucall MMIO region](https://lists.infradead.org/pipermail/linux-riscv/2026-October/101697.html)。

### 4.5 模型 & 技术｜[\[PATCH\] RISC-V：KVM：为 IMSIC 唤醒状态使用 raw lock](https://lists.infradead.org/pipermail/linux-riscv/2026-October/101689.html)

北京时间：2026-10-17 20:29:06｜来源类型：官方邮件列表（linux-riscv 官方邮件列表归档）｜事件状态：讨论新进展

**动态**：原方案针对调度器持有 raw runqueue lock 时调用 IMSIC put：现有 rwlock_t 在 PREEMPT_RT 下可睡眠，原提议另加 raw spinlock 保护 CPU/HGEI 唤醒状态。讨论新进展中，Anup Patel 建议直接把 vsfile_lock 改为 raw spinlock；Andrew Jones 同意，说明原读锁的并发区很短，将发送 v2。

**影响**：明确 RISC-V 虚拟化中调度上下文与锁类型的约束，并把设计从嵌套两把锁转向简化锁结构。

**风险与限制**：这是具体设计回复与后续修改承诺，尚不能称 v2 已发布或修复已合入；更大临界区的实时行为仍需验证。

### 4.6 模型 & 技术｜[fix(csr)：mhpmcounterN/mhpmeventN 在 M-mode 中始终存在](https://github.com/riscv/riscv-unified-db/pull/2747)

北京时间：2026-10-17 08:07:00｜来源类型：GitHub 实现 PR（riscv/riscv-unified-db）｜事件状态：已合并

**动态**：RISC-V Unified Database 移除 mhpmcounterN、mhpmeventN 的 HPM_COUNTER_EN 存在性条件：M-mode 下依据 Sm 建模；高半部保留 RV32 的 XLEN 条件。

**影响**：修正机器可读 CSR 存在性模型，避免把硬件计数使能配置误当作 CSR 是否存在，影响从统一数据库生成的架构测试和配置工具。

**风险与限制**：这是数据库语义修正，不意味着所有计数器都必须实现完整功能，也不是新 ISA 扩展批准。

### 4.7 模型 & 技术｜[npu-test：添加 RoPE 算子示例](https://github.com/OpenXiangShan/AME_gem5/pull/10)

北京时间：2026-10-17 20:36:23｜来源类型：GitHub 实现 PR（OpenXiangShan/AME_gem5）｜事件状态：已合并

**动态**：AME_gem5 新增 FP32 交错 RoPE 的 4×4 tile 示例，将列配对 gather、正余弦、逐元素乘加接入矩阵指令及 make test；同时提供 host golden 路径，主机编译关闭浮点融合以匹配目标分步运算。

**影响**：将 Transformer 的位置旋转计算纳入香山相关矩阵扩展的算子仿真链路，增加模型工作负载与指令验证之间的连接。

**风险与限制**：示例与仿真不代表量产芯片速度；PR 列出的验证命令也不能自动证明所有模型精度和全系统运行已通过。

## 五、AI 业界重磅

### 5.1 模型 & 技术｜[DeepSeek V4.1 Flash-1017 新版本发布：性能提升、耗时缩短，许愿式编程一键成真](https://www.ithome.com/1/014/328.htm)

北京时间：2026-10-17 16:17:20｜来源类型：媒体原创报道（IT之家）｜事件状态：已发布

**动态**：DeepSeek V4.1 Flash-1017 已在官网、App 和 API 更新，并开放权重；官方说明采用带自反馈的强化学习改善从零构建项目能力，面向 DeepSeek Harness、CodeBuddy、WorkBuddy 优化。同次开放 Search/Fetch，各为 5 元/千次成功请求，相关编程工具中的免计费阶段至北京时间 11 月 1 日 00:00。

**影响**：模型更新和检索接口一并扩大 coding agent 可用的模型与工具链选择。

**风险与限制**：基准优势属于官方报告且依赖 harness；按次搜索费用与模型 token 费用是两套计费维度，不能把更短输出等同于所有任务成本下降。

相关原文：[DeepSeek V4.1 Flash 新版本：许愿式编程一键成真](https://api-docs.deepseek.com/zh-cn/news/news261017/)；[更新日志](https://api-docs.deepseek.com/zh-cn/updates/)；[Search & Fetch](https://api-docs.deepseek.com/guides/search/)；[deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)。

### 5.2 行业 & 人事｜[山东深耕垂域大模型赋能实体产业](https://www.news.cn/local/20261017/83d164a5f7854854bbbffc81073a3fc3/c.html)

北京时间：2026-10-17 17:56:14｜来源类型：媒体原创报道（新华社）｜事件状态：原创报道

**动态**：新华社现场报道山东垂域模型应用：尚智覆盖 40 多个工业场景，相关企业称吨钢综合成本降低 3—6 元；全省遴选 100 个重点推广模型，并集中推介覆盖制造、矿山、港航等领域的 7 个标杆成果。

**影响**：提供模型在实体产业的部署与业务效果线索，显示通用模型之外，行业工艺知识、定制化部署及异构算力服务正在形成具体应用形态。

**风险与限制**：效益数字来自受访企业，不能视为统一独立测评；文中提到的多源异构调度平台仍是即将上线的安排。

## 六、总结与趋势观察

**轻量运行时的部署接口继续扩展。** DLPack 提供跨框架张量互换，Wasm/WebGPU 打开浏览器执行路径，diffusion 扩展补入组件化模型导出与运行；三者分别推进数据交换、执行平台和模型类型，适用边界各不相同。 [依据 1](https://github.com/pytorch/executorch/pull/23090)、[依据 2](https://github.com/pytorch/executorch/pull/23938)、[依据 3](https://github.com/pytorch/executorch/pull/23661)。

**数值和权限语义成为正确性工作的共同重点。** PPCDoubleDouble RFC 质疑通用浮点假设，Linalg 修复向量写回维度与掩码，RISC-V 补丁处理写保护对仅执行页的影响；这些事件都要求保留对象自身的精确语义。 [依据 1](https://discourse.llvm.org/t/92540/1)、[依据 2](https://github.com/llvm/llvm-project/pull/226490)、[依据 3](https://lists.infradead.org/pipermail/linux-riscv/2026-October/101698.html)。

**性能证据需要同时说明计算路径、规模和测量范围。** Intel FP8 优化给出 B580 特定 GEMM 的延迟，MX4 解码新增生产通信形状，MLIR PCH 优化报告局部对象的构建 CPU 时间；这些数字分别描述算子、基准覆盖和开发构建，不能相互替代。 [依据 1](https://github.com/intel/intel-xpu-backend-for-triton/pull/8456)、[依据 2](https://github.com/meta-pytorch/tritonbench/pull/1309)、[依据 3](https://github.com/llvm/llvm-project/pull/233106)。

## 附录：信源说明

本期来源包括 LLVM 项目论坛、linux-riscv 官方邮件列表、上游 GitHub 原始实现记录，以及 IT之家和新华社的原创报道；DeepSeek 官方公告、更新日志与模型仓库提供对应发布事实。

主要项目涉及 ExecuTorch、LLVM/MLIR、ONNX-MLIR、triton-riscv、Buddy MLIR、Intel Triton、TritonBench、RISC-V Unified Database 与 AME_gem5。提案和原型按讨论状态理解；性能数字限于原文的硬件、形状及测量方法，源码合并不等于所有下游产品已采用。
