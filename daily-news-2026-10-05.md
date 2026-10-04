# Codex 技术情报每日动态（2026-10-05）

调研窗口：北京时间 2026-10-04 05:17:05—2026-10-05 06:00:26（起点不含，终点含）。

覆盖方向：PyTorch、LLVM/MLIR、Triton & TileLang、RISC-V，以及 AI 模型研究与推理基础设施。

信息口径：以项目原始公告、技术讨论和代码记录为准；性能数据注明作者测试条件，主干或项目分支变更不等同于稳定版发布。

## 今日要闻

- **智能体结果评测**：[ThinkingBox](https://huggingface.co/blog/microsoft/thinkingbox) 在 507 个有状态工作流中，每项重复 20 次，以最终数据库状态和副作用区分一次成功与持续可靠。
- **嵌入式链接器设计**：[Xtensa lld RFC](https://discourse.llvm.org/t/91608/24) 继续讨论 ABI 信息的存放与读取方式；当前提案聚焦静态嵌入式链接。
- **编译器构建成本**：[LLVM 的 C++20 讨论](https://discourse.llvm.org/t/91692/8) 新增宿主构建实测，报告平均约 5% 的编译时间增幅，采用 PCH 时约 4%。

## 今日索引

- **PyTorch**：MPS 上采样反向传播、Triton 内核导入、设备编译池及 AOTI/Vulkan 正确性。
- **LLVM/MLIR**：CIR 重载、XeGPU 布局回退、x86 ABI 与两项活跃技术讨论。
- **Triton & TileLang**：bf16 常量与舍入语义、Intel 直方图及指针对齐传播。
- **RISC-V**：向量浮点、指针屏蔽、中断测试，以及 Sun252i 的分页、闪存和 XIP。
- **AI 业界**：ThinkingBox、Qwen TRT-RTX 导出、XPU 图捕获及 AMD/NVIDIA 推理路径。

## 一、PyTorch 生态核心动态

### 1.1 模型 & 技术｜[\[MPS\] 将上采样反向传播迁移到 Metal](https://github.com/pytorch/pytorch/pull/199687)

北京时间：2026-10-05 02:25:21｜来源类型：GitHub 官方 PR｜事件状态：已进入主干

**事实**：用 Metal gather 内核替代 MPSGraph resize 梯度模板。作者报告，channels_last 的 (2,150,80,80)→(640,640) 双线性案例由 56.526 ms 降至 2.608 ms，约 21.68 倍。

**重要性**：改善 Apple GPU 上高通道数上采样的训练开销。

**风险与限制**：收益依赖形状与布局；表中一个 contiguous nearest 案例为 0.96 倍，不能视为普遍加速。

主干记录：[提交](https://github.com/pytorch/pytorch/commit/cc6e6f80e506b9d63857558d46c8868bca29daa0)。

### 1.2 模型 & 技术｜[\[inductor\] 在用户自定义内核中支持 Triton 模块全局变量](https://github.com/pytorch/pytorch/pull/196237)

北京时间：2026-10-05 03:40:55｜来源类型：GitHub 官方 PR｜事件状态：已进入主干

**事实**：Inductor 重建自定义 Triton 内核源码时，识别 triton 与 triton.* 模块全局变量，并生成保留原别名的限定导入；覆盖 source_triton、source_tl 一类写法。

**重要性**：让自定义内核的模块导入方式与编译封装衔接，减少模型编译链路中的源码重建缺口。

**风险与限制**：处理范围限于 Triton 模块；不为其他任意 Python 模块提供相同保证。

主干记录：[提交](https://github.com/pytorch/pytorch/commit/7c7dfb52efac8ff1d2e0ee2d2ae6e01952036217)。

### 1.3 模型 & 技术｜[\[inductor\] 为使用 Triton 后端的设备唤醒 AsyncCompile 池，而不只是 cuda/xpu](https://github.com/pytorch/pytorch/pull/199635)

北京时间：2026-10-04 12:05:41｜来源类型：GitHub 官方 PR｜事件状态：已进入主干

**事实**：异步编译池的预热判断改用 ir.is_triton()。MTIA、配置为 Triton 的 CPU，以及已注册 TritonScheduling 的外部设备可进入这一路径。

**重要性**：设备接入不再受 cuda/xpu 名称白名单限制，对自定义编译后端集成有直接意义。

**风险与限制**：外部设备必须完成调度器注册；这不是新增设备内核支持，也没有给出端到端速度测量。

主干记录：[提交](https://github.com/pytorch/pytorch/commit/9dab147463e69119277da42da8a7a55fc289d141)。

### 1.4 模型 & 技术｜[\[aoti\] 克隆通过回退视图算子与常量共享存储的输出](https://github.com/pytorch/pytorch/pull/199323)

北京时间：2026-10-04 09:52:04｜来源类型：GitHub 官方 PR｜事件状态：已进入主干

**事实**：AOTI 通过 output_aliases_constant 追踪回退视图与常量的别名关系，统一三种 wrapper 的输出克隆判断；避免模型释放后返回值仍指向已释放常量存储。

**重要性**：修复 lite mode 等 ATen 回退路径中的悬空输出和错误数据风险。

**风险与限制**：为保证所有权会引入必要的拷贝；主干修复不等同于稳定版本已分发。

主干记录：[提交](https://github.com/pytorch/pytorch/commit/17e0b959d582387ed37352f512e9d371cc3cc2a1)。

### 1.5 模型 & 技术｜[\[aoti\] 对没有 C shim 的 _quantized 算子使用代理执行器](https://github.com/pytorch/pytorch/pull/199336)

北京时间：2026-10-04 09:52:04｜来源类型：GitHub 官方 PR｜事件状态：已进入主干

**事实**：明确列举四个具备 C shim 的 _quantized 算子，其余走代理执行器，修复 lite mode 为 wrapped_quantized_linear 生成不存在 C 符号而导致的编译失败。

**重要性**：恢复量化模型导出后落入 AOTI lite 回退路径的可编译性。

**风险与限制**：作者本地缺少 FBGEMM，新增完整算子测试在本地跳过；辅助脚本覆盖了分派路径，不能据此宣称所有量化模型已验证。

主干记录：[提交](https://github.com/pytorch/pytorch/commit/e8f41c0ef97e73484550856d369c702f69fc10e1)。

### 1.6 模型 & 技术｜[\[Vulkan\] 将归约着色器无法处理的归约保留在 CPU](https://github.com/pytorch/executorch/pull/23245)

北京时间：2026-10-04 22:57:38｜来源类型：GitHub 官方 PR｜事件状态：已合并

**事实**：ExecuTorch Vulkan 分区器将空维度列表、无法处理的完整归约以及受纹理 batch/channel 折叠影响的四维归约留在 CPU；argmax/argmin 限于符合实现约束的连续缓冲区和末维归约。

**重要性**：避免错误下放导致构图失败或错误归约结果，明确移动 GPU 部署边界。

**风险与限制**：CPU 回退可能增加数据移动和延迟；并未扩展着色器可执行的归约种类。

## 二、LLVM/MLIR 最新进展

### 2.1 模型 & 技术｜[\[CIR\] 解析 .cir cc1 输入并支持由其执行 -emit-cir](https://github.com/llvm/llvm-project/pull/228288)

北京时间：2026-10-05 02:41:48｜来源类型：GitHub 官方 PR｜事件状态：已合并

**事实**：ClangIR 可以将 .cir 输入解析、验证为 ModuleOp，并重新发射 CIR；要求 cir.triple 与目标 triple 一致，不重复执行非幂等的 CIR-to-CIR 降低流程。

**重要性**：为保存、检查和重载中间表示补上编译链路环节。

**风险与限制**：当前只支持这一路输出；其他输出仍不支持，部分 verifier 错误还不能定位到源文件位置。

### 2.2 模型 & 技术｜[\[mlir\]\[XeGPU\] 为 sg-to-lane convert_layout 添加 SLM 往返回退](https://github.com/llvm/llvm-project/pull/227884)

北京时间：2026-10-04 10:32:26｜来源类型：GitHub 官方 PR｜事件状态：已合并

**事实**：当现有 shuffle 或布局折叠模式均无法处理时，XeGPU 将每 lane 的片段按源布局写入共享局部内存，再按目标布局读回。

**重要性**：扩展 sg-to-lane 分配流程能合法化的布局组合，避免特定子组切分使整个流水线中止。

**风险与限制**：回退模式优先级低于专用模式；额外 SLM 读写不保证性能优于 shuffle。

### 2.3 模型 & 技术｜[\[clang\]\[X86\] 使 \`regparm\` 与 GCC 保持一致](https://github.com/llvm/llvm-project/pull/227130)

北京时间：2026-10-04 20:30:28｜来源类型：GitHub 官方 PR｜事件状态：已合并

**事实**：调整 x86 regparm 分类：f16、f16b、long double、f128 计入浮点类型；复数走栈，union 按整数处理，并修改单个非零大小字段包装结构体的处理。

**重要性**：减少 GCC 与 Clang 混合构建时的调用约定偏差。

**风险与限制**：原作者用 abi-cafe 做了经验测试；依赖旧行为的代码仍需关注 ABI 变化。

### 2.4 深度洞见｜[\[RFC\] 向 lld/ELF 添加 Xtensa 支持（面向嵌入式目标的静态链接）](https://discourse.llvm.org/t/91608/24)

北京时间：2026-10-05 02:36:18｜来源类型：官方技术论坛｜事件状态：讨论新进展

**事实**：提案为 Xtensa 嵌入式工具链增加静态 ET_REL 链接，兼容 GNU 与 LLVM 目标文件及其重定位语义。讨论新进展：alexrp 指出 MIPS ABI 信息可由 PT_MIPS_ABIFLAGS 到达，而 Xtensa 尚无对等机制；采用 e_flags 或新 PT_XTENSA_INFO 都需要补充 psABI。

**重要性**：设计分歧落在运行时可见的 ABI 描述方式，是嵌入式 LLVM 链路落地前的规范问题。

**风险与限制**：仍为 RFC 讨论；初期范围不含动态链接、TLS 或完整链接器 relaxation。

### 2.5 深度洞见｜[\[问题\] 我们是否计划用 C++20 构建 LLVM？](https://discourse.llvm.org/t/91692/8)

北京时间：2026-10-04 16:34:24｜来源类型：官方技术论坛｜事件状态：讨论新进展

**事实**：围绕 LLVM 宿主构建标准升级的讨论出现实测：cor3ntin 报告使用 Clang 和 GCC 时，切换 C++20 的平均编译耗时约增加 5%，采用 PCH 时约增加 4%。

**重要性**：提供构建基础设施升级的具体成本信息，而非仅有版本迁移意向。

**风险与限制**：结果依赖宿主编译器版本和启用项目；作者未能以 C++20 加 PCH 构建 MLIR/Flang，这也不是已决定升级的公告。

## 三、Triton & TileLang 技术动态

### 3.1 模型 & 技术｜[\[FRONTEND\] 直接构建 bf16 标量常量，而不经过 std::to_string](https://github.com/triton-lang/triton/pull/12103)

北京时间：2026-10-05 04:39:39｜来源类型：GitHub 官方 PR｜事件状态：已合并

**事实**：bf16 常量改为直接从 float 构建，避免 std::to_string 的六位小数表示将 1e-8 截为零；原问题影响 tl.full 及 bf16 乘小常量。

**重要性**：修复会静默改变数值计算的前端语义缺陷。

**风险与限制**：作者在 L40S 上展示七个测试值中五个修复前失败；该证据不代表所有数据类型和硬件的全面验证。

### 3.2 模型 & 技术｜[\[INTERPRETER\] 保留转换舍入模式并修复共享的 RTNE](https://github.com/triton-lang/triton/pull/12028)

北京时间：2026-10-04 08:48:40｜来源类型：GitHub 官方 PR｜事件状态：已合并

**事实**：解释器保留默认窄化转换的最近偶数舍入，并修正浮点中点、尾数进位与次正规数处理；示例 fp32 1.99999 转 bf16 恢复为 2.0。

**重要性**：减少解释器调试结果与预期数值语义之间的偏差。

**风险与限制**：变更针对解释执行和共享转换逻辑；不能据此推导 GPU 内核性能提升。

### 3.3 模型 & 技术｜[\[TritonIntelGPU\] 每个直方图输入只计数一次（移植 triton-lang/triton#11767）](https://github.com/intel/intel-xpu-backend-for-triton/pull/8279)

北京时间：2026-10-05 02:33:39｜来源类型：GitHub 官方 PR｜事件状态：已合并

**事实**：Intel 后端对跨 lane/warp 复制的输入只保留代表线程执行原子累加，移除按复制因子再除回的步骤。作者报告 B580 小直方图循环在 R=4—16 时耗时降至原来的 0.34—0.65 倍。

**重要性**：减少冗余共享内存原子操作，改善具有复制布局的内核。

**风险与限制**：无复制的 R=1 案例性能不变；结果限于所述 B580 微基准。

### 3.4 模型 & 技术｜[\[Analysis\] 在张量选择中保留指针对齐](https://github.com/triton-lang/triton/pull/12109)

北京时间：2026-10-05 04:39:17｜来源类型：GitHub 官方 PR｜事件状态：已合并

**事实**：张量选择保持指针组完整时保留字节对齐，拆分时计入 pointee 大小，使对齐选择后的存储仍可向量化；同一对齐信息链路的另一项变更将 tt.divisibility 传递为 LLVM 参数 align 属性。

**重要性**：让指针对齐信息从分析传播至 LLVM，减少后端丢失优化条件。

**风险与限制**：仅在已证明的对齐约束下适用；原文未给出端到端性能数字。

关联事实来源：[\[LLVM\] Preserve pointer argument alignment during conversion](https://github.com/triton-lang/triton/pull/12107)。

## 四、RISC-V 核心新闻

### 4.1 模型 & 技术｜[主测试生成器中的向量浮点](https://github.com/riscv/riscv-arch-test/pull/2349)

北京时间：2026-10-04 07:56:16｜来源类型：GitHub 官方 PR｜事件状态：已合并

**事实**：RISC-V 架构测试主生成器加入 Vf16、Vf32、Vf64、Zvfbfmin、Zvfbfwma 和 Zvfhmin 支持，同时处理 bf16 溢出与 NaN/窄化转换覆盖点。

**重要性**：新增可执行的向量浮点验证能力，与 RVV 软件栈和硬件一致性验证直接相关。

**风险与限制**：“通过全部仿真并达 100% 覆盖”是作者对本次测试集的报告，不等于所有实现通过认证。

### 4.2 模型 & 技术｜[为所有指针屏蔽扩展（Ssnpm、Smmpm、SmnpmS、SmnpmU）添加测试生成器](https://github.com/riscv/riscv-arch-test/pull/2049)

北京时间：2026-10-04 07:38:45｜来源类型：GitHub 官方 PR｜事件状态：已合并

**事实**：测试生成器覆盖 Ssnpm、SmnpmS、SmnpmU、Smmpm，并以 ZpmCommon.py 集中共享逻辑，取代此前仅针对 Ssnpm 的方案。

**重要性**：把多个特权级的指针屏蔽扩展纳入统一生成与覆盖检查。

**风险与限制**：原作者明确 Zicfiss 指令仍未命中，后续问题 #2032 尚待解决。

### 4.3 模型 & 技术｜[InterruptsS 覆盖点](https://github.com/riscv/riscv-arch-test/pull/2697)

北京时间：2026-10-04 09:00:33｜来源类型：GitHub 官方 PR｜事件状态：已合并

**事实**：增加 InterruptsS 覆盖点，并在支持 SSTC 时于启动代码中将 stimecmp 设为最大值。

**重要性**：扩展 supervisor 中断行为的架构验证基础。

**风险与限制**：作者报告的 100% 覆盖明确排除 H 扩展，不能外推为 Hypervisor 验证完成。

### 4.4 模型 & 技术｜[riscv：为机器模式内核提供 Sv32 MMU，并支持按需分页](https://github.com/YuzukiHD/zephyr/commit/f32e2d1bea4e534ca8b6ae41a7ede24f38afe7d9)

北京时间：2026-10-04 09:45:57｜来源类型：项目 GitHub 提交｜事件状态：已提交项目分支

**事实**：YuzukiHD 的 Zephyr 分支让内核保持机器模式，通过 MPRV 和 MPP=user 翻译数据访问，指令映像保持固定的一对一映射；页错误处理模拟 A/D 位，并处理按需调页。

**重要性**：探索在该 RISC-V 板级软件栈中使用虚拟内存与按需分页的实现路径。

**风险与限制**：陷阱与中断处理使用物理地址，CLINT/PLIC 需特殊访问器；这是项目分支实现，不能视为 Zephyr 上游通用支持。

### 4.5 模型 & 技术｜[samples：从 SPIF 闪存窗口运行整个 Zephyr 映像（XIP）](https://github.com/YuzukiHD/zephyr/commit/0b8ad4037cd4830d72a5cef3c9f0afe7b1c51d0b)

北京时间：2026-10-04 14:59:28｜来源类型：项目 GitHub 提交｜事件状态：已提交项目分支

**事实**：Zephyr 示例将 CONFIG_XIP 映像链接到 SPIF flash 窗口，由配套 SyterKit xip-boot 完成映射和跳转启动；spi0 因与 flash 共用引脚而禁用。

**重要性**：打通闪存映射到操作系统映像执行的板级启动链路。

**风险与限制**：实现位于 YuzukiHD 项目分支，依赖相应引脚与映像布局；不能据此宣称通用板卡均可直接使用。

关联事实来源：[apps: yuzukineko xip-boot](https://github.com/YuzukiHD/SyterKit/commit/82ac8e3057c2892be71fd091ac5d323d67b03c17)。

### 4.6 模型 & 技术｜[为 Sun252i 添加 SPI NOR 闪存控制器（SPIF）支持](https://github.com/YuzukiHD/rt-thread/commit/e1c70aed4dad9113869d291bdbedb41013f1ddb3)

北京时间：2026-10-04 17:28:02｜来源类型：项目 GitHub 提交｜事件状态：已提交项目分支

**事实**：RT-Thread 分支新增 Sun252i SPIF 驱动，提供读取、编程、擦除和采样点调优；EVB 设备树配置 100 MHz 及调优、参数存储偏移。

**重要性**：给该板级生态补齐另一条 RTOS 闪存访问路径。

**风险与限制**：100 MHz 是所提交的配置，正文没有将其转换为已测吞吐量；支持状态限于该项目分支。

### 4.7 模型 & 技术｜[更新向量浮点测试用例（#6）](https://github.com/XUANTIE-RV/damo-rv-ats/commit/4216f6a25f8020ebb232125af24dd143e6c1a8a8)

北京时间：2026-10-04 22:33:33｜来源类型：项目 GitHub 提交｜事件状态：已提交项目分支

**事实**：玄铁 DAMO-RV-ATS 新增 vf 测试目录与构建规则，以及 vfadd、vfclass、vfcvt 等向量浮点用例；差异包含寄存器对齐、重叠、掩码和部分非法 SEW 检查，并增加规范规则文件。

**重要性**：将向量浮点指令纳入该兼容性与随机验证框架，是新增 ISA 验证能力。

**风险与限制**：这里确认的是代码与测试能力新增；提交未提供可据以宣称全部处理器通过兼容性认证的结果。

## 五、AI 业界重磅

### 5.1 模型 & 技术｜[修复 Qwen3.6-35B-A3B 的 TRT-RTX 导出和推理](https://github.com/microsoft/onnxruntime-genai/pull/2654)

北京时间：2026-10-04 11:17:03｜来源类型：GitHub 官方 PR｜事件状态：已合并

**事实**：ONNX Runtime GenAI 修复 Qwen3.6-35B-A3B INT4 QDQ 导出：对 TRT-RTX 展开不支持的融合算子，并把 MoE 输出重新接回残差流，处理禁止 CPU 回退时加载失败及生成空白的问题。

**重要性**：直接影响量化模型从构建器到 RTX 执行提供程序的部署正确性。

**风险与限制**：作者的全模型验证是文本 smoke test，运行时关闭 CUDA graph 和共享 KV buffer；不覆盖 MTP、paged attention 或完整多模态生成。

### 5.2 模型 & 技术｜[\[Bugfix\]\[XPU\] 使 DeepSeek V4 FP8 稀疏解码可被图捕获](https://github.com/vllm-project/vllm/pull/59159)

北京时间：2026-10-04 18:31:15｜来源类型：GitHub 官方 PR｜事件状态：已合并

**事实**：vLLM 将 xpu_sparse_decode_fp8 改为静态形状及向量化索引，移除 .item() 主机同步、数据相关形状和逐 token Python 循环，消除 Level Zero 图捕获不支持这些行为造成的启动失败。

**重要性**：改善 DeepSeek-V4-Flash 在 XPU 图执行模式下的部署可用性。

**风险与限制**：作者同时记录 pipeline parallelism 下仍有另一处启动失败，不能视为所有并行配置均已可用。

### 5.3 模型 & 技术｜[\[ROCm\] GLM-5.2 解码路径：适配解码形状的 MoE/MLA 分块、拆分推测 softmax，以及 bf16 GEMM 路由](https://github.com/sgl-project/sglang/pull/41725)

北京时间：2026-10-05 02:19:37｜来源类型：GitHub 官方 PR｜事件状态：已合并

**事实**：SGLang 针对 gfx950 的 GLM-5.2 解码调整 MoE 填充、MLA 分块、跨 CTA softmax 与 bf16 GEMM 路由；作者在 GLM-5.2-MXFP4、TP8/EP1、104k 上下文配置报告端到端提升 5.7%。

**重要性**：优化低 batch 解码下工作组不足、启动开销及路由失配。

**风险与限制**：收益针对所述配置；多个优化设有行数和布局门槛，prefill 不能直接套用。

### 5.4 模型 & 技术｜[\[Diffusion\] FLUX 3 Action：用于观测编码和去噪步骤的 CUDA graphs](https://github.com/sgl-project/sglang/pull/42171)

北京时间：2026-10-04 22:22:09｜来源类型：GitHub 官方 PR｜事件状态：已合并

**事实**：SGLang 为观测编码及去噪步骤增加两类 CUDA graph；RTX 5090 上 sd-fp8r 的中位延迟由 158 ms 降至 140 ms，峰值显存增加约 304 MiB。

**重要性**：减少具身动作生成链路的主机启动开销，原文同时披露延迟与内存代价。

**风险与限制**：仅适用于单 GPU 且权重驻留的配置；多 GPU、组件卸载或捕获失败会回退 eager。

### 5.5 模型 & 技术｜[feat(cake_gemm)：用于 SM100 / SM103 的掩码分组 FP8 批量 DeepGEMM MoE GEMM 后端（backend="cake"）](https://github.com/flashinfer-ai/flashinfer/pull/6044)

北京时间：2026-10-05 01:48:24｜来源类型：GitHub 官方 PR｜事件状态：已合并

**事实**：FlashInfer 的 batch_deepgemm_fp8_nt_groupwise 增加显式 cake 后端，提供针对 B200/GB300 的专用调度，以及无需分配的 prepared launch 和原生打包 UE8M0 scales 路径。

**重要性**：扩展 MoE FP8 GEMM 的部署和图捕获接口，默认 deepgemm 路径保持不变。

**风险与限制**：仅接受规定 SM 数量、形状、对齐和类型；范围外报错，不做静默回退，也没有启用自动后端选择。

### 5.6 深度洞见｜[智能体说任务完成了。数据库却不这么认为。](https://huggingface.co/blog/microsoft/thinkingbox)

北京时间：2026-10-04 06:56:48｜来源类型：官方技术博客｜事件状态：已发布

**事实**：Microsoft 与 Hugging Face 介绍 ThinkingBox：覆盖 507 个有状态业务工作流，每项运行 20 次，按最终数据库状态及副作用评分；477 项仅用状态判定，30 项另加回答规则。通过 OpenEnv 提供评测接口。

**重要性**：将工具调用成功、任务完成和多次运行的一致性分开，给智能体部署提供可执行的结果核查方法。

**风险与限制**：20/20 是有限重复实验的观测值，不保证未来绝对可靠；文中成本估算也不等于生产账单。

## 六、总结与趋势观察

- **数值与状态正确性仍是部署的基本约束**：[Triton bf16 常量修复](https://github.com/triton-lang/triton/pull/12103) 和 [Qwen TRT-RTX 导出修复](https://github.com/microsoft/onnxruntime-genai/pull/2654) 分别处理静默数值变化与 MoE 残差接线错误；程序能运行并不足以证明计算结果正确。

- **优化越来越依赖具体执行条件**：[MPS Metal 内核](https://github.com/pytorch/pytorch/pull/199687) 与 [FLUX 3 Action CUDA graphs](https://github.com/sgl-project/sglang/pull/42171) 都提供了显著受形状、布局或执行配置约束的测量，收益必须连同适用范围和代价阅读。

- **RISC-V 能力建设同时推进验证与板级落地**：[向量浮点测试生成器](https://github.com/riscv/riscv-arch-test/pull/2349)、[玄铁向量浮点用例](https://github.com/XUANTIE-RV/damo-rv-ats/commit/4216f6a25f8020ebb232125af24dd143e6c1a8a8) 增加指令验证能力；[Zephyr XIP](https://github.com/YuzukiHD/zephyr/commit/0b8ad4037cd4830d72a5cef3c9f0afe7b1c51d0b) 则补齐闪存到系统映像执行的启动路径。

## 附录：信源说明

本期来源包括 Microsoft/Hugging Face 技术博客、LLVM 官方论坛，以及 PyTorch、ExecuTorch、LLVM、Triton、RISC-V 架构测试、玄铁、YuzukiHD、ONNX Runtime GenAI、vLLM、SGLang 和 FlashInfer 的原始代码记录。论坛结论保留提案或讨论状态；项目分支不代表上游采用；测试结果和性能数字限于原作者披露的环境与方法。
