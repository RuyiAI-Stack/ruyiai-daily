# Codex 技术情报每日动态（2026-11-21）

调研窗口：北京时间（2026-11-20 07:11:02，2026-11-21 06:00:34]。

覆盖方向：PyTorch、LLVM/MLIR、Triton & TileLang、RISC-V、AI 模型与研究、推理部署和产业动态。

信息口径：以原始公告、技术讨论和项目变更为依据；区分已交付能力、设计提案与作者测量，并保留适用范围。

## 今日要闻

- **Inkling 2 Mini 公布循环 MoE 架构与本地推理集成。** 约 26B 物理参数、每 token 约 3B 激活参数，通过中间层权重循环复用形成 48 次逻辑层访问，支持图文输入与 1M 上下文。[Hugging Face 原文](https://huggingface.co/blog/inkling2mini)
- **OpenAI 披露恶意蒸馏防御的新评测。** 在其自适应黑盒测试中，GPT-6.2 Astra 对此前/当前请求受保护推理的提取成功率分别为 0.2% 和 0.9%；结果限于该测试设置。[官方原文](https://openai.com/index/combatting-adversarial-distillation)

## 今日索引

- **一、PyTorch 生态核心动态**：分布式任务失败边界、稀疏确定性、ESP 量化、CUDA 程序迁移与 Torch-TensorRT 前端调整。
- **二、LLVM/MLIR 最新进展**：固定默认 locale、`ub.assume` 与 OpenCL 整数除法设计、alive2 服务、形状余数和 dispatch 内存约束。
- **三、Triton & TileLang 技术动态**：Gluon L2 预取、TMA 生命周期与不规则 descriptor 修正、Intel gather 和寄存器预算分块。
- **四、RISC-V 核心新闻**：A210 上游适配、指令长度规范、GUITION 板卡、香山触发器、多 hart 验证和分发/机器人部署链路。
- **五、AI 业界重磅**：Inkling 2 Mini、harness adapters、推理保护、药物化学基准、Connected Apps 与野火无线电转写。

## 一、PyTorch 生态核心动态

### 1.1 模型 & 技术｜[\[c10d\]\[nccl2\] Fail old work during reconfiguration (#198873)（\[c10d\]\[nccl2\] 在重新配置期间使旧工作失败 (#198873)）](https://github.com/pytorch/pytorch/commit/6ae12d1fab68e33e3147f3f23bcd88e76f245137)

北京时间：2026-11-21 04:04:13｜来源类型：官方主干 commit｜事件状态：已进入主干。

**事实**：nccl2 重新配置退役 communicator 时，使该代仍未完成的 Work 失败，保留已完成任务并阻止过渡期间的新提交；修复旧任务输出尚不完整却报告成功的情况。

**重要性**：为弹性分布式训练提供更可靠的失败边界，避免把未完成 collective 当作有效结果。

**风险与限制**：改动针对 nccl2 的重配置路径；作者给出 H100 回归验证，不能据此宣称所有容错场景均已覆盖。

关联证据：[原始变更](https://github.com/pytorch/pytorch/pull/198873)。

### 1.2 模型 & 技术｜[Make sparse-dense matmul deterministic on CUDA under deterministic mode (#202968)（在确定性模式下使 CUDA 上的稀疏-稠密矩阵乘法具有确定性 (#202968)）](https://github.com/pytorch/pytorch/commit/42cde06e48644307d57ad5b5065b152247b687ba)

北京时间：2026-11-21 00:33:11｜来源类型：官方主干 commit｜事件状态：已进入主干。

**事实**：启用确定性算法时，CUDA 的多种稀疏矩阵乘法切换到确定性 cuSPARSE 算法；ROCm 对没有已知确定性实现的相关操作改为报错或按 warn_only 警告。 作者在 H100、CUDA 13.4/cuSPARSE 12.8.6 上的复现由 26 个不确定案例降到 0；高密度单行样本中 COO/CSR mm 耗时分别变为原来的 5.8/17.9 倍。

**重要性**：把可复现性承诺落实到稀疏算子，并明确其性能代价。

**风险与限制**：默认模式不变；ROCm 尚未实机验证，性能数字仅适用于文中样本。

关联证据：[原始变更](https://github.com/pytorch/pytorch/pull/202968)。

### 1.3 模型 & 技术｜[\[CD\] Support customization of downstream Linux wheel builds (#201033)（\[CD\] 支持定制下游 Linux wheel 构建 (#201033)）](https://github.com/pytorch/pytorch/commit/d3633dc1fdbdf6d8e39887fe807a9e2831589c6b)

北京时间：2026-11-21 00:14:17｜来源类型：官方主干 commit｜事件状态：已进入主干。

**事实**：新增显式启用的 custom runtime dependency policy，允许下游通过 hook 控制共享库复制、使用部署环境运行库并覆盖平台 tag；官方默认构建策略保留。

**重要性**：云镜像、裸机和硬件厂商可复用上游 wheel 修复流程，减少维护另一套打包链的成本。

**风险与限制**：这不是新增官方 wheel 变体；外部依赖兼容性、动态链接发现和平台 tag 的真实性仍由下游负责。

关联证据：[原始变更](https://github.com/pytorch/pytorch/pull/201033)。

### 1.4 模型 & 技术｜[\[Espressif\] Adopt the shared Fused Quant pipeline（\[Espressif\] 采用共享 Fused Quant 流水线）](https://github.com/pytorch/executorch/pull/25260)

北京时间：2026-11-21 03:25:10｜来源类型：官方 GitHub PR｜事件状态：已合并。

**事实**：ESP lowering 前接入共享 Fused Quant recipe、channels_last 默认 pass 流程与可移植分解；合格的正有限标量乘法可折叠进量化参数。 同系列 ESP-NN int8 add 与 multiply broadcast 接入提供对应执行路径；不支持的广播、量化或数值范围继续保留 portable 执行。

**重要性**：为 ESP 端模型导出、布局和量化算子衔接提供一致路径。

**风险与限制**：不是任意算子均可加速；零/负标量、部分广播等仍可能 fallback，禁止 fallback 时会拒绝。

关联证据：[原始变更](https://github.com/pytorch/executorch/pull/24579)；[原始变更](https://github.com/pytorch/executorch/pull/24578)。

### 1.5 工具 & 产品｜[CUDA backend: add a PTE device-id remap tool（CUDA 后端：添加 PTE 设备 ID 重映射工具）](https://github.com/pytorch/executorch/pull/25315)

北京时间：2026-11-20 12:26:17｜来源类型：官方 GitHub PR｜事件状态：已合并。

**事实**：新增离线工具，同时改写 PTE 中 planned-memory、tensor extra info 与 delegate spec 的 CUDA 设备编号，支持密集编号和显式映射；无需修改运行时。

**重要性**：让显式导出到 cuda:3 等设备的程序可以迁移到编号不同的机器。

**风险与限制**：映射后设备必须存在；运行时仍在当前设备执行 delegate，折叠到单设备也可能保留主机中转和额外拷贝。

### 1.6 工具 & 产品｜[Remove TorchScript frontend while preserving native runtime APIs（移除 TorchScript 前端，同时保留原生运行时 API）](https://github.com/pytorch/TensorRT/pull/5071)

北京时间：2026-11-21 04:56:55｜来源类型：官方 GitHub PR｜事件状态：已合并。

**事实**：移除 TorchScript 编译前端、Python torch_tensorrt.ts 和 torchtrtc CLI；保留 Dynamo/export 部署依赖的原生运行时算子、TorchBind 与公共 C++ 运行时 API。

**重要性**：旧 TorchScript 编译入口用户需要规划迁移；原生引擎运行能力与前端退场的边界被明确。

**风险与限制**：该合并不等于所有运行时被删除；作者尚未验证端到端 .pt2 推理和 Windows 执行。

### 1.7 工具 & 产品｜[Make ModelOpt optional for unquantized inference（使 ModelOpt 对非量化推理成为可选项）](https://github.com/pytorch/TensorRT/pull/5081)

北京时间：2026-11-21 04:56:58｜来源类型：官方 GitHub PR｜事件状态：已合并。

**事实**：非量化推理不再强制安装或导入 ModelOpt；量化转换器按可用算子注册，ModelOpt 与 TorchAO 的安装及测试开关分离。

**重要性**：减少不具备 ModelOpt 支持的平台运行非量化模型时的依赖阻碍。

**风险与限制**：ModelOpt 量化用户仍需自行安装；作者没有完成全 GPU 编译及 Windows ARM64 验证。

## 二、LLVM/MLIR 最新进展

### 2.1 深度洞见｜[RFC: Fixed C.UTF-8 Default Locale and _l-Only Localisation（RFC：固定 C.UTF-8 默认 locale 与仅通过 _l 实现本地化）](https://discourse.llvm.org/t/92924/1)

北京时间：2026-11-20 18:36:08｜来源类型：官方技术论坛｜事件状态：RFC 已提出。

**事实**：提议把非 _l 的 libc 函数固定到构建时默认 locale（通常为 C.UTF-8），需要本地化的调用显式传入 locale_t；目的包括消除隐式全局/TLS 状态和减小嵌入式、裸机及 GPU 目标的负担。

**重要性**：locale 设计直接影响解析一致性、代码体积和可移植运行库的接口选择。

**风险与限制**：仍为 RFC；提议明确违反 setlocale/uselocale 的 C/POSIX 语义，且需要补齐部分非标准 _l 接口。

### 2.2 深度洞见｜[A New `ub.assume` Operation（新的 `ub.assume` 操作）](https://discourse.llvm.org/t/92834/18)

北京时间：2026-11-20 17:15:12｜来源类型：官方技术论坛｜事件状态：讨论新进展。

**事实**：主线提案为 MLIR 增加能给 SSA 值附加范围等事实的纯操作，以 poison 表达违反假设，便于稀疏数据流分析。 本次作者解释无结果操作无法承载 poison、可能要求立即 UB，并提出分别使用 poison 语义的 ub.assume 与立即 UB 的 ub.assert 的可能方向。

**重要性**：影响 MLIR 的范围推导、投机执行与降低到 LLVM 时的正确性，关联模型导入后的优化基础设施。

**风险与限制**：尚无最终设计共识；poison 与立即 UB 混用仍存在组合问题，不能写成新操作已合入。

### 2.3 深度洞见｜[\[RFC\]\[OpenCL\] Defined integer '/' and '%', and non-fatal UBSan reports for them（\[RFC\]\[OpenCL\] 为整数 '/' 和 '%' 定义行为，并为其提供非致命 UBSan 报告）](https://discourse.llvm.org/t/92493/17)

北京时间：2026-11-20 21:58:05｜来源类型：官方技术论坛｜事件状态：讨论新进展。

**事实**：方案改为把既有 llvm.masked.{u,s}{div,rem} 扩展到标量，而不引入新的 poison intrinsic；-fdefined-int-divrem 默认仅计划用于 OpenCL C 与 C++ for OpenCL。 窗口内评审进一步指出，不宜通过 target hook 改变通用 SelectionDAG opcode 的语义；可能陷阱的版本需要 chain，评审者建议新 opcode。

**重要性**：明确整数除零、溢出和 sanitizer 行为的设计边界，对 GPU 与 RISC-V lowering 均有意义。

**风险与限制**：仍处于设计及分步 PR 阶段；CUDA、HIP、SYCL 未被默认扩展，SelectionDAG 方案尚有分歧。

关联证据：[原始回复](https://discourse.llvm.org/t/92493/18)。

### 2.4 工具 & 产品｜[New alive2 compiler explorer instance（新的 alive2 Compiler Explorer 实例）](https://discourse.llvm.org/t/92604/12)

北京时间：2026-11-21 04:00:06｜来源类型：官方技术论坛｜事件状态：讨论新进展。

**事实**：维护者报告在线实例增加 llubi、alive-exec，以及用于 RV64 与 AArch64 后端的实验性 translation validation 工具 backend-tv；同时缩减首次访问下载量。

**重要性**：让编译器后端变换的验证更容易进入开发者工作流。

**风险与限制**：backend-tv 明确为实验工具，不能替代所有目标代码正确性验证；本次为服务功能更新而非新 ISA 支持。

### 2.5 模型 & 技术｜[\[fx_importer\]\[Torch\]\[TorchToArith\] Match torch.remainder for integer remainder（\[fx_importer\]\[Torch\]\[TorchToArith\] 使整数余数匹配 torch.remainder）](https://github.com/llvm/torch-mlir/pull/4891)

北京时间：2026-11-20 15:17:36｜来源类型：官方 GitHub PR｜事件状态：已合并。

**事实**：符号尺寸的 % 改为导入 aten.remainder，整数常量折叠与 TorchToArith lowering 按 Python 余数语义修正符号；例如 -33 % 8 得到 7，而非 C 语义的 -1。

**重要性**：修通含负值尺寸补齐公式的 FX/ONNX 形状导入链，与 Buddy/RuyiAI 关注的模型导入直接相关。

**风险与限制**：范围是 remainder 的导入、折叠与降低；不能由此推断任意动态形状模型均可编译。

### 2.6 模型 & 技术｜[\[DispatchCreation\] Add an option to limit the bytes a dispatch binds（\[DispatchCreation\] 添加限制 dispatch 绑定字节数的选项）](https://github.com/iree-org/iree/pull/25299)

北京时间：2026-11-20 16:10:05｜来源类型：官方 GitHub PR｜事件状态：已合并。

**事实**：新增 dispatch binding size 上限选项，在融合及 producer clone 时估算外部静态张量和结果占用，避免大常量拆分后又被融合成超出目标地址空间的 dispatch。

**重要性**：为受 32 位映射或并发绑定内存限制的目标保留可执行的编译结果。

**风险与限制**：默认不设上限；动态形状张量按零字节估计，因此该限制目前只覆盖静态形状。

## 三、Triton & TileLang 技术动态

### 3.1 模型 & 技术｜[\[BAKCEND\] Expose bulk L2 prefetch in NVIDIA Gluon（\[BAKCEND\] 在 NVIDIA Gluon 中暴露批量 L2 预取）](https://github.com/triton-lang/triton/pull/12867)

北京时间：2026-11-21 05:47:14｜来源类型：官方 GitHub PR｜事件状态：已合并。

**事实**：新增 hopper.bulk.prefetch(pointer, num_bytes, pred=True)，在 SM90 及更新设备上发出 cp.async.bulk.prefetch.L2.global；支持标量、张量输入和条件预取。

**重要性**：允许 Gluon 内核显式把后续权重范围提前搬入 L2。

**风险与限制**：依赖 SM90+；代码生成测试不等于所有模型获得性能提升。

### 3.2 模型 & 技术｜[\[NVIDIA\] Keep TMA store sources live through the completion wait（\[NVIDIA\] 使 TMA store 源保持存活直到完成等待）](https://github.com/triton-lang/triton/pull/12865)

北京时间：2026-11-21 02:06:08｜来源类型：官方 GitHub PR｜事件状态：已合并。

**事实**：给 async_tma_store_wait 传递源 descriptor，让共享内存分配的生命周期延长到传输完成；Gluon 的 store_wait 可用 deps=[src] 表达该依赖。

**重要性**：避免 TMA store 尚未读完时，布局转换的 scratch 复用同一片内存而污染写回数据。

**风险与限制**：修复的是异步源生命周期；调用方仍须正确表达等待及依赖，不能泛化为全部并发问题已解决。

### 3.3 模型 & 技术｜[\[NVIDIA\] Use small-allocation TMA mode for ragged descriptors（\[NVIDIA\] 对不规则 descriptor 使用小分配 TMA 模式）](https://github.com/triton-lang/triton/pull/12700)

北京时间：2026-11-20 08:18:30｜来源类型：官方 GitHub PR｜事件状态：已合并。

**事实**：针对 Blackwell ragged TMA descriptor 选到 LARGE 模式而使受保护的 64 KiB backing buffer 访问出错的问题，改为 SMALL 模式并应用 NVIDIA 已公布的构造 workaround。 作者在 GB200 上报告 32 个 descriptor 测试及 Compute Sanitizer 通过，配对内核性能变化约在 0.22% 内。

**重要性**：修复不规则矩阵/边界行的实际硬件访问问题。

**风险与限制**：完整 checkout 构建和双 rank 模型重跑仍待完成；不是吞吐提升公告。

### 3.4 模型 & 技术｜[\[Intel\] Keep descriptor gathers native and lower them as fast as pointer loads（\[Intel\] 保持 descriptor gather 原生形式，并使其 lowering 与指针加载一样快）](https://github.com/intel/intel-xpu-backend-for-triton/pull/8790)

北京时间：2026-11-20 21:56:31｜来源类型：官方 GitHub PR｜事件状态：已合并。

**事实**：修正行偏移符号扩展和对齐证明，以向量化 predicated loads 降低 descriptor gather；保留合并布局，只对实测合格的 f16/bf16 dot-A 与部分 fp8 转换 tile 启用原生路径。 作者的 B580 vLLM fp8 KV gather 测试中，48 种形状有 23 种耗时为原来的 0.93–0.98 倍；其余约在 2% 内，bf16 KV gather 仍走 pointer rewrite。

**重要性**：改善 paged-KV gather 的正确性与布局决策，保持性能受益的适用边界。

**风险与限制**：部分快速路径仍关闭；不是所有 descriptor 或 GPU 都更快，可通过环境开关恢复全面 pointer rewrite。

### 3.5 模型 & 技术｜[\[StageLargeFMADotsViaSLM\] Choose the K tile from a register budget（\[StageLargeFMADotsViaSLM\] 根据寄存器预算选择 K tile）](https://github.com/intel/intel-xpu-backend-for-triton/pull/8772)

北京时间：2026-11-20 13:48:03｜来源类型：官方 GitHub PR｜事件状态：已合并。

**事实**：按每线程 512 字节预算同时计入 A/B tile 切片及解码后的类型，选择最小到 8 的二次幂 K tile，缓解 FMA lowering 的寄存器溢出。 作者在 BMG 的 149 个 GEMM 内核、best-of-2 测量中报告 B50 总体时间比为 0.294、B570 为 0.392–0.418；一个 B50 样本退化到 1.12 倍。

**重要性**：以实际寄存器活跃数据量调节分块，为 Intel GPU 的非矩阵指令 GEMM 提供具体优化。

**风险与限制**：测量限定所列 BMG 内核与配置；单个回退案例说明该经验预算不是通用最优值。

## 四、RISC-V 核心新闻

### 4.1 模型 & 技术｜[\[PATCH v4 00/13\] riscv: add initial Zhihe A210 support using SCMI clocks and resets（\[PATCH v4 00/13\] riscv：使用 SCMI 时钟和复位添加初始 Zhihe A210 支持）](https://lists.infradead.org/pipermail/linux-riscv/2026-November/104545.html)

北京时间：2026-11-21 03:15:46｜来源类型：官方邮件列表｜事件状态：v4 补丁系列已提出。

**事实**：为四个 C908V、四个 C920V2 加 E902 管理核的 A210 和 Melon Pi V1/V2 提交 Linux 初始支持，Linux 通过 E902 上的 Zephyr SCMI 服务管理时钟/复位。 v4 明确厂商同意采用 SCMI、撤掉 RFC 前缀，并把已在 linux-next 的前置改动拆出；作者报告基于 7.4-rc3 在 Melon Pi V2 上八核上线并启动串口 shell。

**重要性**：厂商固件协议的承诺和启动链说明，让 A210 上游适配具备更清楚的集成路线。

**风险与限制**：尚非 Linux 正式合并或发行；原生时钟/复位驱动另行提交，不能把旧系列的硬件能力重写成新芯片发布。

### 4.2 模型 & 技术｜[Clarify that instructions are always a multiple of IALIGN bits long（澄清指令长度始终是 IALIGN 位数的整数倍）](https://github.com/riscv/riscv-isa-manual/pull/3549)

北京时间：2026-11-20 07:50:11｜来源类型：官方 GitHub PR｜事件状态：已合并。

**事实**：规范正文补充所有指令长度均为 IALIGN 的整数倍这一约束，避免顺序执行后的指令地址因指令长度本身失去对齐。

**重要性**：把原先隐含的编码/取指条件写成明确规范，对模拟器、解码器和异常验证有直接意义。

**风险与限制**：属于已合并的规范澄清，不是引入新指令或提升硬件性能。

### 4.3 工具 & 产品｜[GUITION ESP32-S31 development kit features 7-inch capacitive touch display, Ethernet port（GUITION ESP32-S31 开发套件配备 7 英寸电容触摸显示屏、以太网端口）](https://www.cnx-software.com/2026/11/20/guition-esp32-s31-development-kit-features-7-inch-capacitive-touch-display-ethernet-port/)

北京时间：2026-11-20 18:44:16｜来源类型：原始专业媒体报道｜事件状态：已发布。

**事实**：CNX 的原始产品报道介绍 JC1060F570C：ESP32-S31 RISC-V 高/低功耗双核、7 英寸 1024×600 电容屏、16 MB PSRAM/flash，以及实际板载 10/100 Mbps 以太网。

**重要性**：为网络 HMI 与轻量端侧应用提供集成触屏、音频和网络的 RISC-V 板卡选择。

**风险与限制**：SoC 支持能力不等于板卡引出能力；该板仅百兆网口，报道未提供独立 AI 性能测量。

### 4.4 模型 & 技术｜[feat(trigger): support itrigger, etrigger, icount（feat(trigger)：支持 itrigger、etrigger、icount）](https://github.com/OpenXiangShan/XiangShan/pull/6942)

北京时间：2026-11-20 15:02:20｜来源类型：官方 GitHub PR｜事件状态：已合并。

**事实**：kunminghu-v3 合入中断、异常与指令计数触发器；icount 支持完整 14 位 COUNT、PENDING、特权级及上下文匹配，并通过 Decode/CSR 握手串行化需跟踪的架构指令。

**重要性**：扩展香山的调试触发与软件可观测性，属于产品分支真实新增能力。

**风险与限制**：三个实现提交作者日期为 11 月 6/10 日，合并为 20 日；这里报道产品分支合入，不能视作当天首次提出设计，也不证明流片硬件已具备。

### 4.5 模型 & 技术｜[cpu: Verify serialized AMO results in multi-hart difftest（cpu：在多 hart 差分测试中验证串行化 AMO 结果）](https://github.com/OpenXiangShan/GEM5/pull/1357)

北京时间：2026-11-20 18:56:56｜来源类型：官方 GitHub PR｜事件状态：已合并。

**事实**：在串行 RMW 前后记录 golden/实际内存字节，提交时令该 hart 的私有 NEMU 执行一次 AMO，同时比较目的寄存器与写后内存，覆盖 rd=x0、失败 CAS 和 AMOCAS.Q。 关联 scalar-load checker 改为按 cache 读取时采样的 golden 字节与本 hart 转发历史核验，去掉可能掩盖错误的值修补。

**重要性**：防止 DUT 的同一错误响应同时充当“参考值”，增强多核内存一致性与原子操作验证可信度。

**风险与限制**：这是 Classic 多上下文差分验证增强；并不证明所有 RVWMO 或多核执行路径已无缺陷。

关联证据：[原始变更](https://github.com/OpenXiangShan/GEM5/pull/1356)。

### 4.6 模型 & 技术｜[Sargantana: move to core_tile v4.0 (sargantana4-rv64gcbh)（Sargantana：迁移到 core_tile v4.0（sargantana4-rv64gcbh））](https://github.com/riscv/riscv-arch-test/pull/3248)

北京时间：2026-11-21 04:58:36｜来源类型：官方 GitHub PR｜事件状态：已合并。

**事实**：验证配置迁移到明确提交的 core_tile v4.0 RTL，扩展压缩、虚拟化、分页和 pointer masking 等声明，并调整 hypervisor 测试的无提交 watchdog。 移除 v4.0 已修复的 22 项排除，同时记录新的分页/异常类问题及仍需排除的测试；V/Zve 仍未声明。

**重要性**：让架构测试面向真实新 RTL 及其能力运行，属于验证覆盖变化而非普通测试维护。

**风险与限制**：新问题清单并非均已提交上游 issue；不得把剩余排除或有限 LMUL 能力写成完整向量验证支持。

### 4.7 模型 & 技术｜[feat(humanoid): use remote operator feedback for G1 body TF（feat(humanoid)：使用远程 operator 反馈生成 G1 机身 TF）](https://github.com/spacemit-robotics/ros2_humanoid/commit/ed745686a1ac28d1ecafefdcb3de5da88f782090)

北京时间：2026-11-20 17:43:23｜来源类型：官方主干 commit｜事件状态：已进入主干。

**事实**：G1 body TF 从本地共享内存读取改为只读 WebSocket operator 反馈，可在导航板运行并使用本地 ROS 时间戳；增加重连、样本去重和陈旧反馈处理。

**重要性**：把机器人状态发布与运控所在进程/板卡解耦，提供更明确的跨板部署链路。

**风险与限制**：只是状态反馈和 TF 集成，不是控制算法或稳定性提升的证明；部署仍须处理时钟及反馈中断。

### 4.8 工具 & 产品｜[workflows: add PyPI sdist source mode（workflows：添加 PyPI sdist 源模式）](https://github.com/riseproject-dev/python-wheels/pull/5002)

北京时间：2026-11-21 03:53:38｜来源类型：官方 GitHub PR｜事件状态：已合并。

**事实**：RISC-V wheel 构建工作流新增可选 pypi-sdist 源模式，按 PyPI JSON API 解析准确发行版并核验 SHA-256，再安全解包到现有准备/打补丁目录；checkout 仍为默认。

**重要性**：为 RISE 包分发配方增加可追溯的源码获取路线，降低逐包定制构建的负担。

**风险与限制**：这只证明构建配方合入，不证明某个新 wheel 已上传，更不能证明 Triton-RISC-V 分发合作原信或已上架版本。

### 4.9 工具 & 产品｜[Support reporting download progress in porcelain mode（支持在 porcelain 模式中报告下载进度）](https://github.com/ruyisdk/ruyi/pull/521)

北京时间：2026-11-20 22:58:28｜来源类型：官方 GitHub PR｜事件状态：已合并。

**事实**：为 requests fetcher 增加结构化下载请求与进度事件，用 req_id 关联，按 1% 或 1 MiB 步进报告，覆盖断点续传、完成和失败；porcelain 模式抑制人类进度条与 stdout 故障帮助。

**重要性**：使 IDE 和自动化部署工具能可靠消费 Ruyi 下载生命周期，而不解析终端文本。

**风险与限制**：仅覆盖该下载后端，不能把结构化事件支持泛化到所有传输方式；未宣称已有 IDE 版本集成。

## 五、AI 业界重磅

### 5.1 模型 & 技术｜[Welcome Inkling 2 Mini, Cutting-edge Performance for your Local Setup（欢迎 Inkling 2 Mini，为你的本地配置带来前沿性能）](https://huggingface.co/blog/inkling2mini)

北京时间：2026-11-20 08:00:00｜来源类型：官方博客｜事件状态：已发布。

**事实**：发布多模态循环 Transformer MoE：约 26B 物理参数、每 token 约 3B 激活参数、1M 上下文；24 个物理层中的中间 12 层循环三次，形成 48 次逻辑层访问。 提供 NVFP4/MXFP8 混合精度权重及 Transformers、llama.cpp/Llama App、MLX 和 SGLang 的推理集成说明。

**重要性**：以权重复用探索中等规模本地模型的容量与内存折衷，也给新模型架构导入提出适配需求。

**风险与限制**：循环不消除计算及各次访问的 KV 状态；可变循环研究版本尚无充分质量评估，SGLang 文中配置要求 SM103。

### 5.2 模型 & 技术｜[Introducing harness adapters: adapt your favorite model to your harness to improve performance and cost（介绍 harness adapters：让你偏好的模型适配你的 harness，以改善性能和成本）](https://huggingface.co/blog/harness-adapters-at-scale)

北京时间：2026-11-20 08:00:00｜来源类型：官方博客｜事件状态：已发布。

**事实**：公开在 pi、Hermes 和 OpenCode 内对 Qwen3.8-27B 进行 RL 的 LoRA 配方、适配器及评测轨迹；三个 specialist 在各自 harness 的编码基准均优于基础模型。 Hermes 的 SWE-bench Pro 最难 51 题在关闭 thinking 时从 29.4% 提高到 43.1%；共同解出任务的工具调用减少，但跨业务 AutomationBench 的迁移收益大多未超出噪声。

**重要性**：把智能体运行框架纳入训练环境，显示推理成本优化不仅取决于基础模型或算子速度。

**风险与限制**：不同 harness 的上下文保留策略不同，不能横向混比；merged adapter 的权重选取用过 Terminal-Bench，相关结果略偏乐观。

### 5.3 模型 & 技术｜[Combatting adversarial distillation（对抗恶意蒸馏）](https://openai.com/index/combatting-adversarial-distillation)

北京时间：2026-11-20 18:00:00｜来源类型：官方博客｜事件状态：已发布。

**事实**：OpenAI 披露请求级推理保护、跨会话/模型复用限制及活动级检测的组合防御；在其六个攻击模型汇总的自适应黑盒测试中，GPT-6.2 Astra 对此前/当前请求受保护推理的提取成功率分别为 0.2%/0.9%。

**重要性**：给已部署前沿模型提供具体的推理泄露评测与分层保护案例。

**风险与限制**：数据来自厂商自测和指定攻击设置，不代表对全部蒸馏或推理泄露免疫；所述归因是 OpenAI 的调查结论。

### 5.4 模型 & 技术｜[Can AI help scientists identify promising drug candidates?（AI 能帮助科学家识别有前景的候选药物吗？）](https://openai.com/index/introducing-medchembench)

北京时间：2026-11-20 08:00:00｜来源类型：官方博客｜事件状态：已发布。

**事实**：公开 MedChemBench，以 117 篇论文/专利中的 761 个分子构建 99 个药物化学问题，分别测结构重建、科学解释、化合物选择与属性预测。 在 xhigh 下，GPT-6.2 Astra 获总可用分数的 84.5%，GPT-6 Astra 为 83.3%；结构重建相对较强，化合物选择和数值预测仍明显困难。

**重要性**：把科学 AI 的评价拆到实际推理环节，避免只用综合分数替代实验价值判断。

**风险与限制**：原文不能排除模型见过历史材料；单次、无工具/反馈评测不证明能发现新药或改善实验室决策。

### 5.5 工具 & 产品｜[Get your year-end tasks done with new Connected Apps.（通过新的 Connected Apps 完成年末任务。）](https://blog.google/products-and-platforms/products/search/connected-apps-holiday-errands/)

北京时间：2026-11-21 02:00:00｜来源类型：官方博客｜事件状态：已发布。

**事实**：面向美国逐步推出 Albertsons、Safeway、Shaw’s 和 Target 的 Connected Apps，接入 Search/Gemini 并基于 Universal Commerce Protocol；Experian 从 Gemini 扩至 Search，H&R Block 提供税额/退款估算。

**重要性**：智能体应用从信息回答延伸到购物与服务流程，连接协议和账户权限成为交付路径的一部分。

**风险与限制**：地区和 rollout 有限；Angi、Thumbtack 仍是未来几周计划，不可写成已经上线。

### 5.6 工具 & 产品｜[How Watch Duty is using Gemini to get wildfire alerts out faster（Watch Duty 如何使用 Gemini 更快地发出野火警报）](https://blog.google/company-news/outreach-and-initiatives/google-org/watch-duty-gemini-tool/)

北京时间：2026-11-21 01:00:00｜来源类型：官方博客｜事件状态：已发布。

**事实**：Google 与 Watch Duty 公布无线电转写部署结果：退化 VHF 语音词错率从 38% 降到 22.34%，约 400 路活跃 feed 每日处理超过 200 小时语音，向记者标记新 dispatch 用时不到 30 秒。

**重要性**：展示模型在噪声、行业术语和人工核验链路中的落地价值，可作为部署质量评估案例。

**风险与限制**：这是本次公开的合作结果，现场案例发生于 7 月；超过 10,000 路只是设计/测试规模，警报仍需人员核验。

## 六、总结与趋势观察

- **部署产物的边界变得更明确。** PyTorch 的下游 wheel 策略允许定制外部运行库，ExecuTorch 的 PTE 工具则显式重映射设备编号；两者分别处理构建环境与目标机器差异，兼容性仍需在实际部署环境验证。[wheel 策略](https://github.com/pytorch/pytorch/commit/d3633dc1fdbdf6d8e39887fe807a9e2831589c6b)、[PTE 工具](https://github.com/pytorch/executorch/pull/25315)
- **编译正确性越来越依赖精确语义和资源条件。** torch-mlir 修正 Python 余数语义，IREE 限制静态 dispatch 绑定字节；一个约束运算含义，一个约束可执行资源，两类条件都不能从“编译成功”自动推导。[torch-mlir](https://github.com/llvm/torch-mlir/pull/4891)、[IREE](https://github.com/iree-org/iree/pull/25299)
- **RISC-V 验证链在同时校准参考值与真实 RTL。** GEM5 加强多 hart 原子操作及 load 检查，架构测试迁移到 Sargantana v4.0 并公开保留的排除项；参考模型独立性与配置准确性同样影响验证结论。[GEM5](https://github.com/OpenXiangShan/GEM5/pull/1357)、[架构测试](https://github.com/riscv/riscv-arch-test/pull/3248)
- **智能体效率出现两条互补路线。** Inkling 2 Mini 复用循环层权重以控制模型存储，harness adapters 通过运行框架内训练减少工具交互；前者仍承担循环计算和 KV 状态，后者的跨任务迁移也未证明普遍有效。[Inkling 2 Mini](https://huggingface.co/blog/inkling2mini)、[harness adapters](https://huggingface.co/blog/harness-adapters-at-scale)

## 附录：信源说明

本期来源包括 PyTorch、ExecuTorch、Torch-TensorRT、LLVM/MLIR、torch-mlir、IREE、Triton 及 Intel 后端的项目原始记录，RISC-V 规范、Linux 邮件列表、RuyiSDK、RISE、香山与进迭机器人软件记录，以及 Hugging Face、OpenAI、Google 官方文章和 CNX 原始产品报道。

项目合并代表相应主干或产品分支的状态，不能替代正式发行、量产或硬件交付；RFC 和讨论保留提案性质。性能数字和部署规模限于原文所列设备、负载、测量或验证条件，厂商及作者自述不等于独立复测。
