# Codex 技术情报每日动态（2026-09-08）

调研窗口：北京时间（2026-09-07 05:27:50，2026-09-08 06:00:24]。

覆盖方向：PyTorch、LLVM/MLIR、Triton/TileLang、RISC-V 与 AI 模型、基础设施及应用。

信息口径：以项目官方公告、技术论坛和上游代码记录为依据；区分已合并实现、设计提案与作者报告的实验结果。

## 今日要闻

- [vLLM 发布 Tenstorrent 插件架构说明](https://vllm.ai/blog/2026-09-07-vllm-tt-plugin)：通过独立插件实现分阶段调度、Galaxy 单进程 lane 数据并行和设备端采样，尚不支持多主机、LoRA 与推测解码。
- [ClangIR 默认构建提案](https://discourse.llvm.org/t/91730/36)继续讨论：作者将缩减 MLIR 构建依赖、stage-2 进展及构建成本列为切换默认值前的条件，未提议默认启用 ClangIR 代码生成。
- [Google 与国泰航空扩展 AI 尾迹规避试验](https://blog.google/innovation-and-ai/models-and-research/google-research/contrail-avoidance-ultra-long-haul-flights/)：超过 80 架次采用规避路线，估算尾迹增温影响减少约 40%。
- [AArch64 三值选择累加到 SDOT 的 RFC](https://discourse.llvm.org/t/91742/1)提出面向低比特模型的指令改写，作者报告特定设备上 Bonsai-1.7B 解码由约 18 提升至 27 tok/s。

## 今日索引

- **PyTorch 生态核心动态**：VLM CPU 卸载、DeepSeek V4 检查点和 Arm 量化/舍入语义。
- **LLVM/MLIR 最新进展**：Buddy Whisper 服务、TOSA 降精度、StableHLO 与 ClangIR/Arith RFC。
- **Triton & TileLang 技术动态**：Gluon 缓存策略、RDNA 与 Intel 降级，以及 TileLang 缓存和原子操作。
- **RISC-V 核心新闻**：香山 VFMA 与访存重放、Andes 访存能力及 LLVM RISC-V 编译终止性。
- **AI 业界重磅**：Tenstorrent 插件架构、AI 尾迹规避，以及低精度注意力、昇腾推理和 MoE 后端。

## 一、PyTorch 生态核心动态

### 1.1 模型 & 技术｜[让 VLM 视觉编码器采用解码器的 CPU 卸载策略](https://github.com/pytorch/torchtitan/pull/4512)

北京时间：2026-09-08 06:00:01｜来源类型：官方 GitHub PR｜事件状态：已合并

**事实**：TorchTitan 将 CPUOffloadPolicy 传入视觉编码器，并让 Qwen3.5、Kimi K2.7、Kimi K3 和 Muse Glimmer 四条 VLM 路径使用相同的卸载开关；此前开启 CPU offload 时，编码器参数留在 CPU、梯度落在 CUDA，首次反向传播即可失败。

**重要性**：修复多模态训练的显存卸载路径，使视觉编码器和解码器的分片参数策略一致。

**风险与限制**：作者报告两张 H20 上六项相关测试通过；这不能替代大型模型的完整训练验证。

### 1.2 模型 & 技术｜[修复带有可选 MTP 的 DeepSeek V4 检查点转换](https://github.com/pytorch/torchtitan/pull/4498)

北京时间：2026-09-07 06:08:58｜来源类型：官方 GitHub PR｜事件状态：已合并

**事实**：TorchTitan 修复默认 DeepSeek V4 恢复时的 len(None)、MTP attention sink 导出缺键以及官方 `mtp.<index>` 前缀映射，并保留 MTP 专家权重的 DTensor placement。

**重要性**：打通模型权重导入、恢复与反向导出边界，对依赖 PyTorch 模型适配的部署链路有直接价值。

**风险与限制**：验证覆盖 CPU/Gloo、0/1/2 个 MTP 层和指定官方张量；完整量化权重加载、多 GPU 训练不在已执行范围。

### 1.3 模型 & 技术｜[Arm 后端：在 rescale pass 中保留 Q/DQ](https://github.com/pytorch/executorch/pull/22559)

北京时间：2026-09-08 05:24:13｜来源类型：官方 GitHub PR｜事件状态：已合并

**事实**：ExecuTorch 的 Arm rescale pass 遇到仍连接直接 DQ 输入或 Q 输出的算子时，不再插入 INT32 RESCALE，避免把部分量化 add、mul 的浮点输入误判成整数路径。

**重要性**：保持混合量化模型的图边界和数值语义，减少模型降级到 TOSA 时的错误转换。

**风险与限制**：适用范围是保留 Q/DQ 的部分量化节点；已完全折叠的整数路径沿用原有处理。

### 1.4 模型 & 技术｜[Arm 后端：修复 round() 分解，使其采用中点舍入到偶数](https://github.com/pytorch/executorch/pull/21065)

北京时间：2026-09-07 12:06:07｜来源类型：官方 GitHub PR｜事件状态：已合并

**事实**：ExecuTorch 修正 aten.round 的 Arm 分解：此前采用中点远离零，现改为与 torch.round 一致的中点舍入到偶数，例如 0.5→0、2.5→2、-2.5→-2。

**重要性**：消除委派计算与 PyTorch 参考语义之间可复现的静默数值差异。

**风险与限制**：作者验证了 TOSA 与 U55/U65/U85 相关用例；影响集中在 round 的分解路径。


## 二、LLVM/MLIR 最新进展

### 2.1 模型 & 技术｜[\[Models\] Buddy_server 支持 whisper 模型](https://github.com/buddy-compiler/buddy-mlir/pull/920)

北京时间：2026-09-07 13:48:03｜来源类型：官方 GitHub PR｜事件状态：已合并

**事实**：Buddy 为 Whisper 增加独立的 buddy-server 音频转写后端、whisper_transcription.so 插件及 .rax 转写插件载荷，现有 buddy-cli runner 继续保留。模型构建链路通过 PyTorch/Transformers 导入生成 MLIR 和参数。

**重要性**：将 Whisper 从命令行运行扩展到统一服务端插件链路，是 Buddy 模型导入、编译打包和服务部署的一次直接衔接。

**风险与限制**：提交增加了运行时与 HTTP 集成测试，但未给出真实语音数据集的识别精度或吞吐基准。

### 2.2 模型 & 技术｜[\[mlir\]\[tosa\] 添加 F32 到 F16 类型收窄 pass](https://github.com/llvm/llvm-project/pull/218900)

北京时间：2026-09-07 21:16:27｜来源类型：官方 GitHub PR｜事件状态：已合并

**事实**：MLIR 新增 tosa-narrow-f32-to-f16，可转换 F32 张量、标量属性、稠密常量和资源常量；提供保守/激进重写、可选函数边界转换及保留 F32 累加器的选项。

**重要性**：为 TOSA 模型降级提供统一的精度收窄环节，可用于控制部署数据类型和存储开销。

**风险与限制**：这是会损失精度的转换；函数边界和累加器是否降精度需要显式区分，不能推定模型精度不变。

### 2.3 模型 & 技术｜[\[RFC\] 向 StableHLO 规范添加 Exp2Op 和 Log2Op。](https://github.com/openxla/stablehlo/pull/3007)

北京时间：2026-09-08 00:09:58｜来源类型：官方 RFC（GitHub）｜事件状态：提案

**事实**：StableHLO 提议增加一等 exp2/log2 对应的 Exp2Op、Log2Op 一元逐元素运算，并支持可选 result_accuracy 属性；动机是避免借 ln(2) 缩放带来的精度损失，让后端直接选择以 2 为底的硬件指令。

**重要性**：补齐模型表示到后端指令的数值语义，对 StableHLO 导入和编译后端具有相关性。

**风险与限制**：当前是规范 RFC，尚未合并；不能据此宣称各后端已支持。

### 2.4 模型 & 技术｜[\[RFC\] AArch64：将三值选择累加降级为 SDOT](https://discourse.llvm.org/t/rfc-aarch64-ternary-select-accumulate-to-sdot-lowering/91742/1)

北京时间：2026-09-07 16:28:34｜来源类型：官方技术论坛｜事件状态：提案

**事实**：该 RFC 提议在 AArch64 IR 层识别三值权重的 select/and 累加模式，改写为查表加 SDOT，并由目标 dot-product 能力约束。作者在 Snapdragon 8 Elite 上报告 2048×2048 微基准约 4.3 倍加速、Bonsai-1.7B 单 token 解码约 18→27 tok/s。

**重要性**：直接针对低比特神经网络的 CPU 指令选择，展示模型计算模式与 ISA 的结合。

**风险与限制**：数据为提案作者的特定设备实验，尚非上游正式功能；pass 位置、成本模型和数值等价性仍需评审。

### 2.5 深度洞见｜[RFC：默认启用 ClangIR 构建-](https://discourse.llvm.org/t/rfc-enable-clangir-build-by-default/91730/36)

北京时间：2026-09-07 08:27:29｜来源类型：官方技术论坛｜事件状态：讨论新进展

**事实**：提案拟默认构建 ClangIR，并将 MLIR 依赖和 CIR 测试纳入 Clang 构建，仍可用 CLANG_ENABLE_CIR=OFF 关闭，且不默认启用 -fclangir 代码生成。讨论新进展：作者承认部分测量显示构建耗时约翻倍，提出在切换默认值前缩减 MLIR 方言、工具等构建目标，并等待 stage-2 和社区可接受的构建成本条件。

**重要性**：这会改变使用 LLVM/MLIR 的团队构建依赖和交付成本，主线是 ClangIR 的默认可用性，而非默认替换现有代码生成。

**风险与限制**：仍处 RFC 阶段；模块、调试信息、sanitizer 和非 x86 支持等限制仍是原提案列出的缺口。

### 2.6 深度洞见｜[RFC：向 Arith 添加 IEEE 754-2019 minimumNumber/maximumNumber 运算](https://discourse.llvm.org/t/rfc-add-ieee-754-2019-minimumnumber-maximumnumber-operations-to-arith/91723/6)

北京时间：2026-09-07 15:05:19｜来源类型：官方技术论坛｜事件状态：讨论新进展

**事实**：提案为 Arith 增加 minimumnumf/maximumnumf 和匹配的 vector.reduction 类型，以表达 IEEE 754-2019 在 NaN 与有符号零上的语义。讨论新进展：chelini 提交实现 PR #221658；后续回复围绕独立算子与单算子属性两种表示路线继续讨论。

**重要性**：模型导入和归约降级需要区分传播 NaN、忽略 NaN 等语义，不能把不同 min/max 运算直接互换。

**风险与限制**：尚未形成统一设计结论；实现 PR 的出现不等于合并或后端普遍支持。

关联讨论：[回复 7](https://discourse.llvm.org/t/91723/7)、[回复 8](https://discourse.llvm.org/t/91723/8)、[回复 9](https://discourse.llvm.org/t/91723/9)。

### 2.7 深度洞见｜[RFC：\[编译时间\] 限制超大循环嵌套中的运行时循环展开](https://discourse.llvm.org/t/rfc-compile-time-cap-run-time-loop-unrolling-in-very-large-loop-nests/91747/1)

北京时间：2026-09-07 19:42:47｜来源类型：官方技术论坛｜事件状态：提案

**事实**：RFC 提议增加 unroll-runtime-max-nest-loops，默认阈值 128：顶层循环巢的循环总数超限时，抑制需要生成余数循环的运行时展开，避免反复使整个循环巢的 SCEV 缓存失效而产生近 O(N²) 编译成本。

**重要性**：作者报告 SPEC CPU 2026 的 765.roms 编译时间下降 34.7%；它针对生成代码中的宽循环巢，而不只是嵌套深度。

**风险与限制**：完整/部分展开和显式 pragma 不受该默认限制；仍为提案，测量收益只适用于触发阈值的工作负载。

### 2.8 工具 & 产品｜[MLIR Suite：为 Zed 编辑器提供 MLIR、TableGen 和 PDLL 支持](https://discourse.llvm.org/t/mlir-suite-mlir-tablegen-and-pdll-support-for-the-zed-editor/91302/2)

北京时间：2026-09-07 11:49:13｜来源类型：官方技术论坛｜事件状态：讨论新进展

**事实**：MLIR Suite 为 Zed 提供 MLIR、TableGen、PDLL 语法高亮、符号轮廓和三个上游 LLVM 语言服务器的客户端。讨论新进展：作者确认项目改名为 zed-mlir，并已以 MLIR 名称进入 Zed 扩展注册表，可从编辑器扩展页安装。

**重要性**：为经常阅读 MLIR、TableGen 和 PDLL 的编译器开发提供可直接安装的工具入口。

**风险与限制**：这是社区维护的编辑器扩展；扩展上架不代表 LLVM 官方对全部自定义方言作兼容保证。


## 三、Triton & TileLang 技术动态

### 3.1 模型 & 技术｜[\[Gluon\] 暴露丰富的缓存策略](https://github.com/triton-lang/triton/pull/11482)

北京时间：2026-09-08 04:48:29｜来源类型：官方 GitHub PR｜事件状态：已合并

**事实**：Triton 将 cache_modifier/eviction_policy 组合重构为 TT_CachePolicyAttr，并增加 NVIDIA 专用 TTNG_CachePolicyAttr，支持独立 L1/L2 策略、fractional 策略与可配置 L2 预取大小；Gluon load/store 新增 cache_policy 参数。

**重要性**：把缓存选择从有限字符串扩展为结构化策略，便于算子作者细调内存访问行为。

**风险与限制**：NVIDIA 专属策略不等于跨后端统一能力；提交未给出通用性能收益。

### 3.2 模型 & 技术｜[\[AMD\] 为 RDNA 使用 AMD dot_scaled 分解路径](https://github.com/triton-lang/triton/pull/11629)

北京时间：2026-09-08 01:58:10｜来源类型：官方 GitHub PR｜事件状态：已合并

**事实**：RDNA 的 dot_scaled 改用 AMD 专用分解路径，在 f32→bf16 下转换使用 round-to-zero，替代通用路径中成本更高的 round-to-nearest-even 处理。

**重要性**：让 RDNA 复用更贴近 AMD 目标的低精度矩阵乘实现。

**风险与限制**：舍入方式是该实现的关键差异；原文未提供可泛化的吞吐倍率，不能按指令变化推定端到端提升。

### 3.3 模型 & 技术｜[添加 triton intel gpu gather 算子降级。](https://github.com/intel/intel-xpu-backend-for-triton/pull/7548)

北京时间：2026-09-07 10:51:00｜来源类型：官方 GitHub PR｜事件状态：已合并

**事实**：Intel Triton 后端为 ttig.descriptor_gather 增加 LLVM 降级：满足布局和 tile 约束时走 subgroup gather-load 快路径，否则生成带谓词的 LLVM load。

**重要性**：补齐描述符式 gather 到 Intel GPU 代码生成的链路，关系到不规则访存算子的可移植性。

**风险与限制**：快路径受布局约束；合并记录不包含跨型号端到端性能结果。

### 3.4 模型 & 技术｜[\[CUDA\]\[Cache\] 发布不可变的二进制缓存目录](https://github.com/tile-ai/tilelang/pull/3177)

北京时间：2026-09-08 02:46:47｜来源类型：官方 GitHub PR｜事件状态：已合并

**事实**：TileLang 将 CUDA 二进制与包含长度、格式、SHA-256 的元数据作为不可变目录原子发布，竞争写入保留首个结果；损坏或空读取按缓存未命中处理，不再删除其他进程可能正在读取的条目。

**重要性**：解决共享文件系统中二进制与校验信息分别发布造成的竞态，改善多进程编译缓存的部署可靠性。

**风险与限制**：旧平铺缓存保留但不再读取；损坏目录需离线清理。作者使用故障注入，未执行真实 3FS 集成测试。

### 3.5 模型 & 技术｜[\[BugFix\]\[Transform\] 在 SM90 上保持共享 FP32 原子操作为标量](https://github.com/tile-ai/tilelang/pull/3129)

北京时间：2026-09-07 11:37:14｜来源类型：官方 GitHub PR｜事件状态：已合并

**事实**：TileLang 的原子向量宽度选择加入目标内存作用域：共享内存 FP32 原子操作保持标量，SM90+ 全局 FP32 保留 x4，FP16/BF16 保留 x2，避免 H100 上生成不受支持的共享 float4 atomicAdd。

**重要性**：修复可导致 CUDA_ERROR_INVALID_VALUE 的内核启动失败，使原子访存变换符合 CUDA 内存空间约束。

**风险与限制**：这是特定数据类型和内存作用域的正确性修复，不代表所有原子操作均可向量化。


## 四、RISC-V 核心新闻

### 4.1 模型 & 技术｜[添加 vfma 支持并重构向量流水线元数据](https://github.com/OpenXiangShan/XiangShan/pull/6503)

北京时间：2026-09-07 16:19:30｜来源类型：官方 GitHub PR｜事件状态：已合并

**事实**：香山向量后端接入 VFMA 功能单元，新增 VFMacWrapper，并更新核心配置、数据结构和流水线，使其携带向量浮点乘加所需信息。

**重要性**：推进 RISC-V 向量浮点执行链路，为需要向量乘加的数值与模型算子提供硬件实现基础。

**风险与限制**：该事件是 RTL 后端变更，不等于芯片流片或量产，也没有应用级性能数据。

### 4.2 模型 & 技术｜[\[RISCV\] Andes：为 45 系列建模快速非对齐访问](https://github.com/llvm/llvm-project/pull/221166)

北京时间：2026-09-07 13:04:28｜来源类型：官方 GitHub PR｜事件状态：已合并

**事实**：LLVM 为 andes-n45、nx45、a45、ax45、ax45mpv 配置非对齐标量访存能力，使其由 -mcpu 选择。原实现经 -mtune 引入该能力，可能在实际不支持的 CPU 上生成会崩溃的非对齐访问。

**重要性**：明确 RISC-V CPU 能力与性能调优选项的边界，避免单纯调优改变目标可执行性。

**风险与限制**：该能力限定于列出的 Andes 45 系列；不能推广到全部 RISC-V CPU。

### 4.3 模型 & 技术｜[fix(LoadQueueReplay)：处理非对齐头部重放的延迟唤醒](https://github.com/OpenXiangShan/XiangShan/pull/6480)

北京时间：2026-09-07 16:48:53｜来源类型：官方 GitHub PR｜事件状态：已合并

**事实**：香山保留上一周期 TLB hint 和额外一周期 DCache 唤醒记录，并为经 S4 延迟入队的非对齐 load 头部重放打标，使其能消费先到达的完成响应。

**重要性**：避免唤醒早于重放记录入队时，访存完成后队列仍被阻塞。

**风险与限制**：历史响应只供已标记的延迟记录使用，普通 load 不能匹配陈旧 TLB/MSHR ID；没有系统级吞吐结论。

### 4.4 模型 & 技术｜[\[RISCV\] 修复 SETCC 与 SIGN_EXTEND_INREG 导致的无限 DAGCombine 循环](https://github.com/llvm/llvm-project/pull/221593)

北京时间：2026-09-08 03:03:43｜来源类型：官方 GitHub PR｜事件状态：已合并

**事实**：LLVM RISC-V 后端阻止特定 SETCC 化简反复生成 SIGN_EXTEND_INREG、又被通用 DAGCombiner 还原成 AND 的循环，处理 C1=0 且 X 的第 31 位已知为零的条件。

**重要性**：消除编译器在该输入形态下无法终止的严重正确性问题。

**风险与限制**：修复针对特定 DAG 模式；不意味着所有编译耗时异常均由此解决。


## 五、AI 业界重磅

### 5.1 模型 & 技术｜[我们在亚太地区开展的新凝结尾迹规避试验](https://blog.google/innovation-and-ai/models-and-research/google-research/contrail-avoidance-ultra-long-haul-flights/)

北京时间：2026-09-07 16:00:00｜来源类型：官方博客｜事件状态：已公告

**事实**：Google 与国泰航空扩展 AI 凝结尾迹规避试验：超过 80 架次航班采用规避路线，Google 基于卫星图像估算其尾迹增温影响降低约 40%，并启动规模更大的第二阶段。

**重要性**：把 AI 预测、卫星图像和气象信息接入实际飞行调度与驾驶舱，是预测模型进入真实运营流程的案例。

**风险与限制**：40% 指试验航班凝结尾迹的估计增温影响，不能解读为航班燃油消耗或全部碳排放下降 40%；第二阶段仍在试验。

### 5.2 模型 & 技术｜[feat(sm120)：为 DeepSeek V4 Flash 添加 NVFP4 稀疏 MLA 支持](https://github.com/flashinfer-ai/flashinfer/pull/4955)

北京时间：2026-09-07 19:35:39｜来源类型：官方 GitHub PR｜事件状态：已合并

**事实**：FlashInfer 为 SM120/SM121 的 DeepSeek V4 Flash 路径增加可选 NVFP4 稀疏 MLA，缓存 ABI 为每 token 384 字节，包含量化 NoPE 与 BF16 RoPE 数据，并分别实现 streaming prefill 和 split-K decode。作者在 RTX PRO 5000 Blackwell 上报告指定预填充形状最高约 1.67 倍算子加速。

**重要性**：减少低精度稀疏注意力缓存与访存开销，并提供独立校准策略。

**风险与限制**：FP8 仍为默认。长输出服务实验整体吞吐提升 3.67%，但稳定解码窗口低于 FP8 约 0.57%；算子收益不能直接当作服务吞吐收益。

### 5.3 模型 & 技术｜[\[NPU\] 添加 NPU arch35 支持并增强 DeepSeek-V4 中的 DSV4 处理](https://github.com/sgl-project/sglang/pull/37373)

北京时间：2026-09-07 21:08:06｜来源类型：官方 GitHub PR｜事件状态：已合并

**事实**：SGLang 接入 Ascend Atlas A5/Arch35 检测，为 DeepSeek-V4 增加稀疏注意力、压缩器/索引器、FP8 KV cache、MXFP8 linear 和 MXFP4 MoE 路径，并覆盖 TP 与 DeepEP 调度。

**重要性**：为 DeepSeek-V4 在昇腾 950 平台的模型执行与量化部署建立专门路径。

**风险与限制**：作者仍标注压缩器 A5 算子的精度差异待修复；单项 AIME26 结果不代表完整精度回归通过。

### 5.4 模型 & 技术｜[\[NPU\] 将 DFlash2 推测解码适配到 Ascend NPU](https://github.com/sgl-project/sglang/pull/35629)

北京时间：2026-09-07 09:08:40｜来源类型：官方 GitHub PR｜事件状态：已合并

**事实**：SGLang 将采用分块扩散草稿模型的 DFlash2 适配至 Ascend NPU，修正目标验证阶段的注意力元数据初始化，并在 NPU selector 验证中采用 argmax，保留 CUDA 原路径。

**重要性**：扩展 NPU 的推测解码选择，消除 DFlash2 检查点在目标验证路径的设备假设障碍。

**风险与限制**：已验证环境依赖 triton-ascend；NPU 的 selector 验证策略与 CUDA 不同，不能推定采样行为完全相同。

### 5.5 模型 & 技术｜[\[EPD\] 添加通过 Mooncake TransferEngine 传输编码器缓存的 ECMooncakeConnector](https://github.com/vllm-project/vllm/pull/41567)

北京时间：2026-09-07 14:21:44｜来源类型：官方 GitHub PR｜事件状态：已合并

**事实**：vLLM 增加 ECMooncakeConnector，利用 Mooncake TransferEngine、HTTP registry 和 ZMQ 协调，让拆分部署的消费者拉取编码器缓存张量，不依赖共享文件系统。

**重要性**：为多模态编码与后续推理拆分部署提供直接缓存传输通路。

**风险与限制**：提交提供双进程 CUDA 集成用例和 EPD 全流程脚本；运行需要额外依赖，未给出通用性能倍率。

### 5.6 模型 & 技术｜[feat(moe)：添加集成 MOK 的 megakernel 后端](https://github.com/NVIDIA/Megatron-LM/pull/6572)

北京时间：2026-09-07 20:14:23｜来源类型：官方 GitHub PR｜事件状态：已合并

**事实**：Megatron-LM 增加可插拔 MoE megakernel 接口，以 Mixture-of-Kittens 作为首个后端，覆盖路由之后的 dispatch、专家计算和 combine，参数及检查点所有权仍由 MCore 管理。

**重要性**：把专家通信与计算接入统一后端，同时保留原生梯度和检查点体系，属于 MoE 训练架构变化。

**风险与限制**：当前要求融合梯度累积，共享专家保持 BF16 且与路由专家中间维度一致，不支持可选共享专家输出 gate；性能实验使用不同资源分配，不能视作严格受控对比。

### 5.7 深度洞见｜[在 Tenstorrent 硬件上提供 LLM 服务：深入了解 vLLM TT 插件](https://vllm.ai/blog/2026-09-07-vllm-tt-plugin)

北京时间：2026-09-07 18:12:10（官网发布记录）｜来源类型：官方博客｜事件状态：已公告

**事实**：Tenstorrent 通过独立于 vLLM 主仓库的平台插件接入服务接口：调度步骤分别只执行 prefill 或 decode，Galaxy 的单次执行模型在一个进程内协调多个独立 KV cache lane，并支持设备端采样及不兼容请求的主机回退。模型实现由 TT-Metal 提供，插件负责架构注册和运行时接入。

**重要性**：展示非 GPU 网格架构如何利用 vLLM 扩展点表达调度、并行与采样差异，为异构推理后端设计提供具体案例。

**风险与限制**：当前不支持多主机服务、LoRA、推测解码和 prompt logprobs；prefix cache 与异步解码重叠按模型能力启用，现有安装流程绑定 vLLM 0.26.0。文中未给出统一性能对比。

### 5.8 行业 & 人事｜[支持亚太地区的 16 个绿色 AI 项目](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/ai-planet-accelerator-apac/)

北京时间：2026-09-07 09:00:00｜来源类型：官方博客｜事件状态：已公告

**事实**：Google DeepMind 的 AI for the Planet 亚太加速器选定 16 个组织，本周在新加坡启动训练营，未来三个月提供 AI 技术栈、专门模型、技术支持和导师资源，覆盖生物多样性、农业与气候。

**重要性**：将遥感、生物声学和环境 AI 工具与地方应用团队连接起来，推进研究工具的场景验证。

**风险与限制**：这是项目支持计划启动，不能当作 16 个项目已实现环境收益或商业规模化的证明。

## 六、总结与趋势观察

- **模型服务的边界继续细化**：Buddy 将 Whisper 接入独立转写插件，vLLM 为编码器缓存加入 Mooncake 传输；两者分别推进模型服务接口与拆分部署的数据通路（[Buddy](https://github.com/buddy-compiler/buddy-mlir/pull/920)、[vLLM](https://github.com/vllm-project/vllm/pull/41567)）。
- **低精度收益与数值语义需要同时处理**：TOSA 新增 F32→F16 转换，FlashInfer 扩展 NVFP4 稀疏 MLA；与此同时，ExecuTorch 修复 Q/DQ 边界，Arith RFC 补齐 min/max 语义，说明表示、精度和硬件映射必须一起核对（[TOSA](https://github.com/llvm/llvm-project/pull/218900)、[FlashInfer](https://github.com/flashinfer-ai/flashinfer/pull/4955)、[ExecuTorch](https://github.com/pytorch/executorch/pull/22559)、[Arith](https://discourse.llvm.org/t/91723/6)）。
- **编译成本成为设计约束**：ClangIR 默认构建讨论聚焦依赖和构建时间，超大循环巢 RFC 则直接限制会触发反复 SCEV 失效的展开，两项工作分别处理工具链构建成本和用户程序编译成本（[ClangIR](https://discourse.llvm.org/t/91730/36)、[LoopUnroll](https://discourse.llvm.org/t/91747/1)）。

## 附录：信源说明

本期来源包括 Google、vLLM 官方博客、LLVM Discourse、GitHub 项目官方 PR/RFC。主要项目为 TorchTitan、ExecuTorch、Buddy Compiler、LLVM/MLIR、StableHLO、Triton、TileLang、香山、FlashInfer、SGLang、vLLM 和 Megatron-LM。

PR 合并表示上游代码状态变化，不等于已进入稳定发行版；RFC 与论坛讨论表示设计进展。性能及精度数字均适用于来源列出的设备、模型、数据类型和测试条件，不能直接外推到其他部署。
