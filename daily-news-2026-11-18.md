# Codex 技术情报每日动态（2026-11-18）

**调研窗口**：北京时间（2026-11-17 10:20:52, 2026-11-18 06:00:34]。

**覆盖方向**：PyTorch、LLVM/MLIR、Triton & TileLang、RISC-V，以及 AI 模型、基础设施、芯片与产业。

**信息口径**：以原始公告、正式发行、具体技术讨论和代码合并记录为依据；提案、实验、仿真、已合并实现与正式交付分别标明，性能数据保留测试条件。

## 今日要闻

- [OpenAI 发布 GPT-6.2 Astra 与 Sol，同时明确 API 可用范围、价格与更严格的安全限制。](https://openai.com/index/introducing-gpt-6-2-astra-sol/)
- [微软在 Ignite 公布 Microsoft IQ 与企业智能体平台扩展，并列明多个产品的正式可用状态。](https://blogs.microsoft.com/blog/2026/11/17/ignite-2026-building-the-frontier-ecosystem/)
- [NVIDIA 季度收入达 1158 亿美元，披露 Vera Rubin NVL72 已出货；下一季收入指引为 1300 亿美元。](https://nvidianews.nvidia.com/news/nvidia-announces-financial-results-for-third-quarter-fiscal-2027)
- [LLVM 23.1.6 正式发布，覆盖 Clang、lld、libc++ 与 MLIR 等子项目。](https://github.com/llvm/llvm-project/releases/tag/llvmorg-23.1.6)
- [PyTorch 确认不发布 2.15.1，未完成事项转入计划于 12 月 17 日发布的 2.16.0。](https://dev-discuss.pytorch.org/t/3536/1)

## 今日索引

- **PyTorch**：2.15.1 取消；ExecuTorch 的 QNN 量化、CUDA 权重导出、CPU 算子及语音部署更新，TorchAO 补齐 core ATen 的 PTQ 融合。
- **LLVM/MLIR**：LLVM 23.1.6、RemoveDeadValues 与浮点/数组语义讨论、浏览器 pass 工具，以及 RuyiAI XAME 和 FX 符号形状导入。
- **Triton & TileLang**：Triton-RISC-V bf16 ABI、Gluon bulk copy、AMD 静态 LDS、Rubin Early A Release 与 Ascend 测量口径。
- **RISC-V**：机器人推理演示、HPC 原始访谈、车规芯片出货采访，以及缓存 CSR、Sail BF16 和机器人仿真集成。
- **AI 业界**：GPT-6.2、微软 Ignite、Gemini 任务执行、稀疏注意力分析与 NVIDIA 财报及治理公告。

## 一、PyTorch 生态核心动态

### 1.1 工具 & 产品｜[不发布 PyTorch 2.15.1](https://dev-discuss.pytorch.org/t/3536/1)

北京时间：2026-11-18 03:51:58｜来源类型：官方社区原帖｜事件状态：已确认取消

**事实**：PyTorch 团队确认取消 2.15.1；未完成事项转入 2.16.0 milestone，下一版本计划于 2026 年 12 月 17 日发布。

**重要性**：下游可据此调整补丁升级预期、回归测试和版本锁定。

**风险与限制**：12 月 17 日仍是计划日期，公告未给出提前发行的承诺。

### 1.2 模型 & 技术｜[Qualcomm AI Engine Direct - 为近乎失活的通道赋予有效的逐通道权重 scale](https://github.com/pytorch/executorch/pull/23161)

北京时间：2026-11-18 05:39:22｜来源类型：官方 GitHub PR｜事件状态：已合并

**事实**：逐通道量化将落在 eps 下限的权重 scale 替换为有效通道 scale 的中位数，避免 HTP 计算异常与 int32 bias scale 过小。 作者在 Galaxy S26 Ultra／SM8850、200 张 VOC2012 验证图像上报告，16a8w DeepLabV3-ResNet101 的 mIoU 从 0.10% 恢复到 78.84%。

**重要性**：修复会使分割精度严重失真的量化路径，直接影响端侧模型正确性。

**风险与限制**：数字来自作者指定样本与设备；不能外推至完整数据集或其他后端。

### 1.3 模型 & 技术｜[跨方法将每份 CUDA 权重只写入一次](https://github.com/pytorch/executorch/pull/25059)

北京时间：2026-11-17 11:57:48｜来源类型：官方 GitHub PR｜事件状态：已合并

**事实**：低内存编译模式按 CPU 来源视图及版本跟踪参数，使多方法导出复用已写出的权重；默认深拷贝模式不变。 A100 80GB 上 Muse Glimmer 的 5 个 target 层加 1 个 draft 层、13 个方法测试中，写入量从 27.3 GB 降至 7.4 GB，materialize 时间从 313 秒降至 163 秒。

**重要性**：减少多入口部署工件生成的主机传输和重复写盘成本。

**风险与限制**：这是特定导出夹具的编译阶段结果，不是模型推理提速；buffer 与常量保留原处理方式。

### 1.4 模型 & 技术｜[添加优化的 embedding_bag 内核（#24851）](https://github.com/pytorch/executorch/pull/24851)

北京时间：2026-11-17 10:59:50｜来源类型：官方 GitHub PR｜事件状态：已合并

**事实**：新增 FP32 embedding_bag CPU 内核，支持 int32/int64 索引、sum/mean/max、padding 和 sum 的逐样本权重；使用向量化和线程池。 在 4096×128 权重表、256 bags×32 indices 的 warm FP32 sum 测试中，单线程相对 portable 内核的增益为 EPYC 7.54 倍、Apple Silicon 4.14 倍、iPhone 17 5.90 倍。

**重要性**：为检索和推荐类算子提供可量化的本地 CPU 执行优化。

**风险与限制**：结果仅为内核调用；Pixel 10 测量波动，未证明多线程收益；不支持的布局和类型回退 portable。

### 1.5 模型 & 技术｜[将 Pocket TTS 导出到 XNNPACK（#25120）](https://github.com/pytorch/executorch/pull/25120)

北京时间：2026-11-17 10:56:24｜来源类型：官方 GitHub PR｜事件状态：已合并

**事实**：新增 Pocket TTS 的 FP32、INT4、INT8 XNNPACK 导出和 Python 文本转音频 runner；一个 pocket_tts.pte 包含 text_embed、flow_lm、mimi_decode。 复用流式 transformer、KV cache 和 Mimi 组件，保留 Pocket 的模型维度与时间归一化；量化策略限于 FlowLM。

**重要性**：补充语音模型从 PyTorch 到本地 CPU 工件的部署链路。

**风险与限制**：runner 面向短语音；检查点、声线及状态契约有限定，不表示支持任意 TTS 模型或全部流式场景。

### 1.6 模型 & 技术｜[在 prepare_pt2e 中将 batch norm 折叠进 aten.convolution（#5102）](https://github.com/pytorch/ao/pull/5102)

北京时间：2026-11-18 02:40:36｜来源类型：官方 GitHub PR｜事件状态：已合并

**事实**：PTQ 的卷积与 batch norm 折叠识别非转置 aten.convolution，覆盖模型先分解为 core ATen 再调用 prepare_pt2e 的路径。 QAT 遇到相关分解卷积与 batch norm 时新增警告，而非声称已经完成融合。

**重要性**：减少模型导入顺序导致的量化图差异，与 ATen 导入和后端转换直接相关。

**风险与限制**：转置 aten.convolution 保持原样；量化器标注和 QAT 融合的支持范围没有随 PTQ 一起扩展。

## 二、LLVM/MLIR 最新进展

### 2.1 模型 & 技术｜[LLVM 23.1.6](https://github.com/llvm/llvm-project/releases/tag/llvmorg-23.1.6)

北京时间：2026-11-18 00:32:45｜来源类型：正式 Release｜事件状态：正式发布

**事实**：LLVM 23.1.6 发布，覆盖 LLVM、Clang、lld、libc++ 与 MLIR；发行说明列出循环向量化、目标代码生成和链接器等修复。

**重要性**：为下游编译器和工具链提供新的稳定维护版本。

**风险与限制**：源码与各平台二进制的供应节奏不同；第三方二进制不等同发行经理审核的工件。

### 2.2 模型 & 技术｜[\[RFC\] RemoveDeadValues：死函数中的已声明副作用与推导活跃性](https://discourse.llvm.org/t/92871/1)

北京时间：2026-11-17 18:58:46｜来源类型：官方社区原帖｜事件状态：RFC／提案

**事实**：提案分析私有、无调用者函数中的 RemoveDeadValues：保留有副作用的操作却删除其输入生产者，并用 ub.poison 填充，从而违反要求特定来源操作数的方言验证规则。 建议使保留操作的输入及生产者继续活跃，讨论副作用声明的优先级，以及死函数签名裁剪是否应交由 symbol-dce。

**重要性**：涉及 MLIR 优化后 IR 的有效性，对自定义方言及模型 lowering 的正确性有直接价值。

**风险与限制**：这是设计问题与修复方向的 RFC，尚非已合并的通用修复。

### 2.3 模型 & 技术｜[\[RFC\]\[mlir\]\[Linalg\]\[Vector\] Linalg 和 Vector 操作上的 fast-math 标志](https://discourse.llvm.org/t/92764/12)

北京时间：2026-11-17 18:57:34｜来源类型：官方社区原帖｜事件状态：讨论新进展

**事实**：主提案统一 Linalg／Vector fast-math 标志的标量化含义、重写传播与逐操作控制，并提议明确 reduction 重排及乘加融合的现有行为，用共享 helper 与 opt-in guard 传递标志。新回复中，Jianhui-Li 提议以向量化为抽象边界；作者补充，named op 泛化为 generic 时仍需生成相应标志。 rengolin 的新回复反对用隐式行为覆盖 reduction 重结合和 NaN 语义，要求由了解语义的前端或 pass 明确设置必要标志。

**重要性**：直接影响 PyTorch／JAX 图经 Linalg、Vector 到 LLVM 的浮点契约，不能把底层优化权限视为自动成立。

**风险与限制**：讨论仍有分歧，尚未形成统一方言规范。

相关原文：[\[RFC\]\[mlir\]\[Linalg\]\[Vector\] Fast-math flags on Linalg and Vector ops](https://discourse.llvm.org/t/92764/1)；[\[RFC\]\[mlir\]\[Linalg\]\[Vector\] Fast-math flags on Linalg and Vector ops](https://discourse.llvm.org/t/92764/8)；[\[RFC\]\[mlir\]\[Linalg\]\[Vector\] Fast-math flags on Linalg and Vector ops](https://discourse.llvm.org/t/92764/10)。

### 2.4 模型 & 技术｜[\[RFC\] 使用 llvm.assume.chained 向 LLVM 传递数组维度大小](https://discourse.llvm.org/t/92820/6)

北京时间：2026-11-18 02:48:34｜来源类型：官方社区原帖｜事件状态：讨论新进展

**事实**：主提案新增返回原指针的 llvm.assume.chained，以 operand bundles 携带 Flang 数组维度大小与步长，让派生的 load／store 保留信息，用于多维数组恢复、依赖分析和循环交换。作者在新回复中承认，仅匹配 extent 不足以安全应用源语言范围保证。 当前原型记录 Flang 归一化后的源下标，仅在 delinearization 恢复对应表达式时使用保证；直接附于 load/store 的表示方式仍待实现与比较。

**重要性**：把性能提示收紧到逐访问的可验证语义，关系 Fortran lowering 与数组优化的正确性。

**风险与限制**：作者仍将修订 RFC；回复所述实验未构成已上游的新 intrinsic 或完成的实现。

相关原文：[\[RFC\] Communicating Array Extents to LLVM Using llvm.assume.chained](https://discourse.llvm.org/t/92820/1)；[\[RFC\] Communicating Array Extents to LLVM Using llvm.assume.chained](https://discourse.llvm.org/t/92820/4)。

### 2.5 工具 & 产品｜[MLIR Step-Through：在浏览器中逐步执行 pass 流水线，并查看差异](https://discourse.llvm.org/t/92868/1)

北京时间：2026-11-17 12:14:33｜来源类型：官方社区原帖｜事件状态：已发布

**事实**：作者公开浏览器工具：用 Emscripten 编译的上游 mlir-opt 在 Web Worker 内运行，展示各 pass 的 IR，并比较步骤或不同流水线的差异。 IR 留在浏览器标签页，分享状态写入 URL fragment。

**重要性**：为检查模型编译流水线中的 IR 变化提供轻量入口。

**风险与限制**：作者因雇佣协议无法公开源码；长 URL 与 GitHub Pages 资源额度存在限制，不表示 MLIR 官方新产品。

### 2.6 模型 & 技术｜[\[RISCV\]\[XAME\] 添加 ZTT v0.6 矩阵后端支持](https://github.com/RuyiAI-Stack/llvm-project/pull/36)

北京时间：2026-11-17 15:05:00｜来源类型：官方 GitHub PR｜事件状态：已合并

**事实**：RuyiAI 下游 LLVM 增加 +xame 路径、138 个共享指令编码、riscv.ztt.matrix／acc 类型与 intrinsics，并经 SelectionDAG lowering 保留矩阵状态。 默认 profile 为 8×8、16 个 M／8 个 ACC 寄存器；gem5 profile 为 4×4、32 个 M／4 个 ACC。作者报告 4×4 整数矩阵乘在 AME_gem5 中结果正确。

**重要性**：打通 Buddy MLIR 生成的 ZTT v0.6 IR 与下游 RISC-V 后端，是团队编译与验证链路的重要进展。

**风险与限制**：XAME 接口及调用约定是暂定、编译器本地定义，未成为标准 RISC-V psABI；模拟器结果不等同硅片验证。Buddy 对应依赖更新 PR 仍未合并。

相关原文：[\[llvm\] Bump LLVM to 542365c9cd38](https://github.com/buddy-compiler/buddy-mlir/pull/1064)。

### 2.7 模型 & 技术｜[\[fx_importer\] 支持符号形状表达式中的 FloorDiv](https://github.com/llvm/torch-mlir/pull/4889)

北京时间：2026-11-18 00:23:04｜来源类型：官方 GitHub PR｜事件状态：已合并

**事实**：将 torch.export 对符号尺寸 // 生成的 FloorDiv 转为 affine floordiv，解除启用 import_symbolic_shape_expressions 时部分 strided convolution、chunk、unfold 模型的导入失败。

**重要性**：补齐 PyTorch FX 动态形状进入 MLIR 的具体缺口。

**风险与限制**：变化限于 FloorDiv 表达式翻译；不能推导所有动态形状或后端执行都已支持。

## 三、Triton & TileLang 技术动态

### 3.1 模型 & 技术｜[\[backend\] 修复 bf16 转换辅助函数的 launcher ABI](https://github.com/RuyiAI-Stack/triton-riscv/pull/121)

北京时间：2026-11-17 18:37:02｜来源类型：官方 GitHub PR｜事件状态：已合并

**事实**：bf16 转换辅助函数改用与 llc 调用点一致的浮点寄存器传参和返回方式，修复原先整数原型导致 fp32→bf16 输出为零或垃圾值的问题。 作者给出 x86 反汇编、riscv64 无 +zvfbfmin 的 libcall 检查，以及三个 Python 版本 riscv64 CI 的数值测试通过结果。

**重要性**：修复影响 Triton-RISC-V 常见 bf16 存储路径的严重数值错误。

**风险与限制**：当前 lowering 不生成的其他 bf16 libcall 家族不在本次修复范围；部分扩展路径只有寄存器类检查，不能泛化成所有转换均已执行验证。

### 3.2 模型 & 技术｜[\[Gluon\] 暴露 cp.async.bulk（非 TMA）](https://github.com/triton-lang/triton/pull/12758)

北京时间：2026-11-17 23:39:26｜来源类型：官方 GitHub PR｜事件状态：已合并

**事实**：新增 hopper.bulk 的 async_load、async_store、store_wait，覆盖线性 global↔shared bulk copy，并让编译器与 sanitizer 跟踪内存依赖。 作者在 GB300 上报告 341 个定向 pytest 用例通过，包含 copy、cluster、warp specialization 与 ConSan／GSan；Hopper 仅做代码生成覆盖。

**重要性**：为自定义内核提供显式 bulk copy 与同步接口，扩展 Gluon 的搬运和验证能力。

**风险与限制**：要求 SM90+、16 字节对齐和特定 PTX 版本；原始字节顺序不做 swizzle，GSan 在发出时记录访问而非持续跟踪完成。

### 3.3 模型 & 技术｜[\[AMD\] 通过静态共享内存向 LLVM 暴露 AMD LDS 分配](https://github.com/triton-lang/triton/pull/12715)

北京时间：2026-11-17 14:32:50｜来源类型：官方 GitHub PR｜事件状态：已合并

**事实**：按 ttg.shared 将 global_smem 定义为静态大小的内部数组，让 LLVM 在 occupancy 与寄存器预算中看见 LDS 用量；HIP 与 AOT launcher 不再请求动态 LDS。

**重要性**：改变编译器资源模型，可使高 LDS 内核的寄存器选择与真实共享内存约束一致。

**风险与限制**：AMD 动态 LDS 支持被移除；超限仍由启动前 OutOfResources 检查处理，原文没有给出普适性能增益。

### 3.4 模型 & 技术｜[\[Rubin\] 为 tcgen05_commit 添加可用的 Early A Release](https://github.com/triton-lang/triton/pull/12696)

北京时间：2026-11-17 19:08:53｜来源类型：官方 GitHub PR｜事件状态：已合并

**事实**：Rubin 的 tcgen05_commit 新增可选 early_a_release，用于共享内存 A 矩阵读取完成的 barrier 通知，并防止相关 commit 被 MMA lowering 合并或重排。

**重要性**：可更早表达 A 缓冲区的复用边界，而无需把 A-read 完成与整个 MMA 完成混为一谈。

**风险与限制**：只适用 Rubin 与其 PTX 9.4 最低要求；非 Rubin 目标及 ConSan 用法被拒绝，A-read 完成不代表 MMA 整体完成。

### 3.5 模型 & 技术｜[\[Ascend\] 使用 aclrtDeviceL2CacheFlush 清空基准测试的 L2 缓存](https://github.com/tile-ai/tilelang/pull/3891)

北京时间：2026-11-17 10:55:44｜来源类型：官方 GitHub PR｜事件状态：已合并

**事实**：Ascend 基准由读 scratch buffer 驱逐 L2 改为采样前调用 aclrtDeviceL2CacheFlush；同步清空位于 NPU event／msprof 计时及 runtime estimation 之外。

**重要性**：明确测量时的缓存状态，减少依赖替换策略的延迟偏差，影响后续算子性能比较的口径。

**风险与限制**：需要设备和 runtime 支持该 API；本次修订未重跑测试，不能把测量方法改变写成算子自身提速。

## 四、RISC-V 核心新闻

### 4.1 模型 & 技术｜[DeepComputing 大使 Iulian Arcus 在 Auki Labs 机器人舞台展示 RISC-V AI 推理实况](https://deepcomputing.io/deepcomputing-ambassador-iulian-arcus-shows-risc-v-ai-inference-in-action-at-auki-labs-robotics-stage/)

北京时间：2026-11-17 14:33:08｜来源类型：官方博客／公告｜事件状态：已发布

**事实**：DeepComputing 披露香港现场演示：机器人采集工厂数据，Adumbrate 用于动态保险评估；Qwen3.8 运行在 nuGPU 本地推理系统上，Tenstorrent Wormhole 加速器与 PodTorrent 分布式网络参与 token 服务。

**重要性**：展示 RISC-V 相关加速硬件、模型服务与机器人业务链路的实际集成案例。

**风险与限制**：这是合作方演示与厂商叙述；未给出吞吐、时延、功耗或独立验证，不代表纯 RISC-V CPU 独立完成模型推理。

### 4.2 深度洞见｜[V 问答……RISC-V HPC 重量级人物 Doug Norton 和 Nick Brown 博士](https://riscv.org/blog/v-questions-with-doug-norton-nick-brown/)

北京时间：2026-11-17 19:40:16｜来源类型：官方博客／公告｜事件状态：已发布

**事实**：原始访谈中，EPCC 的 Nick Brown 与 InspireSemi 的 Doug Norton 讨论 HPC 的软件缺口、单节点 RVV 优化和跨代移植。 Brown 引用上一年 gap analysis：英国国家超算约 55% 用户的代码可在 RISC-V 上运行，但可运行不等于已优化；Norton 强调 RISE、MPI／OpenMP 和标准工具链的作用。

**重要性**：为 RISC-V 软件栈补齐与 HPC 部署提供原始一线语境。

**风险与限制**：55% 是访谈引用的既有调查，不是本日新增基准；各厂商实现对可移植性的影响仍待验证。

### 4.3 融资 & 商业｜[计算万物 湘约未来丨全球首款量产上车 湖南造车规芯片筑牢自主基石](https://hn.rednet.cn/content/646055/51/16480375.html)

北京时间：2026-11-17 16:53:58｜来源类型：原始媒体报道｜事件状态：原始采访报道

**事实**：红网记者采访紫荆半导体高管：公司称 M100 已批量发货，截至 2026 年 10 月公司车规芯片累计出货超过 1000 万颗。 公司介绍四季度规划进入具身智能芯片，仍属规划；报道未给出已发布机器人芯片型号。

**重要性**：补充 RISC-V 车规产品商业化的原始采访与明确出货口径。

**风险与限制**：“全球首款”来自受访企业及报道标题，未作独立市场穷尽验证；累计出货是公司车规芯片总量，不应全部归给 M100。

### 4.4 模型 & 技术｜[riscv：按处理器核心拆分玄铁缓存 CSR 定义](https://github.com/YuzukiHD/SyterKit/commit/61ac087b0ffdb82a7e6a5bf868948df14af381ca)

北京时间：2026-11-17 23:35:42｜来源类型：官方 GitHub commit｜事件状态：已进入默认分支

**事实**：SyterKit 为 C906、C907、E907 拆分扩展 CSR 定义，按手册字段重建缓存初始化，避免各内核共用冲突位号；分别初始化和启用所配置的缓存。

**重要性**：修正裸机与引导框架中的硬件寄存器语义，关系玄铁板卡启动和缓存行为。

**风险与限制**：提交有静态断言及手册依据，未提供覆盖各板卡的硬件性能或完整回归结果。

### 4.5 模型 & 技术｜[修复 vfwmaccbf16.vf 的 EMUL 检查及无活跃元素时的 invalid 标志](https://github.com/riscv/sail-riscv/pull/2115)

北京时间：2026-11-17 21:40:21｜来源类型：官方 GitHub PR｜事件状态：已合并

**事实**：Sail 将 BF16 向量 widening 指令的 EMUL 断言移到合法性检查后，并推迟标量 BF16 转换到首个活跃元素，避免无活跃元素时错误置浮点 invalid 标志。

**重要性**：修正 ISA 可执行参考模型中的异常与掩码语义，关系差分验证的参考答案。

**风险与限制**：这是模型语义修复，不是硅片能力升级；不能据此断言所有硬件都存在或已修复同样问题。

### 4.6 模型 & 技术｜[feat(manipulation)：集成设备组合、共享 MuJoCo runtime 与运动完成判定](https://github.com/spacemit-robotics/manipulation/pull/4)

北京时间：2026-11-17 11:31:46｜来源类型：官方 GitHub PR｜事件状态：已合并

**事实**：合入设备组合与共享标定、自有 MuJoCo runtime、运动学、RGB-D 深度缓冲和 K3 渲染路径，收紧运动 deadline 与末端执行器完成判定。 旧 MuJoCo 角色别名被移除，配置改用按设备型号命名的角色；作者称本机和 K3 已完成构建与硬件安全回归。

**重要性**：将仿真、标定、设备配置和完成判定连成明确的机器人部署接口。

**风险与限制**：属于破坏性配置变更；回归为作者报告，不能把仿真与有限安全检查视作实际生产认证。

### 4.7 模型 & 技术｜[feat：添加 Hiking 感知控制仿真](https://github.com/spacemit-robotics/humanoid_unitree_g1/pull/20)

北京时间：2026-11-17 16:04:02｜来源类型：官方 GitHub PR｜事件状态：已合并

**事实**：G1 应用增加 Hiking encoder／actor、关节映射与增益，经 ZMQ 发布深度并支持策略切换；新增 16 cm 楼梯仿真场景。

**重要性**：扩展从深度输入到人形机器人策略运行的可复现仿真路径。

**风险与限制**：README 明确先支持仿真；缺少实时深度的离线测试跳过 Hiking，不能写成实机爬楼部署完成。

## 五、AI 业界重磅

### 5.1 重磅｜[推出 GPT-6.2 Astra 和 Sol](https://openai.com/index/introducing-gpt-6-2-astra-sol/)

北京时间：2026-11-18 01:45:00｜来源类型：官方博客／公告｜事件状态：已发布

**事实**：OpenAI 发布 GPT-6.2 Astra 与 Sol，面向 ChatGPT Work、Codex 和 API；官方强调专业任务、代码、计算机操作与 token 效率。 标准 API 每百万 token 输入／缓存输入／输出价格：Astra 为 8／0.40／40 美元，Sol 为 2／0.10／10 美元。

**重要性**：为专业工作和工程智能体提供新的模型与成本选择。

**风险与限制**：性能结论来自官方及其列示评测；Cyber、Bio/Chem 与 AI 自我改进风险评估有不同等级，发布伴随更严格限制，不能把能力提升视为无约束可用。

### 5.2 重磅｜[Ignite 2026：构建前沿生态系统](https://blogs.microsoft.com/blog/2026/11/17/ignite-2026-building-the-frontier-ecosystem/)

北京时间：2026-11-18 02:36:03｜来源类型：官方博客／公告｜事件状态：已发布

**事实**：微软在 Ignite 公布 Copilot 新工作形态及 Microsoft IQ，后者组织 Fabric、Foundry、Microsoft 365 和第三方上下文。 公告称 Microsoft-Decision-1.1、Perception 和 Cobalt 200 虚拟机正式可用，并介绍 Agent 365 的分析与治理扩展。

**重要性**：企业智能体的模型、上下文、控制平面和云资源继续形成一体化部署接口。

**风险与限制**：一篇 keynote 涵盖多个不同状态产品；不能把所有新形态与预告都写成全面上线，也不能将供应商成本宣称当独立测量。

### 5.3 深度洞见｜[协同设计稀疏注意力，实现高效长上下文推理](https://developer.nvidia.com/blog/co-designing-sparse-attention-for-efficient-long-context-inference/)

北京时间：2026-11-18 01:30:00｜来源类型：官方博客／公告｜事件状态：已发布

**事实**：NVIDIA 技术分析将长上下文瓶颈拆成 indexer、Top-k 与核心 attention，提出减少需计算分数的 key、融合 Top-k、块级选择和 head sharing 等设计原则。 文中报告 Guess-Verify-Refine Top-k 在 DeepSeek V3.2／V4 decode 上比标准 radix select 快 2.1—2.4 倍。

**重要性**：为推理内核和模型协同设计提供具体资源与数据搬运分析。

**风险与限制**：这是方法与指定实现的测量，不是所有稀疏模型端到端提速；FP16、块大小和 key 数量的建议依赖误差范围与硬件。

### 5.4 融资 & 商业｜[NVIDIA 公布 2027 财年第三季度财务业绩](https://nvidianews.nvidia.com/news/nvidia-announces-financial-results-for-third-quarter-fiscal-2027)

北京时间：2026-11-18 05:20:00｜来源类型：官方博客／公告｜事件状态：已发布

**事实**：NVIDIA 报告 2027 财年第三季度收入 1158 亿美元、同比增 103%；数据中心收入 1076 亿美元、同比增 110%，GAAP 净利润 653 亿美元。 公司表示 Vera Rubin NVL72 已出货，下一季度收入指引为 1300 亿美元、上下浮动 3%。

**重要性**：展示 AI 数据中心需求的收入规模及新平台交付进度。

**风险与限制**：下一季度是前瞻指引而非已实现收入；指引不假设来自中国的数据中心计算收入。

### 5.5 工具 & 产品｜[让 Gemini 为你工作](https://blog.google/innovation-and-ai/products/gemini-app/tasks-in-gemini-free-new-features/)

北京时间：2026-11-18 01:00:00｜来源类型：官方博客／公告｜事件状态：已发布

**事实**：Google 以 Gemini 4 Hydrogen 扩展任务执行，免费功能开始在美国、日本、韩国、加拿大、澳大利亚和印度推出，部分高级功能先在美国推出。 原文介绍代拨电话、预约和账单等任务，并说明敏感任务使用硬件隔离虚拟机及授权边界。

**重要性**：个人助手从对话扩大到跨账户、多步骤的实际任务执行。

**风险与限制**：功能逐步推出且有地区差异；示例不保证每个账户、商户或任务均可完成。

### 5.6 行业 & 人事｜[NVIDIA 董事会设立国家安全与出口管制监督委员会](https://nvidianews.nvidia.com/news/nvidia-board-of-directors-establishes-national-security-and-export-control-oversight-committee)

北京时间：2026-11-17 18:00:00｜来源类型：官方博客／公告｜事件状态：已宣布

**事实**：NVIDIA 董事会设立常设委员会，监督国家安全政策、出口合规与全球贸易风险；Suzanne Nora Johnson 任主席，Stephen C. Neal 和 John O. Dabiri 任成员。 国家安全与全球贸易合规团队日常工作向委员会和临时 CEO Colette Kress 报告，委员会将聘请外部独立专家。

**重要性**：公司将芯片贸易与国家安全合规提升为董事会层面的常设监督职能。

**风险与限制**：这是治理安排，不构成对相关调查或任何个人责任的裁决，也不直接改变出口许可规则。

## 六、总结与趋势观察

- **数值正确性需要贯穿模型量化、调用约定与 ISA 参考模型。** QNN 的极小 scale、bf16 helper 的寄存器类别，以及 Sail 的无活跃元素异常标志，分别展示了不同层次的数值边界；一层的修复不能代替另一层的验证。 参见：[Qualcomm AI Engine Direct - 为近乎失活的通道赋予有效的逐通道权重 scale](https://github.com/pytorch/executorch/pull/23161)、[\[backend\] 修复 bf16 转换辅助函数的 launcher ABI](https://github.com/RuyiAI-Stack/triton-riscv/pull/121)、[修复 vfwmaccbf16.vf 的 EMUL 检查及无活跃元素时的 invalid 标志](https://github.com/riscv/sail-riscv/pull/2115)。

- **编译资源与测量条件正在变得更显式。** AMD LDS 进入静态资源模型，Ascend 基准明确 L2 清空时机，ExecuTorch 区分低内存导出的共享权重来源；这些改动分别影响编译决策、测量可比性和工件生成成本。 参见：[\[AMD\] 通过静态共享内存向 LLVM 暴露 AMD LDS 分配](https://github.com/triton-lang/triton/pull/12715)、[\[Ascend\] 使用 aclrtDeviceL2CacheFlush 清空基准测试的 L2 缓存](https://github.com/tile-ai/tilelang/pull/3891)、[跨方法将每份 CUDA 权重只写入一次](https://github.com/pytorch/executorch/pull/25059)。

- **智能体产品同时扩展任务能力与控制面。** Microsoft IQ／Agent 365 将企业上下文与治理放在平台层，Gemini 扩展电话、预约与账户任务并明确隔离和授权边界；功能范围、地区和上线状态仍各有约束。 参见：[Ignite 2026：构建前沿生态系统](https://blogs.microsoft.com/blog/2026/11/17/ignite-2026-building-the-frontier-ecosystem/)、[让 Gemini 为你工作](https://blog.google/innovation-and-ai/products/gemini-app/tasks-in-gemini-free-new-features/)。

## 附录：信源说明

本期主要来源为 PyTorch 与 LLVM 官方社区、项目 GitHub 的正式 Release 和具体实现记录，以及 OpenAI、Microsoft、Google、NVIDIA、DeepComputing、RISC-V International 的原始发布和红网的原始采访。技术项目包括 ExecuTorch、TorchAO、LLVM/MLIR、torch-mlir、Buddy、RuyiAI、Triton、TileLang、Sail、SyterKit 与进迭机器人软件。

项目维护者与厂商报告的测试和性能数据保留设备、模型、输入规模和验证方式；内核测量不等于端到端模型表现，模拟器和仿真结果不等于硅片或实机部署，原始采访中的企业陈述不等于独立市场验证。提案、计划日期和未合并依赖更新不视为正式交付。
