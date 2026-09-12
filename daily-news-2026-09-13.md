# Codex 技术情报每日动态（2026-09-13）

调研窗口：北京时间 2026-09-10 06:40:34 至 2026-09-13 07:04:09（窗口起点按上一期已发布正文 verified evidence 最大 occurredAt；采用半开区间 `(start,end]`）。

覆盖方向：PyTorch、LLVM/MLIR、Triton/TileLang、RISC-V 软件与硬件生态、AI 模型与基础设施。

信息口径：正文仅采用已打开并核验的官方博客、公告、Release、论坛/RFC、GitHub 官方页面或有编辑责任的原始报道；聚合器和搜索结果只作候选发现。

## 今日要闻

- [介绍 Agents API](https://openai.com/index/introducing-the-agents-api/)：OpenAI 将 Agents API 以 public beta 形式开放，提供 Codex harness、托管沙箱、工具搜索、程序化调用和多智能体并发能力。
- [检测并应对 AI 滥用：2026 年 9 月](https://www.anthropic.com/threat-intelligence-report-september-2026)：Anthropic 威胁情报团队报告过去八个月识别并阻断使用 Claude 进行恶意活动的行动，并给出网络安全、监控、诈骗和生物风险案例。
- [PyTorch 中国大会 2026：推进开源 AI 栈](https://pytorch.org/blog/pytorch-conference-china-2026-advancing-the-open-source-ai-stack/)：PyTorch Foundation 总结 9 月 8—9 日上海会议，公布 Alibaba Cloud、Ant Group、Cambricon 加入基金会，并介绍 Accelerator Integration Working Group 的多后端测试矩阵。
- [🔍 RuyiSDK 软件包页面上线：一站式查取与共建 RISC-V 软硬件开发资源](https://ruyisdk.cn/t/2829)：RuyiSDK 发布软件包索引页面，按设备和软件包组织镜像、版本及适配关系，并支持 x86_64、aarch64、riscv64。
- [Helion x 🤗 HF Kernels：构建并发布开箱即用的高性能内核](https://pytorch.org/blog/helion-x-%f0%9f%a4%97-hf-kernels-building-and-shipping-out-of-the-box-performant-kernels/)：Helion 获 Hugging Face Kernels 支持；文章给出预调优配置、跨 CUDA/ROCm/XPU 打包流程，并报告注意力预调优形状几何平均加速 1.20 倍。

## 今日索引

- 1：一、PyTorch 生态核心动态。
- 2：二、LLVM/MLIR 最新进展。
- 3：三、Triton & TileLang 技术动态。
- 4：四、RISC-V 核心新闻。
- 5：五、AI 业界重磅。

## 一、PyTorch 生态核心动态

### 1.1 重磅｜[PyTorch 中国大会 2026：推进开源 AI 栈](https://pytorch.org/blog/pytorch-conference-china-2026-advancing-the-open-source-ai-stack/)

北京时间：2026-09-10T08:01:29+08:00；来源类型：official_blog；事件状态：published。

PyTorch Foundation 总结 9 月 8—9 日上海会议，公布 Alibaba Cloud、Ant Group、Cambricon 加入基金会，并介绍 Accelerator Integration Working Group 的多后端测试矩阵。 这项变化的影响是：开源栈与异构硬件适配合作扩大。 风险与限制：仍需关注实现范围、可用性或后续验证；原文未承诺的效果不作推断。

### 1.2 模型 & 技术｜[Qualcomm AI Engine Direct：增加 LPAI 自定义算子支持和示例](https://github.com/pytorch/executorch/pull/22659)

北京时间：2026-09-12T06:00:40+08:00；来源类型：official_github_pr；事件状态：published。

合并变更加入 Qualcomm AI Engine Direct 的 LPAI 自定义算子支持与示例。 这项变化的影响是：扩大 ExecuTorch 面向 Qualcomm 低功耗 AI 路径的后端覆盖。 风险与限制：仍需关注实现范围、可用性或后续验证；原文未承诺的效果不作推断。

### 1.3 模型 & 技术｜[增加批量序列 KV 缓存布局并重组控制接口](https://github.com/pytorch/executorch/pull/22662)

北京时间：2026-09-12T06:10:48+08:00；来源类型：official_github_pr；事件状态：published。

合并变更为 ExecuTorch 增加批量序列 KV cache 布局并重组控制接口。 这项变化的影响是：直接影响端侧生成式模型的缓存组织。 风险与限制：仍需关注实现范围、可用性或后续验证；原文未承诺的效果不作推断。

### 1.4 工具 & 产品｜[Torch-TensorRT v2.14.0](https://github.com/pytorch/TensorRT/releases/tag/v2.14.0)

北京时间：2026-09-10T13:39:50+08:00；来源类型：official_github_release；事件状态：published。

Torch-TensorRT 发布 v2.14.0，Release 页面列出本版本变更与构建产物。 这项变化的影响是：为 PyTorch 推理部署提供正式版本节点。 风险与限制：仍需关注实现范围、可用性或后续验证；原文未承诺的效果不作推断。

## 二、LLVM/MLIR 最新进展

### 2.1 模型 & 技术｜[RFC：在 llvm-profgen 中支持基本采样](https://discourse.llvm.org/t/91782)

北京时间：2026-09-10T21:49:57+08:00；来源类型：official_forum；事件状态：published。

RFC 提议 llvm-profgen 直接读取含采样指令指针的 perf basic events，并通过地址映射、任务生命周期和调试信息生成 LLVM sample profile。 这项变化的影响是：减少对树外 AutoFDO 工具链的依赖。 风险与限制：仍需关注实现范围、可用性或后续验证；原文未承诺的效果不作推断。

### 2.2 模型 & 技术｜[RFC：增加每函数代码模型属性](https://discourse.llvm.org/t/91786)

北京时间：2026-09-11T03:34:10+08:00；来源类型：official_forum；事件状态：published。

RFC 提议在 LLVM IR 函数定义和声明上增加 code_model 属性，解决 ThinLTO 跨模块导入时函数代码模型丢失问题。 这项变化的影响是：改善 LTO 与手写汇编混用场景的代码生成一致性。 风险与限制：仍需关注实现范围、可用性或后续验证；原文未承诺的效果不作推断。

### 2.3 模型 & 技术｜[[LLVMCPU] 为 RISC-V + FP16 端口的 DataTiledMMAAttr 增加 `vlen` 属性](https://github.com/iree-org/iree/pull/24858)

北京时间：2026-09-11T18:38:17+08:00；来源类型：official_github_pr；事件状态：published。

合并变更在 IREE LLVMCPU 路径为 RISC-V FP16 DataTiledMMAAttr 增加 vlen 属性。 这项变化的影响是：把 RISC-V 向量长度信息带入矩阵乘 lowering。 风险与限制：仍需关注实现范围、可用性或后续验证；原文未承诺的效果不作推断。

### 2.4 深度洞见｜[RFC：用于分布式异构计算的 MLIR 方言](https://discourse.llvm.org/t/86960)

北京时间：2026-09-10T22:20:59+08:00；来源类型：official_forum；事件状态：published。

讨论新进展：DHIR 已能接收 affine MLIR、初步 lowering linalg.generic，并将 Torch-MLIR 工作负载生成可由 mpirun 执行的二进制；设计采用 schedule、replicate、converge 与运行时设备映射。 这项变化的影响是：连接 Torch-MLIR、Linalg 与异构分布式执行。 风险与限制：仍需关注实现范围、可用性或后续验证；原文未承诺的效果不作推断。

### 2.5 深度洞见｜[RFC：将 CIR 流程边界设为一等驱动产物](https://discourse.llvm.org/t/90998)

北京时间：2026-09-12T08:52:32+08:00；来源类型：official_forum；事件状态：published。

讨论新进展：作者提交 Draft-PR 原型，提出 clang/cc1 往返 CIR，并增加 target-specific lowering 前的 CIR 输出点。 这项变化的影响是：为 CIR 工具链和可重放编译流程定义稳定边界。 风险与限制：仍需关注实现范围、可用性或后续验证；原文未承诺的效果不作推断。

### 2.6 深度洞见｜[MLIR 中的源匹配与重写（CIR 和 FIR）](https://discourse.llvm.org/t/91771)

北京时间：2026-09-13T02:55:51+08:00；来源类型：official_forum；事件状态：published。

讨论新进展：SMR/PGL 作者说明当前尚不支持过程内分析，但计划扩展模式与分析能力，使用户无需修改编译器即可表达更复杂优化。 这项变化的影响是：为 MLIR 源级习语匹配和重写提供研究型路径。 风险与限制：仍需关注实现范围、可用性或后续验证；原文未承诺的效果不作推断。

### 2.7 工具 & 产品｜[iree 候选版本 iree-3.12.0rc20260912](https://github.com/iree-org/iree/releases/tag/iree-3.12.0rc20260912)

北京时间：2026-09-12T18:39:37+08:00；来源类型：official_github_release；事件状态：published。

IREE 发布 3.12.0 RC 候选，Release 页面提供该候选版本标识。 这项变化的影响是：为编译器和运行时集成提供新的预发布测试节点。 风险与限制：仍需关注实现范围、可用性或后续验证；原文未承诺的效果不作推断。

## 三、Triton & TileLang 技术动态

### 3.1 模型 & 技术｜[[Refactor][Language] 使 T.Kernel 目标中立并采用方言所有的启动注解](https://github.com/tile-ai/tilelang/pull/3186)

北京时间：2026-09-10T17:33:51+08:00；来源类型：official_github_pr；事件状态：published。

合并变更使 TileLang T.Kernel 与目标解耦，并把 launch 注解归属到 dialect。 这项变化的影响是：为多后端 kernel 表达保留更清晰的目标边界。 风险与限制：仍需关注实现范围、可用性或后续验证；原文未承诺的效果不作推断。

### 3.2 模型 & 技术｜[[BC-Breaking][Frontend] 修复 min/max 类型提升](https://github.com/triton-lang/triton/pull/11472)

北京时间：2026-09-10T23:16:03+08:00；来源类型：official_github_pr；事件状态：published。

合并变更修正 Triton 前端 min/max 类型提升规则，并标注为 BC-breaking。 这项变化的影响是：类型语义变化可能影响现有内核源码与编译结果。 风险与限制：仍需关注实现范围、可用性或后续验证；原文未承诺的效果不作推断。

### 3.3 模型 & 技术｜[[KERNELS] 支持未缩放 FP8 E4M3 与 FP16/BF16 矩阵乘](https://github.com/triton-lang/triton/pull/11652)

北京时间：2026-09-11T02:17:06+08:00；来源类型：official_github_pr；事件状态：published。

合并变更支持未缩放 FP8 E4M3 与 FP16/BF16 的矩阵乘路径。 这项变化的影响是：扩展 Triton 低精度内核组合。 风险与限制：仍需关注实现范围、可用性或后续验证；原文未承诺的效果不作推断。

### 3.4 模型 & 技术｜[[SPIRV] 使用 `SPV_INTEL_subgroup_scaled_matrix_multiply_accumulate` 扩展](https://github.com/intel/intel-xpu-backend-for-triton/pull/7953)

北京时间：2026-09-11T03:26:21+08:00；来源类型：official_github_pr；事件状态：published。

合并变更在 Triton Intel XPU 后端使用 Intel subgroup scaled matrix multiply accumulate 扩展。 这项变化的影响是：推进 Intel XPU 对缩放矩阵乘的后端支持。 风险与限制：仍需关注实现范围、可用性或后续验证；原文未承诺的效果不作推断。

### 3.5 模型 & 技术｜[Helion x 🤗 HF Kernels：构建并发布开箱即用的高性能内核](https://pytorch.org/blog/helion-x-%f0%9f%a4%97-hf-kernels-building-and-shipping-out-of-the-box-performant-kernels/)

北京时间：2026-09-12T01:45:57+08:00；来源类型：official_blog；事件状态：published。

Helion 获 Hugging Face Kernels 支持；文章给出预调优配置、跨 CUDA/ROCm/XPU 打包流程，并报告注意力预调优形状几何平均加速 1.20 倍。 这项变化的影响是：降低高性能内核分发与冷启动成本。 风险与限制：仍需关注实现范围、可用性或后续验证；原文未承诺的效果不作推断。

### 3.6 模型 & 技术｜[[BACKEND] 单独打包固定版本的 AMD LLVM 代码生成](https://github.com/triton-lang/triton/pull/11485)

北京时间：2026-09-12T01:49:00+08:00；来源类型：official_github_pr；事件状态：published。

合并变更为 Triton 单独打包固定版本的 AMD LLVM codegen。 这项变化的影响是：提高 AMD 后端代码生成依赖的可复现性。 风险与限制：仍需关注实现范围、可用性或后续验证；原文未承诺的效果不作推断。

### 3.7 模型 & 技术｜[[Refactor][JIT] 用后端能力标志替代被调用者分配输出的目标探测](https://github.com/tile-ai/tilelang/pull/3218)

北京时间：2026-09-12T22:07:06+08:00；来源类型：official_github_pr；事件状态：published。

合并变更用后端能力标志替代对 callee 分配输出的目标探测。 这项变化的影响是：减少 JIT 目标判断与后端耦合。 风险与限制：仍需关注实现范围、可用性或后续验证；原文未承诺的效果不作推断。

### 3.8 深度洞见｜[增加并优化 Hopper 因果 pingpong 注意力](https://github.com/meta-pytorch/tritonbench/pull/1263)

北京时间：2026-09-12T03:33:35+08:00；来源类型：official_github_pr；事件状态：published。

合并变更新增并优化 Hopper causal pingpong attention 基准。 这项变化的影响是：补充 Hopper 注意力内核的可重复性能评估。 风险与限制：仍需关注实现范围、可用性或后续验证；原文未承诺的效果不作推断。

## 四、RISC-V 核心新闻

### 4.1 模型 & 技术｜[T-SBI 实现 Zicntr*](https://github.com/riscv/riscv-arch-test/pull/2161)

北京时间：2026-09-11T10:01:33+08:00；来源类型：official_github_pr；事件状态：published。

架构测试仓库增加 Zicntr* 的 T-SBI 实现测试。 这项变化的影响是：为特权规范相关实现提供可执行一致性检查。 风险与限制：仍需关注实现范围、可用性或后续验证；原文未承诺的效果不作推断。

### 4.2 模型 & 技术｜[feat(vector)：支持向量除法指令](https://github.com/OpenXiangShan/XiangShan/pull/6552)

北京时间：2026-09-12T18:14:58+08:00；来源类型：official_github_pr；事件状态：published。

合并变更为香山处理器增加向量除法指令支持。 这项变化的影响是：扩展向量执行单元覆盖，关联 AI/HPC 软件栈适配。 风险与限制：仍需关注实现范围、可用性或后续验证；原文未承诺的效果不作推断。

### 4.3 模型 & 技术｜[feat(firmware)：增加 RustSBI 和 Asterinas 启动镜像](https://github.com/YuzukiHD/YuzukiNeko/commit/51e18e075ec8fd98b7db8ddbda8a867bfe726739)

北京时间：2026-09-12T21:12:04+08:00；来源类型：official_github_commit；事件状态：published。

YuzukiNeko 默认分支提交加入 RustSBI 与 Asterinas 启动镜像。 这项变化的影响是：为该 RISC-V 平台增加不同固件/操作系统启动路径。 风险与限制：仍需关注实现范围、可用性或后续验证；原文未承诺的效果不作推断。

### 4.4 深度洞见｜[重构 `memory_exception` 以接收访问的附加信息](https://github.com/riscv/sail-riscv/commit/411a571c2ed353f5b6636a7f766a35c4986b1681)

北京时间：2026-09-12T03:04:24+08:00；来源类型：official_github_commit；事件状态：published。

Sail RISC-V 形式化模型重构 memory_exception，使其接收更多访问上下文信息。 这项变化的影响是：提高异常语义模型对访问属性的表达能力。 风险与限制：仍需关注实现范围、可用性或后续验证；原文未承诺的效果不作推断。

### 4.5 工具 & 产品｜[🔍 RuyiSDK 软件包页面上线：一站式查取与共建 RISC-V 软硬件开发资源](https://ruyisdk.cn/t/2829)

北京时间：2026-09-11T11:40:05+08:00；来源类型：official_forum；事件状态：published。

RuyiSDK 发布软件包索引页面，按设备和软件包组织镜像、版本及适配关系，并支持 x86_64、aarch64、riscv64。 这项变化的影响是：降低 RISC-V 开发板与系统资源查找成本。 风险与限制：仍需关注实现范围、可用性或后续验证；原文未承诺的效果不作推断。

### 4.6 工具 & 产品｜[发布 riscv-isa-release-be8cb00-2026-09-11](https://github.com/riscv/riscv-isa-manual/releases/tag/riscv-isa-release-be8cb00-2026-09-11)

北京时间：2026-09-11T22:40:26+08:00；来源类型：official_github_release；事件状态：published。

RISC-V ISA manual 发布 2026-09-11 版本，Release 页面提供对应规范快照。 这项变化的影响是：为实现者和工具链提供新的 ISA 文档基线。 风险与限制：仍需关注实现范围、可用性或后续验证；原文未承诺的效果不作推断。

## 五、AI 业界重磅

### 5.1 重磅｜[介绍 Agents API](https://openai.com/index/introducing-the-agents-api/)

北京时间：2026-09-10T08:00:00+08:00；来源类型：official_blog；事件状态：published。

OpenAI 将 Agents API 以 public beta 形式开放，提供 Codex harness、托管沙箱、工具搜索、程序化调用和多智能体并发能力。 这项变化的影响是：把长时运行 agent 基础设施以 API 形式提供给开发者。 风险与限制：仍需关注实现范围、可用性或后续验证；原文未承诺的效果不作推断。

### 5.2 重磅｜[检测并应对 AI 滥用：2026 年 9 月](https://www.anthropic.com/threat-intelligence-report-september-2026)

北京时间：2026-09-10T08:00:00+08:00；来源类型：official_blog；事件状态：published。

Anthropic 威胁情报团队报告过去八个月识别并阻断使用 Claude 进行恶意活动的行动，并给出网络安全、监控、诈骗和生物风险案例。 这项变化的影响是：为模型滥用检测和响应提供一手案例。 风险与限制：仍需关注实现范围、可用性或后续验证；原文未承诺的效果不作推断。

### 5.3 模型 & 技术｜[使用 BioNeMo Inference Runtime 进行高吞吐结构预测](https://developer.nvidia.com/blog/high-throughput-structure-prediction-with-bionemo-inference-runtime/)

北京时间：2026-09-10T23:00:00+08:00；来源类型：official_blog；事件状态：published。

NVIDIA 介绍 BioNeMo Inference Runtime 用于高吞吐结构预测的部署方法。 这项变化的影响是：连接生成式模型推理基础设施与生命科学工作负载。 风险与限制：仍需关注实现范围、可用性或后续验证；原文未承诺的效果不作推断。

### 5.4 模型 & 技术｜[全栈 NIM 优化如何让 Nemotron 3 Ultra 服务的用户数增加 2.5 倍](https://developer.nvidia.com/blog/how-full-stack-nim-optimizations-deliver-2-5x-more-users-on-nemotron-3-ultra/)

北京时间：2026-09-11T00:55:32+08:00；来源类型：official_blog；事件状态：published。

NVIDIA 介绍 Nemotron 3 Ultra 的全栈 NIM 优化，并报告可服务用户数提升 2.5 倍。 这项变化的影响是：把推理优化从单算子扩展到服务栈级指标。 风险与限制：仍需关注实现范围、可用性或后续验证；原文未承诺的效果不作推断。

### 5.5 深度洞见｜[扩展在线存储以服务超过 10 亿 ChatGPT 用户](https://openai.com/index/scaling-storage-one-billion-users-part-one)

北京时间：2026-09-11T18:00:00+08:00；来源类型：official_blog；事件状态：published。

OpenAI 介绍为超过 10 亿 ChatGPT 用户扩展在线存储的工程方案。 这项变化的影响是：显示模型产品规模化的存储与可靠性约束。 风险与限制：仍需关注实现范围、可用性或后续验证；原文未承诺的效果不作推断。

### 5.6 工具 & 产品｜[在 API 中使用 GPT‑Live‑1 构建更自然的语音体验](https://openai.com/index/introducing-gpt-live-1-in-the-api/)

北京时间：2026-09-10T08:00:00+08:00；来源类型：official_blog；事件状态：published。

OpenAI 发布 GPT-Live-1 API，主打全双工语音会话与更细的语音 agent 控制。 这项变化的影响是：降低语音 agent 架构复杂度与交互延迟。 风险与限制：仍需关注实现范围、可用性或后续验证；原文未承诺的效果不作推断。

### 5.7 工具 & 产品｜[使用 Gradio Workflow 重建 AUTOMATIC1111](https://huggingface.co/blog/gradio-workflow-1111)

北京时间：2026-09-10T08:00:00+08:00；来源类型：official_blog；事件状态：published。

Hugging Face 以 gr.Workflow 重建 AUTOMATIC1111，展示多模型节点、REST endpoint 与 MCP 工具输出。 这项变化的影响是：把图形工作流、模型调用和 agent 工具接口合并到同一应用层。 风险与限制：仍需关注实现范围、可用性或后续验证；原文未承诺的效果不作推断。

### 5.8 工具 & 产品｜[Gemini 应用现已在 Windows 上可用](https://blog.google/innovation-and-ai/products/gemini-app/gemini-app-now-on-windows/)

北京时间：2026-09-11T00:00:00+08:00；来源类型：official_blog；事件状态：published。

Google 宣布 Gemini 应用可在 Windows 使用。 这项变化的影响是：扩大桌面端模型产品分发入口。 风险与限制：仍需关注实现范围、可用性或后续验证；原文未承诺的效果不作推断。

## 六、总结与趋势观察

- 开源 AI 栈正在把模型、硬件和部署接口放在同一协作层：PyTorch Conference China 的异构适配工作组、Agents API 的托管 harness、Helion/Kernels 的内核分发共同体现了这一方向。
- 编译器中间表示和内核 DSL 的边界继续被显式化：DHIR、CIR 往返 RFC、Triton 类型语义与 TileLang 目标解耦分别从分布式调度、可重放编译和多后端表达推进基础设施演进。
- RISC-V 生态的重点同时落在规范安全语义和可用软件入口：SPMP 标准化、Sail 异常模型、架构测试、香山向量除法与 RuyiSDK 资源索引覆盖了从规范到部署的不同层次。

## 附录：信源说明

本期使用官方项目博客与 Feed、PyTorch/LLVM/RuyiSDK 论坛、GitHub 官方 API 页面、RISC-V 与上海交通大学原始发布、NVIDIA/Google/OpenAI/Anthropic/Hugging Face 官方文章。GitHub 记录主要用于技术脉动；有限分页、动态页面和聚合候选不作为独立事实来源，人物逐人检索及各栏目逐源审计详见研究日志。
