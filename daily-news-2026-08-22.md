# Codex 技术情报日报

日期：2026年8月22日  
统计时段：8月21日 06:00—8月22日 06:00（北京时间）  
今日要闻：

- Triton 将 Hopper/Blackwell 的 MXFP4 权重布局转换显存峰值降低 50%—91.3%，热态转换提速 4.24—21.26 倍。
- ONNX Runtime WebGPU 让 PagedAttention 直接读取分页 KV cache，实测解码几何平均约提速 2 倍。
- Anthropic 将 Mythos 5 接入 Claude Security，并设立 3500 万美元开源软件安全额度基金。
- RISC-V International 介绍 EDACrux 公测套件，其 RVFI 检查器可把退休指令还原为可读轨迹并执行六项 ISA 一致性检查。

## 一、PyTorch 生态核心动态

### 模型 & 技术｜[ExecuTorch 新增独立于 PTE 的原生 `.ptn` 产物链路](https://github.com/pytorch/executorch/pull/22007)

*8月21日 14:05 · 官方合并 · 尚未随版本发布*

ExecuTorch 合并了一组原生后端终端路径：`.ptn` 将 native graph 与 tensor data 打包在同一产物中，格式保持后端无关；[delegate 可把常量交回打包器](https://github.com/pytorch/executorch/pull/22008)，最终由 [`to_native`](https://github.com/pytorch/executorch/pull/22009) 直接生成 native artifact，无需先走 ExecuTorch program emission。

这在 PTE 之外建立了更直接的后端原生产物边界，对需要自有运行时或部署封装的后端具有架构意义。当前公开说明较简略，改动刚进入主干，支持哪些后端、格式稳定性与兼容承诺仍需等待正式文档和版本发布。

### 模型 & 技术｜[PyTorch 2.14 关闭 XPU 默认融合以解除 Qwen 服务崩溃](https://github.com/pytorch/pytorch/pull/194312)

*8月22日 04:23 · 官方合并 · 进入 release/2.14 分支*

PyTorch 确认 XPU 上自动启用的 `batch_linear_lhs` 会产生非连续 split view，破坏布局敏感的下游 consumer，并令 vLLM 服务 Qwen3.5/3.8 时崩溃。2.14 发布分支现已默认关闭该融合，保留显式测试路径。

这是一项直接影响 Intel XPU 上 Qwen 推理可用性的发布阻断修正。它采用禁用特性的方式恢复稳定性，真正的 layout-safety 修复仍在后续 PR 中，因此相关融合的性能收益暂时不可默认获得。

### 模型 & 技术｜[ExecuTorch 将 channels-last 边界布局纳入方法契约](https://github.com/pytorch/executorch/pull/22027)

*8月22日 04:53 · 官方合并 · 尚未随版本发布*

ExecuTorch 新增布局边界吸收：图输入或输出处的 `permute_copy` 可被移出图，并以 channels-last 输入/输出义务写入方法契约。在公开的 36 组用例中，图内 permute 数从 156 降至 137；其中移除的 19 个 copy 有 17 个转化为调用方义务，只有调用方本就持有目标布局或边界位于 delegate 内部时才近似免费。

这把布局转换从隐含图操作变成显式调用契约，便于后端与宿主共同优化边界数据流。当前没有 in-tree pipeline 启用该 pass，收益也不能仅用图内 copy 数衡量；若调用方仍需转换，成本只是转移而非消失。

## 二、LLVM/MLIR 最新进展

### 模型 & 技术｜[IREE Generic Runtime 打通 freestanding 与 RISC-V Bazel 构建](https://github.com/iree-org/iree/pull/24820)

*8月21日 22:47 · 官方合并 · 尚未随版本发布*

IREE 的 Generic 平台不再默认依赖 POSIX socket、signal handling 和 POSIX async proactor，并补齐 freestanding 所需的平台 fallback；同一改动加入 Generic riscv64 的 Bazel 配置，修复 RISC-V bring-up 中发现的构建问题。

这把 IREE runtime 的部署边界从类 POSIX 系统继续推向裸机与受限 RISC-V 环境，对 Buddy Compiler 与 RISC-V 软件栈研究轻量运行时落地具有直接参考价值。风险是 riscv64 配置仍处于 bring-up，且这次变化解决的是可构建性，不代表具体硬件上的模型、驱动和性能已完成验证。

### 模型 & 技术｜[LLVM 补齐一组 RISC-V P 扩展 packed intrinsics 与 codegen](https://github.com/llvm/llvm-project/pull/217534)

*8月22日 01:49 · 官方合并 · 尚未随版本发布*

LLVM/Clang 合并了 RISC-V P 扩展 packed widening multiply 支持：RV32 intrinsic 写入 `riscv_packed_simd.h`，后端可把通用 widening-multiply IR 选择为规范指令或 RV64 组合序列。同一窗口还加入了 [packed multiply-high-accumulate](https://github.com/llvm/llvm-project/pull/217591) 和 [packed saturating/rounding shift](https://github.com/llvm/llvm-project/pull/217692) 的 builtin、header wrapper 与 codegen。

这些变化扩大了编译器对嵌入式定点 DSP 工作负载的直接表达面，对 RISC-V 软件栈评估 P 扩展工具链成熟度有参考价值。相关支持仍处于上游主干，覆盖的是 P 扩展的一组操作而非完整实现，且尚未随 LLVM 正式版本发布。

## 三、Triton & TileLang 技术动态

### 模型 & 技术｜[Triton 显著降低 MXFP4 权重布局转换的时间与显存峰值](https://github.com/triton-lang/triton/pull/11360)

*8月22日 01:31 · 官方合并 · 尚未随版本发布*

Triton 融合 Hopper MXFP4 bit packing/unpacking 与可选的 Blackwell shuffled-value 转换。对 128 MiB 输入，直接转换在 H100 与 GB300 上将峰值分配降低 50%—91.3%，热态耗时从 4.24 倍改善到 21.26 倍；另一项 [AMD 后端改动](https://github.com/triton-lang/triton/pull/11377)新增 MXFP4/MXFP8 `scaled_downcast`，在 CDNA4/CDNA5 满足连续布局约束时降到打包硬件指令。

这组变化覆盖 NVIDIA 权重准备效率与 AMD 低精度硬件 lowering，强化了 Triton 上游对 MX 格式跨后端支持。数字只描述布局转换而非矩阵乘或端到端加载；新的 JIT kernel 在空缓存首次调用时可能更慢，AMD 路径也受架构与 scale 布局约束。

## 四、RISC-V 核心新闻

### 工具 & 产品｜[EDACrux 用 RVFI 退休轨迹检查 RISC-V 实现一致性](https://riscv.org/blog/ferrite-engineering-edacrux/)

*8月21日 21:03 · 官方博客 · 公开测试*

RISC-V International 介绍了 Ferrite Engineering 的 EDACrux：WaveCrux、NetCrux、LintCrux 与 SimCrux 可在波形、RTL 原理图、lint 与回归视图间交叉定位。WaveCrux 的 RVFI Commit Inspector 能把退休指令及寄存器、内存和 trap 效果还原为可读轨迹，并执行六项 ISA 一致性检查；Ibex、CVA6、PicoRV32、SERV、VeeR 等暴露 RVFI 的核心可接入。

它把指令级退休状态、波形与规范一致性检查放进统一调试界面，有助于缩短 RISC-V 核心验证中的问题定位链路。当前仍是公开测试；官方承诺免费版会保留，并计划在测试结束时以 Apache 2.0 开放核心代码，但源码开放与专业版定价尚未落地。

## 五、业界重磅新闻

### 模型 & 技术｜[ONNX Runtime WebGPU PagedAttention 绕过 KV gather 与 Q 重排](https://github.com/microsoft/onnxruntime/pull/31727)

*8月21日 11:29 · 官方合并 · 尚未随版本发布*

ONNX Runtime 的 WebGPU PagedAttention 新增直接分页 decode 与融合分页 prefill，可从 paged KV cache 读取数据，并在适用路径跳过 Q 的 unpack/repack。D3D12 测试矩阵报告 decode 几何平均约 2.0 倍、等长 prefill 约 1.13 倍、变长 prefill 约 1.29 倍；decode dispatch 从 6 次降到 3 次，prefill 可从 5 次降到 2 次。

这减少了浏览器与跨平台 WebGPU 推理中的 KV 搬运、临时显存和 CPU dispatch 开销。结果来自单一开发机；融合 prefill 目前限定 FP16，并受 head size、block size 和共享内存预算约束，不满足条件时仍回退到原路径。

### 重磅｜[Anthropic 将 Mythos 5 接入 Claude Security，并投入 3500 万美元支持开源安全](https://claude.com/blog/bringing-claude-mythos-5-to-more-defenders)

*8月22日 00:05 · 官方公告 · 企业版公开测试*

Claude Enterprise 客户现可在 Claude Security 中使用 Mythos 5 扫描自有代码库，结果包含 CWE 分类、置信度、严重性与建议修复；扫描不把 Mythos 的直接访问开放到其他界面，补丁仍须人工批准。Anthropic 同时宣布价值 3500 万美元的 Defender Advantage Fund，用额度支持开源漏洞修复、自动化扫描与更广泛的防御方案，并计划扩展 Cyber Verification Program。

这把高能力网络安全模型封装为受约束的防御输出，并把资源投向开源供应链安全。Claude Security 仍是公开测试，基金提供的是模型额度而非现金，首批受助者尚未公布；能力与误报率也需要独立、真实项目验证。
