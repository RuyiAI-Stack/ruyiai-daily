# Codex 技术情报每日动态（2026-09-01）

- 调研窗口：北京时间 2026-09-01 05:49:11 至 2026-09-01 06:54:50。
- 覆盖方向：PyTorch/ExecuTorch、LLVM/MLIR、Triton/TileLang、RISC-V 软件栈，以及 AI 模型与推理基础设施。
- 信息口径：仅采用可核验的官方发布、官方论坛与已合并代码变更；时间按首次发布、实质回复或合并时间计算。

## 今日索引

- PyTorch 生态核心动态：本窗口无新增重大动态。
- LLVM/MLIR 最新进展：本窗口无新增重大动态。
- Triton & TileLang 技术动态：本窗口无新增重大动态。
- RISC-V 核心新闻：机器可读规范库补入 Sdtrig 的 `tinfo` 与 `tcontrol` CSR。
- AI 业界重磅：SGLang unified memory 扩展为面向 mamba + hybrid-SWA 模型的三子池结构。

## 一、PyTorch 生态核心动态

本窗口无新增重大动态。

## 二、LLVM/MLIR 最新进展

本窗口无新增重大动态。

## 三、Triton & TileLang 技术动态

本窗口无新增重大动态。

## 四、RISC-V 核心新闻

### 4.1 模型 & 技术｜[feat(Sdtrig)：添加 tinfo 和 tcontrol CSR](https://github.com/riscv/riscv-unified-db/pull/2552)

> 北京时间：2026-09-01 06:30:43｜来源类型：官方 GitHub PR｜事件状态：已合并

事实：riscv-unified-db 为 Sdtrig 扩展补入 `tinfo` 与 `tcontrol` 两个 CSR 的机器可读定义。

重要性：这补齐了调试触发扩展的机器可读规范对象，便于文档、验证与工具链从统一数据库生成一致产物。风险：该变更只覆盖两个 CSR 的数据库定义，不代表完整 Sdtrig 实现或硬件支持已经完成。

## 五、AI 业界重磅

### 5.1 模型 & 技术｜[feat(unified-memory)：用于 mamba + hybrid-SWA 模型的三个子池](https://github.com/sgl-project/sglang/pull/35177)

> 北京时间：2026-09-01 06:10:13｜来源类型：官方 GitHub PR｜事件状态：已合并

事实：SGLang 将 unified memory pool 从固定两个子池推广为任意数量，并为同时包含 full-attention KV、sliding-window KV 与 recurrent state 的 mamba + hybrid-SWA 模型建立共享同一缓冲区的三个子池，其中中间子池可在两侧相邻子池之间按需移动。同一连续变更组还修复四类 hybrid model 启动或正确性问题，并加入按字节预算分配、单请求可行性下限与守恒校验（关联变更 [#35154](https://github.com/sgl-project/sglang/pull/35154)、[#35158](https://github.com/sgl-project/sglang/pull/35158)）。

重要性：统一内存池由双池扩展到三池，使混合注意力与递归状态模型可以在运行时共享空闲容量，而不被启动时固定分区锁死。风险：主 PR 页面显示部分检查未通过；变更面广且涉及调度、分配与恢复路径，生产部署仍需针对具体模型和硬件复验。

## 六、总结与趋势观察

本窗口内的两项有效变化都指向可机器验证的基础设施边界：SGLang 为混合模型增加共享内存池结构与守恒检查，RISC-V unified-db 则继续把调试扩展对象沉淀为机器可读规范。

## 附录：信源说明

本期采用 SGLang 与 RISC-V unified-db 的官方 GitHub 合并记录。GitHub 合并时间用于代码事件排序；PR 描述中的能力范围只适用于对应变更，不等同于完整产品支持或跨平台验证。
