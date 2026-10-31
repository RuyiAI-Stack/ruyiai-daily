# Codex 技术情报每日动态（2026-11-01）

调研窗口：北京时间（2026-10-31 05:49:41，2026-11-01 06:00:22]。

覆盖方向：PyTorch、LLVM/MLIR、Triton & TileLang、RISC-V，以及 AI 模型、软件与产业动态。

信息口径：以原始技术文档、官方讨论、仓库变更和署名报道为依据；区分提案、合并、公开原型与正式产品，性能数字保留测试条件。

## 今日要闻

- DeepSeek提出Router与Worker分工的Super-agent架构，并公开研究原型，探索持续在线的任务协作。 [原文](https://www.ithome.com/1/020/615.htm)
- MLIR讨论限制arith扩展运算中的index类型，聚焦32/64位折叠结果不一致及替代方案。 [原文](https://discourse.llvm.org/t/rfc-disallow-index-type-in-arith-extended-ops/92674/1)
- OpenRouter发布常驻Agent教程，给出定时执行、持久化恢复、单次预算和外部操作审批的集成路径。 [原文](https://openrouter.ai/blog/tutorials/build-always-on-ai-agent/)

## 今日索引

- **PyTorch**：Vulkan能力协商与JSON基准、量化范围选择、MPS去重和低精度内核部署。
- **LLVM/MLIR**：index位宽、推测执行审计、IPO与数据访存剖析讨论，以及Python绑定和NVPTX工具链。
- **Triton & TileLang**：gfx1250描述符缓冲区入口及Intel后端assume信息保留。
- **RISC-V**：Hypervisor与Zicfilp验证、CVA6测试配置、矩阵参考模型和NekoBox网络媒体集成。
- **AI 业界**：DeepSeek长期协作Agent设计与OpenRouter常驻Agent教程。

## 一、PyTorch 生态核心动态

### 1.1 模型 & 技术｜[\[MPS\] 将 unique_consecutive 迁移到 Metal (#202743)](https://github.com/pytorch/pytorch/commit/3df239ff803e27ff43bbcc5921ce946b836414dd)

北京时间：2026-11-01 02:47:49｜来源类型：官方仓库提交｜事件状态：代码已进入默认分支。

**事实**：MPS 的 unique_consecutive(dim=...) 改用 Metal 行内核并移除 MPSGraph 路径，修复 int64 被转成 int32 后截断的问题；作者在 M5 Pro 的连续输入用例上报告 fp32 几何平均加速 2.30 倍、int64 为 2.57 倍。

**重要性**：同时改善去重算子的整数正确性和执行效率，对含大整数标识的数据处理链路有直接影响。

**风险与边界**：非连续输入仍需 contiguous 拷贝，非零维度还含转置拷贝；性能仅适用于作者给出的设备和形状。原始 ghstack PR 当前未标记合并。

关联原文：[\[MPS\] Migrate unique_consecutive to Metal](https://github.com/pytorch/pytorch/pull/202743)。

### 1.2 模型 & 技术｜[\[ET-VK\]\[runtime\] 推导 SPIR-V 要求并启用 Vulkan 着色器特性](https://github.com/pytorch/executorch/pull/24340)

北京时间：2026-11-01 00:41:45｜来源类型：官方 PR｜事件状态：已合并。

**事实**：Vulkan 后端改为读取已编译 SPIR-V 的 OpCapability，并将要求传递到代码生成、逻辑设备配置和运行时检查；不支持或无法识别的能力会在驱动创建管线前被拒绝。

**重要性**：这使端侧模型部署的能力协商与实际着色器保持一致，减少从 GLSL 扩展文本推断所遗漏的设备要求。

**风险与边界**：这是主干运行时变化；能力映射仍须对应具体 GPU，不能推导为所有 Vulkan 设备都支持新算子。

关联原文：[\[ET-VK\]\[runtime\] Derive SPIR-V requirements and enable Vulkan shader features](https://github.com/pytorch/executorch/pull/24387)。

### 1.3 模型 & 技术｜[HistogramObserver：添加可选的 consider_full_range，以将完整范围与贪心搜索结果一起评分 (#5005)](https://github.com/pytorch/ao/pull/5005)

北京时间：2026-10-31 17:40:15｜来源类型：官方 PR｜事件状态：已合并。

**事实**：HistogramObserver 新增默认关闭的 consider_full_range：保留原有贪心搜索，再将其结果与完整观察范围比较，只有完整范围的估计误差更低时才选用它。

**重要性**：避免稀疏远尾在首次裁剪时必然被丢弃，同时给 PTQ 调用方保留逐模型验证后启用的迁移路径。

**风险与边界**：默认数值保持既有行为；开启后仍可能改变量化参数和端到端精度，作者明确要求按调用点核对。

### 1.4 模型 & 技术｜[修复使用 CuteDSL 4.8 时分组 NVFP4 内核的编译](https://github.com/pytorch/ao/pull/5042)

北京时间：2026-10-31 17:18:57｜来源类型：官方 PR｜事件状态：已合并。

**事实**：分组 NVFP4 内核的 clc_response 原以 Int32 数组声明，但底层产生 128 位共享内存读取；变更显式要求 16 字节对齐，修复 CuteDSL 4.8 分配布局变化引发的编译失败。

**重要性**：恢复 MoE 训练原型内核在该依赖版本下的构建路径，并暴露调度器抽象的隐含对齐契约。

**风险与边界**：修复覆盖该共享存储对象的布局要求，不代表整个 NVFP4 训练栈完成稳定性或性能认证。

### 1.5 模型 & 技术｜[fix(kernels)：从 cuTile PTX 中移除 nv_intermediate_source_section](https://github.com/pytorch/TensorRT/pull/4956)

北京时间：2026-10-31 09:15:01｜来源类型：官方 PR｜事件状态：已合并。

**事实**：Torch-TensorRT 在整理 cuTile PTX 参数后，移除 tileiras 附加的 .nv_intermediate_source_section；作者在 RTX 5090、615.71.10 驱动及 CUDA 13.4 环境报告该节会导致 CUDA_ERROR_INVALID_PTX。

**重要性**：直接涉及 cuTile 内核能否被驱动加载，是编译产物到推理执行之间的部署阻塞。

**风险与边界**：作者报告的设备组合有限；该修复不会替代驱动、PTX ISA 与工具链版本兼容性检查。

### 1.6 工具 & 产品｜[\[ET-VK\]\[benchmark\] 添加 JSON 算子和着色器基准用例](https://github.com/pytorch/executorch/pull/24344)

北京时间：2026-11-01 00:42:21｜来源类型：官方 PR｜事件状态：已合并。

**事实**：ExecuTorch 新增 etvk_ops_bench：用 JSON 描述输入、数据类型、存储方式及 shader/workgroup 组合测试；迁移 64 组、4,560 个用例定义，提供独立 CPU 参照、批次 GPU 计时和 Android GPU 时钟快照。

**重要性**：把端侧 GPU 调优从手写 C++ 用例推进到可重复的配置与验证流程；相关运行时接口支持任意显式工作组尺寸、原始时间戳和 SDPA shader 覆盖。

**风险与边界**：这是新的基准与调优能力，不能把用例数当作模型覆盖率；CI 工作流集成仍计划另行完成。

关联原文：[\[ET-VK\]\[runtime\] Support arbitrary explicit local workgroup sizes](https://github.com/pytorch/executorch/pull/24341)；[\[ET-VK\]\[runtime\] Expose raw shader timestamps and dispatch arguments](https://github.com/pytorch/executorch/pull/24342)；[\[ET-VK\]\[sdpa\] Expose shader override](https://github.com/pytorch/executorch/pull/24343)。


## 二、LLVM/MLIR 最新进展

### 2.1 深度洞见｜[\[RFC\] 对文档中含有 UB 的 Pure / AlwaysSpeculatable 运算进行全树审计（以及可能的 TableGen lint？）](https://discourse.llvm.org/t/rfc-tree-wide-audit-of-pure-alwaysspeculatable-ops-with-documented-ub-and-maybe-a-tablegen-lint/92649/10)

北京时间：2026-11-01 01:00:55｜来源类型：官方技术论坛｜事件状态：讨论新进展。

**事实**：提案原本希望排查带 UB、陷阱或区域副作用的 Pure/AlwaysSpeculatable 运算，避免 LICM 将其移出保护条件。窗口内作者承认“56个候选”仅为正则原始输出，计划改用解析后的 TableGen records；最新回复放弃构建期文档文字匹配，转向结构不变量或可选 trait 检查。

**重要性**：为下游编译器区分无内存副作用、可安全推测执行和未定义值提供更可靠的审计方向。

**风险与边界**：56不能当作已证实缺陷数；审计重跑及检查机制仍处于计划与讨论阶段。

关联原文：[\[RFC\] Tree-wide audit of Pure / AlwaysSpeculatable ops with documented UB (and maybe a TableGen lint?)](https://discourse.llvm.org/t/rfc-tree-wide-audit-of-pure-alwaysspeculatable-ops-with-documented-ub-and-maybe-a-tablegen-lint/92649/8)；[\[RFC\] Tree-wide audit of Pure / AlwaysSpeculatable ops with documented UB (and maybe a TableGen lint?)](https://discourse.llvm.org/t/rfc-tree-wide-audit-of-pure-alwaysspeculatable-ops-with-documented-ub-and-maybe-a-tablegen-lint/92649/9)。

### 2.2 深度洞见｜[\[RFC\] 优化 linkonce_odr 函数（前端提示、内联和内部克隆）](https://discourse.llvm.org/t/rfc-optimizing-linkonce-odr-functions-frontend-hints-inlining-and-internal-cloning/92423/16)

北京时间：2026-10-31 21:41:33｜来源类型：官方技术论坛｜事件状态：讨论新进展。

**事实**：该提案通过前端“可能TU局部”提示、内联奖励及克隆后internalize，让既有IPO优化适用于部分 linkonce_odr 函数。窗口内作者认可按实际优化机会进行“即时克隆”，说明草案已有 always-clone 模式用于比较提示以外的机会，并计划扩大代码库和基准实验。

**重要性**：把代码体积控制与优化收益判断拉回具体调用点，可用于评估C++模板及模型运行时编译的优化空间。

**风险与边界**：新增的是探索方向和实验机制说明；更大基准尚待执行，不能将早先单一基准的收益外推。

### 2.3 深度洞见｜[\[RFC\] 禁止在 arith 扩展运算中使用 index 类型](https://discourse.llvm.org/t/rfc-disallow-index-type-in-arith-extended-ops/92674/1)

北京时间：2026-10-31 21:31:50｜来源类型：官方技术论坛｜事件状态：RFC 已提出。

**事实**：提案建议将 addui_extended、subui_extended、mulsi_extended、mului_extended 限制为固定宽度整数：index 未确定目标位宽，按 64 位折叠与先降为 32 位可能产生不同高半部结果。当天回复进一步给出 maxsi、divui、shrui、cmpi 和 index_castui 的位宽反例，并讨论是否统一迁往 index dialect。

**重要性**：这关系到 MLIR/Linalg 等流水线的常量折叠与目标布局一致性，对模型编译和非 LLVM 后端都存在语义影响。

**风险与边界**：仍是 RFC；保留 index 并只做32/64位一致折叠也是备选方案，没有宣布统一禁用。

关联原文：[\[RFC\] Disallow index type in arith extended ops](https://discourse.llvm.org/t/rfc-disallow-index-type-in-arith-extended-ops/92674/2)。

### 2.4 深度洞见｜[\[RFC\] 为 llvm-profgen 添加 ITM 数据跟踪支持](https://discourse.llvm.org/t/rfc-add-itm-data-trace-support-to-llvm-profgen/92465/18)

北京时间：2026-10-31 19:54:48｜来源类型：官方技术论坛｜事件状态：讨论新进展。

**事实**：提案将Arm ITM访存跟踪与ETM指令跟踪一起解码，按ELF静态符号汇总访问次数，供LLD把热静态数据放入快存储层。窗口回复支持在维护复杂度过高时舍弃独立的instrumentation格式，优先减轻双格式负担，并提出合入后探索AMD IBS及Intel PEBS输入。

**重要性**：嵌入式分层存储布局可从实际数据访问获得剖析依据，也明确了后续跨硬件采样输入的接口方向。

**风险与边界**：格式选择仍在讨论，IBS/PEBS属于未来意向，不能写成已支持或已合入。

### 2.5 模型 & 技术｜[\[compiler-rt\]\[profile\] 汇总随通道变化的 GPU 计数器增量](https://github.com/llvm/llvm-project/pull/235978)

北京时间：2026-11-01 05:43:59｜来源类型：官方 PR｜事件状态：已合并。

**事实**：GPU profiling helper改用ballot统计step=1的活动lane，再由首个活动lane累加；旧实现把首lane的step乘以活动lane数，对64-lane中一半条件为真的select会记64而非32。

**重要性**：修正条件相关GPU剖析数据的系统性偏差，避免优化决策建立在错误计数上。

**风险与边界**：编译器生成的step限定为0或1；这是计数语义修复，没有给出应用性能收益。

### 2.6 模型 & 技术｜[\[ELF\] 使用 --lto-emit-asm 和 --lto-emit-llvm 支持 NVPTX 位码](https://github.com/llvm/llvm-project/pull/200679)

北京时间：2026-11-01 04:55:31｜来源类型：官方 PR｜事件状态：已合并。

**事实**：LLD将nvptx/nvptx64位码映射到仅供内部LTO管线使用的EM_CUDA，并只允许以 --lto-emit-asm 或 --lto-emit-llvm 输出，支持替代自定义 llvm-link/opt/llc 链路。

**重要性**：为Rust等前端把GPU位码链接交给标准LLD提供路径，改变编译器到PTX的工具链集成方式。

**风险与边界**：CUDA驱动才会引用的kernel仍须显式导出，否则LTO可能删除；有效PTX还依赖配套去除 .addrsig 的变更。

### 2.7 模型 & 技术｜[\[MLIR\]\[Python\] 在长时间运行的操作期间释放 GIL。](https://github.com/llvm/llvm-project/pull/236861)

北京时间：2026-11-01 01:37:14｜来源类型：官方 PR｜事件状态：已合并。

**事实**：MLIR Python绑定在运行PassManager和创建ExecutionEngine等长操作期间释放GIL，并在可能回调Python的路径重新获取锁。

**重要性**：有助于Python驱动的模型导入、编译与执行工具维持并发响应，对Buddy和RuyiAI常用编译链具有接口层价值。

**风险与边界**：释放GIL不自动保证所有外部回调、共享IR或用户对象线程安全，也没有量化编译加速承诺。


## 三、Triton & TileLang 技术动态

### 3.1 模型 & 技术｜[\[AMD\]\[Gluon\] 为 gfx1250 添加描述符缓冲区运算](https://github.com/triton-lang/triton/pull/12512)

北京时间：2026-10-31 08:46:48｜来源类型：官方 PR｜事件状态：已合并。

**事实**：Gluon通过 ttgl.amd.gfx1250.buffer 暴露make_buffer_descriptor、load和store，并提供soffset、scope与gfx1250 CachePolicy；验证器限制不合法的last_use用法。

**重要性**：新增硬件后端的显式内存操作入口，便于kernel作者控制缓存策略与设备/系统作用域。

**风险与边界**：gfx1250没有buffer到LDS的load_to_shared；num_records仍为32位，靠近边界的向量访问还受4字节分片行为约束。

### 3.2 模型 & 技术｜[\[FoldTrueCmpI\] 保留传给 `llvm.intr.assume` 的比较](https://github.com/intel/intel-xpu-backend-for-triton/pull/8579)

北京时间：2026-10-31 08:03:57｜来源类型：官方 PR｜事件状态：已合并。

**事实**：范围分析曾利用assume本身把其条件证明为true，FoldTrueCmpI随后将所有tl.assume条件折叠，导致后续make_ttir pass丢失假设；修复保留直接供llvm.intr.assume使用的比较。

**重要性**：恢复kernel作者传入的范围信息在后续编译阶段的有效性，关系到指针连续性等优化所需的语义契约。

**风险与边界**：仅修复假设信息被消除的路径；从假设推导出的其他比较仍可折叠，没有承诺普遍吞吐提升。


## 四、RISC-V 核心新闻

### 4.1 工具 & 产品｜[nekobox：添加支持渐进式 MP4 播放的 DLNA 投屏](https://github.com/YuzukiHD/zephyr-components/commit/d9f22baf0e9e11dc17bf557ee5e2c9ece197607d)

北京时间：2026-10-31 07:44:09｜来源类型：官方仓库提交｜事件状态：新集成代码已公开。

**事实**：NekoBox新增Zephyr UPnP AV MediaRenderer和DLNA接收界面，可从手机或PC接收H.264/AAC MP4；写入TF卡后按默认512 KiB缓冲逐步播放，支持chunked或无Content-Length的HTTP响应。

**重要性**：为YuzukiNeko板端媒体系统增加可复用的网络输入、存储与播放集成路径，文档同步交代配置开关和设备限制。

**风险与边界**：投屏默认限640×360或竖屏360×640；不支持HTTPS、播放列表及fragmented MP4，moov位于末尾时仍需完整下载。这是软件集成更新。

关联原文：[dlna README](https://raw.githubusercontent.com/YuzukiHD/zephyr-components/d9f22baf/dlna/README.md)。

### 4.2 模型 & 技术｜[fix(hosttlb)：保留矩阵寄存器选择器参数](https://github.com/OpenXiangShan/NEMU/pull/1274)

北京时间：2026-10-31 21:13:13｜来源类型：官方 PR｜事件状态：已合并。

**事实**：矩阵访存实现以char m_name区分A/B/C寄存器，但公开头文件仍声明bool isacc，调用经头文件先把选择器压为true；声明改成char，并让实现包含公开头文件以暴露未来签名不一致。

**重要性**：修正参考模拟器矩阵寄存器选择的实际语义，避免差分验证使用错误操作数。

**风险与边界**：范围是host-TLB矩阵访问接口，不代表新增矩阵ISA或硬件支持。

### 4.3 模型 & 技术｜[perf(mma)：简化有符号 INT8 参考归约](https://github.com/OpenXiangShan/difftest/pull/1019)

北京时间：2026-10-31 20:37:42｜来源类型：官方 PR｜事件状态：已合并。

**事实**：通过符号位偏置表达有符号字节解码，并去掉每4乘积组内冗余28位wrap，保留32位半归约与饱和边界；作者在Ryzen 9 7950X3D单核、batch=512完整tile验证中报告I8I8I32吞吐提升2.67至2.71倍。

**重要性**：降低矩阵差分验证的CPU参考成本，有利于扩大高强度回归的验证吞吐。

**风险与边界**：该数字限参考模型与指定配置；短时FP16/FP8及CUDA对照不构成GPU或端到端系统加速证明。

### 4.4 模型 & 技术｜[perf(mma)：流水化 CUDA 传输、计算和结果检查](https://github.com/OpenXiangShan/difftest/pull/1018)

北京时间：2026-10-31 18:12:47｜来源类型：官方 PR｜事件状态：已合并。

**事实**：MMA CUDA参考验证将32请求范围分配至4条非阻塞stream，使用pinned描述符，默认批量上限提高至512；每完成128请求发布结果前缀，让CPU比对与后续GPU工作重叠。相关LightSSS更新把CUDA参考计算放入独立持久进程。

**重要性**：同时处理验证吞吐与快照恢复的CUDA上下文生命周期，直接服务香山矩阵单元仿真和错误重放。

**风险与边界**：作者结果衡量参考模型而非整机速度；LightSSS部署需将difftest-mma-worker与模拟器一同交付。

关联原文：[fix(mma): support LightSSS with the CUDA backend](https://github.com/OpenXiangShan/difftest/pull/1017)。

### 4.5 模型 & 技术｜[Sv：用于两阶段地址转换的 H、HSm、HZicbo 和 HZicboSm 组](https://github.com/riscv/riscv-arch-test/pull/3004)

北京时间：2026-10-31 08:56:23｜来源类型：官方 PR｜事件状态：已合并。

**事实**：ACT为Sv套件增加Hypervisor两阶段地址转换、MPRV/MPV、缓存块操作及PMP等测试组，并扩展Svadu、Svpbmt、Svnapot的H组和规则映射。相关Sail变更把MXR限定于显式访问，修复隐式VS页表读取的G-stage权限语义。

**重要性**：从参考模型语义到架构覆盖组一起推进RISC-V虚拟化验证，属于新增验证能力。

**风险与边界**：ACT正文仍标记cp_implicit_mxr生成器关闭；Sail修复已合入并不证明ACT覆盖点已重新启用或全套实现通过。

关联原文：[Apply MXR only to explicit memory accesses](https://github.com/riscv/sail-riscv/pull/2066)。

### 4.6 模型 & 技术｜[添加 OpenHW CVA6 cv32a65x 和 cv32a60x ACT 配置](https://github.com/riscv/riscv-arch-test/pull/2845)

北京时间：2026-10-31 07:36:17｜来源类型：官方 PR｜事件状态：已合并。

**事实**：ACT新增CVA6双发射cv32a65x与单发射cv32a60x的Verilator testharness配置，包含UDB、Sail 0.16、链接脚本及统一构建运行入口。

**重要性**：为两种CVA6实现接入同一架构测试体系提供具体可运行配置，拓展验证对象。

**风险与边界**：cv32a65x RTL虽有8个TOR-only PMP条目，配置当前仍声明0，等待后续支持；新配置不等同认证完成。

### 4.7 模型 & 技术｜[Zicfilp：用于 VS-mode 和 VU-mode 落点的 H 组](https://github.com/riscv/riscv-arch-test/pull/3002)

北京时间：2026-10-31 07:07:09｜来源类型：官方 PR｜事件状态：已合并。

**事实**：Zicfilp新增H和SmH测试，覆盖VS/VU环境的LPE控制、ELP保存与恢复，以及异常进入HS/M模式时的状态选择；加入对应covergroup、规则映射和测试计划。

**重要性**：把控制流完整性扩展推进到虚拟化权限交互的可执行验证路径。

**风险与边界**：需要H与Zicfilp等配置；QEMU当前排除相关组，不能据此声称模拟器或芯片已通过认证。


## 五、AI 业界重磅

### 5.1 模型 & 技术｜[DeepSeek 提出 Super-agent 架构设计：Router 与 Worker 双层架构，探索更自主的 Agent 系统](https://www.ithome.com/1/020/615.htm)

北京时间：2026-10-31 21:36:35｜来源类型：科技媒体原始报道｜事件状态：研究原型已公开。

**事实**：DeepSeek提出Super-agent设计：Router负责用户交互、全局上下文和任务分派，Worker在隔离会话中并行执行；事件过滤、线程订阅、状态持久化及外部Watchdog共同支撑长期运行。团队同时公开dsh-superagent-prototype，可面向GitHub仓库并可选接入飞书。

**重要性**：将长期协作Agent的任务接收、授权、执行与人类介入变成可查看的系统边界，便于研究持续在线的开发协作方式。

**风险与边界**：架构中的部分方向尚未在公开原型实现；原型不是正式DeepSeek产品。异步任务机制尚未默认启用，合成负载实验也不衡量真实编码质量。

关联原文：[dsh-superagent](https://github.com/deepseek-ai/dsh-superagent-prototype)。

### 5.2 工具 & 产品｜[如何构建你自己的常驻 AI Agent](https://openrouter.ai/blog/tutorials/build-always-on-ai-agent/)

北京时间：2026-10-31 08:00:00｜来源类型：官方博客｜事件状态：教程已发布。

**事实**：OpenRouter发布基于Agent SDK 0.11.0的TypeScript常驻Agent教程：定时检查Hacker News、保存已见条目及会话状态、崩溃后恢复，并以6步和每轮0.05美元限制执行；对外发消息的工具须批准后才运行。

**重要性**：给出从调度、持久化到成本和外部操作授权的完整集成路径，可供仓库和状态监控类Agent复用。

**风险与边界**：这是教程示例；调度及文件存储由应用负责，崩溃发生在摘要写入后仍可能产生重复条目，不能把演示视作托管服务保证。

## 六、总结与趋势观察

- **编译语义需要与目标和优化阶段保持一致。** [index位宽讨论](https://discourse.llvm.org/t/rfc-disallow-index-type-in-arith-extended-ops/92674/1)和[Pure审计讨论](https://discourse.llvm.org/t/rfc-tree-wide-audit-of-pure-alwaysspeculatable-ops-with-documented-ub-and-maybe-a-tablegen-lint/92649/10)分别暴露折叠规则与推测执行契约的边界；[assume信息保留](https://github.com/intel/intel-xpu-backend-for-triton/pull/8579)则说明即便假设本身正确，转换顺序也可能使后续优化失去依据。

- **RISC-V验证同时扩展架构覆盖和参考执行能力。** [两阶段地址转换测试](https://github.com/riscv/riscv-arch-test/pull/3004)与[虚拟化落点测试](https://github.com/riscv/riscv-arch-test/pull/3002)增加权限交互覆盖；[CUDA验证流水化](https://github.com/OpenXiangShan/difftest/pull/1018)处理参考计算与快照恢复的并发边界。这些进展说明验证路径更具体，仍不等同硬件认证完成。

- **持续在线Agent的公开实践开始明确调度、状态与授权边界。** [DeepSeek原型设计](https://www.ithome.com/1/020/615.htm)采用隔离Worker与外部监督，[OpenRouter教程](https://openrouter.ai/blog/tutorials/build-always-on-ai-agent/)展示持久化、预算与工具审批；两者都保留实现和运行可靠性的具体限制。

## 附录：信源说明

本期采用PyTorch/ExecuTorch、TorchAO、Torch-TensorRT、LLVM/MLIR、Triton、Intel XPU Triton后端、RISC-V架构测试与Sail、香山NEMU/DiffTest、YuzukiHD项目的官方技术材料，以及OpenRouter博客和IT之家署名报道。技术论坛以具体主题或回复为依据，仓库内容区分实现合并与代码提交。

提案和研究原型的描述不代表正式产品承诺；作者提供的性能数据适用于其设备、输入、精度与测量范围。测试配置、覆盖组或参考模型更新不构成芯片认证结论。
