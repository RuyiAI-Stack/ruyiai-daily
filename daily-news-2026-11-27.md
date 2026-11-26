# Codex 技术情报每日动态（2026-11-27）

调研窗口：北京时间 2026-11-26 05:49:07 至 2026-11-27 06:00:30（左开右闭）。
覆盖方向：PyTorch、LLVM/MLIR、Triton & TileLang、RISC-V，以及 AI 模型研究、软件基础设施与芯片推理系统。
信息口径：采用原始公告、正式发行、技术讨论与上游代码；性能数字保留测试条件，社区部署报告与厂商结论均注明边界。

## 今日要闻

- [腾讯混元公开 DFly2：针对 DSA 稀疏注意力目标模型，复用目标层 KV cache 与至多 2048 个索引位置，避免另建 draft KV cache；适配 Hy4/Hy4 flash 的权重与 AngelSpec 训练代码已开放。](https://mp.weixin.qq.com/s/d1HMHrSEChCJiBRZ0tNOyg)
- [llama.cpp v1.1.0 正式发行，纳入 Inkling 2 家族、Hy4 flash 等模型，以及多种语音与音频模型；DFlash 草稿方法拆分成独立选项。](https://github.com/ggml-org/llama.cpp/releases/tag/v1.1.0)
- [NVIDIA 在此前 Parse 3.5 模型基础上公开三套训练数据：Synthetic Charts V1 约 998 万例、Synthetic Tables V1 100 万例、OCR Synthetic Multilingual v2 790 万例，并提供 SFT/RLVR 微调配方。](https://huggingface.co/blog/nvidia/building-on-nemotron-35-parse-open-data-fine-tunin)

## 今日索引

- **PyTorch**：RuyiAI 专题进入中国日日程；XPU GEMM、NVGEMM 动态形状、DLPack 与 autocast 路径更新。
- **LLVM/MLIR**：ValueBounds 分支边界反例、CVE 检查提案、Buddy XSAIAME 方言，以及模型导入、NEON 微内核和 RISC-V 主机特征修正。
- **Triton & TileLang**：Triton-RISC-V 经 VIR 统一降低；Ascend 同步分析与主机映射、Intel 块访存与 launch 开销更新。
- **RISC-V**：K3 超声识别部署、D560 恢复链路、P550 频率设置，以及 PMP 与 GEM5 存储语义修正。
- **AI 业界**：DFly2、Parse 训练材料、多智能体实验、llama.cpp v1.1.0 和 AMD 基准仪表盘。

## 一、PyTorch 生态核心动态

### 1.1 工具 & 产品｜[Explore the PyTorch Day China 2026 Schedule（查看 PyTorch Day China 2026 日程）](https://pytorch.org/blog/explore-the-pytorch-day-china-2026-schedule/)

北京时间：2026-11-26 10:44:16｜来源类型：官方博客｜事件状态：活动日程已公布

PyTorch Foundation 公布 12 月 12 日北京活动日程，由阿里云、AMD 与基金会共同举办。专题包括“PyTorch on RISC-V: Upstream Support and Deployment with RuyiAI”，讲者为中科院软件所 Hongbin Zhang、Yixuan Chen。
同批公布 FlyDSL 加速 AMD GPU、Mooncake 智能体数据基础设施、跨 NVIDIA/AMD 位级一致强化学习训练等议题。

**重要性**：提供了团队相关的 RuyiAI 上游支持与部署交流入口，也把异构训练和推理链路放在同一日程中。

**限制与风险**：这是议程公告；演讲尚未发生，不能据此认定相关技术或性能结果已发布。

### 1.2 模型 & 技术｜[\[Inductor\]\[XPU\] Add persistent and non-persistent Triton mm templates for BF16/FP16 (#200893)（\[Inductor\]\[XPU\] 为 BF16/FP16 添加持久化与非持久化 Triton mm 模板 (#200893)）](https://github.com/pytorch/pytorch/commit/de52ac8b8b297a3d49941c8d58189ca1c46b9e16)

北京时间：2026-11-26 17:25:29｜来源类型：上游主干提交｜事件状态：已进入主干

Inductor 为 Xe2 及以后 XPU 的 BF16/FP16 矩阵乘增加两组候选：持久化模板面向大 M prefill，非持久化模板面向小 M decode，并保留 triton_mm 回退。
作者在 Arc Pro B60 的 101 个 LLM GEMM 形状上测得 Triton 路径几何平均提高 1.20 倍，包含 oneDNN 回退的系统比较提高 1.066 倍；胜过 oneDNN 的形状从 45 个增至 71 个。

**重要性**：扩充 PyTorch 在 Intel GPU 上针对训练与推理不同尺寸的自动调优空间。

**限制与风险**：测量限定硬件、布局与形状；候选数量增加也会增加自动调优成本。

### 1.3 模型 & 技术｜[\[NVGEMM\] Restore dense GEMM dynamic shapes with guarded kernel requirements (#206536) (#206536)（\[NVGEMM\] 通过带守卫的内核要求恢复稠密 GEMM 动态形状 (#206536) (#206536)）](https://github.com/pytorch/pytorch/commit/b7c9761dd541cfabe6f24596d603422a35754e01)

北京时间：2026-11-27 04:45:30｜来源类型：上游主干提交｜事件状态：已进入主干

NVGEMM 将布局、视图偏移对齐、尺寸跨度和 launch 网格条件统一为符号要求：候选阶段抑制守卫，只给自动调优胜出的内核安装守卫。
满足静态证明时允许 mm 使用 unbacked 形状；为 addmm/bmm 的错位视图与 CLC 网格越界增加回退，并限制依赖形状的 epilogue 融合。

**重要性**：恢复 Blackwell 稠密 GEMM 的动态形状路径，同时把性能启发式与正确性证明分开。

**限制与风险**：不是任意动态形状均能使用 NVGEMM；无法证明的约束仍回退，不能从编译通过推断所有工作负载提速。

### 1.4 模型 & 技术｜[Add PrivateUse1 hook to customize DLPack device metadata (#206675)（添加用于自定义 DLPack 设备元数据的 PrivateUse1 钩子 (#206675)）](https://github.com/pytorch/pytorch/commit/5f228673c58f6d36691c0cbc5c677fdf0c349372)

北京时间：2026-11-27 05:44:21｜来源类型：上游主干提交｜事件状态：已进入主干

PrivateUse1 后端可以通过 getDLDeviceType 钩子返回 DLPack 设备类型，并在导入时匹配已注册类型。
`__dlpack_device__` 改用实际注册的后端名，解决重命名设备后仍硬编码 privateuse1 的互操作失败。

**重要性**：改善外部加速器后端与 DLPack 张量交换的集成路径。

**限制与风险**：类型受到允许集合与注册检查约束；这不代表新增某款硬件的完整 PyTorch 支持。

### 1.5 模型 & 技术｜[fix(autocast): model torch.autocast regions: cast for writes into region outputs, and reclassify_all（fix(autocast): 对 torch.autocast 区域建模：为写入区域输出进行转换，并添加 reclassify_all）](https://github.com/pytorch/TensorRT/pull/5012)

北京时间：2026-11-26 06:34:59｜来源类型：上游 PR｜事件状态：已进入主干

Torch-TensorRT 在重分类模型 autocast 区域时追踪区域间写入，使后续区域读到经过写入的转换结果；补充视图别名和导出再加载的处理。
新增显式选择的 reclassify_all，可重分类 enabled=False 区域；默认 keep 保持原有策略并增加日志。

**重要性**：修复跨混合精度区域的原地写入丢失问题，直接影响导出模型数值语义。

**限制与风险**：写入操作或其前置视图被转换到其他 dtype 时仍有已知限制；不是所有别名写入都已覆盖。

## 二、LLVM/MLIR 最新进展

### 2.1 模型 & 技术｜[\[RFC\] Value Bounds need deferred conditional bounds（\[RFC\] Value Bounds 需要延迟条件边界）](https://discourse.llvm.org/t/92802/4)

北京时间：2026-11-26 22:22:31｜来源类型：技术社区｜事件状态：讨论新进展

提案让 ValueBoundsOpInterface 先登记带条件的约束，在普通工作队列排空后再求证条件并传播新边界，避免算术边界随操作数遍历顺序变化。
讨论新进展：krsch-mobileye 给出 scf.if 反例，当前分析混入两个分支后会把实际 abs(x) 的结果推导为 0；作者倾向不重复分析 IR，scf.if 的延迟分析 API 尚待设计。

**重要性**：这关系到 MLIR/Linalg 优化使用的形状与数值边界是否可靠，而非单纯求解器速度问题。

**限制与风险**：方案仍在评审；维护者提出无条件路径开销、嵌套合流耗时和是否可选启用等问题，未宣布修复落地。

### 2.2 深度洞见｜[\[RFC\] CVE coverage campaign（\[RFC\] CVE 覆盖行动）](https://discourse.llvm.org/t/92983/1)

北京时间：2026-11-27 01:34:20｜来源类型：技术社区｜事件状态：提案讨论中

bearbones 发起从当年 CVE 中寻找 clang-tidy 检查或编译器警告机会的提案，明确候选经过人工筛查。
首个案例是逻辑或与三元表达式优先级混淆导致的缓冲区问题，建议扩展 readability-math-missing-parentheses，或增加逻辑括号警告与 bugprone 检查。

**重要性**：把真实漏洞模式转成静态检测覆盖候选，为安全回归与代码检查提供具体切入点。

**限制与风险**：目前是跟踪提案，没有新增检查正式发布，也未证明能检测全部同类漏洞。

### 2.3 模型 & 技术｜[\[midend\]\[XSAIAME\] Add the XSAIAME dialect and its Python bindings（\[midend\]\[XSAIAME\] 添加 XSAIAME 方言及其 Python 绑定）](https://github.com/buddy-compiler/buddy-mlir/pull/1071)

北京时间：2026-11-26 17:18:32｜来源类型：上游 PR｜事件状态：已进入主干

Buddy 新增 xsai_ame 方言，以带类型的 SSA tile 和 sync 句柄描述 XSAI ZAMES 矩阵与 ZMASYNC 同步扩展，提供 load/store、mma、mzero 及同步操作。
方言注册进 buddy-opt/InitAll，并提供 C API、ODS Python builder 与 buddy_mlir.dialects.xsai_ame；动态 tile 维度显式携带 i64 尺寸，内存同步操作声明副作用。

**重要性**：为团队 RISC-V 矩阵扩展编译链路建立可验证的中间表示与 Python 构建接口。

**限制与风险**：本次只包含 IR 与绑定；到 llvm.riscv.xsai.* intrinsic 的降低及 Linalg 转换留待后续。元素类型组合由硬件后端决定，15 种元素 token 不代表所有硬件均支持全部组合。

### 2.4 模型 & 技术｜[\[TorchOnnxToTorch\] Split DepthToSpace CRD channels as (C, b, b)（\[TorchOnnxToTorch\] 将 DepthToSpace CRD 通道拆分为 (C, b, b)）](https://github.com/llvm/torch-mlir/pull/4927)

北京时间：2026-11-27 00:37:35｜来源类型：上游 PR｜事件状态：已进入主干

ONNX DepthToSpace 的 CRD 模式现在使用 [N,C,b,b,H,W] 重塑顺序，DCR 仍采用原有顺序。
原实现让 CRD 模型输出像素错位；新回归使用不同拆分尺寸，纠正此前测试也采用 DCR 布局的问题。

**重要性**：保护 ONNX 到 Torch/MLIR 的模型导入语义，尤其是包含通道到空间重排的视觉模型。

**限制与风险**：该补丁针对 CRD 降低，不构成新的全模型精度或性能保证。

### 2.5 模型 & 技术｜[\[Runtime\] Add NEON conv_nchwc ukernel tile for arm_64（\[Runtime\] 为 arm_64 添加 NEON conv_nchwc 微内核分块）](https://github.com/iree-org/iree/pull/25290)

北京时间：2026-11-26 22:27:48｜来源类型：上游 PR｜事件状态：已进入主干

IREE 为 AArch64 f32 分块卷积增加 OW_TILE=8、K0=8、C0=8 的 NEON 实现，替代该形状此前的通用回退。
内核使用 16 个累加器、每组 4 个输入通道及 vfmaq_laneq_f32，并抑制内层展开以避免寄存器溢出。

**重要性**：补齐编译器已选择的数据布局与运行时微内核之间的专用执行路径。

**限制与风险**：仅支持当前 f32 与分块形状；没有提供跨 CPU 的量化加速结果，ABI 和通用回退保留。

### 2.6 模型 & 技术｜[\[RISCV\] Clear Zvl*b in getHostCPUFeatures when vector is unavailable（\[RISCV\] 当向量不可用时清除 getHostCPUFeatures 中的 Zvl*b）](https://github.com/llvm/llvm-project/pull/244253)

北京时间：2026-11-27 02:50:43｜来源类型：上游 PR｜事件状态：已进入主干

当 Linux hwprobe 表明 V/Zve32x 不可用时，LLVM 显式关闭 zvl32b，连带关闭更大 Zvl 特征，避免 CPU 型号隐含的最小向量长度残留。
针对内核禁用 V 后 JIT 或 -mcpu=native 出现“zvl*b requires v or zve”编译失败的情形；作者用 qemu-user 复现并验证修复。

**重要性**：改善 RISC-V 系统运行时探测与 LLVM JIT 的兼容性，避免硬件型号特征覆盖操作系统实际开放能力。

**限制与风险**：作者尚未在真实硬件验证；这是特征组合修正，不能写成新增向量硬件能力。

## 三、Triton & TileLang 技术动态

### 3.1 模型 & 技术｜[\[backend\] Unify standard CPU lowering and matmul through VIR（\[backend\] 通过 VIR 统一标准 CPU 降低和矩阵乘）](https://github.com/RuyiAI-Stack/triton-riscv/pull/135)

北京时间：2026-11-26 17:41:01｜来源类型：上游 PR｜事件状态：已进入主干

标准 CPU 流水线将固定宽度与 RVV 矩阵乘统一经 VIR 降低，按目标向量宽度和寄存器数配置分块；增加掩码访存、归约及临时缓冲处理。
从目标特征中最大 +zvlNb 推导 RVV 宽度，支持显式覆盖；新增 target_tune，使 opt/llc 的调度与成本模型选择和 ISA 特征分离。

**重要性**：直接衔接 RuyiAI 的 Triton—Buddy/VIR—LLVM—RISC-V 部署链路，贯通标准CPU内核的向量化与矩阵乘路径。

**限制与风险**：PR 的性能表是各实验历史测量，不能当作本次组合补丁的新总加速；仍需匹配所需 pass 与选项的 Buddy 版本，PR 说明中的组合验证尚未全部完成。

### 3.2 模型 & 技术｜[\[Ascend\] Propagate dependency distances as ranges in InsertSync（\[Ascend\] 在 InsertSync 中以区间传播依赖距离）](https://github.com/tile-ai/tilelang/pull/4486)

北京时间：2026-11-26 11:32:40｜来源类型：上游 PR｜事件状态：已进入主干

InsertSync 把可达依赖距离存成不相交闭区间，并对待处理区间去重，避免按单个距离组合导致队列膨胀。
作者在 AArch64、Ascend 950DT/CANN 9.3.0、相同 GCC 13.3 与 TVM 下，以空缓存固定 CPU 核冷编译 FlashMLA：nk=topk=960 从 1078.1 秒降至 38.7 秒；去重前后产生相同 TIR。

**重要性**：使长 KV 分块下的同步分析编译成本大幅下降，缓解 Ascend 内核迭代瓶颈。

**限制与风险**：数字是指定内核的编译时间，不是 NPU 执行加速；数值验证与 186 项调度测试由作者报告。

### 3.3 模型 & 技术｜[\[Feature\]\[Ascend\] Support explicit host-mapped buffer parameters（\[Feature\]\[Ascend\] 支持显式主机映射缓冲区参数）](https://github.com/tile-ai/tilelang/pull/3909)

北京时间：2026-11-26 09:44:09｜来源类型：上游 PR｜事件状态：已进入主干

Ascend 复用 T.annotate_host_mapped_buffers 声明显式主机映射参数；Cython 与 TVM-FFI 查询 aclrtHostGetDevicePointer，传递设备可见地址并保留 dtype、形状和设备检查。
覆盖全主机参数、自动分配 NPU 输出、Graph 捕获重放与缓存恢复；映射查询失败会消费相应 ACL 错误，避免污染后续 launch。

**重要性**：提供保留张量验证的 CPU/NPU 共享存储集成路径。

**限制与风险**：注册、生命周期与同步由调用者负责；映射参数必须显式提供，PTO 尚不支持，未提出性能结论。

### 3.4 模型 & 技术｜[\[Intel\] Block loads for a GEMM operand scaled by a loaded vector（\[Intel\] 为由已加载向量缩放的 GEMM 操作数使用块加载）](https://github.com/intel/intel-xpu-backend-for-triton/pull/8938)

北京时间：2026-11-27 00:19:38｜来源类型：上游 PR｜事件状态：已进入主干

布局转换删除在 dot 操作数提升后再次处理大块 tile 的转换，让与小向量逐行缩放的 GEMM 输入直接使用 2D block load。
修复依赖此前输出 block store 支持；当缩放向量与 tile 元素位宽一致时才有作用，f32 缩放 f16 tile 的情况仍不覆盖。

**重要性**：减少矩阵乘循环中的共享内存往返，完善 Intel GPU 的高效访存链路。

**限制与风险**：未给出可泛化的端到端性能数值，且受元素宽度与既有 store 路径约束。

### 3.5 模型 & 技术｜[\[Intel\] Lower descriptor scatters to sub-group block writes（\[Intel\] 将描述符散写降低为子组块写入）](https://github.com/intel/intel-xpu-backend-for-triton/pull/8984)

北京时间：2026-11-26 23:48:34｜来源类型：上游 PR｜事件状态：已进入主干

为 ttig.descriptor_scatter 增加原生降低：满足 dot 来源、16/32 位元素与列 16 字节对齐条件时，使用 sub_group_block_write。
按 8/4/2/1 寄存器行段生成块写；跨列边界与不兼容布局保留逐 lane 掩码写，负列索引同样屏蔽。

**重要性**：补齐 tensor descriptor 非连续散写的块访问能力，避免一律展开为指针存储。

**限制与风险**：8 位元素和主机端描述符仍展开；只对满足分析条件的来源采用原生路径。

### 3.6 模型 & 技术｜[\[Intel\] Skip the scratch allocator setup in XPULauncher when no scratch is needed (#8334 slowdown seen in #8981)（\[Intel\] 不需要 scratch 时跳过 XPULauncher 的 scratch 分配器设置（#8981 中观察到的 #8334 性能下降））](https://github.com/intel/intel-xpu-backend-for-triton/pull/9013)

北京时间：2026-11-26 21:00:28｜来源类型：上游 PR｜事件状态：已进入主干

XPULauncher 在全局和 profiling scratch 均为零时跳过闭包与分配器查询，传给底层 launch 的零大小参数保持 None。
作者在 Windows B580 的 BertForQuestionAnswering、batch=2 上验证回归恢复；一台主机 compiled/eager 从约 0.717—0.722 回到 0.660—0.663。

**重要性**：减少小批量推理中每次 launch 的 Python 固定成本，具有具体回归定位和工作负载测量。

**限制与风险**：性能来自 release/3.9.1 加补丁实验；最后把判断移至构造函数的小调整未重新计时，不能视为所有模型提升。

## 四、RISC-V 核心新闻

### 4.1 工具 & 产品｜[A €50 USB mic and the K3 RISC-V board, listening for bats with BattyBirdNET（一支 50 欧元的 USB 麦克风和 K3 RISC-V 开发板，用 BattyBirdNET 聆听蝙蝠）](https://forum.spacemit.com/t/1812/1)

北京时间：2026-11-27 05:16:30｜来源类型：技术社区｜事件状态：社区部署报告

社区作者 opensolvers 公布 Banana Pi SM10（SpacemiT K3）常驻超声监听部署：384 kHz USB 音频重采样至 256 kHz，BattyBirdNET 对 0.56 秒窗口评分。
作者报告单窗约 0.15 秒，使用 fp32 ONNX Runtime CPU EP 在 X100 核运行；明确没有使用 A100 的 IME2，并公开脚本与 systemd 单元。

**重要性**：提供有输入采样率、运行时和时延边界的 RISC-V 边缘 AI 实际部署样本。

**限制与风险**：这是作者单板报告；安静房间也产生物种分数，界面用持续阈值抑制误判。模型为 CC BY-NC-SA 4.0，仓库不含权重。

### 4.2 模型 & 技术｜[eic7702: Build the D560 recovery pair and SPI update（eic7702: 构建 D560 恢复镜像对和 SPI 更新）](https://github.com/openRuyi-Project/ruyifirmware/commit/ac1ab937770b6abd814c03770de63dcc6db34466)

北京时间：2026-11-27 02:31:21｜来源类型：上游主干提交｜事件状态：已进入主干

Ruyi Firmware 为 D560 的两个 die 分别构建 USB BootROM 镜像，并生成供 Setup/Fastboot 使用的 firmware.bin；恢复步骤随包提供。
更新路径保留两个 die 的校准区和 die 0 的 UEFI 变量；关联 EDK II 变更暴露双启动 NOR，Linux 变更增加 EIC7702、SOM560 与 D560 设备描述。

**重要性**：把双芯片平台支持延伸到可操作的启动、恢复与系统描述链路。

**限制与风险**：这是开发分支代码与打包能力，尚不能等同经过完整板级验证的正式固件发行；刷写仍须遵守板卡恢复步骤。

相关变更：[eic7702: Document the D560 raw Fastboot targets](https://github.com/openRuyi-Project/ruyifirmware/commit/4475b7ba5c6d12b2fd3414720fa2fe8b56ad125f)；[EswinPlatformPkg: Expose both D560 boot NORs to Fastboot](https://github.com/openRuyi-Project/edk2-ruyi/commit/436837ce780696ba621015e4252bb48a08af88a1)；[RUYI: riscv: dts: eswin: Add the D560 board](https://github.com/openRuyi-Project/linux/commit/b75960312bb2d0cdad6b173edbe4796edce5da16)；[RUYI: riscv: dts: eswin: Describe the EIC7702 SoC](https://github.com/openRuyi-Project/linux/commit/bc0283a7c3df2b55175c86ba53526917d08bb838)；[RUYI: riscv: dts: eswin: Add the SOM560 module](https://github.com/openRuyi-Project/linux/commit/16103b4e010536a26bba262d4616d58038df8fdc)。

### 4.3 模型 & 技术｜[p550: Boot at 1.6 GHz and offer 1.8 GHz in Setup（p550: 以 1.6 GHz 启动并在 Setup 中提供 1.8 GHz）](https://github.com/openRuyi-Project/ruyifirmware/commit/942ea5af3675ef34a8674d567fd5fd00aa06d8b7)

北京时间：2026-11-27 02:31:21｜来源类型：上游主干提交｜事件状态：已进入主干

P550 固件在 DDR 训练前通过 GPIO94 选择 0.9 V，在内存自检后配置 PLL 并等锁，再切换至 1.6 GHz；锁定失败保持晶振源。
Linux 默认可选至 1.6 GHz，Setup 提供显式 1.8 GHz 选项；关联 EDK II 检查供电能力，使用新变量防止旧 1.6 GHz 授权静默选择更高频率。

**重要性**：把启动频率、Linux OPP 与固件设置对齐，给 P550 提供更明确的系统性能配置路径。

**限制与风险**：需核对 SoM 上相应 GPIO 供电器件或固定 0.9 V 改装配置；没有功耗、温度和长期稳定性测量，不能写成无条件超频保证。

相关变更：[EswinPlatformPkg: Offer a 1.8 GHz Linux CPU mode on Premier P550](https://github.com/openRuyi-Project/edk2-ruyi/commit/96f0fceba33f932d0e7642c79ccaeeba385b8cbf)；[RUYI: riscv: dts: eswin: Add the 1.6 GHz CPU OPP on HiFive Premier P550](https://github.com/openRuyi-Project/linux/commit/5a461e89ed35afa741adb7a36d0cc4c992956e8a)。

### 4.4 模型 & 技术｜[Fix PMP priority of unlocked entries in M-mode #3177（修复 M 模式下未锁定条目的 PMP 优先级 #3177）](https://github.com/openhwfoundation/cva6/pull/3767)

北京时间：2026-11-26 06:25:05｜来源类型：上游 PR｜事件状态：已进入主干

CVA6 PMP 现在由编号最低的匹配条目决定访问，未锁定条目在 M 模式下放行，不再跳过它而错误采用后面的锁定条目。
修复覆盖取指与 load/store；作者在 cv32a65x、profile_rva23s64 仿真验证原实现产生访问异常而补丁后通过。

**重要性**：纠正特权规范匹配顺序与 RTL 行为之间的偏差，影响固件在重叠 PMP 区域中的执行。

**限制与风险**：S/U 模式判定和页表遍历器路径未变；作者未完成全部 cook.py/Spike 回归，不能扩大成所有配置已验证。

### 4.5 模型 & 技术｜[mem-cache: Delay SBuffer response until block ready（mem-cache: 将 SBuffer 响应延迟到缓存块就绪）](https://github.com/OpenXiangShan/GEM5/pull/1330)

北京时间：2026-11-26 11:44:28｜来源类型：上游 PR｜事件状态：已进入主干

GEM5 在 DCache 接收 store miss 后，把非错误 SBuffer WriteResp 延迟到 CacheBlk::whenReady，避免 LSU 转发快照过早消失导致较年轻 load 读到旧值。
LSQ 总线条目增加线程归属，并在 load 提交与 squash 时清理；作者报告原失败 Soplex 配置在 20M warmup 与 20M ROI 下通过。

**重要性**：修复香山性能模拟中的存储转发与缓存可读生命周期衔接，提高验证结果可信度。

**限制与风险**：保留提前释放物理 SBuffer 容量和原缓存数据更新时间；这是模拟器语义修正，不能当作芯片 RTL 已变更。

## 五、AI 业界重磅

### 5.1 模型 & 技术｜[让 Hy4 flash 吐字提速 4.2 倍，混元 DFly2 开源](https://mp.weixin.qq.com/s/d1HMHrSEChCJiBRZ0tNOyg)

北京时间：2026-11-26 14:33:50｜来源类型：官方公告｜事件状态：已公开

腾讯混元公开 DFly2：针对 DSA 稀疏注意力目标模型，复用目标层 KV cache 与至多 2048 个索引位置，避免另建 draft KV cache；适配 Hy4/Hy4 flash 的权重与 AngelSpec 训练代码已开放。
官方报告同等训练比较中 agentlongbench 的 A12B 个例最高 4.20 倍；32K 输入、512 输出、并发 512、同样起草 4 token 时，端到端输出吞吐较 MTP 高 26%。

**重要性**：将投机解码与稀疏注意力、长上下文及 PD 分离共同设计，为部署侧减少冗余缓存和 prefill 开销提供具体路径。

**限制与风险**：最高倍率与并发吞吐来自不同测量配置，不能相乘；复用目标结构限制了直接迁移到其他模型的范围。

### 5.2 模型 & 技术｜[Building on Nemotron 3.5 Parse: Open Data, Fine-Tuning, and Faster Inference（在 Nemotron 3.5 Parse 上构建：开放数据、微调和更快推理）](https://huggingface.co/blog/nvidia/building-on-nemotron-35-parse-open-data-fine-tunin)

北京时间：2026-11-26 13:05:36｜来源类型：官方博客｜事件状态：已公开

NVIDIA 在此前 Parse 3.5 模型基础上公开三套训练数据：Synthetic Charts V1 约 998 万例、Synthetic Tables V1 100 万例、OCR Synthetic Multilingual v2 790 万例，并提供 SFT/RLVR 微调配方。
新增 DFlash2 草稿模型相关推理结果：单 H200、BF16、同一批 2078 页，512 并发从无投机的 13.48 页/秒增至 20.44 页/秒；单并发提速为 7.62 倍。

**重要性**：把文档模型从权重开放扩展到数据、训练验证与推理加速材料，便于评估复现和领域适配。

**限制与风险**：不是 Parse 3.5 模型本身今天首次发布；性能由厂商测量，额外多阶段裁剪重解析会增加请求与计算。

### 5.3 模型 & 技术｜[Swarms, Flocks, Gangs: How to build, train, and evaluate multi-agent systems（蜂群、鸟群、团伙：如何构建、训练和评估多智能体系统）](https://huggingface.co/blog/training-swarms)

北京时间：2026-11-26 08:00:00｜来源类型：官方博客｜事件状态：已公开

Hugging Face 公开共享 Qwen3-0.6B 策略、4 个独立上下文智能体的信息汇总实验及 TRL 复现路径；训练采用团队奖励与组内相对优势。
原始 75 步实验训练成功率从前五步均值 6.56% 升至末五步 82.34%，每队生成 token 从 12826.9 降至 8026.0；checkpoint 75 在 32 个保留问题上解出 26 个。

**重要性**：提供把通信次数、token 成本与团队成功率一起记录的可复现多智能体训练样本。

**限制与风险**：该任务把私有信息分散给不同成员，单智能体天然不能独立完成；不能据此证明多智能体在相同预算下优于单智能体。

### 5.4 工具 & 产品｜[v1.1.0（v1.1.0）](https://github.com/ggml-org/llama.cpp/releases/tag/v1.1.0)

北京时间：2026-11-26 17:16:47｜来源类型：正式 Release｜事件状态：正式发行

llama.cpp v1.1.0 正式发行，纳入 Inkling 2 家族、Hy4 flash 等模型，以及多种语音与音频模型；DFlash 草稿方法拆分成独立选项。
旧 llama_batch API 被弃用，推荐 llama_batch_ext/llama_process；量化器增加按目标 bpw 或文件大小选取逐张量量化，llama-server 扩展指定工具选择及界面能力。

**重要性**：为本地推理的多模型、投机解码、音频与服务接口提供同一正式版本集成点。

**限制与风险**：弃用接口需要下游迁移；模型支持列表不能代替各硬件后端逐模型测试，发行说明未保证所有组合可用。

### 5.5 工具 & 产品｜[Introducing the llama.cpp benchmark dashboard for AMD Ryzen and Radeon（推出面向 AMD Ryzen 和 Radeon 的 llama.cpp 基准仪表盘）](https://huggingface.co/blog/ggml-org/llamacpp-amd-bench)

北京时间：2026-11-26 19:37:17｜来源类型：官方博客｜事件状态：已公开

ggml-org 发布 llama.cpp × AMD 基准仪表盘及公开原始数据，当前硬件为 Radeon 8060S（gfx1151）与 Radeon AI Pro R9700S（gfx1201），可按构建、模型和量化格式查看。
区分 PP512 与 TG128，两者默认上下文深度为零；每周工作流使用预构建二进制，对 Flash Attention 开关各配置重复 10 次，并提供上下文扩展视图。

**重要性**：把本地推理性能变化关联到构建、量化与具体硬件，便于部署选型和回归定位。

**限制与风险**：这是可追溯测量工具，不能把仪表盘发布日期当全部测量日期，也不能用零上下文吞吐替代长上下文性能。

## 六、总结与趋势观察

- **优化从算子延伸到完整部署成本**：DFly2 同时复用稀疏索引与 KV，Parse 提供数据、训练和推理材料，TileLang 则压缩同步分析的冷编译耗时。这些结果分别作用于训练材料、服务执行和开发迭代，不能合并为同一加速倍率。
- **系统边界需要可检查的约束**：ValueBounds 的分支反例、NVGEMM 的符号守卫，以及 CVA6 的 PMP 优先级修正都指向同一问题：只有把实际运行条件与分析模型对齐，性能路径才有可靠语义。
- **RISC-V 部署链路更具体**：Triton-RISC-V 的 VIR 流水线、D560 恢复镜像与 K3 的 CPU 音频识别展示了编译、固件和应用三层进展；开发分支状态与单板报告仍需各自验证。

## 附录：信源说明

主要采用 PyTorch Foundation、LLVM/MLIR Discourse、torch-mlir、IREE、Triton-RISC-V、TileLang、Intel Triton 后端、openRuyi、CVA6、香山 GEM5、腾讯混元、NVIDIA、Hugging Face 与 ggml-org 的原始材料。技术讨论保留提案或回复状态，上游代码进入主干不等于正式版本已发行。厂商性能测量与社区部署报告只适用于文中所述环境；基准工具发布不等于所有结果在同日测量。
