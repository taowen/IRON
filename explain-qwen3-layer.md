# Qwen3 融合层引擎详解

> 在 AMD XDNA NPU 上把 Qwen3-8B 的一个 Transformer decode 层做成一台数据流机器

---

# 一、项目是什么

## 一句话

在 AMD XDNA NPU 上，把 Qwen3-8B 模型的一个 Transformer decode 层做成一台"数据流机器"：PDI/CDO、stream switch、BD ring 和 lock 预配置，runtime 只 patch descriptor/RTP 并启动一次 layer run。

本文后续作为开发目标使用时要区分两层含义：

- **目标态**：Qwen3 decode 单层在 NPU 上数值正确、速度可用，并且可以多层串联。
- **当前态**：`qwen3-layer` 已经能在真机跑通 full-layer 物理闭环，但 attention、RMSNorm/RoPE、SwiGLU 和最终 hidden 输出仍是校准/替换对象。当前通过的是真实 Q4NX projection + current K/V + KV scan + fixed-point attention + Q4NX O/up/gate/down 的集成边界，不等于生产数值已经完成。
- **性能路线**：宏观数据流继续按 MyLM-style fused layer engine 走；后续不是推翻 row1/c1r1/c6r1/topology，而是把当前 high-level AIE C++ main16 Q4NX role kernel 逐步替换成 MyLM-style raw/scheduled core body。

## 背景：GPU 怎么做一层推理

GPU 把 Transformer 层拆成一连串独立算子：

```
Q投影 → K投影 → V投影 → attention → O投影 → up → gate → SwiGLU → down
```

每个算子的流程都是：从显存读上一步输出 → 计算 → 结果写回显存 → 通知下一个算子。
中间结果反复搬运，显存带宽被权重和激活共同争抢。

## NPU 融合引擎的做法

不拆成独立算子，而是把整层做成一台预先配好的数据流机器：

- **构建阶段**：预生成 core program、memtile BD ring、stream switch、lock 和 packet route
- **执行阶段**：runtime patch 当前 token、BO 地址、scan descriptor/RTP，然后启动 layer run。NPU 自动运转，CPU 不参与层内 phase 调度

**核心收益**：大部分中间激活（Q/K/V 投影输出、attention 结果、FFN 中间值）都不回主存，只在 tile 之间直接流动。主存流量主要变成 Q4NX 权重流入（约 115 MiB/层）和 KV cache scan/write；单 token decode 下权重仍是最大流量项。

## 与 MyLM 的关系

当前 IRON 设计和 MyLM 在 operator 级数据流上已经对齐：hidden/RMSNorm replay → main16 Q/K/V → c1r3 postprocess/current K/V → Shape-A/B attention → packet2/O replay → main16 O/up/gate/down → final hidden。真正差距在 AIE 物理执行形态：

- 术语约定：后文正式用 AMD/MLIR-AIE 机制来描述，不再使用我们之前的工程别名当成架构术语。准确分层是：MLIR-AIE/IRON 描述 tile、buffer、BD、lock、stream route 和 runtime sequence；`aie.core` region 是 tile-local core program；`func.func ... attributes {link_with = "...o"}` 把 C++/AIE API/Peano 编译出的 linked AIE core kernel 链进该 tile 的 core ELF；更低层的 `elf_file`/raw ET_EXEC ELF 则是直接指定完整 core program。
- MyLM 的 `layer.xclbin` 是 raw fused engine：core program、BD ring、stream switch、lock phase 预配置，runtime 只 patch descriptor/RTP。
- MyLM main16 是统一的 14,868-byte raw segmented program，`0x1f0` 处加载一次 Q4NX microkernel，各 phase body 调用它。
- IRON 当前 main16 已经不是大段 MLIR phase control。active full-decode 里 MLIR core 基本只是一条 `func.call @q4nx_main16_full_scheduler(...)`，该函数通过 `link_with` 链接进每个 main16 tile 的 C++ AIE core object，作为 tile-local core program entry 实现 phase dispatch。fresh full-decode `main_core_2_2.elf` 反汇编是 `.text=6080`、1 个 core-entry 引用、`jl=14`、`acq=18`、`rel=18`。因此旧的“MLIR 展开 phase loop”问题已经基本迁移到 linked AIE core kernel 中。
- 最新差距集中在 Q4NX microkernel 指令形态：MyLM `0x1f0..0x1850` hot loop 有 `vmac.f=264`、`vextbcst.16=256`、`vunpack=64`、`vups.4x=64`、`vst=0`。IRON 当前 production 已从普通 AIE API MAC 迁到数值正确的 signed native BF16 MAC + 32-dim full unroll，`q4nx_chunk_accum_fast` 现在有 `vmac.f=64`、`vextbcst.16=64`、`vunpack=64`，并通过 QKV compact、full-layer QKV prefix、attention-O 和 full decode；但仍有 `vlda=282`、`vst=194`、`vconv.bf16.fp32=160`，MAC 密度和 MyLM raw scheduled loop 还有明显差距。source assembly probe 已能稳定表达一个 MyLM-style canonical group 的 `33 vmac.f + 32 vextbcst.16 + 8 vunpack + 8 vups.4x + 0 vst`。5x 级性能差距现在主要不是 Python generator 或 MLIR loop annotation，而是 linked C++ AIE Q4NX microkernel 没有达到 MyLM 的 raw scheduled unpack/dequant/MAC 密度。
- MLIR-AIE 工具链探针确认：当前 wheel 的 `bin/aiecc.py` 只是转发到 C++ `bin/aiecc`，core compile 子命令由 C++ driver 生成。aiecc 把 core lowering 成 `main_core_*.peanohack.ll`，再固定调用 Peano `opt --passes=default<O1> -inline-threshold=10` 和 `llc -O2 --march=aie2p --function-sections`。`tools/audit_aiecc_driver.py` 的 fast-fail dry-run 证明把 `--disable-loop-unrolling` 或 `--opt-disable=loop-unroll` 传给 `aiecc` 本身都不会改变子 `opt` 命令，`--aie-loop-aware`/hardware-loop forcing 这类 Peano `llc` flag 也不会被当前 `aiecc` driver 转发到子 `llc`。所以 MLIR-AIE 继续负责 topology/BD/lock/route/runtime/package；main16 性能核心继续走 linked C++ AIE core kernel 或 raw ET_EXEC ELF。
- `tools/replay_main16_core_compile.py` 已把手动 core compile 重放扩展到 16 个 main tile，并通过 `tools/package_externalized_design.py` 跑了真机 ELF-backed packaging。QKV-prefix token31 从 stock `142702.9 us` 到 replacement `124872.8 us`，full decode token31 从同次 rebuilt stock `29404.6 us` 到 replacement `28990.9 us`，数值均 PASS。这证明绕过 stock loop unroll 有真实收益，但 full decode 只提升约 1.4%，仍不是 MyLM 级性能路径。
- transaction packaging probe 确认：aiecc 会把每个 compiled core ELF 写回 `aie.core` 的 `elf_file`，再把 ELF `.text` 降成 `config_blockwrite_data`。fresh QKV prefix 中 16 个 main16 core 的 `.text=9952 bytes`，transaction 里正好有 16 个同尺寸 payload；MyLM main16 raw image 是 `14868 bytes`。因此不能靠 in-place transaction patch 扩容，正确路线是生成替换 main16 ELF 后重新打包 transaction/xclbin。
- `tools/repack_core_program_txn.py` 已经把 transaction-level 重新打包固化：先删除旧 `config_blockwrite_data_*` 和旧 `aie.runtime_sequence @configure()`，保留 `aie.core {elf_file=...}` 与 runtime sequence，然后用 `aie-opt --convert-aie-to-transaction=elf-dir=...` 从当前 ELF 重建 core-program blockwrite。直接对已 transaction 化的 MLIR 再跑 pass 会复制 payload；`aie-translate --aie-npu-to-binary` 生成的也不是 aiecc 的小 `design.bin` runtime instruction artifact。
- 可运行的 raw core 接入点已经确认并固化：`tools/package_externalized_design.py` 会复制 donor project，生成 all-core ELF-backed source MLIR，必要时替换 16 个 `main_core_X_Y.elf`，再用 `aiecc --no-compile` 重新生成 runtime inst、transaction、PDI 和 xclbin。probe 里 ELF-backed build 保持 `design.bin=2372 bytes`、16 个 main16 `9952-byte` payload 和 `213952-byte` xclbin。工具会检查 main16 ELF 没被覆盖、transaction 里有 16 个匹配 payload。不能只把 main16 改成 `elf_file` empty-core 后再正常编译；aiecc 会把空 core 编译回 donor ELF，曾观察到 main16 `.text` 被覆盖成 `160 bytes`。
- replacement ELF 还必须是 ET_EXEC + `PT_LOAD`。MyLM disasm helper 那种 ET_REL `.text` wrapper 虽然 `llvm-size` 能看到 `14868 bytes`，但 `aiecc --no-compile` 不会把它降成 core-program blockwrite。新增 `tools/wrap_raw_aie_program.py` 后，MyLM main16 raw image 能被打包成 16 个 `14868-byte` payload，`design.xclbin=290240 bytes`。这只证明工具链能承载 MyLM-sized raw core，不代表能直接运行 MyLM raw image：MyLM raw core 还假设自己的 whole-core phase program、row1 compact timing 和 Q4NX microkernel body。
- 新的工具链边界已经更精确：Peano/LLVM-AIE 不是完全生成不了硬件 loop。`tools/probe_external_lock_dispatcher.py` 用 Peano `clang++ --target=aie2p-none-unknown-elf -O2` 编译一个 linked C++ AIE core kernel，里面直接调用 AIE2P `acquire_greater_equal()` / `release()` builtin；反汇编结果是 `disasm_acq=1`、`disasm_rel=1`、`disasm_lc_ls_le=3`。这证明 linked C++ AIE core code 可以自己持有 lock 和硬件 loop。当前 active main16 phase control 已经在 linked C++ AIE core object 里；剩余失败点是 Q4NX microkernel 的指令调度没有达到 MyLM raw loop 形态。
- AIEVec/XLLVM 路线可以作为 raw binary 前的中间层。`qwen3-layer/tools/probe_aievec_q4nx_codegen.py` 已确认 `aievec.matmul` 能降到 `xllvm.intr.aie2p.I512.I512.ACC2048.mac.conf`，`aievec.ups/srs` 能降到 BF16 accumulator conversion；但当前 AIEVec/XLLVM op 面只有 `cast/ext/matmul/shift/shuffle/srs/ups`，没有直接暴露 MyLM hot loop 需要的完整 `vunpack/vextbcst.16` 组合。最新决策是对 Q4NX hot body 走 source assembly：`experiments/aie_intrinsics_api_probe/run_asm_probe.py` 已证明 AIE2P `.s` 可以稳定生成 canonical middle group 的 `33 vmac.f + 32 vextbcst.16 + 8 vunpack + 8 vups + 0 vst`，`run_asm_link_probe.py` 已证明 C++ object 和 `.s` object 可以用 `ld.lld -r` 合成一个 role object，`run_asm_npu_smoke.py` 已证明 full 8-group exact lane body 能在真机上数值匹配 BF16 reference。关键教训是：full static lane unroll 会导致 CDO program-memory overflow；大 source-level hardware loop 不可靠；`add #0x40` 会编码成 `-64`；scale/offset/activation 要用 pointer register 推进，不能用 loop-carried `dj0` offset 寄存器。这个能力也已经迁入 `qwen3-layer/npu_build.py`：当前唯一 main16 role object `main_projection_q4nx_fast.o` 由数值正确的 C++ scheduler/kernel 和 `main_projection_q4nx_asm.s` 探针合成，`.s` 探针由 `tools/generate_main16_q4nx_asm.py` 从寄存器/指令计划生成并由 `tools/check_main16_asm_integration.py --strict` 检查；其中 `q4nx_accum_lane_exact_body_shape` 已经是接近 production scheduler 的五指针 ABI：`p0=packed lane data, p1=scale lane, p2=offset lane, p3=activation, p4=dst bf16`，并且不在汇编里 release lock。最终 core ELF 中未引用的 assembly probe 被 `--gc-sections` 丢弃，没有挤占 program memory。进一步试过让手写 `.s` 入口 `j #cpp_symbol` 尾跳回 C++ 语义函数，最终 ELF 会变成 `j #0`，目标函数被 GC；所以不能靠 asm wrapper 过渡。另一个已排除的捷径是 MyLM-style `offset * group_sum` 近似：它破坏当前 per-dim BF16 rounding reference，`qwen3-8b-qkv-compact-output` 最大误差到 `0.112304688`，不能作为 active decode path。当前 active path 先保留 exact per-dim rounding，用 signed native MAC 拿到 `vextbcst.16 + vmac.f`；真正要替换的是保持这个数值语义、但把 `vlda/vst/vconv` 压到 MyLM 量级的完整 source-assembly hot body。`tools/check_main16_asm_integration.py --strict` 固化检查这个边界、生成器漂移、生产 wrapper 指令门禁和坏跳转/坏 lane 残留。
- `tools/check_main16_raw_abi.py` 已把 main16 外层 ABI 变成机器检查。当前 row0 main16 结果是 `raw_main16_abi_ready=true`：activation 是 BD0/1、base `0x8000/0xc000`、L0->L1；weight 是 BD2/3、base `0x2800/0x4000`、L2->L3；record 是 BD4/5、base `0x3c1c/0x541c`、17 dword ping/pong、L5->L4。row1/c1r1 compact tree 也已经迁到 record-granular：main16 17-dword record → row1 65-dword column record → c1r1 257-dword global record。
- 直接替换成 MyLM whole-core main16 ELF 的卡点已经收窄：不是 DMA0/DMA1/record ring，而是 MyLM raw whole-core program 的隐藏 contract。MyLM CDO 会为每个 main tile 初始化 `0x3c60/0x3c64/0x3c68/0x3c80/0x3d00`，其中 `0x3c60..0x3d00` 是 record ping 后面的本地控制/scratch 区；MyLM main16 程序会直接读写这些地址。MyLM c1r2 还通过 `bd3 len=2049` 发出 `1 control dword + 2048 payload` 的 full-vector replay，main16 core program 依赖这个 activation-side control 选择 normal phase chain。IRON 当前 active c1r2 语义上有相同 replay count，但 linked AIE core kernel 仍按 IRON ABI 接收 2048-dword payload，并不等价于 MyLM whole-core program ABI。
- MyLM main16 的 record header 是小 phase word：QKV=`0x1`、O=`0x4`、upgate=`0x8`、down=`0x4`。IRON 当前 header 是 richer debug/value contract：`phase/block/group/row/packet_id`，低位 packet id 为 `10..15`。不过 active IRON 的硬件路由不是直接读这个数据 header；bridge 端 `aie.dma_bd(... packet=#aie.packet_info<...>)` 和 `aie.packet_flow(...)` 决定 packet routing。header mismatch 主要会影响 value contract、debug/reference，以及任何读取 `compact[0]` 的 kernel。c1r3/c6r2 当前接收 header-stripped payload，c1r2 O/down 当前从 `compact+1` 读 payload。
- MyLM 的 16 个 main tile 使用同一个程序并不是因为还有未解的 tile metadata RTP，而是因为生产数据流把 `group/row/block` 隐含在物理 tile、row1 fanout、权重 chunk 顺序和 compact gather 位置里。IRON 的 `group/row/block` header 是调试友好设计，不是 MyLM 必需机制。

因此后续性能优化的主线仍是 MyLM-style raw/scheduled main16 kernel，但重点已经从
“修 compact 粒度/移动 phase control”转成“替换 Q4NX microkernel 的指令形态”。如果之后选择直接采用 MyLM whole-core AIE program，而不是只替换 Q4NX microkernel，就必须同时迁移 c1r2 2049-dword activation control、main16 `0x3c60..0x3d20` 本地控制区、小 phase header contract 和严格的 MyLM weight schedule。

### MLIR-AIE 工具链审计结论

当前问题不是“MLIR-AIE 完全不能用”，而是边界要切准：

- 继续用 MLIR-AIE 生成 topology：tile、buffer 地址、BD ring、lock、stream switch、runtime sequence、transaction、PDI/xclbin。
- 不再指望 `aie.core` 里的 `scf.for` phase control 被 stock `aiecc` 编译成 MyLM 那种 raw phase body。源码和 probe 都确认，当前 C++ `aiecc` 会固定走 `AIECoreToStandard -> LLVM lowering -> peanohack -> Peano opt --passes=default<O1> -inline-threshold=10 -> llc -O2 --march=aie2p`。这个 child `opt` 会展开 constant-trip phase loop，`aiecc --disable-loop-unrolling`、`--opt-disable=loop-unroll`、Peano `--aie-loop-aware`/hardware-loop forcing 这类参数不会转发到 child `opt/llc`。
- Peano/LLVM-AIE 本身可以生成 AIE hardware loop。`probe_external_lock_dispatcher.py` 已证明 linked C++ AIE core code 里直接调用 AIE2P lock builtin 可以得到 `acq/rel` 和 `lc/ls/le` loop 形态。失败点是 stock MLIR `aie.core` compile path，不是 AIE2P backend 没能力。
- 官方可用的替换入口是 `aie.core { aie.end } {elf_file = ...}`。`AIE_CoreOp` verifier 要求带 `elf_file` 的 core body 必须为空；`convert-aie-to-transaction{elf-dir=...}` 会读取这些 ELF 并生成 core-program blockwrite。
- 可运行 packaging 路线已经固定为 all-core ELF-backed source MLIR + `aiecc --no-compile`。只把 main16 改成 `elf_file` empty-core 再让 `aiecc` 正常编译会把空 core 覆盖成无效小 ELF；直接 patch 已 transaction 化 MLIR 也会遇到 payload 重复或 routing/resource pipeline 重入问题。

所以接下来不是再调 MLIR loop annotation，也不是继续堆 C++ kernel 变种。工程路线是：保留 MLIR-AIE 生成的 topology/BD/lock/route/runtime/package，保留当前 linked AIE core kernel ABI，把 `q4nx_chunk_accum_fast` 改成更接近 MyLM raw loop 的 lower-level scheduled intrinsic kernel。优先试 AIEVec/XLLVM 能覆盖的 MAC/UPS/SRS 部分；缺失的 unpack/broadcast 先用 AIE2P compatibility intrinsic 或扩展 AIEVec/XLLVM。若这条路仍然达不到 MyLM 指令形态，再走 raw ET_EXEC ELF 生成。

本机 MyLM 8B benchmark probe 约 `4.93 tok/s`，即约 `203 ms/token`。按
36 层粗分是 `5.6 ms/layer`，还不扣 lm_head/runtime 等非层开销；MyLM 文档公开数
据 `11.9 tok/s @1k` 则对应约 `2.3 ms/layer` 的更紧预算。当前 IRON 单层
`25.067 ms`，所以“4-5x 提升”不是夸张目标，而是至少要追上本机 MyLM probe
才成立。

## 类比

| GPU | NPU 融合引擎 |
|-----|-------------|
| 工厂里每道工序做完，半成品送回仓库，下一道再取 | 流水线车间，半成品直接从一个工位传到下一个 |
| 每个算子一次 kernel launch | 每层一次 runtime patch + 启动 |
| 显存带宽 = 权重 + 中间结果争抢 | 主存带宽 = 权重 + KV cache + 层输入/输出 |

## 适用场景

NPU 融合引擎特别适合**单 token decode**：计算量小、瓶颈在带宽、融合能减少搬运。这正是端侧（笔记本、手机）LLM 推理的核心场景。

Qwen3-8B 共 36 层，每层都可以用这类数据流机器实现。本项目当前实现的是其中一层的真机物理数据流 frontier；最终目标是把它推进到数值正确、性能足够好的 decode engine。

---

# 二、硬件布局

## 三层结构

NPU 阵列从下到上分三层：

```
┌─────────────────────────────────────────────────┐
│  Row 5  ┃  计算层（Compute Tile）               │
│  Row 4  ┃  每个 tile 是一个小处理器，           │
│  Row 3  ┃  跑 C/C++ 程序做实际运算              │
│  Row 2  ┃                                       │
├─────────╋───────────────────────────────────────┤
│  Row 1  ┃  中转层（Memtile）                    │
│         ┃  较大片上缓存，负责拆分/缓冲/转发      │
├─────────╋───────────────────────────────────────┤
│  Row 0  ┃  接口层（Shim Tile）                  │
│         ┃  主存入口——数据从这里进出 NPU          │
└─────────────────────────────────────────────────┘
```

- **Shim（Row 0）**：NPU 的大门。权重、输入向量、KV 缓存都通过这里进入芯片，计算结果也从这里写回主存
- **Memtile（Row 1）**：几百 KB 的片上缓存。接收大块数据、拆成小块、用 ping-pong 缓冲让传输和计算重叠、汇聚多路结果
- **Compute Tile（Row 2-5）**：真正干活的地方。每个 tile 有几十 KB 本地内存和 DMA 引擎，跑编译好的 C/C++ 程序

## 8列 × 6行 棋盘

本设计使用 8 列 × 6 行分区，每个 tile 用 `c{列}r{行}` 标记：

```
       c0     c1     c2     c3     c4     c5     c6     c7
      ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
row5  │      │      │      │      │      │      │      │      │  Compute
      ├──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┤
row4  │      │      │      │      │      │      │      │      │  Compute
      ├──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┤
row3  │      │      │      │      │      │      │      │      │  Compute
      ├──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┤
row2  │      │      │      │      │      │      │      │      │  Compute
      ├──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┤
row1  │      │      │      │      │      │      │      │      │  Memtile
      ├──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┤
row0  │      │      │      │      │      │      │      │      │  Shim
      └──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘
```

共 48 格：32 个 Compute Tile + 8 个 Memtile + 8 个 Shim。

## 数据搬运机制

NPU 内部不靠 CPU 搬数据，有一套专用硬件机制：

### DMA（直接内存访问）

每个 tile 的"自动搬运工"。给它一份任务单，按单搬数据，不需要 CPU 介入。每个 tile 有多个 DMA channel，可同时搬运多份数据。

### BD（Buffer Descriptor，缓冲区描述符）

DMA 的"任务单"，指定：
- 从哪个缓冲区的哪个偏移开始
- 搬多少数据
- 搬完后下一个任务是什么（链式 BD）
- 是否给数据打上 packet ID（用于路由）
- 开始前等哪个锁，完成后释放哪个锁

硬约束：Shim tile 的 BD ID 只有 0..15，用完就没有了。这限制了同一时刻能描述多少并发传输。

### Lock（锁）

生产者-消费者之间的同步信号：

```
生产者填满 buffer A → 释放"A 满了"锁
消费者等到"A 满了"  → 取走数据 → 释放"A 空了"锁
生产者等到"A 空了"  → 继续填 A
```

锁配错 = 死锁（永远等下去）。这是 NPU 编程中最常见的 bug 来源。

### Stream（流）

Tile 之间的固定物理连线。数据沿这些线从一个 tile 流向另一个。编译器把逻辑连接映射到物理路线。

### Packet 路由

同一根物理线上跑多路数据：每个数据包贴一个 packet ID，接收端根据 ID 筛选"这个包是给我的"还是"让它继续走"。

```
tile A ──[packet 8]──→ tile X（匹配 ID 8，接收 K）
         [packet 9]──→ tile Y（匹配 ID 9，接收 V）
```

本设计中 packet 路由大量用在 KV 写回（packet8/9）、attention 返回（packet2）、down 激活返回（packet1）、full-vector replay（packet0）上。

---

# 三、角色分工

整个 48-tile 阵列分成两大阵营：**Main16**（中间 4 列，专做矩阵乘）和 **Edge16**（两侧 4 列，各种辅助操作）。它们的需求完全不同：

| | Main16 | Edge16 |
|---|---|---|
| 视野 | 极窄（32×256 小块） | 全局（完整 4096 维） |
| 运算 | Q4NX 反量化 + 投影 MAC | 归一化/softmax/SiLU/逐元素乘 |
| 程序 | 16 个 tile 完全相同 | 每个 tile 不同 |

## Main16：16 台相同的车床

```
       c0  c1 ┃ c2   c3   c4   c5 ┃ c6  c7
row5          ┃[M13][M14][M15][M16]┃
row4          ┃[M09][M10][M11][M12]┃
row3          ┃[M05][M06][M07][M08]┃
row2          ┃[M01][M02][M03][M04]┃
              ┃     Main16         ┃
```

**特点**：

- **完全相同**——跑一模一样的程序，各自处理不同输出行
- **只做一类事**——128 dword 激活 + 1280 dword Q4NX row-chunk 权重 → 反量化/MAC/flush → 17 dword compact record
- **视野极窄**——每个 tile 一次只看 32 行 × 256 列的窗口，看不到完整 4096 维向量
- **时分复用**——同一组 16 tile 依次跑 7 个投影阶段（Q→K→V→O→up→gate→down）。阶段不是只靠权重区分，还由 body schedule、record header、record slot、replay count 和 Edge 侧消费路径共同定义

MyLM 证明这个 ABI 可以做成更硬的 raw core program：16 个 main tile 都加载同一个 14,868-byte program image，其中 `0x1f0..0x1850` 是共享 Q4NX microkernel，`0x1870/0x1e80/0x2490/0x2aa0` 分别是 QKV/O/upgate/down phase body，`0x36d0` 是 dispatcher。IRON 当前的 `main_projection_q4nx_fast.cc` 是正确性 baseline；后续优化目标是在不改 DMA0/DMA1/record ABI 的前提下替换 compute body。

这里要区分两种替换粒度：

- **只替换 Q4NX microkernel**：保留 IRON 当前 linked AIE core kernel ABI、record header、c1r2 replay 和 bridge packet ABI，只把 `q4nx_chunk_accum_fast` 变成更接近 MyLM `0x1f0..0x1850` 的 scheduled loop。这是当前最短路径。
- **替换 MyLM whole-core AIE program**：必须同时匹配 MyLM 的 activation control word、`0x3c60..0x3d20` 本地 control/scratch、phase header `0x1/0x4/0x8/0x4`、weight schedule 和 phase replay order。这不是 C 函数级替换。

每个 tile 有两个 DMA 输入：
- **DMA0**：激活输入（128 dword = 256 bf16）
- **DMA1**：权重输入（1280 dword = 5120 字节 Q4NX chunk）

输出：**MM2S1** 发送 17-dword compact record（1 dword header + 16 dword payload）。

## Edge16：一条异构装配线

两侧每个 tile 干不同的活：

```
       c0         c1        ┃ main16 ┃    c6         c7
row5  [Shape-B]  [    ]     ┃        ┃   [    ]    [Shape-B]
row4  [Shape-A]  [    ]     ┃        ┃   [    ]    [Shape-A]
row3  [Shape-B]  [后处理]   ┃        ┃   [    ]    [Shape-B]
row2  [Shape-A]  [全向量站] ┃        ┃   [SwiGLU]  [Shape-A]
row1  [KV整形]   [共享桥]   ┃  行1   ┃   [枢纽]    [KV整形]
row0  [K写回]    [────────── shim ──────────────]   [V写回]
```

### c1r2 全向量站

**目标职责**：承载完整 4096-bf16 hidden 向量的全局操作

- 第一次 RMSNorm（进入投影前）
- O 投影后的残差加 + 第二次 RMSNorm（喂给 up/gate）
- Down 投影后的最终残差 + 层输出

**为什么在这里**：RMSNorm 和残差必须看到完整 4096 维向量。目标态 c1r2 要提供 2048-dword full-vector ABI、sum-of-squares/rsqrt 和 residual 计算；当前实现只保留这个物理位置和 replay/output 契约。

**对外接口**：2049-dword packet0 replay（1 control + 2048 payload），replay 次数 = +12（Q/K/V）→ +48（up/gate）。最终输出是 2048-dword hidden_out。c1r2 本地只常驻一个 2048-dword hidden/residual buffer、一个 2048-dword vector/norm buffer、一个 2049-dword replay buffer、一个 257-dword compact receive buffer 和一个 2048-dword output buffer；它不接收 6144-dword full_input，也不常驻 8 条 O/down compact record。

**当前状态**：`full_vector_station.cc` 已经按 phase station 运行：host raw hidden 进入 hidden/residual buffer，arg2 尾部的 input/post RMSNorm 权重复用 2048-dword vector buffer，O compact 流式累加到 residual，post RMSNorm replay 后驱动 up/gate，down compact 流式加回 residual 形成 hidden_out。RMSNorm 仍是当前 bounded fixed-scale 数值路径，不是最终生产 rsqrt kernel；后续要和 c1r3/attention/SwiGLU 一起按 `Qwen3LayerReference` 分阶段收紧。

### c1r3 后处理站

**目标职责**：Q/K norm + RoPE 位置编码 + current K/V 路由

- 目标上接收 main16 产出的 Q+K+V 向量记录
- 对 Q 和 K 做 RMS 归一化
- 对 Q 和 K 做 RoPE 旋转位置编码（V 不做）
- 输出 Q（2048 dword）发给 attention 侧
- 输出 current K/V 经 packet8/9 写入主存缓存

**关键约束**：需要 runtime-start lock 门控，否则 c1r3 可能在 host 写 current-token RTP 前就开始执行。

**当前状态**：`postprocess_qkv.cc` 已经实现 Q/K/V body payload 到 Qwen3 attention ABI 的打包、Q/K RMSNorm、RoPE 和 current K/V packet8/9 写回。当前 full-layer 里 c1r3 收到的是 header-stripped body payload（Q=2048 dword，K/V=512 dword），不是 12 个带 header 的 257-dword global compact。生产差距是 Q/K RMSNorm、RoPE scale/角度和 cache writeback 的数值误差还要按 Qwen3 reference 继续收紧。

### c6r1 枢纽（Memtile）

**职责**：分发 + 汇聚的中转站，承担两个方向的工作

- **前向**：接收 attention ABI 下的 Q payload（2048 dword，当前 full-layer 使用 bf16 Qwen3 payload），拆成 4 份 512-dword 窗口发给 4 个 Shape-A
- **反向**：接收 4 个 Shape-B 的 attention 结果，拼成 2048 dword，以 packet2 发布
- **FFN 汇聚**：收集 24 个 SwiGLU slice，拼成 6144-dword FFN intermediate，以 packet1 发布

### c1r1 共享激活桥（Memtile）

**职责**：packet 数据 → 切片 → 广播给 main16

- 接收 c6r1 发来的 packet（2048-dword attention 或 6144-dword FFN）
- DMA4 接收，256-dword ping-pong bridge
- DMA1 广播到 main16 DMA0 的 128-dword activation ring

**关键**：attention 返回（packet2）和 down 激活返回（packet1）**复用同一条物理桥**。

### Shape-A × 4（评分）

**位置**：c0r2, c0r4, c7r2, c7r4

**目标职责**：QK 点积评分 + block-level softmax

- 接收一个 512-dword Q 窗口（8 heads × 128 dim）
- 每 16 个历史 token 为一个 block，计算 score 和 block_max/block_sum
- 把 carrier（权重+统计量）交给配对的 Shape-B
- 最后一个 block 用 tail-token RTP 把 padding token 权重清零

**当前状态**：`edge_attention.cc` 只保留 active `qwen3_attention_bf16_*` kernel ABI：Q/K/V 为 bf16 payload，Shape-A 生成 bf16 block weights + float scalar carrier。生产差距是 `fast_exp`、online merge 和 bf16 舍入误差仍需和 Qwen3 reference 校准。

### Shape-B × 4（求和）

**位置**：c0r3, c0r5, c7r3, c7r5

**目标职责**：加权 V 求和 + online softmax merge

- 接收 Shape-A 的 carrier + 对应 block 的 16 个历史 V
- 用 block 权重加权 V，与本地累加器做 online-softmax merge
- 历史扫描结束后输出 `accum / running_sum` = 512-dword attention 结果

**当前状态**：active full-layer 中 Shape-B 使用 float accumulator/state 合并 bf16 block，并输出 bf16 packet2 给 O phase。旧 int32/Q12 kv16 path 只作为历史 helper 留在源码中，不是当前生产化 frontier。

### c0r1 / c7r1 KV 整形（Memtile）

**职责**：从 shim scan 接收历史 K/V，分流到 Shape tile

- c0r1 处理 group 0-3（左侧）
- c7r1 处理 group 4-7（右侧）
- K 送 Shape-A，V 送 Shape-B

### c6r2 SwiGLU 切片站

**目标职责**：执行 `SiLU(gate) × up`

- 接收 512-dword 输入（低半区 = up slice，高半区 = gate slice）
- 输出 256-dword（512 bf16）SwiGLU slice
- 24 个 slice 组成完整 12288-bf16 FFN intermediate

**当前状态**：`swiglu.cc` 的 bf16 path 已经去掉人为 `slice_scale`，按 `SiLU(gate) × up` 形态执行；当前为了适配 AIE tile 程序使用本地 bounded table sigmoid 近似，后续还要按 Qwen3 生产 kernel 的误差预算校准。

### Row1 c2-c5 列 compact tile

**职责**：per-column record 汇聚 + Q4NX weight 分发

- S2MM0-3：接收 main16 的 compact record，汇聚成 column-level layout
- S2MM4/5：接收来自 shim 的 Q4NX 权重
- MM2S0-3：把权重分发到各行 main16 DMA1
- MM2S5：column compact 输出到 c1r1/global bridge（不能和 MM2S0-3 权重 fanout 混为一类）

---

# 四、整层数据流——一个 token 的完整旅程

跟随一个 token，看它从进入 NPU 到离开的完整过程。

## 起点：准备

一个 token 的隐藏状态 = 4096 个 bf16 数字组成的向量（8 KB）。同时准备好的还有：
- 本层全部权重（Q/K/V/O/up/gate/down，Q4NX 格式，约 115 MiB）
- 历史 KV 缓存（之前所有 token 的 K 和 V）

NPU 不会一次性读入所有权重，而是按阶段分批流入。

## 步骤 1：第一次 RMSNorm（目标态）

```
主存 → hidden vector(4096 bf16) → c1r2 全向量站
                                    ↓
                              计算平方均值
                              除以均方根
                              乘以缩放参数
                                    ↓
                            归一化后的 hidden
```

RMSNorm 必须看到完整 4096 维向量，所以目标上放在 c1r2。归一化后的 hidden 准备分发给 Main16。

当前 `qwen3-8b-decode-layer` 已经让 host raw hidden 和模型 RMSNorm 权重进入 c1r2 phase station，并由 c1r2 发起 Q/K/V replay；但这里仍属于当前 NPU 数值路径，rsqrt/scale 的生产误差预算还没有收紧到最终 Qwen3 decode 要求。

## 步骤 2：Q/K/V 投影

```
c1r2 发出 12 次 packet0 full-vector replay
  ↓
c1r1 共享桥切成 256-bf16 chunk
  ↓
Main16 DMA0 接收激活（128 dword/次）
Main16 DMA1 同时接收 Q4NX 权重（1280 dword/次）
  ↓
16 个 tile 并行乘累加
  ↓
输出 compact record（17 dword）
```

**拆分方式**：
- Q 投影（4096→4096）：8 个 N-block，每 tile 8 条 record
- K 投影（4096→1024）：2 个 N-block，每 tile 2 条 record
- V 投影（4096→1024）：2 个 N-block，每 tile 2 条 record

每个 N-block = 16 tile × 32 bf16 = 512 个输出元素。main16 的 MAC/flush 核心对各 phase 复用，但不是完全 phase-blind：record header、body buffer、weight chunk base 和后续消费者都由 phase schedule 决定。

## 步骤 3：Q/K/V 后处理

```
main16 Q/K/V record/body payload → row1/c1r1 汇聚 → c1r3 后处理站
                                        ↓
                                  目标: Q/K 做 RMS norm
                                  目标: Q/K 做 RoPE（V 不做）
                                        ↓
                              Q(2048 dw) → 发给 attention
                              K/V → packet8/9 写回主存
```

RoPE 让模型知道"这是第几个词"——按维度两两配对，旋转一个与位置相关的角度。

当前实现已经有 c1r3 的物理位置、runtime-start lock、packet8/9 current K/V 写回和 attention ABI 打包。当前 full-layer 传给 c1r3 的是 Q/K/V body payload，Q payload 为 2048 dword，K/V payload 各 512 dword；真正的 Q/K RMSNorm、RoPE 常量和模型权重仍是待校准项。

## 步骤 4：Current K/V 写回

```
c1r3 → packet8 → shim(c0r0) → 写入 K cache BO
c1r3 → packet9 → shim(c7r0) → 写入 V cache BO
```

当前 token 的 K 和 V 必须写入缓存，供未来 token 的 attention 使用。当前 generator 使用每侧两个 shim S2MM runtime BD 做 even/odd scatter（active 常量是 BD2/BD3），实现 head-interleaved 写入；不要沿用旧实验里的高编号 BD 描述。

**关键约束**：写回必须在 KV scan 之前完成（`npu.sync`），否则当前 token 看不到自己。

## 步骤 5：Attention

分三步：读历史 → 评分 → 加权求和

### 5a. 读出历史 KV 缓存

```
c0r0 shim MM2S ch0: K group 0-3 per block → c0r1
c0r0 shim MM2S ch1: V group 0-3 per block → c0r1
c7r0 shim MM2S ch0: K group 4-7 per block → c7r1
c7r0 shim MM2S ch1: V group 4-7 per block → c7r1
```

按 16 个 token 为一个 block 扫描。使用 iterated BD + queue repeat count 复用 descriptor。

### 5b. Shape-A 评分

```
每个 Shape-A 拿到:
  - 自己那组的 Q 窗口（512 dword = 8 heads × 128 dim）
  - 逐 block 流入的历史 K

每 16 个 token:
  当前 fixed-point: score = Q · K / 128
  当前 fixed-point: 求 block_max
  当前 fixed-point: weight = Q12_exp(block_max - score)
  打包成 carrier → 交给 Shape-B
```

最后一个 block 用 tail-token RTP 把 padding 位置的权重清零。生产 Qwen3 attention 不能把 `dot/128` 和 Q12 lookup 当作最终数值；这里后续要替换/校准到 Q/K norm、RoPE 和模型 attention scale 对应的 softmax。

### 5c. Shape-B 加权求和

```
每个 Shape-B 拿到:
  - Shape-A 传来的 carrier（权重 + max/sum）
  - 对应的 16-token V block

目标数学:
  block_weighted_v = sum(weight[t] × V[t])
  online-softmax merge:
    new_max = max(running_max, block_max)
    accum = accum × old_scale + block_weighted_v × new_scale
    running_sum = running_sum × old_scale + block_sum × new_scale

扫描结束: output = accum / running_sum
```

当前实现用 Q12/int32 近似这套 online merge。每个 Shape-B 产出 512 dword（8 heads × 128 dim）；full-layer 中该输出会转换为 bf16 后进入 O phase。

## 步骤 6：Attention 结果返回 → O 投影

```
4 个 Shape-B → c6r1 的 2048-dw return buffer（按 head 顺序拼接）
  ↓ 四个窗口全部就位后
c6r1 以 packet2 一次性发布
  ↓
c1r1 共享激活桥（DMA4 接收 → 256-dw ping-pong → DMA1 广播）
  ↓
Main16 DMA0 接收 128-dw activation chunk
Main16 DMA1 同时接收 O 的 Q4NX 权重
  ↓
O 投影（4096→4096，8 个 N-block）
```

从 main16 的角度：又来了 256 bf16 激活 + 权重 chunk，跟做 Q/K/V 时一模一样。

## 步骤 7：O 后残差 + 第二次 RMSNorm

目标数据流：

```
O compact record → row1/c1r1 汇聚 → c1r2 全向量站
  ↓
c1r2 从 O compact 重建完整 4096-bf16 O 结果
  + 与原始 hidden 残差加
  + 第二次 RMSNorm
  ↓
归一化后的 hidden 准备发给 up/gate
```

当前 frontier 已经把 O compact 送回 c1r2，并用 bounded numeric scale + int32 sqrt 生成 up/gate replay；这能稳定通过真机，但还不是 Qwen3 的 production RMSNorm/residual。

## 步骤 8：Up/Gate 投影

```
c1r2 发出 48 次 packet0 full-vector replay
  ↓
c1r1 bridge → Main16 DMA0
Main16 DMA1 同时接收 up/gate Q4NX 权重
  ↓
偶数 replay → up[slice] record
奇数 replay → gate[slice] record
共 48 条 record = 24 对 (up, gate)
```

物理执行顺序：up 先于 gate（adjacent pair 调度）。

## 步骤 9：SwiGLU

```
48 条 up/gate record → row1 column compact → c1r1 global compact
  ↓
c1r1 output BD 跳过 header → 48 个 256-dw payload half → c6r2
  ↓
c6r2 每收两半（low=up, high=gate）:
  目标: output = SiLU(gate) × up
  ↓
24 个 256-dw SwiGLU slice → c6r1 的 6144-dw gather buffer
```

当前 `swiglu.cc` 的 bf16 path 已经执行 `SiLU(gate) × up` 形态并去掉 `slice_scale`；后续必须把本地 bounded table sigmoid 近似校准到 Qwen3 生产 SiLU/SwiGLU 的误差预算。

## 步骤 10：Down 投影

```
c6r1 gather buffer 满 → packet1 发布 6144 dword
  ↓
c1r1 共享激活桥（同一条物理桥，之前走 packet2）
  ↓
Main16 DMA0: 48 个 128-dw activation chunk（12288/256=48）
Main16 DMA1: down Q4NX 权重
  ↓
down 投影（12288→4096，每 tile 累加 48 次，8 个 N-block）
  ↓
down compact record
```

## 步骤 11：最终残差 → 层输出

```
down compact → c1r2 全向量站
  ↓
c1r2 从 compact 重建 4096-bf16 down 结果
  + 与步骤 7 后的 hidden 残差加
  ↓
layer_output = hidden + down_output
  ↓
写回主存（作为下一层的输入）
```

这是目标态。当前 `qwen3-8b-decode-layer` 已经输出 2048-dword hidden payload，并和 `Qwen3LayerReference` 的 hidden_out 做 `abs_tol=0.05, rel_tol=0.20` 比较；这证明 down compact 已经接回 c1r2 并形成完整层输出。后续重点不是再证明物理闭环，而是把 c1r2/c1r3/attention/SwiGLU 的 stage-local 误差预算收紧，并证明该 hidden_out 可以作为多层推理输入。

## 全程总结

```
时间 ──────────────────────────────────────────────────────────────→

主存→c1r2:  [hidden in]
c1r2→main:  [───── 12 replay (Q/K/V) ─────]
main→c1r3:                                  [Q/K/V records]
c1r3→shim:                                      [pkt8/9 KV写回]
shim→edge:                                          [KV scan────]
Shape-A/B:                                          [attention──]
c6r1→c1r1:                                                      [pkt2]
main:                                                           [O投影]
c1r2:                                                                [目标: 残差+norm]
c1r2→main:                                                              [48 replay]
main:                                                                   [up/gate──]
c6r2:                                                                              [SwiGLU]
c6r1→main:                                                                               [pkt1 down]
main:                                                                                    [down投影─]
c1r2:                                                                                              [目标: 残差→out]
```

---

# 五、关键设计决策

## 5.1 三次闭环：数据在 Main 和 Edge 之间来回

目标整层计算形成三次 Main↔Edge 闭环，每次都把大块中间激活留在片上，而不是写回主存再读回：

```
闭环 1（最复杂）：
  Main 产出 Q/K/V → c1r3 后处理 → c6r1 拆 Q
  → Shape-A/B attention → c6r1 汇聚 → packet2
  → c1r1 bridge → Main 做 O 投影

闭环 2：
  Main 产出 O record → c1r2 目标残差+RMSNorm
  → 48 次 packet0 replay → Main 做 up/gate

闭环 3：
  Main 产出 up/gate record → c6r2 SwiGLU
  → c6r1 gather → packet1
  → c1r1 bridge → Main 做 down 投影
```

如果不做融合，每次闭环通常意味着一次写主存再读回。这里的收益是消除大块中间激活往返；KV cache、权重和层输入/输出仍然经过主存。

## 5.2 共享激活桥复用

c1r1 的 DMA4/DMA1 bridge 不是 attention 专用——attention 返回（packet2）和 FFN 下行（packet1）**走同一条物理桥**，只是 packet ID 不同。

```
c6r1 packet2（attention 2048 dw） ─┐
                                    ├→ c1r1 DMA4 → 256-dw ping-pong → DMA1 → Main16 DMA0
c6r1 packet1（FFN 6144 dw）     ──┘
```

好处：节省一条完整的 memtile→compute 数据通路。Main16 端不感知差异——都是 128-dw activation chunk。

## 5.3 Row1 通道所有权

Row1 memtile 有多个 S2MM/MM2S channel，它们被严格划分：

| Channel | 用途 | 不可混用原因 |
|---------|------|-------------|
| S2MM 0-3 | compact record fan-in（main16 record 汇聚） | 如果权重也走这里，lock/BD 冲突 |
| S2MM 4/5 | Q4NX weight 入口（从 shim 进来） | 专用于权重流，不被 compact 干扰 |
| MM2S 0-3 | weight 分发到 main16 DMA1 各行 | 与 compact output 分离，避免 fanout/compact phase 互相抢锁 |
| MM2S 5 | column/global compact 输出 | 独立于 weight fanout，按 phase trace 打 packet |

这不是装饰性选择——旧版曾尝试把权重走 S2MM0/1，会和 compact fan-in 冲突。`physical_contract.py` 现在显式禁止旧路由。

## 5.4 BD/Descriptor 复用

Shim BD ID 只有 0..15，而 KV scan 需要描述多个 block 的传输。如果每个 block 一个 BD，超过 7 个 block 就没有 BD 给 current K/V 写回用了。

**解决方案**：iterated BD + queue repeat count

```
单个 scan BD:
  buffer_length = 4096（一个 side 的 K 或 V 流）
  iteration_size = <rounded_blocks>
  iteration_stride = 8191（跨 block 前进）

push_queue:
  repeat_count = <rounded_blocks - 1>（让同一个 BD 执行完整 scan）
```

只要 `iteration_size` 和 `repeat_count` 配对编程，一个 K scan BD 和一个 V scan BD 就能在当前 AIEX descriptor 限制内复用扫描多个 rounded block。

**陷阱**：只写 `iteration_size` 不写 repeat count → 只发送第一个 segment → timeout。

## 5.5 Instruction Patch（避免重编译）

不同 token 位置的 decode 参数不同（block 数、cache 大小、tail-token 数等），但完整重编译 xclbin 很慢。

**方案**：固定最大容量 PDI + 只 patch `design.bin`（instruction stream）

可 patch 的参数：
- current-token RTP
- Shape-A/B block-count RTP
- Shape-A tail-token RTP
- current K/V 写地址
- scan BD 的 iteration_size/stride
- push_queue repeat_count

**已验证**：当前 active full-layer decode runner 使用 token127 容量 xclbin/PDI：
- token127 → token31 的 patched instruction stream 逐 word 等于直接编译的 token31 stream
- token0、token1、token31、token91 都用 token127 容量 xclbin + patched `design.bin` 真机通过
- token127 使用原生容量 `design.bin` 真机通过

这证明 active full-layer instruction patch 是精确的，可以作为"固定容量 PDI + 每 token patch"的生产方案雏形；完整生产 runtime 还需要把模型权重、KV cache 管理、final hidden_out、多层串联和提交边界整合进同一套 runtime。

## 5.6 RTP + Runtime-Start Lock

**问题**：main16 core 在 PDI 启动后自主产生 Q/K/V compact，不依赖 host DMA。如果 c1r3 没有门控，它可能在 runtime sequence 写 current-token RTP 之前就开始执行 → current slot 变成 token0。

**方案**：
1. Runtime sequence 先写 RTP（`aiex.npu.rtp_write`）
2. 再设置 runtime-start lock（`aiex.set_lock(%post_runtime_start, 1)`）
3. c1r3 core 在执行前先 acquire 这个 lock

这建立了 host RTP 写入和 core 执行之间的明确先后关系。

## 5.7 Up/Gate 的 Reusable-Slot 策略

up/gate 共 48 条 record（24 对），不能展开成 48 个独立 BD phase（太多 BD，也无法从 up/gate 跳转到 down）。

**方案**：body-level trace = `q,k,v,o,upgate,down`

- `upgate` 是一个 48-record 长 body
- main16 把 48 条 record 逐条写入 17-dword record ping/pong stream
- row1 用 2D BD stride scatter 成 `48 × 65` column compact
- c1r1 scatter 成 `48 × 257` global compact
- c1r1 output BD 跳过每个 header，把 48 个 payload half 连续送进 c6r2

**Lock 约束**：memtile BD block 最多一个 Release。所以用 stage-level counting lock（初值=4，每个 row acquire 1，output BD 一次 release 4）。

## 5.8 Split K/V Scan

单通道 K/V scan 每个 block 需要 4 个 BD（K0/V0/K1/V1），4 个 block = 16 BD 就把 shim BD 用完。

**方案**：K 和 V 拆到两个物理 channel

```
ch0: K0/K1 per block → row1 S2MM ch0
ch1: V0/V1 per block → row1 S2MM ch1
```

早期 split 方案把每个 block 每侧降到 2 个 scan BD；当前实现进一步压缩为每侧一个 K scan BD + 一个 V scan BD，通过 `iteration_size` 和 queue repeat 扫多个 block，给 current K/V even/odd 写回保留独立 BD。

## 5.9 数值稳定性选择

- 不对 Q4NX bf16 输出做精确 hash——一个 ULP 差异会被 hash 放大成假 mismatch
- c1r2 replay 使用 bounded numeric scale（当前 256）+ 有界 int32 sqrt，对 bf16 LSB 抖动不敏感
- 生成代码优先产出有界整数/数值形式，不依赖复杂 hash 或 wide integer lowering
- Shape-A/B active path 使用显式 bf16/float carrier 和 bounded `fast_exp`，后续需继续校准到 Qwen3 bf16/fp32 reference

---

# 六、量化与数据格式

## 6.1 Q4NX 权重格式

所有权重以 4-bit 量化存储。每个 chunk 的内存布局：

```
┌─────────────────────────────────────┐
│ scales      : 32行 × 8组 × bf16     │  512 字节
│ offsets     : 32行 × 8组 × bf16     │  512 字节
│ int4_data   : 32行 × 256列 / 2      │  4096 字节
├─────────────────────────────────────┤
│ 合计                                 │  5120 字节 = 1280 dword
└─────────────────────────────────────┘
```

- 32 行 × 256 列 = 一个 chunk 覆盖 32 个输出维度、256 个输入维度
- group size = 32：每 32 个权重共享一组 scale 和 offset
- 8 组 = 256 / 32

**在线反量化**（tile 内部执行，不生成中间全精度矩阵）：
```
weight_fp = int4_value × scale + offset
output[row] += weight_fp × activation[col]
```

注意：`model.q4nx` 的第二段不是未缩放的整数 zero point。实测 Qwen3-8B-NPU2
chunk 中该字段是负的 bf16 offset（常见约 `-0.09..-0.015`），和 MyLM
`qwen3_npu` 对齐的反量化必须使用 `q * scale + offset`。旧的
`(q - zero_point) * scale` 会让 O/up/gate/down 系统性偏负，单层还能产生
自洽信号，但 36 层最终 token 会漂移。

**整层权重规模**：

| 阶段 | 输入维度 | 输出维度 | patch 数 | row-chunk/patch | 总 row-chunk |
|------|---------|---------|---------|-----------------|--------------|
| Q | 4096 | 4096 | 64 | 32 | 2048 |
| K | 4096 | 1024 | 16 | 32 | 512 |
| V | 4096 | 1024 | 16 | 32 | 512 |
| O | 4096 | 4096 | 64 | 32 | 2048 |
| up | 4096 | 12288 | 192 | 32 | 6144 |
| gate | 4096 | 12288 | 192 | 32 | 6144 |
| down | 12288 | 4096 | 64 | 96 | 6144 |
| **总计** | | | **608** | | **23552** |

这里的 row-chunk 是一个 main tile row 消费的 5120 字节 Q4NX chunk。一个 row1 patch 覆盖同列两个 main rows，所以 4096 输入维度阶段是 `2 rows × 16 input chunks = 32 row-chunk/patch`；down 是 `2 × 48 = 96`。

23552 row-chunk × 5120 字节 ≈ 120,586,240 字节 ≈ 115 MiB。这是单层 Q4NX weight BO 的主流量级。

## 6.2 bf16（bfloat16）

16 位浮点格式：1 位符号 + 8 位指数 + 7 位尾数。与 float32 共享指数范围，精度低但范围大。

Main16 projection 的激活和 compact payload 以 bf16 为主。一个 dword（32 bit）打两个 bf16：
```
dword.lo16 = value[2i]
dword.hi16 = value[2i+1]
```

但不要把这理解成所有内部 ABI 都是 bf16：当前 attention ABI 是 packed s16/fixed-point carrier；full-layer 中 Shape-B 末端再把 attention 输出转成 bf16 packet2 供 O phase 消费。MLIR 里很多 buffer 用 `i32` 承载 packed bf16 或 packed s16。

## 6.3 Compact Record（17 dword）

Main16 的输出格式：

```
┌────────────┬──────────────────────────────────┐
│ header     │ payload                           │
│ 1 dword    │ 16 dword                         │
├────────────┼──────────────────────────────────┤
│ control/id │ 32 个 bf16 = 一个 tile 的输出行   │
└────────────┴──────────────────────────────────┘
```

每个 dword payload 打两个 bf16 输出值：`payload[i] = (output[2i+1] << 16) | output[2i]`。

**Global compact（257 dword）**：16 个 tile 的 record 经 row1 column compact → c1r1 global compact 汇聚后，变成 1 header + 16×16 = 256 dword payload = 512 bf16 = 一个完整 N-block 的 512 个输出元素。

## 6.4 Full-Vector Packet0（2049 dword）

c1r2 发出的 full-vector replay 格式：

```
┌──────────┬─────────────────────────────────────────┐
│ control  │ payload                                   │
│ 1 dword  │ 2048 dword = 4096 bf16 = 完整 hidden      │
└──────────┴─────────────────────────────────────────┘
```

payload 采用自然 dim-major：`payload[i].lo16 = hidden[2i]`, `payload[i].hi16 = hidden[2i+1]`。

MyLM 的 whole-core main16 program 依赖这个 control dword 选择 normal full-layer phase path。当前 IRON active linked AIE core kernel 语义上保留 `+12/+48` replay count，但不是直接执行 MyLM whole-core program，所以不能把 MyLM raw main16 ELF 当作普通 `link_with` 函数塞进来。

**replay 次数**：
- Q/K/V 阶段：12 次（Q 8 N-block + K 2 + V 2）
- up/gate 阶段：48 次（24 up + 24 gate）
- 最终 full-vector output/run slot：1 次，当前 active full-layer 已经把 down compact 加回 residual 并输出 2048-dword hidden_out payload。

## 6.5 Shape Carrier（80 dword）

Shape-A 每处理完 16 个历史 token（一个 block），交给 Shape-B 的数据包。active full-layer 的 Qwen3 bf16 carrier 仍是 80 dword，但语义已经从旧 kv16/Q12 变成 bf16 weights + float block scalar：

```
base [0x100 = 256 字节 = 64 dword]:
  8 heads × 16 tokens 的 bf16 权重
  base[h][t] = bf16(exp(score[h][t] - block_max[h])) 的近似

scalar [0x40 = 64 字节 = 16 dword]:
  8 × (block_max, block_sum) float pair
  scalar[2h+0] = block_max[h]
  scalar[2h+1] = block_sum[h]
```

当前实现里 carrier 通过 Shape-A 本地 buffer + MM2S0 → Shape-B S2MM1 的 stream/DMA 发送，并用 empty/full lock 做生产者-消费者同步。

## 6.6 Attention 输出（2048 dword）

目标输出是 4 个 Shape-B 各产出 512 dword bf16 = 8 heads × 128 dim：

```
Shape-B 0 (c0r3): heads  0.. 7 → 512 dword
Shape-B 1 (c0r5): heads  8..15 → 512 dword
Shape-B 2 (c7r3): heads 16..23 → 512 dword
Shape-B 3 (c7r5): heads 24..31 → 512 dword
────────────────────────────────────────────
合计: 2048 dword = 4096 bf16 = 32 heads × 128 dim
```

在 c6r1 拼接后以 packet2 一次性发布。

当前 full-layer attention-O slice 已经走 `qwen3_attention_bf16_*` 路径，把 Shape-B accumulator 转成 bf16 packet2 供 Q4NX O phase 消费；旧 standalone kv16 debug attention 不再是 active frontier。生产差距在于这个 bf16 path 仍需要和 Qwen3 reference attention 逐段校准。

## 6.7 SwiGLU 输入/输出

**输入**（512 dword = c6r2 接收）：
```
input[0x000..0x1ff] = up slice   (256 dword = 512 bf16)
input[0x200..0x3ff] = gate slice (256 dword = 512 bf16)
```

**目标输出**（256 dword）：
```
output = SiLU(gate) × up = 512 bf16
```

当前 `swiglu.cc` 输出同样是 256 dword / 512 bf16 的 slice，数值路径已经是 `up * gate * sigmoid(gate)`，但 sigmoid 仍是 AIE-local bounded table 近似，尚未完成生产误差校准。

24 个 slice × 256 dword = 6144 dword = 12288 bf16 = 完整 FFN intermediate。

## 6.8 KV Cache 布局

逻辑布局：token-major 的 `K[token][kv_head][dim_pair]` / `V[token][kv_head][dim_pair]`

- 8 个 KV head/group × 128 dim = 1024 bf16/token = 512 dword/token
- 按 16-token block 对齐扫描
- NPU scan BO 使用 block-major 布局：`block → head → token_in_block → dim_pair`
- packet8/9 current writeback 直接写入 block-major BO 中当前 token 的 8 个 head slice

物理 scan：左右两侧各负责 4 个 KV head。每侧每个 K 或 V 平面用 `buffer_length=4096` dword 描述一个 rounded block；`iteration_size=<rounded_blocks>` 和 queue repeat count 让同一个 BD 反复扫描多个 block。

---

# 七、代码结构

`qwen3-layer/` 目录包含当前 fused-layer 物理实现和后续生产化开发目标：契约定义、数据流图、MLIR 生成器、NPU kernel、可运行 case 和共享工具。

## 7.1 契约与生成检查

| 文件 | 职责 |
|------|------|
| `contract.py` | 全层 ABI 的单一真相源：维度常量、phase 定义、patch/chunk 数量、packet 大小 |
| `physical_contract.py` | 生成后检查：row1 channel 所有权（S2MM4/5=weight, S2MM0-3=compact）、禁止旧路由 |
| `resource_manifest.py` | 显式 tile-local buffer/BD/lock ownership manifest，检查 main16 QKV residency 和 phase overlap |
| `check_contract.py` | 集成检查：active generator、resource manifest、token gate、retired-code absence |
| `projection_schedule.py` | Q/K/V body record 数、O/upgate/down tail weight chunk base、总 weight BO layout 的唯一派生源 |

## 7.2 MLIR 生成器

| 文件 | 职责 |
|------|------|
| `compact_dataflow.py` | row1/c1r1 compact gather + bridge + hub + row1 S2MM4/5 weight fanout 生成器 |
| `attention_dataflow.py` | Shape-A/B tile placement、hub BD、KV output BD 和 packet2 attention hub 的单一真相源 |
| `weight_stream.py` | row1 Q4NX patch-ring 生成器：host/shim weight ingress → main16 DMA1 |
| `cases/full_layer_engine_generate.py` | 当前唯一 full-layer fused-engine MLIR generator；active slice 都从这里裁剪 phase 范围 |
| `cases/currentkv_cache_dataflow.py` | current-token K/V 写回、rounded KV scan 和 Shape-A/B runtime-start helper 生成器 |
| `mlir_utils.py` | 共享 MLIR-AIE 工具：BD 声明、lock 分配、queue 配置、runtime sequence 生成、字段校验 |

`mlir_utils.py` 在生成阶段会校验：
- Lock 平衡
- Memtile BD bank 规则（偶通道用 BD 0-23，奇通道用 BD 24+）
- Scan/write BD 不重叠
- `npu.writebd` ID/field 范围
- Push-queue repeat count
- RTP 写入在 runtime-start lock 之前

## 7.3 NPU Kernel（C++）

| 文件 | 运行在 | 职责 |
|------|--------|------|
| `main_projection_q4nx_fast.cc` | c2-c5, r2-r5 | Q4NX 反量化 + MAC + flush + compact record emit |
| `edge_attention.cc` | c0/c7, r2-r5 | 当前 bf16 attention-O path 和 Shape-A/B online-softmax 近似；目标是收紧到 Qwen3 reference |
| `postprocess_qkv.cc` | c1r3 | 当前 header-stripped Q/K/V body payload → packed attention ABI + packet8/9 current K/V；目标是补齐 Q/K norm + RoPE |
| `full_vector_station.cc` | c1r2 | 当前 hidden replay、O compact replay、down compact → 2048-dword hidden_out；目标是生产 RMSNorm/residual 数值 |
| `swiglu.cc` | c6r2 | 当前 bf16 up/gate contract SwiGLU 近似；目标是生产 SiLU(gate)×up |

共享头文件：
- `record_format.h`：compact record header 布局和打包/解包 helper
- `qwen3_constants.h`：kernel 侧常量（维度、phase 边界等）

## 7.4 可运行 Case

`cases/` 目录是模块化的 case 注册机制。active case 通常由三件套组成：
- `*_generate.py`：生成该 case 的 MLIR-AIE
- `*_reference.py`：CPU 参考实现，用于结果比对
- `*_runner.py`：NPU 运行器，调用 XRT 提交任务并验证

当前不再为每个过渡实验维护独立 debug dataflow。full-layer、QKV prefix、attention-O 等边界都从 `full_layer_engine_generate.py` 派生，区别只是裁剪 phase 范围和 runner/reference 验证边界。

`run_npu.py` 是顶层 CLI 调度器：

```bash
# 检查生成 MLIR 结构
python qwen3-layer/run_npu.py --check-only

# 编译 xclbin（不跑 NPU）
python qwen3-layer/run_npu.py --build-only

# 默认 case = qwen3-8b-decode-layer
python qwen3-layer/run_npu.py

# 指定 case 和 token
python qwen3-layer/run_npu.py --case full-layer-attention-o-bf16 --current-token 31

# 真实模型 full-layer frontier
python qwen3-layer/run_npu.py --case qwen3-8b-decode-layer \
    --current-token 31 --model-path /var/home/taowen/flm/models/Qwen3-8B-NPU2
```

**当前活跃 case**：

| Case | 验证边界 |
|------|---------|
| `qwen3-8b-decode-layer` | 真实模型 full-layer frontier：hidden→Q/K/V→attention→O→up/gate→SwiGLU→down→hidden_out |
| `qwen3-8b-c1r2-input-norm-replay` | c1r2 input RMSNorm + packet0 replay 边界 |
| `qwen3-8b-qkv-cache-write-bridge` | 真实模型 Q/K/V prefix → c1r3 postprocess → packet8/9 current K/V 写回 |
| `full-layer-qkv-prefix` | 从 full-layer generator 裁剪出的 Q/K/V prefix slice，验证 main16 QKV residency 和 current K/V handoff |
| `full-layer-attention-o-bf16` | 从 full-layer generator 裁剪出的 Q→KV scan→attention→packet2→O slice |

历史 case（旧 608-patch weight-stream oracle、deterministic full-layer tail、standalone down/SwiGLU/Q4NX bridge、旧 kv16 attention debug case）已下线，源文件已经删除或从 public registry 移除；有价值约束被并入共享生成器、active reference、`physical_contract.py` 和 `resource_manifest.py`。

## 7.5 共享工具

| 文件 | 职责 |
|------|------|
| `npu_build.py` | 编译流水线：扫描 MLIR `link_with` → 编译 kernel .o → aiecc → xclbin |
| `run_stage_budget.py` | 运行 active NPU case，统一输出 c1r2、current-slot K/V、valid-cache K/V、capacity-unchanged K/V、attention-O、hidden_out 的 `stage_budget:` 统计 |
| `cases/decode_instruction_patch.py` | full-layer decode instruction patch：current token、KV write offset、scan iteration 和 queue repeat |
| `q4nx_reference.py` | Q4NX chunk 的 CPU 参考数学（反量化 + MAC） |
| `qkv_compact_reference.py` | Q/K/V/O compact record layout helper |
| `cases/full_layer_engine_reference.py` | full-layer physical reference：weight layout、cache writeback、attention/O/FFN/final hidden 验证 |
| `cases/qwen3_8b_decode_layer_reference.py` | 真实 Qwen3-8B 单层 CPU reference：RMSNorm、Q/K norm、RoPE、attention、SwiGLU、hidden_out |
| `cases/*kv16_reference.py` | 仍被 active reference 复用的 KV cache scan/layout helper；不是 public runnable debug case |

## 7.6 构建产物

`build/` 是本地生成产物目录，可能残留已删除旧 case 的历史目录。active case 运行时会生成对应子目录，通常包含：
- `design.mlir`：生成的 MLIR-AIE 源码
- `design.xclbin`：编译后的 NPU 二进制（包含 kernel ELF + 路由配置）
- `design.bin`：instruction stream（可被 patch）
- `design-tokenX-to-tokenY.bin`：patched instruction stream

`qwen3-8b-decode-layer` 的 active build 目录是
`build/qwen3-8b-decode-layer-capacity-token127/`。token0/1/31/91 等目标
token 共享同一个 `design.xclbin`，只切换 patched instruction stream。

---

# 八、当前进度与剩余差距

## 已验证（真机通过）

### 物理数据流闭环

- 7 个 projection phase（Q/K/V/O/up/gate/down）全部使用 Q4NX weight stream + main16 DMA1
- Row1 S2MM4/5 weight ingress + MM2S0-3 fanout 与 S2MM0-3 compact gather 共存
- c1r2 packet0 full-vector replay → c1r1 bridge → main16 DMA0 activation ring
- c6r1 packet2（attention）和 packet1（FFN）复用 c1r1 shared bridge
- 48-record upgate body trace + reusable-slot compact + c6r2 payload-half ABI
- `qwen3-8b-decode-layer` 能用真实 MyLM Qwen3-8B-NPU2 权重跑通单层 full-layer frontier，并输出 2048-dword hidden payload。当前比较已经接到 `Qwen3LayerReference`，hidden_out 使用 `abs_tol=0.01, rel_tol=0.05`；token31 最新真机结果是 `25.067 ms`、`final_hidden_out max_abs=0.0078125`，current K/V、valid cache 和 capacity-unchanged cache 都是 0 mismatch。这已经是可运行的单层 decode frontier，但还不是多层 production 数值预算
- main16 projection 已收敛为单一 active role `main_projection_q4nx_fast.cc`；旧 `main_projection_q4nx.cc` 已从 active generator/link path 删除。MyLM-style 17-dword ping/pong record、row1 65-dword column compact、c1r1 257-dword global compact 都已进入 active generator。之前的 program-memory overflow 是双版本 main projection 同时链接造成的；之前的 phase-sized compact mismatch 是历史问题，后续只保留 record-granular compact contract
- `run_stage_budget.py` 已把 c1r2 input norm、current-slot K/V、valid-cache K/V、capacity-unchanged K/V、attention-O 和 full hidden_out 的真机统计统一成 `stage_budget:` 输出；K/V 行会打印最大误差坐标，失败时打印首个 mismatch 坐标；token31/token91 已覆盖 qkv、attention、full stage
- `run_reference_decode.py` 已能跑真实 Qwen3-8B 多层 CPU reference 和 MyLM prefix dump 对照。当前确认 Q4NX 解码公式是 `int4 * scale + offset`，不是旧的 `(int4 - zero_point) * scale`；raw token `9707` 的 layer1/layer4/layer8/layer16/layer24/layer32 top token 与 MyLM probe 对齐。尚未确认 full 36-layer Python reference 是最终 oracle：layer35 开始出现近似误差放大，layer36 目前 Python top token 是 `11`，MyLM probe top token 是 `323`

### 稳定集成边界

- `qwen3-8b-c1r2-input-norm-replay`：真实模型 hidden + input RMSNorm 权重进入 c1r2，验证 packet0 replay ABI
- `qwen3-8b-qkv-cache-write-bridge`：真实模型 Q/K/V prefix、c1r3 packet8/9 current K/V 写回和 host cache layout
- `full-layer-qkv-prefix`：从 full-layer topology 裁剪 Q/K/V prefix，验证 full-layer main16 QKV buffer residency 不被 O/up/down 额外资源破坏
- `full-layer-attention-o-bf16`：从 full-layer topology 裁剪 Q→KV scan→attention→packet2→O，验证 production bf16 attention-O handoff，无 debug drain

### Current K/V + Attention 物理约束

- Packet8/9 当前 K/V 写回 → host KV cache BO（two-BD even/odd scatter）
- Rounded KV scan（split K/V channel）+ iterated BD + queue repeat count
- Shape-A block scoring + tail-token RTP mask
- Shape-B online-softmax 近似 merge + attention output
- Packet2 返回 → c6r1/c1r1 replay → O phase handoff

### Instruction Patch

- token127 容量 PDI 可复用，每 token 只 patch instruction stream 的方向已经在 full-layer schedule 上验证过
- 这个结论目前仍是生产 runtime 的雏形：模型权重管理、KV cache 管理、final hidden_out、多层串联和提交边界还没有整合成完整 runtime

### 关键约束验证

- Shim BD 0..15 上限（AIECC 拒绝 ID>15）
- Memtile BD bank 规则（偶通道 BD 0-23，奇通道 BD 24+）
- BD block 单 release 限制 → stage-level counting lock
- RTP-before-runtime-lock 顺序（否则 current slot = token0）
- 每个 S2MM channel 一个 row（不能用单 channel 多 BD 代替 multi-channel）

## 剩余差距

### 数值与性能收敛（最大差距）

| 模块 | 当前状态 | 目标 |
|------|---------|------|
| Attention（Shape-A/B） | bf16 attention-O path 已接入 O phase，并通过 full decode hidden_out gate | 收紧 stage-local error budget，确认 online softmax/merge 与 Qwen3 reference 的长期多层误差 |
| c1r2 RMSNorm/replay/final output | hidden replay、O residual/postnorm、down residual 和 2048-dword hidden_out 已在物理路径闭环 | 收紧 RMSNorm/residual stage budget，确认输出可直接作为下一层输入 |
| c6r2 SwiGLU | bf16 输入/输出 ABI 已对齐，`slice_scale` 已删除，执行 AIE-local bounded table `SiLU(gate) * up` | 校准到 Qwen3 SiLU/SwiGLU production budget，并优化 table/compute cost |
| c1r3 Q/K norm + RoPE | Q/K/V body payload、packet8/9 current K/V、attention ABI 和 full decode hidden_out 已接通 | 收紧 Q/K RMSNorm、RoPE、scale/rotation constant 的 stage-local budget |
| Main16 Q4NX kernel | 真 Q4NX transport/MAC 和真实模型权重 stream 已跑通；当前只保留 `main_projection_q4nx_fast.o` 一个 main16 role object，由 `main_projection_q4nx_fast.cc` 和生成的 `main_projection_q4nx_asm.s` 合成。active 数值路径已经迁到 exact per-dim rounding + signed native BF16 MAC + 32-dim full unroll，`q4nx_chunk_accum_fast` 反汇编为 `vmac.f=64`、`vextbcst.16=64`、`vextbcst.32=0`，但仍有 `vlda=282`、`vst=194`。`.s` 文件包含未引用的 MyLM-style group-shape probe，以及 `q4nx_accum_lane_exact_body_shape` 五指针 exact-lane candidate；两者都由 `tools/generate_main16_q4nx_asm.py` 生成，避免后续完整 hot body 手写寄存器轮转。当前 full-layer generator 从 `accum` 直接 emit 17-dword compact record ping/pong，activation/weight/record BD、lock、base 已迁到 MyLM ABI，`check_main16_raw_abi.py` 显示 row0 ready=true。row1/c1r1 compact tree 已经改成 MyLM-style record granularity：main16 17-dword record、row1 `17+16+16+16 -> 65`、c1r1 `65+64+64+64 -> 257`。active full-decode main16 ELF 只调用一次 linked AIE core kernel 入口，旧 MLIR phase-control 展开不再是当前根因。Source-assembly full lane smoke 已证明 8-group exact per-dim rounding 可以在真机上数值正确，当前已迁成 production role object 里的 callable candidate，但还没有接入 active `q4nx_chunk_accum_fast` | 下一步保留 topology/linked-core ABI，把 Q4NX lane hot body 改成数值等价的 source assembly，并继续保持单一 main16 role object |

### MyLM-style 主性能方向

现在的结论不是"数据流要重做"，而是"数据流已经足够接近 MyLM，main16 执行形态还不够硬"。下一阶段性能路线固定为：

1. 保留现有 full-layer topology、row1 S2MM4/5 weight ingress、row1 compact gather、c1r1/c6r1 replay、c1r2/c1r3/attention 物理边界。
2. 只保留 `main_projection_q4nx_fast.cc` 作为 active main16 C++ implementation，不再维护 QKV-only、QKVO-only、scheduled-unroll 等并存变种。
3. 所有 active slice 也统一链接这个 main16 object，只裁剪 phase/topology，不裁剪 kernel object。
   - DMA0 activation chunk：128 dword
   - DMA1 Q4NX weight chunk：1280 dword
   - output compact record：17 dword
4. full-layer 22-dim scheduled probe 证明“能放下”不等于“更快”：它保持数值正确，但 token31 full decode 从 baseline `29.236 ms` 回归到 `30.092-30.963 ms`。这个 probe 已删除，active full decode 保持 `main_projection_q4nx_fast.o`。
5. C++ helper 内部不能随意“删防御分支”：一次尝试把 partial-row/bounds branch 改成无条件写，导致 full-layer main16 ELF `.text = 0x4760` 并触发 AIE program-memory overflow，已回退。当前保留的是更小的 generator-level phase-body 清理。
6. 最新 root cause 已经从旧 phase-control 形态下移到 Q4NX microkernel 形态。fresh active full-decode `main_core_2_2.elf` 只有一次 `q4nx_main16_full_scheduler` 调用，`.text=6080`、`jl=14`、`acq=18`、`rel=18`；旧的 `129` 次 Q4 helper call/`232` 次 lock control 是历史 direct-emit 问题。工具链 probe 仍然有价值：stock aiecc 的 child `opt/llc` flags 不可透传，`elf_file` 是 whole-core 替换入口；但下一步不是再调 no-unroll metadata/replay。active C++ hot loop 已经用 signed native MAC 证明 `vextbcst.16 + vmac.f` 可以数值正确地进生产，剩余瓶颈是 wrapper 里过多 `vlda/vst/vconv` 和低 MAC 密度；下一步是把 `q4nx_chunk_accum_fast` 的 lane hot body 改成保持当前 rounding 语义的 MyLM-style source assembly。只有当新 microkernel 通过现有 reference 且 end-to-end 变快后，才替换这个唯一 active main16 linked-core implementation。
7. core program packaging 路径已经明确：transaction MLIR 里的 `aie.core` 会带 `elf_file`，随后 `.text` 被降成 `config_blockwrite_data`。fresh QKV prefix 证明 16 个 `main_projection_q4nx_fast.o` core 的 `.text=9952 bytes` 对应 16 个同尺寸 blockwrite payload；MyLM main16 image 是 `14868 bytes`。所以 raw main16 不能靠简单扩大已有 transaction payload，应该生成替换 ELF 并重新走 transaction/xclbin packaging。
8. 已新增 `tools/repack_core_program_txn.py` 作为 transaction 层检查工具：它对 fresh QKV-prefix probe 重新生成了 286 个 blockwrite global，其中 16 个 main16 payload 仍是 9952 bytes，证明没有重复 payload。但它不是最终 runner 入口，`aie-translate --aie-npu-to-binary` 输出也不是 aiecc 的小 `design.bin`。
9. 可运行入口是 `tools/package_externalized_design.py`：先从 donor transaction 复制所有 core 的 `elf_file/link_files` 回 source MLIR，把 core body 替换成 `elf_file` empty-core；再在 throwaway project dir 中替换选中的 `main_core_X_Y.elf`，最后让 aiecc 只做 packaging。probe 证明这条路能生成 `design.bin`、transaction、PDI 和 xclbin，且不会重复 core-program payload。不要只把 main16 改成 `elf_file` 后正常编译；这会把空 core 重新编译并覆盖 donor ELF。
10. raw image 包装格式已经确认：replacement core 需要 ET_EXEC + loadable `.text` `PT_LOAD`。新增 `tools/wrap_raw_aie_program.py` 后，MyLM 的 14,868-byte main16 raw image 能打进我们的 xclbin package，transaction 里出现 `bytes=14868 count=16`。但直接把 MyLM raw main16 放进 IRON 仍不会自洽，因为 raw 程序不只是 ABI，还包含 MyLM 自己的 whole-core phase program、lock phase order 和 Q4NX microkernel 内部调度。
11. IRON main16 record ABI 已迁到 MyLM-style 17-dword ping/pong：BD4/5、`0x3c1c/0x541c`、L5->L4。row1/c1r1 也已迁到 per-record compact tree：4 个 main row 先合成一个 65-dword column record，再由 c1r1 合成一个 257-dword global record；c1r3 以 3072-dword QKV payload buffer 接收 12 个 global payload。后续 compact 方向是保持这套 contract 并用静态检查防回退，不再回到旧 phase-sized receive/output 聚合。
12. MLIR-AIE/Peano 工具链结论也要更精确：`acquire_greater_equal()`/`release()` 反汇编里常出现在 `j/jl/ret` 后面，是 AIE 分支延迟槽，不能直接当作“被 hoist 到循环外”。空 `asm volatile("" ::: "memory")` 会让 AIE2P backend 崩在 IRTranslator，不能当 barrier。真正稳定的检查应该从 packet/BD/lock ABI 和最终 ELF shape 两边做，而不是只数 `acq/rel` 总数。

MyLM 公开仓库不提供 Qwen3 NPU kernel 源码；`qwen3_npu`/`qwen3_npu_sequence` 的实现来自 `src/lib/libqwen3_npu.so`，AIE 程序来自 `src/xclbins/Qwen3-8B-NPU2/layer.xclbin`。我们只复用其 ABI、program layout 和调度形态，不复用 proprietary binary。当前证据记录在 `qwen3-layer/main16_q4nx_mylm_compare.md` 和 `qwen3-layer/main16_q4nx_mylm_secret.md`。

### CPU reference 与 MyLM 对齐

当前不能再把 full 36-layer Python reference 当作已验证正确的最终 token oracle。它的价值是逐层 dump 和定位：

- layer1 logits 与 MyLM top-k 对齐，bf16 logits `max_abs=0.3671875`
- layer8/16/24/32 prefix top token 都是 `143358`
- layer35 K/V 与 MyLM 已有明显数值差异，尤其 V payload `mean_abs≈0.31`
- layer36 logits 与 MyLM 明显分叉，说明最后尾部还需要按 MyLM/NPU 数值路径继续校准

因此后续开发要优先把最后层的 `hidden_in -> Q/K/V -> attention -> O -> FFN -> hidden_out -> final_norm/lm_head` 拆成 stage dump 对齐，而不是用 full token `11` 作为成功标准。

### 生产集成

- KV cache 运行时保持 block-major scan layout（当前是 test harness 手动填充）
- 多层串联（当前验证单层）
- RMSNorm/RoPE 权重已经能从模型文件进入 aux prefix，但生产误差和 stage oracle 还要收紧
- 生产 host runtime（当前是 Python integration runner）
- 最终输出已经是 2048-dword hidden payload；后续要证明该 tolerance 在多层串联中可接受，或继续收紧到下一层输入预算
- 继续增强 stage-local mismatch 归因：K/V stats 已能输出 `token/head/dim/npu_offset`；下一步要把 attention-O、c1r2 RMSNorm、SwiGLU/down 也提升到同等级别的 head/dim/block/edge 归因

### 与 MyLM 对齐（仍需校准的布局细节）

- c1r2 内部 register-level ping/pong/value layout
- Shape carrier exact lane order
- N-block tile value order
- up/gate adjacent-pair 顺序
- Attention output head-pair order

这些不是下一步 main16 性能优化的前置条件。它们被标记为"IRON-ABI-v0 定义"；只有当我们要逐项贴近 MyLM register/lane 级数值路径，或者未来要验证 binary-level 兼容性时，才需要继续校准。

## 进展时间线摘要

```
early    deterministic contract kernel（验证物理路由）
  ↓
middle   Q4NX weight stream oracle（验证 weight ingress 能力）
  ↓
         各模块独立 bridge case（bridge/shape/swiglu/c1r2）
  ↓
         full-layer contract tail（O→up/gate→SwiGLU→down 闭环）
  ↓
         Q4NX O/up/gate/down（替换 contract kernel 为真实 Q4NX）
  ↓
         current K/V attention 边界（KV writeback + scan + Shape）
  ↓
         Q4NX Q/K/V body（替换 deterministic Q/K/V producer）
  ↓
current  qwen3-8b-decode-layer + full-layer-qkv-prefix + full-layer-attention-o-bf16
         （全部 7 phase Q4NX + current K/V + KV scan + bf16 attention-O + hidden_out frontier）
  ↓
next     MyLM-style raw/scheduled main16 Q4NX kernel → stage-local 数值预算收敛
         → attention/c1r2/c6r2 production budget → 多层 → runtime 集成
```
