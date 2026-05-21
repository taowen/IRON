<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# IRON：为什么这样设计

这份文档用问答链的方式解释 IRON 的设计决策。每个答案自然引出下一个
问题，而不是孤立地解释一个 API。


# 第一部分：为什么需要一个专门的编程框架

## 为什么 NPU 需要 IRON

GPU 有统一虚拟地址空间。程序员写 kernel，说"对这块内存做这个运算"，
硬件的 cache 层次和调度器负责把数据送到计算单元。程序员不需要知道数据
什么时候从哪里搬到哪里。

AMD AIE 阵列的硬件模型完全不同。每个 compute tile 有一块很小的本地
存储（L1，大约 64KB），没有 cache，没有虚拟地址，没有操作系统。tile
不能发出 load/store 到主存——它只能操作本地存储里的数据。

那数据怎么到达 tile？靠独立的 DMA 引擎。DMA 引擎不是 CPU，没有分支
指令，没有条件跳转。它按照一组预编程的描述符（Buffer Descriptor, BD）
机械地执行搬运：从哪个地址开始，搬多少字节，步长是多少，重复几次。

这就是核心约束：**计算单元不能自己取数据，DMA 引擎不能临时做决定。**
两者都需要外部告诉它们"做什么"，而且这个"做什么"必须在执行前完全确定。

IRON 就是把这两件事——tile 的计算逻辑 和 DMA 的搬运计划——写在同一个
Python 程序里的框架。它不是"方便的 API"，而是**唯一的途径**：如果你
不描述数据搬运，DMA 就不会搬；如果 DMA 不搬，tile 就没有数据可算。

看最简单的完整例子 `iron/operators/axpy/design.py`：

```python
# 声明数据通道：tile 从这里取数据
of_in1 = ObjectFifo(tile_ty, name="in1_0")
of_in2 = ObjectFifo(tile_ty, name="in2_0")
of_out = ObjectFifo(tile_ty, name="out_0")

# 声明 tile 的计算逻辑
def core_body(of_in1, of_in2, of_out, axpy):
    for _ in range_(N_div_n):
        elem_in1 = of_in1.acquire(1)
        elem_in2 = of_in2.acquire(1)
        elem_out = of_out.acquire(1)
        axpy(elem_in1, elem_in2, factor, elem_out, per_tile_elements)
        of_in1.release(1)
        of_in2.release(1)
        of_out.release(1)

# 声明 DMA 的搬运计划
rt = Runtime()
with rt.sequence(tensor_ty, tensor_ty, tensor_ty) as (A, B, C):
    rt.start(*my_workers)
    tg = rt.task_group()
    rt.fill(of_in1.prod(), A, tap, task_group=tg)
    rt.fill(of_in2.prod(), B, tap, task_group=tg)
    rt.drain(of_out.cons(), C, tap, wait=True, task_group=tg)
    rt.finish_task_group(tg)
```

三个部分：ObjectFifo（数据通道）、Worker（计算逻辑）、Runtime（搬运
计划）。缺任何一个，程序都不完整。这不是框架强加的结构，而是硬件要求的。


## 为什么数据搬运路径要在编译期决定

上面说 DMA 引擎靠 BD 工作。BD 里写着什么？

```python
TensorAccessPattern(
    (1, num_elements),      # 描述的是一个什么形状的大 tensor
    chunk * i,              # 从哪个偏移开始
    [1, 1, 1, chunk],       # 4 层循环的大小
    [0, 0, 0, 1],           # 4 层循环的步长
)
```

这个 `TensorAccessPattern`（TAP）最终编译成 BD 写入 runtime `.bin`。
BD 的 offset、sizes、strides 是二进制里的固定数值。NPU 运行时，DMA
引擎读取这些数值，按部就班地搬数据。

为什么不能运行时改？因为 BD 直接驱动硬件状态机。DMA 引擎没有"读一个
变量再决定搬多少"的能力——它的控制流就是"按 BD 序列一条接一条执行"。
改 BD 等于改硬件程序，而硬件程序的修改需要重新编译。

这和 Intel NPU 的 blob 模型很像。Intel NPU 把所有 DMA 任务描述符在编译时
写死在 ELF 文件里，运行时管理核只是按顺序派发。AMD AIE 的 runtime `.bin`
也是同样的性质：编译时固化，运行时按剧本执行。

这个约束决定了 IRON 的根本特征：**TAP 是编译期常量，不是运行时变量。**
写 `TensorAccessPattern(offset=26*128)` 意味着 DMA 永远从 26×128 的
位置开始搬。如果你需要从 27×128 开始搬——对不起，那是另一份编译产物。


## 为什么 compute tile 必须用 acquire/release

tile 不能 load/store 到主存，DMA 负责搬数据。但 tile 和 DMA 是两个
独立的硬件单元，同时在跑。tile 怎么知道数据已经搬到了？

答案是 `ObjectFifo` 的 acquire/release 协议：

- DMA 往 ObjectFifo 的某个 slot 写完数据 → slot 变为"可获取"
- tile 调用 `acquire(1)` → 如果有可用 slot，立即拿到指针继续算；如果没有，tile **挂起等待**
- tile 算完调用 `release(1)` → slot 归还给 DMA，DMA 可以写入新数据

这不是可选的设计模式。tile 上没有 `poll()` 或 `check_if_ready()` 这种东西。
acquire 是唯一的同步机制。如果 DMA 没有按计划往 FIFO 里灌数据，tile 就
永远挂在 acquire 上——不是报错，不是超时，是永远卡死。

反过来也一样：如果 tile 不调用 release，DMA 没有空 slot 可以写入新数据，
DMA 也会卡住。

这意味着 ObjectFifo 的 acquire/release 节奏必须和 DMA 的填充/排空节奏
**完美匹配**。不是"大部分时候匹配"，是**每次、每个 slot、每个方向都匹配**。
一次不匹配就是死锁。

这就是为什么后面谈到"多 phase Worker"时问题变得很尖锐——如果一个 Worker
写了 `acquire(B)` 但 host 没有给 B 发 fill（因为想跳过某个 phase），Worker
就永远等在那里。"跳过"不是 Worker 内部一个 if 能做的事，它是整个 dataflow
协议层的问题。


# 第二部分：编译期固化的后果

## 为什么 28 层 Transformer 能用一份编译产物跑完

一个 LLM decode step 要过 28 个 Transformer 层。每层做同样的操作（RMSNorm
→ Q/K/V 投影 → Attention → FFN），只是权重不同。

如果每层都编译一份独立的 xclbin，就需要 28 次 host→device 提交，每次的
调度开销都是实打实的延迟。

IRON 的 persistent 模式解决了这个问题。看 Qwen3 的 design：

```python
# Worker 不退出，循环 28 次
def gemv_worker(weight_fifo, x_fifo, out_fifo, matvec_kernel):
    x = x_fifo.acquire(1)
    for i_idx in range_(q_size // tile_size_output // num_columns):
        c = out_fifo.acquire(1)
        for j_idx in range_(tile_size_output // tile_size_input):
            w = weight_fifo.acquire(1)
            matvec_kernel(tile_size_input, output_row_offset, w, x, c)
            weight_fifo.release(1)
        out_fifo.release(1)
    x_fifo.release(1)
```

Worker 启动后活在 tile 上，循环 acquire 权重。DMA 按照预设的 TAP 序列，
依次灌入第 1 层的权重、第 2 层的权重……第 28 层的权重。Worker 不需要知道
"现在是第几层"——它只知道"acquire 来的下一块数据就是我要处理的"。

这能工作的前提是：**28 层的数据搬运模式可以在编译时完全展开为一个确定性的
TAP 序列。** 层数固定、每层维度相同、权重在 DDR 中按 segment-major 方式
连续排列——DMA 只要把 offset 每次跳一个 segment 的距离就行。

和 Intel NPU 的做法类似：Intel 把 32 层展开成一个扁平的 DMA+DPU+SHAVE 任务
列表，管理核按顺序派发。AMD IRON 把 28 层展开成一个 TAP 序列，DMA 引擎
按顺序执行。都是**编译时展开，运行时无循环**。


## 为什么换一个 decode 位置就需要重新编译

层与层之间维度不变，所以一份 TAP 序列搞定所有层。但 decode 位置不一样。

生成第 26 个 token 时：
- KV cache 要写入 offset = 26 × head_dim 的位置（追加新的 K/V）
- KV cache 要读取 offset = 0 到 26 × head_dim 的范围（查看所有历史）
- RoPE 角度是 θ×26（旋转位置编码的角度和位置绑定）
- Attention mask 长度是 27（当前 token 能看到 0~26）

生成第 27 个 token 时，上述四个数值全部变了。

这些数值分别住在两个地方：

**runtime `.bin`**：DMA 搬运 KV cache 的 TAP offset 和 size。TAP 是 BD
的高级表达，编译后固化为 runtime `.bin` 中的二进制数值。

**core ELF**：Worker 内部的 RoPE 角度、attention mask 长度等。这些值被编译
为 AIE 指令中的立即数，固化在 `.text` 段。

实际做过 artifact-diff 验证：把 pos26 和 pos27 的编译产物逐字节对比：

```
same-bucket pos26 -> pos27:
  runtime .bin: DMA offset words 不同
  core ELF:    .text 段的 instruction words 不同
```

两层产物同时变化。这就是为什么"换一个 decode 位置"不是改一个参数就行的事——
它需要一整套新的编译产物：新的 runtime `.bin`（DMA 路径变了）和新的 core ELF
（tile 内部的常量变了）。

当前的解决方案是 `--precompile-generate-positions`：在 token 循环开始前，
为每个可能的位置提前编译好所有 artifact，运行时根据当前位置选用对应的那套。
这消除了热循环中的编译延迟，但 artifact 数量仍然是 O(生成长度)。

对比 Intel NPU 的做法：Intel 用固定大小的 KV cache + attention mask 来避免
这个问题。NPU blob 永远处理一个 seq_len=1152 的 KV cache tensor，只是 mask
里 1 的个数不同。代价是每次 attention 都要遍历完整的 1152 长度（即使只有 10
个 token 有效）。AMD IRON 当前的做法是为每个精确位置编译一份"只读有效部分"
的 artifact——计算更精确但编译开销更大。两种都不是终态。


## 为什么 RTP 不能解决位置问题

IRON 有一个运行时参数机制：`Buffer(use_write_rtp=True)`。host 可以在运行时
写入一个标量值，Worker 启动后读取它。GEMM 的设计中用了这个机制：

```python
# 声明 RTP buffer
rtps = [[
    Buffer(
        np.ndarray[(2,), np.dtype[np.int32]],
        initial_value=np.array([0, 0], dtype=np.int32),
        use_write_rtp=True,
    )
    for col in range(n_aie_cols)
] for row in range(n_aie_rows)]

# Worker 启动后读取 RTP
def core_fn(my_rtp, barrier, ...):
    barrier.wait_for_value(1)       # 等待 host 写完 RTP
    rtp_K_div_k = my_rtp[0]        # 读取"K 维度循环多少次"
    rtp_n_tiles = my_rtp[1]        # 读取"总共处理多少 tile"
    for _ in range_(rtp_n_tiles):
        for _ in range_(rtp_K_div_k):
            ...

# Host 在运行时设置 RTP 值
rt.inline_ops(set_rtps, rtps)       # 写 K_div_k 和 n_tiles
rt.set_barrier(barrier, 1)          # 释放 Worker 去读
```

RTP 能做的事：让 Worker 内部的循环次数或分支选择变为运行时确定的，
而不是编译时固化的。

但 RTP 住在 tile 本地存储里。它影响的是 **Worker 的程序流**，不是
**DMA 的搬运路径**。位置问题有两个面：

1. DMA 搬运 KV cache 的 offset → 固化在 runtime `.bin` 的 BD 里 → **RTP 碰不到**
2. Worker 内部的 RoPE 角度 / mask 长度 → 固化在 core ELF 的立即数里 → **RTP 理论上能替代，但需要重写 kernel**

即使 RTP 能解决第二个面（Worker 从 RTP 读取位置值，自己算 RoPE 角度），
第一个面仍然解不了：DMA 不知道从 DDR 的哪个 offset 开始搬 KV cache，
因为那个 offset 写死在 BD 里。

所以 RTP 不是"解决位置问题的方案"，而是"可能解决位置问题一半的候选机制"。
另一半（DMA offset 的动态化）需要完全不同的路径——比如让 host 在每次
dispatch 前修改 BD（类似 Intel NPU 的 JIT 重定位），或者找到一种让 DMA
从 Worker 读取目标地址的机制。这些都尚未在 IRON 中被证明可行。


## 为什么不能直接 patch 编译产物

一个看起来最直接的想法：runtime `.bin` 里只有几个 offset word 变了，
直接在二进制里找到那些字节、替换成新值，不就省掉重新编译了吗？

两个原因让这条路不安全：

**第一，core ELF 也变了。** AIE 指令集没有公开的重定位表。你不知道 ELF
的 `.text` 段里哪些字节是"位置相关的立即数"，哪些是"不变的操作码"。
没有工具能可靠地定位需要 patch 的位置，更不能验证 patch 后指令仍然合法。

**第二，即使只 patch runtime `.bin`，缺乏验证手段。** BD 里的 offset/size
不只决定读哪块数据，还隐含了和其他 DMA channel 的 interleave 关系。改了
一个 BD 的 offset 后，如果它和另一个 BD 的地址范围重叠了——没有运行时错误，
只有计算结果里出现莫名的数值偏差。debug 这种问题极其困难。

对比 Intel NPU：Intel 的 blob 支持输入/输出 tensor 的 JIT 重定位（通过
ELF 的 `VPU_SHF_USERINPUT` 标记）。但那是编译器有意留下的"可重定位项"，
有明确的格式和工具链支持。AMD IRON 的 runtime `.bin` 目前没有这种设计——
所有 offset 都是"最终值"，没有"占位符"的概念。

之前也尝试过通过 ObjectFifo 元数据（在 attention 数据流中夹带位置信息）
来绕过重新编译。结果是 ERT_CMD_STATE_TIMEOUT 或 attention 输出出现 NaN
残差——机制不稳定，已被拒绝。

所以当前的 ground truth 是：**每个 decode 位置需要独立编译的 artifact，
没有已验证的 patch/复用路径。** exact-position precompile 把编译移出了
热循环，但没有减少 artifact 总量。


# 第三部分：并行度的真实边界

## 为什么不能直接用所有 tile 做一个大 GEMV

Qwen3-0.6B 的 decode 阶段，GEMV（矩阵-向量乘法）是最耗时的操作。
直觉上这是一个容易并行的操作：把权重矩阵按行分片，每个 tile 算自己
那几行，最后拼起来。NPU2 有 32 个 compute tile——用 32 路并行，速度
应该接近 32 倍？

问题出在"每个 tile 需要数据"这一步。每个 tile 需要：
- 一条从 DDR 搬权重的输入通道（ObjectFifo + fill）
- 一条把结果写回 DDR 的输出通道（ObjectFifo + drain）

每条 ObjectFifo 在 shim tile 上占用一个 DMA 端点和若干 BD。shim tile
的 DMA 端点数是**硬件固定的**。这不是配置参数，是物理上的连线数。

32 个 tile × 2 条通道（输入 + 输出）= 64 个端点。超过 shim DMA 的容量。

更具体地说，`rt.fill()` 每被调用一次，就在 shim tile 上分配一组 DMA
资源。当 fill/drain 的数量超过硬件能提供的资源时，`resolve_program()` 或
后续的 `aiecc` 编译会失败——通常报 BD 分配错误或 endpoint 超限。

实际失败案例：当前 Qwen3 路径尝试过 3-way MLP gate/up（同时做 gate
和 up 投影，每个用多列），消耗了全部 32 core。结果 Fibonacci timing
反而比 2-column 方案更慢——不是因为计算不够快，而是 DMA 带宽被分散了，
每个 tile 等数据的时间变长了。另一次尝试 row-group=8 暴露了 BD 资源
不足的硬限制。

一个重要的区分：`resolve_program(SequentialPlacer())` 只做逻辑放置——
它检查 tile 坐标和 ObjectFifo 连接是否合法，但**不检查 BD 总数是否
超限**。你可能看到 resolve 通过了，信心满满去跑 aiecc，然后在
`aie.dma_bd` lowering 阶段才报出资源不够。所以 "resolve 通过" ≠ "能跑"。


## 为什么 GEMM 能扩到多 tile 但 GEMV 不能照搬

GEMM 的 design.py 展示了解决端点问题的方法：**L3→L2→L1 三级拓扑**。

```python
# A 矩阵的输入路径：L3 → L2 → L1
A_l3l2_fifos[i] = ObjectFifo(A_l2_ty, name=f"A_L3L2_{i}")

# L2 tile 做 split：一个 L3 端点扇出给多个 L1 tile
a_tmp_fifos = (
    A_l3l2_fifos[i].cons().split(
        of_offsets,
        obj_types=[A_l1_ty] * n_rows,
        placement=Tile(i, 1),       # 放在 mem tile（L2）上
    )
)

# C 矩阵的输出路径：L1 → L2 → L3
C_l2l3_fifos[col] = ObjectFifo(C_l2_ty, name=f"C_L2L3_{col}")

# L2 tile 做 join：多个 L1 tile 的输出汇聚到一个 L3 端点
c_tmp_fifos = (
    C_l2l3_fifos[col].prod().join(
        of_offsets,
        obj_types=[C_l1_ty] * n_rows,
        placement=Tile(col, 1),     # 放在 mem tile（L2）上
    )
)
```

关键洞察：L2 mem tile 充当中继。一个 L3 DMA 端点连到一个 L2 tile，
L2 tile 内部用 split/forward/join 分发给多个 L1 compute tile。这样
4 列 × 4 行 = 16 个 compute tile 只需要 4 个 L3 输入端点（每列一个）
和 4 个 L3 输出端点——端点数是 O(columns)，不是 O(tiles)。

GEMM 能这样做是因为矩阵 B 沿 K 维度被切成多块，同一列的所有行共享
同一份 B 数据（B 广播到该列的所有行）。L2 的 forward 就是在做这个广播。
矩阵 A 沿行切分，每行有自己的片段，L2 的 split 按行分发不同的 A 块。

GEMV 的形状是 M=1：输入向量只有一行。这意味着：
- 没有 M 维度可以切分给多行 tile（GEMM 的 A 沿 M 切分的前提不存在）
- 所有 tile 需要读同一份输入向量（全广播），但输出是权重矩阵不同行的结果
- 权重矩阵沿行分片后，每个 tile 读不同的权重块——这个分发模式和 GEMM 不同

所以 GEMM 的 split/join 模式不能直接复制到 GEMV 上。GEMV 需要自己的
拓扑设计——可能是 L2 广播输入向量 + 各 tile 独立读权重 + L2 拼接输出。
但这个新拓扑的端点消耗、BD 需求、L2 tile 的存储压力都还没有验证过。

这就是为什么 experiments.md 里 Experiment 2B 专门问"GEMV 能否用 L2
split/join 代替 32 条独立的 L3 stream"——这不是显然的事，需要证明。


## 为什么性能估算不能从理论带宽推导

一个常见的误区：NPU2 的 DDR 带宽是 X GB/s，28 层 GEMV 的权重总量是
Y MB，所以理论最快时间是 Y/X 毫秒，现在用了 Z 毫秒，只要把利用率从
30% 提到 90% 就能快 3 倍。

这种推理忽略了几乎所有真实开销：

**DMA task overhead。** 每个 `rt.fill()` / `rt.drain()` 不只是"传数据"，
还有 DMA 引擎解析 BD、建立 channel、同步 ObjectFifo 的固定开销。如果把
一个大传输切成 32 个小传输（为了并行），这个 per-task 开销乘以 32。

**ObjectFifo split/join overhead。** L2 tile 的中继不是零成本的——数据
从 L2 FIFO 的一端写入、另一端读出，涉及 L2 tile 内部的 DMA 配置。

**Activation 来回趟数。** GEMV 不只搬权重。每一层的输出（1024 个 bfloat16）
要写回 DDR，下一层再读进来。28 层 = 27 次 DDR round trip。

**Runtime DMA task dispatch。** host runtime 发起每个 fill/drain 需要
写寄存器/BD，这些写操作本身有延迟。如果 28 层 × 每层 12 个 phase ×
多列 = 几千个 DMA task，dispatch 延迟会累积。

**CPU 尾部。** 当前 Qwen3 的 final norm 和 LM head 仍在 CPU 上执行。
如果 NPU 部分再快也好，CPU 尾部是固定延迟的下限。

正确的估算方式：用实测的 phase probe 数据（单层各阶段的实际耗时）作为
基准，然后预测新拓扑会改变哪些项。如果无法实测，至少要把上述每一项作为
独立的加法项列出来，而不是除以一个理论带宽了事。

另一个经常出错的地方：**用错误的维度做估算。** 之前一份外部设计文档用
Q=1536、K/V=256 来规划 tile 分配和流量。实际 Qwen3-0.6B 的维度是
Q=2048、K/V=1024。维度不对意味着每层的权重字节数差 2-4 倍，所有的
流量估算、L1 容量计算、tile 分配方案全部失效。用真实维度是前提，不是优化。


# 第四部分：调度机制的边界

## 为什么 task_group 不是通用的 phase 调度器

看 GEMM 里 task_group 的用法：

```python
tg = rt.task_group()

# 一批 fill 和 drain 操作
for col in range(n_aie_cols):
    rt.fill(A_fifo.prod(), A, tap, task_group=tg)
    rt.fill(B_fifo.prod(), B, tap, task_group=tg)
    rt.drain(C_fifo.cons(), C, tap, wait=True, task_group=tg)

rt.finish_task_group(tg)
```

task_group 的语义：**一组 DMA 操作被打包。finish 表示"等待这组操作全部
完成后再继续"。** 这是一个 DMA 级别的 barrier，不是一个 Worker 级别的
同步原语。

GEMM 中 Worker 的生命周期是：

```
host 写 RTP → set_barrier(1) → Worker 醒来 → 读 RTP → 跑完所有循环 → 结束
```

Worker 只醒来一次，读一次 RTP，然后一口气跑完。host 和 Worker 之间只有
一个同步点（barrier）。

如果想让同一组 Worker 在一次 dispatch 中执行多个 phase（比如先做 attention，
再做 MLP，对 28 层都这样），需要的是：

```
Worker 醒来
  → 跑 phase A（attention）
  → 等待 host 信号 "phase A 数据处理完毕，phase B 数据已就绪"
  → 跑 phase B（MLP gate+up）
  → 等待 host 信号
  → 跑 phase C（MLP down）
  → ... 重复 28 层 ...
→ 结束
```

这要求 host 在 Worker **运行过程中** 多次更新 RTP 和 DMA 状态。
而当前的 WorkerRuntimeBarrier 只证明了"一次性释放"：

```python
barrier.wait_for_value(1)       # Worker 等一次
rtp_K_div_k = my_rtp[0]        # 读一次 RTP
# 之后再也不看 barrier 和 RTP
```

如果让 Worker 跑到 phase B 前再次 `wait_for_value(2)`——这意味着 host
的 runtime sequence 必须在 Worker 还活着的时候执行 `set_barrier(2)` 和
更新 RTP。但 host 的 runtime sequence 本身也是一个编译期展开的操作序列——
host 怎么知道 Worker "已经跑完 phase A 了"？

GEMM 里不存在这个问题：host 在 Worker 启动前就把所有 DMA 都发出去了
（通过 task_group），Worker 自己按 ObjectFifo 的 acquire/release 节奏
消费数据。host 和 Worker 之间没有运行时的来回通信。

多 phase 方案需要的是一种**双向运行时通信协议**。task_group 是单向的
（host→DMA→Worker），没有 Worker→host 的信号路径。这是根本性的能力缺口，
不是简单的 API 扩展。


## 为什么"Worker 内部用 if 跳过 phase"不可行

直觉的设计：Worker 内部写一个循环，每轮读 RTP 决定"这一轮做什么"：

```python
def universal_worker(phase_rtp, fifo_A, fifo_B, fifo_C, ...):
    for layer in range_(28):
        phase = phase_rtp[0]
        if phase == ATTENTION:
            a = fifo_A.acquire(1)
            b = fifo_B.acquire(1)
            # ... attention 计算 ...
            fifo_A.release(1)
            fifo_B.release(1)
        elif phase == MLP:
            c = fifo_C.acquire(1)
            # ... MLP 计算 ...
            fifo_C.release(1)
```

看起来合理。但 ObjectFifo 的 acquire 是硬件阻塞的。

场景：Worker 进入 ATTENTION 分支，执行 `fifo_A.acquire(1)`。但 host 这一轮
想让这个 Worker 跳过 attention（因为 attention 被分配给另一组 tile 了），
所以 host 没有给 fifo_A 发 fill。

结果：Worker 永远挂在 `fifo_A.acquire(1)` 上。不是报错，不是超时，是永远等待。

换一种设计：host 总是给每个 FIFO 都发 fill，Worker 根据 RTP 决定用不用。
那 "跳过" 的 phase 仍然消耗了：
- DMA 引擎的一次传输（搬了数据进来但没人用）
- ObjectFifo 的一个 slot（被占用但没被 release，影响流水深度）
- shim DMA 的端点和 BD 资源（第三部分刚讲的稀缺资源）

这意味着"跳过"并没有省任何东西——反而浪费了硬件资源。

正确的多 phase 设计必须在 **dataflow 协议层** 就把 phase 边界想清楚：
哪些 FIFO 在哪些 phase 里是活跃的，非活跃的 FIFO 不分配 DMA 资源。
这可能需要 IRON 层面的新机制（比如条件 ObjectFifo、或多 runtime sequence
交替调度），而不是 Worker 内部的控制流。


## 为什么 Attention 不能留到最后再解决

如果 GEMV（权重投影和 FFN）占了 decode token time 的绝大部分，为什么
不先把 GEMV 优化到极致，attention 以后再说？

因为 attention 不只是"一个小操作"——它是**位置依赖问题最集中的地方**。

Attention 包含：
- RoPE：角度 = 位置 × 频率常数（位置变，角度变）
- KV cache 写入：写入偏移 = 当前位置 × head_dim（位置变，地址变）
- KV cache 读取：读取范围 = 0 到 当前位置 × head_dim（位置变，长度变）
- Softmax mask：有效长度 = 当前位置 + 1（位置变，mask 变）

上面讨论的"为什么每个位置需要重编译"——四个变化来源中有三个在 attention 里。
如果你的"RTP 解决位置"方案不能在 attention 上工作，那它对 GEMV 再有效
也没用——因为 attention 仍然需要 per-position 编译。

更具体地说：之前唯一一次尝试在不重编译的情况下传递位置信息给 attention
（通过 ObjectFifo 元数据通道携带位置值），就是在 attention 阶段失败的：
产生 ERT_CMD_STATE_TIMEOUT 或 NaN 残差。不是在 GEMV 阶段失败的。

所以 attention 是方案可行性的**最早判定点**。如果一个"动态位置"机制
在 attention 上行不通，整个方向需要重新评估——不是"最后 5% 的集成工作"
出了问题，而是"核心假设不成立"。

实验设计的正确顺序是：先证明 RTP 能改变简单 Worker 的行为（最小机制证明），
然后立刻在 attention 子图上验证——因为 attention 是约束最强的场景。如果
它在 attention 上通过了，GEMV 上大概率也能通过（GEMV 的位置依赖更弱）。


# 第五部分：验证方法论

## 为什么"编译通过"不等于"能跑"

IRON 程序从 Python 到硬件有多个编译阶段：

```
Python (IRON DSL)
  → resolve_program(SequentialPlacer())    # 逻辑放置，生成 MLIR
  → aiecc                                  # MLIR → 指令/BD/ELF
  → xclbin 打包                             # 最终二进制
  → NPU 执行
```

`resolve_program()` 检查的是逻辑层面的合法性：tile 坐标不冲突、
ObjectFifo 连接有效、placement 不重叠。它**不检查**：
- BD 总数是否超过 shim tile 的硬件上限
- 每个 DMA channel 的并发 task 数是否合法
- L1 存储是否装得下所有 ObjectFifo 的 buffer

这些资源约束在 `aiecc` 阶段才暴露——通常是在 `aie.dma_bd` lowering
或 endpoint 分配阶段报错。

所以一个设计如果只跑了 `resolve_program()` 就宣称"能工作"，可能只证明了
20% 的合法性。剩下 80% 的硬件约束还没有被检验。

类似地，`aiecc` 通过也不代表"运行时不会挂"。如果 ObjectFifo 的
acquire/release 和 DMA 的 fill/drain 节奏不匹配（比如 Worker 循环
次数错了），编译完全合法，但运行时 Worker 会挂在某个 acquire 上。
这种死锁在编译期完全检测不到。

所以真正的验证链是：

```
resolve_program() 通过    → 逻辑合法
aiecc 通过               → 资源合法
preflight checks 通过    → 预检合法（endpoint count、max_dma_tasks 等）
运行不挂                  → dataflow 平衡
数值匹配 F.linear        → 计算正确
多位置不变               → 复用性成立
```

每一层都可能独立失败。不能跳过中间步骤。


## 为什么 segment-major packing 在新拓扑下要重新验证

当前 Qwen3 的权重按 "segment-major" 布局：按层、按算子、按列排列在
一块连续内存中。TAP 的 offset 跳一个 segment 的距离就到下一层的权重。

这个布局是为 2-column persistent graph 设计的。每列的 GEMV Worker 读取
自己那一列的权重分片，TAP 的 offset 和 size 精确对应这个分片。

如果换一种拓扑——比如 full-array GEMV 用 8 列而不是 2 列——每列读取
的权重分片大小变了，TAP 的 offset 间隔变了。原来的 segment-major
布局可能不再对齐。

更关键的是 alignment 约束：DMA 的 BD 要求起始地址按特定字节对齐
（通常是 32 字节或 64 字节）。如果新的列数导致某个分片的起始偏移
不满足对齐，DMA 搬运会产生 strided access 或直接报错。

所以每次改变拓扑，即使算法上权重还是"按行分片给各列"，都需要重新验证：
- manifest 的 offset 是否正确
- 每个分片的 shape 和 dtype 是否符合预期
- 起始地址是否满足对齐要求
- TAP 的 sizes/strides 是否完整覆盖了分片
- 端到端数值是否匹配 `torch.nn.functional.linear`


## 为什么实验必须按特定顺序做

新 megakernel 设计中的假设之间有严格的依赖关系：

```
假设 1：RTP 能在 Worker 跨 invocation 时改变行为
  ↓ 如果不能：整个"动态位置"方向失败
  ↓ 如果能：

假设 2：同一 Worker 能在一次 dispatch 中执行多个 phase
  ↓ 如果不能：必须保持静态分配（每组 tile 固定做一种操作）
  ↓ 如果能：

假设 3：attention 的位置依赖能通过 RTP + phase 协议解决
  ↓ 如果不能：位置问题仍然需要 per-position 编译
  ↓ 如果能：

结论：可以开始在 Qwen3 layer graph 中集成新架构
```

如果从底部开始（直接改 Qwen3 layer graph），然后失败了——你不知道是
假设 1 不成立、假设 2 不成立、还是假设 3 不成立。debug 的搜索空间是
三个假设的笛卡尔积。

如果从顶部开始（先验证假设 1），失败后立即知道方向要变。通过后才检验
假设 2，再通过后才碰 attention。每一步的搜索空间是一维的。

第一个实验的验收条件：

```
同一份 xclbin
两个不同的 runtime scalar 值
Worker 行为因此不同（输出不同）
不需要重新编译
不会挂死
preflight 和 aiecc 都通过
```

如果这么简单的东西都做不到，后面所有复杂方案都不需要讨论。


# 第六部分：当前位置与完整边界

## 已证明了什么

```
一份 xclbin 跑完 28 层 persistent decode body     → 已证明（token 匹配 PyTorch）
exact-position precompile 消除热循环编译            → 已证明（多位置 precompile 后 token loop 匹配）
segment-major packing 在 2-column 拓扑下工作       → 已证明
GEMM 中 RTP + WorkerRuntimeBarrier 的单次启动      → 已证明（K_div_k 运行时确定）
task_group 做一批同构 DMA 操作的批量同步             → 已证明
L3→L2→L1 split/join 在 GEMM 形状下合法             → 已证明
```

## 没有证明什么

```
同一 xclbin 跑不同 decode 位置                     → 未证明（artifact diff 说它们不同）
RTP 跨 invocation 改变 Worker 行为                 → 未证明（GEMM 只用了一次性启动）
GEMV 形状下 L2 split/join 合法                     → 未证明（M=1 改变了拓扑需求）
task_group 做多 phase 调度                         → 未证明（缺少 Worker→host 的信号路径）
RTP 替代 core ELF 中的位置立即数                    → 未证明（需要 kernel 重写 + 跨 invocation 稳定性）
attention 子图动态化                               → 未证明（之前的元数据方案已失败）
新拓扑下 segment-major packing 仍然对齐             → 未证明
```

## 完整推理的边界在哪里

一个"megakernel"如果只覆盖 decode body（28 层 Transformer），就不是
完整的推理方案。完整的 Qwen3 推理从用户输入到文本输出涉及：

```
用户输入文本
  → tokenizer（CPU）
  → embedding lookup（CPU 或 NPU）
  → prefill（处理整个 prompt，初始化 KV cache）
  → decode token loop:
      → 28 层 Transformer（NPU）
      → KV cache 更新
      → final RMSNorm（当前 CPU）
      → LM head 矩阵乘法（当前 CPU）
      → argmax / sampling（CPU）
  → detokenize（CPU）
→ 输出文本
```

当前 IRON 只管 "28 层 Transformer" 这一步。其他所有边界都由 host CPU 处理。
每条边界都需要明确的数据传输 owner：

| 边界 | 数据 | 当前 owner | 频率 |
|------|------|-----------|------|
| 用户输入 → tokenizer | 字符串 | CPU | 每轮对话 |
| tokenizer → embedding | token_id | CPU | per-token |
| embedding → 第 1 层输入 | hidden vector (1024 bf16) | XRT buffer | per-token |
| 第 28 层输出 → final norm | hidden vector (1024 bf16) | XRT buffer → CPU | per-token |
| final norm → LM head | hidden vector (1024 bf16) | CPU | per-token |
| LM head → argmax | logits (vocab_size bf16) | CPU | per-token |
| KV cache 读 | per-layer per-head (pos × head_dim) | 编译产物内部 TAP | per-token per-layer |
| KV cache 写 | per-layer per-head (1 × head_dim) | 编译产物内部 TAP | per-token per-layer |

一个自称"新 megakernel 架构"的设计文档必须能完整填写这张表。如果只谈
decode body 的优化而不说清"数据怎么进来、怎么出去、谁管 KV cache 的
持久化"，那它是一个**算子优化方案**，不是一个**推理架构**。

对比 Intel NPU 的做法（前面用户提供的文档中描述的）：他们明确定义了
NPUW 管 KV cache 缓冲区的分配/清零/拷贝、prefill 和 generate 是两个
独立 blob、blob 之间通过 `copy_kvcache()` 传递状态。每条边界都有明确
的 owner 和生命周期。这就是"完整边界图"该有的样子。


## exact-position precompile 是基线，不是终态

当前的最佳工作方案：

```
生成 10 个 token:
  → 编译 pos17 的 artifact（假设 prompt 长度是 17）
  → 编译 pos18 的 artifact
  → ... 在 token 循环开始前一次性编译所有需要的位置 ...
  → token 循环开始：
      step 1: 选用 pos17 artifact, 执行, 得到 token
      step 2: 选用 pos18 artifact, 执行, 得到 token
      ...
```

这消除了热循环中的编译延迟（不再是"生成一个 token 前先编译一次"）。
但：
- 编译仍然发生，只是移到了循环外
- artifact 数量 = 生成 token 数
- 如果要生成 128 个 token，需要 128 份 artifact（runtime .bin + core ELF）
- 冷启动时间和存储空间线性增长

不要把"编译不在热循环里"等同于"解决了位置依赖问题"。precompile 是
一个工程上有用的基线——它让你可以先把 decode body 的其他问题（如列数
优化、phase 优化）隔离出来研究，而不被每次编译的开销干扰。但最终目标
仍然是：**同一份 artifact 能跑多个位置。** 这需要 RTP 或其他动态机制
的介入，而这些机制目前还在"未证明"列表里。
