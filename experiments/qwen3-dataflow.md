# Qwen3 单层 Decode 数据流

本文面向了解 Transformer 算法但不熟悉 NPU 硬件的读者，逐步解释 Qwen3-8B 模型的
一个 Transformer decode 层如何在 AMD XDNA NPU 上作为一台"数据流机器"运行。

---

## 第一部分：为什么需要一种新的计算方式

### 1.1 GPU 的"算完写回"模型

在 GPU 上跑一个 Transformer 层，程序员通常把计算拆成一个个独立的算子：

```
Q 投影 → K 投影 → V 投影 → attention → O 投影 → gate → up → SwiGLU → down
```

每个算子的工作流程都是：

1. 从显存（GPU 全局内存）读入上一步的输出
2. 用成百上千个 CUDA 核心并行计算
3. 把结果写回显存
4. 通知下一个算子来读

这个模式简单直观，但有一个代价：**中间结果反复搬运**。一个 4096 维的向量被计算
出来后写入显存，下一个算子马上又把它读出来。显存带宽虽然很高（几百 GB/s），但当
模型足够大、推理足够频繁时，这些"写了又读"的操作累积起来就成了瓶颈。

打个比方：这像一间工厂，每道工序做完都要把半成品送回仓库，下一道工序再从仓库
取出来。工人很快，但搬运的时间浪费了。

### 1.2 NPU 的设计哲学：让数据流动起来

AMD XDNA NPU 的设计从一开始就想绕开这个问题。

它不是一块大的统一计算阵列加统一显存，而是由许多小的计算单元（称为 tile）排成
一个阵列。每个 tile 只有几十 KB 的本地存储，但 tile 之间有专用的硬件连线，可以
直接把数据从一个 tile 传到相邻的 tile，**不经过主存**。

这意味着：如果我们能提前规划好数据的流向，就可以让一个 token 的大部分中间激活
从进入 NPU 的那一刻起，在各个 tile 之间流动完成计算，只在必须持久化的边界写回
主存。KV cache 读写、输入/输出 hidden 和权重仍然经过主存。

打个比方：这像一条真正的流水线车间 — 半成品从一个工位直接传到下一个工位的手里，
不回仓库。

### 1.3 什么是"融合层引擎"

"融合"的意思是：**不把 Transformer 层拆成一个个独立算子依次调用，而是把整层
做成一台预先配好的数据流机器。**

具体来说：

- **配置阶段**：主机（CPU）把所有路由、缓冲区、同步信号配置好，告诉 NPU "权重
  在这个地址，KV 缓存在那个地址，输入在这里"。
- **执行阶段**：NPU 自动运转 — 权重流入、激活在 tile 间传递、注意力在片上完成、
  结果流出。主机不参与中间调度。

设计目标是：跑完整个 Transformer 层（7 个投影阶段 + 归一化 + attention +
激活函数），NPU 只需要一次启动，主机不需要反复提交 kernel。整层数据流由
七阶段投影骨架、attention-to-O 直接 handoff、SwiGLU-to-down 共享桥、c1r2
全向量站、c6r2 SwiGLU 切片站，以及 main16 紧凑 record 协议共同组成；所有
边界与 lane 顺序在本文档后续章节给出明确定义。

| | GPU 方式 | NPU 融合方式 |
|---|---|---|
| 启动次数 | 每个算子一次 kernel launch | 整层一次启动 |
| 中间结果 | 写回显存，下个算子再读 | 大部分在 tile 之间直接流动 |
| 主机参与 | 每步都要调度 | 只在开始时配置地址 |

这就是"融合层引擎"的含义：不是把算子一个个调用，而是搭建一台数据流机器，
开机一次，跑完整层。

---

## 第二部分：硬件地图 — 认识 NPU 阵列

### 2.1 三层结构

NPU 阵列从下到上分三层，每层承担不同职责：

```
┌─────────────────────────────────────────────────┐
│  Row 5  ┃  计算层（Compute Tile）               │
│  Row 4  ┃  每个 tile 是一个小处理器，           │
│  Row 3  ┃  跑 C/C++ 程序做实际运算              │
│  Row 2  ┃                                       │
├─────────╋───────────────────────────────────────┤
│  Row 1  ┃  中转层（Memtile）                    │
│         ┃  较大的片上缓存，负责拆分/缓冲/转发   │
├─────────╋───────────────────────────────────────┤
│  Row 0  ┃  接口层（Shim Tile）                  │
│         ┃  主存入口 — 数据从这里进出 NPU        │
└─────────────────────────────────────────────────┘
```

**接口层（Row 0，Shim）**：NPU 的大门。主存里的数据（权重、输入向量、KV 缓存）
都要通过这里进入芯片，计算结果也从这里写回主存。

**中转层（Row 1，Memtile）**：比计算 tile 有更大的存储空间（几百 KB），但它不做
复杂计算。它的主要工作是：
- 接收从主存流入的大块数据
- 把大块拆成计算 tile 能消化的小块
- 用 ping-pong 缓冲让传输和计算重叠
- 把多路结果汇聚起来

**计算层（Row 2-5，Compute Tile）**：真正干活的地方。每个 tile 是一个小型处理器，
有自己的本地内存（几十 KB）和 DMA 引擎。它跑编译好的 C/C++ 程序 — 在我们的
设计中主要是"读入权重块 + 读入激活 → 乘累加 → 输出结果"。

### 2.2 8 列 × 6 行的棋盘

在本设计中，NPU 使用 8 列 × 6 行的分区。每个 tile 用 `c{列}r{行}` 标记：

```
       c0     c1     c2     c3     c4     c5     c6     c7
      ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
row5  │      │      │      │      │      │      │      │      │
      ├──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┤
row4  │      │      │      │      │      │      │      │      │
      ├──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┤
row3  │      │      │      │      │      │      │      │      │
      ├──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┤
row2  │      │      │      │      │      │      │      │      │
      ├──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┤
row1  │      │      │      │      │      │      │      │      │  ← Memtile
      ├──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┤
row0  │      │      │      │      │      │      │      │      │  ← Shim
      └──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘
```

共 48 个格子，其中 32 个是计算 tile（row2-5），8 个是 memtile（row1），8 个是
shim（row0）。

**硬约束**：tile 之间的连线是物理的、有限的。不能像软件里那样"任意两个模块通信"
— 如果两条数据流需要走同一根物理线路，就会冲突。这也是为什么布局设计如此重要。

### 2.3 数据怎么在 tile 之间移动

NPU 内部不靠 CPU 搬数据。它有一套专用的硬件机制：

**DMA（直接内存访问）**

每个 tile 都有自己的 DMA 引擎 — 可以理解为一个"自动搬运工"。你给它一份任务单，
它就按单子把数据从一个地方搬到另一个地方，不需要 CPU 介入。

每个 tile 有多个 DMA 通道（channel），可以同时搬运多份数据。

**BD（缓冲区描述符 — DMA 的"任务单"）**

BD 告诉 DMA：
- 从哪个缓冲区的哪个偏移开始读（或写）
- 搬多少数据
- 搬完之后下一个任务是什么（链式 BD）
- 是否要给数据打上 packet ID（用于路由）
- 开始前要等哪个锁，完成后要释放哪个锁

一个 BD 就像一张"快递单"：写明取件地址、送达地址、包裹大小、完成后通知谁。

**锁（Lock）— 同步信号**

锁是生产者和消费者之间的"交接信号"：

```
生产者填满缓冲区 A → 释放"A 满了"锁
消费者等到"A 满了"锁 → 取走数据 → 释放"A 空了"锁
生产者等到"A 空了"锁 → 继续填 A
```

这就是 ping-pong 缓冲的原理：两块缓冲区交替使用，一块在被计算时，另一块在被
填充。锁保证两边不会打架。

如果锁配错了（比如生产者释放了一个没人等的锁），硬件就会死锁 — 永远等下去。
这是 NPU 编程中最常见的 bug 来源之一。

**流（Stream）— 物理连线**

Tile 之间有固定的物理连线。数据沿着这些线从一个 tile 流向另一个 tile。
编译器负责把逻辑上的"A 要把数据发给 B"映射到物理上的具体路线。

**Packet 路由 — 在一条线上跑多路数据**

有时候我们希望同一根物理线上能传输不同目的地的数据。Packet 路由就像在每个数据包
上贴一个标签（packet ID），接收端根据标签决定"这个包是给我的"还是"让它继续走"。

这类似于一条公路上跑着不同颜色的卡车，每个收货站只接收自己颜色的卡车。

```
                    packet ID = 14
tile A ──────────────[■]──────────────→ tile X（匹配 ID 14，接收）
                      │
                      └── packet ID = 15
                           ──────────→ tile Y（匹配 ID 15，接收）
```

在我们的设计中，packet 路由大量使用在 KV 缓存写入（packet14 写 K，packet15
写 V）和 attention 结果返回（packet2）的路径上。

---

## 第三部分：分工 — 谁干什么活

### 3.1 Main16：16 台相同的车床

NPU 阵列中间 4 列（c2-c5）× 4 行（row2-5）= 16 个计算 tile 组成**主计算区**。

```
       c0  c1 ┃ c2   c3   c4   c5 ┃ c6  c7
row5          ┃[M13][M14][M15][M16]┃
row4          ┃[M09][M10][M11][M12]┃
row3          ┃[M05][M06][M07][M08]┃
row2          ┃[M01][M02][M03][M04]┃
              ┃     Main16         ┃
```

这 16 个 tile 的特点：

**完全相同** — 它们跑一模一样的程序，只是各自处理不同的输出行。就像 16 台型号
相同的数控机床，加工同一种零件的不同部分。

**只做一件事** — 矩阵-向量乘法。具体来说：
- 输入通道 0 接收 256 个 bf16 激活值（输入向量的一小段）
- 输入通道 1 接收一个 5120 字节的权重块
- 内部做乘累加，产出 32 个输出值

**视野极窄** — 每个 tile 一次只看到 32 行 × 256 列的"窗口"。它看不到完整的
4096 维向量，也不知道自己是在算 Q 投影还是 K 投影。它只管"拿到数字就乘起来"。

**时分复用** — 这是最巧妙的设计。同一组 16 个 tile 依次执行 7 个不同的投影阶段：

```
阶段 1: Q 投影   (4096 → 4096)
阶段 2: K 投影   (4096 → 1024)
阶段 3: V 投影   (4096 → 1024)
阶段 4: O 投影   (4096 → 4096)
阶段 5: up 投影   (4096 → 12288)
阶段 6: gate 投影 (4096 → 12288)
阶段 7: down 投影 (12288 → 4096)
```

tile 不需要重新配置或重新分配。每个阶段只是换了流入的权重 — 对 tile 来说，
"又来了一批 5120 字节的权重块"，照常乘累加就行。

物理 projection 顺序是 `Q, K, V, O, up, gate, down` —— up 先于 gate 在
main16 上执行。算法上 SwiGLU = SiLU(gate) × up 仍然成立；c6r2 在接收端按
slice 配对，每个 slice 的 up payload 落在输入缓冲区低半区（`0x000..0x3ff`）、
紧跟着的 gate payload 落在高半区（`0x400..0x7ff`），合成一组完整 SwiGLU 输入。

### 3.2 Edge16：一条异构装配线

两侧 4 列（c0, c1, c6, c7）× 4 行 = 16 个 tile 组成**边缘辅助区**。

与 Main16 完全不同的是：**这 16 个 tile 每个都干不同的活**。它们不是 16 台相同
的机床，而是一条装配线上的不同工位，每个工位有不同的程序、不同的输入输出格式、
不同的职责。

```
       c0         c1        ┃ main16 ┃    c6         c7
row5  [求和-B]   [    ]     ┃        ┃   [    ]    [求和-B]
row4  [评分-A]   [    ]     ┃        ┃   [    ]    [评分-A]
row3  [求和-B]   [后处理]   ┃        ┃   [    ]    [求和-B]
row2  [评分-A]   [中转]     ┃        ┃   [辅助]    [评分-A]
row1  [KV整形]   [返回桥]   ┃  行1   ┃   [枢纽]    [KV整形]
row0  [K写回]    [────────── shim ──────────────]   [V写回]
```

各工位的职责：

**后处理站（c1r3）— "加位置信息"**

这个 tile 接收完整的 Q+K+V 向量（3072 dword）以及 q/k norm 权重和 RoPE 常量
（192 dword），执行：
- 对 Q 和 K 做 RMS 归一化（让数值范围稳定）
- 对 Q 和 K 做 RoPE 旋转位置编码（让模型知道"这是第几个词"）

处理后输出：2048-dword Q 向量经片上电路发给 attention 侧，K/V 经 packet14/15
写入主存缓存。Q/K norm 在 RoPE 之前完成；V 不做 norm 也不做 RoPE。

**分发枢纽（c6r1 memtile）— "分拣中心"**

这是一个 Row1 的 memtile，不是计算 tile。它承担两个方向的工作：
- 前向：接收完整的 Q 向量（2048 dword），拆成 4 份，发给 4 个评分 tile
- 反向：接收 4 个求和 tile 的结果，拼成一整块，转发给返回桥

**评分工位（shape-A）— "打分员"，共 4 个（c0r2, c0r4, c7r2, c7r4）**

每个接收一个 512-dword Q 窗口（= 1024 bf16 = 8 个 query head × 128 dim），
对应 2 个 KV group。4 个 tile 合计覆盖全部 32 个 query head。Shape-A 以
**16 个历史 token 为一个 block** 推进：
- 用当前 Q 和该 block 内 16 个历史 K 做点积，得到 8 head × 16 token 的
  原始 score
- 在 block 内做局部 softmax，记录每个 head 的 `block_max` 与 `block_sum`
- 把该 block 的 16 个权重 + max/sum 通过本地 carrier 交给配对的 shape-B

**求和工位（shape-B）— "加权平均员"，共 4 个（c0r3, c0r5, c7r3, c7r5）**

每个 shape-B 与对应 shape-A 物理相邻。Shape-A 通过本地相邻存储 + lock
immediate 同步把 carrier 交给 Shape-B —— 不是 DMA tensor stream，而是
轻量的片上交接。每次交接的 carrier 由两块组成：

```
base   [0x100] = 256 字节
                 head-major 8 records × 16-token bf16 权重
                 base[h][t] = exp(score[h][t] - block_max[h])
scalar [0x040] =  64 字节
                 8 × (block_max, block_sum) fp32 pair
                 scalar[2h+0] = block_max[h], scalar[2h+1] = block_sum[h]
```

Shape-B 本地状态（共 0x1000 字节累加器 + 64 字节 running 状态）：

```
accum[8][128]      fp32   每个 head 的 weighted-V 累加
running_max[8]     fp32
running_sum[8]     fp32
```

Shape-B 每收到一个 16-token block 的 carrier：
- 用 base 中的 16 个权重对该 block 内 16 个历史 V 做加权求和，得到
  block_weighted_v
- 用 carrier 的 block_max / block_sum 与 running_max / running_sum 做标准
  online-softmax merge：

```
new_max   = max(running_max, block_max)
old_scale = exp(running_max - new_max)
new_scale = exp(block_max  - new_max)
accum     = accum       * old_scale + block_weighted_v * new_scale
running_sum = running_sum * old_scale + block_sum       * new_scale
running_max = new_max
```

历史扫描结束后输出 `accum / running_sum`，得到 512 dword（= 1024 bf16 =
8 heads × 128 dim）的 attention 结果。

**返回桥（c1r1 memtile）— "格式转换站"**

接收枢纽拼好的完整 attention 结果（2048 dword），拆成 16 份 O 输入切片（每份
256 bf16 = 128 dword），扇出到全部 16 个 main tile。这让 main16 能把 attention
结果当作普通激活继续做 O 投影。

**KV 整形（c0r1, c7r1 memtile）— "历史供给站"**

按 KV group 分工：c0r0 从主存扫描 group 0-3 的 K 和 V（k03/v03），送入 c0r1；
c7r0 扫描 group 4-7 的 K 和 V（k47/v47），送入 c7r1。Row1 memtile 内部再做
K/V 分流：
- K 历史经 ring buffer（地址 0x20000/0x24000 区域）→ 推给 shape-A
- V 历史经 ring buffer（地址 0x28000/0x2c000 区域）→ 推给 shape-B

每侧 K 和 V 各自拆成两路 2048-dword head-pair stream（即每侧共 4 路）。

**全向量站（c1r2）— "归一化 / 残差 / 最终输出"**

c1r2 是承载 2048-dword 全 hidden 向量流和 sum-of-squares / 倒数平方根代码的
tile。它担任：
- 输入 hidden 的第一次 RMSNorm（分发给 Main16 之前）
- O 投影后的残差加 + 第二次 RMSNorm（喂给 gate/up）
- down 投影后的最终残差 + 层输出

之所以放在这里：RMSNorm 和残差都必须看到完整 4096-bf16 hidden 向量，c1r2 的
2048-dword full-vector ABI、sum-of-squares/rsqrt 风格代码、以及 RMSNorm 权重
boundary 都精确匹配这个角色。

**SwiGLU 切片站（c6r2）— "FFN 激活工位"**

c6r2 接收 512-dword 输入（512 bf16 gate slice + 512 bf16 up slice），执行
`SiLU(gate) × up`，输出 256-dword（= 512 bf16）SwiGLU slice。一共 24 个 slice
拼出 12288-bf16 FFN intermediate。

**FFN 汇聚站（c6r1，复用枢纽 memtile）—**

c6r1 在 attention 之外的另一个角色：把 24 个 SwiGLU slice 在 6144-dword
gather buffer 中拼成完整的 12288-bf16 FFN intermediate，再用 packet0 发布给
down 投影。

### 3.3 为什么要分成两组

核心原因：**矩阵乘和其他操作对硬件的需求完全不同。**

| | Main16 需要 | Edge16 需要 |
|---|---|---|
| 视野 | 极窄（32×256 小块） | 全局（完整 4096 维） |
| 运算 | 只有乘累加 | 归一化/softmax/SiLU/逐元素乘 |
| 数据模式 | 规则、重复、大量 | 不规则、一次性、小量 |
| 程序 | 16 个 tile 完全相同 | 每个 tile 不同 |
| 流水线 | 能高效 ping-pong | 需要灵活调度 |

如果把所有操作都塞进一种 tile：
- 做矩阵乘时需要的"窄接口高吞吐"会被非线性操作拖慢
- 做 attention 时需要的"全局视野"又是矩阵乘 tile 提供不了的

所以分成两组是自然的选择：**Main 是 16 台标准化的数控机床（只会切削），Edge 是
质检、喷漆、组装等专用工位。** 数据在两组之间来回流动，就像半成品在机床和
装配线之间来回一样。

---

## 第四部分：数据的旅程 — 一个 token 的完整经历

现在让我们跟随一个 token，看它从进入 NPU 到离开的完整过程。

### 4.1 起点：一个 4096 维向量从主存进入 NPU

一个 token 的隐藏状态是什么？就是一个 4096 个数字组成的向量（bf16 格式，
共 8192 字节）。这个向量编码了模型对这个 token 当前的"理解" — 它的语义、位置、
上下文信息都浓缩在这 4096 个数字里。

同时准备好的还有：
- 本层的全部权重（Q/K/V/O/gate/up/down 共约 115 MB），存在主存中
- 历史 KV 缓存（之前所有 token 的 K 和 V），也在主存中

NPU 不会一次性把 115 MB 权重全读进来。它按阶段分批流入 — 先流 Q 的权重，
用完再流 K 的权重，如此循环。

### 4.2 热身：RMSNorm 归一化

在做任何投影之前，输入向量需要"归一化" — 把 4096 个数字的范围调整到稳定区间。

**RMSNorm 做什么：**
1. 计算所有 4096 个数字的平方均值
2. 除以这个均值的平方根
3. 乘以一组可学习的缩放参数

这一步必须看到完整的 4096 维向量（因为要算均值），所以它在 Edge 侧的 c1r2
全向量站上完成。归一化后的向量被送入 Row1 memtile，准备分发给 Main16。c1r2
也承担后续 O 后的残差+RMSNorm 和最终残差，是 hidden/RMSNorm/residual/final-output
所有 full-vector 工作的归口。

### 4.3 第一轮投影：Q / K / V

**Q、K、V 是什么（一句话版本）：**
- Q（Query）= "我在找什么信息"，4096 维
- K（Key）= "我能提供什么线索"，1024 维
- V（Value）= "我携带的实际信息"，1024 维

为什么 K 和 V 只有 1024 维？这是 GQA（分组查询注意力）的设计 — 每 4 个 query
head 共享 1 组 key/value head。这节省存储和计算，但不影响质量。

**投影的本质**：矩阵 × 向量。比如 Q 投影就是把一个 4096×4096 的权重矩阵和
输入向量相乘，得到 4096 维的 Q 向量。

**在 NPU 上怎么做这个巨大的矩阵乘**：

不可能一次做完。4096×4096 的权重有 1600 多万个参数。策略是拆成小块，每块在一个
tile 上算：

```
整个权重矩阵
  → 切成 "patch"：每个 patch = 64 输出行 × 4096 输入列
  → 每个 patch 再切成 "chunk"：32 输出行 × 256 输入列 = 5120 字节
```

每个 Main tile 的工作流程：

```
重复 16 次 {
    从输入通道 0 接收：256 个 bf16 激活值（输入向量的第 i 段）
    从输入通道 1 接收：一个 5120 字节权重 chunk
    在线反量化权重：(4bit值 - 零点) × 缩放
    乘累加到本地 32 个 fp32 寄存器
}
// 16 × 256 = 4096，覆盖完整输入维度
// 32 个输出行的结果就绪
```

16 个 Main tile 并行工作，每个负责 32 行。4 列各处理 2 个 patch（每列 4 tile ×
32 行 = 128 行 = 2 个 64 行 patch），4 列合计 = 8 个 patch/轮（称为一个 N-block）。
Q 投影有 64 个 patch = 8 个 N-block。

**激活广播**：同一列的 4 个 tile 需要相同的激活切片。Row1 memtile 用 4 路 DMA
把同一份 256 bf16 分别推给 row2/3/4/5。

**流水线**：前一个 chunk 在被计算时，下一个 chunk 已经在传输途中（ping-pong
缓冲）。计算和传输完全重叠，没有空等。

K 投影和 V 投影与 Q 完全相同，只是输出维度更小（1024 维 = 16 个 patch），
而且流入不同地址的权重。

### 4.4 后处理：RoPE 位置编码

三个投影完成后，Q（4096 bf16）+ K（1024 bf16）+ V（1024 bf16）全部到达后处理
tile（c1r3）。

**RoPE 位置编码做什么**：

Transformer 本身不知道 token 的顺序。RoPE 的做法是把向量按维度两两配对，每对
旋转一个与"位置"相关的角度 — 位置越远，旋转角度越大。这样模型就能从向量的
数值中"读出"两个 token 之间的相对距离。

c1r3 同时对 Q 和 K 做 RoPE（V 不需要位置编码）。

**处理完后，输出分两路走：**

**路线 A — Q 发给 attention（片上流动）：**

```
c1r3 → 片上电路 → c6r1 枢纽
                    ├→ 窗口0 (heads 0-7)   → c0r2 评分 tile
                    ├→ 窗口1 (heads 8-15)  → c0r4 评分 tile
                    ├→ 窗口2 (heads 16-23) → c7r2 评分 tile
                    └→ 窗口3 (heads 24-31) → c7r4 评分 tile
```

完整的 Q 向量（32 个 head × 128 维 = 2048 dword）被拆成 4 个 512-dword 窗口，
每个窗口 = 8 个 query head × 128 dim。每个 attention tile 只看到自己负责的
那一组。

**路线 B — 当前 K/V 写入缓存（走 packet 路由到主存）：**

```
K: c1r3 → packet14 → 向南穿过多个 tile → c0r0 shim → 写入主存 KV 缓存
V: c1r3 → packet15 → 向东穿过多个 tile → c7r0 shim → 写入主存 KV 缓存
```

当前 token 的 K 和 V 被存入 KV 缓存，供将来的 token 做 attention 时使用。

### 4.5 Attention：找到相关信息

**直觉**：当前 token 拿着自己的"问题"（Q），回头看所有历史 token 的"标签"（K），
算出"相关度分数"，然后按分数加权提取历史 token 的"信息"（V）。

**硬件实现分三步：**

**第一步：读出历史 KV 缓存**

历史的 K 和 V 存在主存的 KV 缓存 BO 中。按 KV group 分工扫描：
- c0r0 shim 读出 group 0-3 的 K 和 V（k03/v03）→ 送入 c0r1 memtile
- c7r0 shim 读出 group 4-7 的 K 和 V（k47/v47）→ 送入 c7r1 memtile

Row1 memtile 内部做 K/V 分流：每侧 K 和 V 各自拆成两路 2048-dword head-pair
stream — K 历史送 shape-A，V 历史送 shape-B。

**第二步：Shape-A 评分（c0r2, c0r4, c7r2, c7r4）**

每个 Shape-A tile 拿到：
- 自己那组的当前 Q 窗口（512 dword = 8 heads × 128 dim）
- 逐块流入的历史 K

工作方式：
```
对每个历史 token 的 K:
    score = Q · K / √128        // 点积除以缩放因子
    更新 running_max            // 在线 softmax：边扫描边更新最大值
    更新 sum_exp                // 累加 exp(score - max)
```

这就是"online softmax" — 不需要先把所有 score 算完再做 softmax，
一边扫描一边就能得到最终结果。大幅节省存储。

每凑够 16 个历史 token，Shape-A 把该 block 的 carrier
（`base[0x100]` = 8 heads × 16 token bf16 权重，
`scalar[0x40]` = 8 × (block_max, block_sum) fp32 pair）经本地相邻存储
+ lock immediate 同步交给 Shape-B，详见 3.2 节 Shape-B 工位的 carrier 模型。

**第三步：Shape-B 加权求和（c0r3, c0r5, c7r3, c7r5）**

每个 Shape-B tile 拿到：
- Shape-A 每 block 传来的 carrier（权重 + max/sum）
- 与之配对的 16-token V block

工作方式：
```
对每个 16-token block:
    block_weighted_v = sum(weight[t] × V[t]) for t in 0..15
    online-softmax merge:
        accum         = accum * old_scale + block_weighted_v * new_scale
        running_sum   = running_sum * old_scale + block_sum  * new_scale
        running_max   = max(running_max, block_max)
```

历史扫描结束后输出 `accum / running_sum`，每个 Shape-B tile 产出 512 dword
（= 1024 bf16 = 8 heads × 128 dim）的 attention 结果。

### 4.6 Attention 结果返回主计算区

Attention 完成后，结果必须回到 Main16 — 因为接下来是 O 投影（又一个大矩阵乘）。

**关键：结果不经过主存，直接从 Edge 流回 Main。**

返回路径：

```
4 个 Shape-B tile 填入 c6r1 的 2048-dword return buffer，按 head 顺序拼接:
  c0r3 (512 dw)  → window0  (heads  0.. 7)
  c0r5 (512 dw)  → window1  (heads  8..15)
  c7r3 (512 dw)  → window2  (heads 16..23)
  c7r5 (512 dw)  → window3  (heads 24..31)

四个 return window 通过链式 lock 串成顺序依赖，全部填满后 c6r1 才一次性
发布完整的 2048-dword 块（packet2）→ c1r1 DMA4 共享桥。

c1r1 共享激活桥（DMA4 ↔ DMA1，256-dword ping-pong）：
  2048 dword / 256 = 8 次 bridge 迭代，每次 multicast 256 dword
Main16 入口（DMA0 128-dword activation ring）：
  256 dword / 128 = 每次 bridge 切成 2 个 main16 chunk
  共 16 个 O chunk，chunk c 对应 attention head 2c 与 2c+1
```

注意：packet2 的发布不是"来一块转一块"，而是严格等待四个 512-dword return
window 全部就位后，一次性发布完整的 2048-dword 块；lock 的链式依赖保证了这
个顺序。

**c1r1 的桥不是 attention 专用** —— 它是 c6r1 packet 源（attention packet2、
FFN packet0）→ c1r1 DMA4 → DMA1 multicast → main16 DMA0 ring 的统一通路，
attention-to-O 和 SwiGLU-to-down 复用同一条物理桥（见 4.9）。

从 Main16 的视角看：输入通道 0 上又来了 256 个 bf16 数字。跟之前做 Q/K/V 投影时
接收激活切片一模一样。Main tile 不知道也不关心这些数字是来自主存还是来自 Edge —
它只管拿到数字就跟权重做乘累加。

### 4.7 O 投影

与 Q 投影执行方式完全相同：
- 输入通道 0：attention 结果的 256 bf16 切片（从 Edge 直接流入）
- 输入通道 1：O 权重 chunk（从主存经 Row1 流入）
- 输出：4096 维向量

64 个 patch，和 Q 投影一样的规模。

### 4.8 残差连接 + 第二次 RMSNorm

O 投影完成后，按 Transformer 算法需要做两件事：

1. **残差加**：`hidden = 原始输入 + O 投影输出`

   为什么需要残差？这是深度网络的标准技巧 — 让梯度更容易传播，防止模型"忘记"
   输入信息。

2. **RMSNorm**：再做一次归一化，为 FFN 做准备

两个操作都需要看到完整的 4096 维向量，由 **c1r2 全向量站**承担。Main16 的 17-dword
O 输出记录走 compact/aux 路径汇入 c1r2，c1r2 在本地重建/累加完整 4096-bf16
O 结果，与之前缓存的 hidden 残差相加，再施加第二次 RMSNorm 权重，归一化后的
完整 hidden 重新喂回 Main16 入口，作为 gate/up 投影的激活。

c1r2 同时承担三种 full-vector 工作：进入 attention 前的第一次 RMSNorm、O 后的
残差 + 第二次 RMSNorm、down 后的最终残差与层输出（见 4.10）。

host-visible layer boundary 上**不暴露** O / 残差 / RMSNorm 的 DDR
descriptor，所有这些状态都留在片上由 c1r2 维护。c1r2 内部维护一份 2048-dword
（4096 bf16）full-vector buffer，提供 sum-of-squares / rsqrt 计算做归一化，
并把归一化后的 full-vector 通过 packet0 切成 16 个 256-bf16 chunk 喂给
main16。每个 phase 的 chunk 数量与 main16 的 N-block 数对齐：

```
+12 = Q/K/V 输入 chunk replay  (8 + 2 + 2 N-blocks)
+48 = up/gate 输入 chunk replay (24 + 24 slices)
+1  = 最终 hidden 输出
```

c1r2 的 full-vector layout 采用自然 dim-major：`hidden[dim]`，dim = 0..4095，
每个 dword 打两个 bf16（`payload_dword[i].lo16 = hidden[2i]`，
`payload_dword[i].hi16 = hidden[2i+1]`），第 c 个 chunk 覆盖
`dim c*256..c*256+255`。Q/K/V 输入 replay、up/gate 输入 replay、最终 hidden
输出共用同一个 full-vector layout。

### 4.9 FFN：SwiGLU 结构

FFN（前馈网络）使用 SwiGLU 结构。按 Transformer 算法，需要完成以下计算：

**Up / Gate 投影（Main16，物理 order：up 先于 gate）**

各为 4096 → 12288 的矩阵乘（每个 192 个 patch）。物理 projection 顺序是
`Q, K, V, O, up, gate, down` —— main16 上 up 先于 gate 执行。算法上
SwiGLU = SiLU(gate) × up 仍然成立；c6r2 在接收端按 slice 配对，每个 slice
的 up payload 落在输入缓冲区低半区（`0x000..0x3ff`）、gate payload 落在高
半区（`0x400..0x7ff`），合成一组完整 SwiGLU 输入。

最终需做逐元素相乘：

```
FFN_intermediate = SiLU(gate_output) × up_output
```

SiLU 和逐元素相乘不是矩阵乘，由 **c6r2 SwiGLU 切片站**完成。Main16 产出的
gate/up 记录被汇入 c6r2，每次 c6r2 接收 512-dword 输入（512 bf16 gate slice +
512 bf16 up slice），输出 256-dword（512 bf16）SwiGLU slice。每片输出送入
c6r1 的 6144-dword gather buffer：

```
24 个 slice × 256 dword = 6144 dword = 12288 bf16 完整 FFN intermediate
```

Buffer 填满后，c6r1 用 packet0 发布 6144-dword 块，经与 attention 返回路径
**完全相同的 c1r1 共享激活桥**（DMA4 ↔ DMA1 256-dword ping-pong）扇出到
Main16 入口的 128-dword activation ring：

```
6144 dword / 256 = 24 次 bridge 迭代
6144 dword / 128 = 48 个 down 激活 chunk
```

48 正好等于 down 的输入维度划分（12288 / 256），与 Main16 的累加次数一致。

**Down 投影（Main16，阶段 7）**

12288 → 4096 的矩阵乘（64 个 patch）。输入维度变大了，所以每个 tile 需要累加
48 个 chunk（48 × 256 = 12288）而不是之前的 16 个。

scheduler 以 slice 为单位 adjacent pair 提交：每生成一对 (up[slice],
gate[slice]) 的 257-dword compact packet，c6r2 即输出一个 256-dword 的
SwiGLU slice，整层共 24 个 slice。

### 4.10 终点：最终残差 → 层输出

Down 投影的输出与 4.8 步残差后的 hidden state 再做一次残差加：

```
layer_output = hidden + down_output
```

这就是本层的最终输出 — 一个 4096 维 bf16 向量。它将作为下一层的输入，重复以上
所有过程。

整个 Qwen3-8B 模型有 36 层，每层都是这样一台数据流机器。

---

## 第五部分：设计的精妙之处

### 5.1 三次闭环：数据在 Main 和 Edge 之间来回

整层计算中，数据在 Main16 和 Edge16 之间形成三次闭环。每次从 Main 出去，经过
Edge 侧不同的工位处理后，再流回 Main：

```
闭环 1（attention → O，最复杂，经过 6 种不同角色）：
  Main 产出 Q/K/V
    → c1r3 后处理（q/k norm + RoPE）
    → c6r1 枢纽（拆 Q 为 4 窗口；K/V 经 packet14/15 写 KV cache）
    → c0r1/c7r1 KV 整形（历史 K/V scan 回流）
    → Shape-A × 4（QK 评分 + online softmax）
    → Shape-B × 4（加权求和）
    → c6r1 枢纽（汇聚四份 attention 结果）
    → c6r1 packet2 → c1r1 共享激活桥 → Main 做 O 投影

闭环 2（O → 残差/RMSNorm → up/gate）：
  Main 产出 O 的 17-dword 记录（每 tile 8 条 = 8 N-blocks）
    → compact/aux 路径
    → c1r2 全向量站（残差加 + 第二次 RMSNorm，下一阶段 +48 chunk replay）
    → 回到 Main 做 up/gate 投影（物理 order：up 先于 gate）

闭环 3（up/gate → SwiGLU → down）：
  Main 产出 up + gate 记录（每 tile 48 条 = 24 + 24 slices）
    → c6r2 SwiGLU 切片站（512-dw 输入 → 256-dw 输出）
    → c6r1 6144-dword gather buffer
    → c6r1 packet0 → c1r1 共享激活桥 → Main 做 down 投影
```

闭环 1 是整个设计最复杂的路径 — 数据物理上穿越了几乎整个 8 列阵列，经过后处理、
分发、评分、求和、汇聚、共享桥共 6 种不同工位。

**关键观察**：闭环 1（attention → O）和闭环 3（SwiGLU → down）**复用同一条
c1r1 共享激活桥**（c6r1 packet 源 → c1r1 DMA4 256-dword bridge → DMA1
multicast → Main16 DMA0 128-dword activation ring）—— 不是两套独立的返回机制，
只是 packet2 与 packet0 切换源数据。Main16 端保持普通 projection activation ABI。

**每次闭环都避免了一次主存往返。** 如果不做融合，每次 Main 产出的结果都要写回
主存，Edge 侧再从主存读出来；Edge 算完再写回主存，Main 再读出来。三次闭环就是
六次省下来的主存往返。

### 5.2 时分复用：同一组 tile 跑 7 个阶段

传统做法是给每个投影分配独立的硬件资源。但 NPU 只有 16 个 Main tile — 不够
同时铺开 7 个投影。

解决方案：**让同一组 tile 依次做 7 个阶段，每个阶段只切换权重输入流。**

从 tile 的视角看：

```
时间 ─────────────────────────────────────────────→

输入通道 1:  [Q权重块][Q权重块]...[K权重块]...[V权重块]...[O权重块]...
输入通道 0:  [激活片][激活片]...[激活片]...[激活片]...[attn结果片]...
```

tile 不需要被告知"现在是第几阶段"。它只看到：又来了一个 5120 字节权重块 + 又来了
256 个 bf16 激活。做乘累加，输出结果。换个权重流 = 换个阶段。

好处：
- 16 个 tile 的利用率接近 100% — 除了阶段切换的短暂间隙，一直在算
- 不需要复杂的资源调度 — 硬件配置一次，7 个阶段自动依次完成
- 减少面积浪费 — 不需要 7×16 = 112 个 tile

代价：
- 需要精确的相位同步 — Edge 侧必须在正确的时刻把结果送回 Main
- 权重必须按正确顺序流入 — Q 权重流完了才能流 K 权重
- 总延迟是串行的 — 7 个阶段依次执行，不能并行

对于 decode（每次只处理 1 个 token）来说，这个代价是可接受的：单 token 的
计算量不大，瓶颈在权重带宽而不在并行度。

### 5.3 流水线重叠：传输和计算完全并行

每个 Main tile 处理一个 chunk 需要的时间 ≈ 传输下一个 chunk 需要的时间。
利用 ping-pong 双缓冲，两者完全重叠：

```
时间 ─────────────────────────────────────────────→

缓冲区 A:  [传入 chunk1] [计算 chunk1] [传入 chunk3] [计算 chunk3] ...
缓冲区 B:             [传入 chunk2] [计算 chunk2] [传入 chunk4] ...
```

在任何时刻：
- 一个缓冲区正在被 DMA 填充下一个 chunk
- 另一个缓冲区正在被计算核心消费当前 chunk

结果：**从第一个 chunk 之后，传输延迟被完全隐藏。** 计算核心永远不需要空等数据。

这就是 DMA 和 Lock 协作的价值：DMA 按 BD 指示搬数据，Lock 保证"填完才能算，
算完才能填"，两者交替推进。

### 5.4 减少中间激活的主存往返：瓶颈转移

在 GPU 上，每个算子的中间结果都要写回显存再读出来。一层 Transformer 的中间
数据搬运量（Q+K+V+attention+O+gate+up+FFN+down）远超权重本身。

在融合引擎中：

| 数据类型 | 是否经过主存 | 大小 |
|---|---|---|
| 权重（Q4NX 格式） | 是（流入） | ~115 MB |
| KV 缓存写入 | 是（当前 K/V 经 row0 写出） | ~4 KB |
| KV 缓存读出 | 是（历史 rounded scan 回 NPU） | 随上下文长度增长 |
| 输入 / 最终输出向量 | 是（进出各 8 KB） | 16 KB |
| RMSNorm / RoPE 旁路输入 | 是（流入） | 小 |
| attention 输出 | 否（片上） | — |
| O 投影输出 | 否（片上） | — |
| 第二次残差与 RMSNorm 中间值 | 否（片上） | — |
| gate / up 中间值 | 否（片上） | — |
| SwiGLU 输出 | 否（片上） | — |
| down 投影输出（残差前） | 否（片上） | — |

这是设计的明确选择：host-visible layer boundary 上**根本没有** O / 残差 /
RMSNorm / FFN 中间值的 DDR descriptor，这些状态都由片上工位承担
（c1r2 全向量站、c6r2 SwiGLU 切片、c6r1 6144-dword gather、c1r1 共享桥）。

注意：当前 K/V 的路径并非"片上直接喂 attention" —— 它经 row0 写入 KV cache BO，
随后作为历史的一部分被 rounded scan 读回 NPU。这条路径不可省（未来 token 需要
访问），但其他中间激活都不经主存。

对比 GPU 方式省掉的主存搬运：
- Q 投影输出 → 后处理（~8 KB）
- attention 输出 → O 投影（~8 KB × 2 来回）
- O 输出 → 残差 + RMSNorm → gate/up 激活
- gate/up → SwiGLU → down 激活（12288 bf16 = 24 KB × 2 来回）
- down 输出 → 最终残差 → 层输出

设计的核心价值是：**尽可能把主存带宽留给权重流入** — 这才是单 token decode 的
真正瓶颈（115 MB 权重远大于任何中间结果）。

融合引擎把瓶颈从"主存带宽被中间结果和权重争抢"变成"以权重带宽为主" —
一个更简单、更可优化的问题。完整 engine 的 runtime boundary 应该只暴露
hidden-in/hidden-out + weights + norm/RoPE 旁路 + KV cache descriptor，
**不应该暴露任何 O / residual / FFN temporary BO**。

---

## 第六部分：对比总结

### 6.1 GPU 与 NPU 融合引擎的核心差异

| 维度 | GPU 方式 | NPU 融合引擎 |
|------|---------|-------------|
| 中间结果 | 每步写回显存，下步再读 | 大部分在 tile 之间直接流动 |
| 资源分配 | 每个算子独占一批 SM | 同一组 tile 时分复用跑 7 个阶段 |
| 调度方式 | 主机逐算子提交 kernel | 一次配置，整层自动执行 |
| 主机参与 | 每步都要 kernel launch | 开始时配置地址，然后不介入 |
| 瓶颈 | 显存带宽（权重 + 中间结果争抢） | 主要是权重带宽 |
| attention | 独立 kernel，Q/K/V 在显存中转 | Q 片上直接分发；K/V 经 KV cache 回读 |
| 编程模型 | 写 kernel，框架调度 | 配置数据流图，设计物理布局 |

### 6.2 各自的优势场景

**GPU 更适合**：
- 大 batch（多个 token 同时处理）— 计算密度高，能喂满 SM
- prefill 阶段（一次处理整个 prompt）— 大矩阵乘效率高
- 需要快速迭代的研究 — 编程模型简单，改算子很容易

**NPU 融合引擎更适合**：
- 单 token decode — 计算量小，瓶颈在带宽，融合减少搬运
- 端侧部署（笔记本、手机）— 功耗低，不需要独立显存
- 长上下文推理 — KV 缓存可以放在主存，NPU 按需流入

### 6.3 一句话总结

> 把一个 Transformer 层想象成缝一件衣服：GPU 是每缝一针就把布取下来放回桌上
> 再拿起来；NPU 融合引擎是布一直在缝纫机里，针脚连续走完整件。

**数字概览**：

```
一层 decode 的全部工作（设计目标）：
  608 个 patch，权重总流量 ~115 MB
    （普通 patch K=4096: ~160 KB；down patch K=12288: ~480 KB）
  7 个投影阶段 + 2 次 RMSNorm + 1 次 attention + 1 次 SwiGLU
  16 个 Main tile 并行，各跑 608 个 patch 中分到的份额
  整层完成后输出 1 个 4096 维向量
```

这就是融合层引擎的设计全貌 — 一台由 32 个 tile 组成的数据流机器，每层启动一次，
权重流入的同时计算就在发生，几乎所有中间激活不离开芯片。

**整层数据流契约**：

- **Main16 投影 ABI**：每个 tile 一次接收 128 dword 激活 + 1280 dword
  Q4NX 权重，输出一条 17 dword compact record（1 dword header + 16 dword
  payload）。每个 dword 打两个 bf16：
  ```
  payload_dword[i].lo16 = output[2i + 0]
  payload_dword[i].hi16 = output[2i + 1]
  ```
  每个 tile 一次产出 32 个连续 output element。每个 phase 的 record 数：
  - Q：每 tile 12 条 = 8 N-blocks（K/V 复用同一族，各占其中 2 N-blocks）
  - O：每 tile 8 条
  - up/gate：每 tile 48 条 = 24 + 24 slices
  - down：每 tile 8 条
- **N-block 与 tile 顺序**：一个 N-block = 16 tile × 32 bf16 = 512 bf16 =
  256 dword，tile 顺序按 row1 自然汇聚 (c2..c5) × (r2..r5)，第 n 个 N-block
  覆盖 global output `n*512 .. n*512 + 511`
- **共享激活桥**：c6r1 packet → c1r1 DMA4 256-dword bridge → DMA1 multicast
  → Main16 DMA0 128-dword ring。attention 返回（packet2）与 down 输入返回
  （packet0）复用同一条物理桥
- **FFN SwiGLU 放置**：c6r2 接收 512-dw 输入（`input[0x000..0x3ff]` = up，
  `input[0x400..0x7ff]` = gate）→ 256-dw SwiGLU slice →
  c6r1 6144-dw gather → packet0 发布
- **Full-vector 放置**：c1r2 承担 hidden/RMSNorm/residual/final-output。
  full-vector 采用自然 dim-major `hidden[dim]`，dim = 0..4095，dword 打两个
  bf16；通过 packet0 切成 16 个 256-bf16 chunk 喂给 main16；phase 顺序
  `+12 → +48 → +1`（Q/K/V → up/gate → 最终 hidden 输出）
- **Q / KV 布局**：
  - Q 为 head-major `Q[head][dim]`，head=0..31, dim=0..127
  - K/V 为 GQA group-major `K[group][dim]` / `V[group][dim]`，
    group=0..7, dim=0..127；GQA 映射 `kv_group = q_head // 4`
  - Q 分发为 4 个 head-major 窗口：heads 0-7 / 8-15 / 16-23 / 24-31，
    每窗口 = 8 heads × 128 bf16 = 1024 bf16 = 512 dword
  - KV 缓存逻辑布局 `K[layer][group][token][dim]` /
    `V[layer][group][token][dim]`，写入与回读使用同一布局
- **Shape-A/B carrier**：每 16 个历史 token 一个 block。
  - `base[0x100]` = head-major 8 records × 16-token bf16 权重，
    `base[h][t] = exp(score[h][t] - block_max[h])`
  - `scalar[0x40]` = 8 × (block_max, block_sum) fp32 pair，
    `scalar[2h+0] = block_max[h]`, `scalar[2h+1] = block_sum[h]`
  - 通过本地相邻存储 + lock immediate 同步
  - Shape-B 用标准 online-softmax merge 累加 `accum[8][128]` fp32
- **Attention 输出 → O chunk 顺序**：head-major，
  `O_chunk[c]` = heads `2c` 与 `2c+1` 的所有 128 dim，每 chunk = 256 bf16 =
  128 dword，正好匹配 main16 activation ring 的 128-dword chunk
- **物理 projection 顺序**：`Q, K, V, O, up, gate, down`

---

## 附录

### A. 模型参数速查

```
模型:           Qwen3-8B
hidden_size:    4096    （隐藏层维度）
num_q_heads:    32      （query 头数）
num_kv_heads:   8       （key/value 头数，GQA）
head_dim:       128     （每个头的维度）
intermediate:   12288   （FFN 中间维度）
num_layers:     36      （层数）
GQA ratio:      4       （每 4 个 query head 共享 1 个 kv head）
```

### B. 量化格式 Q4NX 详解

所有权重以 4-bit 量化存储。每个权重块（chunk）的内存布局：

```
┌─────────────────────────────────────┐
│ scales      : 32行 × 8组 × bf16     │  512 字节
│ zero_points : 32行 × 8组 × bf16     │  512 字节
│ int4_data   : 32行 × 256列 / 2      │  4096 字节
├─────────────────────────────────────┤
│ 合计                                 │  5120 字节
└─────────────────────────────────────┘
```

解释：
- 32 行 × 256 列：一个 chunk 覆盖 32 个输出维度、256 个输入维度
- group size = 32：每 32 个权重共享一组 scale 和 zero_point
- 8 组 = 256 列 / 32：每行有 8 个量化组

在线反量化公式（tile 内部执行，不生成中间全精度矩阵）：
```
weight_fp = (int4_value - zero_point) × scale
output[row] += weight_fp × activation[col]
```

### C. 608 个 Patch 的完整清单

| 阶段 | 输入维度 | 输出维度 | patch 数 | 每 patch 的 chunk 数 | 每 tile 累加次数 |
|------|---------|---------|---------|-------------------|---------------|
| Q 投影 | 4096 | 4096 | 64 | 16 | 16 |
| K 投影 | 4096 | 1024 | 16 | 16 | 16 |
| V 投影 | 4096 | 1024 | 16 | 16 | 16 |
| O 投影 | 4096 | 4096 | 64 | 16 | 16 |
| up 投影 | 4096 | 12288 | 192 | 16 | 16 |
| gate 投影 | 4096 | 12288 | 192 | 16 | 16 |
| down 投影 | 12288 | 4096 | 64 | 48 | 48 |
| **合计** | | | **608** | | |

说明：
- 每个 patch = 64 输出行 × 完整输入维度
- 每个 tile 处理 32 输出行（半个 patch）
- 一个 N-block = 4 列 × 每列 2 patch = 8 patch（16 tile 并行完成）
- Q 投影：64 patch ÷ 8 patch/N-block = 8 个 N-block
- down 投影特殊：输入维度 12288 = 48 × 256，所以每 tile 需要累加 48 个 chunk

### D. 物理布局图

```
         c0          c1         c2    c3    c4    c5       c6          c7
      ┌────────┬────────┬──────┬──────┬──────┬──────┬────────┬────────┐
row5  │Shape-B │        │ M13  │ M14  │ M15  │ M16  │        │Shape-B │
      ├────────┼────────┼──────┼──────┼──────┼──────┼────────┼────────┤
row4  │Shape-A │        │ M09  │ M10  │ M11  │ M12  │        │Shape-A │
      ├────────┼────────┼──────┼──────┼──────┼──────┼────────┼────────┤
row3  │Shape-B │后处理   │ M05  │ M06  │ M07  │ M08  │        │Shape-B │
      ├────────┼────────┼──────┼──────┼──────┼──────┼────────┼────────┤
row2  │Shape-A │全向量站 │ M01  │ M02  │ M03  │ M04  │SwiGLU   │Shape-A │
      ├────────┼────────┼──────┼──────┼──────┼──────┼────────┼────────┤
row1  │KV整形  │共享桥   │      行1 memtile 中转       │枢纽/汇聚│KV整形  │
      ├────────┼────────┼──────┴──────┴──────┴──────┼────────┼────────┤
row0  │K写回   │                  shim / 主存入口                │V写回   │
      └────────┴─────────────────────────────────────────────────┴────────┘
```

图例：
- M01-M16：Main16 计算 tile（矩阵-向量乘法）
- Shape-A：attention 评分 tile（QK 点积 + online softmax）
- Shape-B：attention 输出 tile（加权 V 求和）
- 后处理（c1r3）：Q/K current-side norm + RoPE
- 全向量站（c1r2）：4096-bf16 hidden 的第一次 RMSNorm、O 后残差+第二次 RMSNorm、最终残差/层输出
- SwiGLU（c6r2）：512-dw gate/up slice → 256-dw SwiGLU slice
- 共享桥（c1r1 memtile）：DMA4 256-dw 双缓冲 ↔ DMA1 multicast，attention 返回与 down 返回共用
- 枢纽/汇聚（c6r1 memtile）：Q 分发 + attention 结果汇聚（packet2）+ FFN intermediate gather（packet0）

### E. 术语对照表

| 术语 | 英文 | 含义 |
|------|------|------|
| tile | tile | NPU 阵列中的一个计算/存储单元 |
| shim | shim tile | Row 0 的接口单元，连接主存和 NPU 内部 |
| memtile | memory tile | Row 1 的大容量存储单元，做数据中转 |
| DMA | Direct Memory Access | 自动搬运数据的硬件引擎 |
| BD | Buffer Descriptor | DMA 的任务单，指定从哪搬、搬多少、何时搬 |
| Lock | Lock | 生产者-消费者之间的同步信号 |
| Stream | Stream | tile 之间的物理数据连线 |
| Packet | Packet | 带标签的数据包，用于在同一条线路上分路 |
| BO | Buffer Object | 主机可见的主存缓冲区（XRT 管理） |
| ping-pong | ping-pong buffer | 双缓冲，交替使用让传输和计算重叠 |
| chunk | chunk | 权重的最小处理单位（32行×256列=5120字节） |
| patch | patch | 一次投影调度的基本单位（64行×完整输入维度） |
| Q4NX | Q4NX | 4-bit 量化权重格式 |
| bf16 | bfloat16 | 16位浮点格式，1位符号+8位指数+7位尾数 |
| RMSNorm | Root Mean Square Norm | 基于均方根的归一化 |
| RoPE | Rotary Position Embedding | 旋转位置编码 |
| GQA | Grouped Query Attention | 分组查询注意力，多个 Q head 共享 KV head |
| SwiGLU | SiLU-gated GLU | FFN 中使用的激活机制 |
| online softmax | online softmax | 边扫描边计算 softmax，不需要两遍扫描 |
| 融合层引擎 | fused layer engine | 整层做成一个预配置的数据流机器 |
