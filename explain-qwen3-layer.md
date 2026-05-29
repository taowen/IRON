# Qwen3 融合层引擎详解

> 在 AMD XDNA NPU 上把 Qwen3-8B 的一个 Transformer decode 层做成一台数据流机器

---

# 一、项目是什么

## 一句话

在 AMD XDNA NPU 上，把 Qwen3-8B 模型的一个 Transformer decode 层做成一台"数据流机器"——配置一次、启动一次、整层跑完。

## 背景：GPU 怎么做一层推理

GPU 把 Transformer 层拆成一连串独立算子：

```
Q投影 → K投影 → V投影 → attention → O投影 → up → gate → SwiGLU → down
```

每个算子的流程都是：从显存读上一步输出 → 计算 → 结果写回显存 → 通知下一个算子。
中间结果反复搬运，显存带宽被权重和激活共同争抢。

## NPU 融合引擎的做法

不拆成独立算子，而是把整层做成一台预先配好的数据流机器：

- **配置阶段**：CPU 一次性把路由、缓冲区、地址全配好
- **执行阶段**：NPU 自动运转——权重流入、激活在 tile 之间传递、attention 在片上完成、结果流出。CPU 不参与中间调度

**核心收益**：大部分中间激活（Q/K/V 投影输出、attention 结果、FFN 中间值）都不回主存，只在 tile 之间直接流动。主存带宽几乎全部留给权重流入（~115 MB/层）。

## 类比

| GPU | NPU 融合引擎 |
|-----|-------------|
| 工厂里每道工序做完，半成品送回仓库，下一道再取 | 流水线车间，半成品直接从一个工位传到下一个 |
| 每个算子一次 kernel launch | 整层一次启动 |
| 显存带宽 = 权重 + 中间结果争抢 | 主存带宽 ≈ 只有权重流入 |

## 适用场景

NPU 融合引擎特别适合**单 token decode**：计算量小、瓶颈在带宽、融合能减少搬运。这正是端侧（笔记本、手机）LLM 推理的核心场景。

Qwen3-8B 共 36 层，每层都是这样一台数据流机器。本项目实现的就是其中一层的完整物理数据流。

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
| 运算 | 只有乘累加 | 归一化/softmax/SiLU/逐元素乘 |
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
- **只做一件事**——128 dword 激活 + 1280 dword Q4NX 权重 → 乘累加 → 17 dword compact record
- **视野极窄**——每个 tile 一次只看 32 行 × 256 列的窗口，不知道自己在算 Q 还是 down
- **时分复用**——同一组 16 tile 依次跑 7 个投影阶段（Q→K→V→O→up→gate→down），靠切换权重流区分阶段

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

**职责**：承载完整 4096-bf16 hidden 向量的所有全局操作

- 第一次 RMSNorm（进入投影前）
- O 投影后的残差加 + 第二次 RMSNorm（喂给 up/gate）
- Down 投影后的最终残差 + 层输出

**为什么在这里**：RMSNorm 和残差必须看到完整 4096 维向量。c1r2 提供 2048-dword full-vector ABI 和 sum-of-squares/rsqrt 计算。

**对外接口**：2049-dword packet0 replay（1 control + 2048 payload），replay 次数 = +12（Q/K/V）→ +48（up/gate）→ +1（最终输出）。

### c1r3 后处理站

**职责**：Q/K norm + RoPE 位置编码 + current K/V 路由

- 接收 main16 产出的 Q+K+V compact record
- 对 Q 和 K 做 RMS 归一化
- 对 Q 和 K 做 RoPE 旋转位置编码（V 不做）
- 输出 Q（2048 dword）发给 attention 侧
- 输出 current K/V 经 packet8/9 写入主存缓存

**关键约束**：需要 runtime-start lock 门控，否则 c1r3 可能在 host 写 current-token RTP 前就开始执行。

### c6r1 枢纽（Memtile）

**职责**：分发 + 汇聚的中转站，承担两个方向的工作

- **前向**：接收完整 Q（2048 dword），拆成 4 份 512-dword 窗口发给 4 个 Shape-A
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

**职责**：QK 点积评分 + block-level softmax

- 接收一个 512-dword Q 窗口（8 heads × 128 dim）
- 每 16 个历史 token 为一个 block，计算 score 和 block_max/block_sum
- 把 carrier（权重+统计量）交给配对的 Shape-B
- 最后一个 block 用 tail-token RTP 把 padding token 权重清零

### Shape-B × 4（求和）

**位置**：c0r3, c0r5, c7r3, c7r5

**职责**：加权 V 求和 + online softmax merge

- 接收 Shape-A 的 carrier + 对应 block 的 16 个历史 V
- 用 block 权重加权 V，与本地累加器做 online-softmax merge
- 历史扫描结束后输出 `accum / running_sum` = 512-dword attention 结果

### c0r1 / c7r1 KV 整形（Memtile）

**职责**：从 shim scan 接收历史 K/V，分流到 Shape tile

- c0r1 处理 group 0-3（左侧）
- c7r1 处理 group 4-7（右侧）
- K 送 Shape-A，V 送 Shape-B

### c6r2 SwiGLU 切片站

**职责**：执行 `SiLU(gate) × up`

- 接收 512-dword 输入（低半区 = up slice，高半区 = gate slice）
- 输出 256-dword（512 bf16）SwiGLU slice
- 24 个 slice 组成完整 12288-bf16 FFN intermediate

### Row1 c2-c5 列 compact tile

**职责**：per-column record 汇聚 + Q4NX weight 分发

- S2MM0-3：接收 main16 的 compact record，汇聚成 column-level layout
- S2MM4/5：接收来自 shim 的 Q4NX 权重
- MM2S0-3：把权重分发到各行 main16 DMA1

---

# 四、整层数据流——一个 token 的完整旅程

跟随一个 token，看它从进入 NPU 到离开的完整过程。

## 起点：准备

一个 token 的隐藏状态 = 4096 个 bf16 数字组成的向量（8 KB）。同时准备好的还有：
- 本层全部权重（Q/K/V/O/up/gate/down，Q4NX 格式，~115 MB）
- 历史 KV 缓存（之前所有 token 的 K 和 V）

NPU 不会一次性读入所有权重，而是按阶段分批流入。

## 步骤 1：第一次 RMSNorm

```
主存 → hidden vector(4096 bf16) → c1r2 全向量站
                                    ↓
                              计算平方均值
                              除以均方根
                              乘以缩放参数
                                    ↓
                            归一化后的 hidden
```

RMSNorm 必须看到完整 4096 维向量，所以放在 c1r2。归一化后的 hidden 准备分发给 Main16。

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

每个 N-block = 16 tile × 32 bf16 = 512 个输出元素。tile 不知道自己在算 Q 还是 V，只管"拿到数字就乘"。

## 步骤 3：Q/K/V 后处理

```
main16 compact record → row1 汇聚 → c1r3 后处理站
                                        ↓
                                  Q/K 做 RMS norm
                                  Q/K 做 RoPE（V 不做）
                                        ↓
                              Q(2048 dw) → 发给 attention
                              K/V → packet8/9 写回主存
```

RoPE 让模型知道"这是第几个词"——按维度两两配对，旋转一个与位置相关的角度。

## 步骤 4：Current K/V 写回

```
c1r3 → packet8 → shim(c0r0) → 写入 K cache BO
c1r3 → packet9 → shim(c7r0) → 写入 V cache BO
```

当前 token 的 K 和 V 必须写入缓存，供未来 token 的 attention 使用。写回使用两个链式 BD（BD14 写偶位置，BD15 写奇位置），实现 head-interleaved scatter。

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
  score = Q · K / 128
  求 block_max
  weight = Q12_exp(block_max - score)
  打包成 carrier → 交给 Shape-B
```

最后一个 block 用 tail-token RTP 把 padding 位置的权重清零。

### 5c. Shape-B 加权求和

```
每个 Shape-B 拿到:
  - Shape-A 传来的 carrier（权重 + max/sum）
  - 对应的 16-token V block

每个 block:
  block_weighted_v = sum(weight[t] × V[t])
  online-softmax merge:
    new_max = max(running_max, block_max)
    accum = accum × old_scale + block_weighted_v × new_scale
    running_sum = running_sum × old_scale + block_sum × new_scale

扫描结束: output = accum / running_sum
```

每个 Shape-B 产出 512 dword（8 heads × 128 dim）。

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

```
O compact record → row1/c1r1 汇聚 → c1r2 全向量站
  ↓
c1r2 从 O compact 重建完整 4096-bf16 O 结果
  + 与原始 hidden 残差加
  + 第二次 RMSNorm
  ↓
归一化后的 hidden 准备发给 up/gate
```

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
  output = SiLU(gate) × up
  ↓
24 个 256-dw SwiGLU slice → c6r1 的 6144-dw gather buffer
```

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
c1r2:                                                                [残差+norm]
c1r2→main:                                                              [48 replay]
main:                                                                   [up/gate──]
c6r2:                                                                              [SwiGLU]
c6r1→main:                                                                               [pkt1 down]
main:                                                                                    [down投影─]
c1r2:                                                                                              [残差→out]
```

---

# 五、关键设计决策

## 5.1 三次闭环：数据在 Main 和 Edge 之间来回

整层计算形成三次 Main↔Edge 闭环，每次都避免了一次主存往返：

```
闭环 1（最复杂）：
  Main 产出 Q/K/V → c1r3 后处理 → c6r1 拆 Q
  → Shape-A/B attention → c6r1 汇聚 → packet2
  → c1r1 bridge → Main 做 O 投影

闭环 2：
  Main 产出 O record → c1r2 残差+RMSNorm
  → 48 次 packet0 replay → Main 做 up/gate

闭环 3：
  Main 产出 up/gate record → c6r2 SwiGLU
  → c6r1 gather → packet1
  → c1r1 bridge → Main 做 down 投影
```

如果不做融合，每次闭环 = 两次主存往返（出去写一次、回来读一次）。三次闭环 = 省下六次主存搬运。

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
| MM2S 0-3 | weight 分发到 main16 DMA1 各行 | 同时也复用做 compact output |

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

只要 `iteration_size` 和 `repeat_count` 配对编程，一个 BD 就能扫描任意数量的 block。

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

**已验证**：token1007 容量 PDI patch 到 token91 后：
- patched instruction stream 逐 word 等于重新编译的 token91 stream
- 真机通过

这证明 patch 是精确的，可以作为"固定最大上下文 PDI + 每 token patch"的生产方案雏形。

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
- main16 把 48 条 record 写入本地 `upgate_records` buffer
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

每个 block 每侧只需 2 个 scan BD，释放空间给 current K/V 写回。

## 5.9 数值稳定性选择

- 不对 Q4NX bf16 输出做精确 hash——一个 ULP 差异会被 hash 放大成假 mismatch
- c1r2 replay 使用 bounded numeric scale（当前 256）+ 有界 int32 sqrt，对 bf16 LSB 抖动不敏感
- 生成代码优先产出有界整数/数值形式，不依赖复杂 hash 或 wide integer lowering
- Shape-A/B 使用显式固定点（Q12 exp weights + int32 accumulator），后续需校准到 Qwen3 bf16/fp32

---

# 六、量化与数据格式

## 6.1 Q4NX 权重格式

所有权重以 4-bit 量化存储。每个 chunk 的内存布局：

```
┌─────────────────────────────────────┐
│ scales      : 32行 × 8组 × bf16     │  512 字节
│ zero_points : 32行 × 8组 × bf16     │  512 字节
│ int4_data   : 32行 × 256列 / 2      │  4096 字节
├─────────────────────────────────────┤
│ 合计                                 │  5120 字节 = 1280 dword
└─────────────────────────────────────┘
```

- 32 行 × 256 列 = 一个 chunk 覆盖 32 个输出维度、256 个输入维度
- group size = 32：每 32 个权重共享一组 scale 和 zero_point
- 8 组 = 256 / 32

**在线反量化**（tile 内部执行，不生成中间全精度矩阵）：
```
weight_fp = (int4_value - zero_point) × scale
output[row] += weight_fp × activation[col]
```

**整层权重规模**：

| 阶段 | 输入维度 | 输出维度 | patch 数 | chunk/patch | 总 chunk |
|------|---------|---------|---------|------------|---------|
| Q | 4096 | 4096 | 64 | 16 | 1024 |
| K | 4096 | 1024 | 16 | 16 | 256 |
| V | 4096 | 1024 | 16 | 16 | 256 |
| O | 4096 | 4096 | 64 | 16 | 1024 |
| up | 4096 | 12288 | 192 | 16 | 3072 |
| gate | 4096 | 12288 | 192 | 16 | 3072 |
| down | 12288 | 4096 | 64 | 48 | 3072 |
| **总计** | | | **608** | | **11776** |

608 patch × 5120 字节/chunk × 16 chunk/patch ÷ 16 tile ≈ 每 tile 3,082,240 字节权重流量。
总权重 ≈ 115 MB。

## 6.2 bf16（bfloat16）

16 位浮点格式：1 位符号 + 8 位指数 + 7 位尾数。与 float32 共享指数范围，精度低但范围大。

NPU 内部激活全部使用 bf16。一个 dword（32 bit）打两个 bf16：
```
dword.lo16 = value[2i]
dword.hi16 = value[2i+1]
```

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

**replay 次数**：
- Q/K/V 阶段：12 次（Q 8 N-block + K 2 + V 2）
- up/gate 阶段：48 次（24 up + 24 gate）
- 最终输出：1 次

## 6.5 Shape Carrier（80 dword）

Shape-A 每处理完 16 个历史 token（一个 block），交给 Shape-B 的数据包：

```
base [0x100 = 256 字节 = 64 dword]:
  8 heads × 16 tokens 的 Q12 权重
  base[h][t] = round(4096 × exp(-(block_max[h] - score[h][t]) / 8))

scalar [0x40 = 64 字节 = 16 dword]:
  8 × (block_max, block_sum) int32 pair
  scalar[2h+0] = block_max[h]
  scalar[2h+1] = block_sum[h]
```

通过本地相邻存储 + lock immediate 同步（不是 DMA tensor stream）。

## 6.6 Attention 输出（2048 dword）

4 个 Shape-B 各产出 512 dword = 8 heads × 128 dim：

```
Shape-B 0 (c0r3): heads  0.. 7 → 512 dword
Shape-B 1 (c0r5): heads  8..15 → 512 dword
Shape-B 2 (c7r3): heads 16..23 → 512 dword
Shape-B 3 (c7r5): heads 24..31 → 512 dword
────────────────────────────────────────────
合计: 2048 dword = 4096 bf16 = 32 heads × 128 dim
```

在 c6r1 拼接后以 packet2 一次性发布。

## 6.7 SwiGLU 输入/输出

**输入**（512 dword = c6r2 接收）：
```
input[0x000..0x1ff] = up slice   (256 dword = 512 bf16)
input[0x200..0x3ff] = gate slice (256 dword = 512 bf16)
```

**输出**（256 dword）：
```
output = SiLU(gate) × up = 512 bf16
```

24 个 slice × 256 dword = 6144 dword = 12288 bf16 = 完整 FFN intermediate。

## 6.8 KV Cache 布局

逻辑布局：`K[layer][group][token][dim]` / `V[layer][group][token][dim]`

- 8 个 KV group × 128 dim = 1024 bf16/token
- 按 16-token block 对齐扫描
- 写入和回读使用同一布局

物理 scan：每侧（左/右）用 `buffer_length=4096`（一个 side 的连续 K 或 V 流）描述整个 block 序列。

---

# 七、代码结构

`qwen3-layer/` 目录包含完整的融合层实现：契约定义、数据流图、MLIR 生成器、NPU kernel、可运行 case 和共享工具。

## 7.1 契约与数据流图

| 文件 | 职责 |
|------|------|
| `contract.py` | 全层 ABI 的单一真相源：维度常量、phase 定义、patch/chunk 数量、packet 大小 |
| `dataflow.py` | 静态数据流图——Node（tile+角色）和 Edge（source/target/payload/packet）的类型化描述 |
| `physical_contract.py` | 生成后检查：row1 channel 所有权（S2MM4/5=weight, S2MM0-3=compact）、禁止旧路由 |
| `check_contract.py` | 集成检查：contract + dataflow + 生成 MLIR 三者一致性 |
| `projection_schedule.py` | Q/K/V body record 数、O/upgate/down tail weight chunk base、总 weight BO layout 的唯一派生源 |

## 7.2 MLIR 生成器

| 文件 | 职责 |
|------|------|
| `generate.py` | 从 dataflow 图生成物理骨架 MLIR-AIE（tile 声明、ObjectFifo、lock、BD） |
| `emit_mlir.py` | 输出 `build/qwen3_dataflow.mlir` 文件 |
| `compact_dataflow.py` | row1/c1r1 compact gather + bridge + hub + row1 S2MM4/5 weight fanout 生成器 |
| `weight_stream.py` | row1 Q4NX patch-ring 生成器：host/shim weight ingress → main16 DMA1 |
| `qkv_compact_dataflow.py` | 四阶段 Q/K/V/O compact bridge，current-K/V attention 集成边界用 |
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
| `main_projection_q4nx.cc` | c2-c5, r2-r5 | Q4NX 反量化 + MAC + flush + compact record emit |
| `edge_attention.cc` | c0/c7, r2-r5 | Shape-A（QK score + Q12 exp）、Shape-B（weighted V + online merge） |
| `postprocess_qkv.cc` | c1r3 | Q/K norm + RoPE + current K/V 打包 |
| `full_vector_station.cc` | c1r2 | RMSNorm + 残差加 + full-vector replay + compact 展开 |
| `swiglu.cc` | c6r2 | bf16 up/gate → SiLU(gate)×up → 256-dw slice |
| `debug_contract.cc` | — | 临时确定性桥和 smoke-test kernel（不属于生产路径） |

共享头文件：
- `record_format.h`：compact record header 布局和打包/解包 helper
- `qwen3_constants.h`：kernel 侧常量（维度、phase 边界等）

## 7.4 可运行 Case

`cases/` 目录是模块化的 case 注册机制。每个 case 由三件套组成：
- `*_generate.py`：生成该 case 的 MLIR-AIE
- `*_reference.py`：CPU 参考实现，用于结果比对
- `*_runner.py`：NPU 运行器，调用 XRT 提交任务并验证

`run_npu.py` 是顶层 CLI 调度器：

```bash
# 检查生成 MLIR 结构
python qwen3-layer/run_npu.py --check-only

# 编译 xclbin（不跑 NPU）
python qwen3-layer/run_npu.py --build-only

# 默认 case = currentkv-full-layer-q4nx-down-bridge
python qwen3-layer/run_npu.py

# 指定 case 和 token
python qwen3-layer/run_npu.py --case currentkv-kvscan-attention-kv16-o-bridge --current-token 91

# Instruction patch（不重编译 xclbin）
python qwen3-layer/run_npu.py --case currentkv-kvscan-attention-kv16-o-bridge \
    --current-token 91 --patch-from-token 1007
```

**当前活跃 case**：

| Case | 验证边界 |
|------|---------|
| `currentkv-full-layer-q4nx-down-bridge` | 完整闭环：hidden→Q/K/V→attention→O→up/gate→SwiGLU→down→output |
| `currentkv-kvscan-attention-kv16-o-bridge` | KV cache 写回 + scan + attention + O |
| `q4nx-qkv-body-post-bridge` | Q/K/V handoff 诊断：hidden replay → Q4NX Q/K/V → c1r3 postprocess |

历史 case（bridge、shape、单阶段 smoke）已下线——它们的约束被并入共享生成器和 `physical_contract.py`。

## 7.5 共享工具

| 文件 | 职责 |
|------|------|
| `npu_build.py` | 编译流水线：扫描 MLIR `link_with` → 编译 kernel .o → aiecc → xclbin |
| `q4nx_reference.py` | Q4NX chunk 的 CPU 参考数学（反量化 + MAC） |
| `qkv_compact_reference.py` | Q/K/V/O compact record layout helper |
| `bridge_generate/reference/runner.py` | c6r1/c1r1/main16 bridge case（共享桥验证） |
| `swiglu_generate/reference/runner.py` | up/gate compact → c6r2 SwiGLU case |
| `c1r2_generate/reference/runner.py` | O compact → c1r2 replay → main16 case |
| `shape_generate/reference/runner.py` | Q fanout + KV split + Shape-A/B + O bridge case |

## 7.6 构建产物

`build/` 目录下每个 case 有一个子目录，包含：
- `design.mlir`：生成的 MLIR-AIE 源码
- `design.xclbin`：编译后的 NPU 二进制（包含 kernel ELF + 路由配置）
- `design.bin`：instruction stream（可被 patch）
- `design-tokenX-to-tokenY.bin`：patched instruction stream

---

# 八、当前进度与剩余差距

## 已验证（真机通过）

### 物理数据流闭环

- 7 个 projection phase（Q/K/V/O/up/gate/down）全部使用 Q4NX weight stream + main16 DMA1
- Row1 S2MM4/5 weight ingress + MM2S0-3 fanout 与 S2MM0-3 compact gather 共存
- c1r2 packet0 full-vector replay → c1r1 bridge → main16 DMA0 activation ring
- c6r1 packet2（attention）和 packet1（FFN）复用 c1r1 shared bridge
- 48-record upgate body trace + reusable-slot compact + c6r2 payload-half ABI
- 全层 27-core NPU 数据流闭环（默认 token127，NPU 时间 ~110 ms）

### Current K/V + Attention

- Packet8/9 当前 K/V 写回 → host KV cache BO（two-BD even/odd scatter）
- Rounded KV scan（split K/V channel）+ iterated BD + queue repeat count
- Shape-A block scoring + tail-token RTP mask
- Shape-B online-softmax merge + attention output
- Packet2 返回 → O phase handoff

### Instruction Patch

- token127 xclbin + patched token91 design.bin → 逐 word 等于重新编译的 token91
- token1007 cache-capacity PDI（63 rounded blocks）patch 到 token91 → 真机通过
- 证明高上下文容量 PDI 可复用，每 token 只 patch instruction stream

### 关键约束验证

- Shim BD 0..15 上限（AIECC 拒绝 ID>15）
- Memtile BD bank 规则（偶通道 BD 0-23，奇通道 BD 24+）
- BD block 单 release 限制 → stage-level counting lock
- RTP-before-runtime-lock 顺序（否则 current slot = token0）
- 每个 S2MM channel 一个 row（不能用单 channel 多 BD 代替 multi-channel）

## 剩余差距

### 数值校准（最大差距）

| 模块 | 当前状态 | 目标 |
|------|---------|------|
| Attention（Shape-A/B） | kv16 固定点 Q12 exp + int32 accumulator | 校准到 Qwen3 bf16/fp32 softmax |
| c1r2 RMSNorm/replay | bounded numeric scale (256) + int32 sqrt | 校准到 Qwen3 RMSNorm 精度 |
| c6r2 SwiGLU | bf16 输入/输出 ABI 已对齐，近似 SiLU | 校准到 Qwen3 SwiGLU 精度 |
| c1r3 Q/K norm + RoPE | 有实现 | 校准 exact scale/rotation constant |

### 生产集成

- KV cache 运行时保持 block-major scan layout（当前是 test harness 手动填充）
- 多层串联（当前验证单层）
- RMSNorm/RoPE 权重从模型文件加载
- 生产 host runtime（当前是 Python test runner）

### 与 MyLM 对齐（可选方向）

- c1r2 内部 register-level ping/pong/value layout
- Shape carrier exact lane order
- N-block tile value order
- up/gate adjacent-pair 顺序
- Attention output head-pair order

这些被标记为"IRON-ABI-v0 定义"，如果要复用 MyLM binary kernel 需要逐项校准。

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
         currentkv attention 边界（KV writeback + scan + Shape）
  ↓
         Q4NX Q/K/V body（替换 deterministic Q/K/V producer）
  ↓
current  currentkv-full-layer-q4nx-down-bridge
         （全部 7 phase Q4NX + attention + instruction patch）
  ↓
next     数值校准 → 生产精度 → 多层 → runtime 集成
```
