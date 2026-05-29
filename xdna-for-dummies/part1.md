# 第一部分：硬件基础

---

## 第一章：为什么需要另一种计算模型

### 三种机器，三种哲学

**CPU：万能但慢。** CPU 是冯诺依曼机器——取指令、解码、执行、写回，一步一步来。它什么都能算，但每一步都要"想"：下一条指令是什么？数据在哪？要不要跳转？这套机制让 CPU 极其灵活，但也意味着大量时间花在"决定做什么"而不是"真正做计算"上。

**GPU：猛但吃不饱。** GPU 靠成千上万个小核心同时跑同一段程序（SIMT）。它的哲学是"用吞吐量换延迟"——单个线程不快，但成千上万个一起跑，总吞吐量惊人。前提是：你得有足够多的并行工作喂给它。

**NPU：预配置的数据流机器。** NPU 的哲学完全不同——不让 CPU 每步调度，而是把整个计算图预先配置成一台硬件机器。数据流进去，结果流出来，中间没有人做决定。

### 问题出在哪

考虑 LLM 推理的 decode 阶段：每次只生成一个 token。这意味着：

```
输入：一条 4096 维向量（8 KB）
权重：一层约 115 MB
输出：一条 4096 维向量（8 KB）
```

计算量很小（一次矩阵向量乘），但数据搬运量巨大（115 MB 权重要从内存读进来）。这是典型的 **memory-bound** 场景。

GPU 在这种场景下的问题：
- 成千上万个核心，但只有一条向量要算，大部分核心空转
- 每个算子做完，中间结果写回显存，下一个算子再读回来，带宽被中间结果和权重一起争抢
- kernel launch 开销在微秒级，而单 token 计算本身也就几十微秒

### NPU 的解法

NPU 不是"更快的 CPU"或"更小的 GPU"。它是一种完全不同的机器：

1. **预配置**：编译阶段就把所有 DMA 任务、数据路径、同步信号配好
2. **数据驱动**：没有"取指令"循环，数据到了就自动触发计算
3. **片上交接**：中间结果不回主存，直接从一个计算单元流向下一个
4. **一次启动**：host 只做一次 runtime 提交，NPU 自动跑完整个计算图

### 类比

| | CPU | GPU | NPU |
|---|---|---|---|
| 像什么 | 一个全能工匠，每步看图纸 | 千人抄写班，同时抄同一段 | 自动流水线，接通管道开阀门就跑 |
| 擅长 | 复杂控制流 | 大规模并行 | 固定数据流，带宽优先 |
| 调度方式 | 每条指令都要取/解码 | host launch kernel | 预配置 + 一次性启动 |
| 中间结果 | 寄存器/cache | 写回显存 | 片上直接流动 |

### 什么时候该用 NPU

NPU 特别适合：
- 计算图固定（每次 decode 做同样的事）
- memory-bound（瓶颈在带宽而非算力）
- 中间结果大但下游马上消费（融合能省带宽）
- 端侧设备（笔记本、手机），功耗和带宽都有限

不适合：
- 计算图动态变化（每次形状不同）
- 需要复杂控制流（大量条件分支）
- compute-bound 且需要极高算力密度

---

## 第二章：硬件地图——Tile 阵列长什么样

### 三层结构

NPU 芯片内部是一个二维网格，从下到上分三层：

```
┌─────────────────────────────────────────────────────┐
│  Row 5  │                                           │
│  Row 4  │  计算层（Compute Tile）                    │
│  Row 3  │  每个格子是一个小处理器，跑 C/C++ 程序      │
│  Row 2  │                                           │
├─────────┼───────────────────────────────────────────┤
│  Row 1  │  中转层（Memtile）                         │
│         │  大片上缓存，负责拆分/缓冲/转发             │
├─────────┼───────────────────────────────────────────┤
│  Row 0  │  接口层（Shim Tile）                       │
│         │  主存入口——数据从这里进出 NPU               │
└─────────────────────────────────────────────────────┘
```

**Shim（Row 0）**——NPU 的大门。所有数据都通过 Shim 进出芯片：权重从主存读入、计算结果写回主存、host 的控制指令也从这里进来。Shim 自己不做计算，只负责搬运。

**Memtile（Row 1）**——中转站。有几百 KB 的片上缓存（比 Compute Tile 大得多），但不跑用户程序。它的工作是：接收大块数据拆成小块、把小块结果汇聚成大块、用双缓冲让传输和计算重叠。你可以把它想象成物流中心的分拣区。

**Compute Tile（Row 2+）**——真正干活的地方。每个 tile 有几十 KB 本地内存、一个小处理器核心和自己的 DMA 引擎。它跑编译好的 C/C++ 程序，但只能访问自己的本地内存——看不到其他 tile 的数据。

### 坐标系统

每个格子用 `c{列}r{行}` 标记。例如：
- `c3r2` = 第 3 列、第 2 行的 Compute Tile
- `c0r1` = 第 0 列的 Memtile
- `c5r0` = 第 5 列的 Shim

一个典型的 XDNA 分区是 8 列 × 6 行 = 48 格：

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

32 个 Compute Tile + 8 个 Memtile + 8 个 Shim = 48 格。

### 每层的资源对比

| | Compute Tile | Memtile | Shim |
|---|---|---|---|
| 本地内存 | ~64 KB | ~512 KB | 无（只做搬运） |
| 处理器核心 | 有（跑 C/C++） | 无 | 无 |
| DMA 引擎 | 有（几个 channel） | 有（更多 channel） | 有 |
| Lock | ~16 个 | ~64 个 | 有限 |
| BD slot | 若干 | 偶通道 0-23，奇通道 24-47 | 最多 16 个 |

### 关键认知

1. **Compute Tile 很小**。64 KB 装不下一个 4096×4096 的 bf16 矩阵（那需要 32 MB）。所以大算子必须拆成很多小块，让每个 tile 一次只处理一小片。

2. **Tile 之间看不到彼此的内存**。c3r2 不能直接读 c3r3 的 buffer。数据必须通过 DMA + stream 显式搬运。

3. **物理连接是固定的**。不是任意两个 tile 之间都能直接通信。数据沿着物理连线从一个 tile 流向相邻的 tile，远距离通信需要经过中间节点。

---

## 第三章：数据怎么搬——DMA 和 Buffer Descriptor

### DMA：不需要 CPU 的搬运工

在 CPU 世界里，搬数据通常是 `memcpy`——CPU 亲自从源地址读、往目标地址写。NPU 里不这样做。每个 tile 都有自己的 DMA 引擎，你给它一份"任务单"，它就自己搬，不需要任何处理器参与。

DMA 有两个方向：
- **S2MM**（Stream to Memory-Mapped）：从 stream 接收数据，写入本地 buffer。相当于"收件"
- **MM2S**（Memory-Mapped to Stream）：从本地 buffer 读数据，发送到 stream。相当于"寄件"

每个 tile 有多个 DMA channel，可以同时搬运多份数据。例如一个 Compute Tile 可能有：
- S2MM channel 0：接收激活输入
- S2MM channel 1：接收权重输入
- MM2S channel 1：发送计算结果

### BD：DMA 的任务单

BD（Buffer Descriptor）是 DMA 的"任务单"。一个 BD 描述一次搬运任务的所有细节：

```
一个 BD 包含：
┌─────────────────────────────────────────┐
│ buffer_addr   : 从哪个 buffer 的哪开始   │
│ buffer_length : 搬多少数据               │
│ stride        : 如果不是连续的，怎么跳    │
│ next_bd       : 搬完后下一个任务是什么    │
│ packet_id     : 给数据贴什么路由标签      │
│ acquire_lock  : 开始前等哪个锁           │
│ release_lock  : 完成后释放哪个锁         │
└─────────────────────────────────────────┘
```

DMA 不需要"指令"告诉它"现在搬这个"。BD 和 lock 组合在一起就构成了完整的自动行为：lock 就绪 → 按 BD 描述搬运 → 释放 lock → 跳到 next BD。

### BD Ring：循环任务

多个 BD 可以首尾相连形成一个环：

```
BD0 → BD1 → BD0 → BD1 → ...（ping-pong）
```

DMA 会自动在环里轮转。配合 lock，这就变成了一个自动的生产者-消费者 FIFO：

```
BD0 (ping): acquire ping_empty → 搬数据到 ping buffer → release ping_full → next=BD1
BD1 (pong): acquire pong_empty → 搬数据到 pong buffer → release pong_full → next=BD0
```

DMA 不停地转这个环，但每一步都要先拿到 lock 才能继续。这就是"数据驱动"——不是有人告诉它"现在搬"，而是 lock 就绪了它就自动搬。

### 2D Stride：不连续的搬运

有时候你要搬的数据不是一整块连续内存，而是一个矩阵的某些行或某个子矩阵。BD 的 stride 字段可以描述"每搬 N 字节，跳过 M 字节"：

```
buffer:  [████----████----████----]
          row0    row1    row2

stride 描述：
  d0_size = 4（每行 4 个元素）
  d1_size = 3（搬 3 行）
  d1_stride = 8（每行起始间隔 8 个元素）
```

这让你不需要先把数据整理成连续块再搬——直接告诉 DMA 怎么跳就行。

### 硬约束：BD 是稀缺资源

关键限制：**Shim tile 每个只有 16 个 BD slot**。

这意味着如果你的算子需要描述 20 种不同的搬运模式，16 个 BD 不够用。你必须用 BD ring 复用（同一个 BD 反复执行）、iteration field（让一个 BD 自动递增地址）、或 queue repeat count（同一个 BD 反复入队）。

Memtile 的 BD 更多一些，但有 bank 规则：**偶数 channel 只能用 BD 0-23，奇数 channel 只能用 BD 24-47**。违反这个规则不会编译报错，但运行时会静默死锁。第九章会展示这在实际编排中的影响。

### Iterated BD：一个 BD 扫描多个 Block

如果你要用一个 BD 反复传输相似的数据（比如扫描 KV cache 的每个 16-token block），可以用 iteration：

```
一个 BD:
  buffer_length = 4096（一个 block 的数据量）
  iteration_size = N（扫描 N 个 block）
  iteration_stride = 8191（每个 block 之间的偏移）

配合 push_queue:
  repeat_count = N - 1（让这个 BD 执行 N 次）
```

只写 `iteration_size` 不写 `repeat_count` 是常见 bug——DMA 只会发送第一个 segment 然后停下来，导致 timeout。

---

## 第四章：同步怎么做——Lock 和信用机制

### 问题：谁先谁后？

NPU 里没有"调用-返回"的概念。producer 和 consumer 各自独立运行。如果没有同步机制：

- producer 还没写完，consumer 就来读 → 读到垃圾数据
- consumer 还没读完，producer 又覆盖 → 数据丢失
- 某个 DMA 启动太早，读到上一轮的旧数据 → 结果错误

在 CPU 上你可能用 mutex 或 condition variable。NPU 上用的是 **Lock**——一种硬件计数器。

### Lock 的本质

一个 Lock 就是一个整数，支持两个操作：

- **acquire(lock, value)**：等到 lock 的值 ≥ value，然后减去 value
- **release(lock, value)**：给 lock 的值加上 value

这两个操作是原子的。如果 acquire 时计数不够，DMA 或 core 会**停下来等**，直到别人 release 足够的计数。

### 最小例子：单 buffer 同步

```
lock: buffer_empty (初值 = 1)
lock: buffer_full  (初值 = 0)

Producer:                     Consumer:
  acquire(empty, 1)             acquire(full, 1)
  写入 buffer                    从 buffer 读取
  release(full, 1)              release(empty, 1)
```

时序：
1. Producer acquire empty（初值=1，成功）→ 写入
2. Consumer acquire full（初值=0，等待...）
3. Producer release full（变成 1）
4. Consumer acquire full（成功）→ 读取
5. Consumer release empty（变成 1）
6. Producer 可以再次 acquire empty → 循环

### Ping-Pong 双缓冲

单 buffer 的问题：producer 写的时候 consumer 在等，consumer 读的时候 producer 在等。永远不能重叠。

双缓冲解决这个问题——两个 buffer 交替使用：

```
buffer: ping, pong
lock:   ping_empty(1), ping_full(0), pong_empty(1), pong_full(0)

Producer:                        Consumer:
  acquire(ping_empty)              acquire(ping_full)
  写入 ping                         读取 ping
  release(ping_full)               release(ping_empty)
  acquire(pong_empty)              acquire(pong_full)
  写入 pong                         读取 pong
  release(pong_full)               release(pong_empty)
  ... 循环 ...                      ... 循环 ...
```

现在 producer 写 pong 的同时 consumer 可以读 ping——传输和计算重叠了。

在 XDNA 上，这通常表达为 BD ring：两个 BD（一个管 ping，一个管 pong）首尾相连。DMA 自动交替使用两个 buffer，每次都先 acquire lock 再搬运。

### Counting Lock：一对多

如果一个 producer 的数据要给 4 个 consumer 用呢？

用 counting lock：初值设为 consumer 数量。

```
lock: slot_empty (初值 = 4)   ← 4 个 consumer 都要 release 才能再写
lock: slot_full  (初值 = 0)

Producer:
  acquire(slot_empty, 4)   ← 等所有 consumer 都读完
  写入 buffer
  release(slot_full, 4)    ← 通知所有 consumer

Consumer 0..3（每个各自）:
  acquire(slot_full, 1)    ← 只拿 1 份信用
  读取 buffer
  release(slot_empty, 1)   ← 归还 1 份信用
```

4 个 consumer 各自 release 1，加起来 = 4，producer 才能重新拿到 4 开始下一轮。

### 死锁的常见原因

NPU 死锁的表现是 **timeout**——runtime 等不到 NPU 完成。定位困难，因为你看不到"哪个 lock 在等"。常见原因：

1. **acquire 没配 release**：某个路径上忘了 release，下游永远等不到
2. **lock 初值错误**：empty lock 初值应该 = buffer slot 数，full lock 初值应该 = 0
3. **BD bank 规则违反**：Memtile 偶通道用了高编号 BD（属于奇通道）→ 编译不报错但运行死锁
4. **多 producer 共用 channel**：两个不相关的 producer 往同一个 physical channel 写，顺序不确定 → lock 计数混乱
5. **counting lock 计数不平衡**：producer release 4 但只有 3 个 consumer release back → 永久少 1

记住一条经验法则：**NPU timeout ≈ 先查 lock 配对，再查 BD bank，再查 iteration/repeat**。

---

## 第五章：数据怎么路由——Stream 和 Packet

### Stream：固定的物理管道

Tile 之间通过 **stream** 传输数据。Stream 是编译时确定的物理连线——从哪个 tile 的哪个 MM2S channel 出发，到哪个 tile 的哪个 S2MM channel 接收。

在 MLIR-AIE 里，一条 stream 的声明长这样：

```
aie.flow(%tile_a, DMA : 0, %tile_b, DMA : 1)
```

意思是：tile_a 的 MM2S channel 0 连接到 tile_b 的 S2MM channel 1。

stream 有两种模式：
- **circuit flow**：独占一条物理线，这条线在整个运行期间只给这对 producer/consumer 用
- **packet flow**：多路数据共享同一条物理线，用 packet ID 区分

### 问题：线不够用

一个 tile 的 MM2S channel 只有几个（通常 2-6 个）。但一个复杂算子可能需要十几路不同的数据流。如果每路都用 circuit flow 独占一条线，物理资源很快耗尽。

### Packet 路由：同一条线，多路复用

Packet 路由的思路：在数据的前面加一个 header（包含 packet ID），接收端根据 ID 筛选：

```
物理线 ═══════════════════════════════════════►
         ┌─────────┐  ┌─────────┐  ┌─────────┐
数据包:   │ pkt_id=1│  │ pkt_id=2│  │ pkt_id=1│
         └─────────┘  └─────────┘  └─────────┘
              │              │              │
              ▼              ▼              ▼
         接收端 A        接收端 B       接收端 A
         (过滤 id=1)    (过滤 id=2)   (过滤 id=1)
```

在 MLIR-AIE 里：

```
aie.packet_flow(0x01) {        // packet ID = 1
  aie.packet_source<%tile_a, DMA : 0>
  aie.packet_dest<%tile_b, DMA : 0>
}
aie.packet_flow(0x02) {        // packet ID = 2
  aie.packet_source<%tile_a, DMA : 0>   // 同一个源 channel
  aie.packet_dest<%tile_c, DMA : 0>     // 不同的目标
}
```

同一个 MM2S channel 可以发出不同 packet ID 的数据，不同接收端根据 ID 各取所需。

### 实际例子

在一个融合层中，c6r1（hub tile）需要发送两种完全不同的数据：
- **attention 结果**（2048 dword）→ 给 O 投影用
- **FFN 中间值**（6144 dword）→ 给 down 投影用

但它们去往同一个目的地（c1r1 shared bridge）。解法：

```
attention 结果: packet ID = 2
FFN 中间值:    packet ID = 1
```

c1r1 收到后按 packet ID 区分。从下游 Main16 的角度看，两者都变成了 128-dword activation chunk——它不需要知道上游的区别。

### 硬约束

**Packet ID 必须全局唯一。** 如果两个不相关的数据流用了同一个 packet ID，接收端会收到错误的数据，而且这个错误可能表现为跨列死锁——极难定位。

在设计阶段，所有 packet ID 应该集中登记在一个地方，新增路径时必须查表避免冲突。

### Circuit Flow vs Packet Flow 的选择

| | Circuit Flow | Packet Flow |
|---|---|---|
| 带宽 | 独占，无 header 开销 | 共享，每包多 1 dword header |
| 延迟 | 最低 | 略高（路由器筛选） |
| 灵活性 | 一对一固定 | 一对多、多对一都行 |
| 资源消耗 | 每路一条物理线 | 多路复用一条线 |
| 适合 | 高带宽持续流（权重流） | 间歇性多路数据（packet/record） |

---

## 第六章：从 CPU 思维转换到数据流思维

这一章没有新硬件概念，但可能是最重要的一章。

### CPU 程序员的心智模型

```python
def transformer_layer(hidden):
    q = linear(hidden, w_q)      # 做完 Q 再做 K
    k = linear(hidden, w_k)
    v = linear(hidden, w_v)
    attn = attention(q, k, v)    # 等 Q/K/V 都好了再做 attention
    out = linear(attn, w_o)      # 等 attention 好了再做 O
    return out
```

CPU 思维：一步一步来，每步都"调用"一个函数，等它"返回"，拿到结果再做下一步。

### 数据流思维

在 NPU 上，同样的计算变成：

```
配置阶段（编译时）：
  - Tile A 的 BD ring: 接收 hidden chunk → 接收 weight chunk → 本地 MAC → 发送 record
  - Tile B 的 BD ring: 接收 record → 汇聚 → 发给下游
  - Lock: 连接 A 和 B 的 ping-pong 同步
  - Stream: A 的 MM2S → B 的 S2MM

执行阶段（运行时，第十章详述）：
  - Host 填好输入 BO、权重 BO
  - Host 提交 runtime sequence
  - NPU 自动运转，数据流过所有阶段
  - Host 等待完成，读回结果
```

没有"调用"，没有"返回"。数据到了 + lock 就绪 = 自动开始。

### 关键概念映射

| CPU 概念 | NPU 对应 | 说明 |
|---|---|---|
| 函数调用 | 数据到达 + lock 就绪 | 没有调用者，是数据驱动的 |
| 返回值 | 写入下游 buffer + release lock | 没有返回栈 |
| for 循环 | BD ring 自动轮转 | DMA 自动重复，不需要循环变量 |
| if/else | packet ID 路由 | 编译时确定路径，不是运行时判断 |
| 全局变量 | RTP buffer | host 写入，tile 读取，需要 lock 门控 |
| malloc | 不存在 | 所有 buffer 编译时分配 |
| 线程同步 | lock acquire/release | 硬件原子操作，零开销 |

### 什么东西不能带进来

1. **函数调用栈**。Tile 不能"调用"另一个 tile 的函数。它只能往 stream 发数据，期望对方配好了 BD 会收。

2. **动态内存分配**。所有 buffer 在编译时就确定了大小和位置。运行时不能 malloc。

3. **任意寻址**。Tile 不能随机访问另一个 tile 的内存。数据必须通过 DMA + stream 显式搬运。

4. **隐式顺序**。CPU 上前一行代码一定在后一行之前执行。NPU 上如果没有 lock 连接的两个 DMA 操作是**完全无序**的。想要顺序，必须用 lock 显式表达。

5. **精确异常**。NPU 不会在某个 tile "第 42 行"报错。出错的表现是 timeout 或数据不对。

### 设计时的思考方式

当你拿到一个算子要在 NPU 上实现时，不要想"我要写什么代码"。要想：

```
数据从哪来？        → 定义输入 stream/BD
每个 tile 做什么？  → 定义 core 程序
结果去哪？          → 定义输出 stream/BD
谁等谁？            → 定义 lock
```

这四个问题回答清楚了，实现就是把它们连起来。

---

## 第七章：第一个完整例子——让两个 Tile 传一个向量

### 目标

最简单的有意义的 XDNA 程序：Tile A 产生 256 个 bf16 数据，通过 stream 传给 Tile B，Tile B 读到后写入 host 可读的 buffer。

```
         stream
Tile A ─────────► Tile B ─────► Host 读回
(producer)        (consumer)
```

### 步骤 1：声明 Tile 和 Buffer

```mlir
// 声明两个 Compute Tile
%tile_a = aie.tile(2, 2)   // c2r2
%tile_b = aie.tile(3, 2)   // c3r2

// Tile A 的输出 buffer（ping-pong）
%a_ping = aie.buffer(%tile_a) : memref<128xi32>   // 128 dword = 256 bf16
%a_pong = aie.buffer(%tile_a) : memref<128xi32>

// Tile B 的输入 buffer（ping-pong）
%b_ping = aie.buffer(%tile_b) : memref<128xi32>
%b_pong = aie.buffer(%tile_b) : memref<128xi32>
```

为什么用 `i32` 而不是 `bf16`？因为 DMA 以 dword（32-bit）为单位搬运，一个 dword 装两个 bf16。128 个 i32 = 256 个 bf16。

### 步骤 2：声明 Lock

```mlir
// Tile A 的输出同步
%a_ping_empty = aie.lock(%tile_a, 0) {init = 1}   // 初始"空"，可以写
%a_ping_full  = aie.lock(%tile_a, 1) {init = 0}   // 初始"没满"，不能读
%a_pong_empty = aie.lock(%tile_a, 2) {init = 1}
%a_pong_full  = aie.lock(%tile_a, 3) {init = 0}

// Tile B 的输入同步
%b_ping_empty = aie.lock(%tile_b, 0) {init = 1}
%b_ping_full  = aie.lock(%tile_b, 1) {init = 0}
%b_pong_empty = aie.lock(%tile_b, 2) {init = 1}
%b_pong_full  = aie.lock(%tile_b, 3) {init = 0}
```

初值是关键：empty lock 初值 = 1（表示 buffer 一开始是空的，producer 可以写），full lock 初值 = 0（buffer 一开始没数据，consumer 不能读）。

### 步骤 3：声明 BD 和 DMA

```mlir
// Tile A 的发送 DMA（MM2S channel 0）
%tile_a_mem = aie.mem(%tile_a) {
  ^ping:
    aie.use_lock(%a_ping_full, Acquire, 1)   // 等 core 写完
    aie.dma_bd(%a_ping : memref<128xi32>)    // 发送 ping buffer
    aie.use_lock(%a_ping_empty, Release, 1)  // 通知 core 可以再写
    aie.next_bd ^pong
  ^pong:
    aie.use_lock(%a_pong_full, Acquire, 1)
    aie.dma_bd(%a_pong : memref<128xi32>)
    aie.use_lock(%a_pong_empty, Release, 1)
    aie.next_bd ^ping                        // 循环回 ping
  ^start:
    aie.dma_start(MM2S, 0, ^ping, ^end)      // 启动 MM2S channel 0
  ^end:
    aie.end
}

// Tile B 的接收 DMA（S2MM channel 0）
%tile_b_mem = aie.mem(%tile_b) {
  ^ping:
    aie.use_lock(%b_ping_empty, Acquire, 1)  // 等 consumer 读完
    aie.dma_bd(%b_ping : memref<128xi32>)    // 接收到 ping buffer
    aie.use_lock(%b_ping_full, Release, 1)   // 通知 consumer 数据来了
    aie.next_bd ^pong
  ^pong:
    aie.use_lock(%b_pong_empty, Acquire, 1)
    aie.dma_bd(%b_pong : memref<128xi32>)
    aie.use_lock(%b_pong_full, Release, 1)
    aie.next_bd ^ping
  ^start:
    aie.dma_start(S2MM, 0, ^ping, ^end)
  ^end:
    aie.end
}
```

注意 BD 的 lock 方向：
- 发送 BD：acquire **full**（等 core 写完）→ 发送 → release **empty**（通知 core 可再写）
- 接收 BD：acquire **empty**（等 consumer 读完）→ 接收 → release **full**（通知 consumer 数据来了）

### 步骤 4：连接 Stream

```mlir
aie.flow(%tile_a, DMA : 0, %tile_b, DMA : 0)
// Tile A 的 MM2S channel 0 → Tile B 的 S2MM channel 0
```

一行代码，建立物理连线。

### 步骤 5：写 Core 程序

```mlir
// Tile A 的 core：产生数据
aie.core(%tile_a) {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index

  // 写 ping buffer
  aie.use_lock(%a_ping_empty, Acquire, 1)    // 等 buffer 空
  scf.for %i = %c0 to %c128 step %c1 {
    %val = ... // 产生数据
    memref.store %val, %a_ping[%i] : memref<128xi32>
  }
  aie.use_lock(%a_ping_full, Release, 1)     // 通知 DMA 可以发了

  // 写 pong buffer
  aie.use_lock(%a_pong_empty, Acquire, 1)
  scf.for %i = %c0 to %c128 step %c1 {
    %val = ...
    memref.store %val, %a_pong[%i] : memref<128xi32>
  }
  aie.use_lock(%a_pong_full, Release, 1)

  aie.end
}

// Tile B 的 core：消费数据
aie.core(%tile_b) {
  // 读 ping buffer
  aie.use_lock(%b_ping_full, Acquire, 1)     // 等数据到达
  // ... 使用 %b_ping 中的数据 ...
  aie.use_lock(%b_ping_empty, Release, 1)    // 通知 DMA 可以再收

  // 读 pong buffer
  aie.use_lock(%b_pong_full, Acquire, 1)
  // ... 使用 %b_pong 中的数据 ...
  aie.use_lock(%b_pong_empty, Release, 1)

  aie.end
}
```

### 完整时序

```
时间 ──────────────────────────────────────────────────►

Tile A core:  [写 ping]         [写 pong]
Tile A DMA:            [发 ping]          [发 pong]
              ─────stream─────────stream──────────►
Tile B DMA:            [收 ping]          [收 pong]
Tile B core:                    [读 ping]          [读 pong]
```

双缓冲让 Tile A 写 pong 和 Tile B 读 ping 重叠。

### 常见错误清单

| 错误 | 表现 | 原因 |
|------|------|------|
| lock 初值反了 | timeout | empty 初值=0 → producer 一开始就等不到 |
| BD 长度不匹配 | timeout 或数据截断 | 发送 128 dword 但接收 BD 写的是 64 |
| channel 方向写错 | 编译报错或死锁 | S2MM/MM2S 搞混 |
| 忘记 next_bd | DMA 只搬一次就停 | 没有形成 ring |
| 没连 stream flow | 数据发出去没人收 | timeout |
| Core 在 DMA 启动前就 release | DMA 读到未初始化数据 | 时序假设错误 |

### 从这个例子到真实算子

真实的 NPU 算子就是这个例子的放大版：
- 2 个 tile → 48 个 tile
- 1 条 stream → 几十条 stream + packet 路由
- 简单 copy → Q4NX 反量化 + MAC + online softmax
- 1 次传输 → BD ring 循环几百次

但底层机制完全一样：**buffer + lock + BD + stream**。理解了这个最小例子，就理解了 XDNA 编程的核心。
