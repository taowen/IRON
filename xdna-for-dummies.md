# XDNA NPU 编程入门：从零开始理解数据流机器

面向已有 CPU/GPU 编程经验但从未接触过 NPU 数据流编程的读者。

全书分四部分：
- **第一部分（硬件基础）**：建立心智模型，能看懂一个最小例子
- **第二部分（编程实操）**：Compute Tile / Memtile / Host / 编译，能动手写代码
- **第三部分（设计 Pattern）**：11 个通用规律，拿到新算子知道怎么拆
- **第四部分（综合实战）**：调试方法 + 端到端案例

---

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

---

# 第二部分：编程实操

> 第一部分讲了硬件长什么样、数据怎么流。这一部分讲**怎么写代码**：Compute Tile 上的 C++ kernel、Memtile 的纯 DMA 编排、Host 侧的提交流程、以及从 Python 到 xclbin 的编译管线。

---

## 第八章：Compute Tile 编程——在小盒子里写 C++

### 编程模型：收一块 → 算 → 发一块

第二章提到，Compute Tile 只有 ~64 KB 本地内存，看不到其他 tile。这决定了 kernel 的基本形态——不是"接收完整输入、返回完整输出"，而是一个流式循环：

```
初始化 accumulator

loop:
    acquire input_full     ← 等 DMA 搬完一个 chunk
    acquire output_empty   ← 等下游腾出空间

    compute(input_buf → output_buf)

    release input_empty    ← 通知 DMA 可以搬下一个 chunk
    release output_full    ← 通知下游数据已就绪

    goto loop
```

Core 程序只管 local buffer 里的数据；DMA 在后台通过第四章介绍的 ping-pong BD ring 把 stream 数据搬进搬出。两者并行运行，通过 lock 同步。

### 一个真实 kernel 的结构

以 Q4NX 投影 kernel 为例（简化自 `main_projection_q4nx.cc`）：

```c
// 本地状态：accumulator，生命周期 = 整个 phase
static float accum[32];  // 32 行输出

// 每次收到一个 activation chunk + weight chunk 就调用
void q4nx_chunk_accum(
    bfloat16 *weight_chunk,      // 1280 dword = 5120 字节 Q4NX
    bfloat16 *activation_chunk,  // 128 dword = 256 个 bf16
    int32_t num_rows             // 32
) {
    // 从 Q4NX chunk 中解包 scale, zero_point, int4 data
    bfloat16 *scales = weight_chunk;
    bfloat16 *zeros = weight_chunk + 32 * 8;
    uint8_t *data = (uint8_t *)(weight_chunk + 32 * 8 * 2);

    for (int row = 0; row < num_rows; row++) {
        float row_acc = 0.0f;
        for (int group = 0; group < 8; group++) {
            float scale = scales[group * 32 + row];
            float zero = zeros[group * 32 + row];
            for (int dim = 0; dim < 32; dim++) {
                int col = group * 32 + dim;
                uint8_t q4 = /* 从 data 中解包 4-bit */;
                float weight = (float(q4) - zero) * scale;
                row_acc += weight * float(activation_chunk[col]);
            }
        }
        accum[row] += row_acc;
    }
}

// 所有 chunk 累加完毕后，flush 输出
void q4nx_flush_output(bfloat16 *output, int32_t num_rows) {
    for (int row = 0; row < num_rows; row++) {
        output[row] = bfloat16(accum[row]);
        accum[row] = 0.0f;  // 为下一个 phase 清零
    }
}
```

注意这里的设计：
- `accum` 是 tile-local 的 static 变量，跨多个 chunk 累加
- 每次只处理 256 列输入（一个 chunk），不是完整 4096 列
- 量化反量化在 tile 内部在线完成，不生成中间全精度矩阵
- flush 后 accumulator 清零，准备服务下一个 output block 或 phase

### 两种角色的 Kernel

**同构 kernel**：多个 tile 跑完全相同的程序，只是处理不同的数据切片。

```
Main16（16 个 tile）:
  全部跑 main_projection_q4nx.cc
  每个 tile 负责 32 行输出
  区别只在于：DMA 给它们喂的 weight chunk 对应不同的输出行
```

好处：一份 kernel 代码编译一次，16 个 tile 链接同一个 .o 文件。

**异构 kernel**：每个 tile 跑不同的程序，因为它们的角色完全不同。

```
c1r2: full_vector_station.cc  — RMSNorm / residual / replay
c1r3: postprocess_qkv.cc      — Q/K norm + RoPE + current K/V 路由
c6r2: swiglu.cc                — SiLU(gate) × up
c0r2: edge_attention.cc        — Shape-A 评分
c0r3: edge_attention.cc        — Shape-B 加权求和（同一个源文件，不同入口）
```

异构 kernel 的 MLIR 声明中每个 tile 的 `link_with` 指向不同的 .o 文件。

### 输出格式：Record

Compute Tile 的输出不是"一个大 tensor"，而是一个个小的 **record**：

```
一个 record = 17 dword:
┌──────────────┬───────────────────────────────────┐
│ header (1 dw)│ payload (16 dw = 32 bf16)         │
└──────────────┴───────────────────────────────────┘
```

header 里带路由信息（phase、block 坐标、tile 坐标），下游的 Memtile 根据 header 知道这块数据该放哪。

为什么不直接写大 tensor？因为 16 个 tile 并发产出，如果每个都直接写大 tensor 的某个位置，地址计算和 BD 配置会极其复杂。record 是一个稳定的小包格式，让汇聚逻辑变得简单规整。

### 常见陷阱

| 陷阱 | 后果 | 怎么避免 |
|------|------|---------|
| buffer 超过本地内存 | 编译报错或踩到别的 buffer | 计算每个 buffer 的字节数，确保总和 < 64 KB |
| 忘记清零 accumulator | 上一个 phase 的残留值污染当前结果 | flush 后显式清零 |
| 输出大小和 BD length 不匹配 | DMA 少搬或多搬，下游收到错误数据 | record 大小和 BD 的 buffer_length 必须一致 |
| 使用浮点除法 | AIE2P 没有硬件除法，编译器会生成很慢的软件除法 | 用乘倒数、查表或整数近似 |
| 大循环不展开 | 性能差，pipeline stall | 对热循环用 `#pragma unroll` 或手动展开 |

---

## 第九章：Memtile 编程——纯 DMA 编排

### Memtile 不跑程序

第二章提到 Memtile 没有处理器核心、但有 ~512 KB 大内存。它的全部行为靠 BD + lock 描述——你不写 C++ 代码，而是配置 DMA 的任务单，让硬件自动完成接收、缓冲、转发。

### 四种典型用途

**1. 扇出（Fan-out）**

一份数据来自 Shim，要分发给 4 行 Compute Tile：

```
Shim ──► Memtile buffer
             ├──► Row2 tile
             ├──► Row3 tile
             ├──► Row4 tile
             └──► Row5 tile
```

实现方式：一个 S2MM BD 接收，4 个 MM2S BD 从同一个 buffer 的不同偏移读取并发送。或者用 counting lock——S2MM 写满后 release 4，4 个 MM2S 各 acquire 1。

**2. 汇聚（Gather/Compact）**

4 行 Compute Tile 各产出一个小 record，Memtile 汇聚成一个大块：

```
Row2 tile ──┐
Row3 tile ──┼──► Memtile buffer (拼接) ──► 输出
Row4 tile ──┤
Row5 tile ──┘
```

实现方式：4 个 S2MM channel（每行一个），各自用 2D stride BD 把 record 写入 buffer 的正确偏移。写满后一个 MM2S BD 连续发出整个 compact 块。

**3. 双缓冲中转**

上游速度和下游速度不同。Memtile 用 ping-pong 吸收差异：

```
上游 ──► [ping] ──► 下游
         [pong]
         交替使用
```

DMA 往 ping 写的时候，下游从 pong 读。上下游完全解耦。

**4. 切片/重排**

一个大数据块进来，Memtile 用 2D stride BD 切成多个小块分发：

```
[2048 dword 完整向量]
    ├── BD0: offset=0,    len=512 ──► Shape-A 0
    ├── BD1: offset=512,  len=512 ──► Shape-A 1
    ├── BD2: offset=1024, len=512 ──► Shape-A 2
    └── BD3: offset=1536, len=512 ──► Shape-A 3
```

### BD Bank 规则的实际影响

第三章提到了 Memtile BD bank 规则（偶通道 BD 0-23，奇通道 BD 24-47）。在实际编排中这意味着：你给 channel 分配 BD 编号时必须查表，而不是随意递增。

例如：权重 ingress 用 S2MM channel 4（偶数），那它的 BD 必须在 0-23 范围内。如果不小心给它分配了 BD 30（属于奇数 bank），编译不报错，但 NPU 会死——静默死锁，表现为 timeout。

### Channel 所有权

Memtile 有多个 S2MM 和 MM2S channel。在复杂设计中，不同 channel 必须严格划分职责：

```
Row1 Memtile channel 分工示例：
┌─────────────────────────────────────────────────┐
│ S2MM 0-3: compact record 汇聚（从 Compute Tile）│
│ S2MM 4-5: 权重入口（从 Shim）                   │
│ MM2S 0-3: 权重分发（到 Compute Tile 各行）       │
│ MM2S 5:   compact 输出（到 bridge/downstream）   │
└─────────────────────────────────────────────────┘
```

为什么不能混用？因为每个 channel 有自己的 BD ring 和 lock 状态。如果权重流和 compact 流共用一个 channel，它们的 lock 会互相干扰，BD ring 的轮转顺序也会混乱。

这不是"推荐做法"，而是"不这样做就死锁"。旧版本曾尝试把权重走 S2MM0/1（本该归 compact 用），直接导致跨阶段死锁。

### 一个具体例子：权重双缓冲分发

把第四章的 ping-pong 和 counting lock 组合起来，看一个真实的 Memtile 编排：

```
目标：从 Shim 读入 Q4NX 权重 chunk，双缓冲后分发给 4 行 Compute Tile

结构：
  S2MM channel 4: 从 Shim 接收 → patch0 ping/pong buffer
  S2MM channel 5: 从 Shim 接收 → patch1 ping/pong buffer
  MM2S channel 0: 从 patch0 ping/pong → Row2
  MM2S channel 1: 从 patch0 ping/pong → Row3
  MM2S channel 2: 从 patch1 ping/pong → Row4
  MM2S channel 3: 从 patch1 ping/pong → Row5

每个 patch buffer:
  大小 = ROWS_PER_PATCH × CHUNK_BF16 = 2 × 2560 = 5120 bf16

Lock（counting lock，初值=2 表示两个 consumer row）:
  patch0_ping_empty (init=2)
  patch0_ping_full  (init=0)
  patch0_pong_empty (init=2)
  patch0_pong_full  (init=0)
```

S2MM BD ring（ingress 侧）：
```
^ping: acquire(ping_empty, 2) → 搬入 ping → release(ping_full, 2) → next=^pong
^pong: acquire(pong_empty, 2) → 搬入 pong → release(pong_full, 2) → next=^ping
```

MM2S BD ring（每个 row 各自）：
```
^ping: acquire(ping_full, 1) → 从 ping 发送一行的份额 → release(ping_empty, 1) → next=^pong
^pong: acquire(pong_full, 1) → 从 pong 发送一行的份额 → release(pong_empty, 1) → next=^ping
```

两行共享同一个 buffer：S2MM release 2 份信用后两个 MM2S 各 acquire 1，两个都读完各 release 1 回来凑够 2，S2MM 才能覆盖——这正是第四章 counting lock 的应用。

---

## 第十章：Host 侧——怎么把任务提交给 NPU

### BO：Host 和 NPU 的共享内存

BO（Buffer Object）是 XRT 提供的共享内存抽象。Host 程序和 NPU 都能访问同一块物理内存：

```python
import numpy as np

# 分配 BO
input_bo = np.zeros(4096, dtype=np.int32)   # host 端 numpy array
weight_bo = np.zeros(30_000_000, dtype=np.int32)  # 约 115 MB 权重

# 填入数据
input_bo[:] = prepare_hidden_vector()
weight_bo[:] = load_q4nx_weights()
```

Host 侧填好 BO 后，NPU 的 Shim DMA 就能从这些 BO 地址开始搬运。

### Runtime Sequence：告诉 NPU 怎么用 BO

光有 BO 不够——NPU 还需要知道"从 BO 的哪个偏移开始搬、搬多少、搬到哪"。这通过 **runtime sequence** 描述：

```
runtime sequence 的三个核心指令：

npu.writebd    配置一个 Shim BD（地址、长度、stride 等）
npu.push_queue 把 BD 推入执行队列（开始搬运）
npu.sync       等待所有 DMA 完成
```

一个典型的 runtime sequence：

```mlir
aiex.runtime_sequence(%input: memref<...>, %weight: memref<...>, %output: memref<...>) {
  // 配置 Shim BD0: 从 input BO 读 hidden vector
  aiex.npu.writebd {bd_id = 0, buffer_length = 2048, ...}
  aiex.npu.address_patch {bd_id = 0, arg_idx = 0}  // 绑定到第一个 BO
  aiex.npu.push_queue {channel = 0, bd_id = 0, repeat_count = 0}

  // 配置 weight BD: 从 weight BO 读权重
  aiex.npu.writebd {bd_id = 2, buffer_length = 1280, ...}
  aiex.npu.address_patch {bd_id = 2, arg_idx = 1}  // 绑定到第二个 BO
  aiex.npu.push_queue {channel = 0, bd_id = 2, repeat_count = 607}  // 608 patches

  // 等权重送完
  aiex.npu.sync {channel = 0, direction = 0}

  // 配置输出 BD: 结果写回 output BO
  aiex.npu.writebd {bd_id = 4, buffer_length = 257, ...}
  aiex.npu.address_patch {bd_id = 4, arg_idx = 2}  // 绑定到第三个 BO
  aiex.npu.push_queue {channel = 0, bd_id = 4, repeat_count = 0}

  // 等输出完成
  aiex.npu.sync {channel = 0, direction = 1}
}
```

### XRT API 流程

从 Python 来看，完整的 NPU 使用流程：

```python
import npu_build

# 1. 编译（通常只做一次）
npu_build.compile_mlir(mlir_path, xclbin_path, insts_path)

# 2. 加载 xclbin
handle = npu_build.load_kernel(xclbin_path, insts_path)

# 3. 准备 BO
input_bo = np.zeros(..., dtype=np.int32)
weight_bo = np.zeros(..., dtype=np.int32)
output_bo = np.zeros(..., dtype=np.int32)

# 填入数据
input_bo[:] = hidden_vector
weight_bo[:] = q4nx_weights

# 4. 提交执行
npu_build.run(handle, [input_bo, weight_bo, output_bo])

# 5. 读回结果
result = output_bo.copy()

# 6. 清理
npu_build.cleanup()
```

整个过程中，host 程序不做层内调度。它只负责：填数据 → 提交 → 等结果。NPU 内部的 DMA 轮转、lock 同步、tile 间数据流全部自动完成。

### 5 个 BO 参数上限

XRT 当前限制：一次 kernel 执行最多传入 **5 个 BO 参数**。

这意味着如果你的算子需要 input、weight、kv_cache_k、kv_cache_v、output = 5 个 BO，你已经把限额用完了。如果还需要更多，必须：

- **打包**：把多个逻辑 buffer 拼接到同一个 BO 的不同偏移
- **偏移 patch**：runtime sequence 中用 `address_patch` + 偏移来指向 BO 内的子区域

例如：把 K cache 和 V cache 打包到同一个 `kv_bo` 里，K 从 offset 0 开始，V 从 offset N 开始。runtime sequence 配置 BD 时分别指向不同偏移。

### RTP 和 Runtime-Start Lock

有些动态参数需要在每次运行时告诉 NPU（比如当前 token 位置），但又不值得重新编译整个 xclbin。这通过 **RTP（Runtime Parameter）** 实现：

```mlir
// Runtime sequence 中写 RTP
aiex.npu.rtp_write(%tile, %rtp_index, %value)

// 写完 RTP 后释放 runtime-start lock，允许 core 开始执行
aiex.set_lock(%runtime_start_lock, 1)
```

**关键时序**：RTP 必须在 core 读取之前写好。如果 core 没有 runtime-start lock 门控，它可能在 host 写 RTP 之前就启动，读到旧值。这是一个实际踩过的坑——表现为"current token 永远是 0"。

---

## 第十一章：编译流水线——从 Python 到 xclbin

### 为什么用 Python 生成 MLIR

XDNA 的硬件描述文件（MLIR-AIE）非常冗长：一个 48-tile 设计的 MLIR 可能有几千行。其中大量是重复结构：

- 16 个 Main tile 的 DMA 配置几乎相同，只是坐标和偏移不同
- 4 列 Memtile 的权重分发逻辑完全对称
- Lock 编号和 BD 编号需要精确分配，手工容易出错

Python generator 的价值：

```python
# 用循环生成 16 个 Main tile 的 BD
for col in (2, 3, 4, 5):
    for row in (2, 3, 4, 5):
        emit_main_tile_dma(col, row, ...)

# 用参数化函数生成 4 列 Memtile 的权重 stream
for group in range(4):
    emit_weight_stream(group, ...)
```

手写 MLIR 容易复制粘贴出错，参数化生成则能确保一致性。

### MLIR-AIE 的核心概念

MLIR-AIE 是描述 NPU 完整配置的中间表示。第七章的最小例子已经展示了它的基本语法（tile、buffer、lock、dma_bd、flow、core）。在大型设计中，还需要两个额外能力：

```mlir
// link_with：链接外部编译好的 C++ kernel
aie.core(%tile) {
} {link_with = "main_projection_q4nx.o"}

// runtime_sequence：描述 host 侧的 BD 配置和提交
aiex.runtime_sequence(%input: memref<...>, %weight: memref<...>, %output: memref<...>) {
  aiex.npu.writebd { ... }
  aiex.npu.push_queue { ... }
  aiex.npu.sync { ... }
}
```

核心 kernel 用 C++ 写、编译成 .o，在 MLIR 里用 `link_with` 引用。runtime sequence 则对应第十章介绍的 host 侧提交逻辑。

### 编译步骤

完整的编译管线：

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Python      │     │  MLIR-AIE    │     │  xclbin      │
│  generator   │────►│  (.mlir)     │────►│  + insts     │
└──────────────┘     └──────────────┘     └──────────────┘
       │                     │                     │
    contract.py         design.mlir           design.xclbin
    dataflow.py                               design.bin
    generate.py
    emit_mlir.py
```

详细步骤：

**1. Python 生成 MLIR**
```bash
python qwen3-layer/emit_mlir.py  # 输出 build/design.mlir
```

**2. 编译 Kernel .o 文件**
```bash
clang++ -O2 --target=aie2p-none-unknown-elf \
  -c main_projection_q4nx.cc -o main_projection_q4nx.o
```

每个 kernel 源文件编译成一个 AIE ELF object。MLIR 中的 `link_with` 指向这些 .o 文件。

**3. aiecc 编译 MLIR → xclbin + instruction stream**
```bash
aiecc --aie-generate-xclbin --xclbin-name=design.xclbin \
      --aie-generate-npu-insts --npu-insts-name=design.bin \
      design.mlir
```

`aiecc` 做的事情：
- 解析 MLIR，提取 tile 配置、路由表、lock 初值
- 链接 kernel ELF 到对应 tile
- 生成 stream switch 路由配置
- 生成 PDI/CDO（NPU 初始化序列）
- 打包成 xclbin
- 生成 instruction stream（runtime sequence 编译后的二进制）

### 关键产物

| 文件 | 内容 | 何时使用 |
|------|------|---------|
| `design.mlir` | 完整的硬件描述 | 调试时阅读、结构检查 |
| `design.xclbin` | NPU 配置二进制（kernel ELF + 路由 + 初始化） | 加载到 NPU |
| `design.bin` | instruction stream 二进制 | 提交到 NPU 执行 runtime sequence |
| `*.o` | 编译后的 kernel ELF | 被 aiecc 链接进 xclbin |

### 结构检查：编译前的防线

编译 xclbin 很慢（几十秒到几分钟），而且编译成功不代表运行正确。很多错误（BD bank 违规、lock 不平衡、packet ID 冲突）编译器不检查。

所以在编译之前，用 Python 做**结构检查**：

```python
# check_contract.py 的思路

# 检查 contract 常量一致性
errors = contract.validate_contract()

# 检查 dataflow 图的边和节点一致
errors += dataflow.validate_dataflow()

# 检查生成的 MLIR 中的物理约束
mlir_text = Path("build/design.mlir").read_text()
errors += physical_contract.validate_q4nx_down_full_layer_ownership("full-layer", mlir_text)
```

这些检查验证：
- Row1 channel 所有权是否正确（S2MM4/5 = weight，不是 compact）
- 每个 Main16 tile 是否有正确的 DMA 启动标记
- 是否链接了正确的 kernel .o 文件
- 是否使用了被禁止的旧路由
- Compact output 是否在 MM2S5（不能和 weight fanout 混）

这比"编译不报错就运行"安全得多。相当于在发射前检查接线图，而不是直接上电看看会不会爆炸。

### Instruction Patch：避免重编译

编译 xclbin 很慢，但每个 token 位置的 runtime 参数不同（block 数、tail-token 数、cache 偏移等）。

解法：**编译一次最大容量 xclbin，每次运行只 patch instruction stream**。

```python
# 从 token1007 容量的 design.bin patch 到 token91
patched_insts = patch_instruction_stream(
    base_insts=load("design-token1007.bin"),
    target_token=91,
    patches=[
        # (offset, new_value) 列表
        (rtp_offset, 91),            # current token RTP
        (block_count_offset, 6),     # rounded blocks
        (iteration_offset, 6),       # scan iteration size
        (repeat_offset, 5),          # queue repeat count
        ...
    ]
)
save("design-token1007-to-token91.bin", patched_insts)
```

已验证：patch 后的 instruction stream 逐 word 等于重新编译出来的结果。这意味着"编译一次 + 每 token patch"是可行的生产方案。

---

## 本部分小结

| 层次 | 你写什么 | 它变成什么 |
|------|---------|-----------|
| Compute Tile | C++ kernel（.cc → .o） | 在 tile 上循环执行的小程序 |
| Memtile | Python 生成 BD/lock/buffer 配置 | MLIR 中的 `aie.mem` 块，纯 DMA 自动运转 |
| Host | Python runner 填 BO + 提交 | XRT 调用，NPU 一次性跑完整个图 |
| 粘合层 | Python generator（contract + dataflow → MLIR） | design.mlir → design.xclbin + design.bin |

关键认知：**你不是在写一个程序，你是在描述一台机器**。编译后这台机器就固定了——运行时只是"开机 + 喂数据 + 等结果"。

---

# 第三部分：设计 Pattern

> 这些 pattern 不是某个模型的特殊做法，而是在 XDNA 上做任何算子都会反复遇到的通用规律。
> 每个 pattern 的结构：问题类型 → 通用解法 → 适用条件 → 失败信号 → 实例。

---

## 第十二章：Pattern 1——大算子拆成 Tile-Local 工作单元

### 硬件问题

Compute Tile 只有 ~64 KB 本地内存。一个 4096→4096 投影有 32 MB 权重。而且 tile 之间看不到彼此的内存——不能让一个 tile 看到完整问题。

### 通用拆法

每个 tile 固定负责：
- **固定输出切片**：N 行输出（不是整个输出维度）
- **固定输入窗口**：K 列的 activation chunk + 对应 weight chunk
- **固定 accumulator**：N 个元素，跨 chunk 保持
- **固定输出 ABI**：record = header + payload（下游根据 header 知道这块数据来自哪个 tile）

tile 不需要理解整个模型。它只知道：这次来了哪块输入、哪块权重、我负责哪几行、算完写成什么格式。

### 适用条件

- 输出可以按 tile 切分（tile 之间不需要共享 partial output）
- reduce 维度可以分块累加
- local memory 放得下当前 chunk + accumulator + output record
- 输入/权重流顺序能和 tile 内循环对齐

### 失败信号

- tile 里出现大量和全局阶段相关的 if/switch
- 一个 tile 要保存太多不同 phase 才用到的状态
- 输出切片需要频繁跨 tile 合并

### 实例

```
tile0(2,2): owns rows 0-3    tile1(3,2): owns rows 4-7
    ↑ weight chunks              ↑ weight chunks
    ↑ activation chunks          ↑ activation chunks (same data)
    ↓ 5-dword record             ↓ 5-dword record
    [header|4 payload]           [header|4 payload]
```

8×16 matvec 拆到 2 个 tile，每 tile 负责 4 行。真实 qwen3-layer 中：16 个 tile 各负责 32 行输出 × 256 列 chunk × Q4NX weight chunk (1280 dwords) → 17-dword record。

---

## 第十三章：Pattern 2——低 batch matvec 优先优化数据流

### 硬件问题

单 token decode 的 matvec 是 memory-bound：115 MB 权重从主存流入，只做一次向量乘。如果只盯 MAC 数量，会错过真正瓶颈——数据搬运。设计变量是"权重怎么流、activation 怎么复用"。

### 通用拆法

核心原则：
- **权重流进来一次就完成本地使用**——不缓存不回头，ping-pong 只为 DMA/compute overlap
- **activation 是否 replay 取决于来源**——来自 host/replay 便宜就 block-major，来自片上 packet 不便重放就 chunk-major
- **weight stream 顺序必须和 compute loop 对齐**——chunk-major pack，不是逻辑行主序

两种调度策略：

| | Block-major | Chunk-major multi-block |
|---|---|---|
| 做法 | 一个输出 block 看完整 K 个 input chunks | 一个 activation chunk 同时服务多个 output blocks |
| activation | 来自 host replay，便宜 | 来自片上 packet，不想重放 |
| 适合 | Q/K/V（c1r2 replay 12 次） | O/down（packet2/packet1 来自片上） |

### 适用条件

- reduce 是线性累加
- 权重流量远大于激活流量
- weight stream 顺序能和 compute loop 对齐（chunk-major pack）
- tile 有足够 accumulator 保存当前要服务的 output block

### 失败信号

- 反量化结果被物化成完整矩阵
- activation 从片上返回后又被重复搬很多遍
- weight stream 顺序和 compute loop 不一致，导致额外 buffer
- 为减少几次 MAC 却引入更大的搬运

---

## 第十四章：Pattern 3——多 tile 输出先变成稳定 record，再汇聚

### 硬件问题

多个 Compute Tile 并发产出小块结果。如果让每个 tile 直接写大 tensor 的任意位置：
- 16 路写入地址复杂，packet/BD 数量膨胀
- 下游难以判断数据边界
- debug 时分不清错在路由、布局还是数学

### 通用拆法

定义稳定 record ABI，on-chip 层级汇聚：

```
tile record (17 dw) = 1 header + 16 payload
  ↓ per-row S2MM channel 收入 memtile
column compact (65 dw) = row0 full record + row1..3 payload only (header stripped)
  ↓ 4 column 汇聚
global compact (257 dw) = 1 header + 256 payload = 512 bf16
```

header 去重：row0 保留完整 record（含 header），其他 row 只保留 payload（通过 BD offset 跳过 header）。

### 适用条件

- tile 输出粒度固定
- 下游消费顺序固定
- metadata（header）很小，payload 可以连续拼
- 汇聚路径比全局 scatter 更简单

### 失败信号

- header 越来越大，开始携带业务逻辑
- BD block 有多个 release → 编译报错（硬约束：一个 BD block 最多一个 release）
- record 顺序需要运行时条件判断修正
- compact 规则在多个文件各写一份

---

## 第十五章：Pattern 4——producer/consumer 用 ping-pong + credit lock 表达

### 硬件问题

XDNA 内部数据流不是 CPU 每步调度。producer 和 consumer 速度不同，没有信用机制就会：
- producer 覆盖 consumer 还没读的数据
- consumer 读到未写满的 buffer
- DMA channel 等不到锁而 timeout

### 通用拆法

双缓冲 + empty/full lock，表达为 BD ring：

```
producer BD ring:
  ^ping: acquire(empty,1) → write ping → release(full,1) → next=^pong
  ^pong: acquire(empty,1) → write pong → release(full,1) → next=^ping

consumer BD ring:
  ^ping: acquire(full,1) → read ping → release(empty,1) → next=^pong
  ^pong: acquire(full,1) → read pong → release(empty,1) → next=^ping
```

一对多 fanout 用 counting lock：
```
lock: empty (init = N)  ← N 个 consumer 都 release 才能覆盖
producer: acquire(empty, N) → write → release(full, N)
consumer_i: acquire(full, 1) → read → release(empty, 1)
```

### 失败信号

- **BD 只有 acquire 没有 release**（或反过来）→ 永久死锁
- **lock 初值错**：empty 应 = slot 数（ping-pong 时 = 2），full 应 = 0
- **BD bank 规则违反**：memtile 偶通道用 BD 0-23，奇通道用 BD 24-47
- **多 producer 共用 channel**：顺序不确定 → lock 计数混乱
- **counting lock 不平衡**：producer release N 但只有 N-1 个 consumer release back

---

## 第十六章：Pattern 5——scarce channel 先分所有权，再用 packet 复用

### 硬件问题

物理 DMA channel 很少（每个 tile 只有几个 MM2S/S2MM），但逻辑数据流很多。常见错误：
- 以为换一个 packet ID 就能解决 channel 冲突（不行——physical channel 必须有单一 owner）
- 两个无关 producer 共用一个 channel → lock/BD 生命周期冲突 → 死锁

### 通用拆法

分两层：

```
1. Channel ownership（硬约束）:
   规定每个物理 channel 属于哪一个数据流 owner
   这是不可违反的——混用 = 死锁

2. Packet ID（逻辑复用）:
   在已有 owner 的物理路径上，用 packet ID 区分多路逻辑数据
   receiver 根据 ID 筛选"这个包是给我的"
```

先固定 channel ownership，再分配 packet ID。两者不能混为一谈。

### 失败信号

- 以为换 packet ID 就能解决 channel 冲突（本质是 ownership 问题）
- 同一 channel 同时承担两个互不相关的 producer 顺序
- packet ID 没有集中登记，新增路径靠记忆避冲突
- packet ID 全局重复 → 跨列死锁

### 实例

```
Row1 Memtile channel 分工：
  S2MM 0-3: compact record 汇聚（从 Compute Tile）
  S2MM 4-5: 权重入口（从 Shim）
  MM2S 0-3: 权重分发（到 Compute Tile 各行）
  MM2S 5:   compact 输出（到 bridge/downstream）
```

旧版曾把权重走 S2MM0/1 → 和 compact 冲突 → 死锁。

---

## 第十七章：Pattern 6——layout 是算子设计的一部分

### 硬件问题

逻辑 tensor layout 往往是人类友好的格式（token-major、行主序、自然索引），但 DMA 需要的是：
- 连续地址（一个 1D BD 就能描述）
- 可表达的 stride（不超过硬件 field 范围）
- 少量 BD（shim 只有 16 个 slot）

如果沿用逻辑 layout，可能数学很清楚，硬件却很难搬——需要 2D stride、多个 BD、或复杂 gather。

### 通用拆法

从 DMA 消费单位反推物理 layout：

```
下游一次读多少？→ 连续放在一起
是否需要按 window 切？→ 按 window 排列
能否用一个 BD 描述？→ 不能就提前 pack
```

热路径数据 pack 成硬件友好布局。pack 成本只付一次（host 侧或编译时），后面换来更少 BD、更少 stride、更少 copy。

### 失败信号

- 为了保持逻辑 layout，用了很多小 BD
- stride 字段接近或超过硬件范围
- scan 需要复杂 gather
- host reference 和 NPU layout 各写一套，容易漂移

### 实例

- **KV cache block-major**：逻辑 `K[token][head][dim]` → 物理 `K[block][head][token_in_block][dim]`，scan 按 16-token block 读，block-major 让每个 block 连续
- **Q4NX 权重预 pack**：模型权重 `[output_dim, input_dim]` → NPU 需要 `[chunk0_32rows, chunk1_32rows, ...]` 按 tile 消费顺序排列

---

## 第十八章：Pattern 7——大中间矩阵改成 block carrier + online merge

### 硬件问题

某些算子有巨大中间矩阵（如 attention score matrix: tokens × heads × context），完整物化会压垮 tile local memory 或片外带宽。

### 通用拆法

按 block 流式处理，每个 block 提取**最小充分统计量（carrier）**：

```
每个 block:
  计算局部结果
  提取 carrier（远小于完整中间矩阵）
  carrier 必须包含跨 block 合并所需的全部信息

merge 端:
  保存 running state
  用 online 公式合并每个 carrier
  最后输出最终结果
```

关键约束：**carrier 必须是充分统计量**。对于 online softmax merge，carrier 不是简单的 max，而是 `(block_max, block_sum, weighted_V)` 三元组——少了任何一个就无法正确合并。

### 适用条件

- 跨 block 合并有稳定数学公式（online softmax、streaming max、running mean）
- carrier 能表达局部 block 的充分信息
- tail block 可以 mask
- running state 放得进 tile local memory

### 失败信号

- carrier 越做越大，接近原始中间矩阵 → pattern 失效
- merge 需要回看过去 block 的完整数据 → 不满足 online 条件
- tail token/padding 靠后处理修补
- fixed-point scale 没有和 reference 对齐

### 实例

Shape-A（block processor）每处理 16 个 token 产出 80-dword carrier：
```
carrier 结构:
  base[0x100]: 8 heads × 16 tokens 的 Q12 权重
  scalar[0x40]: 8 × (block_max, block_sum) int32 pair
```

Shape-B（merge processor）：接收 carrier + 对应 block 的 V → 本地做 weighted sum + online merge。

---

## 第十九章：Pattern 8——小追加和大扫描分成两个数据流

### 硬件问题

状态型算子同时有两种访问：
- **append/update**：每次只写很小的新状态（如当前 token 的 K/V），地址随 runtime 参数变化
- **scan/read**：每次顺序读大量历史状态（如整个 KV cache），适合大块 DMA

把两种访问混在一个路径里，descriptor 复杂且同步不清楚。更严重：**scan 可能读到旧值**（append 还没完成就开始 scan）。

### 通用拆法

在 **runtime descriptor 层面**拆成两个阶段，用 `npu.sync` 强制顺序：

```
Phase 1: append (shim S2MM, buffer_offset = f(position))
  → push_queue → npu.sync (确认写入可见)

Phase 2: scan (shim MM2S, buffer_offset = 0, length = 全量)
  → push_queue → npu.sync (等 scan 结果回来)
```

关键：这不是"在同一个 core 里先写后读"——是两个 runtime sequence 阶段，descriptor 层面分离，中间有硬件 sync 保证可见性。

### 失败信号

- scan 偶尔读到旧值（没有 sync 保证可见性）
- 给每个 block 一个 BD → 超 shim 16 BD 上限
- 只有 iteration_size 没有 repeat_count → 只扫一个 segment
- current slot 没有毒化测试，写没写进去无法区分

---

## 第二十章：Pattern 9——动态参数走 RTP/descriptor patch

### 硬件问题

数据流拓扑不变，但每次运行有少量参数变化（token position、block count、tail mask、buffer offset）。如果每次重编译 xclbin → 太慢（几十秒）。如果让 tile 内部维护大量 runtime state → 程序复杂且同步危险。

### 通用拆法

把动态性分层：

| 动态参数类型 | 处理方式 | 例子 |
|---|---|---|
| tile 需要读的标量 | RTP buffer + runtime-start lock | current token position |
| DMA 需要的地址/长度 | `npu.writebd` 的 buffer_offset/buffer_length | KV write offset |
| BD 的 iteration/repeat | `npu.writebd` 字段 | scan block count |
| instruction stream 常数 | binary patch design.bin | RTP value, iteration word |
| 拓扑/channel/BD 数量改变 | 重新编译 | 只有这种情况需要 |

关键时序：**RTP 必须在 core 读取之前写好**。用 runtime-start lock 门控：
```
runtime sequence: rtp_write → set_lock(runtime_start, 1)
core: acquire(runtime_start, 1) → 安全读 RTP
```

### 失败信号

- tile 在 RTP 写入前启动 → 读到旧值（表现为 current token = 0）
- patch 后 instruction stream 没有和直接编译结果逐 word 比对
- target token 需要的 block 数超过 base capacity
- 为了避免 patch，把大量条件判断塞进 core loop

---

## 第二十一章：Pattern 10——融合不是一个大 kernel，而是稳定 ABI 间片上交接

### 硬件问题

多个算子连续执行，中间激活很大。如果每个算子都写回主存再读回，带宽浪费严重。但把所有数学写进一个巨大 kernel 会让 tile 变成全能 station，程序复杂、debug 困难。

### 通用拆法

按**稳定 ABI** 融合——融合边界是数据格式契约，不是函数调用边界：

```
算子 A 输出固定格式 record [header | payload]
  header 编码: operator_id + payload_count
  payload 格式: 固定大小、固定 layout

算子 B 直接从片上接收 record（通过 aie.flow）
  解析 header 验证来源
  按 ABI 约定处理 payload
  输出自己的 record

中间 record 不回主存，host 不分配 intermediate BO
```

ABI 包含：record 大小、header 编码、payload layout、消费次数、lock/BD contract。

### 失败信号

- 一个 tile 变成全能 station，保存太多阶段状态
- ABI 含混（header 格式随 call site 变化）
- 不同 layout 的算子被硬捏在一起，产生大量重排
- host 需要频繁进入层内 phase 调度

### 实例

完整融合层的 ABI 链：
```
c1r2 full-vector replay (packet0, 2049 dw, header 0xC1000000|idx)
  → main16 Q/K/V compact (17-dw record → 65-dw column → 257-dw global)
  → c1r3 postprocess → attention (packet2, 2048 dw)
  → main16 O compact → c1r2 replay
  → main16 up/gate compact → c6r2 SwiGLU (payload half, 256 dw × 24)
  → c6r1 gather (packet1, 6144 dw) → main16 down compact
```

每个箭头是一个稳定 ABI 边界：大小固定、header 格式固定、lock contract 固定。

---

## 第二十二章：Pattern 11——用集成边界验证数据流

### 硬件问题

XDNA 错误跨越多个层面：Python generator → MLIR-AIE → BD/lock/packet route → C++ kernel → host BO layout。单独测试一个 helper 函数或一个 Python 模块，很难发现真实硬件路径上的 timeout、错包、错 layout。

### 通用拆法

验证稳定的集成边界——选择一个完整的物理路径闭环：

```
Level 1: 结构检查（编译前，秒级）
  验证 BD/channel/packet/lock ownership 是否正确
  不等编译就能发现接线错误

Level 2: 编译检查（build-only，分钟级）
  验证 MLIR 能通过 aiecc
  发现 BD ID 超范围、lock 引用错误等

Level 3: 真机运行 + 分级验证（需要 NPU）
  a) 中间状态毒化: 关键 buffer 预填 poison 值
  b) 数学正确性: 和 CPU reference（同物理 layout）逐元素对比
  c) 定位: poison 残留 = 路径断了; 值不对 = 计算错了
```

毒化不只用于最终 output——关键中间状态（如 current K/V slot、compact buffer、scan input）更应该毒化。

### 失败信号

- 大量测试只验证 Python helper，没有验证生成的 MLIR
- reference 使用逻辑 layout，NPU 使用物理 layout（对比永远 mismatch）
- 只看最终输出，不读回关键中间状态（定位困难）
- 旧实验 case 保留太多，约束没有沉淀到共享检查

---

# 第四部分：调试——NPU 出错时怎么定位

---

## 第二十三章：NPU 出错时怎么定位

### 调试前先换一个心智模型

在 CPU 程序里，bug 往往表现为异常、崩溃、错误返回值，调试方式是打断点、看调用栈、打印变量。

XDNA NPU 上不是这样。大多数错误只有两种外在表现：

1. **timeout**：某个 DMA 或 core 一直等不到 lock、stream、packet 或 host sync 完成。
2. **数据不对**：程序跑完了，但 output、cache、record、header 或中间 buffer 和 reference 不一致。

调试的目标不是找"哪条语句错了"，而是找：

```
哪一个 producer 没有 release？
哪一个 consumer acquire 了不存在的信用？
哪条 stream 没有数据？
哪个 packet 被送到了错误的接收者？
哪个 BD 搬了错误的地址、长度或次数？
```

换句话说：**定位第一个断掉的交接边界**。

### 先判断是哪一类失败

| 现象 | 优先怀疑 |
|------|----------|
| 编译不过 | MLIR 语法、tile/buffer/lock 名字、kernel link、资源分配 |
| routing 失败 | stream route 不可达、flow 太多、物理拓扑冲突 |
| run timeout | lock 配对、BD repeat、packet route、channel 所有权、producer 没启动 |
| output 全是 poison | 输出路径没有写到，或最后一个 DMA 没跑 |
| output 部分 poison | record 数量不够、tail mask 错、repeat count 错 |
| header 正确 payload 错 | 计算 kernel 或 payload layout 错 |
| header 错 payload 像随机数 | record ABI、packet ID、compact layout 错 |
| current K/V cache 没更新 | append path、sync 顺序、writeback BD 地址错 |
| token0 对，token17 错 | KV scan block/tail、RTP、descriptor patch、online merge 错 |

### timeout：从"谁在等谁"开始

timeout 本质上是某个等待条件永远不满足。排查时先写出这一条边的生产消费契约：

```
producer:
  tile/channel:
  BD:
  buffer length:
  packet ID:
  release lock:
  release count:
  repeat count:

consumer:
  tile/channel:
  BD:
  expected length:
  packet ID or circuit flow:
  acquire lock:
  acquire count:
  repeat count:
```

然后逐项核对：

1. **release count 和 acquire count 是否相等**
2. **BD ring 是否真的会重复**（第二轮、第三轮才出错很常见）
3. **channel 所有权有没有冲突**
4. **BD bank 是否符合硬件规则**（偶数 channel 用 BD 0-23，奇数用 24-47）
5. **packet ID 是否唯一，route 是否匹配**

### 数据错误：先看数据有没有到，再看值对不对

核心方法是 **poison + reference + 中间状态读回**。

**1. Poison：证明路径是否真的写过**

把输出和关键中间 buffer 预填成明显的毒值（如 `0xDEADBEEF`）。运行后检查：

| 结果 | 含义 |
|------|------|
| 全部还是 poison | 这条路径完全没写到 |
| 前半段变了，后半段 poison | repeat count、tail、BD length 或 record 数量不够 |
| poison 出现在某些固定 stride 位置 | layout/stride 错 |
| poison 被 attention 读到 | append-before-scan 或 sync 顺序错 |

**2. Reference 必须模拟物理 layout**

NPU 上的数据不是逻辑 tensor layout。正确的 reference 应该复刻：header 编码、record 顺序、bf16 round、Q4NX pack/dequant 顺序、KV cache block/tail layout。

**3. Header 先于 payload 检查**

record 类输出要先看 header：
- header 正确 payload 错：重点查 kernel 数值
- header 错 payload 也错：先查 record ABI、compact 顺序、packet route

### RTP 和 descriptor patch 的调试

最容易出两类 bug：

1. **core 早于 RTP 启动**：表现为 token0 正常但其他 token 看起来像 token0
2. **patch 后没有和直接编译结果对比**：patch "能跑"但跑的是错误 token 的扫描范围

### 调试清单

遇到 timeout：

```
1. 确认是编译、加载、运行哪个阶段 timeout
2. 找最后一个应该产生 output 的边界
3. 写出 producer/consumer 契约
4. 核对 lock acquire/release count
5. 核对 BD length、repeat、next
6. 核对 channel ownership 和 BD bank
7. 核对 packet ID 和 flow route
8. 检查 source-side replay 次数
```

遇到数据不对：

```
1. output 和关键中间 buffer 预填 poison
2. 先看 poison 是否消失
3. 先查 header，再查 payload
4. CPU reference 使用物理 layout
5. token0、token17、token127 分层运行
6. 非有限值先定位第一处来源
7. 容差只用于解释 bf16/Q4NX 舍入，不用于掩盖系统性错误
```

遇到动态 token 错：

```
1. RTP 是否在 core 启动前写入
2. runtime-start lock 是否门控所有相关 core
3. descriptor patch 是否覆盖 length/offset/repeat/iteration
4. patched design.bin 是否和直接编译 token 的关键字段一致
5. KV append 和 scan 之间是否有 sync
```

### 最重要的原则

NPU 调试不是"试一个 patch 看看会不会过"。每次失败都要能回答：

```
失败发生在哪个物理边界？
这个边界的 producer/consumer 契约是什么？
哪个字段违反了契约？
为什么这个修复能防止同类问题再次发生？
```

如果回答不了，就说明还没有找到根因。

---

# 附录

## 附录 A：术语表

| 术语 | 含义 |
|------|------|
| Tile | NPU 阵列中的一个处理单元格 |
| Shim | Row 0，主存入口/出口 |
| Memtile | Row 1，片上大缓冲 + DMA 编排 |
| Compute Tile | Row 2+，跑 C++ kernel 的计算单元 |
| DMA | 直接内存访问，自动搬运数据 |
| BD | Buffer Descriptor，DMA 的任务单 |
| Lock | 生产者/消费者同步的计数器 |
| Stream | Tile 间的固定物理连线 |
| Packet | 数据包路由标签，复用物理线 |
| S2MM | Stream to Memory-Mapped（接收方向） |
| MM2S | Memory-Mapped to Stream（发送方向） |
| BO | Buffer Object，host/NPU 共享内存 |
| RTP | Runtime Parameter，运行时可写参数 |
| xclbin | 编译后的 NPU 配置二进制 |
| instruction stream | runtime sequence 编译后的指令流 |
| ping-pong | 双缓冲交替使用 |
| credit lock / counting lock | 初值>1 的 lock，表达多消费者信用 |
| Q4NX | 4-bit 量化权重格式 |
| bf16 | bfloat16，1+8+7 浮点格式 |
| dword | 32-bit word |

## 附录 B：硬件限制速查

| 资源 | 限制 |
|------|------|
| Shim BD | 每个 Shim tile 最多 16 个 |
| Memtile BD bank | 偶通道 BD 0-23，奇通道 BD 24-47 |
| Compute Tile 本地内存 | ~64 KB（型号相关） |
| Memtile 内存 | ~512 KB（型号相关） |
| DMA channel | 每个 tile 若干 S2MM + MM2S |
| Lock 数量 | 每个 tile 有限（~16） |
| Packet ID | 全局唯一，跨列不可重复 |
| XRT buffer 参数 | 单次 kernel 最多 5 个 BO |
| BD block release | 一个 BD block 最多一个 release |

## 附录 C：Pattern 速查表

| # | 问题类型 | 通用解法 |
|---|----------|----------|
| 1 | 算子太大, 单 tile 放不下 | tile-local 工作单元 |
| 2 | 低 batch matvec memory-bound | stream weight, reuse activation |
| 3 | 多 tile 小输出汇聚 | record → column compact → global compact |
| 4 | producer/consumer 速率不同 | ping-pong + credit lock |
| 5 | channel 稀缺且流量多 | channel ownership + packet ID |
| 6 | 逻辑 layout 不适合 DMA | hardware-first 物理 layout |
| 7 | 中间矩阵太大 | block carrier + online merge |
| 8 | 小追加 + 大扫描 | append 和 scan 分阶段 |
| 9 | runtime 参数变化 | RTP / descriptor patch |
| 10 | 中间激活回主存太贵 | 稳定 ABI 间片上交接 |
| 11 | 硬件路径难定位 | 集成边界验证 |

## 附录 D：推荐阅读路径

**入门路线**（第一周）：
1. 通读第一部分（第 1-7 章），建立硬件心智模型
2. 跑通第七章的最小例子

**动手路线**（第二周）：
3. 读第二部分（第 8-11 章），理解各层编程方式
4. 自己写一个 2-tile 的 ping-pong 传输

**设计路线**（第三周）：
5. 读第三部分 Pattern 1-6（第 12-17 章），理解拆分和同步设计
6. 动手做量化 Matvec 实战

**融合路线**（第四周）：
7. 读 Pattern 7-11（第 18-22 章），理解融合和验证
8. 做融合进阶
9. 遇到 bug 时翻第二十三章

## 附录 E：可运行示例

每个 Pattern 都有对应的可运行示例：

```bash
.venv/bin/python xdna-for-dummies/patterns/p01-tile-local-work-units/run_npu.py
.venv/bin/python xdna-for-dummies/patterns/p02-dataflow-first-matvec/run_npu.py
.venv/bin/python xdna-for-dummies/patterns/p03-record-column-compact/run_npu.py
.venv/bin/python xdna-for-dummies/patterns/p04-ping-pong-credit-lock/run_npu.py
.venv/bin/python xdna-for-dummies/patterns/p05-channel-ownership-packet/run_npu.py
.venv/bin/python xdna-for-dummies/patterns/p06-layout-is-design/run_npu.py
.venv/bin/python xdna-for-dummies/patterns/p07-block-carrier-online-merge/run_npu.py
.venv/bin/python xdna-for-dummies/patterns/p08-append-scan-separation/run_npu.py
.venv/bin/python xdna-for-dummies/patterns/p09-rtp-descriptor-patch/run_npu.py
.venv/bin/python xdna-for-dummies/patterns/p10-fusion-abi-handoff/run_npu.py
.venv/bin/python xdna-for-dummies/patterns/p11-integration-boundary-test/run_npu.py
```
