# 如何在 IRON 框架上编写 Megakernel

本文记录了我们在 AMD XDNA NPU 上使用 IRON 框架构建 Qwen3-0.6B 28层 decode megakernel 的完整经验。文章以「提出问题 → 分析约束 → 给出方案」的结构展开。目标读者是已经理解 transformer 推理但从未接触过 NPU 编程的工程师。

---

## 背景：IRON 是什么？

IRON 是一个开源的 Python 框架，用于编程 AMD Ryzen AI 处理器中的 NPU（Neural Processing Unit）。NPU 的硬件架构代号是 XDNA，核心是多个 AI Engine (AIE) 处理单元组成的 tile 阵列。

### 目标硬件：AMD XDNA NPU

```
┌────────────────────────────────────────────────────────────┐
│                    Host CPU (x86)                           │
│                    DDR Memory (L3)                          │
└────────────────────┬───────────────────────────────────────┘
                     │ PCIe / shared memory
┌────────────────────▼───────────────────────────────────────┐
│  XDNA NPU                                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Shim Tiles (DMA controllers)                        │  │
│  │  负责 L3 ↔ NPU 之间的数据搬运                          │  │
│  └──────────────────────┬───────────────────────────────┘  │
│  ┌──────────────────────▼───────────────────────────────┐  │
│  │  Memory Tiles (L2, 大容量 SRAM)                       │  │
│  │  负责数据中继：广播(1→N)、聚合(N→1)、缓冲              │  │
│  └──────────────────────┬───────────────────────────────┘  │
│  ┌──────────────────────▼───────────────────────────────┐  │
│  │  Compute Tiles (L1, 每个 64KB SRAM + AIE 处理核)      │  │
│  │  NPU2: 32 个 compute tiles                           │  │
│  │  实际执行向量/标量计算的地方                             │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────┘
```

**关键特征**：
- 分布式存储架构——tile 之间不共享 L1，数据搬运全靠 DMA
- 没有全局缓存一致性——程序员必须显式管理数据流动
- DMA 引擎是自主执行的——一旦配置好，不需要 CPU 介入

### 与 GPU/CPU 编程的根本差异

| | CPU/GPU | XDNA NPU |
|--|---------|----------|
| 内存模型 | 共享内存 + 缓存一致性 | 分布式本地存储，无一致性 |
| 数据搬运 | 硬件自动（cache miss → fetch） | 程序员显式编排 DMA |
| 并行模型 | 线程/warp 共享全局内存 | 独立 tile，通过 FIFO 通信 |
| 控制流 | 运行时分支、动态调度 | 静态编译时确定的执行序列 |
| 可重编程 | 热加载 kernel | 需要重新 dispatch 新 artifact |

最关键的心智转换：**在 GPU 上你思考「并行执行什么」，在 NPU 上你思考「数据如何流动」**。计算只是数据流经 tile 时的副作用。

### IRON 的编程模型

IRON 提供 Python DSL，编译流程为：

```
Python 代码 (design.py)
    ↓ IRON Python API
MLIR-AIE 中间表示 (.mlir)
    ↓ aiecc 编译器
每个 tile 的机器码 (.elf) + DMA 配置 (.insts)
    ↓ 打包
Xclbin (可加载的 NPU 二进制镜像)
    ↓ XRT 运行时
NPU 硬件执行
```

IRON 不是一个调度框架（不做自动算子融合或图优化）——它是一个**close-to-metal 的编程接口**，让你直接控制每个 tile 做什么、数据怎么流动。

### IRON 的核心抽象与硬件的映射关系

| IRON 概念 | 硬件对应物 | 作用 |
|-----------|-----------|------|
| `Worker` | Compute Tile 上的执行线程 | 定义一个 tile 要执行的计算序列 |
| `Kernel` | 编译后的 C++ 函数 (.o) | 在 tile 上实际执行的向量/标量代码 |
| `ObjectFifo` | DMA 通道 + L1/L2 Buffer slots | 数据流控制：生产者/消费者模式 |
| `Buffer` | Tile-local L1 SRAM 变量 | 跨迭代持久化状态（DMA 不参与） |
| `TensorAccessPattern` | DMA Buffer Descriptor (BD) | 告诉 DMA 怎么读写内存 |
| `Runtime` | NPU 指令序列 (runtime.bin) | Host 侧的 DMA 编排 |
| `Program` | 整个静态 NPU 部署图 | 最终编译产物 |
| `SequentialPlacer` | 物理 tile 分配器 | 决定 Worker/FIFO 放在哪些物理 tile 上 |

后文会逐一展开这些概念在 megakernel 中的具体使用方式。

---

## 核心矛盾：模型太大，芯片太小

一个 transformer decode step 的逻辑很简单：

```
for layer in 0..27:
    x = rmsnorm(x) @ Wq, Wk, Wv
    x = attention(x, kv_cache)
    x = x + residual
    x = rmsnorm(x) @ Wgate, Wup → silu → @ Wdown
    x = x + residual
```

但 XDNA NPU 的单个 compute tile 只有 **64KB** 本地存储（L1）。Qwen3-0.6B 的 Q 投影权重矩阵就有 1024×2048 = 4MB (bf16)。一个 tile 连一个权重矩阵都放不下，更不用说 28 层。

**朴素方案（每个算子一个 tile，每层一张图，复制28份）**的实验结果：单层就消耗 25 个 compute core。NPU2 总共只有 32 个 tile。两层都放不下。这个方向彻底行不通。

那么问题来了：**如何用 8 个 tile 跑完 28 层 transformer？**

---

## IRON 核心概念一：Worker 与 Kernel

在 IRON 中，`Worker` 代表一个 compute tile 上的执行线程。你用 Python 写 Worker 的 body 函数，定义它要做什么计算、按什么顺序获取/释放数据。编译后，这个 body 变成那个 tile 上实际执行的程序。

```python
# 一个最简单的 Worker：获取一个 packet，调 kernel 处理，释放
def my_worker_body(input_fifo, output_fifo, my_kernel):
    data = input_fifo.acquire(1)        # 等 DMA 送来数据
    result = output_fifo.acquire(1)     # 获取输出 buffer slot
    my_kernel(data, result, 1024)       # 调用 C++ kernel
    input_fifo.release(1)               # 归还输入 slot（DMA 可以填新数据）
    output_fifo.release(1)             # 归还输出 slot（DMA 可以送走结果）

worker = Worker(my_worker_body, [input_fifo.cons(), output_fifo.prod(), kernel])
```

`Kernel` 是编译后的 C++ 函数。它在 tile 的 AIE 核上执行实际的向量/标量计算。IRON 框架不关心 kernel 内部做什么——它只负责把数据送到 kernel 面前、把结果送走。

```c
// C++ kernel 示例：接收 L1 buffer 指针，做计算
extern "C" void my_kernel(const bfloat16 *input, bfloat16 *output, int32_t size) {
    for (int i = 0; i < size; i++)
        output[i] = input[i] * 2.0f;
}
```

**Worker 和 Kernel 的关系**：Worker 是调度者（决定何时获取数据、调用哪个 kernel、何时释放），Kernel 是执行者（做实际的数学运算）。一个 Worker 可以在不同时刻调用不同的 Kernel——这正是 megakernel 实现 "phase 切换" 的基础。

---

## 解决方案总览：Phase-Owned 状态机

答案是**时间复用**。不做空间展开（每个算子占一个 tile），而是让同一组 tile 循环处理所有层和所有算子：

```
8 个固定 Worker，每个循环执行 28 层 × 13 个 phase/层
DMA 按顺序流式地推送不同的数据（权重、激活值）
Worker 的「角色」随到达的数据而改变
```

NPU 计算图是静态的——它不随层数增长。增加层数只是增加循环次数和数据流长度。资源消耗被 `num_lanes`（8）约束，而不是被 `num_layers × num_phases`（364）约束。

这就是我们所说的「megakernel」：**一次 dispatch，一个静态图，跑完整个模型的 decode body**。

---

## IRON 核心概念二：ObjectFifo —— 硬件流控通道

`ObjectFifo` 是 IRON 中最核心的数据流抽象。它**不是软件队列**——编译后直接对应 NPU 的 DMA 通道 + L1 buffer slot。

### 硬件层面发生了什么

```
L3 (DDR)                L1 (Tile 本地 SRAM)
┌──────────┐           ┌─────────┐
│ 预排好的  │  DMA BD   │ slot 0  │ ← Worker 正在处理这个 slot
│ 连续数据  │ ────────→ │ slot 1  │ ← DMA 正在填充这个 slot（如果 depth=2）
│          │           └─────────┘
└──────────┘
```

- **`depth`**：L1 中预分配的 buffer slot 数量。depth=1 表示 DMA 和计算串行交替；depth=2 表示 DMA 可以在 Worker 计算时预填充下一个 slot（双缓冲）。
- **`acquire(1)`**：Worker 声明「我需要一个有数据的 slot」。如果没有 → 阻塞等待 DMA 填好。
- **`release(1)`**：Worker 声明「我用完了这个 slot」。DMA 引擎收到信号后可以覆写该 slot。

### ObjectFifo 的两端

```python
fifo = ObjectFifo(my_type, name="my_fifo", depth=2)

# 生产者端：DMA 从 L3 往里填
rt.fill(fifo.prod(), host_buffer, tap)     # Runtime 配置

# 消费者端：Worker 从里面取
data = fifo.cons().acquire(1)              # Worker 代码
```

ObjectFifo 还支持拓扑变换：
- `.cons().forward(depth=2)`：在 L2 创建广播中继节点（一进多出）
- `.prod().join(offsets=[...], ...)`：在 L2 创建聚合节点（多进一出）

### 流控的本质

acquire/release 构成了 **backpressure（背压）机制**：
- 如果 Worker 处理慢，所有 slot 都满 → DMA 自动暂停（不会覆盖未处理的数据）
- 如果 DMA 送得慢，所有 slot 都空 → Worker 自动等待（不会读到垃圾数据）

这就是为什么不需要任何显式同步原语——**FIFO 的 acquire/release 语义就是同步**。

---

## 问题一：切 Phase 是怎么实现的？没有动态分发吗？

没有 `if/switch/dispatch_table`。Phase 切换是纯粹的**结构性**实现。

### Worker 代码的真实结构

```python
# phase_owned_stages.py 中的 Worker body
def lane_worker_body(shared_fifo, packet_fifo, lane_output_fifo,
                     hidden_state, state, ...):
    init_kernel(state)                              # 初始化 checksum
    for layer in range_(num_layers):                # 循环 28 层
        shared = shared_fifo.acquire(1)             # 等待广播数据到达
        packet = packet_fifo.acquire(1)             # 等待本 lane 的第一个 packet
        lane_output = lane_output_fifo.acquire(1)   # 获取输出 buffer slot

        # Phase 0: RMSNorm + Q 投影
        q_shard_kernel(shared, packet, hidden_state, state, lane_output,
                       packet_elements, hidden_size, q_rows_per_packet,
                       q_output_values_per_lane, layer)
        packet_fifo.release(1)

        # Phase 1: K 投影（复用 shared 数据）
        packet = packet_fifo.acquire(1)
        projection_kernel(shared, packet, hidden_state, state, lane_output,
                         packet_elements, hidden_size, q_rows_per_packet,
                         q_output_values_per_lane, k_output_values_per_lane)
        packet_fifo.release(1)

        # Phase 2: V 投影（shared 数据用完，释放）
        packet = packet_fifo.acquire(1)
        projection_kernel(shared, packet, hidden_state, state, lane_output, ...)
        shared_fifo.release(1)           # shared 数据本层用完
        packet_fifo.release(1)

        # Phase 3: Q RoPE
        packet = packet_fifo.acquire(1)
        norm_rope_kernel(packet, state, lane_output, ...)
        packet_fifo.release(1)

        # Phase 4: K RoPE
        packet = packet_fifo.acquire(1)
        norm_rope_kernel(packet, state, lane_output, ...)
        packet_fifo.release(1)

        # Phase 5-8: Attention（4个 chunk，online softmax）
        attention_init_kernel(attention_state, attention_acc, head_dim)
        for _ in range_(4):              # 4 个 KV cache chunk
            packet = packet_fifo.acquire(1)
            attention_update_kernel(packet, attention_state, attention_acc,
                                   attention_chunk_size, head_dim)
            packet_fifo.release(1)
        attention_finalize_kernel(attention_state, attention_acc, state,
                                lane_output, context_output_base, head_dim)

        # Phase 9: O 投影 + residual
        packet = packet_fifo.acquire(1)
        o_kernel(packet, state, lane_output, ...)
        packet_fifo.release(1)

        # Phase 10: gate/up 投影
        packet = packet_fifo.acquire(1)
        gate_up_kernel(packet, state, lane_output, ...)
        packet_fifo.release(1)

        # Phase 11: down 投影 + residual
        packet = packet_fifo.acquire(1)
        down_kernel(packet, state, lane_output, ...)
        packet_fifo.release(1)
        lane_output_fifo.release(1)      # 本层所有输出写完，释放

        # Phase 12: 更新 tile-local hidden state
        packet = packet_fifo.acquire(1)
        next_hidden_kernel(packet, hidden_state, state, ...)
        packet_fifo.release(1)
```

### Phase 切换的底层原理

1. **每个 `release` 之后，程序计数器自然走到下一行代码**——下一行调用了不同的 kernel
2. **下一个 `acquire` 阻塞直到 DMA 送来下一个 packet**——这个 packet 包含不同的数据（比如从 Q 权重变成 K 权重）
3. **DMA 的推送顺序由 TAP 决定**——host 在 L3 缓冲区里按 phase 顺序预排好了所有数据
4. **没有任何运行时判断**——Worker 代码的 kernel 调用顺序和 DMA 数据推送顺序是编译时对齐的

```
Worker代码:    q_shard → k_proj → v_proj → q_rope → k_rope → attn×4 → o → gate_up → down → next
                 ↕         ↕        ↕        ↕        ↕        ↕       ↕     ↕        ↕      ↕
DMA送达顺序: packet0 → packet1 → packet2 → packet3 → packet4 → 5,6,7,8 → 9 → 10    → 11   → 12
```

这个设计的精妙之处：**DMA 引擎不知道什么是 "phase"，Worker 也不关心数据从哪来**。整个系统的正确性只依赖一个不变量：数据打包顺序 = kernel 调用顺序。

---

## 问题二：28 层的权重放在哪里？怎么送到 tile？

### 约束

- 一个 tile 的 L1 只有 64KB
- L2 (memory tile) 更大但仍远远不够存全模型
- Qwen3-0.6B 全模型权重约 1.2GB (bf16)

### 方案：L3 (主机 DDR) 预排缓冲 + DMA 流式推送

所有权重被预先打包进一个连续的 L3 缓冲区，每个 lane 有自己的独立流：

```
lane_packets[lane 0] = [
    layer0_phase0_q_weights,        # 4 行 Q 权重 × 1024 = 8KB
    layer0_phase1_k_weights,        # 4 行 K 权重 × 1024 = 8KB
    layer0_phase2_v_weights,        # 4 行 V 权重 × 1024 = 8KB
    layer0_phase3_q_rope_data,      # head + norm + cos + sin = 1KB
    layer0_phase4_k_rope_data,      # head + norm + cos + sin = 1KB
    layer0_phase5_attn_chunk0,      # q + k_cache_chunk + v_cache_chunk + mask = 25KB
    layer0_phase6_attn_chunk1,      # 同上
    layer0_phase7_attn_chunk2,      # 同上
    layer0_phase8_attn_chunk3,      # 同上
    layer0_phase9_o_weights,        # context + residual + 4 行 O × 2048 = 20KB
    layer0_phase10_gate_up_weights, # hidden + norm + 4 行 gate × 1024 + 4 行 up × 1024 = 20KB
    layer0_phase11_down_weights,    # ffn_hidden + residual + 4 行 down × 3072 = 30KB
    layer0_phase12_next_hidden,     # 下一层的 hidden state = 2KB
    layer1_phase0_q_weights,
    ...
    layer27_phase12_next_hidden
]
```

### DMA TAP 配置

一个 TAP (TensorAccessPattern) 描述了 DMA 怎么读 L3 缓冲区：

```python
# 每个 lane 的 TAP：从自己的偏移开始，连续读取全部 phase packets
packet_taps = [
    TensorAccessPattern(
        (input_elements,),                            # L3 缓冲区总大小
        lane * total_phase_packets * packet_elements, # 本 lane 的起始偏移
        [1, 1, 1, total_phase_packets * packet_elements],  # 传输总量
        [0, 0, 0, 1],                                 # 连续读取（stride=1）
    )
    for lane in range(num_lanes)
]
```

**关键数字：**

| 指标 | 数值 |
|------|------|
| 每 lane 流长度 | 28 层 × 13 phase × 15,368 bf16 = ~10.8MB |
| 单 packet 大小 | 15,368 bf16 = ~30KB |
| DMA 任务数 | 12（不随层数增长）|
| L1 缓冲模式 | 单缓冲 (depth=1)，DMA 和计算交替 |

DMA 任务数不随层数增长的原因：**一个 TAP 可以描述任意长度的连续传输**。DMA 引擎机械地沿缓冲区推进，每当 Worker `release` 一个 buffer slot，DMA 就送入下一个 packet。DMA 没有「层」或「phase」的概念——它只看到一段连续内存。

### 为什么用连续布局而不是分散布局？

如果每一层的每个 phase 权重分散存放（比如按 Qwen3 safetensors 原始布局），DMA 需要复杂的多维 stride 来跳跃读取，或者需要大量 BD（Buffer Descriptor）来描述每一段。XDNA 的 BD 数量有限（~16/memtile block），很快就会耗尽。

预排连续布局的好处：
1. 一个 TAP 搞定全部数据传输
2. DMA 可以 burst 读取（带宽效率最高）
3. 不占用额外 BD 资源

代价：host 需要一次性的预打包（把 safetensors 权重重排为 phase 顺序）。这个打包在推理开始前做一次，之后每次 decode 直接送预排数据。

---

## 问题三：如何跨 Lane 共享数据而不炸掉端点？

有些数据对所有 8 个 lane 都一样（hidden state 用于做 RMSNorm、layernorm weight）。如果从 L3 直接向 8 个 tile 广播同一份数据：

- L3 → tile0, L3 → tile1, ..., L3 → tile7 需要 8 个独立的 shim DMA 端点
- XDNA 的 shim DMA 端点数量有限，8 路扇出会导致 placer 失败

### 方案：Fabric Group + L2 中继广播

将 8 个 lane 分成 2 组（每组 4 个），每组用一个 L2 memory tile 做中继：

```
                    ┌── tile0
         ┌─ L2_g0 ─┼── tile1
L3 shim ─┤         ├── tile2
         │         └── tile3
         │         ┌── tile4
         └─ L2_g1 ─┼── tile5
                    ├── tile6
                    └── tile7
```

**L2 的作用**：一个 shim DMA 端点填充 L2，L2 内部的广播机制将数据复制到组内 4 个 tile。Shim 端点数从 8 降到 2。

### 代码实现

```python
# L3 → L2 的 FIFO（每组一个，depth=2 允许双缓冲）
shared_l3_fifos = [
    ObjectFifo(shared_packet_ty,
               name=f"shared_packets_l3l2_g{group}", depth=2)
    for group in range(fabric_group_count)  # 2 组
]

# L2 → L1 的广播（.cons().forward() 建立 L2 广播节点）
shared_fifos = [
    shared_l3_fifos[group].cons().forward(
        name=f"shared_packets_broadcast_g{group}", depth=2)
    for group in range(fabric_group_count)
]

# 每个 Worker 消费自己所在组的广播 FIFO
workers = [
    Worker(lane_worker_body, [
        shared_fifos[lane // fabric_group_size].cons(),  # 组内广播
        packet_fifos[lane].cons(),                        # 私有流
        lane_output_fifos[lane].prod(),                   # 输出
        ...
    ])
    for lane in range(num_lanes)
]
```

### 输出方向：Join 聚合

类似地，输出方向需要将 4 个 lane 的部分结果拼接成完整输出：

```python
# 4 个 lane 的输出 join 到一个 group buffer
q_join_fifos[group].prod().join(
    offsets=[lane_in_group * output_values_per_lane
             for lane_in_group in range(fabric_group_size)],  # [0, 200, 400, 600]
    obj_types=[lane_output_ty] * fabric_group_size,
    depths=[1] * fabric_group_size,
)
```

Join 是零拷贝的 DMA 聚合：每个 lane 写入 group buffer 的固定偏移位置，L2 组装完整后 drain 回 L3。

### 什么时候用 L2，什么时候跳过？

| 场景 | 路径 | 原因 |
|------|------|------|
| 共享数据广播 (fan-out ≥ 2) | L3 → L2 → L1 × N | 节省 shim 端点 |
| 输出聚合 (fan-in ≥ 2) | L1 × N → L2 → L3 | 组装后统一 drain |
| 私有 lane 数据 | L3 → L1 直连 | 无共享需求，跳过 L2 |

---

## IRON 核心概念三：Buffer —— Tile-Local 持久化存储

`Buffer` 是 IRON 中表示 tile 本地变量的抽象。和 `ObjectFifo` 的根本区别：

| | ObjectFifo | Buffer |
|--|-----------|--------|
| DMA 参与？ | 是，DMA 负责填充/排出 | 否，DMA 完全不知道它的存在 |
| 生命周期 | 临时：acquire 获取，release 归还 | 永久：从 Worker 启动到结束一直存在 |
| 用途 | 流式数据传输（权重、激活值） | 跨迭代持久化状态 |
| L1 占用 | 通过 depth 控制 slot 数 | 初始化后永久占用 |

```python
# Buffer 声明：在 tile 的 L1 中分配 2KB，初始化为零
hidden_state = Buffer(
    initial_value=np.zeros(shape=(1024,), dtype=bfloat16),
    name="lane_hidden_state",
)

# Worker 中直接使用（不需要 acquire/release）
def worker_body(fifo, hidden_state, my_kernel):
    for layer in range_(28):
        packet = fifo.acquire(1)
        my_kernel(packet, hidden_state, ...)  # kernel 直接读写 hidden_state
        fifo.release(1)
        # hidden_state 在下一次迭代中保持上次写入的值
```

Buffer 的硬件对应物就是 tile L1 SRAM 中的一段固定地址。它在编译时被分配物理地址，kernel 通过指针直接访问。

---

## 问题四：怎么在 28 层迭代之间传递激活状态？

每一层的输出 hidden state 是下一层的输入。如果每层结束都送回 L3 再送回来：

- 28 层 × 2KB × 2 方向 = 112KB 额外带宽
- 每次都要等 DMA 往返延迟
- 浪费 L3 带宽给同一个 tile 自己用的数据

### 方案：Tile-Local Buffer 跨迭代持久化

```python
# 声明 tile 本地 Buffer（不在任何 FIFO 中）
hidden_states = [
    Buffer(
        initial_value=np.zeros(shape=(hidden_size,), dtype=bfloat16),
        name=f"lane_{lane}_hidden_state",
    )
    for lane in range(num_lanes)
]
```

这个 Buffer 的生命周期：

```
Layer 0:  从 shared 广播流复制初始 hidden state → hidden_state[]
Layer 0:  各 phase 使用 hidden_state[] 做 RMSNorm
Layer 0:  next_hidden_kernel 从 packet 中读入 layer1 的 hidden → 覆盖 hidden_state[]
Layer 1:  各 phase 使用更新后的 hidden_state[]
...
Layer 27: 最后一层也是同样的模式
```

**关键设计点**：

1. `hidden_state` 是 tile 本地变量，不是 FIFO 对象——DMA 不参与它的读写
2. 代价：1024 bf16 = 2KB 永久占用 L1（从 64KB 预算中扣除）
3. 收益：完全消除了层间的 L3 往返

### 诊断状态：`state[0]` checksum

除了 hidden_state，每个 tile 还维护一个 float32 累加器：

```c
// 每个 kernel 都做：
float checksum = state[0];
// ... 计算过程中累加中间结果 ...
state[0] = checksum;
```

这个 checksum 纯粹是诊断用的——和 Python reference 逐比特对比，如果最终 checksum 一致，说明 308 个 phase 全部计算正确。它不参与数据流。

### Attention 状态：online softmax 跨 chunk 持久化

Attention 的 online softmax 需要跨 4 个 chunk 维护状态：

```python
# 每个 lane 两个额外的 tile-local Buffer
attention_states = [
    Buffer(initial_value=np.zeros(shape=(2,), dtype=np.float32), ...)
    # attention_state[0] = running max
    # attention_state[1] = running sum of exp(score - max)
]
attention_accs = [
    Buffer(initial_value=np.zeros(shape=(head_dim,), dtype=np.float32), ...)
    # attention_acc[0..127] = running weighted V accumulator
]
```

每处理一个 chunk，更新 max 和 sum，修正历史累积值。4 个 chunk 全部处理完后，`finalize` kernel 做 `acc / sum` 得到最终 attention context。这些状态全在 tile-local，不经过 DMA。

---

## 问题五：KV Cache 形状动态变化，但 ObjectFIFO 形状必须静态，怎么办？

### XDNA 的硬约束

```
ObjectFIFO 对象类型：编译时固定，不能运行时改
TensorAccessPattern：编译时固定，DMA BD 里写死了 offset/sizes/strides
DMA Buffer Descriptor：没有分支/条件跳转，机械执行序列
```

但 KV cache 随着 decode 进行在增长：position 0 时 cache 里只有 1 个 token，position 200 时有 201 个 token。

### 方案：固定 max-shape 读取 + 运行时 mask 填零

```
NPU 始终读取:     kv_cache[max_seq_len=256, head_dim=128]  ← 形状固定
运行时 mask 值:   mask[i] = 1.0 if i <= position else 0.0
Online softmax:   mask[i] = 0 的行被 skip（不参与 score 计算）
```

**Attention kernel 的关键逻辑**：

```c
// phase_owned_kernels.cc 中的 attention_update_packed_bf16
void new_mega_phase_attention_update_packed_bf16(
    const bfloat16 *packet,       // [q_head | k_chunk | v_chunk | mask_chunk]
    float *attention_state,        // [running_max, running_sum]
    float *attention_acc,          // [head_dim] weighted V accumulator
    int32_t chunk_size,            // 64
    int32_t head_dim)              // 128
{
    const bfloat16 *q = packet;
    const bfloat16 *k = q + head_dim;
    const bfloat16 *v = k + chunk_size * head_dim;
    const bfloat16 *mask = v + chunk_size * head_dim;

    // 第一遍：找本 chunk 的 max score（跳过 mask=0 的行）
    float chunk_max = -INFINITY;
    for (int row = 0; row < chunk_size; row++) {
        if (mask[row] <= 0.5f) continue;        // ← 运行时 skip 无效位置
        float score = dot(q, k[row]) * scale;
        chunk_max = max(chunk_max, score);
    }

    // 修正历史累积值
    float old_max = attention_state[0];
    float new_max = max(old_max, chunk_max);
    float correction = exp(old_max - new_max);  // 重新缩放旧值
    for (int dim = 0; dim < head_dim; dim++)
        attention_acc[dim] *= correction;

    // 第二遍：累积本 chunk 的 weighted V
    float chunk_sum = 0;
    for (int row = 0; row < chunk_size; row++) {
        if (mask[row] <= 0.5f) continue;
        float score = dot(q, k[row]) * scale;
        float weight = exp(score - new_max);
        chunk_sum += weight;
        for (int dim = 0; dim < head_dim; dim++)
            attention_acc[dim] += weight * v[row][dim];
    }

    attention_state[0] = new_max;
    attention_state[1] = old_sum * correction + chunk_sum;
}
```

### Host 在 dispatch 之间的责任

```python
# 每生成一个 token 后，host 做：
kv_cache[layer, head, position, :] = present_kv   # 一次 memcpy
position += 1
mask[position] = 1.0                                # 标记新位置有效
cos, sin = precompute_rope(position)                # 预计算旋转嵌入
# 重新打包 attention chunk packets（更新 k_cache 和 mask 数据）
```

### NPU artifact 完全与位置无关

同一个编译产物从 position 0 到 position 255 都能正确工作。位置信息只通过数据值（mask 内容、RoPE cos/sin 值、cache 内容）传入，不改变图拓扑或 DMA 配置。

### 代价与缓解

**代价**：在 position 1 时，NPU 仍然要读取 256 个 cache 位置的完整数据，其中 255 个是无效的。浪费了大量带宽。

**缓解**：
1. Attention kernel 中 `if (mask[row] <= 0.5f) continue;` 跳过无效行——不参与计算
2. 更根本的方案：编译多个 max_seq_len bucket（64, 128, 256），短上下文用小 bucket

### Chunk 策略的意义

将 max_seq_len=256 分为 4 个 chunk（每 chunk 64 行）：

```
Chunk 0: cache[0:64]    → 1 个 packet (q + k_chunk + v_chunk + mask = 25KB)
Chunk 1: cache[64:128]  → 1 个 packet
Chunk 2: cache[128:192] → 1 个 packet
Chunk 3: cache[192:256] → 1 个 packet
```

为什么要分 chunk？因为完整的 K cache [256, 128] = 64KB，已经等于整个 L1 容量。分成 4 chunk 后每个 packet 约 25KB，加上其他 buffer 能放进 L1。

Online softmax 保证分 chunk 处理的结果和一次性处理完全一致——这是数学上的等价变换，不是近似。

---

## 问题六：为什么不能把所有操作融合成一个巨大 kernel？

从 Worker 的角度看，所有 phase 都是同一个 tile 上的顺序执行。为什么不写一个 kernel 做完整层？

### 约束：L1 容量决定 phase 边界

每次 `acquire` 最多送 `packet_elements` = 15,368 bf16 ≈ 30KB 的新数据。一个完整层需要的权重数据：

| Phase | 所需数据 | 大小 |
|-------|---------|------|
| Q 投影 | 4 行 × 1024 | 8KB |
| K 投影 | 4 行 × 1024 | 8KB |
| V 投影 | 4 行 × 1024 | 8KB |
| Q/K RoPE | head + norm + cos + sin | 1KB × 2 |
| Attention ×4 | q + k_chunk + v_chunk + mask | 25KB × 4 |
| O 投影 | context(4KB) + residual + 4 行 × 2048 | 20KB |
| gate+up | hidden(2KB) + norm(2KB) + 8 行 × 1024 | 20KB |
| down | ffn(6KB) + residual + 4 行 × 3072 | 30KB |

总计：~220KB。L1 只有 64KB 且还要放代码、输出 buffer、tile-local state。

**Phase 边界不是计算边界，而是数据搬运边界**。每个 phase 对应「L1 里能放下的一个 packet 的处理」。

### 已经融合的操作

当前设计已经尽可能融合了：

| Kernel | 融合了什么 |
|--------|-----------|
| `q_shard_kernel` | RMSNorm + Q 投影行乘（一个 kernel 做两件事） |
| `gate_up_kernel` | post-attention RMSNorm + gate 投影 + up 投影（三合一） |
| `down_kernel` | down 投影 + residual add（二合一） |
| `o_kernel` | O 投影 + residual add（二合一） |
| `norm_rope_kernel` | QK Norm + RoPE 旋转（二合一） |

### 真正不可融合的边界

**只有 attention 是真正不可融合的**——因为它需要 KV cache 数据，而 cache 内容取决于历史 token 且由 host 管理。

如果 L1 无限大，最小 phase 结构是：

```
Phase A: 所有 pre-attention 线性变换 (RMSNorm + Q + K + V + RoPE)
Phase B: Attention (需要 KV cache，host 参与)      ← 唯一硬边界
Phase C: 所有 post-attention 线性变换 (O + residual + RMSNorm + gate + up + silu + down + residual)
```

当前 13 个 phase = ceil(220KB / 30KB) + 数据依赖链断点。packet 容量是唯一瓶颈。

### 未来优化方向

如果 XDNA 允许单 kernel 内多次 acquire/release（streaming model），可以在一个 kernel 中循环读取多个 weight packet，只要 kernel 代码能正确交替「等数据→算→释放→等下一个」。目前 IRON 的 kernel ABI 要求一次调用只处理一份已 acquire 的数据。

---

## 问题七：行分片并行是怎么工作的？

### 原理

每个 lane 计算同一矩阵乘法的不同输出行。以 Q 投影为例（`q_rows_per_packet=4`, `num_lanes=8`）：

```
Lane 0: 计算 Q 的第 0-3 行 × hidden    → output[0:4]
Lane 1: 计算 Q 的第 4-7 行 × hidden    → output[4:8]
Lane 2: 计算 Q 的第 8-11 行 × hidden   → output[8:12]
...
Lane 7: 计算 Q 的第 28-31 行 × hidden  → output[28:32]
```

每个 lane 的 packet 里只包含它负责的 4 行权重（4 × 1024 = 8KB）。所有 lane 通过广播获得相同的 hidden state。计算是**尴尬并行**的——lane 之间无需通信。

### Kernel 代码（Q 投影）

```c
// phase_owned_kernels.cc
void new_mega_phase0_q_shard_bf16(
    const bfloat16 *shared_packet,   // [hidden_state | norm_weight]（广播来的）
    const bfloat16 *lane_packet,     // [q_row0 | q_row1 | q_row2 | q_row3]（私有的）
    bfloat16 *hidden_state,          // tile-local hidden buffer
    float *state,                    // checksum
    bfloat16 *lane_output,           // 输出 buffer
    int32_t packet_size,
    int32_t hidden_size,             // 1024
    int32_t q_rows_per_packet,       // 4
    int32_t q_output_values_per_lane,// 8（含 padding）
    int32_t layer_id)
{
    // Layer 0: 从广播流初始化 tile-local hidden state
    if (layer_id == 0) {
        for (int i = 0; i < hidden_size; i++)
            hidden_state[i] = shared_packet[i];
    }

    // RMSNorm: 计算 1/sqrt(mean(x^2) + eps)
    float mean_square = 0.0f;
    for (int i = 0; i < hidden_size; i++) {
        float x = (float)hidden_state[i];
        mean_square += x * x;
    }
    float inv_rms = rsqrt(mean_square / hidden_size + 1e-6f);

    // 对每一行做 dot(normalized_x, weight_row)
    const bfloat16 *weight = shared_packet + hidden_size;  // norm weight
    for (int row = 0; row < q_rows_per_packet; row++) {
        const bfloat16 *q_row = lane_packet + row * hidden_size;
        float acc = 0.0f;
        for (int i = 0; i < hidden_size; i++) {
            float xnorm = (float)hidden_state[i] * inv_rms * (float)weight[i];
            acc += xnorm * (float)q_row[i];
        }
        lane_output[row] = (bfloat16)acc;
    }

    // Padding: 输出对齐到 8 的倍数
    for (int row = q_rows_per_packet; row < q_output_values_per_lane; row++)
        lane_output[row] = 0;
}
```

### 输出 Join：零拷贝 DMA 聚合

计算完成后，4 个 lane 的部分输出通过 L2 join 拼接成组输出：

```
Group 0 的 join buffer:
[lane0_output(200 bf16) | lane1_output(200 bf16) | lane2_output(200 bf16) | lane3_output(200 bf16)]
  offset=0                 offset=200               offset=400               offset=600
```

每个 lane 写入自己的固定偏移——不需要任何同步机制，因为 ObjectFIFO 的 acquire/release 语义保证了写入顺序。L2 收到所有 lane 的数据后，DMA drain 回 L3。

### 输出布局（每层每 lane）

```python
output_values_per_lane = (
    q_output_values_per_lane           # 8  (4行 padded 到 8)
    + k_output_values_per_lane         # 8
    + v_output_values_per_lane         # 8
    + q_rope_output_values_per_lane    # 8
    + k_rope_output_values_per_lane    # 8
    + context_output_values_per_lane   # 128 (head_dim)
    + attention_output_values_per_lane # 8
    + gate_up_output_values_per_lane   # 16 (gate + up 各 4 行)
    + residual_output_values_per_lane  # 8
)  # 总计 = 200 per lane, per layer
```

---

## IRON 核心概念四：TensorAccessPattern —— DMA 的「寻址程序」

在传统 CPU 编程中，数据访问模式由代码中的循环和指针运算决定。在 XDNA NPU 中，数据搬运由 DMA 引擎自主执行——你不能在运行时写循环来控制 DMA 的行为。取而代之的是**编译时配置的 Buffer Descriptor (BD)**。

`TensorAccessPattern` (TAP) 就是 IRON 对 BD 的 Python 抽象。它被编译进 artifact 中，告诉 DMA 引擎如何遍历 L3 缓冲区。

**类比**：如果把 DMA 引擎想象成一台「自动读卡器」，TAP 就是你预先编好的「读卡程序卡」——一旦插入（dispatch），读卡器就按程序自主工作，不需要 CPU 指导。

### TAP 与 ObjectFifo 的联动

```
TAP 决定：DMA 从 L3 的哪个位置读/写、按什么步长
ObjectFifo 决定：读出的数据放进 L1 的哪个 slot、什么时候推进

两者组合 = 完整的数据搬运策略
```

**关键性质**：TAP 是编译时固定的。这意味着 DMA 的寻址模式不能在运行时改变——这正是 KV cache 动态性必须通过 mask 而非 TAP 来处理的根本原因。

---

## 问题八：TensorAccessPattern (TAP) 的详细机制

### 本质

TAP 是 DMA Buffer Descriptor 的高层抽象。它告诉 DMA 引擎：「从缓冲区的哪里开始，按什么步长，读多少数据」。本质上是一个可配置的 memcpy，支持多维 stride。

### 4-元组语义

```python
TensorAccessPattern(
    buffer_shape,    # L3 缓冲区的完整形状
    offset,          # 起始偏移（以 element 为单位）
    sizes,           # 4维迭代计数 [d0, d1, d2, d3]
    strides,         # 4维步长 [s0, s1, s2, s3]
)
```

DMA 引擎按这个模式机械执行：

```
for i0 in range(sizes[0]):
  for i1 in range(sizes[1]):
    for i2 in range(sizes[2]):
      for i3 in range(sizes[3]):
        read buffer[offset + i0*strides[0] + i1*strides[1] + i2*strides[2] + i3*strides[3]]
```

### 三个关键 TAP 实例

**1. Shared 广播 TAP（所有层的 hidden + norm weight 连续传输）**

```python
shared_tap = TensorAccessPattern(
    (shared_input_elements,),              # 28 × 2048 = 57344 元素
    0,                                      # 从头开始
    [1, 1, 1, shared_input_elements],       # 一次传输全部
    [0, 0, 0, 1],                           # 连续读
)
# 效果：连续传输 57344 个 bf16，ObjectFIFO 每次 acquire 得到 2048 个（一层份）
```

**2. Lane Packet TAP（每个 lane 独立的权重流）**

```python
packet_taps[lane] = TensorAccessPattern(
    (input_elements,),                      # 全 L3 buffer 大小
    lane * total_phase_packets * packet_elements,  # Lane N 从这里开始
    [1, 1, 1, total_phase_packets * packet_elements],
    [0, 0, 0, 1],
)
# 效果：从 lane 的起始偏移连续读取 ~10MB，ObjectFIFO 每次 acquire 得到一个 packet (30KB)
```

**3. Output Drain TAP（带 stride 的分层写回）**

```python
out_taps[group] = TensorAccessPattern(
    (output_elements,),
    group * fabric_group_size * output_values_per_lane,  # 组偏移
    [num_layers, 1, 1, output_values_per_group],  # 28 个层的输出
    [output_values_per_layer, 0, 0, 1],            # 层间 stride
)
# 效果：每层输出写入 L3 的不同位置，层间有间隔（因为另一个 group 的数据在中间）
```

### TAP 的关键性质

1. **静态**：编译进 artifact，运行时不能修改
2. **与 FIFO 流控联动**：DMA 只在 Worker release buffer slot 后才推进
3. **一个 TAP 描述整个传输**：无论多长，DMA 自主执行，不需要 CPU 介入
4. **stride 范围有限**：最大 1,048,576。超过这个值会导致 lowering 失败

---

## IRON 核心概念五：Placer —— 从逻辑图到物理布局

当你在 Python 中写 `Worker`、`ObjectFifo`、`Buffer` 时，你只描述了逻辑关系（谁和谁通信、数据怎么流）。`Placer` 负责将这些逻辑实体映射到物理 tile 坐标。

```python
# 逻辑描述：8 个 Worker + 各种 FIFO
program = Program(device, runtime)

# 物理映射：SequentialPlacer 按顺序分配 tile
program.resolve_program(SequentialPlacer())
```

`SequentialPlacer` 的行为：
- 按列扫描物理 tile（col 0 → col 8）
- 为每个 Worker 分配一个 compute tile
- 为每个 L2 广播/join 节点分配一个 memory tile
- 为每个 L3 连接分配一个 shim DMA 端点

**Placer 失败意味着设计不可实现**——物理资源不够放下你的逻辑设计。常见失败原因：

- 一个 compute tile 被分配了 > 2 个输入 ObjectFifo → 端点不够
- 一个 memory tile 上堆了太多 FIFO 转发节点 → BD 数量耗尽
- 总 Worker 数超过 32 → 物理 tile 不够

这就是为什么 megakernel 设计必须从物理约束出发，而不是从计算图出发。

---

## 问题九：硬件资源的硬限制是什么？

### 必须遵守的限制

| 资源 | 限制值 | 超过后果 |
|------|--------|---------|
| Compute tiles 总数 | 32 | 无法放置更多 Worker |
| L1 每 tile | 64KB | packet + output + code + state 超了 → 编译失败 |
| 每 tile 输入 ObjectFIFO | ≤ 2 | 更多输入端点 → placer 失败 |
| 每 tile 输出 ObjectFIFO | ≤ 1-2 | 同上 |
| DMA BD 总数/memtile block | ~16 | 太多 FIFO 经过同一 memtile → 失败 |
| FIFO 对象对齐 | 16 字节 | 非对齐 → preflight 拒绝 |
| Packet 大小 | 8 bf16 的倍数 | DMA 对齐要求 |
| TAP stride 范围 | 最大 1,048,576 | lowering 失败 |
| `aie.mem` block 数 | ≤ 16 | 超过 → MLIR lowering 报错 |

### 当前生产配置的资源使用

```
compute_cores = 8（占 32 的 25%）
max_compute_tile_inputs = 2（shared + packet，刚好用满）
max_compute_tile_outputs = 1（lane_output → join）
total_dma_tasks = 12
max_fifo_buffered_bytes = 30736（< 64KB）
runtime_memrefs = 3（shared_l3, packets_l3, outputs_l3）
```

### 为什么只用 8 个 tile？

不是不想用更多——是当前设计已经把 8 个 tile 的输入/输出端点全部用满了（2 输入 + 1 输出）。要用更多 tile 做更粗粒度并行，需要新的拓扑设计（比如流水线而不是数据并行）。但 NPU2 的端点/BD 约束使得任何拓扑的设计空间都很窄。

### Preflight 检查清单

在真正下发到硬件之前，preflight 验证：

```python
assert compute_cores == num_lanes                    # 8
assert max_compute_tile_inputs <= 2
assert max_compute_tile_outputs <= 2
assert max_fifo_buffered_bytes <= 64 * 1024
assert all(obj_bytes % 16 == 0 for obj in all_fifos)  # 16 字节对齐
assert runtime_memrefs <= 5                           # host buffer 数量
```

这些检查在编译阶段就拦截错误——比在硬件上 hang 住然后盲目 debug 好无数倍。

---

## 问题十：怎么调试一个 28 层 × 13 phase 的 megakernel？

当一次 dispatch 执行 364 个 phase，输出错了怎么定位？

### 第一层：Checksum 累加器

每个 kernel 都把中间结果累加到 `state[0]`：

```c
float checksum = state[0];
for (int row = 0; row < q_rows_per_packet; row++) {
    float acc = /* dot product */;
    checksum += acc;              // 累加结果
    output[row] = (bfloat16)acc;
}
state[0] = checksum;
```

Python reference 用完全相同的累加顺序计算相同的 checksum。如果最终 checksum 匹配，则所有 364 个 phase 计算正确。如果不匹配，可以二分搜索是哪一层/哪个 phase 引入了错误。

### 第二层：逐 Phase 输出对比

每个 phase 把结果写入输出 buffer 的特定偏移：

```
output[layer * output_values_per_layer + lane * output_values_per_lane + phase_offset]
```

Runner 同时对比：
1. **cycle-accurate reference**：和 C++ kernel 完全相同的数学（相同的累加顺序、相同的精度）
2. **Qwen3 PyTorch reference**：用 `F.linear` 做全精度计算的标准答案

如果 NPU 输出和 cycle-accurate reference 一致但和 PyTorch 不同——说明精度问题（bf16 累加误差），不是 bug。如果和 cycle-accurate reference 都不同——说明某个 kernel 算错了或数据打包错了。

### 第三层：Preflight 结构检查

在下硬件之前验证图拓扑的正确性：

```
✓ compute core 数量 == 预期 (8)
✓ max tile inputs ≤ 2
✓ max tile outputs ≤ 2
✓ FIFO buffer 大小 ≤ L1 预算
✓ DMA 任务数在限制内
✓ 所有对象 16 字节对齐
```

如果 preflight 失败 → 问题在图拓扑（FIFO 连接方式、Worker 数量等）。如果 preflight 通过但输出错 → 问题在 kernel 数学或 packet 打包。

### 第四层：`event0()` / `event1()` 硬件 trace

每个 kernel 首尾有 `event0()` 和 `event1()` 调用，可以在硬件 trace 中看到每个 phase 的精确执行时间。如果某个 phase 异常慢或根本没执行，trace 会直接暴露。

---

## 问题十一：数据打包（Packet Packing）的具体布局

这是实现中最容易出错的环节——offset 错一个字节就全盘崩溃。

### Phase 0 (Q 投影) 的 Packet 布局

**Shared packet（广播）**：
```
[hidden_state[1024] | input_norm_weight[1024]]
 offset=0            offset=1024
 2KB                 2KB
```

**Lane packet（私有）**：
```
[q_weight_row0[1024] | q_weight_row1[1024] | q_weight_row2[1024] | q_weight_row3[1024] | padding...]
 offset=0             offset=1024            offset=2048            offset=3072
 2KB                  2KB                    2KB                    2KB
```

### Phase 5-8 (Attention Chunks) 的 Packet 布局

```
[q_head[128] | k_cache_chunk[64×128] | v_cache_chunk[64×128] | mask_chunk[64]]
 offset=0     offset=128              offset=8320              offset=16512
 256B         16KB                    16KB                     128B
```

总计 = 128 + 8192 + 8192 + 64 = 16,576 bf16 ≈ 32KB（< packet_elements）

### Phase 9 (O 投影) 的 Packet 布局

```
[attention_context[2048] | residual_shard[4] | o_weight_row0[2048] | ... | o_weight_row3[2048]]
 offset=0                  offset=2048         offset=2052
 4KB                       8B                  4KB × 4 = 16KB
```

### Phase 10 (Gate/Up) 的 Packet 布局

```
[attn_residual[1024] | post_norm_weight[1024] | gate_row0[1024] | ... | gate_row3 | up_row0 | ... | up_row3]
 0                    1024                      2048                     6144         10240
 2KB                  2KB                       2KB × 4                  2KB × 4
```

### Phase 11 (Down 投影) 的 Packet 布局

```
[ffn_hidden[3072] | residual_shard[4] | down_row0[3072] | down_row1[3072] | down_row2[3072] | down_row3[3072]]
 0                  3072                3076              6148              9220              12292
 6KB                8B                  6KB               6KB               6KB               6KB
```

总计 = 3072 + 4 + 4×3072 = 15,364 bf16 ≈ 30KB（几乎卡满 packet 容量上限！）

### 为什么 down 是最紧的？

因为 `intermediate_size=3072` 比 `hidden_size=1024` 大 3 倍。down 投影的每一行有 3072 个值，加上 ffn_hidden 向量本身也是 3072。这使得 down phase 的 packet 几乎用满了 15,368 的容量。如果要增加 `q_rows_per_packet`（从 4 行变成 8 行），必须同步增加 `packet_elements` 上限。

### 打包时必须同步更新的 5 个位置

修改任何 phase 的 offset 布局时，以下 5 处必须一起改：

1. **C++ kernel**：读取 packet 数据时的指针偏移
2. **runner.py 的 packet 填充代码**：往 lane_packets 里写数据的偏移
3. **runner.py 的 reference 计算**：Python 参考实现读 packet 的偏移
4. **ops.py 的 minimum_packet_elements 计算**：确保 packet 容量够用
5. **README 的 phase 文档**：人类可读的布局说明

遗漏任何一处都会导致难以调试的 mismatch。

---

## IRON 核心概念六：Runtime 与 Program —— 从 Python 到硬件二进制

### 编译流程全景

```
┌─────────────────────────────────────────────────────────────────────┐
│  Python 设计文件 (design.py)                                         │
│  ┌─────────┐  ┌──────────┐  ┌────────┐  ┌─────────┐  ┌─────────┐  │
│  │ Worker  │  │ObjectFifo│  │ Buffer │  │ Kernel  │  │ Runtime │  │
│  └────┬────┘  └────┬─────┘  └───┬────┘  └────┬────┘  └────┬────┘  │
└───────┼─────────────┼────────────┼────────────┼────────────┼────────┘
        │             │            │            │            │
        ▼             ▼            ▼            ▼            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Program(device, runtime).resolve_program(SequentialPlacer())        │
│                                                                      │
│  Placer 决定：                                                       │
│  - Worker A → 物理 tile (col=0, row=2)                              │
│  - Worker B → 物理 tile (col=1, row=2)                              │
│  - ObjectFifo X → 哪条 DMA 通道                                     │
│  - L2 广播节点 → 哪个 memory tile                                    │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  MLIR-AIE IR → aiecc 编译器                                          │
│  ├── Worker body → 每个 tile 的 ELF 二进制                           │
│  ├── Kernel .cc → 编译链接进 ELF                                     │
│  ├── ObjectFifo → DMA Buffer Descriptor 配置                         │
│  ├── TAP → BD 的 offset/sizes/strides 字段                          │
│  └── Runtime sequence → NPU 控制器指令序列                            │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Xclbin 包 → XRT 运行时加载到 NPU → 执行                            │
└─────────────────────────────────────────────────────────────────────┘
```

### Runtime 的角色

`Runtime` 不运行在 tile 上——它生成的是 **host-side 的 DMA 编排指令**。当 NPU 开始执行一个 dispatch 时，Runtime sequence 告诉 shim DMA 控制器：

1. 启动哪些 Worker
2. 从 host 的哪个 buffer、按哪个 TAP 往哪个 ObjectFifo 填数据
3. 从哪个 ObjectFifo、按哪个 TAP 把结果排回 host 的哪个 buffer
4. 什么时候算完成

Runtime sequence 本身也是编译时固定的——它被编入 artifact，不能运行时修改。Host 只能选择「把哪些数据放进 L3 buffer 再 dispatch」。

---

## 问题十二：Runtime Sequence 是怎么把所有东西组装起来的？

### Runtime 是什么

Runtime 描述了一次 dispatch 的 DMA 行为——谁填谁、谁排谁、什么顺序。它不是 Worker 代码的一部分，而是 host-side 控制面。

### 完整的 Runtime Sequence

```python
rt = Runtime()
with rt.sequence(shared_l3_ty, packets_l3_ty, outputs_l3_ty) as (shared_l3, packets_l3, outputs_l3):
    # 启动所有 Worker（Worker 会立即 acquire 并阻塞，等待 DMA 送数据）
    rt.start(*workers)

    # 创建一个 task group（所有 DMA 任务属于同一组）
    tg = rt.task_group()

    # 填充：shared 广播 FIFO（每组一个，同一份 shared_tap）
    for group in range(2):
        rt.fill(shared_l3_fifos[group].prod(), shared_l3, shared_tap, task_group=tg)

    # 填充：每个 lane 的私有 packet FIFO
    for lane in range(8):
        rt.fill(packet_fifos[lane].prod(), packets_l3, packet_taps[lane], task_group=tg)

    # 排出：每个 group 的 join 输出
    for group in range(2):
        rt.drain(q_join_fifos[group].cons(), outputs_l3, out_taps[group],
                 wait=True, task_group=tg)

    # 等待所有 DMA 完成
    rt.finish_task_group(tg)
```

### 执行时序

```
t=0:   rt.start() → 8 个 Worker 开始执行，立即 acquire 阻塞
t=0:   rt.fill() × 10 → 10 个 DMA 任务启动
       - 2 个 shared fill（L3 → L2 广播）
       - 8 个 packet fill（L3 → L1 直连）
t=ε:   DMA 送到第一个 packet → Worker 的 acquire 返回 → 开始计算
       ...Worker 算完 release → DMA 自动送下一个 packet → Worker 再 acquire...
t=T:   所有 28×13 个 phase 执行完毕 → 所有 output release 完毕
       → drain 将 output 送回 L3
       → rt.finish_task_group() 返回
       → dispatch 结束
```

**关键点**：所有 DMA 任务同时启动，但由 FIFO backpressure 自然限流。不需要软件手动排序「先送 phase 0 再送 phase 1」——DMA 按 TAP 顺序推进，Worker 按代码顺序 acquire，两者通过 FIFO 同步。

### 3 个 Host Buffer（Runtime Memrefs）

整个 dispatch 只需要 3 个 host-side 缓冲区：

```
shared_l3:   57,344 bf16 = 112KB    (28层的 hidden + norm weight)
packets_l3:  按配置 ~86MB            (8 lane × 28层 × 13 phase × 30KB)
outputs_l3:  按配置 ~44KB            (28层 × 8 lane × 200 value)
```

这满足 `runtime_memrefs ≤ 5` 的硬约束。如果设计需要更多缓冲区（比如把 KV cache 作为单独输入），必须确保不超过 5 个。

---

## 问题十三：GEMV Scaling 实验教会了我们什么？

在固定 megakernel 之前，我们做了每种投影的独立 column scaling 实验。

### 核心发现：扩展不是单调的

| 投影 | 矩阵尺寸 | 最佳 column 数 | 延迟 (µs) | vs 1-col 加速比 |
|------|----------|---------------|-----------|----------------|
| Q | 2048×1024 | 4 | 598 | 1.40x |
| K | 1024×1024 | 2 | 393 | 1.41x |
| V | 1024×1024 | 1 | 972 | 1.00x（不值得并行）|
| O | 1024×2048 | 4 | 515 | 1.92x |
| gate | 3072×1024 | 4 | 522 | 2.54x |
| up | 3072×1024 | 8 | 546 | 2.41x |
| down | 1024×3072 | 8 | 487 | 2.78x |

统一 4 column：4,718µs。统一 8 column：5,264µs（反而慢了！）。Per-shape 最优：4,033µs。

### 教训

1. **不要假设 "more lanes is always faster"**——通信开销（L2 broadcast/join 延迟）在小矩阵上抵消了并行收益
2. **V 投影用 1 column 最快**——因为 K/V 矩阵比 Q 小（GQA），额外 lane 带来的通信开销大于计算收益
3. **down 投影从 8 column 收益最大**——因为矩阵最大（3072 维），每 lane 的计算量足以分摊通信

### 在 Megakernel 中的应用

当前 megakernel 对所有投影使用统一的 8 lane（因为它们共享同一组 Worker）。但每个 phase 只传输自己需要的行数——如果某个投影不需要这么多 lane，额外 lane 只是做了空计算（padding 行输出 0）。未来优化方向是让 lane 在不需要的 phase 里跳过计算（而不是算 0）。

---

## 问题十四：为什么 shared_fifo 用 depth=2 而 packet_fifo 用 depth=1？

### Depth 的含义

FIFO `depth` = L1 中预分配的 buffer slot 数量。`depth=2` 意味着 DMA 可以在 Worker 处理 slot 0 的同时向 slot 1 预填充——经典的双缓冲。

### 为什么 shared 用 depth=2？

Shared data 从 L3 → L2 → L1 经过两跳。如果 depth=1：

```
Worker acquire → 计算 → release → DMA 开始填充 → 等待两跳延迟 → Worker acquire（被阻塞）
```

Worker 在每层开始时被 DMA 延迟阻塞。depth=2 允许 DMA 在 Worker 计算时提前准备下一层的数据：

```
Worker 处理 layer N 的 shared data（slot 0）
同时 DMA 将 layer N+1 的 shared data 填入 slot 1
```

### 为什么 packet 只用 depth=1？

Packet 是 ~30KB。如果 depth=2 就需要 60KB 的 L1 空间——已经用完整个 64KB L1 了（还没算其他 buffer）。所以只能 depth=1：DMA 和计算串行交替，不能重叠。

```
总 L1 预算 = 64KB
- packet buffer (depth=1): 30KB
- shared buffer (depth=2): 4KB × 2 = 8KB
- lane_output buffer: ~400B
- hidden_state: 2KB
- state + attention_state + attention_acc: ~520B
- 代码 + 栈: ~3KB
总计 ≈ 44KB（还有 ~20KB 余量）
```

如果 packet 用 depth=2（60KB），加上其他 buffer 就超 64KB 了。L1 是最紧的资源瓶颈。

---

## 问题十五：为什么 `acquire` 必须和 `release` 精确配对？

### 硬件行为

ObjectFIFO 是硬件流控机制。`acquire(1)` 消耗一个「token」（表示 buffer slot 可读）。`release(1)` 归还一个 token（表示 slot 空闲，DMA 可以填入新数据）。

如果 acquire 和 release 不配对：
- 多 acquire 少 release → DMA 被永远阻塞（所有 slot 都被 Worker 持有），deadlock
- 少 acquire 多 release → 非法状态，硬件行为未定义

### Shared FIFO 的持有策略

Shared data 被 3 个连续的 phase 使用（Q/K/V 投影都需要 hidden + norm weight）：

```python
shared = shared_fifo.acquire(1)       # 获取本层的 shared data

# Phase 0: Q 投影（使用 shared）
q_shard_kernel(shared, ...)

# Phase 1: K 投影（继续使用同一个 shared）
projection_kernel(shared, ...)

# Phase 2: V 投影（shared 最后一次使用）
projection_kernel(shared, ...)
shared_fifo.release(1)                # 释放！DMA 可以开始送下一层的 shared data
```

如果在 phase 0 之后就 release，DMA 可能在 phase 1 还在读 shared 数据时就覆盖了它。持有直到最后一个消费者用完才释放。

### Lane Output FIFO 的持有策略

Output buffer 跨越整层的所有 phase：

```python
lane_output = lane_output_fifo.acquire(1)   # 层开始时获取

# Phase 0-11 的所有 kernel 都往 lane_output 的不同 offset 写入
q_shard_kernel(..., lane_output, ...)        # 写 offset [0:8]
projection_kernel(..., lane_output, ...)     # 写 offset [8:16]
...
down_kernel(..., lane_output, ...)           # 写 offset [184:200]

lane_output_fifo.release(1)                 # 整层输出写完才释放
                                            # → 触发 join DMA，把数据送到 L2
```

---

## 设计食谱总结

1. **从数据搬运约束开始设计**，不是从计算图。问：每个操作需要多少数据？怎么送到 tile？能放进 30KB 的 packet 吗？

2. **预排所有权重到 per-lane L3 连续缓冲**。一个 DMA 任务流式传输整个序列。Phase "切换"就是 DMA 在缓冲区中前进。

3. **用 tile-local Buffer 保存跨迭代状态**。避免层间的 L3 往返。2KB 的 L1 代价换取 28 次 DMA 往返的节省。

4. **按 4 lane 分组做 broadcast/join**。不要尝试 8 路直接扇出——shim 端点扛不住。

5. **严守 tile 端点限制**：2 输入 + 1 输出。这是最紧约束——它决定了拓扑复杂度上限。

6. **以 packet 为设计单元**。每个 kernel 消费恰好一个 packet，产出到预分配的 output slot。Packet 大小是根本预算。

7. **用固定形状应对动态行为**。KV cache 用 max-shape + mask。位置信息通过数据值传入，不改图拓扑。

8. **在 L1 容量范围内激进融合**。唯一的硬边界是 attention（KV cache 交互）。其他操作融合到 packet 装不下为止。

9. **结构性验证优先于数值验证**。Preflight 抓拓扑错误，checksum 抓数学错误，per-phase 输出对比定位错误位置。

10. **不要靠复制扩展，靠增加循环次数扩展**。图保持静态；只有数据流变长。
