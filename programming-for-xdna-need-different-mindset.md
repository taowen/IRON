## 问题：Decode 为什么慢

LLM 推理的 decode 阶段，每次只生成一个 token：

```
输入: 1 × 4096 向量 (8 KB)
权重: 一层约 115 MB（Qwen3-8B, Q4 量化后）
输出: 1 × 4096 向量 (8 KB)
```

打个比方：你开一辆大货车去仓库拉 115 MB 的货（权重），只为加工一件 8 KB 的小零件（当前 token 的 hidden 向量）。这 115 MB 当然都参与了计算，但每个权重只被用来服务一个 token，算术强度很低。瓶颈主要在"跑这趟路"上，不在"加工"上。这就是典型的 **memory-bound** 场景。

GPU 在这个场景下容易遇到三个结构性问题：

**中间结果反复写回显存。** 一层 transformer 通常会被拆成多个 kernel：RMSNorm、Q/K/V projection、attention、O projection、MLP 等。每个 kernel 做完要把几 KB 到几十 KB 的中间结果写回 HBM，下一个 kernel 再读回来。这些中间结果本身不大，但每次往返都要和权重流争内存带宽，还要付 kernel launch 的固定开销。就像加工完一颗螺丝钉，非要跑一趟仓库把半成品存进去再取出来，路上的时间可能比加工时间还长。

**Kernel launch 延迟叠加。** 每个 kernel launch 都有固定开销。单个小 batch decode kernel 的实际计算时间很短时，这些固定开销会变得很显眼。

**大规模并行资源不容易喂满。** GPU 是 SIMT 机器，需要足够多的并行工作来摊薄调度和访存开销。batch=1 decode 只有一条当前 token 向量，很多执行 lane 会被访存、同步和小矩阵形状限制住。

核心矛盾：**GPU 擅长高并行度、高算术强度的任务；batch=1 decode 需要的首先不是更多峰值算力，而是更高的带宽利用率和更少的中间搬运。**

---

## XDNA 的回答：一张 8×6 的 Tile 阵列

XDNA NPU 不是更小的 GPU。它是一种完全不同的机器——**空间数据流架构**。

想象一个工厂车间：不是一个万能机器人拿着零件到处跑（CPU），也不是一千个机器人同时干同一道工序（GPU），而是**一条固定的流水线**——每个工位干一件事，零件从一头进去、成品从另一头出来，中间不下线。XDNA 就是这条流水线的芯片实现。

下面用一个 8 列 × 6 行的 XDNA 风格物理网格举例。不同产品的阵列规模会变化，但基本组成类似：底部 shim 负责主存接口，中间 memtile 负责片上搬运和缓存，上方 compute tile 负责运行程序。以 Qwen3 decode 的一层融合执行为例，可以把这些 tile 分配成如下角色：

```
       c0     c1     c2     c3     c4     c5     c6     c7
      ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
row5  │attn  │      │ main │ main │ main │ main │      │attn  │  Compute Tile
row4  │attn  │      │ main │ main │ main │ main │      │attn  │  (每个 64 KB, 跑 C++)
row3  │attn  │post  │ main │ main │ main │ main │      │attn  │
row2  │attn  │vector│ main │ main │ main │ main │swiglu│attn  │
      ├──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┤
row1  │kv-L  │bridge│ wt+c │ wt+c │ wt+c │ wt+c │ hub  │kv-R  │  Memtile (512 KB, 纯 DMA)
      ├──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┤
row0  │shim-K│shim-H│shim-W│shim-W│shim-W│shim-W│      │shim-V│  Shim (主存入口)
      └──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘
```

从下往上看，像一栋三层楼的工厂：

**一楼（Shim, Row 0）是收发室。** 所有数据从主存进出都要经过这里。它不做计算，只负责搬运。

**二楼（Memtile, Row 1）是中转仓库。** 512 KB 的大缓存，但不跑程序——完全靠 DMA 硬件配置来做分拣：把大块权重拆成小块分发给楼上的工位，把楼上各工位的小块产出汇总打包。

**三到六楼（Compute Tile, Row 2-5）是车间。** 每个工位只有 64 KB 内存，跑编译好的 C++ 程序。关键限制：**每个工位看不到隔壁的内存**，数据只能通过物理连线（stream）和 DMA 传入传出。

图中各角色的含义：
- **main**（16 个）：做矩阵向量乘的主力。Q/K/V/O/Up/Gate/Down 七个投影全由这 16 个 tile 按时间轮转完成。
- **vector**（1 个，c1r2）：持有完整 hidden 向量，做 RMSNorm 和 residual add。整层中被复用两轮。
- **post**（1 个，c1r3）：Q/K 的 head-wise norm + RoPE 位置编码。
- **attn**（8 个，左右各 4）：分组做 attention 的 score 计算和加权求和。
- **swiglu**（1 个，c6r2）：做 SiLU(gate) × up 激活函数。
- **bridge/hub/wt+c/kv-L/kv-R**：Memtile 层的各种中转节点。

这套硬件有几个关键约束，直接决定了编程方式：

| 约束 | 编程影响 |
|---|---|
| 每个 Compute Tile 只有 64 KB | 不可能加载完整权重，必须流式处理 |
| Tile 之间内存不可见 | 不能共享内存，必须显式搬运 |
| DMA 与 Compute 独立运行 | 可以双缓冲，搬运和计算完全重叠 |
| 同步靠硬件 Lock（计数器） | 不需要软件调度，数据到了自动触发 |
| Stream 是编译时确定的物理连线 | 运行时不能改路由，拓扑完全静态 |
| Memtile BD 有 bank/编号规则 | BD 必须放到对应 DMA channel 能访问的 bank |
| Packet ID 需要避免冲突 | 同一 packet 路由域内重复会导致数据送错目标 |

**所有这些约束指向同一个目标：让数据以极低的软件开销在芯片内部流动。** 一旦配好管道，runtime 不再逐个 operator 调度；数据按预设路径流过 tile、memtile 和 stream switch。

代价是：你必须在编译时确定所有路径。运行时不能临时改主意。

---

## 全层融合：用整张 Tile 阵列跑一层

### 一次提交跑完整层

GPU 通常把一层 transformer 拆成多个 kernel 分别提交。XDNA 更适合把一层组织成**一次提交**的静态数据流：host 填输入、填权重、启动 NPU，然后等待输出。

下面是完整的物理路径。第一次看可以跳过细节，先感受"数据从头到尾没有回过主存"这个事实：

```
[Host → BO → Shim]
    ↓ hidden vector (8 KB) 进入芯片
[c1r2 vector station]
    做 RMSNorm，产出 activation（4096 bf16）
    广播 12 次给 main16（Q 需要 8 次，K/V 各 2 次）
    ↓
[main16: 16 个 tile 并行做 Q/K/V 投影]
    每个 tile 负责 32 行输出，做 Q4NX 反量化 + 乘加
    产出 record → 汇聚成 compact（512 bf16）
    ↓
[c1r3 postprocess]
    对 Q/K 做 head-wise norm + RoPE 旋转位置编码
    current K/V 写回主存（下次 decode 要用）
    Q payload 送去做 attention
    ↓
[8 个 attention tile: 流式 online softmax]
    每次处理 16 个历史 token，产出 carrier
    逐 block 累加合并，产出 attention output
    ↓
[main16: O 投影]  →  c1r2 做 residual add
    ↓
[c1r2: 第二次 RMSNorm，广播 48 次给 main16]
    ↓
[main16: Up/Gate 投影]  →  [c6r2: SwiGLU 激活]  →  [main16: Down 投影]
    ↓
[c1r2: 第二次 residual add → 输出 → Shim → Host 读回]
```

理想情况下，整条路径中权重按顺序从主存流入，最终 hidden 向量写回主存，中间激活（Q/K/V/O/attention/up/gate/down 结果）尽量全部在片上流转。这样做的价值不只是少写几 KB 中间结果，而是避免把每个 operator 都变成一次独立的主存往返和调度事件。

### 按计算类别复用 Tile

注意 `main16` 在路径中出现了四次（Q/K/V → O → Up/Gate → Down）。这不是四组不同的 tile，是**同一组 16 个 tile 跑了七轮**。c1r2 也出现了两次（两轮 RMSNorm + 两次 residual add），同样是一个 tile。

Qwen3-8B 一层有 7 个投影。如果每个投影独占 16 个 tile，需要 112 个——芯片没有那么多。解法是**按计算类别分 tile，按时间复用**：

| Tile 组 | 做什么 | 怎么复用 |
|---|---|---|
| main16 (16 个) | 量化矩阵向量乘 | 7 个投影轮流跑，kernel 代码完全相同 |
| c1r2 (1 个) | RMSNorm + residual | 两轮 norm + 两次 add，是一个持续运行的状态机 |
| c1r3 (1 个) | Q/K norm + RoPE | 只在 QKV 阶段用一次 |
| attn (8 个) | score + merge | 只在 attention 阶段用一次 |
| c6r2 (1 个) | SwiGLU | 只在 MLP 阶段用一次 |

main16 的 projection kernel 对所有投影是**同一类程序**。Tile 不需要理解自己在做 Q 投影还是 Down 投影；它只需要处理"来了一个 weight chunk + 一个 activation chunk，做乘加，满了就输出"这个 ABI。区别只在于 DMA 给它喂的数据来自不同 phase 的权重流。

"换投影"不需要停机重配。权重流是连续的：Q 的最后一个 chunk 搬完，下一个进来的自然就是 K 的第一个 chunk。BD ring 自动轮转，零切换开销。

c1r2 这样的 full-vector station 则不同。它负责持有完整 hidden/residual 向量，并按 phase 顺序做 input RMSNorm、attention residual add、post RMSNorm、down residual add 等工作。它不是"做完退出再被调用"，而是一个持续运行的状态机，每完成一个阶段就等下一个 lock。

**这就是一张有限 tile 阵列能覆盖整层的关键：计算量最大的矩阵向量乘由 16 个 tile 按时间轮转完成七轮，其他特殊计算由专职 tile 在对应阶段接入。**

---

## 让 Decode 变快的五个编程思维转换

### 一、权重必须"流过"，不能"加载"

一个 4096→4096 的投影有 14 MB Q4NX 权重。一个 tile 有 64 KB——连 256×256 的 bf16 小矩阵（128 KB）都放不下。

GPU 程序员习惯 `weight[row][col]` 随机索引。XDNA 上完全不可能。权重必须设计成一条流——像传送带上的零件，到了就用，用完就让下一个进来：

```c
// 每个 main tile 跑的 kernel（简化版）
static float accum[32];  // 跨 chunk 保持的累加器

void q4nx_chunk_accum(bfloat16 *packed_chunk, bfloat16 *activation, int32_t rows) {
    // packed_chunk = 5 KB 的 Q4NX 权重块（32 行 × 256 列）
    // 解包格式：前面是 scale 和 zero_point，后面是 4-bit 数据
    bfloat16 *scales = packed_chunk;
    bfloat16 *zeros = packed_chunk + 32 * 8;
    uint8_t *data = (uint8_t *)(packed_chunk + 32 * 8 * 2);

    for (int row = 0; row < rows; row++) {
        float row_acc = 0.0f;
        for (int group = 0; group < 8; group++) {
            float scale = float(scales[group * 32 + row]);
            float zero = float(zeros[group * 32 + row]);
            for (int dim = 0; dim < 32; dim++) {
                // 从 4-bit 解压成 float，乘以 activation，累加
                uint8_t q4 = /* 解包一个 4-bit 值 */;
                row_acc += (float(q4) - zero) * scale * float(activation[group*32 + dim]);
            }
        }
        accum[row] += row_acc;  // 累加，不是覆盖——跨 chunk 汇总
    }
}
```

Q4NX 是一种 4-bit 量化格式：每 32 个权重共享一对 `(scale, zero_point)`，反量化公式是 `weight = (q4_value - zero) * scale`。4-bit 主体数据只有 bf16 的 1/4，实际格式还会包含 scale/zero 等元数据。计算时必须在线解压。关键是**不需要把整个矩阵解压出来**——每到一个 5 KB 的 chunk，在 tile 内部解压、乘加、累加到 32 个 float，然后这个 chunk 就可以被下一个覆盖了。

一个投影有 16 个这样的 chunk（256 列 × 16 = 4096 列），accumulator 累加 16 次后 flush 成输出。整个过程中 tile 里同时只需要放下：2 个权重 buffer（ping-pong）+ 1 个 activation buffer + 1 个 accumulator + 1 个 output buffer ≈ 15 KB，远小于 64 KB。

DMA 和 compute 用 **ping-pong 双缓冲**完全重叠：

```
时间 ──────────────────────────────────────────────────────►
DMA:     [搬 chunk0→ping] [搬 chunk1→pong] [搬 chunk2→ping] ...
Compute:                  [算 ping]        [算 pong]        ...
```

没有 ping-pong 时，DMA 和 compute 必须串行（共用一个 buffer 会互相踩）。有了两个 buffer + lock 保证"搬完才能算，算完才能覆盖"，两者就可以同时工作。

### 二、DMA 是自转的硬件引擎

CPU 搬数据靠 `memcpy`——软件逐字节拷贝。GPU 搬数据靠 `cudaMemcpy` 或 kernel 里的 load——需要软件发起每次搬运。

XDNA 的 DMA 是**独立的硬件状态机**。你给它一张 BD（Buffer Descriptor，任务单），它自己搬，不需要任何人再管：

```
一个 BD 包含五个字段：
  地址      : 从哪个 buffer 搬 / 搬到哪个 buffer
  长度      : 搬多少数据
  启动条件  : 等哪个 lock 信号才开始（acquire）
  完成通知  : 搬完后释放哪个 lock 信号（release）
  下一个 BD : 搬完后自动执行哪张任务单
```

两个 BD 首尾相连就形成 **BD ring**——DMA 永远在转这个环：

```
BD0(ping): 等 ping 空 → 搬数据到 ping → 通知 ping 满 → 跳到 BD1
BD1(pong): 等 pong 空 → 搬数据到 pong → 通知 pong 满 → 跳回 BD0
    ↑                                                         │
    └─────────────────── 无限循环 ────────────────────────────┘
```

启动时 runtime sequence 只需要把 BD ring 推进 DMA 队列，并给出重复次数，DMA 就会自动执行一串搬运任务。整个投影阶段的权重搬运不需要 host 对每个 chunk 逐次介入。

这里的 lock 是一个**硬件计数器**——`acquire(lock, N)` 表示"等计数器 ≥ N 再减 N"，`release(lock, N)` 表示"计数器加 N"。如果权重要分发给 2 个 tile 共享，lock 初值设为 2：producer 搬完 release 2，两个 consumer 各 acquire 1，各自读完各 release 1 回来凑够 2，producer 才能覆盖 buffer。就像自助餐的取餐位：厨师放好菜，两位客人各自取完、各还一个空盘，厨师看到 2 个空盘才上新菜。

BD 编号也有硬件可见性约束。Memtile 的 BD 存储器按 bank 组织，某个 DMA channel 只能访问特定范围的 BD。如果分配错了，那张"任务单"对控制器不可见，DMA 永远不执行，表现为系统静默死锁。高层编译器不一定会替你检查这类物理约束。

### 三、Attention 不物化矩阵，靠 carrier 流式合并

标准 attention 算 `softmax(QK^T/√d) · V`。其中 `QK^T` 产生一个 score matrix，维度是 `[32 heads × context_length]`。context=1024 时，如果按 bf16 存完整 score matrix 就要约 64 KB；如果还要保留 softmax 中间状态、V 加权结果和程序本身，就超过了一个 tile 能舒服承载的范围。

一种适合 XDNA 的做法是**把 KV cache 切成小 block，每次只看一小块**。例如按 16-token block 处理时，Shape-A tile 对 16 个 token 算出局部 softmax 统计量（scores → block max → exp → block sum），然后把结果打包成一个小的 **carrier**：

```c
// Shape-A: 处理一个 16-token block，产出 carrier
for (int32_t q_head = 0; q_head < 8; q_head++) {
    // 1. 算 16 个 score（Q 和这 16 个 K 的点积）
    // 2. 找这 16 个 score 的 max
    // 3. exp(score - max) 得到未归一化 weight
    // 4. 打包成 carrier: (16 个 weight, block_max, block_sum)
    scalars[q_head * 2] = running_max;
    scalars[q_head * 2 + 1] = weight_sum;
}
```

carrier 里的 `per-token weights`、`block_max`、`block_sum` 是 online softmax 合并所需的充分统计量。有了这些值，下游就能和之前的 block 正确合并，不需要回看原始 score。

Shape-B tile 做 **online merge**。核心思想是：之前 block 的 softmax 用的是旧的 max。如果新 block 出现了更大的 max，之前那些值的相对重要性被高估了，需要用 `exp(old_max - new_max)` 缩小之前的累加结果：

```c
// Shape-B: 每收到一个 carrier + 对应的 V block
float new_max = max(old_max, block_max);
float old_scale = fast_exp(old_max - new_max);    // < 1，缩小旧结果
float block_scale = fast_exp(block_max - new_max); // ≤ 1，缩放新 block

for (dim = 0; dim < 128; dim++) {
    float block_weighted_v = sum(weight[token] * V[token][dim]);
    accum[dim] = accum[dim] * old_scale + block_weighted_v * block_scale;
}
state_max = new_max;
state_sum = old_sum * old_scale + block_sum * block_scale;
```

打个比方：你在统计全班的加权平均分。传统做法是等所有人的分数都出来再算。Online 做法是每来 16 个人的分数就更新一次加权平均——如果新来的这批有更高分的人，就把之前的权重都按比例调低。数学上可以证明最终结果和一次性算完全相同。

所有 block 扫完后，Shape-B 只需要保存 running max、running sum 和 output accumulator 等小状态。而 KV cache 可以边从主存搬入边被消费：第一个 block 在计算时，第二个 block 已经在路上了。

### 四、不能依赖通用数学库——要控制近似函数

AIE2P 这类 tile 处理器更像嵌入式向量核心，不适合把 CPU/GPU 上的通用数学库原样搬过来。即使编译器能提供 `sqrtf`、`expf` 这类函数，它们也可能带来很大的代码体积和 cycle 成本。高性能 kernel 通常会自己控制近似函数。

RMSNorm 需要 `1/√x`——用 Newton-Raphson 迭代：

```c
static float fast_rsqrt(float value) {
    // 分段给出初始猜测（确保在正确答案 2 倍以内）
    float y = 1.0f;
    if (value < 0.015625f) y = 8.0f;
    else if (value < 0.25f) y = 2.0f;
    else if (value > 4.0f) y = 0.25f;
    // ... 更多分段 ...

    // 10 次迭代，每次精度翻倍：y_new = y * (1.5 - 0.5*x*y²)
    const float half = value * 0.5f;
    for (int iter = 0; iter < 10; iter++)
        y = y * (1.5f - half * y * y);
    return y;
}
```

Newton 法对初始猜测敏感——如果猜差了 10 倍，10 次迭代可能不够收敛。所以用 if-else 分段保证第一次猜测就接近正确值。

Softmax 的 exp——利用 `e^x = 2^(x/ln2)` 拆成整数幂和小数部分：

```c
static float fast_exp(float value) {
    int32_t n = floor(value * 1.4426950f);   // x/ln2 的整数部分
    float frac = value - n * 0.6931472f;     // 小数部分 ∈ [0, ln2)
    float poly = 1 + frac + 0.5*frac² + 0.167*frac³ + 0.042*frac⁴;  // Taylor 展开
    return pow2(n) * poly;
}
```

`pow2(n)` 是零计算开销——直接构造 IEEE 754 float 的位模式：`2^n` 的 float 表示就是 mantissa 全 0、exponent = n+127，一条整数移位指令搞定。

SwiGLU 的 sigmoid 可以用小查找表 + 线性插值实现。几十到上百个采样点只占几百字节，放在 64 KB local memory 里很便宜，精度对 bfloat16 路径通常已经足够。

这些近似为什么重要？因为 64 KB tile 里**代码和数据共享空间**。如果通用 math 库让代码体积膨胀、执行 cycle 增多，就会拖慢整个流水线节拍：DMA 搬完了数据，compute 还没算完，双缓冲失去意义。定制近似的目标不是炫技，而是让 kernel 足够小、足够快，并且误差可控。

### 五、Packet 路由和 Channel 所有权

每个 tile 只有几个 DMA channel（2-6 个）。但融合层有几十路数据流。如果每路独占一条物理线，资源很快耗尽。

XDNA 用 **packet routing** 解决：在数据前面加一个 header（packet ID），多路数据共享同一条物理线，接收端的硬件路由器根据 ID 筛选。就像同一条马路上跑不同车牌号的车，每个收费站只放行特定车牌。

```python
# 融合层中的 packet 路由示意：多种数据共享同一段物理通路
packet A: activation replay
packet B: attention result
packet C: FFN intermediate
```

硬约束：**同一 packet 路由域内的 packet ID 必须避免冲突**。如果两条不相关的路碰巧用了同一个 ID，硬件路由器可能把数据送错地方，不一定 crash，但结果会悄悄变错。

与 packet 路由对应的是 **circuit flow**——独占一条物理线的点对点连接，带宽更高但占一条线的资源。权重这种持续大流量用 circuit flow，间歇性的 compact 结果用 packet flow。

另一个物理约束是 **channel 所有权**。每个 DMA channel 有自己的 BD ring 和 lock 状态，同一时间只能服务一路数据流。如果两路不相关的数据共用一个 channel——producer A 的 release 会被误认为是 producer B 的数据就绪信号，consumer 拿到"满"信号去读，读到的是错误数据，后续全错或死锁。

这不是"最佳实践"，是**物理约束**。开发这类程序时，通常需要在编译前额外做结构检查：每个 channel 的 owner 是谁、BD ring 属于谁、lock pair 是否只被同一条数据流使用。因为很多错误编译器不会报，运行时只会表现为 timeout 或数值漂移。

---

## Record ABI：让 16 个并行 Tile 的输出可预测

16 个 main tile 并行计算同一个投影的不同输出行。如果让每个 tile 直接写最终 tensor 的对应位置，需要精确管理 16 路不同的全局偏移，BD 配置会迅速变复杂，出错时也很难区分"路由错了"还是"计算错了"。

解法是定义一个稳定的小包格式 **record**，类似网络协议中的帧：

```c
// 17 dwords = 1 header + 32 bf16 payload
int32_t header = (phase << 24) | (block << 20) | (group << 16) | (row << 8) | 0xD0;
```

header 编码了"这是哪个投影、哪个 block、哪列、哪行的结果"。所有 tile 产出完全相同格式的 record，下游按固定规则汇聚。

汇聚是层级的，像快递系统：快递员→站点→分拣中心：

```
16 tiles 各产出 17-dword record（1 header + 32 bf16）
  ↓ 每列 4 个 tile 的 record 进入该列 Memtile
column compact: 65 dwords（1 header + 128 bf16）
  ↓ 4 列的 column compact 进入 c1r1
global compact: 257 dwords（1 header + 512 bf16 = 完整 output block）
```

16 tiles × 32 行/tile = 512 行，正好是一个 output block。这个汇聚全由 Memtile 的 DMA 配置完成——BD 的 offset 字段跳过 header 只取 payload，length 字段精确裁剪，stride 字段拼接到正确位置。**没有一行代码，全是硬件配置。**

debug 时 header 是关键线索：header 对但 payload 错 → 问题在计算 kernel；header 就错了 → 问题在路由配置。

---

## 编译一次，用 Runtime 参数覆盖多个 Token

NPU 编译（例如 MLIR-AIE → xclbin）通常比一次 decode 慢得多。但 decode 每个 token 的参数不同：生成第 5 个 token 时 scan 5 个历史 token，生成第 500 个时 scan 500 个。Block 数、tail token 数、KV cache 写入位置全变。每个 token 都重编译根本不可行。

一种常见做法是把拓扑编译成最大容量版本，然后在运行时只 patch 少量描述符和 RTP（runtime parameter）。就像一份已经签好的合同，只需要改几个空格里的数字：

```python
patched = patch_instruction_stream(
    base=load("max_context.bin"),
    patches=[
        (rtp_offset, current_token),
        (block_count_offset, blocks_to_scan),
        (iteration_offset, blocks_to_scan),
        (repeat_offset, blocks_to_scan - 1),
        (kv_write_offset, current_token_write_offset),
    ]
)
```

安全性验证的思路是：对若干代表性 token 位置，独立生成一份完整 instruction stream，然后和 patch 版本逐 word 比对。两者一致，才说明 patch 只改了该改的字段。

有一个时序陷阱：**RTP 必须在 tile 的 core 读取之前写好**。NPU 启动后 core 可能很快跑到读取 RTP 的位置，但 host 写 RTP 要通过 instruction stream、shim 和片上路径传进去，有延迟。如果 core 已经读了但值还没到，就会读到初始值。

解法是 **runtime-start lock**：core 一开始就 acquire 一个 lock 阻塞，host 写完 RTP 后才 release 这个 lock。这样 core 在 RTP 准备好之前不会继续。没有这道闸门时，典型现象是很多 token 的行为都像 token 0。

---

## Debug：三级验证代替事后猜测

CPU 程序出 bug 有 segfault、stack trace 可以定位。NPU 的错误只有两种表现：**timeout**（某个 lock 永远等不到）或**数值不对**（跑完了但结果和 reference 不一致）。没有行号、没有 stack trace、没有 core dump。

因此更实用的方法不是"跑了看看"，而是**三级验证，逐级缩小问题范围**：

### 第一级：结构检查（秒级，不编译不跑）

在生成硬件描述后、编译之前，用脚本验证物理约束：

```python
# 验证 contract 常量一致性（record 大小、block 数、replay 次数）
errors = validate_contract()
# 验证 dataflow 图（边的 dwords 是否和节点角色一致）
errors += validate_dataflow()
# 验证生成的 MLIR 中物理标记的出现次数
errors += validate_generated_mlir(mlir_text, schedule)
# 验证 KV cache 物理 layout 契约
errors += validate_cache_layout_contract(schedule)
```

具体检查例如：确保 16 个 main tile 都正确启动了权重接收 DMA，确保 weight stream 和 compact gather 没有抢同一个 channel，确保 packet ID 在同一 packet 域内没有冲突。这类检查能在秒级发现大量接线错误。

### 第二级：编译检查（分钟级，不跑真机）

通过编译器生成可加载的 NPU 二进制。编译失败通常是 BD ID 超范围、lock 引用错误、路由不可达。编译成功说明硬件描述语法正确，但**不代表运行正确**：BD bank 规则、lock 配对错误、启动顺序竞态等问题可能要到运行时才暴露。

### 第三级：真机运行 + CPU reference 逐 lane 比对

这是最终验证。流程：

1. **准备 CPU oracle**：用 Python 完整模拟 NPU 的物理路径（包括 Q4NX 反量化、RMSNorm 的 bf16 舍入、RoPE 的具体精度、在线 attention 的 block 边界行为），产出 expected output。

2. **毒化关键 slot**：对 KV cache 中 current_token 对应的位置预填一个明显不可能的值（如 19.0 / -19.0）。运行后如果这个 slot 被正确覆盖了，说明 append 路径通了；如果还是毒值，说明 current K/V 写回失败。

```python
initial_k_cache = bf16_cache_payload(
    schedule,
    _poison_current_cache(inputs.k_cache, schedule.current_token, 19.0)
)
```

3. **分层验证输出**：先验 cache writeback（current K/V 是否正确写回），再验最终 output。每一层应该单独设置容差：

```python
# cache writeback: 只经过少量运算，容差应该很小
validate_cache("K", expected_k, got_k, tight_tolerance)
# final output: 经过多阶段近似，容差按完整数值路径设定
validate_output(expected_hidden, got_hidden, pipeline_tolerance)
```

容差分层的原因：cache writeback 只经过少量运算（norm + RoPE），误差小；最终 output 经过 Q4NX 量化、多个投影、attention merge、SwiGLU，误差会累积。容差只用于解释 bf16/Q4NX 的合理舍入，不用于掩盖系统性错误。

4. **先查 non-finite**：如果 output 里出现 NaN 或 Inf，优先报告第一个出现的 lane——这通常指向某个 accumulator 溢出或读到了未初始化数据。

5. **报告 lane 级定位**：不是笼统说"结果不对"，而是精确报告"lane 42: expected=1.234, got=5.678, abs_err=4.444"——直接定位到第几维出了问题。

### 路径闭环验证的思路

每个验证目标都应该是一条**完整的物理路径闭环**，而不是单个函数：

```
closed_loop_1: host hidden → c1r2 RMSNorm → main16 QKV → c1r3 norm+RoPE → KV writeback → scan → attention → O
closed_loop_2: attention + O → c1r2 residual + post RMSNorm → main16 up/gate → c6r2 SwiGLU
closed_loop_3: SwiGLU → main16 down → compact → c1r2 residual → output
```

如果 closed_loop_1 对了但 closed_loop_2 出错，问题就缩小到 O residual add、post RMSNorm 或 up/gate 投影之间。逐步增加闭环覆盖范围（从单 tile 到跨 tile 到全层），比只写孤立函数测试更适合这种空间数据流程序。

---

## 总结

1. Decode 慢在带宽——一层要读大量权重，却只服务一个 token
2. GPU 浪费带宽在中间结果往返和 kernel launch 上
3. XDNA 把整层融合成片上管道——中间结果永远不出芯片
4. 同一组 tile 按时间轮转做完多个投影——有限 tile 阵列覆盖完整层
5. 但这要求完全不同的编程方式：流式权重、BD ring 自转、online softmax、近似数学函数、packet 路由、record ABI、runtime patch
6. 代价是 debug 极难——很多错误只表现为 timeout 或数值错误

| | GPU decode | XDNA decode |
|---|---|---|
| 中间激活 | 每个 kernel 写回 HBM | 片上直接流向下一个 tile |
| 权重搬运 | 每个 kernel 各自读 | 一条流水线读一次 |
| 计算/搬运重叠 | warp 级 | tile 级 DMA/compute ping-pong |
| 调度开销 | 多个 kernel launch | 一次 NPU 提交 |
| Tile 复用 | N/A（每个 kernel 独立） | 16 tile 跑 7 个投影 |
| Softmax | 物化完整 score matrix | 16-token carrier + online merge |
| 编程模型 | 写 kernel 函数 | 描述一台数据流机器 |

Decode 的瓶颈常常是带宽。XDNA 通过全层融合、片上数据流和 tile 时间复用，减少中间结果往返和 host 调度开销。这不是单纯靠更高的频率或更宽的总线做到的，而是靠一种完全不同的编程模型：你不再只写一个个 kernel 让硬件执行，而是在编译时设计一台专用的数据流机器。运行时做的事情越少，数据就越接近"开闸后自然流过"。
