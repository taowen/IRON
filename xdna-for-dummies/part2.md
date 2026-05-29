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
