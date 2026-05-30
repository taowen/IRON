# 第四部分：调试

---

## 第二十三章：NPU 出错时怎么定位

### 调试前先换一个心智模型

在 CPU 程序里，bug 往往表现为异常、崩溃、错误返回值，调试方式是打断点、看调用栈、打印变量。

XDNA NPU 上不是这样。大多数错误只有两种外在表现：

1. **timeout**：某个 DMA 或 core 一直等不到 lock、stream、packet 或 host sync 完成。
2. **数据不对**：程序跑完了，但 output、cache、record、header 或中间 buffer 和 reference 不一致。

这不是因为硬件“难以调试”，而是因为你配置的不是一段顺序程序，而是一台静态数据流机器。机器启动后，几十个 DMA ring、core loop、lock 计数器、stream route 同时运行。没有一个全局调用栈能告诉你“卡在哪一行”。

所以调试的目标不是找“哪条语句错了”，而是找：

```
哪一个 producer 没有 release？
哪一个 consumer acquire 了不存在的信用？
哪条 stream 没有数据？
哪个 packet 被送到了错误的接收者？
哪个 BD 搬了错误的地址、长度或次数？
```

换句话说：**定位第一个断掉的交接边界**。

### 先判断是哪一类失败

不要一上来就改 kernel。先把失败分类：

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

这个分类很重要。timeout 和数值错误的排查路径完全不同。

### timeout：从“谁在等谁”开始

timeout 本质上是某个等待条件永远不满足。常见等待条件只有几类：

```
core 等 input_full lock
DMA 等 empty/full lock
DMA 等 stream 上有 packet
host 等 shim DMA sync
下游等上游 release 的 counting lock
```

排查 timeout 时，不要先猜数学计算。先写出这一条边的生产消费契约：

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

然后逐项核对。

#### 1. release count 和 acquire count 是否相等

单消费者 ping-pong 最简单：

```
producer release full +1
consumer acquire full -1
consumer release empty +1
producer acquire empty -1
```

一对多 fanout 必须用 counting lock：

```
producer 写满一个 shared buffer 后 release full +4
四个 consumer 各 acquire full -1
四个 consumer 读完后各 release empty +1
producer 下一次覆盖前 acquire empty -4
```

如果 producer 只 release +1，而有 4 个 consumer acquire，前三个或后三个一定会饿死。

#### 2. BD ring 是否真的会重复

很多 timeout 不是第一轮出错，而是第二轮、第三轮出错。

例如 O/down phase 的 main tile 按 output block 计算：

```
for n_block in 0..7:
    读完整 activation
    读这个 n_block 的 weight chunks
    输出一个 compact record
```

这意味着同一份 O activation 或 down activation 要被 replay 8 次。如果上游只发一次，main tile 算完第一个 output block 后就会继续等 DMA0，最后 timeout。

这种错误看起来像“main tile 挂住了”，根因却在上游 source-side replay 的 BD/lock 设计。

#### 3. channel 所有权有没有冲突

不要把 packet ID 当成物理 channel 的替代品。

packet ID 只是流上的标签，不能让两个逻辑流安全地共享同一个 DMA channel 的 BD ring 和 lock 状态。复杂设计里必须先分 channel 所有权：

```
row1 S2MM0..3  compact gather
row1 S2MM4..5  weight ingress
row1 MM2S0..3  weight fanout
row1 MM2S5     compact output
```

如果 weight ingress 和 compact gather 都抢 S2MM0/1，即使 packet ID 不同，也会在 BD ring、lock、buffer ownership 上冲突。

#### 4. BD bank 是否符合硬件规则

Memtile BD 有 bank 规则：

```
偶数 channel 使用 BD 0..23
奇数 channel 使用 BD 24..47
```

违反这个规则不一定在编译时报错，常见表现是运行时静默 timeout。所以 generator 里要把 channel → BD range 做成结构检查，不要靠手写编号。

#### 5. packet ID 是否唯一，route 是否匹配

packet flow 的调试顺序：

1. 发送端 BD 是否设置了正确 packet ID。
2. 接收端 packet route 是否订阅这个 ID。
3. 同一个物理区域内是否有 packet ID 重复。
4. packet payload length 是否和接收端 BD length 一致。

packet ID 重复时，数据可能被错误 tile 消费，原本的 consumer 就会 timeout。

### 数据错误：先看数据有没有到，再看值对不对

数据错误比 timeout 更适合分层验证。核心方法是 **poison + reference + 中间状态读回**。

#### 1. Poison：证明路径是否真的写过

把输出和关键中间 buffer 预填成明显的毒值：

```
0x7EADBEEF
0xDEADBEEF
0x7FC00000  // NaN-like pattern
```

运行后检查：

| 结果 | 含义 |
|------|------|
| 全部还是 poison | 这条路径完全没写到 |
| 前半段变了，后半段 poison | repeat count、tail、BD length 或 record 数量不够 |
| poison 出现在某些固定 stride 位置 | layout/stride 错 |
| poison 被 attention 读到 | append-before-scan 或 sync 顺序错 |

poison 的价值是把“计算错了”和“数据根本没到”分开。

#### 2. Reference 必须模拟物理 layout

NPU 上的数据不是逻辑 tensor layout。它经常是：

```
record: header + payload
column compact: 4 个 row record 拼成 65 dword
global compact: 4 个 column compact 拼成 257 dword
KV cache: block-major
Q4NX weight: 按 tile、row block、chunk 预 pack
```

如果 CPU reference 用逻辑 layout，而 NPU 用物理 layout，对比结果没有意义。正确的 reference 应该复刻：

- header 编码
- record 顺序
- bf16 round
- Q4NX pack/dequant 顺序
- KV cache block/tail layout
- packet payload 的实际 dword 顺序

这就是为什么集成测试的 reference 往往比数学公式复杂：它验证的不是抽象算子，而是硬件 ABI。

#### 3. Header 先于 payload 检查

record 类输出要先看 header：

```
header 正确，payload 错：
  producer tile 和 compact 路径大概率是对的，重点查 kernel 数值和 payload layout

header 错，payload 也错：
  先查 record ABI、compact 顺序、packet route、BD length

header 对，但 record 顺序错：
  查 row/column compact 的 stride 和 release 顺序
```

不要在 header 都错的时候调数学 kernel。那通常是在调错层。

### RTP 和 descriptor patch 的调试

动态 decode 会变：

```
current token
KV block count
tail token count
scan offset
queue repeat count
RTP value
```

但拓扑不应该每个 token 重编译。常见做法是编译一个最大容量 xclbin，然后 patch instruction stream 或 descriptor。

这里最容易出两类 bug。

#### 1. core 早于 RTP 启动

如果 core 没有 runtime-start lock 门控，它可能在 host 写 RTP 之前就读参数。表现是：

```
token0 正常
其他 token 看起来像 token0
current K/V 写回位置固定
RoPE position 不变
```

解决方式：所有读取 RTP 的 core 在入口先 acquire runtime-start lock；runtime sequence 写完 RTP 后再 release。

#### 2. patch 后没有和直接编译结果对比

patch 是危险操作。必须有一个检查：

```
compile token17 directly  → design-token17.bin
compile token127 then patch to token17 → patched.bin
compare relevant instruction fields
```

至少要验证：

- RTP write 值
- writebd length
- address offset
- iteration count
- queue repeat count
- sync channel/direction

否则 patch 可能“能跑”，但跑的是错误 token 的扫描范围。

### 数值错误：不要忽略非有限值

bf16/Q4NX/online softmax 路径中，数值错误常见来源有：

| 来源 | 表现 |
|------|------|
| accumulator 没清零 | 上一个 phase 污染下一个 phase |
| RMSNorm sumsq 溢出 | Q/K 变 NaN，attention 全坏 |
| bf16 round 不一致 | 小范围误差，通常固定在少数 lane |
| Q4NX 解包顺序错 | 大范围系统性偏差 |
| RoPE position 错 | token0 对，非零 token 错 |
| softmax max/sum merge 错 | 短 context 对，长 context 错 |
| V layout 错 | attention score 看似对，weighted V 错 |

遇到 NaN 或 inf，不要用更大的容差掩盖。先定位第一个产生非有限值的边界：

```
Q/K/V projection compact
→ c1r3 Q/K RMSNorm
→ RoPE payload
→ Shape-A QK score
→ Shape-B online softmax
→ weighted V
→ O phase input
```

每一层都可以临时把 payload 替换成已知有限值，确认 transport 和计算哪个先坏。调完必须删掉这些 probe，不能让 dummy producer 混进主路径。

### 推荐的三层验证流程

不要只写小单元测试。XDNA 的 bug 经常出现在跨层交界处，所以验证边界应该尽量接近真实物理路径。

### 单元测试、集成测试和探针分别做什么

NPU 项目里也需要测试，但不能照搬普通软件里的测试金字塔。普通单元测试擅长验证纯函数；XDNA 失败最多的地方却不是纯函数，而是：

```
Python generator 生成的 MLIR
BD/channel/lock/packet 的物理契约
C++ kernel 的本地 buffer ABI
host BO 的物理 layout
runtime sequence 的 descriptor patch
真实硬件上的 DMA/stream 时序
```

这些东西单独测一个 helper 函数很难覆盖。更合理的分工是：

| 类型 | 应该测什么 | 不应该承担什么 |
|------|------------|----------------|
| 单元测试 | 纯 layout 编码、header pack/unpack、Q4NX 解包、reference 小函数 | 证明硬件路径正确 |
| 集成测试 | 一个稳定物理边界的完整闭环：host → NPU → host，带 poison 和 reference | 覆盖每个内部 helper 的所有分支 |
| 探针 | 临时切断某条边，用已知 payload 定位 transport 还是 compute 错 | 长期留在主路径里当 fallback |

#### 单元测试：只测真正稳定的纯逻辑

单元测试适合放在变化少、没有硬件状态的地方：

```
record header 编码/解码
logical index → physical offset
Q4NX nibble unpack
bf16 pack/unpack
CPU reference 的小块数学
descriptor patch offset 解析
```

这些测试的价值是防止低级编码错误。但不要给每个 generator 内部函数都写单元测试。generator 的内部接口经常随着数据流设计变化而变化，测太细会让重构成本变高，而且仍然证明不了真实 NPU 能跑。

#### 集成测试：测试稳定边界

集成测试应该选择业务上稳定的边界，而不是代码上最小的函数边界。例如：

```
host hidden → c1r2 replay → main16 Q/K/V compact → c1r3 postprocess → host drain
current K/V writeback → sync → KV scan → attention output
attention packet2 → O phase → c1r2 post RMSNorm replay
SwiGLU packet1 → down phase → final compact/output
```

每个集成测试都应该有三件事：

1. **结构检查**：不跑硬件，先验证 MLIR 里 channel、BD、lock、packet、kernel link 符合契约。
2. **poison 检查**：关键输出和中间状态预填毒值，确认路径真的写过。
3. **物理 layout reference**：CPU reference 按 NPU 实际 record/cache/packet layout 对比。

这种测试比小单元测试慢，但覆盖的是 NPU 真正容易坏的地方。

#### 探针：用于二分定位，不是设计的一部分

探针是临时诊断代码。它的作用是把一条复杂路径切开，快速判断问题在哪一侧。

常用探针：

```
constant payload probe:
  把 producer 输出替换成固定有限值，验证 transport/route/consumer 是否能跑通

raw passthrough probe:
  跳过某个 postprocess，直接转发上游 payload，验证上游数据是否已经坏了

header-only probe:
  payload 不重要，只检查 record header 和顺序

poison probe:
  在 cache/output/compact buffer 放毒值，验证谁写了、谁没写

single-token probe:
  current-token 0 跑最短路径，先排除基本 append/handoff 问题

multi-token probe:
  current-token 17/127 跑 KV block、tail、RoPE、online merge
```

探针必须有退出标准：

```
probe 证明 transport 正常 → 删除 probe，把真实 compute 恢复
probe 证明 compute 异常 → 修 compute，然后用集成测试覆盖
probe 证明 layout 错 → 把 layout 约束沉淀到结构检查
probe 证明 lock/replay 错 → 把 release/acquire count 写进 contract 检查
```

不要把 probe 变成 permanent fallback。主路径里长期保留 dummy producer、raw passthrough、debug drain，会让后续每次读代码都分不清“生产逻辑”和“临时绕路”。

#### 推荐比例

一个实用的比例是：

```
少量单元测试：
  只覆盖稳定纯逻辑和容易写错的编码函数

更多结构检查：
  每次生成 MLIR 都跑，秒级发现物理契约错误

少而强的集成测试：
  选稳定数据流边界，真机跑，带 poison 和 physical reference

临时探针：
  定位问题时添加，根因确认后删除
```

这和普通软件项目不一样。XDNA 上最贵的 bug 不是“某个小函数返回值错”，而是“看起来所有局部都对，但真实硬件上某个 producer/consumer 边界没有闭合”。

#### Level 1：结构检查

不编译，只生成 MLIR 并检查字符串/结构/contract：

```bash
.venv/bin/python qwen3-layer/run_npu.py \
  --case currentkv-full-layer-q4nx-down-bridge \
  --check-only
```

结构检查应该覆盖：

- tile role 是否正确
- kernel object 是否链接到正确 tile
- row1 channel 所有权是否符合设计
- packet ID 是否唯一
- main16 DMA0/DMA1/MM2S record 是否齐全
- runtime sequence 是否写 RTP
- 禁止的旧路径是否没有出现

这一步秒级，应该频繁跑。

#### Level 2：编译检查

编译 MLIR、kernel、PDI、xclbin，但不运行：

```bash
.venv/bin/python qwen3-layer/run_npu.py \
  --case currentkv-full-layer-q4nx-down-bridge \
  --build-only
```

这一步验证：

- MLIR-AIE 能解析
- resource allocation 能过
- routing 能过
- kernel ELF 能链接
- xclbin 和 instruction stream 能生成

它不能证明运行正确，但能证明物理资源没有明显冲突。

#### Level 3：真机运行 + reference 对比

真机运行必须选有意义的 token：

```bash
# 最小闭环：只验证 append/current token 基础路径
.venv/bin/python qwen3-layer/run_npu.py \
  --case currentkv-full-layer-q4nx-down-bridge \
  --current-token 0

# 跨 KV block：验证 scan、tail、online merge
.venv/bin/python qwen3-layer/run_npu.py \
  --case currentkv-full-layer-q4nx-down-bridge \
  --current-token 17

# 默认长一些的集成边界
.venv/bin/python qwen3-layer/run_npu.py \
  --case currentkv-full-layer-q4nx-down-bridge \
  --current-token 127
```

token0 通过只能证明最短路径成立。token17、token127 这类非零 token 才能暴露：

- current K/V 写回位置
- RoPE position
- KV block scan
- tail mask
- descriptor patch
- online softmax merge

### 一个完整排查例子：O/down timeout

现象：

```
Q/K/V 能产出
attention 能返回 packet2
O phase 开始后 NPU timeout
```

错误直觉：

```
是不是 32 个 compute tile 放不下 full layer？
是不是 weight channel 冲突？
是不是 attention 没发回来？
```

实际根因：

```
main16 的 O/down body 是 record-major：
  O 有 8 个 output N-block
  down 也有 8 个 output N-block

每个 N-block 都要重新读完整 activation。
如果上游只发布一次 activation，main16 算完第一个 block 后继续等 DMA0，没人再发，于是 timeout。
```

正确设计：

```
source tile 持有完整 activation buffer
self-loop BD 发布同一份 activation 8 次
counting lock 每次给 bridge 一份信用
bridge 只做 packet-to-circuit 转发，不持有完整大 buffer
main16 按 record-major 消费 8 轮
```

这个例子说明：timeout 的表面位置在 main16，根因却是上游 replay count 和 phase schedule 不匹配。

### 一个完整排查例子：Q transport 看似坏了

现象：

```
真实 attention 输出 NaN
看起来像 c1r3 → Shape-A 的 Q payload 路径坏了
```

分层 probe：

1. 把 Q payload 全写 0：attention 有有限输出，说明 transport/route 能工作。
2. 把 Q payload 改成 raw Q body：输出仍有限，说明 main16 Q compact 和 c1r3→hub→Shape-A 路径能工作。
3. 恢复 Q RMSNorm + RoPE：NaN 复现，说明问题在 c1r3 数值处理。

根因：

```
Q RMSNorm 的 sumsq 在有限但很大的输入上溢出，scale 变成非有限值。
```

修复：

```
RMSNorm 先做 max-abs scaling：
  max_abs = max(abs(x))
  sumsq = Σ (x / max_abs)^2
  scale = rsqrt(sumsq / dim + eps / max_abs^2) / max_abs
```

这个例子说明：不要把所有 NaN 都归因于 stream/packet。先用有限 dummy payload 分离 transport 和 compute。

### 调试时该保留什么，删掉什么

应该保留：

- 结构检查函数
- physical contract 检查
- 毒化检查
- host reference 的物理 layout 模拟
- 可复现的集成 case
- 简短的根因注释

应该删掉：

- dummy producer
- raw payload passthrough probe
- 只验证 Python helper 的小测试
- 已被新路径替代的旧 case
- 为了兼容旧实验留下的大量 if/else

临时 probe 是必要的，但它们不是产品代码。调完以后要把结论沉淀成 contract 或 integration check，而不是把 probe 留在主路径里。

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

NPU 调试不是“试一个 patch 看看会不会过”。每次失败都要能回答：

```
失败发生在哪个物理边界？
这个边界的 producer/consumer 契约是什么？
哪个字段违反了契约？
为什么这个修复能防止同类问题再次发生？
```

如果回答不了，就说明还没有找到根因。
