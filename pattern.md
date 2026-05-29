# XDNA 数据流 Pattern: 从具体算子提炼通用规律

这份文档的目标不是说明 `qwen3-layer` 这个模型怎么实现，而是从它的 projection、attention、KV cache、SwiGLU、full-vector station 等具体算子里抽出一般规律。

一个有用的 pattern 应该长这样:

```text
问题类型: 什么样的算子/数据流会遇到这个问题
通用解法: 在 XDNA 上通常怎么拆
适用条件: 什么前提下这个解法成立
失败信号: 什么时候说明你用错了
代码例子: qwen3-layer 中哪个地方体现了这个规律
```

`qwen3-layer` 只是案例。读者应该学习的是“遇到某类问题如何拆”，而不是背 Qwen3 的阶段顺序。

---

## Pattern 1: 大算子拆成 tile-local 工作单元

### 问题类型

一个算子的输入、权重或输出太大，单个 compute tile 放不下完整张量，也不应该让一个 tile 看到完整问题。

典型例子:

- 大矩阵乘或 matvec
- 大维度 projection
- 大向量上的逐块 reduce
- 多 head attention 的 per-head/per-window 计算

### 通用解法

把大算子拆成很多小的 tile-local 工作单元:

```text
每个 compute tile 只负责:
  固定输出切片
  固定输入窗口
  固定本地 accumulator
  固定输出 record
```

tile 不应该理解整个模型或整个图。它只需要知道:

```text
这次来了哪块输入
这次来了哪块权重/状态
我负责输出哪几行/哪几个 lane
算完以后写成什么 ABI
```

### 适用条件

- 输出可以切分，tile 之间不需要频繁共享 partial output。
- reduce 维度可以分块累加。
- 每个 tile 的 local memory 能放下当前 chunk、accumulator 和输出 record。
- 输入/权重流顺序能和 tile 内循环对齐。

### 失败信号

- tile 里出现大量和全局阶段相关的 `if/switch`。
- 一个 tile 要保存太多不同 call site 才用到的状态。
- 为了复用同一个 kernel，不断往参数里塞只给部分阶段使用的字段。
- 输出切片需要频繁跨 tile 合并。

### qwen3-layer 例子

`main_projection_q4nx.cc` 中 Main16 tile 的工作单元是:

```text
32 个输出行
256 个输入维度
128 dword activation chunk
1280 dword Q4NX weight chunk
17 dword output record
```

这不是“Qwen3 专属做法”，而是大 projection 在 XDNA 上的通用拆法: 输出按 tile 切，reduce 维度按 chunk 流入，本地累加，最后发小 record。

---

## Pattern 2: 低 batch matvec 优先优化数据流, 不是算术峰值

### 问题类型

单 token decode、batch 很小的 matvec 通常不是 compute-bound，而是 memory-bound。尤其是量化权重 projection:

```text
一条 activation 向量
大量权重
输出一条向量
```

如果只盯着 MAC 数量，会错过主要瓶颈。

### 通用解法

核心原则:

```text
权重流进来一次就尽量完成本地使用
activation 如果便宜就 replay
activation 如果来自片上结果就尽量复用
量化权重不要先反量化成大矩阵
```

常见两种调度:

```text
block-major:
  对一个输出 block 读完整输入 chunks
  简单, 适合 activation replay 成本低的情况

chunk-major:
  一个 activation chunk 进来
  同时服务多个输出 block
  适合 activation 来自片上 packet, 不想多次重放的情况
```

### 适用条件

- reduce 是线性累加。
- 权重流量大于中间激活流量。
- activation chunk 可以按固定粒度切。
- tile 有足够 accumulator 保存当前要同时服务的输出 block。

### 失败信号

- 为了减少几次 MAC，却引入了更大的片上/片外搬运。
- 反量化结果被物化成完整矩阵。
- activation 从片上返回后又被重复发很多遍。
- weight stream 顺序和 compute loop 顺序不一致，导致额外 buffer 或复杂控制。

### qwen3-layer 例子

`currentkv_full_layer_q4nx_down_generate.py` 里:

- Q/K/V 和 up/gate 使用 block-major。
- O/down 使用 chunk-major multi-block。

这反映的是一般规律: 同样是 projection，输入来源不同，最佳数据流也不同。pattern 不是“Qwen3 的 O/down 要这么写”，而是“当片上 activation 不便重放时，用 chunk-major 让一次输入服务多个输出块”。

---

## Pattern 3: 多 tile 输出先变成稳定 record, 再汇聚

### 问题类型

多个 compute tile 并发产出小块结果，下游需要看到一个更大的逻辑张量。

如果直接让每个 tile 写大 tensor 的任意位置，会带来:

- 多路写入地址复杂
- packet/BD 数量膨胀
- 下游难以判断边界
- debug 时不知道错在路由、布局还是数学

### 通用解法

定义一个稳定的小 record ABI:

```text
record = header + payload
header = 只放路由/阶段/块坐标等必要元数据
payload = 下游真正要消费的连续数据
```

然后做层级汇聚:

```text
tile record
  -> column compact
  -> global compact
  -> downstream packet/buffer
```

每一级只做一件事: 拼接、去掉重复 header、保持 payload 连续。

### 适用条件

- tile 输出粒度固定。
- 下游消费顺序固定。
- metadata 很小，payload 可以连续拼。
- 汇聚路径比直接全局 scatter 更简单。

### 失败信号

- header 越来越大，开始携带业务逻辑。
- 下游有的 call site 要 header，有的不要，导致 BD slice 到处特殊处理。
- record 顺序需要靠运行时条件判断修正。
- column/global compact 的规则在多个文件里各写一份。

### qwen3-layer 例子

`record_format.h` 定义 projection record header。  
`qkv_compact_reference.py` 和 `compact_dataflow.py` 体现了:

```text
17 dword tile record
65 dword column compact
257 dword global compact
```

这个例子要学的是“多 tile 小输出先收敛成稳定 record ABI”，不是 17/65/257 这些数字本身。

---

## Pattern 4: producer/consumer 用 ping-pong + credit lock 表达

### 问题类型

XDNA 内部数据流不是 CPU 每步调度。producer 和 consumer 速度不同，如果没有明确的信用机制，就会:

- producer 覆盖 consumer 还没读的数据
- consumer 读到未写满的 buffer
- 某个 DMA channel 等不到锁而 timeout

### 通用解法

用双缓冲和 empty/full lock:

```text
buffer: ping, pong
lock:   ping_empty, ping_full, pong_empty, pong_full

producer:
  acquire empty
  写 buffer
  release full

consumer:
  acquire full
  读 buffer
  release empty
```

如果一个 producer 服务多个 consumer，用 counting lock 表达 fanout credit。不要用隐式假设“大家会按顺序刚好读完”。

### 适用条件

- producer/consumer 是固定数据流。
- 每个 slot 的容量和消费次数可预先确定。
- DMA BD ring 可以长期循环。

### 失败信号

- 某个 BD 只有 acquire 没有 release，或反过来。
- lock 初值靠猜。
- 多个独立 producer 共用一个 channel，靠 packet 到达顺序碰运气。
- odd/even DMA channel 使用了错误 BD bank。

### qwen3-layer 例子

`weight_stream.py` 生成 row1 Q4NX weight stream 的 ping/pong patch buffer。  
`compact_dataflow.py` 生成 c1r1 shared activation bridge 的 ping/pong packet buffer。  
`mlir_utils.py` 检查 DMA lock 平衡和 memtile BD bank。

这些都是同一个通用解法: 不把 FIFO 当抽象概念，而是显式落成 buffer、BD ring 和 lock credit。

---

## Pattern 5: scarce channel 先分所有权, 再用 packet 复用逻辑流

### 问题类型

物理 DMA channel 很少，但逻辑数据流很多。常见冲突:

- 权重流和中间结果抢同一个 memtile S2MM
- 多类 packet 共用一条物理线
- current state 写回和普通 compact traffic packet ID 冲突

### 通用解法

分两层处理:

```text
channel ownership:
  规定每个物理 channel 属于哪类流量
  这是硬约束, 不能随便复用

packet ID:
  在已经拥有的物理路径上区分逻辑数据
  这是逻辑复用, 不能替代 channel ownership
```

先固定 channel ownership，再分配 packet ID。

### 适用条件

- 物理路径稳定。
- packet source/dest 明确。
- 接收端只需要筛选少数 packet 类型。
- 不同 packet 的生命周期不会互相破坏 lock credit。

### 失败信号

- 以为换一个 packet ID 就能解决 channel 冲突。
- 同一 channel 同时承担两个互不相关的 producer 顺序。
- packet ID 没有集中登记，新增路径靠记忆避冲突。
- debug 时分不清“包发错了”和“channel 被抢了”。

### qwen3-layer 例子

`physical_contract.py` 固定 row1:

```text
S2MM0..3: compact gather
S2MM4..5: weight ingress
MM2S0..3: weight fanout
MM2S5:    compact output
```

`dataflow.py` 和 `compact_dataflow.py` 再定义 packet0/1/2/8/9/10..15 的逻辑语义。

一般规律是: channel 是物理所有权，packet 是逻辑标签，两者不能混为一谈。

---

## Pattern 6: layout 是算子设计的一部分

### 问题类型

逻辑张量 layout 往往是给数学看的人类格式，但 DMA 需要的是:

- 连续地址
- 可表达的 stride
- 少量 BD
- 合法 field range
- 合法 bank/channel 组合

如果沿用逻辑 layout，可能数学很清楚，硬件却很难搬。

### 通用解法

从 DMA 消费单位反推物理 layout:

```text
下游一次读多少?
是否需要按 window 切?
是否能用一个 BD 描述?
是否能用 2D stride 描述?
是否值得提前 pack 一次?
```

热路径上的数据应该 pack 成硬件友好的布局。pack 的成本只付一次，后面换来更少 BD、更少 stride、更少中间 copy。

### 适用条件

- 这个 tensor 会被多次消费，或在关键路径上很大。
- layout 转换成本小于后续 DMA 简化收益。
- reference 能覆盖同样的物理 layout。

### 失败信号

- 为了保持逻辑 layout，用了很多小 BD。
- stride 字段接近或超过硬件范围。
- scan 需要复杂 gather。
- host reference 和 NPU layout 各写一套，容易漂移。

### qwen3-layer 例子

几个具体实例:

- Q payload 以 head window 切，方便 fanout 到 Shape-A。
- KV cache 在 NPU BO 中使用 block-major，方便 16-token block scan。
- current K/V stream 先 even 后 odd，方便两个 linked BD scatter。
- `qwen3_model.py` 把真实模型 Q4NX 权重 pack 成 Main16 weight stream，而不是让 NPU 按模型原始 tensor layout 随机访问。

这些例子共同说明: layout 不是收尾格式转换，而是数据流设计的核心。

---

## Pattern 7: 大中间矩阵改成 block carrier + online merge

### 问题类型

某些算子有巨大的中间矩阵，但最终只需要 reduce 后的结果。

典型例子:

- attention score matrix
- 分块 softmax
- streaming top-k / max / sum
- 长序列上的 weighted reduction

如果把完整中间矩阵落地，会压垮 local memory 或片外带宽。

### 通用解法

把中间矩阵按 block 流式处理:

```text
每个 block:
  计算局部结果
  提取最小充分统计量 carrier
  把 carrier 交给 reduce/merge 端

merge 端:
  保存 running state
  用 online 公式合并每个 block
  最后输出最终结果
```

carrier 应该比完整中间矩阵小很多，并且只包含跨 block 合并必须的信息。

### 适用条件

- 跨 block 合并有稳定数学公式。
- carrier 能表达局部 block 的充分信息。
- tail block 可以 mask。
- running state 能放在 tile local memory。

### 失败信号

- carrier 越做越大，接近原始中间矩阵。
- merge 需要回看过去 block 的完整数据。
- tail token/padding 靠后处理修补。
- fixed-point scale 没有和 reference 对齐。

### qwen3-layer 例子

`edge_attention.cc` 中 Shape-A 产生 80-dword carrier，Shape-B 做 online softmax merge。

当前实现是 kv16 fixed-point 物理 frontier，不是生产 Qwen3 attention 数值。  
但 pattern 是通用的: “大 score 矩阵不落地，用 block carrier 和 online merge 代替”。

---

## Pattern 8: 小追加和大扫描要分成两个数据流

### 问题类型

状态型算子经常同时有两种访问:

```text
append/update:
  每次只写很小的新状态
  地址随 runtime 参数变化

scan/read:
  每次顺序读大量历史状态
  适合大块 DMA
```

把这两种访问混在一个路径里，会让 descriptor 复杂、同步不清楚。

典型例子:

- KV cache 当前 token 写回 + 历史扫描
- streaming buffer append + window scan
- ring buffer 写指针更新 + bulk read

### 通用解法

拆成两个阶段:

```text
1. append/update 当前状态
2. sync 确认写入完成
3. bulk scan 历史状态
```

append 可以用少数 2D/scatter BD。  
scan 应该用顺序 BD、iterated BD 或 queue repeat。

### 适用条件

- 当前写入必须被本次 scan 看到。
- scan 的布局可以 block 化。
- append 地址能由 runtime 参数派生。
- descriptor 数量比状态长度小得多。

### 失败信号

- scan 偶尔读到旧值。
- 给每个 block 分配一个 BD，很快超过硬件上限。
- 只配置 iteration field，但 queue 没有 repeat。
- current slot 没有毒化测试，写没写进去无法区分。

### qwen3-layer 例子

`currentkv_kvscan_attention_kv16_generate.py` 中:

- current K/V 用 linked S2MM BD 写回。
- `npu.sync` 后再启动 KV scan。
- scan 用 `iteration_size` + `repeat_count` 复用 BD。
- reference 用 poisoned current slots 验证写回必须发生。

一般规律是: append 和 scan 是两种不同数据流，要显式分阶段。

---

## Pattern 9: 动态参数走 RTP/descriptor patch, 不走 tile 内复杂状态

### 问题类型

拓扑不变，但每次运行有少量参数变化:

- token position
- block count
- tail valid count
- buffer offset
- scan repeat count
- descriptor stride/iteration

如果每次重编译，太慢。  
如果让 tile 内部维护很多 runtime state，程序复杂且同步危险。

### 通用解法

把动态性分层:

```text
tile 需要读的值:
  RTP buffer
  runtime-start lock 保证 RTP 先写再执行

DMA 需要的值:
  runtime descriptor writebd/address_patch

instruction stream 中的常数字段:
  在 capacity 足够的前提下 patch design.bin

topology/channel/BD 数量改变:
  重新生成和编译
```

### 适用条件

- 动态参数只改变次数、offset、tail mask，不改变物理图。
- base PDI 的 capacity 覆盖 target run。
- patch sites 可精确定位并校验。

### 失败信号

- tile 在 RTP 写入前启动。
- patch 后 instruction stream 没有和直接编译结果比对过。
- target token 需要的 block 数超过 base capacity。
- 为了避免 patch，把大量条件判断塞进 core loop。

### qwen3-layer 例子

`currentkv_instruction_patch.py` patch:

- current-token RTP
- Shape block-count RTP
- Shape-A tail-token RTP
- current K/V write offset
- scan iteration word
- K/V scan queue repeat word

这里体现的通用规律是: 静态数据流机器通过少量 runtime 参数变成多次可复用的执行实例。

---

## Pattern 10: 融合不是一个大 kernel, 而是稳定 ABI 之间的片上交接

### 问题类型

多个算子连续执行，中间激活很大。如果每个算子都写回主存再读回，带宽和 latency 都会浪费。

典型例子:

- projection -> postprocess -> attention -> projection
- projection -> activation function -> projection
- norm/residual -> projection

### 通用解法

按稳定 ABI 融合:

```text
算子 A 输出稳定 packet/record
算子 B 直接从片上接收
中间不回主存
host 只在图边界配置和提交
```

融合边界应该是数据 ABI，而不是函数调用边界。  
不要为了“融合”把所有数学写进一个巨大的 kernel。

### 适用条件

- 中间数据大，且下游马上消费。
- producer 和 consumer 能用固定 packet/record 对接。
- 每个 tile 角色仍然清晰。
- host 不需要观察中间值。

### 失败信号

- 一个 tile 变成全能 station，保存太多阶段状态。
- 为了融合，ABI 变得含混。
- 不同 layout 的算子被硬捏在一起，产生大量重排。
- host 需要频繁进入层内 phase 调度。

### qwen3-layer 例子

`currentkv_full_layer_q4nx_down_generate.py` 把多个边界连成一次 NPU run:

```text
hidden replay -> Q/K/V -> attention -> O -> up/gate -> SwiGLU -> down -> output
```

要学习的是“稳定 packet/record ABI 之间片上交接”，不是 Qwen3 的具体层顺序。

---

## Pattern 11: 用集成边界验证数据流, 小单元测试不够

### 问题类型

XDNA 错误常常跨越多个层面:

```text
Python generator
MLIR-AIE
BD/lock/packet route
runtime descriptor
C++ kernel
host BO layout
```

单独测试一个 helper 函数，很难发现真实硬件路径上的 timeout、错包、错 layout。

### 通用解法

验证稳定的集成边界:

```text
一条完整物理路径
一个真实 DMA/packet/lock 闭环
一个 host reference 覆盖同样物理 layout
必要时读回中间 BO 定位失败点
```

结构检查和真机/构建检查都要有:

```text
结构检查:
  contract, dataflow, MLIR marker, BD/channel/packet 约束

集成运行:
  端到端输入输出
  关键中间状态读回
  poisoned data 区分路径失败和数学失败
```

### 适用条件

- 边界相对稳定，不会因为小重构频繁变化。
- reference 能准确模拟物理 layout。
- 检查能覆盖 descriptor、packet、lock、route。

### 失败信号

- 大量测试只验证 Python helper，没有验证生成 MLIR。
- reference 使用逻辑 layout，NPU 使用物理 layout。
- 只看最终输出，不读回关键状态，定位困难。
- 旧实验 case 保留太多，约束没有沉淀到共享检查。

### qwen3-layer 例子

`check_contract.py` 是结构集成检查。  
`run_npu.py --check-only` 和 `--build-only` 是可运行边界检查。  
`currentkv` reference 用 poisoned current-token slots 验证 writeback-before-scan。

这体现的通用规律是: XDNA 项目的测试边界应该跟硬件数据流边界一致。

---

## 如何从一个新算子推导 XDNA pattern

遇到新算子时，不要先问“有没有现成 kernel”。先按下面顺序拆问题。

### 1. 判断瓶颈

```text
是权重大?
是激活大?
是中间矩阵大?
是状态 scan 大?
是 host/NPU 往返多?
```

瓶颈不同，pattern 不同。

### 2. 定义 tile-local 工作单元

```text
一个 tile 负责哪些输出?
一次读多少输入?
需要保存哪些 accumulator?
输出最小 ABI 是什么?
```

如果 tile 工作单元说不清楚，后面的 DMA 和 lock 都会混乱。

### 3. 设计物理 layout

```text
下游一次连续读多少?
能否用一个 BD 描述?
需要 2D stride 吗?
是否要提前 pack?
```

layout 要从 DMA 消费方式反推。

### 4. 设计同步和所有权

```text
谁是 producer?
谁是 consumer?
buffer slot 有几个?
empty/full lock 初值是多少?
channel 是否已经有 owner?
packet ID 是否冲突?
```

先定物理所有权，再谈逻辑复用。

### 5. 处理动态参数

```text
这个参数改变 topology 吗?
只是改变 offset/count/tail 吗?
tile 要不要读它?
descriptor 要不要 patch 它?
```

能走 RTP/descriptor patch 的，就不要塞进 tile 内复杂控制状态。

### 6. 选择验证边界

```text
能否生成结构检查?
能否做 host physical reference?
能否读回关键状态?
能否用 poisoned input 验证先后关系?
```

验证边界越贴近真实数据流，越能发现真正的 XDNA bug。

---

## 速查表

| 问题类型 | 通用解法 | qwen3-layer 例子 |
|---|---|---|
| 算子太大, 单 tile 放不下 | tile-local 工作单元 | Main16 projection |
| 低 batch matvec memory-bound | stream weight, reuse activation, online dequant | Q4NX projection |
| 多 tile 小输出汇聚 | record -> column compact -> global compact | compact dataflow |
| producer/consumer 速率不同 | ping/pong buffer + credit lock | weight stream, bridge |
| channel 稀缺且流量多 | channel ownership + packet ID | row1 ownership, packet0/1/2/8/9 |
| logical layout 不适合 DMA | hardware-first physical layout | KV block-major, weight stream pack |
| 中间矩阵太大 | block carrier + online merge | Shape-A/B attention |
| 小追加 + 大扫描 | append/writeback 和 scan 分阶段 | current K/V + KV scan |
| runtime 参数变化 | RTP, descriptor patch, instruction patch | current token schedule |
| 中间激活回主存太贵 | 稳定 ABI 上融合 | full-layer physical frontier |
| 硬件路径难定位 | 大范围集成边界验证 | check_contract, active cases |

---

## 推荐阅读顺序

1. `contract.py`: 看 ABI 尺寸和阶段参数如何集中定义。
2. `dataflow.py`: 看通用数据流图如何描述 tile 角色和边。
3. `main_projection_q4nx.cc`: 看大 projection 如何被拆成 tile-local 工作。
4. `compact_dataflow.py`: 看多 tile 输出如何变成 record/compact。
5. `weight_stream.py`: 看 ping/pong + lock + BD ring 如何表达 FIFO。
6. `edge_attention.cc`: 看大中间矩阵如何变成 carrier + online merge。
7. `currentkv_kvscan_attention_kv16_generate.py`: 看 append/scan 和动态 schedule。
8. `qwen3_model.py`: 看真实模型 layout 如何 pack 成硬件 layout。

读这些文件时，不要问“Qwen3 为什么这样排阶段”。要问:

```text
这个文件解决的是哪类硬件问题?
它用了哪个通用 pattern?
这个 pattern 换到另一个算子还能不能成立?
成立需要哪些前提?
```
