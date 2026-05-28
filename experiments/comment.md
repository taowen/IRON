# qwen3-dataflow.md 审查意见

审查对象：`experiments/qwen3-dataflow.md`。

总体结论：这份文档的高层方向基本对，但现在把两类东西混在了一起：

```text
已经由 MyLM 逆向确认的 physical/control ABI:
  tile 分工、packet/drop_header、BD 长度、row1 compact、c1r1/c6r1/c6r2/c1r2
  物理职责、main16 replay count

IRON 自己定义的 value ABI:
  payload lane order、head/window order、carrier scalar 语义、c1r2 full-vector
  value layout、attention-to-O chunk order、up/gate N-block 配对
```

如果把本文档当成 `IRON-ABI-v0` 设计说明，大部分自定义 value layout 是可以自洽的。
如果把它当成 MyLM 精确复刻说明，下面这些地方会导致数值错误、调度错误，甚至在硬件
上无法跑通。

## 必须修正

### 1. Q/K/V compact record 数写错

位置：`906..919`。

文档写：

```text
Q：每 tile 12 条 = 8 N-blocks（K/V 复用同一族，各占其中 2 N-blocks）
```

这是错误的。正确理解是：

```text
Q/K/V family 总共每 tile 12 条:
  Q = 8 N-blocks
  K = 2 N-blocks
  V = 2 N-blocks
```

不能说 Q 自己有 12 条。否则 scheduler 会多发 Q 记录，后续 c1r3 收到的 Q/K/V
边界也会错。

### 2. `c1r2` 的 `+12/+48/+1` 被写成了 chunk replay

位置：`639..648`、`929..932`。

文档写 `+12 = Q/K/V 输入 chunk replay`、`+48 = up/gate 输入 chunk replay`。
这不对。exp91 的结论是：

```text
+12 = Q/K/V full-vector packet0 replay count
+48 = up/gate full-vector packet0 replay count
+1  = final hidden-output boundary transfer
```

每一次 replay 是一个 `2049-dword` manual-header full-vector packet：

```text
1 control/header dword + 2048 dword payload = 4096 bf16 hidden vector
```

这个 full-vector packet 后续才经 c1r1 bridge 切成 main16 需要的 activation chunks。
如果实现成 `+12/+48` 个 256-bf16 chunk，会少发一个数量级，main16 无法完成投影。

### 3. `c6r2` up/gate 半区顺序有两处写反

位置：`364..368`、`675..678`、`1040`。

文档有些地方已经写对了：

```text
input[0x000..0x3ff] = up
input[0x400..0x7ff] = gate
```

但 `c6r2 接收 512-dword 输入（512 bf16 gate slice + 512 bf16 up slice）`
这类描述仍然是旧说法。exp95/exp96 的结论是物理低半区先放 up，高半区放 gate：

```text
c6r2 input low half  = up payload
c6r2 input high half = gate payload
SwiGLU = SiLU(gate) * up
```

这里不是措辞问题。c6r2 kernel 的 load path 先把第二半区送入 SiLU/table-index
activation，再用 `dj0=-0x400` 取第一半区做乘法。

### 4. `up/gate` 的 adjacent-pair 调度和“物理 order up 后 gate”有冲突

位置：`244..247`、`661..667`、`700..702`、`951`。

文档同时说：

```text
物理 projection 顺序是 Q, K, V, O, up, gate, down
```

又说：

```text
for slice:
  produce up[slice]
  produce gate[slice]
  c6r2 立即消费这一对
```

这两句话不能同时作为 MyLM 已知事实。exp96 只证明了：

```text
两个 257-dword c1r1 compact packet 经 c6r2 drop_header 后形成一个 512-dword input
低半区语义是 up，高半区语义是 gate
```

尚未证明 MyLM 的 48 个 up/gate N-block 是严格 `up0, gate0, up1, gate1...`，还是
`up0..up23, gate0..gate23` 加某个小重排。

如果硬件实际按 `up0..up23` 后接 `gate0..gate23` 直接喂 c6r2，那么 c6r2 会把
`up0+up1` 当成一组 SwiGLU 输入，这是错误的；c6r2 本地也没有足够空间缓存 24 个
up slice 等 gate 全部到来。因此 adjacent pair 只能作为 IRON 自定义调度，不能写成
MyLM 已确认事实。

### 5. Shape-A/B carrier 的 scalar 语义被写成了确定事实

位置：`287..307`、`318..334`、`550..553`、`941..948`。

文档定义：

```text
base[h][t] = exp(score[h][t] - block_max[h])
scalar[2h+0] = block_max[h]
scalar[2h+1] = block_sum[h]
```

这可以作为 IRON-ABI-v0 的自定义 value ABI，但不是 MyLM 已确认事实。当前 MyLM
逆向只确认：

```text
base[0x100]   = eight 0x20-byte records
scalar[0x40] = online-softmax 相关 normalization/scale state
Shape-B 按 0x00,0x40,0xc0,0x80 顺序消费 head-pair block
```

`scalar` 是否就是 `(block_max, block_sum)`、lane 顺序如何、`base` 内 head/token
顺序如何，都还没有完全校准。复用 MyLM binary kernel 时不能按本文档这个定义直接
喂数据。

### 6. “局部 softmax”表述容易导向错误实现

位置：`287..294`、`539..553`。

Shape-A 可以为 16-token block 产生：

```text
exp(score - block_max)
block_max
block_sum
```

然后由 Shape-B 做跨 block 的 online-softmax merge。它不能把每个 block 独立归一化
成最终 softmax 再交给 Shape-B；那样跨 block 的概率质量会错。

建议把“block 内做局部 softmax”改成“block 内计算 unnormalized exp weights 和
block-level max/sum statistics”。

### 7. `c1r2` “一份 2048-dword full-vector buffer”说法过强

位置：`639..641`、`651..655`。

c1r2 的物理角色已经确认：hidden/RMSNorm/residual/final-output full-vector station。
但内部 register-level ping/pong pointer、value layout、残差缓存方式仍未校准。

它至少需要跨 phase 保存：

```text
输入 hidden residual，用于 O 后残差
O/down projection compact 汇聚结果
O 后 residual/RMSNorm 结果，用于 FFN 和最终 down residual
```

所以“内部维护一份 full-vector buffer”不能作为可实现的物理约束。正确说法应是：

```text
c1r2 对外暴露 2048-dword full-vector packet ABI；
内部 ping/pong/value layout 仍是 calibration 项。
```

### 8. “main tile 不需要知道阶段，只是换权重”不准确

位置：`241..247`、`778..779`。

矩阵乘主体可以复用同一段 projection kernel，但 main16 dispatcher/output path 不是
完全 phase-blind。已确认的 scheduler-critical 控制包括：

```text
Q/K/V family: 0x1 header/control, 12 records
O family:     0x4 header/control, 8 records
up/gate:      0x8 header/control, 48 records
down:         0x4 header/control, 8 records
```

Edge 侧的 c1r2/c6r2/c6r1 接收路径依赖这些 phase family、record count 和 body order。
如果实现时真的只“换权重流”，不切换 output record family 和 downstream phase，
数据会走错消费者。

## 需要降级为设计假设

### 9. main16 payload lane order 和 tile order

位置：`908..922`。

文档定义：

```text
payload_dword[i].lo16 = output[2i]
payload_dword[i].hi16 = output[2i+1]
tile 顺序 = (c2..c5) × (r2..r5)
```

这可以作为 IRON-ABI-v0，但 MyLM 的 exact lane order 仍是 remaining work。exp96
确认了 16 records -> 257-dword compact packet 的形状，不等于确认了所有 bf16 lane
在 MyLM kernel 内的语义顺序。

### 10. attention output 到 O 的 head-pair chunk order

位置：`582..597`、`948..950`。

文档定义：

```text
O_chunk[c] = heads 2c 和 2c+1 的所有 128 dim
```

这是一个合理的 IRON layout，尺寸也匹配：

```text
2 heads * 128 bf16 = 256 bf16 = 128 dword
```

但它仍是 value ABI 设计，不是 MyLM binary ABI 已完全校准的事实。若复用 MyLM O
kernel，需要重新校准 Shape-B return window 的 exact head order 和 c1r1 bridge
切片顺序。

### 11. Q/K/V current writeback 和 history scan 的时序需要写清楚

位置：`508..531`、`840..842`。

decode attention 应该覆盖当前 token 自己，即 scan 范围通常是 `[0..current]`。
文档现在说当前 K/V “供将来的 token 使用”，又说历史 scan 读“之前所有 token”，容易
被实现成 attention 不包含当前 token。

如果 MyLM/IRON 的策略是先 packet14/15 写当前 K/V，再 rounded scan 包含当前 token，
文档应明确写出这个顺序和同步约束。否则算法上会漏掉 self token。

### 12. 性能描述写成了已实现事实

位置：`473..474`、`778..783`、`794..810`。

“计算和传输完全重叠”“利用率接近 100%”“计算核心永远不需要空等”都只能作为目标。
当前实验已经证明 small chunk-ring 的 same-channel BD/lock phase 一旦不精确就会
timeout。没有生产 Q4NX kernel 和完整 BD/lock phase 复刻之前，不能把这些写成事实。

建议改成：

```text
目标是通过 ping-pong BD ring 让 DMA/compute overlap；
是否完全隐藏传输延迟取决于 kernel latency、BD phase、lock order 和 stream route。
```

## 小问题

1. 位置 `409`：权重顺序写成 `Q/K/V/O/gate/up/down`，后文又写物理顺序
   `Q/K/V/O/up/gate/down`。建议全文统一为物理顺序 `up` before `gate`，并在算法
   公式处单独说明 `SwiGLU = SiLU(gate) * up`。
2. 位置 `366`、`676`、`1040`：所有 “gate/up slice” 字样都应改成 “up/gate slice”
   或明确写成 “low half up, high half gate”。
3. 位置 `59..63`：“所有边界与 lane 顺序在本文档后续章节给出明确定义”建议改成
   “physical/control boundary 已按 MyLM 逆向收敛，value lane order 在 IRON-ABI-v0
   中给出自定义定义”。否则会误导读者以为 lane order 已由 MyLM 完整证明。

## 可以保留的核心结论

以下部分没有发现物理上不可行的问题：

```text
main16 复用跑 Q/K/V/O/up/gate/down
c1r3 做 current Q/K norm + RoPE，K/V packet14/15 写回
c0/c7 两侧做 KV rounded scan，K 给 Shape-A，V 给 Shape-B
Shape-A/B 用 neighbor-local carrier 交接 attention state
c6r1 packet2 -> c1r1 bridge -> main16 O phase
c1r2 承担 full-vector RMSNorm/residual/final-output
c6r2 做 512-dword up+gate input -> 256-dword SwiGLU slice
c6r1 packet0 -> c1r1 bridge -> main16 down phase
host boundary 不暴露 O/residual/FFN temporary BO
```

但这些结论要和上面的 value ABI 假设分开写，避免把“我们自己定义的自洽 layout”
误认为“MyLM 已经证明的 exact binary ABI”。
