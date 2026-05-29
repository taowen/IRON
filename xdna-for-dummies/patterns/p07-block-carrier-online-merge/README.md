# Pattern 7: 大中间矩阵改成 Block Carrier + Online Merge

## 硬件问题

某些算子有巨大中间矩阵（如 attention score matrix: tokens × heads × context），完整物化会压垮 tile local memory 或片外带宽。

## 通用拆法

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

## 适用条件

- 跨 block 合并有稳定数学公式（online softmax、streaming max、running mean）
- carrier 能表达局部 block 的充分信息
- tail block 可以 mask
- running state 放得进 tile local memory

## 失败信号

- carrier 越做越大，接近原始中间矩阵 → pattern 失效
- merge 需要回看过去 block 的完整数据 → 不满足 online 条件
- tail token/padding 靠后处理修补
- fixed-point scale 没有和 reference 对齐

## 本例演示（简化版）

256 个 i32 分 4 个 block 找全局 max：

```
host ──[block0: 64dw]──► tile: find_block_max → carrier (max_val, block_id)
host ──[block1: 64dw]──► tile: find_block_max → carrier → online_merge(carrier, state)
host ──[block2: 64dw]──► tile: find_block_max → carrier → online_merge(carrier, state)
host ──[block3: 64dw]──► tile: → state ──► host (2 dw)
```

carrier = 2 dwords << 中间数据 256 dwords。

**注意**：这个 max demo 只展示了 carrier 的基本结构。真正的 attention carrier 需要支持 online softmax merge（running_max + running_sum + weighted_V 三元组），不是简单 max。

## 真机结果

```
  NPU time: 538.9 us
  expected: max=9999, block=2
  got:      max=9999, block=2
  PASS: online merge found global max without materializing full intermediate
```

## 去 qwen3-layer 抄哪里

**`qwen3-layer/edge_attention.cc`** — 真实 attention carrier + online merge：

Shape-A（block processor）每处理 16 个 token 产出 80-dword carrier：
```c
// carrier 结构:
// base[0x100]: 8 heads × 16 tokens 的 Q12 权重 (NOT weighted V!)
// scalar[0x40]: 8 × (block_max, block_sum) int32 pair

// 注意: carrier 只包含权重和统计量，不包含 weighted_V。
// V 是由 Shape-B 自己从 KV scan 流接收的——V 在 merge 侧消费。
// 如果 V 改在 producer 侧消费（Shape-A 直接算 weighted_V），
// 那 carrier 结构会变成 (block_max, block_sum, weighted_V[dim])，
// 大小完全不同。carrier 的边界取决于"V 在哪侧消费"。
```

Shape-B（merge processor）：接收 carrier + 对应 block 的 V → 本地做 weighted sum + online merge：
```c
// Shape-B 同时从两个源接收:
//   1. carrier (from Shape-A): weights + max/sum
//   2. V block (from KV scan): 16 tokens × head_dim
//
// running state: accum[head][dim], running_max[head], running_sum[head]
// 每收到一个 (carrier, V_block):
//   block_weighted_v = sum(carrier.weight[t] × V[t])  ← V 在 merge 侧消费
//   online merge: rescale accum, add block_weighted_v
```

相关文件：
- `qwen3-layer/cases/currentkv_kvscan_attention_kv16_generate.py` — Shape-A/B tile placement 和 carrier stream 配置
- `qwen3-layer/contract.py` — `SHAPE_CARRIER_DWORDS = (0x100 + 0x40) // 4 = 80`

## 运行

```bash
.venv/bin/python xdna-for-dummies/patterns/p07-block-carrier-online-merge/run_npu.py
```
