# Pattern 2: 低 Batch Matvec 优先优化数据流，不是算术峰值

## 硬件问题

单 token decode 的 matvec 是 memory-bound：115 MB 权重从主存流入，只做一次向量乘。如果只盯 MAC 数量，会错过真正瓶颈——数据搬运。设计变量是"权重怎么流、activation 怎么复用"。

## 通用拆法

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

## 适用条件

- reduce 是线性累加
- 权重流量远大于激活流量
- weight stream 顺序能和 compute loop 对齐（chunk-major pack）
- tile 有足够 accumulator 保存当前要服务的 output block

## 失败信号

- 反量化结果被物化成完整矩阵
- activation 从片上返回后又被重复搬很多遍
- weight stream 顺序和 compute loop 不一致，导致额外 buffer
- 为减少几次 MAC 却引入更大的搬运

## 本例演示

block-major 调度 + weight ping-pong：

```
host weights ──[ping-pong stream]──► tile0(2,2): MAC 4 chunks → record
host weights ──[ping-pong stream]──► tile1(3,2): MAC 4 chunks → record
host activation ──[replay 4x]──► tile0
host activation ──[replay 4x]──► tile1
```

关键实现点：
- **Weight BD ring**: `^wt_ping (bd0, next=1) ↔ ^wt_pong (bd1, next=0)` — DMA 填 pong 时 core 算 ping
- **Weight chunk-major pack**: host 预 pack 成 `[chunk0_rows, chunk1_rows, ...]`，DMA 用简单 1D BD
- **Activation replay**: host 对每个 chunk 分别发一次 activation（block-major = 便宜时直接重发）

## 真机结果

```
  2 tiles, each owns 4 output rows
  Weight: streams through ping-pong (DMA fills next while core uses current)
  Activation: replayed from host (block-major = cheap to replay)

  NPU time: 617.5 us
  tile0: header=0x00004D04 payload=[1496, 3672, 5848, 8024]
  tile1: header=0x00014D04 payload=[10200, 12376, 14552, 16728]
  PASS
```

## 去 qwen3-layer 抄哪里

**`qwen3-layer/cases/currentkv_full_layer_q4nx_down_generate.py`** — 完整 7-phase 调度：

- **Q/K/V = block-major**：c1r2 发 packet0 full-vector replay 12 次 → c1r1 bridge 切成 256-bf16 chunk → main16 DMA0。activation 来自 replay（便宜），每个 output block 看完整 16 个 input chunk
- **O/down = chunk-major multi-block**：packet2/packet1 来自片上（不便重放），一个 activation chunk 同时服务 8 个 output block（通过 `q4nx_chunk_accum_block_slice_i32`）

注意：chunk-major 不是唯一正确方案。如果 source 侧有能力 replay（比如在 memtile 缓存一份 activation 后多次发出），block-major 同样合法。关键是 buffer/BD/lock credit 必须自洽——选 block-major 还是 chunk-major 取决于"activation 来源的 replay 成本"，不是哪个方案更高级。

相关文件：
- `qwen3-layer/weight_stream.py` — ping-pong BD ring（`PATCH_BUFFER_BF16`, `next_bd_id` 链）
- `qwen3-layer/projection_schedule.py` — `QKV_BODY_PLANS` vs `CURRENT_TAIL_PLANS` 派生
- `qwen3-layer/main_projection_q4nx.cc` — `q4nx_chunk_accum_single` (block-major) vs `q4nx_chunk_accum_block_slice_i32` (chunk-major)

## 运行

```bash
.venv/bin/python xdna-for-dummies/patterns/p02-dataflow-first-matvec/run_npu.py
```
