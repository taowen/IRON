# Pattern 1: 大算子拆成 Tile-Local 工作单元

## 硬件问题

Compute Tile 只有 ~64 KB 本地内存。一个 4096→4096 投影有 32 MB 权重。而且 tile 之间看不到彼此的内存——不能让一个 tile 看到完整问题。

## 通用拆法

每个 tile 固定负责：
- **固定输出切片**：N 行输出（不是整个输出维度）
- **固定输入窗口**：K 列的 activation chunk + 对应 weight chunk
- **固定 accumulator**：N 个元素，跨 chunk 保持
- **固定输出 ABI**：record = header + payload（下游根据 header 知道这块数据来自哪个 tile）

tile 不需要理解整个模型。它只知道：这次来了哪块输入、哪块权重、我负责哪几行、算完写成什么格式。

## 适用条件

- 输出可以按 tile 切分（tile 之间不需要共享 partial output）
- reduce 维度可以分块累加
- local memory 放得下当前 chunk + accumulator + output record
- 输入/权重流顺序能和 tile 内循环对齐

## 失败信号

- tile 里出现大量和全局阶段相关的 if/switch
- 一个 tile 要保存太多不同 phase 才用到的状态
- 输出切片需要频繁跨 tile 合并

## 本例演示

```
tile0(2,2): owns rows 0-3    tile1(3,2): owns rows 4-7
    ↑ weight chunks              ↑ weight chunks
    ↑ activation chunks          ↑ activation chunks (same data)
    ↓ 5-dword record             ↓ 5-dword record
    [header|4 payload]           [header|4 payload]
```

8×16 matvec 拆到 2 个 tile，每 tile 负责 4 行。每个 tile：
- 收 4 个 weight chunk（每个 4 rows × 4 cols = 16 dwords）
- 收 4 个 activation chunk（每个 4 dwords，和权重 chunk 对齐）
- 本地 accumulator 跨 4 个 chunk 持续累加
- 最后输出 1 个 record：header（tile_id 编码）+ 4 个 matvec 结果

## 真机结果

```
  2 tiles, each owns 4 output rows
  Each tile: weight chunk (16 dw) + act chunk (4 dw) → MAC → accumulate
  Output: 5-dword record per tile (1 header + 4 payload)

  NPU time: 595.0 us
  tile0 record: header=0x0000AB04 payload=[1496, 3672, 5848, 8024]
  tile1 record: header=0x0001AB04 payload=[10200, 12376, 14552, 16728]
  PASS: both tiles emitted correct records (header + matvec payload)
```

## 去 qwen3-layer 抄哪里

**`qwen3-layer/main_projection_q4nx.cc`** — main16 tile kernel，真实工作单元：

```
32 输出行 × 256 输入列 chunk × Q4NX weight chunk (1280 dwords)
→ 17-dword record (1 header + 16 payload = 32 bf16)
```

相关文件：
- `contract.py` — M_PER_TILE=32, K_CHUNK=256, RECORD_DWORDS=17
- `record_format.h` — header 编码（phase, group, row）
- `projection_schedule.py` — 各 phase 的 chunk/record 数量

## 运行

```bash
.venv/bin/python xdna-for-dummies/patterns/p01-tile-local-work-units/run_npu.py
```
