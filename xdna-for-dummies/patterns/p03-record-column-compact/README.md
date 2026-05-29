# Pattern 3: 多 Tile 输出先变成稳定 Record，再层级汇聚

## 硬件问题

多个 Compute Tile 并发产出小块结果。如果让每个 tile 直接写大 tensor 的任意位置：
- 16 路写入地址复杂，packet/BD 数量膨胀
- 下游难以判断数据边界
- debug 时分不清错在路由、布局还是数学

## 通用拆法

定义稳定 record ABI，on-chip 层级汇聚：

```
tile record (17 dw) = 1 header + 16 payload
  ↓ per-row S2MM channel 收入 memtile
column compact (65 dw) = row0 full record + row1..3 payload only (header stripped)
  ↓ 4 column 汇聚
global compact (257 dw) = 1 header + 256 payload = 512 bf16
```

header 去重：row0 保留完整 record（含 header），其他 row 只保留 payload（通过 BD offset 跳过 header）。

## 适用条件

- tile 输出粒度固定
- 下游消费顺序固定
- metadata（header）很小，payload 可以连续拼
- 汇聚路径比全局 scatter 更简单

## 失败信号

- header 越来越大，开始携带业务逻辑
- BD block 有多个 release → 编译报错（硬约束：一个 BD block 最多一个 release）
- record 顺序需要运行时条件判断修正
- compact 规则在多个文件各写一份

## 本例演示

on-chip memtile compact（真实跑通）：

```
prod0(2,2) ──► memtile S2MM ch0 → compact[0:17]  (full record: header + payload)
prod1(2,3) ──► memtile S2MM ch1 → compact[17:33] (payload only: header stripped!)
                                      ↓
                memtile MM2S ch5 → shim → host (33 dwords compact)
```

关键实现点：
- **Per-row S2MM channel**：row0→ch0 (BD 0, even bank), row1→ch1 (BD 24, odd bank)
- **Header stripping**：prod1 BD 从 offset=1 开始发（跳过 header dword）
- **Counting lock drain**：`mt_full` 初值=0，每个 row 写完 release 1，drain `AcquireGreaterEqual(2)` 等两个都完成
- **BD block 单 release 约束**：drain 只 release `drain_token`（不能同时 release row0_empty 和 row1_empty）

## 真机结果

```
  expected: [16, 0, 1, 2, ..., 15, 100, 101, ..., 115]
             ↑header  ↑row0 payload   ↑row1 payload (no header!)
  got:      [16, 0, 1, 2, ..., 15, 100, 101, ..., 115]
  PASS: 2 producer records compacted into single output by memtile
```

## 去 qwen3-layer 抄哪里

**`qwen3-layer/compact_dataflow.py`** — 完整的三级 compact（4 rows × 4 columns → global）：

```python
# Column level: per-row S2MM with phase-aware BD ring
COLUMN_RECEIVE_BDS = ((0,1,2,3,4,5), (24,25,26,27,28,29), ...)
# Header stripping: row0 gets full RECORD_DWORDS, row1+ gets RECORD_PAYLOAD_DWORDS
_segment(row):
    if row == 0: return 0, RECORD_DWORDS
    else: return RECORD_DWORDS + (row-1) * RECORD_PAYLOAD_DWORDS, RECORD_PAYLOAD_DWORDS
# Counting drain: AcquireGreaterEqual(ROWS_PER_COLUMN) on full lock
```

相关文件：
- `qwen3-layer/record_format.h` — header 编码
- `qwen3-layer/qkv_compact_reference.py` — compact 的 CPU 参考
- `recipes/hierarchical-record-compaction/` — 更完整的多列 compact 真机例子

## 运行

```bash
.venv/bin/python xdna-for-dummies/patterns/p03-record-column-compact/run_npu.py
```
