# Pattern 8: 小追加和大扫描要分成两个数据流

## 硬件问题

状态型算子同时有两种访问：
- **append/update**：每次只写很小的新状态（如当前 token 的 K/V），地址随 runtime 参数变化
- **scan/read**：每次顺序读大量历史状态（如整个 KV cache），适合大块 DMA

把两种访问混在一个路径里，descriptor 复杂且同步不清楚。更严重：**scan 可能读到旧值**（append 还没完成就开始 scan）。

## 通用拆法

在 **runtime descriptor 层面**拆成两个阶段，用 `npu.sync` 强制顺序：

```
Phase 1: append (shim S2MM, buffer_offset = f(position))
  → push_queue → npu.sync (确认写入可见)

Phase 2: scan (shim MM2S, buffer_offset = 0, length = 全量)
  → push_queue → npu.sync (等 scan 结果回来)
```

关键：这不是"在同一个 core 里先写后读"——是两个 **runtime sequence 阶段**，descriptor 层面分离，中间有硬件 sync 保证可见性。

## 适用条件

- 当前写入必须被本次 scan 看到
- scan 布局可 block 化（iterated BD + repeat）
- append 地址由 runtime 参数派生
- descriptor 数量 << 状态长度

## 失败信号

- scan 偶尔读到旧值（没有 sync 保证可见性）
- 给每个 block 一个 BD → 超 shim 16 BD 上限
- 只有 iteration_size 没有 repeat_count → 只扫一个 segment
- current slot 没有毒化测试，写没写进去无法区分

## 本例演示

```
Phase 1 (append):
  writer tile(2,2) → MM2S → shim S2MM ch0
  BD: buffer_offset = APPEND_POSITION * 4 = 32 bytes (position-dependent!)
  → 4 dwords 写入 cache BO 的 [8:12] 位置

npu.sync (确认 append 完成)

Phase 2 (scan):
  shim MM2S ch0 → BD: buffer_offset=0, length=64 (整个 cache BO)
  → scanner tile(2,3) 收到完整 cache（含刚追加的数据）→ 求和输出
```

验证：cache 位置 [8:12] 预填 poison (0x7EADBEEF)，如果 scan 在 append 前执行，sum 会包含 poison → 测试失败。

## 真机结果

```
  Phase 1: writer tile → shim S2MM → cache BO at offset 32 bytes
  npu.sync (ensures append visible in BO)
  Phase 2: shim MM2S → entire cache BO (64 dw) → scanner tile → sum

  NPU time: 595.8 us
  cache[8:12] after: [9008, 9009, 9010, 9011]
  expected sum: 38076
  got sum:      38076
  PASS: Phase 1 append → sync → Phase 2 scan sees appended data
```

## 去 qwen3-layer 抄哪里

**`qwen3-layer/cases/currentkv_kvscan_attention_kv16_generate.py`**：

```python
# Phase 1: current K/V append (position-dependent linked BD scatter)
_push_current_cache_write(column, channel, schedule):
    # BD offset = f(current_token, head_layout)
    # even/odd linked S2MM BD
    # push_queue

# npu.sync (确认 K/V 写入 cache BO)
npu_sync(column, channel)

# Phase 2: KV scan (iterated BD + repeat)
_push_kv_scan(column, channel, schedule):
    # iteration_size = rounded_blocks
    # repeat_count = rounded_blocks - 1
    # 一个 BD 扫描整个 cache (包含刚写入的 current token)
```

相关文件：
- `qwen3-layer/cases/currentkv_instruction_patch.py` — patch append 的 write offset 和 scan 的 iteration/repeat
- `qwen3-layer/mlir_utils.py` — 验证 scan BD 的 iteration/repeat 配对

## 运行

```bash
.venv/bin/python xdna-for-dummies/patterns/p08-append-scan-separation/run_npu.py
```
