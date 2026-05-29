# Pattern 10: 融合不是一个大 Kernel，而是稳定 ABI 之间的片上交接

## 硬件问题

多个算子连续执行，中间激活很大。如果每个算子都写回主存再读回，带宽浪费严重。但把所有数学写进一个巨大 kernel 会让 tile 变成全能 station，程序复杂、debug 困难。

## 通用拆法

按**稳定 ABI** 融合——融合边界是数据格式契约，不是函数调用边界：

```
算子 A 输出固定格式 record [header | payload]
  header 编码: operator_id + payload_count
  payload 格式: 固定大小、固定 layout

算子 B 直接从片上接收 record（通过 aie.flow）
  解析 header 验证来源
  按 ABI 约定处理 payload
  输出自己的 record

中间 record 不回主存，host 不分配 intermediate BO
```

ABI 包含：record 大小、header 编码、payload layout、消费次数、lock/BD contract。

## 适用条件

- 中间数据大，下游马上消费
- producer/consumer 用固定 record 对接
- 每个 tile 角色清晰
- host 不需要观察中间值

## 失败信号

- 一个 tile 变成全能 station，保存太多阶段状态
- ABI 含混（header 格式随 call site 变化）
- 不同 layout 的算子被硬捏在一起，产生大量重排
- host 需要频繁进入层内 phase 调度

## 本例演示

```
host → tile_a(2,2) → [17-dword record: header 0xA010 | payload] → tile_b(2,3) → [record: header 0xB010 | payload] → host
                      ↑ ON-CHIP (no host BO for intermediate!)
```

- **tile_a** (`op_a_emit_record`): 接收 16 dword 输入，产出 17-dword record = `[header: (0xA0<<8)|16 | payload: input+10]`
- **tile_b** (`op_b_consume_record`): 接收 17-dword record，解析 header 验证 op_id，产出自己的 record = `[header: (0xB0<<8)|16 | payload*2 | source_op_id]`
- Host 只配置 input BD 和 output BD，中间的 17-dword record 从未在 host 侧分配

tile_b 能从 header 读出 `source_op_id = 0xA0`，证明 ABI 契约被正确传递。

## 真机结果

```
  host → tile_a(emit record: header+payload) → tile_b(consume record) → host
  ABI: 17-dword record = 1 header (op_id|count) + 16 payload

  NPU time: 628.7 us
  header: got=0x0000B010 expected=0x0000B010
  payload[0:4]: got=[22, 24, 26, 28] expected=[22, 24, 26, 28]
  source_op_id: got=0xA0 expected=0xA0
  PASS: record ABI handoff on-chip, header+payload verified
```

## 去 qwen3-layer 抄哪里

**`qwen3-layer/cases/currentkv_full_layer_q4nx_down_generate.py`** — 完整融合层的 ABI 链：

```
c1r2 full-vector replay (packet0, 2049 dw, header 0xC1000000|idx)
  → main16 Q/K/V compact (17-dw record → 65-dw column → 257-dw global)
  → c1r3 postprocess → attention (packet2, 2048 dw)
  → main16 O compact → c1r2 replay
  → main16 up/gate compact → c6r2 SwiGLU (payload half, 256 dw × 24)
  → c6r1 gather (packet1, 6144 dw) → main16 down compact
```

每个箭头是一个稳定 ABI 边界：大小固定、header 格式固定、lock contract 固定。

相关文件：
- `qwen3-layer/contract.py` — C1R2_PACKET_DWORDS=2049, COMPACT_PACKET_DWORDS=257, ATTENTION_PACKET_DWORDS=2048
- `qwen3-layer/compact_dataflow.py` — packet ID 分配（0=replay, 1=FFN, 2=attention）
- `qwen3-layer/record_format.h` — record header 编码
- `qwen3-layer/full_vector_station.cc` — replay record header = `0xC1000000 | replay_idx`

## 运行

```bash
.venv/bin/python xdna-for-dummies/patterns/p10-fusion-abi-handoff/run_npu.py
```
