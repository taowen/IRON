# Pattern 9: 动态参数走 RTP/Descriptor Patch，不走 Tile 内复杂状态

## 硬件问题

数据流拓扑不变，但每次运行有少量参数变化（token position、block count、tail mask、buffer offset）。如果每次重编译 xclbin → 太慢（几十秒）。如果让 tile 内部维护大量 runtime state → 程序复杂且同步危险。

## 通用拆法

把动态性分层：

| 动态参数类型 | 处理方式 | 例子 |
|---|---|---|
| tile 需要读的标量 | RTP buffer + runtime-start lock | current token position |
| DMA 需要的地址/长度 | `npu.writebd` 的 buffer_offset/buffer_length | KV write offset |
| BD 的 iteration/repeat | `npu.writebd` 字段 | scan block count |
| instruction stream 常数 | binary patch design.bin | RTP value, iteration word |
| 拓扑/channel/BD 数量改变 | 重新编译 | 只有这种情况需要 |

关键时序：**RTP 必须在 core 读取之前写好**。用 runtime-start lock 门控：
```
runtime sequence: rtp_write → set_lock(runtime_start, 1)
core: acquire(runtime_start, 1) → 安全读 RTP
```

## 适用条件

- 动态参数只改变次数、offset、tail mask，不改变物理图
- base PDI 的 capacity 覆盖 target run
- patch sites 可精确定位并校验

## 失败信号

- tile 在 RTP 写入前启动 → 读到旧值（表现为 current token = 0）
- patch 后 instruction stream 没有和直接编译结果逐 word 比对
- target token 需要的 block 数超过 base capacity
- 为了避免 patch，把大量条件判断塞进 core loop

## 本例演示

tile 读 RTP 标量，加到输入数组上。编译一次，用不同 RTP 值运行两次：

```
Run 1: rtp_write(5)  → set_lock → core reads RTP=5  → output = input + 5
Run 2: rtp_write(100) → set_lock → core reads RTP=100 → output = input + 100
```

这只展示了 RTP 层。**真实系统还有 descriptor patch 和 instruction patch**（见下方 qwen3-layer 参考）。

## 真机结果

```
  --- Run with RTP = 5 ---
  NPU time: 560.8 us
  PASS: output = input + 5

  --- Run with RTP = 100 ---
  NPU time: 394.9 us
  PASS: output = input + 100
```

## 去 qwen3-layer 抄哪里

**`qwen3-layer/cases/currentkv_instruction_patch.py`** — 完整的三层 patch：

```python
# Layer 1: RTP (tile reads)
npu_rtp_write("post_current_token", 0, schedule.current_token)
npu_rtp_write("shape_a0_blocks", 0, schedule.kv_blocks)
npu_rtp_write("shape_a0_tail_tokens", 0, schedule.tail_tokens)

# Layer 2: Descriptor patch (DMA BD fields)
# current K/V write offset (position-dependent)
# scan iteration_size (block-count-dependent)
# push_queue repeat_count

# Layer 3: Instruction stream binary patch
# token1007 capacity PDI patched to token91:
# 修改 design.bin 中的 RTP value word、iteration word、repeat word
# 验证: patched stream == 重新编译的 token91 stream (逐 word 相等)
```

相关文件：
- `qwen3-layer/cases/currentkv_full_layer_q4nx_down_generate.py` — `npu_rtp_write` + `npu_set_lock` 用法
- `qwen3-layer/mlir_utils.py` — `npu_rtp_write` helper、RTP-before-lock 顺序检查

## 运行

```bash
.venv/bin/python xdna-for-dummies/patterns/p09-rtp-descriptor-patch/run_npu.py
```
