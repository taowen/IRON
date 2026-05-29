# Pattern 6: Layout 是算子设计的一部分

## 硬件问题

逻辑 tensor layout 往往是人类友好的格式（token-major、行主序、自然索引），但 DMA 需要的是：
- 连续地址（一个 1D BD 就能描述）
- 可表达的 stride（不超过硬件 field 范围）
- 少量 BD（shim 只有 16 个 slot）

如果沿用逻辑 layout，可能数学很清楚，硬件却很难搬——需要 2D stride、多个 BD、或复杂 gather。

## 通用拆法

从 DMA 消费单位反推物理 layout：

```
下游一次读多少？→ 连续放在一起
是否需要按 window 切？→ 按 window 排列
能否用一个 BD 描述？→ 不能就提前 pack
```

热路径数据 pack 成硬件友好布局。pack 成本只付一次（host 侧或编译时），后面换来更少 BD、更少 stride、更少 copy。

## 适用条件

- 这个 tensor 被多次消费，或在关键路径上很大
- layout 转换成本 < 后续 DMA 简化收益
- host reference 能覆盖同样物理 layout

## 失败信号

- 为了保持逻辑 layout，用了很多小 BD
- stride 字段接近或超过硬件范围
- scan 需要复杂 gather
- host reference 和 NPU layout 各写一套，容易漂移

## 本例演示

两个 tile 需要 even 和 odd 索引的元素——在自然布局中交错存放（非连续）：

```
自然布局 [0,1,2,3,4,5,...]:
  tile0 需要 [0,2,4,6,...] → 非连续！每隔一个取一个
  tile1 需要 [1,3,5,7,...] → 非连续！

预 pack 为 [all_evens | all_odds]:
  tile0 取前 16 dwords → 连续！1 个 BD
  tile1 取后 16 dwords → 连续！1 个 BD
```

## 真机结果

```
  NPU time: 509.2 us
  tile0 (evens sum): expected=240 got=240
  tile1 (odds sum):  expected=256 got=256
  PASS: pre-packed even/odd layout → simple 1D BDs, correct result
```

## 去 qwen3-layer 抄哪里

**`qwen3-layer/qwen3_model.py`** — Q4NX 权重从模型原始 tensor 预 pack 成 main16 chunk stream layout：
```python
# 模型权重: [output_dim, input_dim] 逻辑布局
# NPU 需要: [chunk0_32rows, chunk1_32rows, ...] 按 tile 消费顺序排列
# pack 一次, main16 用简单 1D BD 就能流式读取
```

**`qwen3-layer/cases/currentkv_kvscan_attention_kv16_generate.py`** — KV cache block-major layout：
```
逻辑: token-major K[token][head][dim]
物理: block-major K[block][head][token_in_block][dim]
原因: scan 按 16-token block 读, block-major 让每个 block 连续
```

**Current K/V even/odd scatter**（本例直接对应）：
```
current K/V 写入时先 even heads 后 odd heads
让 linked BD scatter 只需要两个 BD（even/odd）而不是 8 个（per-head）
```

## 运行

```bash
.venv/bin/python xdna-for-dummies/patterns/p06-layout-is-design/run_npu.py
```
