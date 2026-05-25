# 如何正确地在 IRON 上写 Megakernel

前一篇文章 (`how-to-write-megakernel-in-IRON.md`) 记录了手写 megakernel 的完整过程。那篇文章的技术细节是准确的，但它展示的是**一种不应该模仿的写法**。本文解释为什么，以及正确的做法是什么。

---

## 前文的核心错误：把编译器的活儿手动做了

前文描述的 28 层 × 13 phase megakernel 手动完成了以下工作：

1. 手算每个 phase 的 packet 布局（5 处必须同步修改）
2. 手写 Worker body 的 phase 调用顺序
3. 手配 TAP 的 4-元组（offset/sizes/strides）
4. 手管 L1 容量预算（packet 30KB + shared 8KB + state 2KB...）
5. 手排所有权重到 per-lane L3 连续缓冲
6. 手算 acquire/release 配对

这些工作**已经有现成的抽象可以代劳**——`FusedMLIROperator` + runlist。前文之所以没用它，不是因为它不够好，而是当时 `FusedMLIROperator` 还未支持 full ELF 编译模式和 external buffer 类型。现在它支持了。

---

## 正确写法：声明 runlist，让编译器处理底层

以 `torch2iron/src/models/exported_llama3/generated/decode_fused.py` 为例，完整的 16 层 Llama decode megakernel 只需要：

### 第一步：实例化参数化算子

```python
gemv_attn_query_op = GEMV(
    M=config.n_heads * config.head_dim,
    K=config.emb_dim,
    num_aie_columns=8,
    tile_size_input=4,
    tile_size_output=config.head_dim // 2,
    context=elf_ctx,
)

rms_norm_op = RMSNorm(
    size=config.emb_dim,
    num_aie_columns=1,
    num_channels=1,
    tile_size=config.emb_dim,
    weighted=True,
    context=elf_ctx,
)

llama_chunked_attention_op = LlamaChunkedAttention(
    max_seq_len=prompt_len,
    num_kv_groups=config.n_kv_groups,
    q_heads_per_group=config.n_heads // config.n_kv_groups,
    head_dim=config.head_dim,
    chunk_size=DECODE_ATTN_CHUNK_SIZE,
    context=elf_ctx,
)
# ... 其他算子类似
```

每个算子是独立的、可单独测试的 Python 对象。它自带 MLIR 生成、kernel 编译、arg spec 声明。

### 第二步：写 runlist

```python
runlist = []
for layer_idx in range(config.n_layers):
    runlist.extend([
        (rms_norm_op, "x", f"W_norm1_{layer_idx}", "x_norm"),
        (gemv_attn_query_op, f"W_attn_query_{layer_idx}", "x_norm", "queries"),
        (gemv_attn_key_value_op, f"W_attn_key_{layer_idx}", "x_norm", "keys"),
        (gemv_attn_key_value_op, f"W_attn_value_{layer_idx}", "x_norm", "values"),
        (rope_queries_op, "queries", "rope_angles", "queries"),
        (rope_keys_op, "keys", "rope_angles", "keys"),
        (copy_present_kv_op, "keys", f"present_keys_{layer_idx}"),
        (copy_present_kv_op, "values", f"present_values_{layer_idx}"),
        (strided_copy_packet_key_op, "keys", f"packet_cache_{layer_idx}"),
        (strided_copy_packet_value_op, "values", f"packet_cache_{layer_idx}"),
        (llama_chunked_attention_op, "queries", f"packet_cache_{layer_idx}", "attn_context"),
        (gemv_attn_output_op, f"W_attn_output_decode_{layer_idx}", "attn_context", "attn_output"),
        (residual_add_op, "x", "attn_output", "x"),
        (rms_norm_op, "x", f"W_norm2_{layer_idx}", "x_norm"),
        (gemv_ffn_up_gate_op, f"W_ffn_gate_{layer_idx}", "x_norm", "ffn_gate"),
        (gemv_ffn_up_gate_op, f"W_ffn_up_{layer_idx}", "x_norm", "ffn_up"),
        (silu_ffn_op, "ffn_gate", "ffn_gate"),
        (eltwise_mul_ffn_op, "ffn_gate", "ffn_up", "ffn_hidden"),
        (gemv_ffn_down_op, f"W_ffn_down_{layer_idx}", "ffn_hidden", "ffn_output"),
        (residual_add_op, "x", "ffn_output", "x"),
    ])
runlist += [
    (rms_norm_op, "x", "W_final_norm", "hidden_out"),
    (gemv_out_head_op, "W_out_head", "hidden_out", "logits"),
]
```

这就是整个 decode 模型的计算图。**没有手写的 packet offset，没有 TAP，没有 Worker body，没有 acquire/release 配对。**

### 第三步：声明 buffer 分类

```python
fused_op = FusedMLIROperator(
    "fused_op",
    runlist,
    input_args=["x", "rope_angles"],          # 每 token 变化的输入
    output_args=["logits", *present_kv_names], # dispatch 后读取的输出
    buffer_sizes={...},                         # 显式指定的 buffer 大小
    external_args={
        "weight": weight_names,                # 权重（只传一次）
        "lm_head": ["W_out_head"],             # 大权重单独放
        "kv_cache": packet_cache_names,        # KV cache（增量同步）
    },
    context=elf_ctx,
).compile()
```

`FusedMLIROperator` 接收 runlist 后自动完成：

| 自动完成的工作 | 前文手动做的对应物 |
|---|---|
| `_calculate_buffer_layout()`: 对所有 named buffer 做 liveness analysis，分配 L3 偏移 | 手算 packet offset，5 处同步修改 |
| 将每个 op 的 MLIR 串联，buffer 名替换为偏移 | 手写 Worker body 的 phase 调用序列 |
| 编译为单个 ELF | 手配 TAP + DMA 编排 |
| buffer_sizes 自动推导 | 手管 L1 容量预算 |

### 第四步：运行时只需要做极少的事

```python
# decode 一个 token
fused.get_buffer("rope_angles").torch_view()[:] = angles_slice.flatten()
fused.get_buffer("x").torch_view()[:] = embedding.flatten()
fused.input_buffer.to("cpu")
fused()
# 读出 present K/V，追加到 cache 的正确位置
append_decode_kv_cache(config, fused, max_seq_len, current_slot, position)
```

---

## 前文错误一：手动编排 Phase = 人肉充当编译器

前文的 Worker body 硬编码了 13 个 phase 的 kernel 调用顺序：

```python
# 前文写法（错误示范）
q_shard_kernel(shared, packet, ...)
packet_fifo.release(1)
packet = packet_fifo.acquire(1)
projection_kernel(shared, packet, ...)  # K
packet_fifo.release(1)
packet = packet_fifo.acquire(1)
projection_kernel(shared, packet, ...)  # V
shared_fifo.release(1)
...
```

正确写法中，这些全部由 runlist 的条目顺序隐式定义：

```python
# 正确写法
(gemv_attn_query_op, "W_attn_query_0", "x_norm", "queries"),
(gemv_attn_key_value_op, "W_attn_key_0", "x_norm", "keys"),
(gemv_attn_key_value_op, "W_attn_value_0", "x_norm", "values"),
```

好处：
- 增删 op 只改 runlist 一处，不影响其他任何文件
- buffer 名是数据流的边——编译器自动推导生命周期和复用
- 同一个 op 实例可复用（不同层只换 weight buffer 名）

---

## 前文错误二：数据打包全部手算

前文最易出错的环节：

> 修改任何 phase 的 offset 布局时，以下 5 处必须一起改：
> 1. C++ kernel 的指针偏移
> 2. runner.py 的 packet 填充代码
> 3. runner.py 的 reference 计算
> 4. ops.py 的 minimum_packet_elements 计算
> 5. README 的 phase 文档

正确写法中**没有这个问题**。每个算子通过 `get_arg_spec()` 声明自己的输入/输出形状：

```python
class LlamaChunkedAttention(MLIROperator):
    def get_arg_spec(self):
        return [
            AIERuntimeArgSpec("in", (self.num_kv_groups, self.q_heads_per_group, self.head_dim)),
            AIERuntimeArgSpec("in", (self.packed_elements,)),
            AIERuntimeArgSpec("out", (self.num_kv_groups, self.q_heads_per_group, self.head_dim)),
        ]
```

`FusedMLIROperator._calculate_buffer_layout()` 读取所有 op 的 arg spec，自动计算偏移。内部表示和物理布局的映射是编译器的内部实现细节，不暴露给使用者。

---

## 前文错误三：KV Cache 动态性用 ELF Patching 解决

前文（以及 IRON 原始的 llama 示例）用 magic number + 二进制 patch 来处理 KV cache 的写入位置：

```python
# 前文写法（错误示范）
strided_copy_cache_magic = 0xDEADBEE0
# ... 编译后找到 magic 的位置 ...
patched_elf_data = ops.fused_elf_data.copy()
patch_elf(patched_elf_data, patches)
ops.fused.reload_elf(patched_elf_data)  # 每 token 重建 xrt::hw_context！
```

这是一个反模式：
- **每 token 重建 `xrt::hw_context`**，延迟约数毫秒
- ELF 二进制搜索 magic number，脆弱且不可维护
- 依赖未文档化的编译器行为（magic 在 ELF 中的位置）

### 正确方案：Packet Cache + Host-side Append

torch2iron 使用 **packet cache** 布局：KV cache 按 chunk 组织，与 `LlamaChunkedAttention` op 的消费格式完全一致。

```
packet_cache 内存布局（per layer）:
┌─── group 0 ──────────────────────────────────────────────────────────────┐
│ chunk 0: [K_rows(64×head_dim) | V_rows(64×head_dim) | mask(64)]         │
│ chunk 1: [K_rows(64×head_dim) | V_rows(64×head_dim) | mask(64)]         │
│ ...                                                                      │
│ chunk N: [K_rows(64×head_dim) | V_rows(64×head_dim) | mask(64)]         │
├─── group 1 ──────────────────────────────────────────────────────────────┤
│ (same layout)                                                            │
└──────────────────────────────────────────────────────────────────────────┘
```

NPU 内部用 `StridedCopy` 把当前 token 的 K/V 写入 `current_cache_slot`（编译时固定的位置）。Host 在 dispatch 后把 present K/V 从 output 读出，追加到 cache 的实际位置：

```python
def append_decode_kv_cache(config, fused, max_seq_len, current_cache_slot, position):
    if position == current_cache_slot:
        return  # NPU 已经写在正确位置了
    for layer_idx in range(config.n_layers):
        present_key = fused.get_buffer(f"present_keys_{layer_idx}").data
        present_value = fused.get_buffer(f"present_values_{layer_idx}").data
        packet_cache = fused.get_buffer(f"packet_cache_{layer_idx}")
        # 写入 position 对应的 chunk/row
        sync_decode_packet_cache_slot(config, max_seq_len, packet_cache,
                                      present_key, present_value, position)
```

关键区别：
- **ELF 完全静态**——编译一次，永不 patch，永不 reload
- `StridedCopy` 的 output_offset 编译时固定（指向 `current_cache_slot`）
- Host 用增量 sync 把 present K/V 写到 cache 的正确位置（只 sync 变化的几十字节）
- 下次 dispatch 时 NPU 读取完整 packet_cache（包含历史 + 新数据）

---

## 前文错误四：没有利用 external_args 分离 buffer 生命周期

前文把所有东西塞进三类 buffer：input（每 token 变）、output、scratch（NPU 常驻）。这导致：
- 权重在 scratch 里，和中间 buffer 混在一起，编译器无法做跨 variant 共享
- KV cache 在 scratch 里，不能增量同步

torch2iron 的 `FusedMLIROperator` 支持 `external_args`：

```python
external_args={
    "weight": weight_names,      # ~65MB，只传一次，多个 decode variant 共享同一份
    "lm_head": ["W_out_head"],   # ~256KB，单独放避免 weight buffer 过大
    "kv_cache": packet_cache_names,  # 按需增量 sync 单个 slot
}
```

好处：
- **权重共享**：多个 decode variant（seq64, seq128, seq256...）使用同一块物理 weight buffer
- **增量同步**：`sync_decode_packet_range()` 只 sync 变化的 K/V slot（几十字节），不用搬整个 cache
- **Buffer 替换**：`fused.replace_buffer("kv_cache", prefill_fused.kv_cache_buffer)` 可以零拷贝切换 variant

---

## 前文错误五：固定一个 max_seq_len 然后全量计算

前文用一个固定的 `max_seq_len=256`，在 position 1 时 attention 仍然读取全部 256 行。torch2iron 使用 **decode variants**：

```python
DECODE_VARIANT_SEQ_LENS = (64, 128, 256, 512, 1024, 2048)
```

运行时根据当前 token 位置选择最小够用的 variant：

```python
def ensure_decode_variant(self, num_valid_tokens):
    target_seq_len = self._select_decode_variant_seq_len(num_valid_tokens + 1)
    target = self.aie_ops.decode.variants[target_seq_len]
    if self.active_decode_variant is target:
        return target
    # 迁移 KV cache 到新 variant
    copy_decode_packet_cache_tokens(config, src_fused, ..., target_fused, ...)
    self.active_decode_variant = target
```

- Position 1-63 → 用 seq64 ELF（attention 只读 64 行）
- Position 64-127 → 迁移到 seq128 ELF
- Position 128-255 → 迁移到 seq256 ELF
- ...

计算浪费从 "97% at position 1" 降低到 "最多 50%（刚进入下一个 bucket 时）"。

所有 variant 共享同一份 weight buffer，只有 KV cache 和 scratch 是独立的。ELF 切换时只需迁移已有的 cache token（几 KB），不需要重新加载权重（几十 MB）。

---

## 正确写法 vs 前文写法的对比总结

| 维度 | 前文（手写 megakernel） | 正确写法（FusedMLIROperator + runlist） |
|------|----------------------|---------------------------------------|
| Phase 编排 | Worker body 硬编码 364 行 | runlist 20 行/层 × N 层 |
| Buffer 布局 | 手算 offset，5 处同步 | `_calculate_buffer_layout()` 自动 |
| KV cache 动态性 | ELF patching + reload | host-side append + 增量 sync |
| 算子开发 | 修改 kernel.cc 后需同步 runner/packer/reference | 算子独立编译测试，runlist 即插即用 |
| 多 seq_len 支持 | 编译一个固定大小 | 多 variant ELF + 运行时切换 |
| 权重管理 | 在 scratch 里 | external_args 支持跨 variant 共享 |
| 调试 | checksum 二分 364 个 phase | 每个算子可单独测试 |
| 代码生成 | 全手写 | `torch.export` → codegen → runlist |

---

## 什么时候仍然需要手写 megakernel？

前文的手写方法在以下场景仍有价值：

1. **极限性能优化**：当你需要跨 phase 共享 L1 slot、精确控制 DMA double-buffering depth、或在一个 kernel 内做多次 acquire/release（streaming model），runlist 抽象会限制你
2. **非标准拓扑**：当数据流不是简单的线性 pipeline（比如需要环形通信、或 tile 间的直连 FIFO），FusedMLIROperator 目前不支持
3. **新算子原型**：在编写新的 `MLIROperator` 子类之前，先用手写 design.py 验证硬件行为

但对于**模型级别的 megakernel 组装**——把现有算子串成完整推理——正确做法永远是 runlist + FusedMLIROperator。手写 megakernel 应该只存在于单个算子的内部实现中（比如 `LlamaChunkedAttention` 内部的 worker/fifo/TAP 编排），而不是模型级别。

---

## 设计原则总结

1. **算子是边界**。每个算子封装自己的 tile 拓扑、DMA 配置、kernel 代码。对外只暴露 `(input_shapes) → (output_shapes)` 的 arg spec。

2. **Runlist 是计算图**。数据流通过 named buffer 连接算子，顺序即执行顺序。增删重排只改 runlist，不影响算子本身。

3. **编译器管布局**。liveness analysis 决定 buffer 复用，`_calculate_buffer_layout()` 分配偏移。开发者不碰物理地址。

4. **Buffer 类型分离生命周期**。`input_args`（每 token 变）、`output_args`（每 token 读）、`external_args`（权重/cache，按需同步）、`scratch`（纯中间结果，NPU 常驻）。

5. **ELF 保持静态**。所有动态行为（位置、cache 长度）通过数据值传入，不改图拓扑。多 variant ELF 覆盖不同 seq_len bucket，运行时切换比 patching 更快更干净。

6. **增量同步代替全量传输**。`sync_decode_packet_range()` 只 DMA 变化的字节——一个 K/V slot 只有 `head_dim × 2 bytes`，不是整个 cache。

7. **可以自动生成**。runlist 的结构性足够强，可以从 `torch.export` 的 FX Graph 自动 codegen（见 `torch2iron.export.codegen`）。手写 megakernel 无法自动化。
