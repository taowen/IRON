<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Performance Comparison Failures

Used during SageAttention bring-up.

## Case 1: Scalar Dequant Erases QK Speedup

### Symptom

The first runnable SageAttention path was correct enough to compare, but it was
slower than bf16 MHA:

```text
SageAttention Latency (us): 3045.2
MHA Latency (us): 446.2
Speedup: 0.147x
```

### Cause

The QK stage used `int8 x int8 -> int32`, then converted every int32 score to
bf16 logits with a scalar loop. The scalar dequant loop erased the intended QK
speedup.

### Fix Used

Vectorize the int32-to-bf16 dequant loop:

```cpp
constexpr int vec_len = 16;
const auto scale = aie::broadcast<float, vec_len>((float)dequant_scale);
for (int i = 0; i < DIM_M * DIM_N; i += vec_len) {
    auto qk_i32 = aie::load_v<vec_len>(scratch_i32 + i);
    auto qk_f32 = aie::to_float<float>(qk_i32, 0);
    auto scaled = aie::mul(qk_f32, scale);
    aie::store_v(logits_out + i, scaled.to_vector<bfloat16>());
}
```

## Case 2: Fusing QK and Softmax Was Correct but Slower

### Symptom

The fused QK+softmax worker passed numerical verification after fixing the
layout bug, but longer performance probes showed it was slower than MHA:

```text
64  sage 333.3988  mha 295.6772  speedup 0.887x
128 sage 542.2411  mha 387.9595  speedup 0.715x
256 sage 1242.9985 mha 689.3278  speedup 0.555x
```

### Cause

The fusion removed the `memA -> outA` ObjectFIFO transfer, but it forced the QK
kernel to convert the blocked accumulator layout into row-major logits itself.
That row-major writeback cost more than the saved FIFO stage.

### Fix Used

Revert to explicit stages:

```text
QK Worker -> ObjectFIFO forward layout conversion -> softmax Worker -> PV Worker
```

Then parallelize query blocks across two columns. This keeps the layout
conversion in DMA and exposes more parallelism. After runtime scale metadata
was added, the QK-to-softmax FIFO carries int32 scores and softmax performs the
dequant step.

## Case 3: Shape and Sample Count Matter

### Symptom

At `seq_len=128`, SageAttention and the one-pipeline MHA baseline were close
enough that a short five-iteration average could pass or fail depending on a
small outlier:

```text
SageAttention Latency (us): 387.2
MHA Latency (us): 385.3
Speedup: 0.995x
```

### Cause

For short sequences, QK is not dominant enough. Runtime overhead, DMA setup,
softmax, and PV hide much of the int8 QK benefit. A short timing sample also
makes the assertion too sensitive.

### Fix Used

The performance test now uses `seq_len=256`, five warmups, and thirty timed
iterations. With runtime scale metadata enabled, three pytest iterations
produced:

```text
SageAttention Latency (us): 667.5
MHA Latency (us): 685.6
Speedup: 1.027x

SageAttention Latency (us): 622.4
MHA Latency (us): 727.8
Speedup: 1.169x

SageAttention Latency (us): 641.8
MHA Latency (us): 665.1
Speedup: 1.036x
```

The compile-time scale prototype was faster, but it was not a real runtime
operator. Runtime scale metadata is the more correct design point even though
the current int32 score FIFO leaves less speedup.

The benchmark command was:

```bash
source /opt/xilinx/xrt/setup.sh
source .venv/bin/activate
pytest iron/operators/sage_attention/test.py --iterations 3 -s -v
```

## Case 4: Fewer Host Dispatches Did Not Improve Qwen3 Decode Yet

### Symptom

A two-layer persistent Qwen3 chunk reduced the number of full-layer NPU calls,
but did not reduce steady-state decode time.

Measured on the same prompt, runtime-packed weights, and `max_new_tokens=3`:

```text
chunk=1 token 1: npu_layer_time_us_total=251371.797 decode_s=0.259349
chunk=1 token 2: npu_layer_time_us_total=248143.551 decode_s=0.254117

chunk=2 token 1: npu_layer_time_us_total=254985.202 decode_s=0.259366
chunk=2 token 2: npu_layer_time_us_total=261342.450 decode_s=0.266101
```

### Cause

The bottleneck is still NPU-side layer work, not Python dispatch. The two-layer
graph adds hidden feedback/final routing and emits two rounds of DMA tasks
inside one runtime sequence. That saves host calls, but the saved host overhead
is currently too small to beat the extra NPU/dataflow cost.

### Fix Used

Keep the two-layer path only as a correctness-validated chunking checkpoint and
avoid extra setup overhead:

```text
do not compile the single-layer op for chunk=2 when the model has an even layer count
do not allocate duplicate single-layer weight/cache XRT buffers for chunk=2
use weight_pair and cache_pair BOs so preflight stays at five runtime memrefs
```

Accepted correctness evidence:

```text
two-layer-full-layer: two_layer_hidden_errors=0
generate --fast-generate --layer-chunk-size 2: token_match=True
```

Current conclusion:

```text
Layer chunking is structurally useful for reducing host dispatch count, but it
is not yet a throughput win. The next performance work should reduce NPU work
inside a layer, especially debug-free/full-output-free single-layer dispatch,
weight/cache DMA volume, or a deeper persistent token loop.
```

### Follow-up: N-Layer Final-Only Output Is Correct But Only Slightly Faster

The next probe removed the debug drains from the full-layer graph and exposed a
single `n-layer-final-only` path. It drains only `final_hidden[1024]` plus the
inout KV cache current-position updates.

Correctness command:

```bash
source /opt/xilinx/xrt/setup.sh
source .venv/bin/activate
python iron/applications/qwen3_0_6b/persistent/main.py \
  --stage n-layer-final-only \
  --layer-chunk-size 4 \
  --verify \
  --build-dir build_qwen3_persistent_n_layer_4 \
  --prompt 'Count from one to five.' \
  --raw-prompt
```

Accepted evidence:

```text
chunk=1 preflight: compute_cores=19 max_dma_tasks_per_fifo=1
chunk=2 preflight: compute_cores=21 max_dma_tasks_per_fifo=2
chunk=4 preflight: compute_cores=21 max_dma_tasks_per_fifo=4
chunk=4 verify: chunk_hidden_errors=0
chunk=4 verify: layer0..3 keys/values cache current errors=0
```

Fast-generate comparison on the same prompt and `max_new_tokens=3`:

```text
n-layer-final-only chunk=1 token 1:
npu_layer_time_us_total=245563.972 decode_s=0.251359
n-layer-final-only chunk=1 token 2:
npu_layer_time_us_total=242661.658 decode_s=0.246359

n-layer-final-only chunk=2 token 1:
npu_layer_time_us_total=250323.856 decode_s=0.255839
n-layer-final-only chunk=2 token 2:
npu_layer_time_us_total=251366.210 decode_s=0.255056

n-layer-final-only chunk=4 token 1:
npu_layer_time_us_total=249153.122 decode_s=0.253014
n-layer-final-only chunk=4 token 2:
npu_layer_time_us_total=249887.935 decode_s=0.252974
```

Conclusion:

```text
Removing debug output is worth keeping because it reduces output BO size and
keeps internal boundaries on chip. Generalizing to n-layer chunks also removes
the single/two-layer code split. It is still not the main bottleneck: the next
real performance target is reducing repeated per-layer weight/cache DMA and
external-kernel scalar work, not only host output volume or dispatch count.
```

### Follow-up: Prefix KV Blocks Are A Real Decode Win

Symptom:

```text
decode position 6 still streamed all max_seq_len=256 K/V cache rows
```

Cause:

```text
The attention score/context workers and Runtime.fill TAPs used
max_seq_len // cache_block_seq blocks unconditionally. For early decode
positions, most transferred K/V blocks were future rows that the causal mask
would ignore.
```

Fix used:

```text
For final-output-only decode, process active_cache_blocks =
ceil((position + 1) / cache_block_seq). Keep debug checkpoint streams full-size
so diagnostic output layouts do not change. Current K/V writeback is drained
once per chunk with a chunk-shaped TAP instead of one drain per layer.
```

Accepted evidence on the same prompt:

```text
n-layer-final-only chunk=4 verify:
npu_time_us=29134.841
chunk_hidden_errors=0
layer0..3 keys/values cache current errors=0

n-layer-final-only chunk=4 verify at decode_position=70:
npu_time_us=39632.917
chunk_hidden_errors=0
layer0..3 keys/values cache current errors=0

generate --fast-generate --layer-chunk-size 4 --max-new-tokens 3:
token 1: token_match=True npu_layer_time_us_total=184098.641 decode_s=0.187769
token 2: token_match=True npu_layer_time_us_total=184764.213 decode_s=0.187131
```

Follow-up fix:

```text
chunk=8 no longer fails at aiecc after repacking chunk weights segment-major.
It still timed out at runtime when historical K/V cache was filled once per
layer. Grouping cache fill and current writeback by four layers reduced the
real graph to max_dma_tasks_per_fifo=2 and made chunk=8 generate match the CPU
reference tokens.
```

Second follow-up:

```text
chunk=28 failed when cache fill/writeback stayed grouped by four layers:
@qwen3_rc_k_rope_0 and @qwen3_rc_v_0 exhausted BD IDs during NPU lowering.
Switching layer_iterations > 8 to one full-depth cache TAP made chunk=28
compile with max_dma_tasks_per_fifo=1 and run as one NPU dispatch per decoded
token.

default prompt: token_match=True, new_text='Paris'
Fibonacci raw prompt: token_match=True through decode positions 17..20,
new_text=' 5, 8'
npu_layer_time_us_total: about 190-200ms
```
