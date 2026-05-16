<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Qwen3 Megakernel Symptom Lookup

Each entry starts from an observed symptom and records only diagnostics that
were used in this repository.

## Symptom Index

Environment, artifact cache, and host ABI:

- [Full-ELF Runtime APIs Are Missing](#full-elf-runtime-apis-are-missing)
- [Clean Graph Edits Appear To Do Nothing](#clean-graph-edits-appear-to-do-nothing)
- [Host Buffer Assignment TypeError](#host-buffer-assignment-typeerror)
- [Runtime Segfaults In XRT BO Validation](#runtime-segfaults-in-xrt-bo-validation)
- [First Iteration Is Zero, Later Iterations Improve](#first-iteration-is-zero-later-iterations-improve)

Full-ELF layout and numeric boundary:

- [One-Step Decode Token Is Wrong](#one-step-decode-token-is-wrong)
- [Layout-Only Changes Move The Token](#layout-only-changes-move-the-token)
- [QKV Numeric Errors Appear After Runtime Packing](#qkv-numeric-errors-appear-after-runtime-packing)
- [RoPE Outputs Fail Under GEMV Tolerance](#rope-outputs-fail-under-gemv-tolerance)

Persistent IRON dataflow and resources:

- [Persistent QKV Cannot Place On 8 Columns](#persistent-qkv-cannot-place-on-8-columns)
- [Persistent QKV Exceeds Output DMA Channels](#persistent-qkv-exceeds-output-dma-channels)
- [Static MLIR Shows One FIFO Drained Twice](#static-mlir-shows-one-fifo-drained-twice)
- [Attention Score Worker Exceeds Input DMA Channels](#attention-score-worker-exceeds-input-dma-channels)
- [K Cache Matrix Does Not Fit In L1](#k-cache-matrix-does-not-fit-in-l1)
- [Debug Pass-Through FIFO Exceeds L1](#debug-pass-through-fifo-exceeds-l1)
- [K Cache Block DMA Exhausts BD IDs](#k-cache-block-dma-exhausts-bd-ids)
- [Multidimensional TAP Is Legal But NPU BD Rejects It](#multidimensional-tap-is-legal-but-npu-bd-rejects-it)
- [Runtime Start Does Not Create A Phase Barrier](#runtime-start-does-not-create-a-phase-barrier)
- [Attention Scores Overwrite The First GQA Head](#attention-scores-overwrite-the-first-gqa-head)

## Full-ELF Runtime APIs Are Missing

Symptom:

```text
compile-only succeeds, but full-ELF runtime cannot execute
```

Diagnostic:

```bash
source /opt/xilinx/xrt/setup.sh
.venv/bin/python - <<'PY'
import sys
import pyxrt
print(sys.version)
print(pyxrt.__file__)
for name in ["elf", "ext", "hw_context", "kernel", "bo", "device"]:
    print(name, hasattr(pyxrt, name))
PY
```

Evidence found:

```text
Python 3.12 pyxrt: elf False, ext False
Python 3.14 pyxrt: elf True, ext True
```

Root cause:

```text
The active Python environment did not expose the XRT full-ELF APIs needed by
FusedFullELFCallable.
```

Fix:

```text
Rebuild .venv with Python 3.14, then reinstall requirements.txt and
requirements_examples.txt.
```

Recheck:

```bash
source /opt/xilinx/xrt/setup.sh
.venv/bin/python -m pytest iron/operators/mem_copy/test.py -q --iterations 1
```

Observed recheck:

```text
64 passed
```

## Clean Graph Edits Appear To Do Nothing

Symptom:

```text
After a runlist or buffer-layout edit, compile_and_load_s is unexpectedly tiny.
The result still looks like the old graph.
```

Diagnostic:

```text
Treat the tiny compile time as evidence of cached artifacts. Inspect or remove
the build directory before judging the edit.
```

Command used:

```bash
rm -rf build_qwen3_megakernel
source /opt/xilinx/xrt/setup.sh
.venv/bin/python iron/applications/qwen3_0_6b/qwen3_megakernel.py \
  --model Qwen/Qwen3-0.6B \
  --num-layers 1 \
  --max-seq-len 256 \
  --verify-one-step
```

Root cause:

```text
FusedMLIROperator reused existing artifacts. Runtime-only iteration is fast,
but graph edits require a clean build to prove the generated ELF changed.
```

Fix:

```text
Use --clean-build or a new build directory after runlist, buffer layout, or
patch-site changes.
```

## Host Buffer Assignment TypeError

Symptom:

```text
TypeError: can't assign a numpy.ndarray to a torch.BFloat16Tensor
```

Diagnostic:

```text
Check the host buffer ABI before blaming AIE kernels. The failure occurs while
filling a host-side full-ELF buffer, not inside the NPU program.
```

Root cause:

```text
The RoPE LUT helper returned a NumPy ml_dtypes.bfloat16 view, while
FusedFullELFCallable.get_buffer(...).torch_view() expected Torch tensor
assignment.
```

Fix:

```text
Return a contiguous torch.bfloat16 tensor for the RoPE LUT runtime input.
```

## One-Step Decode Token Is Wrong

Symptom:

```text
prompt_next_token: 33067
npu_next_token: 1172 text=' only'
ref_next_token: 11853 text='imize'
logits_max_abs: 25.687500
logits_mean_abs: 4.346396
```

Diagnostic:

```text
Do not inspect final logits first. Drain stage-local debug copies and find the
first wrong semantic tensor.
```

Useful debug outputs:

```text
x_norm
queries_raw
queries_norm
queries
keys_raw
keys_norm
keys
values
attn_scores
attn_weights
attn_context
attn_out
ffn_out
logits
```

Evidence found:

```text
values had only GEMV-level error, while queries and keys were already badly
wrong after the extra Q/K path.
```

Root cause:

```text
In-place Q/K RMSNorm and RoPE were unsafe across the fused full-ELF buffer
layout:

q_norm("queries" -> "queries")
rope_q("queries" -> "queries")
k_norm("keys" -> "keys")
rope_k("keys" -> "keys")
```

Fix:

```text
Make Q/K phases explicit:

gemv_q -> queries_raw -> q_norm -> queries_norm -> rope_q -> queries
gemv_k -> keys_raw    -> k_norm -> keys_norm    -> rope_k -> keys
```

Accepted recheck:

```text
npu_next_token: 11853 text='imize'
ref_next_token: 11853 text='imize'
logits_max_abs: 0.437500
logits_mean_abs: 0.073747
```

## First Iteration Is Zero, Later Iterations Improve

Symptom:

```text
iteration 0: x_norm, queries_raw, keys_raw, values all zero
iteration 1: qkv values mostly aligned
```

Diagnostic:

```bash
source /opt/xilinx/xrt/setup.sh
.venv/bin/python iron/applications/qwen3_0_6b/qwen3_megakernel.py \
  --model Qwen/Qwen3-0.6B \
  --num-layers 1 \
  --max-seq-len 256 \
  --debug-stage qkv \
  --verify-repeat 3
```

Evidence found:

```text
The first bad tensor was x_norm, before Q/K/V GEMV, Q/K RMSNorm, or RoPE.
```

Root cause:

```text
XRTSubBuffer.torch_view() marked only the sub-buffer as CPU-resident. The
parent BO could still look NPU-current, so parent.to("npu") skipped the host
write.
```

Fix:

```text
Let XRTSubBuffer keep its parent tensor and mark the parent dirty from
torch_view().
```

Accepted recheck:

```text
iteration 0: npu_next_token=11853 text='imize'
iteration 1: npu_next_token=11853 text='imize'
iteration 2: npu_next_token=11853 text='imize'
```

## Layout-Only Changes Move The Token

Symptom:

```text
A graph edit that should be layout-only changes the one-layer output token.
```

Experiments that were reverted:

```text
1. Expanding RoPE LUTs from one shared angle row to one row per head.
2. Splitting the reused x_norm buffer into attn_x_norm and mlp_x_norm.
3. Replacing multi-row RoPE with one RoPE(rows=1) call per head.
```

Diagnostic:

```text
Run a clean A/B build. In full-ELF fusion, scratch layout and patch locations
are correctness-relevant.
```

Current baseline:

```text
Q/K in-place norm/RoPE removed: keep
RoPE angle_rows=1 shared LUT: keep
Multi-row RoPE run: keep
x_norm reused between attention and MLP norm: keep
```

## Persistent QKV Cannot Place On 8 Columns

Symptom:

```text
ValueError: Failed to find a tile matching column 0: tried until column 8.
Try using a device with more columns.
```

Diagnostic:

```text
This is a placement/resource failure from resolve_program(), not a runtime
numeric issue. Reduce the persistent stage first, then scale placement
explicitly.
```

Root cause established for bring-up:

```text
The first hand-authored QKV persistent stage was too optimistic as an 8-column
layout before resource ownership was explicit.
```

Fix used for current checkpoint:

```text
Set num_aie_columns=1 for the semantic checkpoint.
```

Status:

```text
This is not a final performance placement. It is the accepted correctness
bring-up fallback.
```

## Persistent QKV Exceeds Output DMA Channels

Symptom:

```text
error: 'aie.tile' op number of output DMA channel exceeded! %tile_0_3
```

Diagnostic:

```text
Inspect the ObjectFIFO graph around the tile named in the aiecc error. Count
producer outputs from that tile before editing kernels.
```

Evidence found:

```text
weight_worker produced four separate outputs: one debug x_norm drain plus
Q, K, and V copies.
```

Root cause:

```text
The design duplicated the same logical x_norm token as multiple producer
outputs, exhausting tile DMA resources.
```

Fix:

```text
Use one ObjectFIFO with multiple consumers:

hidden_fifo -> rmsnorm_worker -> weight_worker ->
single xnorm broadcast FIFO -> Q/K/V matvec workers and optional debug drain
```

## Runtime Segfaults In XRT BO Validation

Symptom:

```text
Fatal Python error: Segmentation fault
Current thread:
  hostruntime.py line 274 in run
C stack:
  libxrt_coreutil.so.2 ... validate_bo_at_index
  xrt::run::set_arg_at_index
```

Diagnostic:

```text
Compare generated MLIR runtime_sequence arguments with xclbin
main_kernels.json BO metadata.
```

Commands used:

```bash
rg -n "aie.runtime_sequence|dma_bd\\(%arg" \
  build_qwen3_persistent/*.mlir

cat build_qwen3_persistent/*.mlir.prj/main_kernels.json
```

Evidence found before the fix:

```text
MLIR runtime_sequence had 9 memref arguments.
main_kernels.json exposed only bo0..bo4, i.e. 5 host BO arguments.
```

Root cause:

```text
The host passed more runtime BOs than the kernel metadata advertised. XRT
crashed while validating a BO argument beyond the metadata.
```

Fix:

```text
Pack QKV runtime buffers into 3 BOs:

hidden[1024]
packed_weights[4195328] = norm_weight + Wq + Wk + Wv
packed_outputs[5120] = x_norm + queries_raw + keys_raw + values
```

Accepted evidence after the fix:

```text
aie.runtime_sequence(
  %arg0: memref<1024xbf16>,
  %arg1: memref<4195328xbf16>,
  %arg2: memref<5120xbf16>)
```

The metadata still exposes `bo0` through `bo4`, so three runtime BOs are within
the available host argument range.

## QKV Numeric Errors Appear After Runtime Packing

Symptom:

```text
x_norm_errors: 0
queries_raw_errors: 59
keys_raw_errors: 26
values_errors: 34
```

Diagnostic:

```text
Use the first accepted NPU tensor as the local reference input for the next
stage. Do not compare every downstream stage directly to the full PyTorch path.
```

Check used:

```python
torch_from_npu_x = F.linear(npu_x_norm.view(1, 1, -1), W).flatten()
```

Evidence found:

```text
against_torch_from_npu_x queries_raw: max=0.000000 mean=0.000000
against_torch_from_npu_x keys_raw:    max=0.000000 mean=0.000000
against_torch_from_npu_x values:      max=0.000000 mean=0.000000
```

Root cause:

```text
The verifier used the wrong boundary. Q/K/V projection was correct for the
actual NPU x_norm it consumed. The apparent Q/K/V errors were upstream RMSNorm
drift being projected by Wq/Wk/Wv.
```

Fix:

```text
Verify Q/K/V against F.linear(actual_npu_x_norm, W). Print full-reference drift
separately.
```

Accepted recheck:

```text
x_norm_max_abs: 0.007812
x_norm_errors: 0
queries_raw_max_abs: 0.000000
queries_raw_full_ref_max_abs: 0.015625
queries_raw_errors: 0
keys_raw_max_abs: 0.000000
keys_raw_full_ref_max_abs: 0.007812
keys_raw_errors: 0
values_max_abs: 0.000000
values_full_ref_max_abs: 0.005859
values_errors: 0
```

## Static MLIR Shows One FIFO Drained Twice

Symptom:

```text
The design intends to write k_rope and values both to debug outputs and to KV
cache, but generated ObjectFIFO declarations show only one shim consumer.
```

Diagnostic:

```bash
rg -n "objectfifo @qwen3_rc_(v_0|k_rope_0|q_rope_0|k_raw_0|q_raw_0)" \
  build_qwen3_persistent/*.mlir
```

Evidence found:

```text
qwen3_rc_k_rope_0(%tile_2_2, {%shim_noc_tile_2_0}, ...)
qwen3_rc_v_0(%tile_1_2, {%shim_noc_tile_3_0}, ...)
```

The Python design had two `rt.drain(... k_rope_fifos[col].cons() ...)` calls
and two `rt.drain(... v_fifos[col].cons() ...)` calls. The MLIR showed this was
not a broadcast to two independent consumers; it was one consumer endpoint.

Root cause:

```text
Two runtime drains from the same FIFO consumer would consume the token stream
twice. For k_rope/v there were only 8 produced head tokens, but two drains
would require 16 consumed tokens.
```

Fix:

```text
Make cache the only drain for k_rope and values. Read semantic `keys` and
`values` for verification from the current-position cache slice instead of
adding separate debug drains.
```

Recheck:

```text
runtime_sequence kept five BOs.
k_rope and v ObjectFIFOs each had one shim consumer.
cache current-position errors were zero.
```

## RoPE Outputs Fail Under GEMV Tolerance

Symptom:

```text
queries_errors: 10
keys_errors: 6
queries_max_abs: 0.125000
keys_max_abs: 2.000000
```

Diagnostic:

```text
Compare NPU RoPE against both the model-level apply_rope reference and a host
reference that follows aie_kernels/generic/rope.cc two-halves ordering.
Then check the existing RoPE operator test tolerance.
```

Evidence found:

```text
q_apply max 0.125 mean 0.004813
q_cc    max 0.125 mean 0.004813
k_apply max 2.000 mean 0.011349
k_cc    max 2.000 mean 0.011349
```

The existing RoPE operator test uses:

```text
rel_tol=0.05
abs_tol=0.5
```

Root cause:

```text
The persistent stage verifier reused the strict GEMV abs_tol=1e-6 for RoPE
outputs. The RoPE layout and LUT were correct; the verifier was using the wrong
operator-specific tolerance.
```

Fix:

```text
Use rel_tol=0.05 and abs_tol=0.5 for `queries` and `keys` in
input-rmsnorm-qkv-rope-cache. Keep the stricter GEMV-style tolerance for Q/K/V
projection and cache copy checks.
```

Accepted recheck:

```text
queries_errors: 0
keys_errors: 0
keys_cache_current_errors: 0
values_cache_current_errors: 0
keys_cache_prefix_errors: 0
values_cache_prefix_errors: 0
```

## Attention Score Worker Exceeds Input DMA Channels

Symptom:

```text
error: 'aie.tile' op number of input DMA channel exceeded! %tile_2_3
```

Diagnostic:

```bash
rg -n "tile_2_3|objectfifo @qwen3_rc_" build_qwen3_persistent/*.mlir
```

Evidence found:

```text
qwen3_rc_q_rope_0  -> tile_2_3
qwen3_rc_k_rope_0  -> tile_2_3
qwen3_rc_k_cache_0 -> tile_2_3
```

Root cause:

```text
The score worker had three input ObjectFIFOs: Q RoPE, current K RoPE, and
K-cache block. The tile input DMA channels were exhausted before runtime.
```

Fix direction:

```text
Do not build score as a three-input worker. Pack Q and current K in a prior
two-input worker, then run score as qk_pair + k_cache_block.
```

## K Cache Matrix Does Not Fit In L1

Symptom:

```text
Failed to allocate buffer: "qwen3_rc_k_cache_0_cons_buff_0" with size: 65536 bytes
allocated buffers exceeded available memory
```

Diagnostic:

```text
Read the aiecc MemoryMap. It printed two 65536-byte K-cache consumer buffers on
one compute tile, before score and Q buffers were counted.
```

Root cause:

```text
The first score design tried to move a full [256, 128] bf16 K-cache head as one
ObjectFIFO object. With depth=2 that alone needs 128KB, above the tile L1 budget.
```

Fix:

```text
Stream K cache as [64, 128] blocks. This makes each K-cache object 16KB and
keeps depth=2 within L1.
```

## Debug Pass-Through FIFO Exceeds L1

Symptom:

```text
Adding an independent K-cache debug stream makes aiecc fail allocation on the
debug worker tile, even though the production K-cache stream already fits.
```

Evidence found:

```text
The debug worker carried both input and output ObjectFIFOs with [64, 128] bf16
objects. With depth=2 on both edges, the tile needs 16KB * 4 = 64KB for debug
FIFO objects before stack and other allocations.
```

Root cause:

```text
The debug copy used normal depth=2 FIFOs on both sides. For a pass-through
diagnostic worker that does no compute overlap, double-buffering both input and
output consumed the whole tile L1 budget.
```

Fix:

```text
Set both K-cache debug pass-through FIFOs to depth=1. The diagnostic still
proves the runtime TAP/stream order, while production K-cache keeps depth=2 for
the score worker.
```

## K Cache Block DMA Exhausts BD IDs

Symptom:

```text
Allocator exhausted available buffer descriptor IDs
```

Diagnostic:

```bash
rg -n "dma_configure_task_for @qwen3_rc_k_cache_0" build_qwen3_persistent/*.mlir
```

Evidence found:

```text
One Python rt.fill per q-head/cache-block generated 64 DMA tasks for the same
K-cache FIFO.
```

Root cause:

```text
The logical tiling was correct, but the runtime expression used too many
separate DMA tasks. BD exhaustion is a runtime/tap expression bug, not a math
kernel bug.
```

Fix direction:

```text
Prefer one multidimensional TensorAccessPattern for the repeated block stream.
If repeated GQA access would require illegal stride=0, reuse K blocks inside a
worker instead of asking DMA to reread the same block.
```

## Multidimensional TAP Is Legal But NPU BD Rejects It

Symptoms:

```text
Stride 2 must be a positive integer
Size 0 exceeds the [0:1023] range
```

Evidence found:

```text
[size = 2, stride = 0] was rejected.
[size = 8192, stride = 1] was rejected in a multidimensional BD.
```

Root cause:

```text
TAP can describe access maps that are semantically clear but illegal for NPU BD
lowering. Reusing a tensor region with stride=0 across a non-unit dimension is
not accepted, and individual BD dimension sizes must stay in range.
```

Fix:

```text
Express K-cache blocks as [kv_head, block, row, dim] with strides
[max_seq_len * head_dim, block_rows * head_dim, head_dim, 1].
```

## Runtime Start Does Not Create A Phase Barrier

Symptom:

```text
Host runtime returns ERT_CMD_STATE_TIMEOUT after adding a second Runtime task group.
```

Diagnostic:

```bash
sed -n '350,470p' build_qwen3_persistent/*.mlir
rg -n "core|start|dma_await_task" build_qwen3_persistent/*.mlir
```

Evidence found:

```text
All workers are emitted as always-running aie.core loops. A later rt.start()
does not create the intended "start Q path after K cache write" barrier.
```

Root cause:

```text
The attempted phase split relied on runtime start ordering, but the actual
program is a static dataflow graph with persistent cores. Q/score consumers
still participated in ObjectFIFO synchronization from the beginning.
```

Fix direction:

```text
Use explicit data dependencies, phase tokens, or a different dataflow shape.
For attention score, packing Q/current-K before score avoids the phase barrier.
```

## Attention Scores Overwrite The First GQA Head

Symptom:

```text
upstream Q/K/V, RoPE, and KV cache checks pass
attn_scores_errors: nonzero
qk_pair_errors: 0
k_cache_stream_prefix_errors: 0
```

First diagnostic:

```bash
source /opt/xilinx/xrt/setup.sh
.venv/bin/python iron/applications/qwen3_0_6b/qwen3_persistent.py \
  --model Qwen/Qwen3-0.6B \
  --stage input-rmsnorm-qkv-rope-cache-scores-softmax \
  --verify \
  --verify-repeat 1 \
  --dump-proof \
  --clean-build
```

Evidence found:

```text
The verifier drained qk_pair and an independent K-cache stream:

qk_pair_errors: 0
k_cache_stream_prefix_errors: 0
attn_scores_errors: 413

So the failing boundary is after qk_pair packing and after K-cache DMA/TAP.
```

Second diagnostic:

```bash
sed -n '145,260p' \
  build_qwen3_persistent/*.mlir.prj/main_core_2_4.peanohack.ll
```

Generated LLVM evidence:

```text
call void @llvm.aie2p.acquire(i32 48, i32 -1)
%15 = phi ptr ... @qwen3_rc_attn_scores_0_buff_0 ...

%25 = phi ptr ... @qwen3_rc_attn_scores_0_buff_0 ...

call void @qwen3_attention_scores_bf16(... ptr %16 ... i32 0)
call void @qwen3_attention_scores_bf16(... ptr %26 ... i32 1)
```

There is a lock acquire before the first `score_debug_fifo.acquire(1)`, but not
before the second `score_debug_fifo.acquire(1)`. Both score pointers can name
the same ObjectFIFO buffer before any release.

Root cause:

```text
The Worker treated two sequential acquire(1) calls on the same producer FIFO as
two different output tokens. IRON/ObjectFIFO acquire is stateful: if enough
objects are already acquired, a later acquire of the same or smaller size does
not acquire additional objects. The second score kernel overwrote the first GQA
head's score buffer.
```

Confirmed API rule:

```text
ObjectFifo acquire only performs new lock acquires if necessary. If one object
is already acquired, another acquire(1) in the same process returns within the
already-acquired set.
```

Fix:

```text
The score Worker now uses one acquire(2), indexes subviews [0] and [1], and
releases two objects after both q_select score rows are produced.
```

Recheck:

```text
Generated MLIR:
  aie.objectfifo.acquire @qwen3_rc_attn_scores_0(Produce, 2)
  aie.objectfifo.subview.access %2[0]
  aie.objectfifo.subview.access %2[1]
  aie.objectfifo.release @qwen3_rc_attn_scores_0(Produce, 2)

Runtime verifier:
  qk_pair_errors: 0
  k_cache_stream_prefix_errors: 0
  attn_scores_errors: 0
  attn_weights_errors: 0
  preflight: ok ... non_advancing_acquires=0
```

Follow-up verifier bug found:

```text
After score passed, six attn_weights elements still failed because the expected
weights used softmax(float32_matmul_scores). The NPU softmax consumes bf16
attn_scores from the FIFO, so the local reference must softmax the bf16-rounded
score tensor converted back to float32.
```

## PV/context Fails During L1 Buffer Allocation

Symptom:

```text
Failed to allocate buffer: "qwen3_rc_v_cache_0_cons_buff_0" with size: 16384 bytes
error: 'aie.tile' op allocated buffers exceeded available memory
tile_3_3 MemoryMap includes:
  qwen3_rc_v_context_debug_0_buff_0
  qwen3_rc_v_context_block_0_buff_0
  qwen3_rc_v_context_block_0_buff_1
  qwen3_rc_v_cache_0_cons_buff_0
  qwen3_rc_v_cache_0_cons_buff_1
```

Root cause:

```text
The V merge tile buffered too many 64x128xbf16 blocks at once. The object size
was legal, but depth=2 on both V-cache and V-context block FIFOs plus a debug
block exceeded L1.
```

Fix:

```text
Keep the 64-token V block shape, but make the V-cache and V-context block
FIFOs depth=1 so the tile streams one block at a time.
```

Recheck:

```text
preflight: ok ... max_fifo_buffered_bytes=32768 max_tile_inputs=2 max_tile_outputs=2
```

## PV/context Inputs Pass But Context Has A Few Large Errors

Symptom:

```text
attn_weights_errors: 0
v_context_stream_prefix_errors: 0
v_context_stream_current_errors: 0
attn_context_errors: 6
```

Root cause:

```text
The context external kernel did per-row bf16 read/modify/write accumulation.
That was the first unproven boundary after all input streams passed.
```

Fix:

```text
Accumulate each 64-row V block in a local float array and write bf16 context
once per block.
```

Recheck:

```text
attn_context_errors: 0
attn_context_max_abs: 0.000000
```

## New FIFO Has Consumer But No Producer

Symptom:

```text
ValueError: Prod endpoint not set for ObjectFifo(... name='qwen3_rc_o_weight_0',
prod=None, cons=[...])
```

Root cause:

```text
The O projection worker consumed qwen3_rc_o_weight_0, but the matching
Runtime.fill() was accidentally added to an earlier QKV-only design function,
not the RoPE/cache/context implementation that owns the worker.
```

Fix:

```text
Search the generated design path, not only the source text. Every new
worker-side input FIFO must have a Runtime.fill(), Worker producer, or link in
the same Program variant.
```

## New Stage Exceeds SequentialPlacer Worker Capacity

Symptom:

```text
ValueError: Failed to find a tile matching column 3: tried until column 8.
Try using a device with more columns.
```

Root cause:

```text
The accepted context checkpoint already used 15 compute Workers on the current
NPU2 placement. Adding separate context-flatten, O-projection, and residual
Workers pushed the design past the 16 compute-tile budget used by the
SequentialPlacer in this environment.
```

Fix:

```text
Fuse context flatten into the context Worker and disable the older K-cache
debug copy Worker in the deeper O-projection checkpoint. Keep debug at the
new boundary instead: attn_context, attn_context_flat, attn_o_proj,
attn_residual.
```

Recheck:

```text
preflight: ok ... max_tile_inputs=2 max_tile_outputs=2
```

## Disabled Debug Stream Creates Zero-Length TAP

Symptom:

```text
ValueError: All sizes must be >= 1, but got [1, 1, 1, 0]
```

Root cause:

```text
K-cache debug output size was set to zero for the O-projection checkpoint, but
the TAP was still generated from include_scores_softmax instead of the more
specific include_k_cache_debug flag.
```

Fix:

```text
When a debug stream is optional, guard object FIFOs, workers, fills, drains,
and TAPs with the same boolean.
```

## MLP Gate/Up Fails Full Reference But Passes Local Boundary

Symptom:

```text
mlp_x_norm_errors: 0
ffn_gate_errors: 108
ffn_up_errors: 145
```

Evidence after changing only the verifier to use the actual NPU
`mlp_x_norm` as the GEMV input boundary:

```text
ffn_gate_max_abs: 0.000000
ffn_gate_errors: 0
ffn_up_max_abs: 0.000000
ffn_up_errors: 0
ffn_gate_full_ref_max_abs: 0.015625
ffn_up_full_ref_max_abs: 0.007812
```

Root cause:

```text
The gate/up GEMV workers were correct. The original check compared them to a
full PyTorch reference fed by PyTorch mlp_x_norm, while the NPU workers consume
the bf16 mlp_x_norm FIFO produced by the NPU weighted RMSNorm worker.
```

Fix:

```text
For each stage, compare the first output at the full reference boundary, then
build downstream local references from the actual NPU FIFO output that the next
Worker consumes.
```

## SiLU Negative Inputs Exceed The Positive-Only Operator Tolerance

Symptom:

```text
ffn_gate_silu_errors: 3
ffn_gate_silu_max_abs: 0.019531
Mismatch in ffn_gate_silu[1331]: expected -0.090820, got -0.100586
```

Root cause:

```text
The AIE SiLU kernel uses the tanh-form approximation. The standalone SiLU test
used random positive inputs in [0, 4), but Qwen3 gate projection produces
negative values where the approximation has a larger absolute error against
PyTorch exact SiLU.
```

Fix:

```text
Verify SiLU at the actual gate FIFO boundary and use an explicit absolute
tolerance for the approximation. Do not treat the downstream ffn_hidden as a
new multiply bug if it matches actual_silu * actual_up.
```

Recheck:

```text
ffn_gate_silu_errors: 0
ffn_hidden_errors: 0
```
