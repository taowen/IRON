<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Qwen3 Megakernel Dataflow Symptoms

[Back to symptom index](symptoms.md).

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

## Worker Loop Index Does Not Match External Kernel ABI

Symptom:

```text
MLIRError: Verification failed:
'func.call' op operand type mismatch:
expected operand type 'i32', but provided 'index'
```

Diagnostic used:

```text
Read the failing `func.call` signature. In Worker code, `range_()` loop
variables are MLIR `index` values; external kernels declared with `np.int32`
expect i32.
```

Root cause:

```text
The phase-owned production skeleton passed `layer` and `phase` loop variables
directly to `new_mega_phase_packet_accum_bf16`, whose Kernel declaration used
np.int32 for both arguments.
```

Fix used:

```python
layer_i32 = index.casts(T.i32(), layer)
phase_i32 = index.casts(T.i32(), phase)
packet_kernel(packet, state, packet_elements, layer_i32, phase_i32)
```

Recheck:

```text
MLIR generation proceeds past verification; subsequent failures, if any, are
not this ABI type mismatch.
```

## Kernel Arity Fails During `resolve_program()`

Symptom:

```text
ValueError: Kernel 'new_mega_phase0_q_shard_bf16' expects 12 argument(s),
but 10 were provided.
```

Trigger:

```text
While adding a new argument to the O partial kernel, the Python Kernel
declaration for q_shard was accidentally widened instead. The Worker call site
still passed the correct q_shard arguments, so IRON failed before aiecc.
```

Diagnostic:

```text
Treat this as a Python-side Kernel ABI declaration bug, not a placement or
external C++ math bug. Compare exactly three places:

1. C++ function signature.
2. `Kernel(...)` type list in `phase_owned_stages.py`.
3. The Worker call site argument list.
```

Root cause:

```text
The type list for one external symbol was edited while the call site belonged
to a different symbol. `resolve_program()` caught the arity mismatch before MLIR
lowering, resource allocation, routing, or runtime execution.
```

Fix:

```text
Restore q_shard to its original five integer parameters, and add the new
`fabric_group_size` integer only to `new_mega_phase_o_partial_shard_bf16`.
```

Recheck:

```text
`resolve_program()` proceeds into full aiecc. The next diagnostics, if any, are
actual allocation/routing/kernel issues rather than Python Kernel arity drift.
```

## AIE Kernel Cannot Use Host Math sqrtf

Symptom:

```text
phase_owned_kernels.cc: error: use of undeclared identifier 'sqrtf'
```

Diagnostic used:

```text
Search existing AIE kernels before changing graph code. The repo's RMSNorm
kernels use AIE API math, not libc math.
```

Evidence found:

```text
aie_kernels/aie2p/rms_norm.cc uses:
  float inv_rms = aie::invsqrt(rms);
```

Root cause:

```text
The first phase-owned RMSNorm phase used `sqrtf` in an AIE cross-compiled
kernel. The target environment did not expose that host math symbol.
```

Fix used:

```cpp
const float inv_rms = aie::invsqrt(mean_square + 0.000001f);
```

Recheck:

```text
The kernel compiles past the undeclared sqrtf failure.
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

## Inactive FIFO Skip Requires A Different Static Graph

Symptom:

```text
A phase-based Worker design wants to skip an inactive phase without filling
dummy FIFO tokens.
```

Diagnostic:

```text
Build two small variants from the same source:

active:
  Worker acquires phase0 and optional FIFOs.
  Runtime has phase0, optional, output BOs.

skip:
  Worker never creates/acquires the optional FIFO.
  Runtime has only phase0 and output BOs.

Compare MLIR, runtime .bin, xclbin, and runtime argument counts.
```

Evidence from C2:

```text
active arg_count=3, skip arg_count=2
active and skip both matched CPU reference
same_artifact=False
runtime_bin size changed 420 -> 300 bytes
xclbin size changed 10570 -> 10138 bytes
```

Root cause:

```text
Removing an inactive FIFO token is a static dataflow/ABI change in the current
IRON Runtime/ObjectFIFO model. It avoids dummy DMA only by compiling a different
graph. It does not prove same-artifact dynamic phase skipping.
```

Fix direction:

```text
Use static phase ownership or separate artifacts for phase sets that need
different FIFO dependencies. Do not rely on dummy tokens as a scalable
megakernel mechanism, and do not assume a Worker-side conditional can remove
Runtime.fill tasks from the same artifact.
```

## External Kernel Symbol Is Declared With Two Memref Shapes

Symptom:

```text
MLIRError: Verification failed:
error: redefinition of symbol named 'new_mega_copy_bf16'
see current operation:
  func.func ... @new_mega_copy_bf16(memref<3072xbf16>, memref<3072xbf16>, i32)
see existing symbol definition here
```

Trigger:

```text
D1.3c used the same external C symbol for both hidden-sized copies
memref<1024xbf16> and FFN-sized copies memref<3072xbf16>.
```

Diagnostic:

```text
This is an MLIR symbol/ABI failure during Program verification. The NPU
compiler has not reached placement, resource allocation, or external-kernel
math.
```

Root cause:

```text
IRON emits one private func declaration per Kernel object. Reusing the same C
symbol name with different memref shapes creates duplicate symbol definitions
in one MLIR module. The down-projection row-shard experiment also showed that
declaring the same symbol twice with the same memref shape can still produce a
redefinition error before placement.
```

Fix used:

```text
Give each distinct memref signature a distinct external symbol name:
  new_mega_copy_bf16      for memref<1024xbf16>
  new_mega_copy_ffn_bf16  for memref<3072xbf16>

When the signature is identical, reuse the same Kernel object instead of
creating a second Kernel declaration with the same symbol name.
```

## DMA BD Transfer Length Is Not 4-Byte Aligned

Symptom:

```text
aiecc fails during resource allocation:

error: 'aie.dma_bd' op transfer length must be multiple of 4
note: see current operation: "aie.dma_bd"(...)
      : (memref<1xbf16>) -> ()
```

Diagnostic:

```text
Inspect the memref in the failing `aie.dma_bd` note. If it is a one-element
BF16/F16/i16 ObjectFIFO object, the DMA transfer length is only 2 bytes even
though the shape and FIFO endpoint counts are otherwise legal.
```

Evidence from D0:

```text
The static phase ownership skeleton used an output FIFO object shaped
memref<1xbf16>. The first compile failed before preflight with the 4-byte
transfer-length diagnostic above.
```

Root cause:

```text
AIE DMA BD transfer lengths must be 4-byte aligned. A scalar BF16 FIFO object
is a legal-looking IRON type but not a legal DMA transfer payload.
```

Fix:

```text
Pad scalar BF16/F16 DMA-visible objects to at least two elements and ignore the
padding element semantically. Prefer 16-byte object alignment for megakernel
FIFOs because the preflight linter already enforces that stricter rule.
```

## Worker Closure Calls An Unresolved Kernel

Symptom:

```text
real_graph_probe: fail stage=n-layer-final-only cols=2 layers=1 phase=preflight
ValueError: Kernel must be resolved before it can be called.
```

Diagnostic:

```text
Re-run the Program generator directly with a Python traceback. If the stack
points inside a Worker body and the failing call is a Kernel object from the
outer Python closure, inspect the Worker(...) argument list.
```

Evidence found:

```text
The fused postnorm/gate-up worker called hidden_copy from the outer closure:

  hidden_copy(xnorm_buffer, xnorm_out, hidden_size)

but Worker(...) did not include hidden_copy in its argument list. IRON resolves
Kernel objects that are passed into the worker arguments; the closure reference
remained unresolved when the core body was emitted.
```

Fix:

```text
Add the copy Kernel as an explicit Worker argument and call that parameter from
the Worker body:

  copy_kernel(xnorm_buffer, xnorm_out, hidden_size)
```

Recheck:

```text
The next preflight progressed past Worker resolution and reached the real graph
resource/TAP checks.
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
.venv/bin/python iron/applications/qwen3_0_6b/persistent/main.py \
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

## Producer Order Deadlocks Despite Balanced FIFO Counts

Symptom:

```text
preflight: ok ... compute_cores=24 max_tile_inputs=2 max_tile_outputs=2
aie.utils.hostruntime.hostruntime.HostRuntimeError:
  Kernel returned ert_cmd_state.ERT_CMD_STATE_TIMEOUT
```

Diagnostic:

```text
After preflight passes, compare the production order inside the new Worker
against the first acquire order in the downstream Worker. Balanced token counts
are not enough; a bounded FIFO can still deadlock if the first required token
is produced late.
```

Evidence found during attention2 QK merge:

```text
The first qk_matvec_worker prototype consumed one combined QK weight stream and
produced all Q heads first, then all K heads.

Downstream qk_pair_final_worker consumed in the opposite phase order:
  current K first
  then the two Q heads mapped to that KV head

Q RoPE output filled its small FIFO while qk_pair waited for K, and the QK
matvec Worker could not advance far enough to produce K.
```

Root cause:

```text
The QK weight stream was contiguous as all Q shard rows followed by all K shard
rows. That layout was statically legal but temporally incompatible with the
attention score pipeline.
```

Fix used:

```text
Pack each attention2 QK shard by KV-head group:
  K head h
  Q heads mapped to K head h
  next K head
  next mapped Q heads

Then make qk_matvec_worker emit K first and the corresponding Q heads second,
matching qk_pair_final_worker.
```

Recheck:

```text
attention2 attention-only verify after interleaved QK:
  npu_time_us=3938.720
  attention_probe_residual_errors=0
  layer0_keys_cache_current_errors=0
  layer0_values_cache_current_errors=0
```

## Attention2 Multi-Layer Pack Order Mismatches Runtime TAP

Symptom:

```text
attention2 + full MLP2 with hidden+metadata fusion works for layer_iterations=1
but fails at layer_iterations=2:

chunk_hidden_errors: 788
layer0_values_cache_current_errors: nonzero
layer1_keys_cache_current_errors: nonzero
layer1_values_cache_current_errors: nonzero
```

Diagnostic:

```text
Do not inspect the external QKV kernel first. Compare the packed weight
artifact order against the Runtime.fill TAP order. The TAP can be legal while
still consuming a different segment order than the packer wrote.
```

Evidence found:

```text
The packer wrote:

  qk_l0, v_l0, qk_l1, v_l1

The attention2 runtime TAP consumed per column:

  qk_col0_all_layers, v_col0_all_layers,
  qk_col1_all_layers, v_col1_all_layers
```

Root cause:

```text
The multi-layer packed weight artifact was layer-major, but the multi-column
runtime stream was segment-major. All offsets were in bounds, so preflight and
compilation could not catch the semantic order mismatch.
```

Fix:

```text
Pack attention2 weights in the same segment-major order used by Runtime.fill:

  qk_col0_all_layers
  v_col0_all_layers
  qk_col1_all_layers
  v_col1_all_layers
```

Recheck:

```text
pytest -q \
  iron/applications/qwen3_0_6b/test.py::test_qwen3_attention2_segment_major_weight_chunk_matches_layer_major_artifact

layer_iterations=2 verify:
  chunk_hidden_errors=0
  current K/V cache errors=0
```

## Hidden/Metadata Split Times Out At Larger Chunks

Symptom:

```text
attention2 + full MLP2 with hidden+metadata runtime fusion passes preflight
and compiles for layer_iterations=8, but runtime returns:

ERT_CMD_STATE_TIMEOUT
```

Diagnostic:

```text
After preflight passes, inspect the temporal order of split child stream
consumption. A split can create balanced token counts but still stall if one
child stream is drained far ahead of the other with small FIFO depths.
```

Evidence found:

```text
The runtime input object is split into:

  hidden child token
  QK/RoPE metadata child token

for every layer. The fused initial/RMS worker consumed layer0 hidden and then
discarded all later hidden child tokens up front, while downstream workers
needed metadata tokens in layer order.
```

Root cause:

```text
The two split child streams advanced in different phase orders. At larger
chunks, early hidden-token discard could backpressure the split/metadata path
and prevent the layer-ordered dataflow from reaching the later metadata token.
```

Fix:

```text
Discard exactly one unused hidden child token per subsequent layer immediately
before that layer consumes feedback. This keeps hidden and metadata child
streams advancing in the same layer order.
```

Recheck:

```text
layer_iterations=8:
  no runtime timeout
  chunk_hidden_errors=0
```

## Scalar State Mistaken For Activation Transfer

Symptom:

```text
The phase-owned graph appears to have persistent state, but the only state is a
single float checksum. Phase 0 computes from packet inputs, and later phases do
not carry a hidden[1024] activation into the next layer.
```

Diagnostic:

```text
List every value that must cross a phase or layer boundary. For each one,
identify whether it is carried by ObjectFifo or by an activation-sized
tile-local Buffer. A scalar checksum does not count.
```

Evidence found:

```text
state[0] was only updated for debugging.
No tile-local hidden vector existed.
Layer N+1 phase 0 could not depend on layer N residual output except through
host-provided packet data.
```

Root cause:

```text
The skeleton proved Worker/FIFO resource shape but not activation lifetime.
Real Qwen3 decode needs hidden[1024] BF16 to survive across phases/layers.
```

Fix:

```text
Add a tile-local hidden_state[1024] BF16 Buffer per lane Worker.
Initialize it from the shared stream for layer 0.
Make phase 0 read hidden_state for RMSNorm/Q shard.
Make the next_layer_token phase write hidden_state for the next layer.
Keep state[0] only as a checksum.
```

Recheck:

```text
num_lanes=8, fabric_group_size=4:
  preflight_compute_cores=8
  preflight_max_compute_tile_inputs=2
  preflight_max_compute_tile_outputs=1
  phase_owned_errors=0
```

## O Projection Packet Width Uses Hidden Size Instead Of Attention Size

Symptom:

```text
RuntimeError: The expanded size of the tensor (1024) must match the existing
size (2048) at non-singleton dimension 0.
```

Diagnostic:

```text
Do not assume every phase packet vector is hidden_size wide. Check the actual
weight shape and the tensor being packed:

  attention_context.shape
  o_proj.weight.shape
  hidden_size
  num_attention_heads * head_dim
```

Evidence found:

```text
Qwen3-0.6B:
  hidden_size = 1024
  num_attention_heads = 16
  head_dim = 128
  attention_size = 2048
  o_proj.weight shape = [1024, 2048]
  attention_context length = 2048
```

Root cause:

```text
The o_proj packet layout reused hidden_size for the attention context and O
weight row width. That was correct for residual/RMSNorm/down output space, but
wrong for Q/O attention space in Qwen3-0.6B.
```

Fix:

```text
Add attention_size as an explicit operator/design/ABI parameter.
Use attention_size for:
  attention_context packet field
  o_proj weight row packet field
  o_proj kernel dot loop bound

Keep hidden_size for:
  residual shards
  RMSNorm vectors
  MLP down output rows
  tile-local hidden_state
```

Recheck:

```text
attention_size=2048
phase_owned_errors=0
qwen3_phase_output_errors=0
```

## Kernel Declaration Has One More Argument Than The C++ ABI

Symptom:

```text
resolve_program fails before aiecc:

ValueError: Kernel 'new_mega_phase0_q_shard_bf16' expects 11 argument(s),
but 10 were provided.

ValueError: Kernel 'new_mega_phase0_q_shard_bf16' expects 9 argument(s),
but 10 were provided.
```

Diagnostic:

```text
Count the Python Kernel(...) declaration and the C++ function signature side by
side. This is a static ABI declaration error, not an ObjectFifo or placement
bug.
```

Evidence found:

```text
After adding attention_size to the new o_proj kernel, one stale np.int32 was
left in the q_shard Kernel(...) declaration. The q_shard C++ signature still
took 10 arguments, and the worker call passed 10 arguments.

After adding a generic K/V projection kernel, the opposite version of the same
bug happened: q_shard lost one np.int32 in the Python Kernel(...) declaration,
while the C++ signature and Worker call still used 10 arguments.
```

Root cause:

```text
The wrong Kernel(...) declaration was edited. IRON validates the declared
argument count when resolving the Worker body, so it failed before MLIR/AIECC
lowering.
```

Fix:

```text
Add or remove the np.int32 in the q_shard Kernel(...) declaration so the count
matches the C++ signature and Worker call.
When adding a dimension parameter, update only the affected Kernel(...)
declaration, C++ signature, and worker call together.
```

Recheck:

```text
compile-only:
  preflight_compute_cores=8
  preflight_max_compute_tile_inputs=2
  preflight_max_compute_tile_outputs=1
```
