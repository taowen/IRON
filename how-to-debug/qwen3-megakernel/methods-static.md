<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Qwen3 Megakernel Static Diagnostic Methods

[Back to method index](diagnostic-methods.md).

## 4. Compare runtime_sequence With main_kernels.json

Use when runtime crashes while setting XRT kernel arguments.

```bash
rg -n "aie.runtime_sequence|dma_bd\\(%arg" build_qwen3_persistent/*.mlir
cat build_qwen3_persistent/*.mlir.prj/main_kernels.json
```

This diagnosed the persistent QKV segfault:

```text
faulthandler stack: xrt::run::set_arg_at_index -> validate_bo_at_index
MLIR: 9 runtime memrefs
metadata: bo0..bo4 only
```

Fix proven by the same method:

```text
MLIR: 3 runtime memrefs
metadata: bo0..bo4
```

## 9. Read aiecc Resource Errors As Graph Errors

Use when a persistent Program fails during placement or allocation.

Examples already diagnosed:

```text
Failed to find a tile matching column 0 ... tried until column 8
```

This meant the first 8-column persistent layout was too ambitious for the
unproven stage. The accepted checkpoint used one column.

```text
error: 'aie.tile' op number of output DMA channel exceeded! %tile_0_3
```

This named the tile whose ObjectFIFO producer outputs needed inspection. The
fix was a single broadcast `x_norm` FIFO instead of multiple duplicated output
FIFOs.

## 13. Count Tile FIFO Inputs Before Changing Kernels

Use when aiecc reports input or output DMA channel exhaustion.

```bash
rg -n "tile_2_3|objectfifo @qwen3_rc_" build_qwen3_persistent/*.mlir
```

The score bring-up used this to prove the failing tile had three input FIFOs:

```text
Q RoPE
current K RoPE
K cache block
```

The fix was a graph change, not a C++ dot-product change: introduce a qk-pair
packing stage so score has only two inputs.

## 14. Read L1 MemoryMap Literally

Use when aiecc says buffers do not fit.

```text
Failed to allocate buffer: "...k_cache...cons_buff_0" with size: 65536 bytes
MemoryMap:
  k_cache buff 0: 65536 bytes
  k_cache buff 1: 65536 bytes
```

This means the ObjectFIFO object shape is too large for the tile, even before
math kernel scratch is considered. For attention, stream K/V by sequence blocks
instead of materializing a full [seq, head_dim] object in L1.

The PV/context bring-up hit the same class with smaller objects but too much
buffering on one tile:

```text
tile_3_3:
  qwen3_rc_v_context_debug_0_buff_0  16384 bytes
  qwen3_rc_v_context_block_0_buff_0  16384 bytes
  qwen3_rc_v_context_block_0_buff_1  16384 bytes
  qwen3_rc_v_cache_0_cons_buff_0     16384 bytes
  qwen3_rc_v_cache_0_cons_buff_1     16384 bytes
```

The fix was not to change the PV math. The graph kept 64-token V blocks but
made the V-cache and V-context block FIFOs single-buffered so the merge tile
streams blocks instead of hoarding them in L1.

## 15. Inspect DMA Task Count, Not Just TAP Correctness

Use when BD IDs are exhausted.

```bash
rg -n "dma_configure_task_for @qwen3_rc_k_cache_0" build_qwen3_persistent/*.mlir
```

Correct access order can still be expressed incorrectly if Python emits one
`rt.fill` per logical tile. Prefer a single legal multidimensional TAP, then
verify the generated `aie.dma_bd` dimensions.

## 16. Validate TAP Against NPU BD Limits

Use when NPU lowering rejects a generated `aie.dma_bd`.

Checks from the score bring-up:

```text
Non-unit dimensions cannot use stride=0.
Large flattened dimensions such as size=8192 can be illegal.
Break large contiguous blocks into [row, dim] dimensions.
```

Good K-cache block shape:

```text
sizes   = [kv_heads, blocks, block_rows, head_dim]
strides = [max_seq_len * head_dim, block_rows * head_dim, head_dim, 1]
```

## 19. Run Persistent Artifact Preflight

Use after `op.compile()` and before `op.get_callable()` for hand-authored
persistent stages.

Implemented checks:

```text
MLIR runtime_sequence memref count == operator arg spec count
MLIR runtime_sequence memref count <= main_kernels.json HOST bo* count
ObjectFIFO object bytes * depth <= L1 budget
compute tile input/output ObjectFIFO count <= expected channel budget
DMA task count per FIFO <= expected BD budget
```

This turns already diagnosed failures into Python errors before runtime:

```text
Runtime BO metadata mismatch instead of XRT BO validation segfault
ObjectFIFO L1 budget mismatch instead of aiecc MemoryMap failure
Compute tile input ObjectFIFO overuse instead of DMA channel allocation failure
FIFO DMA task overuse instead of BD ID exhaustion
```

Current implementation:

```text
iron/applications/qwen3_0_6b/qwen3_preflight.py
```

The persistent CLI now prints a `preflight: ok ...` summary immediately after
compile when these checks pass.

## 20. Inspect Repeated ObjectFIFO Acquire Lowering

Use when a Worker acquires more than one object from the same FIFO before any
release and the data looks overwritten or shifted.

Command used:

```bash
sed -n '145,260p' \
  build_qwen3_persistent/*.mlir.prj/main_core_2_4.peanohack.ll
```

This diagnosed the score worker failure:

```text
qk_pair_errors: 0
k_cache_stream_prefix_errors: 0
attn_scores_errors: 413
```

The generated LLVM showed a lock acquire for the first score output, but not for
the second output:

```text
call void @llvm.aie2p.acquire(i32 48, i32 -1)
%15 = phi ptr ... @qwen3_rc_attn_scores_0_buff_0 ...
%25 = phi ptr ... @qwen3_rc_attn_scores_0_buff_0 ...
```

Rule:

```text
Do not assume acquire(1), acquire(1) means two FIFO objects. ObjectFIFO acquire
is stateful and only acquires additional objects if the requested total is
larger than what the process already holds.
```

Accepted shapes for two live output tokens:

```text
one acquire(2) and two indexed subviews
two separate ObjectFIFOs
one packed object whose layout explicitly contains both logical outputs
```

Recheck after changing the graph:

```text
Generated LLVM must show either an acquire of size 2 or separate lock acquires
for the two output FIFOs before the two score kernel calls.
```

Observed fixed MLIR:

```text
aie.objectfifo.acquire @qwen3_rc_attn_scores_0(Produce, 2)
aie.objectfifo.subview.access %2[0]
aie.objectfifo.subview.access %2[1]
aie.objectfifo.release @qwen3_rc_attn_scores_0(Produce, 2)
```

Observed verification:

```text
attn_scores_errors: 0
attn_weights_errors: 0
preflight: ok ... non_advancing_acquires=0
```

## 23. Check Producer Endpoints Before Reading Placer Errors As Resource Errors

Use when `resolve_program()` reports:

```text
Prod endpoint not set for ObjectFifo(...)
```

This is a graph construction error, not a compute kernel issue. For the
O-projection checkpoint, `qwen3_rc_o_weight_0` had a consumer Worker but no
producer because the `Runtime.fill()` was added to the wrong Program variant.

Diagnosis:

```bash
rg -n "qwen3_rc_o_weight|rt.fill\\(" iron/applications/qwen3_0_6b/qwen3_persistent_design.py
```

Required invariant:

```text
Every worker input ObjectFIFO is produced by exactly one source in that same
design variant: Runtime.fill, another Worker, or an ObjectFIFO link.
```

## 24. Count Workers Against The Actual Placer Budget

Use when adding a small downstream phase makes `SequentialPlacer` fail with:

```text
Failed to find a tile matching column ...
```

The context checkpoint used 15 Workers. Adding three more Workers for
flatten, O projection, and residual exceeded the current 16 compute-tile
placement budget. The fix was structural:

```text
context Worker also packs attn_context_flat
drop K-cache debug copy Worker in the deeper checkpoint
keep O projection Worker and residual Worker
```

Do not respond by changing FIFO depths or kernel math until the Worker count
and host-debug workers are accounted for.

## 25. Optional Debug Streams Need One Boolean

Use when a disabled debug stream produces a TAP or DMA error:

```text
All sizes must be >= 1, but got [1, 1, 1, 0]
```

The safe pattern is to derive one boolean, then use it consistently:

```python
include_k_cache_debug = include_scores_softmax and not include_o_proj
```

Apply that same flag to:

```text
debug size
ObjectFIFO creation
Worker creation
Runtime.fill
Runtime.drain
TensorAccessPattern creation
host verifier slices
```

