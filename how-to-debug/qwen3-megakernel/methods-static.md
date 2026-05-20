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

The same method caught the first two-layer persistent chunk:

```text
MLIR: 7 runtime memrefs
operator arg spec: 7
main_kernels.json: bo0..bo4 only
```

The correct fix was to reduce the runtime ABI, not to bypass preflight:

```text
hidden
weight_pair = layer0 weights || layer1 weights
rope_angles
final_hidden
cache_pair = layer0 KV cache || layer1 KV cache
```

Then prove the generated TAP offsets point inside the packed pair buffers:

```bash
rg -n "aie.runtime_sequence|dma_bd\\(%arg" build_qwen3_persistent_two_layer/*.mlir
```

Accepted evidence:

```text
aie.runtime_sequence(%arg0 hidden, %arg1 weight_pair, %arg2 angles,
                     %arg3 final_hidden, %arg4 cache_pair)
preflight: ok runtime_memrefs=5 arg_specs=5 metadata_host_bos=5
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

The persistent graph probe made this threshold concrete for a current-KV-like
stream:

```bash
python iron/applications/qwen3_0_6b/persistent/graph_probe.py \
  --patterns separate repeat grouped \
  --layers 8 9 28 64 \
  --group-layers 4 \
  --preflight-only \
  --clean-build
```

Accepted evidence:

```text
8 separate layer transfers compile: max_dma_tasks_per_fifo=8
9 separate layer transfers fail preflight: max_dma_tasks_per_fifo=9
28 grouped-by-4 transfers compile: max_dma_tasks_per_fifo=7
64 grouped-by-4 transfers fail preflight: max_dma_tasks_per_fifo=16
64-layer repeated TAP compiles: max_dma_tasks_per_fifo=1
real Qwen3 chunk=8 compiles/runs after segment-major weights and grouped cache
DMA: max_dma_tasks_per_fifo=2
real Qwen3 chunk=28 compiles/runs after full-depth cache TAP grouping:
max_dma_tasks_per_fifo=1
```

So the current Qwen3 preflight limit is eight DMA tasks per FIFO. This is not a
general XDNA architectural constant; it is the measured safe boundary for the
Qwen3 persistent graph shapes in this repo.

Do not infer from the synthetic grouped-by-4 result that the real graph should
also use grouped-by-4 forever. The real chunk=28 experiment showed that the
same FIFO still emitted too many writeback tasks at full depth. The diagnostic
sequence is:

```text
1. Count DMA tasks per FIFO in generated MLIR.
2. If the same FIFO gets many similar tasks, try one legal full-depth TAP.
3. Re-run preflight and token verification before treating the new TAP as a
   performance path.
```

## 16. Validate TAP Against NPU BD Limits

Use when NPU lowering rejects a generated `aie.dma_bd`.

Checks from the score bring-up:

```text
Non-unit dimensions cannot use stride=0.
Large flattened dimensions such as size=8192 can be illegal.
Break large contiguous blocks into [row, dim] dimensions.
Repeated scatter dimensions can still exhaust BD IDs even when task count is
small; inspect repeat_count and the generated bd_dim_layout_array.
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
DMA task count per FIFO <= measured Qwen3 BD budget
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

For synthetic scaling work, `graph_probe.py --preflight-only` now generates MLIR
and runs this check before invoking `aiecc`. This avoids hiding the root cause
behind the long NPU lowering error dump.

## 34. Validate Packed Weight Artifact Before Runtime

Use when a persistent generate path starts from preprocessed weights on disk.

Diagnostic command used:

```bash
source /opt/xilinx/xrt/setup.sh
.venv/bin/python iron/applications/qwen3_0_6b/persistent/main.py \
  --model Qwen/Qwen3-0.6B \
  --prepare-weights \
  --packed-weights-dir build_qwen3_packed_weights_test
```

The manifest must prove these facts before any NPU run:

```text
format == qwen3_iron_packed_weights_v1
dtype == bfloat16
weight_order matches pack_full_layer_weights()
num_layers matches config
per_layer_numel matches the compiled op
total file bytes == manifest total_bytes
each layer offset == layer_id * per_layer_numel
each layer byte offset is 64B aligned
```

Recheck with exact slicing:

```text
packed_weight_layer_slice(layer_i) == pack_full_layer_weights_for_layer(model, i)
```

This catches wrong packed offsets as a Python error instead of letting a legal
but wrong DMA stream corrupt later layer numerics.

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
rg -n "qwen3_rc_o_weight|rt.fill\\(" iron/applications/qwen3_0_6b/persistent/design.py
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

## 35. Probe Real Graph Column Scaling

Use when decode is correct but the measured NPU time is still the dominant
cost. Do not infer the next performance direction from synthetic chunk tests
alone; compile/preflight the real stage variants.

Diagnostic command used:

```bash
source /opt/xilinx/xrt/setup.sh
. .venv/bin/activate
python iron/applications/qwen3_0_6b/persistent/real_graph_probe.py \
  --stages qkv mlp-gate-up full-mlp n-layer-final-only \
  --columns 1 2 4 8 \
  --preflight-only \
  --allow-failures \
  --clean-build
```

Accepted evidence:

```text
real_graph_probe: ok stage=qkv cols=4 ... compute_cores=14
real_graph_probe: ok stage=mlp-gate-up cols=2 ... compute_cores=9
real_graph_probe: ok stage=full-mlp cols=4 ... compute_cores=13
real_graph_probe: ok stage=n-layer-final-only cols=1 ... compute_cores=19
real_graph_probe: fail stage=n-layer-final-only cols=2 ...
  ValueError: n-layer final-only attention path is currently single-column only
```

Interpretation:

```text
If producer subgraphs scale to multiple columns but the full graph is rejected
by a single-column guard, the next speed work is to restructure the attention
closure and its placement. Re-running chunk sweeps will not unlock the unused
columns.
```

The reusable probe is:

```text
iron/applications/qwen3_0_6b/persistent/real_graph_probe.py
```

## 36. Decouple A Shared Column Knob

Use when one public parameter controls several unproven subgraphs and a direct
change would make failures ambiguous.

The n-layer column-scaling experiment used this method:

```text
requested --num-aie-columns=2 or 4
attention path internally stays num_columns=1
MLP down path receives mlp_down_columns=2 or 4
layer_iterations>1 fails early until a residual join exists
```

Accepted evidence:

```text
n-layer-final-only layer_iterations=1 cols=2:
  preflight ok, chunk_hidden_errors=0

n-layer-final-only layer_iterations=1 cols=4:
  preflight ok, chunk_hidden_errors=0
```

This isolates the performance experiment to one boundary. If it fails, the
search space is the down weight TAP, compact residual tile drain, or broadcast
from `ffn_hidden`, not Q/K/V head ownership or attention context packing.

## 37. Treat AIECC BD Legality As A Separate Check

Use when `real_graph_probe.py --preflight-only` accepts a graph but full
`aiecc` lowering fails in `aie.dma_bd`.

Observed command:

```bash
source /opt/xilinx/xrt/setup.sh
. .venv/bin/activate
python iron/applications/qwen3_0_6b/persistent/main.py \
  --stage n-layer-final-only \
  --layer-chunk-size 2 \
  --num-aie-columns 2 \
  --verify \
  --clean-build
```

Observed failure:

```text
'aie.dma_bd' op Stride 3 exceeds the [1:1048576] range
```

Diagnosis:

```text
The generated MLIR had a legal high-level TensorAccessPattern, but lowering
could not encode the layer stride into a BD. This is different from FIFO
object size, tile input count, or DMA task count.
```

Fix pattern:

```text
If a repeated TAP has a huge stride, first try splitting it into a bounded
number of linear fills. Then re-run preflight to make sure max_dma_tasks_per_fifo
stays under the measured safe limit.
```

## 38. Identify Which Placer Resource Was Exhausted

Use when `resolve_program()` fails before MLIR preflight with a generic
SequentialPlacer message:

```text
Failed to find a tile matching column N: tried until column 8
```

Do not assume this means "too many compute workers." Monkey-patch
`SequentialPlacer._place_endpoint` in a one-off probe to print whether the
failed endpoint is a runtime input/output endpoint, a memtile endpoint, or a
compute endpoint:

```python
from aie.iron.placers import SequentialPlacer

orig_place = SequentialPlacer._place_endpoint

def debug_place(self, ofe, tiles, common_col, channels, device,
                output=False, link_tiles=[], link_channels={}):
    try:
        return orig_place(self, ofe, tiles, common_col, channels, device,
                          output, link_tiles, link_channels)
    except Exception:
        print("endpoint=", repr(ofe))
        print("output=", output)
        print("remaining_tiles=", tiles)
        print("channels_used=", {str(k): sum(c for _, c in v)
                                 for k, v in channels.items()})
        raise

SequentialPlacer._place_endpoint = debug_place
```

Evidence from the packed MLP2 experiment:

```text
endpoint=<aie.iron.runtime.endpoint.RuntimeEndpoint ...>
output=True
remaining_tiles=[]
```

Interpretation:

```text
The graph exhausted shim/runtime output endpoints because it emitted too many
Runtime.fill tasks. The correct fix was packed host-side weight layout, not
changing the compute kernel.
```

## 39. Treat Tile Input/Output Count As A Packing Constraint

Use when preflight reports:

```text
Compute tile %tile_X_Y has 3 input ObjectFIFOs; limit=2
Compute tile %tile_X_Y has 3 output ObjectFIFOs; limit=2
```

Diagnostic command:

```bash
rg -n "tile_X_Y|objectfifo @qwen3_" build_dir/*.mlir
```

The gate/up shard experiment found:

```text
bad input shape:
  xnorm FIFO
  gate_weight FIFO
  up_weight FIFO

bad output shape:
  post_norm FIFO
  gate_up_shard_0 FIFO
  gate_up_shard_1 FIFO
```

Fix pattern:

```text
Pack logically adjacent streams before the tile:
  gate rows followed by up rows in one per-column weight FIFO

Move small metadata that would create a third output to runtime or packed
layout:
  post_norm weights as one contiguous segment
```

Recheck:

```text
preflight: max_tile_inputs=2 max_tile_outputs=2
```

## 40. Benchmark Routing Workers Before Accepting Them

Use when a graph is correct after adding a copy/split/join Worker to reduce
DMA tasks or tile ports.

Accepted rule from the MLP2 experiment:

```text
If the routing Worker touches model weights, run timing before accepting it.
```

Evidence:

```text
NPU-side gate/up split-copy:
  cols=2 layers=2 verify passes
  npu_time_us about 152935

Host-packed MLP2:
  cols=2 layers=2 verify passes
  late iterations about 9837-9916 us
```

Interpretation:

```text
A routing Worker is reasonable for small activation tiles such as residual
joins. It is usually wrong for multi-megabyte weight repacking. Put that
layout work in the prepacked weight artifact or host packing step instead.
```
