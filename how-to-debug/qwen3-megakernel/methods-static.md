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

The same method caught the first production input-QKV fusion attempt:

```text
MLIR: 6 runtime memrefs
operator arg spec: 6
main_kernels.json: bo0..bo4 only
```

The correct fix was again to reduce the runtime ABI, not to add another host
BO:

```text
qkv_packed_input = input_metadata || q_proj || k_proj || v_proj
past_stream
o_pack
mlp_pack
packed_output
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

The same method diagnosed the A0 fixed-chunk attention experiment. The failed
graph had four input FIFOs into one Worker:

```text
Q
K chunk
V chunk
mask chunk
```

The accepted graph packed `K/V/mask` into one chunk stream and kept `Q` as the
second input. This let the same fixed-TAP attention artifact run multiple
positions using mask data only.

The D1.3c production MLP graph used the same method twice:

```text
1. aiecc named tile_1_2 as output-DMA exhausted. Generated MLIR showed that
   one Worker produced four FIFOs: attn_residual_debug, final_residual,
   ffn_hidden, and ffn_hidden_debug. The fix was to split post-norm and
   gate/up into separate Workers and keep production debug drains bounded.

2. Preflight rejected tile_1_3 with three input ObjectFIFOs: xnorm,
   gate_weight, and up_weight. The fix was to pack gate/up row groups into
   one FIFO and use a pair kernel, not to change TAP strides or placement.
```

Rule:

```text
For production single-graph growth, count endpoint fan-in/fan-out from the
generated MLIR after every new phase. Debug drains count as real endpoints.
```

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

Do not assume this means "too many compute workers." First use the built-in
real graph probe trace:

```bash
source /opt/xilinx/xrt/setup.sh
. .venv/bin/activate
python iron/applications/qwen3_0_6b/persistent/real_graph_probe.py \
  --stages n-layer-final-only \
  --columns 2 \
  --attention-columns 2 \
  --layer-iterations 1 \
  --preflight-only \
  --allow-failures \
  --trace-placement
```

If you need a custom probe, monkey-patch
`SequentialPlacer._place_endpoint` to print whether the failed endpoint is a
runtime input/output endpoint, a memtile endpoint, or a compute endpoint:

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

Attention2 + full MLP2 used the same method again:

```text
before fix:
  runtime_output=16 placed, then the 17th output endpoint failed

accepted one-layer fix:
  combine hidden and QK/RoPE metadata into one runtime input and split it
  inside the graph
  placement_trace_counts runtime_output=16, runtime_input=6, other_input=1
```

Do not treat every endpoint overflow as permission to combine large hot
streams. The attempted K/V cache-pair stream passed placement, but exposed BD
dimension and access-order problems. Prefer combining small metadata streams
first, then verify with full compile and local numeric boundaries.

The three-way gate/up branch repeated this class:

```text
accepted direct gate/up+SiLU baseline:
  placement_trace_counts runtime_output=16, runtime_input=5, other_input=1

three-way gate/up:
  placement_trace_fail_key=runtime_output
  placement_trace_fail_type=RuntimeEndpoint
  placement_trace_fail_common_col=3
  placement_trace_fail_counts runtime_output=16, runtime_input=6, other_input=1
```

This proved the branch was blocked by an extra host->NPU weight stream before
any numerical or C++ kernel claim could be made. The next version must reduce
runtime endpoints, for example by packing/splitting gate/up shard streams,
before adding another gate/up Worker.

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

## 41. Estimate Static Work Before Choosing A Widening Target

Use when a graph fits and runs, but the next speed branch is unclear.

```bash
python iron/applications/qwen3_0_6b/persistent/work_estimator.py \
  --mlir build_qwen3_score_softmax_fused_generate_default/\
Qwen3PersistentNLayerFinalOnly_h1024_q2048_kv1024_hd128_msl256_pos26_\
ffn3072_col2_attncol2_mlpgatecol2_attnprobe0_tsi4_tso128_\
epsilon1en06_layers28_npu2.mlir \
  --layers 28 \
  --position 26 \
  --show-kernels
```

What the estimator does:

```text
parse generated MLIR
count func.call operations under finite scf.for trip counts
ignore the persistent infinite outer loop
apply simple Qwen3-specific work models for matvec, attention QK/PV, softmax,
copy, RMSNorm, and join kernels
```

What it is not:

```text
It is not hardware trace, does not model NoC stalls, and does not prove runtime
latency. Use it to choose the next experiment, then validate with preflight,
token checks, and timing.
```

Current accepted graph result at position 26:

```text
MLP gate/up matvec:      176.161M estimated MACs/token
QKV projection matvec:   117.441M estimated MACs/token
MLP down matvec:          88.080M estimated MACs/token
O-proj matvec:            58.720M estimated MACs/token
attention QK + PV:         3.096M estimated MACs/token
```

Diagnosis:

```text
The next short-position graph-body target is MLP gate/up. Direct O-proj widening
is lower priority, and short-position attention QK/PV is not the current
compute target.
```

## 43. Read AIE API Compile Errors As Kernel-Boundary Evidence

Use when MLIR/preflight succeeds but full compile fails inside an external
kernel.

The direct gate/up+SiLU experiment hit this after the graph had already passed
preflight:

```text
preflight:
  compute_cores=30
  max_tile_inputs=2
  max_tile_outputs=2

clang++:
  error: no matching function for call to 'tanh'
  candidate template ignored: could not match 'vector<float, Elems>' against
  'float'
```

Root cause:

```text
The AIE API tanh overload used in this repo is vector-only. The fused kernel
reduced each gate row to a scalar and then tried to call aie::tanh(float).
This was not an IRON graph/resource failure and not a token-level numeric
failure. It was an external-kernel API boundary.
```

Fix pattern:

```text
1. Read the exact overload error and inspect the existing repo kernel that
   implements the same operation.
2. Preserve the existing approximation path when possible.
3. Recompile the real full graph, not only a host C++ snippet.
```

Accepted fix:

```text
Broadcast the scalar bf16 gate/up row results into a 16-lane vector.
Run the same tanh-form SiLU approximation as qwen3_silu_mul_bf16.
Store lane 0 into the direct hidden shard.
```

Second-order trap:

```text
The first fix placed vector constants in the wrong function scope. The next
clang++ failure was:
  use of undeclared identifier 'silu_vec_len'
  use of undeclared identifier 'half'
  use of undeclared identifier 'one'

That was a C++ macro/scope issue, not a new IRON placement problem.
```

Recheck:

```text
clang-format --dry-run --Werror aie_kernels/generic/qwen3_attention.cc
full generate compile/run with --mlp-gate-up-direct-silu
token_match=True before any performance claim
```

The D1.3c production bring-up hit a smaller version of the same class while
adding helper kernels to `fixed_attention.cc`:

```text
clang++:
  no matching function for call to 'add'
  no member named 'to_vector' in 'aie::vector<__bf16, 16>'
```

Evidence:

```text
The existing qkv-rope-attention-o-fused stage failed at external-kernel compile
even though the new MLP graph was not yet connected. The whole source file is
compiled into fixed_attention.o, so unused helper functions can break older
stages.
```

Fix used:

```text
Mirror the already-working repo kernels:
  rms_norm.cc uses 16-lane mul_square into vector<float>
  silu.cc keeps tanh input as an expression that supports to_vector<float>()

Then re-run an older accepted stage as a compile gate before wiring the new
graph.
```

## 44. Full AIECC After ObjectFIFO Shape Changes

Use when a graph changes ObjectFIFO object type, depth, or acquire count.
Python preflight is necessary, but it does not prove that the later
`AIEObjectFifoStatefulTransform` resource allocation can assign all memory
blocks and BDs.

Command pattern:

```bash
source /opt/xilinx/xrt/setup.sh
. .venv/bin/activate

python iron/applications/qwen3_0_6b/persistent/real_graph_probe.py \
  --stages n-layer-final-only \
  --columns 2 \
  --attention-columns 2 \
  --mlp-gate-up-columns 2 \
  --mlp-gate-up-pair-rows \
  --mlp-gate-up-direct-silu \
  --mlp-gate-up-row-group 8 \
  --layer-iterations 28 \
  --preflight-only \
  --trace-placement \
  --build-dir build_qwen3_mlp_gateup_rg8_preflight \
  --clean-build

PYTHONUNBUFFERED=1 python -X faulthandler \
  iron/applications/qwen3_0_6b/persistent/main.py \
  --stage generate --fast-generate --verify-generate \
  --max-new-tokens 2 \
  --layer-chunk-size 28 \
  --num-aie-columns 2 \
  --attention-columns 2 \
  --mlp-gate-up-columns 2 \
  --mlp-gate-up-pair-rows \
  --mlp-gate-up-direct-silu \
  --mlp-gate-up-row-group 8 \
  --require-packed-weights \
  --build-dir build_qwen3_mlp_gateup_rg8_generate_default \
  --clean-build
```

The row-group 8 experiment proved why both commands are needed:

```text
preflight:
  ok, max_fifo_buffered_bytes=32768, max_dma_tasks_per_fifo=1

full aiecc:
  error: 'aie.mem' op has more than 16 blocks
  note: no space for this BD
```

Diagnosis:

```text
The FIFO object byte budget was not the failing resource. The graph created too
many individual FIFO blocks on the memory tile by using depth=16 of
hidden_weight_ty objects. This only surfaced in the full AIE ObjectFIFO
stateful transform.
```

Fix pattern:

```text
If the logical kernel wants many adjacent rows, prefer fewer larger FIFO
objects over a deep FIFO of single rows, as long as the stream layout remains
block-aligned:

bad for row-group 8:
  16 objects of hidden_weight_ty

better:
  4 objects of (4, hidden_size) bf16
```

Do not accept the branch after compile/token correctness alone. Re-run the
static estimator and timing; the fixed row-group 8 graph reduced calls/token
but still slowed the default prompt, so it was rejected as a performance path.

## 45. Use ObjectFifo Split To Reduce Runtime Endpoints

Use when a graph fails because a new independent `Runtime.fill` would consume
one more host->NPU endpoint, but the data can be packed as adjacent slices of
one logical stream.

Pattern used for the three-way gate/up follow-up:

```python
parent = ObjectFifo(pair_ty, name="gate_up_weight_12_pair", depth=8)
child1, child2 = parent.cons().split(
    offsets=[0, hidden_size],
    obj_types=[hidden_weight_ty, hidden_weight_ty],
    names=["gate_up_weight_1", "gate_up_weight_2"],
    depths=[8, 8],
)
```

Host-side packing must match the split object boundary exactly:

```text
[child1 object 0][child2 object 0]
[child1 object 1][child2 object 1]
...
```

Recheck sequence:

```text
1. layout test: segment-major packing from model weights equals packing from
   the prepacked layer-major artifact
2. real_graph_probe --preflight-only --trace-placement
3. full generate compile/run, because preflight does not prove all AIECC
   ObjectFIFO lowering constraints
4. token_match=True before timing
5. same-prompt timing against the accepted baseline
```

Evidence:

```text
independent three-way gate/up:
  fails placing the 17th RuntimeEndpoint output

split parent stream:
  preflight ok
  placement_trace_counts runtime_output=16, runtime_input=5, other_input=2
  token_match=True
```

Decision rule:

```text
Endpoint reduction is a resource fix, not a speed claim. The three-way split
graph solved placement and matched tokens, but it was slower than the accepted
two-way graph on the Fibonacci multi-token prompt. Keep the method; reject the
branch unless measured token time improves.
```

## 46. Diff Position Artifacts Before Choosing Patch Or Buckets

Use when decode is correct but compile still happens per position. Do not
choose runtime instruction patching or bucketed precompile from intuition; first
compare the generated artifacts.

Generate adjacent same-bucket artifacts:

```bash
source /opt/xilinx/xrt/setup.sh
. .venv/bin/activate

python iron/applications/qwen3_0_6b/persistent/real_graph_probe.py \
  --stages n-layer-final-only \
  --columns 2 \
  --attention-columns 2 \
  --mlp-gate-up-columns 2 \
  --mlp-gate-up-pair-rows \
  --mlp-gate-up-direct-silu \
  --layer-iterations 28 \
  --position 26 \
  --build-dir build_qwen3_position_diff_full_pos26 \
  --clean-build

python iron/applications/qwen3_0_6b/persistent/real_graph_probe.py \
  --stages n-layer-final-only \
  --columns 2 \
  --attention-columns 2 \
  --mlp-gate-up-columns 2 \
  --mlp-gate-up-pair-rows \
  --mlp-gate-up-direct-silu \
  --layer-iterations 28 \
  --position 27 \
  --build-dir build_qwen3_position_diff_full_pos27 \
  --clean-build
```

Then summarize:

```bash
python iron/applications/qwen3_0_6b/persistent/artifact_position_diff.py \
  build_qwen3_position_diff_full_pos26 \
  build_qwen3_position_diff_full_pos27
```

Evidence from the accepted direct-SiLU graph:

```text
pos26 -> pos27:
  MLIR line_count=1434/1434
  changed_diff_lines=64
  runtime .bin changed_bytes=8
  runtime .bin has eight u32 patch candidates, each +256 bytes
  xclbin changed_bytes=126
  main_aie_cdo_elfs.bin changed_bytes=56
  six main_core_*.elf files changed
```

The artifact diff tool now also maps changed core-ELF bytes back to section and
symbol context. Same-bucket pos26 -> pos27 produced `.text` patch candidates in
these core functions:

```text
main_core_1_5.elf: core_1_5/FUNC, 16 changed instruction words
main_core_2_2.elf: core_2_2/FUNC,  4 changed instruction words
main_core_2_3.elf: core_2_3/FUNC,  8 changed instruction words
main_core_4_2.elf: core_4_2/FUNC, 16 changed instruction words
main_core_4_3.elf: core_4_3/FUNC,  4 changed instruction words
main_core_4_4.elf: core_4_4/FUNC,  8 changed instruction words
```

The observed u32 deltas were instruction encodings, not scalar data slots:

```text
core_elf_patch_candidate:
  section=.text
  symbol=core_1_5/FUNC
  value=54526600->56623752
  delta=2097152
```

Interpretation:

```text
The runtime instruction stream only needs current K/V DMA offset changes inside
the same cache block. But the AIE core ELFs also change because attention
score, V merge, context, and mask workers bake position or position+1 as
immediate integer arguments.
```

So this is not a pure `.bin` patch problem:

```text
patching only the runtime .bin is insufficient
bucketed precompile alone is insufficient while core ELFs bake exact position
raw ELF/CDO byte patching is not a safe first implementation target unless
  every changed instruction word is mapped to a stable relocation/encoding rule
```

Check a cache-block boundary separately:

```bash
python iron/applications/qwen3_0_6b/persistent/real_graph_probe.py \
  --stages n-layer-final-only \
  --columns 2 \
  --attention-columns 2 \
  --mlp-gate-up-columns 2 \
  --mlp-gate-up-pair-rows \
  --mlp-gate-up-direct-silu \
  --layer-iterations 28 \
  --position 63 \
  --preflight-only \
  --build-dir build_qwen3_position_diff_probe_pos63 \
  --clean-build

python iron/applications/qwen3_0_6b/persistent/real_graph_probe.py \
  --stages n-layer-final-only \
  --columns 2 \
  --attention-columns 2 \
  --mlp-gate-up-columns 2 \
  --mlp-gate-up-pair-rows \
  --mlp-gate-up-direct-silu \
  --layer-iterations 28 \
  --position 64 \
  --preflight-only \
  --build-dir build_qwen3_position_diff_probe_pos64 \
  --clean-build

python iron/applications/qwen3_0_6b/persistent/artifact_position_diff.py \
  build_qwen3_position_diff_probe_pos63 \
  build_qwen3_position_diff_probe_pos64
```

Observed boundary:

```text
pos63 -> pos64:
  changed_diff_lines=140
  loop_bound=12
  K/V cache read length 32768 -> 65536 bf16 elements
  score/context/V-merge loops 1 block -> 2 blocks
```

Decision rule:

```text
First move position and valid length into a numerically verified runtime path,
or prove a stable ELF/CDO patch rule. After that, choose between patching the
remaining `.bin` DMA offset words inside a cache block and precompiling one
variant per active cache-block count.

Until then, exact-position precompile is useful as a measurement/runtime
selection baseline only. It removes JIT compilation from the token loop, but it
does not solve artifact count or setup-time growth.
```

## 47. Check ObjectFIFO Object Alignment After Metadata Tails

Use this when a small metadata tail is appended to an existing ObjectFIFO
object.

Diagnostic used:

```text
memref<130xbf16> -> 260 bytes
memref<386xbf16> -> 772 bytes
memref<258xbf16> -> 516 bytes
```

These are not 16-byte multiples. The graph may still pass MLIR verification and
placement, so this must be a preflight check before runtime.

Implemented guard:

```text
Qwen3PreflightError:
  ObjectFIFO object alignment mismatch:
  object_bytes=..., expected a 16-byte multiple.
```

Result from the runtime-position metadata experiment:

```text
Padding the tail to 8 bf16 values fixed the alignment issue, but the branch
still produced NaN attention-probe output and full-layer timeout. Alignment was
a real preflight blind spot, not the final root cause.
```

Rule:

```text
Pad metadata tails to an aligned object size, then still run attention-probe or
single-layer generate. Passing alignment only proves the FIFO shape is less
suspicious; it does not prove numeric correctness.
```

## 50. Run Compileall Before NPU Debugging

Use this when a new CLI stage or production runner path fails before there is
clear MLIR, XRT, ObjectFIFO, or numeric evidence.

Command used:

```bash
. .venv/bin/activate
python -m compileall iron/applications/new-mega/production
```

This diagnosed the first `qkv-rope-attention` production integration failure:

```text
File "iron/applications/new-mega/production/main.py", line 182
    else:
    ^^^^
SyntaxError: invalid syntax
```

Root cause:

```text
The stage success-message dispatch was edited from two branches to three
branches but left as if/else/else. The failure boundary was the Python entry
script, not the NPU graph.
```

Fix proven by this method:

```text
Change the branch to if fixed-attention / elif qkv-rope-present / else
qkv-rope-attention. Re-run compileall before sourcing XRT and launching the
stage.
```
