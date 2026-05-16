<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Qwen3 Megakernel Diagnostic Methods

These are reusable checks that have already found real Qwen3 megakernel bugs in
this tree.

## Method Selection

| Failure class | Use methods |
| --- | --- |
| Environment or Python/XRT binding mismatch | 1, 2 |
| Cached build artifacts or stale graph | 3 |
| Runtime BO metadata crash | 4 |
| Wrong final token or stage value | 5, 7, 8 |
| Runtime patch-site risk | 6 |
| Placement, ObjectFIFO, DMA, or L1 resource failure | 9, 11, 13, 14, 15, 16 |
| Full-ELF scratch layout changes correctness | 10 |
| Operator-specific numeric mismatch | 12 |
| Persistent phase ordering or timeout | 17 |
| Structured attention-score mismatch | 18 |

## 1. Classify The Failure Boundary First

Use this order before changing code:

```text
environment/API failure
compile or placement failure
host runtime argument failure
host buffer ABI failure
data movement or first-run state failure
numeric semantic failure
performance/placement scaling issue
```

The command output should decide the bucket. Do not infer it from the operator
name that happened to be running.

## 2. Probe pyxrt Capabilities

Use when compile-only succeeds but full-ELF runtime cannot start.

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

Used result:

```text
Python 3.12 binding missed pyxrt.elf and pyxrt.ext.
Python 3.14 binding provided both.
```

Recheck after rebuilding the environment:

```bash
source /opt/xilinx/xrt/setup.sh
.venv/bin/python -m pytest iron/operators/mem_copy/test.py -q --iterations 1
```

## 3. Inspect Artifacts Before Rerunning

Use when graph edits appear to have no effect.

```bash
find build_qwen3_megakernel -maxdepth 2 -type f \
  \( -name '*.mlir' -o -name '*.elf' \) | sort
```

Rules confirmed during bring-up:

```text
Unexpectedly tiny compile time after a graph edit means cached artifacts may
have been reused.

Use --clean-build or a new build directory after runlist, scratch layout, or
runtime patch changes.
```

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

## 5. Add Stage-Local Debug Drains

Use when final logits or tokens are wrong.

Do not make the real scratch tensor an output directly if that changes its
buffer class or lifetime. Add a pass-through debug copy immediately after the
semantic value is produced.

QKV debug points used:

```text
x_norm
queries_raw
queries_norm
queries
keys_raw
keys_norm
keys
values
```

Command used:

```bash
source /opt/xilinx/xrt/setup.sh
.venv/bin/python iron/applications/qwen3_0_6b/qwen3_megakernel.py \
  --model Qwen/Qwen3-0.6B \
  --num-layers 1 \
  --max-seq-len 256 \
  --build-dir build_qwen3_megakernel_debug \
  --debug-stage qkv \
  --dump-patches \
  --verify-repeat 3
```

This method localized one wrong-token run to the Q/K path before attention.

## 6. Assert Runtime Patch Sites

Use when the fused graph patches runtime constants into the ELF.

Patch families used by the one-layer decode graph:

```text
cache write byte offsets for current decode position
softmax active sequence length
```

Expected one-layer signals:

```text
2 key-cache patch sites per layer
2 value-cache patch sites per layer
2 zero-base cache patch sites from shared StridedCopy design
num_layers + 1 softmax patch sites
no duplicate cache patch locations
```

Patch locations can move when the runlist changes. Counts and target buffer
identities should not.

## 7. Repeat The Same Input

Use when a single run is wrong but the wrongness may depend on state.

```bash
source /opt/xilinx/xrt/setup.sh
.venv/bin/python iron/applications/qwen3_0_6b/qwen3_megakernel.py \
  --model Qwen/Qwen3-0.6B \
  --num-layers 1 \
  --max-seq-len 256 \
  --verify-one-step \
  --verify-repeat 3
```

This exposed the parent-BO dirty bug:

```text
iteration 0: x_norm and qkv outputs all zero
iteration 1: qkv values mostly aligned
```

The first bad tensor was before GEMV, so the fix belonged to host buffer sync,
not Q/K/V math.

## 8. Use Local Reference And Full Reference Separately

Use when a stage after an accepted stage reports numeric mismatch.

Pattern:

```text
local reference: uses actual NPU output from the previous accepted stage
full reference: uses the all-PyTorch path and measures accumulated drift
```

For persistent QKV, the decisive local check was:

```python
torch_from_npu_x = F.linear(npu_x_norm.view(1, 1, -1), W).flatten()
```

This proved Q/K/V projection was locally exact while full-reference drift came
from upstream RMSNorm precision.

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

## 10. Treat Layout-Only Changes As Correctness Changes

Use when a change claims to affect only layout or scratch names.

Clean A/B recheck is required because full-ELF fusion makes layout,
patch-sites, and scratch lifetime part of correctness.

Reverted examples:

```text
RoPE LUT one row per head
separate attention and MLP x_norm buffers
one RoPE(rows=1) call per head
```

## 11. Check ObjectFIFO Consumers In Generated MLIR

Use after adding broadcast or multiple debug/cache drains.

```bash
rg -n "objectfifo @qwen3_rc_" build_qwen3_persistent/*.mlir
```

This found a real bug before runtime: two drains were written against the same
`k_rope` and `v` FIFO consumer, but the MLIR exposed only one shim consumer.
The fix was to make cache the only drain for those streams and verify semantic
values from the cache slice.

Rule:

```text
If two logical consumers are intended, generated MLIR must show two consumer
endpoints or an explicit forward/split/join dataflow. Reusing `.cons()` twice
from the same endpoint is not proof of broadcast.
```

## 12. Use Operator-Specific Tolerance

Use when local reference is correct but a downstream math operator reports a
small number of numeric mismatches.

Checks used:

```text
1. Compare against a host reference that follows the external kernel's operation
   order.
2. Check that operator's existing test tolerance.
3. Only then update the persistent-stage verifier.
```

This diagnosed the RoPE mismatch. Existing RoPE tests use
`rel_tol=0.05, abs_tol=0.5`; applying GEMV's `abs_tol=1e-6` to RoPE produced
false failures.

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

## 17. Treat Runtime Phase Assumptions As Suspect

Use when adding a later `rt.start()` or second task group causes a timeout.

Inspection command:

```bash
sed -n '350,470p' build_qwen3_persistent/*.mlir
rg -n "core|start|dma_await_task" build_qwen3_persistent/*.mlir
```

If workers appear as persistent `aie.core` loops, a later runtime start is not a
phase barrier. Add an explicit FIFO dependency or restructure the dataflow so
the always-running workers can safely block.

## 18. Use Error Cardinality To Find Layout Bugs

Use when a numeric mismatch has a structured count.

Example:

```text
216 attention score mismatches = 8 odd heads * 27 valid positions
```

This points at GQA pair ordering or qk-pair packing, not at all Q heads, not at
the K-cache prefix, and not at the final output drain.
