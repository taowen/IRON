<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Qwen3 Megakernel Runtime Diagnostic Methods

[Back to method index](diagnostic-methods.md).

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
.venv/bin/python iron/applications/qwen3_0_6b/full_elf/main.py \
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
.venv/bin/python iron/applications/qwen3_0_6b/full_elf/main.py \
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

## 31. Clone XRT Tensor Views Before Crossing Debug Boundaries

Use when a tensor returned by `XRTTensor.to_torch()` is carried into a later
layer, saved to disk, or passed to a separate diagnostic.

`XRTTensor.to_torch()` returns a zero-copy torch view over the mapped XRT BO.
`contiguous()` does not copy when the view is already contiguous. If the BO is
destroyed at the end of the loop iteration, the torch tensor can still point at
unstable mapped memory.

Symptom found:

```text
layer_17_qkv_diagnostic_bundle_begin: .../qkv_boundary_layer_17.npz
layer_17_qkv_diagnostic_bundle_tensor_begin: hidden
process exits with code -1 and no Python traceback
```

Accepted fix:

```python
def host_owned_tensor(tensor):
    return tensor.detach().clone().contiguous()

current_hidden = host_owned_tensor(actual["layer_residual"])
full_layer_v = host_owned_tensor(actual["v_context_stream_current"])
```

Recheck:

```text
layer_17_qkv_diagnostic_bundle_tensor_done: hidden shape=(1024,)
...
layer_17_qkv_diagnostic_bundle: build_qwen3_persistent_multilayer/diagnostics/qkv_boundary_layer_17.npz
```

Do this before interpreting a serialization crash as an NPU dataflow failure.

## 32. Split Wall Time From NPU Time

Use when decode is numerically correct but throughput is far worse than the
sum of AIE kernel times suggests.

Diagnostic command used:

```bash
source /opt/xilinx/xrt/setup.sh
.venv/bin/python iron/applications/qwen3_0_6b/persistent/main.py \
  --stage generate \
  --fast-generate \
  --verify-generate \
  --max-new-tokens 4 \
  --build-dir build_qwen3_persistent_generate \
  --prompt 'Count from one to five.' \
  --raw-prompt
```

Before the fast path, persistent generate spent about 1.3-1.6 s of wall time
per decoded token while the reported NPU layer time was about 250 ms. That gap
was too large to explain with compute kernels.

Break the wall time into these buckets before changing the graph:

```text
weight_pack_s
weight_xrt_s
cache_xrt_s
hidden_sync_s
rope_sync_s
op_call_s
output_drain_s
cpu_final_lm_head_s
```

The diagnosed bottleneck was repeated host/runtime work: per-token, per-layer
weight packing, `XRTTensor` creation, and full KV-cache drain/fill. Reusing
packed weight BOs and keeping each layer's cache as an XRT inout buffer changed
the measured decode wall time to about 0.25-0.27 s/token while preserving token
matches against the cached CPU reference.

Do this before moving math into a larger megakernel. If wall time is dominated
by Python/XRT setup, changing the external kernel arithmetic will not address
the observed problem.
