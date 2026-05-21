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

Use when Python or XRT behaves differently across virtual environments.

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
find build_qwen3_persistent -maxdepth 2 -type f \
  \( -name '*.mlir' -o -name '*.xclbin' -o -name '*.bin' \) | sort
```

Rules confirmed during bring-up:

```text
Unexpectedly tiny compile time after a graph edit means cached artifacts may
have been reused.

Use --clean-build or a new build directory after graph, scratch layout, or
runtime buffer changes.
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
.venv/bin/python iron/applications/qwen3_0_6b/persistent/main.py \
  --model Qwen/Qwen3-0.6B \
  --stage input-rmsnorm-qkv \
  --build-dir build_qwen3_persistent_debug \
  --verify \
  --verify-repeat 3
```

This method localized one wrong-token run to the Q/K path before attention.

## 7. Repeat The Same Input

Use when a single run is wrong but the wrongness may depend on state.

```bash
source /opt/xilinx/xrt/setup.sh
.venv/bin/python iron/applications/qwen3_0_6b/persistent/main.py \
  --model Qwen/Qwen3-0.6B \
  --stage input-rmsnorm-qkv \
  --verify \
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
weight_disk_load_s
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

## 33. Prove Packed Weight BO Slices With Token Match

Use after introducing a global packed weight artifact. The first runtime proof
should still use the known-good single-layer Program, but each layer's weight
argument should be an XRT sub-buffer of one parent packed-weight BO.

Diagnostic command used:

```bash
source /opt/xilinx/xrt/setup.sh
.venv/bin/python iron/applications/qwen3_0_6b/persistent/main.py \
  --stage generate \
  --fast-generate \
  --require-packed-weights \
  --verify-generate \
  --max-new-tokens 3 \
  --packed-weights-dir build_qwen3_packed_weights_test \
  --build-dir build_qwen3_persistent_generate \
  --prompt 'Count from one to five.' \
  --raw-prompt
```

Accepted evidence:

```text
fast_generate_weight_source: packed_artifact
fast_generate_weight_pack_s: 0.000000
token_match: True for positions 6 and 7
```

If this fails while manifest exact-slice tests pass, inspect XRT sub-buffer
offsets and sync state before changing AIE kernels.

## 34. Warm Up Column-Scaling Measurements

Use when a graph change verifies numerically but a single timing sample says
the new column count is slower.

Run repeated iterations from the same compiled artifact:

```bash
source /opt/xilinx/xrt/setup.sh
.venv/bin/python iron/applications/qwen3_0_6b/persistent/main.py \
  --stage post-attn-rmsnorm-full-mlp \
  --num-aie-columns 4 \
  --verify \
  --verify-repeat 5 \
  --build-dir build_qwen3_full_mlp_cols4_verify
```

Confirmed failure mode:

```text
single clean cols=4 run: npu_time_us about 3716
repeat cols=4 late iterations: about 1609-1691
```

For performance decisions, compare warm iterations or a latency distribution,
not iteration 0. Keep `--verify` enabled while changing graph shape so a fast
number does not hide a broken FIFO/TAP layout.

## 42. Run A Phase Sensitivity Probe Before Widening

Use when the graph is correct, NPU time dominates, and the next performance
target is unclear. This is a dynamic probe built from existing verified stages;
it is not hardware trace and not a strict additive decomposition.

Command used:

```bash
source /opt/xilinx/xrt/setup.sh
. .venv/bin/activate

python iron/applications/qwen3_0_6b/persistent/phase_timing_probe.py \
  --repeat 5 \
  --json-output build_qwen3_phase_probe_summary.json
```

For the current direct gate/up+SiLU baseline, add:

```bash
python iron/applications/qwen3_0_6b/persistent/phase_timing_probe.py \
  --repeat 5 \
  --mlp-gate-up-direct-silu \
  --build-dir-prefix build_qwen3_phase_probe_direct \
  --json-output build_qwen3_phase_probe_direct_summary.json
```

What it runs:

```text
full_layer_current_l1:
  current one-layer n-layer-final-only graph

attention_only_current_l1:
  same graph stopped after attention/O-proj/residual

qkv_standalone_l1:
  standalone input RMSNorm + QKV projection

mlp_full_standalone_l1:
  standalone post-attention RMSNorm + full MLP

mlp_gate_up_standalone_l1:
  standalone post-attention RMSNorm + gate/up + SiLU/mul

mlp_down_standalone_l1:
  standalone MLP down projection + residual-add
```

Read the result as a sensitivity test:

```text
current full-layer - current attention-only:
  meaningful MLP-side increment for the current graph family

standalone stage times:
  useful diagnostic signals, but not an additive timing split
```

Current direct-graph evidence:

```text
current full-layer warm median:    3.896 ms
current attention-only median:     1.684 ms
current MLP-side median increment: 2.212 ms
MLP median-increment share:        56.8%

standalone QKV warm median:        0.823 ms
attention-only median minus QKV:   0.861 ms
```

Diagnosis:

```text
The next local optimization should stay on the MLP side unless hardware trace
contradicts this. QKV has the highest call count statically, but this probe
shows the current-graph MLP-side increment is still larger than the
attention-only side.
```

Two caveats:

```text
Do not pass --require-packed-weights to non-fast-generate stage probes. The CLI
guard rejects it because packed-weight enforcement is only used by
--fast-generate.

Standalone MLP/down probes can show higher variance than current-graph probes.
Use warm median/min and the current full-vs-attention differential for target
selection.
```

## 48. Treat Control Packets As A Last-Resort Runtime Patch Path

Use this when considering `NpuControlPacketOp` to avoid per-position rebuilds.

Observed constraint:

```text
NpuControlPacketOp can write 32-bit payloads to tile registers, so in theory it
can rewrite DMA Buffer Descriptor fields such as transfer size, stride, repeat,
or offset.
```

Why it was not used as the first fix:

```text
IRON ObjectFifo/Runtime does not expose a safe "rewrite BD while quiescent"
workflow.
The compiler currently emits control packets as static initialization data.
Using it directly requires proving tile register addresses, payload encoding,
stream routing, and that no DMA engine is reading the BD being modified.
```

Decision rule:

```text
Prefer artifact diff, runtime metadata, or bucket variants first. Only use
control packets after a small isolated BD rewrite proof demonstrates:
  exact target BD register address
  before/after register payload
  DMA quiescence point
  token-correct run after the rewrite
```

## 49. Cleanup Runtime Between Independent Artifact Sweeps

Use when one diagnostic script compiles or loads many unrelated xclbins and
later configurations fail in XRT hardware-context creation.

Symptom:

```text
DRM_IOCTL_AMDXDNA_CREATE_HWCTX IOCTL failed (err=-22): Invalid argument
```

Check:

```text
Run the failing configuration by itself in a fresh process. If it passes, the
operator shape is not the root cause.
```

Fix for sweep scripts:

```python
try:
    run_one_operator_configuration()
finally:
    aie_utils.DefaultNPURuntime.cleanup()
```

Do not use this as a performance hot-loop mechanism. It is for diagnostic
sweeps that intentionally load many different artifacts.
