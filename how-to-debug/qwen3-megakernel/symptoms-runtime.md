<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Qwen3 Megakernel Runtime Symptoms

[Back to symptom index](symptoms.md).

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

## Pytest Cannot Import pyxrt

Symptom:

```text
ImportError: Cannot import pyxrt (err=No module named 'pyxrt')... is XRT installed?
```

Diagnostic:

```bash
.venv/bin/python - <<'PY'
import os
print(os.environ.get("XILINX_XRT"))
print(os.environ.get("PYTHONPATH"))
PY
```

Then rerun with XRT sourced:

```bash
source /opt/xilinx/xrt/setup.sh
.venv/bin/pytest iron/applications/qwen3_0_6b/test.py -q -m 'not extensive'
```

Evidence found:

```text
The same pytest command failed before sourcing XRT and passed after
source /opt/xilinx/xrt/setup.sh.
```

Root cause:

```text
The Python environment was correct, but the shell did not include XRT's Python
bindings in PYTHONPATH.
```

Fix:

```text
Source /opt/xilinx/xrt/setup.sh before pytest, operator runs, and persistent
bring-up scripts.
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

## Diagnostic Bundle Crashes While Serializing A Layer Tensor

Symptom:

```text
layer_17_qkv_diagnostic_bundle_tensor_begin: hidden
process exits with code -1 and no Python traceback
```

Diagnostic:

```text
Check whether the tensor came from XRTTensor.to_torch() in a previous loop
iteration. A contiguous slice can still be a zero-copy view over the XRT BO.
```

Root cause:

```text
The multi-layer driver carried actual["layer_residual"].contiguous() into the
next layer. Because the slice was already contiguous, no copy happened. The
next layer's hidden could outlive the previous iteration's XRT BO.
```

Fix:

```text
Clone any XRT-derived tensor that crosses a layer, process, or serialization
boundary.
```

Accepted recheck:

```text
layer_17_qkv_diagnostic_bundle_tensor_done: hidden shape=(1024,)
layer_17_qkv_diagnostic_bundle: build_qwen3_persistent_multilayer/diagnostics/qkv_boundary_layer_17.npz
```

## Decode Wall Time Is Much Larger Than NPU Time

Symptom:

```text
npu_layer_time_us_total: about 250000
decode_s: about 1.3-1.6
```

Diagnostic:

```text
Print separate setup, sync, call, drain, and CPU final-head timers. Compare
`decode_s` with `npu_layer_time_us_total` before changing AIE kernels.
```

Evidence found after adding the split:

```text
fast_generate_setup_s: 0.493570
fast_generate_weight_pack_s: 0.121498
fast_generate_weight_xrt_s: 0.301134
fast_generate_cache_xrt_s: 0.024689
fast_op_call_s: 0.250842
fast_output_drain_s: 0.001958
decode_s: 0.256877
token_match: True
```

Root cause:

```text
The slow path rebuilt per-layer weight BOs and copied the full KV cache through
host memory on every decoded token. The external kernels were not the primary
wall-time bottleneck.
```

Fix:

```text
Add --fast-generate: pack weights once, reuse each layer's weight XRTTensor,
keep each layer's KV cache in an XRT inout buffer across decode positions, and
drain only the layer residual needed for the host-driven layer loop.
```

Accepted recheck:

```text
--fast-generate --verify-generate --max-new-tokens 4
token_match: True for positions 6, 7, and 8
decode_s: 0.249-0.257 per NPU-decoded token
```
