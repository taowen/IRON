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

