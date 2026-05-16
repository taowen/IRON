# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Qwen3-0.6B Inference

This directory contains a correctness-first Qwen3-0.6B inference path. It loads
the original Hugging Face safetensors directly and does not require converting
weights into a block-f16 format before running.

The current implementation is a PyTorch CPU reference for the dense Qwen3 model
used to validate architecture details before moving the graph onto IRON
operators. It includes Q/K RMSNorm, Qwen3 RoPE, GQA, SwiGLU, tied embeddings,
and greedy generation.

## Run

```bash
python iron/applications/qwen3_0_6b/qwen3_cpu.py \
  --model Qwen/Qwen3-0.6B \
  --prompt "What is the capital of France? Answer with only the city name." \
  --max-new-tokens 8 \
  --verify-hf
```

The script downloads the required Hugging Face files through
`huggingface_hub` when `--model` is a repo id. Pass a local model directory to
reuse an existing checkout.

## Decode Megakernel Bring-Up

`qwen3_megakernel.py` builds a performance-first single-token decode graph using
IRON full-ELF fusion. It is a decode path, not prefill: prompt/KV-cache creation
is still handled by the CPU reference, and the fused graph consumes one token,
updates K/V cache, and produces logits.

Use the static linter before compiling:

```bash
python iron/applications/qwen3_0_6b/qwen3_megakernel.py \
  --model Qwen/Qwen3-0.6B \
  --num-layers 1 \
  --max-seq-len 256 \
  --lint-only
```

Compile a layer-limited smoke megakernel:

```bash
source /opt/xilinx/xrt/setup.sh
python iron/applications/qwen3_0_6b/qwen3_megakernel.py \
  --model Qwen/Qwen3-0.6B \
  --num-layers 1 \
  --max-seq-len 256 \
  --compile-only
```

Verify one single-layer decode step against the PyTorch cached reference:

```bash
source /opt/xilinx/xrt/setup.sh
python iron/applications/qwen3_0_6b/qwen3_megakernel.py \
  --model Qwen/Qwen3-0.6B \
  --num-layers 1 \
  --max-seq-len 256 \
  --verify-one-step \
  --verify-repeat 3
```

On the current NPU2 environment this produces the same next token as the
reference for the default prompt:

```text
npu_next_token: 11853 text='imize'
ref_next_token: 11853 text='imize'
logits_max_abs: 0.437500
logits_mean_abs: 0.073747
```

Use a separate build directory for debug graphs so debug drains do not share
cached artifacts with the normal performance path:

```bash
python iron/applications/qwen3_0_6b/qwen3_megakernel.py \
  --model Qwen/Qwen3-0.6B \
  --num-layers 1 \
  --max-seq-len 256 \
  --build-dir build_qwen3_megakernel_debug \
  --clean-build \
  --debug-stage qkv \
  --dump-patches \
  --verify-repeat 3
```

Runtime execution requires a `pyxrt` build that exposes full-ELF APIs
(`pyxrt.elf` and `pyxrt.ext`). The current fallback is intentional: compile-only
mode validates MLIR/aiecc generation and buffer layout without requiring that
runtime extension.

## Persistent Megakernel Bring-Up

`qwen3_persistent.py` is the hand-authored IRON path that moves toward the
AlpinDale-style decode megakernel. It does not use `FusedMLIROperator`; it
builds an explicit `Program` with `Worker` and `ObjectFifo` stages.

The first implemented stage is the layer-0 single-token input RMSNorm:

```bash
source /opt/xilinx/xrt/setup.sh
python iron/applications/qwen3_0_6b/qwen3_persistent.py \
  --model Qwen/Qwen3-0.6B \
  --stage input-rmsnorm \
  --verify \
  --verify-repeat 3 \
  --dump-proof
```

Observed result on the current NPU2 environment:

```text
implementation: hand-authored IRON Program/Worker/ObjectFifo
dispatch_shape: hidden[1024] + norm_weight[1024] -> x_norm[1024]
max_abs: 0.007812
mean_abs: 0.001019
errors: 0
```

This is not yet the full Qwen3 decode megakernel. It is the first persistent
dataflow checkpoint before adding the rest of decode.

The second implemented stage adds Q/K/V projection behind the same persistent
dataflow boundary:

```bash
source /opt/xilinx/xrt/setup.sh
python iron/applications/qwen3_0_6b/qwen3_persistent.py \
  --model Qwen/Qwen3-0.6B \
  --stage input-rmsnorm-qkv \
  --verify \
  --verify-repeat 1 \
  --dump-proof
```

This stage uses three runtime BOs and tap offsets rather than nine separate
runtime buffers:

```text
runtime_bos: hidden[1024], packed_weights[4195328], packed_outputs[5120]
worker_graph: hidden_fifo -> rmsnorm_worker -> weight_worker ->
  single xnorm broadcast FIFO -> Q/K/V matvec workers
```

Verification uses the actual NPU `x_norm` as the local Q/K/V reference input,
then prints full-reference drift separately. Observed local Q/K/V max error is
zero on the current NPU2 environment; the full-reference drift comes from the
upstream bf16 RMSNorm boundary.

The third implemented stage adds Q/K RMSNorm, RoPE, and KV cache write:

```bash
source /opt/xilinx/xrt/setup.sh
python iron/applications/qwen3_0_6b/qwen3_persistent.py \
  --model Qwen/Qwen3-0.6B \
  --stage input-rmsnorm-qkv-rope-cache \
  --verify \
  --verify-repeat 2 \
  --dump-proof
```

This stage keeps the runtime within the five BOs exposed by the generated
xclbin metadata:

```text
runtime_bos: hidden[1024], packed_weights[4195584], rope_angles[128],
  packed_outputs[9216], packed_cache[524288]
```

`packed_outputs` contains `x_norm`, raw Q/K, Q/K norm, and RoPE Q. RoPE K and
V are validated through the current-position KV cache write to avoid draining
the same ObjectFIFO twice. On the current NPU2 environment, repeat verification
passes with cache current-position and prefix-preservation errors at zero.

The fourth stage adds attention score computation and softmax while keeping the
same five runtime BOs:

```bash
source /opt/xilinx/xrt/setup.sh
python iron/applications/qwen3_0_6b/qwen3_persistent.py \
  --model Qwen/Qwen3-0.6B \
  --stage input-rmsnorm-qkv-rope-cache-scores-softmax \
  --verify \
  --verify-repeat 1 \
  --dump-proof
```

This checkpoint is now accepted on the current NPU2 environment: upstream
Q/K/V, RoPE, qk-pair packing, K-cache stream debug, attention scores, and
attention weights all pass local-reference verification. The root cause of the
previous score failure is recorded in `how-to-debug/qwen3-megakernel/`: the
score Worker used two `acquire(1)` calls on the same output FIFO before
release, so both logical outputs could alias one FIFO object. The fixed graph
uses `acquire(2)` with indexed subviews, and preflight now rejects this
non-advancing acquire pattern.

The next checkpoint adds the accurate-version PV/context half of attention:

```bash
source /opt/xilinx/xrt/setup.sh
python iron/applications/qwen3_0_6b/qwen3_persistent.py \
  --model Qwen/Qwen3-0.6B \
  --stage input-rmsnorm-qkv-rope-cache-scores-softmax-context \
  --verify \
  --verify-repeat 1 \
  --dump-proof
```

This stage streams historical V-cache blocks, merges the current token V in a
separate Worker, and computes `attn_context = softmax(scores) @ V` without
adding runtime BOs. It was accepted on the current NPU2 environment with
`attn_context_errors: 0`. During bring-up, a V merge tile initially exceeded L1
because three 64x128 bf16 block FIFOs were buffered on the same tile; the
accepted graph uses single-buffer V-cache/context block FIFOs. A second issue
was isolated to the context external kernel: per-row bf16 read/modify/write
accumulation produced a few large cancellation-sensitive errors, while all
input streams were correct. The fixed kernel accumulates each block in local
float and writes bf16 once per block.

The next checkpoint adds attention output projection and residual add:

```bash
source /opt/xilinx/xrt/setup.sh
python iron/applications/qwen3_0_6b/qwen3_persistent.py \
  --model Qwen/Qwen3-0.6B \
  --stage input-rmsnorm-qkv-rope-cache-scores-softmax-context-o-proj \
  --verify \
  --verify-repeat 1 \
  --dump-proof
```

This stage compiles a second GEMV object for `DIM_K=2048` under a distinct
`qwen3_o_proj_*` symbol, because the QKV GEMV object is compiled for
`DIM_K=1024`. It was accepted on the current NPU2 environment with
`attn_context_errors: 0`, `attn_context_flat_errors: 0`,
`attn_o_proj_errors: 0`, and `attn_residual_errors: 0`.

During bring-up, three real graph/resource issues were found before changing
kernel math: an O-projection weight FIFO had no producer in the active Program
variant, the naive three-worker extension exceeded the current 16 Worker
SequentialPlacer budget, and disabling an older debug stream left behind a
zero-length TAP. These are recorded in `how-to-debug/qwen3-megakernel/`.

The next accepted checkpoint isolates the MLP front half rather than appending
it to the already full attention graph:

```bash
source /opt/xilinx/xrt/setup.sh
python iron/applications/qwen3_0_6b/qwen3_persistent.py \
  --model Qwen/Qwen3-0.6B \
  --stage post-attn-rmsnorm-mlp-gate-up \
  --verify \
  --verify-repeat 1 \
  --dump-proof
```

This stage starts from the verified `attn_residual[1024]`, runs
post-attention RMSNorm, `gate_proj`, `up_proj`, SiLU, and
`ffn_hidden = silu(gate) * up`. It was accepted on the current NPU2 environment
with `mlp_x_norm_errors: 0`, `ffn_gate_errors: 0`, `ffn_up_errors: 0`,
`ffn_gate_silu_errors: 0`, and `ffn_hidden_errors: 0`.

During bring-up, comparing gate/up directly to the full PyTorch reference
misidentified the boundary. Recomputing the local reference from the actual NPU
`mlp_x_norm` proved the GEMV outputs were exact at that boundary. The remaining
SiLU discrepancy was the existing AIE tanh-approx SiLU on negative gate inputs,
so this checkpoint uses an operator-specific absolute tolerance for
`ffn_gate_silu`.

The next accepted checkpoint isolates the MLP down projection and layer
residual add:

```bash
source /opt/xilinx/xrt/setup.sh
python iron/applications/qwen3_0_6b/qwen3_persistent.py \
  --model Qwen/Qwen3-0.6B \
  --stage post-attn-mlp-down-residual \
  --verify \
  --verify-repeat 1 \
  --dump-proof
```

This stage starts from `ffn_hidden[3072]` and `attn_residual[1024]`, runs
`down_proj`, and computes `layer_residual = attn_residual + ffn_out`. It was
accepted on the current NPU2 environment with `ffn_out_errors: 0` and
`layer_residual_errors: 0`.

Placement scaling, runtime position patching, full-layer persistent
composition, and multi-token decode are still future persistent stages.

## Weight Format Decision

For this bring-up, keep the Hugging Face safetensors as the source of truth.
The existing IRON GEMM/GEMV APIs accept ordinary contiguous bf16/fp16 tensors
or simple transposed/padded views; block-f16 conversion is therefore not needed
to validate Qwen3 correctness.

A later NPU performance path should cache prepacked per-operator weights and a
manifest, but that is a performance artifact rather than the canonical model
format.
