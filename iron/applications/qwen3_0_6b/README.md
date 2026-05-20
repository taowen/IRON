# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Qwen3-0.6B Inference

This directory contains a correctness-first Qwen3-0.6B inference path. It can
load the original Hugging Face safetensors directly, and the persistent
performance path can also use a prepacked bf16 weight artifact generated from
those safetensors. Block-f16 conversion is not required for the current path.

The current implementation is a PyTorch CPU reference for the dense Qwen3 model
used to validate architecture details before moving the graph onto IRON
operators. It includes Q/K RMSNorm, Qwen3 RoPE, GQA, SwiGLU, tied embeddings,
and greedy generation.

Implementation layout:

```text
qwen3_cpu.py              CPU model/reference runner
qwen3_decode_reference.py cached CPU decode reference
persistent/              supported IRON persistent Program path
full_elf/                experimental full-ELF fused scaffold
```

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

## Persistent Megakernel Path

`persistent/main.py` is the supported Qwen3 NPU path. The attention design has
been narrowed to one active implementation: `n-layer-final-only`. It runs one
or more full transformer layers in a persistent IRON Program and drains only the
final hidden state for CPU final norm/LM head.

Useful supporting checkpoints that do not use `attention_design.py` are still
available for isolated RMSNorm/QKV and MLP debugging:

```bash
source /opt/xilinx/xrt/setup.sh
python iron/applications/qwen3_0_6b/persistent/main.py \
  --model Qwen/Qwen3-0.6B \
  --stage input-rmsnorm-qkv \
  --verify

python iron/applications/qwen3_0_6b/persistent/main.py \
  --model Qwen/Qwen3-0.6B \
  --stage post-attn-rmsnorm-full-mlp \
  --verify
```

Compile or verify the active attention+MLP path:

```bash
source /opt/xilinx/xrt/setup.sh
python iron/applications/qwen3_0_6b/persistent/main.py \
  --model Qwen/Qwen3-0.6B \
  --stage n-layer-final-only \
  --layer-chunk-size 4 \
  --verify \
  --prompt "Count from one to five." \
  --raw-prompt
```

Fast generate uses the same n-layer final-only Program, cached XRT weight/cache
buffers, and CPU final norm/LM head:

```bash
source /opt/xilinx/xrt/setup.sh
python iron/applications/qwen3_0_6b/persistent/main.py \
  --model Qwen/Qwen3-0.6B \
  --stage generate \
  --fast-generate \
  --layer-chunk-size 4 \
  --verify-generate \
  --max-new-tokens 3
```

Current measurement on the default prompt and `max_new_tokens=3`:

```text
chunk=4 after prefix-KV optimization:
token_match=True, npu_layer_time_us_total ~= 184-201ms

chunk=7 compile/preflight optimization:
the static graph would reduce 28 layers to 4 chunk dispatches per token, and
compile-only preflight reports compute_cores=21, total_dma_tasks=81,
max_dma_tasks_per_fifo=7
```

Use the real graph probe to decide the next speed target:

```bash
source /opt/xilinx/xrt/setup.sh
. .venv/bin/activate
python iron/applications/qwen3_0_6b/persistent/real_graph_probe.py \
  --stages qkv mlp-gate-up n-layer-final-only \
  --columns 1 2 4 8 \
  --layer-iterations 4 \
  --preflight-only \
  --allow-failures
```

Current result: QKV preflights at 4 columns and MLP gate/up preflights at 2
columns, but the n-layer attention+MLP path is still accepted only at 1 column.
The next performance direction is to lift that column-scaling limit.

## Weight Format Decision

For this bring-up, keep the Hugging Face safetensors as the source of truth.
The existing IRON GEMM/GEMV APIs accept ordinary contiguous bf16/fp16 tensors
or simple transposed/padded views; block-f16 conversion is therefore not needed
to validate Qwen3 correctness.

The persistent performance path can cache prepacked bf16 weights on disk:

```bash
source /opt/xilinx/xrt/setup.sh
python iron/applications/qwen3_0_6b/persistent/main.py \
  --model Qwen/Qwen3-0.6B \
  --prepare-weights
```

The default output directory is `<model_dir>/qwen3_iron_packed/`. Override it
with `--packed-weights-dir`. The artifact is:

```text
weights.bf16.bin   contiguous raw bf16 per-layer weights for all transformer layers
manifest.json      format/config/offset/shape table for every layer segment
```

On the current development machine, `Qwen/Qwen3-0.6B` resolves to the local
Hugging Face snapshot pinned in `qwen3_cpu.py`, and the prepared artifact lives
under:

```text
~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca/qwen3_iron_packed
```

Use the artifact during fast generate:

```bash
source /opt/xilinx/xrt/setup.sh
python iron/applications/qwen3_0_6b/persistent/main.py \
  --model Qwen/Qwen3-0.6B \
  --stage generate \
  --fast-generate \
  --require-packed-weights \
  --verify-generate \
  --max-new-tokens 3
```

By default, fast generate executes the n-layer final-only Program with
`--layer-chunk-size 1`; the current accepted performance setting is
`--layer-chunk-size 4`. The packed artifact removes runtime weight packing and
establishes a global weight buffer plus per-layer offset manifest; final
norm/LM head still run on the CPU.
