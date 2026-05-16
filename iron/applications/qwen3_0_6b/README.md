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

## Weight Format Decision

For this bring-up, keep the Hugging Face safetensors as the source of truth.
The existing IRON GEMM/GEMV APIs accept ordinary contiguous bf16/fp16 tensors
or simple transposed/padded views; block-f16 conversion is therefore not needed
to validate Qwen3 correctness.

A later NPU performance path should cache prepacked per-operator weights and a
manifest, but that is a performance artifact rather than the canonical model
format.
