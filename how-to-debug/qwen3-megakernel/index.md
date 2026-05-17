<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Qwen3 Megakernel Debug

Use these notes as a debugging map, not as a chronological log.

- [Symptom lookup](symptoms.md): known failures, evidence, root causes, fixes,
  and recheck commands.
- [Diagnostic methods](diagnostic-methods.md): reusable checks that have already
  found real bugs in this bring-up.
- [Lessons](lessons.md): design constraints and preflight checks inferred from
  the diagnosed failures.

Scope:

```text
persistent/           supported hand-authored IRON Program/Worker/ObjectFifo path
full_elf/             experimental full-ELF FusedMLIROperator scaffold
```

Use `persistent/` as the active correctness and performance-debug baseline.
Use `full_elf/` as a reference for fused-buffer and LM-head ideas, not as the
currently accepted generate path.

The current accepted persistent checkpoints are:

```text
input-rmsnorm      hidden + norm_weight -> x_norm
input-rmsnorm-qkv  hidden + packed_weights -> packed x_norm/Q/K/V
input-rmsnorm-qkv-rope-cache
                   hidden + packed weights + rope_angles -> Q/K RoPE + KV cache write
```

The accepted score/softmax checkpoint below compiles, runs, and verifies on the
current NPU2 environment:

```text
input-rmsnorm-qkv-rope-cache-scores-softmax
input-rmsnorm-qkv-rope-cache-scores-softmax-context
input-rmsnorm-qkv-rope-cache-scores-softmax-context-o-proj
post-attn-rmsnorm-mlp-gate-up
post-attn-mlp-down-residual
post-attn-rmsnorm-full-mlp
input-rmsnorm-qkv-rope-cache-scores-softmax-context-o-proj-full-mlp
multi-layer-full-layer
```

Score/softmax root cause that was fixed:

```text
input RMSNorm, QKV projection, Q/K RMSNorm, RoPE, and KV cache write pass.
qk_pair debug drain and independent K-cache stream debug drain pass.
Attention score failed because the score Worker acquired the same output FIFO
with acquire(1), acquire(1) before release; generated LLVM showed both logical
score outputs could point at the same FIFO object.
The fixed graph uses acquire(2) and indexed subviews, and preflight now rejects
non-advancing ObjectFIFO acquire patterns.
```

PV/context root causes that were fixed:

```text
The first V merge graph failed in aiecc resource allocation, not at runtime.
MemoryMap showed tile_3_3 held two v_cache blocks, two v_context blocks, and
one debug V block. Each block was 64x128xbf16, so the tile exceeded L1.
The accepted graph uses depth=1 for the V-cache and V-context block FIFOs.

After V stream and weights both verified, attn_context still had six large
errors. Comparing per-row bf16 accumulation and float accumulation proved the
inputs were not the failing boundary. The context kernel was fixed to
accumulate one block in local float and write bf16 once per block.
```

O-projection root causes that were fixed:

```text
The first graph had qwen3_rc_o_weight_0 consumed by a Worker but not produced
in the active Program variant. The fix was to move the Runtime.fill into the
RoPE/cache/context implementation.

The next graph exceeded the current SequentialPlacer budget: context already
used 15 Workers and the naive extension added three more. The accepted graph
fuses context flatten into the context Worker and drops the older K-cache debug
copy Worker in this deeper checkpoint.

After K-cache debug was disabled, a stale TAP was still generated with length
0. Optional debug streams now use one boolean for size, FIFO, Worker, fill,
drain, TAP, and verifier slicing.
```

MLP gate/up root causes that were diagnosed:

```text
The isolated MLP checkpoint initially showed gate/up mismatches against the
full PyTorch reference even though mlp_x_norm passed. Recomputing gate/up from
the actual NPU mlp_x_norm proved the GEMV workers and packed weights were
correct; the full-reference difference was upstream bf16 boundary drift.

The remaining SiLU mismatches were only on negative gate values. The existing
AIE SiLU kernel uses a tanh approximation, and the old standalone SiLU test
only covered positive inputs. The checkpoint now verifies SiLU with the same
local input boundary and an explicit absolute tolerance for that approximation.
```

MLP down/residual checkpoint status:

```text
The isolated MLP down checkpoint compiled, passed preflight, and verified
without new debug symptoms. It starts from ffn_hidden and attn_residual host
inputs, computes down_proj, drains ffn_out, then checks layer_residual through
the residual add Worker.

Accepted evidence:
ffn_out_errors: 0
layer_residual_errors: 0
```

Full MLP checkpoint status:

```text
The composed post-attention full MLP checkpoint compiled, passed preflight, and
verified without new debug symptoms. It starts from attn_residual, computes
post-attention RMSNorm, gate/up, SiLU, elementwise multiply, down projection,
and layer residual in one IRON Program while draining every boundary.

Accepted evidence:
runtime_memrefs: 3
arg_specs: 3
max_fifo_buffered_bytes: 49152
non_advancing_acquires: 0
mlp_x_norm_errors: 0
ffn_gate_errors: 0
ffn_up_errors: 0
ffn_gate_silu_errors: 0
ffn_hidden_errors: 0
ffn_out_errors: 0
layer_residual_errors: 0
```

Attention + full MLP checkpoint status:

```text
The composed attention/O-projection/full-MLP checkpoint compiled, passed
preflight, and verified on NPU2. This is not a final performance placement, but
it proves one decode-token layer can keep attention residual, full MLP hidden,
down projection, and layer residual in one IRON Program.

Accepted evidence:
runtime_memrefs: 5
arg_specs: 5
compute_cores: 19
max_fifo_buffered_bytes: 32768
max_tile_inputs: 2
max_tile_outputs: 2
non_advancing_acquires: 0
attn_residual_errors: 0
ffn_hidden_errors: 0
ffn_out_errors: 0
layer_residual_errors: 0
```

Multi-layer full-layer reuse status:

```text
The accepted full-layer Program is now reused as a single-layer decode
primitive. The host switches layer weights and the per-layer KV cache slice,
then feeds each layer_residual into the next invocation.

This checkpoint still stops before final RMSNorm, LM head, token selection, and
multi-token decode. It exists to prove layer-by-layer reuse before building a
larger persistent state machine.

Accepted evidence:
num_layers=1 hidden_after_layers_errors: 0
num_layers=2 hidden_after_layers_errors: 0
num_layers=4 hidden_after_layers_errors: 0
num_layers=4 hidden_after_layers_max_abs: 0.125000
```

Persistent generate status:

```text
The default generate path remains the debug flow: host switches layer weights,
drains the full debug output and updated KV cache, and refills it for the next
invocation.

The accepted fast-generate path now uses one n-layer-final-only operator class
for --layer-chunk-size 1, 2, and 4. It reuses packed weight XRT buffers, keeps
each chunk's KV cache resident in its XRT inout buffer across decode positions,
and drains only the final hidden state after each chunk because the chunk loop
is still host-driven.

Accepted evidence:
--fast-generate --verify-generate --max-new-tokens 3
chunk=1 token_match: True for positions 6 and 7
chunk=2 token_match: True for positions 6 and 7
chunk=4 token_match: True for positions 6 and 7

n-layer-final-only:
chunk_hidden_errors: 0 for chunk=1,2,4
cache current errors: 0 for accepted chunk checks
preflight: runtime_memrefs=5 compute_cores<=21 non_advancing_acquires=0
```

Do not debug from final logits first. Start from the symptom, prove the failing
boundary, and only then change code.
