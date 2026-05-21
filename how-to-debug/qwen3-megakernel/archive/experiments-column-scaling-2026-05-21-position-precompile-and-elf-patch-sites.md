<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Position Precompile And ELF Patch Sites

This archives step 7 evidence after rejecting the runtime-position metadata
stream. The question was whether to pursue raw ELF/CDO patching,
`NpuControlPacketOp`/BD rewriting, bucket variants, or a safer exact-position
runtime-selection baseline.

## ELF/CDO Patch-Site Evidence

Command:

```bash
python iron/applications/qwen3_0_6b/persistent/artifact_position_diff.py \
  build_qwen3_position_diff_full_pos26 \
  build_qwen3_position_diff_full_pos27
```

Same-bucket pos26 -> pos27:

```text
MLIR changed_diff_lines=64
runtime .bin changed_bytes=8
runtime .bin has eight u32 patch candidates, all DMA byte offsets +256
xclbin changed_bytes=126
main_aie_cdo_elfs.bin changed_bytes=56
changed_core_elves=6
```

The enhanced diff tool mapped core-ELF changes to `.text`, not a data section:

```text
main_core_1_5.elf: core_1_5/FUNC, 16 changed instruction words
main_core_2_2.elf: core_2_2/FUNC,  4 changed instruction words
main_core_2_3.elf: core_2_3/FUNC,  8 changed instruction words
main_core_4_2.elf: core_4_2/FUNC, 16 changed instruction words
main_core_4_3.elf: core_4_3/FUNC,  4 changed instruction words
main_core_4_4.elf: core_4_4/FUNC,  8 changed instruction words
```

Example patch candidate:

```text
core_elf_patch_candidate:
  file_offset=460
  section=.text
  vaddr=0x14c
  symbol=core_1_5/FUNC
  symbol_offset=300
  value=54526600->56623752
  delta=2097152
```

Decision:

```text
Raw ELF/CDO byte patching is not selected as the next implementation path.
The differences are instruction encodings in core functions, not obvious scalar
metadata slots. It would require a stable relocation/encoding proof before it
could be safer than changing the IRON graph.
```

`NpuControlPacketOp` stays last-resort for the same reason: it can theoretically
rewrite low-level registers/BDs, but current IRON code does not expose a safe
runtime workflow for computing the target register payloads and proving DMA
quiescence.

## Exact-Position Precompile Baseline

Implemented flag:

```text
--precompile-generate-positions
```

Behavior:

```text
compile every exact decode-position variant needed by --max-new-tokens before
the token loop, then reuse those artifacts through the existing chunk_op_cache.
```

This does not reduce total setup cost and does not solve artifact count growth.
It is accepted only as a safe measurement/runtime-selection baseline: it moves
JIT compile out of the token loop without adding a new NPU dispatch or changing
the graph body.

Small layer1 validation:

```text
raw prompt: Fibonacci numbers: 1, 1, 2, 3,
layer_chunk_size=1
positions precompiled: 17, 18
token_step 1: token_match=True, next token 5
token_step 2: token_match=True, next token ,
new_text=' 5,'
```

Full accepted graph validation using existing build cache:

```text
raw prompt: Fibonacci numbers: 1, 1, 2, 3,
layer_chunk_size=28
generate_position_17_n_layer_28_compile_s: 13.361
generate_position_18_n_layer_28_compile_s: 6.685
generate_precompile_positions_s: 20.324 positions=2 first_position=17 last_position=18

token_step 1:
  decode_position=17
  npu_layer_time_us_total=100861.695
  token_match=True
  npu_next_token=20 text='5'

token_step 2:
  decode_position=18
  npu_layer_time_us_total=100676.740
  token_match=True
  npu_next_token=11 text=','

new_text=' 5,'
```

Decision:

```text
Exact-position precompile is accepted as the hot-loop measurement baseline.
It is not the final "dynamic position" solution because setup cost and artifact
count still scale with generated tokens.
```

Next implication:

```text
Further per-position speed work should not start from raw ELF patching. It
should either:
  1. find a numerically safe runtime scalar path for position/valid length, or
  2. keep exact-position variants only as a benchmark harness while optimizing
     the MLP/body time.
```
