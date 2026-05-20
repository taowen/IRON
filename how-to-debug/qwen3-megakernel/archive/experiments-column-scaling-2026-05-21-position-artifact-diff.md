<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Position Artifact Diff

This archive records the first artifact-level diagnostic for removing
per-position compile from the accepted Qwen3 persistent generate path.

## Question

The accepted graph compiles a new artifact for each decode position. Before
choosing runtime instruction patching or bucketed precompile, we need to know
what actually changes between adjacent positions.

## Commands

Generate same-bucket MLIR quickly:

```bash
source /opt/xilinx/xrt/setup.sh
. .venv/bin/activate

python iron/applications/qwen3_0_6b/persistent/real_graph_probe.py \
  --stages n-layer-final-only \
  --columns 2 \
  --attention-columns 2 \
  --mlp-gate-up-columns 2 \
  --mlp-gate-up-pair-rows \
  --mlp-gate-up-direct-silu \
  --layer-iterations 28 \
  --position 26 \
  --preflight-only \
  --build-dir build_qwen3_position_diff_probe_pos26 \
  --clean-build

python iron/applications/qwen3_0_6b/persistent/real_graph_probe.py \
  --stages n-layer-final-only \
  --columns 2 \
  --attention-columns 2 \
  --mlp-gate-up-columns 2 \
  --mlp-gate-up-pair-rows \
  --mlp-gate-up-direct-silu \
  --layer-iterations 28 \
  --position 27 \
  --preflight-only \
  --build-dir build_qwen3_position_diff_probe_pos27 \
  --clean-build
```

Compile full artifacts for binary-level evidence:

```bash
python iron/applications/qwen3_0_6b/persistent/real_graph_probe.py \
  --stages n-layer-final-only \
  --columns 2 \
  --attention-columns 2 \
  --mlp-gate-up-columns 2 \
  --mlp-gate-up-pair-rows \
  --mlp-gate-up-direct-silu \
  --layer-iterations 28 \
  --position 26 \
  --build-dir build_qwen3_position_diff_full_pos26 \
  --clean-build

python iron/applications/qwen3_0_6b/persistent/real_graph_probe.py \
  --stages n-layer-final-only \
  --columns 2 \
  --attention-columns 2 \
  --mlp-gate-up-columns 2 \
  --mlp-gate-up-pair-rows \
  --mlp-gate-up-direct-silu \
  --layer-iterations 28 \
  --position 27 \
  --build-dir build_qwen3_position_diff_full_pos27 \
  --clean-build
```

Summarize the artifacts:

```bash
python iron/applications/qwen3_0_6b/persistent/artifact_position_diff.py \
  build_qwen3_position_diff_full_pos26 \
  build_qwen3_position_diff_full_pos27
```

Also check a cache-block boundary:

```bash
python iron/applications/qwen3_0_6b/persistent/artifact_position_diff.py \
  build_qwen3_position_diff_probe_pos63 \
  build_qwen3_position_diff_probe_pos64
```

## Same-Bucket Result: pos26 -> pos27

Full compile time:

```text
pos26 compile elapsed_s=68.289
pos27 compile elapsed_s=68.099
```

MLIR structure:

```text
line_count=1434/1434
changed_diff_lines=64
dma_bd_count=21/21
```

Changed MLIR categories:

```text
attention_scores_position=8
attention_context_position=8
merge_v_position=4
mask_length=8
constant=28
dma_bd=8
```

Changed runtime DMA offsets:

```text
qwen3_rc_k_rope_0 offset 3328   -> 3456   delta +128 bf16 elements
qwen3_rc_v_0      offset 265472 -> 265600 delta +128 bf16 elements
qwen3_rc_k_rope_1 offset 134400 -> 134528 delta +128 bf16 elements
qwen3_rc_v_1      offset 396544 -> 396672 delta +128 bf16 elements
```

Runtime instruction binary:

```text
size=2756/2756
changed_bytes=8
u32 patch candidates:
  offset=1956 value=6656   -> 6912   delta +256 bytes
  offset=2024 value=6656   -> 6912   delta +256 bytes
  offset=2104 value=530944 -> 531200 delta +256 bytes
  offset=2172 value=530944 -> 531200 delta +256 bytes
  offset=2252 value=268800 -> 269056 delta +256 bytes
  offset=2320 value=268800 -> 269056 delta +256 bytes
  offset=2400 value=793088 -> 793344 delta +256 bytes
  offset=2468 value=793088 -> 793344 delta +256 bytes
```

Project-level binaries:

```text
main_mem_topology.json identical
main_kernels.json      identical
main_aie_cdo_init.bin  identical
main_aie_cdo_enable.bin identical

main_aie_cdo_elfs.bin  different, changed_bytes=56
main.pdi               different, changed_bytes=56
xclbin                 different, changed_bytes=126
```

Changed core ELFs:

```text
main_core_1_5.elf changed_bytes=16
main_core_2_2.elf changed_bytes=4
main_core_2_3.elf changed_bytes=8
main_core_4_2.elf changed_bytes=16
main_core_4_3.elf changed_bytes=4
main_core_4_4.elf changed_bytes=8
```

The changed core LLVM confirms why the ELFs changed:

```text
qwen3_attention_scores_bf16(..., i32 26, ...)
qwen3_attention_scores_bf16(..., i32 27, ...)

mask_bf16(..., i32 27, i32 256)
mask_bf16(..., i32 28, i32 256)

qwen3_merge_current_v_bf16(..., i32 26, ...)
qwen3_merge_current_v_bf16(..., i32 27, ...)
```

## Cache-Block Boundary Result: pos63 -> pos64

Preflight still passes with the same headline resource counts:

```text
compute_cores=30
total_dma_tasks=21
max_dma_tasks_per_fifo=1
max_tile_inputs=2
max_tile_outputs=2
```

MLIR structure stays the same size, but more constants change:

```text
line_count=1434/1434
changed_diff_lines=140
dma_bd_count=21/21
loop_bound=12
dma_bd=16
```

The active cache block count changes:

```text
score/context/V-merge loops: 1 block -> 2 blocks
K/V cache read lengths:      32768 -> 65536 bf16 elements
current K/V write offsets:   +128 bf16 elements
```

This is a cache-length bucket boundary. It is not only a current-position
offset change.

## Interpretation

There are two separate position-specialized surfaces:

```text
1. Runtime instruction stream:
   current K/V write DMA offsets change by one head_dim each position.
   In the pos26->pos27 binary, this is exactly eight u32 patch candidates.

2. AIE core code / embedded ELF:
   attention score, V merge, context, and mask calls bake position or
   position+1 as immediate integer arguments.
```

Therefore, patching only the runtime `.bin` is not sufficient. The graph would
still run AIE cores compiled for the wrong position and mask length.

Bucketed precompile alone is also not sufficient for the current graph. A
bucket artifact compiled for one position still bakes that exact position into
the core ELF. Buckets only become useful after the per-position scalar values
move to runtime metadata or after the ELF/CDO patch sites are proven.

## Decision

Step 5 is accepted: the adjacent-position artifact difference is now diagnosed.

The next implementation experiment should not start with arbitrary bucket
variants. It should first remove or patch the AIE-core position constants:

```text
preferred direction:
  extend the existing attention2 runtime metadata so attention workers receive
  position and valid length dynamically

then:
  keep one artifact per active cache-block count, or patch the eight runtime
  `.bin` DMA offset words inside a block

defer:
  raw xclbin/CDO ELF binary patching, unless runtime metadata proves too
  expensive or impossible
```

For max_seq_len=256 and cache_block_seq=64, a bucket plan would mean four
active-cache-block variants after dynamic position metadata:

```text
positions 0..63    -> one cache block
positions 64..127  -> two cache blocks
positions 128..191 -> three cache blocks
positions 192..255 -> four cache blocks
```
