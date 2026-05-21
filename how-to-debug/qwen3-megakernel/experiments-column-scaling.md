<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Column Scaling Experiments

This is the active decision page for Qwen3 persistent megakernel performance
work. Keep raw logs and closed branch detail in `archive/`; keep this page
focused on the current baseline, constraints, and the next executable steps.

Rule:

```text
When a new failure appears, update the relevant symptom/method page first.
Then record only the experiment decision here.
```

## Archive Map

| archive | contents |
| --- | --- |
| `archive/experiments-column-scaling-2026-05-21.md` | original single-column baseline, early chunk/resource failures, first full-depth timings |
| `archive/experiments-column-scaling-2026-05-21-attention2-mlp2.md` | attention2 and packed two-column MLP bring-up |
| `archive/experiments-column-scaling-2026-05-21-attention2-mlp2-multilayer.md` | multi-layer attention2/MLP2 scaling and resource evidence |
| `archive/experiments-column-scaling-2026-05-21-attention2-drift-diagnostics.md` | layer17/18/19 strict-reference drift diagnostics |
| `archive/experiments-column-scaling-2026-05-21-softmax-fusion-projection-boundary.md` | score-side softmax fusion, prompt suite, O-proj widening limit |
| `archive/experiments-column-scaling-2026-05-21-mlp-gateup-pair-fusion.md` | paired-row MLP gate/up layout and paired matvec baseline |
| `archive/experiments-column-scaling-2026-05-21-current-baseline-phase-probe.md` | accepted-baseline timing, static estimator, phase sensitivity probe |
| `archive/experiments-column-scaling-2026-05-21-direct-gateup-silu.md` | direct hidden-producing gate/up+SiLU compile fix, prompt suite, estimator, phase probe |
| `archive/experiments-column-scaling-2026-05-21-three-way-gateup.md` | three-way gate/up rejection: RuntimeEndpoint output placement boundary |
| `archive/experiments-column-scaling-2026-05-21-rowgroup8.md` | gate/up row-group 8 branch: BD/block failure, block-object fix, slower timing rejection |
| `archive/experiments-column-scaling-2026-05-21-three-way-gateup-split.md` | packed/split three-way gate/up: endpoint blocker solved, token-correct, multi-token timing rejection |
| `archive/experiments-column-scaling-2026-05-21-phase-probe-after-split.md` | accepted-graph phase probe after rejecting static MLP widening branches |
| `archive/experiments-column-scaling-2026-05-21-state-machine-resource-probe.md` | resource-scaling check before descriptor/state-machine work |
| `archive/experiments-column-scaling-2026-05-21-position-artifact-diff.md` | adjacent-position MLIR/bin/xclbin diff and bucket boundary diagnosis |
| `archive/experiments-column-scaling-2026-05-21-runtime-position-metadata-rejected.md` | runtime position metadata stream attempt, timeout/NaN diagnosis, and rejection |
| `archive/experiments-column-scaling-2026-05-21-position-precompile-and-elf-patch-sites.md` | ELF patch-site proof and exact-position precompile runtime-selection baseline |

Symptom and method notes remain split by diagnosis type:

```text
symptoms-resources.md
symptoms-dataflow.md
symptoms-numeric.md
symptoms-runtime.md
methods-static.md
methods-runtime.md
methods-numeric.md
methods-layout.md
```

## Current Accepted Baseline

```text
stage: generate --fast-generate
operator: n-layer-final-only
layer_chunk_size: 28
num_aie_columns: 2
attention layout: two-column attention2 with fused score/softmax Workers
MLP layout: packed two-column segment-major gate/up/down weights
MLP gate/up rows: paired as [4 gate rows][4 matching up rows]
MLP gate/up compute: qwen3_mlp_gate_up_pair_silu4_rows_shard_bf16
MLP activation: fused into direct hidden-producing gate/up kernel
final norm / LM head: CPU F.linear path
weights: prepacked bf16 weights on disk
```

Known-good gate:

```text
default prompt: token_match=True, new_text='Paris', NPU time 105.293 ms
prompt suite:  5/5 token steps matched for Fibonacci, weekdays, numeric, opposite
suite mean:    100.961 ms after direct gate/up+SiLU
precompile mode: --precompile-generate-positions moves exact-position compiles
                 before the token loop; Fibonacci full chunk=28 matched 2/2
preflight:     compute_cores=30, ObjectFIFOs=53, max_tile_inputs=2,
               max_tile_outputs=2, max_fifo_buffered_bytes=32768
static calls:  76,160 calls/token; qwen3_silu_mul_shard_bf16 has no call sites
```

Current interpretation:

```text
Host dispatch is no longer the main bottleneck. The accepted path is one NPU
decode dispatch for the transformer body.

The layer-iteration resource probe changed the state-machine diagnosis:
the accepted graph already reuses the main Worker/ObjectFIFO graph across
layer_iterations. Resources do not grow linearly with layers.

The missing "true megakernel" piece is now narrower: position/cache-dependent
compile artifacts and runtime metadata are still static per decode position.
The accepted measurement baseline is exact-position precompile with runtime
selection: it removes JIT compile from the token loop, but setup cost and
artifact count still scale with generated tokens.

Raw ELF/CDO patching is not selected as the next implementation path. Same-
bucket pos26->pos27 changes instruction words in six core ELFs, not only
runtime `.bin` DMA offset words. `NpuControlPacketOp` remains a last-resort
low-level path because IRON does not expose a safe runtime BD rewrite workflow.

The latest direct-graph phase probe says the MLP-side increment is still the
larger measured target:
  full-layer warm median: 3.882 ms
  attention-only median:  1.661 ms
  MLP-side increment:     2.221 ms, 57.2% of one-layer warm median
```

## Hard Constraints

Keep these constraints unless a new diagnostic proves they are too conservative:

```text
max_tile_inputs <= 2
max_tile_outputs <= 2
no extra host dispatch in the accepted decode path
preflight before numerical validation
token validation before timing claims
every rejected branch needs a named root cause
```

Known rejected or deferred branches:

```text
direct input-RMSNorm Worker fusion:
  rejected by preflight with max_tile_inputs=3 > 2

direct O-proj row sharding:
  needs more than the two freed cores unless residual/MLP boundary is redesigned

three-way MLP gate/up with independent weight stream:
  rejected by placement trace; adding the third gate/up Runtime.fill hits the
  runtime-output endpoint boundary before MLIR preflight

three-way MLP gate/up with packed/split parent stream:
  rejected by timing; it solved the endpoint blocker but consumed all 32 cores
  and regressed the multi-token Fibonacci mean

generic NPU final LM head:
  second-dispatch GEMV was slower than CPU F.linear

gate/up row-group 8:
  rejected by timing; the first FIFO shape exposed a BD/block resource failure,
  and the block-object fix compiled but was slower
```

## Closed Decisions

The detailed logs live in `archive/`. Keep only decision-quality summaries here:

| decision | result | root cause or proof |
| --- | --- | --- |
| packed two-column MLP | accepted | full-depth chunk=28 runs and materially beats the old single-column path |
| two-column attention2 + fused score/softmax | accepted | prompt-suite token IDs matched and phase probe made MLP the larger remaining target |
| paired gate/up row layout | accepted | reduced gate/up stream pressure while preserving token correctness |
| direct hidden-producing gate/up+SiLU | accepted | removed separate `qwen3_silu_mul_shard_bf16` call sites; prompt-suite mean improved 101.683 -> 100.961 ms |
| gate/up row-group 8 | rejected | first form failed full `aiecc` with `aie.mem` >16 blocks; block-object form compiled and matched tokens but slowed default prompt 106.233 -> 109.638 ms |
| three-way gate/up with split parent stream | rejected | `ObjectFifo.split` kept runtime outputs at 16 and token IDs matched, but Fibonacci mean regressed 102.215 -> 104.442 ms and the graph consumed all 32 compute cores |
| phase probe after split rejections | accepted | accepted graph still shows MLP-side median increment 2.221 ms, 57.2% of one-layer warm median |
| layer-iteration resource probe | accepted | accepted graph compute cores/endpoints do not grow linearly with layers; descriptor work should target position-specific compile/runtime constants, not worker-per-layer removal |
| adjacent-position artifact diff | accepted | pos26->pos27 changes only position/mask constants plus four current-K/V DMA offsets in MLIR, but both runtime `.bin` and embedded AIE ELFs change, so `.bin` patching alone is insufficient |
| runtime position metadata through attention streams | rejected | score/mask metadata variants compiled and attention-probe ran, but full-layer generate timed out; attention-probe produced NaN residuals, so the branch is numerically unsafe and default code is restored to static position |
| ELF/CDO patch-site proof | rejected as implementation path | changed core-ELF bytes map to `.text` instruction words in six core functions; no stable scalar metadata slot or relocation rule has been proven |
| exact-position precompile | accepted as measurement baseline | full chunk=28 Fibonacci precompiled positions 17/18 before the token loop and matched 2/2 decode steps; setup/artifact count still scales with token count |
| direct input RMSNorm fusion | rejected | preflight hit `max_tile_inputs=3 > 2` |
| direct O-proj row sharding | deferred | two freed cores are not enough without redesigning residual/MLP boundary |
| three-way gate/up with independent weight stream | rejected | placement trace shows accepted graph already uses 16 runtime outputs; third gate/up stream needs a 17th output endpoint |
| NPU final LM head as a second dispatch | rejected | slower than CPU `F.linear` tail |

## Next 10 Steps

Every step must end as `accepted`, `rejected`, or `blocked` with a named root
cause. Do not start the next graph rewrite until the current step has a
diagnosis-quality result.

| step | status | purpose | finish condition |
| ---: | --- | --- | --- |
| 1 | rejected | Try a no-new-runtime-output MLP optimization by increasing direct gate/up row group size | full 8-row row-FIFO form failed `aiecc` resource allocation; block-object non-fused form compiled and matched tokens but was slower than the accepted direct-SiLU baseline |
| 2 | rejected | Test packed/split gate/up streams behind an existing endpoint | endpoint blocker solved with `ObjectFifo.split`, but the three-way graph did not beat the accepted two-way direct-SiLU baseline on Fibonacci |
| 3 | accepted | Re-run phase probe on the best accepted graph | MLP-side median increment remains 2.221 ms, so the graph body is still MLP-side dominated |
| 4 | accepted | Check whether the accepted graph's resources grow linearly with layer count | `layers=1/8/28` preflight shows compute cores 29/30/30 and runtime outputs 16/16/16, so worker-per-layer growth is not the current blocker |
| 5 | accepted | Diff adjacent-position artifacts and identify what changes between decode positions | pos26/27 and pos63/64 artifact diffs identify runtime `.bin` DMA offsets, AIE-core position constants, and cache-block bucket boundaries |
| 6 | rejected | Move position and valid length out through widened attention metadata streams | attempted score/mask/V metadata streams reached `ERT_CMD_STATE_TIMEOUT` or NaN attention-probe output; default path is restored and this exact stream-widening design is rejected |
| 7 | accepted | Choose safer per-position reuse mechanism after metadata rejection | raw ELF/CDO patching rejected by instruction-word patch-site proof; exact-position precompile accepted only as hot-loop measurement/runtime-selection baseline |
| 8 | active | Use exact-position precompile to separate hot-loop timing from setup-time artifact generation | run multi-token prompt suites with precompiled exact positions; token IDs must match PyTorch and token loop must not compile new variants |
| 9 | next | Revisit MLP body speed using the precompiled hot-loop baseline | new graph-body change must beat the accepted direct-SiLU baseline on a multi-token prompt suite, not only default prompt |
| 10 | later | Revisit final norm / LM head only after the NPU body improves | end-to-end token time including output/argmax beats CPU `F.linear` tail |

Rejected step 1 result:

```text
First form:
  both gate/up streams used acquire(16) single-row FIFO objects
  preflight passed, but full aiecc failed:
    'aie.mem' op has more than 16 blocks
    no space for this BD

Diagnosed fix:
  keep fused col0 at row-group 4 because post_norm shares that stream
  use 4-row FIFO objects only for the non-fused widened stream

Final result:
  token_match=True, new_text='Paris'
  static calls/token improved 76,160 -> 70,784
  measured default prompt NPU time regressed 106.233 -> 109.638 ms

Decision:
  rejected as a speed branch; accepted baseline remains 4-row direct gate/up+SiLU
```

Rejected step 2 result:

```text
Resource result:
  ObjectFifo.split packed gate/up shard1+shard2 behind one runtime stream
  preflight passed with runtime_output=16 and compute_cores=32
  full compile/generate token_match=True

Timing:
  default prompt warm run: three-way split 104.377 ms vs baseline 105.774 ms
  Fibonacci positions 17-21:
    three-way split mean 104.442 ms
    accepted two-way baseline mean 102.215 ms

Decision:
  rejected as a performance path. It solved the endpoint blocker but did not
  produce robust speedup, and it consumes all 32 compute cores.
```

Accepted step 3 result:

```text
Phase probe after rejecting row-group 8 and three-way split:
  full-layer warm median:    3.882 ms
  attention-only median:     1.661 ms
  MLP-side median increment: 2.221 ms
  MLP median-increment share: 57.2%

Decision:
  the accepted graph is still MLP-side dominated, but static MLP shard-count
  increases did not produce robust speedup. Move to descriptor/state-machine
  resource reuse instead of another static gate/up split.
```

Accepted step 4 result:

```text
Resource probe:
  layers=1:  compute_cores=29 runtime_output=16 max_dma_tasks_per_fifo=1
  layers=8:  compute_cores=30 runtime_output=16 max_dma_tasks_per_fifo=2
  layers=28: compute_cores=30 runtime_output=16 max_dma_tasks_per_fifo=1

Decision:
  the accepted graph already reuses workers/FIFOs across layer_iterations for
  the main resource counts. A "state-machine" rewrite should not target
  worker-per-layer removal. The useful target is position-specific compile and
  runtime constants/TAP metadata.
```

Accepted step 5 result:

```text
Same-bucket pos26 -> pos27:
  MLIR line_count=1434/1434
  changed_diff_lines=64
  runtime .bin size=2756/2756, changed_bytes=8
  eight u32 runtime patch candidates, all DMA byte offsets +256
  main_aie_cdo_elfs.bin changed_bytes=56
  six core ELFs changed because position/mask are compiled as immediates

Cache-block boundary pos63 -> pos64:
  MLIR line_count=1434/1434
  changed_diff_lines=140
  K/V cache read length 32768 -> 65536 bf16 elements
  score/context/V-merge loops 1 block -> 2 blocks

Decision:
  raw runtime .bin patching alone is insufficient, because AIE core ELFs also
  bake position and valid length. Bucketed precompile alone is also insufficient
  until those scalar values become runtime metadata or proven ELF/CDO patch
  sites. The first dynamic metadata attempt is now rejected; the current safe
  baseline is exact-position precompile while a different runtime scalar path
  is investigated.
```

Rejected step 6 result:

```text
Attempt:
  move position and valid length through widened existing attention metadata
  streams so score/mask/V/context workers no longer bake immediate constants.

Result:
  variants reached ERT_CMD_STATE_TIMEOUT or attention-probe NaN residuals.
  Padding ObjectFIFO metadata tails fixed a real alignment blind spot, but did
  not fix the numeric/runtime failure.

Decision:
  this exact stream-widening design is rejected. Keep the accepted static-
  position graph as the correctness baseline.
```

Accepted step 7 result:

```text
ELF/CDO patch-site proof:
  runtime .bin has only eight u32 DMA-offset patch candidates for pos26->pos27
  but six core ELFs also change. The changed core-ELF words map to `.text`
  instruction words inside core_* functions, with no proven stable relocation
  rule. Raw ELF/CDO patching is therefore rejected as the next implementation.

NpuControlPacketOp / BD rewrite:
  remains last-resort. It is a real hardware capability, but current IRON does
  not expose a safe workflow for runtime BD/register rewrite plus DMA
  quiescence proof.

Exact-position precompile:
  accepted as a measurement/runtime-selection baseline. It compiles every exact
  position needed by --max-new-tokens before the token loop and then reuses
  chunk_op_cache. It does not solve total setup time or artifact count growth.
```

Active step 8 result so far:

```text
Implemented:
  --precompile-generate-positions

Validation:
  raw prompt: Fibonacci numbers: 1, 1, 2, 3,
  layer_chunk_size=28
  generate_precompile_positions_s: 20.324
  positions: 17, 18
  token_step 1: token_match=True, npu_next_token=20 text='5'
  token_step 2: token_match=True, npu_next_token=11 text=','
  new_text=' 5,'

Interpretation:
  precompiled exact variants can be selected at runtime without adding a new
  host dispatch and without compiling inside the token loop. This is now the
  correct baseline for comparing graph-body speed changes.
```

## Canonical Recheck Commands

Accepted baseline default prompt:

```bash
source /opt/xilinx/xrt/setup.sh
. .venv/bin/activate

PYTHONUNBUFFERED=1 python -X faulthandler \
  iron/applications/qwen3_0_6b/persistent/main.py \
  --stage generate --fast-generate --verify-generate \
  --max-new-tokens 2 \
  --layer-chunk-size 28 \
  --num-aie-columns 2 \
  --attention-columns 2 \
  --mlp-gate-up-columns 2 \
  --mlp-gate-up-pair-rows \
  --mlp-gate-up-direct-silu \
  --require-packed-weights \
  --build-dir build_qwen3_mlp_gateup_direct_silu_generate_default
```

Raw prompt pattern:

```bash
PYTHONUNBUFFERED=1 python -X faulthandler \
  iron/applications/qwen3_0_6b/persistent/main.py \
  --stage generate --fast-generate --verify-generate \
  --raw-prompt --prompt "<prompt>" \
  --max-new-tokens 6 \
  --layer-chunk-size 28 \
  --num-aie-columns 2 \
  --attention-columns 2 \
  --mlp-gate-up-columns 2 \
  --mlp-gate-up-pair-rows \
  --mlp-gate-up-direct-silu \
  --require-packed-weights \
  --build-dir build_qwen3_mlp_gateup_direct_silu_generate_<name>
```

Preflight accepted baseline before timing:

```bash
python iron/applications/qwen3_0_6b/persistent/real_graph_probe.py \
  --stages n-layer-final-only \
  --columns 2 \
  --attention-columns 2 \
  --mlp-gate-up-columns 2 \
  --mlp-gate-up-pair-rows \
  --mlp-gate-up-direct-silu \
  --layer-iterations 28 \
  --preflight-only \
  --build-dir build_qwen3_mlp_gateup_direct_silu_preflight \
  --clean-build
```

Static estimator:

```bash
python iron/applications/qwen3_0_6b/persistent/work_estimator.py \
  --mlir <generated-pos26-mlir> \
  --layers 28 \
  --position 26 \
  --show-kernels
```

Phase sensitivity probe:

```bash
python iron/applications/qwen3_0_6b/persistent/phase_timing_probe.py \
  --repeat 5 \
  --mlp-gate-up-direct-silu \
  --build-dir-prefix build_qwen3_phase_probe_direct \
  --json-output build_qwen3_phase_probe_direct_summary.json
```

Rejected row-group 8 preflight repro:

```bash
python iron/applications/qwen3_0_6b/persistent/real_graph_probe.py \
  --stages n-layer-final-only \
  --columns 2 \
  --attention-columns 2 \
  --mlp-gate-up-columns 2 \
  --mlp-gate-up-pair-rows \
  --mlp-gate-up-direct-silu \
  --mlp-gate-up-row-group 8 \
  --layer-iterations 28 \
  --preflight-only \
  --trace-placement \
  --build-dir build_qwen3_mlp_gateup_rg8_preflight \
  --clean-build
```
