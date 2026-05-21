<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Qwen3 Megakernel Runtime Symptoms

[Back to symptom index](symptoms.md).

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

## Experiment Flag Is Ignored By Generate

Symptom:

```text
The command includes a new experiment flag, but generate_position prints an
operator_name that still contains the old option value.

Example:
  command includes --mlp-gate-up-pair-rows
  operator_name contains mlpgatepair0
```

Diagnostic:

```text
Do not trust the CLI command alone. Read the operator_name printed by the
generate path, then trace which factory function constructs the op for that
stage.
```

Commands used:

```bash
rg -n "Qwen3PersistentNLayerFinalOnly|mlp_gate_up_columns" \
  iron/applications/qwen3_0_6b/persistent/main.py \
  iron/applications/qwen3_0_6b/persistent/generate_runner.py
```

Evidence found:

```text
main.py passed mlp_gate_up_pair_rows to the single-stage n-layer op path.
--stage generate returned earlier through run_generate().
generate_runner.py constructed the fast-generate op and did not pass the new
field.
```

Root cause:

```text
There were two Qwen3PersistentNLayerFinalOnly construction sites. Only one was
updated for the new experiment flag.
```

Fix:

```text
Pass the flag through generate_runner.py and recheck the printed operator_name.
```

Accepted evidence:

```text
operator_name=..._mlpgatepair1_...
token_match=True
```

## Clean Graph Edits Appear To Do Nothing

Symptom:

```text
After a graph or buffer-layout edit, compile/preflight time is unexpectedly tiny.
The result still looks like the old graph.
```

Diagnostic:

```text
Treat the tiny compile time as evidence of cached artifacts. Inspect or remove
the build directory before judging the edit.
```

Command used:

```bash
rm -rf build_qwen3_persistent_probe
source /opt/xilinx/xrt/setup.sh
.venv/bin/python iron/applications/qwen3_0_6b/persistent/real_graph_probe.py \
  --stages n-layer-final-only \
  --columns 1 \
  --layer-iterations 4 \
  --preflight-only \
  --build-dir build_qwen3_persistent_probe
```

Root cause:

```text
The operator reused existing artifacts. Runtime-only iteration is fast, but
graph edits require a clean build to prove generated MLIR/xclbin changed.
```

Fix:

```text
Use --clean-build or a new build directory after runlist, buffer layout, or
patch-site changes.
```

## Diagnostic Chunk Is Rejected Before Compile

Symptom:

```text
The next column-scaling diagnostic command fails before MLIR generation:

ValueError: Qwen3 n-layer final-only currently supports chunk sizes 1..8 or
the experimental full-depth chunk size 28; got 12.
```

Diagnostic:

```text
Read the operator guard and then inspect the actual design grouping logic. A
host-side validation error can be stale after the graph has learned a more
general resource pattern.
```

Evidence found:

```text
ops_nlayer.py rejected intermediate chunks >8 and !=28.
attention_design.py already used the same full-depth cache DMA grouping for
all layer_iterations>8:

  layer_writeback_dma_group_size = layer_iterations
  layer_cache_dma_group_size = layer_iterations
```

Root cause:

```text
The guard encoded an old validation state. It blocked the diagnostic needed to
locate the first numeric failure between chunk=8 and chunk=28, even though the
generated graph strategy was no longer limited to those two cases.
```

Fix:

```text
Allow diagnostic chunks 1..28 and keep the hard Qwen3-0.6B model limit at 28.
Update the unit test to construct 9, 12, and 28, and still reject 29.
```

Recheck:

```text
layer_iterations=12/16/18/19/20 all compile and preflight with
max_dma_tasks_per_fifo=1.
```

## Independent Drains Return Zero Until Each Drain Waits

Symptom:

```text
input-qkv-rope-present returns quickly and one output family is correct, but
other independently drained output regions are all zero:

new_mega_input_q_raw_errors: 1423
new_mega_input_k_raw_errors: 755
new_mega_input_v_raw_errors: 0

diagnostic slice:
  qcur nz 0 len 2048
  kcur nz 0 len 1024
  vcur nz 1024 len 1024
```

Diagnostic used:

```text
Do not start by changing matvec math. Slice the packed output by logical drain
region and count nonzero values per region.
```

Evidence found:

```text
The graph had 12 independent drains: current/debug shards for Q, K, and V.
Only the last V debug drain had wait=True. Q and K drains had no dependency on
that final V debug drain, so the host could read the output buffer before those
drain tasks had completed.
```

Root cause:

```text
Runtime task_group issue order is not a correctness dependency between
independent drains. Waiting on one unrelated drain does not prove the other
drains are visible in the host buffer.
```

Fix used:

```text
Set wait=True on each independent drain in the diagnostic boundary graph.
For a performance graph, either join outputs into a single dependent drain or
keep explicit waits on every independently host-visible region.
```

Recheck:

```text
input-qkv-rope-present:
  q_raw_errors=0
  k_raw_errors=0
  v_raw_errors=0
  current_errors=0
```

## CLI Stage Fails Before Any NPU Work

Symptom:

```text
python -m compileall iron/applications/new-mega/production
...
File "iron/applications/new-mega/production/main.py", line 182
    else:
    ^^^^
SyntaxError: invalid syntax
```

Diagnostic:

```bash
. .venv/bin/activate
python -m compileall iron/applications/new-mega/production
```

Evidence found:

```text
The new qkv-rope-attention CLI path added a third accepted-stage branch, but
the success message used if/else/else instead of if/elif/else. The failure was
pure Python control flow, before MLIR generation, XRT, ObjectFIFO, or kernel
execution.
```

Root cause:

```text
The entry script had an invalid branch structure after adding a new production
stage. Running the NPU command directly would have hidden this as a generic
"stage fails" symptom.
```

Fix:

```text
Change the success message dispatch to if fixed-attention / elif
qkv-rope-present / else qkv-rope-attention.
```

Recheck:

```text
compileall passed, then qkv-rope-attention ran two prompts with
qkv_errors=0 and context_errors=0.
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

## CREATE_HWCTX Fails After Many Distinct Xclbins In One Process

Symptom:

```text
RuntimeError: DRM_IOCTL_AMDXDNA_CREATE_HWCTX IOCTL failed (err=-22):
Invalid argument
```

Context:

```text
B1 real-shape GEMV scaling loaded many different GEMV xclbins in one Python
process. The first 24 configurations ran, then the following `down`
configurations failed while creating an XRT hardware context.
```

Diagnostic:

```text
Rerun the failed shape alone in a fresh process before changing the operator.
```

Evidence:

```text
down 1024x3072 at columns 1/2/4/8 passed in a fresh process.
```

Root cause:

```text
The failing boundary was runtime context lifetime/resource cleanup from loading
many distinct xclbins in one process. It was not a GEMV shape, TAP, placement,
or numeric failure.
```

Fix:

```python
import aie.utils as aie_utils

aie_utils.DefaultNPURuntime.cleanup()
```

Call cleanup after each independent operator configuration in sweep scripts.
Do not add this inside the hot token loop; this is for experiments that load
many different artifacts in one process.

Accepted evidence after the fix:

```text
aie.runtime_sequence(
  %arg0: memref<1024xbf16>,
  %arg1: memref<4195328xbf16>,
  %arg2: memref<5120xbf16>)
```

The metadata still exposes `bo0` through `bo4`, so three runtime BOs are within
the available host argument range.

Second occurrence diagnosed during two-layer persistent chunking:

```text
Qwen3PreflightError: Runtime BO metadata mismatch:
MLIR runtime_sequence has 7 memref arguments
main_kernels.json exposes only 5 HOST bo* arguments
```

Root cause:

```text
The two-layer Program had one memref each for hidden, weights0, weights1,
angles, outputs, cache0, cache1. The generated xclbin metadata still exposed
only five host BO slots, so the runtime ABI was invalid even though aiecc
produced artifacts.
```

Fix:

```text
Pack adjacent layer weights into weight_pair[2 * packed_weights_size] and
adjacent layer KV caches into cache_pair[2 * packed_cache_size]. Use TAP base
offsets to access layer 0 and layer 1 inside those pair buffers.
```

Accepted recheck:

```text
stage: two-layer-full-layer
preflight: ok runtime_memrefs=5 arg_specs=5 metadata_host_bos=5
compute_cores=21 max_dma_tasks_per_fifo=2 non_advancing_acquires=0
```

Third occurrence diagnosed during production input-QKV fusion:

```text
Qwen3PreflightError: Runtime BO metadata mismatch:
MLIR runtime_sequence has 6 memref arguments
main_kernels.json exposes only 5 HOST bo* arguments
```

Root cause:

```text
The first fused qkv-rope-attention-o-mlp graph added Q/K/V projection weights
as a separate runtime memref. That changed the runtime ABI from five BOs to
six BOs even though the xclbin metadata still exposed five HOST BO slots.
```

Fix:

```text
Pack input metadata and Q/K/V weights into one first runtime buffer:

qkv_packed_input =
  hidden || input_norm_weight || q_norm_weight || k_norm_weight || rope_lut ||
  q_proj_weight || k_proj_weight || v_proj_weight

Use fixed TAP offsets to feed metadata and weights to the appropriate Workers.
```

Accepted recheck:

```text
stage: qkv-rope-attention-o-mlp-fused
preflight: ok runtime_memrefs=5 arg_specs=5 metadata_host_bos=5
current_errors=0 attn_out_errors=0 ffn_hidden_errors=0
ffn_out_errors=0 layer_residual_errors=0
```

## AIECC Fails With File Name Too Long

Symptom:

```text
aiecc.py: error: Error creating temporary directory: File name too long
```

Diagnostic:

```text
Inspect the generated artifact path, not the graph math. If the filename
contains every dataclass parameter, the failure can happen before meaningful
MLIR/AIE resource allocation.
```

Root cause:

```text
After adding input-projection tile parameters to the fused production operator,
the inherited artifact name grew past the filesystem/tool temporary-directory
limit.
```

Fix:

```text
Override the production operator name with a short stable artifact prefix:

NewMegaFusedLayerInputQKV_s{max_seq_len}_c{chunk_size}_h{hidden}_i{intermediate}_{dev}
```

Recheck:

```text
The same graph compiles and runs after shortening only the artifact name.
```

## First Iteration Is Zero, Later Iterations Improve

Symptom:

```text
iteration 0: x_norm, queries_raw, keys_raw, values all zero
iteration 1: qkv values mostly aligned
```

Diagnostic:

```bash
source /opt/xilinx/xrt/setup.sh
.venv/bin/python iron/applications/qwen3_0_6b/persistent/main.py \
  --model Qwen/Qwen3-0.6B \
  --stage input-rmsnorm-qkv \
  --verify \
  --verify-repeat 3 \
  --build-dir build_qwen3_persistent_debug
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

Second occurrence during attention2 n-layer local diagnostics:

```text
Fatal Python error: Segmentation fault
Current thread:
  qwen3_cpu.py line 194 in rms_norm
  stage_runner.py line 183 in _run_n_layer_local_cache_diagnostic
```

Evidence:

```text
The crashing tensor was the prefix chunk output returned by XRTTensor.to_torch().
It was then passed into Torch RMSNorm. Converting it with host_owned_tensor()
before CPU math removed the segfault.
```

Additional fix:

```text
Any tensor returned by XRTTensor.to_torch() and later used by PyTorch CPU math,
serialization, or a second diagnostic phase must be wrapped with
host_owned_tensor().
```

## N-Layer Chunk 6/8 Compiles But Times Out At Runtime

Symptom:

```text
n-layer-final-only --layer-chunk-size 6 --verify
preflight: ok ... max_dma_tasks_per_fifo=6 ...
HostRuntimeError: Kernel returned ert_cmd_state.ERT_CMD_STATE_TIMEOUT
```

Diagnostic sequence used:

```text
1. Repack n-layer weights segment-major and prove compile/preflight improves.
2. Run chunk=4,5,6,7,8 with the same prompt position and clean builds.
3. Split current K/V writeback into groups of 4 and rerun chunk=6.
4. Split historical K/V cache fill into groups of 4 and rerun chunk=6/8.
```

Evidence found:

```text
segment-major weights, per-layer cache fill:
chunk=4 verify: pass, max_dma_tasks_per_fifo=4
chunk=5 verify: pass, max_dma_tasks_per_fifo=5
chunk=6 verify: timeout, max_dma_tasks_per_fifo=6

segment-major weights, writeback grouped by 4, per-layer cache fill:
chunk=6 verify: timeout, max_dma_tasks_per_fifo=6

segment-major weights, cache fill grouped by 4, writeback grouped by 4:
chunk=6 verify: pass, max_dma_tasks_per_fifo=2
chunk=8 generate: token_match=True over consecutive decode positions 9 and 10
```

Root cause:

```text
The timeout was not caused by external-kernel math or current K/V writeback.
The blocker was issuing six or more independent historical K/V cache fill tasks
to the same shallow cache ObjectFIFO in one runtime task group. The DMA stream
can fill ahead only to FIFO depth, then backpressure blocks the task schedule
needed to make forward progress.
```

Fix:

```text
Use segment-major chunk weights so each weight FIFO gets one linear DMA stream
for the whole layer chunk. Group historical K/V cache fills and current K/V
writeback by at most four layers. The real chunk=8 graph then stays at
max_dma_tasks_per_fifo=2 and avoids the runtime timeout.
```

Residual check:

```text
chunk=8 n-layer stage verify:
final hidden passes with 0 errors.
V cache current passes with 0 errors.
K cache current has a few deterministic strict-check errors in some partial
chunk tests, but full generate with --layer-chunk-size 8 matched CPU reference
tokens on the default prompt and raw prompt "The sequence is 1, 2,".
```

Full-depth recheck:

```text
After switching layer_iterations > 8 to one full-depth cache TAP, chunk=28
fast-generate matched CPU reference tokens on:

default prompt:
  Paris -> EOS

raw prompt "Fibonacci numbers: 1, 1, 2, 3,":
  new_text: ' 5, 8'
  token_match=True for decode positions 17, 18, 19, and 20

npu_layer_time_us_total: about 190-200ms
fast_op_call_s: about 0.191-0.200s
```

The stage-level full hidden check is stricter than token correctness here:
chunk=28 `n-layer-final-only --verify` showed hidden/cache drift, but greedy
tokens matched over the checked prompts. Treat chunk=28 as the performance
generate path and keep smaller chunks for numerical bisection.

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

Evidence after switching fast-generate to a disk packed artifact:

```text
fast_generate_weight_source: packed_artifact
fast_generate_weight_pack_s: 0.000000
fast_generate_weight_disk_load_s: 0.220869
fast_generate_weight_xrt_s: 0.292092
fast_op_call_s: 0.245879
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

Then add --prepare-weights and --require-packed-weights so fast-generate uses
one disk-backed global packed-weight BO with per-layer XRT sub-buffers.
```

Accepted recheck:

```text
--fast-generate --verify-generate --max-new-tokens 4
token_match: True for positions 6, 7, and 8
decode_s: 0.249-0.257 per NPU-decoded token
```

## Decode Compiles A New Artifact For Each Position

Symptom:

```text
generate_position_26_n_layer_28_compile_s: about 68s
generate_position_27_n_layer_28_compile_s: about 68s
```

The NPU body is correct, but practical decode still rebuilds an artifact for
each position.

Diagnostic:

```text
Diff adjacent-position artifacts before choosing a fix.
Use method 46, not a guess between instruction patching and bucket variants.
```

Evidence found:

```text
pos26 -> pos27:
  runtime .bin changed_bytes=8
  eight u32 patch candidates, all current-K/V DMA offsets +256 bytes
  main_aie_cdo_elfs.bin changed_bytes=56
  six main_core_*.elf files changed

pos63 -> pos64:
  K/V cache read length 32768 -> 65536 bf16 elements
  attention loops 1 active cache block -> 2 active cache blocks
```

Root cause:

```text
The graph has two position-specialized surfaces:
  runtime instruction DMA offsets
  AIE core immediate constants for position and valid softmax length
```

Fix direction:

```text
Do not patch only the runtime .bin; that leaves core ELFs compiled for the
wrong position. Do not rely on bucketed precompile alone; a bucket artifact
still bakes one exact position until position and valid length are runtime
metadata.

First move attention position metadata out of core immediates or prove ELF/CDO
patch sites. Then patch the remaining runtime DMA offset words or precompile
one variant per active cache-block count.
```

## Runtime Position Metadata Variant Times Out

Symptom:

```text
HostRuntimeError: Kernel returned ert_cmd_state.ERT_CMD_STATE_TIMEOUT
```

This happened on the first real NPU decode after adding runtime position and
valid-length metadata to attention streams.

Diagnostics used:

```text
1. Re-run with --max-new-tokens 2, not 1.
   max-new-tokens=1 only reports the prompt reference token and does not enter
   the NPU decode loop.

2. Try layer_chunk_size=1.
   The timeout still happened, so it was not a 28-layer watchdog problem.

3. Remove MLP with --attention-probe-only.
   The attention-only graph returned, which moved the failure from pure
   dataflow deadlock toward attention numeric corruption feeding the full layer.
```

Evidence:

```text
attention-probe-only:
  npu_time_us: 5500.721
  attention_probe_residual_max_abs: nan
  layer0_keys_cache_current_errors: 802

restored static-position path:
  token_match: True at decode_position=26
```

Root cause found so far:

```text
The tested stream-widening design for runtime position metadata is unsafe.
It either adds extra synchronization branches to the attention dataflow or
produces NaN attention output before MLP.
```

Fix used:

```text
Restore static position arguments for the accepted path. Keep the artifact-diff
diagnosis and pursue a different per-position reuse mechanism.
```

Do not diagnose this as "probably a long 28-layer dispatch" until
`layer_chunk_size=1 --max-new-tokens 2` has been run.

## Column Scaling Looks Slower On A Single Run

Symptom:

```text
full-MLP cols=4 verifies numerically, but one clean run reports a worse
npu_time_us than cols=1.
```

Diagnostic:

```text
Repeat the same compiled artifact several times and compare warm iterations.
Keep the same prompt, same stage, same build directory, and --verify enabled.
```

Command used:

```bash
source /opt/xilinx/xrt/setup.sh
.venv/bin/python iron/applications/qwen3_0_6b/persistent/main.py \
  --stage post-attn-rmsnorm-full-mlp \
  --num-aie-columns 4 \
  --verify \
  --verify-repeat 5 \
  --build-dir build_qwen3_full_mlp_cols4_verify
```

Evidence found:

```text
cols=1 late iterations: about 2.33-2.37 ms
cols=2 late iterations: about 1.85-1.88 ms after warmup
cols=4 late iterations: about 1.61-1.69 ms after warmup
all checked debug buffers: errors=0
```

Root cause:

```text
The first runtime call includes cold-start/runtime state effects and is not a
stable estimate for graph scaling. The graph was correct; the diagnostic method
was too weak.
```

Fix:

```text
Use repeated warm measurements or a latency histogram before accepting or
rejecting a column-scaling change.
```

## Column-Sharded Projection Verifies But Slows Down

Symptom:

```text
A graph-internal row-sharded GEMV compiles and every debug buffer has
errors=0, but npu_time_us is worse than the single-worker version.
```

Diagnostic:

```text
Do not stop at correctness. Inspect generated MLIR to prove whether the new
workers are really parallel, then compare the amount of work moved to columns
against the extra broadcast, join, copy, and DMA tasks.
```

Commands used:

```bash
rg -n "new_mega_mlp_o_weight_|new_mega_mlp_attn_out_shard|tile\\(" \
  build_new_mega_production_column_o_run/*.mlir

source /opt/xilinx/xrt/setup.sh
. .venv/bin/activate
python iron/applications/new-mega/production/main.py \
  --stage qkv-rope-attention-o-mlp-fused \
  --build-dir build_new_mega_production_column_o_run
```

Evidence found:

```text
O-only row sharding:
  MLIR placed O shard workers on tile_0_4 and tile_0_5
  join worker was on tile_1_2
  context FIFO broadcast to both O shard workers
  verification errors=0
  npu_time_us=11886.655

previous D1.3c baseline:
  npu_time_us was about 10.8-11.2ms
```

Root cause:

```text
The O projection is not large enough in this graph for row sharding to pay for
the added ObjectFifo broadcast, shard join, copy, and additional weight DMA.
The graph was parallel, but the chosen phase was too small.
```

Fix:

```text
Move the same row-sharded pattern to a larger phase before accepting it as a
performance change. In production, adding two-column gate/up sharding together
with O sharding kept correctness and reduced the default prompt to 9581.209us.
Adding two-column down-projection sharding reduced the same stage further to
8585.632us.
```

## HuggingFace Snapshot Download Fails During Local Verification

Symptom:

```text
httpx.RemoteProtocolError: Server disconnected without sending a response
```

Diagnostic:

```bash
ls ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots
```

Root cause:

```text
The model was already cached locally, but passing the repo id made
resolve_model_dir() call snapshot_download(), which still contacted
HuggingFace before returning the cached snapshot.
```

Fix used:

```text
Pass the local snapshot directory through --model for compile/verify runs.
This keeps network failures out of megakernel diagnosis.
```

## NPU Run Appears Hung Before Any Result Is Printed

Symptom:

```text
production phase-owned run prints only XRT environment setup and then no
result for many minutes after adding O row chunks.
```

Diagnostic:

```text
Do not classify this as an ObjectFIFO deadlock until proving the process has
entered device execution. Check whether the new host-side packet builder or
reference grew a nested Python scalar loop.
```

Evidence from D1.5w:

```text
O projection was split into 32 row chunks.
The packet-level reference still used nested Python loops:

  lane * layer * O chunk * producer lane * target row * context dim

That inflated the host reference into hundreds of millions of Python scalar
multiply-adds. The NPU graph had not produced a failure yet.
```

Fix:

```text
Vectorize the reference at the same semantic boundary:

  group_partials += producer_weight_block @ producer_context

After this change, the same run completed and reported real NPU diagnostics.
```

Rule:

```text
When a megakernel phase is expanded by chunking, update the diagnostic
reference complexity at the same time. A correct graph can look hung if the
host verifier scales as Python scalar loops.
```
