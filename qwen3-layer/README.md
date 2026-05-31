# qwen3-layer

This directory implements the qwen3-layer contract described in
`experiments/qwen3-dataflow.md` and contains a runnable NPU integration
backend.

The current implementation is the active qwen3 full-layer NPU integration path:

- `c1r2` full-vector station with `2049`-dword packet0 replay contract.
- `c1r1` shared activation bridge for packet2/O, packet0 replay, and packet1/down.
- `c1r3` Q/K norm + RoPE postprocess station with packet8/9 current KV routes.
- `c6r1` Q fanout, attention return gather, and FFN intermediate gather.
- `c6r2` up/gate SwiGLU slice station.
- Main16 projection workers and row1 compact routes.
- Shape-A/B carrier and return windows.

## Files

- `contract.py`: Qwen3 layer constants and ABI checks.
- `check_contract.py`: integration check for active generators, resource
  manifests, token gates, and retired-code absence.
- `mlir_utils.py`: shared MLIR-AIE BD, lock, queue, and runtime sequence
  helpers used by runnable cases.
- `physical_contract.py`: executable channel ownership checks for the
  compact-only full-layer cases and the MyLM-aligned Q4NX dual-input ABI.
- `resource_manifest.py`: explicit tile-local buffer, lock, and BD ownership
  manifest checks for main16 QKV residency and phase overlap.
- `weight_stream.py`: shared row1 Q4NX patch-ring generator for host/shim
  weight ingress into main16 DMA1.
- `projection_schedule.py`: single source for Q/K/V body record counts and
  O/upgate/down Q4NX tail weight chunk bases.
- `compact_dataflow.py`: shared row1/c1r1 compact gather, bridge, hub, and
  MyLM-aligned row1 S2MM4/5 weight fanout generator used by the frontier.
- `attention_dataflow.py`: shared Shape-A/B tile placement, hub BD ownership,
  KV output BDs, and packet2 attention return hub used by active attention
  generators.
- `qkv_compact_reference.py`: shared Q/K/V/O compact record layout helpers for
  active attention integration references.
- `cases/full_layer_engine_generate.py`: the single full-layer fused-engine
  MLIR generator. Active slices import this generator and crop the physical
  phase range instead of keeping separate debug dataflows.
- `cases/full_layer_engine_reference.py`: shared physical reference helpers,
  constants, weight layout, cache writeback, attention, O/FFN, and final hidden
  validation used by the active runners.
- `cases/kv_cache_dataflow.py`: shared current-token K/V cache
  writeback, rounded KV scan, and Shape-A/B runtime-start helper generation.
- `q4nx_reference.py`: shared Q4NX chunk reference math used by integration
  checks. It now follows the MyLM 5120-byte Q4NX chunk layout instead of the
  old row-major synthetic chunk layout.
- `qwen3_model.py`: typed MyLM `model.q4nx` parser for Qwen3-8B, including
  layer projection tensors, RMSNorm/QK norm weights, and row1/main16 weight
  stream construction.
- `qwen3_download.py`: small downloader for the MyLM Qwen3-8B-NPU2 model files
  listed in `qwen3_model.py`.
- `run_reference_decode.py`: CPU full-model Qwen3-8B reference decode runner
  over embedding, layers, final RMSNorm, and lm_head. It can stop at a layer
  prefix and dump bf16 layer tensors for MyLM comparison.
- `tools/build_mylm_forward_probe.sh`: builds the local MyLM prefix probe used
  to compare Qwen3 layer prefixes against `/var/home/taowen/projects/MyLM`.
- `tools/compare_bf16_dump.py`: compares raw bf16 dumps and reports top-k,
  max error, mean error, and the first mismatching lane.
- `tools/compare_main16_q4nx_disasm.py`: compares MyLM c2r2 raw main16 Q4NX
  disassembly with the active IRON main16 object and full-core phase-control
  shape.
- `tools/check_main16_raw_abi.py`: compares the active generated main16
  BD/lock/buffer ABI against the extracted MyLM raw-main16 contract.
- `tools/check_main16_program_shape.py`: rejects replacement main16 ELFs that
  still look like MLIR-expanded per-chunk C++ helper control instead of a
  MyLM-style fixed tile-local core program.
- `tools/probe_aiecc_core_codegen.py`: inspects one generated core's aiecc
  LLVM/Peano pipeline and optionally replays a manual no-unroll core compile.
- `tools/audit_aiecc_driver.py`: runs a fast-fail `aiecc -n -v` audit and
  checks which driver flags actually change the child Peano `opt/llc`
  commands.
- `tools/probe_external_lock_dispatcher.py`: proves that a linked C++ AIE core
  dispatcher can compile AIE2P lock builtins and generate a hardware loop. It
  is a compile-shape probe; final linked ELFs still need packet/BD/lock
  contract checks and hardware validation.
- `tools/replay_main16_core_compile.py`: replays all 16 generated main16 core
  compiles with Peano `opt -disable-loop-unrolling`, producing replacement
  ELFs for ELF-backed package probes.
- `tools/inspect_core_program_txn.py`: inspects transaction MLIR and confirms
  which `elf_file` payloads become core-program `config_blockwrite_data`.
- `tools/repack_core_program_txn.py`: strips stale core-program transaction
  payloads and regenerates them from the current `aie.core` `elf_file` ELFs.
- `tools/externalize_core_programs.py`: replaces generated source MLIR core
  bodies with ELF-backed `aie.core` declarations copied from a donor
  transaction, so `aiecc --no-compile` can package replacement core ELFs.
- `tools/package_externalized_design.py`: copies the donor project directory,
  optionally replaces all 16 main16 ELFs, runs the all-core ELF-backed
  `aiecc --no-compile` package step, and checks the resulting transaction
  payloads.
- `tools/wrap_raw_aie_program.py`: wraps raw AIE2P program bytes as an
  ET_EXEC ELF with a loadable `.text` `PT_LOAD` segment for replacement-core
  package probes.
- `main16_q4nx_mylm_compare.md`: current main16 performance/reverse-analysis
  conclusion and the next raw Q4NX microkernel direction.
- `npu_build.py`: shared MLIR, xclbin, and NPU runtime helpers. It scans
  generated MLIR `link_with` attributes and compiles the required role objects,
  so runners do not duplicate kernel-object ownership.
- `run_stage_budget.py`: runs active NPU integration cases and prints stable
  `stage_budget:` / `perf_budget:` lines for c1r2, Q/K/V, attention-O, full
  hidden_out, row1 weight fanout, and main16 Q4NX compute.
- `main_projection_q4nx_fast.cc`: main16 Q/K/V/O/up/gate/down Q4NX projection,
  flush, and record emit kernels.
- `main_projection_q4nx_asm.s`: generated source-assembly main16 Q4NX probe
  object linked into the single main16 role object; unreferenced sections are
  garbage collected until a production exact lane body is wired in. It contains
  both the canonical MyLM-style group probe and a callable exact-lane body
  candidate that writes FP32 partial sums. Regenerate or check it with
  `tools/generate_main16_q4nx_asm.py`.
- `edge_attention.cc`: Shape-A/B edge attention kernels for KV scan, online
  softmax, weighted V, and accumulator merge.
- `postprocess_qkv.cc`: c1r3 Q/K norm, RoPE/layout pack, and current K/V
  writeback payload kernels.
- `full_vector_station.cc`: c1r2 hidden/O compact replay, full-vector station,
  and final compact/output helpers.
- `swiglu.cc`: c6r2 up/gate SwiGLU kernels that produce down activations.
- `record_format.h` / `qwen3_constants.h`: shared header layout and constants.
- `cases/`: modular case registry and wrappers for executable integration
  boundaries, including the Q/K/V -> Shape-A/B -> O compact path.
- `run_npu.py`: thin CLI dispatcher for real NPU integration cases.

## Check

```bash
.venv/bin/python qwen3-layer/check_contract.py
```

## Run On NPU

```bash
.venv/bin/python qwen3-layer/run_npu.py --check-only
.venv/bin/python qwen3-layer/run_npu.py --check-only --download-model
.venv/bin/python qwen3-layer/run_npu.py --case qwen3-8b-decode-layer --model-path /var/home/taowen/flm/models/Qwen3-8B-NPU2
.venv/bin/python qwen3-layer/run_npu.py --build-only
.venv/bin/python qwen3-layer/run_npu.py
.venv/bin/python qwen3-layer/run_npu.py --case qwen3-8b-c1r2-input-norm-replay
.venv/bin/python qwen3-layer/run_npu.py --case qwen3-8b-qkv-cache-write-bridge --current-token 31
.venv/bin/python qwen3-layer/run_npu.py --case full-layer-qkv-prefix --current-token 31
.venv/bin/python qwen3-layer/run_npu.py --case full-layer-attention-o-bf16 --current-token 31
.venv/bin/python qwen3-layer/run_npu.py --case row1-weight-stream-perf
.venv/bin/python qwen3-layer/run_npu.py --case main16-q4nx-compute-perf
.venv/bin/python qwen3-layer/run_stage_budget.py --tokens 31,91
.venv/bin/python qwen3-layer/run_stage_budget.py --stages row1-weight,main16-compute --tokens 127
.venv/bin/python qwen3-layer/run_reference_decode.py --prompt Hello --max-new-tokens 1 --stop-layer 1 --expect-token-ids 51920
```

This runner is intentionally an integration boundary, not a tiny unit test. The
default case is now `qwen3-8b-decode-layer`: it validates the real MyLM
Qwen3-8B-NPU2 assets, sends raw hidden plus RMSNorm/QK norm aux weights through
the c1r2/c1r3 path, constructs the real layer Q4NX weight stream in the main16
ABI, and compares the 2048-dword final hidden payload with
`Qwen3LayerReference`. token31 is the default because it keeps a multi-block KV
scan while still producing a non-zero end-to-end signal through the production
bf16 attention-O slice. The current single-layer frontier passes the real-model
hidden-out contract with `abs_tol=0.01, rel_tol=0.05`; this is a working decode
frontier, not yet the final multi-layer production error budget.

The runnable registry is deliberately small. The public cases are:

- `qwen3-8b-decode-layer`: the real model full-layer decode frontier.
- `full-layer-qkv-prefix`: the full-layer physical Q/K/V prefix slice.
- `full-layer-attention-o-bf16`: the production bf16 attention -> O slice.
- `qwen3-8b-qkv-cache-write-bridge`: the real-model current K/V writeback slice.
- `qwen3-8b-c1r2-input-norm-replay`: the real-model c1r2 RMSNorm replay boundary.
- `row1-weight-stream-perf`: full-layer weight stream through row1 S2MM4/5 and
  row1 MM2S0..3 into main16 DMA1 sinks, with compute disabled and main16 done
  gathered through the same packetized row1 -> c1r1 bridge shape as full-layer.
- `main16-q4nx-compute-perf`: full-layer Q4NX chunk count on all main16 cores
  with activation/weight DMA disabled, using a row1 done gather for completion.

Historical migration cases for the old 608-patch weight-stream oracle,
deterministic full-layer tail, patched descriptor runner, and standalone
down/SwiGLU/Q4NX bridges are no longer runnable modes. Their useful constraints
now live in shared generators: `weight_stream.py` owns row1 S2MM4/5 weight
ingress and row1 MM2S fanout, `compact_dataflow.py` owns the frontier compact
bridge plus row1 weight-stream composition, and `physical_contract.py` validates
that the frontier does not regress to the old row1 S2MM0/1 weight route.

`--check-only` validates generated MLIR structure, token-gate schedules, and
physical contracts. `--build-only` verifies routing, core compilation,
instruction generation, PDI, xclbin generation, and instruction patching. For
the full decode case, the runner builds one token127 capacity xclbin/PDI and
patches `design.bin` for the requested token. `--current-token` still selects
one target `DecodeSchedule`: current-token RTP value, current-write byte offset,
Shape-A/B block count, Shape-A tail valid-token count, scan BD
`iteration_size`, and queue `repeat_count` all come from that schedule; the
K/V BO is allocated at the capacity schedule size.

The two full-layer slices are cut from the full-layer topology instead of
handwritten debug dataflow. The prefix slice validates hidden replay, row1
weight ingress, main16 Q/K/V residency, and c1r3 packet8/9 current K/V
writeback. The attention-O slice continues through the production
`qwen3_attention_bf16_*` path, packet2 handoff, and main16 O phase without a
deterministic/debug attention producer.

The two perf slices are diagnostic boundaries, not alternate backends. The row1
slice answers whether host/shim -> row1 -> main16 DMA1 can stream the full
115MiB layer weight payload continuously; it now waits for all 16 main sinks to
return the 1472-chunk done count through row1 -> c1r1 -> shim_out, so the timing
is not just a host input-queue drain. The main16 slice answers whether the
current Q4NX kernel throughput alone is already a layer-time bottleneck. They
share the same role objects and ABI constants as the full-layer generator, so a
performance regression in these cases is actionable instead of a separate debug
dataflow artifact.

Recent token31 measurements with the single active `main_projection_q4nx_fast.cc`
role:

- `row1-weight-stream-perf`: `8.717 ms`, `12.883 GiB/s`
- `main16-q4nx-compute-perf`: `13.649 ms`, repeated after the stage run
- `qwen3-8b-c1r2-input-norm-replay`: `1.876 ms`
- `qwen3-8b-qkv-cache-write-bridge`: `7.963 ms`
- `full-layer-attention-o-bf16`: `12.346 ms`
- `qwen3-8b-decode-layer`: `24.802 ms` after record-granular compact and
  source-side down replay, `final_hidden_out max_abs=0.0078125`

These numbers do not support row1 weight fanout as the primary reason main16 is
slow: the isolated full-layer weight stream is faster than the isolated main16
Q4NX compute loop. They do show two large remaining costs: the main16 Q4NX
microkernel itself and the attention/O integrated slice. If a future regression
looks like main16 starvation, the next diagnostic should be a generated
progress-counter slice that records activation-acquired, weight-acquired,
compute-done, and record-released counts per main tile, rather than adding debug
taps to the production full-layer path.

Older C++ unroll probes were removed from the code path. They improved narrow
slices but either did not fit full-layer program memory or regressed full decode
to `30.092-30.963 ms`. The repository now keeps one main16 C++ implementation:
the fastest verified full-decode version.

The full-layer generator now emits 17-dword compact records directly from the
active accumulator. The old `q4nx_output` scratch buffer, block-accumulator
variant, and `q4nx_flush_output_fast` ABI are gone from the active path. The
main remaining gap is still the Q4NX microkernel instruction shape, not another
record-copy helper.

The aiecc core-codegen boundary is now explicit. A generated QKV-prefix
main16 core enters Peano as compact LLVM IR with four `q4nx` references, then
the stock aiecc child pipeline runs:

```text
opt --passes=default<O1> -inline-threshold=10
llc -O2 --march=aie2p --function-sections
```

That `opt` step fully unrolls the constant-trip phase loops before `llc` can
turn them into the MyLM-style phase body. `tools/probe_aiecc_core_codegen.py`
checks this against the generated project:

```bash
.venv/bin/python qwen3-layer/tools/probe_aiecc_core_codegen.py \
  --manual-output-dir /tmp/iron_aiecc_core_codegen_probe
```

The driver flag audit is separate from that generated-project probe:

```bash
.venv/bin/python qwen3-layer/tools/audit_aiecc_driver.py
```

It confirms that `--disable-loop-unrolling` and
`--opt-disable=loop-unroll` do not reach the child Peano `opt` command. The
Peano `llc` hidden help exposes AIE loop-scheduler/hardware-loop controls such
as `--aie-loop-aware`, but passing those flags to this `aiecc` driver also does
not change the child `llc` command. `-O0` and `-O3` do change the child
optimization level, but they still do not provide the MyLM-style scheduled
phase body.

The important nuance is that LLVM-AIE/Peano itself can still emit AIE hardware
loops. The stock aiecc problem is specifically the MLIR-core phase control
being optimized by the fixed child `opt` command before `llc`. This probe puts
the phase control in a linked C++ AIE core object instead:

```bash
.venv/bin/python qwen3-layer/tools/probe_external_lock_dispatcher.py --force
```

It compiles a tiny linked C++ AIE dispatcher with AIE2P
`acquire_greater_equal()` and `release()` builtins. The expected shape is one
`acq`, one `rel`, and `lc/ls/le`
hardware-loop setup in the object disassembly. AIE branches have delay slots,
so lock instructions printed after `j/jl/ret` are not automatically outside the
loop. The latest QKV-prefix timeout was traced to a different issue: main16 and
row1 now decode as unpacketized compact sources, and row1 has the MyLM-style
`17+16+16+16 -> 65` packer. c1r1 now has the matching
`65+64+64+64 -> 257` record packer. The runnable
`qwen3-8b-qkv-compact-output` slice drains all 12 Q/K/V global compact records
to host as a 3084-dword stream, so it does not leave the AIE graph blocked after
the first record. Headers are bit-exact and the current Q4NX payload delta is
bounded to 1 bf16 ULP, which is a microkernel/reference rounding issue rather
than a compact transport failure. The static contract check rejects the old
mixed phase-sized bridge before another hardware timeout.

That means the next production direction does not have to throw away MLIR-AIE
topology generation: keep MLIR for buffers, locks, BD rings, stream routes,
runtime sequence, PDI, and xclbin packaging. The MyLM-style main16 source is
now paired with a record-granular c1r1 compact tree before the phase
dispatcher is moved out of the MLIR `aie.core` body into a linked C++ AIE core
kernel or raw core program.

On the latest full-decode probe after the 17-dword record ping/pong migration,
stock aiecc reports `llvm_q4_refs=16` but `opt_q4_refs=260`, with
`.text=14416`, `disasm_jl=157`, and `disasm_acq=250`. Earlier manual replay of
the same aiecc core compile shape with `opt -disable-loop-unrolling` proved the
child `opt` unroll can be avoided outside the stock driver, but passing
`--disable-loop-unrolling` to `aiecc` itself does not change the printed child
`opt` command. This makes the boundary clear: either patch/replay the core
compile and package replacement ELFs, or move to raw/scheduled main16 programs;
generator cleanup alone will not produce the desired phase body.

`tools/replay_main16_core_compile.py` makes the useful part of that manual
path repeatable for all 16 main tiles:

```bash
.venv/bin/python qwen3-layer/tools/replay_main16_core_compile.py \
  --project-dir /tmp/iron_full_decode_txn_probe/prj \
  --output-dir /tmp/iron_full_decode_main16_disable_unroll_elfs \
  --force
```

Packaging those replacement ELFs through
`tools/package_externalized_design.py` preserves the generated topology and
runtime sequence while changing only the main16 core programs. The hardware
result is correct but small:

- Older phase-sized compact donor, `full-layer-qkv-prefix` token31:
  `124872.8 us`, PASS, versus stock `142702.9 us`.
- `qwen3-8b-decode-layer` token31: `28990.9 us`, PASS, versus the same rebuilt
  stock design at `29404.6 us`.

This proves replacement-ELF packaging works and stock `opt` loop unrolling
costs real time. It also proves this alone is not the MyLM-class fix: the
replacement no-unroll core still uses ordinary branch loops around the C++
Q4NX helper instead of a raw scheduled phase body.

For runnable raw main16 experiments, use the all-core ELF-backed package tool
rather than feeding transaction MLIR back into `aiecc`:

```bash
.venv/bin/python qwen3-layer/tools/package_externalized_design.py \
  --input-mlir /tmp/iron_fresh_qkv_prefix/design.mlir \
  --donor-transaction-mlir /tmp/iron_txn_probe_fresh/design.txn.mlir \
  --donor-project-dir /tmp/iron_txn_probe_fresh/prj \
  --output-dir /tmp/iron_raw_main16 \
  --force
```

This path keeps the MLIR-AIE generated topology, routing, BD/lock setup,
runtime sequence, PDI, and xclbin packaging, but it does not let aiecc recompile
or overwrite the core ELFs. The package tool records
`core_program_inspection.txt` and fails if the 16 main16 ELF text sizes change
during `--no-compile`, or if the transaction does not contain 16 matching
main16 payloads. The verified probe produced a `2372`-byte runtime instruction
stream, a `213952`-byte xclbin, and 16 main16 `9952`-byte payloads.

Do not convert only main16 to an `elf_file` empty-core and then run a normal
compile. In the probe, aiecc compiled the empty main16 core and overwrote the
donor ELF with a tiny `160`-byte `.text` program. The safe path is all-core
ELF-backed source MLIR plus `--no-compile`, using a throwaway copied project
directory when replacing selected ELFs.

The lower-level transaction-only repack path is still useful for inspecting
payload regeneration:

```bash
.venv/bin/python qwen3-layer/tools/repack_core_program_txn.py \
  --txn /tmp/iron_txn_probe_fresh/design.txn.mlir \
  --elf-dir /tmp/iron_txn_probe_fresh/prj \
  --source-output /tmp/iron_txn_probe_fresh/repack_source.mlir \
  --output-txn /tmp/iron_txn_probe_fresh/repacked.txn.mlir
.venv/bin/python qwen3-layer/tools/inspect_core_program_txn.py \
  --txn /tmp/iron_txn_probe_fresh/repacked.txn.mlir
```

This intentionally strips stale `config_blockwrite_data_*` globals and the old
`@configure()` transaction sequence before rerunning
`aie-opt --convert-aie-to-transaction=elf-dir=...`. Running that pass directly
on an already-converted transaction duplicates core-program payloads.

Raw replacement ELFs must be loadable executables. A minimal ET_REL wrapper with
only a `.text` section is enough for disassembly, but `aiecc --no-compile` does
not convert it into core-program blockwrite payloads. The wrapper used for
package probes is:

```bash
.venv/bin/python qwen3-layer/tools/wrap_raw_aie_program.py \
  /tmp/mylm_solidify_L31/programs/c2r2_program.bin \
  /tmp/iron_mylm_main16_exec_elfs/main_core_2_2.elf
```

Using ET_EXEC plus `PT_LOAD`, the package tool successfully generated 16
main16 `14868`-byte payloads from the extracted MyLM raw main16 images. This is
a toolchain proof only, not a runnable replacement: the MyLM raw core assumes
the same outer main16 ABI that IRON now matches, but it also assumes MyLM's raw
whole-core phase program, row1 compact timing, and hand-written Q4NX microkernel
body.

Check the raw-core ABI gap explicitly with:

```bash
.venv/bin/python qwen3-layer/tools/check_main16_raw_abi.py
```

Current output is `raw_main16_abi_ready=true` for the checked row0 main16 tile:

- activation now matches MyLM: BD0/1, length 128 dwords, bases
  `0x8000/0xc000`, locks L0->L1.
- weight now matches MyLM: BD2/3, length 1280 dwords, bases
  `0x2800/0x4000`, locks L2->L3.
- record now matches MyLM: BD4/5, length 17 dwords, bases
  `0x3c1c/0x541c`, locks L5->L4.

The next raw-main16 migration is not more main16 source-side ABI work. It is
first converting row1/bridge compact gather to the same record-granular
65/257-dword quanta, then replacing the MLIR-generated main16 phase control
with a MyLM-style raw scheduled phase body that consumes the matching
activation, weight, and record rings.

MyLM and this implementation are aligned on the layer dataflow principle:
main16 consumes activation/weight/record rings, row1 splits compact and weight
traffic onto separate channels, c1r2/c6r1 act as source-side replay stations,
and attention feeds O through packet2 without host/debug drain. The remaining
gap is implementation hardness: MyLM uses raw scheduled core programs and
static BD/lock bodies, while this tree still emits MLIR phase control around the
single role-level AIE C++ kernel. The hard disassembly evidence is that MyLM
`c2r2` has 5 calls into the shared Q4NX loop, 11 total `jl`, and 16/15
`acq`/`rel` instructions; the current IRON full main core has 129 Q4 helper
calls, 157 total `jl`, and 232/232 `acq`/`rel`. That is why small Python
generator cleanup cannot plausibly produce a 5x gain.

The CPU decode reference is not yet the final multi-layer oracle. It now uses
the correct MyLM Q4NX formula `weight = int4 * scale + offset`, where the second
5120-byte chunk segment is a bf16 offset rather than an integer zero point.
Against the MyLM probe, raw token `9707` (`Hello`) has matching top-k through
the early and middle prefixes: layer1 top token `51920`, layer4 top token
`70765`, and layers8/16/24/32 top token `143358`. The remaining reference
numerical gap is the final tail: layer35 starts to reorder near-tied logits and
layer36 diverges from MyLM (`323` vs the Python reference `11`). Until that is
closed, full 36-layer Python logits should be treated as diagnostic data, while
the stable NPU integration oracle remains the single-layer `Qwen3LayerReference`
and the MyLM prefix dump comparison.

For MyLM prefix comparison:

```bash
.venv/bin/python qwen3-layer/run_reference_decode.py --prompt Hello --max-new-tokens 1 --stop-layer 1 --dump-dir qwen3-layer/build/reference-dump-smoke --dump-layers 0 --top-k 5
qwen3-layer/tools/build_mylm_forward_probe.sh
qwen3-layer/build/mylm_forward_probe --layers 1 --token-id 9707 --cache-layer 0 --dump-prefix qwen3-layer/build/mylm-smoke --top-k 5
.venv/bin/python qwen3-layer/tools/compare_bf16_dump.py --expected qwen3-layer/build/mylm-smoke.logits.bf16 --got qwen3-layer/build/reference-dump-smoke/pos0000.logits.bf16 --abs-tol 0.5 --top-k 5
```

The stage-budget K/V lines are deliberately split by scope. `current_*_slot`
reports only the newly written token, `valid_*_cache` reports logical tokens
`0..current_token`, and `capacity_*_unchanged` verifies the unused capacity BO
tail when the full decode runner patches a token127 build down to a smaller
target token. Non-zero max error lines include `max_abs_at=token/head/dim/...`;
failing lines also include `first_mismatch=...`.

The up/gate bridge intentionally uses one S2MM channel per row in each row1
column compact tile. A single S2MM channel with multiple packet BDs is not a
valid replacement for ordered packet-id scheduling when independent main tiles
produce rows concurrently; the memtile consumes arrivals in stream order, so
row0 gate can be captured by the row1 up BD. Splitting rows across channels
keeps the packet handoff deterministic while still using legal memtile BD banks
for AIE2p.

The streaming KV scan case keeps two similar hardware constraints explicit.
First, the four row1 KV output channels cannot share one generic "payload full"
token when the shim is slicing K0/V0/K1/V1 independently; each slot needs its
own empty/full lock, otherwise an MM2S channel can consume the readiness token
for the wrong slot and the run can timeout. Second, generated runtime
descriptors must not patch an address argument that the xclbin kernel metadata
does not expose. The kvscan case therefore uses four BO arguments
`Q, left_kv_cache, right_kv_cache, output`, and `mlir_utils.py` validates the
maximum `arg_idx`.
The current-K/V cache writeback keeps two generator rules explicit. AIERT
requires each DMA BD block that uses locks to carry both acquire and release
operations; the earlier row1 bypass merge failed when one BD only acquired a
lock. AIE2p memtile BDs are also banked by channel parity: even DMA channels
use BD 0..23, odd DMA channels use BD 24+. The current writeback path uses two
linked shim S2MM BDs per side instead of eight head-slice BDs. c1r3 emits the
current K/V stream as even dwords for all heads followed by odd dwords for all
heads; BD14 writes the even positions and BD15 writes the odd positions with
`d0_size=32`, `d0_stride=1`, `d1_size=8`, and `d1_stride=1023`.

The scan side is now split by K/V plane to avoid the next descriptor wall. A
single-channel scan needs one runtime BD for each K0/V0/K1/V1 slice per block,
so four blocks would consume BD0..15 before current writeback has any legal BD
left. The passing split design sends K0/K1 over shim MM2S channel 0 and V0/V1
over shim MM2S channel 1. Each rounded block therefore uses two scan BDs instead
of four. The first static split version reached seven rounded blocks with
BD0..6 for K-side scans, BD7..13 for V-side scans, and BD14/15 for current
writeback. The current version keeps only one K scan BD and one V scan BD per
side: `buffer_length=4096`, `iteration_size=<rounded_blocks>`,
`iteration_stride=8191`, and `push_queue repeat_count=<rounded_blocks - 1>`.
`iteration_size` alone only configures the BD address generator; without the
queue repeat count the hardware executes one 4096-dword segment and the
downstream block loop waits forever. Row1 mirrors the split channels with S2MM
channel 0 for K and channel 1 for V; the V input uses high-bank memtile BDs
28/29, while the existing output BDs remain 2/24/4/26. The current token is
patched through `aiex.npu.rtp_write` into a c1r3-local `post_current_token`
buffer consumed by the `postprocess_qkv.cc` role.
That RTP write must be paired with a runtime-start lock. Without the lock,
main16 can produce Q/K/V compacts and c1r3 can enter
`qwen3_postprocess_q4nx_body_payload` before the runtime sequence writes the RTP; the
observed failure was a clean token0 current-slot writeback while the descriptor
path and cache scatter were otherwise correct. The current-token RTP is consumed
by the c1r3 postprocess role, not by a shared bridge object. The old token15
literal was the root cause of the first current-slot cache writeback mismatch.
`mlir_utils.py` validates the lock-balance, memtile-BD-bank, scan/write
BD-disjointness, `npu.writebd` ID/field ranges, push-queue repeat count, and
RTP-before-runtime-lock ordering before aiecc reaches CDO generation.

The 2026-05-29 scale probe found the descriptor boundary precisely. AIECC
rejects `aiex.npu.writebd` IDs above 15. Reusing overlapping shim BD IDs after
`npu.sync` compiled but timed out on NPU. A single 2D current-write BD also
failed: `d0_stride=1` completed but only filled every other dword, while
`d0_stride=0` compiled but timed out. The first passing three-block design used
the explicit two-BD even/odd scatter described above. The next passing
seven-block design compressed scan descriptors by splitting K and V onto two
shim/memtile channels. The current scheduled design adds iterated scan BDs
plus queue repeats, Shape block-count RTPs, and runtime-start locks. The
standalone scan scale probe emits 447 runtime instructions and passed token127
with eight rounded blocks and token91 with six rounded blocks. The active
full-layer runner now builds a token127 capacity xclbin/PDI and has passed
token0, token1, token31, and token91 by patching `design.bin`; token127 passes
with the unpatched capacity instruction stream. This proves the high-level AIEX
path can express descriptor reuse for the KV scan, as long as the BD iteration
fields and queue repeat count are programmed together.

The Shape-A/B attention math stays in explicit fixed point for now. Shape-A
stores packed 16-bit Q12 exp weights and int32 block max/sum values; Shape-B
keeps int32 accumulators and applies the same Q12 scale during online merge.
This is closer to production online softmax than the old stair-step contract,
but still needs calibration against the Qwen3 bf16/fp32 kernel instead of
depending on compiler lowering of wide integer math.
The same rule now applies to c1r2 and main16 FFN contract kernels: generated
tile code should prefer bounded signed int32 loops, shifts, and masks. A probe
that used an unsigned LCG, unsigned modulo, and 64-bit integer sqrt produced a
stable but wrong scale on NPU before c6r2. The passing c1r2 path uses a simple
signed compact-to-lane mix, a bounded int32 sqrt, 64 int32 projection
accumulators, and Q8 s16-pair packing before fixed-point SwiGLU.

## Check

```bash
.venv/bin/python qwen3-layer/check_contract.py
```

The executable cases prove the current hardware contracts. The active registry
keeps the real full-layer decode case plus four stable integration slices:
c1r2 input RMSNorm replay, real-model current-K/V cache writeback,
full-layer Q/K/V prefix, and production bf16 attention -> O. The old `current`
case name, patched descriptor runner, and 608-patch weight-stream oracle were
retired because they implied a second fused-layer backend. The useful pieces of
that path are now explicit shared code: host Q4NX weight BO layout, row1
S2MM4/5 ingress, and AIE Q4NX chunk accumulation.
The frontier now uses main16 Q4NX kernels for Q/K/V/O/up/gate/down over the
same row1 S2MM4/5 -> main16 DMA1 weight stream. Remaining numerical work is to
finish production Qwen3 calibration for c1r2 RMSNorm/replay, c1r3 Q/K norm +
RoPE, bf16 attention, and c6r2 bounded table sigmoid.
In IRON MLIR, S2MM and MM2S channel ids are directional
namespaces: the current main16 record output is emitted as `MM2S1`, which is
role-equivalent to MyLM's third main16 record stream even though the exact
printed id is not `DMA2`. Remaining work is to keep the production Qwen3 cache
in this block-major scan layout at runtime and finish numeric calibration for
attention, c1r2 RMSNorm/replay, and SwiGLU.
The full-layer generator now has an explicit compact phase trace shared by main
record DMA, row1 compact DMA, and bridge compact DMA. Each trace item carries a
unique MLIR label, logical phase, record slot, packet id, and payload slice. The
validated trace is now the body-level `q,k,v,o,upgate,down` schedule. `upgate`
is a 48-record long body, not two single-record phases: main16 emits
interleaved up/gate records through the same 17-dword record ping/pong stream,
row1 uses a 65-dword ping/pong record packer, and c1r1 uses a 257-dword
ping/pong global record packer. The `qwen3-8b-qkv-compact-output` NPU slice
proves the Q/K/V prefix can drain all 12 global compact records without a debug
drain or phase-sized bridge. c1r3 should receive the 12 Q/K/V payload records as
one 3072-dword local buffer and c6r2 should consume adjacent up/gate payload
pairs. This avoids both the invalid 53-phase materialized trace and the
standalone `up -> gate -> up` ring that cannot transition to down.
The c1r2 bridge also captures the lock rule needed for that port: a memtile DMA
BD block can release only one lock. Reusable row/group fan-in therefore uses a
stage-level counting empty lock, acquired once by each input row/group and
released once by the compact output BD with count 4. Per-row empty releases
compiled into multiple release ops and were rejected by MLIR-AIE; not releasing
the empty tokens caused the second up/gate pair to timeout on NPU.
For body-level scatter/gather, the MLIR-AIE dimension syntax must use named
struct parameters, for example `[<size = 48, stride = 257>, <size = 256,
stride = 1>]`; the shorter `[<48, 257>, ...]` form is shown in old dialect
comments but is rejected by the current parser.
Current K/V intentionally uses packet8/9 rather than packet14/15: full-layer
execution already reserves packet14/15 for FFN up/gate and down compact traffic.
Do not wire this by materializing both 8192-dword KV sides inside the c1r3
postprocess tile; that would exceed the intended compute-tile local-memory
budget. The full-layer version needs a streaming KV path through shim/row1/edge
tiles, matching the qwen3-dataflow design.
