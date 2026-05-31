# Main16 Q4NX MyLM Comparison

This note records the current performance conclusion for the main16 projection
kernel. It is based on local reverse notes, extracted MyLM disassembly, and the
local MyLM decode benchmark probe.

## Sources

- MyLM reverse note:
  `/var/home/taowen/projects/MyLM/tools/re/fused-layer-engine/current-understanding.md`
- MyLM Q4NX layout note:
  `/var/home/taowen/projects/MyLM/tools/re/Q4NX_LAYOUT.md`
- MyLM disassembly tool:
  `/var/home/taowen/projects/MyLM/tools/re/aie_program_disassemble.py`
- MyLM phase summary tool:
  `/var/home/taowen/projects/MyLM/tools/re/aie_main16_phase_control_summary.py`
- Captured c2r2 disassembly:
  `/tmp/mylm_solidify_L31/disasm/c2r2.s`
- Captured program layout:
  `/tmp/mylm_solidify_L31/programs/program_segments.tsv`,
  `/tmp/mylm_solidify_L31/programs/program_images.tsv`
- Generated IRON comparison report:
  `qwen3-layer/main16_q4nx_mylm_secret.md`
- IRON active object:
  `qwen3-layer/main_projection_q4nx_fast.o`
- Local MyLM 8B probe:
  `tools/re/bin/qwen3_decode_bench --model /var/home/taowen/flm/models/Qwen3-8B-NPU2`

If `/tmp/mylm_solidify_L31` is missing, regenerate it from the MyLM reverse
tooling before running the comparison script.

The visible MyLM tree does not contain the Qwen3 NPU kernel source. The public
headers declare `qwen3_npu` and `qwen3_npu_sequence`, while the implementations
come from `src/lib/libqwen3_npu.so` and the AIE programs from
`src/xclbins/Qwen3-8B-NPU2/layer.xclbin`. That makes MyLM useful as a concrete
ABI/dataflow/performance target, but not as a source-level kernel to copy.

## Fixed ABI Match

MyLM and IRON are now aligned on the outer main16 ABI:

| Path | MyLM ABI | IRON target |
| --- | --- | --- |
| activation | main16 DMA0, 128 dwords | c1r1 -> main16 DMA0 |
| weight | main16 DMA1, 1280 dwords | row1 S2MM4/5 -> row1 MM2S -> main16 DMA1 |
| record | main16 output, 17 dwords | main16 -> row1 compact gather |

The remaining gap is not the chunk format or the row1 channel split. The gap is
how the main16 core consumes that ABI.

## Current Toolchain Boundary

The latest active full-decode build no longer puts the main16 phase control
inside a large MLIR `aie.core` body. MLIR-AIE generates topology, local buffers,
BD rings, locks, stream switch routing, runtime sequence, transaction, PDI, and
xclbin packaging. The main16 core body itself is now a small wrapper that calls
the linked C++ AIE core object:

```text
func.call @q4nx_main16_full_scheduler(...)
```

A fresh full-decode compile of
`qwen3-layer/build/qwen3-8b-decode-layer-capacity-token127/design.mlir` produced:

| ELF | `.text` | core-entry refs | `jl` | `acq` | `rel` | `lc/ls/le` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| current IRON `main_core_2_2.elf` | 6080 | 1 `q4nx_main16_full_scheduler` | 14 | 18 | 18 | 6 |
| MyLM `c2r2.elf` | 14868 | raw dispatcher + 5 Q4 calls | 14 | 16 | 15 | 18 |

So the current bottleneck is no longer the old MLIR-expanded per-chunk phase
control. That failure mode existed and was useful to diagnose, but the active
single-version path has already moved phase control into a linked C++ AIE core
object.
The remaining performance gap is the Q4NX microkernel shape inside
`main_projection_q4nx_fast.o`.

## Supply Attribution

The current evidence does not point to row1 weight fanout as the main reason
main16 is slow. A token31 stage-budget run measured:

| Boundary | NPU time |
| --- | ---: |
| row1 weight stream, compute disabled | 8.717 ms |
| main16 Q4NX compute, DMA0/DMA1 disabled | 13.649 ms |
| c1r2 input RMSNorm replay | 1.876 ms |
| QKV cache-write bridge | 7.963 ms |
| attention -> O slice | 12.346 ms |
| full decode layer | 24.802 ms |

The row1 slice streams the full layer weight payload through row1 S2MM4/5 and
row1 MM2S fanout to all 16 main DMA1 sinks, then waits for the sinks to return
their 1472-chunk done counts. It is still faster than the isolated main16
compute loop. That means the next main16 speedup should not assume DMA1
starvation without new evidence.

The attention/O slice is large enough to be a second real bottleneck. It does
not prove that edge attention is starving main16, but it does mean full-layer
latency is not explained by row1/main16 alone. To prove a future starvation
claim at lock granularity, add a generated diagnostic slice with per-main-tile
progress counters for activation-acquired, weight-acquired, compute-done, and
record-released. Keep that as a slice mode so the production full-layer path
does not acquire debug drains or extra host-visible state.

## MyLM Raw Program Layout

The extracted Qwen3-8B `layer.xclbin` loads one uniform 14,868-byte main16
program image on all 16 projection tiles. For `c2r2` the program is segmented as:

| Offset | Bytes | Meaning |
| --- | ---: | --- |
| `0x0000` | 492 | entry/setup |
| `0x01f0` | 5760 | shared Q4NX microkernel |
| `0x1870` | 1544 | Q/K/V body |
| `0x1e80` | 1544 | O body |
| `0x2490` | 1544 | up/gate body |
| `0x2aa0` | 1560 | down body |
| `0x30c0` | 1544 | alternate body |
| `0x36d0` | 504 | dispatcher |
| `0x38d0` | 324 | tail/helper |

This is the first hard reason MyLM fits and runs fast: the Q4NX loop is loaded
once and called by the phase bodies. It is not seven separate C++ kernels, and
it is not a full unroll replicated inside every phase.

## MyLM Main16 Phase Body

`aie_main16_phase_control_summary.py` identifies the normal main16 phase
bodies:

| Body | Phase | Records/tile |
| --- | --- | ---: |
| `0x1870` | Q/K/V | 12 |
| `0x1e80` | O | 8 |
| `0x2490` | up/gate | 48 |
| `0x2aa0` | down | 8 |

The Q/K/V body at `0x1870` directly wraps the raw Q4NX microkernel:

- acquire weight-ready lock near `0x1a04`
- acquire activation-ready lock near `0x1a12`
- call Q4NX microkernel at `0x1d70 -> 0x1f0`
- release weight-empty lock near `0x1d90`
- release activation-empty lock near `0x1d9c`
- acquire/release the record/output side after compute

The microkernel setup at `0x1f0` programs a large zero-overhead loop:

```text
0x238: movxm ls, #0x260
0x23e: movxm le, #0x1850
```

The hot loop starts at `0x260` and is dominated by scheduled vector dequant,
broadcast, and MAC slots.

## Opcode Shape

Run:

```bash
.venv/bin/python qwen3-layer/tools/compare_main16_q4nx_disasm.py --top 20
```

Current summary:

| Range | Instruction lines | Op slots | Key ops |
| --- | ---: | ---: | --- |
| MyLM `0x1f0..0x1850` | 976 | 1560 | `vmac.f=264`, `vextbcst.16=256`, `vups.4x=64`, `vunpack=64` |
| MyLM `0x260..0x1850` | 963 | 1532 | same hot-loop shape |
| MyLM `0x1870..0x1e80` | 264 | 393 | phase lock/control plus body call |
| IRON active `q4nx_chunk_accum_fast` hot body | 716 | 1477 | `vmac.f=64`, `vextbcst.16=64`, `vextbcst.32=0`, `vunpack=64`, `vst=194`, `vlda=282` |
| IRON source-asm group probe | 88 | 88 | `vmac.f=33`, `vextbcst.16=32`, `vunpack=8`, `vups.4x=8`, `vst=0` |
| IRON `q4nx_chunk_accum_slice_i32_fast` | 2 | 2 | wrapper around the active single-version main16 Q4NX kernel |

This does not mean one MyLM call does the same dynamic work as one IRON helper
call; the loop counters differ. The important evidence is structural: MyLM has
a long raw scheduled loop with hundreds of vector MAC/broadcast slots in one
body. The current production IRON kernel remains the exact per-dim-rounding
Q4NX path, but it now uses signed native BF16 MAC with a 32-dim full unroll.
That was the first production-safe move toward the MyLM instruction family:
QKV compact, full-layer QKV prefix, attention-O, and full decode all passed.
The remaining gap is not `vextbcst.16` selection anymore; it is the excessive
`vlda/vst/vconv` traffic and lower MAC density versus the raw scheduled loop.
The assembly work is therefore kept as the next replacement target until the
full lane body is bit-compatible with the current Q4NX reference.

The newer semantic analyzer makes the MyLM loop count close from first
principles, not just opcode frequency:

```text
q4_chunk=32x256
output_lane_passes=2
activation_groups=8
activation_lanes_per_group=32
static vextbcst.16=256 = 8 * 32
dynamic vmac.f=528 = 512 q*scale*activation MACs + 16 offset*group_sum MACs
complete_lane_groups=8
```

So the fast mechanism is now concrete: phase bodies precompute eight
activation-group sums, then the Q4 loop keeps each 32-lane activation vector
live and broadcasts every lane from the vector register while consuming packed
int4 payload. The offset term is paid once per quant group via the group-sum
scratch, not once per activation dimension as in the simpler C++ dequant loop.

The corresponding Peano probe now reaches the same per-group shape:

```text
probe_native_q4_group_sum_correction_unroll32_signed:
  vmac.f=33
  vextbcst.16=32
  vst=1
```

That is exactly one quant group for one 16-row output lane:
`32 q*scale*activation MACs + 1 offset*group_sum MAC`. It is a useful
implementation target, but it is not production-equivalent yet because the
current reference rounds `bf16(q * scale + offset)` per dimension, while the
group-sum form moves the offset out of the per-dim rounding.

An exact-rounding full-unroll probe was added to test whether we could keep
the current `bf16(q * scale + offset)` contract while simply making the C++
intrinsic body bigger. Smaller exact fragments were also tested to find the
spill threshold:

```text
probe_native_q4_exact_rounding_unroll4_signed:
  vmac.f=4
  vextbcst.16=4
  vst=2

probe_native_q4_exact_rounding_unroll8_signed:
  vmac.f=8
  vextbcst.16=8
  vst=15

probe_native_q4_exact_rounding_unroll16_signed:
  vmac.f=16
  vextbcst.16=16
  vst=51

probe_native_q4_exact_rounding_group4_call_chain_signed:
  aggregate one-group body: vmac.f=32, vextbcst.16=32, vst=4
  wrapper: jl=8, hardware_loop=0

probe_native_q4_exact_rounding_group8_call_chain_signed:
  callee dim0: vmac.f=8, vextbcst.16=8, vst=20
  wrapper: jl=4, hardware_loop=0

probe_native_q4_exact_rounding_group16_call_chain_signed:
  callee dim0: vmac.f=16, vextbcst.16=16, vst=55
  wrapper: jl=2, hardware_loop=0

probe_native_q4_exact_rounding_unroll32_signed:
  vmac.f=32
  vextbcst.16=32
  vst=126

probe_native_q4_exact_rounding_chunk_lane_kernel_signed:
  vmac.f=256
  vextbcst.16=256
  vconv.bf16.fp32=640
  vst=1043
```

The 4-lane fragment is low-spill but too small to approach MyLM MAC density.
Splitting a full group into eight noinline 4-lane kernels keeps the local
kernels spill-free, but replaces one group with eight calls and loses hardware
loop formation in the wrapper. The obvious attempt to reduce the call count
does not work: 8-lane noinline already has `vst=20`, and 16-lane noinline has
`vst=55`. The full chunk-lane exact body is worse than the current production
wrapper's `vst=194`, so the exact path cannot be solved by more C++ template
unrolling or small-function tiling. The source-assembly replacement has to own
the exact rounding sequence and register allocation.

The next chunk-level probes clarify the migration boundary:

| Body | `vmac.f` | `vextbcst.16` | `vunpack` | `vups` | `vst` | Meaning |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| MyLM static hot loop | 264 | 256 | 64 | 64 | 0 | target raw schedule |
| Rejected Peano noinline lane kernel | 264 | 256 | 256 | 128 | 5 | shape is closer, semantics differ |
| Rejected Peano exact-rounding lane kernel | 256 | 256 | 256 | 128 | 1043 | semantics match, spill is unusable |
| Peano two-lane static duplicate | 528 | 256 | 512 | 256 | 824 | reject: massive spill |
| Peano two-lane loop body | 264 | 256 | 256 | 128 | 251 | reject: loop causes spill |

So the Peano/source-assembly route is not blocked, but the viable shape is
narrower than "write a C++ loop over lanes". The next candidate must preserve
the exact production rounding contract while reducing the unpack/dequant
schedule; the previous group-sum shortcut is ruled out for the active path.

The route is now being switched to source assembly for that inner body. The
first assembly probe is intentionally small, but it removes the most important
uncertainty: Peano's integrated assembler preserves the exact canonical middle
group inventory we need.

| Body | `vmac.f` | `vextbcst.16` | `lda.s16` | `vunpack` | `vups` | `vst` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MyLM canonical middle group | 33 | 32 | 1 | 8 | 8 | 0 |
| IRON source-assembly group probe | 33 | 32 | 1 | 8 | 8 | 0 |

The integration boundary is now proven in the production build path too. The
active `main_projection_q4nx_fast.o` is still the only main16 role object, but
`npu_build.py` now builds it from:

```text
main_projection_q4nx_fast.cc
main_projection_q4nx_asm.s
```

using `ld.lld -r`. The assembly companion now contains a generated
MyLM-style group-shape probe and a generated callable exact-lane candidate
(`q4nx_accum_lane_exact_body_shape`). `tools/generate_main16_q4nx_asm.py
--check` guards both source shapes before we wire the exact body into the C++
scheduler.
`main16-q4nx-compute-perf --build-only` passed with this multi-source object,
and the final linked core ELF confirmed the unreferenced probe was
garbage-collected:

```text
q4nx_accum_lane_asm_group_shape symbol = absent
q4nx_accum_lane_exact_body_shape symbol = absent
```

The same case also ran on NPU after the change. The most recent run after
restoring the production C++ lane and keeping assembly unreferenced was:

```text
main16-q4nx-compute-perf:
  npu_time = 13649.4 us
  done[0:8] = [1472, 1472, 1472, 1472, 1472, 1472, 1472, 1472]
  tile_chunks_per_sec = 1725492.4
```

`tools/check_main16_asm_integration.py --strict` now checks the same invariant
mechanically:

```text
role_has_asm_group = true
role_has_rejected_lane = false
core_has_asm_group = false
has_bad_symbol_jump = false
asm group = 33 vmac.f, 32 vextbcst.16, 8 vunpack, 8 vups.4x, 0 vst
production q4 = 64 vmac.f, 64 vextbcst.16, 0 vextbcst.32
```

So production does not need a second main16 variant, unused assembly probes do
not consume final program memory, and the exact numerical path now has a hard
native-MAC disassembly gate. The callable assembly ABI has also been verified
in the experiment tree: a C++ wrapper can call a source-assembly function with
normal pointer arguments, and the combined relocatable object preserves the
`R_AIE_1` relocation while the asm body contains `vldb=2`, `vmac.f=1`, and
`vst=1`. The direct NPU smoke now goes further: source asm reads a DMA-filled
BF16 vector, executes `vmac.f`, stores the accumulator through
`vst.conv.bf16.fp32`, releases the output lock, and matches the host reference.
That is the direction to use for production: C++ scheduler calls a complete asm
hot body. One attempted shortcut is explicitly ruled out: a source-assembly wrapper
that does `j #some_cpp_symbol` does not preserve a normal relocatable symbol in
this AIE2P assembler path. It links as `j #0` and the target can be garbage
collected. The checker rejects that pattern. Another shortcut is also ruled
out: the approximate group-sum lane body caused `qwen3-8b-qkv-compact-output`
to fail with `payload_max_abs=0.112304688` and then caused
`full-layer-qkv-prefix` K/V mismatch. The current exact per-dim-rounding
native-MAC kernel gives:

```text
qwen3-8b-qkv-compact-output:
  payload_exact_word_mismatches=1
  payload_max_abs=0.000000477
  PASS

full-layer-qkv-prefix:
  PASS: full-layer physical Q/K/V prefix writes current K/V correctly
```

Therefore the next migration step is not an asm-to-C++ bridge and not the
group-sum approximation; it is to write a callable numerical source-assembly
lane body that preserves the current per-dim Q4NX rounding contract while
keeping the existing C++ scheduler, locks, records, and generator ABI unchanged.

The historical full-core control shape was an even stronger root-cause signal
before the migration to a linked-core phase program:

| Program | Q4 calls | Total `jl` | `acq` | `rel` |
| --- | ---: | ---: | ---: | ---: |
| MyLM `c2r2.s` | 5 calls to `0x1f0` | 11 | 16 | 15 |
| IRON full main core after direct emit | 129 calls to Q4 helpers | 157 | 232 | 232 |

This was not just code size trivia: MyLM has one raw scheduled Q4NX body plus
normal phase bodies, while the older IRON build expanded phase/chunk lock
choreography around helper calls. The active build has since removed that
specific issue by moving phase control into linked C++ AIE core code. A
5x gain now has to come from changing the linked Q4NX microkernel itself, not
from another round of MLIR loop annotation.

## MLIR-AIE Codegen Probe

This section records the older MLIR-body phase-control failure mode. It is kept
because it explains why the current route moved phase control into a linked C++
AIE core kernel, but it is no longer the active main16 shape.

The older MLIR expressed the main16 phase bodies with `scf.for`; the large
static control shape was introduced after lowering. The toolchain probe showed:

| Probe | Main16 q4 refs | `acq` refs | Full-layer token31 |
| --- | ---: | ---: | --- |
| default aiecc O2/O1 | 97 | 233 | 29.159 ms, PASS |
| `aiecc -O 0` | 5 | 17 | 25.838 ms, FAIL: V cache/final hidden |
| `loop_annotation = #no_unroll` | 5 | 17 | 30.263 ms, PASS |
| `loop_annotation` plus `--aie-loop-aware` flags | 5 | 17 | compile-only probe |
| `opt -disable-loop-unrolling` replacement ELFs | 5 | 16 | 28.991 ms, PASS |

Two details matter. First, MLIR-AIE's AIE core lowering preserves loop metadata
only when the attribute is spelled `loop_annotation`; the upstream-style
`llvm.loop_annotation` is dropped before LLVM IR. Second, preserving the loop is
not enough: the surviving loop is a normal branch loop, not the MyLM raw
zero-overhead `lc/ls/le` loop with scheduled branch slots. That is why the
correct no-unroll build is slower than the active default and was not kept as
the production path.

The Peano backend does expose loop-aware scheduler switches, but they are not
an `aiecc` flag-plumbing fix. The current `aiecc` driver does not forward
`--aie-loop-aware --aie-loop-epilogue-analysis --aie-loop-sched-heuristics`
or hardware-loop forcing flags to the child `llc` command. Manual replay with
those flags still did not turn the phase bodies into MyLM-style zero-overhead
loops; the disassembly kept ordinary branch loops for Q/K/V/O/up/gate/down, and
`lc/ls/le` appeared only inside the C++ Q4NX helper. So the problem is not just
a missing command-line flag on the current MLIR shape.

The concrete aiecc child pipeline is now pinned down by
`tools/probe_aiecc_core_codegen.py`. In this installed MLIR-AIE wheel,
`bin/aiecc.py` is only a thin Python wrapper around the C++ `bin/aiecc`
binary; the child Peano command construction happens inside that C++ driver.
For the latest full-decode `main_core_2_2`, aiecc lowers the core to
`main_core_2_2.peanohack.ll`, then invokes:

```text
opt --passes=default<O1> -inline-threshold=10
llc -O2 --march=aie2p --function-sections
```

In that older shape, the pre-opt LLVM IR had 16 `q4nx` references. Stock `opt`
turned that into `opt_q4_refs=260`, and the resulting ELF had `.text=14416`,
`disasm_jl=157`, `disasm_acq=250`, `disasm_rel=250`. Earlier manual replay of
the same aiecc core compile shape with `opt -disable-loop-unrolling` proved
that the child `opt` unroll was avoidable outside the stock driver, but it
still did not create a MyLM-style raw scheduled phase body. A dry-run with
`aiecc --disable-loop-unrolling` still prints the original child `opt` command,
and `tools/audit_aiecc_driver.py` confirms that
`aiecc --opt-disable=loop-unroll` is also not forwarded to that child command;
the AIE loop-scheduler/hardware-loop flags are not forwarded to child `llc`
either. `-O0/-O3` do change the child optimization level, but they do not
create the MyLM raw scheduled phase body. So stock aiecc is not a flag-plumbing
path for this fix. The practical conclusion still holds: use MLIR-AIE for
topology and packaging, and keep performance-critical main16 code in a linked
C++ AIE core kernel or raw scheduled ELF.

The replay path is now executable across all 16 main tiles with
`tools/replay_main16_core_compile.py`. On a fresh full decode project it
produced 16 replacement main cores with `text_bytes=7040`, `opt_q4_refs=5`,
`disasm_acq=16`, and `disasm_rel=16`. Packaging those ELFs with
`tools/package_externalized_design.py` ran correctly on hardware:

```text
full-layer-qkv-prefix token31:
  replacement 124872.8 us PASS
  stock       142702.9 us PASS

qwen3-8b-decode-layer token31:
  replacement 28990.9 us PASS
  stock       29404.6 us PASS
```

So replaying the Peano compile with loop unrolling disabled is a real packaging
and correctness proof, but only a percent-level full-decode speedup. It removes
some generated control bloat; it does not create the MyLM raw phase body or its
scheduled zero-overhead loop.

## Core Program Packaging Probe

The replacement point is now understood more concretely. A fresh
`full-layer-qkv-prefix` transaction probe generated from source shows that
aiecc writes compiled core ELF paths back into the transaction MLIR:

```text
%core_2_2 = aie.core(%tile_2_2) {
  aie.end
} {elf_file = "/tmp/.../main_core_2_2.elf", link_files = [".../main_projection_q4nx_fast.o"]}
```

The transaction generation then lowers each core `.text` section into
`config_blockwrite_data` payloads. The tool
`tools/inspect_core_program_txn.py` reports, for the fresh QKV prefix probe:

```text
role=main_projection_q4nx_fast.o cores=16 text_bytes=9952
matching_config_blocks=16 tiles=c2r2,...,c5r5
mylm_reference: main16_images=16 main16_bytes=14868
```

So an in-place transaction patch cannot grow the payload; it can only replace
same-size program bytes. The practical MyLM-style path is to generate
replacement main16 ELFs and rerun xclbin/transaction packaging, while keeping
the MLIR-AIE generated topology, BDs, locks, stream switches, and runtime
sequence.

The transaction pass itself is now reproducible outside the full `aiecc`
compile. `--convert-aie-to-transaction` takes `elf-dir` as a pass option:

```bash
.venv/lib/python3.12/site-packages/mlir_aie/bin/aie-opt \
  --convert-aie-to-transaction=elf-dir=/tmp/iron_txn_probe_fresh/prj \
  -o /tmp/iron_txn_probe_fresh/repacked.txn.mlir \
  /tmp/iron_txn_probe_fresh/repack_source.mlir
```

The input must not already contain the old `config_blockwrite_data_*` globals
or the old `aie.runtime_sequence @configure()` block. Running the pass directly
on an already-converted transaction duplicates payloads. The helper
`tools/repack_core_program_txn.py` performs the required strip step and then
reruns the pass. On the fresh QKV-prefix probe it regenerates the same one-copy
payload set:

```text
config_blockwrite_globals=286
bytes=9952 count=16
role=main_projection_q4nx_fast.o cores=16 matching_config_blocks=16
```

This establishes a transaction-level inspection path:

1. compile the verified MLIR topology once with `aiecc`;
2. replace the selected `main_core_X_Y.elf` files in the aiecc project dir;
3. run `tools/repack_core_program_txn.py` to rebuild core-program blockwrites;
4. inspect the repacked transaction before attempting runtime packaging.

`aie-translate --aie-npu-to-binary` on the transaction MLIR is not equivalent
to the small runtime `design.bin` produced by `aiecc`; in the probe it emitted a
much larger binary. Treat it as a separate translation artifact, not as proof
that the final xclbin/instruction packaging step is solved.

The runnable packaging path is ELF-backed source MLIR plus
`aiecc --no-compile`. `tools/externalize_core_programs.py` copies every
`aie.core` `elf_file`/`link_files` attribute from a donor transaction back onto
the source MLIR and replaces the core bodies with `elf_file` empty-core
declarations:

```text
%main_2_2_core = aie.core(%main_2_2) {
  aie.end
} {elf_file = "/tmp/.../main_core_2_2.elf", link_files = [".../main_projection_q4nx_fast.o"]}
```

Then `aiecc --no-compile` can regenerate runtime instructions, transaction,
PDI, and xclbin without recompiling or overwriting the ELFs. The verified probe
preserved the expected payload shape:

```text
cores=18
config_blockwrite_globals=286
bytes=9952 count=16
role=main_projection_q4nx_fast.o cores=16 matching_config_blocks=16
design.bin=2372 bytes
```

`tools/package_externalized_design.py` now wraps this as the current replacement
point for a raw/scheduled main16 ELF: it copies the donor project directory,
optionally replaces all 16 `main_core_X_Y.elf` files from a replacement
directory, converts every source core to an ELF-backed empty core, runs
`aiecc --no-compile`, then
checks that the main16 ELF text sizes were not overwritten and that the
transaction contains 16 matching main16 payloads. Feeding transaction MLIR back
to `aiecc` is not a continuation path; it re-enters the routing/resource
pipeline and failed the probe with fixed switchbox connection conflicts.

Partial `elf_file` conversion is also unsafe. A probe that converted only
main16 and let aiecc compile the remaining C++ roles overwrote the donor main16
ELF with an empty-core `.text=160` program. So raw main16 experiments must use
all-core ELF-backed source MLIR plus `--no-compile`, or an explicit copied
project dir with post-build ELF size checks.

There is one more important packaging detail: the replacement ELF must be an
executable with a loadable text segment. The MyLM reverse helper
`aie_raw_to_elf.py` creates an ET_REL wrapper for disassembly, and
`llvm-size` sees `.text=14868`, but `aiecc --no-compile` does not lower that
relocatable section into core-program blockwrites. The IRON helper
`tools/wrap_raw_aie_program.py` creates ET_EXEC + `PT_LOAD` instead. With that
wrapper, the package tool successfully produced:

```text
bytes=14868 count=16
role=main_projection_q4nx_fast.o cores=16 text_bytes=14868 matching_config_blocks=16
design.bin=2372 bytes
design.xclbin=290240 bytes
```

This proves the toolchain can carry a MyLM-sized raw main16 image through our
xclbin packaging path. It does not mean the extracted MyLM raw main16 program is
directly runnable in the IRON topology. MyLM's raw program assumes fixed
main16 ownership:

```text
BD0/1  DMA0 activation ping/pong, locks L0/L1
BD2/3  DMA1 weight ping/pong,     locks L2/L3
BD4/5  compact record ping/pong,  locks L4/L5
activation bases 0x78000/0x7c000
weight bases     0x72800/0x74000
record bases     0x73c1c/0x7541c
```

The active IRON full-layer generator now matches MyLM for the checked row0
main16 outer ABI. Activation uses BD0/1, bases `0x8000/0xc000`, and L0->L1.
Weight uses BD2/3, bases `0x2800/0x4000`, and L2->L3. Record output uses
17-dword ping/pong buffers at `0x3c1c/0x541c`, BD4/5, and L5->L4.

`tools/check_main16_raw_abi.py` now makes this explicit against the generated
full-layer MLIR. The current diagnostic is:

```text
raw_main16_abi_ready=true
activation:
  mylm bd=(0,1) len=(128,128) base=(0x8000,0xc000) locks=L0->L1
  iron bd=(0,1) len=(128,128) base=(0x8000,0xc000) locks=L0->L1
weight:
  mylm bd=(2,3) len=(1280,1280) base=(0x2800,0x4000) locks=L2->L3
  iron bd=(2,3) len=(1280,1280) base=(0x2800,0x4000) locks=L2->L3
record:
  mylm bd=(4,5) len=(17,17) base=(0x3c1c,0x541c) locks=L5->L4
  iron bd=(4,5) len=(17,17) base=(0x3c1c,0x541c) locks=L5->L4
```

The active migration is now source-side and compact-tree complete for the
checked topology: main16 emits 17-dword records, row1 packs
`17+16+16+16 -> 65` dwords, and c1r1 packs `65+64+64+64 -> 257` dwords. This
removes the main16 DMA/lock/raw-ring blocker. The remaining raw-main16 issue is
not the basic BD ring. It is the hidden contract around MyLM's whole-core
dispatcher.

## Direct MyLM Main16 ELF Compatibility

The extracted MyLM main16 program is not a C ABI function that can be called
from the current IRON linked AIE core kernel. It is a whole-core AIE program
that assumes a larger local contract than the three visible DMA rings.

### Record Headers

MyLM phase bodies emit small phase headers:

```text
Q/K/V  header 0x1, 12 records/tile
O      header 0x4,  8 records/tile
upgate header 0x8, 48 records/tile
down   header 0x4,  8 records/tile
```

IRON currently emits richer value headers:

```text
(phase << 24) | (block << 20) | (group << 16) | (row << 8) | packet_id
packet_id: Q/K/V/O/FFN/down = 10/11/12/13/14/15
```

This is a value-layout mismatch, but it is not literally how the current IRON
stream switch routes the records. In the active generator, the stream routing
from bridge to c1r3/c1r2/c6r2 is driven by the bridge output BD packet
attributes and `aie.packet_flow(...)`, while row1/c1r1 compact gather is
record-order/circuit based. The manual record header is kept as the first data
dword of the compact packet and is used by debug/reference contracts, and by
any kernel that chooses to inspect compact[0]. Most current production
consumers either receive header-stripped payloads or ignore the header:

- c1r3 receives Q/K/V payload bodies from bridge output offset `1`.
- c6r2 receives up/gate payload halves from bridge output offset `1`.
- c1r2 O/down code uses `compact + 1` payload lanes.

So direct MyLM headers would not automatically break bridge packet routing, but
they would break the current IRON header-value contract and any validation or
kernel path that expects phase/block/group/row fields. A runnable MyLM raw-main16
experiment must either adapt those consumers/reference checks to MyLM's small
phase headers or patch the raw program's record-header stores.

### Tile-Local Scratch And Control

MyLM initializes and uses local memory immediately after the record ping output:

```text
record ping BD4: base 0x3c1c, len 17 dwords, ends at 0x3c60
0x3c60: CDO mask-write init; record-publish byte/parity state
0x3c64: CDO write 0; Q4/control register state loaded by the microkernel
0x3c68: CDO write 8; helper state used by tail/helper 0x38d0
0x3c80: 128-byte zero scratch/state block
0x3d00: 32-byte function/control table, first word 0x3820
```

The disassembly uses these addresses directly:

```text
0x1f0:  load 0x73c64 / 0x73c62 before the Q4 hot loop
0x1db0/0x23c0/...: use 0x73c60, 0x73c62 before record publish
0x38d0: use 0x73c68 and 0x73d00 in the tail/helper path
```

This memory is not part of the visible BD4 payload. It is adjacent tile-local
program state. IRON's manifest currently proves the 17-dword record buffer
itself, but a whole-core MyLM ELF also requires reserving and initializing this
`0x3c60..0x3d20` control/scratch interval exactly.

### Activation Control Word

MyLM's c1r2 full-vector replay path is 2049 dwords:

```text
c1r2.bd3 base 0x7101c len 2049
1 control dword + 2048-dword hidden/full-vector payload
L3 release +12 for Q/K/V, +48 for up/gate
```

The c1r2 disassembly writes a control/header word at `[p0, #28]`, i.e. at the
start of the 2049-dword output window. The main16 dispatcher then selects its
normal body chain from a control value loaded through the activation-side
setup. This is why MyLM can distinguish the normal
`QKV -> O -> upgate -> down` chain from the alternate `0x30c0` path without a
host-side per-phase call sequence.

The current IRON c1r2 station has the same semantic replay counts, but the
active MLIR sends a 2048-dword `full_replay` payload over packet0 and relies on
the IRON linked-core kernel ABI, not on MyLM's whole-core activation control
word. Therefore the direct MyLM main16 ELF is not compatible with the current
c1r2 replay ABI
until IRON either:

- emits the MyLM-style 2049-dword control+payload replay and matches main16's
  activation chunk alignment, or
- patches/replaces the MyLM dispatcher so it does not depend on that control
  word.

### Tile Coordinates

This is no longer considered a blocker. The 16 MyLM main tiles load the same
program image because the main16 program does not encode logical
`group/row/block` in the record header. The physical tile, row1 fanout order,
weight chunk order, and row1/c1r1 compact gather position determine where each
32-row payload lands in the global 512-output block. IRON's richer
`group/row/block` header is useful for debugging, but MyLM does not need it for
the production compact tree.

### Weight Schedule

The chunk ABI matches, but the whole-core raw program also assumes MyLM's
weight order:

```text
Q 64, K 16, V 16, O 64, up 192, gate 192, down 64 patches
```

Within each phase, body replay count, ping/pong parity, row1 two-patch ingress,
and row fanout order must match the raw dispatcher. A mismatch here will not
look like a broken DMA ring; it will show up as wrong values or a phase waiting
on a lock after consuming the wrong activation/weight pair.

The practical conclusion is:

1. Keep MLIR-AIE for topology, BD/lock, stream switch, runtime sequence, and
   packaging.
2. Do not drop the unmodified MyLM whole-core ELF into IRON and expect it to
   behave like `q4nx_main16_full_scheduler`.
3. The shortest runnable path is to preserve IRON's linked-core kernel/bridge
   ABI and replace only the Q4NX microkernel body. The more aggressive path is
   to adopt MyLM's whole-core AIE program, but then the c1r2 2049-dword replay
   control, `0x3c60..0x3d20` local control interval, small phase headers, and
   exact weight schedule must move together.

## Performance Evidence

Current IRON stage-budget measurements on real NPU:

| Slice | Time |
| --- | ---: |
| row1 weight stream | 8.357 ms, 13.438 GiB/s |
| main16 Q4NX compute-only, active direct-emit ABI | 14.249 ms |
| attention-O bf16 slice, active main16 | 12.194 ms |
| full single layer, active main16 after record-granular compact and source-side down replay | 25.067 ms |

The local MyLM full-model benchmark probe measured about `4.93 tok/s` for
Qwen3-8B at 1k/8k context on this machine, or roughly `203 ms/token`. Spread
over 36 layers, that is about `5.6 ms/layer` including non-layer overhead.
The published MyLM table reports `11.9 tok/s` at 1k and `10.4 tok/s` at 8k,
which corresponds to an even tighter layer budget. Either way, the current IRON
single-layer `25.067 ms` result is still about 4.5x slower than the local MyLM
probe and much farther from the published number.

The row1 slice proves the full-layer weight ingress/fanout path is not the
largest gap by itself. Consolidating the main16 role into one active object made
full-layer slices fit again and improved the C++ Q4NX path materially, but the
main16 compute-only slice is still too slow for a MyLM-class layer budget before
adding attention, O/down replay, or final full-layer traffic.

The profitable C++ tuning space is now narrow because the numerical contract
requires the exact bf16 dequant order `q * scale -> bf16 -> + offset -> bf16 ->
MAC`. Reassociating this into an affine MAC is much faster in isolation, but it
changes Q/K/V prefix values enough to fail the current cache-write oracle.

The deleted C++ unroll probes are useful historical evidence, but not worth
maintaining as active code. A 32-dim unroll improved narrow slices but overflowed
full-layer program memory; a 22-dim full-layer probe fit under the roughly 16 KiB
AIE program-memory limit and remained numerically correct, but regressed token31
full decode from 29.236 ms to 30.092-30.963 ms. The repository now keeps one
main16 C++ implementation: the fastest verified full-decode object,
`main_projection_q4nx_fast.o`.

One generator-level phase-body simplification is active: records are emitted
directly from `accum`, and the old `q4nx_output` scratch plus block-accumulator
variant were removed. After record-granular compact and c6r1 source-side down
replay, token31 full decode runs at `25.067 ms`; this is useful raw-core
preparation, but it confirms the
larger point: the remaining gap is the scheduled main16 core body, not a spare
record-copy helper.
A separate attempt to remove the defensive partial-row/bounds branches from the
C++ helper itself was not kept: it made a full-layer main16 ELF grow to
`.text = 0x4760`, which overflows AIE program memory.

The QKV prefix runner is a correctness gate with downstream tiles intentionally
blocked after the prefix, so its end-to-end NPU time is not a clean main16
throughput number. The attention-O result is more useful for performance: it
exercises QKV, edge attention, packet2 handoff, and O projection in one
full-layer-derived path.

## Conclusion

The active IRON main16 role is now single-version: compute, fill, flush, and
record emit live in `main_projection_q4nx_fast.cc`. The old mixed C++ projection
object was removed from the active generator path because linking both old emit
and fast compute consumed too much main16 program memory in full-layer slices.
The dataflow principle is the same as MyLM; the difference is that MyLM lowers
the same ABI into raw scheduled core bodies, while IRON still surrounds an AIE
C++ role kernel with generated MLIR phase control.

The next main16 experiment should be a raw or generated fixed-schedule Q4NX
microkernel with the existing numerical contract and ABI:

```text
DMA0 activation chunk: 128 dwords
DMA1 Q4NX weight chunk: 1280 dwords
output compact record: 17 dwords
```

`tools/check_main16_program_shape.py` is now the structural gate for that
experiment. It treats the extracted MyLM `c2r2` raw program as the positive
example (`jl=14`, `acq=16`, `rel=15`, `mylm_q4_calls_to_0x1f0=5`) and rejects
the current generated IRON full core (`jl=157`, `acq=250`, `rel=250`, plus C++
Q4 helper symbol references). A candidate replacement main16 ELF must pass this
shape check before it is worth packaging and running full-layer numeric gates.

The latest toolchain probe narrows the implementation route. Peano can generate
AIE hardware loops when phase control lives in a linked C++ AIE core object:

```bash
.venv/bin/python qwen3-layer/tools/probe_external_lock_dispatcher.py --force
```

The probe compiles a linked C++ AIE dispatcher using AIE2P
`acquire_greater_equal()` and `release()` builtins and produces one `acq`, one
`rel`, and `lc/ls/le` loop setup. AIE control-flow instructions have delay
slots, so `acq/rel` lines printed after `j/jl/ret` in objdump are not by
themselves proof that locks escaped the loop. The QKV-prefix linked-core
timeout was then traced through compact ABI issues, and the current source side
has moved to the MyLM-style record granularity: main16 emits 17-dword records,
row1 packs 17+16+16+16 into 65-dword column records, and c1r1 packs
65+64+64+64 into one 257-dword global record before packet routing by phase.

So the current blocker is not "LLVM-AIE cannot emit a hardware loop", and it is
not the old phase-sized compact bridge. The active main16 phase control already
lives in a linked C++ AIE core entry point. The remaining gap is that the C++
Q4NX microkernel generated from `aie::vector` plus native MAC still is not
MyLM's raw scheduled microkernel: it has `vextbcst.16`, but far too much
load/store and conversion traffic. Keep MLIR-AIE for topology and packaging,
and focus the next performance work on replacing the linked Q4NX microkernel
with a source-assembly lane body that preserves the exact production Q4NX
rounding contract.

The source-assembly path is now numerically proven on a real-NPU Q4NX lane
boundary, not only as a shape probe. `run_asm_npu_smoke.py` calls a
hand-written full 8-group exact Q4NX lane body from an AIE core, reads packed
uint4/scale/offset/activation from DMA-filled local memory, uses
`vextbcst.16 + vmac.f`, stores BF16 accumulator output, and matches the BF16
reference for all 8 groups / 256 input dims in one 16-row lane. This also
resolved several source-assembly traps: a full static lane unroll overflows
program memory, `add #0x40` encodes as `-64`, a large source-level hardware
loop did not execute as intended, and loop-carried `dj0` offsets left
offset/activation stuck on group0. The passing version uses a counted branch
loop and pointer registers for scale/offset/activation. This does not complete
the production lane. It removes the ABI/runtime doubt for source assembly and
leaves the real work as wiring this scheduled exact body into the production
main16 scheduler and weight/activation ABI.

## AIEVec/XLLVM Probe

The AIEVec route is worth keeping, but the current installed op surface is not
yet a complete MyLM Q4NX hot-loop vocabulary. Run:

```bash
qwen3-layer/tools/probe_aievec_q4nx_codegen.py
```

The probe uses the local IREE/MLIR-AIE `iree-opt --convert-aievec-to-llvm`
lowering. It proves two useful facts:

- `aievec.matmul` on `npu4` lowers to
  `xllvm.intr.aie2p.I512.I512.ACC2048.mac.conf`.
- `aievec.ups`/`aievec.srs` lower to the BF16 accumulator conversion
  intrinsics used by the AIE backend.

It also proves the current limitation:

```text
aievec_ops=cast,ext,matmul,shift,shuffle,srs,ups
has_xllvm_unpack=False
has_xllvm_extbcst=False
```

So AIEVec/XLLVM is a better-maintained route than hand-writing raw binary for
MAC/UPS/SRS experiments, but it does not directly expose the `vunpack` and
`vextbcst.16` shape that dominates the MyLM Q4NX microkernel. There are two
practical next steps:

1. Use AIEVec to build a small fixed-schedule microkernel probe for the parts
   it can express, then compare disassembly against MyLM.
2. For exact Q4NX parity, either use Peano compatibility intrinsics from the
   AIE2P headers for unpack/broadcast, or extend AIEVec/XLLVM with the missing
   ops.

This makes AIEVec a useful intermediate route, not a direct drop-in solution
for the current `q4nx_chunk_accum_fast` replacement.

## MLIR-AIE Source Audit

The installed toolchain is MLIR-AIE commit
`e4f35d643c7e9f077faf85577dcfb340174d671e`. A source checkout of the same
commit confirms the black-box probe results:

- `tools/aiecc/aiecc.cpp` exposes `-O` as the only relevant core optimization
  knob. There is no `cl::opt` for forwarding `-disable-loop-unrolling`,
  `--aie-loop-aware`, hardware-loop forcing, or arbitrary child `opt`/`llc`
  options.
- `runLLVMLoweringPipeline()` lowers each `aie.core` through
  `AIELocalizeLocks`, `AIENormalizeAddressSpaces`, `AIECoreToStandard`,
  `AIEXToStandard`, `ConvertAIEVecToLLVM`, and generic LLVM conversion. This
  is a functional core extraction path, not a MyLM-style phase scheduler.
- `compileCore()` then applies `peanohack`, invokes Peano
  `opt --passes=default<O{min(O,1)}> -inline-threshold=10`, and invokes
  `llc -O{O} --march=aie2p --function-sections`. At `-O3` it only inserts
  `-disable-loop-idiom-memset`; it still does not expose loop-unroll or AIE
  hardware-loop scheduler flags.
- `AIE_CoreOp` explicitly supports external executable core programs through
  `elf_file`; the verifier requires an `elf_file` core body to contain only
  `aie.end`. This is the correct hook for replacement main16 ELFs.
- The transaction path is also explicit in source:
  `convert-aie-to-transaction{elf-dir=... device-name=...}` reads the
  `elf_file` payloads and generates the core-program blockwrites before NPU
  lowering destroys the AIE device state.

So "MLIR-AIE cannot generate the code we want" should be read narrowly. It can
generate the topology we need: pinned buffers, DMA BD rings, locks, stream
routes, runtime sequence, transaction, and xclbin. The part it does not
generate for us is the MyLM main16 raw Q4NX instruction schedule. That schedule
must come from one of three routes:

1. patch or custom-build `aiecc` so the child `opt/llc` pipeline is under our
   control;
2. keep the current linked main16 core kernel, but rewrite the Q4NX microkernel
   as lower-level scheduled Peano C++/intrinsics and link/package that ELF
   through `elf_file`;
3. generate a raw main16 ET_EXEC ELF directly, then package it with the existing
   all-core ELF-backed `aiecc --no-compile` path.

Route 1 is toolchain work and still would not automatically create the MyLM
micro-schedule. Route 2 is the fastest engineering path because it preserves
the current verified topology and linked-core ABI. Route 3 is the final escape
hatch if C++ cannot match the raw instruction schedule.

## Resolved Q4NX Kernel Semantics

The dedicated MyLM analyzer now checks the raw c2r2 Q4NX body:

```bash
python3 experiments/aie_intrinsics_api_probe/analyze_mylm_main16_kernel.py
```

The key result is:

```text
lc=2
static vmac.f=264
dynamic vmac.f=528
matches_dynamic_vmac=True
```

This explains the kernel at Q4NX algorithm level, not just at opcode level.
One 5120-byte chunk is 32 output rows by 256 activation columns. With 16-row
BF16 vector MACs, the required work is:

```text
main term:        2 output lanes * 8 groups * 32 dims = 512 vector MACs
zero/offset term: 2 output lanes * 8 groups           =  16 vector MACs
total:                                                   528 vector MACs
```

The static hot loop is half of that chunk. It runs twice, once for rows 0..15
and once for rows 16..31. The disassembly supports this directly:

- `p1` loads eight 32-lane activation vectors as `x11`, then rewinds by
  `0x200` bytes before the second output-lane pass.
- `p3` loads eight `s16` activation group sums, then rewinds by `0x10` bytes;
  those group sums feed the zero/offset correction term for both output lanes.
- `p0` does not rewind; it keeps advancing through the two packed int4 payload
  halves.
- `p5` addresses scale lanes from `chunk + 0x000 + group * 2`.
- `p4` addresses zero/offset lanes from `chunk + 0x200 + group * 2`.

This is the concrete performance target, but not yet the active numerical
contract. IRON's current reference still rounds `bf16(q * scale + offset)` per
dimension. A replacement kernel must either preserve that rounding exactly or
change the layer reference and prove end-to-end decode quality; the failed
group-sum production attempt shows that the algebraic shortcut is not a free
drop-in replacement.
