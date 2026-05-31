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
func.call @q4nx_main16_layer_scheduler(..., phase_limit)
```

A fresh full-decode compile of
`qwen3-layer/build/qwen3-8b-decode-layer-capacity-token127/design.mlir` produced:

| ELF | `.text` | core-entry refs | `jl` | `acq` | `rel` | `lc/ls/le` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| current IRON `main_core_2_2.elf` | linked core object | 1 `q4nx_main16_layer_scheduler(..., phase_limit)` + active `q4nx_chunk_accum_asm_zol` | active checker gate | active checker gate | active checker gate | active checker gate |
| MyLM `c2r2.elf` | 14868 | raw dispatcher + 5 Q4 calls | 14 | 16 | 15 | 18 |

So the current bottleneck is no longer the old MLIR-expanded per-chunk phase
control. That failure mode existed and was useful to diagnose, but the active
single-version path has already moved phase control into a linked C++/source-asm
AIE core object. The remaining performance gap is the Q4NX microkernel shape
inside `main_projection_q4nx_fast.o`.

## Supply Attribution

The current evidence does not point to row1 weight fanout as the main reason
main16 is slow. A token31 stage-budget run measured:

| Boundary | NPU time |
| --- | ---: |
| row1 weight stream, compute disabled | 8.717 ms |
| main16 Q4NX compute, DMA0/DMA1 disabled | 18.158 ms |
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
| MyLM `0x1f0..0x1850` | 976 | 1560 | `vmac.f=264`, `vextbcst.16=256`, `vups.4x=64`, `vunpack=64`, `vconv.bf16.fp32=136`, `vst=0` |
| MyLM `0x260..0x1850` | 963 | 1532 | same hot-loop shape |
| MyLM `0x1870..0x1e80` | 264 | 393 | phase lock/control plus body call |
| IRON active `q4nx_chunk_accum_asm_zol` hot body | checker gate | checker gate | `vmac.f=64`, `vextbcst.16=64`, `vextbcst.32=0`, `vunpack=64`, `vups.2x=32`, `vlda=2`, `vldb=66`, `vst=2`, `vconv.bf16.fp32=160` |
| Historical IRON source-asm group probe | 88 | 88 | `vmac.f=33`, `vextbcst.16=32`, `vunpack=8`, `vups.4x=8`, `vst=0`; kept in experiments, not active role object |
| IRON `q4nx_chunk_accum_slice_i32_fast` | 2 | 2 | callable hot-body microbench wrapper; not used by the generated layer scheduler |

This does not mean one MyLM call does the same dynamic work as one IRON helper
call; the loop counters differ. The important evidence is structural: MyLM has
a long raw scheduled loop with hundreds of vector MAC/broadcast slots in one
body. The current production IRON kernel remains the exact per-dim-rounding Q4NX
path, but the active hot body has moved from C++ AIE API code into generated
source assembly. The generated `q4nx_main16_layer_scheduler` now owns the
dispatcher, phase bodies, shared Q4 body, and compact record emit path. Its
shared Q4 body inlines the instruction template instead of calling
`q4nx_chunk_accum_asm_zol`; `tools/check_main16_asm_integration.py --strict`
rejects the old C++ dataflow scheduler and the old QKV/QKVO/full scheduler
variants. The callable `main16-q4nx-compute-perf` case is now only a hot-body
microbench. Production scheduler evidence must come from `full-layer-qkv-prefix`
and `qwen3-8b-decode-layer`.

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

The group boundaries are not eight independent copies of a 33-MAC body. The
corrected analyzer shows:

```text
group0:   vmac.f=28, vconv.bf16.fp32=16, vunpack=12, lda.s16=2
group1-6: vmac.f=33, vconv.bf16.fp32=17, vunpack=8,  lda.s16=1
group7:   vmac.f=38, vconv.bf16.fp32=18, vunpack=4,  lda.s16=0
```

This is the core performance lesson from the reverse engineering work: MyLM is
a cross-group software pipeline with fill/drain, not a loop that simply repeats
one independent "32 main MAC + 1 correction MAC" group. Any source-assembly
replacement has to first reproduce the register/dataflow schedule, including
the `vups.4x` and `vconv` chains, before opcode counts become meaningful.
Exp104 turns this into a generated schedule artifact and records the register
families that cross group boundaries (`acc1`, `acc4`, `vec0`, `vec8`, `vec9`,
`vec10`, and pointer/scalar state).

Exp105 then measures the local dependency gaps on real NPU:

```text
vextbcst.16 -> vmac.f         first passing gap = 1
vbcst.16 -> vmac.f            first passing gap = 1
vldb -> vextbcst.16 -> vmac.f first passing gap = 6
vmac.f -> vst                 first passing gap = 5
```

So the main assembly problem is not simply "insert enough nops after
`vextbcst.16`". The real schedule has to hide vector-load and storeback latency
with useful dequant and accumulator work from other groups.

Exp106 narrows the generator boundary further. The full loop is not eight
unique groups and not eight copies of one group. It is:

```text
fill(group0)
steady_template(group1) * 5
pre_drain(group6)
drain(group7)
```

Groups 1, 2, 3, 4, and 5 have identical instruction text and the same
189-slot hash. Group6 differs from the steady template only at the tail: it
replaces one nop with `mov r21, p3` and inserts `add.nc p3, r21, #-0x10`.
That is the `p3` group-sum scratch rewind before the drain window. The steady
template lane order is also non-linear:

```text
0, 1, 2, 3, 8, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15,
16, 21, 17, 18, 24, 19, 20, 22, 23, 25, 26, 27, 28, 29, 30, 31
```

This gives the next implementation target: a small assembly generator with
four explicit templates, not a per-group macro loop.

Exp107 closes the loop on that claim: it regenerates the complete
`0x260..0x1850` MyLM hot-loop instruction text from those four templates and
gets an exact match:

```text
original_slots=1532
generated_slots=1532
exact_match=True
vmac.f=264
vextbcst.16=256
vunpack=64
vups.4x=64
vconv.bf16.fp32=136
vst=0
```

That is the first point where the reverse-engineered Q4NX body is genuinely
"generator-shaped". Exp108 then checks the numerical-contract decision on real
Qwen3 weights. Layer0/token31 with the synthetic hidden input stays within the
current hidden_out `1e-2` gate (`max_abs=0.0078125`), but a real dumped
layer35 hidden input does not: `full_hidden_out max_abs=44.0`,
`>1e-2=4008`. Therefore the group-sum formula is not a silent drop-in for the
current exact reference.

Exp109 narrows that failure. The scaled-only form is wrong, but the stronger
centered-dequant form is much closer:

```text
centered = bf16(bf16(q * scale) + zero) - zero
result   = sum(centered * activation) + zero * sum(activation_group)
```

On the same layer35 hidden dump this reduces `full_hidden_out max_abs` from
`44.0` to `0.0625`, with only `2` hidden values above `1e-2`. The residual is
not the exp108 systematic miss; it is the floating-point contraction of
32 separate `zero * activation_dim` products into one `zero * group_sum`
correction. That explains why MyLM's hot loop has real `vadd/vsub/vconv`
dataflow around the group-sum correction: the useful target is not
`q*scale + zero_sum`, but a centered coefficient path plus a deliberate choice
about zero-correction accumulation order.

Exp110 then isolates that choice. Recomputing `centered+zero` before the MAC
matches the current reference exactly on the layer35 dump
(`full_hidden_out max_abs=0`), while both a contracted `zero*group_sum` and a
split zero accumulator path still drift (`max_abs=0.125`, `>1e-2=10`). So the
centered coefficient is not lossy; the parity boundary is whether zero enters
the coefficient before the MAC or is added as a separate accumulator
contribution.

The next production work is to map the four-template schedule onto this
centered-Q4NX numerical contract. If exact direct-reference parity is required,
the generated body must recompose the dequant coefficient before each MAC. If
we choose the MyLM-like `32 main MAC + 1 zero MAC` group shape, that is a
separate numerical contract and needs multi-layer token validation.

Exp111 converts this into dynamic instruction cost. The active exact body is
not slow because it does too few or too many MACs: it has the expected `512`
dynamic `vmac.f` per Q4NX chunk, while MyLM has `528` because of the 16
group-correction MACs. The gap is coefficient construction and control:
active exact pays `vconv.bf16.fp32=1280`, `vconv.fp32.bf16=768`,
`vmul.f=512`, `crupsmode=128`, and `nop=3088`; MyLM's raw hot loop is
`272`, `0`, `16`, `0`, and `216` respectively. So the next exact-parity
target is a scheduled dequant/coefficient pipeline, not another MAC-count
experiment.

The historical Peano/source-shape probes can spell a middle-group-like fragment:

```text
probe_native_q4_group_sum_correction_unroll32_signed:
  vmac.f=33
  vextbcst.16=32
  vst=1
```

That is useful only as a legal-instruction probe. It is not a sufficient
kernel plan, because the real loop's first and last groups are fill/drain
windows and the `vups/vconv/vmov` chains carry values across group boundaries.

Semantically, the fragment corresponds to one middle quant group for one
16-row output lane: `32 q*scale*activation MACs + 1 offset*group_sum MAC`.
It is still not an implementation target by itself. It is not
production-equivalent because the current reference rounds
`bf16(q * scale + offset)` per dimension, while the group-sum form moves the
offset out of the per-dim rounding; and it is not MyLM-equivalent because the
real loop carries dequant/register state across group boundaries.

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
`vst=55`. This is why the active path moved to source assembly: the exact path
cannot be solved by more C++ template unrolling or small-function tiling. The
source-assembly replacement now owns the exact rounding sequence and register
allocation; its remaining problem is schedule density, not C++ spill count.

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
active `main_projection_q4nx_fast.o` is still the only main16 role object, and
`npu_build.py` builds it from:

```text
main_projection_q4nx_fast.cc
main_projection_q4nx_asm.s
```

using `ld.lld -r`. The assembly companion now contains only the production
`q4nx_chunk_accum_asm_zol` body. The older MyLM-style group-shape probe and the
callable exact-lane staging bodies were removed from the qwen3-layer role
object; they now live only as experiment evidence. This matters because even
unreferenced probes add one more implementation shape for humans to maintain,
and final ELF garbage collection is not a substitute for keeping the active
role object single-version.

The latest real-NPU run of the single active source-assembly path is:

```text
main16-q4nx-compute-perf:
  npu_time = 18158.1 us
  done[0:8] = [1472, 1472, 1472, 1472, 1472, 1472, 1472, 1472]
  tile_chunks_per_sec = 1297050.5
```

`tools/check_main16_asm_integration.py --strict` now checks the same invariant
mechanically:

```text
role_has_rejected_lane = false
role_has_old_cpp_wrapper = false
role_has_production_asm = true
core_has_production_asm = true
production q4 = 64 vmac.f, 64 vextbcst.16, 0 vextbcst.32
production q4 current = 66 vldb, 2 vlda, 2 vst, 32 vups.2x
```

So production keeps one main16 implementation shape, and the exact numerical
path now has a hard native-MAC disassembly gate. The callable assembly ABI has
also been verified
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
  NPU time = 6571.9 us
  payload_exact_word_mismatches=4
  payload_max_abs=0.000000477
  PASS

full-layer-qkv-prefix:
  PASS: full-layer physical Q/K/V prefix writes current K/V correctly
```

Therefore the next migration step is not an asm-to-C++ bridge and not the
group-sum approximation; it is to write a callable numerical source-assembly
lane body that preserves the current per-dim Q4NX rounding contract while
keeping the existing C++ scheduler, locks, records, and generator ABI unchanged.

The first production-layout callable attempt also narrowed the ABI problem. The
exact lane candidate must use the real Q4NX chunk strides: packed lane data
advances 0x100 bytes per 32-dim group, while scale, offset, and activation
advance 0x40 bytes. It also has to preserve the scalar/pointer registers that
Peano treats as callee-saved. Without saving `r10/r14/r15/p5`,
`main16-q4nx-compute-perf` timed out because the C++ caller's loop state was
corrupted. With those registers saved, the callable path completed but measured
`21.486 ms`, slower than the old C++ active path (`13.637 ms`). The aligned-ZOL
version is now the active production path and improves that first callable
attempt to `18.158 ms`, but it is still slower than the old C++ path. This
latest number includes group-level activation load reuse and a 6-nop handoff
window; an attempted scale/offset register reuse failed the QKV oracle, and
smaller call-state save sets timed out in the full scheduler context. A second
attempt to make the asm body load/store the target accumulator directly with
`vlda`/`vst` timed out before exp99 fixed the ZOL alignment issue.

`experiments/98_aie2p_float_accum_inplace_asm` isolates that second boundary.
The tiny design calls source assembly from a normal C++ wrapper, uses `vlda` to
load an FP32 accumulator from tile-local memory, updates it with `vmac.f`, writes
it back with `vst`, and returns through the C ABI. It passes on real NPU
(`533.7 us`, output is sixteen FP32 `1.0` values). That means the production
timeout is not explained by bare in-place accumulator load/store; the risk is in
the full Q4NX asm body's register preservation, instruction scheduling, live
ranges, or buffer lifetime. The active path therefore stays on the single
fastest known C++ native-MAC kernel; the source-assembly body remains an
unreferenced production-layout candidate until the whole hot body can beat the
active path.

`experiments/99_aie2p_q4_direct_target_stages` narrows that risk further. One
direct-target exact 32-dim group passes on real NPU (`exact-group1`, `dst=480`).
Two groups inside a normal source-level `jnz` loop time out (`exact-group2`).
The same two groups manually unrolled pass (`exact-group2-unrolled`, `dst=960`).
Four groups manually unrolled also pass (`exact-group4-unrolled`, `dst=1920`).
The eight-group `jnz` loop times out, and the eight-group manual unroll fails at
CDO generation with program-memory overflow (`.text` about 20.6 KiB). Reusing a
four-group body twice is still unsafe: a Peano C++ wrapper times out, while a
pure asm wrapper returns but computes three four-group contributions
(`dst=5760`) instead of two (`dst=3840`).

The important new result is the hardware-loop root cause. A minimal source-asm
`lc/ls/le` smoke with unaligned labels runs only once (`dst=1` instead of `2`).
Changing `mova lc` to Peano-style `add.nc lc, ...`, adding a setup gap, or
trying `lc=3` does not fix it. LLVM-AIE's ZOL lowering requires 16-byte
bundle-aligned `ls/le` and at least 112 bytes from setup to loop-end; MyLM's
`ls=0x260/le=0x1850` also satisfy this. After adding `.p2align 4` around the
source-asm loop labels, the smoke passes, two exact Q4 groups pass
(`exact-group2-hwloop-addnc-aligned`, `dst=960`), and the full eight-group lane
passes (`exact-group8-hwloop-addnc-aligned`, `dst=3840`, core `.text=3008`).
The final core ELF resolves the `jnz` target correctly, so the remaining unsafe
path is normal branch/call reuse; aligned source-assembly ZOL is now the viable
production candidate.

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
   behave like `q4nx_main16_layer_scheduler`.
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
| main16 Q4NX compute-only, active source-asm ABI | 18.158 ms |
| attention-O bf16 slice, active main16 | 12.194 ms |
| full single layer, active main16 after source-asm handoff optimization | 28.479 ms |

The local MyLM full-model benchmark probe measured about `4.93 tok/s` for
Qwen3-8B at 1k/8k context on this machine, or roughly `203 ms/token`. Spread
over 36 layers, that is about `5.6 ms/layer` including non-layer overhead.
The published MyLM table reports `11.9 tok/s` at 1k and `10.4 tok/s` at 8k,
which corresponds to an even tighter layer budget. Either way, the current IRON
single-layer `28.479 ms` result is still about 5x slower than the local MyLM
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
replay, the current source-asm path runs token31 full decode at `28.479 ms`.
This is useful raw-core preparation, but it confirms the larger point: the
remaining gap is the scheduled main16 core body, not a spare record-copy helper.
A separate attempt to remove the defensive partial-row/bounds branches from the
C++ helper itself was not kept: it made a full-layer main16 ELF grow to
`.text = 0x4760`, which overflows AIE program memory.

The QKV prefix runner is a correctness gate with downstream tiles intentionally
blocked after the prefix, so its end-to-end NPU time is not a clean main16
throughput number. The attention-O result is more useful for performance: it
exercises QKV, edge attention, packet2 handoff, and O projection in one
full-layer-derived path.

## Conclusion

The active IRON main16 role is now a generated linked role program:
`q4nx_main16_layer_scheduler` dispatches Q/K/V/O/upgate/down phase bodies, those
phase bodies configure shared tile-local control words, and the shared Q4 body
emits IRON compact headers 10..15. `main_projection_q4nx_fast.cc` still carries
small callable helpers for the hot-body microbench, but the production
full-layer path no longer expands a C++ chunk loop or calls
`q4nx_chunk_accum_slice_i32_fast`. The dataflow principle is the same as MyLM;
the remaining difference is that MyLM lowers the same ABI into a raw scheduled
whole-core body, while IRON currently links a generated source-assembly role
object into the MLIR-AIE core.

The next main16 experiment should keep this single object and further compress
the generated Q4NX source-assembly microkernel under the existing numerical
contract and ABI:

```text
DMA0 activation chunk: 128 dwords
DMA1 Q4NX weight chunk: 1280 dwords
output compact record: 17 dwords
```

`tools/check_main16_asm_integration.py --strict` is now the active structural
gate for this source-assembly route. It verifies that the generator and checked
in asm agree, the final core ELF contains `q4nx_chunk_accum_asm_zol`, the old
C++ hot-body wrapper is absent, and the production body still has
`vmac.f/vextbcst.16/ZOL` while avoiding `vextbcst.32`. If we later switch from
linked scheduler plus source-asm hot body to a whole-core replacement ELF, then
`tools/check_main16_program_shape.py` becomes the right gate again: it treats
the extracted MyLM `c2r2` raw program as the positive example and checks the
raw dispatcher shape before packaging and running numeric gates.

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
lives in a linked C++ AIE core entry point. The remaining gap is that the active
source-assembly Q4NX microkernel still is not MyLM's raw scheduled
microkernel: it has `vextbcst.16`, but far too much conversion, dequant, and
call-state traffic. Keep MLIR-AIE for topology and packaging, and focus the
next performance work on shortening this linked Q4NX source-assembly lane body
while preserving the exact production Q4NX
rounding contract.

The source-assembly path is now numerically proven on a real-NPU Q4NX lane
boundary, not only as a shape probe. `run_asm_npu_smoke.py` first proved a
hand-written full 8-group exact Q4NX lane body could be called from an AIE core
and match the BF16 reference for all 8 groups / 256 input dims in one 16-row
lane. `experiments/99_aie2p_q4_direct_target_stages` then resolved the larger
production-shape issue: full static lane unroll overflows program memory, normal
source-level `jnz` around the large exact body times out, and unaligned
source-asm `lc/ls/le` silently runs once. The passing production candidate uses
pointer registers for scale/offset/activation and a bundle-aligned
source-assembly ZOL. `exact-group8-hwloop-addnc-aligned` matches the exact
accumulator value (`dst=3840`) with a small `.text=3008` core. This does not
complete the performance work, but it removed the hardware-loop and program-size
doubt. That body is now wired into the production main16 scheduler and
weight/activation ABI; the next work is to shorten and reschedule it until the
real NPU gate improves, not merely passes.

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
for the current `q4nx_chunk_accum_asm_zol` replacement.

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
scaled-only group-sum production attempt shows that shortcut is not a free
drop-in replacement. Exp109 shows a better numerical target:

```text
centered = bf16(bf16(q * scale) + offset) - offset
result   = sum(centered * activation) + offset * sum(activation_group)
```

This centered form explains why a fast body can still contain `vadd/vsub/vconv`
around the group-sum correction. On real layer35 input it removes the large
systematic error, but a single contracted offset/group-sum MAC is still not
bit-identical to the direct per-dim accumulation order. Exp110 further shows
that split per-dim offset MACs are also not enough if they are added as
separate accumulator contributions; exact parity is restored only when
`centered + offset` is recomposed before the MAC. Exp111 confirms the active
exact body already has the right dynamic MAC count; the excessive cost is the
unscheduled coefficient construction and repeated conversion/control setup.
Exp112 makes the cost floor explicit: even an ideal exact body still needs
`512` vector multiplies and `1536` conversion operations per chunk, while the
MyLM-like group-correction contract needs `16` vector multiplies and `272`
conversions. That means exact assembly can improve materially, but MyLM-like
speed requires either a different numerical contract or a still-unknown fused
rounding path. Exp113 then traces MyLM `vmac.f` operands in the canonical
steady group: `6/66` vector operands and `2/33` accumulator sources are carried
across the group boundary. Exp114 refines that trace to half-register cells:
`25/66` group1 vector operands are mixed-half values and `9/66` have
cross-group cells. The next useful assembly artifact is therefore a scheduled
operand graph with explicit half-register fill/steady/pre-drain/drain state,
not a self-contained 33-MAC macro. The remaining decoder gap is the
scheduled operand mapping through `vmac.f #0x33c`: exp102 already validates the
primitive `#0x33c` lane model on real NPU, while exp114 shows that MyLM often
feeds that MAC with mixed-half `xN` values assembled by prior `vmov/vconv/vext`
slots. Exp115 turns this into a generator-shaped artifact: `fill -> steady` has
23 incoming data cells, and the steady-to-steady data-cell signature is stable
for groups `2..6` (`5121c8604a8461c4`). This gives the next assembly generator
a concrete boundary-state contract rather than a Markdown-only reverse note.
Exp116 verifies that boundary by generating a graph-annotated steady assembly
include from the JSON: stripping comments recovers MyLM group1 exactly
(`189` slots, `33` `vmac.f`, zero missing graph records). Exp117 extends the
same approach to the complete hot loop: `fill + steady*5 + pre_drain + drain`
exact-matches MyLM `0x260..0x1850` (`1532` slots, identical hash) while
annotating all `165` steady-section `vmac.f` instructions from the graph.
Exp118 converts that into a full liveness table: every group boundary carries
the same `27` data cells plus `12` pointer/scalar cells. This is the clearest
current evidence that the fast body is a fixed register-residency software
pipeline. Exp119 then extends the operand graph to all `264` `vmac.f` slots:
the group MAC counts are `28,33,33,33,33,33,33,38`, group1 has a distinct
fill-to-steady operand signature, and groups `2..5` share the same
steady-to-steady operand signature. Exp120 merges those two artifacts into the
first sectionized generator contract: `fill`, `fill_to_steady`,
`steady_to_steady`, `pre_drain`, and `drain` macros re-expand to the exact
original `1532` slots and preserve `264` group MACs. Exp121 uses that contract
as a numerical branch gate: `exact_recompose_coeff` is exact in nominal and
stress synthetic scenarios, while the MyLM-like group correction keeps the
`264/528` MAC shape but has stress `>1e-2` mismatches. The production-safe next
generator should therefore preserve the exp120 live-state sections while
targeting an exact 32-MAC/group body, unless a later real token gate explicitly
accepts the MyLM-like numerical contract. Exp122 makes that rewrite mechanical:
remove one zero-correction MAC per logical group, changing the section MAC shape
from `28,33,33,33,33,33,33,38` to `27,32,32,32,32,32,32,37`, for `256` static
MACs and `512` dynamic MACs.

Exp123 deliberately pauses before modifying production assembly and builds a
slot-level semantics table for MyLM group0 (`0x260..0x52a`). The fill section
has `192` instruction slots, `28` `vmac.f`, `32` `vextbcst.16`, `8` `vups.4x`,
`16` `vconv.bf16.fp32`, and no `vst`. Its boundary1 state is not a closed
per-group result: group0 produces `23` data cells plus `4` control cells that
remain live into group1, while `4` data cells and `8` control cells are carried
from entry. This makes the next generator requirement sharper: preserve the
fill-to-steady live state and only then substitute the exact 32-MAC/group
arithmetic body.

Exp124 then traces group2 with the group0-1 live state already applied. This
is the first true steady-to-steady section: `189` slots, `33` `vmac.f`, `32`
`vextbcst.16`, `8` `vups.4x`, `17` `vconv.bf16.fp32`, and no `vst`. Boundary2
and boundary3 have the same `27` data cells plus `12` control cells and the
same relative producer classes (`12` entry, `3` persistent group0, `24`
previous-group). The transition changes `24` producers (`23` data and `1`
control) from group1 to group2. This is the reusable steady contract the exact
rewrite must preserve.

Exp125 quantifies why the helper boundary is expensive, but the first direct
nocall implementation is not a production path. The current C++ scheduler
performs `192` Q4NX helper calls per main tile for Q/K/V and `1472` per tile
for the full layer. Across main16 that is `3072` QKV helper calls and `23552`
full-layer helper calls. Each helper call executes `13` SAVE slots, `13`
RESTORE slots, two accumulator loads, and two accumulator stores. SAVE/RESTORE
alone therefore costs `4992` QKV or `38272` full-layer instruction slots per
tile.

The tested linked `q4nx_main16_qkv_scheduler_nocall` preserved the old C ABI,
topology, BD, and lock IDs, and static disassembly confirmed that it did not
call or jump to `q4nx_chunk_accum_asm_zol`. It still timed out on the
`full-layer-qkv-prefix` NPU gate, while the old C++ scheduler passes on the
same dataflow. That narrows the failure to hand-written scheduler
control/timing rather than row1/c1r2/c1r3 connectivity. The active path is
therefore back to the known-good C++ scheduler plus generated source-assembly
Q4NX hot body, but the old QKV/QKVO/full scheduler symbols have been collapsed
into one `q4nx_main16_layer_scheduler(..., phase_limit)` entry. QKV prefix,
attention-O, and full layer now differ only by `phase_limit=3/4/7`. A future
nocall attempt needs progress counters or a Peano-shaped whole scheduler before
it can replace the active path.

The single-entry refactor has passed the first real gate: `full-layer-qkv-prefix
token31` runs on NPU and writes current K/V correctly with `phase_limit=3`.
The full decode gate also still passes with `phase_limit=7`; token31 measured
`28.479 ms` and `final_hidden_out max_abs=0.0078125` with zero mismatches.

Exp127 also now has the return-address contract required for a real generated
candidate. The earlier scaffold was compileable but unsafe: nested `jl` calls
would overwrite the caller's `lr` across entry, dispatcher, phase bodies,
shared Q4 body, and record emitter. The generator now emits a legal 64-byte
`lr` frame for every non-leaf generated function and checks that the record
emitter remains a leaf.

Exp126 turns the replacement target into a whole-main16 contract. The next
active candidate should be one generated role program with a shared Q4 body,
Q/K/V/O/up/gate/down phase bodies, and a dispatcher. It should not expose a
second QKV-only scheduler mode. The first replacement must keep IRON compact
headers `10..15` at the row1/c1r1 boundary; MyLM's `0x1/0x4/0x8` headers are
evidence for the raw dispatcher, but the current IRON downstream routing is not
compatible with them as-is.

Exp127 now makes that target concrete without touching the active path. It
generates a compileable scaffold with stable symbols for the entry,
dispatcher, shared Q4 body, compact record emitter, and Q/K/V/O/upgate/down
phase bodies. The manifest fixes the first linked replacement ABI to the
current MLIR call convention: `p0..p5` carry weight, activation, and record
ping-pong buffers, while `r0/r1/r2/r3` carry group, row, num_rows, and the
exclusive phase limit. It also
records the core lock contract (`48..53`), the IRON record header formulas,
and the full dynamic schedule: `1472` Q4 body invocations and `76` compact
records per main tile. The dispatcher uses the same phase-limit shape as the
active `q4nx_main16_layer_scheduler`: `3=qkv`, `4=qkvo`, `6=upgate`, `7=full`.
The record emitter is no longer a pure comment: it
writes the real IRON header, converts the 32-lane FP32 accumulator into the
16-dword BF16 record payload, and clears that accumulator. It also has a
generator check that prevents the emitter from touching caller phase-loop state
registers such as `r11/r12/r16`, because those registers are needed after the
emitter returns. `r27` is reserved as the `sel.eqz` predicate because the
assembler rejected low-register predicates; `r29` is reserved as the zero
insert index because source assembly accepts `vinsert.32 ..., r29, ...` while
rejecting the disassembler's printed `#0` form.

The phase bodies now only configure header-run registers and call the shared Q4
run body once per header run: 3 calls for Q/K/V, then one call each for O,
upgate, and down. The manifest checks `6` static phase-to-Q4 calls for `1472`
dynamic chunk executions. The shared Q4 run body owns the record loop, chunk
loop, DMA lock protocol, ping/pong selection, exact chunk body execution, and
record emission. Its exact chunk body is emitted from the same
`qwen3-layer/tools/main16_q4nx_asm_lib.py` source template as the active
`q4nx_chunk_accum_asm_zol` body, so exp127 no longer carries a second Q4NX
hot-body copy. The embedded chunk body uses the no-save form (`64 vmac.f`, `64
vextbcst.16`, `2 vst`, zero save/restore slots). The run body rematerializes
`r15=1` before parity/lock-release uses and `r14=-1` before acquire uses, and
keeps persistent buffer pointers in general registers (`r19`, `r20`, `r21`,
`r22`, `r26`, `r28`, and `r30`) so the exact chunk body can clobber pointer
registers. The remaining whole-program gap is replacing the current exact
dequant body with a MyLM-density scheduled body before this scaffold can
replace `q4nx_main16_layer_scheduler`.

## Current Exact Body Constraints

MyLM hoists the Q4NX scalar constants and control setup at the raw hot-loop
entry. In the active IRON `q4nx_chunk_accum_asm_zol` external-call body that is
not currently valid.

The tested hoist variants were:

- move `0x4b01`, `0x3c`, `0x33c`, zero, `crupsmode`, and `s0` to function
  entry;
- keep only `crupsmode/s0` at function entry;
- reduce `Q4_HANDOFF` from six `nop` instructions to four.

All three variants compiled and ran `main16-q4nx-compute-perf`, but failed the
real `qwen3-8b-qkv-compact-output` numerical gate. The scalar-hoist variants
produced full payload mismatch; the shorter handoff produced smaller but still
global mismatch. The current exact body therefore intentionally keeps:

```text
#19201 / #60 / #828 / crupsmode / s0: 16 occurrences per chunk body
Q4_HANDOFF: 6 nop instructions before vmov bmll1, bmll0
```

`check_main16_asm_integration.py --strict` now checks these counts explicitly.
This does not mean MyLM needs these repeated setup instructions. It means our
current external-call exact-rounding body has local scheduling and control-state
dependencies that the MyLM raw dispatcher avoids with a different fixed
register plan and instruction schedule.
