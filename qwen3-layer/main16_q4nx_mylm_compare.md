# Main16 Q4NX MyLM Comparison

This note records the current performance conclusion for the main16 projection
kernel. It is based on local reverse notes and disassembly, not on a new MyLM
benchmark run.

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
| IRON baseline `q4nx_chunk_accum_fast` hot body | 575 | 1179 | `vst=154`, `vconv.bf16.fp32=110`, `vmac.f=44`, `vbcst.16=45` |
| IRON scheduled probe hot body | 637 | 1570 | `vlda=221`, `vst=174`, `vconv.bf16.fp32=160`, `vmac.f=64`, `vbcst.16=65` |
| IRON `q4nx_chunk_accum_slice_i32_fast` | 6 | 8 | wrapper around the active single-version main16 Q4NX kernel |
| IRON `q4nx_chunk_accum_block_slice_i32_fast` | checked by script | checked by script | active O/down block-accum wrapper |

This does not mean one MyLM call does the same dynamic work as one IRON helper
call; the loop counters differ. The important evidence is structural: MyLM has
a long raw scheduled loop with hundreds of vector MAC/broadcast slots in one
body, while the IRON kernel exposes a small C++ helper with visible scalar loop
control and far fewer vector MAC slots in the generated hot body.

## Performance Evidence

Current IRON stage-budget measurements on real NPU:

| Slice | Time |
| --- | ---: |
| row1 weight stream | 8.357 ms, 13.438 GiB/s |
| main16 Q4NX compute-only, active 18-dim unroll | 14.263 ms |
| attention-O bf16 slice, active main16 | 13.202 ms |
| full single layer, active main16 | 29.236 ms |

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
microkernel with the existing ABI:

```text
DMA0 activation chunk: 128 dwords
DMA1 Q4NX weight chunk: 1280 dwords
output compact record: 17 dwords
```

Keep the current Python MLIR generator for topology, BD/lock ownership, row1
fanout, and source-side replay. The next implementation step is not another C++
variant; it is either shrinking the full main16 phase body enough to reduce
control overhead, or replacing the phase body with a MyLM-style raw scheduled
core program. Only after a new implementation improves end-to-end full decode
should it replace the single active main16 backend.
