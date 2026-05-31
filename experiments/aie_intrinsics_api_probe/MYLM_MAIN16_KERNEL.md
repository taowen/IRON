# MyLM Main16 Kernel Reverse Notes

This note records the concrete reverse-engineered facts behind MyLM's fast
main16 Q4NX projection kernel. It is intentionally narrower than the full
layer dataflow notes: only the main16 raw core program and the Q4 hot loop are
covered here.

## Data Source

Bundle generated from MyLM Qwen3-8B layer 31 with max context 4096:

```bash
/var/home/taowen/projects/MyLM/tools/re/dump_qwen3_layer_contract.sh \
  31 4096 /tmp/mylm_qwen3_layer_L31_deep
```

The wrapper script stopped in an older `mylm_layer_report.py` field assumption,
but the required files were already generated:

- `/tmp/mylm_qwen3_layer_L31_deep/layer_cdo.csv`
- `/tmp/mylm_qwen3_layer_L31_deep/layer_bd.csv`
- `/tmp/mylm_qwen3_layer_L31_deep/programs/*_program.bin`
- `/tmp/mylm_qwen3_layer_L31_deep/programs/program_segments.tsv`

The raw program images were disassembled with:

```bash
python3 /var/home/taowen/projects/MyLM/tools/re/aie_program_disassemble.py \
  --program-dir /tmp/mylm_qwen3_layer_L31_deep/programs \
  --out-dir /tmp/mylm_qwen3_layer_L31_deep/disasm \
  --tiles c2r2,c2r3,c3r2,c3r3,c4r2,c5r5,c1r2,c1r3,c6r2,c0r2,c0r3,c7r2,c7r3
```

## Why This Is Hard To Read From MyLM

MyLM does not ship the C++/assembly source for the AIE core kernel in this
tree. The layer xclbin contains raw per-tile program-memory segments. That
means there are no function names, no C variables, and no compiler IR.

The main difficulty is therefore not finding `vmac.f`; it is reconstructing
the whole ABI around the raw program:

- AIE branch/call slot windows matter. Several caller argument assignments sit
  after `jl`, so a normal linear reading puts arguments on the wrong side of
  the call.
- AIE load-use latency matters just as much. A naive source-assembly
  `lda; st` can store the old scalar register value even when the DMA buffer,
  pointer register, and lock protocol are correct. Peano-generated C++ hides
  this by placing enough independent work or delay slots between the load and
  the consumer.
- The core program is segmented CDO payload, not a clean ELF. Whole-image
  objdump is fragile; segment-aware raw-to-ELF wrapping is required.
- AIE bundles mix scalar, vector, memory, and pointer operations on one
  disassembly line, so a line is not one operation.
- Hardware loop registers (`ls`, `le`, `lc`) make static instruction counts
  different from runtime instruction counts.
- The program uses local control/scratch bytes around `0x73c60..0x73d00`.
  These are not declared as C symbols; their meaning must be inferred from
  loads, stores, BD bases, and lock protocol.
- Same raw program is used by all 16 main tiles, so tile role metadata is not
  encoded as 16 separate source files. The per-tile difference comes from the
  dataflow, BD/lock sequencing, and weight/activation streams.

The practical hard part was not "find `vmac.f`". It was proving the data
dependence:

```text
phase body activation chunk -> 8 group sums -> p3 scratch
Q4 microkernel p3 loads     -> offset correction MACs
Q4 microkernel p1 loads     -> 8 x 32 activation-lane broadcasts
Q4 microkernel p0 streams   -> packed int4 payload plus scale/offset lanes
```

That is now mechanically checked by
`experiments/aie_intrinsics_api_probe/analyze_mylm_main16_kernel.py`.

The current IRON route has therefore moved one level lower for the Q4NX hot
body. We are not trying to guess a C++ source pattern for every instruction
anymore. `run_asm_probe.py` proves that Peano's integrated assembler accepts a
hand-written AIE2P middle-group block with the same key inventory as MyLM:

```text
vmac.f=33
vextbcst.16=32
lda.s16=1
vunpack=8
vups=8
vst=0
```

`run_asm_link_probe.py` separately proves the intended production packaging
boundary: a C++ wrapper and a source-assembly symbol can be merged into one
relocatable AIE role object with `ld.lld -r`. That keeps the qwen3-layer
generator single-versioned while allowing the inner Q4NX body to become
source assembly.

`run_asm_npu_smoke.py` now proves the same boundary on real NPU for a small
exact Q4NX sub-body. The smoke uses the existing stable host/shim input path,
calls source assembly from an AIE core, reads packed uint4 plus scale/offset
and activation values from a DMA-filled local buffer, runs one complete 32-dim
group of the current `bf16(q * scale + offset)` contract with
`vextbcst.16 + vmac.f`, stores the accumulator through
`vst.conv.bf16.fp32`, releases the output lock from asm, and validates the
host-visible output. The failed intermediate forms were informative: scalar
`lda` followed immediately by `st` wrote zero because the load result was not
ready, the first Q4NX asm smoke stored the old offset accumulator until extra
padding was inserted after the final `vmac.f`, and the first 8-dim stitch was
wrong until scale/offset/activation were reloaded before the second 4-dim
body. The first 32-dim version also hit the `vldb.128` immediate range limit,
so the final smoke advances a pack-base pointer with `padda` before each
4-dim block. This is the concrete reason MyLM-style source assembly has to be
a scheduled hot body with explicit register and pointer ownership, not merely
the same operations written as mnemonics.

## Main16 Program Shape

All 16 main tiles share the same raw program image:

```text
sha256 c50327d37903c4e09642d5f939165dd9b43bb6766348150eb1218e082d0332cd
tiles  c2..c5,r2..r5
```

The c2r2 program segments are:

```text
0x0000  492 B  prologue/top-level ping-pong setup
0x01f0 5760 B  Q4NX microkernel
0x1870 1544 B  Q/K/V phase body
0x1e80 1544 B  O phase body
0x2490 1544 B  up/gate phase body
0x2aa0 1560 B  down phase body
0x30c0 1544 B  alternate path
0x36d0  504 B  dispatcher
0x38d0  324 B  function table / tail
```

The visible DMA/lock ABI matches the IRON main16 ABI:

```text
activation DMA0: bd0/bd1, base 0x78000/0x7c000, len 128 dwords, L0/L1
weight     DMA1: bd2/bd3, base 0x72800/0x74000, len 1280 dwords, L2/L3
record     MM2S: bd4/bd5, base 0x73c1c/0x7541c, len 17 dwords, L5/L4
```

The dispatcher at `0x36d0` calls the normal phase bodies in fixed order:

```text
0x1870 Q/K/V:   header 0x1, records_per_tile 12
0x1e80 O:       header 0x4, records_per_tile 8
0x2490 up/gate: header 0x8, records_per_tile 48
0x2aa0 down:    header 0x4, records_per_tile 8
```

## Q4NX Microkernel ABI

The call slot before `jl #0x1f0` shows the Q4 microkernel arguments:

```text
1d70: jl #0x1f0
1d76: movs p3, r15; mov p0, r18
1d7e: movs p1, r16; movxm p2, #0x73c80
```

From the phase body setup:

- `p0` is the selected Q4NX weight ping/pong buffer.
- `p1` is the selected 128-dword activation ping/pong buffer.
- `p2` is the fixed local accumulator/scratch window at `0x73c80`.
- `p3` is a phase-body-produced 16-bit scratch stream consumed by the
  microkernel. It is not a host-visible stream.

The `p3` scratch is now decoded far enough to explain the Q4NX contract. Every
phase body first loads the 128-dword activation chunk as eight 32-lane BF16
vectors, reduces each 32-lane group with `vadd.f` plus lane-shift steps, stores
eight `s16` values to stack scratch through `st.s16`, then passes that scratch
base as `p3` in the `jl #0x1f0` call slot:

```text
1a2a..1c36: vlda.conv.fp32.bf16 eight activation vectors from p3/r16
1ac6..1d6a: st.s16 eight reduced values into p2/r15 scratch
1d70:       jl #0x1f0
1d76:       movs p3, r15; mov p0, r18
1d7e:       movs p1, r16; movxm p2, #0x73c80
```

The Q4 microkernel consumes eight `lda.s16 r7, [p3], #0x2` values in the static
hot-loop body, one for each 32-lane activation group. The loop body later
rewinds `p3` by 16 bytes before the second output-lane pass:

```text
153a: add.nc p3, r21, #-0x10
```

So the eight reduced values are produced once by the phase body and reused for
both 16-row output lanes. This matches Q4NX's group size of 32 and the dequant
formula `q * scale + zero`: the zero/offset term needs
`sum(activation[32 lanes])` per quant group. The remaining detail is only the
exact reduction rounding mode, not the ownership or purpose of `p3`.

The current analyzer verifies the producer side directly:

```text
phase_body_group_sum_producer:
  activation_vector_loads=8
  scalar_extracts=8
  scratch_stores=8
  vadd.f=40
  vshift=32
  produces_exactly_8_group_sums=True
```

The scratch store detection has to account for both post-increment and indexed
`[p2]` forms. This is a concrete example of why a naive line-grep missed the
mechanism earlier even though the mechanism was present in the disassembly.

The Q4 setup at `0x1f0` programs a zero-overhead loop:

```text
1f0: lc = 0x2, p4 = 0x73c64
218: r1 = 0x200, m0 = 0x400
22c: r4 = 0x33c, p0 += 0x400, r17 = p0
238: ls = 0x260
23e: le = 0x1850
244: r5 = 0x3c, crrnd = *(0x73c62)
24e: crsat = *(0x73c64)
```

The `p0 += 0x400` is important: it skips the Q4NX scale and zero-point headers
and starts the packed int4 payload. The scale/zero regions are still addressed
through derived pointers in the hot loop. Same-bundle pointer moves observe the
pre-increment pointer here: `r17` remains the chunk base, while `p0` advances
to `chunk + 0x400`.

The first hot-loop group shows the pointer model:

```text
260: vlda x8, [p0], #0x40; vldb x11, [p1], #0x40; r16 = r17 + 0x200
26c: ... p5 = r17 + 2 * group
284: ... p4 = r16 + 2 * group
4dc: vldb wl2, [p5], #0x40
4ec: vldb wl6, [p4], #0x40
```

Interpreting the Q4NX chunk format:

```text
r17 = chunk base
p0  = chunk + 0x400 packed int4 payload
p1  = 256-bf16 activation chunk
p5  = chunk + 0x000 + group * 2 scale lanes
p4  = chunk + 0x200 + group * 2 zero/offset lanes
p3  = eight activation group sums
p2  = 0x73c80 accumulator/output scratch
```

The hot loop resets `p1` by 0x200 bytes after reading all eight activation
vectors:

```text
156e: paddb [p1], #-0x200
```

That is the strongest evidence that `lc=0x2` means two output-lane passes over
the same 256-bf16 activation chunk: first for rows 0..15, second for rows
16..31. The int4 pointer keeps advancing across the two Q4NX payload halves.

The MAC control word is now decoded:

```text
r4 = 0x33c = aie2p_compute_control(
    __SIGN_SIGNED, __SIGN_SIGNED,
    amode=2, bmode=3, variant=1,
    zero_acc=0, shift16=0,
    sub0=0, sub1=0, sub2=0, sub_mask=0
)
```

Peano's default `mac_elem_16_conf(a, b, acc, 0, 0, 0)` uses the same BF16
mode fields but with unsigned sign bits, producing `0x3c`. The signed overload

```c++
mac_elem_16_conf(
    lhs, __SIGN_SIGNED,
    rhs, __SIGN_SIGNED,
    acc, 0, 0, 0
)
```

reproduces `#0x33c` and still emits `vextbcst.16 + vmac.f` when `rhs` is a
`v32int16` activation view bitcast back to `v32bfloat16`.

## Hot Loop Instruction Shape

For static range `0x260..0x1850` in c2r2:

```text
vmac.f          264
vextbcst.16     256
vextbcst.32       0
vunpack          64
vups             64
vldb             46
vlda             11
vst               0
lda.s16           8
```

`lc=0x2` makes this static body run twice. The dynamic count is therefore:

```text
vextbcst.16 = 256 * 2 = 512
lda.s16     =   8 * 2 = 16, but it reuses the same 8 scratch values
vbcst.16    =   8 * 2 = 16
vmac.f      = 264 * 2 = 528
```

That dynamic `vmac.f` count is not arbitrary. One Q4NX chunk computes
`32 rows x 256 activation columns`. With 16-row vector MACs:

```text
main term:        2 output lanes * 8 groups * 32 dims = 512 vector MACs
zero/offset term: 2 output lanes * 8 groups           =  16 vector MACs
total:                                                   528 vector MACs
```

This closes the runtime-count question: MyLM's static 264-MAC loop is exactly
half of one 32-row chunk, and the hardware loop repeats it for the two 16-row
payload halves.

The activation vector load pattern is exact:

```text
x11 loads at: 0x260, 0x52a, 0x7de, 0xa92, 0xd46, 0xffa, 0x12ae, 0x1566
```

Each of those eight `x11` vectors has all 32 lanes extracted and broadcast:

```text
lane 0..31: each appears 8 times as vextbcst.16 x?, x11, #lane
total: 8 activation vectors * 32 lanes = 256 vextbcst.16
```

This is now enforced by
`experiments/aie_intrinsics_api_probe/analyze_mylm_main16_kernel.py`, not just
by manual inspection:

```text
semantic_shape:
  q4_chunk=32x256
  output_lane_passes=2
  vector_rows=16
  activation_groups=8
  activation_lanes_per_group=32
first_principles_check:
  main_macs=512
  correction_macs=16
  total_macs=528
  matches_dynamic_vmac=True
  expected_static_activation_extracts=256
  matches_static_vextbcst=True
activation_stream:
  lane_groups=8
  complete_lane_groups=8
```

This is the concrete MyLM trick: the activation is live as a 32-lane vector,
then every lane is broadcast from the vector register while packed Q4 weights
are unpacked/dequantized and MACed. There is no per-lane vector spill in the
hot loop.

The analyzer also splits the static hot loop by activation group. The middle
groups have the canonical shape:

```text
group=1..6:
  lanes=32
  vmac.f=33
  vextbcst.16=32
  vbcst.16=1
  lda.s16=1
  vunpack=8
  vups=8
  vst=0
```

The first and last groups differ by pipeline prologue/epilogue motion, but the
whole static loop still totals exactly `264 vmac.f`, `256 vextbcst.16`,
`8 vbcst.16`, and `8 lda.s16`.

The only vector store in the microkernel area is after the hot loop:

```text
1850: vst lfh0, [p2, dj0]
```

Record output happens in the phase body, not inside the Q4 hot loop:

```text
1df6: st el0, [p2], #4
1dfe: vst.conv.bf16.fp32 bmhh1, [p2, #0x0]
1e02: vst.conv.bf16.fp32 bmhl1, [p2, #0x20]
```

## Direct Contrast With Current IRON Fast Kernel

The earlier production `qwen3-layer/main_projection_q4nx_fast.o` disassembled
roughly as:

```text
vmac.f       44
vextbcst.16  0
vbcst.16    45
vunpack     44
vups        22
vst        154
lda.s16     44
```

This explains the performance gap better than the vague phrase "MyLM uses
better instructions":

- MyLM exposes a 5.6KB raw scheduled Q4 microkernel with hundreds of statically
  placed MAC/broadcast/unpack operations.
- IRON currently relies on a much smaller C++ AIE API kernel with scalar
  activation broadcasts and many vector stores, which are consistent with
  register spills and loop-control overhead.
- The failed production transplant used the default native `mac_elem_16_conf`
  control shape. That path produces `0x3c`, while MyLM uses `0x33c`. This is a
  concrete reason it could improve speed while breaking O/attention numeric
  contracts.

The current active production path has moved to exact per-dim rounding plus
signed native BF16 MAC with a 32-dim full unroll. It is not the group-sum lane
candidate; it keeps the existing `bf16(q * scale + offset)` reference contract
and has passed QKV compact, full-layer QKV prefix, attention-O, and full
decode. Its role-object wrapper now disassembles as:

```text
iron_fast_q4_wrapper:
  vmac.f=64
  vextbcst.16=64
  vextbcst.32=0
  vunpack=64
  vlda=282
  vst=194
  vconv.bf16.fp32=160
```

But it is still not MyLM-equivalent:

```text
MyLM static hot loop:
  vunpack=64
  vups.4x=64
  vconv.bf16.fp32=136

IRON active wrapper:
  vmac.f=64
  vextbcst.16=64
  vlda=282
  vst=194
  vconv.bf16.fp32=160
```

This explains why the recent production move improved the integrated shape but
did not become MyLM-fast: the high-level Peano expression can spell the right
signed MAC/broadcast semantics, but it still carries much more load/store and
conversion traffic than the raw scheduled MyLM loop. The noinline group-sum
lane probe remains useful as an instruction-shape target, but it is not the
active decode path because it changes the accepted rounding contract.

The exact current-reference C++ intrinsic full-unroll route was tested next and
is now rejected on code shape alone. It keeps the accepted per-dim rounding
contract and reaches the desired broadcast/MAC family, but Peano spills so much
that it is worse than the active wrapper:

```text
exact_rounding_unroll32:
  vmac.f=32
  vextbcst.16=32
  vst=126

exact_rounding_chunk_lane_kernel:
  vmac.f=256
  vextbcst.16=256
  vconv.bf16.fp32=640
  vconv.fp32.bf16=257
  vst=1043
```

This narrows the remaining path: for exact Q4NX rounding, the hot body needs
source assembly or another lower-level scheduler that explicitly owns register
allocation. A bigger C++ template body is not a viable production migration.
A smaller exact C++ body does not create a hidden sweet spot either:

```text
exact_rounding_unroll4:  vmac.f=4,  vextbcst.16=4,  vst=2
exact_rounding_unroll8:  vmac.f=8,  vextbcst.16=8,  vst=15
exact_rounding_unroll16: vmac.f=16, vextbcst.16=16, vst=51
exact_rounding_unroll32: vmac.f=32, vextbcst.16=32, vst=126
```

The only low-spill exact C++ body is the 4-lane fragment, but its MAC density is
too low to be a MyLM-grade hot loop. The larger exact fragments spill before
they reach useful group-level density.

The next attempt split one 32-lane exact group into eight noinline 4-lane
kernels. That proves the function boundary can remove local vector spills, but
it does not produce a viable hot loop:

```text
exact_rounding_group4_kernel:
  vmac.f=4
  vextbcst.16=4
  vst=0

exact_rounding_group4_call_chain aggregate for one group:
  vmac.f=32
  vextbcst.16=32
  vst=4
  wrapper_jl=8
  wrapper_hardware_loop=0
```

Scaling that to one 8-group lane body would put 64 calls in the chunk path,
where MyLM has one raw scheduled loop body. So function boundaries are a useful
diagnostic for register pressure, but not the production answer.

The source-assembly callable ABI has been narrowed as well. A C++ wrapper can
call a source-assembly function that receives normal pointer arguments and
executes vector load/MAC/store instructions:

```text
probe_asm_vector_mac_smoke:
  p0=lhs pointer
  p1=rhs pointer
  p2=dst pointer
  vldb=2
  vmac.f=1
  vst=1

probe_asm_vector_mac_smoke_wrapper:
  j #0
  R_AIE_1 probe_asm_vector_mac_smoke
```

This is different from the rejected asm-to-C++ tail-jump. The viable production
direction is C++ scheduler -> complete asm hot body, with no return jump from
asm into a C++ semantic helper.

## What Is Now Known

MyLM's main16 speed comes from a whole-core raw program:

1. A fixed dispatcher drives Q/K/V/O/up/gate/down bodies without host-side
   per-chunk scheduling.
2. Each phase body waits on weight and activation locks, builds local scratch,
   calls the same Q4 microkernel, emits one 17-dword record, then releases the
   DMA rings.
3. The Q4 microkernel keeps activation as eight 32-lane vectors and performs
   512 dynamic vector-lane broadcasts from those registers, because the static
   256-lane schedule is replayed for two 16-row output lanes.
4. Packed int4 data is consumed after the `+0x400` Q4NX header skip, while
   scale lanes come from `chunk+0x000`, zero/offset lanes come from
   `chunk+0x200`, and activation-sum scratch feeds the group correction term.
5. The hot loop performs 528 dynamic 16-lane BF16 MACs per chunk: 512 main
   `q*scale*activation` MACs plus 16 `zero*sum(activation)` correction MACs.
6. The hot loop has no vector stores; stores are confined to accumulator
   scratch after the hot loop and compact record output after the microkernel.
7. The BF16 MAC control word is signed/signed `0x33c`, not the default
   unsigned/unsigned `0x3c` used by the simplest Peano compat overload.
8. The `p3` scratch stream is the per-32-lane activation reduction needed by
   Q4NX zero-point correction; it is produced in the phase body and consumed
   exactly once per quant group by the microkernel.
9. The fast math form is:

   ```text
   acc[row] += sum_d (q4[row,d] * scale[row,group]) * activation[d]
   acc[row] += offset[row,group] * sum_d activation[d]
   ```

   This explains both the eight `p3` scratch loads and the 16 correction MACs
   per 32-row chunk. The earlier IRON mental model did the offset add per
   input dimension; MyLM hoists that term out of the 32-lane group. This is a
   performance clue, not yet IRON's accepted numerical contract.
10. A Peano C++ source form that matches the important instruction contract now
    exists in `intrinsic_probe.cc`: the group-sum correction probes generate
    signed `#0x33c`, `vextbcst.16`, correction `vbcst.16`, and low-spill
    unrolled code. That does not prove production equivalence yet, but it
    removes the previous uncertainty about whether Peano can express this
    shape at all.
11. The full 32-lane Peano probe now matches MyLM's per-group count:

    ```text
    probe_native_q4_group_sum_correction_unroll32_signed:
      vmac.f=33
      vextbcst.16=32
      vst=1   // final output store only
    ```

    Since MyLM's static hot loop has `264 vmac.f`, this probe is exactly
    `1/8` of one static output-lane pass. The remaining production work is
    therefore not "can Peano spell the loop"; it is fitting eight such groups
    and the two output lanes into the real `q4nx_chunk_accum_fast` ABI without
    reintroducing spills or changing the accepted numerical contract.
12. The full 8-group Peano lane probe now reaches the same static MAC count as
    MyLM:

    ```text
    probe_native_q4_group_sum_correction_chunk_lane_kernel_signed:
      vmac.f=264
      vextbcst.16=256
      vst=5
    ```

    This is close enough to be a code-shape candidate, but it is not production
    equivalent. It still emits `vunpack=256`, `vups=128`, and
    `vconv.bf16.fp32=384`, versus MyLM's `vunpack=64`, `vups=64`, and
    `vconv.bf16.fp32=136`. Directly duplicating both lanes in C++ emits
    `vst=824`; wrapping the full body in a simple two-iteration loop emits
    `vst=251`. Therefore the viable Peano path is a noinline lane body reused
    by calls, unless a lower-level generator can reproduce MyLM's `lc=2`
    scheduled loop without spills.
13. The previously vague "MyLM does something with group sums" is now resolved:
    the phase body loads eight activation vectors, reduces them to eight
    `s16` scratch values, and passes that scratch base in the `jl #0x1f0`
    branch-slot setup as `p3`. The Q4 loop consumes exactly eight `lda.s16`
    values and broadcasts each once for the offset correction MAC.

## Remaining Kernel-Level Unknowns

The key performance mechanism is no longer unknown at the "which instruction"
level, and the `p3` scratch mechanism is no longer unknown. The remaining
blockers before cloning the kernel shape in IRON are now more specific:

1. Exact numeric meaning of the remaining control constants and control bytes:
   `r5=0x3c`, `crsat=*(0x73c64)`, `crrnd=*(0x73c62)`,
   `0x73c60/0x73c61/0x73c62/0x73c64/0x73c80/0x73d00`.
2. The signed native MAC plus group-sum scratch is not a drop-in replacement
   for the current production reference. An active transplant failed
   `qwen3-8b-qkv-compact-output` with `payload_max_abs=0.112304688`, because
   the current reference rounds `bf16(q * scale + offset)` per input dimension.
   Any MyLM-style group-sum route must either preserve that rounding by other
   means or deliberately update the layer reference and prove end-to-end decode
   quality.
3. How to make Peano produce MyLM's dequant schedule (`64 vunpack`,
   `64 vups.4x`, `136 vconv`) instead of the current native lane candidate's
   heavier conversion path. This is now a compiler/source-shape problem, not a
   dataflow mystery.
4. Whether the 5.6KB microkernel was authored by handwritten assembly, an
   internal generator, AIEVec/XLLVM, or a lower-level post-link scheduler. The
   shipped MyLM tree gives only the raw result, not the source. This matters
   for maintainability, but it is no longer required to understand the kernel
   algorithm.

The next useful step is now a production-controlled rewrite of the IRON Q4NX
inner loop as source assembly with exact current Q4NX rounding semantics. It
should keep activation vectors live as 32-lane registers to use
`vextbcst.16`, but it cannot rely on the group-sum shortcut unless the
reference contract is intentionally changed.
