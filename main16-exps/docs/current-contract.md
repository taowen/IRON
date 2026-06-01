# Current MyLM Main16 Contract

This is the current observed contract for the MyLM Qwen3-8B c2r2 main16 raw
program. It is based on the existing `128..133` experiments.

## Raw Program Packaging

- MyLM c2r2 raw program can be wrapped as an AIE2P `ET_EXEC` whole-core ELF.
- MLIR-AIE can load it through `aie.core(%tile) { aie.end } { elf_file = ... }`.
- Static segments currently loaded with the program:
  - `.text` at local `0x00000`;
  - `.mylm_static_73c80` at local `0x73c80`, 128 bytes;
  - `.mylm_static_73d00` at local `0x73d00`, 32 bytes.

## Observable Harness ABI

The standalone c2r2 harness uses:

```text
shim2 MM2S0 -> c2r2 DMA0 activation ring
shim2 MM2S1 -> c2r2 DMA1 weight ring
c2r2 MM2S1 -> shim3 S2MM1 compact record output
```

Main16 buffer/lock ABI in the harness:

```text
activation ping/pong: 0x8000 / 0xc000, len 128 dword
weight ping/pong:     0x2800 / 0x4000, len 1280 dword
record ping/pong:     0x3c1c / 0x541c, len 17 dword
locks:
  activation empty/full: core locks 48 / 49
  weight empty/full:     core locks 50 / 51
  record empty/full:     core locks 52 / 53
  start gate:            core lock 54
```

## Program Layout

Observed/summarized MyLM c2r2 layout:

```text
0x0000  entry/setup
0x01f0  shared Q4NX microkernel
0x1870  Q/K/V body
0x1e80  O body
0x2490  up/gate body
0x2aa0  down body
0x30c0  alternate body
0x36d0  dispatcher
0x38d0  tail/helper
```

## Dispatcher Control

The first useful dispatcher control point is not `0x73c80[0]` or
`0x73d00[0]`.

Experiment `132_mylm_main16_dispatcher_stub_probe` shows:

```text
caller [sp - 4] = 0x78200
local  [0x78200] = control value
call   0x36d0
```

Observed behavior:

```text
control = 0 -> first record header 0x4
control = 1 -> first record header 0x1
control = 8 -> first record header 0x4
```

Interpretation:

- this control point is a QKV-vs-alternate gate;
- value `1` enters the normal Q/K/V path;
- non-`1` values take an alternate path that emits header `0x4`;
- it is not a general phase-id selector.

## Q/K/V Record Boundary

Experiment `133_mylm_main16_qkv_record_count_probe` shows:

```text
control = 1, wait 12 records:
  12 x header 0x1

control = 1, wait 13 records:
  12 x header 0x1
   1 x header 0x4
```

Interpretation:

- MyLM Q/K/V prefix emits 12 compact records with header `0x1`;
- dispatcher then continues into the next O-like phase with header `0x4`;
- `control=1` enters a dispatcher sequence, not a single isolated QKV call.

## Full Dispatcher Header Sequence

Experiment `main16-exps/001_dispatcher_header_sequence` shows:

```text
20 records,  320 stream chunks:
  12 x header 0x1
   8 x header 0x4

68 records, 1088 stream chunks:
  12 x header 0x1
   8 x header 0x4
  48 x header 0x8

76 records, 1472 stream chunks:
  12 x header 0x1
   8 x header 0x4
  48 x header 0x8
   8 x header 0x4
```

Interpretation:

- main16 dispatcher order is Q/K/V -> O -> up/gate -> down;
- headers are phase-level MyLM headers, not IRON packet ids;
- Q/K/V, O, and up/gate consume 16 stream chunks per record;
- down consumes 48 stream chunks per record;
- a naive `records * 16` input stream underfeeds the full 76-record sequence.

Experiment `main16-exps/002_stream_consumption_boundary` confirms the
non-timeout stream boundaries:

```text
Q/K/V exact:    12 records,  192 chunks -> record_observed
O exact:        20 records,  320 chunks -> record_observed
up/gate exact:  68 records, 1088 chunks -> record_observed
down exact:     76 records, 1472 chunks -> record_observed
```

Timeout-based underfeed tests are not cheap integration tests. A deliberate
underfeed can leave the raw-core runtime session unsuitable for immediate
follow-up runs even when `xrt-smi examine` still reports topology `6x8`; use
those only as isolated destructive probes. Recover the pinned driver before
continuing:

```bash
sudo systemctl restart amdxdna-pinned.service
```

## Direct Phase Body Entry

Experiment `main16-exps/003_phase_body_direct_entry` shows that the normal phase
bodies can be entered directly without calling dispatcher `0x36d0`, at least for
the header/count/stream contract:

```text
0x1870 Q/K/V:   12 records, 192 chunks -> 12 x header 0x1
0x1e80 O:        8 records, 128 chunks ->  8 x header 0x4
0x2490 up/gate: 48 records, 768 chunks -> 48 x header 0x8
0x2aa0 down:     8 records, 384 chunks ->  8 x header 0x4
```

The direct-entry stub uses the copied dispatcher-style record-buffer profile:

```text
p0 = 0x78000
p1 = 0x78200
p2 = 0x72800
p3 = 0x75400
p4 = 0x73c00
p5 = 0x75400
p6 = 0x73c00
p7 = 0x75400
r0 = phase header
```

Important boundary:

- this resolves phase body entry, phase-local record count, header, and exact
  stream chunk count;
- non-zero Q/K/V payload observability for this direct-entry profile is covered
  by experiment `main16-exps/005_nonzero_payload_probe`.

## Q4NX Hot-Body Shape

Experiment `main16-exps/004_q4nx_hot_body_schedule` summarizes the MyLM Q4NX
hot loop from the local whole-core ELF:

```text
hot loop: 0x260..0x1850
lc:       2
static:
  vmac.f           264
  vextbcst.16      256
  vups.4x           64
  vunpack           64
  vconv.bf16.fp32  136
  lda.s16            8
  vbcst.16           8
  vst                0
dynamic:
  vmac.f           528
  vextbcst.16      512
```

The schedule is organized as eight 32-lane activation groups. Each group covers
all `x11` lanes through `vextbcst.16`; groups 1..6 carry 33 static `vmac.f`,
while group 0 has 28 and group 7 has 38, proving the body is software-pipelined
across group boundaries.

The Q/K/V phase body produces exactly eight scalar group sums before calling the
Q4 microkernel:

```text
activation vector loads: 8
scalar extracts:         8
scratch stores to p2:    8
```

Those values are consumed by the microkernel through `lda.s16 r7, [p3], #0x2`
and rewound once at `0x153a`. Matching MyLM performance therefore requires the
same cross-group register lifetime plan and group-sum `p3` contract, not just
using `vextbcst.16` in isolation.

## Non-Zero Q/K/V Payload Observability

Experiment `main16-exps/005_nonzero_payload_probe` keeps the direct `0x1870`
Q/K/V phase-body entry and feeds deterministic host activation/weight streams
through the same DMA0/DMA1 rings:

```text
zero activation, zero weight:
  payload nonzero words = 0

bf16-one activation, zero weight:
  payload nonzero words = 0

bf16-one activation, Q4NX scale=0x3c80, nibble=1:
  payload nonzero words = 192
  payload word = 0x42004200

bf16-one activation, Q4NX scale=0x3c80, nibble=2:
  payload nonzero words = 192
  payload word = 0x42804280
```

Interpretation:

- direct phase-body entry is numerically connected to the host-fed activation
  and Q4NX weight streams;
- headers/counts are no longer the only observable signal;
- the simple synthetic input is proportional in the Q4 nibble value, which is a
  useful anchor for the next MyLM-style Q4NX reference;
- this still does not prove production Qwen3 parity, because real model chunks
  need per-row/per-group scale, zero, packed nibble, and record-layout mapping.

## Synthetic Q4NX Payload Formula

Experiments `main16-exps/008_q4nx_payload_formula_probe` through
`main16-exps/011_q4nx_chunk_pair_matrix` refine the simple payload formula.

Uniform synthetic inputs match the scalar dot-product expectation:

```text
activation = 1.0, scale = 1/64, nibble = 1 -> payload bf16 32, word 0x42004200
activation = 1.0, scale = 1/64, nibble = 2 -> payload bf16 64, word 0x42804280
activation = 1.0, scale = 1/128,nibble = 1 -> payload bf16 16, word 0x41804180
activation = 1.0, scale = 1/32, nibble = 1 -> payload bf16 64, word 0x42804280
activation = 2.0, scale = 1/64, nibble = 1 -> payload bf16 64, word 0x42804280
```

The first-record-only case also matches record-major ordering:

```text
active chunks 0..15 -> record0 payload 0x42004200, records 1..11 zero
```

The naive one-active-chunk formula was wrong. The direct Q/K/V harness shows an
even paired-chunk schedule:

```text
chunk 0,2,4,...,14 -> bf16 4, word 0x40804080
chunk 1,3,5,...,15 -> zero
sum of single chunks -> 32
all record0 chunks  -> 32
```

Separating activation and weight axes gives the same even-only pattern on both
inputs. A small pair matrix confirms matching-pair behavior:

```text
activation0 * weight0 -> 4
activation0 * weight2 -> 0
activation2 * weight0 -> 0
activation2 * weight2 -> 4
odd-index pairs       -> 0
```

Interpretation for this direct harness:

- each record has 16 stream chunks at the DMA/lock level;
- Q4NX arithmetic observes 8 even indexed activation/weight chunk pairs;
- each observed even pair contributes 256 bf16 products for this uniform input;
- the aggregate still equals 2048 products per record;
- this likely reflects the raw program's paired ping-side chunk schedule or
  logical chunk packing, so host-side `chunk_index` should not be treated as one
  uniform mathematical tile.

This is now a scalar anchor for codegen/reference work. It still does not decode
the production Q4NX packed lane order, scale/zero layout, or non-uniform lane
mapping.

## Q4NX Field Layout Probe

Experiment `main16-exps/012_q4nx_field_layout_probe` fixes one active paired
chunk and varies fields inside the active weight chunk:

```text
activation chunk 0 active
weight chunk 0 active
activation = bf16 1.0
scale      = bf16 1/64
q4 word    = 0x11111111
zero       = 0
```

The baseline with all scale slots and q4 data words active produces bf16 `4` in
all 16 payload lanes, matching the chunk0 contribution from experiments
009..011.

Observed scale-slot mapping:

```text
scale[0]   -> lane 0, value 0.5
scale[1]   -> lane 1, value 0.5
...
scale[8]   -> lane 8, value 0.5
scale[16]  -> lane 0, value 0.5
scale[32]  -> lane 0, value 0.5
scale[64]  -> lane 0, value 0.5
scale[96]  -> lane 0, value 0.5
scale[127] -> lane 15, value 0.5
```

So, for this synthetic setup, scale dword `i` maps to payload lane `i mod 16`.

Observed q4 data word mapping:

```text
q4word even in low region  -> lanes 0..3,  value 1/64 each
q4word odd  in low region  -> lanes 4..7,  value 1/64 each
q4word 512, 768, 1022      -> lanes 8..11, value 1/64 each
q4word 513, 769, 1023      -> lanes 12..15,value 1/64 each
```

The first 32 q4 data dwords and sparse probes through index 385 only affect
lanes 0..7. Sparse probes at 512 and above affect lanes 8..15. This gives a
first production-layout anchor: the q4 data region is split into low-output and
high-output halves, with even/odd q4 words selecting the low/high lane quartet
inside each half.

Still unknown:

- how these field mappings line up with the half-register cell trace.

Experiment `main16-exps/013_q4nx_layout_boundary_nibble_zero_probe` resolves the
first production-layout details left open by experiment 012:

```text
q4 data word 0..511:
  even word -> payload lanes 0..3
  odd word  -> payload lanes 4..7

q4 data word 512..1023:
  even word -> payload lanes 8..11
  odd word  -> payload lanes 12..15
```

The direct harness therefore has an exact q4 low/high output-half split at q4
data word index `512`.

Single-nibble probes show that a payload dword must be treated as two bf16
halves, not as one scalar lane. The earlier `nonzero_lanes` report only looked
at the high 16 bits. The full dword mapping for the active chunk0 pair is:

```text
q4word 0:
  nibble 0 -> payload word 0 low  bf16
  nibble 1 -> payload word 0 high bf16
  nibble 2 -> payload word 1 low  bf16
  nibble 3 -> payload word 1 high bf16
  nibble 4 -> payload word 2 low  bf16
  nibble 5 -> payload word 2 high bf16
  nibble 6 -> payload word 3 low  bf16
  nibble 7 -> payload word 3 high bf16

q4word 1:
  same low/high-half pattern for payload words 4..7

q4word 512:
  same low/high-half pattern for payload words 8..11

q4word 513:
  same low/high-half pattern for payload words 12..15
```

The middle 128 dwords are also active. With all q4 data words set to
`0x11111111`, all scale slots active, and one zero/offset slot set to bf16
`1/64`, the corresponding lane increases by `0.5`:

```text
zero[0]   -> lane 0 +0.5
zero[1]   -> lane 1 +0.5
...
zero[8]   -> lane 8 +0.5
zero[16]  -> lane 0 +0.5
zero[32]  -> lane 0 +0.5
zero[64]  -> lane 0 +0.5
zero[96]  -> lane 0 +0.5
zero[127] -> lane 15 +0.5
```

So scale and zero/offset slots both map as `index mod 16` in this synthetic
setup. The sign is positive for this raw MyLM payload contract; do not assume
the field is a conventional subtractive zero-point without matching the exact
MyLM formula.

Still unknown:

- how these field mappings line up with the half-register cell trace;
- exact parity mapping for phase records beyond record0.

Experiment `main16-exps/014_q4nx_stream_parity_probe` checks the chunk0 nibble
mapping across all 16 stream chunks in record0:

```text
active chunk 0,2,4,...,14:
  q4word 0 with 0x11111111 -> payload words 0..3, value 1/64 in both bf16 halves

active chunk 1,3,5,...,15:
  q4word 0 with 0x11111111 -> no observed payload contribution
```

For every contributing even chunk, single-nibble behavior is identical at the
payload-word-half level:

```text
nibble 0 -> word 0 low  bf16
nibble 1 -> word 0 high bf16
nibble 2 -> word 1 low  bf16
nibble 3 -> word 1 high bf16
nibble 4 -> word 2 low  bf16
nibble 5 -> word 2 high bf16
nibble 6 -> word 3 low  bf16
nibble 7 -> word 3 high bf16
```

Odd chunk nibble probes for chunks 1 and 3 also produce no observed payload.
Within record0, this confirms the raw MyLM direct path uses the same paired
even-chunk schedule observed in experiments 009..011.

Experiment `main16-exps/015_q4nx_tiny_codegen_numeric_gate` turns this corrected
word-half formula into a generated whole-core source-assembly record emitter and
compares it against the real MyLM raw body. It passes four representative
cases:

```text
q4word0_allnibbles
q4word512_allnibbles
q4word0_nibble3
zero0_allq4
```

For each case:

```text
formula record == MyLM direct 0x1870 record
generated tiny source-asm record == formula record
generated tiny source-asm record == MyLM direct 0x1870 record
```

This is still an emit-only generated body, not the final dynamic Q4NX hot loop,
but it proves the source-assembly/codegen packaging, lock consumption, record
release, and corrected payload formula in the same isolated NPU harness.

## Source-Assembly Codegen Rules

Experiments `main16-exps/016_q4nx_dynamic_tiny_arithmetic_gate` through
`main16-exps/019_source_asm_branch_semantics_probe` clarify the rules for
writing generated source assembly in this harness.

Experiment 016 is a useful negative result. It tried to replace experiment 015's
generated constants with dynamic q4/scale/zero loads and scalar branch cascades,
but the resulting program timed out. The failure is not proof that dynamic Q4NX
is impossible; it shows the generator was missing an explicit source-asm
scheduling contract.

Experiment 017 isolates buffer visibility. After acquiring the normal full locks
for the first activation/weight chunk, source assembly can read the DMA-written
buffers through full local addresses:

```text
weight scale0:     [0x72800 + 0x000] -> 0x3c803c80
weight zero0:      [0x72800 + 0x200] -> 0x3c803c80
weight q4word0:   [0x72800 + 0x400] -> 0x11111111
weight q4word512: [0x72800 + 0xc00] -> 0x11111111
activation word0: [0x78000 + 0x000] -> 0x3f803f80
record header:    [0x73c1c + 0x000] -> 0x1
```

The corresponding short addresses such as `0x2800`, `0x8000`, and `0x3c1c`
do not read the DMA buffers correctly in this whole-core source-asm path.

Experiment 018 measures producer/consumer gaps on real NPU. The current
conservative minimum gaps are:

```text
lda -> st:      6 nop
lda -> eq+jz:   6 nop before eq
eq -> jz:       0 nop in the tested path
and -> eq:      0 nop in the tested path
lshl -> or:     0 nop in the tested path
```

These values are not performance guidance. A MyLM-style generator should fill
those gaps with independent useful work. They are correctness lower bounds for
source assembly emitted without Peano scheduling.

Experiment 019 clarifies scalar condition semantics:

```text
eq(a, b): equal -> 1, not equal -> 0
jz after compare: branch when the compare result is 0
```

Do not generate direct `mova rX, #0/1; jz/jnz rX, label` control flow as if the
branch were a normal C `if`. In this harness the reliable primitive is
compare-produced predicate plus `jz`. The current codegen direction should
therefore avoid `jnz` and prefer either `jz`-only control flow or branchless
`sel.eqz`/`sel.nez` where possible.

## Q4NX Alias Lifetime Graph

Experiment `main16-exps/006_q4nx_alias_lifetime_graph` turns the MyLM Q4NX hot
loop into an alias-aware def/use table. The goal is not to rediscover register
names; AM027 already explains the hardware vocabulary:

```text
wlN/whN are the 256-bit halves of xN
bmll/bmlh/bmhl/bmhh, cml/cmh, and dm are accumulator aliases
```

The useful result is the steady-state cross-group live-through signature. For
every activation-group boundary `g0->g1` through `g6->g7`, the same families are
live across the boundary:

```text
acc1, acc4,
vec0, vec10, vec2, vec3, vec4, vec8, vec9
```

Representative first boundary edges:

```text
acc1  0x51e vmac.f        -> 0x52a vmac.f
acc4  0x478 vmov bmll4    -> 0x570 vmac.f dm4
vec0  0x51e vldb x0       -> 0x53a vmac.f
vec10 0x51a vunpack x10   -> 0x52a vmac.f
vec8  0x516 vunpack x8    -> 0x53a vups.4x
```

Semantic counts from the same pass:

```text
activation_load          8
group_sum_load           8
q4_unpack               64
ups_vector_to_accum     64
dequant_add             64
dequant_sub             64
activation_broadcast   256
mac                    264
accum_to_bf16_vector   136
hot-loop vst             0
```

Interpretation:

- MyLM's hot loop is a regular software pipeline after the first group;
- the cross-group state is small and concrete, not an unbounded opaque register
  mess;
- a source-assembly/codegen replacement must reproduce this live-through family
  contract before it can reproduce MyLM's MAC density;
- the current table is still family-level, so partial alias views such as
  `cml/cmh/bmll/bmlh` need lane/view-level decoding before generating a correct
  replacement body.

## Half-Register Boundary Trace

Experiment `main16-exps/007_half_register_boundary_trace` refines the group
boundary to cell granularity:

```text
xN        -> vecN.lo, vecN.hi
wlN/whN   -> one vector half
dmN       -> accN.bmll/bmlh/bmhl/bmhh
cmlN/cmhN -> low/high accumulator halves
bm*       -> one accumulator quadrant
```

For the first steady-state group (`group1`), the experiment observes:

```text
boundary cells into group1:        23
group1 vmac.f count:              33
mixed vector operands:            25
cross-group vector operands:       9
cross-group accumulator operands:  2
```

Key boundary cells into group1:

```text
acc1.*    <- g0 0x51e vmac.f
acc4.bmll <- g0 0x478 vmov
acc4.bmlh/bmhl/bmhh <- g0 0x41c vsub.f
vec9.*    <- g0 0x50e vunpack
vec10.*   <- g0 0x51a vunpack
vec8.*    <- g0 0x516 vunpack, first consumed by g1 0x53a vups.4x
vec4.*    <- g0 0x4ba vconv.bf16.fp32
vec2.hi   <- g0 0x4b2 vextbcst.16 lane 0x1a
vec2.lo   <- g0 0x4dc vldb
```

This explains why a source-assembly replacement cannot treat `xN` as a single
SSA-like vector. Some operands are deliberately mixed from halves produced by
different instructions or even different activation groups. Example from group1:

```text
0x570 vmac.f:
  dm4 uses acc4.bmll from g0 vmov and other acc4 quadrants from g0 vsub.f
  x4  uses low half from g1 vconv and high half from g0 vconv
  x2  uses low half from g0 vldb and high half from g0 activation lane 0x1a
```

The MyLM-style generator target is therefore a cell-state machine across
pipeline fill, steady groups, and drain, not a per-group whole-vector loop.

## Current Unknowns

- mapping from MyLM headers `0x1/0x4/0x8` to IRON packetized record routing;
- mapping from the half-register cell trace to the non-uniform production Q4NX
  payload formula;
- production stream parity mapping beyond record0;
- how to bridge MyLM raw main16 into the full IRON layer dataflow.
