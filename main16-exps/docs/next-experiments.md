# Next Experiments

The next experiments should live under `main16-exps/`, not under the old
top-level numbered sequence.

## 1. Full Dispatcher Header Sequence

Status: done in `main16-exps/001_dispatcher_header_sequence`.

Goal:

```text
control=1, wait 20 records -> expect 12 x 0x1 + 8 x 0x4
control=1, wait 68 records -> expect 12 x 0x1 + 8 x 0x4 + 48 x 0x8
control=1, wait 76 records -> expect 12 x 0x1 + 8 x 0x4 + 48 x 0x8 + 8 x 0x4
```

Reason:

- confirms the dispatcher phase sequence and phase record counts;
- proves the main16 phase-level control path before adding non-zero payloads.

## 2. Stream-Consumption Boundary Checks

Status: exact non-timeout boundaries done in
`main16-exps/002_stream_consumption_boundary`.

Goal:

- underfeed each phase by one stream chunk and confirm timeout;
- feed exact stream chunks and confirm record completion;
- overfeed by one phase boundary and confirm the next header appears.

Reason:

- this turns the observed phase sequence into a lock/stream consumption
  contract;
- it explains timeout causes before full-layer integration.

Note:

- timeout-based underfeed probes are destructive enough to make immediate
  follow-up runs unreliable; run them separately and restart the pinned driver
  before continuing:

```bash
sudo systemctl restart amdxdna-pinned.service
```

## 3. Phase-Isolation Stub

Status: done in `main16-exps/003_phase_body_direct_entry`.

Goal:

- call `0x1870`, `0x1e80`, `0x2490`, and `0x2aa0` directly with copied caller
  state;
- observe whether direct phase-body entry emits the expected header/count.

Reason:

- isolates O/upgate/down without relying on the dispatcher sequence;
- identifies which state must be initialized by the dispatcher before each body.

Result:

- Q/K/V, O, up/gate, and down direct-entry all pass for full phase-local
  header/count/stream contracts.
- The alternate `0x30c0` body is not needed for the normal Qwen3 layer path.

## 4. Q4NX Hot-Body Schedule Summary

Status: done in `main16-exps/004_q4nx_hot_body_schedule`.

Goal:

- make MyLM Q4NX hot-loop counts and first-group register flow reproducible from
  the local whole-core ELF.

Reason:

- separates real MyLM schedule facts from old loose descriptions like "uses
  vextbcst.16";
- gives the source-assembly/codegen work a concrete register-lifetime target.

## 5. Non-Zero Payload Probe

Status: done in `main16-exps/005_nonzero_payload_probe`.

Goal:

- feed deterministic non-zero activation and weight chunks;
- collect record payloads for the QKV prefix;
- compare with a local MyLM-style Q4NX reference once the layout is decoded.

Reason:

- header/count correctness is only control-flow proof;
- performance replacement requires numeric payload semantics.

Result:

- direct `0x1870` Q/K/V entry emits 12 x `0x1` headers for all scenarios;
- zero and activation-only inputs produce all-zero payload;
- bf16-one activation plus synthetic Q4NX nibble 1 produces payload
  `0x42004200`;
- bf16-one activation plus synthetic Q4NX nibble 2 produces payload
  `0x42804280`;
- the MyLM numeric path is observable under the IRON harness, but production
  chunk/reference mapping remains to be decoded.

## 6. Q4NX Alias Lifetime Graph

Status: done in `main16-exps/006_q4nx_alias_lifetime_graph`.

Goal:

- derive an alias-aware def/use table for the MyLM hot loop `0x260..0x1850`;
- summarize vector and accumulator families live across activation-group
  boundaries;
- identify the minimal steady-state cross-group state a MyLM-style source
  assembly/codegen body must preserve.

Reason:

- knowing `wl/wh/x/dm/cml/cmh` vocabulary is not enough;
- MyLM's performance depends on a cross-group software pipeline;
- the next implementation attempt needs a register-family schedule, not only
  opcode counts.

Result:

- every boundary `g0->g1` through `g6->g7` has the same live-through families:
  `acc1, acc4, vec0, vec10, vec2, vec3, vec4, vec8, vec9`;
- semantic counts match the expected MyLM hot-loop shape:
  64 `vunpack`, 64 `vups.4x`, 256 `vextbcst.16`, 264 `vmac.f`, and no hot-loop
  `vst`;
- the result is still family-level, so partial accumulator/vector views need
  lane-level decoding next.

## 7. Half-Register Boundary Trace

Status: done in `main16-exps/007_half_register_boundary_trace`.

Goal:

- split the experiment 006 live-through families into vector halves and
  accumulator quadrants;
- trace group1 `vmac.f` operands back to cell-level producers;
- identify which operands are mixed across instructions or across group0->group1.

Reason:

- MyLM does not treat `xN` or `dmN` as indivisible SSA values;
- a source-assembly/codegen replacement needs to preserve the half-register
  software pipeline, not just the family-level live-through set.

Result:

- group1 has 33 `vmac.f`;
- 25 vector operands are mixed-half;
- 9 vector operands and 2 accumulator operands consume cells from group0;
- `0x570 vmac.f` is the clearest example of mixed cross-group state:
  accumulator quadrants, `x4` halves, and `x2` halves come from different
  producers.

## 8. MyLM Q4NX Payload Formula

Status: scalar synthetic anchor done across
`main16-exps/008_q4nx_payload_formula_probe`,
`main16-exps/009_q4nx_chunk_contribution_map`,
`main16-exps/010_q4nx_activation_weight_axis_map`, and
`main16-exps/011_q4nx_chunk_pair_matrix`.

Goal:

- generate a small set of synthetic Q4NX chunks with known scale, zero, nibble,
  and activation patterns;
- compare the 12-record payloads against candidate Q4NX formulas;
- identify the exact row/group/record layout and rounding contract used by the
  raw `0x01f0` microkernel.

Reason:

- experiment 005 proves the payload changes correctly, but not which formula and
  layout produce those words;
- experiments 006 and 007 identify the live-through register families and
  half-register cells, but not yet the payload formula;
- without this reference, a new source-assembly/codegen kernel cannot be judged
  numerically against MyLM.

Result:

- uniform full-record synthetic inputs match scale/nibble/activation linearity;
- direct-harness record0 uses even activation/weight chunk pairs:
  chunks `0,2,...,14` contribute bf16 `4`, odd chunks contribute zero;
- matching even/even pairs contribute, cross pairs do not;
- aggregate record0 value remains bf16 `32`, so the scalar dot length is 2048
  products, but the host-visible chunk schedule is paired.

## 9. Production Q4NX Layout Probe

Status: first field-layout anchor done in
`main16-exps/012_q4nx_field_layout_probe`, then expanded in
`main16-exps/013_q4nx_layout_boundary_nibble_zero_probe`.

Goal:

- move beyond uniform nibble/scale cases;
- vary one scale slot, one zero slot, and one q4 data lane at a time;
- map those host Q4NX fields to compact payload lanes and to the
  half-register cell trace from experiments 006/007.

Reason:

- experiments 008..011 give the scalar dot anchor, but uniform data hides lane
  order and scale/zero placement;
- a MyLM-style code generator needs the production packed layout, not only the
  aggregate scalar formula.

Result:

- scale dword `i` maps to payload lane `i mod 16` under the current synthetic
  setup;
- q4 data low region affects lanes `0..7`;
- q4 data sparse probes at `512+` affect lanes `8..15`;
- even/odd q4 words select lane quartets inside each half.

Additional result from experiments 013 and 014:

- q4 data word `0..511` maps to low payload lanes and `512..1023` maps to high
  payload lanes;
- even q4 words select the first quartet in a half, odd q4 words select the
  second quartet;
- q4 nibble positions `0/2/4/6` map to the low bf16 half of payload words, and
  `1/3/5/7` map to the high bf16 half;
- zero/offset slots map as `index mod 16` and are numerically active.
- across record0 stream chunks, even active chunks `0,2,...,14` contribute and
  odd active chunks do not;
- the q4word0 word-half nibble mapping is identical for all contributing even
  chunks.

Remaining:

- connect field mappings to the half-register cell trace from experiments
  006/007.
- resolve record-index parity beyond record0.

## 10. Source-Assembly Schedule Prototype

Goal:

- use the field-layout anchors from experiments 008..012 to write a tiny
  MyLM-style generated body for one active record/chunk pair;
- preserve explicit half-register cell state instead of whole-vector SSA values;
- compare synthetic payload against the direct MyLM body before touching the
  production qwen3-layer kernel.

Reason:

- the layout and schedule questions can be tested without full decode;
- replacing qwen3 decode should wait until a small generated schedule is
  numerically aligned with MyLM on isolated probes.

Status: emit-only gate done in
`main16-exps/015_q4nx_tiny_codegen_numeric_gate`.

Result:

- four representative synthetic cases pass exact record comparison against
  MyLM direct `0x1870`;
- the first generated whole-core source-assembly body consumes 16 input stream
  chunks, releases the compact record lock, and emits a record equal to MyLM;
- this proves codegen packaging and the corrected payload formula, but not yet
  dynamic Q4NX arithmetic.

Remaining:

- replace generated constants with dynamic loads of scale/q4/zero using the
  measured source-asm latency/branch contract from experiments 017..019.

## 11. Dynamic Source-Assembly Arithmetic

Status:

- positive buffer visibility gate done in
  `main16-exps/017_q4nx_dynamic_readback_probe`;
- latency model done in `main16-exps/018_source_asm_latency_model`;
- branch semantics done in `main16-exps/019_source_asm_branch_semantics_probe`;
- first all-in-one dynamic arithmetic attempt in
  `main16-exps/016_q4nx_dynamic_tiny_arithmetic_gate` still times out and
  should not be extended in its current branch-cascade shape.

Key facts:

```text
full weight base:     0x72800
full activation base: 0x78000
full record base:     0x73c1c
lda consumer gap:     6 nop minimum in isolated probes
eq result:            equal -> 1, not equal -> 0
jz after compare:     branch when compare result is 0
```

Next goal:

- write a smaller dynamic arithmetic gate that computes only one payload word
  first;
- use `jz`-only control flow or branchless `sel.eqz`/`sel.nez`;
- add a generator hazard check that rejects `lda` consumers closer than the
  measured gap;
- only after the one-word gate passes, expand to 16 payload words and then to
  multi-word accumulation.

## 12. Production Stream Parity Probe

Goal:

- repeat the q4word/nibble probes for active chunk pairs `2/2`, `4/4`, and one
  later record;
- determine whether low nibble positions `0/2/4/6` are unused only for chunk0 or
  belong to a different logical parity path;
- turn the synthetic layout contract into a production stream parity table.

Reason:

- experiment 013 gives an exact field layout for the active chunk0 pair;
- a replacement production kernel needs the full parity mapping across the
  16 DMA chunks per record, not only the first paired chunk.

Status: record0 chunk parity done in
`main16-exps/014_q4nx_stream_parity_probe`.

Remaining:

- repeat the compact probes for record1/record2 to check whether the parity
  table is record-invariant.

## 13. Peano Compiler Route

Status: first compiler gate done in
`main16-exps/020_peano_q4nx_compiler_route`.

Goal:

- stop trying to hand-schedule the full MyLM Q4NX hot body directly;
- use Peano/llvm-aie as the scheduler/register allocator;
- generate a Q4NX-specific source/IR shape and score the resulting assembly
  against the MyLM static target.

Result:

- Peano can emit the signed MAC primitive:
  `#0x33c + vextbcst.16 + vmac.f`;
- the group-sum candidate reaches the macro counts
  `vmac.f=264`, `vextbcst.16=256`, `vextbcst.32=0`;
- the current C++ intrinsic formulation is still too high level:
  it produces `vups.2x`, `vmul.f=256`, and `vconv.bf16.fp32=384`, while MyLM's
  static target is `vups.4x=64`, `vmul.f=8`, and `vconv.bf16.fp32=136`;
- the loop candidate exposes hardware-loop registers but the postpipeliner
  reports no schedule, so this is not yet a production kernel route.

Next goal:

- keep Peano as the backend, but stop emitting per-dim C++ dequant operations;
- generate a narrower Q4NX DSL/MIR-level body that exposes the MyLM
  `vups.4x -> vadd/vsub/vconv/vmac` data path;
- use the same manifest gate from experiment 020 as the acceptance test before
  touching qwen3-layer.

## 14. LLVM-AIE Backend Surface

Status: done in `main16-exps/021_llvm_aie_backend_surface`.

Goal:

- answer whether we can extract useful pieces from `~/projects/llvm-aie` after
  C++ proved too high level;
- identify the smallest useful surfaces for a MyLM-style Q4NX generator.

Result:

- do not copy LLVM backend implementation code into IRON; it is too coupled to
  LLVM pass state and would not make the generator simpler;
- do use `llvm-aie` as:
  - an external assembler/scheduler/disassembler service;
  - the AIE2P instruction database from `AIE2PGenInstrInfo.td`;
  - the AIE2P pattern hints from `AIE2PInstrPatterns.td`;
  - the AIE2P latency/bypass metadata from `AIE2PGenSchedule.td`;
  - MIR templates from the AIE2P binary and postpipeliner tests.

Next goal:

- write `022_q4nx_mir_schedule_probe`;
- start from AIE2P MIR, not C++;
- build a tiny Q4NX group body using `VUPS_4x`, `VEXTBCST_16`, and `VMAC_f`;
- run Peano `llc`/`llvm-mc`/`llvm-objdump` and gate on instruction shape,
  postpipeliner remarks, bundle count, and absence of C++-induced spills.

## 15. Q4NX MIR Schedule Probe

Status: first gate done in `main16-exps/022_q4nx_mir_schedule_probe`.

Goal:

- test the route selected by experiment 021: machine MIR instead of C++;
- see whether Peano can assemble and schedule AIE2P instructions that match the
  MyLM Q4NX hot-body vocabulary.

Result:

- direct MIR basic block passes and objdump contains:
  `vups.4x=1`, `vextbcst.16=1`, `vmac.f=1`, `vextbcst.32=0`, `vst=0`;
- tiny loop passes postpipeliner with `Schedule found` and loop
  `BundleCount=4`;
- one-group macro-shape MIR reaches the desired static counts:
  `vups.4x=8`, `vextbcst.16=32`, `vmac.f=33`, `vst=0`;
- the one-group naive dependency graph does not get a postpipeliner schedule
  and emits a 70-bundle loop block.

Conclusion:

- the MIR route is real and avoids C++ instruction-selection drift;
- instruction counts alone are not enough;
- the next problem is to generate the MyLM-style cross-group register lifetime
  graph so Peano sees independent work instead of a serialized dependency chain.

Next goal:

- write the one-group MIR from experiment 006/007 cell-lifetime data instead of
  the naive generated order;
- separate fill, steady, and drain groups;
- gate on postpipeliner schedule, BundleCount reduction, and then an isolated
  NPU numeric comparison against MyLM direct `0x1870`.

## 16. Q4NX MIR Lifetime Order Probe

Status: done in `main16-exps/023_q4nx_mir_lifetime_order_probe`.

Goal:

- preserve MyLM group1 order/registers for the three headline hot ops:
  `vups.4x`, `vextbcst.16`, and `vmac.f`;
- compare it against a naive one-group order with the same macro counts.

Result:

- both candidates compile and objdump with exact macro counts:
  `vups.4x=8`, `vextbcst.16=32`, `vmac.f=33`;
- both keep `vextbcst.32=0` and `vst=0`;
- the projected order still does not schedule usefully because the producer
  graph is missing.

Conclusion:

- opcode counts and headline order are not enough;
- the real MyLM schedule depends on the surrounding
  `vlda/vldb/vunpack/vadd/vsub/vconv/vmov` producers and register moves.

## 17. Q4NX MIR Opcode Coverage Map

Status: done in `main16-exps/024_q4nx_mir_opcode_coverage_map`.

Goal:

- map the full MyLM group1 opcode vocabulary to AIE2P machine MIR;
- prove the `llvm-aie` route can express the whole local data path without
  falling back to C++ instruction selection.

Result:

- a representative opcode coverage block compiles through `llc` and
  `llvm-objdump`;
- the full MyLM group1 event order, excluding explicit nops, compiles with exact
  counts:
  `vldb=6`, `vlda=1`, `vunpack=8`, `vups.4x=8`, `vadd=8`, `vsub.f=8`,
  `vconv.bf16.fp32=17`, `vextbcst.16=32`, `vmov=50`, `vmov.d=2`,
  `vmul.f=1`, `vmac.f=33`;
- it still has `vextbcst.32=0` and `vst=0`;
- Peano reports no postpipeliner schedule for the isolated group, with loop
  `BundleCount=120`.

Conclusion:

- the remaining blocker is not missing opcode coverage;
- the next generator must emit a steady-state group pair or triplet so
  cross-group values have real producers/consumers and the scheduler can hide
  latency.

Next goal:

- generate a `group0+group1+group2` MIR window from experiment 006 events;
- keep fill/steady/drain boundaries explicit;
- gate on reduced loop bundle count and then run an isolated NPU numeric
  comparison against MyLM direct `0x1870`.

## 18. Q4NX MIR Group Window Probe

Status: done in `main16-exps/025_q4nx_mir_group_window_probe`.

Goal:

- move from one isolated group to steady-state windows that cross group
  boundaries;
- verify that direct MIR still preserves MyLM's opcode vocabulary and avoids
  C++-induced spills/fallbacks.

Result:

- group window `(1, 2)` compiles with exact doubled counts:
  `vups.4x=16`, `vextbcst.16=64`, `vmac.f=66`, `vmov=100`,
  `vconv.bf16.fp32=34`, `vst=0`;
- group window `(1, 2, 3)` compiles with exact tripled counts:
  `vups.4x=24`, `vextbcst.16=96`, `vmac.f=99`, `vmov=150`,
  `vconv.bf16.fp32=51`, `vst=0`;
- both windows keep `vextbcst.32=0`;
- generated loop byte count closely matches the MyLM source span:
  `1392` bytes vs `1382`, and `2080` bytes vs `2074`.

Conclusion:

- the MIR route can preserve MyLM's object-level instruction shape across
  steady-state group boundaries;
- Peano's postpipeliner still does not synthesize a new schedule for these large
  already-ordered windows, but that is no longer the main question;
- the next valuable test is numeric: package a small generated MIR object into
  the direct-QKV harness and compare payloads with MyLM direct `0x1870`.

Next goal:

- produce a tiny generated MIR/object kernel for one observable synthetic case;
- run it in the existing direct-QKV harness;
- compare compact record payload against MyLM direct body and the formula from
  experiment 015.

## 19. Q4NX MIR Full Hotloop Probe

Status: done in `main16-exps/026_q4nx_mir_full_hotloop_probe`.

Goal:

- extend the MIR projection from group windows to the complete MyLM hot loop;
- include group0/group7 scalar pointer setup and drain instructions;
- prove the direct-MIR route can assemble the full MyLM-shaped body before any
  NPU numeric packaging.

Result:

- all `1422` non-nop events translate and assemble;
- the straight-line and hardware-loop variants both pass;
- the ZOL variant keeps the MyLM headline counts:
  `vups.4x=64`, `vextbcst.16=256`, `vmac.f=264`,
  `vconv.bf16.fp32=136`, `vmul.f=8`, `vst=0`;
- no `vextbcst.32` fallback appears;
- generated loop byte count is `5600` for a MyLM source span of `5610` bytes.

Conclusion:

- the blocker is no longer “llvm-aie cannot express MyLM's hot-loop shape”;
- C++ intrinsic tuning is the wrong path for this part;
- the remaining hard work is converting this object-shape proof into a runnable
  numeric kernel with the same ABI as the direct-QKV harness.

Next goal:

- package the generated MIR object as a tiny direct-QKV kernel;
- feed the same synthetic activation/weight cases used by experiments 015/017;
- compare compact payloads against MyLM direct `0x1870` before touching
  qwen3-layer production code.

## 20. Q4NX MIR Phase ABI Probe

Status: done in `main16-exps/027_q4nx_mir_phase_abi_probe`.

Goal:

- explain the missing ABI boundary between experiment 026's generated hot-loop
  object and a runnable direct-QKV numeric test;
- avoid a false numeric failure from jumping into `0x260..0x1850` without the
  MyLM prologue and phase-body state.

Result:

- `0x01f0` sets `lc=2`, `ls=0x260`, `le=0x1850`, constants, rounding/control
  registers, and rebases `p0` by `0x400`;
- the Q/K/V body writes 8 group-sum halfwords before `jl #0x1f0`;
- the call delay slots map `p0=weight`, `p1=activation`, `p2=0x73c80`, and
  `p3=group-sum stream`;
- the post-call phase body emits compact record payload through
  `vst.conv.bf16.fp32`.

Conclusion:

- experiment 026 is sufficient as an object-shape proof, but not as a runnable
  kernel;
- the next generated runnable boundary must include the `0x01f0` prologue
  semantics and the phase-body group-sum/record-emit logic.

Next goal:

- generate a tiny direct-QKV program with `prologue + hot loop + record emit`;
- use one synthetic case first, with the same activation/weight layout as
  experiments 015/017;
- only after this numeric gate passes should the generated MIR path replace any
  qwen3-layer main16 body.

## 21. Q4NX MIR Hotloop Patch Numeric Gate

Status: negative gate done in `main16-exps/028_q4nx_mir_hotloop_patch_numeric_gate`.

Goal:

- patch the generated full-hotloop MIR object from experiment 026 into MyLM's
  raw program;
- keep MyLM's prologue, Q/K/V body, group-sum producer, DMA/locks, and record
  emitter unchanged;
- run one direct-QKV NPU numeric case.

Result:

- the NPU run reaches `record_observed`; topology remains healthy before and
  after the run;
- the first record header is observed, so this is not a stream/lock/phase-entry
  failure;
- payload is wrong: the first synthetic case emits repeated `0x4b014b01`
  payload words instead of the expected `0x3c803c80`.

Conclusion:

- experiment 026's opcode-count gate was too weak;
- the failure is in hot-loop schedule/register semantics;
- the next experiment must validate MIR pass boundaries and `BUNDLE` semantics
  before more NPU numeric patching.

## 22. Q4NX MIR Prebundled Exact Replay

Status: done in `main16-exps/029_q4nx_mir_prebundled_exact_replay`.

Goal:

- stop treating direct machine MIR as if opcode order alone defined executable
  semantics;
- learn the correct MIR form for hand-scheduled AIE2P code;
- reproduce the MyLM `0x260..0x1850` hot-loop bytes exactly before changing any
  instruction.

Result:

- unpadded pre-bundled MIR compiles to `5600` bytes and first differs at
  `0x2ba`;
- the missing bytes are MyLM address-gap padding, not ordinary op events;
- adding explicit `BUNDLE { NOP }` gaps before `0x2c2`, `0x1828`, and `0x1844`
  produces `5616` bytes;
- the padded pre-bundled MIR is byte-exact against MyLM raw hot-loop bytes.

Conclusion:

- for a known MyLM-style schedule, the correct path is:

```text
pre-bundled MIR -> llc --start-after=postmisched --skip-machine-alignment -> object
```

- the wrong path for numeric replacement is:

```text
unbundled MIR -> llc --start-before=postmisched -> scheduler-chosen order
```

- `BUNDLE` boundaries and address-gap NOPs are part of the kernel contract.

Next goal:

- feed the byte-exact padded object into the experiment 028 numeric harness as a
  no-op replacement; it should reproduce MyLM payloads exactly;
- after that passes, mutate one bundle at a time and keep both byte-diff and NPU
  numeric gates tight;
- only once bundle-local mutations pass should this path be considered for
  qwen3-layer main16 replacement.

## 23. Q4NX MIR Byte-Exact No-Op Numeric Gate

Status: done in `main16-exps/030_q4nx_mir_byte_exact_noop_numeric_gate`.

Goal:

- take the byte-exact padded MIR object from experiment 029;
- patch it into the same raw hot-loop range used by experiment 028;
- prove the packaging/runtime path is a no-op before making any semantic
  change.

Result:

- the generated object hot bytes match MyLM's original `0x260..0x1850` bytes;
- the patched full raw program is byte-identical to the original raw program;
- all four synthetic direct-QKV NPU cases pass:
  `q4word0_allnibbles`, `q4word512_allnibbles`, `q4word0_nibble3`, and
  `zero0_allq4`;
- observed payloads match the existing MyLM direct-body expectations.

Conclusion:

- experiment 028 failed because the unbundled/scheduled MIR did not preserve
  executable schedule semantics;
- experiment 030 proves the pre-bundled MIR route can enter the real NPU
  numeric harness without changing behavior;
- this is now the correct baseline for source-level Q4NX hot-loop mutation.

Next goal:

- create the first controlled byte-diff experiment;
- start with a semantics-preserving bundle mutation or padding-only mutation;
- keep three gates mandatory: expected byte diff, no timeout, and exact NPU
  payload match;
- only after a no-op byte-diff mutation passes should we replace a real dataflow
  bundle and compare against MyLM numerically.

## 24. Q4NX MIR Padding Mutation Numeric Gate

Status: negative gate done in
`main16-exps/031_q4nx_mir_padding_mutation_numeric_gate`.

Goal:

- make one intentional hot-loop byte diff after the byte-exact no-op gate;
- choose a padding-only mutation first, not a real Q4NX dataflow bundle;
- require record observation and exact payload match.

Result:

- patch site: `0x2ba`, the first explicit padding gap before source address
  `0x2c2`;
- mutation: `00 00 00 00 -> f8 a0 df 1f`, disassembling as
  `mov r31, r31`;
- NPU still emits records, so topology/locks/runtime remain healthy;
- payload fails: first words become `0x48014801`, not `0x3c803c80`.

Conclusion:

- even a self-move in the first padding gap changes Q4NX numerics;
- this gap is part of the cycle-level timing contract;
- future meaningful mutations must preserve bundle/cycle count exactly, not
  only architectural register state.

## 25. Q4NX MIR Padding Gap Sweep

Status: done in `main16-exps/032_q4nx_mir_padding_gap_sweep`.

Goal:

- apply the same self-move byte mutation to each explicit padding gap from
  experiment 029;
- distinguish a globally unsafe mutation from a site-specific timing hazard.

Result:

- `gap_before_0x2c2` at `0x2ba`: payload mismatch, same
  `0x48014801` signature as experiment 031;
- `gap_before_0x1828` at `0x1820`: payload matches;
- `gap_before_0x1844` at `0x183c`: payload matches.

Conclusion:

- padding bytes are not uniformly spare;
- early padding sits inside a live hazard window and must preserve exact cycle
  count;
- late padding can be used as a byte-diff packaging canary for direct-QKV, but
  it does not prove that real Q4NX bundles can be changed safely.

Next goal:

- stop using padding mutation as a proxy for real optimization;
- build a bundle-level mutation manifest from the byte-exact pre-bundled MIR:
  bundle address, original bytes, replacement bytes, expected cycle count, and
  live def/use cells;
- first mutate only a bundle whose destination is overwritten before any use,
  or whose source operand is provably dead for the selected synthetic case;
- keep the acceptance gates strict: expected byte diff, no timeout, exact
  direct-QKV payload match, and topology healthy after the run.

## 26. Q4NX MIR Bundle Mutation Manifest

Status: done in `main16-exps/033_q4nx_mir_bundle_mutation_manifest`.

Goal:

- stop selecting mutation sites by raw address guessing;
- produce a bundle-level manifest with bytes, ops, semantics, cell defs/uses,
  and mutation class;
- use experiment 031/032 padding results as known canary/timing-critical facts.

Result:

- total bundles: `966`;
- source bundles: `963`;
- padding bundles: `3`;
- timing-critical padding: `0x2ba`;
- byte-diff canary padding: `0x1820`, `0x183c`;
- hard-safe source candidates: `0`;
- weak source candidates after experiment 038 gates: `0`;
- known failed source mutations: `8`.

Important correction:

- the first manifest version treated `vups.4x` as a pure destination overwrite;
- experiment 034 falsified that model on `zero0_allq4`;
- experiments 036/037 measured the actual `vups.4x` transfer behavior, so the
  manifest now models `dm/cml/cmh` directly instead of using a broad implicit
  destination read;
- experiments 034/038 mark the failed `vmov -> nopm` source mutations as known
  failed gates;
- the manifest still treats `lfh*` as external live-out.

Conclusion:

- source-level mutation needs instruction semantics, not only cell overwrite
  order;
- padding canaries are useful for packaging checks but not for performance
  proof;
- no source bundle is currently safe enough to mutate after the zero/offset
  numeric gates.

## 27. Q4NX MIR Dead Vmov NOPM Numeric Gate

Status: negative all-case gate done in
`main16-exps/034_q4nx_mir_dead_vmov_nopm_numeric_gate`.

Goal:

- test the first weak source-bundle candidate from the pre-refinement manifest;
- replace `0x44e: vmov bmhh1, bmhh2` with same-length MV-slot `nopm`;
- run all synthetic direct-QKV cases.

Result:

- `q4word0_allnibbles`: pass;
- `q4word512_allnibbles`: pass;
- `q4word0_nibble3`: pass;
- `zero0_allq4`: fail, with expected `0x4090/0x4080` becoming
  `0x408f/0x407f`.

Conclusion:

- the mutation is not generally safe;
- the bundle appears dead for simple q4-data paths but still affects
  zero/offset correction or a destination-read alias inside `vups.4x`;
- this is exactly why qwen3-layer integration must wait.

Next goal:

- learn the exact `vups.4x` accumulator-cell read/write semantics from
  official docs, llvm-aie TableGen, and small MIR probes;
- update the manifest with instruction-specific cell transfer functions rather
  than generic register-family overwrite rules;
- only after that should we attempt another real source-bundle mutation.

## 28. VUPS.4x Semantics Surface

Status: done in `main16-exps/035_vups4x_semantics_surface`.

Goal:

- answer whether official/compiler material explains `vups.4x` deeply enough
  for MyLM-style bundle mutation;
- compile the four AIE2P `vups.4x` forms from minimal MIR;
- separate compiler-visible facts from value-level semantics that still need
  NPU probing.

Result:

- minimal MIR builds pass for:
  - `VUPS_4x_mv_ups_w2c_upsSign0`;
  - `VUPS_4x_mv_ups_w2c_upsSign1`;
  - `VUPS_4x_mv_ups_x2d_upsSign0`;
  - `VUPS_4x_mv_ups_x2d_upsSign1`.
- `llvm-aie` tells us operand classes, implicit uses/defs, and schedule
  resources:
  - `w2c`: destination class `OP_mCMm`, source class `OP_mWm`;
  - `x2d`: destination class `eDM`, source class `OP_mXm`;
  - both use `crSat`, `crUPSMode`, and `upsSign0/1`;
  - both define `srUPS_of`.

Conclusion:

- MIR and `llvm-aie` are sufficient for encoding and schedule-surface work;
- they are not a value-level semantics reference for `dm/cml/cmh/bm*`
  accumulator sub-cells;
- experiment 034 therefore cannot be fixed by reading TableGen harder.

Next goal:

- write an isolated NPU readback probe for `vups.4x`;
- initialize accumulator quadrants with distinct sentinels;
- execute one `w2c` or `x2d` `vups.4x` variant;
- emit/read back the affected accumulator cells after controlled delays;
- derive the instruction-specific transfer function needed by the bundle
  mutation manifest.

## 29. VUPS.4x Cell Readback Probe

Status: done in `main16-exps/036_vups4x_cell_readback_probe`.

Goal:

- stop guessing `vups.4x` accumulator-cell behavior from TableGen;
- initialize `dm0` quadrants with a sentinel on real NPU;
- execute one target `vups.4x`;
- store one selected quadrant to the record buffer and read it back.

Result:

- all 8 initial real-NPU cases pass without runtime timeout;
- `x2d_upssign0` changes every observed quadrant:
  `bmll0`, `bmlh0`, `bmhl0`, and `bmhh0`;
- `w2c_upssign0 cml0` changes `bmll0` and `bmlh0`;
- `w2c_upssign0 cml0` preserves sentinel values in `bmhl0` and `bmhh0`.

Conclusion:

- the simple compiler-level model was too weak, but the physical behavior is
  now directly measurable;
- `x2d` can be modeled as full-`dm` overwrite for this uniform source/shift
  case;
- `w2c cml` can be modeled as low-half update plus high-half preserve for this
  uniform source/shift case;
- this does not yet explain experiment 034's zero/offset mismatch, because that
  failure may depend on lane pattern, `vadd/vsub/vconv`, or a specific upstream
  `vmov` feeding the preserved half.

Next goal:

- add a lane-pattern version of the same probe;
- use distinct source lanes and distinct quadrant sentinels;
- cover `cml` vs `cmh`, `upssign0` vs `upssign1`, and at least one nonzero
  `s0` shift;
- update the bundle manifest with measured `vups.4x` transfer functions rather
  than broad "implicit destination read" conservatism.

## 30. VUPS.4x Lane Pattern Probe

Status: focused real-NPU gate done in
`main16-exps/037_vups4x_lane_pattern_probe`.

Goal:

- complete the missing value-level `vups.4x` cases that experiment 036 left
  open;
- use a non-uniform source vector so the readback proves lane transfer behavior,
  not only "changed vs sentinel";
- cover `x2d`, `cml`, `cmh`, `upssign1`, and nonzero `s0` under the same
  observable compact-record harness.

Result:

- focused case count: `24`;
- all focused cases returned `record_observed`;
- `x2d` updates all four `dm0` quadrants with lane-pattern data;
- `cml` updates only the low half (`bmll0`, `bmlh0`) and preserves the high
  half (`bmhl0`, `bmhh0`);
- `cmh` updates only the high half (`bmhl0`, `bmhh0`) and preserves the low
  half (`bmll0`, `bmlh0`);
- `upssign1` sign-extends affected lane bytes before writeback;
- `s0=1` shifts the observed lane-pattern values left by one step in the
  sampled cases.

Conclusion:

- the remaining `vups.4x` unknown is no longer "does it overwrite or preserve
  accumulator cells";
- the bundle manifest can now model `x2d`, `cml`, and `cmh` with measured
  half-register transfer functions;
- the next useful work is not another padding mutation, but applying this
  instruction-specific model to real source bundles.

Next goal:

- identify why the repeated `vmov -> nopm` mutations all produce the same
  one-step-lower zero/offset payload;
- model the zero/offset correction chain explicitly before trying another
  source-bundle mutation.

## 31. Q4NX MIR Manifest Candidate NOPM Sweep

Status: negative real-NPU gate done in
`main16-exps/038_q4nx_mir_manifest_candidate_nopm_sweep`.

Goal:

- test whether the updated `vups.4x` transfer model exposes any real source
  bundle that can be changed without byte-copying MyLM;
- replace each weak manifest `vmov` candidate with MV-slot `nopm`;
- run every candidate against every synthetic direct-QKV case.

Result:

- candidates tested: `7`;
- synthetic cases per candidate: `4`;
- all seven candidates pass the three simple q4-data cases;
- all seven candidates fail `zero0_allq4`;
- the failure signature is identical across the sites:
  expected `0x4090/0x4080`, observed `0x408f/0x407f`.

Conclusion:

- the third gap is still real: we do not yet have a self-generated
  non-byte-copy MyLM-style source block that passes the full NPU numeric gate;
- however, the failure is now sharply localized to the zero/offset correction
  path, not generic `vups.4x` overwrite semantics;
- the manifest is now stricter and no longer proposes these false candidates.

Next goal:

- trace the zero/offset correction chain around the repeated failing `vmov`
  sites;
- determine whether the `vmov` is feeding a hidden half-register alias, a
  rounding/bypass hazard, or a cycle-level dependency;
- only after that, generate a small non-byte-copy block that preserves the
  zero/offset path and passes all four direct-QKV cases.

## 32. Q4NX Zero/Offset Vmov Source Probe

Status: done in `main16-exps/039_q4nx_zero_offset_vmov_source_probe`.

Goal:

- stop treating the repeated `vmov bmhh1,bmhh2` failure as a generic unknown;
- mutate one representative site, `0x700`, with source-preserving and
  source-changing replacements;
- run the zero/offset case only, because experiment 038 showed that this is the
  discriminating case.

Result:

- `nopm`: fails;
- `vmov bmhh1,bmhh1`: fails with the same one-step-lower payload;
- `vmov bmhh1,bmll2`: passes;
- `vmov bmhh1,bmlh2`: passes;
- `vmov bmhh1,bmhl2`: passes;
- `vmov bmhh1,bmhh2`: passes.

Conclusion:

- the dependency is not merely cycle count or a writeback to `bmhh1`;
- the zero/offset path requires refreshing `bmhh1` from the `acc2` correction
  value family;
- the synthetic zero/offset case does not distinguish which `acc2` quadrant is
  used at this template point.

Next goal:

- run the alternate `acc2` source replacements against all direct-QKV synthetic
  cases;
- if they pass, record this as the first non-byte-copy source-bundle mutation
  that survives the current NPU numeric gate.

## 33. Q4NX Acc2 Alternate Source Numeric Gate

Status: done in `main16-exps/040_q4nx_acc2_alternate_source_numeric_gate`.

Goal:

- take the passing alternate sources from experiment 039;
- run them against all four direct-QKV synthetic cases;
- decide whether we have a real non-byte-copy source-bundle mutation that
  preserves current numeric coverage.

Result:

- `vmov bmhh1,bmll2`: all four cases pass;
- `vmov bmhh1,bmlh2`: all four cases pass;
- `vmov bmhh1,bmhl2`: all four cases pass;
- original `vmov bmhh1,bmhh2`: all four cases pass.

Conclusion:

- we now have a non-byte-copy source-bundle mutation that passes the current
  direct-QKV synthetic NPU gate;
- this is a tooling and semantics milestone, not a speedup;
- the current synthetic coverage still cannot distinguish the four `acc2`
  quadrants at this site, so it is not strong enough to validate a production
  Q4NX rewrite.

Next goal:

- strengthen the numeric gate with lane/asymmetric zero cases that distinguish
  `acc2` quadrants;
- then use the same replacement machinery to test a tiny generated
  MyLM-style block, not only a single-source operand mutation.

## 34. Q4NX Acc2 Quadrant Discriminator Gate

Status: done in `main16-exps/041_q4nx_acc2_quadrant_discriminator_gate`.

Goal:

- strengthen the experiment 040 numeric gate beyond the four original synthetic
  cases;
- avoid depending on the scalar formula by using unmodified MyLM raw output as
  the reference;
- test whether asymmetric zero slots can distinguish the alternate `acc2`
  source quadrants at site `0x700`.

Result:

- reference cases: `32`;
- cases are zero slot `0..15` crossed with low-half-only and high-half-only
  zero pairs;
- `vmov bmhh1,bmll2`: all 32 cases match MyLM reference;
- `vmov bmhh1,bmlh2`: all 32 cases match MyLM reference;
- `vmov bmhh1,bmhl2`: all 32 cases match MyLM reference.

Conclusion:

- the current asymmetric zero coverage still cannot distinguish `acc2`
  quadrants at this template point;
- the non-byte-copy source mutation from experiment 040 remains valid under a
  stronger gate;
- this strengthens confidence in the tooling, but still does not validate a
  generated MyLM-style block.

Next goal:

- generate a MIR/source-derived hot-loop variant with the alternate source
  rather than patching a single 4-byte instruction directly;
- run it through the same MyLM-reference numeric gate.

## 35. Q4NX Generated MIR Hotloop Variant Gate

Status: done in `main16-exps/042_q4nx_generated_mir_hotloop_variant_gate`.

Goal:

- move beyond direct 4-byte raw patching;
- regenerate the entire `0x260..0x1850` Q4NX hot-loop body from pre-bundled
  MIR;
- change one source operation in MIR and verify the generated hot loop still
  passes the direct-QKV numeric gate.

Mutation:

```text
address: 0x700
original: $bmhh1 = VMOV_alu_mv_mv_x $bmhh2
variant:  $bmhh1 = VMOV_alu_mv_mv_x $bmll2
```

Result:

- generated hot-loop bytes: `5616`;
- byte diffs vs MyLM hot loop: `1`;
- diff: `0x702: cb -> c8`;
- `q4word0_allnibbles`: pass;
- `q4word512_allnibbles`: pass;
- `q4word0_nibble3`: pass;
- `zero0_allq4`: pass.
- asymmetric-zero gate: `32/32` pass.

Conclusion:

- this is the first controlled non-byte-copy generated MIR hot-loop variant
  that survives the current NPU numeric gate;
- the MIR route is now proven for a semantically safe source operand mutation;
- this is still not a performance optimization, because the generated variant
  is intentionally equivalent under current coverage.

Next goal:

- replace a larger local template window, not just a single source operand,
  while preserving the MyLM-reference numeric gate;
- start moving from semantically equivalent mutation toward a generated block
  that can change scheduling without breaking payload.

## 36. Q4NX Generated MIR Template Variant Gate

Status: done in `main16-exps/043_q4nx_generated_mir_template_variant_gate`.

Goal:

- move from a single source operand mutation to a repeated template-level
  generated MIR mutation;
- regenerate the entire `0x260..0x1850` hot loop;
- replace every matching source bundle:

```text
original: $bmhh1 = VMOV_alu_mv_mv_x $bmhh2
variant:  $bmhh1 = VMOV_alu_mv_mv_x $bmll2
```

Result:

- mutation sites: `15`;
- byte diffs vs MyLM hot loop: `15`;
- all diffs are expected source-operand byte changes: `cb -> c8`;
- direct-QKV gate: `4/4` pass;
- asymmetric-zero gate: `32/32` pass.

Conclusion:

- the MIR route can now generate a repeated-template hot-loop variant, not only
  a one-site operand mutation;
- the variant is still semantically equivalent under current coverage and is
  not a performance optimization;
- nevertheless, this is the first proof that our generated MIR pipeline can
  modify a local template across the MyLM hot loop while preserving the
  MyLM-reference NPU numeric gate.

Next goal:

- replace a contiguous local bundle window rather than identical isolated
  operand bundles;
- the window should include producer and consumer ops, but keep the same
  schedule shape first;
- only after a contiguous generated window passes should we try schedule
  changes that could affect performance.

## 37. Q4NX Generated MIR Contiguous Window Gate

Status: done in `main16-exps/044_q4nx_generated_mir_contiguous_window_gate`.

Goal:

- move from isolated/repeated operand mutation to a contiguous local window;
- keep the original bundle schedule shape first;
- verify that a generated MIR window spanning producer and consumer ops can
  still pass the MyLM-reference NPU numeric gates.

Window:

```text
0x6ee..0x704
```

Changed moves:

```text
$bmlh1 = VMOV_alu_mv_mv_x $bmlh2 -> $bmlh1 = VMOV_alu_mv_mv_x $bmll2
$bmhl1 = VMOV_alu_mv_mv_x $bmhl2 -> $bmhl1 = VMOV_alu_mv_mv_x $bmll2
$bmhh1 = VMOV_alu_mv_mv_x $bmhh2 -> $bmhh1 = VMOV_alu_mv_mv_x $bmll2
```

Result:

- byte diffs vs MyLM hot loop: `3`;
- direct-QKV gate: `4/4` pass;
- asymmetric-zero gate: `32/32` pass.

Conclusion:

- the MIR route can now carry a contiguous local window mutation while
  preserving current numeric coverage;
- this is still a semantically equivalent schedule-shape-preserving mutation,
  not a speedup;
- the next meaningful step is to change schedule shape inside a small window,
  not just source operands.

Next goal:

- construct a window-level schedule-changing experiment with the same operations
  and live-ins/live-outs;
- begin with a change that only reorders independent `vmov`/broadcast work
  around the existing `vconv/vmac/vups/vadd` sequence;
- keep the acceptance gates identical: generated full hot loop, direct-QKV,
  and asymmetric-zero MyLM-reference NPU comparison.

## 38. Q4NX Generated MIR High-Half Swap Gate

Status: done in `main16-exps/045_q4nx_generated_mir_highhalf_swap_gate`.

Goal:

- make the first schedule-shape mutation in a local generated MIR window;
- preserve operation count, bundle sizes, and full hot-loop replacement;
- verify the result against unmodified MyLM raw output on NPU.

Changed schedule:

```text
0x6f6: $bmhl1 = VMOV_alu_mv_mv_x $bmhl2
0x700: $bmhh1 = VMOV_alu_mv_mv_x $bmhh2
```

becomes:

```text
0x6f6: $bmhh1 = VMOV_alu_mv_mv_x $bmhh2
0x700: $bmhl1 = VMOV_alu_mv_mv_x $bmhl2
```

Result:

- byte diffs vs MyLM hot loop: `2`;
- direct-QKV gate: `4/4` pass;
- asymmetric-zero gate: `32/32` pass.

Conclusion:

- the MIR route is no longer limited to source operand substitutions;
- it can carry a small schedule-order mutation while preserving current
  MyLM-reference numeric gates;
- this still does not prove a performance win, because the change only swaps
  two local high-half moves and does not reduce instruction count or stalls.

Next goal:

- move from a safe two-move reorder to a performance-relevant local window;
- try pulling a proven-independent move across a `vconv/vups/vadd/vmac` bundle
  boundary while keeping live-ins/live-outs explicit;
- require the same direct-QKV and asymmetric-zero NPU gates before considering
  integration into `qwen3-layer`.

## 39. Q4NX Generated MIR VUPS Advance Gate

Status: done in `main16-exps/046_q4nx_generated_mir_vups_advance_gate`.

Goal:

- make a performance-relevant local schedule mutation, not just a safe move
  swap;
- issue `vups.4x` one bundle earlier;
- keep full hot-loop generation and MyLM-reference NPU comparison unchanged.

Changed schedule:

```text
0x700: $bmhh1 = VMOV_alu_mv_mv_x $bmhh2
0x704: $dm1 = VUPS_4x_mv_ups_x2d_upsSign0 $x6, $s0
       $dm2 = VADD_vmac_cm2_add_reg $dm1, $dm0, $r0
```

becomes:

```text
0x700: $dm1 = VUPS_4x_mv_ups_x2d_upsSign0 $x6, $s0
0x704: $bmhh1 = VMOV_alu_mv_mv_x $bmhh2
       $dm2 = VADD_vmac_cm2_add_reg $dm1, $dm0, $r0
```

Result:

- byte diffs vs MyLM hot loop: `4`;
- smoke gate: pass;
- direct-QKV payload cases: `3/3` pass;
- direct zero/offset case `zero0_allq4`: fail.

Failure:

```text
reference first payload words:
0x40904090, 0x40804080, 0x40804080, ...

observed first payload words:
0xc8c2c8c2, 0xc8c2c8c2, 0xc8c2c8c2, ...
```

Conclusion:

- this local schedule change is not numerically safe;
- the high-half copy before `vups.4x` is part of the zero/offset correction
  contract;
- the previous passing schedule swaps were safe because they did not move this
  copy across the `vups.4x` boundary.

Next goal:

- isolate whether `vups.4x` can be advanced while preserving the high-half copy
  timing;
- check the MIR encoder/resource constraints for candidate bundles like
  `bmhh1-copy + vups.4x` at `0x700`;
- only continue to performance-oriented scheduling after the local dependency
  is represented explicitly in the generated block.

## 40. Q4NX VUPS Preserve-Copy Encoding Probe

Status: done in `main16-exps/047_q4nx_vups_preserve_copy_encoding_probe`.

Goal:

- keep the high-half copy timing that experiment 046 showed is numerically
  important;
- test whether `vups.4x` can still be advanced into a nearby bundle;
- separate numeric dependency from MIR encoder/resource constraints.

Candidates:

- `copy_and_vups_at_700`: keep `bmhh1 <- bmhh2` at `0x700` and add `vups.4x`
  to the same bundle; leave only `vadd` at `0x704`.
- `vext_and_vups_at_6f2`: pair `vups.4x` with the earlier lane broadcast at
  `0x6f2`; leave only `vadd` at `0x704`.

Result:

- both candidates fail in the unpadded MIR encoder with `llc` return code `-6`;
- no NPU numeric run is possible for these two schedules through the current
  pre-bundled MIR route.

Conclusion:

- experiment 046 was not just a bad choice of where to put `bmhh1`; if we keep
  the copy timing, the obvious early `vups.4x` slots are not currently encodable;
- MyLM's local placement around `0x700..0x704` is constrained by both data
  dependency and bundle issue resources;
- performance-oriented edits need a manifest that models bundle resource
  legality before we spend time on NPU gates.

Next goal:

- derive a small bundle-resource manifest from byte-exact MyLM MIR: which opcode
  classes can coexist in one bundle, and which combinations make `llc` abort;
- use that manifest to propose legal local schedule moves instead of guessing;
- keep every legal candidate behind the same MyLM-reference direct/asymmetric
  gates.

## 41. Q4NX Schedule DSL Resource Manifest

Status: done in `main16-exps/048_q4nx_schedule_dsl_resource_manifest`.

Goal:

- stop manually editing MIR as the primary interface;
- introduce a minimal layer above MIR: `Program -> Bundle -> Operation`;
- derive a conservative bundle-resource manifest from the MyLM byte-exact hot
  loop.

Result:

- MyLM hot-loop bundle count: `963`;
- observed opcode-kind bundle signatures: `50`;
- DSL exact replay: `5616` text bytes, byte-exact match;
- `highhalf_swap_045`: accepted and encoded, `2` expected byte diffs;
- `vups_advance_046`: rejected before encoding at `0x704 ('vmov_x', 'vadd')`;
- `copy_and_vups_at_700_047`: rejected before encoding at
  `0x700 ('vmov_x', 'vups4x')`;
- `vext_and_vups_at_6f2_047`: rejected before encoding at
  `0x6f2 ('vextbcst16', 'vups4x')`.

Conclusion:

- the route is viable: MIR becomes an output format, not the handwritten source;
- an observed-signature manifest is conservative, but it already prevents the
  bad `vups.4x` guesses from reaching either `llc` crashes or NPU numeric gates;
- this is still not a performance optimizer; it is the minimum infrastructure
  needed to search legal schedule moves systematically.

Next goal:

- enrich the DSL operation model with explicit defs/uses and live-in/live-out
  checks for local windows;
- build a candidate generator that only proposes schedules with observed bundle
  signatures and unchanged live-outs;
- run generated candidates through compile gates first, then MyLM-reference NPU
  numeric gates.

## 42. Q4NX Schedule DSL Window Interface Gate

Status: done in `main16-exps/049_q4nx_schedule_dsl_window_interface_gate`.

Goal:

- add local window interface checks above the observed-signature resource
  manifest;
- keep the check text-derived and narrow;
- prove that resource legality and interface preservation catch different bad
  schedules.

Result:

- `highhalf_swap_045`: interface ok, resource ok;
- `vups_advance_046`: interface ok, resource not ok;
- `drop_bmhh_copy_negative_control`: interface not ok, resource ok.

The negative control replaces:

```text
$bmhh1 = VMOV_alu_mv_mv_x $bmhh2
```

with:

```text
NOP
```

The checker reports:

```text
live_ins:      $bmhh2 -> <empty>
boundary_defs: $bmhh1 -> <empty>
```

Conclusion:

- the DSL now has two useful pre-encoding gates:
  observed bundle signatures and local window interface preservation;
- `vups_advance_046` shows interface preservation alone is insufficient;
- the negative control shows resource signature alone is insufficient.

Next goal:

- generate candidates automatically inside a small window using only operations
  already present in that window;
- filter candidates by interface and observed bundle signatures;
- compile only surviving candidates, then run NPU numeric gates for the first
  nontrivial survivor.
