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
