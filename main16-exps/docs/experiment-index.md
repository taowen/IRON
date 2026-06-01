# Experiment Index

Existing main16-related experiments remain in `experiments/` for reproducibility.
This file is the index for the current learning thread.

## Whole-Core Route

- `experiments/128_raw_core_program_codegen_feasibility`
  - Proves the raw whole-core ELF/codegen route is viable.
  - Shows `aie.core { elf_file = ... }` can package an AIE2P executable program.

- `experiments/129_mylm_main16_standalone_raw_kernel`
  - Wraps MyLM c2r2 raw program plus static segments.
  - Confirms the program can be packaged with static local-memory segments.

## Observable Harness

- `experiments/130_mylm_main16_record_observable_harness`
  - Adds real activation/weight/record DMA and lock harness.
  - Observes raw MyLM c2r2 emitting compact records.
  - Baseline all-zero input emits header `0x4`.

## Dispatcher Control

- `experiments/131_mylm_main16_phase_control_probe`
  - Patches `73d00[0]` and `73c80[0]` across known body/dispatcher addresses.
  - All variants still emit `0x4`.
  - Rules out a simple static first-dword phase selector.

- `experiments/132_mylm_main16_dispatcher_stub_probe`
  - Replaces entry with a small raw caller stub.
  - Shows `[sp - 4] -> 0x78200` and `[0x78200] = 1` enters Q/K/V.
  - Observed headers:
    - control `0`: `0x4`
    - control `1`: `0x1`
    - control `8`: `0x4`

## Q/K/V Boundary

- `experiments/133_mylm_main16_qkv_record_count_probe`
  - Uses exp132 stub with control `1`.
  - Confirms `12 x 0x1`, then the 13th record is `0x4`.
  - Establishes the Q/K/V prefix boundary in the dispatcher sequence.

## New Main16-Exps Series

- `main16-exps/001_dispatcher_header_sequence`
  - Uses exp132 stub with control `1`.
  - Confirms the full phase header sequence:
    `12 x 0x1 + 8 x 0x4 + 48 x 0x8 + 8 x 0x4`.
  - Confirms the full stream chunk count is `1472`, not `76 * 16`,
    because down consumes `8 * 48` chunks.

- `main16-exps/002_stream_consumption_boundary`
  - Confirms non-timeout exact boundaries:
    - Q/K/V: 12 records, 192 chunks.
    - O: 20 records cumulative, 320 chunks.
    - up/gate: 68 records cumulative, 1088 chunks.
    - down/full: 76 records cumulative, 1472 chunks.
  - Records that underfed timeout probes can dirty the raw-core runtime session
    and should be run separately. Recover with
    `sudo systemctl restart amdxdna-pinned.service`.

- `main16-exps/003_phase_body_direct_entry`
  - Calls `0x1870`, `0x1e80`, `0x2490`, and `0x2aa0` directly.
  - Confirms full phase-local record counts and headers without dispatcher:
    Q/K/V `12 x 0x1`, O `8 x 0x4`, up/gate `48 x 0x8`, down `8 x 0x4`.
  - Establishes direct-entry control at the header/count/stream level.

- `main16-exps/004_q4nx_hot_body_schedule`
  - Disassembles the local MyLM whole-core ELF and summarizes
    `0x260..0x1850`.
  - Confirms `lc=2`, static `vmac.f=264`, `vextbcst.16=256`,
    `vups.4x=64`, `vst=0`.
  - Emits the first-group def/use table and the eight activation-lane group
    boundaries for Q4NX schedule work.

- `main16-exps/005_nonzero_payload_probe`
  - Reuses the direct `0x1870` Q/K/V phase-body entry and feeds deterministic
    host activation/weight streams through DMA0/DMA1.
  - Confirms all 12 headers stay `0x1`.
  - Shows zero and activation-only inputs produce zero payload, while Q4 nibble
    1 and 2 produce non-zero payloads `0x42004200` and `0x42804280`.
  - Establishes a real-NPU numeric observability anchor for decoding the MyLM
    Q4NX payload formula.

- `main16-exps/006_q4nx_alias_lifetime_graph`
  - Builds an alias-aware def/use graph for the MyLM Q4NX hot loop
    `0x260..0x1850`.
  - Groups `wl/wh/x` vector aliases and `bm*/cml/cmh/dm` accumulator aliases
    into families.
  - Confirms the steady-state cross-group live-through signature is identical
    for all group boundaries:
    `acc1, acc4, vec0, vec10, vec2, vec3, vec4, vec8, vec9`.
  - Turns the next source-assembly/codegen target from a vague instruction
    count into a concrete register-family schedule.

- `main16-exps/007_half_register_boundary_trace`
  - Refines experiment 006 to vector-half and accumulator-quadrant cells.
  - Confirms group1 has 33 `vmac.f`, 25 mixed vector operands, 9 cross-group
    vector operands, and 2 cross-group accumulator operands.
  - Shows concrete mixed operands such as `0x570 vmac.f`, where `dm4`, `x4`,
    and `x2` are all assembled from separately scheduled cells.
  - Establishes that a MyLM-style generator needs explicit cell state across
    fill/steady/drain, not whole-vector SSA temporaries.

- `main16-exps/008_q4nx_payload_formula_probe`
  - Tests scalar synthetic Q4NX formulas against direct Q/K/V payloads.
  - Confirms scale, nibble, and activation linearity for full-record uniform
    inputs.
  - Falsifies the naive one-active-chunk model: chunk0 contributes bf16 `4`,
    not bf16 `2`.

- `main16-exps/009_q4nx_chunk_contribution_map`
  - Activates one record0 chunk at a time.
  - Shows even chunks `0,2,...,14` contribute bf16 `4`, while odd chunks
    contribute zero.
  - Confirms the sum of single-chunk contributions equals the all-record0
    aggregate `32`.

- `main16-exps/010_q4nx_activation_weight_axis_map`
  - Separates activation-only and weight-only single chunk sweeps.
  - Shows both axes have the same even-only contribution pattern.

- `main16-exps/011_q4nx_chunk_pair_matrix`
  - Runs a small activation/weight chunk pair matrix.
  - Confirms matching even pairs contribute (`0*0`, `2*2`) and cross pairs such
    as `0*2` do not.

- `main16-exps/012_q4nx_field_layout_probe`
  - Fixes active activation/weight chunk0 and varies fields inside the Q4NX
    weight chunk.
  - Shows scale dword `i` maps to payload lane `i mod 16` for the synthetic
    setup.
  - Shows q4 data words in the low region affect lanes `0..7`, while sparse
    words at `512+` affect lanes `8..15`; even/odd q4 words select lane
    quartets `0..3/4..7` or `8..11/12..15`.

- `main16-exps/013_q4nx_layout_boundary_nibble_zero_probe`
  - Runs 62 direct-QKV NPU cases covering q4 half boundary, single-nibble
    contribution, and zero/offset slots.
  - Confirms the q4 low/high split is exactly at data word `512`.
  - Shows q4 nibble positions `0/2/4/6` map to the low bf16 half of payload
    words, while `1/3/5/7` map to the high bf16 half.
  - Shows zero/offset slot `i` maps as `i mod 16` and contributes positively in
    the synthetic payload contract.

- `main16-exps/014_q4nx_stream_parity_probe`
  - Runs 90 direct-QKV NPU cases across record0 stream chunks.
  - Confirms even active chunks `0,2,...,14` contribute and odd active chunks
    do not.
  - Shows all contributing even chunks share the same q4word0 word-half nibble
    mapping.

- `main16-exps/015_q4nx_tiny_codegen_numeric_gate`
  - Generates whole-core source assembly for four representative synthetic
    record0 cases.
  - Runs both the real MyLM direct `0x1870` body and the generated tiny program
    on NPU.
  - Passes exact record equality for formula vs MyLM, generated source asm vs
    formula, and generated source asm vs MyLM.
  - Establishes the first source-assembly/codegen numeric gate before dynamic
    Q4NX arithmetic.

- `main16-exps/016_q4nx_dynamic_tiny_arithmetic_gate`
  - Attempts to replace generated constants with dynamic scale/q4/zero loads and
    scalar control flow.
  - Currently fails by timeout; keep it as a negative result, not as a production
    candidate.
  - Root cause is not buffer visibility; later experiments show source-asm
    latency and branch-generation rules were missing.

- `main16-exps/017_q4nx_dynamic_readback_probe`
  - Reads candidate DMA buffer addresses from source assembly and writes them
    directly into the compact record.
  - Passes with full local addresses:
    weight `0x72800`, activation `0x78000`, record `0x73c1c`.
  - Confirms experiment 016's first failure was not because the generated core
    could not see DMA-written buffers.

- `main16-exps/018_source_asm_latency_model`
  - Sweeps producer/consumer gaps on real NPU.
  - Establishes conservative source-asm gaps:
    `lda_to_st=6`, `lda_to_eq_jz=6`, `eq_to_jz=0`,
    `and_to_eq_value=0`, `lshl_to_or=0`.
  - Turns source-asm scheduling into a measured contract instead of guessed
    nop insertion.

- `main16-exps/019_source_asm_branch_semantics_probe`
  - Probes scalar `eq` and `jz/jnz` behavior in the source-asm harness.
  - Confirms `eq`: equal -> `1`, not equal -> `0`.
  - Confirms `jz` can be used as branch-on-zero after compare; direct `mova`
    into branch registers and `jnz` are not reliable generation primitives in
    this harness.

- `main16-exps/020_peano_q4nx_compiler_route`
  - Tries the compiler route with a narrow Peano/llvm-aie Q4NX intrinsic
    source instead of hand-scheduling another raw source-assembly body.
  - Confirms Peano can emit the signed primitive we need:
    `#0x33c + vextbcst.16 + vmac.f`.
  - Confirms the group-sum candidate can hit the macro counts
    `vmac.f=264`, `vextbcst.16=256`, `vextbcst.32=0`.
  - Also shows the current C++ intrinsic shape is still not MyLM-quality:
    it emits `vups.2x`, `vmul.f=256`, `vconv.bf16.fp32=384`, and no
    `vups.4x`, while MyLM uses a register-resident `vups.4x` schedule.

- `main16-exps/021_llvm_aie_backend_surface`
  - Inventories which parts of `~/projects/llvm-aie` are useful after the
    C++ intrinsic route proved too loose.
  - Confirms AIE2P has explicit `vextbcst.16`, `vmac.f`, and `vups.4x`
    instruction definitions, selection patterns, and schedule metadata.
  - Selects the next route: use `llvm-aie` as an external compiler service and
    machine-contract source, then generate a tiny Q4NX MIR/source-asm body
    instead of copying LLVM backend C++ into IRON.

- `main16-exps/022_q4nx_mir_schedule_probe`
  - Generates AIE2P machine MIR directly, bypassing C++ lowering.
  - Confirms Peano can assemble and objdump the critical instruction trio:
    `vups.4x`, `vextbcst.16`, and `vmac.f`.
  - Confirms a tiny loop enters postpipeliner and gets a valid schedule.
  - Confirms a naive one-group macro shape reaches `vups.4x=8`,
    `vextbcst.16=32`, `vmac.f=33`, `vst=0`, but postpipeliner cannot schedule
    it yet. The next issue is explicit MyLM-style register lifetime design, not
    C++ instruction selection.

- `main16-exps/023_q4nx_mir_lifetime_order_probe`
  - Compares a naive one-group MIR order with a MyLM group1 projection for
    `vups.4x`, `vextbcst.16`, and `vmac.f`.
  - Confirms both forms compile to the expected op counts with no
    `vextbcst.32` and no `vst`.
  - Shows preserving only the three headline op classes is insufficient; the
    missing producer graph is `vlda/vldb/vunpack/vadd/vsub/vconv/vmov`.

- `main16-exps/024_q4nx_mir_opcode_coverage_map`
  - Maps the full MyLM group1 opcode vocabulary to AIE2P machine MIR.
  - Confirms Peano can compile the full group1 event order, excluding only
    explicit nops, with exact counts:
    `vups.4x=8`, `vextbcst.16=32`, `vmac.f=33`, `vmov=50`,
    `vconv.bf16.fp32=17`, `vst=0`.
  - Shows the isolated group still does not postpipeline (`loop_bundle_count=120`),
    so the next kernel generator must emit fill/steady/drain group pairs, not a
    standalone group.

- `main16-exps/025_q4nx_mir_group_window_probe`
  - Extends the MIR projection to steady-state group windows `(1, 2)` and
    `(1, 2, 3)`.
  - Confirms exact multiplied op counts, no `vextbcst.32`, and no `vst`.
  - Shows generated loop byte counts closely track the MyLM source spans:
    `1392` bytes for a `1382`-byte source window and `2080` bytes for a
    `2074`-byte source window.
  - Confirms the MIR route can preserve MyLM's object-level shape; next work is
    numeric harness packaging, not more C++ intrinsic tuning.

- `main16-exps/026_q4nx_mir_full_hotloop_probe`
  - Compiles the complete MyLM Q4NX hot-loop event stream as direct AIE2P MIR,
    including scalar pointer setup/drain instructions.
  - Confirms all `1422` non-nop events translate and assemble.
  - Confirms the full ZOL object keeps the MyLM headline counts:
    `vups.4x=64`, `vextbcst.16=256`, `vmac.f=264`,
    `vconv.bf16.fp32=136`, `vst=0`, `vextbcst.32=0`.
  - Shows the loop byte count is `5600` bytes for a `5610`-byte MyLM source
    span, so direct MIR is now a credible route for generating the hot body.
  - The next gate is numeric packaging into a direct-QKV harness, not more
    object-shape proof.

- `main16-exps/027_q4nx_mir_phase_abi_probe`
  - Identifies the ABI layer missing between experiment 026's hot-loop object
    and a runnable direct-QKV numeric harness.
  - Confirms MyLM enters `0x260..0x1850` through a `0x01f0` prologue that sets
    `lc=2`, `ls=0x260`, `le=0x1850`, constants, control registers, and rebases
    `p0` by `0x400`.
  - Confirms the Q/K/V phase body writes 8 group-sum halfwords before calling
    the shared microkernel.
  - Confirms the call delay slots map `p0=weight`, `p1=activation`,
    `p2=0x73c80`, and `p3=group-sum stream`.
  - Confirms post-call record emit uses `vst.conv.bf16.fp32`.
  - Establishes the next runnable kernel boundary:
    `prologue + generated hot loop + record emit`, not hot loop alone.

- `main16-exps/028_q4nx_mir_hotloop_patch_numeric_gate`
  - Patches the experiment 026 generated MIR hot-loop bytes into MyLM's raw
    program while preserving MyLM prologue, Q/K/V body, group-sum producer,
    DMA/locks, and record emitter.
  - Runs the direct-QKV NPU harness and observes records without timeout.
  - Fails numerically: the first synthetic case emits `0x4b014b01` payloads
    instead of the expected `0x3c803c80` payloads.
  - Narrows the failure to MIR schedule semantics, not topology, phase entry,
    DMA, locks, or record output.

- `main16-exps/029_q4nx_mir_prebundled_exact_replay`
  - Converts the MyLM hot-loop event stream into pre-bundled MIR and compiles it
    with `--start-after=postmisched --skip-machine-alignment`.
  - Shows the unpadded pre-bundled MIR is not enough: it first differs at
    `0x2ba` because address-gap NOPs are missing.
  - Inserts three explicit address gaps before `0x2c2`, `0x1828`, and `0x1844`.
  - Passes a byte-exact gate against MyLM raw bytes for `0x260..0x1850`
    (`5616` bytes).
  - Establishes the correct MIR contract for hand-scheduled AIE2P code:
    pre-bundled MIR is an encoder input, while unbundled MIR is a scheduler
    input.

- `main16-exps/030_q4nx_mir_byte_exact_noop_numeric_gate`
  - Patches the byte-exact padded MIR object from experiment 029 back into
    MyLM's raw `0x260..0x1850` hot-loop range.
  - Verifies the patched full raw program remains byte-identical to the
    original MyLM raw program before building the direct-QKV harness.
  - Runs all four synthetic direct-QKV NPU numeric cases and matches the MyLM
    observed payloads.
  - Establishes the no-op packaging/runtime baseline for future controlled
    bundle mutations.

- `main16-exps/031_q4nx_mir_padding_mutation_numeric_gate`
  - Replaces four zero bytes in the first explicit hot-loop padding gap
    (`0x2ba`) with `mov r31, r31`.
  - Runs the direct-QKV NPU harness and observes a record without timeout.
  - Fails numerically: the first payload becomes `0x48014801` instead of
    `0x3c803c80`.
  - Shows the first padding gap is not spare code space; it is part of the
    cycle-level timing contract.

- `main16-exps/032_q4nx_mir_padding_gap_sweep`
  - Applies the same `mov r31, r31` byte mutation independently to the three
    explicit hot-loop padding gaps from experiment 029.
  - Confirms `0x2ba` is timing-critical and changes payload.
  - Confirms the late gaps at `0x1820` and `0x183c` preserve the direct-QKV
    synthetic payload for the first case.
  - Establishes safe byte-diff canary sites for packaging tests, while keeping
    real Q4NX bundle mutation separate from padding mutation.

- `main16-exps/033_q4nx_mir_bundle_mutation_manifest`
  - Builds a bundle-level manifest for the byte-exact MyLM hot loop.
  - Records per-bundle bytes, ops, semantics, cell-level defs/uses, and
    mutation class.
  - Uses experiments 036/037's measured `vups.4x` transfer model:
    `dm` updates all quadrants, `cml` updates the low half, and `cmh` updates
    the high half.
  - Tracks experiments 034/038 failed source mutations as known failed gates.
  - Treats `lfh*` writes as external live-out because the surrounding phase-body
    record emit is outside the hot-loop range.
  - Finds no hard-safe or weak source-bundle mutation candidates after the
    zero/offset numeric gates.

- `main16-exps/034_q4nx_mir_dead_vmov_nopm_numeric_gate`
  - Tests the first weak source-bundle candidate from the earlier manifest:
    `0x44e: vmov bmhh1, bmhh2 -> nopm`.
  - Passes `q4word0_allnibbles`, `q4word512_allnibbles`, and
    `q4word0_nibble3`.
  - Fails `zero0_allq4` with a one-step-lower payload pattern
    (`0x4090/0x4080` expected, `0x408f/0x407f` observed).
  - Shows the naive dead-destination proof missed data-dependent
    zero/offset-path semantics.

- `main16-exps/035_vups4x_semantics_surface`
  - Extracts the local `llvm-aie` compiler-visible contract for the four
    AIE2P `vups.4x` forms.
  - Confirms minimal MIR can encode:
    `w2c/upssign0`, `w2c/upssign1`, `x2d/upssign0`, and `x2d/upssign1`.
  - Records operand classes, implicit control/status registers, and schedule
    resources from `AIE2PGenInstrInfo.td` and `AIE2PGenSchedule.td`.
  - Establishes that the public/compiler metadata is not enough to decide
    accumulator sub-cell preservation or implicit destination reads; those
    still require an isolated NPU readback probe.

- `main16-exps/036_vups4x_cell_readback_probe`
  - Runs an isolated source-assembly NPU readback probe for `vups.4x`.
  - Initializes `dm0` quadrants with sentinel `0x3c80`, runs one target
    `vups.4x`, and stores one selected quadrant back through the compact record
    buffer.
  - Confirms `x2d_upssign0` changes all four quadrants:
    `bmll0`, `bmlh0`, `bmhl0`, and `bmhh0`.
  - Confirms `w2c_upssign0 cml0` changes only the low half:
    `bmll0` and `bmlh0`; `bmhl0` and `bmhh0` preserve the sentinel.
  - Establishes the first real NPU value-level transfer fact missing from the
    static bundle manifest.

- `main16-exps/037_vups4x_lane_pattern_probe`
  - Extends experiment 036 with a non-uniform source vector, distinct quadrant
    sentinels, `cml/cmh`, sampled `upssign1`, and sampled nonzero `s0`.
  - Runs 24 focused real-NPU cases and observes a compact record for every case.
  - Confirms `x2d` updates all four `dm0` quadrants.
  - Confirms `cml` updates only `bmll0/bmlh0` and preserves
    `bmhl0/bmhh0`; `cmh` updates only `bmhl0/bmhh0` and preserves
    `bmll0/bmlh0`.
  - Shows `upssign1` sign-extends affected lane bytes and `s0=1` shifts sampled
    lane-pattern values left by one step.
  - Provides the measured transfer facts needed to make the bundle mutation
    manifest stricter than "implicit destination read".

- `main16-exps/038_q4nx_mir_manifest_candidate_nopm_sweep`
  - Takes the seven weak `vmov` source candidates exposed by the measured
    `vups.4x` model and replaces each with MV-slot `nopm`.
  - Runs every candidate against all four direct-QKV synthetic numeric cases on
    real NPU.
  - Shows all candidates pass the simple q4-data cases but fail
    `zero0_allq4` with the same one-step-lower zero/offset payload pattern.
  - Establishes that the apparent dead `vmov` bundles are part of the
    zero/offset correction contract; they are now recorded as known failed in
    experiment 033.

- `main16-exps/039_q4nx_zero_offset_vmov_source_probe`
  - Focuses on representative site `0x700: vmov bmhh1, bmhh2`.
  - Runs `zero0_allq4` after replacing the source with `nopm`, self-copy, or a
    different `acc2` quadrant.
  - Shows `nopm` and `vmov bmhh1,bmhh1` fail with the same one-step-lower
    zero/offset payload.
  - Shows every tested `acc2 -> bmhh1` source passes:
    `bmll2`, `bmlh2`, `bmhl2`, and original `bmhh2`.
  - Narrows the dependency to refreshing `bmhh1` from the `acc2` correction
    value family, not preserving exact original source quadrant.

- `main16-exps/040_q4nx_acc2_alternate_source_numeric_gate`
  - Runs the passing alternate `acc2` sources from experiment 039 against all
    four direct-QKV synthetic cases.
  - Confirms `vmov bmhh1,bmll2`, `vmov bmhh1,bmlh2`, and
    `vmov bmhh1,bmhl2` all match the current numeric gate at site `0x700`.
  - Establishes the first non-byte-copy source-bundle mutation that survives
    the current NPU numeric coverage.
  - Notes that this is not a performance optimization; it is a tooling and
    semantics milestone for future generated source bundles.

- `main16-exps/041_q4nx_acc2_quadrant_discriminator_gate`
  - Strengthens the source-mutation numeric gate with 32 asymmetric zero cases:
    zero slot `0..15` crossed with low-half-only and high-half-only zero pairs.
  - Uses unmodified MyLM raw output as the reference instead of the scalar
    formula.
  - Confirms `vmov bmhh1,bmll2`, `vmov bmhh1,bmlh2`, and
    `vmov bmhh1,bmhl2` all match the MyLM reference for every asymmetric zero
    case at site `0x700`.
  - Shows the current asymmetric zero coverage still cannot distinguish `acc2`
    quadrants at this template point.

- `main16-exps/042_q4nx_generated_mir_hotloop_variant_gate`
  - Regenerates the full `0x260..0x1850` Q4NX hot loop from pre-bundled MIR
    instead of directly patching one raw instruction.
  - Changes the source operation at `0x700` from
    `$bmhh1 = VMOV_alu_mv_mv_x $bmhh2` to
    `$bmhh1 = VMOV_alu_mv_mv_x $bmll2`.
  - Replaces the whole hot-loop range in the raw program with generated bytes.
  - Produces one byte diff versus MyLM (`0x702: cb -> c8`) and passes all four
    direct-QKV synthetic cases against the unmodified MyLM reference.
  - Also passes the 32-case asymmetric-zero reference gate from experiment 041.
  - Establishes the first controlled non-byte-copy generated MIR hot-loop
    variant that survives the current NPU numeric gate.

- `main16-exps/043_q4nx_generated_mir_template_variant_gate`
  - Regenerates the full Q4NX hot loop from pre-bundled MIR and changes every
    matching template source bundle:
    `$bmhh1 = VMOV_alu_mv_mv_x $bmhh2` to
    `$bmhh1 = VMOV_alu_mv_mv_x $bmll2`.
  - Replaces 15 template sites across the hot loop and produces 15 expected
    byte diffs (`cb -> c8` at the encoded source operand byte).
  - Passes all four direct-QKV synthetic cases against the unmodified MyLM
    reference.
  - Passes all 32 asymmetric-zero cases against the unmodified MyLM reference.
  - Establishes that the MIR route can carry a repeated-template generated
    variant, not only a single source-operand change.

- `main16-exps/044_q4nx_generated_mir_contiguous_window_gate`
  - Regenerates the full hot loop while changing one contiguous
    producer/consumer window `0x6ee..0x704`.
  - Changes three local `acc2 -> acc1` quadrant moves to use `bmll2` while
    preserving the original bundle schedule shape:
    `bmlh1`, `bmhl1`, and `bmhh1`.
  - Produces three expected byte diffs and passes all four direct-QKV cases.
  - Passes all 32 asymmetric-zero cases against the unmodified MyLM reference.
  - Establishes that the generated MIR route can carry a contiguous local window
    mutation, not only isolated or repeated source-operand mutations.

- `main16-exps/045_q4nx_generated_mir_highhalf_swap_gate`
  - Regenerates the full hot loop from MIR and swaps the timing of the local
    high-half moves at `0x6f6` and `0x700`.
  - Keeps the same operation count and bundle sizes, but changes schedule order:
    `bmhl1 <- bmhl2` and `bmhh1 <- bmhh2` are exchanged.
  - Produces two expected byte diffs and passes all four direct-QKV cases.
  - Passes all 32 asymmetric-zero cases against the unmodified MyLM reference.
  - Establishes the first generated MIR schedule-shape mutation that survives
    the current NPU numeric gates.

- `main16-exps/046_q4nx_generated_mir_vups_advance_gate`
  - Regenerates the full hot loop from MIR and moves the `vups.4x` at `0x704`
    one bundle earlier to `0x700`.
  - Delays the original `bmhh1 <- bmhh2` high-half copy into the `0x704`
    bundle with the dependent `vadd`.
  - Encodes cleanly and passes the first smoke case plus three direct-QKV
    payload cases.
  - Fails the direct `zero0_allq4` case against the unmodified MyLM reference:
    the output collapses to repeated `0xc8c2c8c2` instead of the expected
    zero/offset-corrected payload.
  - Shows the `vups.4x`/high-half-copy ordering is part of the local numeric
    contract, not just padding that can be freely rescheduled.

- `main16-exps/047_q4nx_vups_preserve_copy_encoding_probe`
  - Tests whether `vups.4x` can be advanced while preserving the original
    `bmhh1 <- bmhh2` timing at `0x700`.
  - Candidate `copy_and_vups_at_700` tries to issue the high-half copy and
    `vups.4x` in the same bundle; current `llc` aborts in the unpadded encoder.
  - Candidate `vext_and_vups_at_6f2` tries to pair `vups.4x` with the earlier
    lane broadcast and keep only `vadd` at `0x704`; current `llc` also aborts.
  - Shows the local `vups.4x` advance path is constrained both numerically
    and by currently encodable bundle resource combinations.

- `main16-exps/048_q4nx_schedule_dsl_resource_manifest`
  - Adds the first layer above raw MIR: `Program -> Bundle -> Operation`.
  - Emits pre-bundled MIR from the DSL and uses `llc` only as the final encoder.
  - Builds a conservative resource manifest from MyLM's byte-exact hot loop:
    only opcode-kind bundle signatures observed in MyLM are accepted.
  - Byte-exactly round-trips the original MyLM Q4NX hot loop: 963 bundles,
    5616 text bytes, zero diffs.
  - Accepts and encodes the safe `045` high-half swap.
  - Rejects the `046/047` `vups.4x` advance candidates before encoding because
    they create unobserved bundle signatures such as `('vmov_x', 'vadd')`,
    `('vmov_x', 'vups4x')`, and `('vextbcst16', 'vups4x')`.

- `main16-exps/049_q4nx_schedule_dsl_window_interface_gate`
  - Adds local window interface checking on top of the DSL:
    live-ins and boundary defs are derived from MIR text.
  - Confirms the safe `045` high-half swap preserves the window interface and
    passes the resource manifest.
  - Confirms `046` keeps the same textual window interface but fails the
    resource manifest, so resource legality and interface preservation are
    separate checks.
  - Adds a negative control that replaces `bmhh1 <- bmhh2` with `NOP`; the
    resource manifest allows single `NOP`, but the interface gate rejects the
    missing `$bmhh2` live-in and `$bmhh1` boundary def.

## Older Main16/Q4NX Disassembly Studies

The `100..125` experiments contain operand-level and Q4NX microkernel learning.
They are still useful, but should not be treated as the current runnable raw
dispatcher harness.

Most relevant:

- `100_mylm_q4nx_body_contract`
- `103_mylm_q4nx_register_flow`
- `104_mylm_q4nx_pipeline_schedule`
- `119_mylm_q4nx_full_operand_graph`
- `120_mylm_q4nx_generator_contract`
- `125_main16_qkv_nocall_scheduler_contract`
- `126_mylm_main16_whole_core_contract`
- `127_main16_whole_program_scaffold`
