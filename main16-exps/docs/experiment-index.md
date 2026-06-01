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
