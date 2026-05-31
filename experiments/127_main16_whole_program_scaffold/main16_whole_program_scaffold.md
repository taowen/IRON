# Main16 Whole-Program Scaffold

This is a compileable scaffold for the MyLM-style main16 role program.
It is intentionally not linked into the active qwen3-layer path.

- Status: `compileable_scaffold_not_active`
- Entry: `q4nx_main16_whole_program_entry`
- Dispatcher: `q4nx_main16_whole_dispatcher`
- Shared Q4 body: `q4nx_main16_whole_q4_body`
- Record emitter: `q4nx_main16_whole_emit_record`
- FP32 accumulator: `q4nx_main16_whole_float_accum`
- Missing symbols: ``
- Call relocations: `20`
- Full dynamic Q4 calls: `1472`
- Full dynamic records: `76`
- Open gaps: `3`

## Linked C ABI

The first whole-program replacement keeps the existing MLIR call ABI so the
current topology, buffers, BD rings, locks, and downstream compact routing stay unchanged.

| Name | Register | Type | Semantic |
| --- | --- | --- | --- |
| `wt_ping` | `p0` | `bfloat16 *` | main16 DMA1 weight ping buffer |
| `wt_pong` | `p1` | `bfloat16 *` | main16 DMA1 weight pong buffer |
| `chunk_ping` | `p2` | `int32_t *` | main16 DMA0 activation ping buffer |
| `chunk_pong` | `p3` | `int32_t *` | main16 DMA0 activation pong buffer |
| `record_ping` | `p4` | `int32_t *` | main16 compact record ping buffer |
| `record_pong` | `p5` | `int32_t *` | main16 compact record pong buffer |
| `group` | `r0` | `int32_t` | main16 column group id |
| `row` | `r1` | `int32_t` | intile row id |
| `num_rows` | `r2` | `int32_t` | valid rows in this tile |
| `phase_limit` | `r3` | `int32_t` | exclusive phase limit: 3=qkv, 4=qkvo, 6=upgate, 7=full |

## State Registers

These registers are initialized once at entry and are owned by the whole-main16 program.
`r27` is intentionally reserved for `sel.eqz` predicates because the assembler rejects low-register predicates.

| Name | Register | Source | Semantic |
| --- | --- | --- | --- |
| `group` | `r16` | `r0` | main16 column group id |
| `row` | `r17` | `r1` | intile row id |
| `num_rows` | `r18` | `r2` | valid rows for future payload copy |
| `phase_limit` | `r31` | `r3` | exclusive phase limit for the single dispatcher entry |
| `wt_ping_ptr` | `r19` | `p0` | weight ping pointer for Q4 body calls |
| `wt_pong_ptr` | `r20` | `p1` | weight pong pointer for Q4 body calls |
| `chunk_ping_ptr` | `r21` | `p2` | activation ping pointer for Q4 body calls |
| `chunk_pong_ptr` | `r22` | `p3` | activation pong pointer for Q4 body calls |
| `float_accum_ptr` | `r30` | `q4nx_main16_whole_float_accum` | 32-lane FP32 projection accumulator |
| `record_ping_ptr` | `r26` | `p4` | compact record ping pointer |
| `record_pong_ptr` | `r28` | `p5` | compact record pong pointer |
| `vector_insert_index` | `r29` | `constant zero` | assembler-mandated index register for vinsert.32 |
| `select_predicate` | `r27` | `derived` | required predicate register for sel.eqz |

## Lock Contract

| Name | MLIR lock | Core lock | Initial | Core action | Semantic |
| --- | ---: | ---: | ---: | --- | --- |
| `activation_empty` | 0 | 48 | 2 | `release` | DMA0 activation ping/pong reusable by producer |
| `activation_full` | 1 | 49 | 0 | `acquire` | DMA0 activation ping/pong ready for core |
| `weight_empty` | 2 | 50 | 2 | `release` | DMA1 weight ping/pong reusable by row1 fanout |
| `weight_full` | 3 | 51 | 0 | `acquire` | DMA1 weight ping/pong ready for core |
| `record_empty` | 4 | 52 | 2 | `acquire` | compact record ping/pong reusable by core |
| `record_full` | 5 | 53 | 0 | `release` | compact record ping/pong ready for row1 gather |

## Record Header Contract

- Body record: `(phase << 24) | (block << 20) | (group << 16) | (row << 8) | packet_id`
- Projection record: `(phase << 24) | (group << 16) | (row << 8) | packet_id`
- Up/gate phase policy: phase 4 on even replay records, phase 5 on odd replay records, packet 14 for both.

## Dispatcher Contract

- Entry-compatible target: `q4nx_main16_layer_scheduler`
- Phase-limit register: `r31`
- Phase limits: `{'qkv': 3, 'qkvo': 4, 'upgate': 6, 'full': 7}`
- Phase body call sites: `{'qkv': 1, 'o': 1, 'upgate': 1, 'down': 1}`
- Phase-limit compare sites: `3`
- Phase-limit exit sites: `3`

## LR Frame Contract

Every generated non-leaf function saves its caller return address once at entry
and restores it immediately before `ret lr`; the record emitter remains a leaf.

- Frame bytes: `64`
- Non-leaf symbols: `{'q4nx_main16_whole_program_entry': {'save': 1, 'restore': 1, 'calls': 1}, 'q4nx_main16_whole_dispatcher': {'save': 1, 'restore': 1, 'calls': 4}, 'q4nx_main16_whole_q4_body': {'save': 1, 'restore': 1, 'calls': 1}, 'q4nx_main16_whole_body_qkv': {'save': 1, 'restore': 1, 'calls': 3}, 'q4nx_main16_whole_body_o': {'save': 1, 'restore': 1, 'calls': 1}, 'q4nx_main16_whole_body_upgate': {'save': 1, 'restore': 1, 'calls': 1}, 'q4nx_main16_whole_body_down': {'save': 1, 'restore': 1, 'calls': 1}}`
- Leaf symbols: `{'q4nx_main16_whole_emit_record': {'save': 0, 'restore': 0, 'calls': 0}}`

## Q4 Body Call Contract

- Status: `shared_q4_run_body_owns_record_chunk_loops`
- Source of truth: `qwen3-layer/tools/generate_main16_q4nx_asm.py`
- Active asm: `qwen3-layer/main_projection_q4nx_asm.s`
- Uses SAVE/RESTORE: `False`
- Static phase-to-Q4 run calls: `6`
- Dynamic chunk executions: `1472`
- Static chunk-body calls inside Q4 run: `0`
- Static emit-record calls inside Q4 run: `1`
- `p1`: selected inside shared Q4 run body from weight ping/pong by chunk parity
- `p2`: selected inside shared Q4 run body from activation ping/pong by chunk parity
- `p0`: 32-lane FP32 projection accumulator selected inside shared Q4 run body
- Selection predicate: `r27 = chunk & 1 inside shared Q4 run body`
- Static counts: `{'vmac.f': 64, 'vextbcst.16': 64, 'vst': 2, 'save_slots': 1, 'restore_slots': 1}`

## Q4 Run Boundary

Phase bodies now call the shared Q4 run body once per header run.
The shared Q4 run body owns the record loop, chunk loop, DMA lock protocol,
ping/pong selection, exact chunk body execution, and record emission.

| Item | Count |
| --- | ---: |
| Static phase-to-Q4 run calls | 6 |
| Expected phase-to-Q4 run calls | 6 |
| Dynamic chunk executions | 1472 |
| Static chunk-body calls inside Q4 run | 0 |
| Static emit-record calls inside Q4 run | 1 |

## Record Emitter Scaffold

- Status: `header_writer_with_fp32_accum_to_bf16_payload_clear`
- ABI: `p0=selected record buffer, p6=FP32 accumulator, r0=phase, r1=block, r2=packet, r3=group, r4=row, r5=num_rows`
- Static record store sites: `2`
- Dynamic record dwords: `17`
- Dynamic payload BF16 values: `32`
- Dynamic accumulator clear floats: `32`
- Forbidden caller state register references: ``

The emitter now writes the real IRON header to the selected ping/pong record buffer.
Payload is converted from the FP32 accumulator to BF16 and the accumulator is cleared.

## Phase Bodies

| Phase | Symbol | MyLM body | MyLM header | Records | Q4 calls | IRON record schedule |
| --- | --- | ---: | --- | ---: | ---: | --- |
| `qkv` | `q4nx_main16_whole_body_qkv` | `0x1870` | `0x1` | 12 | 192 | `Q:phase0/packet10x8/chunks16/base0/body, K:phase1/packet11x2/chunks16/base128/body, V:phase2/packet12x2/chunks16/base160/body` |
| `o` | `q4nx_main16_whole_body_o` | `0x1e80` | `0x4` | 8 | 128 | `O:phase3/packet13x8/chunks16/base192/body` |
| `upgate` | `q4nx_main16_whole_body_upgate` | `0x2490` | `0x8` | 48 | 768 | `upgate:phase4/phase5 by record parity/packet14x48/chunks16/base320/projection` |
| `down` | `q4nx_main16_whole_body_down` | `0x2aa0` | `0x4` | 8 | 384 | `down:phase6/packet15x8/chunks48/base1088/body` |

## Control Skeleton

Static sites are intentionally looped. Dynamic counts are the expected full
main16 phase behavior when those loops run.

### Static Sites

| Phase | Q4 call sites | Emit call sites | Activation acquire sites | Weight acquire sites | Record acquire sites |
| --- | ---: | ---: | ---: | ---: | ---: |
| `qkv` | 3 | 0 | 0 | 0 | 0 |
| `o` | 1 | 0 | 0 | 0 | 0 |
| `upgate` | 1 | 0 | 0 | 0 | 0 |
| `down` | 1 | 0 | 0 | 0 | 0 |

### Dynamic Counts

| Phase | Records | Chunks/Q4 calls | Activation acquires | Weight acquires | Record acquires |
| --- | ---: | ---: | ---: | ---: | ---: |
| `qkv` | 12 | 192 | 192 | 192 | 12 |
| `o` | 8 | 128 | 128 | 128 | 8 |
| `upgate` | 48 | 768 | 768 | 768 | 48 |
| `down` | 8 | 384 | 384 | 384 | 8 |
| `total` | 76 | 1472 | 1472 | 1472 | 76 |

## Next Implementation Work

1. Replace the current exact chunk body shape with a MyLM-density scheduled body.
2. Keep the shared Q4 body accumulating into the `p6` FP32 accumulator.
3. Preserve the current FP32-to-BF16 record emitter contract while changing the Q4 body internals.
4. Preserve IRON compact headers `10..15` until downstream routing is explicitly changed.
5. Only switch active MLIR to this entry after `full-layer-qkv-prefix token31` passes.

## Open Gaps

- shared Q4 run body still uses the current exact chunk body shape
- whole-program register plan must replace current exact dequant body with MyLM-density scheduled body
- active qwen3-layer uses q4nx_main16_layer_scheduler, but this generated whole-program entry is not active yet
