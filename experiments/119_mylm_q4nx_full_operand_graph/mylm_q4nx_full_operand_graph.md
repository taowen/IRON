# MyLM Q4NX Full Operand Graph

This experiment emits operand graph records for every `vmac.f` in the
full MyLM Q4NX hot loop, not only the canonical steady group.

## Checks

- Parsed slots: `1532`
- `vmac.f` records: `264`
- Expected group MAC counts: `True`
- Steady instruction text stable: `True`
- All steady operand signatures stable: `False`
- Steady-to-steady operand signatures stable: `True`
- Mixed vector operands: `202` / `528`
- Cross-group vector operands: `63` / `528`
- Cross-group accumulator operands: `14` / `264`

## Group MAC Counts

| Group | Section | MACs |
| ---: | --- | ---: |
| 0 | `fill` | 28 |
| 1 | `steady` | 33 |
| 2 | `steady` | 33 |
| 3 | `steady` | 33 |
| 4 | `steady` | 33 |
| 5 | `steady` | 33 |
| 6 | `pre_drain` | 33 |
| 7 | `drain` | 38 |

## Steady Hashes

| Group | Text Hash | Operand Signature Hash |
| ---: | --- | --- |
| 1 | `debf92a97adcdff2` | `070ae3b9658fd953` |
| 2 | `debf92a97adcdff2` | `5466ff6129c65f6b` |
| 3 | `debf92a97adcdff2` | `5466ff6129c65f6b` |
| 4 | `debf92a97adcdff2` | `5466ff6129c65f6b` |
| 5 | `debf92a97adcdff2` | `5466ff6129c65f6b` |

## Opcode Counts

| Op | Count |
| --- | ---: |
| `vmac.f` | 264 |
| `vextbcst.16` | 256 |
| `vunpack` | 64 |
| `vups.4x` | 64 |
| `vconv.bf16.fp32` | 136 |
| `vlda` | 11 |
| `vldb` | 46 |
| `lda.s16` | 8 |
| `vbcst.16` | 8 |
| `vst` | 0 |

## Vector Operand Producer Pairs

| Producer Pair | MAC Count |
| --- | ---: |
| `activation_lane + register_move+bf16_coeff` | 49 |
| `activation_lane + register_move+activation_lane` | 40 |
| `activation_lane + bf16_coeff` | 34 |
| `activation_lane + activation_lane` | 32 |
| `bf16_coeff + register_move+activation_lane` | 16 |
| `register_move+activation_lane + register_move+activation_lane` | 16 |
| `scalar_broadcast + vector_load+activation_lane` | 9 |
| `activation_lane + vector_load` | 8 |
| `register_move+activation_lane + register_move+bf16_coeff` | 8 |
| `bf16_coeff + vector_load+activation_lane` | 8 |
| `vector_load + vector_load+activation_lane` | 7 |
| `unpacked_q4 + vector_load` | 7 |
| `unpacked_q4 + unpacked_q4` | 7 |
| `bf16_coeff + vector_load` | 7 |
| `register_move+bf16_coeff + vector_load` | 7 |
| `register_move+activation_lane + unpacked_q4` | 7 |
| `activation_lane + register_move+entry` | 1 |
| `register_move+activation_lane + vector_load+activation_lane` | 1 |

## First Fill MACs

| MAC | Acc | Left | Right |
| --- | --- | --- | --- |
| `g0@0x31e.2:vmac.f` | `vector_arith` | `vector_load` | `activation_lane` |
| `g0@0x330.1:vmac.f` | `accumulator_carry` | `bf16_coeff` | `register_move+activation_lane` |
| `g0@0x342.1:vmac.f` | `accumulator_carry` | `register_move+entry` | `activation_lane` |
| `g0@0x356.1:vmac.f` | `accumulator_carry` | `activation_lane` | `activation_lane` |
| `g0@0x378.1:vmac.f` | `accumulator_carry` | `register_move+activation_lane` | `activation_lane` |
| `g0@0x38e.1:vmac.f` | `accumulator_carry` | `bf16_coeff` | `register_move+activation_lane` |
| `g0@0x3a2.0:vmac.f` | `accumulator_carry` | `register_move+activation_lane` | `activation_lane` |
| `g0@0x3ac.1:vmac.f` | `accumulator_carry` | `register_move+bf16_coeff` | `register_move+activation_lane` |
| `g0@0x3bc.2:vmac.f` | `accumulator_carry` | `register_move+bf16_coeff` | `activation_lane` |
| `g0@0x3d4.2:vmac.f` | `accumulator_carry` | `bf16_coeff` | `activation_lane` |
| `g0@0x3f0.1:vmac.f` | `accumulator_carry` | `activation_lane` | `activation_lane` |
| `g0@0x400.1:vmac.f` | `accumulator_carry` | `register_move+bf16_coeff` | `activation_lane` |

## Generator Boundary

- The JSON output is the first full-loop MAC operand graph: all fill, steady, pre-drain, and drain MACs are represented.
- The steady instruction text is stable. The group1 operand signature differs because it is the fill-to-steady transition; groups2..5 share the steady-to-steady signature.
- The next experiment should consume this graph plus exp118 liveness to emit a modified numeric body and run a synthetic gate before touching `qwen3-layer`.
