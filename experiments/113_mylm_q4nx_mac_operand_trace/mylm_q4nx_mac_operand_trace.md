# MyLM Q4NX MAC Operand Trace

Source disasm: `/tmp/mylm_qwen3_layer_L31_deep/disasm/c2r2.s`

This report traces the latest visible producer for each `vmac.f` vector operand.
It is a conservative register-family analysis, not a full half-register alias decompiler.

## Whole Hot Loop Summary

- Total `vmac.f`: `264`
- Vector operands crossing a group boundary: `42` / `528`

| Producer Kind | Vector Operands |
| --- | ---: |
| `activation_lane` | 196 |
| `register_move` | 169 |
| `bf16_coeff` | 65 |
| `vector_load` | 61 |
| `unpacked_q4` | 28 |
| `vbcst.16` | 9 |

## Steady Group1 Summary

- Group1 `vmac.f`: `33`
- Vector operands crossing into group1: `6` / `66`
- Accumulator source carries crossing into group1: `2` / `33`

| Vector Producer Pair | MAC Count |
| --- | ---: |
| `activation_lane + register_move` | 11 |
| `activation_lane + activation_lane` | 4 |
| `activation_lane + bf16_coeff` | 4 |
| `register_move + register_move` | 3 |
| `bf16_coeff + vector_load` | 2 |
| `bf16_coeff + register_move` | 2 |
| `unpacked_q4 + unpacked_q4` | 1 |
| `register_move + vector_load` | 1 |
| `register_move + unpacked_q4` | 1 |
| `activation_lane + vector_load` | 1 |
| `vbcst.16 + vector_load` | 1 |
| `vector_load + vector_load` | 1 |
| `unpacked_q4 + vector_load` | 1 |

## Group1 MAC Operand Table

| MAC | Accumulator Source | Left Vector | Right Vector |
| --- | --- | --- | --- |
| `g1@0x52a.1:vmac.f` | `dm1` <- accumulator_carry carry `g0@0x51e.1:vmac.f` (-4) | `x9` <- unpacked_q4 carry `g0@0x50e.0:vunpack` (-9) | `x10` <- unpacked_q4 carry `g0@0x51a.0:vunpack` (-6) |
| `g1@0x53a.1:vmac.f` | `dm1` <- vector_arith `g1@0x536.0:vadd` (-2) | `x4` <- bf16_coeff carry `g0@0x4ba.0:vconv.bf16.fp32` (-38) | `x0` <- vector_load carry `g0@0x51e.0:vldb` (-9) |
| `g1@0x54a.0:vmac.f` | `dm3` <- accumulator_carry `g1@0x53a.1:vmac.f` (-3) | `x3` <- register_move carry `g0@0x4f8.0:vmov` (-22) | `x11` <- vector_load `g1@0x52a.0:vldb` (-8) |
| `g1@0x570.2:vmac.f` | `dm4` <- register_move carry `g0@0x478.0:vmov` (-71) | `x4` <- bf16_coeff `g1@0x564.0:vconv.bf16.fp32` (-5) | `x2` <- vector_load carry `g0@0x4dc.0:vldb` (-44) |
| `g1@0x580.1:vmac.f` | `dm4` <- accumulator_carry `g1@0x570.2:vmac.f` (-4) | `x6` <- register_move `g1@0x580.0:vmov` (-1) | `x1` <- unpacked_q4 `g1@0x55e.0:vunpack` (-11) |
| `g1@0x5d0.2:vmac.f` | `dm3` <- vector_arith `g1@0x5bc.1:vmul.f` (-6) | `x6` <- vector_load `g1@0x5a8.0:vldb` (-13) | `x9` <- activation_lane lane #0x1 `g1@0x556.0:vextbcst.16` (-38) |
| `g1@0x5e2.1:vmac.f` | `dm3` <- accumulator_carry `g1@0x5d0.2:vmac.f` (-4) | `x3` <- bf16_coeff `g1@0x588.0:vconv.bf16.fp32` (-27) | `x10` <- register_move `g1@0x5de.0:vmov` (-2) |
| `g1@0x5f4.1:vmac.f` | `dm3` <- accumulator_carry `g1@0x5e2.1:vmac.f` (-5) | `x2` <- register_move `g1@0x58e.0:vmov` (-30) | `x7` <- activation_lane lane #0x4 `g1@0x5ee.1:vextbcst.16` (-2) |
| `g1@0x608.1:vmac.f` | `dm3` <- accumulator_carry `g1@0x5f4.1:vmac.f` (-5) | `x4` <- activation_lane lane #0x5 `g1@0x600.1:vextbcst.16` (-2) | `x7` <- activation_lane lane #0x4 `g1@0x5ee.1:vextbcst.16` (-7) |
| `g1@0x62a.1:vmac.f` | `dm3` <- accumulator_carry `g1@0x608.1:vmac.f` (-9) | `x10` <- register_move `g1@0x5de.0:vmov` (-21) | `x4` <- activation_lane lane #0x5 `g1@0x600.1:vextbcst.16` (-11) |
| `g1@0x640.1:vmac.f` | `dm4` <- accumulator_carry `g1@0x62a.1:vmac.f` (-6) | `x8` <- bf16_coeff `g1@0x636.0:vconv.bf16.fp32` (-4) | `x3` <- register_move `g1@0x636.1:vmov` (-3) |
| `g1@0x654.0:vmac.f` | `dm4` <- accumulator_carry `g1@0x640.1:vmac.f` (-4) | `x3` <- register_move `g1@0x650.0:vmov` (-1) | `x0` <- activation_lane lane #0x7 `g1@0x62a.0:vextbcst.16` (-11) |
| `g1@0x65e.1:vmac.f` | `dm4` <- accumulator_carry `g1@0x654.0:vmac.f` (-4) | `x5` <- register_move `g1@0x65a.0:vmov` (-2) | `x9` <- register_move `g1@0x65e.0:vmov` (-1) |
| `g1@0x66e.2:vmac.f` | `dm4` <- accumulator_carry `g1@0x65e.1:vmac.f` (-5) | `x5` <- register_move `g1@0x65a.0:vmov` (-7) | `x4` <- activation_lane lane #0x9 `g1@0x632.0:vextbcst.16` (-18) |
| `g1@0x686.2:vmac.f` | `dm4` <- accumulator_carry `g1@0x66e.2:vmac.f` (-7) | `x2` <- bf16_coeff `g1@0x614.0:vconv.bf16.fp32` (-33) | `x7` <- activation_lane lane #0xa `g1@0x640.0:vextbcst.16` (-21) |
| `g1@0x6a2.1:vmac.f` | `dm3` <- accumulator_carry `g1@0x686.2:vmac.f` (-7) | `x3` <- activation_lane lane #0xf `g1@0x6a2.0:vextbcst.16` (-1) | `x10` <- activation_lane lane #0xe `g1@0x69a.1:vextbcst.16` (-2) |
| `g1@0x6b2.1:vmac.f` | `dm1` <- accumulator_carry `g1@0x6a2.1:vmac.f` (-4) | `x8` <- register_move `g1@0x6ae.0:vmov` (-2) | `x4` <- activation_lane lane #0x15 `g1@0x6b2.0:vextbcst.16` (-1) |
| `g1@0x6c6.1:vmac.f` | `dm3` <- accumulator_carry `g1@0x6b2.1:vmac.f` (-5) | `x9` <- activation_lane lane #0x11 `g1@0x6c6.0:vextbcst.16` (-1) | `x5` <- register_move `g1@0x6c2.0:vmov` (-2) |
| `g1@0x6d6.1:vmac.f` | `dm3` <- accumulator_carry `g1@0x6c6.1:vmac.f` (-4) | `x1` <- activation_lane lane #0x12 `g1@0x6d2.0:vextbcst.16` (-2) | `x10` <- activation_lane lane #0xe `g1@0x69a.1:vextbcst.16` (-15) |
| `g1@0x6e6.1:vmac.f` | `dm3` <- accumulator_carry `g1@0x6d6.1:vmac.f` (-4) | `x8` <- activation_lane lane #0x13 `g1@0x6e6.0:vextbcst.16` (-1) | `x3` <- register_move `g1@0x6e2.0:vmov` (-2) |
| `g1@0x6f6.2:vmac.f` | `dm3` <- accumulator_carry `g1@0x6e6.1:vmac.f` (-5) | `x2` <- bf16_coeff `g1@0x6f6.0:vconv.bf16.fp32` (-2) | `x7` <- activation_lane lane #0x14 `g1@0x6f2.0:vextbcst.16` (-3) |
| `g1@0x70c.2:vmac.f` | `dm3` <- accumulator_carry `g1@0x6f6.2:vmac.f` (-6) | `x5` <- bf16_coeff `g1@0x70c.0:vconv.bf16.fp32` (-2) | `x9` <- activation_lane lane #0x11 `g1@0x6c6.0:vextbcst.16` (-20) |
| `g1@0x722.1:vmac.f` | `dm3` <- accumulator_carry `g1@0x70c.2:vmac.f` (-5) | `x0` <- activation_lane lane #0x19 `g1@0x71e.0:vextbcst.16` (-2) | `x1` <- activation_lane lane #0x12 `g1@0x6d2.0:vextbcst.16` (-22) |
| `g1@0x730.1:vmac.f` | `dm3` <- accumulator_carry `g1@0x722.1:vmac.f` (-4) | `x3` <- register_move `g1@0x6e2.0:vmov` (-22) | `x8` <- register_move `g1@0x72c.0:vmov` (-2) |
| `g1@0x744.1:vmac.f` | `dm3` <- accumulator_carry `g1@0x730.1:vmac.f` (-5) | `x2` <- register_move `g1@0x73c.1:vmov` (-2) | `x7` <- activation_lane lane #0x14 `g1@0x6f2.0:vextbcst.16` (-23) |
| `g1@0x758.1:vmac.f` | `dm3` <- accumulator_carry `g1@0x744.1:vmac.f` (-5) | `x2` <- register_move `g1@0x73c.1:vmov` (-7) | `x4` <- activation_lane lane #0x15 `g1@0x6b2.0:vextbcst.16` (-44) |
| `g1@0x76c.2:vmac.f` | `dm3` <- accumulator_carry `g1@0x758.1:vmac.f` (-6) | `x5` <- bf16_coeff `g1@0x764.0:vconv.bf16.fp32` (-4) | `x6` <- activation_lane lane #0x1b `g1@0x76c.1:vextbcst.16` (-1) |
| `g1@0x77e.1:vmac.f` | `dm2` <- accumulator_carry `g1@0x76c.2:vmac.f` (-4) | `x8` <- register_move `g1@0x77a.0:vmov` (-2) | `x9` <- register_move `g1@0x77e.0:vmov` (-1) |
| `g1@0x790.2:vmac.f` | `dm1` <- accumulator_carry `g1@0x77e.1:vmac.f` (-6) | `x3` <- register_move `g1@0x78a.1:vmov` (-3) | `x10` <- activation_lane lane #0x1d `g1@0x790.1:vextbcst.16` (-1) |
| `g1@0x7a0.3:vmac.f` | `dm1` <- accumulator_carry `g1@0x790.2:vmac.f` (-6) | `x3` <- register_move `g1@0x78a.1:vmov` (-9) | `x0` <- activation_lane lane #0x1e `g1@0x79c.0:vextbcst.16` (-4) |
| `g1@0x7b4.1:vmac.f` | `dm1` <- accumulator_carry `g1@0x7a0.3:vmac.f` (-4) | `x1` <- vbcst.16 `g1@0x7b0.0:vbcst.16` (-2) | `x2` <- vector_load `g1@0x790.0:vldb` (-12) |
| `g1@0x7c2.1:vmac.f` | `dm1` <- accumulator_carry `g1@0x7b4.1:vmac.f` (-4) | `x8` <- vector_load `g1@0x7a0.0:vlda` (-11) | `x6` <- vector_load `g1@0x7a0.1:vldb` (-10) |
| `g1@0x7d2.1:vmac.f` | `dm1` <- accumulator_carry `g1@0x7c2.1:vmac.f` (-4) | `x5` <- unpacked_q4 `g1@0x7bc.0:vunpack` (-7) | `x7` <- vector_load `g1@0x7b4.0:vldb` (-9) |

## Conclusion

- The steady template is not self-contained: several vector operands and accumulator sources are live before group1 starts.
- A production generator must model the boundary carry state before it emits even the first steady group.
- The useful abstraction is a scheduled operand graph, not a macro that repeats 32 lane broadcasts and MACs in source order.
