# MyLM Q4NX Operand Graph

Source disasm: `/tmp/mylm_qwen3_layer_L31_deep/disasm/c2r2.s`

This graph is pre-lane-select. Exp102 already validates the primitive `vmac.f #0x33c` lane model; this report tracks the half-register state fed into those MACs.

## Boundary State

| Boundary Into Group | Data Cells | Pointer/Scalar Cells | Signature |
| ---: | ---: | ---: | --- |
| 1 | 23 | 3 | `104ce8bf871f382b` |
| 2 | 23 | 3 | `5121c8604a8461c4` |
| 3 | 23 | 3 | `5121c8604a8461c4` |
| 4 | 23 | 3 | `5121c8604a8461c4` |
| 5 | 23 | 3 | `5121c8604a8461c4` |
| 6 | 23 | 3 | `5121c8604a8461c4` |
| 7 | 23 | 4 | `512c8be2d110c5ab` |

- Steady boundaries checked: `2, 3, 4, 5, 6`
- Steady data-cell signature stable: `True`

## Fill To Steady Boundary

| Cell | Producer | First Use |
| --- | --- | --- |
| `acc1.bmhh` | `g0@0x51e.1:vmac.f` | `g1@0x52a.1:vmac.f` |
| `acc1.bmhl` | `g0@0x51e.1:vmac.f` | `g1@0x52a.1:vmac.f` |
| `acc1.bmlh` | `g0@0x51e.1:vmac.f` | `g1@0x52a.1:vmac.f` |
| `acc1.bmll` | `g0@0x51e.1:vmac.f` | `g1@0x52a.1:vmac.f` |
| `acc4.bmhh` | `g0@0x41c.0:vsub.f` | `g1@0x570.2:vmac.f` |
| `acc4.bmhl` | `g0@0x41c.0:vsub.f` | `g1@0x570.2:vmac.f` |
| `acc4.bmlh` | `g0@0x41c.0:vsub.f` | `g1@0x570.2:vmac.f` |
| `acc4.bmll` | `g0@0x478.0:vmov` | `g1@0x570.2:vmac.f` |
| `vec0.hi` | `g0@0x51e.0:vldb` | `g1@0x53a.1:vmac.f` |
| `vec0.lo` | `g0@0x51e.0:vldb` | `g1@0x53a.1:vmac.f` |
| `vec10.hi` | `g0@0x51a.0:vunpack` | `g1@0x52a.1:vmac.f` |
| `vec10.lo` | `g0@0x51a.0:vunpack` | `g1@0x52a.1:vmac.f` |
| `vec2.hi` | `g0@0x4b2.1:vextbcst.16` | `g1@0x570.2:vmac.f` |
| `vec2.lo` | `g0@0x4dc.0:vldb` | `g1@0x570.2:vmac.f` |
| `vec3.hi` | `g0@0x490.0:vconv.bf16.fp32` | `g1@0x54a.0:vmac.f` |
| `vec3.lo` | `g0@0x4f8.0:vmov` | `g1@0x54a.0:vmac.f` |
| `vec4.hi` | `g0@0x4ba.0:vconv.bf16.fp32` | `g1@0x53a.1:vmac.f` |
| `vec4.lo` | `g0@0x4ba.0:vconv.bf16.fp32` | `g1@0x53a.1:vmac.f` |
| `vec6.hi` | `g0@0x4ba.1:vextbcst.16` | `g1@0x580.1:vmac.f` |
| `vec8.hi` | `g0@0x516.0:vunpack` | `g1@0x53a.0:vups.4x` |
| `vec8.lo` | `g0@0x516.0:vunpack` | `g1@0x53a.0:vups.4x` |
| `vec9.hi` | `g0@0x50e.0:vunpack` | `g1@0x52a.1:vmac.f` |
| `vec9.lo` | `g0@0x50e.0:vunpack` | `g1@0x52a.1:vmac.f` |

## First Steady To Steady Boundary

| Cell | Producer | First Use |
| --- | --- | --- |
| `acc1.bmhh` | `g1@0x7d2.1:vmac.f` | `g2@0x7de.1:vmac.f` |
| `acc1.bmhl` | `g1@0x7d2.1:vmac.f` | `g2@0x7de.1:vmac.f` |
| `acc1.bmlh` | `g1@0x7d2.1:vmac.f` | `g2@0x7de.1:vmac.f` |
| `acc1.bmll` | `g1@0x7d2.1:vmac.f` | `g2@0x7de.1:vmac.f` |
| `acc4.bmhh` | `g1@0x6ce.0:vsub.f` | `g2@0x824.2:vmac.f` |
| `acc4.bmhl` | `g1@0x6ce.0:vsub.f` | `g2@0x824.2:vmac.f` |
| `acc4.bmlh` | `g1@0x6ce.0:vsub.f` | `g2@0x824.2:vmac.f` |
| `acc4.bmll` | `g1@0x786.0:vmov` | `g2@0x824.2:vmac.f` |
| `vec0.hi` | `g1@0x7d2.0:vldb` | `g2@0x7ee.1:vmac.f` |
| `vec0.lo` | `g1@0x7d2.0:vldb` | `g2@0x7ee.1:vmac.f` |
| `vec10.hi` | `g1@0x7ce.0:vunpack` | `g2@0x7de.1:vmac.f` |
| `vec10.lo` | `g1@0x7ce.0:vunpack` | `g2@0x7de.1:vmac.f` |
| `vec2.hi` | `g1@0x764.1:vextbcst.16` | `g2@0x824.2:vmac.f` |
| `vec2.lo` | `g1@0x790.0:vldb` | `g2@0x824.2:vmac.f` |
| `vec3.hi` | `g1@0x73c.0:vconv.bf16.fp32` | `g2@0x7fe.0:vmac.f` |
| `vec3.lo` | `g1@0x7ac.0:vmov` | `g2@0x7fe.0:vmac.f` |
| `vec4.hi` | `g1@0x76c.0:vconv.bf16.fp32` | `g2@0x7ee.1:vmac.f` |
| `vec4.lo` | `g1@0x76c.0:vconv.bf16.fp32` | `g2@0x7ee.1:vmac.f` |
| `vec6.hi` | `g1@0x76c.1:vextbcst.16` | `g2@0x834.1:vmac.f` |
| `vec8.hi` | `g1@0x7ca.0:vunpack` | `g2@0x7ee.0:vups.4x` |
| `vec8.lo` | `g1@0x7ca.0:vunpack` | `g2@0x7ee.0:vups.4x` |
| `vec9.hi` | `g1@0x7c2.0:vunpack` | `g2@0x7de.1:vmac.f` |
| `vec9.lo` | `g1@0x7c2.0:vunpack` | `g2@0x7de.1:vmac.f` |

## Steady Group1 MAC Graph Summary

- `vmac.f`: `33`
- Mixed vector operands: `25` / `66`
- Vector operands with carry cells: `9` / `66`

| Vector Producer Pair | MAC Count |
| --- | ---: |
| `activation_lane + register_move+bf16_coeff` | 6 |
| `activation_lane + register_move+activation_lane` | 5 |
| `activation_lane + activation_lane` | 4 |
| `activation_lane + bf16_coeff` | 4 |
| `bf16_coeff + register_move+activation_lane` | 2 |
| `register_move+activation_lane + register_move+activation_lane` | 2 |
| `unpacked_q4 + unpacked_q4` | 1 |
| `bf16_coeff + vector_load` | 1 |
| `register_move+bf16_coeff + vector_load` | 1 |
| `bf16_coeff + vector_load+activation_lane` | 1 |
| `register_move+activation_lane + unpacked_q4` | 1 |
| `activation_lane + vector_load` | 1 |
| `register_move+activation_lane + register_move+bf16_coeff` | 1 |
| `scalar_broadcast + vector_load+activation_lane` | 1 |
| `vector_load + vector_load+activation_lane` | 1 |
| `unpacked_q4 + vector_load` | 1 |

## Steady Group1 MAC Graph

| MAC | Acc | Left | Right |
| --- | --- | --- | --- |
| `g1@0x52a.1:vmac.f` | `acc1.bmll:accumulator_carry!, acc1.bmlh:accumulator_carry!, acc1.bmhl:accumulator_carry!, acc1.bmhh:accumulator_carry!` | `vec9.lo:unpacked_q4!, vec9.hi:unpacked_q4!` | `vec10.lo:unpacked_q4!, vec10.hi:unpacked_q4!` |
| `g1@0x53a.1:vmac.f` | `acc1.bmll:vector_arith, acc1.bmlh:vector_arith, acc1.bmhl:vector_arith, acc1.bmhh:vector_arith` | `vec4.lo:bf16_coeff!, vec4.hi:bf16_coeff!` | `vec0.lo:vector_load!, vec0.hi:vector_load!` |
| `g1@0x54a.0:vmac.f` | `acc3.bmll:accumulator_carry, acc3.bmlh:accumulator_carry, acc3.bmhl:accumulator_carry, acc3.bmhh:accumulator_carry` | `vec3.lo:register_move!, vec3.hi:bf16_coeff!` | `vec11.lo:vector_load, vec11.hi:vector_load` |
| `g1@0x570.2:vmac.f` | `acc4.bmll:register_move!, acc4.bmlh:vector_arith!, acc4.bmhl:vector_arith!, acc4.bmhh:vector_arith!` | `vec4.lo:bf16_coeff, vec4.hi:bf16_coeff!` | `vec2.lo:vector_load!, vec2.hi:activation_lane#0x1a!` |
| `g1@0x580.1:vmac.f` | `acc4.bmll:accumulator_carry, acc4.bmlh:accumulator_carry, acc4.bmhl:accumulator_carry, acc4.bmhh:accumulator_carry` | `vec6.lo:register_move, vec6.hi:activation_lane#0x1b!` | `vec1.lo:unpacked_q4, vec1.hi:unpacked_q4` |
| `g1@0x5d0.2:vmac.f` | `acc3.bmll:vector_arith, acc3.bmlh:vector_arith, acc3.bmhl:vector_arith, acc3.bmhh:vector_arith` | `vec6.lo:vector_load, vec6.hi:vector_load` | `vec9.lo:activation_lane#0x1, vec9.hi:activation_lane#0x1` |
| `g1@0x5e2.1:vmac.f` | `acc3.bmll:accumulator_carry, acc3.bmlh:accumulator_carry, acc3.bmhl:accumulator_carry, acc3.bmhh:accumulator_carry` | `vec3.lo:bf16_coeff, vec3.hi:bf16_coeff` | `vec10.lo:register_move, vec10.hi:activation_lane#0x2` |
| `g1@0x5f4.1:vmac.f` | `acc3.bmll:accumulator_carry, acc3.bmlh:accumulator_carry, acc3.bmhl:accumulator_carry, acc3.bmhh:accumulator_carry` | `vec2.lo:register_move, vec2.hi:activation_lane#0x1a!` | `vec7.lo:activation_lane#0x4, vec7.hi:activation_lane#0x4` |
| `g1@0x608.1:vmac.f` | `acc3.bmll:accumulator_carry, acc3.bmlh:accumulator_carry, acc3.bmhl:accumulator_carry, acc3.bmhh:accumulator_carry` | `vec4.lo:activation_lane#0x5, vec4.hi:activation_lane#0x5` | `vec7.lo:activation_lane#0x4, vec7.hi:activation_lane#0x4` |
| `g1@0x62a.1:vmac.f` | `acc3.bmll:accumulator_carry, acc3.bmlh:accumulator_carry, acc3.bmhl:accumulator_carry, acc3.bmhh:accumulator_carry` | `vec10.lo:register_move, vec10.hi:activation_lane#0x2` | `vec4.lo:activation_lane#0x5, vec4.hi:activation_lane#0x5` |
| `g1@0x640.1:vmac.f` | `acc4.bmll:accumulator_carry, acc4.bmlh:accumulator_carry, acc4.bmhl:accumulator_carry, acc4.bmhh:accumulator_carry` | `vec8.lo:bf16_coeff, vec8.hi:bf16_coeff` | `vec3.lo:register_move, vec3.hi:activation_lane#0x6` |
| `g1@0x654.0:vmac.f` | `acc4.bmll:accumulator_carry, acc4.bmlh:accumulator_carry, acc4.bmhl:accumulator_carry, acc4.bmhh:accumulator_carry` | `vec3.lo:register_move, vec3.hi:activation_lane#0x6` | `vec0.lo:activation_lane#0x7, vec0.hi:activation_lane#0x7` |
| `g1@0x65e.1:vmac.f` | `acc4.bmll:accumulator_carry, acc4.bmlh:accumulator_carry, acc4.bmhl:accumulator_carry, acc4.bmhh:accumulator_carry` | `vec5.lo:register_move, vec5.hi:bf16_coeff` | `vec9.lo:register_move, vec9.hi:activation_lane#0x8` |
| `g1@0x66e.2:vmac.f` | `acc4.bmll:accumulator_carry, acc4.bmlh:accumulator_carry, acc4.bmhl:accumulator_carry, acc4.bmhh:accumulator_carry` | `vec5.lo:register_move, vec5.hi:bf16_coeff` | `vec4.lo:activation_lane#0x9, vec4.hi:activation_lane#0x9` |
| `g1@0x686.2:vmac.f` | `acc4.bmll:accumulator_carry, acc4.bmlh:accumulator_carry, acc4.bmhl:accumulator_carry, acc4.bmhh:accumulator_carry` | `vec2.lo:bf16_coeff, vec2.hi:bf16_coeff` | `vec7.lo:activation_lane#0xa, vec7.hi:activation_lane#0xa` |
| `g1@0x6a2.1:vmac.f` | `acc3.bmll:accumulator_carry, acc3.bmlh:accumulator_carry, acc3.bmhl:accumulator_carry, acc3.bmhh:accumulator_carry` | `vec3.lo:activation_lane#0xf, vec3.hi:activation_lane#0xf` | `vec10.lo:activation_lane#0xe, vec10.hi:activation_lane#0xe` |
| `g1@0x6b2.1:vmac.f` | `acc1.bmll:accumulator_carry, acc1.bmlh:accumulator_carry, acc1.bmhl:accumulator_carry, acc1.bmhh:accumulator_carry` | `vec8.lo:register_move, vec8.hi:bf16_coeff` | `vec4.lo:activation_lane#0x15, vec4.hi:activation_lane#0x15` |
| `g1@0x6c6.1:vmac.f` | `acc3.bmll:accumulator_carry, acc3.bmlh:accumulator_carry, acc3.bmhl:accumulator_carry, acc3.bmhh:accumulator_carry` | `vec9.lo:activation_lane#0x11, vec9.hi:activation_lane#0x11` | `vec5.lo:register_move, vec5.hi:activation_lane#0xd` |
| `g1@0x6d6.1:vmac.f` | `acc3.bmll:accumulator_carry, acc3.bmlh:accumulator_carry, acc3.bmhl:accumulator_carry, acc3.bmhh:accumulator_carry` | `vec1.lo:activation_lane#0x12, vec1.hi:activation_lane#0x12` | `vec10.lo:activation_lane#0xe, vec10.hi:activation_lane#0xe` |
| `g1@0x6e6.1:vmac.f` | `acc3.bmll:accumulator_carry, acc3.bmlh:accumulator_carry, acc3.bmhl:accumulator_carry, acc3.bmhh:accumulator_carry` | `vec8.lo:activation_lane#0x13, vec8.hi:activation_lane#0x13` | `vec3.lo:register_move, vec3.hi:activation_lane#0xf` |
| `g1@0x6f6.2:vmac.f` | `acc3.bmll:accumulator_carry, acc3.bmlh:accumulator_carry, acc3.bmhl:accumulator_carry, acc3.bmhh:accumulator_carry` | `vec2.lo:bf16_coeff, vec2.hi:bf16_coeff` | `vec7.lo:activation_lane#0x14, vec7.hi:activation_lane#0x14` |
| `g1@0x70c.2:vmac.f` | `acc3.bmll:accumulator_carry, acc3.bmlh:accumulator_carry, acc3.bmhl:accumulator_carry, acc3.bmhh:accumulator_carry` | `vec5.lo:bf16_coeff, vec5.hi:bf16_coeff` | `vec9.lo:activation_lane#0x11, vec9.hi:activation_lane#0x11` |
| `g1@0x722.1:vmac.f` | `acc3.bmll:accumulator_carry, acc3.bmlh:accumulator_carry, acc3.bmhl:accumulator_carry, acc3.bmhh:accumulator_carry` | `vec0.lo:activation_lane#0x19, vec0.hi:activation_lane#0x19` | `vec1.lo:activation_lane#0x12, vec1.hi:activation_lane#0x12` |
| `g1@0x730.1:vmac.f` | `acc3.bmll:accumulator_carry, acc3.bmlh:accumulator_carry, acc3.bmhl:accumulator_carry, acc3.bmhh:accumulator_carry` | `vec3.lo:register_move, vec3.hi:activation_lane#0xf` | `vec8.lo:register_move, vec8.hi:activation_lane#0x13` |
| `g1@0x744.1:vmac.f` | `acc3.bmll:accumulator_carry, acc3.bmlh:accumulator_carry, acc3.bmhl:accumulator_carry, acc3.bmhh:accumulator_carry` | `vec2.lo:register_move, vec2.hi:bf16_coeff` | `vec7.lo:activation_lane#0x14, vec7.hi:activation_lane#0x14` |
| `g1@0x758.1:vmac.f` | `acc3.bmll:accumulator_carry, acc3.bmlh:accumulator_carry, acc3.bmhl:accumulator_carry, acc3.bmhh:accumulator_carry` | `vec2.lo:register_move, vec2.hi:bf16_coeff` | `vec4.lo:activation_lane#0x15, vec4.hi:activation_lane#0x15` |
| `g1@0x76c.2:vmac.f` | `acc3.bmll:accumulator_carry, acc3.bmlh:accumulator_carry, acc3.bmhl:accumulator_carry, acc3.bmhh:accumulator_carry` | `vec5.lo:bf16_coeff, vec5.hi:bf16_coeff` | `vec6.lo:activation_lane#0x1b, vec6.hi:activation_lane#0x1b` |
| `g1@0x77e.1:vmac.f` | `acc2.bmll:accumulator_carry, acc2.bmlh:accumulator_carry, acc2.bmhl:accumulator_carry, acc2.bmhh:accumulator_carry` | `vec8.lo:register_move, vec8.hi:activation_lane#0x13` | `vec9.lo:register_move, vec9.hi:activation_lane#0x17` |
| `g1@0x790.2:vmac.f` | `acc1.bmll:accumulator_carry, acc1.bmlh:accumulator_carry, acc1.bmhl:accumulator_carry, acc1.bmhh:accumulator_carry` | `vec3.lo:register_move, vec3.hi:bf16_coeff` | `vec10.lo:activation_lane#0x1d, vec10.hi:activation_lane#0x1d` |
| `g1@0x7a0.3:vmac.f` | `acc1.bmll:accumulator_carry, acc1.bmlh:accumulator_carry, acc1.bmhl:accumulator_carry, acc1.bmhh:accumulator_carry` | `vec3.lo:register_move, vec3.hi:bf16_coeff` | `vec0.lo:activation_lane#0x1e, vec0.hi:activation_lane#0x1e` |
| `g1@0x7b4.1:vmac.f` | `acc1.bmll:accumulator_carry, acc1.bmlh:accumulator_carry, acc1.bmhl:accumulator_carry, acc1.bmhh:accumulator_carry` | `vec1.lo:scalar_broadcast, vec1.hi:scalar_broadcast` | `vec2.lo:vector_load, vec2.hi:activation_lane#0x1a` |
| `g1@0x7c2.1:vmac.f` | `acc1.bmll:accumulator_carry, acc1.bmlh:accumulator_carry, acc1.bmhl:accumulator_carry, acc1.bmhh:accumulator_carry` | `vec8.lo:vector_load, vec8.hi:vector_load` | `vec6.lo:vector_load, vec6.hi:activation_lane#0x1b` |
| `g1@0x7d2.1:vmac.f` | `acc1.bmll:accumulator_carry, acc1.bmlh:accumulator_carry, acc1.bmhl:accumulator_carry, acc1.bmhh:accumulator_carry` | `vec5.lo:unpacked_q4, vec5.hi:unpacked_q4` | `vec7.lo:vector_load, vec7.hi:vector_load` |

## Generator Implication

- `fill -> steady` has a concrete boundary state: acc, packed, coefficient, activation, and pointer cells are already live before group1 starts.
- The steady-to-steady data-cell signature is stable for groups 2..6, so a generator can model one steady state transition instead of eight independent groups.
- The next production experiment should consume the JSON artifact and emit a small assembly template from graph records, with exact numerical validation kept separate from the MyLM-like group-correction contract decision.
