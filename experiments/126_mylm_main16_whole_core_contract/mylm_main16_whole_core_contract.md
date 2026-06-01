# MyLM Main16 Whole-Core Contract

This experiment defines the next direction after the failed partial
QKV-only nocall attempt. The goal is a MyLM-style main16 role program,
not another temporary scheduler variant.

## Active IRON Path

- Has active Q4 asm body `q4nx_chunk_accum_asm_zol`: `True`
- Has active layer scheduler `q4nx_main16_layer_scheduler`: `True`
- Contains rejected nocall symbol: `False`
- Static helper-call relocations to `q4nx_chunk_accum_asm_zol`: `1`

| Active Q4 op | Count |
| --- | ---: |
| `vmac.f` | 64 |
| `vextbcst.16` | 64 |
| `vups.4x` | 0 |
| `vups.2x` | 32 |
| `vunpack` | 64 |
| `vmul.f` | 64 |
| `vconv.bf16.fp32` | 160 |
| `vlda` | 2 |
| `vldb` | 66 |
| `vst` | 2 |
| `add.nc	lc` | 2 |
| `movxm	ls` | 2 |
| `movxm	le` | 2 |

## MyLM Whole-Core Shape

| Segment | Offset | Bytes |
| --- | ---: | ---: |
| `entry/setup` | `0x0000` | 492 |
| `shared Q4NX microkernel` | `0x01f0` | 5760 |
| `Q/K/V body` | `0x1870` | 1544 |
| `O body` | `0x1e80` | 1544 |
| `up/gate body` | `0x2490` | 1544 |
| `down body` | `0x2aa0` | 1560 |
| `alternate body` | `0x30c0` | 1544 |
| `dispatcher` | `0x36d0` | 504 |
| `tail/helper` | `0x38d0` | 324 |

## MyLM Phase Bodies

| Phase | Body | Header | Records/tile |
| --- | ---: | --- | ---: |
| `Q/K/V` | `0x1870` | `0x1` | 12 |
| `O` | `0x1e80` | `0x4` | 8 |
| `up/gate` | `0x2490` | `0x8` | 48 |
| `down` | `0x2aa0` | `0x4` | 8 |

## MyLM Q4 Body Contract

- Hot-loop slots: `1532`
- Static `vmac.f`: `264`
- Group MAC shape: `28,33,33,33,33,33,33,38`

| MyLM Q4 op | Count |
| --- | ---: |
| `add.nc` | 4 |
| `lda.s16` | 8 |
| `lshl` | 2 |
| `mov` | 4 |
| `movx` | 1 |
| `nop` | 108 |
| `nopa` | 1 |
| `nopb` | 1 |
| `paddb` | 2 |
| `vadd` | 64 |
| `vbcst.16` | 8 |
| `vconv.bf16.fp32` | 136 |
| `vextbcst.16` | 256 |
| `vlda` | 11 |
| `vldb` | 46 |
| `vmac.f` | 264 |
| `vmov` | 400 |
| `vmov.d` | 16 |
| `vmul.f` | 8 |
| `vsub.f` | 64 |
| `vunpack` | 64 |
| `vups.4x` | 64 |

## Generator Sections

| Section | Groups | Template | MACs/group | Live in | Live out | Signature |
| --- | --- | ---: | ---: | --- | --- | --- |
| `fill` | `0` | 0 | 28 | `entry` | `boundary1` | `` |
| `fill_to_steady` | `1` | 1 | 33 | `boundary1` | `boundary2` | `070ae3b9658fd953` |
| `steady_to_steady` | `2,3,4,5` | 2 | 33 | `boundary2` | `boundary6` | `5466ff6129c65f6b` |
| `pre_drain` | `6` | 6 | 33 | `boundary6` | `boundary7` | `` |
| `drain` | `7` | 7 | 38 | `boundary7` | `exit` | `` |

## Header Policy

Current IRON downstream routing uses packet/header IDs:

| Phase | IRON header packet id |
| --- | ---: |
| `Q` | 10 |
| `K` | 11 |
| `V` | 12 |
| `O` | 13 |
| `FFN` | 14 |
| `down` | 15 |

MyLM's `0x1/0x4/0x8` phase headers are part of the raw program
contract, but they are not a drop-in replacement for current IRON
row1/c1r1/c1r3 routing. The first whole-core replacement should keep
IRON headers while adopting the MyLM phase/body/register scheduling
shape.

## Decision

- Active path: keep the single q4nx_main16_layer_scheduler(..., phase_limit) entry calling q4nx_chunk_accum_asm_zol until the generated whole-main16 replacement passes the QKV prefix gate.
- Rejected path: do not re-enable the partial q4nx_main16_qkv_scheduler_nocall path; it removed the helper statically but timed out on full-layer-qkv-prefix.
- Next path: generate one main16 role program with MyLM-style sections: shared Q4NX body, Q/K/V body, O body, up/gate body, down body, and dispatcher.
- Header policy: preserve IRON compact headers 10..15 at the row1/c1r1 boundary for the first replacement; MyLM's 0x1/0x4/0x8 headers are a raw-program clue, not a drop-in ABI for current IRON downstream routing.
- First gate: full-layer-qkv-prefix token31 must pass before enabling the generated main16 program in any full-decode path.

## Direction

The next useful implementation is a generated main16 role program with
one shared Q4 body and phase bodies that own the lock, record, and
phase-order protocol. A QKV-only linked asm scheduler is too small a
slice: it changes the call boundary without giving the code generator
the fixed register/control plan that makes MyLM fast.

