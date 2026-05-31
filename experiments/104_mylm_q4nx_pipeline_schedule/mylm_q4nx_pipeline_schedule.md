# MyLM Q4NX Pipeline Schedule

Source disasm: `/tmp/mylm_qwen3_layer_L31_deep/disasm/c2r2.s`

This is a compact schedule artifact for writing the next Q4NX assembly
generator. It intentionally avoids a full instruction listing and keeps
only the facts that affect register lifetime and software pipelining.

## Group Shape

| Group | Range | Slots | Key Ops | Incoming Families |
| ---: | --- | ---: | --- | --- |
| 0 | `0x260..0x52a` | 192 | `vmac.f=28`, `vextbcst.16=32`, `vconv.bf16.fp32=16`, `vups.4x=8`, `vunpack=12`, `vldb=6`, `vlda=5`, `lda.s16=2`, `vbcst.16=1` | `p0`, `p1`, `r18`, `r19`, `r2`, `r16`, `r17`, `r1`, `unpacksign0`, `r20`, `r3`, `p5`, `p2`, `p4`, `p3`, `s0`, `upssign0`, `acc0`, `r0`, `r5`, ... |
| 1 | `0x52a..0x7de` | 189 | `vmac.f=33`, `vextbcst.16=32`, `vconv.bf16.fp32=17`, `vups.4x=8`, `vunpack=8`, `vldb=6`, `vlda=1`, `lda.s16=1`, `vbcst.16=1` | `p1`, `acc1`, `vec9`, `vec10`, `r4`, `s0`, `upssign0`, `acc0`, `r0`, `vec8`, `vec4`, `vec0`, `r5`, `vec3`, `unpacksign0`, `acc4`, `vec2`, `p0`, `p3`, `p5`, ... |
| 2 | `0x7de..0xa92` | 189 | `vmac.f=33`, `vextbcst.16=32`, `vconv.bf16.fp32=17`, `vups.4x=8`, `vunpack=8`, `vldb=6`, `vlda=1`, `lda.s16=1`, `vbcst.16=1` | `p1`, `acc1`, `vec9`, `vec10`, `r4`, `s0`, `upssign0`, `acc0`, `r0`, `vec8`, `vec4`, `vec0`, `r5`, `vec3`, `unpacksign0`, `acc4`, `vec2`, `p0`, `p3`, `p5`, ... |
| 3 | `0xa92..0xd46` | 189 | `vmac.f=33`, `vextbcst.16=32`, `vconv.bf16.fp32=17`, `vups.4x=8`, `vunpack=8`, `vldb=6`, `vlda=1`, `lda.s16=1`, `vbcst.16=1` | `p1`, `acc1`, `vec9`, `vec10`, `r4`, `s0`, `upssign0`, `acc0`, `r0`, `vec8`, `vec4`, `vec0`, `r5`, `vec3`, `unpacksign0`, `acc4`, `vec2`, `p0`, `p3`, `p5`, ... |
| 4 | `0xd46..0xffa` | 189 | `vmac.f=33`, `vextbcst.16=32`, `vconv.bf16.fp32=17`, `vups.4x=8`, `vunpack=8`, `vldb=6`, `vlda=1`, `lda.s16=1`, `vbcst.16=1` | `p1`, `acc1`, `vec9`, `vec10`, `r4`, `s0`, `upssign0`, `acc0`, `r0`, `vec8`, `vec4`, `vec0`, `r5`, `vec3`, `unpacksign0`, `acc4`, `vec2`, `p0`, `p3`, `p5`, ... |
| 5 | `0xffa..0x12ae` | 189 | `vmac.f=33`, `vextbcst.16=32`, `vconv.bf16.fp32=17`, `vups.4x=8`, `vunpack=8`, `vldb=6`, `vlda=1`, `lda.s16=1`, `vbcst.16=1` | `p1`, `acc1`, `vec9`, `vec10`, `r4`, `s0`, `upssign0`, `acc0`, `r0`, `vec8`, `vec4`, `vec0`, `r5`, `vec3`, `unpacksign0`, `acc4`, `vec2`, `p0`, `p3`, `p5`, ... |
| 6 | `0x12ae..0x1566` | 190 | `vmac.f=33`, `vextbcst.16=32`, `vconv.bf16.fp32=17`, `vups.4x=8`, `vunpack=8`, `vldb=6`, `vlda=1`, `lda.s16=1`, `vbcst.16=1` | `p1`, `acc1`, `vec9`, `vec10`, `r4`, `s0`, `upssign0`, `acc0`, `r0`, `vec8`, `vec4`, `vec0`, `r5`, `vec3`, `unpacksign0`, `acc4`, `vec2`, `p0`, `p3`, `p5`, ... |
| 7 | `0x1566..0x1850` | 205 | `vmac.f=38`, `vextbcst.16=32`, `vconv.bf16.fp32=18`, `vups.4x=8`, `vunpack=4`, `vldb=4`, `vbcst.16=1` | `p1`, `acc1`, `vec9`, `vec10`, `r4`, `s0`, `upssign0`, `acc0`, `r0`, `vec8`, `vec4`, `vec0`, `r5`, `vec3`, `unpacksign0`, `acc4`, `vec2`, `p0`, `p5`, `r16`, ... |

## Cross-Group Carry

| Boundary Into Group | Family | Previous Def | First Use Before Local Def |
| ---: | --- | --- | --- |
| 1 | `acc1` | `g0@0x51e.1:vmac.f` | `g1@0x52a.1:vmac.f` |
| 1 | `acc4` | `g0@0x478.0:vmov` | `g1@0x570.2:vmac.f` |
| 1 | `p3` | `g0@0x4d6.0:lda.s16` | `g1@0x78a.0:lda.s16` |
| 1 | `p4` | `g0@0x284.2:add.nc` | `g1@0x7a0.1:vldb` |
| 1 | `p5` | `g0@0x26c.3:add.nc` | `g1@0x790.0:vldb` |
| 1 | `vec0` | `g0@0x51e.0:vldb` | `g1@0x53a.1:vmac.f` |
| 1 | `vec10` | `g0@0x51a.0:vunpack` | `g1@0x52a.1:vmac.f` |
| 1 | `vec2` | `g0@0x4dc.0:vldb` | `g1@0x570.2:vmac.f` |
| 1 | `vec3` | `g0@0x4f8.0:vmov` | `g1@0x54a.0:vmac.f` |
| 1 | `vec4` | `g0@0x4ba.0:vconv.bf16.fp32` | `g1@0x53a.1:vmac.f` |
| 1 | `vec8` | `g0@0x516.0:vunpack` | `g1@0x53a.0:vups.4x` |
| 1 | `vec9` | `g0@0x50e.0:vunpack` | `g1@0x52a.1:vmac.f` |
| 2 | `acc1` | `g1@0x7d2.1:vmac.f` | `g2@0x7de.1:vmac.f` |
| 2 | `acc4` | `g1@0x786.0:vmov` | `g2@0x824.2:vmac.f` |
| 2 | `p3` | `g1@0x78a.0:lda.s16` | `g2@0xa3e.0:lda.s16` |
| 2 | `p4` | `g0@0x284.2:add.nc` | `g2@0xa54.1:vldb` |
| 2 | `p5` | `g0@0x26c.3:add.nc` | `g2@0xa44.0:vldb` |
| 2 | `vec0` | `g1@0x7d2.0:vldb` | `g2@0x7ee.1:vmac.f` |
| 2 | `vec10` | `g1@0x7ce.0:vunpack` | `g2@0x7de.1:vmac.f` |
| 2 | `vec2` | `g1@0x790.0:vldb` | `g2@0x824.2:vmac.f` |
| 2 | `vec3` | `g1@0x7ac.0:vmov` | `g2@0x7fe.0:vmac.f` |
| 2 | `vec4` | `g1@0x76c.0:vconv.bf16.fp32` | `g2@0x7ee.1:vmac.f` |
| 2 | `vec8` | `g1@0x7ca.0:vunpack` | `g2@0x7ee.0:vups.4x` |
| 2 | `vec9` | `g1@0x7c2.0:vunpack` | `g2@0x7de.1:vmac.f` |
| 3 | `acc1` | `g2@0xa86.1:vmac.f` | `g3@0xa92.1:vmac.f` |
| 3 | `acc4` | `g2@0xa3a.0:vmov` | `g3@0xad8.2:vmac.f` |
| 3 | `p3` | `g2@0xa3e.0:lda.s16` | `g3@0xcf2.0:lda.s16` |
| 3 | `p4` | `g0@0x284.2:add.nc` | `g3@0xd08.1:vldb` |
| 3 | `p5` | `g0@0x26c.3:add.nc` | `g3@0xcf8.0:vldb` |
| 3 | `vec0` | `g2@0xa86.0:vldb` | `g3@0xaa2.1:vmac.f` |
| 3 | `vec10` | `g2@0xa82.0:vunpack` | `g3@0xa92.1:vmac.f` |
| 3 | `vec2` | `g2@0xa44.0:vldb` | `g3@0xad8.2:vmac.f` |
| 3 | `vec3` | `g2@0xa60.0:vmov` | `g3@0xab2.0:vmac.f` |
| 3 | `vec4` | `g2@0xa20.0:vconv.bf16.fp32` | `g3@0xaa2.1:vmac.f` |
| 3 | `vec8` | `g2@0xa7e.0:vunpack` | `g3@0xaa2.0:vups.4x` |
| 3 | `vec9` | `g2@0xa76.0:vunpack` | `g3@0xa92.1:vmac.f` |
| 4 | `acc1` | `g3@0xd3a.1:vmac.f` | `g4@0xd46.1:vmac.f` |
| 4 | `acc4` | `g3@0xcee.0:vmov` | `g4@0xd8c.2:vmac.f` |
| 4 | `p3` | `g3@0xcf2.0:lda.s16` | `g4@0xfa6.0:lda.s16` |
| 4 | `p4` | `g0@0x284.2:add.nc` | `g4@0xfbc.1:vldb` |
| 4 | `p5` | `g0@0x26c.3:add.nc` | `g4@0xfac.0:vldb` |
| 4 | `vec0` | `g3@0xd3a.0:vldb` | `g4@0xd56.1:vmac.f` |
| 4 | `vec10` | `g3@0xd36.0:vunpack` | `g4@0xd46.1:vmac.f` |
| 4 | `vec2` | `g3@0xcf8.0:vldb` | `g4@0xd8c.2:vmac.f` |
| 4 | `vec3` | `g3@0xd14.0:vmov` | `g4@0xd66.0:vmac.f` |
| 4 | `vec4` | `g3@0xcd4.0:vconv.bf16.fp32` | `g4@0xd56.1:vmac.f` |
| 4 | `vec8` | `g3@0xd32.0:vunpack` | `g4@0xd56.0:vups.4x` |
| 4 | `vec9` | `g3@0xd2a.0:vunpack` | `g4@0xd46.1:vmac.f` |
| 5 | `acc1` | `g4@0xfee.1:vmac.f` | `g5@0xffa.1:vmac.f` |
| 5 | `acc4` | `g4@0xfa2.0:vmov` | `g5@0x1040.2:vmac.f` |
| 5 | `p3` | `g4@0xfa6.0:lda.s16` | `g5@0x125a.0:lda.s16` |
| 5 | `p4` | `g0@0x284.2:add.nc` | `g5@0x1270.1:vldb` |
| 5 | `p5` | `g0@0x26c.3:add.nc` | `g5@0x1260.0:vldb` |
| 5 | `vec0` | `g4@0xfee.0:vldb` | `g5@0x100a.1:vmac.f` |
| 5 | `vec10` | `g4@0xfea.0:vunpack` | `g5@0xffa.1:vmac.f` |
| 5 | `vec2` | `g4@0xfac.0:vldb` | `g5@0x1040.2:vmac.f` |
| 5 | `vec3` | `g4@0xfc8.0:vmov` | `g5@0x101a.0:vmac.f` |
| 5 | `vec4` | `g4@0xf88.0:vconv.bf16.fp32` | `g5@0x100a.1:vmac.f` |
| 5 | `vec8` | `g4@0xfe6.0:vunpack` | `g5@0x100a.0:vups.4x` |
| 5 | `vec9` | `g4@0xfde.0:vunpack` | `g5@0xffa.1:vmac.f` |
| 6 | `acc1` | `g5@0x12a2.1:vmac.f` | `g6@0x12ae.1:vmac.f` |
| 6 | `acc4` | `g5@0x1256.0:vmov` | `g6@0x12f4.2:vmac.f` |
| 6 | `p3` | `g5@0x125a.0:lda.s16` | `g6@0x150e.0:lda.s16` |
| 6 | `p4` | `g0@0x284.2:add.nc` | `g6@0x1526.1:vldb` |
| 6 | `p5` | `g0@0x26c.3:add.nc` | `g6@0x1514.0:vldb` |
| 6 | `vec0` | `g5@0x12a2.0:vldb` | `g6@0x12be.1:vmac.f` |
| 6 | `vec10` | `g5@0x129e.0:vunpack` | `g6@0x12ae.1:vmac.f` |
| 6 | `vec2` | `g5@0x1260.0:vldb` | `g6@0x12f4.2:vmac.f` |
| 6 | `vec3` | `g5@0x127c.0:vmov` | `g6@0x12ce.0:vmac.f` |
| 6 | `vec4` | `g5@0x123c.0:vconv.bf16.fp32` | `g6@0x12be.1:vmac.f` |
| 6 | `vec8` | `g5@0x129a.0:vunpack` | `g6@0x12be.0:vups.4x` |
| 6 | `vec9` | `g5@0x1292.0:vunpack` | `g6@0x12ae.1:vmac.f` |
| 7 | `acc1` | `g6@0x155a.1:vmac.f` | `g7@0x1566.1:vmac.f` |
| 7 | `acc4` | `g6@0x150a.0:vmov` | `g7@0x15ae.2:vmac.f` |
| 7 | `p4` | `g0@0x284.2:add.nc` | `g7@0x17de.0:vldb` |
| 7 | `p5` | `g0@0x26c.3:add.nc` | `g7@0x17ca.0:vldb` |
| 7 | `r16` | `g0@0x260.3:add.nc` | `g7@0x17d4.0:mov` |
| 7 | `r7` | `g6@0x150e.0:lda.s16` | `g7@0x17ec.0:vbcst.16` |
| 7 | `vec0` | `g6@0x155a.0:vldb` | `g7@0x1578.1:vmac.f` |
| 7 | `vec10` | `g6@0x1556.0:vunpack` | `g7@0x1566.1:vmac.f` |
| 7 | `vec2` | `g6@0x1514.0:vldb` | `g7@0x15ae.2:vmac.f` |
| 7 | `vec3` | `g6@0x1532.0:vmov` | `g7@0x1588.0:vmac.f` |
| 7 | `vec4` | `g6@0x14f0.0:vconv.bf16.fp32` | `g7@0x1578.1:vmac.f` |
| 7 | `vec8` | `g6@0x1552.0:vunpack` | `g7@0x1578.0:vups.4x` |
| 7 | `vec9` | `g6@0x154a.0:vunpack` | `g7@0x1566.1:vmac.f` |

## Vext Consumer Distance

- Matched `vextbcst.16` consumers: `200`
- This is a conservative first-consumer heuristic. Unmatched
  broadcasts are expected until the full alias/lane model is
  decoded.
- Distance is counted in parsed instruction slots between the broadcast
  slot and the first later `vmac.f` slot that consumes the same vector
  family.

| Distance Slots | Count |
| ---: | ---: |
| 1 | 48 |
| 2 | 40 |
| 3 | 9 |
| 4 | 7 |
| 10 | 8 |
| 11 | 16 |
| 14 | 8 |
| 18 | 8 |
| 20 | 1 |
| 21 | 8 |
| 22 | 1 |
| 23 | 6 |
| 24 | 8 |
| 26 | 1 |
| 27 | 6 |
| 28 | 9 |
| 33 | 1 |
| 36 | 8 |
| 38 | 7 |

## First 48 Vext Consumers

| Group | Lane | Vector | Def | Use | Distance |
| ---: | ---: | --- | --- | --- | ---: |
| 0 | `0x1` | `x9` | `g0@0x29a.0:vextbcst.16` | `g0@0x31e.2:vmac.f` | 33 |
| 0 | `0x2` | `x10` | `g0@0x30a.0:vextbcst.16` | `g0@0x330.1:vmac.f` | 11 |
| 0 | `0x4` | `x7` | `g0@0x33c.1:vextbcst.16` | `g0@0x342.1:vmac.f` | 2 |
| 0 | `0x5` | `x4` | `g0@0x34e.1:vextbcst.16` | `g0@0x356.1:vmac.f` | 2 |
| 0 | `0x6` | `x3` | `g0@0x36c.0:vextbcst.16` | `g0@0x38e.1:vmac.f` | 10 |
| 0 | `0x7` | `x0` | `g0@0x378.0:vextbcst.16` | `g0@0x3a2.0:vmac.f` | 11 |
| 0 | `0x8` | `x9` | `g0@0x328.0:vextbcst.16` | `g0@0x3ac.1:vmac.f` | 36 |
| 0 | `0x9` | `x4` | `g0@0x380.0:vextbcst.16` | `g0@0x3bc.2:vmac.f` | 18 |
| 0 | `0xa` | `x7` | `g0@0x38e.0:vextbcst.16` | `g0@0x3d4.2:vmac.f` | 21 |
| 0 | `0xf` | `x3` | `g0@0x3f0.0:vextbcst.16` | `g0@0x3f0.1:vmac.f` | 1 |
| 0 | `0xe` | `x10` | `g0@0x3e8.1:vextbcst.16` | `g0@0x3f0.1:vmac.f` | 2 |
| 0 | `0x15` | `x4` | `g0@0x400.0:vextbcst.16` | `g0@0x400.1:vmac.f` | 1 |
| 0 | `0x11` | `x9` | `g0@0x414.0:vextbcst.16` | `g0@0x414.1:vmac.f` | 1 |
| 0 | `0xd` | `x5` | `g0@0x3de.1:vextbcst.16` | `g0@0x414.1:vmac.f` | 14 |
| 0 | `0x12` | `x1` | `g0@0x420.0:vextbcst.16` | `g0@0x424.1:vmac.f` | 2 |
| 0 | `0x13` | `x8` | `g0@0x434.0:vextbcst.16` | `g0@0x434.1:vmac.f` | 1 |
| 0 | `0x14` | `x7` | `g0@0x440.0:vextbcst.16` | `g0@0x444.2:vmac.f` | 3 |
| 0 | `0x19` | `x0` | `g0@0x46c.0:vextbcst.16` | `g0@0x470.1:vmac.f` | 2 |
| 0 | `0x1b` | `x6` | `g0@0x4ba.1:vextbcst.16` | `g0@0x4ba.2:vmac.f` | 1 |
| 0 | `0x17` | `x9` | `g0@0x464.0:vextbcst.16` | `g0@0x4cc.1:vmac.f` | 28 |
| 0 | `0x1d` | `x10` | `g0@0x4dc.1:vextbcst.16` | `g0@0x4dc.2:vmac.f` | 1 |
| 0 | `0x1e` | `x0` | `g0@0x4e8.0:vextbcst.16` | `g0@0x4ec.3:vmac.f` | 4 |
| 0 | `0x1a` | `x2` | `g0@0x4b2.1:vextbcst.16` | `g0@0x500.1:vmac.f` | 23 |
| 0 | `0x1c` | `x7` | `g0@0x4c4.0:vextbcst.16` | `g0@0x51e.1:vmac.f` | 27 |
| 0 | `0x1f` | `x11` | `g0@0x4ec.2:vextbcst.16` | `g1@0x54a.0:vmac.f` | 24 |
| 1 | `0x1` | `x9` | `g1@0x556.0:vextbcst.16` | `g1@0x5d0.2:vmac.f` | 38 |
| 1 | `0x2` | `x10` | `g1@0x5bc.0:vextbcst.16` | `g1@0x5e2.1:vmac.f` | 11 |
| 1 | `0x4` | `x7` | `g1@0x5ee.1:vextbcst.16` | `g1@0x5f4.1:vmac.f` | 2 |
| 1 | `0x5` | `x4` | `g1@0x600.1:vextbcst.16` | `g1@0x608.1:vmac.f` | 2 |
| 1 | `0x6` | `x3` | `g1@0x61e.0:vextbcst.16` | `g1@0x640.1:vmac.f` | 10 |
| 1 | `0x7` | `x0` | `g1@0x62a.0:vextbcst.16` | `g1@0x654.0:vmac.f` | 11 |
| 1 | `0x8` | `x9` | `g1@0x5da.0:vextbcst.16` | `g1@0x65e.1:vmac.f` | 36 |
| 1 | `0x9` | `x4` | `g1@0x632.0:vextbcst.16` | `g1@0x66e.2:vmac.f` | 18 |
| 1 | `0xa` | `x7` | `g1@0x640.0:vextbcst.16` | `g1@0x686.2:vmac.f` | 21 |
| 1 | `0xf` | `x3` | `g1@0x6a2.0:vextbcst.16` | `g1@0x6a2.1:vmac.f` | 1 |
| 1 | `0xe` | `x10` | `g1@0x69a.1:vextbcst.16` | `g1@0x6a2.1:vmac.f` | 2 |
| 1 | `0x15` | `x4` | `g1@0x6b2.0:vextbcst.16` | `g1@0x6b2.1:vmac.f` | 1 |
| 1 | `0x11` | `x9` | `g1@0x6c6.0:vextbcst.16` | `g1@0x6c6.1:vmac.f` | 1 |
| 1 | `0xd` | `x5` | `g1@0x690.1:vextbcst.16` | `g1@0x6c6.1:vmac.f` | 14 |
| 1 | `0x12` | `x1` | `g1@0x6d2.0:vextbcst.16` | `g1@0x6d6.1:vmac.f` | 2 |
| 1 | `0x13` | `x8` | `g1@0x6e6.0:vextbcst.16` | `g1@0x6e6.1:vmac.f` | 1 |
| 1 | `0x14` | `x7` | `g1@0x6f2.0:vextbcst.16` | `g1@0x6f6.2:vmac.f` | 3 |
| 1 | `0x19` | `x0` | `g1@0x71e.0:vextbcst.16` | `g1@0x722.1:vmac.f` | 2 |
| 1 | `0x1b` | `x6` | `g1@0x76c.1:vextbcst.16` | `g1@0x76c.2:vmac.f` | 1 |
| 1 | `0x17` | `x9` | `g1@0x716.0:vextbcst.16` | `g1@0x77e.1:vmac.f` | 28 |
| 1 | `0x1d` | `x10` | `g1@0x790.1:vextbcst.16` | `g1@0x790.2:vmac.f` | 1 |
| 1 | `0x1e` | `x0` | `g1@0x79c.0:vextbcst.16` | `g1@0x7a0.3:vmac.f` | 4 |
| 1 | `0x1a` | `x2` | `g1@0x764.1:vextbcst.16` | `g1@0x7b4.1:vmac.f` | 23 |

## Conclusion

- Group0 and group7 have different counts from the steady-state middle
  groups. Treating the loop as eight independent 33-MAC groups is wrong.
- Several vector/accumulator families enter a group already live. A
  production generator needs an explicit register-lifetime table, not
  only an opcode template.
- The `vextbcst.16` consumer distances are filled by real dequant and
  accumulator movement work; replacing them with nop padding is only a
  diagnostic crutch.
