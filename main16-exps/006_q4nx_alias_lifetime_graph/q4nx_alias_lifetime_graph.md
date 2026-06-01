# Q4NX Alias Lifetime Graph

- Status: `passed`
- Source: `experiments/130_mylm_main16_record_observable_harness/mylm_c2r2_main16_record_exec.elf`
- Hot range: `0x260..0x1850`
- Event fragments: `1532`
- Boundary live-through edges: `63`

## Semantic Counts

| semantic | count |
| --- | ---: |
| `accum_to_bf16_vector` | `136` |
| `activation_lane_broadcast` | `256` |
| `activation_load` | `8` |
| `dequant_add` | `64` |
| `dequant_sub` | `64` |
| `group_sum_load` | `8` |
| `local_scratch_load` | `1` |
| `mac` | `264` |
| `mul_seed` | `8` |
| `nop` | `108` |
| `nopa` | `1` |
| `nopb` | `1` |
| `paddb` | `2` |
| `q4_unpack` | `64` |
| `register_move` | `416` |
| `scalar_pointer_setup` | `11` |
| `ups_vector_to_accum` | `64` |
| `vbcst.16` | `8` |
| `vldb` | `16` |
| `weight_or_state_load` | `32` |

## Group Family Summary

| group | events | live-in vector/acc families | written vector/acc families |
| ---: | ---: | --- | --- |
| `0` | `192` | `['acc0']` | `['acc1', 'acc2', 'acc3', 'acc4', 'vec0', 'vec1', 'vec10', 'vec11', 'vec2', 'vec3', 'vec4', 'vec5', 'vec6', 'vec7', 'vec8', 'vec9']` |
| `1` | `189` | `['acc0', 'acc1', 'acc4', 'vec0', 'vec10', 'vec2', 'vec3', 'vec4', 'vec8', 'vec9']` | `['acc1', 'acc2', 'acc3', 'acc4', 'vec0', 'vec1', 'vec10', 'vec11', 'vec2', 'vec3', 'vec4', 'vec5', 'vec6', 'vec7', 'vec8', 'vec9']` |
| `2` | `189` | `['acc0', 'acc1', 'acc4', 'vec0', 'vec10', 'vec2', 'vec3', 'vec4', 'vec8', 'vec9']` | `['acc1', 'acc2', 'acc3', 'acc4', 'vec0', 'vec1', 'vec10', 'vec11', 'vec2', 'vec3', 'vec4', 'vec5', 'vec6', 'vec7', 'vec8', 'vec9']` |
| `3` | `189` | `['acc0', 'acc1', 'acc4', 'vec0', 'vec10', 'vec2', 'vec3', 'vec4', 'vec8', 'vec9']` | `['acc1', 'acc2', 'acc3', 'acc4', 'vec0', 'vec1', 'vec10', 'vec11', 'vec2', 'vec3', 'vec4', 'vec5', 'vec6', 'vec7', 'vec8', 'vec9']` |
| `4` | `189` | `['acc0', 'acc1', 'acc4', 'vec0', 'vec10', 'vec2', 'vec3', 'vec4', 'vec8', 'vec9']` | `['acc1', 'acc2', 'acc3', 'acc4', 'vec0', 'vec1', 'vec10', 'vec11', 'vec2', 'vec3', 'vec4', 'vec5', 'vec6', 'vec7', 'vec8', 'vec9']` |
| `5` | `189` | `['acc0', 'acc1', 'acc4', 'vec0', 'vec10', 'vec2', 'vec3', 'vec4', 'vec8', 'vec9']` | `['acc1', 'acc2', 'acc3', 'acc4', 'vec0', 'vec1', 'vec10', 'vec11', 'vec2', 'vec3', 'vec4', 'vec5', 'vec6', 'vec7', 'vec8', 'vec9']` |
| `6` | `190` | `['acc0', 'acc1', 'acc4', 'vec0', 'vec10', 'vec2', 'vec3', 'vec4', 'vec8', 'vec9']` | `['acc1', 'acc2', 'acc3', 'acc4', 'vec0', 'vec1', 'vec10', 'vec11', 'vec2', 'vec3', 'vec4', 'vec5', 'vec6', 'vec7', 'vec8', 'vec9']` |
| `7` | `205` | `['acc0', 'acc1', 'acc4', 'vec0', 'vec10', 'vec2', 'vec3', 'vec4', 'vec8', 'vec9']` | `['acc1', 'acc2', 'acc3', 'acc4', 'vec0', 'vec1', 'vec10', 'vec11', 'vec2', 'vec3', 'vec4', 'vec5', 'vec6', 'vec7', 'vec8', 'vec9']` |

## Boundary Signature

| boundary | live-through families |
| --- | --- |
| `g0->g1` | `['acc1', 'acc4', 'vec0', 'vec10', 'vec2', 'vec3', 'vec4', 'vec8', 'vec9']` |
| `g1->g2` | `['acc1', 'acc4', 'vec0', 'vec10', 'vec2', 'vec3', 'vec4', 'vec8', 'vec9']` |
| `g2->g3` | `['acc1', 'acc4', 'vec0', 'vec10', 'vec2', 'vec3', 'vec4', 'vec8', 'vec9']` |
| `g3->g4` | `['acc1', 'acc4', 'vec0', 'vec10', 'vec2', 'vec3', 'vec4', 'vec8', 'vec9']` |
| `g4->g5` | `['acc1', 'acc4', 'vec0', 'vec10', 'vec2', 'vec3', 'vec4', 'vec8', 'vec9']` |
| `g5->g6` | `['acc1', 'acc4', 'vec0', 'vec10', 'vec2', 'vec3', 'vec4', 'vec8', 'vec9']` |
| `g6->g7` | `['acc1', 'acc4', 'vec0', 'vec10', 'vec2', 'vec3', 'vec4', 'vec8', 'vec9']` |

## Boundary Live-Through Families

| boundary | family | producer | consumer |
| --- | --- | --- | --- |
| `g0->g1` | `acc1` | `0x51e` `vmac.f` `mac` defs=['dm1'] uses=['dm1', 'x5', 'x7', 'r4'] | `0x52a` `vmac.f` `mac` defs=['dm1'] uses=['dm1', 'x9', 'x10', 'r4'] |
| `g0->g1` | `acc4` | `0x478` `vmov` `register_move` defs=['bmll4'] uses=['lfh0'] | `0x570` `vmac.f` `mac` defs=['dm4'] uses=['dm4', 'x4', 'x2', 'r4'] |
| `g0->g1` | `vec0` | `0x51e` `vldb` `weight_or_state_load` defs=['x0'] uses=['p0'] | `0x53a` `vmac.f` `mac` defs=['dm3'] uses=['dm1', 'x4', 'x0', 'r4'] |
| `g0->g1` | `vec10` | `0x51a` `vunpack` `q4_unpack` defs=['x10'] uses=['wh7'] | `0x52a` `vmac.f` `mac` defs=['dm1'] uses=['dm1', 'x9', 'x10', 'r4'] |
| `g0->g1` | `vec2` | `0x4dc` `vldb` `vldb` defs=['wl2'] uses=['p5'] | `0x570` `vmac.f` `mac` defs=['dm4'] uses=['dm4', 'x4', 'x2', 'r4'] |
| `g0->g1` | `vec3` | `0x4f8` `vmov` `register_move` defs=['wl3'] uses=['wh4'] | `0x54a` `vmac.f` `mac` defs=['dm3'] uses=['dm3', 'x3', 'x11', 'r4'] |
| `g0->g1` | `vec4` | `0x4ba` `vconv.bf16.fp32` `accum_to_bf16_vector` defs=['x4'] uses=['cmh1'] | `0x53a` `vmac.f` `mac` defs=['dm3'] uses=['dm1', 'x4', 'x0', 'r4'] |
| `g0->g1` | `vec8` | `0x516` `vunpack` `q4_unpack` defs=['x8'] uses=['wh8'] | `0x53a` `vups.4x` `ups_vector_to_accum` defs=['dm2'] uses=['x8', 's0'] |
| `g0->g1` | `vec9` | `0x50e` `vunpack` `q4_unpack` defs=['x9'] uses=['wl8'] | `0x52a` `vmac.f` `mac` defs=['dm1'] uses=['dm1', 'x9', 'x10', 'r4'] |
| `g1->g2` | `acc1` | `0x7d2` `vmac.f` `mac` defs=['dm1'] uses=['dm1', 'x5', 'x7', 'r4'] | `0x7de` `vmac.f` `mac` defs=['dm1'] uses=['dm1', 'x9', 'x10', 'r4'] |
| `g1->g2` | `acc4` | `0x786` `vmov` `register_move` defs=['bmll4'] uses=['lfh0'] | `0x824` `vmac.f` `mac` defs=['dm4'] uses=['dm4', 'x4', 'x2', 'r4'] |
| `g1->g2` | `vec0` | `0x7d2` `vldb` `weight_or_state_load` defs=['x0'] uses=['p0'] | `0x7ee` `vmac.f` `mac` defs=['dm3'] uses=['dm1', 'x4', 'x0', 'r4'] |
| `g1->g2` | `vec10` | `0x7ce` `vunpack` `q4_unpack` defs=['x10'] uses=['wh7'] | `0x7de` `vmac.f` `mac` defs=['dm1'] uses=['dm1', 'x9', 'x10', 'r4'] |
| `g1->g2` | `vec2` | `0x790` `vldb` `vldb` defs=['wl2'] uses=['p5'] | `0x824` `vmac.f` `mac` defs=['dm4'] uses=['dm4', 'x4', 'x2', 'r4'] |
| `g1->g2` | `vec3` | `0x7ac` `vmov` `register_move` defs=['wl3'] uses=['wh4'] | `0x7fe` `vmac.f` `mac` defs=['dm3'] uses=['dm3', 'x3', 'x11', 'r4'] |
| `g1->g2` | `vec4` | `0x76c` `vconv.bf16.fp32` `accum_to_bf16_vector` defs=['x4'] uses=['cmh1'] | `0x7ee` `vmac.f` `mac` defs=['dm3'] uses=['dm1', 'x4', 'x0', 'r4'] |
| `g1->g2` | `vec8` | `0x7ca` `vunpack` `q4_unpack` defs=['x8'] uses=['wh8'] | `0x7ee` `vups.4x` `ups_vector_to_accum` defs=['dm2'] uses=['x8', 's0'] |
| `g1->g2` | `vec9` | `0x7c2` `vunpack` `q4_unpack` defs=['x9'] uses=['wl8'] | `0x7de` `vmac.f` `mac` defs=['dm1'] uses=['dm1', 'x9', 'x10', 'r4'] |
| `g2->g3` | `acc1` | `0xa86` `vmac.f` `mac` defs=['dm1'] uses=['dm1', 'x5', 'x7', 'r4'] | `0xa92` `vmac.f` `mac` defs=['dm1'] uses=['dm1', 'x9', 'x10', 'r4'] |
| `g2->g3` | `acc4` | `0xa3a` `vmov` `register_move` defs=['bmll4'] uses=['lfh0'] | `0xad8` `vmac.f` `mac` defs=['dm4'] uses=['dm4', 'x4', 'x2', 'r4'] |
| `g2->g3` | `vec0` | `0xa86` `vldb` `weight_or_state_load` defs=['x0'] uses=['p0'] | `0xaa2` `vmac.f` `mac` defs=['dm3'] uses=['dm1', 'x4', 'x0', 'r4'] |
| `g2->g3` | `vec10` | `0xa82` `vunpack` `q4_unpack` defs=['x10'] uses=['wh7'] | `0xa92` `vmac.f` `mac` defs=['dm1'] uses=['dm1', 'x9', 'x10', 'r4'] |
| `g2->g3` | `vec2` | `0xa44` `vldb` `vldb` defs=['wl2'] uses=['p5'] | `0xad8` `vmac.f` `mac` defs=['dm4'] uses=['dm4', 'x4', 'x2', 'r4'] |
| `g2->g3` | `vec3` | `0xa60` `vmov` `register_move` defs=['wl3'] uses=['wh4'] | `0xab2` `vmac.f` `mac` defs=['dm3'] uses=['dm3', 'x3', 'x11', 'r4'] |
| `g2->g3` | `vec4` | `0xa20` `vconv.bf16.fp32` `accum_to_bf16_vector` defs=['x4'] uses=['cmh1'] | `0xaa2` `vmac.f` `mac` defs=['dm3'] uses=['dm1', 'x4', 'x0', 'r4'] |
| `g2->g3` | `vec8` | `0xa7e` `vunpack` `q4_unpack` defs=['x8'] uses=['wh8'] | `0xaa2` `vups.4x` `ups_vector_to_accum` defs=['dm2'] uses=['x8', 's0'] |
| `g2->g3` | `vec9` | `0xa76` `vunpack` `q4_unpack` defs=['x9'] uses=['wl8'] | `0xa92` `vmac.f` `mac` defs=['dm1'] uses=['dm1', 'x9', 'x10', 'r4'] |
| `g3->g4` | `acc1` | `0xd3a` `vmac.f` `mac` defs=['dm1'] uses=['dm1', 'x5', 'x7', 'r4'] | `0xd46` `vmac.f` `mac` defs=['dm1'] uses=['dm1', 'x9', 'x10', 'r4'] |
| `g3->g4` | `acc4` | `0xcee` `vmov` `register_move` defs=['bmll4'] uses=['lfh0'] | `0xd8c` `vmac.f` `mac` defs=['dm4'] uses=['dm4', 'x4', 'x2', 'r4'] |
| `g3->g4` | `vec0` | `0xd3a` `vldb` `weight_or_state_load` defs=['x0'] uses=['p0'] | `0xd56` `vmac.f` `mac` defs=['dm3'] uses=['dm1', 'x4', 'x0', 'r4'] |
| `g3->g4` | `vec10` | `0xd36` `vunpack` `q4_unpack` defs=['x10'] uses=['wh7'] | `0xd46` `vmac.f` `mac` defs=['dm1'] uses=['dm1', 'x9', 'x10', 'r4'] |
| `g3->g4` | `vec2` | `0xcf8` `vldb` `vldb` defs=['wl2'] uses=['p5'] | `0xd8c` `vmac.f` `mac` defs=['dm4'] uses=['dm4', 'x4', 'x2', 'r4'] |
| `g3->g4` | `vec3` | `0xd14` `vmov` `register_move` defs=['wl3'] uses=['wh4'] | `0xd66` `vmac.f` `mac` defs=['dm3'] uses=['dm3', 'x3', 'x11', 'r4'] |
| `g3->g4` | `vec4` | `0xcd4` `vconv.bf16.fp32` `accum_to_bf16_vector` defs=['x4'] uses=['cmh1'] | `0xd56` `vmac.f` `mac` defs=['dm3'] uses=['dm1', 'x4', 'x0', 'r4'] |
| `g3->g4` | `vec8` | `0xd32` `vunpack` `q4_unpack` defs=['x8'] uses=['wh8'] | `0xd56` `vups.4x` `ups_vector_to_accum` defs=['dm2'] uses=['x8', 's0'] |
| `g3->g4` | `vec9` | `0xd2a` `vunpack` `q4_unpack` defs=['x9'] uses=['wl8'] | `0xd46` `vmac.f` `mac` defs=['dm1'] uses=['dm1', 'x9', 'x10', 'r4'] |
| `g4->g5` | `acc1` | `0xfee` `vmac.f` `mac` defs=['dm1'] uses=['dm1', 'x5', 'x7', 'r4'] | `0xffa` `vmac.f` `mac` defs=['dm1'] uses=['dm1', 'x9', 'x10', 'r4'] |
| `g4->g5` | `acc4` | `0xfa2` `vmov` `register_move` defs=['bmll4'] uses=['lfh0'] | `0x1040` `vmac.f` `mac` defs=['dm4'] uses=['dm4', 'x4', 'x2', 'r4'] |
| `g4->g5` | `vec0` | `0xfee` `vldb` `weight_or_state_load` defs=['x0'] uses=['p0'] | `0x100a` `vmac.f` `mac` defs=['dm3'] uses=['dm1', 'x4', 'x0', 'r4'] |
| `g4->g5` | `vec10` | `0xfea` `vunpack` `q4_unpack` defs=['x10'] uses=['wh7'] | `0xffa` `vmac.f` `mac` defs=['dm1'] uses=['dm1', 'x9', 'x10', 'r4'] |
| `g4->g5` | `vec2` | `0xfac` `vldb` `vldb` defs=['wl2'] uses=['p5'] | `0x1040` `vmac.f` `mac` defs=['dm4'] uses=['dm4', 'x4', 'x2', 'r4'] |
| `g4->g5` | `vec3` | `0xfc8` `vmov` `register_move` defs=['wl3'] uses=['wh4'] | `0x101a` `vmac.f` `mac` defs=['dm3'] uses=['dm3', 'x3', 'x11', 'r4'] |
| `g4->g5` | `vec4` | `0xf88` `vconv.bf16.fp32` `accum_to_bf16_vector` defs=['x4'] uses=['cmh1'] | `0x100a` `vmac.f` `mac` defs=['dm3'] uses=['dm1', 'x4', 'x0', 'r4'] |
| `g4->g5` | `vec8` | `0xfe6` `vunpack` `q4_unpack` defs=['x8'] uses=['wh8'] | `0x100a` `vups.4x` `ups_vector_to_accum` defs=['dm2'] uses=['x8', 's0'] |
| `g4->g5` | `vec9` | `0xfde` `vunpack` `q4_unpack` defs=['x9'] uses=['wl8'] | `0xffa` `vmac.f` `mac` defs=['dm1'] uses=['dm1', 'x9', 'x10', 'r4'] |
| `g5->g6` | `acc1` | `0x12a2` `vmac.f` `mac` defs=['dm1'] uses=['dm1', 'x5', 'x7', 'r4'] | `0x12ae` `vmac.f` `mac` defs=['dm1'] uses=['dm1', 'x9', 'x10', 'r4'] |
| `g5->g6` | `acc4` | `0x1256` `vmov` `register_move` defs=['bmll4'] uses=['lfh0'] | `0x12f4` `vmac.f` `mac` defs=['dm4'] uses=['dm4', 'x4', 'x2', 'r4'] |
| `g5->g6` | `vec0` | `0x12a2` `vldb` `weight_or_state_load` defs=['x0'] uses=['p0'] | `0x12be` `vmac.f` `mac` defs=['dm3'] uses=['dm1', 'x4', 'x0', 'r4'] |
| `g5->g6` | `vec10` | `0x129e` `vunpack` `q4_unpack` defs=['x10'] uses=['wh7'] | `0x12ae` `vmac.f` `mac` defs=['dm1'] uses=['dm1', 'x9', 'x10', 'r4'] |
| `g5->g6` | `vec2` | `0x1260` `vldb` `vldb` defs=['wl2'] uses=['p5'] | `0x12f4` `vmac.f` `mac` defs=['dm4'] uses=['dm4', 'x4', 'x2', 'r4'] |
| `g5->g6` | `vec3` | `0x127c` `vmov` `register_move` defs=['wl3'] uses=['wh4'] | `0x12ce` `vmac.f` `mac` defs=['dm3'] uses=['dm3', 'x3', 'x11', 'r4'] |
| `g5->g6` | `vec4` | `0x123c` `vconv.bf16.fp32` `accum_to_bf16_vector` defs=['x4'] uses=['cmh1'] | `0x12be` `vmac.f` `mac` defs=['dm3'] uses=['dm1', 'x4', 'x0', 'r4'] |
| `g5->g6` | `vec8` | `0x129a` `vunpack` `q4_unpack` defs=['x8'] uses=['wh8'] | `0x12be` `vups.4x` `ups_vector_to_accum` defs=['dm2'] uses=['x8', 's0'] |
| `g5->g6` | `vec9` | `0x1292` `vunpack` `q4_unpack` defs=['x9'] uses=['wl8'] | `0x12ae` `vmac.f` `mac` defs=['dm1'] uses=['dm1', 'x9', 'x10', 'r4'] |
| `g6->g7` | `acc1` | `0x155a` `vmac.f` `mac` defs=['dm1'] uses=['dm1', 'x5', 'x7', 'r4'] | `0x1566` `vmac.f` `mac` defs=['dm1'] uses=['dm1', 'x9', 'x10', 'r4'] |
| `g6->g7` | `acc4` | `0x150a` `vmov` `register_move` defs=['bmll4'] uses=['lfh0'] | `0x15ae` `vmac.f` `mac` defs=['dm4'] uses=['dm4', 'x4', 'x2', 'r4'] |
| `g6->g7` | `vec0` | `0x155a` `vldb` `weight_or_state_load` defs=['x0'] uses=['p0'] | `0x1578` `vmac.f` `mac` defs=['dm3'] uses=['dm1', 'x4', 'x0', 'r4'] |
| `g6->g7` | `vec10` | `0x1556` `vunpack` `q4_unpack` defs=['x10'] uses=['wh7'] | `0x1566` `vmac.f` `mac` defs=['dm1'] uses=['dm1', 'x9', 'x10', 'r4'] |
| `g6->g7` | `vec2` | `0x1514` `vldb` `vldb` defs=['wl2'] uses=['p5'] | `0x15ae` `vmac.f` `mac` defs=['dm4'] uses=['dm4', 'x4', 'x2', 'r4'] |
| `g6->g7` | `vec3` | `0x1532` `vmov` `register_move` defs=['wl3'] uses=['wh4'] | `0x1588` `vmac.f` `mac` defs=['dm3'] uses=['dm3', 'x3', 'x11', 'r4'] |
| `g6->g7` | `vec4` | `0x14f0` `vconv.bf16.fp32` `accum_to_bf16_vector` defs=['x4'] uses=['cmh1'] | `0x1578` `vmac.f` `mac` defs=['dm3'] uses=['dm1', 'x4', 'x0', 'r4'] |
| `g6->g7` | `vec8` | `0x1552` `vunpack` `q4_unpack` defs=['x8'] uses=['wh8'] | `0x1578` `vups.4x` `ups_vector_to_accum` defs=['dm2'] uses=['x8', 's0'] |
| `g6->g7` | `vec9` | `0x154a` `vunpack` `q4_unpack` defs=['x9'] uses=['wl8'] | `0x1566` `vmac.f` `mac` defs=['dm1'] uses=['dm1', 'x9', 'x10', 'r4'] |

## Key Event Prefix

| index | group | address | semantic | defs | uses | fragment |
| ---: | ---: | --- | --- | --- | --- | --- |
| `1` | `0` | `0x260` | `activation_load` | `['x11']` | `['p1']` | `vldb	 x11, [p1], #0x40` |
| `5` | `0` | `0x26c` | `q4_unpack` | `['x9']` | `['wl8']` | `vunpack	x9, wl8, unpacksign0` |
| `9` | `0` | `0x278` | `q4_unpack` | `['x8']` | `['wh8']` | `vunpack	x8, wh8, unpacksign0` |
| `13` | `0` | `0x284` | `q4_unpack` | `['x5']` | `['wl7']` | `vunpack	x5, wl7, unpacksign0` |
| `15` | `0` | `0x28e` | `group_sum_load` | `['r7']` | `['p3']` | `lda.s16	 r7, [p3], #0x2` |
| `16` | `0` | `0x28e` | `q4_unpack` | `['x10']` | `['wh7']` | `vunpack	x10, wh7, unpacksign0` |
| `17` | `0` | `0x294` | `q4_unpack` | `['x1']` | `['wl0']` | `vunpack	x1, wl0, unpacksign0` |
| `19` | `0` | `0x29a` | `activation_lane_broadcast` | `['x9']` | `['x11']` | `vextbcst.16	 x9, x11, #0x1` |
| `20` | `0` | `0x29e` | `ups_vector_to_accum` | `['dm2']` | `['x9', 's0']` | `vups.4x	dm2, x9, s0, upssign0` |
| `21` | `0` | `0x29e` | `dequant_add` | `['dm1']` | `['dm2', 'dm0', 'r0']` | `vadd	dm1, dm2, dm0, r0` |
| `22` | `0` | `0x2a6` | `ups_vector_to_accum` | `['dm2']` | `['x8', 's0']` | `vups.4x	dm2, x8, s0, upssign0` |
| `23` | `0` | `0x2a6` | `dequant_add` | `['dm2']` | `['dm2', 'dm0', 'r0']` | `vadd	dm2, dm2, dm0, r0` |
| `24` | `0` | `0x2ae` | `activation_lane_broadcast` | `['x7']` | `['x11']` | `vextbcst.16	 x7, x11, #0x0` |
| `25` | `0` | `0x2ae` | `dequant_sub` | `['dm1']` | `['dm1', 'dm0', 'r5']` | `vsub.f	dm1, dm1, dm0, r5` |
| `26` | `0` | `0x2b6` | `dequant_sub` | `['dm3']` | `['dm2', 'dm0', 'r5']` | `vsub.f	dm3, dm2, dm0, r5` |
| `30` | `0` | `0x2ce` | `accum_to_bf16_vector` | `['x5']` | `['cml2']` | `vconv.bf16.fp32	 x5, cml2` |
| `32` | `0` | `0x2d6` | `ups_vector_to_accum` | `['dm1']` | `['x5', 's0']` | `vups.4x	dm1, x5, s0, upssign0` |
| `33` | `0` | `0x2d6` | `dequant_add` | `['dm4']` | `['dm1', 'dm0', 'r0']` | `vadd	dm4, dm1, dm0, r0` |
| `34` | `0` | `0x2de` | `accum_to_bf16_vector` | `['x3']` | `['cmh2']` | `vconv.bf16.fp32	 x3, cmh2` |
| `37` | `0` | `0x2e6` | `dequant_sub` | `['dm2']` | `['dm4', 'dm0', 'r5']` | `vsub.f	dm2, dm4, dm0, r5` |
| `41` | `0` | `0x2fa` | `accum_to_bf16_vector` | `['x8']` | `['cmh1']` | `vconv.bf16.fp32	 x8, cmh1` |
| `43` | `0` | `0x302` | `ups_vector_to_accum` | `['dm3']` | `['x10', 's0']` | `vups.4x	dm3, x10, s0, upssign0` |
| `44` | `0` | `0x302` | `dequant_add` | `['dm3']` | `['dm3', 'dm0', 'r0']` | `vadd	dm3, dm3, dm0, r0` |
| `45` | `0` | `0x30a` | `activation_lane_broadcast` | `['x10']` | `['x11']` | `vextbcst.16	 x10, x11, #0x2` |
| `46` | `0` | `0x30a` | `mul_seed` | `['dm3']` | `['x5', 'x7', 'r4']` | `vmul.f	dm3, x5, x7, r4` |
| `48` | `0` | `0x312` | `dequant_sub` | `['dm4']` | `['dm3', 'dm0', 'r5']` | `vsub.f	dm4, dm3, dm0, r5` |
| `49` | `0` | `0x31a` | `activation_lane_broadcast` | `['x7']` | `['x11']` | `vextbcst.16	 x7, x11, #0x3` |
| `50` | `0` | `0x31e` | `accum_to_bf16_vector` | `['x4']` | `['cml1']` | `vconv.bf16.fp32	 x4, cml1` |
| `52` | `0` | `0x31e` | `mac` | `['dm3']` | `['dm3', 'x6', 'x9', 'r4']` | `vmac.f	dm3, dm3, x6, x9, r4` |
| `53` | `0` | `0x328` | `activation_lane_broadcast` | `['x9']` | `['x11']` | `vextbcst.16	 x9, x11, #0x8` |
| `56` | `0` | `0x330` | `mac` | `['dm3']` | `['dm3', 'x3', 'x10', 'r4']` | `vmac.f	dm3, dm3, x3, x10, r4` |
| `58` | `0` | `0x33c` | `q4_unpack` | `['x1']` | `['wh0']` | `vunpack	x1, wh0, unpacksign0` |
| `59` | `0` | `0x33c` | `activation_lane_broadcast` | `['x7']` | `['x11']` | `vextbcst.16	 x7, x11, #0x4` |
| `61` | `0` | `0x342` | `mac` | `['dm3']` | `['dm3', 'x2', 'x7', 'r4']` | `vmac.f	dm3, dm3, x2, x7, r4` |
| `63` | `0` | `0x34e` | `accum_to_bf16_vector` | `['x5']` | `['cml1']` | `vconv.bf16.fp32	 x5, cml1` |
| `64` | `0` | `0x34e` | `activation_lane_broadcast` | `['x4']` | `['x11']` | `vextbcst.16	 x4, x11, #0x5` |
| `66` | `0` | `0x356` | `mac` | `['dm3']` | `['dm3', 'x4', 'x7', 'r4']` | `vmac.f	dm3, dm3, x4, x7, r4` |
| `68` | `0` | `0x362` | `accum_to_bf16_vector` | `['x2']` | `['cmh1']` | `vconv.bf16.fp32	 x2, cmh1` |
| `69` | `0` | `0x362` | `ups_vector_to_accum` | `['dm4']` | `['x1', 's0']` | `vups.4x	dm4, x1, s0, upssign0` |
| `70` | `0` | `0x362` | `dequant_add` | `['dm1']` | `['dm4', 'dm0', 'r0']` | `vadd	dm1, dm4, dm0, r0` |
| `71` | `0` | `0x36c` | `activation_lane_broadcast` | `['x3']` | `['x11']` | `vextbcst.16	 x3, x11, #0x6` |
| `73` | `0` | `0x370` | `dequant_sub` | `['dm4']` | `['dm1', 'dm0', 'r5']` | `vsub.f	dm4, dm1, dm0, r5` |
| `74` | `0` | `0x378` | `activation_lane_broadcast` | `['x0']` | `['x11']` | `vextbcst.16	 x0, x11, #0x7` |
| `75` | `0` | `0x378` | `mac` | `['dm4']` | `['dm3', 'x10', 'x4', 'r4']` | `vmac.f	dm4, dm3, x10, x4, r4` |
| `76` | `0` | `0x380` | `activation_lane_broadcast` | `['x4']` | `['x11']` | `vextbcst.16	 x4, x11, #0x9` |
| `77` | `0` | `0x384` | `accum_to_bf16_vector` | `['x8']` | `['cml2']` | `vconv.bf16.fp32	 x8, cml2` |
| `80` | `0` | `0x38e` | `activation_lane_broadcast` | `['x7']` | `['x11']` | `vextbcst.16	 x7, x11, #0xa` |
| `81` | `0` | `0x38e` | `mac` | `['dm4']` | `['dm4', 'x8', 'x3', 'r4']` | `vmac.f	dm4, dm4, x8, x3, r4` |
| `82` | `0` | `0x396` | `activation_lane_broadcast` | `['x10']` | `['x11']` | `vextbcst.16	 x10, x11, #0xb` |
| `85` | `0` | `0x3a2` | `mac` | `['dm4']` | `['dm4', 'x3', 'x0', 'r4']` | `vmac.f	dm4, dm4, x3, x0, r4` |
| `89` | `0` | `0x3ac` | `mac` | `['dm4']` | `['dm4', 'x5', 'x9', 'r4']` | `vmac.f	dm4, dm4, x5, x9, r4` |
| `92` | `0` | `0x3bc` | `accum_to_bf16_vector` | `['x1']` | `['cmh2']` | `vconv.bf16.fp32	 x1, cmh2` |
| `94` | `0` | `0x3bc` | `mac` | `['dm4']` | `['dm4', 'x5', 'x4', 'r4']` | `vmac.f	dm4, dm4, x5, x4, r4` |
| `95` | `0` | `0x3c6` | `accum_to_bf16_vector` | `['x0']` | `['cmh1']` | `vconv.bf16.fp32	 x0, cmh1` |
| `96` | `0` | `0x3c6` | `ups_vector_to_accum` | `['dm3']` | `['x1', 's0']` | `vups.4x	dm3, x1, s0, upssign0` |
| `97` | `0` | `0x3c6` | `dequant_add` | `['dm2']` | `['dm3', 'dm0', 'r0']` | `vadd	dm2, dm3, dm0, r0` |
| `98` | `0` | `0x3d0` | `activation_lane_broadcast` | `['x4']` | `['x11']` | `vextbcst.16	 x4, x11, #0xc` |
| `99` | `0` | `0x3d4` | `q4_unpack` | `['x4']` | `['wl6']` | `vunpack	x4, wl6, unpacksign0` |
| `101` | `0` | `0x3d4` | `mac` | `['dm3']` | `['dm4', 'x2', 'x7', 'r4']` | `vmac.f	dm3, dm4, x2, x7, r4` |
| `102` | `0` | `0x3de` | `q4_unpack` | `['x6']` | `['wh6']` | `vunpack	x6, wh6, unpacksign0` |
| `103` | `0` | `0x3de` | `activation_lane_broadcast` | `['x5']` | `['x11']` | `vextbcst.16	 x5, x11, #0xd` |
| `104` | `0` | `0x3de` | `dequant_sub` | `['dm2']` | `['dm2', 'dm0', 'r5']` | `vsub.f	dm2, dm2, dm0, r5` |
| `105` | `0` | `0x3e8` | `accum_to_bf16_vector` | `['x2']` | `['cml1']` | `vconv.bf16.fp32	 x2, cml1` |
| `106` | `0` | `0x3e8` | `activation_lane_broadcast` | `['x10']` | `['x11']` | `vextbcst.16	 x10, x11, #0xe` |
| `107` | `0` | `0x3f0` | `activation_lane_broadcast` | `['x3']` | `['x11']` | `vextbcst.16	 x3, x11, #0xf` |
| `108` | `0` | `0x3f0` | `mac` | `['dm1']` | `['dm3', 'x3', 'x10', 'r4']` | `vmac.f	dm1, dm3, x3, x10, r4` |
| `109` | `0` | `0x3f8` | `activation_lane_broadcast` | `['x7']` | `['x11']` | `vextbcst.16	 x7, x11, #0x10` |
| `111` | `0` | `0x400` | `activation_lane_broadcast` | `['x4']` | `['x11']` | `vextbcst.16	 x4, x11, #0x15` |
| `112` | `0` | `0x400` | `mac` | `['dm3']` | `['dm1', 'x8', 'x4', 'r4']` | `vmac.f	dm3, dm1, x8, x4, r4` |
| `113` | `0` | `0x408` | `ups_vector_to_accum` | `['dm4']` | `['x4', 's0']` | `vups.4x	dm4, x4, s0, upssign0` |
| `114` | `0` | `0x408` | `dequant_add` | `['dm4']` | `['dm4', 'dm0', 'r0']` | `vadd	dm4, dm4, dm0, r0` |
| `116` | `0` | `0x414` | `activation_lane_broadcast` | `['x9']` | `['x11']` | `vextbcst.16	 x9, x11, #0x11` |
| `117` | `0` | `0x414` | `mac` | `['dm3']` | `['dm3', 'x9', 'x5', 'r4']` | `vmac.f	dm3, dm3, x9, x5, r4` |
| `118` | `0` | `0x41c` | `dequant_sub` | `['dm4']` | `['dm4', 'dm0', 'r5']` | `vsub.f	dm4, dm4, dm0, r5` |
| `119` | `0` | `0x420` | `activation_lane_broadcast` | `['x1']` | `['x11']` | `vextbcst.16	 x1, x11, #0x12` |
| `121` | `0` | `0x424` | `mac` | `['dm3']` | `['dm3', 'x1', 'x10', 'r4']` | `vmac.f	dm3, dm3, x1, x10, r4` |
| `122` | `0` | `0x42c` | `activation_lane_broadcast` | `['x10']` | `['x11']` | `vextbcst.16	 x10, x11, #0x18` |
| `124` | `0` | `0x434` | `activation_lane_broadcast` | `['x8']` | `['x11']` | `vextbcst.16	 x8, x11, #0x13` |
| `125` | `0` | `0x434` | `mac` | `['dm3']` | `['dm3', 'x8', 'x3', 'r4']` | `vmac.f	dm3, dm3, x8, x3, r4` |
| `127` | `0` | `0x440` | `activation_lane_broadcast` | `['x7']` | `['x11']` | `vextbcst.16	 x7, x11, #0x14` |

## Interpretation

This experiment moves past register-name vocabulary. It treats vector and accumulator aliases as shared families and exposes which families are live across activation-group boundaries. Those live-through edges are the concrete shape of MyLM's cross-group software pipeline.

The table is still conservative: `dm/cml/cmh/bm*` partial writes are reported at family granularity. That is enough to identify which families need manual lane/view-level decoding next.
