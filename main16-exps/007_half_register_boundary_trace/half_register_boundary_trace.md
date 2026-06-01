# Half-Register Boundary Trace

- Status: `passed`
- Source: `experiments/130_mylm_main16_record_observable_harness/mylm_c2r2_main16_record_exec.elf`
- Steady group: `1`

## Metrics

| metric | value |
| --- | ---: |
| `group` | `1` |
| `boundary_cells` | `23` |
| `mac_count` | `33` |
| `mixed_vector_operands` | `25` |
| `cross_group_vector_operands` | `9` |
| `cross_group_accumulators` | `2` |
| `shape:accum_to_bf16_vector` | `8` |
| `shape:activation_lane_broadcast` | `24` |
| `shape:activation_load` | `1` |
| `shape:q4_unpack` | `4` |
| `shape:register_move+accum_to_bf16_vector` | `8` |
| `shape:register_move+activation_lane_broadcast` | `13` |
| `shape:vbcst.16` | `1` |
| `shape:vldb+activation_lane_broadcast` | `3` |
| `shape:weight_or_state_load` | `4` |

## Boundary Cells Into Group1

| cell | producer | first group1 use |
| --- | --- | --- |
| `acc1.bmhh` | `g0@0x51e:vmac.f/mac` | `g1@0x52a:vmac.f` |
| `acc1.bmhl` | `g0@0x51e:vmac.f/mac` | `g1@0x52a:vmac.f` |
| `acc1.bmlh` | `g0@0x51e:vmac.f/mac` | `g1@0x52a:vmac.f` |
| `acc1.bmll` | `g0@0x51e:vmac.f/mac` | `g1@0x52a:vmac.f` |
| `acc4.bmhh` | `g0@0x41c:vsub.f/dequant_sub` | `g1@0x570:vmac.f` |
| `acc4.bmhl` | `g0@0x41c:vsub.f/dequant_sub` | `g1@0x570:vmac.f` |
| `acc4.bmlh` | `g0@0x41c:vsub.f/dequant_sub` | `g1@0x570:vmac.f` |
| `acc4.bmll` | `g0@0x478:vmov/register_move` | `g1@0x570:vmac.f` |
| `vec0.hi` | `g0@0x51e:vldb/weight_or_state_load` | `g1@0x53a:vmac.f` |
| `vec0.lo` | `g0@0x51e:vldb/weight_or_state_load` | `g1@0x53a:vmac.f` |
| `vec10.hi` | `g0@0x51a:vunpack/q4_unpack` | `g1@0x52a:vmac.f` |
| `vec10.lo` | `g0@0x51a:vunpack/q4_unpack` | `g1@0x52a:vmac.f` |
| `vec2.hi` | `g0@0x4b2:vextbcst.16/activation_lane_broadcast lane #0x1a` | `g1@0x570:vmac.f` |
| `vec2.lo` | `g0@0x4dc:vldb/vldb` | `g1@0x570:vmac.f` |
| `vec3.hi` | `g0@0x490:vconv.bf16.fp32/accum_to_bf16_vector` | `g1@0x54a:vmac.f` |
| `vec3.lo` | `g0@0x4f8:vmov/register_move` | `g1@0x54a:vmac.f` |
| `vec4.hi` | `g0@0x4ba:vconv.bf16.fp32/accum_to_bf16_vector` | `g1@0x53a:vmac.f` |
| `vec4.lo` | `g0@0x4ba:vconv.bf16.fp32/accum_to_bf16_vector` | `g1@0x53a:vmac.f` |
| `vec6.hi` | `g0@0x4ba:vextbcst.16/activation_lane_broadcast lane #0x1b` | `g1@0x580:vmac.f` |
| `vec8.hi` | `g0@0x516:vunpack/q4_unpack` | `g1@0x53a:vups.4x` |
| `vec8.lo` | `g0@0x516:vunpack/q4_unpack` | `g1@0x53a:vups.4x` |
| `vec9.hi` | `g0@0x50e:vunpack/q4_unpack` | `g1@0x52a:vmac.f` |
| `vec9.lo` | `g0@0x50e:vunpack/q4_unpack` | `g1@0x52a:vmac.f` |

## Group1 MAC Operand Table

| MAC | Accumulator | Left vector | Right vector |
| --- | --- | --- | --- |
| `0x52a` | `dm1` [cross-group]<br>acc1.bmll <- g0@0x51e:vmac.f/mac<br>acc1.bmlh <- g0@0x51e:vmac.f/mac<br>acc1.bmhl <- g0@0x51e:vmac.f/mac<br>acc1.bmhh <- g0@0x51e:vmac.f/mac | `x9` [cross-group]<br>vec9.lo <- g0@0x50e:vunpack/q4_unpack<br>vec9.hi <- g0@0x50e:vunpack/q4_unpack | `x10` [cross-group]<br>vec10.lo <- g0@0x51a:vunpack/q4_unpack<br>vec10.hi <- g0@0x51a:vunpack/q4_unpack |
| `0x53a` | `dm1`<br>acc1.bmll <- g1@0x536:vadd/dequant_add<br>acc1.bmlh <- g1@0x536:vadd/dequant_add<br>acc1.bmhl <- g1@0x536:vadd/dequant_add<br>acc1.bmhh <- g1@0x536:vadd/dequant_add | `x4` [cross-group]<br>vec4.lo <- g0@0x4ba:vconv.bf16.fp32/accum_to_bf16_vector<br>vec4.hi <- g0@0x4ba:vconv.bf16.fp32/accum_to_bf16_vector | `x0` [cross-group]<br>vec0.lo <- g0@0x51e:vldb/weight_or_state_load<br>vec0.hi <- g0@0x51e:vldb/weight_or_state_load |
| `0x54a` | `dm3`<br>acc3.bmll <- g1@0x53a:vmac.f/mac<br>acc3.bmlh <- g1@0x53a:vmac.f/mac<br>acc3.bmhl <- g1@0x53a:vmac.f/mac<br>acc3.bmhh <- g1@0x53a:vmac.f/mac | `x3` [mixed, cross-group]<br>vec3.lo <- g0@0x4f8:vmov/register_move<br>vec3.hi <- g0@0x490:vconv.bf16.fp32/accum_to_bf16_vector | `x11`<br>vec11.lo <- g1@0x52a:vldb/activation_load<br>vec11.hi <- g1@0x52a:vldb/activation_load |
| `0x570` | `dm4` [mixed, cross-group]<br>acc4.bmll <- g0@0x478:vmov/register_move<br>acc4.bmlh <- g0@0x41c:vsub.f/dequant_sub<br>acc4.bmhl <- g0@0x41c:vsub.f/dequant_sub<br>acc4.bmhh <- g0@0x41c:vsub.f/dequant_sub | `x4` [mixed, cross-group]<br>vec4.lo <- g1@0x564:vconv.bf16.fp32/accum_to_bf16_vector<br>vec4.hi <- g0@0x4ba:vconv.bf16.fp32/accum_to_bf16_vector | `x2` [mixed, cross-group]<br>vec2.lo <- g0@0x4dc:vldb/vldb<br>vec2.hi <- g0@0x4b2:vextbcst.16/activation_lane_broadcast lane #0x1a |
| `0x580` | `dm4`<br>acc4.bmll <- g1@0x570:vmac.f/mac<br>acc4.bmlh <- g1@0x570:vmac.f/mac<br>acc4.bmhl <- g1@0x570:vmac.f/mac<br>acc4.bmhh <- g1@0x570:vmac.f/mac | `x6` [mixed, cross-group]<br>vec6.lo <- g1@0x580:vmov/register_move<br>vec6.hi <- g0@0x4ba:vextbcst.16/activation_lane_broadcast lane #0x1b | `x1`<br>vec1.lo <- g1@0x55e:vunpack/q4_unpack<br>vec1.hi <- g1@0x55e:vunpack/q4_unpack |
| `0x5d0` | `dm3`<br>acc3.bmll <- g1@0x5bc:vmul.f/mul_seed<br>acc3.bmlh <- g1@0x5bc:vmul.f/mul_seed<br>acc3.bmhl <- g1@0x5bc:vmul.f/mul_seed<br>acc3.bmhh <- g1@0x5bc:vmul.f/mul_seed | `x6`<br>vec6.lo <- g1@0x5a8:vldb/weight_or_state_load<br>vec6.hi <- g1@0x5a8:vldb/weight_or_state_load | `x9`<br>vec9.lo <- g1@0x556:vextbcst.16/activation_lane_broadcast lane #0x1<br>vec9.hi <- g1@0x556:vextbcst.16/activation_lane_broadcast lane #0x1 |
| `0x5e2` | `dm3`<br>acc3.bmll <- g1@0x5d0:vmac.f/mac<br>acc3.bmlh <- g1@0x5d0:vmac.f/mac<br>acc3.bmhl <- g1@0x5d0:vmac.f/mac<br>acc3.bmhh <- g1@0x5d0:vmac.f/mac | `x3`<br>vec3.lo <- g1@0x588:vconv.bf16.fp32/accum_to_bf16_vector<br>vec3.hi <- g1@0x588:vconv.bf16.fp32/accum_to_bf16_vector | `x10` [mixed]<br>vec10.lo <- g1@0x5de:vmov/register_move<br>vec10.hi <- g1@0x5bc:vextbcst.16/activation_lane_broadcast lane #0x2 |
| `0x5f4` | `dm3`<br>acc3.bmll <- g1@0x5e2:vmac.f/mac<br>acc3.bmlh <- g1@0x5e2:vmac.f/mac<br>acc3.bmhl <- g1@0x5e2:vmac.f/mac<br>acc3.bmhh <- g1@0x5e2:vmac.f/mac | `x2` [mixed, cross-group]<br>vec2.lo <- g1@0x58e:vmov/register_move<br>vec2.hi <- g0@0x4b2:vextbcst.16/activation_lane_broadcast lane #0x1a | `x7`<br>vec7.lo <- g1@0x5ee:vextbcst.16/activation_lane_broadcast lane #0x4<br>vec7.hi <- g1@0x5ee:vextbcst.16/activation_lane_broadcast lane #0x4 |
| `0x608` | `dm3`<br>acc3.bmll <- g1@0x5f4:vmac.f/mac<br>acc3.bmlh <- g1@0x5f4:vmac.f/mac<br>acc3.bmhl <- g1@0x5f4:vmac.f/mac<br>acc3.bmhh <- g1@0x5f4:vmac.f/mac | `x4`<br>vec4.lo <- g1@0x600:vextbcst.16/activation_lane_broadcast lane #0x5<br>vec4.hi <- g1@0x600:vextbcst.16/activation_lane_broadcast lane #0x5 | `x7`<br>vec7.lo <- g1@0x5ee:vextbcst.16/activation_lane_broadcast lane #0x4<br>vec7.hi <- g1@0x5ee:vextbcst.16/activation_lane_broadcast lane #0x4 |
| `0x62a` | `dm3`<br>acc3.bmll <- g1@0x608:vmac.f/mac<br>acc3.bmlh <- g1@0x608:vmac.f/mac<br>acc3.bmhl <- g1@0x608:vmac.f/mac<br>acc3.bmhh <- g1@0x608:vmac.f/mac | `x10` [mixed]<br>vec10.lo <- g1@0x5de:vmov/register_move<br>vec10.hi <- g1@0x5bc:vextbcst.16/activation_lane_broadcast lane #0x2 | `x4`<br>vec4.lo <- g1@0x600:vextbcst.16/activation_lane_broadcast lane #0x5<br>vec4.hi <- g1@0x600:vextbcst.16/activation_lane_broadcast lane #0x5 |
| `0x640` | `dm4`<br>acc4.bmll <- g1@0x62a:vmac.f/mac<br>acc4.bmlh <- g1@0x62a:vmac.f/mac<br>acc4.bmhl <- g1@0x62a:vmac.f/mac<br>acc4.bmhh <- g1@0x62a:vmac.f/mac | `x8`<br>vec8.lo <- g1@0x636:vconv.bf16.fp32/accum_to_bf16_vector<br>vec8.hi <- g1@0x636:vconv.bf16.fp32/accum_to_bf16_vector | `x3` [mixed]<br>vec3.lo <- g1@0x636:vmov/register_move<br>vec3.hi <- g1@0x61e:vextbcst.16/activation_lane_broadcast lane #0x6 |
| `0x654` | `dm4`<br>acc4.bmll <- g1@0x640:vmac.f/mac<br>acc4.bmlh <- g1@0x640:vmac.f/mac<br>acc4.bmhl <- g1@0x640:vmac.f/mac<br>acc4.bmhh <- g1@0x640:vmac.f/mac | `x3` [mixed]<br>vec3.lo <- g1@0x650:vmov/register_move<br>vec3.hi <- g1@0x61e:vextbcst.16/activation_lane_broadcast lane #0x6 | `x0`<br>vec0.lo <- g1@0x62a:vextbcst.16/activation_lane_broadcast lane #0x7<br>vec0.hi <- g1@0x62a:vextbcst.16/activation_lane_broadcast lane #0x7 |
| `0x65e` | `dm4`<br>acc4.bmll <- g1@0x654:vmac.f/mac<br>acc4.bmlh <- g1@0x654:vmac.f/mac<br>acc4.bmhl <- g1@0x654:vmac.f/mac<br>acc4.bmhh <- g1@0x654:vmac.f/mac | `x5` [mixed]<br>vec5.lo <- g1@0x65a:vmov/register_move<br>vec5.hi <- g1@0x600:vconv.bf16.fp32/accum_to_bf16_vector | `x9` [mixed]<br>vec9.lo <- g1@0x65e:vmov/register_move<br>vec9.hi <- g1@0x5da:vextbcst.16/activation_lane_broadcast lane #0x8 |
| `0x66e` | `dm4`<br>acc4.bmll <- g1@0x65e:vmac.f/mac<br>acc4.bmlh <- g1@0x65e:vmac.f/mac<br>acc4.bmhl <- g1@0x65e:vmac.f/mac<br>acc4.bmhh <- g1@0x65e:vmac.f/mac | `x5` [mixed]<br>vec5.lo <- g1@0x65a:vmov/register_move<br>vec5.hi <- g1@0x600:vconv.bf16.fp32/accum_to_bf16_vector | `x4`<br>vec4.lo <- g1@0x632:vextbcst.16/activation_lane_broadcast lane #0x9<br>vec4.hi <- g1@0x632:vextbcst.16/activation_lane_broadcast lane #0x9 |
| `0x686` | `dm4`<br>acc4.bmll <- g1@0x66e:vmac.f/mac<br>acc4.bmlh <- g1@0x66e:vmac.f/mac<br>acc4.bmhl <- g1@0x66e:vmac.f/mac<br>acc4.bmhh <- g1@0x66e:vmac.f/mac | `x2`<br>vec2.lo <- g1@0x614:vconv.bf16.fp32/accum_to_bf16_vector<br>vec2.hi <- g1@0x614:vconv.bf16.fp32/accum_to_bf16_vector | `x7`<br>vec7.lo <- g1@0x640:vextbcst.16/activation_lane_broadcast lane #0xa<br>vec7.hi <- g1@0x640:vextbcst.16/activation_lane_broadcast lane #0xa |
| `0x6a2` | `dm3`<br>acc3.bmll <- g1@0x686:vmac.f/mac<br>acc3.bmlh <- g1@0x686:vmac.f/mac<br>acc3.bmhl <- g1@0x686:vmac.f/mac<br>acc3.bmhh <- g1@0x686:vmac.f/mac | `x3`<br>vec3.lo <- g1@0x6a2:vextbcst.16/activation_lane_broadcast lane #0xf<br>vec3.hi <- g1@0x6a2:vextbcst.16/activation_lane_broadcast lane #0xf | `x10`<br>vec10.lo <- g1@0x69a:vextbcst.16/activation_lane_broadcast lane #0xe<br>vec10.hi <- g1@0x69a:vextbcst.16/activation_lane_broadcast lane #0xe |
| `0x6b2` | `dm1`<br>acc1.bmll <- g1@0x6a2:vmac.f/mac<br>acc1.bmlh <- g1@0x6a2:vmac.f/mac<br>acc1.bmhl <- g1@0x6a2:vmac.f/mac<br>acc1.bmhh <- g1@0x6a2:vmac.f/mac | `x8` [mixed]<br>vec8.lo <- g1@0x6ae:vmov/register_move<br>vec8.hi <- g1@0x636:vconv.bf16.fp32/accum_to_bf16_vector | `x4`<br>vec4.lo <- g1@0x6b2:vextbcst.16/activation_lane_broadcast lane #0x15<br>vec4.hi <- g1@0x6b2:vextbcst.16/activation_lane_broadcast lane #0x15 |
| `0x6c6` | `dm3`<br>acc3.bmll <- g1@0x6b2:vmac.f/mac<br>acc3.bmlh <- g1@0x6b2:vmac.f/mac<br>acc3.bmhl <- g1@0x6b2:vmac.f/mac<br>acc3.bmhh <- g1@0x6b2:vmac.f/mac | `x9`<br>vec9.lo <- g1@0x6c6:vextbcst.16/activation_lane_broadcast lane #0x11<br>vec9.hi <- g1@0x6c6:vextbcst.16/activation_lane_broadcast lane #0x11 | `x5` [mixed]<br>vec5.lo <- g1@0x6c2:vmov/register_move<br>vec5.hi <- g1@0x690:vextbcst.16/activation_lane_broadcast lane #0xd |
| `0x6d6` | `dm3`<br>acc3.bmll <- g1@0x6c6:vmac.f/mac<br>acc3.bmlh <- g1@0x6c6:vmac.f/mac<br>acc3.bmhl <- g1@0x6c6:vmac.f/mac<br>acc3.bmhh <- g1@0x6c6:vmac.f/mac | `x1`<br>vec1.lo <- g1@0x6d2:vextbcst.16/activation_lane_broadcast lane #0x12<br>vec1.hi <- g1@0x6d2:vextbcst.16/activation_lane_broadcast lane #0x12 | `x10`<br>vec10.lo <- g1@0x69a:vextbcst.16/activation_lane_broadcast lane #0xe<br>vec10.hi <- g1@0x69a:vextbcst.16/activation_lane_broadcast lane #0xe |
| `0x6e6` | `dm3`<br>acc3.bmll <- g1@0x6d6:vmac.f/mac<br>acc3.bmlh <- g1@0x6d6:vmac.f/mac<br>acc3.bmhl <- g1@0x6d6:vmac.f/mac<br>acc3.bmhh <- g1@0x6d6:vmac.f/mac | `x8`<br>vec8.lo <- g1@0x6e6:vextbcst.16/activation_lane_broadcast lane #0x13<br>vec8.hi <- g1@0x6e6:vextbcst.16/activation_lane_broadcast lane #0x13 | `x3` [mixed]<br>vec3.lo <- g1@0x6e2:vmov/register_move<br>vec3.hi <- g1@0x6a2:vextbcst.16/activation_lane_broadcast lane #0xf |
| `0x6f6` | `dm3`<br>acc3.bmll <- g1@0x6e6:vmac.f/mac<br>acc3.bmlh <- g1@0x6e6:vmac.f/mac<br>acc3.bmhl <- g1@0x6e6:vmac.f/mac<br>acc3.bmhh <- g1@0x6e6:vmac.f/mac | `x2`<br>vec2.lo <- g1@0x6f6:vconv.bf16.fp32/accum_to_bf16_vector<br>vec2.hi <- g1@0x6f6:vconv.bf16.fp32/accum_to_bf16_vector | `x7`<br>vec7.lo <- g1@0x6f2:vextbcst.16/activation_lane_broadcast lane #0x14<br>vec7.hi <- g1@0x6f2:vextbcst.16/activation_lane_broadcast lane #0x14 |
| `0x70c` | `dm3`<br>acc3.bmll <- g1@0x6f6:vmac.f/mac<br>acc3.bmlh <- g1@0x6f6:vmac.f/mac<br>acc3.bmhl <- g1@0x6f6:vmac.f/mac<br>acc3.bmhh <- g1@0x6f6:vmac.f/mac | `x5`<br>vec5.lo <- g1@0x70c:vconv.bf16.fp32/accum_to_bf16_vector<br>vec5.hi <- g1@0x70c:vconv.bf16.fp32/accum_to_bf16_vector | `x9`<br>vec9.lo <- g1@0x6c6:vextbcst.16/activation_lane_broadcast lane #0x11<br>vec9.hi <- g1@0x6c6:vextbcst.16/activation_lane_broadcast lane #0x11 |
| `0x722` | `dm3`<br>acc3.bmll <- g1@0x70c:vmac.f/mac<br>acc3.bmlh <- g1@0x70c:vmac.f/mac<br>acc3.bmhl <- g1@0x70c:vmac.f/mac<br>acc3.bmhh <- g1@0x70c:vmac.f/mac | `x0`<br>vec0.lo <- g1@0x71e:vextbcst.16/activation_lane_broadcast lane #0x19<br>vec0.hi <- g1@0x71e:vextbcst.16/activation_lane_broadcast lane #0x19 | `x1`<br>vec1.lo <- g1@0x6d2:vextbcst.16/activation_lane_broadcast lane #0x12<br>vec1.hi <- g1@0x6d2:vextbcst.16/activation_lane_broadcast lane #0x12 |
| `0x730` | `dm3`<br>acc3.bmll <- g1@0x722:vmac.f/mac<br>acc3.bmlh <- g1@0x722:vmac.f/mac<br>acc3.bmhl <- g1@0x722:vmac.f/mac<br>acc3.bmhh <- g1@0x722:vmac.f/mac | `x3` [mixed]<br>vec3.lo <- g1@0x6e2:vmov/register_move<br>vec3.hi <- g1@0x6a2:vextbcst.16/activation_lane_broadcast lane #0xf | `x8` [mixed]<br>vec8.lo <- g1@0x72c:vmov/register_move<br>vec8.hi <- g1@0x6e6:vextbcst.16/activation_lane_broadcast lane #0x13 |
| `0x744` | `dm3`<br>acc3.bmll <- g1@0x730:vmac.f/mac<br>acc3.bmlh <- g1@0x730:vmac.f/mac<br>acc3.bmhl <- g1@0x730:vmac.f/mac<br>acc3.bmhh <- g1@0x730:vmac.f/mac | `x2` [mixed]<br>vec2.lo <- g1@0x73c:vmov/register_move<br>vec2.hi <- g1@0x6f6:vconv.bf16.fp32/accum_to_bf16_vector | `x7`<br>vec7.lo <- g1@0x6f2:vextbcst.16/activation_lane_broadcast lane #0x14<br>vec7.hi <- g1@0x6f2:vextbcst.16/activation_lane_broadcast lane #0x14 |
| `0x758` | `dm3`<br>acc3.bmll <- g1@0x744:vmac.f/mac<br>acc3.bmlh <- g1@0x744:vmac.f/mac<br>acc3.bmhl <- g1@0x744:vmac.f/mac<br>acc3.bmhh <- g1@0x744:vmac.f/mac | `x2` [mixed]<br>vec2.lo <- g1@0x73c:vmov/register_move<br>vec2.hi <- g1@0x6f6:vconv.bf16.fp32/accum_to_bf16_vector | `x4`<br>vec4.lo <- g1@0x6b2:vextbcst.16/activation_lane_broadcast lane #0x15<br>vec4.hi <- g1@0x6b2:vextbcst.16/activation_lane_broadcast lane #0x15 |
| `0x76c` | `dm3`<br>acc3.bmll <- g1@0x758:vmac.f/mac<br>acc3.bmlh <- g1@0x758:vmac.f/mac<br>acc3.bmhl <- g1@0x758:vmac.f/mac<br>acc3.bmhh <- g1@0x758:vmac.f/mac | `x5`<br>vec5.lo <- g1@0x764:vconv.bf16.fp32/accum_to_bf16_vector<br>vec5.hi <- g1@0x764:vconv.bf16.fp32/accum_to_bf16_vector | `x6`<br>vec6.lo <- g1@0x76c:vextbcst.16/activation_lane_broadcast lane #0x1b<br>vec6.hi <- g1@0x76c:vextbcst.16/activation_lane_broadcast lane #0x1b |
| `0x77e` | `dm2`<br>acc2.bmll <- g1@0x76c:vmac.f/mac<br>acc2.bmlh <- g1@0x76c:vmac.f/mac<br>acc2.bmhl <- g1@0x76c:vmac.f/mac<br>acc2.bmhh <- g1@0x76c:vmac.f/mac | `x8` [mixed]<br>vec8.lo <- g1@0x77a:vmov/register_move<br>vec8.hi <- g1@0x6e6:vextbcst.16/activation_lane_broadcast lane #0x13 | `x9` [mixed]<br>vec9.lo <- g1@0x77e:vmov/register_move<br>vec9.hi <- g1@0x716:vextbcst.16/activation_lane_broadcast lane #0x17 |
| `0x790` | `dm1`<br>acc1.bmll <- g1@0x77e:vmac.f/mac<br>acc1.bmlh <- g1@0x77e:vmac.f/mac<br>acc1.bmhl <- g1@0x77e:vmac.f/mac<br>acc1.bmhh <- g1@0x77e:vmac.f/mac | `x3` [mixed]<br>vec3.lo <- g1@0x78a:vmov/register_move<br>vec3.hi <- g1@0x73c:vconv.bf16.fp32/accum_to_bf16_vector | `x10`<br>vec10.lo <- g1@0x790:vextbcst.16/activation_lane_broadcast lane #0x1d<br>vec10.hi <- g1@0x790:vextbcst.16/activation_lane_broadcast lane #0x1d |
| `0x7a0` | `dm1`<br>acc1.bmll <- g1@0x790:vmac.f/mac<br>acc1.bmlh <- g1@0x790:vmac.f/mac<br>acc1.bmhl <- g1@0x790:vmac.f/mac<br>acc1.bmhh <- g1@0x790:vmac.f/mac | `x3` [mixed]<br>vec3.lo <- g1@0x78a:vmov/register_move<br>vec3.hi <- g1@0x73c:vconv.bf16.fp32/accum_to_bf16_vector | `x0`<br>vec0.lo <- g1@0x79c:vextbcst.16/activation_lane_broadcast lane #0x1e<br>vec0.hi <- g1@0x79c:vextbcst.16/activation_lane_broadcast lane #0x1e |
| `0x7b4` | `dm1`<br>acc1.bmll <- g1@0x7a0:vmac.f/mac<br>acc1.bmlh <- g1@0x7a0:vmac.f/mac<br>acc1.bmhl <- g1@0x7a0:vmac.f/mac<br>acc1.bmhh <- g1@0x7a0:vmac.f/mac | `x1`<br>vec1.lo <- g1@0x7b0:vbcst.16/vbcst.16<br>vec1.hi <- g1@0x7b0:vbcst.16/vbcst.16 | `x2` [mixed]<br>vec2.lo <- g1@0x790:vldb/vldb<br>vec2.hi <- g1@0x764:vextbcst.16/activation_lane_broadcast lane #0x1a |
| `0x7c2` | `dm1`<br>acc1.bmll <- g1@0x7b4:vmac.f/mac<br>acc1.bmlh <- g1@0x7b4:vmac.f/mac<br>acc1.bmhl <- g1@0x7b4:vmac.f/mac<br>acc1.bmhh <- g1@0x7b4:vmac.f/mac | `x8`<br>vec8.lo <- g1@0x7a0:vlda/weight_or_state_load<br>vec8.hi <- g1@0x7a0:vlda/weight_or_state_load | `x6` [mixed]<br>vec6.lo <- g1@0x7a0:vldb/vldb<br>vec6.hi <- g1@0x76c:vextbcst.16/activation_lane_broadcast lane #0x1b |
| `0x7d2` | `dm1`<br>acc1.bmll <- g1@0x7c2:vmac.f/mac<br>acc1.bmlh <- g1@0x7c2:vmac.f/mac<br>acc1.bmhl <- g1@0x7c2:vmac.f/mac<br>acc1.bmhh <- g1@0x7c2:vmac.f/mac | `x5`<br>vec5.lo <- g1@0x7bc:vunpack/q4_unpack<br>vec5.hi <- g1@0x7bc:vunpack/q4_unpack | `x7`<br>vec7.lo <- g1@0x7b4:vldb/weight_or_state_load<br>vec7.hi <- g1@0x7b4:vldb/weight_or_state_load |

## Interpretation

Group1 is already in the steady state of the MyLM hot loop. Many vector operands are assembled from separately produced halves, and some of those halves cross the group0->group1 boundary. Treating an `xN` register as one indivisible value hides the actual software pipeline.

The next experiment should connect these cell producers to the Q4NX payload formula: which cells are unpacked nibbles, which cells are bf16 coefficients after `vups.4x/vadd/vsub/vconv`, and which cells are activation broadcasts.
