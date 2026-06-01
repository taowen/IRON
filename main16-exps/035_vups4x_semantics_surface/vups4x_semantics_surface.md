# VUPS.4x Semantics Surface

Status: `passed`

## What llvm-aie Teaches

| opcode | asm | dst | src | implicit uses | implicit defs |
| --- | --- | --- | --- | --- | --- |
| `VUPS_4x_mv_ups_w2c_upsSign0` | `vups.4x` | `OP_mCMm` | `OP_mWm` | `crSat, crUPSMode, upsSign0` | `srUPS_of` |
| `VUPS_4x_mv_ups_w2c_upsSign1` | `vups.4x` | `OP_mCMm` | `OP_mWm` | `crSat, crUPSMode, upsSign1` | `srUPS_of` |
| `VUPS_4x_mv_ups_x2d_upsSign0` | `vups.4x` | `eDM` | `OP_mXm` | `crSat, crUPSMode, upsSign0` | `srUPS_of` |
| `VUPS_4x_mv_ups_x2d_upsSign1` | `vups.4x` | `eDM` | `OP_mXm` | `crSat, crUPSMode, upsSign1` | `srUPS_of` |

## Schedule Surface

| opcode | resources | latencies |
| --- | --- | --- |
| `VUPS_4x_mv_ups_w2c_upsSign0` | `SimpleCycle<CRSAT_RA>, SimpleCycle<CRUPSMODE_RA>, PrefixCycle<DM_WM_L0_PORT>, SimpleCycle<EVENT_UPS_OF>` | `/*dst*/3, /*src*/1, /*su*/1, /*srUPS_of*/3, /*crSat*/1, /*crUPSMode*/2, /*upsSign0*/1` |
| `VUPS_4x_mv_ups_w2c_upsSign1` | `SimpleCycle<CRSAT_RA>, SimpleCycle<CRUPSMODE_RA>, PrefixCycle<DM_WM_L0_PORT>, SimpleCycle<EVENT_UPS_OF>` | `/*dst*/3, /*src*/1, /*su*/1, /*srUPS_of*/3, /*crSat*/1, /*crUPSMode*/2, /*upsSign1*/1` |
| `VUPS_4x_mv_ups_x2d_upsSign0` | `SimpleCycle<CRSAT_RA>, PrefixCycle<CRUPSMODE_RA>, PrefixCycle<UPS__H__A>, SimpleCycle<UPS__OMUX>, PrefixCycle<DM_WM_L0_PORT>, PrefixCycle<EVENT_UPS_OF>, PrefixCycle<UPS__H__K>, SimpleCycle<UPS__K>` | `/*dst*/3, /*src*/1, /*su*/1, /*srUPS_of*/3, /*crSat*/1, /*crUPSMode*/2, /*upsSign0*/1` |
| `VUPS_4x_mv_ups_x2d_upsSign1` | `SimpleCycle<CRSAT_RA>, PrefixCycle<CRUPSMODE_RA>, PrefixCycle<UPS__H__A>, SimpleCycle<UPS__OMUX>, PrefixCycle<DM_WM_L0_PORT>, PrefixCycle<EVENT_UPS_OF>, PrefixCycle<UPS__H__K>, SimpleCycle<UPS__K>` | `/*dst*/3, /*src*/1, /*su*/1, /*srUPS_of*/3, /*crSat*/1, /*crUPSMode*/2, /*upsSign1*/1` |

## Minimal MIR Builds

| variant | status | disassembly |
| --- | --- | --- |
| `VUPS_4x_mv_ups_w2c_upsSign0` | `passed` | `0: f8 08 01 18  	vups.4x	cml0, wl0, s0, upssign0` |
| `VUPS_4x_mv_ups_w2c_upsSign1` | `passed` | `0: f8 28 01 18  	vups.4x	cml0, wl0, s0, upssign1` |
| `VUPS_4x_mv_ups_x2d_upsSign0` | `passed` | `0: f8 04 00 18  	vups.4x	dm0, x0, s0, upssign0` |
| `VUPS_4x_mv_ups_x2d_upsSign1` | `passed` | `0: f8 24 00 18  	vups.4x	dm0, x0, s0, upssign1` |

## Interpretation

- MIR/llvm-aie is enough to encode all four `vups.4x` forms reproducibly.
- The available compiler metadata is not a value-level semantics document.
- In particular, it does not prove whether `dm/cml/cmh/bm*` sub-cells are fully overwritten, preserved, or implicitly read.
- Experiment 034 already showed that treating a nearby accumulator-cell move as dead is unsound for the zero/offset path.
- The next useful experiment is not another static mutation. It is an isolated NPU readback probe that initializes accumulator quadrants, executes one `vups.4x`, and records which cells changed.

## Sources

- `projects/llvm-aie/llvm/lib/Target/AIE/aie2p/AIE2PGenInstrInfo.td`
- `projects/llvm-aie/llvm/lib/Target/AIE/aie2p/AIE2PGenSchedule.td`
- `projects/llvm-aie/clang/lib/Headers/aie2p/aie2p_ups.h`
- `projects/llvm-aie/clang/include/clang/Basic/BuiltinsAIE2P.def`
