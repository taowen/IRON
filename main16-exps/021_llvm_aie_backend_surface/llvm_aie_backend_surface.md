# LLVM-AIE Backend Surface

Status: `route-selected`

C++ remains the wrong control surface for the MyLM-style main16 Q4NX hot body. 
The useful `llvm-aie` pieces are lower-level: AIE2P instruction definitions, 
schedule metadata, MIR examples, and the Peano command-line tools.

## Reusable Surfaces

### instruction database

- Use: Parse or reference exact instruction names, operands, implicit regs, and selectable patterns for the Q4NX generator.
- Risk: Generated TD files are backend-internal contracts; pin the llvm-aie revision before depending on names.
- Paths:
  - `llvm/lib/Target/AIE/aie2p/AIE2PGenInstrInfo.td`
  - `llvm/lib/Target/AIE/aie2p/AIE2PInstrPatterns.td`

### schedule database

- Use: Extract instruction latency/bypass facts for a nop-free source-asm checker or for MIR schedule expectations.
- Risk: It is not enough to count opcodes; resource conflicts and VLIW packetization still need llc/llvm-mc validation.
- Paths:
  - `llvm/lib/Target/AIE/aie2p/AIE2PGenSchedule.td`

### MIR examples

- Use: Use as templates for a tiny Q4NX MIR/codegen experiment with explicit registers and postpipeliner remarks.
- Risk: MIR is lower-level and brittle, but it gives more control than C++ and less manual scheduling than raw source asm.
- Paths:
  - `llvm/test/CodeGen/AIE/aie2p/BinaryOutput/vups.mir`
  - `llvm/test/CodeGen/AIE/aie2p/BinaryOutput/vextbcst.mir`
  - `llvm/test/CodeGen/AIE/aie2p/BinaryOutput/vmac.mir`
  - `llvm/test/CodeGen/AIE/aie2p/schedule/postpipeliner/gemm-bfp16-v10.mir`

### external tools

- Use: Keep assembler, disassembler, object writer, and scheduler as external services invoked by experiments.
- Risk: Do not copy these internals into IRON; use command outputs as gates.
- Paths:
  - `.venv/lib/python3.12/site-packages/llvm-aie/bin/llc`
  - `.venv/lib/python3.12/site-packages/llvm-aie/bin/llvm-mc`
  - `.venv/lib/python3.12/site-packages/llvm-aie/bin/llvm-objdump`

### backend implementation reference

- Use: Read-only reference for why schedules fail, how bundles are finalized, and how source/MIR becomes bytes.
- Risk: These files are not small reusable libraries; direct extraction would import too much LLVM state.
- Paths:
  - `llvm/lib/Target/AIE/AIEPostPipeliner.cpp`
  - `llvm/lib/Target/AIE/AIEHazardRecognizer.cpp`
  - `llvm/lib/Target/AIE/AIEFinalizeBundle.cpp`
  - `llvm/lib/Target/AIE/MCTargetDesc/aie2p/AIE2PMCCodeEmitter.cpp`
  - `llvm/lib/Target/AIE/AsmParser/AIE2PAsmParser.cpp`
  - `llvm/lib/Target/AIE/Disassembler/AIE2PDisassembler.cpp`

## Source Evidence

| Surface | Line | Matches | Why |
| --- | ---: | ---: | --- |
| `llvm/lib/Target/AIE/aie2p/AIE2PGenInstrInfo.td` | 6358 | 1 | Defines the exact AIE2P machine instruction spelling and operands for vextbcst.16. |
| `llvm/lib/Target/AIE/aie2p/AIE2PGenInstrInfo.td` | 8860 | 1 | Defines one of the bf16 vmac.f forms used by Q4NX-style MAC bodies. |
| `llvm/lib/Target/AIE/aie2p/AIE2PGenInstrInfo.td` | 11946 | 1 | Defines the vups.4x form closest to the MyLM dequant pipeline. |
| `llvm/lib/Target/AIE/aie2p/AIE2PInstrPatterns.td` | 852 | 1 | Shows the intrinsic/global-isel pattern that can select vups.4x. |
| `llvm/lib/Target/AIE/aie2p/AIE2PInstrPatterns.td` | 1030 | 1 | Shows that .16 extract-broadcast is explicitly modeled, not an accidental asm spelling. |
| `llvm/lib/Target/AIE/aie2p/AIE2PGenSchedule.td` | 8285 | 1 | Provides latency/bypass metadata for a scheduler or static checker. |
| `llvm/lib/Target/AIE/aie2p/AIE2PGenSchedule.td` | 9049 | 1 | Provides latency/bypass metadata for bf16 vmac.f. |
| `llvm/lib/Target/AIE/aie2p/AIE2PGenSchedule.td` | 10069 | 1 | Provides latency/bypass metadata for vups.4x x-to-d. |
| `PERF_OPT_GUIDE.md` | 516 | 1 | Explains why our C++ route fails when register pressure creates spills. |
| `PERF_OPT_GUIDE.md` | 397 | 4 | Documents the compiler-level control surface for software-pipelined loops. |
| `PERF_OPT_GUIDE.md` | 1106 | 1 | Gives a compiler-reported cycle proxy we can gate in experiments. |

## MIR Metrics

### `llvm/test/CodeGen/AIE/aie2p/BinaryOutput/vups.mir`

Contains concrete machine-instruction names for vups.2x and vups.4x.

- `vups_4x`: `148`
- `vups_2x`: `148`
- `vextbcst_16`: `0`
- `vextbcst_32`: `0`
- `vmac_f`: `0`
- `postpipeliner`: `0`
- `schedule_found`: `0`
- `hardware_loop`: `0`

### `llvm/test/CodeGen/AIE/aie2p/BinaryOutput/vextbcst.mir`

Contains concrete machine-instruction names for vextbcst.16.

- `vups_4x`: `0`
- `vups_2x`: `0`
- `vextbcst_16`: `90`
- `vextbcst_32`: `90`
- `vmac_f`: `0`
- `postpipeliner`: `0`
- `schedule_found`: `0`
- `hardware_loop`: `0`

### `llvm/test/CodeGen/AIE/aie2p/BinaryOutput/vmac.mir`

Contains concrete machine-instruction names for vmac.f forms.

- `vups_4x`: `0`
- `vups_2x`: `0`
- `vextbcst_16`: `0`
- `vextbcst_32`: `0`
- `vmac_f`: `390`
- `postpipeliner`: `0`
- `schedule_found`: `0`
- `hardware_loop`: `0`

### `llvm/test/CodeGen/AIE/aie2p/schedule/postpipeliner/gemm-bfp16-v10.mir`

Shows the MIR-level route into postpipeliner, remarks, ZOL, liveins, and fixed registers.

- `vups_4x`: `0`
- `vups_2x`: `0`
- `vextbcst_16`: `0`
- `vextbcst_32`: `0`
- `vmac_f`: `4`
- `postpipeliner`: `1`
- `schedule_found`: `2`
- `hardware_loop`: `4`

## Decision

- Do not continue trying to coerce high-level C++ into MyLM-like assembly.
- Do not copy LLVM backend C++ into IRON; it is too coupled to LLVM pass state.
- Use `llvm-aie` as a compiler service and as machine-contract metadata.
- Next implementation route: generate a tiny Q4NX MIR/basic-block template with explicit AIE2P machine instructions and physical register intent, then run Peano `llc`/`llvm-mc` and gate on assembly counts, postpipeliner remarks, bundle count, and NPU numeric equality.
- Fallback route if MIR is too brittle: keep source assembly, but generate it from TD-derived latency/bypass metadata instead of hand-inserting nops.

## Next Experiment

`022_q4nx_mir_schedule_probe`: start from the AIE2P MIR test style, build a tiny Q4NX group body using `VUPS_4x`, `VEXTBCST_16`, and `VMAC_f`, and verify whether `llc --start-before=postmisched` can produce a scheduled AIE2P object without C++-induced spills.
