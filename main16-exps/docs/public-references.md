# Public References

These are the public references that are useful when reading MyLM main16
disassembly. They explain the hardware and toolchain vocabulary, but they do
not document MyLM's private ABI.

## AMD / Xilinx Architecture And Kernel Docs

- AM020, Versal AI Engine-ML Architecture Manual:
  https://docs.amd.com/r/en-US/am020-versal-aie-ml/Overview
- AM020 functional overview:
  https://docs.amd.com/r/en-US/am020-versal-aie-ml/Functional-Overview
- UG1079, AI Engine Kernel and Graph Programming Guide:
  https://docs.amd.com/r/2023.1-English/ug1079-ai-engine-kernel-coding/Overview
- AI Engine intrinsics documentation index:
  https://download.amd.com/docnav/aiengine/aiengine_intrinsics_start.html
- UG1639, AI Engine-ML v2 Intrinsics User Guide:
  https://docs.amd.com/r/2025.1-English/AI-Engine-ML-v2-Intrinsics-User-Guide-UG1639

## Toolchain References

- Xilinx llvm-aie / Peano:
  https://github.com/Xilinx/llvm-aie
- MLIR-AIE AIE dialect, including `aie.core { elf_file = ... }`:
  https://xilinx.github.io/mlir-aie/AIEDialect.html
- IRON Python API:
  https://xilinx.github.io/mlir-aie/python/html/namespaceiron.html

## What These References Solve

- instruction semantics such as `acq`, `rel`, `vmac.f`, `vextbcst.16`,
  `vups.4x`, `vunpack`, and VLIW grouping;
- register-file and accumulator concepts;
- local tile memory, locks, DMA, streams, and AIE core program loading;
- legal assembler/compiler routes for AIE2P code.

## What They Do Not Solve

- MyLM entry/caller stack ABI;
- MyLM dispatcher control ABI;
- MyLM phase body arguments in `p0..p7` and `r*`;
- meaning of local addresses such as `0x73c00`, `0x73c60`, `0x73c80`,
  `0x78200`;
- MyLM record header and payload protocol;
- Q4NX activation/weight layout and numeric contract.
