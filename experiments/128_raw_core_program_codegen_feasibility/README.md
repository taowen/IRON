# Exp128: Raw Core Program Codegen Feasibility

Goal: independently test whether the raw whole-core route is viable without
using linked C++/asm callable objects as the main16 hot path.

This experiment checks four narrow toolchain facts:

- Peano/llvm-aie can assemble source AIE2P asm and link it into an ET_EXEC
  whole-core ELF.
- `aie-opt --convert-aie-to-transaction=elf-dir=...` lowers an ET_EXEC
  `aie.core { aie.end } {elf_file = ...}` payload into core-program
  `config_blockwrite_data`.
- `aiecc --no-compile` can package the external ELF into an xclbin without
  recompiling the core program.
- An ET_REL object is not enough for this route: it verifies, but produces no
  core-program payload.
- The generated exp127 whole-main16 scaffold can be linked as a whole-core ELF
  and packaged into transaction blockwrites.

Run:

```bash
.venv/bin/python experiments/128_raw_core_program_codegen_feasibility/run.py
```

Optional MyLM raw-program input:

```bash
.venv/bin/python experiments/128_raw_core_program_codegen_feasibility/run.py \
  --mylm-raw /tmp/mylm_qwen3_8b_layer_redump/programs/c2r2_program.bin
```

Outputs:

- `raw_core_program_codegen_feasibility.md`
- `raw_core_program_codegen_feasibility.json`
- generated asm/linker/MLIR/ELF/transaction probe artifacts
