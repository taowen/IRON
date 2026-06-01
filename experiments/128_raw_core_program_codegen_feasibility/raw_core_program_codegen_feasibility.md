# Raw Core Program Codegen Feasibility

- Status: `pass`
- Tiny asm ELF type: `EXEC`
- Tiny asm `.text`: `32` bytes
- Tiny asm transaction payloads: `[32]`
- Tiny asm aiecc xclbin bytes: `7268`
- ET_REL negative-control payloads: `[]`
- Main16 scaffold available: `True`
- MyLM raw available: `True`

## Verdict

| Check | Pass |
| --- | --- |
| `raw_asm_to_exec_elf` | `True` |
| `exec_elf_to_transaction_payload` | `True` |
| `aiecc_no_compile_xclbin_package` | `True` |
| `relocatable_object_is_not_enough` | `True` |
| `main16_scaffold_can_be_whole_core_elf` | `True` |
| `mylm_raw_bytes_can_be_wrapped_as_exec` | `True` |

## Toolchain Route

The viable route is:

```text
Python/codegen AIE2P asm or raw bytes
  -> Peano clang/ld.lld ET_EXEC whole-core ELF
  -> aie.core { aie.end } { elf_file = ... }
  -> aie-opt/aiecc --no-compile transaction/xclbin packaging
```

The negative control matters: a relocatable object can pass through the
MLIR verifier, but it does not generate a core-program blockwrite payload.
So production raw main16 must hand packaging an executable ELF with loadable
segments, not just a normal callable object.

The `aiecc --no-compile` probe produced an xclbin from the tiny external
ELF without recompiling it: `experiments/128_raw_core_program_codegen_feasibility/raw_codegen_probe.xclbin`, 7268 bytes.

## Main16 Scaffold

- ELF: `experiments/128_raw_core_program_codegen_feasibility/main16_scaffold_whole_core.elf`
- Sections: `{'.text': 6198, '.bss': 128, '.comment': 97}`
- Transaction payloads: `[6200, 128]`

This proves the current exp127 generated source asm can be promoted from
ET_REL object to a whole-core executable. Its BSS accumulator also becomes
an explicit zero-fill blockwrite, so fixed local-memory symbols must be
managed deliberately in the linker script.

## MyLM Raw Bytes

- Raw input: `/tmp/mylm_qwen3_8b_layer_redump/programs/c2r2_program.bin`
- Wrapped ELF: `experiments/128_raw_core_program_codegen_feasibility/mylm_c2r2_exec.elf`
- Sections: `{'.text': 14868}`
- Transaction payloads: `[14868]`

This checks the fallback route where our generator emits raw AIE2P bytes
directly and only uses an ELF wrapper for MLIR-AIE/aiebu packaging.
