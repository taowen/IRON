module @raw_core_codegen_probe {
  aie.device(npu2) {
    %t22 = aie.tile(2, 2)
    %c22 = aie.core(%t22) {
      aie.end
    } {elf_file = "mylm_c2r2_exec.elf"}
  }
}
