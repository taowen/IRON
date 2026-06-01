module @mylm_main16_standalone_raw_kernel {
  aie.device(npu2) {
    %t22 = aie.tile(2, 2)
    %activation_empty = aie.lock(%t22, 0) {init = 2 : i32, sym_name = "activation_empty"}
    %activation_full = aie.lock(%t22, 1) {init = 0 : i32, sym_name = "activation_full"}
    %weight_empty = aie.lock(%t22, 2) {init = 2 : i32, sym_name = "weight_empty"}
    %weight_full = aie.lock(%t22, 3) {init = 0 : i32, sym_name = "weight_full"}
    %record_empty = aie.lock(%t22, 4) {init = 2 : i32, sym_name = "record_empty"}
    %record_full = aie.lock(%t22, 5) {init = 0 : i32, sym_name = "record_full"}
    %start_lock = aie.lock(%t22, 6) {init = 0 : i32, sym_name = "start_lock"}
    %c22 = aie.core(%t22) {
      aie.end
    } {elf_file = "mylm_c2r2_main16_exec.elf"}
    aie.runtime_sequence() {
      aiex.set_lock(%start_lock, 1)
    }
  }
}
