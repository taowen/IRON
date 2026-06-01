module @raw_core_codegen_probe {
  aie.device(npu2) {
    memref.global "private" constant @config_blockwrite_data_0 : memref<8xi32> = dense<[458776, 2295832, 268437272, 67108996, 0, 0, 0, 268435480]>
    aie.runtime_sequence @configure() {
      aiex.npu.maskwrite32 {address = 69410816 : ui32, mask = 1 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 69328400 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.maskwrite32 {address = 69328408 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.maskwrite32 {address = 69328384 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.maskwrite32 {address = 69328392 : ui32, mask = 2 : ui32, value = 2 : ui32}
      %0 = memref.get_global @config_blockwrite_data_0 : memref<8xi32>
      aiex.npu.blockwrite(%0) {address = 69337088 : ui32} : memref<8xi32>
      aiex.npu.maskwrite32 {address = 69328400 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 69328408 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 69328384 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 69328392 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 69410816 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.maskwrite32 {address = 69410816 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 69332992 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 69333008 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 69333024 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 69333040 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 69333056 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 69333072 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 69333088 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 69333104 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 69333120 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 69333136 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 69333152 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 69333168 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 69333184 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 69333200 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 69333216 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 69333232 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 69410816 : ui32, mask = 1 : ui32, value = 1 : ui32}
    }
    %tile_2_2 = aie.tile(2, 2)
    %core_2_2 = aie.core(%tile_2_2) {
      aie.end
    } {elf_file = "raw_codegen_probe.elf"}
  }
}

