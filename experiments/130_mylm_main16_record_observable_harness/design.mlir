module @mylm_main16_record_observable_harness {
  aie.device(npu2) {
    %shim_in = aie.tile(2, 0)
    %shim_tap = aie.tile(3, 0)
    %t22 = aie.tile(2, 2)

    aie.flow(%shim_in, DMA : 0, %t22, DMA : 0)
    aie.flow(%shim_in, DMA : 1, %t22, DMA : 1)
    aie.flow(%t22, DMA : 1, %shim_tap, DMA : 1)

    %activation_ping = aie.buffer(%t22) {address = 32768 : i32, sym_name = "activation_ping"} : memref<128xi32>
    %activation_pong = aie.buffer(%t22) {address = 49152 : i32, sym_name = "activation_pong"} : memref<128xi32>
    %weight_ping = aie.buffer(%t22) {address = 10240 : i32, sym_name = "weight_ping"} : memref<2560xbf16>
    %weight_pong = aie.buffer(%t22) {address = 16384 : i32, sym_name = "weight_pong"} : memref<2560xbf16>
    %record_ping = aie.buffer(%t22) {address = 15388 : i32, sym_name = "record_ping"} : memref<17xi32>
    %record_pong = aie.buffer(%t22) {address = 21532 : i32, sym_name = "record_pong"} : memref<17xi32>

    %activation_empty = aie.lock(%t22, 0) {init = 2 : i32, sym_name = "activation_empty"}
    %activation_full = aie.lock(%t22, 1) {init = 0 : i32, sym_name = "activation_full"}
    %weight_empty = aie.lock(%t22, 2) {init = 2 : i32, sym_name = "weight_empty"}
    %weight_full = aie.lock(%t22, 3) {init = 0 : i32, sym_name = "weight_full"}
    %record_empty = aie.lock(%t22, 4) {init = 2 : i32, sym_name = "record_empty"}
    %record_full = aie.lock(%t22, 5) {init = 0 : i32, sym_name = "record_full"}
    %start_lock = aie.lock(%t22, 6) {init = 0 : i32, sym_name = "start_lock"}

    %c22 = aie.core(%t22) {
      aie.end
    } {elf_file = "mylm_c2r2_main16_record_exec.elf"}

    %m22 = aie.mem(%t22) {
      %activation_dma = aie.dma_start(S2MM, 0, ^activation_ping_in, ^weight_start)
    ^activation_ping_in:
      aie.use_lock(%activation_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%activation_ping : memref<128xi32>, 0, 128) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%activation_full, Release, 1)
      aie.next_bd ^activation_pong_in
    ^activation_pong_in:
      aie.use_lock(%activation_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%activation_pong : memref<128xi32>, 0, 128) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%activation_full, Release, 1)
      aie.next_bd ^activation_ping_in

    ^weight_start:
      %weight_dma = aie.dma_start(S2MM, 1, ^weight_ping_in, ^record_start)
    ^weight_ping_in:
      aie.use_lock(%weight_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%weight_ping : memref<2560xbf16>, 0, 2560) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%weight_full, Release, 1)
      aie.next_bd ^weight_pong_in
    ^weight_pong_in:
      aie.use_lock(%weight_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%weight_pong : memref<2560xbf16>, 0, 2560) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%weight_full, Release, 1)
      aie.next_bd ^weight_ping_in

    ^record_start:
      %record_dma = aie.dma_start(MM2S, 1, ^record_ping_out, ^end)
    ^record_ping_out:
      aie.use_lock(%record_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%record_ping : memref<17xi32>, 0, 17) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%record_empty, Release, 1)
      aie.next_bd ^record_pong_out
    ^record_pong_out:
      aie.use_lock(%record_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%record_pong : memref<17xi32>, 0, 17) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%record_empty, Release, 1)
      aie.next_bd ^record_ping_out
    ^end:
      aie.end
    }

    aie.runtime_sequence(%out: memref<1224xi32>, %weights: memref<1474560xi32>, %activation: memref<147456xi32>) {
      aiex.npu.writebd {bd_id = 2 : i32, buffer_length = 1224 : i32, buffer_offset = 0 : i32, burst_length = 64 : i32, column = 3 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 100782148 : ui32, arg_idx = 0 : i32, arg_plus = 0 : i32}
      aiex.npu.push_queue(3, 0, S2MM : 1) {bd_id = 2 : i32, issue_token = true, repeat_count = 0 : i32}
      aiex.npu.writebd {bd_id = 12 : i32, buffer_length = 147456 : i32, buffer_offset = 0 : i32, burst_length = 64 : i32, column = 2 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 67228036 : ui32, arg_idx = 2 : i32, arg_plus = 0 : i32}
      aiex.npu.push_queue(2, 0, MM2S : 0) {bd_id = 12 : i32, issue_token = true, repeat_count = 0 : i32}
      aiex.npu.writebd {bd_id = 14 : i32, buffer_length = 1474560 : i32, buffer_offset = 0 : i32, burst_length = 64 : i32, column = 2 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 67228100 : ui32, arg_idx = 1 : i32, arg_plus = 0 : i32}
      aiex.npu.push_queue(2, 0, MM2S : 1) {bd_id = 14 : i32, issue_token = true, repeat_count = 0 : i32}
      aiex.set_lock(%start_lock, 1)
      aiex.npu.sync {channel = 1 : i32, column = 3 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
    }
  }
}
