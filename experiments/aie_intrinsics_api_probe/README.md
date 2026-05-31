# AIE Intrinsics API Probe

This is a compile/disassembly experiment for AIE2P C++ kernel APIs and source
assembly. The only production-facing migration is a small unreferenced assembly
shape probe linked into the single qwen3 main16 role object; the active
numerical Q4NX path remains the exact C++ kernel until a bit-compatible source
assembly lane exists.

The experiment answers a narrow question: which AIE C++ API constructs lower to
the instruction shapes needed by a faster Qwen3 kernel?

Covered probes:

- `probe_asm_q4_group_shape`: hand-written AIE2P assembly for the minimal
  `32 vextbcst.16 + 33 vmac.f` group skeleton.
- `probe_asm_q4_group_with_prep_shape`: hand-written AIE2P assembly for a
  MyLM-style middle Q4NX group shape: unpack, upshift, activation broadcast,
  main MACs, and one group-sum correction MAC.
- `probe_asm_store_word_wrapper`: C++ wrapper calling a source-assembly symbol
  and linked into one relocatable object with `ld.lld -r`.
- `probe_asm_vector_mac_smoke_wrapper`: C++ wrapper calling a source-assembly
  function that receives three C ABI pointers, does `vldb + vmac.f + vst`, and
  links with relocations intact.
- `probe_asm_bf16_mac_result_release`: direct source-assembly NPU smoke body
  that reads a DMA-filled BF16 vector, computes `src * 1.0` with `vmac.f`,
  stores the accumulator through `vst.conv.bf16.fp32`, and releases the output
  lock from assembly.
- `probe_asm_q4_exact_unroll4_release`: direct source-assembly NPU smoke body
  for the first four exact Q4NX dimensions. It reads packed uint4, scale,
  offset, and activation values from one DMA-filled local buffer, executes the
  current `bf16(q * scale + offset)` per-dim rounding contract, uses
  `vextbcst.16 + vmac.f`, stores the BF16 accumulator, and releases the output
  lock from assembly.
- `probe_asm_q4_exact_group8_release`: direct source-assembly NPU smoke body
  for the first eight exact Q4NX dimensions. It stitches two 4-dim bodies with
  an explicit accumulator handoff and reloads scale/offset/activation before
  the second body because the first body clobbers those parameter registers.
- `probe_asm_q4_exact_group32_release`: direct source-assembly NPU smoke body
  for one complete 32-dim exact Q4NX group. It uses assembler macros to emit
  eight stitched 4-dim blocks, advances a pack-base pointer with `padda` to
  avoid out-of-range `vldb.128` immediates, and matches the BF16 reference on
  real NPU.
- `probe_asm_q4_exact_lane8_loop_release`: direct source-assembly NPU smoke
  body for one full 8-group / 16-row exact Q4NX lane. It reuses one scheduled
  32-dim group body in a counted branch loop, keeps scale/offset/activation on
  pointer registers, avoids the full-unroll program-memory overflow, and
  matches the BF16 reference on real NPU.
- `probe_bf16_load_store`: `aie::load_v` and `aie::store_v`.
- `probe_bf16_broadcast_extract_mac`: `extract`, scalar `broadcast`, and `aie::mac`.
- `probe_q4_unpack_broadcast_mac`: `uint4` load, `aie::unpack`, bf16 conversion, dequant, and MAC.
- `probe_native_q4_unpack_bridge_mac`: Q4 unpack plus native `vextbcst.16` MAC without scale/offset.
- `probe_native_q4_dequant_i16_activation_pair_mac`: native-style Q4 dequant plus two activation-lane MACs.
- `probe_native_q4_dequant_i16_activation_pair_mac_signed`: same native-style pair, but with the signed BF16 MAC control word used by MyLM.
- `probe_native_q4_group_sum_correction_pair_mac_signed`: MyLM-style `q*scale` main term plus one offset/group-sum correction MAC.
- `probe_native_q4_group_sum_correction_unroll8_signed`: same contract unrolled across eight activation lanes to test MAC density and vector spills.
- `probe_native_q4_group_sum_correction_unroll32_signed`: one full 32-lane quant group; should produce `33 vmac.f = 32 main MAC + 1 correction MAC`.
- `probe_native_q4_group_sum_correction_chunk_lane_kernel_signed`: one full 8-group / 16-row lane body, shaped like the static half of MyLM's Q4NX hot loop.
- `probe_native_q4_exact_rounding_unroll4/8/16/32_signed`: one group using the current exact per-dim BF16 rounding contract, with different unroll widths to find Peano's spill threshold.
- `probe_native_q4_exact_rounding_group4_call_chain_signed`: one exact group split into eight noinline 4-lane kernels, to test whether function boundaries can reduce spill without destroying the hot-loop shape.
- `probe_native_q4_exact_rounding_group8_call_chain_signed`: one exact group split into four noinline 8-lane kernels, to test whether fewer calls are possible without reintroducing spill.
- `probe_native_q4_exact_rounding_group16_call_chain_signed`: one exact group split into two noinline 16-lane kernels, to test the largest plausible C++ ABI boundary before source assembly.
- `probe_native_q4_exact_rounding_chunk_lane_kernel_signed`: one full 8-group / 16-row lane body using the exact current rounding contract.
- `probe_native_q4_group_sum_correction_chunk_two_lane_calls_signed`: wrapper that reuses the noinline lane body twice instead of duplicating two lanes in one giant function.
- `probe_native_q4_v32load_dequant_pair_mac`: production-safe `uint4 x 32` load followed by native-style dequant/MAC.
- `probe_lock_counted_loop`: Peano lock builtins and a counted loop.
- `probe_builtin_broadcast_elem_i16`: Peano compat vector-lane broadcast for `v32int16`.
- `probe_builtin_broadcast_elem_bf16`: Peano compat vector-lane broadcast for `v32bfloat16`.
- `probe_builtin_shuffle_bf16`: Peano compat scalar broadcast with shuffle.

Run:

```bash
python3 experiments/aie_intrinsics_api_probe/run_probe.py --force
python3 experiments/aie_intrinsics_api_probe/run_asm_probe.py
python3 experiments/aie_intrinsics_api_probe/run_asm_link_probe.py
.venv/bin/python experiments/aie_intrinsics_api_probe/run_asm_npu_smoke.py
python3 experiments/aie_intrinsics_api_probe/analyze_mylm_main16_kernel.py
python3 experiments/aie_intrinsics_api_probe/compare_q4nx_group_sum_reference.py
```

The script uses:

- Peano clang from `.venv/lib/python3.12/site-packages/llvm-aie`.
- AIE API headers from `.venv/lib/python3.12/site-packages/mlir_aie/include`.
- mlir-aie source at `/var/home/taowen/projects/mlir-aie` for source-code study.
- llvm-aie source at `/var/home/taowen/projects/llvm-aie` for Peano header,
  builtin, instruction-selection, and scheduling study.

The build output goes to `experiments/aie_intrinsics_api_probe/build/`, which is
ignored by git.

See `RESULTS.md` and `MYLM_MAIN16_KERNEL.md` for the latest observed
instruction counts, MyLM hot-loop semantics, and source-code notes.

Current purpose:

1. Learn the system API surface before touching the production qwen3 kernel.
2. Compare API-generated instruction shape with the MyLM-style hot loop.
3. Keep C++ role-object linking plus source assembly runnable, so production can
   replace one complete hot kernel body at a time without introducing a second
   main16 role object. Do not use source assembly as a tail-jump wrapper back
   into C++; on this AIE2P assembler path that resolves to a bad `j #0`.
   The supported direction is C++ scheduler/wrapper calling source assembly; the
   wrapper carries `R_AIE_1` relocations to the asm symbol before final linking.
4. Do not use the MyLM-style group-sum shortcut as a production drop-in. It
   changes the current per-dim BF16 Q4NX rounding contract and failed the QKV
   compact-output slice.
5. Do not migrate the full-lane exact-rounding C++ intrinsic body as-is. Peano
   already spills at 8-lane exact unroll (`vst=15`) and becomes unusable at a
   full 8-group lane body (`vst=1043`). Splitting into noinline 4-lane kernels
   removes the kernel-local spill but replaces one group with eight calls and no
   hardware loop. The larger noinline ABI candidates do not save the C++ route:
   8-lane noinline already has `vst=20`, and 16-lane noinline has `vst=55`.
   The exact body still needs source assembly or a lower-level scheduler.
6. Source assembly is now proven on real NPU with a DMA-filled local input
   buffer, exact Q4NX rounding, and accumulator writeback for a full 8-group
   lane: `run_asm_npu_smoke.py` calls
   `probe_asm_q4_exact_lane8_loop_release`, which performs unpack/upshift,
   `bf16(q * scale + offset)`, `vextbcst.16 + vmac.f`, BF16 accumulator
   storeback, and lock release. The important ABI lessons are that
   hand-written AIE assembly must schedule load-use/MAC-store latency
   explicitly, full static lane unroll overflows program memory, large
   source-level hardware loops are not a safe assumption, and scale/offset/
   activation should use pointer registers rather than loop-carried `dj0`
   offset registers.
