# AIE Intrinsics API Probe Results

Run command:

```bash
.venv/bin/python experiments/aie_intrinsics_api_probe/run_probe.py --force
python3 experiments/aie_intrinsics_api_probe/run_asm_probe.py
python3 experiments/aie_intrinsics_api_probe/run_asm_link_probe.py
.venv/bin/python experiments/aie_intrinsics_api_probe/run_asm_npu_smoke.py
python3 experiments/aie_intrinsics_api_probe/analyze_mylm_main16_kernel.py
python3 experiments/aie_intrinsics_api_probe/compare_q4nx_group_sum_reference.py
```

Source baselines:

- `Xilinx/mlir-aie` cloned at `/var/home/taowen/projects/mlir-aie`, commit `9dd5f63`.
- `Xilinx/llvm-aie` cloned at `/var/home/taowen/projects/llvm-aie`, commit `05a016679`.
- Peano clang and AIE API headers come from the repo virtualenv.

Observed instruction shape:

Source-assembly probes:

| probe | key result |
| --- | --- |
| `probe_asm_q4_group_shape` | source `.s` compiles to `elf32-aie` with `vmac.f=33`, `vextbcst.16=32`, `vbcst.16=1`, `vst=0` |
| `probe_asm_q4_group_with_prep_shape` | source `.s` compiles with `vmac.f=33`, `vextbcst.16=32`, `vbcst.16=1`, `lda.s16=1`, `vlda=1`, `vldb=3`, `vunpack=8`, `vups=8`, `vst=0` |
| `asm_link_probe` | `clang++` C++ wrapper object plus `clang` source-assembly object link into one relocatable AIE object with `ld.lld -r`; no unresolved symbols; C++ wrappers can call asm symbols through `R_AIE_1` relocations |
| `probe_asm_vector_mac_smoke` | C ABI source-assembly function with `p0/p1/p2` pointer arguments; disassembly has `vldb=2`, `vmac.f=1`, `vst=1` |
| `probe_asm_bf16_mac_result_release` | direct source-assembly NPU smoke body; DMA fills `src`, asm executes `vlda x0, [p0]`, consumes it with `vmac.f`, stores accumulator result with `vst.conv.bf16.fp32`, releases the output lock, and host verifies the BF16 result |

C++/intrinsics probes:

| probe | key instructions |
| --- | --- |
| `probe_bf16_load_store` | `vldb=1`, `vst=1`, hardware loop present |
| `probe_bf16_broadcast_extract_mac` | `vmac.f=1`, `vbcst.16=1`, scalar bf16 load, vector load/store, hardware loop present |
| `probe_q4_unpack_broadcast_mac` | `vunpack=2`, `vups=1`, `vmul.f=1`, `vmac.f=1`, `vbcst.16=4`, vector load/store, hardware loop present |
| `probe_native_bf16_vextbcst_mac` | pure Peano native `v32bfloat16/v32accfloat`, `vextbcst.32=1`, `vmac.f=1`, no scalar bf16 load, hardware loop present |
| `probe_native_bf16_acc16_vextbcst_mac` | pure Peano native `v32bfloat16/v16accfloat`, `vextbcst.32=1`, `vmac.f=1`, no scalar bf16 load, hardware loop present |
| `probe_native_bf16_i16_view_acc16_mac` | BF16 source viewed as `v32int16`, but memory scalarized to `lda.s16 + vbcst.16 + vmac.f`; no `vextbcst.16` |
| `probe_arg_i16_vextbcst` | vector argument `v32int16 -> broadcast_elem` lowers to `vextbcst.16` |
| `probe_arg_i16_view_bf16_acc16_mac` | vector argument `v32int16` bitcast back to BF16 and fed to `mac_elem_16_conf`; lowers to `vextbcst.16 + vmac.f` |
| `probe_arg_i16_view_bf16_acc16_mac_signed` | same as above, but using the signed BF16 MAC overload; lowers to `vextbcst.16 + vmac.f` with control `#0x33c`, matching MyLM's Q4 hot loop |
| `probe_native_q4_unpack_bridge_mac` | Q4 unpack plus native i16-view activation MAC; `vmac.f=2`, `vextbcst.16=2`, `vunpack=1`, `vups=1`, `vst=2` |
| `probe_native_q4_dequant_i16_activation_pair_mac` | native-style Q4 dequant plus two activation-lane MACs; `vmac.f=2`, `vmul.f=2`, `vadd=3`, `vextbcst.16=2`, `vst=2` |
| `probe_native_q4_dequant_i16_activation_pair_mac_signed` | same Q4 pair shape with signed BF16 MAC control `#0x33c`; this is the first Peano C++ source shape that matches MyLM's `vextbcst.16 + vmac.f r4=0x33c` contract |
| `probe_native_q4_group_sum_correction_pair_mac_signed` | MyLM-style `q*scale*activation + offset*sum(activation_group)` pair; `vmac.f=3`, `vmul.f=2`, `vadd=1`, `vextbcst.16=2`, `vconv.bf16.fp32=3`, `vst=2` |
| `probe_native_q4_group_sum_correction_unroll8_signed` | same contract unrolled across 8 activation lanes; `vmac.f=9`, `vmul.f=8`, `vadd=4`, `vextbcst.16=8`, `vconv.bf16.fp32=12`, `vst=1` tail output only |
| `probe_native_q4_group_sum_correction_unroll32_signed` | one full MyLM-style 32-lane quant group; `vmac.f=33`, `vmul.f=32`, `vadd=16`, `vextbcst.16=32`, `vconv.bf16.fp32=48`, `vst=1` tail output only |
| `probe_native_q4_exact_rounding_unroll4_signed` | exact current-reference per-dim rounding across 4 activation lanes; `vmac.f=4`, `vextbcst.16=4`, `vconv.bf16.fp32=10`, `vconv.fp32.bf16=5`, `vst=2` |
| `probe_native_q4_exact_rounding_unroll8_signed` | exact current-reference per-dim rounding across 8 activation lanes; `vmac.f=8`, `vextbcst.16=8`, `vconv.bf16.fp32=20`, `vconv.fp32.bf16=9`, `vst=15` |
| `probe_native_q4_exact_rounding_unroll16_signed` | exact current-reference per-dim rounding across 16 activation lanes; `vmac.f=16`, `vextbcst.16=16`, `vconv.bf16.fp32=40`, `vconv.fp32.bf16=17`, `vst=51` |
| `probe_native_q4_exact_rounding_group4_dim0_kernel_signed` | exact current-reference 4-lane kernel with accumulator passed through the function ABI; `vmac.f=4`, `vextbcst.16=4`, `vconv.bf16.fp32=10`, `vconv.fp32.bf16=6`, `vst=0` |
| `probe_native_q4_exact_rounding_group4_call_chain_signed` | one exact 32-lane group split into eight noinline 4-lane calls; wrapper has `jl=8`, `vst=4`, and no hardware loop |
| `probe_native_q4_exact_rounding_group8_dim0_kernel_signed` | exact current-reference 8-lane kernel with accumulator passed through the function ABI; `vmac.f=8`, `vextbcst.16=8`, `vconv.bf16.fp32=20`, `vconv.fp32.bf16=10`, `vst=20` |
| `probe_native_q4_exact_rounding_group8_call_chain_signed` | one exact 32-lane group split into four noinline 8-lane calls; wrapper has `jl=4`, `vst=4`, and no hardware loop, but each callee already spills |
| `probe_native_q4_exact_rounding_group16_dim0_kernel_signed` | exact current-reference 16-lane kernel with accumulator passed through the function ABI; `vmac.f=16`, `vextbcst.16=16`, `vconv.bf16.fp32=40`, `vconv.fp32.bf16=18`, `vst=55` |
| `probe_native_q4_exact_rounding_group16_call_chain_signed` | one exact 32-lane group split into two noinline 16-lane calls; wrapper has `jl=2`, `vst=4`, and no hardware loop, but each callee spills too heavily |
| `probe_native_q4_exact_rounding_unroll32_signed` | one exact current-reference 32-lane quant group; `vmac.f=32`, `vmul.f=32`, `vadd=48`, `vextbcst.16=32`, `vconv.bf16.fp32=80`, `vconv.fp32.bf16=33`, but `vst=126`, so this shape is not viable |
| `probe_native_q4_group_sum_correction_chunk_lane_kernel_signed` | one full 8-group / 16-row lane body; `vmac.f=264`, `vextbcst.16=256`, `vst=5`, but `vunpack=256`, `vups=128`, `vconv.bf16.fp32=384` remain higher than MyLM |
| `probe_native_q4_exact_rounding_chunk_lane_kernel_signed` | one exact current-reference 8-group / 16-row lane body; `vmac.f=256`, `vextbcst.16=256`, `vconv.bf16.fp32=640`, `vconv.fp32.bf16=257`, and `vst=1043`, so full exact C++ unroll spills catastrophically |
| `probe_native_q4_exact_rounding_chunk_two_lane_calls_signed` | wrapper around the exact noinline lane body; avoids wrapper duplication, but the callee body is already too spill-heavy |
| `probe_native_q4_group_sum_correction_chunk_two_lanes_signed` | two lanes statically duplicated; `vmac.f=528`, but `vst=824`, so this shape is not viable |
| `probe_native_q4_group_sum_correction_chunk_lane_loop_signed` | one giant lane body inside a two-iteration loop; hardware loop appears, but `vst=251`, so Peano spills heavily when the body is looped |
| `probe_native_q4_group_sum_correction_chunk_two_lane_calls_signed` | wrapper that calls the noinline lane body twice; wrapper has `jl=1` plus a tail jump and avoids duplicating the hot body |
| `probe_native_q4_dequant_i16_activation_pair_mac_pipelined` | same opcode shape as the non-pipelined pair probe; the Chess loop hints did not change this small loop materially |
| `probe_native_q4_dequant_accfloat_pair_mac` | keeps dequant intermediate in `accfloat`; same `vextbcst.16 + vmac.f` shape with fewer bf16 conversion ops, but changes rounding semantics |
| `probe_native_q4_v32load_dequant_pair_mac` | production-safe `uint4 x 32` load shape; `vmac.f=2`, `vextbcst.16=2`, `vunpack=2`, `vups=1`, `vst=2` |
| `probe_lock_counted_loop` | `acq=1`, `rel=1`, hardware loop present |
| `probe_builtin_broadcast_elem_i16` | scalar extract plus `vbcst.16=1`, not `vextbcst` |
| `probe_builtin_broadcast_elem_bf16` | `vextbcst.32=1`, vector load/store |
| `probe_builtin_shuffle_bf16` | `vbcstshfl.16=1`, vector store |

Conclusions:

1. The normal C++ AIE API can generate the basic pieces we need: vector load/store, hardware counted loops, lock acquire/release, Q4 unpack, bf16 conversion, floating multiply, and floating MAC.
2. Scalar `aie::broadcast<bfloat16, N>(value)` lowers to scalar load plus `vbcst.16`. This is usable, but it is not the same as vector-lane extract-and-broadcast.
3. Peano compat `broadcast_elem(v32bfloat16, idx)` lowers to `vextbcst.32`. That is the first confirmed direct path from C++ source to a `vextbcst` instruction on this local toolchain.
4. Peano compat `shuffle_bfloat16(value, mode)` lowers to `vbcstshfl.16`; this is scalar broadcast plus shuffle, not vector-lane extract.
5. Peano compat `broadcast_elem(v32int16, idx)` can lower to `vextbcst.16`, but only when the source is already a vector value. If the source is a memory vector used only for the broadcast, Peano scalarizes it to `lda.s16 + vbcst.16`.
6. `aie::vector<uint4, 16>` does not compile on this installed AIE API; Q4 vector loads need a supported vector size such as `uint4 x 32`, matching the existing qwen3 kernel style.
7. Do not bridge Peano compat `v32bfloat16` back into `aie::vector<bfloat16, 16>` for production. That mixed type path is not what MyLM's raw scheduled loop does, and it risks introducing hidden extracts, stores, or register pressure.
8. A same-type native path is viable for BF16 MAC: `v32bfloat16` plus `broadcast_elem(v32bfloat16, idx)` feeding `mac_elem_32` or `mac_elem_16_conf` directly emits `vextbcst.32 + vmac.f` with no scalar bf16 load.
9. `vextbcst.16 + vmac.f` is possible from Peano, but it is not produced by BF16's `broadcast_elem(v32bfloat16, idx)` builtin. It requires a 16-bit integer vector view (`v32int16`) that remains in a vector register and is then bitcast back to `v32bfloat16` before the MAC.
10. The native BF16 MAC proof is not yet a production Q4NX kernel. Q4NX unpack, uint-to-bf16 conversion, scale/offset, and accumulator storeback still need to close inside the same native type system before replacing `main_projection_q4nx_fast.cc`.
11. Directly transplanting the default native `mac_elem_16_conf` path into the production main16 kernel is not numerically equivalent to the existing `aie::mac` path. It can pass Q/K/V compact and cache-writeback tests, but fails the attention-O and full-layer hidden-output contracts.
12. MyLM's Q4 hot loop does not use the default BF16 MAC control word. It materializes `r4 = 0x33c`, while Peano's default `mac_elem_16_conf(a, b, acc, 0, 0, 0)` materializes `0x3c`.
13. `0x33c` decodes to `aie2p_compute_control(__SIGN_SIGNED, __SIGN_SIGNED, 2, 3, 1, 0, 0, 0, 0, 0, 0)`. The signed overload
    `mac_elem_16_conf(a, __SIGN_SIGNED, b, __SIGN_SIGNED, acc, 0, 0, 0)` reproduces this exact control word.
14. The earlier failed production transplant used the default/native unsigned control shape. That is now a concrete root-cause candidate for the attention-O numeric mismatch; the next production attempt must use the signed overload and re-run the full integration boundary.
15. MyLM's Q4NX math shape is not the scalar-looking `((q * scale) + offset) * activation` loop used by the current production reference. It hoists the offset term per 32-column quant group:

    ```text
    sum_d ((q_d * scale) + offset) * activation_d
      = sum_d (q_d * scale) * activation_d
        + offset * sum_d activation_d
    ```

    The phase body computes the eight `sum_d activation_d` values into a local
    `s16` scratch stream; the Q4 hot loop consumes them through `p3`. The new
    group-sum probes are the first C++/Peano form that expresses this contract.
16. The group-sum rewrite materially improves generated shape before touching
    production: the old signed per-pair probe has `2 vmac.f / 64 lines` and
    several dequant conversions/adds; the group-sum pair has `3 vmac.f / 61
    lines`; the unroll8 group-sum probe has `9 vmac.f / 90 lines`; the full
    unroll32 quant-group probe has `33 vmac.f / 189 lines`. Both unrolled
    probes have exactly one `vst`, which is the final output store, not a
    loop-body vector spill. This is not enough to claim production correctness,
    but it explains a real part of MyLM's speed rather than just naming the
    instruction.
17. The exact current-reference full-unroll C++ intrinsic route is now ruled
    out as a production migration. It does preserve the per-dim rounding shape,
    and it can generate the desired `vextbcst.16 + vmac.f` count, but Peano
    spills too aggressively. The spill threshold appears immediately after the
    smallest exact bodies:

    ```text
    exact_rounding_unroll4:
      vmac.f=4
      vextbcst.16=4
      vst=2

    exact_rounding_unroll8:
      vmac.f=8
      vextbcst.16=8
      vst=15

    exact_rounding_unroll16:
      vmac.f=16
      vextbcst.16=16
      vst=51

    exact_rounding_group4_call_chain:
      aggregate one-group body: vmac.f=32, vextbcst.16=32, vst=4
      wrapper: jl=8, hardware_loop=0

    exact_rounding_group8_call_chain:
      callee dim0: vmac.f=8, vextbcst.16=8, vst=20
      wrapper: jl=4, hardware_loop=0

    exact_rounding_group16_call_chain:
      callee dim0: vmac.f=16, vextbcst.16=16, vst=55
      wrapper: jl=2, hardware_loop=0

    exact_rounding_unroll32:
      vmac.f=32
      vextbcst.16=32
      vst=126

    exact_rounding_chunk_lane_kernel:
      vmac.f=256
      vextbcst.16=256
      vst=1043
    ```

    The 4-lane exact body avoids the catastrophic spill, but it is too small to
    match MyLM's MAC density. Splitting a 32-lane group into eight noinline
    4-lane kernels keeps those kernels spill-free, but the wrapper now has
    eight calls per group and does not form a hardware loop. The natural attempt
    to reduce that call count also fails: the 8-lane noinline callee already has
    `vst=20`, and the 16-lane callee has `vst=55`. The exact C++ ABI boundary
    therefore has no useful middle ground between too many calls and too much
    spill. The full chunk-lane body is worse than the active production wrapper
    (`vst=194`) despite having more static MACs. The next exact path must
    therefore be source assembly or a lower-level scheduler that explicitly owns
    register allocation.
18. The source-assembly path is now proven locally. Peano's `clang` integrated
    assembler accepts hand-written AIE2P `.s`, and `ld.lld -r` can merge that
    object with a C++ object into one role object. This is the right production
    integration boundary: keep MLIR/IRON and the C++ scheduler as the owner of
    phase control, but replace the Q4NX hot lane body with a source assembly
    symbol only after the C ABI and numerical storeback are nailed down. The
    callable ABI smoke now covers more than a scalar store:

    ```text
    probe_asm_vector_mac_smoke:
      p0=lhs, p1=rhs, p2=dst
      vldb=2
      vmac.f=1
      vst=1

    probe_asm_vector_mac_smoke_wrapper:
      j #0 with R_AIE_1 relocation to the asm symbol
    ```

    This proves the supported direction for production: C++ scheduler calls a
    complete source-assembly hot body. It does not rely on asm tail-jumping back
    into C++.
19. Direct source-assembly execution is also proven on real NPU. The smoke
    case first isolated several false leads:

    ```text
    qwen3-8b-c1r2-input-norm-replay:
      existing host/shim -> c1r2 input path still passes exact reference

    C++ external copy on the same smoke dataflow:
      dst[0] = src[0] passes

    source asm constant store:
      dst[0] = 0x1234 passes

    naive source asm scalar copy:
      lda r1, [p0, #0]
      st  r1, [p1, #0]
      fails with got 0
    ```

    The root cause was not NPU runtime, host buffer flush, lock ownership, or
    route setup. It was AIE load-use scheduling. Peano-generated C++ places a
    long enough gap between `lda` and the dependent `st`; the naive source asm
    consumed the scalar register too early. With explicit latency padding, the
    source-assembly scalar path passes, and the vector-copy path passes:

    ```text
    vlda x0, [p0, #0]
    ...
    vmac.f dm1, dm1, x0, x0, r4
    vst wl0, [p1, #0]
    vst wh0, [p1, #0x20]

    expected[0:8] = [4096, 4097, 4098, 4099, 4100, 4101, 4102, 4103]
    got[0:8]      = [4096, 4097, 4098, 4099, 4100, 4101, 4102, 4103]
    ```

    The first passing accumulator smoke wrote the BF16 MAC result, not the
    original vector:

    ```text
    vlda x0, [p0, #0]
    vbcst.16 x2, 0x3f80
    vmac.f dm0, dm0, x0, x2, 0x33c
    vst.conv.bf16.fp32 bmll0, [p1, #0]

    expected[0:8] = [-1084178560, -1088372928, -1094664448, -1107247488,
                     1040187392, 1052786304, 1059077888, 1063272256]
    got[0:8]      = [-1084178560, -1088372928, -1094664448, -1107247488,
                     1040187392, 1052786304, 1059077888, 1063272256]
    ```

    The smoke has now been advanced past a complete 32-dim Q4NX exact-rounding
    group to a full 8-group / 16-row exact lane. The earlier 4-dim, 8-dim, and
    32-dim versions remain in the linked object as smaller accumulator-store,
    accumulator-handoff, and one-group proofs:

    ```text
    packed uint4 bytes        -> vunpack/vups
    scale + offset            -> bf16(q * scale + offset)
    activation lanes 0..31    -> vextbcst.16 + vmac.f
    accumulator               -> vst.conv.bf16.fp32

    expected[0:8] = [1035091350, 1038761422, 1041317379, 1043152415,
                     1045052988, 1046888024, 1048657524, 1049575048]
    got[0:8]      = [1035091350, 1038761422, 1041317379, 1043152415,
                     1045052988, 1046888024, 1048657524, 1049575048]
    ```

    The current lane smoke is:

    ```text
    expected[0:8] = [-1112883810, -1111310922, -1109738034, -1108165146,
                     -1106919938, -1106133493, -1105347049, -1104560605]
    got[0:8]      = [-1112883810, -1111310922, -1109738034, -1108165146,
                     -1106919938, -1106133493, -1105347049, -1104560605]
    ```

    This is the first end-to-end proof that a source-assembly hot body can read
    DMA-filled tile-local Q4NX inputs for all 8 quant groups in one lane,
    execute the current exact per-dim rounding contract, convert accumulator
    output back to BF16, release the output lock, and be observed by the host.
    It also explains why source assembly cannot be written as a plain linear
    mnemonic list; it must own instruction latency and branch-delay scheduling
    the same way MyLM's raw program does. The first Q4NX smoke attempt stored
    the offset vector, not the MAC result, until extra padding was inserted
    after the final `vmac.f`. The first 8-dim stitch failed until
    scale/offset/activation were reloaded before the second 4-dim body; those
    vector registers are working registers, not immutable call parameters. The
    first 32-dim group compile failed because `vldb.128 [p0,#imm]` cannot
    encode offsets past 127; the final version advances `p5` with `padda`
    before each 4-dim block. A full static lane unroll then failed CDO
    generation with program-memory overflow, so the passing lane uses a counted
    branch loop. A first loop version also proved that `add #0x40` encodes as
    `-64` on this path and that loop-carried `dj0` offset registers left
    offset/activation stuck on group0; the passing version keeps
    scale/offset/activation in pointer registers `p2/p3/p4`.
20. The same packaging boundary is now active in `qwen3-layer`: the single
    main16 role object `main_projection_q4nx_fast.o` is built from
    `main_projection_q4nx_fast.cc` plus `main_projection_q4nx_asm.s`. The `.s`
    file currently contains only an unreferenced group-shape probe. The final
    `main_core_2_2.elf` has no `q4nx_accum_lane_asm_group_shape` symbol because
    the probe section is garbage-collected. This proves source assembly can be
    introduced without adding a second main16 role or spending final program
    memory until the numerical asm lane is actually called.
21. A source-assembly wrapper that tail-jumps to a C++ semantic function is not
    a valid transition step on this toolchain. Testing `j #cpp_symbol` from
    hand-written AIE2P `.s` produced a final `j #0`, and the intended C++ target
    was garbage-collected. The production route must therefore keep the current
    C++ lane until the source-assembly replacement contains the complete
    numerical hot body. The checker now rejects this bad symbol-jump pattern so
    it cannot accidentally reach a real NPU run.
22. A MyLM-style group-sum lane body was also tested as an active production
    replacement and rejected. The pointer ABI bug in the first attempt was a
    real issue: `uint4* + kLaneNibbles` compiled to the chunk end (`+0x1400`)
    instead of the second lane data offset (`+0x0c00`), so production code must
    use explicit byte offsets for packed Q4NX lane addressing. After fixing the
    pointer, the active group-sum body still failed `qwen3-8b-qkv-compact-output`
    with `payload_max_abs=0.112304688`. The root cause is semantic, not routing:
    group-sum hoists the offset term out of the per-dim `bf16(q * scale +
    offset)` rounding used by the current reference. The production path was
    restored to the exact C++ kernel; `qwen3-8b-qkv-compact-output` then passed
    with `payload_max_abs=0.000000477`, and `full-layer-qkv-prefix` passed K/V
    writeback.

Full-group probe comparison:

| probe | lines | `vmac.f` | lines / `vmac.f` | `vst` | `vextbcst.16` |
| --- | ---: | ---: | ---: | ---: | ---: |
| signed per-pair dequant | 64 | 2 | 32.00 | 2 | 2 |
| group-sum pair | 61 | 3 | 20.33 | 2 | 2 |
| group-sum unroll8 | 90 | 9 | 10.00 | 1 | 8 |
| group-sum unroll32 | 189 | 33 | 5.73 | 1 | 32 |
| exact-rounding unroll4 | 87 | 4 | 21.75 | 2 | 4 |
| exact-rounding unroll8 | 108 | 8 | 13.50 | 15 | 8 |
| exact-rounding unroll16 | 202 | 16 | 12.62 | 51 | 16 |
| exact-rounding unroll32 | 374 | 32 | 11.69 | 126 | 32 |
| exact-rounding group4 kernel | 52 | 4 | 13.00 | 0 | 4 |
| exact-rounding group8 kernel | 99 | 8 | 12.38 | 20 | 8 |
| exact-rounding group16 kernel | 188 | 16 | 11.75 | 55 | 16 |
| exact-rounding group4 call-chain aggregate | 539 | 32 | 16.84 | 4 | 32 |

The group4 call-chain aggregate is the sum of eight `dim0..dim28` 4-lane
kernels plus the wrapper. Its low `vst` is not enough: the wrapper has `jl=8`
for one 32-lane group and no hardware loop, so scaling to an 8-group lane body
would put 64 function calls in the hot chunk path.

The new group8/group16 probes close the obvious escape route. Reducing the
call count by making each exact callee wider immediately brings back the same
spill problem: group8 has 20 vector stores for 8 useful MACs, and group16 has
55 vector stores for 16 useful MACs. This makes the source-assembly direction a
consequence of measured codegen rather than a preference.

`33 vmac.f` is the exact per-group/per-output-lane count expected from the
MyLM analyzer: 32 activation lanes plus one offset/group-sum correction. MyLM's
static hot loop is `8 * 33 = 264 vmac.f`, and `lc=2` repeats it for the second
16-row output lane.

Chunk-level probe comparison:

| body | lines | `vmac.f` | lines / `vmac.f` | `vextbcst.16` | `vunpack` | `vups` | `vconv.bf16.fp32` | `vst` | control |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| MyLM static hot loop | 963 | 264 | 3.65 | 256 | 64 | 64 | 136 | 0 | `0x33c` |
| Peano noinline lane kernel | 1154 | 264 | 4.37 | 256 | 256 | 128 | 384 | 5 | `0x33c` |
| Peano exact-rounding lane kernel | 2698 | 256 | 10.54 | 256 | 256 | 128 | 640 | 1043 | `0x33c` |
| Peano two-lane static duplicate | 3732 | 528 | 7.07 | 256 | 512 | 256 | 768 | 824 | `0x33c` |
| Peano two-lane loop body | 2181 | 264 | 8.26 | 256 | 256 | 128 | 384 | 251 | `0x33c` |

The noinline lane kernel is the best current Peano shape: it reaches the same
264 static MAC count and nearly the same line/MAC density as MyLM, with only
five vector stores. The bad shapes are also informative:

- statically duplicating both output lanes creates massive spill;
- asking Peano to put the full body under a lane loop creates a hardware loop
  but also spills hundreds of vectors;
- MyLM's actual `lc=2` reuse is therefore not something Peano gets for free
  from a simple C++ loop.

The remaining performance gap inside the noinline lane kernel is mostly
unpack/conversion scheduling. MyLM needs `vunpack=64`, `vups=64`,
`vconv.bf16.fp32=136` for the 264-MAC static loop. The current Peano lane body
needs `vunpack=256`, `vups=128`, `vconv.bf16.fp32=384`, because the source
expresses Q4 unpack per 2-column pair instead of reusing wider unpacked vectors
the way MyLM's raw schedule does.

The assembly probes remove the codegen uncertainty from this specific gap:

| Body | `vmac.f` | `vextbcst.16` | `lda.s16` | `vunpack` | `vups` | `vst` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MyLM canonical middle group | 33 | 32 | 1 | 8 | 8 | 0 |
| Source assembly group-with-prep probe | 33 | 32 | 1 | 8 | 8 | 0 |
| Source assembly exact Q4NX unroll4 NPU smoke | 4 | 4 | 0 | 4 | 2 | 0 |
| Source assembly exact Q4NX group8 NPU smoke | 8 | 8 | 0 | 8 | 4 | 0 |
| Source assembly exact Q4NX group32 NPU smoke | 32 | 32 | 0 | 32 | 16 | 0 |
| Source assembly exact Q4NX lane8 loop NPU smoke | 256 | 256 | 0 | 256 | 128 | 0 |

The first two rows prove that the local assembler can preserve the MyLM-style
per-group instruction inventory, including no vector stores in the hot group.
The exact Q4NX smoke rows prove the source-assembly path is numerically
callable on real NPU and can carry accumulator state across stitched 4-dim
blocks for one complete 8-group lane. It is not yet a production kernel: the
smoke still uses one fixed local input buffer and a simplified single-output
lane ABI. The next hard problem is wiring this lane body into the production
main16 scheduler and weight/activation ABI while reducing the remaining
unpack/convert traffic toward the MyLM inventory.

Numerical reference probe:

`compare_q4nx_group_sum_reference.py` compares the current reference

```text
bf16((q * scale) + offset) * activation
```

against the MyLM-style algebraic form

```text
sum(bf16(q * scale) * activation) + offset * sum(activation_group)
```

on eight deterministic random Q4NX chunks. The group-sum form is not bit-exact
to the current reference because the current path rounds `(q * scale + offset)`
per input dimension before the MAC. In this probe the `group_sum_fp32` variant
stayed below `0.0031` max absolute error and the `group_sum_bf16` variant stayed
below `0.0056`; neither produced `> 1e-2` row errors. This is promising for a
production attempt, but it means the migration must be validated at the
attention-O and full-layer boundaries rather than treated as a local algebraic
no-op.

Production transplant attempt:

| candidate | main16 compute | integration result | conclusion |
| --- | --- | --- | --- |
| native dequant plus native `mac_elem_16_conf` | `12658.1 us`, faster than the baseline `~14310 us` | Q/K/V compact and K/V cache writeback passed exactly; `full-layer-attention-o-bf16` failed with `attention_o max_abs=0.0625 mean_abs=0.011252445 mismatches=161`; `qwen3-8b-decode-layer` failed with `final_hidden_out max_abs=0.25 mean_abs=0.045732379 mismatches=457` | faster, but not production-correct |
| existing AIE dequant plus native `mac_elem_16_conf` | `13773.9 us` | failed `full-layer-attention-o-bf16` with the same attention-output mismatch shape | the numerical break is caused by the native MAC substitution, not just by dequant rounding |
| source-assembly group-sum lane body | compute-only ran, but compute-only has no value check | `qwen3-8b-qkv-compact-output` failed with `payload_max_abs=0.112304688`; `full-layer-qkv-prefix` failed K/V writeback | source assembly path works, but this math contract is not a drop-in replacement |
| exact per-dim rounding plus signed native MAC full unroll | `qwen3-8b-qkv-compact-output=5851.7-6104.3 us` | QKV compact passed with max_abs `0.000000477`; full-layer QKV prefix passed at `112431.6-130712.2 us`; attention-O passed at `11052.8-11099.2 us`; full decode token0 passed at `23778.8 us` with final hidden max_abs `0.015625`, mismatches `0` | first production-safe migration into the `vextbcst.16 + vmac.f` instruction family; still not MyLM-grade because wrapper disasm has `vlda=282`, `vst=194`, `vconv.bf16.fp32=160` |

Relevant source facts:

- AIEVec has a vector-lane `broadcast` op and a scalar `broadcast_scalar` op in `/var/home/taowen/projects/mlir-aie/include/aie/Dialect/AIEVec/IR/AIEVecOps.td`.
- The AIEVec-to-C++ emitter prints vector broadcast as `broadcast_elem(...)` and scalar broadcast as `broadcast_to_v...(...)` in `/var/home/taowen/projects/mlir-aie/lib/Targets/AIEVecToCpp/TranslateAIEVecToCpp.cpp`.
- XLLVM exposes AIE2P bf16 MAC and BF16 UPS/SRS ops in `/var/home/taowen/projects/mlir-aie/include/aie/Dialect/XLLVM/IR/XLLVMAIE2IntrOps.td`.
- Peano's AIE2P headers expose supported unpack vector sizes in `.venv/lib/python3.12/site-packages/llvm-aie/lib/clang/21/include/aie2p/aie2p_ldst.h`.
- Peano's source header exposes `broadcast_elem(v32bfloat16, idx)` through `__builtin_aie2p_vextract_broadcast_bf32_bf512` in `/var/home/taowen/projects/llvm-aie/clang/lib/Headers/aie2p/aie2p_scl2vec.h`.
- The AIE2P backend selects `vextbcst.32` for that builtin in `/var/home/taowen/projects/llvm-aie/llvm/lib/Target/AIE/aie2p/AIE2PInstrPatterns.td`.
- Peano's source header implements `broadcast_elem(v32int16, idx)` as `broadcast_s16(ext_elem(...))`, not as a dedicated builtin. The AIE2P backend has a DAG combine for extract+broadcast on `v32i16` that selects `vextbcst.16`, but ordinary memory sources can be scalarized before that combine fires.
- The AIE2P instruction definitions and schedule include `vextbcst.*`, `vbcstshfl.*`, `vunpack`, and `vmac.f` in `/var/home/taowen/projects/llvm-aie/llvm/lib/Target/AIE/aie2p/`.

Current answer to `vextbcst.16` vs `vextbcst.32`:

- The 16-lane accumulator is not the selector. `probe_native_bf16_acc16_vextbcst_mac` still emits `vextbcst.32` because the broadcast source is `v32bfloat16`, which maps to `llvm.aie2p.vextract.broadcast32.bf512`.
- The source vector view is the selector. A `v32int16` source that reaches backend selection as a vector value emits `vextbcst.16`.
- If the `v32int16` source comes directly from memory and only one lane is needed, Peano chooses `lda.s16 + vbcst.16`. That is a reasonable scalarization, but it is not MyLM's vector-register `vextbcst.16` shape.
- MyLM's `vextbcst.16` therefore likely means the operand is already live in a 16-bit vector/register view, not that MyLM is simply using a 16-accumulator MAC.

Next production step:

This experiment resolves the first semantic gap: Peano C++ can express MyLM's
`vextbcst.16 + vmac.f` signed BF16 MAC shape, and the exact per-dim-rounding
version is now in the active qwen3-layer main16 path.

The MyLM analyzer resolves the next gap: static `lc=2`, `vmac.f=264` is exactly
one 32x256 Q4NX chunk when interpreted as two 16-row lane passes:

```text
main term:        2 output lanes * 8 groups * 32 dims = 512 vector MACs
zero/offset term: 2 output lanes * 8 groups           =  16 vector MACs
total:                                                   528 vector MACs
```

The analyzer now also checks activation-lane coverage mechanically:

```text
activation_groups=8
complete_lane_groups=8
expected_static_activation_extracts=256
matches_static_vextbcst=True
```

The pointer evidence is concrete: `p1` loads the same eight 32-lane activation
vectors and rewinds by `0x200`; `p3` loads eight activation group sums and
rewinds by `0x10`; `p0` keeps advancing through the two int4 payload halves.

The next production step is no longer "prove signed native MAC works" or
"prove exact source assembly can run on NPU"; both are done for a full 8-group
lane boundary. The next step is to integrate the exact source-assembly contract
into the active main16 scheduler: preserve `bf16(q * scale + offset)` per input
dimension for the production weight/activation ABI, while removing the current
wrapper's excess `vlda/vst/vconv` traffic and approaching the MyLM group
inventory of
`33 vmac.f + 32 vextbcst.16 + 8 vunpack + 8 vups + 0 vst`.
Therefore the remaining work is no longer "find an instruction"; it is to
write the complete scheduled lane body whose memory ABI, accumulator storeback,
and rounding semantics are production-equivalent. Activation should still stay
live as a vector-register source for `vextbcst.16`, but the failed group-sum
production attempt means we cannot hoist the offset term unless the reference
contract is deliberately changed and validated end to end.
