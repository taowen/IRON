# Experiment 99: AIE2P Q4 Direct-Target Stages

This experiment continues from experiment 98. Experiment 98 proved that a
normal C++ AIE core wrapper can call source assembly that loads, updates, and
stores an FP32 accumulator in tile-local memory.

This experiment adds production-shaped Q4NX instructions around that same
direct accumulator target:

- `exact-group1`: one 32-dim Q4NX exact-rounding group writes back to the
  FP32 accumulator.
- `exact-group2`: two loop-carried 32-dim groups write back to the FP32
  accumulator.
- `exact-group2-hwloop`: two groups using source-assembly `lc/ls/le`
  zero-overhead loop registers.
- `hwloop-smoke-lc2`: minimal source-assembly `lc/ls/le` smoke.
- `hwloop-smoke-lc2-aligned`: same smoke with 16-byte bundle-aligned `ls/le`
  and enough setup-to-loop-end distance.
- `exact-group2-hwloop-addnc*`: two-group variants that test Peano-style
  `add.nc lc, ...` setup, a setup gap, off-by-one LC, and explicit bundle
  alignment.
- `exact-group2-unrolled`: the same two groups without a source-level `jnz`.
- `exact-group8`: the full eight 32-dim groups for one production lane write
  back to the FP32 accumulator.
- `exact-group8-hwloop-addnc-aligned`: the full eight 32-dim groups using one
  bundle-aligned source-assembly zero-overhead loop.
- `exact-group4-unrolled`: four groups without a source-level `jnz`.
- `exact-group4x2-call`: calls the same four-group body twice with shifted
  pointers.
- `exact-group4x2-asm-call`: same structure, but the two calls are issued from
  source assembly instead of a Peano-generated C++ wrapper.
- `exact-group8-unrolled`: the same eight groups without a source-level `jnz`.

It does not change the active Qwen3 decode path. The point is to locate whether
the direct-target timeout comes from the base `vlda/vst` boundary, one exact
Q4NX group, or the full multi-group body.

Run:

```bash
.venv/bin/python experiments/99_aie2p_q4_direct_target_stages/run.py --stage hwloop-smoke-lc2
.venv/bin/python experiments/99_aie2p_q4_direct_target_stages/run.py --stage hwloop-smoke-lc2-aligned
.venv/bin/python experiments/99_aie2p_q4_direct_target_stages/run.py --stage exact-group1
.venv/bin/python experiments/99_aie2p_q4_direct_target_stages/run.py --stage exact-group2
.venv/bin/python experiments/99_aie2p_q4_direct_target_stages/run.py --stage exact-group2-hwloop
.venv/bin/python experiments/99_aie2p_q4_direct_target_stages/run.py --stage exact-group2-hwloop-addnc
.venv/bin/python experiments/99_aie2p_q4_direct_target_stages/run.py --stage exact-group2-hwloop-addnc-gap
.venv/bin/python experiments/99_aie2p_q4_direct_target_stages/run.py --stage exact-group2-hwloop-addnc-lc3
.venv/bin/python experiments/99_aie2p_q4_direct_target_stages/run.py --stage exact-group2-hwloop-addnc-aligned
.venv/bin/python experiments/99_aie2p_q4_direct_target_stages/run.py --stage exact-group2-unrolled
.venv/bin/python experiments/99_aie2p_q4_direct_target_stages/run.py --stage exact-group8
.venv/bin/python experiments/99_aie2p_q4_direct_target_stages/run.py --stage exact-group8-hwloop-addnc-aligned
.venv/bin/python experiments/99_aie2p_q4_direct_target_stages/run.py --stage exact-group4-unrolled
.venv/bin/python experiments/99_aie2p_q4_direct_target_stages/run.py --stage exact-group4x2-call
.venv/bin/python experiments/99_aie2p_q4_direct_target_stages/run.py --stage exact-group4x2-asm-call
.venv/bin/python experiments/99_aie2p_q4_direct_target_stages/run.py --stage exact-group8-unrolled
```

Passing means the stage completed on real NPU and produced the exact FP32
accumulator value for the all-one toy input. Timing is diagnostic only; this is
not an optimized kernel.

Observed results:

| Stage | Result | Meaning |
| --- | --- | --- |
| `hwloop-smoke-lc2` | FAIL, `dst=1` vs expected `2` | Plain source-assembly `lc/ls/le` setup can assemble and run while still not activating the zero-overhead loop. |
| `hwloop-smoke-lc2-aligned` | PASS, `~537 us`, `dst=2` | Source-assembly ZOL works when `ls/le` are 16-byte bundle aligned and the setup-to-LE distance satisfies the AIE2P constraint. |
| `exact-group1` | PASS, `~534 us`, `dst=480` | One direct-target exact group is safe. |
| `exact-group2` | NPU timeout | Source-level `jnz` loop around the exact group is not safe. |
| `exact-group2-hwloop` | FAIL, `dst=480` vs expected `960` | The source-assembly `lc/ls/le` encoding assembled and ran, but this range setup only executed one group. |
| `exact-group2-hwloop-addnc` | FAIL, `dst=480` vs expected `960` | Peano-style `add.nc lc, ...` alone does not fix the loop; LC write form was not the root cause. |
| `exact-group2-hwloop-addnc-gap` | FAIL, `dst=480` vs expected `960` | Adding a setup window without aligning `ls/le` still does not activate the loop. |
| `exact-group2-hwloop-addnc-lc3` | FAIL, `dst=480` vs expected `960` | This is not an off-by-one LC trip-count issue. |
| `exact-group2-hwloop-addnc-aligned` | PASS, `~581 us`, `dst=960` | Bundle alignment is the missing condition for source-assembly ZOL around the exact Q4 body. |
| `exact-group2-unrolled` | PASS, `~432 us`, `dst=960` | The second group, pointer advancement, and accumulator lifetime are safe when no `jnz` loop is used. |
| `exact-group8` | NPU timeout | Same failure class as `exact-group2`; this is not an eighth-group address-range issue. |
| `exact-group8-hwloop-addnc-aligned` | PASS, `~563 us`, `dst=3840`, core `.text=3008` | A full 256-dim lane can be represented as one exact 32-dim group body repeated by aligned source-assembly ZOL without CDO program-memory overflow. |
| `exact-group4-unrolled` | PASS, `~439 us`, `dst=1920` | Four groups fit and run without a source-level loop. |
| `exact-group4x2-call` | NPU timeout | C++ wrapper reuse of the four-group body is not safe enough. |
| `exact-group4x2-asm-call` | FAIL, `dst=5760` vs expected `3840` | Pure asm reuse returns, but current `jl`/delay-slot setup executes the wrong effective amount of work. |
| `exact-group8-unrolled` | CDO program-memory overflow | Full manual unroll is too large for a production core body. |

The final core ELF resolves the `jnz` target correctly, so the `jnz` timeout is
not a simple relocation-to-zero bug. The hardware-loop false starts had a
different root cause: source assembly must obey AIE2P ZOL placement rules, not
just write `lc/ls/le`. LLVM-AIE encodes this as a 16-byte bundle-aligned loop
start/end plus at least 112 bytes from loop setup to the loop-end label. MyLM's
`ls=0x260` and `le=0x1850` also follow this alignment. Once the source assembly
uses `.p2align 4` on `ls/le`, the exact Q4 body loops correctly.

The current production lesson is positive: one full 256-dim lane no longer
requires full manual unroll or unsafe function reuse. It can be expressed as a
single exact 32-dim group body under an aligned source-assembly zero-overhead
loop, with exact output and a small core program.

The validator checks exact accumulator values for the all-one toy input:
`480 * number_of_32_dim_groups`. This caught two important false positives:
the first `lc/ls/le` attempt ran but executed only one group, and the pure asm
4x2 call path ran but produced three four-group contributions instead of two.
