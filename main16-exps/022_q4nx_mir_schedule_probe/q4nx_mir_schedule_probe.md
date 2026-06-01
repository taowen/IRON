# Q4NX MIR Schedule Probe

Status: `partial`

This experiment bypasses C++ and emits AIE2P machine MIR directly.
It tests whether Peano can assemble the Q4NX-critical instruction trio:
`vups.4x`, `vextbcst.16`, and `vmac.f`.

## Results

### `basic_machine_block`

Status: `pass`

Prove direct AIE2P machine MIR can assemble the three key Q4NX instructions.

- `llc_returncode`: `0`
- `objdump_returncode`: `0`
- `vups_4x`: `1`
- `vups_2x`: `0`
- `vextbcst_16`: `1`
- `vextbcst_32`: `0`
- `vmac_f`: `1`
- `vst`: `0`
- `vlda`: `0`
- `schedule_found`: `0`
- `schedule_missed`: `0`
- `bundle_count`: `0`
- `ret`: `1`

### `loop_postpipeline_probe`

Status: `pass`

Probe whether the same direct machine-instruction shape can enter the postpipeliner/ZOL route.

- `llc_returncode`: `0`
- `objdump_returncode`: `0`
- `vups_4x`: `1`
- `vups_2x`: `0`
- `vextbcst_16`: `1`
- `vextbcst_32`: `0`
- `vmac_f`: `1`
- `vst`: `0`
- `vlda`: `0`
- `schedule_found`: `1`
- `schedule_missed`: `0`
- `bundle_count`: `3`
- `ret`: `1`

### `one_group_shape_probe`

Status: `partial`

Probe the MyLM-style one-group macro shape: 8 vups.4x, 32 vextbcst.16, and 33 vmac.f.

- `llc_returncode`: `0`
- `objdump_returncode`: `0`
- `vups_4x`: `8`
- `vups_2x`: `0`
- `vextbcst_16`: `32`
- `vextbcst_32`: `0`
- `vmac_f`: `33`
- `vst`: `0`
- `vlda`: `0`
- `schedule_found`: `0`
- `schedule_missed`: `2`
- `bundle_count`: `3`
- `ret`: `1`

Reasons:
- postpipeliner did not report Schedule found

## Conclusion

- Direct machine MIR is a viable route if the basic block passes: it avoids C++ lowering and gives exact instruction selection.
- The next meaningful gate is not C++ tuning; it is MIR/codegen scheduling with explicit register lifetimes.
- If the loop/postpipeline probe remains partial, we should still use MIR for object generation and use TD-derived latency checks for a source-asm/codegen scheduler.

## Next Step

Extend this from a three-instruction probe to one MyLM-style Q4NX activation group:

- generate 8 `vups.4x`, 32 `vextbcst.16`, and 33 `vmac.f` shape for one group;
- keep registers explicit enough to prevent C++-style spills;
- compare objdump counts and then run an isolated NPU numeric gate against MyLM direct `0x1870`.
