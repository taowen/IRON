# Q4NX Exact vs MyLM Contract Floor

## Chunk Shape

- Rows per chunk: `32`
- Rows per vector pass: `16`
- Lane passes: `2`
- Groups: `8`
- Dims per group: `32`
- Vector dims per chunk: `512`
- 4-dim packed blocks per chunk: `128`

## Contract Notes

### `active_exact`

- Current q4nx_chunk_accum_asm_zol dynamic counts from exp111.
- Numerically exact against the current reference, but not software-pipelined.

### `exact_lower_bound`

- Semantic lower bound for bf16(bf16(q*scale)+zero) before MAC.
- Assumes a perfect 4-dim unpack/vups schedule and no redundant control setup.
- Still needs two bf16 roundings and one bf16->fp32 conversion per vector dim.

### `mylm_group_contract`

- MyLM raw hot-loop dynamic counts from exp111.
- Uses 512 main MACs plus 16 zero/group-sum correction MACs.
- Not bit-equivalent to current exact reference per exp110.

## Dynamic Cost Model

| Op | Active Exact | Exact Lower Bound | MyLM-like Contract | Active -> Lower | Lower/MyLM |
| --- | ---: | ---: | ---: | ---: | ---: |
| `vmac.f` | 512 | 512 | 528 | 0.0% | 0.97x |
| `vextbcst.16` | 512 | 512 | 512 | 0.0% | 1.00x |
| `vunpack` | 512 | 128 | 128 | 75.0% | 1.00x |
| `vups.4x` | 0 | 128 | 128 | n/a | 1.00x |
| `vups.2x` | 256 | 0 | 0 | 100.0% | n/a |
| `vmul.f` | 512 | 512 | 16 | 0.0% | 32.00x |
| `vadd.f` | 512 | 512 | 0 | 0.0% | n/a |
| `vsub.f` | 256 | 0 | 128 | 100.0% | 0.00x |
| `vconv.bf16.fp32` | 1280 | 1024 | 272 | 20.0% | 3.76x |
| `vconv.fp32.bf16` | 768 | 512 | 0 | 33.3% | n/a |
| `crupsmode` | 128 | 0 | 0 | 100.0% | n/a |
| `crunpacksize` | 512 | 0 | 0 | 100.0% | n/a |
| `nop` | 3088 | 0 | 216 | 100.0% | 0.00x |

## Decision

- Exact parity can still improve the active body by removing redundant unpack/control/conversion traffic.
- The exact lower bound still needs `512` vector multiplies and `1536` conversion operations per chunk.
- MyLM-like group correction needs only `16` vector multiplies and `272` conversion operations per chunk.
- Therefore exact-parity assembly can narrow the gap, but it cannot reach the MyLM raw hot-loop shape unless there is an unknown fused instruction path for the two bf16 roundings.
- The performance route to MyLM-like speed should now branch explicitly: optimize exact as a safe baseline, and separately run multi-layer token gates for the MyLM-like numerical contract.

## Immediate Target

- Safe exact target: reduce active `vunpack` `512 -> 128`, `vconv.bf16.fp32` `1280 -> 1024`, and `vconv.fp32.bf16` `768 -> 512`.
- Risk-bearing fast target: accept the MyLM-like zero/group-sum correction contract, then prove token quality across many layers before migrating production.
