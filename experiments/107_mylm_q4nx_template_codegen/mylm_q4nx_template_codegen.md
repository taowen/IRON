# MyLM Q4NX Template Codegen

This report proves that the MyLM hot loop can be represented as four
explicit schedule templates rather than eight hand-copied groups.

## Replay Check

- Original slots: `1532`
- Generated slots: `1532`
- Original hash: `9093b2372c1324ace60247f52e7edbec7ed77f880bc75c86a705f1f4c515e66b`
- Generated hash: `9093b2372c1324ace60247f52e7edbec7ed77f880bc75c86a705f1f4c515e66b`
- Exact text match: `True`

## Opcode Counts

| Op | Count |
| --- | ---: |
| `vmac.f` | 264 |
| `vextbcst.16` | 256 |
| `vunpack` | 64 |
| `vups.4x` | 64 |
| `vconv.bf16.fp32` | 136 |
| `vst` | 0 |
| `vlda` | 11 |
| `vldb` | 46 |
| `lda.s16` | 8 |
| `vbcst.16` | 8 |

## Template Boundaries

| Generated Range | Template | Slots |
| --- | --- | ---: |
| `0:192` | `fill` | 192 |
| `192:381` | `steady0` | 189 |
| `381:570` | `steady1` | 189 |
| `570:759` | `steady2` | 189 |
| `759:948` | `steady3` | 189 |
| `948:1137` | `steady4` | 189 |
| `1137:1327` | `pre_drain` | 190 |
| `1327:1532` | `drain` | 205 |

## Production Boundary

The next production step is not to paste this MyLM body into IRON. The
useful boundary is to port this generator shape to the IRON exact-Q4NX
contract, then make each template pass the existing qwen3-layer numeric
gates. MyLM's group-sum formula remains a separate numerical-contract
decision.
