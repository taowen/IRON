# Q4NX Acc2 Alternate Source Numeric Gate

Status: `passed`

- site: `0x700`

## Results

| replacement | asm | cases | all match | failed cases | observed first words |
| --- | --- | ---: | --- | --- | --- |
| `from_bmll2` | `vmov bmhh1, bmll2` | `4` | `True` | `none` | `0x1, 0x40904090, 0x40804080, 0x40804080, 0x40804080, 0x40804080` |
| `from_bmlh2` | `vmov bmhh1, bmlh2` | `4` | `True` | `none` | `0x1, 0x40904090, 0x40804080, 0x40804080, 0x40804080, 0x40804080` |
| `from_bmhl2` | `vmov bmhh1, bmhl2` | `4` | `True` | `none` | `0x1, 0x40904090, 0x40804080, 0x40804080, 0x40804080, 0x40804080` |
| `original_bmhh2` | `vmov bmhh1, bmhh2` | `4` | `True` | `none` | `0x1, 0x40904090, 0x40804080, 0x40804080, 0x40804080, 0x40804080` |

## Interpretation

- A passing alternate source is a real non-byte-copy source-bundle mutation under the current direct-QKV synthetic gate.
- This does not make the mutation a performance optimization; it only proves we can change a source bundle without breaking current numeric coverage.
- If all `acc2` quadrants pass, the synthetic cases do not distinguish the quadrant value at this template point.
