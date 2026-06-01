# Q4NX Acc2 Quadrant Discriminator Gate

Status: `passed`

- site: `0x700`
- reference cases: `32`

## Results

| replacement | cases | all match reference | failed cases | first failed observed |
| --- | ---: | --- | --- | --- |
| `from_bmll2` | `32` | `True` | `none` | `` |
| `from_bmlh2` | `32` | `True` | `none` | `` |
| `from_bmhl2` | `32` | `True` | `none` | `` |

## Interpretation

- The reference is the unmodified MyLM raw program, not the scalar formula.
- A pass means the asymmetric zero cases still cannot distinguish the tested `acc2` source quadrants at this site.
- A fail identifies the first case that can distinguish the quadrants and should become part of the default source-mutation gate.
