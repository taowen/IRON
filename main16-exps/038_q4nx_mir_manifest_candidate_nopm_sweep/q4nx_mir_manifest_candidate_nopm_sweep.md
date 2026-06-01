# Q4NX MIR Manifest Candidate NOPM Sweep

Status: `failed`

## Results

| address | cases | all match | failed cases | observed first words |
| --- | ---: | --- | --- | --- |
| `0x700` | `4` | `False` | `zero0_allq4` | `0x1, 0x3c803c80, 0x3c803c80, 0x3c803c80` |
| `0x9b4` | `4` | `False` | `zero0_allq4` | `0x1, 0x3c803c80, 0x3c803c80, 0x3c803c80` |
| `0xc68` | `4` | `False` | `zero0_allq4` | `0x1, 0x3c803c80, 0x3c803c80, 0x3c803c80` |
| `0xf1c` | `4` | `False` | `zero0_allq4` | `0x1, 0x3c803c80, 0x3c803c80, 0x3c803c80` |
| `0x11d0` | `4` | `False` | `zero0_allq4` | `0x1, 0x3c803c80, 0x3c803c80, 0x3c803c80` |
| `0x1484` | `4` | `False` | `zero0_allq4` | `0x1, 0x3c803c80, 0x3c803c80, 0x3c803c80` |
| `0x173e` | `4` | `False` | `zero0_allq4` | `0x1, 0x3c803c80, 0x3c803c80, 0x3c803c80` |

## Interpretation

- Each row is a real source-bundle byte mutation, not a byte-exact replay.
- A passing row means the updated manifest found a source bundle that can be removed for the tested synthetic cases.
- A failing row means the static dead-def model is still missing timing, alias, or data-dependent semantics for that site.
- In the full sweep, every candidate failed only `zero0_allq4`, with the expected zero/offset payload one bf16 step higher than the observed payload.
- This is still a source-bundle mutation gate, not a complete self-generated Q4NX hot block.
