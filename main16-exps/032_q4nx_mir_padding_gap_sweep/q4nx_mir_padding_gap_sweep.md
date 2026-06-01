# Q4NX MIR Padding Gap Sweep

Status: `failed`

## Mutation

- original bytes: `00000000`
- replacement bytes: `f8a0df1f`
- replacement asm: `mov r31, r31`

## Results

| site | address | matches expected | observed first words |
| --- | --- | --- | --- |
| `gap_before_0x2c2` | `0x2ba` | `False` | `0x1, 0x48014801, 0x48014801, 0x48014801` |
| `gap_before_0x1828` | `0x1820` | `True` | `0x1, 0x3c803c80, 0x3c803c80, 0x3c803c80` |
| `gap_before_0x1844` | `0x183c` | `True` | `0x1, 0x3c803c80, 0x3c803c80, 0x3c803c80` |

## Interpretation

- The same self-move byte mutation is applied to each explicit hot-loop padding gap independently.
- Any payload mismatch means that gap is a timing contract, not spare code space.
- A passing site can be used as a byte-diff packaging canary, but not as evidence that real Q4NX bundles are safe to reorder.
