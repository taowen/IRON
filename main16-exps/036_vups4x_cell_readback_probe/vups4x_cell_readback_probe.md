# VUPS.4x Cell Readback Probe

Status: `passed`

- sentinel bf16: `0x3c80`
- probe bf16: `0x4000`

## Results

| case | status | unique record words | first record words |
| --- | --- | --- | --- |
| `x2d_upssign0_bmll0` | `record_observed` | `0x40, 0x0` | `0x40, 0x0, 0x40, 0x0` |
| `x2d_upssign0_bmlh0` | `record_observed` | `0x40, 0x0` | `0x40, 0x0, 0x40, 0x0` |
| `x2d_upssign0_bmhl0` | `record_observed` | `0x40, 0x0` | `0x40, 0x0, 0x40, 0x0` |
| `x2d_upssign0_bmhh0` | `record_observed` | `0x40, 0x0` | `0x40, 0x0, 0x40, 0x0` |
| `w2c_upssign0_bmll0` | `record_observed` | `0x40, 0x0` | `0x40, 0x0, 0x40, 0x0` |
| `w2c_upssign0_bmlh0` | `record_observed` | `0x40, 0x0` | `0x40, 0x0, 0x40, 0x0` |
| `w2c_upssign0_bmhl0` | `record_observed` | `0x3c80, 0x0` | `0x3c80, 0x3c80, 0x3c80, 0x3c80` |
| `w2c_upssign0_bmhh0` | `record_observed` | `0x3c80, 0x0` | `0x3c80, 0x3c80, 0x3c80, 0x3c80` |

## Interpretation

- This is a value-semantics probe, not a performance probe.
- The compact record header is intentionally not meaningful here; the selected vector cell is stored from word 0.
- `x2d` should reveal whether a full `dm` destination overwrites all four accumulator quadrants.
- `w2c` should reveal whether a `cml/cmh` destination preserves the other half of `dm0` initialized by the sentinel.
- If output words are neither sentinel-like nor probe-like, the next experiment must use lane-pattern inputs rather than uniform broadcasts.
