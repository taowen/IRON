# Q4NX MIR Byte-Exact No-Op Numeric Gate

Status: `passed`

## Patch

- source object: `main16-exps/029_q4nx_mir_prebundled_exact_replay/build/mylm_hotloop_prebundled_padded.o`
- patched raw: `main16-exps/030_q4nx_mir_byte_exact_noop_numeric_gate/build/patched_raw/mylm_c2r2_program.byte_exact_mir_noop.bin`
- hot range: `0x260..0x1850`
- patch bytes: `5616`
- hot bytes match original: `True`
- full raw matches original: `True`
- first diff: `None`

## Cases

| case | status | matches expected | observed first words |
| --- | --- | --- | --- |
| `q4word0_allnibbles` | `record_observed` | `True` | `0x1, 0x3c803c80, 0x3c803c80, 0x3c803c80` |
| `q4word512_allnibbles` | `record_observed` | `True` | `0x1, 0x0, 0x0, 0x0` |
| `q4word0_nibble3` | `record_observed` | `True` | `0x1, 0x0, 0x3c800000, 0x0` |
| `zero0_allq4` | `record_observed` | `True` | `0x1, 0x40904090, 0x40804080, 0x40804080` |

## Interpretation

- This is a no-op replacement: the MIR object from experiment 029 is byte-exact against MyLM's hot loop.
- A pass means the MIR object extraction, raw patching, packaging, and direct-QKV runtime path preserve MyLM numerics.
- This is the baseline for future one-bundle-at-a-time MIR mutations.
