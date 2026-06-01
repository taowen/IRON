# Q4NX MIR Hotloop Patch Numeric Gate

Status: `failed`

## Patch

- source object: `main16-exps/026_q4nx_mir_full_hotloop_probe/build/mylm_full_hotloop_zol.o`
- patched raw: `main16-exps/028_q4nx_mir_hotloop_patch_numeric_gate/build/patched_raw/mylm_c2r2_program.mir_hotloop_patch.bin`
- hot range: `0x260..0x1850`
- MIR loop range: `0x10..0x15f0`
- patch bytes: `5600`
- nop fill bytes: `16`

## Cases

| case | status | matches expected | observed first words |
| --- | --- | --- | --- |
| `q4word0_allnibbles` | `record_observed` | `False` | `0x1, 0x4b014b01, 0x4b014b01, 0x4b014b01` |

## Interpretation

- This is the first direct numeric gate for the MIR-generated MyLM-style hot body.
- It preserves MyLM's prologue, phase body, group-sum producer, record emitter, DMA, and lock behavior.
- A pass means the generated MIR hot loop is numerically compatible for the selected synthetic cases.
- A fail should be debugged as a hot-loop semantic/register-schedule mismatch, not as a dataflow or phase-body issue.
