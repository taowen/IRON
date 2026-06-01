# Q4NX MIR Padding Mutation Numeric Gate

Status: `failed`

## Patch

- patched raw: `main16-exps/031_q4nx_mir_padding_mutation_numeric_gate/build/patched_raw/mylm_c2r2_program.padding_mov_self.bin`
- mutation address: `0x2ba`
- original bytes: `00000000`
- replacement bytes: `f8a0df1f`
- replacement asm: `mov r31, r31`
- diff count: `4`

## Byte Diff

| address | original | replacement |
| --- | --- | --- |
| `0x2ba` | `0x00` | `0xf8` |
| `0x2bb` | `0x00` | `0xa0` |
| `0x2bc` | `0x00` | `0xdf` |
| `0x2bd` | `0x00` | `0x1f` |

## Cases

| case | status | matches expected | observed first words |
| --- | --- | --- | --- |
| `q4word0_allnibbles` | `record_observed` | `False` | `0x1, 0x48014801, 0x48014801, 0x48014801` |

## Interpretation

- This is the first intentional hot-loop byte-diff after the byte-exact no-op gate.
- The mutation replaces two scalar NOP slots in the first explicit padding gap with a self-move.
- A pass means this specific padding gap has enough slack for a one-cycle shorter no-op mutation.
- A fail means padding bytes are part of the timing contract and future mutations must preserve cycle count exactly.
