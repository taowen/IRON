# Source ASM Branch Semantics Probe

- Status: `passed`

## Payload

| test | observed | meaning |
| --- | --- | --- |
| `eq_equal` | `0x1` | `value_1` |
| `eq_notequal` | `0x0` | `value_0` |
| `jz_zero` | `0x11110001` | `branch_taken` |
| `jz_one` | `0x22220002` | `fallthrough` |
| `jnz_zero` | `0x22220002` | `fallthrough` |
| `jnz_one` | `0x22220002` | `fallthrough` |
| `eq_equal_jz` | `0x22220002` | `fallthrough` |
| `eq_notequal_jz` | `0x11110001` | `branch_taken` |
| `eq_equal_jnz` | `0x22220002` | `fallthrough` |
| `eq_notequal_jnz` | `0x22220002` | `fallthrough` |

## Interpretation

`eq` and conditional branch semantics must be taken from this table when generating scalar control flow. The experiment exists because the mnemonic spelling alone was not a safe guide for `eq` + `jz/jnz`.
