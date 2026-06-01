# Q4NX Schedule DSL Window Interface Gate

Status: `passed`

| candidate | interface ok | resource ok | window | diffs |
| --- | --- | --- | --- | --- |
| `highhalf_swap_045` | `True` | `True` | `0x6f6..0x700` | `` |
| `vups_advance_046` | `True` | `False` | `0x700..0x704` | `` |
| `drop_bmhh_copy_negative_control` | `False` | `True` | `0x700..0x700` | `live_ins: $bmhh2 -> <br>boundary_defs: $bmhh1 -> ` |

## Interpretation

- The high-half swap keeps the same local window interface and passes the resource manifest.
- The `vups.4x` advance keeps the same textual live-in/def interface, but is still rejected by the resource manifest.
- The negative control drops the high-half copy; the resource manifest allows a single `NOP`, but the window interface rejects the changed live-ins and boundary defs.
- This confirms that the DSL needs both checks: resource legality and window interface preservation catch different classes of bad schedules.
