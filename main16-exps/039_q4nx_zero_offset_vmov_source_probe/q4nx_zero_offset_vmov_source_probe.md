# Q4NX Zero/Offset Vmov Source Probe

Status: `passed`

- site: `0x700`
- case: `zero0_allq4`

## Results

| replacement | asm | bytes | matches zero case | observed first words |
| --- | --- | --- | --- | --- |
| `nopm` | `nopm` | `f84a0318` | `False` | `0x1, 0x408f408f, 0x407f407f, 0x407f407f, 0x407f407f, 0x407f407f, 0x407f407f, 0x407f407f, 0x407f407f, 0x407f407f, 0x407f407f, 0x407f407f` |
| `self_bmhh1` | `vmov bmhh1, bmhh1` | `f812c719` | `False` | `0x1, 0x408f408f, 0x407f407f, 0x407f407f, 0x407f407f, 0x407f407f, 0x407f407f, 0x407f407f, 0x407f407f, 0x407f407f, 0x407f407f, 0x407f407f` |
| `from_bmll2` | `vmov bmhh1, bmll2` | `f812c819` | `True` | `0x1, 0x40904090, 0x40804080, 0x40804080, 0x40804080, 0x40804080, 0x40804080, 0x40804080, 0x40804080, 0x40804080, 0x40804080, 0x40804080` |
| `from_bmlh2` | `vmov bmhh1, bmlh2` | `f812c919` | `True` | `0x1, 0x40904090, 0x40804080, 0x40804080, 0x40804080, 0x40804080, 0x40804080, 0x40804080, 0x40804080, 0x40804080, 0x40804080, 0x40804080` |
| `from_bmhl2` | `vmov bmhh1, bmhl2` | `f812ca19` | `True` | `0x1, 0x40904090, 0x40804080, 0x40804080, 0x40804080, 0x40804080, 0x40804080, 0x40804080, 0x40804080, 0x40804080, 0x40804080, 0x40804080` |
| `original_bmhh2` | `vmov bmhh1, bmhh2` | `f812cb19` | `True` | `0x1, 0x40904090, 0x40804080, 0x40804080, 0x40804080, 0x40804080, 0x40804080, 0x40804080, 0x40804080, 0x40804080, 0x40804080, 0x40804080` |

## Interpretation

- `nopm` and `vmov bmhh1, bmhh1` fail the same way, so preserving cycle count or self-state is not enough.
- Every tested `acc2 -> bmhh1` source passes the zero/offset case, so the dependence is on refreshing `bmhh1` from the `acc2` correction value family.
- The source quadrant is not distinguished by this synthetic zero/offset case; the next gate must run a passing alternate source against all direct-QKV cases.
