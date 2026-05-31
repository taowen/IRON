# Q4NX Exact Section Rewrite Contract

This experiment turns the exp121 exact-body decision into a section-level
rewrite contract. It keeps exp120's live-state boundaries and removes one
MyLM zero-correction MAC from each logical group.

## Checks

- Sections: `5`
- Source static MACs: `264`
- Exact static MACs: `256`
- Removed zero-correction MACs: `8`
- Exact dynamic MACs: `512`
- Exact synthetic gate passes: `True`
- MyLM stress gate passes: `False`

## Numeric Gate Inputs

- Exact max abs: `0.000000000`
- Exact `>1e-2`: `0`
- MyLM-like stress max abs: `0.015625000`
- MyLM-like stress `>1e-2`: `16`

## Section Rewrite

| Section | Groups | Source MACs/group | Exact MACs/group | Live In | Live Out |
| --- | --- | ---: | ---: | --- | --- |
| `fill` | `0` | 28 | 27 | `entry` | `boundary1` |
| `fill_to_steady` | `1` | 33 | 32 | `boundary1` | `boundary2` |
| `steady_to_steady` | `2,3,4,5` | 33 | 32 | `boundary2` | `boundary6` |
| `pre_drain` | `6` | 33 | 32 | `boundary6` | `boundary7` |
| `drain` | `7` | 38 | 37 | `boundary7` | `exit` |

## Generated Include

- `/var/home/taowen/projects/IRON/experiments/122_q4nx_exact_section_rewrite_contract/generated_q4nx_exact_section_rewrite.s.inc`

## Production Implication

- The next source-assembly generator should use this rewrite contract, not the MyLM 33-MAC/group contract.
- It must preserve exp120 section live state while emitting exact coefficient recomposition before each MAC.
- MyLM-like group correction remains a separate quality/performance branch gated by real token tests.
