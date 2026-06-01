# Q4NX Schedule DSL Resource Manifest

Status: `passed`

- bundle count: `963`
- observed bundle signatures: `50`

## Exact Replay

| candidate | status | matches original | text bytes | diffs |
| --- | --- | --- | ---: | --- |
| `exact_replay` | `encoded` | `True` | `5616` | `` |

## Candidate Gates

| candidate | expected | status | violations | diff count | first diffs |
| --- | --- | --- | --- | ---: | --- |
| `highhalf_swap_045` | `encoded` | `encoded` | `` | `2` | `0x6fc:8a->cb, 0x702:cb->8a` |
| `vups_advance_046` | `rejected_by_manifest` | `rejected_by_manifest` | `0x704 ('vmov_x', 'vadd')` | `None` | `` |
| `copy_and_vups_at_700_047` | `rejected_by_manifest` | `rejected_by_manifest` | `0x700 ('vmov_x', 'vups4x')` | `None` | `` |
| `vext_and_vups_at_6f2_047` | `rejected_by_manifest` | `rejected_by_manifest` | `0x6f2 ('vextbcst16', 'vups4x')` | `None` | `` |

## Interpretation

- The DSL round-trips the original MyLM hot loop through pre-bundled MIR and `llc` byte-exactly.
- The safe high-half schedule swap from experiment 045 is accepted because its opcode-kind bundle signatures match observed MyLM signatures.
- The `vups.4x` advance schedules from experiments 046/047 are rejected before encoding because they create unobserved bundle signatures.
- This is a conservative first resource manifest. It should be expanded with measured legal combinations, but it already prevents blind schedule guesses from reaching NPU gates.
