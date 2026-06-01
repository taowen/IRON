# Q4NX Layout Boundary, Nibble, and Zero Probe

- Status: `passed`
- Active pair: activation chunk `0`, weight chunk `0`

## Results

| case | group | status | nonzero lanes | lane values prefix | npu_time |
| --- | --- | --- | --- | --- | --- |
| `boundary_q4word_0384` | `boundary` | `record_observed` | `[0, 1, 2, 3]` | `[0.015625, 0.015625, 0.015625, 0.015625, 0, 0, 0, 0...]` | `976267` |
| `boundary_q4word_0385` | `boundary` | `record_observed` | `[4, 5, 6, 7]` | `[0, 0, 0, 0, 0.015625, 0.015625, 0.015625, 0.015625...]` | `956390` |
| `boundary_q4word_0448` | `boundary` | `record_observed` | `[0, 1, 2, 3]` | `[0.015625, 0.015625, 0.015625, 0.015625, 0, 0, 0, 0...]` | `999601` |
| `boundary_q4word_0449` | `boundary` | `record_observed` | `[4, 5, 6, 7]` | `[0, 0, 0, 0, 0.015625, 0.015625, 0.015625, 0.015625...]` | `1004510` |
| `boundary_q4word_0480` | `boundary` | `record_observed` | `[0, 1, 2, 3]` | `[0.015625, 0.015625, 0.015625, 0.015625, 0, 0, 0, 0...]` | `954898` |
| `boundary_q4word_0481` | `boundary` | `record_observed` | `[4, 5, 6, 7]` | `[0, 0, 0, 0, 0.015625, 0.015625, 0.015625, 0.015625...]` | `971379` |
| `boundary_q4word_0496` | `boundary` | `record_observed` | `[0, 1, 2, 3]` | `[0.015625, 0.015625, 0.015625, 0.015625, 0, 0, 0, 0...]` | `935011` |
| `boundary_q4word_0497` | `boundary` | `record_observed` | `[4, 5, 6, 7]` | `[0, 0, 0, 0, 0.015625, 0.015625, 0.015625, 0.015625...]` | `976397` |
| `boundary_q4word_0508` | `boundary` | `record_observed` | `[0, 1, 2, 3]` | `[0.015625, 0.015625, 0.015625, 0.015625, 0, 0, 0, 0...]` | `961841` |
| `boundary_q4word_0509` | `boundary` | `record_observed` | `[4, 5, 6, 7]` | `[0, 0, 0, 0, 0.015625, 0.015625, 0.015625, 0.015625...]` | `953225` |
| `boundary_q4word_0510` | `boundary` | `record_observed` | `[0, 1, 2, 3]` | `[0.015625, 0.015625, 0.015625, 0.015625, 0, 0, 0, 0...]` | `1007316` |
| `boundary_q4word_0511` | `boundary` | `record_observed` | `[4, 5, 6, 7]` | `[0, 0, 0, 0, 0.015625, 0.015625, 0.015625, 0.015625...]` | `996234` |
| `boundary_q4word_0512` | `boundary` | `record_observed` | `[8, 9, 10, 11]` | `[0, 0, 0, 0, 0, 0, 0, 0...]` | `952173` |
| `boundary_q4word_0513` | `boundary` | `record_observed` | `[12, 13, 14, 15]` | `[0, 0, 0, 0, 0, 0, 0, 0...]` | `983742` |
| `boundary_q4word_0514` | `boundary` | `record_observed` | `[8, 9, 10, 11]` | `[0, 0, 0, 0, 0, 0, 0, 0...]` | `957322` |
| `boundary_q4word_0515` | `boundary` | `record_observed` | `[12, 13, 14, 15]` | `[0, 0, 0, 0, 0, 0, 0, 0...]` | `951141` |
| `nibble_q4word_0000_n0` | `nibble` | `record_observed` | `[]` | `[0, 0, 0, 0, 0, 0, 0, 0...]` | `1016142` |
| `nibble_q4word_0000_n1` | `nibble` | `record_observed` | `[0]` | `[0.015625, 0, 0, 0, 0, 0, 0, 0...]` | `998579` |
| `nibble_q4word_0000_n2` | `nibble` | `record_observed` | `[]` | `[0, 0, 0, 0, 0, 0, 0, 0...]` | `947253` |
| `nibble_q4word_0000_n3` | `nibble` | `record_observed` | `[1]` | `[0, 0.015625, 0, 0, 0, 0, 0, 0...]` | `948856` |
| `nibble_q4word_0000_n4` | `nibble` | `record_observed` | `[]` | `[0, 0, 0, 0, 0, 0, 0, 0...]` | `980616` |
| `nibble_q4word_0000_n5` | `nibble` | `record_observed` | `[2]` | `[0, 0, 0.015625, 0, 0, 0, 0, 0...]` | `1039836` |
| `nibble_q4word_0000_n6` | `nibble` | `record_observed` | `[]` | `[0, 0, 0, 0, 0, 0, 0, 0...]` | `956731` |
| `nibble_q4word_0000_n7` | `nibble` | `record_observed` | `[3]` | `[0, 0, 0, 0.015625, 0, 0, 0, 0...]` | `938107` |
| `nibble_q4word_0001_n0` | `nibble` | `record_observed` | `[]` | `[0, 0, 0, 0, 0, 0, 0, 0...]` | `1430684` |
| `nibble_q4word_0001_n1` | `nibble` | `record_observed` | `[4]` | `[0, 0, 0, 0, 0.015625, 0, 0, 0...]` | `930963` |
| `nibble_q4word_0001_n2` | `nibble` | `record_observed` | `[]` | `[0, 0, 0, 0, 0, 0, 0, 0...]` | `957273` |
| `nibble_q4word_0001_n3` | `nibble` | `record_observed` | `[5]` | `[0, 0, 0, 0, 0, 0.015625, 0, 0...]` | `971439` |
| `nibble_q4word_0001_n4` | `nibble` | `record_observed` | `[]` | `[0, 0, 0, 0, 0, 0, 0, 0...]` | `960438` |
| `nibble_q4word_0001_n5` | `nibble` | `record_observed` | `[6]` | `[0, 0, 0, 0, 0, 0, 0.015625, 0...]` | `947795` |
| `nibble_q4word_0001_n6` | `nibble` | `record_observed` | `[]` | `[0, 0, 0, 0, 0, 0, 0, 0...]` | `971499` |
| `nibble_q4word_0001_n7` | `nibble` | `record_observed` | `[7]` | `[0, 0, 0, 0, 0, 0, 0, 0.015625...]` | `948706` |
| `nibble_q4word_0512_n0` | `nibble` | `record_observed` | `[]` | `[0, 0, 0, 0, 0, 0, 0, 0...]` | `971549` |
| `nibble_q4word_0512_n1` | `nibble` | `record_observed` | `[8]` | `[0, 0, 0, 0, 0, 0, 0, 0...]` | `936243` |
| `nibble_q4word_0512_n2` | `nibble` | `record_observed` | `[]` | `[0, 0, 0, 0, 0, 0, 0, 0...]` | `947164` |
| `nibble_q4word_0512_n3` | `nibble` | `record_observed` | `[9]` | `[0, 0, 0, 0, 0, 0, 0, 0...]` | `993340` |
| `nibble_q4word_0512_n4` | `nibble` | `record_observed` | `[]` | `[0, 0, 0, 0, 0, 0, 0, 0...]` | `941223` |
| `nibble_q4word_0512_n5` | `nibble` | `record_observed` | `[10]` | `[0, 0, 0, 0, 0, 0, 0, 0...]` | `1010431` |
| `nibble_q4word_0512_n6` | `nibble` | `record_observed` | `[]` | `[0, 0, 0, 0, 0, 0, 0, 0...]` | `985815` |
| `nibble_q4word_0512_n7` | `nibble` | `record_observed` | `[11]` | `[0, 0, 0, 0, 0, 0, 0, 0...]` | `955870` |
| `nibble_q4word_0513_n0` | `nibble` | `record_observed` | `[]` | `[0, 0, 0, 0, 0, 0, 0, 0...]` | `985726` |
| `nibble_q4word_0513_n1` | `nibble` | `record_observed` | `[12]` | `[0, 0, 0, 0, 0, 0, 0, 0...]` | `943838` |
| `nibble_q4word_0513_n2` | `nibble` | `record_observed` | `[]` | `[0, 0, 0, 0, 0, 0, 0, 0...]` | `949749` |
| `nibble_q4word_0513_n3` | `nibble` | `record_observed` | `[13]` | `[0, 0, 0, 0, 0, 0, 0, 0...]` | `985515` |
| `nibble_q4word_0513_n4` | `nibble` | `record_observed` | `[]` | `[0, 0, 0, 0, 0, 0, 0, 0...]` | `995845` |
| `nibble_q4word_0513_n5` | `nibble` | `record_observed` | `[14]` | `[0, 0, 0, 0, 0, 0, 0, 0...]` | `1003879` |
| `nibble_q4word_0513_n6` | `nibble` | `record_observed` | `[]` | `[0, 0, 0, 0, 0, 0, 0, 0...]` | `992618` |
| `nibble_q4word_0513_n7` | `nibble` | `record_observed` | `[15]` | `[0, 0, 0, 0, 0, 0, 0, 0...]` | `991606` |
| `zero_000` | `zero` | `record_observed` | `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]` | `[4.5, 4, 4, 4, 4, 4, 4, 4...]` | `1042712` |
| `zero_001` | `zero` | `record_observed` | `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]` | `[4, 4.5, 4, 4, 4, 4, 4, 4...]` | `1090120` |
| `zero_002` | `zero` | `record_observed` | `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]` | `[4, 4, 4.5, 4, 4, 4, 4, 4...]` | `961490` |
| `zero_003` | `zero` | `record_observed` | `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]` | `[4, 4, 4, 4.5, 4, 4, 4, 4...]` | `960158` |
| `zero_004` | `zero` | `record_observed` | `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]` | `[4, 4, 4, 4, 4.5, 4, 4, 4...]` | `964806` |
| `zero_005` | `zero` | `record_observed` | `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]` | `[4, 4, 4, 4, 4, 4.5, 4, 4...]` | `959466` |
| `zero_006` | `zero` | `record_observed` | `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]` | `[4, 4, 4, 4, 4, 4, 4.5, 4...]` | `958605` |
| `zero_007` | `zero` | `record_observed` | `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]` | `[4, 4, 4, 4, 4, 4, 4, 4.5...]` | `975155` |
| `zero_008` | `zero` | `record_observed` | `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]` | `[4, 4, 4, 4, 4, 4, 4, 4...]` | `1007926` |
| `zero_016` | `zero` | `record_observed` | `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]` | `[4.5, 4, 4, 4, 4, 4, 4, 4...]` | `1015481` |
| `zero_032` | `zero` | `record_observed` | `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]` | `[4.5, 4, 4, 4, 4, 4, 4, 4...]` | `956111` |
| `zero_064` | `zero` | `record_observed` | `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]` | `[4.5, 4, 4, 4, 4, 4, 4, 4...]` | `993790` |
| `zero_096` | `zero` | `record_observed` | `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]` | `[4.5, 4, 4, 4, 4, 4, 4, 4...]` | `969425` |
| `zero_127` | `zero` | `record_observed` | `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]` | `[4, 4, 4, 4, 4, 4, 4, 4...]` | `999391` |

## Interpretation

Boundary cases locate the q4 low/high output-half split. Nibble cases show which payload lanes each 4-bit field affects. Zero cases test whether the middle 128 dwords participate in this synthetic MyLM Q4NX contract.
