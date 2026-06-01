# Q4NX Chunk Contribution Map

- Status: `passed`
- Setup: activation bf16 `1.0`, scale bf16 `1/64`, q4 nibble `1`

## Per-Chunk Contribution

| chunk | status | first record word | first record value | nonzero records | npu_time |
| ---: | --- | --- | ---: | --- | --- |
| `0` | `record_observed` | `0x40804080` | `4.0` | `[0]` | `1006185` |
| `1` | `record_observed` | `0x0` | `0.0` | `[]` | `930373` |
| `2` | `record_observed` | `0x40804080` | `4.0` | `[0]` | `1036452` |
| `3` | `record_observed` | `0x0` | `0.0` | `[]` | `1000474` |
| `4` | `record_observed` | `0x40804080` | `4.0` | `[0]` | `1023868` |
| `5` | `record_observed` | `0x0` | `0.0` | `[]` | `987310` |
| `6` | `record_observed` | `0x40804080` | `4.0` | `[0]` | `993301` |
| `7` | `record_observed` | `0x0` | `0.0` | `[]` | `1006856` |
| `8` | `record_observed` | `0x40804080` | `4.0` | `[0]` | `986309` |
| `9` | `record_observed` | `0x0` | `0.0` | `[]` | `1027345` |
| `10` | `record_observed` | `0x40804080` | `4.0` | `[0]` | `943529` |
| `11` | `record_observed` | `0x0` | `0.0` | `[]` | `1324589` |
| `12` | `record_observed` | `0x40804080` | `4.0` | `[0]` | `945342` |
| `13` | `record_observed` | `0x0` | `0.0` | `[]` | `954729` |
| `14` | `record_observed` | `0x40804080` | `4.0` | `[0]` | `1044627` |
| `15` | `record_observed` | `0x0` | `0.0` | `[]` | `942226` |

## Aggregate

- Sum of single-chunk values: `32.0`
- All record0 chunks value: `32.0`
- Sum matches all-record0: `True`

## Interpretation

This map explains why experiment 008's one-active-chunk formula failed. The scalar full-record formula is not made of 16 equal 128-element chunk contributions. Use this map as the next clue for decoding the Q4NX microkernel's intra-record chunk/lane schedule.
