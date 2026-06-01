# Source ASM Latency Model

- Status: `passed`

## Minimum Safe Gaps

| dependency | min nops |
| --- | --- |
| `lda_to_st` | `6` |
| `lda_to_eq_jz` | `6` |
| `eq_to_jz` | `0` |
| `and_to_eq_value` | `0` |
| `lshl_to_or` | `0` |

## Sweep

| dependency | gap | observed | expected | pass | runtime |
| --- | --- | --- | --- | --- | --- |
| `lda_to_st` | `0` | `0x0` | `0x3c803c80` | `False` | `record_observed` |
| `lda_to_eq_jz` | `0` | `0xa55a5a5a` | `0x5a5aa55a` | `False` | `record_observed` |
| `lda_to_st` | `1` | `0x0` | `0x3c803c80` | `False` | `record_observed` |
| `lda_to_eq_jz` | `1` | `0xa55a5a5a` | `0x5a5aa55a` | `False` | `record_observed` |
| `lda_to_st` | `2` | `0x0` | `0x3c803c80` | `False` | `record_observed` |
| `lda_to_eq_jz` | `2` | `0xa55a5a5a` | `0x5a5aa55a` | `False` | `record_observed` |
| `lda_to_st` | `3` | `0x0` | `0x3c803c80` | `False` | `record_observed` |
| `lda_to_eq_jz` | `3` | `0xa55a5a5a` | `0x5a5aa55a` | `False` | `record_observed` |
| `lda_to_st` | `4` | `0x0` | `0x3c803c80` | `False` | `record_observed` |
| `lda_to_eq_jz` | `4` | `0xa55a5a5a` | `0x5a5aa55a` | `False` | `record_observed` |
| `lda_to_st` | `5` | `0x0` | `0x3c803c80` | `False` | `record_observed` |
| `lda_to_eq_jz` | `5` | `0xa55a5a5a` | `0x5a5aa55a` | `False` | `record_observed` |
| `lda_to_st` | `6` | `0x3c803c80` | `0x3c803c80` | `True` | `record_observed` |
| `lda_to_eq_jz` | `6` | `0x5a5aa55a` | `0x5a5aa55a` | `True` | `record_observed` |
| `lda_to_st` | `7` | `0x3c803c80` | `0x3c803c80` | `True` | `record_observed` |
| `lda_to_eq_jz` | `7` | `0x5a5aa55a` | `0x5a5aa55a` | `True` | `record_observed` |
| `lda_to_st` | `8` | `0x3c803c80` | `0x3c803c80` | `True` | `record_observed` |
| `lda_to_eq_jz` | `8` | `0x5a5aa55a` | `0x5a5aa55a` | `True` | `record_observed` |
| `eq_to_jz` | `0` | `0x5a5aa55a` | `0x5a5aa55a` | `True` | `record_observed` |
| `and_to_eq_value` | `0` | `0x0` | `0x0` | `True` | `record_observed` |
| `lshl_to_or` | `0` | `0x3c800000` | `0x3c800000` | `True` | `record_observed` |
| `eq_to_jz` | `1` | `0x5a5aa55a` | `0x5a5aa55a` | `True` | `record_observed` |
| `and_to_eq_value` | `1` | `0x0` | `0x0` | `True` | `record_observed` |
| `lshl_to_or` | `1` | `0x3c800000` | `0x3c800000` | `True` | `record_observed` |
| `eq_to_jz` | `2` | `0x5a5aa55a` | `0x5a5aa55a` | `True` | `record_observed` |
| `and_to_eq_value` | `2` | `0x0` | `0x0` | `True` | `record_observed` |
| `lshl_to_or` | `2` | `0x3c800000` | `0x3c800000` | `True` | `record_observed` |
| `eq_to_jz` | `3` | `0x5a5aa55a` | `0x5a5aa55a` | `True` | `record_observed` |
| `and_to_eq_value` | `3` | `0x0` | `0x0` | `True` | `record_observed` |
| `lshl_to_or` | `3` | `0x3c800000` | `0x3c800000` | `True` | `record_observed` |
| `eq_to_jz` | `4` | `0x5a5aa55a` | `0x5a5aa55a` | `True` | `record_observed` |
| `and_to_eq_value` | `4` | `0x0` | `0x0` | `True` | `record_observed` |
| `lshl_to_or` | `4` | `0x3c800000` | `0x3c800000` | `True` | `record_observed` |
| `eq_to_jz` | `5` | `0x5a5aa55a` | `0x5a5aa55a` | `True` | `record_observed` |
| `and_to_eq_value` | `5` | `0x0` | `0x0` | `True` | `record_observed` |
| `lshl_to_or` | `5` | `0x3c800000` | `0x3c800000` | `True` | `record_observed` |
| `eq_to_jz` | `6` | `0x5a5aa55a` | `0x5a5aa55a` | `True` | `record_observed` |
| `and_to_eq_value` | `6` | `0x0` | `0x0` | `True` | `record_observed` |
| `lshl_to_or` | `6` | `0x3c800000` | `0x3c800000` | `True` | `record_observed` |
| `eq_to_jz` | `7` | `0x5a5aa55a` | `0x5a5aa55a` | `True` | `record_observed` |
| `and_to_eq_value` | `7` | `0x0` | `0x0` | `True` | `record_observed` |
| `lshl_to_or` | `7` | `0x3c800000` | `0x3c800000` | `True` | `record_observed` |
| `eq_to_jz` | `8` | `0x5a5aa55a` | `0x5a5aa55a` | `True` | `record_observed` |
| `and_to_eq_value` | `8` | `0x0` | `0x0` | `True` | `record_observed` |
| `lshl_to_or` | `8` | `0x3c800000` | `0x3c800000` | `True` | `record_observed` |
| `eq_to_jz` | `9` | `0x5a5aa55a` | `0x5a5aa55a` | `True` | `record_observed` |
| `and_to_eq_value` | `9` | `0x0` | `0x0` | `True` | `record_observed` |
| `lshl_to_or` | `9` | `0x3c800000` | `0x3c800000` | `True` | `record_observed` |
| `eq_to_jz` | `10` | `0x5a5aa55a` | `0x5a5aa55a` | `True` | `record_observed` |
| `and_to_eq_value` | `10` | `0x0` | `0x0` | `True` | `record_observed` |
| `lshl_to_or` | `10` | `0x3c800000` | `0x3c800000` | `True` | `record_observed` |
| `eq_to_jz` | `11` | `0x5a5aa55a` | `0x5a5aa55a` | `True` | `record_observed` |
| `and_to_eq_value` | `11` | `0x0` | `0x0` | `True` | `record_observed` |
| `lshl_to_or` | `11` | `0x3c800000` | `0x3c800000` | `True` | `record_observed` |
| `eq_to_jz` | `12` | `0x5a5aa55a` | `0x5a5aa55a` | `True` | `record_observed` |
| `and_to_eq_value` | `12` | `0x0` | `0x0` | `True` | `record_observed` |
| `lshl_to_or` | `12` | `0x3c800000` | `0x3c800000` | `True` | `record_observed` |
| `eq_to_jz` | `13` | `0x5a5aa55a` | `0x5a5aa55a` | `True` | `record_observed` |
| `and_to_eq_value` | `13` | `0x0` | `0x0` | `True` | `record_observed` |
| `lshl_to_or` | `13` | `0x3c800000` | `0x3c800000` | `True` | `record_observed` |
| `eq_to_jz` | `14` | `0x5a5aa55a` | `0x5a5aa55a` | `True` | `record_observed` |
| `and_to_eq_value` | `14` | `0x0` | `0x0` | `True` | `record_observed` |
| `lshl_to_or` | `14` | `0x3c800000` | `0x3c800000` | `True` | `record_observed` |
| `eq_to_jz` | `15` | `0x5a5aa55a` | `0x5a5aa55a` | `True` | `record_observed` |
| `and_to_eq_value` | `15` | `0x0` | `0x0` | `True` | `record_observed` |
| `lshl_to_or` | `15` | `0x3c800000` | `0x3c800000` | `True` | `record_observed` |
| `eq_to_jz` | `16` | `0x5a5aa55a` | `0x5a5aa55a` | `True` | `record_observed` |
| `and_to_eq_value` | `16` | `0x0` | `0x0` | `True` | `record_observed` |
| `lshl_to_or` | `16` | `0x3c800000` | `0x3c800000` | `True` | `record_observed` |

## Interpretation

These are conservative real-NPU gaps for source assembly emitted without Peano scheduling. A future MyLM-style generator should fill the gaps with independent work, not literal nops. The values are still useful as a hazard-checking lower bound for generated tiny kernels.
