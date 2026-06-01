# Q4NX Dynamic Readback Probe

- Status: `passed`

## Observations

| name | base | offset | observed | expected | match |
| --- | --- | --- | --- | --- | --- |
| `full_weight_scale0` | `0x72800` | `0x0` | `0x3c803c80` | `0x3c803c80` | `True` |
| `full_weight_zero0` | `0x72800` | `0x200` | `0x3c803c80` | `0x3c803c80` | `True` |
| `full_weight_q4word0` | `0x72800` | `0x400` | `0x11111111` | `0x11111111` | `True` |
| `full_weight_q4word512` | `0x72800` | `0xc00` | `0x11111111` | `0x11111111` | `True` |
| `local_weight_scale0` | `0x2800` | `0x0` | `0x140ad228` | `0x3c803c80` | `False` |
| `local_weight_zero0` | `0x2800` | `0x200` | `0x9dc5529b` | `0x3c803c80` | `False` |
| `local_weight_q4word0` | `0x2800` | `0x400` | `0x3c803c80` | `0x11111111` | `False` |
| `local_weight_q4word512` | `0x2800` | `0xc00` | `0x3c803c80` | `0x11111111` | `False` |
| `full_activation_word0` | `0x78000` | `0x0` | `0x3f803f80` | `0x3f803f80` | `True` |
| `local_activation_word0` | `0x8000` | `0x0` | `0x11111111` | `0x3f803f80` | `False` |
| `full_record_header` | `0x73c1c` | `0x0` | `0x1` | `0x1` | `True` |
| `local_record_header` | `0x3c1c` | `0x0` | `0xb468d0` | `0x1` | `False` |
| `full_weight_pong_scale0` | `0x74000` | `0x0` | `0x0` | `0x0` | `True` |
| `full_activation_pong_word0` | `0x7c000` | `0x0` | `0x0` | `0x0` | `True` |
| `full_weight_scale1` | `0x72800` | `0x4` | `0x3c803c80` | `0x3c803c80` | `True` |
| `full_weight_q4word1` | `0x72800` | `0x404` | `0x0` | `0x0` | `True` |

## Interpretation

A pass means source assembly can see the DMA-written full-address weight and activation buffers immediately after the normal full-lock acquire. Experiment 016 can then treat any remaining mismatch as generated arithmetic/control-flow, not buffer visibility.
