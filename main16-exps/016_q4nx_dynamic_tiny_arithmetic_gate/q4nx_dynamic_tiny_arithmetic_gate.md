# Q4NX Dynamic Tiny Arithmetic Gate

- Status: `failed`

## Results

| case | formula == MyLM | dynamic == expected | dynamic == MyLM | dynamic status |
| --- | --- | --- | --- | --- |
| `dyn_q4word0_allnibbles` | `True` | `False` | `False` | `timeout` |

## Interpretation

This generated whole-core source-assembly body reads q4/scale/zero fields from the active weight chunk, computes payload-word low/high bf16 halves through a tiny count-to-bf16 path, consumes the same 16 stream chunks, and emits one compact record. It is intentionally narrow and not yet the MyLM software-pipelined hot loop.
