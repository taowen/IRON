# Experiment 43: Attention + O Projection + FFN Closed Tail

This experiment folds the post-attention tail into one internal dataflow:

```text
current + history
  -> edge attention
  -> O projection
  -> FFN norm/gate/up/SwiGLU/down
  -> final output
```

The math is deterministic `i32` arithmetic.  This is not a performance kernel
and not a real Qwen3 layer.  It proves the handoff ABI needed by a fused-layer
engine.

## Contract

- Runtime-visible inputs:
  - `current`: 512 `i32`
  - `history`: 2048 `i32`
  - `o_weight`: 2048 `i32`
  - `ffn_weight`: 4096 `i32`
- Runtime-visible output:
  - `output`: 512 `i32`
- Internal-only tensors:
  - attention output: 512 `i32`, split as two 256-dword streams.
  - O projection output: 512 `i32`, split as two 256-dword streams.
  - FFN gate/up/SwiGLU buffers: 3 x 512 `i32`, tile-local only.

There is no host-visible `attention_output`, `o_output`, `gate`, `up`, or
`swiglu` BO.

## What This Proves

- O projection output can enter FFN phase without a DDR round trip.
- Gate and up intermediate results can stay in FFN tile-local buffers.
- SwiGLU can consume local gate/up and feed local down projection state.
- Only the final layer-tail output is drained to host.

The two large cross-tile vector handoffs use the Exp42 rule: split each
512-dword vector into two independently locked 256-dword halves.
