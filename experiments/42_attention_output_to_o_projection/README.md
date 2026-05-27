# Experiment 42: Attention Output To O Projection

This experiment targets the next missing MyLM/FastFlowLM fused-layer ABI:
the edge attention output must return to a projection worker without first
becoming a host-visible DDR tensor.

The math is intentionally deterministic `i32` arithmetic.  The experiment is
not a high-performance O projection kernel; it proves the dataflow boundary.

## Contract

- `edge_attn` receives:
  - `current`: 512 `i32` dwords from DDR.
  - `history`: 2048 `i32` dwords from DDR.
- `edge_attn` produces a 512-dword attention output.
- The attention output flows directly from `edge_attn` to `o_proj`.
- `o_proj` receives:
  - the internal 512-dword attention output.
  - `o_weight`: 2048 `i32` dwords from DDR.
- `o_proj` computes 512 `i32` output dwords and drains them to DDR.

The runtime sequence has only four host-visible buffers:

```text
current, history, o_weight, output
```

There is deliberately no host-visible `attention_output` buffer.  The
attention output is an internal tile-to-tile stream.

The internal 512-dword stream is split into two 256-dword DMA BDs on both the
producer and consumer side.  This matches the practical boundary observed while
debugging: a single 512-dword core-to-core BD produced correct data only up to
the first 256 dwords on this route, while the two-BD phase handoff is exact.

## Why This Matters

Exp41 proved a small 17-dword state can move directly between edge tiles.
This experiment proves the larger attention-value handoff needed after
shape-B/edge attention: a full 512-dword attention result can be consumed by
an O-projection phase without a DDR round trip.

If this passes, the remaining open problem is no longer whether the handoff is
representable; it becomes how to attach the real MyLM projection fabric and
Q4NX O weights to this boundary.
