# Experiment 44: Projected Current-Write Attention Closed FFN

This experiment connects the two retained paths that were previously proven
separately:

```text
hidden
  -> Q/current-K/current-V projection
  -> current K/V writeback into the KV cache BO
  -> scan the updated KV cache
  -> attention output
  -> O projection
  -> FFN norm/gate/up/SwiGLU/down
  -> final output
```

The first version is intentionally small: one KV group, four Q heads, fixed
`L=31`, deterministic `i32` arithmetic.  It proves the fused-layer dataflow
boundary, not final Q4NX performance.

## Contract

- Runtime-visible inputs:
  - `hidden`: 512 `i32`
  - `kv_cache`: 8192 `i32` (`K[32,128] + V[32,128]`)
  - `o_weight`: 2048 `i32`
  - `ffn_weight`: 4096 `i32`
- Runtime-visible output:
  - `output`: 512 `i32`
- Internal-only tensors:
  - projected query: 512 `i32`, split as two 256-dword streams.
  - attention output: 512 `i32`, split as two 256-dword streams.
  - O output: 512 `i32`, split as two 256-dword streams.
  - FFN gate/up/SwiGLU: tile-local buffers.

The current token is `L - 1 = 30`.  The projection tile writes current K/V to
the runtime KV cache BO, the runtime syncs that writeback, and only then the
attention tile reads K/V history from the same BO.  The CPU reference applies
the same writeback before computing attention.

There are no host-visible Q/K/V, attention output, O output, gate, up, or
SwiGLU buffers.

## Result

Real NPU verification passes:

```text
PASS: current K/V writeback verified in KV cache BO.
PASS: projected current-write attention closed FFN verified.
```

Measured runtime for this scalar `i32` contract was about `1235 us` on NPU2.

One implementation constraint was important: the AIE tile kernels keep this
experiment's arithmetic in `int32`.  An earlier diagnostic version used
`int64_t` plus a custom truncating divide in the FFN path.  That forced a
64-bit division helper into the AIE object and corrupted the high half of a
tile-local intermediate buffer.  The value ranges in this contract do not need
64-bit math, so the final version uses native `int32` division.
