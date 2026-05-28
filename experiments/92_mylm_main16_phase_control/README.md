# Experiment 92: MyLM Main16 Phase Control

This experiment closes the scheduler-critical part of the main16 17-dword
record header question.

Exp85/86 showed that each main16 tile emits a 17-dword compact record:

```text
1 dword control/header + 32 bf16 projection values
```

The unresolved part was whether the header bitfields and phase order were still
too opaque to reproduce. This experiment joins the dispatcher call windows,
body-local replay counters, and Qwen3 phase block counts.

The normal mode body order is:

```text
body 0x1870: header 0x1, 12 records/tile = Q/K/V
body 0x1e80: header 0x4,  8 records/tile = O
body 0x2490: header 0x8, 48 records/tile = up/gate
body 0x2aa0: header 0x4,  8 records/tile = down
```

The important correction is that the dispatcher setup slots after `jl/j` must
be treated as body-entry setup. The same slot-window model is required by the
top-level loop; otherwise the loop counter would not advance before the next
iteration.

This means the fused-layer scheduler does not need the full bit-level
decomposition of `0x1/0x4/0x8` yet. It needs the concrete control words above
and the static body order. `0x4` is reused for O and down, so those phases are
distinguished by position and replay count. `0x8` covers the combined up/gate
family; exp96 later shows row1 compacts each global replay into a 257-dword
packet before `c6r2` consumes adjacent payload packets as one SwiGLU input.
