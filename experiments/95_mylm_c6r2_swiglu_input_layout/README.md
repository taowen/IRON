# 95_mylm_c6r2_swiglu_input_layout

This experiment resolves the half ordering inside the `c6r2` SwiGLU input
window.

Previous experiments proved the size fit:

```text
c6r2 input  = 512 dwords = 0x800 bytes
c6r2 output = 256 dwords = 0x400 bytes
```

Exp95 asks which input half is `up` and which half is `gate`.

## Run

```bash
.venv/bin/python experiments/95_mylm_c6r2_swiglu_input_layout/run.py --reuse
```

Artifact:

```text
/tmp/iron_exp95_mylm_c6r2_swiglu_input_layout/c6r2_swiglu_input_layout.txt
```

## Result

`c6r2` target `0x400` treats the 0x800-byte input as two 0x400-byte halves:

```text
input + 0x000: up slice,   512 bf16
input + 0x400: gate slice, 512 bf16
output:        SiLU(gate) * up, 512 bf16
```

Evidence:

- `0x43c` advances the input pointer by `+0x400`.
- `0x490` reads `wl11` from the advanced pointer.
- `0x4a4..0x4ae` immediately feeds that value through `vfloor`, clamp, and
  table-index style code, which is the gate/SwiGLU activation path.
- `0x4b2` reads the paired first-half data via `dj0=-0x400`.
- later multiply/convert/store instructions produce one 0x400-byte output.

This matches MyLM's physical patch order:

```text
Q, K, V, O, up, gate, down
```

Exp96 later closes the compact-record route that assembles this input. The
remaining unknown is the element/lane order inside each 512-bf16 slice and the
exact up/gate N-block pairing order.
