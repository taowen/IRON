# 96_mylm_upgate_c6r2_compact_route

This experiment closes the transport shape that exp95 left open: how 17-dword
main16 up/gate records become one 512-dword `c6r2` SwiGLU input.

## Run

```bash
.venv/bin/python experiments/96_mylm_upgate_c6r2_compact_route/run.py --reuse
```

Artifact:

```text
/tmp/iron_exp96_mylm_upgate_c6r2_compact_route/upgate_c6r2_compact_route.txt
```

## Result

The route is a two-level row1 compact tree plus final header drop at `c6r2`.

Per column, `c2r1..c5r1` gather four main16 records:

```text
17 + 16 + 16 + 16 = 65 dwords
```

The first incoming record keeps the manual packet header. The next three
incoming DMA masters have `drop_header=1`, so they contribute payload only.

`c1r1` then gathers the four column packets:

```text
65 + 64 + 64 + 64 = 257 dwords
```

That is one packet header plus the full 16-main-tile payload:

```text
1 header + 16 tiles * 16 payload dwords = 257 dwords
```

Finally, `c6r2 DMA_0` has `drop_header=1` and its input BD length is 512
dwords:

```text
2 * (257 - 1) = 512 dwords
```

Combining this with exp95, one `c6r2` input is:

```text
input[0x000..0x3ff] = up payload,   512 bf16
input[0x400..0x7ff] = gate payload, 512 bf16
```

## Implication

The fused engine should not model up/gate as a direct stream of 17-dword
records into `c6r2`. The production ABI is:

```text
16 main16 records
  -> column row1 compact packets
  -> c1r1 global compact packet
  -> c6r2 header-drop input
```

The remaining value-order problem is narrower: the exact element order inside
each 16-dword main16 payload, and the exact up/gate N-block pairing order.
