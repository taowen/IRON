# 88_mylm_aux_compact_record_roles

This experiment narrows the remaining post-O/FFN question by auditing the two
aux compute tiles that sit around the compact-record path: `c1r2` and `c6r2`.

It does not claim the complete FFN schedule is decoded. The purpose is to stop
treating these tiles as unknown transport shims and classify what their BD
sizes, route adjacency, and core programs actually imply.

## Run

```bash
.venv/bin/python experiments/88_mylm_aux_compact_record_roles/run.py --reuse
```

Artifact:

```text
/tmp/iron_exp88_mylm_aux_compact_record_roles/aux_compact_record_roles.txt
```

## Result

What is now supported by BD and disassembly evidence:

- `c1r2` has two 2048-dword inputs, which are full 4096-bf16 hidden vectors,
  and its program contains a sum-of-squares / reciprocal-sqrt style full-vector
  path. It should be treated as a full-vector RMSNorm/residual-style aux
  compute station.
- `c6r2` has compact 512-dword input and 256-dword output windows, receives the
  exact route-id-8 compact path, and contains clamp/gather/format conversion
  code. It should be treated as a compact record/indexed-format station.
- packet8 still has no packet-enabled BD source. It is a local compact-record
  route/bridge, not a normal packet DMA tensor stream.

Exp89 refines the `c6r2` role further: the 512-dword input and 256-dword output
fit one `SwiGLU(gate) * up` slice, and `c6r1` gathers 6144 dwords, exactly the
full 12288-bf16 FFN intermediate for down. Exp95 fixes the physical half order
as `up` then `gate`; exp96 closes the row1 compact route that assembles those
halves.

What remains unknown:

- the exact phase-dependent identity of `c1r2` inputs and outputs;
- the 2049-dword `c1r2` output control/header field;
- the exact `c6r2` payload element order and up/gate N-block pairing order;
- the exact schedule for post-O residual, RMSNorm, up/gate, SwiGLU, and down
  over these aux paths.
