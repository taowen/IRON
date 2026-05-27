# Experiment 39: Projected Current-Write Full Attention

Exp38 proved that the full `32Q/8KV` attention fabric fits and routes, but it
still accepted query and cache as host-prepared tensors. Exp39 closes the two
missing boundaries:

```text
hidden[4096] bf16
  -> per-worker NPU query projection
  -> per-KV-group NPU current K/V projection
  -> current K/V writeback into the cache BO
  -> sync
  -> rounded K/V cache scan
  -> row1 token-major to dim-group-major reshape
  -> full 32Q/8KV attention fabric
  -> 4096-dword attention output
```

## What It Proves

- Query is no longer host-provided. All 32 attention workers generate their own
  Q vector from the hidden vector before consuming K/V history.
- Current K/V is no longer host-prewritten. The row-0 worker in each KV group
  generates the group current K/V, streams it to shim S2MM, and updates the same
  cache BO that attention later scans.
- The current-write sync boundary scales from a single GQA group to all eight
  KV groups.
- The full exp38 row1 reshape/fanout and packet-gather fabric still fits after
  adding hidden ingress and current-write egress.

## Scope

The projection kernel is deliberately deterministic and compact. It proves the
layer-engine dataflow boundary, not final Q4NX projection performance. The full
Q4NX online MVM contract was proven separately in the projection experiments;
this experiment focuses on joining projected Q/K/V, current writeback, and the
full attention fabric without host-provided query/current tensors.

Still out of scope:

- Q/K norm and RoPE,
- MyLM's exact edge/aux attention microkernel,
- O projection,
- FFN,
- residual and RMSNorm.

## Result

Real NPU run:

```text
L=17  PASS  time=54069.1us  max_abs=0.0000385
L=31  PASS  time=60946.1us  max_abs=0.0000153
L=32  PASS  time=61595.4us  max_abs=0.0000156
L=79  PASS  time=85254.1us  max_abs=0.0000129
```

The important result is not latency. This experiment deliberately uses a
simple deterministic projection kernel so the dataflow is easy to verify. The
contract result is that projected Q, projected current K/V, current cache
writeback, sync, updated-cache scan, full row1 fanout, and 32-head packet gather
can coexist in one `8 x 4` compute fabric.

## Run

```bash
source /opt/xilinx/xrt/setup.sh
.venv/bin/python experiments/39_projected_current_write_full_attention/run_npu.py
```
