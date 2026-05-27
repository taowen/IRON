# Experiment 24: MyLM Fabric Boundary

Status: deprecated negative evidence. Do not extend this experiment.

This experiment copied the outer split of the MyLM layer but not the real
edge/KV dataflow boundary. It remains useful as a failure case showing why a
raw-history single edge worker is not a valid MyLM replica. Continue with
`25_mylm_edge_bd_ring`.

This experiment is the first reduced replica that follows the current MyLM
reverse-engineering contract instead of scaling the single-worker shortcut.

It splits the layer into two physical fabrics:

- `proj0` on column 0: finite Q/K/V projection from `hidden + packed Q4NX`
- `edge0` on column 1: current K/V cache write, rounded history scan, online
  softmax attention, and layer epilogue

The edge tile does not receive Q/K/V from the host. Projection output crosses an
AIE fabric boundary through the column-1 memtile:

```text
hidden, QKV weights -> mem0 -> proj0

proj0 query      -> mem1 query bridge       -> edge0 input channel 0
proj0 current K  -> mem1 current bridge     -> edge0 input channel 1
proj0 current V  -> mem1 current bridge     -> edge0 input channel 1

KV cache history -> mem1 static ping-pong   -> edge0 input channels 0/1
edge0 current K/V -> KV cache BO
edge0 output      -> output BO
```

## What This Proves

- Projection and KV/attention are no longer one overloaded worker.
- Q/K/V weight consumption is finite; there is no infinite weight ring.
- The same edge input channel has an explicit phase sequence:
  - channel 0: resident query, then K history
  - channel 1: current K, current V, then V history
- Current K/V are written to the cache before history reads are queued.
- History reads still use `ceil(L / 16)` rounded scan tiles and kernel-side tail
  masking.
- The projection output stays inside the AIE graph until the observable current
  cache write and final output.

## What This Still Does Not Prove

- The full MyLM 16-main-tile projection fabric.
- The exact c0/c7/c1/c6 attention worker placement.
- GQA fanout across 8 KV groups.
- Current K/V bypass; this experiment uses the safe write-cache-then-read
  contract.

## Why This Is Deprecated

The key simplification is wrong:

```text
raw KV history -> edge0 worker
```

MyLM has a larger edge path:

```text
row0 shim descriptor
  -> c0/c7 row1 static BD rings
  -> 4096-dword raw plane tile reshaped into 2048-dword streams
  -> c1/c6 auxiliary packet/control path
  -> attention-side workers
```

Debugging showed that small contexts can pass while contexts crossing a
16-token history tile fail or produce NaNs. The host descriptor formulas were
correct; the dataflow boundary was not. Keeping this experiment around prevents
us from reintroducing the same shortcut.

## Run

```bash
/var/home/taowen/projects/IRON/.venv/bin/python run_npu.py
```
