# Experiment 17: per-slot ping-pong and shim BD ownership

This experiment isolates the first failure found in experiment 16.

The original minimal failure was:

```text
gate-buffer-extra:
  core consumes gate chunk0 + gate chunk1
  runtime still queues up chunk0 + up chunk1
  core copies gate_buf to output
```

With experiment 16's shared `wt_empty/wt_full` and reused shim BD1/BD2, the
output matched `gate chunk0 + up chunk1`, proving that a later weight push
overwrote an earlier queued descriptor/data path.

## Result

Two fixes are needed for this boundary:

1. Use per-slot ping/pong locks on both memtile and core-local weight buffers.
2. Do not rewrite a shim BD descriptor after queueing it and before a sync.

The second point is the critical finding from this experiment. Per-slot locks
alone did not fix `gate-buffer-extra`; the failure remained identical. After
switching the runtime weight pushes to unique shim BD descriptors, the probes
passed.

## Commands Run

```bash
/var/home/taowen/projects/IRON/.venv/bin/python phase_probe.py gate-buffer
/var/home/taowen/projects/IRON/.venv/bin/python phase_probe.py gate-buffer-extra
/var/home/taowen/projects/IRON/.venv/bin/python phase_probe.py gate-after-up
/var/home/taowen/projects/IRON/.venv/bin/python run_npu.py
```

Observed probe results:

```text
gate-buffer:       bad=0/128, max_abs=0.5000
gate-buffer-extra: bad=0/128, max_abs=0.5000
gate-after-up:     bad=0/128, max_abs=0.5000
```

The complete FFN no longer produced NaNs or phase-wide corruption. It ran with
1/128 elements outside the current tolerance, so the remaining issue is no
longer the phase/descriptor ownership bug.

## Design Rule

For a runtime sequence that queues multiple DMA pushes without a sync:

```text
writebd(BD1, offset0); push(BD1)
writebd(BD1, offset1); push(BD1)
```

is unsafe because the hardware may consume `BD1` after it has already been
rewritten. Use distinct descriptors, or wait/sync before descriptor reuse.

For data buffers that are reused by phase:

```text
ping_empty / ping_full
pong_empty / pong_full
```

is the correct ownership model. A single counted `empty/full` pair is too weak
to express which physical slot is protected.
