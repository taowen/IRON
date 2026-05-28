# Experiment 91: MyLM c1r2 Phase Order

This experiment closes the main `c1r2` scheduling ambiguity left after exp89
and exp90.

It joins four evidence sources:

- the runtime layer transaction,
- CDO lock initialization,
- decoded `c1r2` BDs,
- segment-aware `c1r2` core disassembly.

The important result is that `c1r2` is not waiting on an unexplained software
lock. Runtime writes `c1r2` mode `1` and lock `L6=1` before descriptor patches,
which selects the normal full-layer path and releases the first `acq #0x36`.

The `L3` release magnitudes then map directly to Qwen3 projection N-block
counts:

```text
+12 = Q/K/V activation replays = 8 + 2 + 2
+48 = up/gate activation replays = 24 + 24
+1  = one final hidden-output transfer
```

So `c1r2.bd3` is the manual-header full-vector packet0 replay channel for
projection activations and the final hidden boundary. The exact register-level
ping-pong pointer order still needs value calibration, but the normal
full-layer phase order is now constrained enough to implement the engine
schedule.
