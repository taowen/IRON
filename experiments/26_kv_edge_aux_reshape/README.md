# Experiment 26: Edge Selector And Sideband Calibration

Exp26 continues from exp25, but it should not jump straight to attention or KV
reshape.  The MyLM reverse-engineering now shows three different edge path
types:

```text
packet current path:
  c1r3 ch3 packet14/15 -> shape-A current input

compact sideband path:
  main row2/row3 17-dword MM2S -> route8/route15 area -> c1r2/c6r2 aux

history path:
  row1 KV ring -> shape-A/B history inputs
```

The old "raw KV -> aux reshape -> attention-friendly stream" target was too
coarse.  It would hide exactly the thing that kept breaking earlier
experiments: selector meaning, packet/non-packet routes, and compact sideband
handoff are separate contracts.

## Final Goal

Build a small runnable NPU experiment that calibrates the edge routing contract
with deterministic payloads.

The first complete exp26 should contain:

- one packet source that emits a 512-dword current vector using packet14 or
  packet15 semantics;
- one unpacketized 17-dword sideband source, matching the route8-style compact
  main-fabric output;
- one history source that delivers a 2048-dword half-plane stream;
- one shape-A-like debug consumer with two input channels:
  - 512 dwords current;
  - 2048 dwords history;
- a host-visible debug output that records checksums or slices from each input
  so selector routing can be verified;
- optionally one shape-B-like follow-up consumer to test direct compact-state
  handoff without a DMA-visible output.

## Acceptance Criteria

- The experiment runs on real NPU.
- The current payload reaches the intended shape-A-like input.
- The 2048-dword history payload reaches the history input independently of the
  current path.
- The 17-dword sideband is routed as an unpacketized stream, not as an
  `aie.packet_source` path.
- Selector sweeps make it clear which route selector value feeds which consumer
  input port.
- Debug output is host-visible and deterministic; CPU reference can verify it
  exactly.

## Current Status

Runnable checkpoint achieved:

```bash
/var/home/taowen/projects/IRON/.venv/bin/python run_npu.py
```

Observed real-NPU result:

```text
PASS: left14
PASS: right15
SUCCESS: exp26 selector, history ring, and shape-B handoff verified.
```

The current implementation proves these concrete paths:

```text
current:
  c1r3 receives 512 dwords from shim1
  left14:  c1r3 emits packet14 -> c0r2 current input
  right15: c1r3 emits packet15 -> c7r2 current input

history:
  left14:  shim0 -> row1 memtile ping-pong ring -> c0r2 history input
  right15: shim7 -> row1 memtile ping-pong ring -> c7r2 history input
  each variant loads a 4096-dword rounded history source and forwards the first
  2048 dwords to the shape-A-like worker

sideband:
  c2r2 receives 17 dwords from shim2
  c2r2 emits a non-packet stream
  c6r2 receives the 17-dword compact sideband, writes a host-visible debug
  checksum, and forwards a transformed 17-dword stream to c6r3
  c6r3 verifies the shape-B-like compact-state handoff and writes its own
  host-visible debug checksum
```

The generated CDO/BD decode also shows the expected packet/non-packet split:

```text
left14:
  c1r3 bd2: len=512 packet=14
  c0r2 bd0: len=512 current input
  c0r2 bd1: len=2048 history input

right15:
  c1r3 bd2: len=512 packet=15
  c7r2 bd0: len=512 current input
  c7r2 bd1: len=2048 history input

shared sideband path:
  c2r2 bd2: len=17 packet_en=0
  c6r2 bd0: len=17 sideband input
  c6r2 -> c6r3: non-packet compact-state handoff
```

This is not the final MyLM edge attention path yet.  It is now a working
calibration baseline for packet current, row1-routed history input, and compact
sideband handoff.

## Non-Goals

- No Q4 projection.
- No softmax.
- No value accumulation.
- No full GQA attention.
- No full MyLM fused layer.

## Why This Is The Right Next Step

MyLM's edge shape A consumes current + history and has no DMA-visible output.
Shape B consumes history and emits a 512-dword vector, but has no visible
score/state input.  Route8 has no packet-enabled BD source and the main-fabric
route8 output is only 17 dwords.  Therefore the missing contract is not a large
KV tensor transform; it is a selector/sideband handoff problem.

Exp26 is done when we can write down and verify:

```text
packet route current input -> shape-A port X
history stream             -> shape-A port Y
compact sideband route     -> aux / shape-B handoff candidate
```

Only after this is proven should exp27 add actual attention math.

## Next Work

- Replace the checksum-only shape-A worker with a deterministic KV slice
  transform so token/head/dim layout can be verified element-by-element.
- Compare the generated row1 ring BDs with MyLM's 4096/2048/2048 lock-ring
  shape, then add the missing second half-plane stream if the route contract
  stays stable.
- Use a numeric probe for the 17-dword compact state to distinguish sideband
  metadata from partial attention state.
