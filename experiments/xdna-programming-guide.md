# XDNA Programming Guide For The Fused-Layer Experiments

This note is the current compact model for the IRON fused-layer experiments.
It intentionally replaces the older chronological log. The experiment
directories keep the reproducible evidence; this file explains why the dataflow
is shaped this way and what still needs to be decoded.

## Reading Path

Read in this order:

1. `54_real_qwen_patch_schedule`
2. `63_mylm_packetized_patch_phase`
3. `65_mylm_fused_layer_engine_v0`
4. `66_mylm_real_attention_o_phase`
5. `67_mylm_global_qkv_o_layout`
6. `70_mylm_stream_switch_physical_path_replay`
7. `78_mylm_row0_current_kv_writeback`
8. `79_mylm_row1_history_split`
9. `80_mylm_shape_ab_return_phase`
10. `81_mylm_shape_b_hidden_payload`
11. `82_mylm_attention_o_bridge`
12. `83_mylm_layer_boundary_contract`
13. `84_mylm_shape_ab_carrier_lock`
14. `87_mylm_shape_ab_carrier_block_order`
15. `93_mylm_shape_ab_carrier_lane_usage`
16. `94_mylm_shape_a_carrier_producer_layout`
17. `95_mylm_c6r2_swiglu_input_layout`
18. `96_mylm_upgate_c6r2_compact_route`
19. `85_mylm_main16_phase_record`
20. `86_mylm_dispatcher_packet8_aux`
21. `88_mylm_aux_compact_record_roles`
22. `89_mylm_fullvector_ffn_dataflow`
23. `90_mylm_main16_activation_bridge`
24. `91_mylm_c1r2_phase_order`
25. `92_mylm_main16_phase_control`

Removed experiments are listed in `README.md`. In short, the old edge-ring,
linked-BD, early main16/O, and short-lived reverse probes were folded into this
path once exp63 and exp70/78-90 made the physical model more precise.

## Hardware Constraints

The useful partition is an 8-column AIE2P array:

```text
row0: shim / host DMA boundary
row1: memtiles / BD rings / packet and circuit fanout
row2..row5: compute tiles
```

The current fused-layer design uses:

- main16: `c2..c5/r2..r5`, reused for Q/K/V/O/up/gate/down projection work,
- edge tiles: `c0/c7 rows2..5`, used for attention-side work,
- aux tiles: `c1/c6`, used for Q/K/V postprocess, current distribution,
  return bridge, and compact records.

XDNA does not give enough compute tiles to allocate every Qwen3 operator as a
separate live kernel. The design must be a static dataflow engine: core
programs, memtile BDs, stream switches, and locks are preconfigured; runtime
patches descriptors and starts a layer run.

## Qwen3 Patch Shape

The real Qwen3 projection schedule is:

```text
Q:    4096 -> 4096    64 patches
K:    4096 -> 1024    16 patches
V:    4096 -> 1024    16 patches
O:    4096 -> 4096    64 patches
up:   4096 -> 12288  192 patches
gate: 4096 -> 12288  192 patches
down: 12288 -> 4096   64 patches
```

Total: `608` patches.

One hidden-dim MyLM patch is:

```text
64 output rows x 4096 K = 0x28000 bytes
```

The down patch is:

```text
64 output rows x 12288 K = 0x78000 bytes
```

The main16 tile ABI is:

```text
activation input: 128 dwords = 256 bf16
Q4NX weight input: 1280 dwords
record output: 17 dwords = 1 dword header + 32 bf16 values
```

One 17-dword record is one main tile's 32-row slice. It is not a full hidden
vector and should not be stretched into a fake Q/K/V tensor.

## Patch Queue

Exp62 proved why a literal same-channel row1 BD chain is fragile: the schedule
crossed into illegal odd-channel BD banks and timed out. Exp63 fixed the patch
handoff by keeping one host logical queue but packetizing each shim BD so row1
can land patches in legal banks.

Practical rule:

```text
Host/shim patch queue can stay logical and linear.
Row1 landing BDs must respect physical channel-bank limits.
Packet routing is the right abstraction for the patch phase.
```

Direct CDO generation should not be used just because the layer is complex.
Use it only when a phase schedule cannot be expressed through the existing
MLIR-AIE path or when descriptor-program overhead becomes the real bottleneck.

## Attention Dataflow

The old wrong model was a global Q/K/V sideband collector. MyLM evidence points
to a narrower physical split.

Current-token path:

```text
main16 Q/K/V projection records
  -> c1r3 current Q/K/V postprocess
  -> full Q circuit output to c6r1.bd24
  -> c6r1 bd25/bd2/bd26/bd3
  -> Shape-A current windows:
     c0r2, c0r4, c7r2, c7r4

c1r3 packet14 -> row0 current K writeback
c1r3 packet15 -> row0 current V writeback
```

KV history path:

```text
row0 scans rounded KV cache tiles
  K03/V03 on c0 side, K47/V47 on c7 side
  -> row1 memtile split
  -> K history, 2048-dword head-pair streams, to Shape-A
  -> V history, 2048-dword head-pair streams, to Shape-B
```

Attention compute path:

```text
Shape-A:
  current Q window + K history
  -> compact hidden carrier through neighbor-local aliases and locks

Shape-B:
  hidden carrier + V history
  -> local fp32 accumulator
  -> 512-dword bf16 output window
```

Return to O:

```text
four Shape-B windows fill c6r1 0x28000..0x28800
  -> c6r1.bd29 publishes one 2048-dword packet2 block
  -> c1r1 bridges eight 256-dword quanta
  -> all main16 activation rings
  -> O consumes chunks 0..15 in linear packet2 order
```

Exp67 fixes the O layout independently of routing:

```text
chunk c carries heads 2*c and 2*c+1, flattened from Attn[32][128]
```

## Shape-A/B Carrier

Exp80 and exp84 fix the lock ownership:

```text
Shape-A owns local hidden locks:
  L7/L5 ready
  L4 empty

Shape-B accesses those locks through north-neighbor immediates:
  0x7 / 0x5 / 0x4
```

Exp84 fixes the alias relation:

```text
Shape-A aliases:     0x62400, 0x66000
Shape-B local bases: 0x72400, 0x76000
delta:               +0x10000
```

Exp81 gives the current capacity model:

```text
0x76000..0x76140: compact softmax carrier capacity
  0x100 bytes: 8 heads x 16 token bf16 weights
  0x040 bytes: 8 heads x 2 fp32 online-softmax scalars

0x76140..0x77140: Shape-B-local fp32 output accumulator
  8 heads x 128 dim x fp32 = 0x1000 bytes
```

This is a sizing and usage model. Exp87 narrows the block phase order, but it
does not decode every value field.

Exp87 tightens the block phase order:

```text
Shape-A:
  rel local L7
  acq local L4
  helper 0x650 stores base[0x100]
  post-call store through dj0 = 0x100
  rel local L5

Shape-B:
  acq north-neighbor L7
  acq north-neighbor L5
  hidden consume window splits p0=base, p1/p3=base+0x100
  rel north-neighbor L4
```

This proves a two-ready-phase carrier protocol and the carrier split. Shape-B's
setup bundle contains:

```text
paddb [p1], #0x100; mov p0,p1
```

The same-bundle pointer rule is the same as in the Q4 microkernel:
`mov p0,p1` observes the pre-increment base while `paddb` advances `p1` for the
next slot. Therefore Shape-B consumes:

```text
base + 0x000: 0x100-byte bf16 softmax-weight block through p0
base + 0x100: 0x040-byte fp32 online-softmax scalar block through p1/p3
```

Exp93 tightens Shape-B's actual carrier reads:

```text
target 0x410 base reads: 0x00, 0x40, 0xc0, 0x80
target 0x600 base reads: 0x00, 0x40, 0xc0, 0x80
target 0x410 scalar:     full 0x40-byte vector load
target 0x600 scalar:     first eight dwords, offsets 0x00..0x1c
```

So the algorithmic fit is more precise: `base[0x100]` is consumed as four
0x40 vector blocks, each exactly large enough for two query heads x 16 token
bf16 weights. The scalar block is online-softmax scale/normalization state; it
should not be named raw max/sum until value calibration proves each lane. The
remaining unknowns are head-pair order, head order inside each pair, token lane
order, and scalar lane semantics.

Exp94 tightens the producer side:

```text
Shape-A helper 0x650:
  8 x 0x20-byte `vst wl0` carrier-base stores = base[0x100]
```

So the practical carrier ABI is:

```text
base record:      one query head x 16 token bf16 weights = 0x20 bytes
base window:      8 records = one Shape-A tile's local Q heads
Shape-B read:     four 0x40 head-pair blocks
scalar window:    16 fp32 online-softmax lanes, still opaque
```

First principles plus the Q-window route make the implementation assumption:

```text
c0r2 -> global Q heads 0..7
c0r4 -> global Q heads 8..15
c7r2 -> global Q heads 16..23
c7r4 -> global Q heads 24..31
```

This global-head grouping is strong enough for the fused-engine skeleton, but
the exact local head order and token lane order still require value calibration.

## Main16 Records And Dispatcher

Exp85 fixes the visible record shape:

```text
record +0x00: 1 dword header
record +0x04: 16 bf16 values
record +0x24: 16 bf16 values
```

Exp86 fixes the header source:

```text
each visible phase body:
  st  r0, [sp, #-12]
  lda el0, [sp, #-12]
  st  el0, [record +0x00]
```

The first dword is the body input register `r0`, not a computed vector value.
Exp92 fixes the scheduler-critical header values and replay counts. The direct
caller uses the post-control-flow setup slots as the body-entry window; the
same slot-window model is required by the top-level loop, otherwise its loop
counter would not advance before the next iteration. The Q4 call gives the
independent proof: `jl #0x1f0` is followed by p0/p1/p2 setup slots, and the
microkernel consumes those pointers immediately. In normal mode:

```text
body 0x1870: header 0x1, 12 records/tile = Q/K/V
body 0x1e80: header 0x4,  8 records/tile = O
body 0x2490: header 0x8, 48 records/tile = up/gate
body 0x2aa0: header 0x4,  8 records/tile = down
```

The alternate `0x30c0` body uses header `0x4` and 304 records/tile, equal to
`4 * (12 + 8 + 48 + 8)`, but Qwen3 layer transactions select the normal
mode-1 path.

Exp86 also fixes the main16 phase-program shape:

```text
dispatcher 0x36d0:
  if *phase_flag == 1:
    0x1870 -> 0x1e80 -> 0x2490 -> 0x2aa0
  else:
    0x30c0
```

This is not seven independent Q/K/V/O/up/gate/down kernels. It is one static
program with a conditional multi-body path and a single-body path.

The bit-level decomposition of `0x1/0x4/0x8` is still not decoded, but fused
scheduling no longer depends on it. Header `0x4` is reused for O and down, so
those phases are distinguished by body order and replay count. Header `0x8`
covers the combined up/gate family; exp96 shows row1 compacts each global
replay before `c6r2` pairs adjacent payload packets for SwiGLU.

## Packet Routes

AIE2P packet routing is not a single route selector. A packet slave slot
matches packet id/mask and supplies `arbitor + msel`; local packet masters
receive the packet when arbitor and `msel_enable` match.

Important decoded routes:

```text
packet14: c1r3 current K writeback -> row0 shim
packet15: c1r3 current V writeback -> row0 shim
packet2:  c6r1 attention return block -> c1r1 -> main16 O
packet8:  no packet-enabled BD source
```

Packet8 has exact route-id rows across:

```text
c1r2 -> c2r2 -> c3r2 -> c4r2 -> c5r2 -> c6r2
```

But because packet8 has no packet-enabled BD source, it must be treated as an
unpacketized compact/aux record path, not as a packet DMA tensor stream.

Exp88 narrows the aux tile roles around that compact path. Exp89 refines the
FFN interpretation, exp90 closes the reused activation bridge back into main16,
and exp91 constrains the `c1r2` full-vector phase order:

```text
c1r2:
  two 2048-dword inputs
  2049/2048-dword outputs
  full-vector sum-of-squares / reciprocal-sqrt style code
  hidden-input / RMSNorm / residual / final-output station

c6r2:
  512-dword input
  256-dword output
  clamp/gather/indexed-format conversion code
  SwiGLU(gate) * up slice station

c6r1:
  6144-dword gather/publish buffer
  12288 bf16 FFN intermediate for down projection
```

The first-principles fit is now stronger. A 2048-dword stream is exactly one
4096-element bf16 hidden vector, and RMSNorm/residual require full-vector
visibility. The transaction maps hidden input, two RMSNorm weight vectors, and
final hidden output through `c1r0/c1r2`, while q/k norm + RoPE side data maps to
`c1r3`.

For FFN, one `c6r2` input is 512 dwords:

```text
input[0x000..0x3ff] = 512 bf16 up slice
input[0x400..0x7ff] = 512 bf16 gate slice
SwiGLU output       = 512 bf16 SiLU(gate) * up = 256 dwords
24 slices x 256 dwords = 6144 dwords = 12288 bf16
```

Exp95 fixes the half order: target `0x400` advances the input pointer by
`+0x400`, feeds the second half into `vfloor` / clamp / table-index activation
code, then reads the paired first half through `dj0=-0x400` for the multiply.
That matches the MyLM physical patch order `up` before `gate`, and the
6144-dword total exactly matches the `c6r1` gather and packet0 publish buffer
for the down-projection activation vector.

Exp96 closes the transport shape into that 512-dword input. It is not a direct
stream of 17-dword main16 records into `c6r2`; row1 compacts the records in two
levels:

```text
per column c2r1..c5r1:
  17 + 16 + 16 + 16 = 65 dwords
  first record keeps the manual packet header
  the next three DMA masters use drop_header=1

c1r1:
  65 + 64 + 64 + 64 = 257 dwords
  one header + 16 main-tile payloads

c6r2 DMA_0:
  drop_header=1
  2 * (257 - 1) = 512 dwords per input BD
```

So one c6r2 input is two global 256-dword payload packets after header drop.
Exp95 gives their semantic order: up payload first, gate payload second. The
remaining schedule detail is the exact N-block pairing order, not the physical
row1 compact route.

Exp90 shows that packet2/O and packet0/down reuse the same physical activation
ingress:

```text
c6r1 packet source -> c1r1 DMA4 256-dword ping-pong input
                  -> c1r1 DMA1 multicast
                  -> main16 DMA0 128-dword bd0/bd1 activation ring
```

O publishes 2048 dwords, so it takes 8 c1r1 bridge iterations and becomes 16
main16 activation chunks. Down publishes 6144 dwords, so it takes 24 bridge
iterations and becomes 48 chunks. This is exactly the Qwen3 size model:

```text
O input    = 4096 bf16  = 2048 dwords  = 16 x 256-bf16 chunks
down input = 12288 bf16 = 6144 dwords  = 48 x 256-bf16 chunks
```

Exp91 resolves the apparent `c1r2` lock-phase ambiguity. The layer transaction
writes `c1r2` mode `1` at tile offset `0x3020`, which maps to core-local
`0x73020`, and writes `c1r2` lock `L6=1` at `0x1f060`. That selects the normal
full-layer path and satisfies the first `acq #0x36`.

The `c1r2.bd3` output is a 2049-dword manual-header full-vector packet0 replay
channel:

```text
base 0x7101c:
  1 dword packet/control header
  2048 dwords = one 4096-bf16 hidden vector
```

The core releases its ready lock `L3` by phase replay counts:

```text
+12 = Q/K/V input replays = 8 + 2 + 2 N-blocks
+48 = up/gate input replays = 24 + 24 N-blocks
+1  = final hidden-output boundary transfer
```

So the normal `c1r2` order is:

```text
hidden + RMS1 -> pre-attention RMSNorm -> 12 packet0 replays for Q/K/V
O result + residual + RMS2 -> post-attention RMSNorm -> 48 replays for up/gate
down result + residual -> final hidden -> one host-boundary output
```

The `+8` branch in the disassembly matches one 4096-output projection replay
count, but the Qwen3 layer transaction selects mode `1`, so that branch is not
the normal fused-layer path. The remaining `c1r2` unknown is register-level
ping/pong pointer order and value calibration, not the phase schedule.

## Layer Boundary

Exp83 decodes the fused-layer transaction boundary:

```text
arg0: hidden input and final hidden output
arg1: Q/K/V/O/up/gate/down Q4NX weights
arg2: two 4096-element RMSNorm weight vectors
arg3: q/k norm and compact RoPE side data
arg4: KV-cache current writes and rounded history scans
```

There are no host-visible descriptors for:

```text
O output
post-attention residual
post-attention RMSNorm
up/gate intermediate
SwiGLU output
down intermediate
```

Those states must live inside the static main16/edge/aux dataflow.

## Practical Design Rules

Do:

- keep patch transport, current K/V writeback, Q-window distribution, history
  split, Shape-A/B carrier, and the reused O/down activation bridge as separate
  physical paths;
- use packet routing only where there is a packet-enabled BD source or a
  decoded packet bridge;
- treat 17-dword records as compact per-tile records;
- treat `c1r2` `+12/+48/+1` releases as full-vector replay counts, not as
  arbitrary lock magic;
- keep deterministic payloads and host-visible debug only in calibration
  experiments.

Do not:

- model packet14/15 as Shape-A current DMA input;
- model packet8 as a packet DMA tensor stream;
- model Shape-A as a terminal DMA producer;
- add DDR scratch for O/residual/RMSNorm/FFN intermediates;
- search for seven separate main16 operator kernels.

## Remaining Work

1. Calibrate the remaining Shape-A/B carrier value order: local head order of
   the eight 0x20 base records, token lane order inside one record, and
   online-softmax scalar lane semantics in `scalar[0x40]`.
2. Calibrate register-level `c1r2` ping/pong pointer order and value layout.
3. Decode optional bit-level meaning of main16 headers `0x1/0x4/0x8`; the
   scheduler-critical values and replay counts are known.
4. Calibrate the exact element order inside each 16-dword main16 payload and the
   exact up/gate N-block pairing order into `c6r2`.
5. Replace diagnostic kernels with high-throughput RMSNorm, Q/K norm, RoPE,
   online softmax, SwiGLU, down, and Q4NX kernels.
6. Integrate one fused layer into layer-to-layer runtime submission and lm_head.
