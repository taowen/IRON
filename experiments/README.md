# Retained Fused-Layer Experiments

This directory now keeps only the experiments that still support the current
MyLM-style fused-layer direction. Older one-off probes were removed after their
positive lessons were covered by later milestones or their negative lessons were
folded into this summary.

If the raw MLIR-AIE code is hard to read, start with
[XDNA Programming Guide For The Fused-Layer Experiments](xdna-programming-guide.md).
It explains the tile model, row1 memtile rings, DMA BDs, locks, packet routes,
Q4NX patch shape, main16/edge16 split, and the remaining full-layer questions.

## Current Understanding

The target is our own full-layer fused Qwen3 decode engine. MyLM is the
hardware-mapping reference, not an instruction-for-instruction compatibility
target.

The full layer fits by time-reusing one physical fabric instead of allocating
separate tiles for Q/K/V/O/FFN. The main projection fabric is the 16 compute
tiles `c2..c5/r2..r5`. The same main16 fabric is reused across Q, K, V, O, up,
gate, and down phases. The edge/aux side uses `c0/c1/c6/c7` for KV scan,
current-token packets, attention-side replay, compact phase records, and
handoff back to main16.

MyLM's high-level shape is static PDI/CDO dataflow, not a chain of ordinary
host operators. Core programs, memtile BD rings, stream switches, and locks are
configured up front. Runtime patches descriptor addresses/lengths and starts a
layer run. This is why the current limiting issue is exact phase handoff and
BD/lock scheduling, not broad tile-count feasibility.

The main-tile projection ABI we currently trust is:

- activation slice on input channel 0: `128 dwords` / `256 bf16`,
- Q4NX weight chunk on input channel 1: `1280 dwords` / `5120 bytes`,
- compact debug/sideband record: `17 dwords`.

The real Qwen3 projection patch schedule is:

| phase | input dim | output dim | patches |
| --- | ---: | ---: | ---: |
| Q | 4096 | 4096 | 64 |
| K | 4096 | 1024 | 16 |
| V | 4096 | 1024 | 16 |
| O | 4096 | 4096 | 64 |
| up | 4096 | 12288 | 192 |
| gate | 4096 | 12288 | 192 |
| down | 12288 | 4096 | 64 |

Total: `608` MyLM-sized patches. A hidden-dim patch is `64 output rows x
4096 K = 0x28000 bytes`; the down patch is `64 output rows x 12288 K =
0x78000 bytes`.

## Retained Chain

- `25_mylm_edge_bd_ring`: reproduces the edge/KV row1 static BD-ring skeleton
  for rounded context lengths `L=17/31/32/128`.
- `26_kv_edge_aux_reshape`: calibrates packet14/15 current variants,
  2048-dword half-plane streams, and the 17-dword sideband path.
- `38_full_attention_fabric`: proves the full `32Q/8KV` attention resource map
  fits with row1 KV reshape/fanout.
- `39_projected_current_write_full_attention`: adds NPU-produced Q/current K/V
  and current cache writeback before the full attention fabric.
- `48_main16_fullk_q4nx_phase_replay`: proves main16 can replay full-K Q4NX
  projection phases on the same physical tiles.
- `52_fullk_edge_slice_replay_q4nx_o_phase`: proves a full-K O projection can
  consume edge-replayed 256-bf16 slices without a host-visible attention vector.
- `53_full_layer_phase_chain_contract`: strongest end-to-end contract so far.
  It runs seven layer-shaped phases over the main16/edge16 fabric with compact
  phase records. It deliberately uses deterministic replay and simplified
  `K=4096` phases.
- `54_real_qwen_patch_schedule`: maps the real Qwen3 projection dimensions to
  the 608-patch phase schedule.
- `55_mylm_linked_bd_chain`: proves a linked shim-BD chain can be patched once
  and consumed by a static row1 ping-pong ring.
- `56_mylm_linked_qwen_schedule`: combines linked descriptors with the real
  Qwen3 phase/block schedule.
- `57_mylm_exact_patch_manifest`: fixes the exact MyLM patch unit and validates
  patch order, offsets, phase ranges, and BD slot alternation.
- `58_mylm_patch_pair_row1_split`: proves two 64-row patches can be split by
  row1 into four 32-row compute streams for one main column.
- `59_mylm_exact_nblock_projection`: proves exact MyLM-sized patches can feed
  the full 512-row main16 projection block. It still keeps full patches resident
  in row1, so it is not the final memory shape.
- `61_mylm_chunk_ring_slot_locks`: fixes the output route/collector mismatch,
  gives ping and pong independent slot locks, and proves that a full `0x28000`
  patch descriptor can stream through a small row1 Q4NX chunk ring on real NPU.
- `62_mylm_full_nblock_chunk_ring`: current strongest row1/main16 evidence.
  It scales the small-ring schedule to four main columns and four rows per
  column with two exact `0x28000` patches per column. The split-channel variant
  passes on real NPU; the same-channel variant times out because memtile even
  DMA channels can only use BD `0..23`, while the 32-phase input chain crosses
  into odd-channel BD slots `24..47`.
- `63_mylm_packetized_patch_phase`: current strongest patch-handoff evidence.
  It keeps one linked host `MM2S ch0` patch queue, packetizes each shim BD, and
  lets packet routing send patch0/patch1 to legal memtile `S2MM ch0/ch1` BD
  banks. It passes at full main16 scale, so this patch phase does not require
  direct CDO/transaction generation.
- `64_mylm_attention_to_o_direct_handoff`: current strongest attention-to-O
  handoff evidence. Edge tiles replay deterministic attention-result slices
  directly into the full main16 O phase while O weights arrive through exp63's
  packetized patch queue and row1 small chunk rings. It passes on real NPU with
  no host-visible attention drain.
- `65_mylm_fused_layer_engine_v0`: current fused-engine baseline. It combines
  the real Qwen3 seven-phase schedule, exact 608-patch MyLM manifest,
  packetized patch descriptors, row1 small chunk rings, and direct
  deterministic attention-result-to-O handoff. It passes on real NPU using
  high-level MLIR-AIE generated descriptors.

## Deleted Experiments

The older experiments `11`, `13`, `18..24`, `27..37`, `40..47`, `50`, and
`60` were removed. They were valuable while exploring, but keeping them made the
evidence chain harder to read. The retained chain preserves the current useful
results: KV ring shape, current/cache writeback, full attention resource map,
main16 phase replay, edge-to-main O replay, real patch schedule, and exact-patch
row1 split.

## Remaining Questions

1. How do we express MyLM-style direct CDO ownership in our codebase?
   Exp62 shows why a literal same-channel row1 BD chain fails; exp63 shows that
   packetized linked descriptors can keep one host logical queue while routing
   phases to legal row1 BD banks. Exp65 shows the v0 fused layer descriptor
   program is still expressible through high-level MLIR-AIE. Direct
   CDO/transaction work should now be reserved for phase ownership that
   packetized descriptors cannot express or for reducing descriptor-program
   overhead.
2. What is the production attention ABI?
   We still need to replace deterministic 17-dword records with real Q/current
   K/V, rounded KV scan, online softmax state, and weighted V. Exp64 proves the
   return-to-O stream shape for deterministic attention results, but not the
   production attention producer.
3. Where should attention state live?
   The exact placement of running max/sum/output accumulators and the exact
   shape-A/shape-B or packet14/15 consumer relationship remain inferred rather
   than implemented as final code.
4. How do Q/K/V outputs hand off to edge attention and return to main16 O
   without debug drains?
   Exp64 proves the return-to-O half without an attention debug drain, and
   exp65 proves that return path composes with the full seven-phase schedule.
   The remaining gap is connecting exp39-style real Q/K/V attention output to
   that same O handoff ABI in one continuous schedule.
5. How do we replace contract kernels with fast kernels?
   The retained experiments mostly use deterministic or scalar kernels to prove
   dataflow. The final engine needs high-throughput online Q4NX kernels with
   sustained DMA/compute overlap.
6. How do layer-level runlist and lm_head integrate?
   Once one fused layer is correct, the final runtime still needs layer-to-layer
   submission, per-layer weights/cache/state, and an lm_head path.
