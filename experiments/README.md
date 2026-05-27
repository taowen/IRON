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
- `60_mylm_chunk_ring_projection`: current frontier. It changes row1 residency
  from full patches to a small Q4NX chunk ring. It compiles, but the hardware
  run times out, pointing at unresolved same-channel BD/lock phase ordering.
- `resource_plan_audit.py`: computes the first-principles resource plan and
  checks that retained experiments still cover the intended milestones.

## Deleted Experiments

The older experiments `11`, `13`, `18..24`, `27..37`, `40..47`, and `50` were
removed. They were valuable while exploring, but keeping them made the evidence
chain harder to read. The retained chain preserves the current useful results:
KV ring shape, current/cache writeback, full attention resource map, main16
phase replay, edge-to-main O replay, real patch schedule, and exact-patch row1
split.

## Remaining Questions

1. How do we express MyLM-style direct CDO ownership in our codebase?
   High-level MLIR-AIE/IRON routing can prove contracts, but exp60 shows that
   the exact same-channel descriptor and chunk-ring schedule may require us to
   own stream-switch, BD, and lock programming more directly.
2. What is the exact row1 small chunk-ring schedule for a full `0x28000` patch?
   Exp59 proves the exact patch ABI with full-patch residency. Exp60 proves the
   intended smaller residency compiles but not yet that the lock/DMA phase order
   is correct.
3. What is the production attention ABI?
   We still need to replace deterministic 17-dword records with real Q/current
   K/V, rounded KV scan, online softmax state, weighted V, and the return stream
   to O projection.
4. Where should attention state live?
   The exact placement of running max/sum/output accumulators and the exact
   shape-A/shape-B or packet14/15 consumer relationship remain inferred rather
   than implemented as final code.
5. How do Q/K/V outputs hand off to edge attention and return to main16 O
   without debug drains?
   The contracts have proven pieces of this, but the full production ABI still
   needs one continuous schedule.
6. How do we replace contract kernels with fast kernels?
   The retained experiments mostly use deterministic or scalar kernels to prove
   dataflow. The final engine needs high-throughput online Q4NX kernels with
   sustained DMA/compute overlap.
7. How do layer-level runlist and lm_head integrate?
   Once one fused layer is correct, the final runtime still needs layer-to-layer
   submission, per-layer weights/cache/state, and an lm_head path.
