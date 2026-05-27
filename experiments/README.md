# Retained Fused-Layer Experiments

Only the experiments that still describe the current MyLM/FastFlowLM
reproduction path are kept here.

## Stage Summary: Toward Our Fused Layer Engine

The current goal is not to clone MyLM/FastFlowLM instruction-for-instruction.
MyLM is used as a hardware-mapping reference. The implementation target is our
own Qwen3 fused layer engine.

The retained experiments now answer the main physical-resource question: a
full decode layer can be shaped around the same `main16 + edge16` fabric instead
of allocating separate compute tiles for Q/K/V/O/FFN. The main projection fabric
is `c2..c5/r2..r5`; it time-reuses the same 16 compute tiles across projection
phases. The edge/aux fabric on `c0/c1/c6/c7` handles attention-side replay and
phase handoff. Runtime should be layer-level: it patches coarse BO descriptors
and starts the engine, while phase sequencing is enforced inside the static
BD/lock/worker dataflow.

The strongest current milestone is `53_full_layer_phase_chain_contract`. It
runs on real NPU and connects seven layer-shaped phases
`Q,K,V,O,gate,up,down` through one continuous per-column Q4NX weight stream,
edge replay, compact `17 dword` phase records, main16 full-K Q4NX phases,
core-local SwiGLU, and final debug drain. This proves the long-lifetime
phase-chain skeleton can fit and advance without deadlock.

The remaining work is no longer a broad resource feasibility question. It is
implementation work:

- Replace deterministic edge replay with real attention: Q/current K/V,
  KV-cache scan, online softmax, and weighted V.
- Replace the simplified `K=4096` phase set with the real Qwen3 patch schedule:
  `Q,K,V,O,up,gate,down = 64,16,16,64,192,192,64` patches.
- Replace contract kernels with high-throughput online Q4NX microkernels and
  keep DMA/compute overlapped through static row1 rings.
- Keep MyLM's exact packet/sideband details as useful diagnostics, not as a
  blocking compatibility target.

- `resource_plan_audit.py`: computes the first-principles Qwen3 fused-layer
  resource plan and compares it against retained experiments plus MyLM
  reverse-engineering notes/BD CSVs.
- `11_chunked_q4nx`: minimal Q4NX chunk accumulation contract.
- `13_projection_column`: low-level projection column with memtile broadcast,
  writebd runtime, and static BD rings.
- `18_integrated_ffn_contract`: retained FFN dataflow contract with projection
  phase reuse, core-local SwiGLU, memtile gather, and down projection.
- `19_current_write_kv_attention`: current K/V writeback followed by rounded
  KV-cache history scan.
- `20_online_softmax_kv_attention`: online softmax attention over the same
  current-write KV boundary.
- `21_single_layer_decode_contract`: local query, current-write KV attention,
  and epilogue in one decode contract.
- `22_q4nx_query_attention`: real Q4NX query projection feeding KV attention
  without a host-side query input.
- `23_qkv_current_attention`: generated Q/K/V, current K/V cache writeback,
  phase transition from weight chunks to KV history, and cache-backed attention.
- `24_mylm_fabric_boundary`: deprecated negative evidence. It shows the raw
  history single-edge-worker shortcut is not MyLM's edge/KV boundary.
- `25_mylm_edge_bd_ring`: current path. Encodes the MyLM edge/KV static BD-ring
  contract and runs a real-NPU checksum skeleton for L=17/31/32/128.
- `26_kv_edge_aux_reshape`: current selector/sideband calibration. It runs on
  real NPU and verifies packet14/15 current variants, row1-routed 2048-dword
  history input, non-packet 17-dword sideband, and a c6r2 -> c6r3
  shape-B-like handoff with host-visible checksums.
- `27_shape_ab_attention_contract`: single 16-token shape-A/shape-B attention
  contract. It uses packet14 current, row1 K/V history rings, a 17-word compact
  sideband, and verifies tail-masked weighted-value output on real NPU.
- `28_multitile_shape_ab_attention`: multi-tile online attention contract. It
  sends current once, streams multiple K/V history tiles through row1 rings, and
  verifies that a 17-word sideband can drive running softmax/value accumulation.
- `29_qkv_phase_reuse_projection`: Q/K/V projection phase-reuse contract. It
  keeps one hidden activation resident while the same physical projection tiles
  consume Q, then K, then V full-K Q4NX weight streams. This proves the
  descriptor-count fix for `K=4096`: one phase-sized runtime BD feeds a static
  row1 chunking ring.
- `30_fullk_qkv_attention`: full-K Q/K/V projection-to-attention contract. It
  combines phase-sized Q/K/V weight streams with current K/V cache writeback,
  then scans rounded KV history tiles and verifies online attention on real NPU
  for non-16-aligned and multi-tile context lengths.
- `31_gqa_full_attention`: full GQA attention-group contract. It verifies a
  full `512 x 4096` Q phase, `128 x 4096` K/V phases, current K/V writeback,
  rounded KV scan, and online GQA attention for one 4Q:1KV group on real NPU.
- `32_parallel_gqa_attention`: parallel GQA attention-head contract. It splits
  the four Q heads of one GQA group across four columns, lets column 0 produce
  the shared current K/V cache entry, and verifies that all columns can scan the
  updated cache and assemble the same 512-dim output faster than exp31.
- `33_row1_kv_reshape_attention`: row1 KV-cache reshape contract. It loads
  token-major rounded K/V history planes into row1 memtile storage, uses static
  BD dimensions to emit dim-group-major streams, and verifies tail-masked
  `4Q:1KV` online attention on real NPU for L=17/31/32/79.
- `34_parallel_row1_reshape_attention`: parallel row1 KV-cache reshape
  contract. It combines exp33's row1 dim-group-major K/V reshape with
  exp32-style four-column GQA head parallelism and verifies tail-masked
  one-head online attention per column on real NPU for L=17/31/32/79. It still
  duplicates K/V DDR reads per column, so it is not the MyLM one-read
  edge/fanout path yet.
- `35_one_read_kv_fanout_attention`: one-read KV fanout contract. It moves
  K/V history scan to a single central row1 memtile, reshapes token-major
  history with static BD dimensions, fans the reshaped K/V streams to four
  attention workers, and verifies the same four-head output as exp34 on real
  NPU for L=17/31/32/79. Runtime K/V patches drop from
  `4 * 2 * num_tiles` to `2 * num_tiles`.
- `36_current_write_one_read_fanout_attention`: current-write plus one-read
  fanout contract. It writes current K/V into the cache BO on NPU, syncs that
  writeback, then scans the updated cache once through the exp35 fanout path.
  Real-NPU runs verify bit-exact current cache writeback and four-head
  attention output for L=17/31/32/79.
- `37_projected_one_read_fanout_attention`: projected one-read fanout attention
  contract. It computes Q heads and shared current K/V from hidden + Q4NX
  weights on NPU, writes current K/V into the cache BO, then scans the updated
  cache once through a central row1 fanout. It also records the key negative
  result: two static producers cannot both target one worker DMA port even if
  they are phase-separated; the phase handoff must be represented by one row1
  source feeding hidden first and K/V history later.
- `38_full_attention_fabric`: full attention-fabric contract. It scales the
  exp37 attention side to `32Q/8KV`: eight columns for eight KV groups, four
  compute rows per column for the four Q heads sharing each KV group, one row1
  K/V read per group, row1 reshape/fanout, packet gather, and per-group drain.
  Real-NPU runs pass for L=17/31/32/79, proving the full attention resource map
  fits and routes before reattaching projection/current-write.
- `39_projected_current_write_full_attention`: projected current-write full
  attention contract. It removes exp38's host-provided query/current tensors:
  every worker projects Q from hidden on NPU, each KV group's row-0 worker
  projects current K/V on NPU, current K/V is written into the cache BO, then
  the full `32Q/8KV` row1 fanout/packet-gather attention fabric scans the
  updated cache. Real-NPU runs pass for L=17/31/32/79. The projection kernel is
  deterministic and scalar, so this proves the fused-layer dataflow boundary
  rather than final Q4NX projection performance.
- `40_projection_record_handoff_abi`: projection-record handoff ABI contract.
  It validates the `17 = 1 header + 16 dword payload` record hypothesis, routes
  four compute-tile records through row1 into a `65/64` column replay shape, and
  checks the derived `257`, `2049`, and `6144` ladder sizes against a CPU
  reference. This targets the missing MyLM phase-handoff ABI rather than
  attention or FFN math.
- `41_edge_shape_ab_handoff_probe`: edge shape-A -> shape-B handoff probe. It
  proves a 17-dword compact state can be produced by one compute tile, sent
  directly to another compute tile without a host-visible state BO, and consumed
  with a separate history stream to produce exact host-checked output.
- `42_attention_output_to_o_projection`: attention-output to O-projection
  handoff probe. It proves a full 512-dword attention result can flow directly
  from an edge attention tile into an O-projection tile and be consumed with an
  O-weight stream, with no host-visible attention output BO.
- `43_attention_o_ffn_closed_tail`: closed post-attention layer-tail probe. It
  connects edge attention, O projection, and FFN tail in one internal dataflow:
  O output enters FFN without DDR, gate/up/SwiGLU stay tile-local, and only the
  final layer-tail output drains to host.
- `44_projected_current_attention_closed_ffn`: projected current-write
  attention closed-FFN probe. It projects query/current K/V from hidden, writes
  current K/V into the KV cache BO, syncs that writeback before scanning the
  updated cache, and then keeps attention/O/FFN intermediates internal until
  final output drain.
- `45_main16_edge_return_o_phase`: MyLM-style main16/edge resource-reuse
  contract. It routes 16 main-fabric projection records into edge/aux tiles,
  returns per-row attention shards back to the same main16 physical tiles, runs
  the O phase there, and only drains the final gathered output. This replaces
  the invalid "add separate O/FFN workers after full attention" model.
- `46_main16_edge_return_ffn_tail`: main16 edge-return FFN-tail contract. It
  extends exp45 by keeping O output, gate, up, and SwiGLU tile-local on the
  same main16 fabric, then packet-gathers only the final FFN-tail output.
- `47_main16_multiphase_ffn_replay`: main16 multi-phase FFN replay contract. It
  splits O, gate, up, and down weights into four lock-ordered slices on the
  same main tile S2MM input channel, proving phase replay over the same
  physical main16 resources rather than separate host-visible operators.
- `48_main16_fullk_q4nx_phase_replay`: main16 full-K Q4NX phase replay
  contract. It replaces exp47's toy phase weights with four full-K Q4NX
  streams, using phase-sized runtime descriptors feeding a static row1
  fat-chunk ring and the same main16 physical compute fabric.
- `50_edge_slice_replay_q4nx_o_phase`: edge-slice replay contract. It tightens
  the discarded exp49 idea by making the 512-dword edge return a stream of four 128-dword slices,
  each paired with one `32 x 256` Q4NX O-weight chunk. This avoids modeling the
  edge path as a full activation tensor producer and matches the MyLM-style
  phase handoff shape we need before extending O into the layer tail.
- `52_fullk_edge_slice_replay_q4nx_o_phase`: full-K O contract over the
  corrected edge-slice ABI. It streams four 512-dword edge shards as sixteen
  128-dword slices and pairs those slices with sixteen Q4NX `32 x 256` chunks
  on the same main16 physical projection fabric, proving `K=4096` O projection
  without a host-visible attention vector. The final host drain is a debug
  record stream with `(group,row)` headers because MyLM's real layer handoff is
  not a row1 packet-gather ordering contract.
- `53_full_layer_phase_chain_contract`: full layer-shaped phase-chain contract.
  It connects Q/K/V/O/gate/up/down as seven sequential full-K Q4NX phases on
  the same main16/edge16 physical fabric. Each phase consumes sixteen
  256-bf16 activation slices from edge and sixteen Q4NX chunks from a continuous
  per-column weight stream; compact 17-dword records gate phase replay. Real
  NPU runs pass, proving the long-lifetime phase chain fits and advances. It
  deliberately uses deterministic sideband records and `K=4096` for every phase,
  so it does not yet prove MyLM's exact sideband semantics or Qwen3's expanded
  FFN dimensions.

Earlier syntax probes, one-off diagnostics, and superseded failure
reproductions were removed so the directory stays focused on the implementation
path that is still useful.
