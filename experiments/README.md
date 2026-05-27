# Retained Fused-Layer Experiments

Only the experiments that still describe the current MyLM/FastFlowLM
reproduction path are kept here.

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

Earlier syntax probes, one-off diagnostics, and superseded failure
reproductions were removed so the directory stays focused on the implementation
path that is still useful.
