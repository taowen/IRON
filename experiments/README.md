# Retained Fused-Layer Experiments

Only the experiments that still describe the current MyLM/FastFlowLM
reproduction path are kept here.

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

Earlier syntax probes, one-off diagnostics, and superseded failure
reproductions were removed so the directory stays focused on the implementation
path that is still useful.
