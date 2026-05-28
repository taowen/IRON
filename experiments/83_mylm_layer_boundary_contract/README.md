# 83_mylm_layer_boundary_contract

Audits the MyLM fused layer runtime boundary from the decoded transaction, BD
table, and main16 disassembly.

The goal is to prove whether post-attention residual, post-attention RMSNorm,
up/gate, SwiGLU, and down intermediates leave the layer engine through DDR. They
do not: the transaction only exposes hidden input/output, RMSNorm/RoPE side
inputs, KV cache writeback/scan, and weight patches. The remaining post-O and
FFN state therefore has to live inside static AIE dataflow.
