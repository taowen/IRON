# Retained Fused-Layer Experiments

This directory keeps only experiments that still support the current
MyLM-style fused-layer direction. The compact dataflow model lives in
[xdna-programming-guide.md](xdna-programming-guide.md); individual directories
are for reproduction and narrow evidence.

## Current Direction

The target is our own full-layer fused Qwen3 decode engine. MyLM is the
hardware-mapping reference, not an instruction-for-instruction compatibility
target.

The stable high-level model is:

- main16 `c2..c5/r2..r5` is time-reused for Q/K/V/O/up/gate/down projection;
- edge/aux `c0/c1/c6/c7` handles current Q/K/V, KV history, attention state,
  compact records, and return to O;
- runtime patches descriptors/RTPs and starts a layer run, rather than exposing
  a host-visible operator chain.

## Kept Evidence

Patch queue and projection schedule:

- `54_real_qwen_patch_schedule`: real Qwen3 608-patch schedule.
- `57_mylm_exact_patch_manifest`: exact MyLM patch unit and phase ranges.
- `58_mylm_patch_pair_row1_split`: patch-pair to four 32-row streams.
- `59_mylm_exact_nblock_projection`: full main16 projection with full-patch
  row1 residency.
- `61_mylm_chunk_ring_slot_locks`, `62_mylm_full_nblock_chunk_ring`: row1
  small-ring lock and same-channel timeout evidence.
- `63_mylm_packetized_patch_phase`: retained patch transport result; one
  logical host queue lands in legal full-main16 row1 BD banks.

Attention and O handoff:

- `38_full_attention_fabric`: `32Q/8KV` attention resource map fits.
- `39_projected_current_write_full_attention`: NPU-produced current Q/K/V and
  current cache writeback baseline.
- `64_mylm_attention_to_o_direct_handoff`: deterministic attention result
  streams to O without a host-visible drain.
- `65_mylm_fused_layer_engine_v0`: fused-engine skeleton with exact patch queue.
- `66_mylm_real_attention_o_phase`: local online-softmax/weighted-V producer
  feeds O on real NPU.
- `67_mylm_global_qkv_o_layout`: `Attn[32][128]` to O `16 x 256` layout.
- `82_mylm_attention_o_bridge`: packet2 return bridge to O chunks.

MyLM physical reverse evidence:

- `70_mylm_stream_switch_physical_path_replay`: CDO/stream-switch replay base.
- `78_mylm_row0_current_kv_writeback`: packet14/15 are current K/V writeback.
- `79_mylm_row1_history_split`: K history to Shape-A, V history to Shape-B.
- `80_mylm_shape_ab_return_phase`: Shape-A hidden carrier, Shape-B 512-dword
  return.
- `81_mylm_shape_b_hidden_payload`: compact carrier capacity and fp32
  accumulator capacity.
- `83_mylm_layer_boundary_contract`: no DDR-visible O/residual/FFN
  intermediates.
- `84_mylm_shape_ab_carrier_lock`: Shape-A owns L7/L5/L4; Shape-B uses
  north-neighbor lock immediates.
- `87_mylm_shape_ab_carrier_block_order`: Shape-A publishes carrier ready L7
  then L5; Shape-B splits the carrier into `base[0x100]` and
  `scalar[0x40]` through call-slot setup.
- `85_mylm_main16_phase_record`: `17 dword` record shape and full-Q split.
- `86_mylm_dispatcher_packet8_aux`: main16 dispatcher/body shape, packet8 as
  unpacketized compact/aux path, and header source from body input `r0`.
- `88_mylm_aux_compact_record_roles`: `c1r2` is a full-vector
  RMSNorm/residual-style aux compute station; `c6r2` is a compact
  record/indexed-format station on the packet8-side route.
- `89_mylm_fullvector_ffn_dataflow`: `c1r2` is the hidden-input/RMSNorm/final
  output full-vector station; `c6r2` is the SwiGLU slice station; `c6r1`
  gathers the 12288-bf16 FFN intermediate and publishes packet0 for down.
- `90_mylm_main16_activation_bridge`: packet2/O and packet0/down reuse the same
  c1r1 DMA4-to-DMA1 256-dword bridge, then feed the main16 128-dword activation
  ring as 16 O chunks or 48 down chunks.
- `91_mylm_c1r2_phase_order`: runtime writes the `c1r2` mode flag and L6 start
  gate; `c1r2.bd3` releases `+12/+48/+1` full-vector packet0 replays for
  Q/K/V, up/gate, and final hidden output.
- `92_mylm_main16_phase_control`: main16 body replay counts and compact-record
  control words are scheduler-known: `12 x 0x1` for Q/K/V, `8 x 0x4` for O,
  `48 x 0x8` for up/gate, and `8 x 0x4` for down.
- `93_mylm_shape_ab_carrier_lane_usage`: Shape-B consumes `base[0x100]` as
  four 0x40 head-pair-sized blocks in read order `0x00,0x40,0xc0,0x80`; the
  scalar block is online-softmax scale/normalization state, not proven raw
  max/sum lanes.
- `94_mylm_shape_a_carrier_producer_layout`: Shape-A helper `0x650` produces
  `base[0x100]` as eight 0x20 stores, so the carrier base record quantum is one
  query head x 16 token bf16 weights; Shape-B consumes adjacent records as
  0x40 head-pair blocks.
- `95_mylm_c6r2_swiglu_input_layout`: `c6r2` consumes each 512-dword SwiGLU
  input as `up[0x400]` followed by `gate[0x400]`, matching MyLM's physical
  patch order `up` before `gate`.
- `96_mylm_upgate_c6r2_compact_route`: 16 main16 up/gate records are compacted
  by row1 into one 257-dword c1r1 packet; `c6r2 DMA_0` drops two packet headers
  and receives two 256-dword payloads as one 512-dword `up+gate` input.

AIE2P kernel/toolchain learning:

- `aie_intrinsics_api_probe`: broad Peano/AIE API/source-assembly instruction
  shape and NPU smoke probes for Q4NX hot-loop candidates.
- `97_aie2p_callable_asm_contract`: focused callable source-assembly contract:
  AIE2P C ABI, vector preserve-all limits, wrapper relocation shape, and
  BF16/float accumulator storeback forms.
- `98_aie2p_float_accum_inplace_asm`: callable source assembly can load,
  update, and store a tile-local FP32 accumulator without relying on C++ vector
  state.
- `99_aie2p_q4_direct_target_stages`: aligned AIE2P source-assembly ZOL can
  run one full exact 256-dim Q4NX lane without source-level branch timeout or
  manual full unroll overflow.
- `100_mylm_q4nx_body_contract`: executable target contract for the next
  MyLM-style Q4NX body: group-sum producer, `528` dynamic `vmac.f`, zero hot
  `vst`, and the scheduled dequant path with `64` `vunpack`, `64` `vups.4x`,
  and `136` hot-loop `vconv.bf16.fp32` slots.
- `101_mylm_q4nx_group_sum_asm`: first runnable source-assembly numeric gate
  for the MyLM-style group-sum route. It matches the intended `lc=8`,
  `vextbcst.16`, signed `vmac.f`, no-hot-store shape, but fails the synthetic
  output gate; this proves opcode-count matching is insufficient without
  MyLM's exact operand/register layout.
- `102_aie2p_vmac_operand_layout`: focused NPU calibration for
  `vmac.f #0x33c` operands: scalar broadcast, `vextbcst.16` lanes, group-sum
  correction, and a 32-lane sum, used to debug exp101 before any production
  migration.
- `103_mylm_q4nx_register_flow`: register-level annotation table for MyLM
  hot-loop `0x260..0x52a`, making defs/uses, alias families, and software
  pipeline dependencies explicit before writing more Q4NX assembly.
- `104_mylm_q4nx_pipeline_schedule`: compact full-hot-loop schedule summary:
  all eight MyLM Q4NX groups, cross-group register-family carry, and
  `vextbcst.16` consumer distance distribution.
- `105_aie2p_dependency_latency`: real-NPU latency probe for core assembly
  dependencies. It measures the first passing gaps for `vextbcst.16 -> vmac.f`,
  `vbcst.16 -> vmac.f`, `vldb -> vextbcst.16`, and `vmac.f -> vst`.
- `106_mylm_q4nx_steady_state_template`: extracts the actual repeatable
  MyLM Q4NX template. Groups 1..5 are text-identical steady-state bodies,
  group6 is a pre-drain variant, and group0/group7 are explicit fill/drain
  templates.
- `107_mylm_q4nx_template_codegen`: replays the complete MyLM
  `0x260..0x1850` hot loop from four templates and exact-matches all `1532`
  instruction slots, proving the schedule is now generator-shaped.
- `108_qwen3_q4nx_group_sum_contract`: compares IRON's exact Q4NX formula and
  the MyLM-style group-sum formula on real Qwen3-8B weights. Layer0 synthetic
  hidden stays within `1e-2`, but layer35 real dumped hidden diverges badly,
  so group-sum is not a silent drop-in for the current reference.
- `109_qwen3_q4nx_centered_dequant_contract`: tests the stronger
  centered-dequant form `bf16(bf16(q*scale)+zero)-zero` plus a zero group-sum
  correction. This removes exp108's layer35 systematic drift
  (`hidden_out max_abs 44.0 -> 0.0625`), but a contracted one-MAC zero
  correction still leaves two hidden values above `1e-2`; exact production
  assembly needs this centered path and must decide whether to keep direct
  per-dim zero accumulation order.
- `110_qwen3_q4nx_zero_order_contract`: separates centered coefficient
  construction from zero accumulation order. On layer35, recomposing
  `centered+zero` before the MAC is exact (`hidden_out max_abs=0`), while both
  contracted group-sum zero and split zero accumulator paths drift. Exact
  parity therefore requires zero in the coefficient before MAC; the MyLM-like
  `32+1 MAC/group` route is a deliberate numerical tradeoff.
- `111_q4nx_dynamic_instruction_cost`: expands active IRON and MyLM Q4NX
  source loops by hardware-loop trip count. Active exact already has the right
  `512` dynamic MAC count; the gap is coefficient construction and control
  traffic (`vconv.bf16.fp32 1280 vs 272`, `vconv.fp32.bf16 768 vs 0`,
  `crupsmode 128 vs 0`, `nop 3088 vs 216`).
- `112_q4nx_exact_vs_mylm_contract_floor`: computes the instruction-cost
  floor for exact current Q4NX semantics versus the MyLM-like group-correction
  contract. Exact parity can still reduce redundant traffic, but its lower
  bound still needs `512` vector multiplies and `1536` conversion operations
  per chunk; the MyLM-like contract needs `16` vector multiplies and `272`
  conversions.
- `113_mylm_q4nx_mac_operand_trace`: traces latest visible producers for every
  MyLM Q4NX `vmac.f` operand. In the canonical steady group1, `6/66` vector
  operands and `2/33` accumulator sources are carried across the group
  boundary, confirming that the next assembly generator needs explicit
  boundary state, not a self-contained 33-MAC group macro.
- `114_mylm_q4nx_half_register_alias`: refines exp113 to vector-half and
  accumulator-quadrant cells. In steady group1, `25/66` vector operands are
  mixed-half values and `9/66` have cross-group cells. This proves whole-vector
  `xN` producer tracking is still too coarse; the next generator needs
  half-register boundary state and a later `vmac.f` config-lane decoder.
- `115_mylm_q4nx_operand_graph`: turns the half-register trace into a
  generator-shaped operand graph with Markdown and JSON outputs. It proves the
  steady-to-steady data-cell boundary signature is stable for groups `2..6`
  (`5121c8604a8461c4`), so a generator can model one steady transition instead
  of eight independent groups.
- `116_mylm_q4nx_graph_annotated_codegen`: consumes the exp115 JSON graph and
  emits a graph-annotated steady-state assembly include. Stripping generated
  comments recovers MyLM group1 exactly (`189` slots, `33` `vmac.f`, zero
  missing graph records), proving the operand graph is now a usable codegen
  input rather than a Markdown-only reverse artifact.
- `117_mylm_q4nx_full_graph_codegen`: expands the complete MyLM
  `fill + steady*5 + pre_drain + drain` hot loop from templates and annotates
  the five steady sections from the exp115 graph. Stripping comments exact
  matches MyLM `0x260..0x1850` (`1532` slots, same hash, `165` annotated
  steady `vmac.f`), giving the next modified-body experiment a full-loop
  codegen entry point.
- `118_mylm_q4nx_cell_liveness`: turns the MyLM hot-loop disassembly into a
  full half-register/accumulator-quadrant liveness table. Each group boundary
  carries `27` data cells and `12` pointer/scalar cells, proving the fast body
  is a fixed register-residency software pipeline that a generator must model
  explicitly before production assembly changes.
- `119_mylm_q4nx_full_operand_graph`: extends exp115 from one steady group to
  all `264` MyLM Q4NX `vmac.f` slots. It confirms the full group MAC shape
  `28,33,33,33,33,33,33,38`, shows group1 is a fill-to-steady transition, and
  verifies groups `2..5` share a stable steady-to-steady operand signature.
- `120_mylm_q4nx_generator_contract`: merges exp118 liveness with exp119 MAC
  graph into a sectionized generator contract. The emitted include splits the
  MyLM hot loop into `fill`, `fill_to_steady`, `steady_to_steady`, `pre_drain`,
  and `drain` macros, then re-expands to the exact original `1532` slots with
  `264` group MACs.
- `121_q4nx_modified_body_numeric_gate`: uses the exp120 section contract to
  compare the two numerical body branches. `exact_recompose_coeff` is exact in
  both nominal and stress synthetic gates; MyLM-like group correction preserves
  `264/528` MAC shape but shows stress mismatches, so it needs real token
  acceptance before production.
- `122_q4nx_exact_section_rewrite_contract`: converts exp121's exact decision
  into the next generator contract. It preserves exp120 live-state sections,
  removes one zero-correction MAC per logical group, and changes the target
  body from `264` static MACs to exact `256` static MACs.
- `123_mylm_q4nx_group0_instruction_semantics`: annotates MyLM
  `0x260..0x52a` at instruction-slot granularity. Group0 has `192` slots,
  `28` `vmac.f`, `32` `vextbcst.16`, no `vst`, and produces `23` data plus
  `4` control cells that live into group1. This is the first full fill-section
  def/use table for learning the assembly before production rewrites.
- `124_mylm_q4nx_steady_transition_semantics`: annotates MyLM group2 with the
  group0-1 live state already applied. It proves the reusable steady transition
  is `189` slots, `33` `vmac.f`, `32` `vextbcst.16`, `8` `vups.4x`, no `vst`,
  and preserves the same `27` data plus `12` control boundary cells while
  replacing `24` previous-group producers with group2 producers.
- `125_main16_qkv_nocall_scheduler_contract`: quantifies the current
  per-chunk helper-call boundary. Q/K/V costs `192` helper calls per tile
  (`3072` across main16), full layer costs `1472` helper calls per tile
  (`23552` across main16), and SAVE/RESTORE alone is `4992` QKV or `38272`
  full-layer instruction slots per tile before counting accumulator memory
  roundtrips. A first linked nocall scheduler satisfied the static no-helper
  contract but timed out on `full-layer-qkv-prefix`, so it is not active code.
- `126_mylm_main16_whole_core_contract`: converts "move toward MyLM" into the
  next executable boundary. It keeps the active C++ scheduler plus
  `q4nx_chunk_accum_asm_zol` path as the correctness baseline, rejects the
  partial QKV-only nocall route, and defines the next target as one generated
  main16 role program with MyLM-style Q4 body, phase bodies, dispatcher, and
  IRON-compatible compact headers.
- `127_main16_whole_program_scaffold`: generates and compiles the first
  whole-main16 program scaffold. It defines stable entry, dispatcher, shared
  Q4 body, Q/K/V/O/upgate/down body symbols, the linked C ABI
  (`p0..p5` buffers, `r0..r2` tile metadata, and `r3` phase limit), lock ownership, and the exact
  IRON header schedule that differs from MyLM (`Q/K/V = packet10/11/12`,
  O=13, up/gate=14, down=15). Its manifest now covers the full phase schedule:
  `1472` dynamic Q4 calls and `76` compact records per main tile. The record
  emitter scaffold now writes the real IRON header, converts 32 FP32 accumulator
  lanes to the 16-dword BF16 record payload, and clears that FP32 accumulator.
  It also checks that the emitter does not touch the caller's phase-loop state
  registers. Each phase body now only configures a header run and calls the
  shared Q4 run body; there are `6` static phase-to-Q4 calls for `1472` dynamic
  chunk executions. The shared Q4 run body owns the record loop, chunk loop,
  DMA lock protocol, ping/pong selection, exact chunk body, and record emitter
  call. Its exact chunk body is emitted from the same
  `qwen3-layer/tools/main16_q4nx_asm_lib.py` source template as the active
  `q4nx_chunk_accum_asm_zol` body, so exp127 no longer carries a second Q4NX
  hot-body copy. The embedded chunk body uses the no-save form (`64 vmac.f`,
  `64 vextbcst.16`, `2 vst`, `0` save/restore slots). The dispatcher now uses
  the same single-entry phase-limit shape as active `q4nx_main16_layer_scheduler`:
  `3=qkv`, `4=qkvo`, `6=upgate`, and `7=full`. The remaining whole-program gap
  is replacing that exact dequant body with a MyLM-density scheduled body. This
  is the replacement target for future generated Q4 body work, not active code.
  The active single-scheduler refactor has passed `full-layer-qkv-prefix token31`
  and `qwen3-8b-decode-layer token31` on NPU; full decode measured `28.479 ms`
  with `final_hidden_out max_abs=0.0078125`. The scaffold also now has a real
  nested-call return-address contract: every non-leaf generated function saves
  and restores `lr` with a 64-byte frame, while the record emitter is checked as
  a leaf.

## Folded Experiments

Deleted because their useful conclusions are covered above:

- `25`, `26`: early edge/KV ring and selector skeletons, folded into
  exp70/78-84.
- `48`, `52`, `53`: early main16/O/full-layer milestones, folded into
  exp65/66/83/85.
- `55`, `56`: linked-descriptor precursors, folded into exp63.
- `68`, `69`: early reverse probes, folded into exp70/78-86.
- `71..77`: Q-window, packet-mask, and shim-return probes, folded into
  exp70/82/85.
- Older removed experiments: `11`, `13`, `18..24`, `27..37`, `40..47`, `50`,
  and `60`.

## Remaining Questions

1. Calibrate the remaining Shape-A/B compact carrier value order: local head
   order of the eight 0x20 base records, token lane order inside one record,
   and online-softmax scalar lane semantics in `scalar[0x40]`.
2. Calibrate register-level `c1r2` ping/pong pointer order and value layout.
3. Decode optional bit-level meaning of main16 headers `0x1/0x4/0x8`; the
   scheduler-critical values and replay counts are known.
4. Calibrate the exact element order inside each 16-dword main16 payload and the
   exact up/gate N-block pairing order into `c6r2`.
5. Replace diagnostic kernels with production RMSNorm, Q/K norm, RoPE, online
   softmax, SwiGLU, down, and Q4NX kernels.
6. Integrate layer-to-layer runtime submission, per-layer weights/KV state, and
   lm_head.
