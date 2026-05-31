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
