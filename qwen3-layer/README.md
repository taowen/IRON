# qwen3-layer

This directory implements the qwen3-layer contract described in
`experiments/qwen3-dataflow.md` and contains a runnable NPU integration
backend.

The current implementation is the qwen3-dataflow physical skeleton:

- `c1r2` full-vector station with `2049`-dword packet0 replay contract.
- `c1r1` shared activation bridge for packet2/O, packet0 replay, and packet1/down.
- `c1r3` Q/K norm + RoPE postprocess station with packet8/9 current KV routes.
- `c6r1` Q fanout, attention return gather, and FFN intermediate gather.
- `c6r2` up/gate SwiGLU slice station.
- Main16 projection workers and row1 compact routes.
- Shape-A/B carrier and return windows.

## Files

- `contract.py`: Qwen3 layer constants and ABI checks.
- `dataflow.py`: typed qwen3-dataflow graph.
- `generate.py`: MLIR-AIE physical skeleton generator from the target graph.
- `check_contract.py`: integration check for contract, graph, and generated MLIR.
- `emit_mlir.py`: writes `qwen3-layer/build/qwen3_dataflow.mlir`.
- `mlir_utils.py`: shared MLIR-AIE BD, lock, queue, and runtime sequence
  helpers used by runnable cases.
- `physical_contract.py`: executable channel ownership checks for the
  compact-only full-layer cases and the MyLM-aligned Q4NX dual-input ABI.
- `weight_stream.py`: shared row1 Q4NX patch-ring generator for host/shim
  weight ingress into main16 DMA1.
- `projection_schedule.py`: single source for Q/K/V body record counts and
  O/upgate/down Q4NX tail weight chunk bases.
- `compact_dataflow.py`: shared row1/c1r1 compact gather, bridge, hub, and
  MyLM-aligned row1 S2MM4/5 weight fanout generator used by the frontier.
- `attention_dataflow.py`: shared Shape-A/B tile placement, hub BD ownership,
  KV output BDs, and packet2 attention return hub used by active attention
  generators.
- `qkv_compact_reference.py`: shared Q/K/V/O compact record layout helpers for
  active attention integration references.
- `qkv_compact_dataflow.py`: shared four-phase Q/K/V/O compact bridge used by
  current-K/V attention integration boundaries; the old standalone qkv-shape
  generator/reference has been removed.
- `q4nx_reference.py`: shared Q4NX chunk reference math used by integration
  checks.
- `npu_build.py`: shared MLIR, xclbin, and NPU runtime helpers. It scans
  generated MLIR `link_with` attributes and compiles the required role objects,
  so runners do not duplicate kernel-object ownership.
- `main_projection_q4nx.cc`: main16 Q/K/V/O/up/gate/down Q4NX projection,
  flush, and record emit kernels.
- `edge_attention.cc`: Shape-A/B edge attention kernels for KV scan, online
  softmax, weighted V, and accumulator merge.
- `postprocess_qkv.cc`: c1r3 Q/K norm, RoPE/layout pack, and current K/V
  writeback payload kernels.
- `full_vector_station.cc`: c1r2 hidden/O compact replay, full-vector station,
  and final compact/output helpers.
- `swiglu.cc`: c6r2 up/gate SwiGLU kernels that produce down activations.
- `debug_contract.cc`: temporary deterministic bridge and smoke-test contract
  kernels. This is not part of the production full-layer path.
- `record_format.h` / `qwen3_constants.h`: shared header layout and constants.
- `cases/`: modular case registry and wrappers for executable integration
  boundaries, including the Q/K/V -> Shape-A/B -> O compact path.
- `run_npu.py`: thin CLI dispatcher for real NPU integration cases.

## Check

```bash
.venv/bin/python qwen3-layer/check_contract.py
```

## Run On NPU

```bash
.venv/bin/python qwen3-layer/run_npu.py --check-only
.venv/bin/python qwen3-layer/run_npu.py --build-only
.venv/bin/python qwen3-layer/run_npu.py
.venv/bin/python qwen3-layer/run_npu.py --case currentkv-kvscan-attention-kv16-o-bridge
.venv/bin/python qwen3-layer/run_npu.py --case currentkv-kvscan-attention-kv16-o-bridge --current-token 91
.venv/bin/python qwen3-layer/run_npu.py --case currentkv-kvscan-attention-kv16-o-bridge --current-token 91 --patch-from-token 127
.venv/bin/python qwen3-layer/run_npu.py --case currentkv-kvscan-attention-kv16-o-bridge --current-token 91 --patch-from-token 1007
.venv/bin/python qwen3-layer/run_npu.py --case currentkv-full-layer-q4nx-down-bridge
.venv/bin/python qwen3-layer/run_npu.py --case currentkv-full-layer-q4nx-down-bridge --current-token 91 --patch-from-token 1007
.venv/bin/python qwen3-layer/run_npu.py --case q4nx-qkv-body-post-bridge
```

This runner is intentionally an integration boundary, not a tiny unit test. The
default case is `currentkv-full-layer-q4nx-down-bridge`, the current
qwen3-dataflow fused-layer frontier. Historical migration cases for the old
608-patch weight-stream oracle, deterministic full-layer tail, and standalone
down/SwiGLU/Q4NX bridges have been retired from the runnable registry. Their
useful pieces now live in shared generators: `weight_stream.py` owns row1
S2MM4/5 weight ingress and row1 MM2S fanout, `compact_dataflow.py` owns the
frontier compact bridge plus row1 weight-stream composition, and
`physical_contract.py` validates that the frontier does not regress to the old
row1 S2MM0/1 weight route.
The `currentkv-full-layer-q4nx-down-bridge` case now starts from a host hidden
vector replayed by c1r2, runs Q4NX Q/K/V on main16, converts the bf16 Q/K/V
body into the current kv16 attention ABI in c1r3, and writes current K/V through
packet8/9 before scanning rounded KV cache blocks. The attention result returns
as a bf16 packet2 payload, main16 consumes it in Q4NX O, c1r2 replays the O
compact as bf16 full-vector packet0 payloads, main16 runs Q4NX up/gate with
row1 S2MM4/5 weights on DMA1, c6r2 consumes bf16-input SwiGLU, and main16
finally runs Q4NX down with the same DMA0/DMA1 activation/weight ABI. It still
drains the 257-dword down compact for tolerant validation. Its instruction patch path
has been audited with the larger Q4NX weight stream: `%weights` stays on arg2,
output stays on arg3, token1007 -> token91 patched `design.bin` is
word-identical to a direct token91 recompile, and both the default token127 run
and patched token91 run pass on NPU. One resolved failure mode here was the old
c1r2 bit-hash replay: bf16 O compact values matched numerically, but hashing
their bit patterns amplified one-ulp differences into final mismatches. The
current bridge uses numeric bf16 compact expansion with a bounded fixed scale.
The runnable registry now keeps only active boundaries that still feed the
current frontier. `--check-only` validates generated MLIR structure, and
`--build-only` verifies routing, core compilation, instruction generation, PDI,
and xclbin generation. Older bridge, shape, and single-phase smoke cases are no
longer registry entries because the full frontier now covers their physical
routes with production role objects; their useful constraints were moved into
shared generators and `physical_contract.py`.
The `currentkv-kvscan-attention-kv16-o-bridge` case adds the next decode
boundary. Main16 emits Q/K/V compacts, c1r3 expands Q and emits current K/V as
packet8/9, shim DMA writes the current-token slices into block-major K and V
cache BOs with the two-BD even/odd scatter, the runtime syncs those writes, and
split shim K/V scan streams read the rounded 16-token context blocks from the
same BOs before Shape-A/B consumes the windows.
Shape-A emits one carrier per block and masks the rounded tail block with a
tail-token RTP, Shape-B merges the block states online, then packet2 returns a
single attention output to the O phase. The host history buffers deliberately
contain poisoned current-token slots, so the output only matches when packet8
and packet9 update the cache before scan. The runner also reads the K/V cache
BOs back and validates the overwritten slots directly, which separates
writeback failures from later attention/O failures.
The `q4nx-qkv-body-post-bridge` case is the kept Q/K/V handoff diagnostic. It
replaces the deterministic Q/K/V producer: host hidden enters the c1r2-position
full-vector station, 12 packet0 replays go through c1r1 into main16 DMA0, Q/K/V
weights enter through row1 S2MM4/5 and main16 DMA1, and c1r3 drains Q plus
current K/V layout to host. The smaller Q-only diagnostic is no longer a
runnable registry entry because the QKV boundary covers the same initial hidden
replay and weight timing with the full Q/K/V postprocess ABI.
`--current-token` selects a single decode schedule: current-token RTP value,
current-write byte offset, cache BO size, Shape-A/B block count, Shape-A tail
valid-token count, scan BD `iteration_size`, and queue `repeat_count` all come
from the same `DecodeSchedule`. Shape-A/B block count and Shape-A tail count are
RTPs protected by runtime-start locks, not core constants. `--patch-from-token`
compiles the xclbin/PDI for a larger cache-capacity schedule and patches only
`design.bin` to the target token. Both token127 -> token91 and token1007 ->
token91 have passed on NPU. token1007 is the current AIEX descriptor ceiling:
63 rounded 16-token blocks, covering a 1024-token cache capacity. The
token127 -> token91 patched instruction stream is word-identical to a freshly
compiled token91 stream.
The `currentkv-full-layer-q4nx-down-bridge` case is the current full-layer
closed loop. It keeps hidden replay, Q4NX Q/K/V body, bf16-to-kv16 attention
ABI conversion, packet8/9 current-K/V writeback, rounded KV scan, Shape-A/B
kv16 attention, packet2 O handoff, bf16 c1r2 replay, Q4NX O, Q4NX up/gate,
bf16-input c6r2 SwiGLU, packet1 down handoff, Q4NX down, and final compact
drain in one real NPU run. It also carries the NPU constraints that matter for
future work: c1r2 packet0 replay owns MM2S1 BD1, c1r1 S2MM3 needs high-bank
compact BDs, c6r2 consumes payload halves without compact headers, and row1
compact gather must stay disjoint from row1 S2MM4/5 Q4NX weight ingress.

The up/gate bridge intentionally uses one S2MM channel per row in each row1
column compact tile. A single S2MM channel with multiple packet BDs is not a
valid replacement for ordered packet-id scheduling when independent main tiles
produce rows concurrently; the memtile consumes arrivals in stream order, so
row0 gate can be captured by the row1 up BD. Splitting rows across channels
keeps the packet handoff deterministic while still using legal memtile BD banks
for AIE2p.

The streaming KV scan case keeps two similar hardware constraints explicit.
First, the four row1 KV output channels cannot share one generic "payload full"
token when the shim is slicing K0/V0/K1/V1 independently; each slot needs its
own empty/full lock, otherwise an MM2S channel can consume the readiness token
for the wrong slot and the run can timeout. Second, generated runtime
descriptors must not patch an address argument that the xclbin kernel metadata
does not expose. The kvscan case therefore uses four BO arguments
`Q, left_kv_cache, right_kv_cache, output`, and `mlir_utils.py` validates the
maximum `arg_idx`.
The current-K/V cache writeback keeps two generator rules explicit. AIERT
requires each DMA BD block that uses locks to carry both acquire and release
operations; the earlier row1 bypass merge failed when one BD only acquired a
lock. AIE2p memtile BDs are also banked by channel parity: even DMA channels
use BD 0..23, odd DMA channels use BD 24+. The current writeback path uses two
linked shim S2MM BDs per side instead of eight head-slice BDs. c1r3 emits the
current K/V stream as even dwords for all heads followed by odd dwords for all
heads; BD14 writes the even positions and BD15 writes the odd positions with
`d0_size=32`, `d0_stride=1`, `d1_size=8`, and `d1_stride=1023`.

The scan side is now split by K/V plane to avoid the next descriptor wall. A
single-channel scan needs one runtime BD for each K0/V0/K1/V1 slice per block,
so four blocks would consume BD0..15 before current writeback has any legal BD
left. The passing split design sends K0/K1 over shim MM2S channel 0 and V0/V1
over shim MM2S channel 1. Each rounded block therefore uses two scan BDs instead
of four. The first static split version reached seven rounded blocks with
BD0..6 for K-side scans, BD7..13 for V-side scans, and BD14/15 for current
writeback. The current version keeps only one K scan BD and one V scan BD per
side: `buffer_length=4096`, `iteration_size=<rounded_blocks>`,
`iteration_stride=8191`, and `push_queue repeat_count=<rounded_blocks - 1>`.
`iteration_size` alone only configures the BD address generator; without the
queue repeat count the hardware executes one 4096-dword segment and the
downstream block loop waits forever. Row1 mirrors the split channels with S2MM
channel 0 for K and channel 1 for V; the V input uses high-bank memtile BDs
28/29, while the existing output BDs remain 2/24/4/26. The current token is
patched through `aiex.npu.rtp_write` into a c1r3-local `post_current_token`
buffer consumed by the `postprocess_qkv.cc` role.
That RTP write must be paired with a runtime-start lock. Without the lock,
main16 can produce Q/K/V compacts and c1r3 can enter
`currentkv_postprocess_payload` before the runtime sequence writes the RTP; the
observed failure was a clean token0 current-slot writeback while the descriptor
path and cache scatter were otherwise correct. The current-token RTP is consumed
by the c1r3 postprocess role, not by a shared bridge object. The old token15
literal was the root cause of the first current-slot cache writeback mismatch.
`mlir_utils.py` validates the lock-balance, memtile-BD-bank, scan/write
BD-disjointness, `npu.writebd` ID/field ranges, push-queue repeat count, and
RTP-before-runtime-lock ordering before aiecc reaches CDO generation.

The 2026-05-29 scale probe found the descriptor boundary precisely. AIECC
rejects `aiex.npu.writebd` IDs above 15. Reusing overlapping shim BD IDs after
`npu.sync` compiled but timed out on NPU. A single 2D current-write BD also
failed: `d0_stride=1` completed but only filled every other dword, while
`d0_stride=0` compiled but timed out. The first passing three-block design used
the explicit two-BD even/odd scatter described above. The next passing
seven-block design compressed scan descriptors by splitting K and V onto two
shim/memtile channels. The current scheduled design adds iterated scan BDs
plus queue repeats, Shape block-count RTPs, and runtime-start locks. It now
emits 447 runtime instructions and has passed token127 with eight rounded
blocks, token91 with six rounded blocks and a non-tail current slot,
token127 xclbin/PDI patched to token91, and token1007 cache-capacity PDI
patched to token91 without aiecc recompilation. This proves
the high-level AIEX path can express descriptor reuse for the KV scan, as long
as the BD iteration fields and queue repeat count are programmed together.

The Shape-A/B attention math stays in explicit fixed point for now. Shape-A
stores packed 16-bit Q12 exp weights and int32 block max/sum values; Shape-B
keeps int32 accumulators and applies the same Q12 scale during online merge.
This is closer to production online softmax than the old stair-step contract,
but still needs calibration against the Qwen3 bf16/fp32 kernel instead of
depending on compiler lowering of wide integer math.
The same rule now applies to c1r2 and main16 FFN contract kernels: generated
tile code should prefer bounded signed int32 loops, shifts, and masks. A probe
that used an unsigned LCG, unsigned modulo, and 64-bit integer sqrt produced a
stable but wrong scale on NPU before c6r2. The passing c1r2 path uses a simple
signed compact-to-lane mix, a bounded int32 sqrt, 64 int32 projection
accumulators, and Q8 s16-pair packing before fixed-point SwiGLU.

## Emit

```bash
.venv/bin/python qwen3-layer/emit_mlir.py
```

The executable cases prove the current hardware contracts. The active registry
keeps the Q4NX Q/K/V body handoff diagnostic, the current-K/V cache writeback
plus KV-scan/attention boundary, and `currentkv-full-layer-q4nx-down-bridge` as
the closed-loop frontier. The old
`current` case name and the 608-patch weight-stream oracle were retired because
they implied a second fused-layer backend. The useful pieces of that path are
now explicit shared code: host Q4NX weight BO layout, packetized patch
descriptors, row1 S2MM4/5 ingress, and AIE Q4NX chunk accumulation.
The frontier now uses main16 Q4NX kernels for Q/K/V/O/up/gate/down over the
same row1 S2MM4/5 -> main16 DMA1 weight stream, including token1007-capacity
PDI reuse through patched token91 instructions. Remaining numerical work is to
calibrate the kv16 attention approximation, c1r2 RMSNorm/replay, and c6r2
SwiGLU against production Qwen3 kernels.
In IRON MLIR, S2MM and MM2S channel ids are directional
namespaces: the current main16 record output is emitted as `MM2S1`, which is
role-equivalent to MyLM's third main16 record stream even though the exact
printed id is not `DMA2`. Remaining work is to keep the production Qwen3 cache
in this block-major scan layout at runtime and finish numeric calibration for
attention, c1r2 RMSNorm/replay, and SwiGLU.
The full-layer generator now has an explicit compact phase trace shared by main
record DMA, row1 compact DMA, and bridge compact DMA. Each trace item carries a
unique MLIR label, logical phase, record slot, packet id, and payload slice. The
validated trace is now the body-level `q,k,v,o,upgate,down` schedule. `upgate`
is a 48-record long body, not two single-record phases: main16 writes
interleaved up/gate records into an `upgate_records` buffer, row1/c1r1 use
2D BD strides to build `48 x 65` and `48 x 257` compact layouts, and the final
c1r1 output BD skips every compact header so c6r2 consumes 48 payload halves as
24 distinct adjacent pairs. This avoids both the invalid 53-phase materialized
trace and the standalone `up -> gate -> up` ring that cannot transition to
down.
The c1r2 bridge also captures the lock rule needed for that port: a memtile DMA
BD block can release only one lock. Reusable row/group fan-in therefore uses a
stage-level counting empty lock, acquired once by each input row/group and
released once by the compact output BD with count 4. Per-row empty releases
compiled into multiple release ops and were rejected by MLIR-AIE; not releasing
the empty tokens caused the second up/gate pair to timeout on NPU.
For body-level scatter/gather, the MLIR-AIE dimension syntax must use named
struct parameters, for example `[<size = 48, stride = 257>, <size = 256,
stride = 1>]`; the shorter `[<48, 257>, ...]` form is shown in old dialect
comments but is rejected by the current parser.
Current K/V intentionally uses packet8/9 rather than packet14/15: full-layer
execution already reserves packet14/15 for FFN up/gate and down compact traffic.
Do not wire this by materializing both 8192-dword KV sides inside the c1r3
postprocess tile; that would exceed the intended compute-tile local-memory
budget. The full-layer version needs a streaming KV path through shim/row1/edge
tiles, matching the qwen3-dataflow design.
