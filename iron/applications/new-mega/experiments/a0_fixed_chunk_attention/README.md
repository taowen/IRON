<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# A0 Fixed-Chunk Decode Attention

Status: accepted.

## Question

Can one fixed IRON artifact run decode attention for multiple token positions
without RTP, dynamic TAP offsets, dynamic BD patching, or per-position
recompilation?

## Hypothesis

Yes. For decode attention, make `max_seq_len`, `head_dim`, and `chunk_size`
compile-time constants. Always DMA the full preallocated KV cache and full mask
through fixed TAPs. Encode the live position only as mask values:

```text
mask[i] = 1 when i <= position
mask[i] = 0 when i > position
```

The Worker always processes the same number of chunks. Fully masked chunks do
no numerical work beyond acquire/release and mask checks, and online softmax
makes chunked processing equivalent to full softmax.

## Minimal Design

```text
max_seq_len = 256
head_dim = 128
chunk_size = 64
num_chunks = 4

Q:                one 128-element vector
packed K/V/mask: fixed 4 chunks, each chunk is K[64,128] || V[64,128] || mask[64]
O:                one 128-element vector
```

The first draft used four input streams (`Q`, `K`, `V`, `mask`) and failed
resource allocation because one compute tile had too many input DMA channels.
The accepted version packs `K/V/mask` into one chunk stream and keeps `Q` as the
second input stream.

The AIE Worker keeps:

```text
state[0] = running_max
state[1] = running_sum
acc[128] = running unnormalized output
```

The C kernel computes:

```text
for each fixed chunk:
  score = q @ k_chunk.T / sqrt(head_dim)
  ignore rows with mask=0
  update running_max/running_sum/acc with online softmax

out = acc / running_sum
```

## Acceptance

```text
same xclbin and runtime .bin are reused for all tested positions
NPU output matches CPU attention reference for partial and full chunks
no position value appears in DesignGenerator arguments
no Runtime fill/drain TAP changes between tested positions
```

## Run

```bash
source /opt/xilinx/xrt/setup.sh
. .venv/bin/activate
python iron/applications/new-mega/experiments/a0_fixed_chunk_attention/run.py
```

## Result

Accepted on NPU2.

Command:

```bash
source /opt/xilinx/xrt/setup.sh
. .venv/bin/activate
PYTHONUNBUFFERED=1 python -X faulthandler \
  iron/applications/new-mega/experiments/a0_fixed_chunk_attention/run.py \
  --build-dir build_new_mega_a0_fixed_chunk_attention
```

Observed:

```text
artifact_reused_for_all_positions: True
position=0   valid_chunks=1 max_abs=0.000000 cpu_match=True
position=26  valid_chunks=1 max_abs=0.000244 cpu_match=True
position=63  valid_chunks=1 max_abs=0.000069 cpu_match=True
position=64  valid_chunks=2 max_abs=0.000122 cpu_match=True
position=127 valid_chunks=2 max_abs=0.000065 cpu_match=True
position=200 valid_chunks=4 max_abs=0.000046 cpu_match=True
position=255 valid_chunks=4 max_abs=0.000021 cpu_match=True
decision: accepted
```

Additional max-sequence sweep:

```text
max_seq_len=512  num_chunks=8   accepted, positions 0/26/255/256/511 matched
max_seq_len=1024 num_chunks=16  accepted, positions 0/26/255/512/1023 matched
max_seq_len=2048 num_chunks=32  accepted, positions 0/26/255/1024/2047 matched
max_seq_len=4096 num_chunks=64  accepted, positions 0/26/255/2048/4095 matched
```

Conclusion:

```text
Position-specific attention artifacts are avoidable for decode attention when
KV/mask movement uses fixed max-sequence TAPs and the live position is encoded
only in runtime mask data.
```

This does not solve current-K/V cache writeback by itself. It does show that
the attention read side does not need RTP, dynamic BD patching, or a new xclbin
per position.
