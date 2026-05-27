# Experiment 67: MyLM Global QKV/O Layout Contract

This experiment keeps exp66's full fused-layer schedule and replaces the local
edge-tile attention diagnostic with a canonical global Q/K/V and O-layout
contract.

The point is narrow: prove that the existing packetized full-layer engine can
consume an attention producer whose output is indexed as the real
`Attn[32][128]` vector and sliced exactly the way the O phase expects. This is
not yet the physical MyLM all-to-all Q/K/V collector.

## Contract

- Main projection fabric remains `c2..c5/r2..r5`.
- Edge/aux fabric remains `c0/c1/c6/c7`, rows `2..5`.
- The phase order remains `Q, K, V, O, UP, GATE, DOWN`.
- The global weight BO remains the exact `608` MyLM/Qwen patch manifest.
- Q/K/V phase records still carry the producing main tile's `32` bf16 payload,
  so the compact phase handoff remains active.
- The O producer indexes a canonical global attention vector:

  ```text
  Attn[32][128] flattened as global_idx = head * 128 + dim
  O chunk c covers global_idx = c * 256 .. c * 256 + 255
  O chunk c therefore packs heads 2*c and 2*c + 1
  ```

- The corresponding projection-output ownership is:

  ```text
  Q block b, main group g, row r, lane l -> head = b * 4 + g, dim = r * 32 + l
  K/V block b, main group g, row r, lane l -> kv_head = b * 4 + g, dim = r * 32 + l
  GQA mapping: q_head h consumes kv_head floor(h / 4)
  ```

- Every edge tile computes the same value for a given `global_idx`. That makes
  the O input layout independent of which edge tile replays the 256-bf16 slice.
- There is no debug drain between attention and O.

## Schedule

| phase | input dim | output dim | 512-row blocks | K chunks | patches |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q | 4096 | 4096 | 8 | 16 | 64 |
| K | 4096 | 1024 | 2 | 16 | 16 |
| V | 4096 | 1024 | 2 | 16 | 16 |
| O | 4096 | 4096 | 8 | 16 | 64 |
| UP | 4096 | 12288 | 24 | 16 | 192 |
| GATE | 4096 | 12288 | 24 | 16 | 192 |
| DOWN | 12288 | 4096 | 8 | 48 | 64 |

Total: `608` exact MyLM patches.

## What This Proves

- The fused layer engine can run with O phase input slices laid out as the real
  flattened `32 heads x 128 dim` attention result.
- The Q/K/V projection-output coordinate system is now explicit and executable
  in both the AIE kernel and CPU reference.
- The direct attention-result-to-O handoff from exp64/65/66 still works when
  the producer is target-independent instead of local-edge dependent.

## What This Does Not Prove

- Physical all-to-all Q/K/V distribution from main16 into the edge/aux
  attention fabric.
- Production current K/V cache writeback and rounded KV-cache DMA scan.
- Production online attention microkernel performance.
- Direct CDO/transaction ownership.

## Run

```bash
.venv/bin/python experiments/67_mylm_global_qkv_o_layout/run_npu.py
```
