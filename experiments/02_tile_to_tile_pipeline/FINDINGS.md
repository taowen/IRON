# Experiment 02: Tile-to-Tile Pipeline — Findings

## Result: IRON CAN express tile-to-tile intermediate dataflow without DDR

Generated MLIR (line 13):
```
aie.objectfifo @normed(%tile_0_2, {%tile_0_3, %tile_0_4}, 2) : !aie.objectfifo<memref<128xbf16>>
```

This proves:
- **One norm tile produces → two projection tiles consume**, via ObjectFifo
- **The intermediate (`normed`) is NOT a DDR argument** — it does not appear in `runtime_sequence`
- **1-to-N broadcast** from a single producer to multiple consumers works natively
- **Heterogeneous workers** — norm worker (tile 0,2) and projection workers (tiles 0,3/0,4) have different bodies in the same graph

## What this proves vs what the fused layer engine needs

| Fused engine pattern | Proved? | Evidence |
|---------------------|---------|----------|
| Intermediate stays on-chip (not DDR) | **Yes** | `@normed` is tile-to-tile, not in `runtime_sequence` |
| One producer broadcasts to N consumers | **Yes** | `%tile_0_2 -> {%tile_0_3, %tile_0_4}` |
| Heterogeneous worker roles in one graph | **Yes** | 3 cores with different bodies |
| DDR only at layer boundary (input/output) | **Yes** | Only `hidden_in`, `W_*`, `C_*` connect to shim |

## What this does NOT prove (honest gaps)

1. **Memtile involvement for large intermediates**: The `@normed` fifo here is 128 bf16 = 256 bytes,
   small enough to fit in local memory or pass via stream. A real hidden vector is 4096 bf16 = 8KB.
   At that size, the ObjectFifo would route through memtile (row 1). Whether the placer/router
   handles this correctly at scale is not tested.

2. **Replay/reuse of the same data**: This design produces `normed` once and broadcasts to 2 consumers.
   In the fused engine, `hidden_norm` is replayed to Q/K/V projections sequentially (the same data
   consumed 3 times by the same set of tiles in 3 different phases). That's a different pattern:
   it requires either depth>1 ping-pong with replay, or re-sending the same data.
   This experiment broadcasts once to parallel consumers, not serial replay.

3. **Chaining multiple stages without DDR**: This only has Norm→GEMV. The fused engine has:
   Norm→Q→RoPE→Attention→O→Norm2→up/gate→SwiGLU→down. Can we chain 3+ heterogeneous
   stages all tile-to-tile in one graph? Resource pressure (tiles, streams, BDs) may hit limits.

4. **The norm worker does nothing computationally**: It just acquires and releases. A real RMSNorm
   kernel would need to read the input, compute variance, normalize, and write to the normed fifo.
   The data path works structurally but we haven't proven it works with real kernel linkage.

5. **No multi-phase (Q then K then V) on the SAME tiles with this intermediate**:
   Experiment 01 showed multi-phase on same tiles. This shows tile-to-tile.
   The combined pattern (norm → fanout to projection tiles, then same tiles do Q/K/V/O/up/gate/down
   phases sequentially, with norm output replayed each time) is not yet tested.

6. **Scale**: 2 projection tiles, K=128. Real layer: 16 projection tiles, K=4096, 7 projections.
   The DMA/BD/lock/stream-switch resource budget at full scale may break.

## Conclusion

The IRON MLIR primitives (`ObjectFifo`, `Worker`, `Program`) CAN express the fundamental
tile-to-tile dataflow pattern needed by fused layer engines. The barriers are:

1. **The `FusedMLIROperator` high-level API does NOT use this pattern** — it forces DDR boundaries
   between every operator. But you can bypass it by writing a monolithic design.

2. **Combining multi-phase (Exp 01) + tile-to-tile (Exp 02)** into one coherent design at
   production scale (16 tiles, 7 projections, attention fabric, KV cache) is a significant
   engineering challenge, even though each piece is individually expressible.

3. **The unsolved hard problems** are not "can IRON express X" but:
   - BD/task count explosion with 608 weight patches
   - KV cache scan with 16-token tile rounding + tail mask
   - Attention worker fabric fitting in edge/aux tiles
   - Stream switch routing at full 8-column scale
   These are resource/scale problems, not expressibility problems.
