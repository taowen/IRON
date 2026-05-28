# MLIR-AIE Recipes

这些 recipe 是从 `experiments` 里已经能真机运行的 NPU 实验整理出来的小例子。
每个 recipe 都是独立目录，包含 `generate.py`、`reference.py`、`run_npu.py` 和
tile kernel 源码，可以直接编译并在 NPU 上跑。

## Recipes

- [packetized-shim-memtile-split](packetized-shim-memtile-split/README.md)
  - 学 packetized shim descriptor 如何把一个 host queue 分到 row1 两个物理输入通道。
- [row1-fanout-packet-gather](row1-fanout-packet-gather/README.md)
  - 学 32Q/8KV attention fabric 的 row1 fanout、worker packet gather 和多长度验证。
- [edge-to-main-direct-handoff](edge-to-main-direct-handoff/README.md)
  - 学 edge 侧 producer 如何不落 host，直接喂给 main16 projection。
- [packet-to-multicast-bridge](packet-to-multicast-bridge/README.md)
  - 学 packet source 如何经过 ping/pong bridge fanout 到多个 worker。
- [hierarchical-record-compaction](hierarchical-record-compaction/README.md)
  - 学多 producer record 如何逐级压缩成一个紧凑 packet，再喂给单个 consumer。
- [slot-locked-streaming-ring](slot-locked-streaming-ring/README.md)
  - 学大 host descriptor 如何用小 slot ring 和 lock token 流式喂给 worker。

## 运行约定

在仓库根目录运行：

```bash
.venv/bin/python recipes/packetized-shim-memtile-split/run_npu.py
.venv/bin/python recipes/row1-fanout-packet-gather/run_npu.py
.venv/bin/python recipes/edge-to-main-direct-handoff/run_npu.py
.venv/bin/python recipes/packet-to-multicast-bridge/run_npu.py
.venv/bin/python recipes/hierarchical-record-compaction/run_npu.py
.venv/bin/python recipes/slot-locked-streaming-ring/run_npu.py
```

这些例子会在各自目录下生成 `build/`。真机运行需要本机已有 MLIR-AIE、Peano 和 XRT
环境；`run_npu.py` 会把 `/var/opt/xilinx/xrt/bin` 加进 `PATH`。
