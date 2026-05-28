# Packetized Shim Memtile Split

来源：`experiments/63_mylm_packetized_patch_phase`

这个 recipe 展示一个关键 MLIR-AIE pattern：host 只 push 一个 linked `MM2S ch0`
queue，但每个 shim BD 带不同 packet id，让 row1 memtile 收到后分到两个合法的
物理 S2MM channel。

## 学什么

- `aiex.npu.writebd enable_packet=1` 如何给 host-to-NPU descriptor 打 packet id。
- `aie.packet_flow` 如何把 patch0/patch1 路由到 row1 的不同 DMA channel。
- row1 小 chunk ring 如何用独立 ping/pong lock 避免 same-channel BD bank timeout。
- main16 worker 如何同时消费 activation stream 和 Q4NX weight chunk stream。

## 关键路径

```text
host linked queue
  -> shim packet16/17
  -> row1 patch0 S2MM ch0, patch1 S2MM ch1
  -> row1 MM2S rows 0..3
  -> main16 S2MM weight input
```

输出只用于校验：

```text
main16 output packet -> row1 gather -> shim S2MM -> host output
```

## 运行

```bash
# 小规模单列版本，适合先验证工具链和 NPU 可用性。
PACKET_SPLIT_VARIANT=onecol .venv/bin/python recipes/packetized-shim-memtile-split/run_npu.py

# 完整 main16 版本。
PACKET_SPLIT_VARIANT=full .venv/bin/python recipes/packetized-shim-memtile-split/run_npu.py
```

默认是 `full`。通过后会打印 NPU time、record headers 和 CPU/NPU 数值误差。

## 文件

- `generate.py`: 生成 MLIR-AIE。
- `reference.py`: 生成 Q4NX patch layout 和 CPU reference。
- `projection_nblock.cc`: main/edge tile kernel。
- `run_npu.py`: 编译 kernel、生成 xclbin/insts、运行并校验输出。
