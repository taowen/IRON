# Packet To Multicast Bridge

这个 recipe 展示一个通用数据流模式：

```text
packet source -> bridge ping/pong buffer -> multicast worker inputs -> summaries
```

适合复用在“一个上游 tile 发布整块 payload，下游多个 worker 以固定 chunk
粒度重复消费”的场景。例子里提供两个 payload 大小，验证同一条桥可以承载不同
packet id 和不同 fanout 次数。

## Pattern

- 上游 hub 用 packet flow 把整块 payload 发到 bridge。
- bridge 用 256-dword ping/pong buffer 接收，再切成 128-dword worker chunk。
- 16 个 worker 都接收同一份 chunk stream，并在本地做 summary。
- worker summary 再 packet 回 memtile，最后由 shim 写回 host 校验。

## Run

```bash
.venv/bin/python recipes/packet-to-multicast-bridge/run_npu.py
.venv/bin/python recipes/packet-to-multicast-bridge/run_npu.py --case fanout-6144d
```

## Files

- `generate.py`: 生成 packet source、bridge、multicast workers 和 output gather 的 MLIR-AIE。
- `reference.py`: 生成输入 payload 和 CPU expected summary。
- `dataflow_kernels.cc`: worker summary kernel。
- `run_npu.py`: 编译、运行并校验。
