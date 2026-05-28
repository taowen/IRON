# Row1 Fanout Packet Gather

来源：`experiments/38_full_attention_fabric`

这个 recipe 展示 full `32Q/8KV` attention fabric 如何映射到 `8 columns x 4 rows`。
它保留 host-provided query/KV cache，重点学习 row1 对 K/V 的 reshape/fanout，
以及 32 个 worker 输出如何 packet gather 回 row1。

## 学什么

- 每个 KV group 一列，4 个 compute row 对应 4 个 Q heads。
- row1 memtile 只读取一次 K/V history，然后 fanout 给 4 个 worker。
- worker 输入是静态相位顺序：query，K tile，V tile。
- worker 输出用 packet gather 回 row1，再由 row1 合并成每组 output block。

## 关键路径

```text
query host BO -> row1 query slice -> 4 workers
KV host BO    -> row1 reshape K/V -> 4 workers
worker output -> packet gather -> row1 output block -> host output
```

这个 recipe 不包含 Q/K/V projection、current K/V writeback、RoPE、下游 projection 或 FFN。
它只回答 attention fabric 的物理资源和路由问题。

## 运行

```bash
.venv/bin/python recipes/row1-fanout-packet-gather/run_npu.py
```

脚本会编译一次 kernel，并依次跑 `L=17,31,32,79` 四个上下文长度。每个 case 都会
和 CPU reference 比较。

## 文件

- `generate.py`: 按 context length 生成 MLIR-AIE。
- `reference.py`: 生成 query/KV case 和 CPU attention reference。
- `reshape_attention.cc`: worker tile kernel。
- `run_npu.py`: 编译、运行多长度 case 并校验输出。
