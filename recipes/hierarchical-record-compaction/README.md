# Hierarchical Record Compaction

这个 recipe 展示一个通用数据流模式：

```text
many producers -> column compactors -> global compactor -> paired consumer
```

适合复用在“很多 worker 产出带 header 的小 record，需要逐级去掉重复 header 并拼成
紧凑 packet”的场景。例子使用两路独立 record stream，最终在 consumer 侧按
low/high 两个半区接收。

## Pattern

- 每个 producer 产出两条 17-dword record。
- 每个 column compactor 把 4 个 producer record 合并成 65-dword compact packet。
- global compactor 把 4 个 column compact packet 合并成 257-dword global packet。
- consumer 丢弃 global header，接收两个 256-dword 半区，并验证 paired layout。

## Run

```bash
.venv/bin/python recipes/hierarchical-record-compaction/run_npu.py
```

## Files

- `contract.py`: 这个 recipe 的尺寸常量。
- `generate.py`: 生成多 producer、column/global compactor 和 consumer 的 MLIR-AIE。
- `reference.py`: CPU reference，定义 record compact 和 paired consumer layout。
- `dataflow_kernels.cc`: producer record 和 consumer merge kernel。
- `run_npu.py`: 编译、运行并校验。
