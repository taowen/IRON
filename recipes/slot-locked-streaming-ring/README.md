# Slot-Locked Streaming Ring

这个 recipe 展示一个通用数据流模式：

```text
large host descriptor -> small ping/pong slots -> slot locks -> worker streams
```

适合复用在“host descriptor 很大，但中间 memtile 只保留少量 streaming slot”的场景。
例子里的 payload 是 packed numeric chunk，重点不是具体计算，而是 ping/pong slot 的
lock 计数、BD channel 和 worker 输入流配合。

## Pattern

- host 只提交大块 descriptor。
- 中间 memtile 不做 full-payload residency，只维护两个小 slot。
- ping 和 pong slot 使用独立 empty/full lock。
- producer 对 full lock release 多 token，让多个消费者各 acquire 一次后才能复用 slot。
- worker 同时消费 activation stream 和 packed payload stream。

## Run

```bash
.venv/bin/python recipes/slot-locked-streaming-ring/run_npu.py
```

## Files

- `generate.py`: 生成 host descriptor、slot-locked ring、worker streams 和 output gather 的 MLIR-AIE。
- `reference.py`: 生成 packed payload 和 CPU reference。
- `streaming_kernels.cc`: worker 侧 packed chunk 计算 kernel。
- `run_npu.py`: 编译、运行并校验。
