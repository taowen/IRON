# Edge To Main Direct Handoff

来源：`experiments/64_mylm_attention_to_o_direct_handoff`

这个 recipe 展示 edge 侧 producer 的结果不落 host，直接作为 main16 projection 的
activation 输入。这里用 attention-slice replay 作为 producer，但 pattern 本身是
通用的 edge-to-main direct stream handoff。

## 学什么

- edge tile 产出 deterministic activation slice，直接流到对应 main tile 的 activation input。
- projection weight 仍然走 packetized patch queue，经 row1 小 chunk ring 喂给 main16。
- main16 worker 同时消费 edge activation 和 weight chunk，产出 projection record。
- host-visible output 只保留最终 record，用于校验。

## 关键路径

```text
edge replay slice          -> main16 DMA0 activation input
host weight packet queue   -> row1 split/chunk ring -> main16 DMA1 weight input
main16 output packet       -> row1 gather -> host output
```

这个 recipe 的 edge producer 是确定性 contract kernel，不是 production attention。
它专门用来学习 direct handoff 的 transport ABI。

## 运行

```bash
.venv/bin/python recipes/edge-to-main-direct-handoff/run_npu.py
```

通过后会打印 projection record headers、CPU/NPU 前几个输出值和误差。

## 文件

- `generate.py`: 生成 direct handoff + packetized weight path 的 MLIR-AIE。
- `reference.py`: 生成 weight 和 CPU reference。
- `projection_nblock.cc`: edge replay 和 main16 projection tile kernel。
- `run_npu.py`: 编译、运行并校验输出。
