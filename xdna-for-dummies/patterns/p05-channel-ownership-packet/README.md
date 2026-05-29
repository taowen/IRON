# Pattern 5: Scarce Channel 先分所有权，再用 Packet 复用逻辑流

## 硬件问题

物理 DMA channel 很少（每个 tile 只有几个 MM2S/S2MM），但逻辑数据流很多。常见错误：
- 以为换一个 packet ID 就能解决 channel 冲突（不行——physical channel 必须有单一 owner）
- 两个无关 producer 共用一个 channel → lock/BD 生命周期冲突 → 死锁

## 通用拆法

分两层：

```
1. Channel ownership（硬约束）:
   规定每个物理 channel 属于哪一个数据流 owner
   这是不可违反的——混用 = 死锁

2. Packet ID（逻辑复用）:
   在已有 owner 的物理路径上，用 packet ID 区分多路逻辑数据
   receiver 根据 ID 筛选"这个包是给我的"
```

先固定 channel ownership，再分配 packet ID。两者不能混为一谈。

## 适用条件

- 物理路径稳定（编译时确定）
- packet source/dest 明确
- 接收端只需筛选少数 packet 类型
- 不同 packet 的生命周期不会互相破坏 lock credit

## 失败信号

- 以为换 packet ID 就能解决 channel 冲突（本质是 ownership 问题）
- 同一 channel 同时承担两个互不相关的 producer 顺序
- packet ID 没有集中登记，新增路径靠记忆避冲突
- packet ID 全局重复 → 跨列死锁

## 本例演示

```
host → producer(2,2) OWNS MM2S ch0
                  ├── packet_id=0 ──► worker0(2,3): adds +10
                  └── packet_id=1 ──► worker1(2,4): adds +20
```

关键：producer 的 **同一个 MM2S ch0** 发送两种 packet。BD ring 交替发 pkt0 和 pkt1：
```mlir
^send_pkt0:
  aie.dma_bd(...) {packet = #aie.packet_info<pkt_id = 0>}
  aie.next_bd ^send_pkt1
^send_pkt1:
  aie.dma_bd(...) {packet = #aie.packet_info<pkt_id = 1>}
  aie.next_bd ^send_pkt0
```

worker0 只收到 packet 0 的数据，worker1 只收到 packet 1 的数据。

## 真机结果

```
  Producer OWNS MM2S ch0 exclusively
  Same physical channel carries TWO logical streams:
    packet_id=0 → worker0 (adds +10)
    packet_id=1 → worker1 (adds +20)

  NPU time: 630.9 us
  worker0 (pkt0): expected=[11, 12, 13, 14] got=[11, 12, 13, 14]
  worker1 (pkt1): expected=[53, 54, 55, 56] got=[53, 54, 55, 56]
  PASS: single channel owner, packet ID demuxes to correct consumers
```

## 去 qwen3-layer 抄哪里

**`qwen3-layer/physical_contract.py`** — channel ownership 的显式检查：

```python
ROW1_COMPACT_S2MM = (0, 1, 2, 3)    # 只用于 compact gather
ROW1_WEIGHT_S2MM = (4, 5)            # 只用于 weight ingress
ROW1_WEIGHT_MM2S = (0, 1, 2, 3)     # 只用于 weight fanout
ROW1_COMPACT_MM2S = 5                # 只用于 compact output
```

旧版曾把权重走 S2MM0/1 → 和 compact 冲突 → 死锁。`physical_contract.py` 现在显式禁止这种旧路由。

**packet ID 复用真实例子**：
- `compact_dataflow.py` 中 c1r1 MM2S ch5 用 packet ID 区分 Q/K/V/O/upgate/down 的 compact 输出
- c6r1 hub 的 MM2S 用 packet2（attention）和 packet1（FFN down）复用同一条到 c1r1 bridge 的物理线

## 运行

```bash
.venv/bin/python xdna-for-dummies/patterns/p05-channel-ownership-packet/run_npu.py
```
