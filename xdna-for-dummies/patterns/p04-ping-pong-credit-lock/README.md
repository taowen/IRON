# Pattern 4: Producer/Consumer 用 Ping-Pong + Credit Lock 表达

## 硬件问题

XDNA 内部数据流不是 CPU 每步调度。producer 和 consumer 速度不同，没有信用机制就会：
- producer 覆盖 consumer 还没读的数据
- consumer 读到未写满的 buffer
- DMA channel 等不到锁而 timeout

## 通用拆法

双缓冲 + empty/full lock，表达为 BD ring：

```
producer BD ring:
  ^ping: acquire(empty,1) → write ping → release(full,1) → next=^pong
  ^pong: acquire(empty,1) → write pong → release(full,1) → next=^ping

consumer BD ring:
  ^ping: acquire(full,1) → read ping → release(empty,1) → next=^pong
  ^pong: acquire(full,1) → read pong → release(empty,1) → next=^ping
```

一对多 fanout 用 counting lock：
```
lock: empty (init = N)  ← N 个 consumer 都 release 才能覆盖
producer: acquire(empty, N) → write → release(full, N)
consumer_i: acquire(full, 1) → read → release(empty, 1)
```

## 适用条件

- producer/consumer 是固定数据流
- 每个 slot 的容量和消费次数可预先确定
- DMA BD ring 可以长期循环

## 失败信号

- **BD 只有 acquire 没有 release**（或反过来）→ 永久死锁，表现为 timeout
- **lock 初值错**：empty 应 = slot 数（ping-pong 时 = 2），full 应 = 0
- **BD bank 规则违反**：memtile 偶通道(0,2,4...)用 BD 0-23，奇通道(1,3,5...)用 BD 24-47。违反不报编译错，运行时静默死锁
- **多 producer 共用 channel**：顺序不确定 → lock 计数混乱
- **counting lock 不平衡**：producer release N 但只有 N-1 个 consumer release back → 永久少 1

## 本例演示

```
producer(2,2) ──[ping-pong BD ring]──► consumer(2,3) ──[ping-pong BD ring]──► host
```

producer 产生 2 batch 数据，consumer 加 42。展示：
- BD ring 的 `next_bd_id` 循环（0↔1）
- `wt_empty` init=2（两个 slot）
- acquire/release 严格配对

## 真机结果

```
  NPU time: 534.8 us
  expected[0:8]: [42, 43, 44, 45, 46, 47, 48, 49]
  got[0:8]:      [42, 43, 44, 45, 46, 47, 48, 49]
  PASS
```

## 去 qwen3-layer 抄哪里

**`qwen3-layer/weight_stream.py`** — row1 weight ping-pong 分发：

```python
# Counting lock: init = ROWS_PER_PATCH = 2 (两行 consumer)
# S2MM 写满后 release 2, 每个 row 的 MM2S 各 acquire 1
weight_stream_lock_defs(tile, lock_base):
    init = ROWS_PER_PATCH  # counting lock fanout
```

**`qwen3-layer/mlir_utils.py`** — lock 平衡检查、BD bank 规则验证。

**`qwen3-layer/compact_dataflow.py`** — c1r1 shared bridge 的 ping-pong：
```python
BRIDGE_PACKET_IN_BDS = (6, 7)   # S2MM ch4 (偶数 → BD 0-23 ✓)
BRIDGE_PACKET_OUT_BDS = (28, 29) # MM2S ch1 (奇数 → BD 24-47 ✓)
```

## 运行

```bash
.venv/bin/python xdna-for-dummies/patterns/p04-ping-pong-credit-lock/run_npu.py
```
