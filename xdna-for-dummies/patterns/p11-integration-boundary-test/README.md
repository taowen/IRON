# Pattern 11: 用集成边界验证数据流，小单元测试不够

## 硬件问题

XDNA 错误跨越多个层面：Python generator → MLIR-AIE → BD/lock/packet route → C++ kernel → host BO layout。单独测试一个 helper 函数或一个 Python 模块，很难发现真实硬件路径上的 timeout、错包、错 layout。

NPU 出错时只有两种表现：timeout（lock 等不到）或数据不对。你需要方法快速定位是哪一层出了问题。

## 通用拆法

验证稳定的集成边界——选择一个完整的物理路径闭环：

```
Level 1: 结构检查（编译前，秒级）
  验证 BD/channel/packet/lock ownership 是否正确
  不等编译就能发现接线错误

Level 2: 编译检查（build-only，分钟级）
  验证 MLIR 能通过 aiecc
  发现 BD ID 超范围、lock 引用错误等

Level 3: 真机运行 + 分级验证（需要 NPU）
  a) 中间状态毒化: 关键 buffer 预填 poison 值
  b) 数学正确性: 和 CPU reference（同物理 layout）逐元素对比
  c) 定位: poison 残留 = 路径断了; 值不对 = 计算错了
```

毒化不只用于最终 output——关键中间状态（如 current K/V slot、compact buffer、scan input）更应该毒化。

## 适用条件

- 边界相对稳定，不因小重构频繁变化
- reference 能准确模拟物理 layout（不是逻辑 layout）
- 检查能覆盖 descriptor、packet、lock、route

## 失败信号

- 大量测试只验证 Python helper，没有验证生成的 MLIR
- reference 使用逻辑 layout，NPU 使用物理 layout（对比永远 mismatch）
- 只看最终输出，不读回关键中间状态（定位困难）
- 旧实验 case 保留太多，约束没有沉淀到共享检查

## 本例演示

```
host → compute (x*3+7) → host

Output buffer 预填 poison (0x7EADBEEF)
三级验证:
  Level 1: MLIR 结构检查（tile/flow/lock/kernel link 标记）
  Level 2: Poison 检查（任何残留 = DMA 路径断了）
  Level 3: 数值检查（和 CPU reference 对比）
```

**注意**：这个 demo 的结构检查很浅（只看 marker）。真实系统的结构检查应该验证 BD/channel ownership、lock 平衡、packet ID 唯一性——见 `physical_contract.py`。

## 真机结果

```
  Level 1: Structure check → PASS
  Level 2: Poison check → PASS (no poison remains)
  Level 3: Value check → PASS (output = input*3 + 7)
  ALL THREE LEVELS PASS
```

## 去 qwen3-layer 抄哪里

**结构检查层**：`qwen3-layer/check_contract.py` + `qwen3-layer/physical_contract.py`
```python
# 不只是 marker 检查——验证物理约束:
validate_q4nx_down_full_layer_ownership():
    # S2MM4/5 = weight (不能在 compact S2MM0-3 上)
    # 每个 main16 tile 有正确的 DMA start 标记
    # 没有链接被禁止的旧 object
    # compact output 在 MM2S5 (不和 weight fanout 混)
```

**毒化验证层**：`qwen3-layer/cases/currentkv_full_layer_q4nx_down_reference.py`
```python
# current K/V slot 毒化: 验证 writeback-before-scan
# 如果 scan 读到 poison → 证明 append 没在 sync 前完成
```

**运行层**：`qwen3-layer/run_npu.py`
```bash
--check-only   # Level 1: 结构检查
--build-only   # Level 2: 编译检查
(默认)         # Level 3: 真机运行 + reference 对比
```

## 运行

```bash
.venv/bin/python xdna-for-dummies/patterns/p11-integration-boundary-test/run_npu.py
```
