# XDNA NPU 编程入门：从零开始理解数据流机器

面向已有 CPU/GPU 编程经验但从未接触过 NPU 数据流编程的读者。

全书分四部分：
- **第一部分（硬件基础）**：建立心智模型，能看懂一个最小例子
- **第二部分（编程实操）**：Compute Tile / Memtile / Host / 编译，能动手写代码
- **第三部分（设计 Pattern）**：11 个通用规律，拿到新算子知道怎么拆
- **第四部分（综合实战）**：调试方法 + 端到端案例

---

# 第一部分：硬件基础

## 第一章：为什么需要另一种计算模型

- CPU 冯诺依曼 vs GPU SIMT vs NPU 数据流
- 单 token LLM decode 的瓶颈：memory-bound，GPU 空转
- NPU 的核心思路：预配置数据流机器，一次启动，片上交接
- 类比：人工车间 / 千人抄写班 / 自动流水线

## 第二章：硬件地图——Tile 阵列长什么样

- 三层结构：Shim → Memtile → Compute Tile
- 每层的物理资源对比
- 坐标系统 `c{列}r{行}`
- 8 列 × 6 行 = 48 格 NPU 分区

## 第三章：数据怎么搬——DMA 和 Buffer Descriptor

- DMA：自动搬运工，S2MM（收件）/ MM2S（寄件）
- BD：任务单（地址、长度、stride、next BD、lock、packet ID）
- BD ring：循环任务
- 2D stride：不连续搬运
- 硬约束：Shim 16 个 BD、Memtile BD bank 规则
- Iterated BD + queue repeat：一个 BD 扫描多个 block

## 第四章：同步怎么做——Lock 和信用机制

- Lock = 硬件计数器 + acquire/release
- 最小例子：单 buffer empty/full lock
- Ping-pong 双缓冲
- Counting lock：一对多 fanout
- 死锁常见原因和调试直觉

## 第五章：数据怎么路由——Stream 和 Packet

- Stream：编译时确定的物理连线
- Packet 路由：同一物理线多路复用，按 ID 筛选
- circuit_flow vs packet_flow
- 硬约束：packet ID 全局唯一

## 第六章：从 CPU 思维转换到数据流思维

- 调用→数据到达，返回值→release lock，循环→BD ring，分支→packet ID
- 什么东西不能带进来：调用栈、malloc、任意寻址、隐式顺序

## 第七章：第一个完整例子——让两个 Tile 传一个向量

- 目标：Tile A 产生 256 bf16 → stream → Tile B
- 完整步骤：tile/buffer/lock/BD/stream/core
- 时序图
- 常见错误清单

---

# 第二部分：编程实操

## 第八章：Compute Tile 编程——在小盒子里写 C++

- 约束：~64 KB 本地内存，看不到其他 tile
- 编程模型：收一块→算→发一块，循环往复
- 和 DMA 的协作：core 只管 local buffer，DMA 后台搬运
- AIE API 基础：向量运算、类型转换
- 两种角色的 kernel：同构（projection）vs 异构（postprocess/swiglu/attention）
- 输出格式：record（header + payload）
- 常见陷阱：buffer 超限、accumulator 未清零、输出大小和 BD 不匹配

## 第九章：Memtile 编程——纯 DMA 编排

- Memtile 没有处理器，全靠 BD + lock 描述行为
- 四种典型用途：扇出、汇聚、双缓冲中转、切片重排
- BD bank 规则：偶通道 BD 0-23，奇通道 BD 24-47
- Channel 所有权：为什么不能混用
- 实例：权重双缓冲分发（ping-pong + counting lock）

## 第十章：Host 侧——怎么把任务提交给 NPU

- BO（Buffer Object）：host/NPU 共享内存
- Runtime sequence：writebd / push_queue / sync
- XRT API 流程：加载 xclbin → 分配 BO → 填数据 → 提交 → 等待 → 读回
- 5 个 BO 参数上限及其应对（打包 + 偏移 patch）
- RTP + runtime-start lock：动态参数的安全写入

## 第十一章：编译流水线——从 Python 到 xclbin

- 为什么用 Python 生成 MLIR 而不是手写
- MLIR-AIE 核心概念：tile / buffer / lock / dma_start / flow / core / link_with
- 编译步骤：Python → .mlir → clang++ kernel .o → aiecc → xclbin + design.bin
- 结构检查：编译前用代码验证约束（contract / physical_contract）
- Instruction patch：编译一次最大容量 xclbin，每次运行只 patch design.bin

---

# 第三部分：设计 Pattern

> 这些 pattern 不是某个模型的特殊做法，而是在 XDNA 上做任何算子都会反复遇到的通用规律。
> 每个 pattern 的结构：问题类型 → 通用解法 → 适用条件 → 失败信号 → 实例。

## 第十二章：Pattern 1——大算子拆成 tile-local 工作单元

- 问题：输入/权重/输出太大，单 tile 放不下
- 解法：每个 tile 只负责固定输出切片 + 固定输入窗口 + 本地 accumulator
- 失败信号：tile 里出现大量全局 if/switch、保存太多阶段状态
- 实例：16 tile 各 32 行 × 256 列投影 MAC

## 第十三章：Pattern 2——低 batch matvec 优先优化数据流

- 问题：单 token decode = memory-bound，盯 MAC 数量会错过瓶颈
- 解法：权重流入一次用完、activation 便宜就 replay、在线反量化不展开
- 两种调度：block-major vs chunk-major
- 实例：Q/K/V block-major，O/down chunk-major multi-block

## 第十四章：Pattern 3——多 tile 输出先变成稳定 record，再汇聚

- 问题：多 tile 并发产出，下游需要大逻辑张量
- 解法：record ABI（header + payload）→ column compact → global compact
- 失败信号：header 越来越大、record 顺序靠运行时条件修正
- 实例：17 dw → 65 dw → 257 dw

## 第十五章：Pattern 4——producer/consumer 用 ping-pong + credit lock 表达

- 问题：producer/consumer 速率不同
- 解法：双缓冲 + empty/full lock；一对多用 counting lock
- 失败信号：BD 只有 acquire 没有 release、lock 初值靠猜
- 实例：权重 stream ping/pong、shared activation bridge

## 第十六章：Pattern 5——scarce channel 先分所有权，再用 packet 复用

- 问题：物理 channel 少，逻辑流量多
- 解法：channel ownership（硬约束）+ packet ID（软标签），两层分离
- 失败信号：以为换 packet ID 能解决 channel 冲突
- 实例：row1 S2MM0-3/4-5/MM2S0-3/5 的固定分工

## 第十七章：Pattern 6——layout 是算子设计的一部分

- 问题：逻辑 layout 对 DMA 不友好（非连续、stride 超范围、BD 数量膨胀）
- 解法：从 DMA 消费单位反推物理 layout，热路径提前 pack
- 失败信号：为保持逻辑 layout 用了很多小 BD
- 实例：KV cache block-major、Q4NX 权重预 pack

## 第十八章：Pattern 7——大中间矩阵改成 block carrier + online merge

- 问题：attention score matrix 落地会压垮内存
- 解法：每 block 提取最小充分统计量 carrier → merge 端用 online 公式合并
- 失败信号：carrier 越做越大、merge 需要回看过去 block
- 实例：Shape-A 80-dw carrier + Shape-B online softmax merge

## 第十九章：Pattern 8——小追加和大扫描分成两个数据流

- 问题：状态型算子同时有 append（少量写）和 scan（大量读）
- 解法：append → sync → scan，分阶段设计
- 失败信号：scan 读到旧值、每 block 一个 BD 超硬件上限
- 实例：current K/V 写回 → npu.sync → KV scan

## 第二十章：Pattern 9——动态参数走 RTP/descriptor patch

- 问题：拓扑不变但 token position/block count/tail mask 每次不同
- 解法：tile 读 RTP（加 runtime-start lock）、DMA 用 writebd patch、topology 变化才重编译
- 失败信号：tile 在 RTP 写入前启动、patch 后没和直接编译结果比对
- 实例：current-token RTP、scan iteration、queue repeat count

## 第二十一章：Pattern 10——融合不是一个大 kernel，而是稳定 ABI 间片上交接

- 问题：多算子连续执行，中间激活回主存太浪费
- 解法：算子 A 输出稳定 packet/record → 算子 B 直接片上接收，host 只在图边界操作
- 失败信号：一个 tile 变全能 station、ABI 含混、不同 layout 硬捏一起
- 实例：hidden→Q/K/V→attention→O→up/gate→SwiGLU→down 全程片上交接

## 第二十二章：Pattern 11——用集成边界验证数据流

- 问题：XDNA 错误跨越 Python/MLIR/BD-lock-packet/C++ kernel/host BO 多层
- 解法：验证完整物理路径闭环 + host reference 同物理 layout + poisoned data
- 失败信号：大量测试只验证 Python helper、只看最终输出不读中间状态
- 实例：check_contract.py、run_npu.py --check-only、poisoned current slots

---

# 第四部分：调试——NPU 出错时怎么定位

- timeout 定位：哪个 lock acquire 等不到
- 数据错误定位：读回中间 BO、poisoned data、host reference 同物理 layout
- 常见 bug 分类：lock 配对错、BD bank 违规、packet ID 冲突、channel 所有权冲突、RTP 时序错

---

# 附录

## 附录 A：术语表

| 术语 | 含义 |
|------|------|
| Tile | NPU 阵列中的一个处理单元格 |
| Shim | Row 0，主存入口/出口 |
| Memtile | Row 1，片上大缓冲 + DMA 编排 |
| Compute Tile | Row 2+，跑 C++ kernel 的计算单元 |
| DMA | 直接内存访问，自动搬运数据 |
| BD | Buffer Descriptor，DMA 的任务单 |
| Lock | 生产者/消费者同步的计数器 |
| Stream | Tile 间的固定物理连线 |
| Packet | 数据包路由标签，复用物理线 |
| S2MM | Stream to Memory-Mapped（接收方向） |
| MM2S | Memory-Mapped to Stream（发送方向） |
| BO | Buffer Object，host/NPU 共享内存 |
| RTP | Runtime Parameter，运行时可写参数 |
| xclbin | 编译后的 NPU 配置二进制 |
| instruction stream | runtime sequence 编译后的指令流 |
| ping-pong | 双缓冲交替使用 |
| credit lock / counting lock | 初值>1 的 lock，表达多消费者信用 |
| Q4NX | 4-bit 量化权重格式 |
| bf16 | bfloat16，1+8+7 浮点格式 |
| dword | 32-bit word |

## 附录 B：硬件限制速查

| 资源 | 限制 |
|------|------|
| Shim BD | 每个 Shim tile 最多 16 个 |
| Memtile BD bank | 偶通道 BD 0-23，奇通道 BD 24-47 |
| Compute Tile 本地内存 | ~64 KB（型号相关） |
| Memtile 内存 | ~512 KB（型号相关） |
| DMA channel | 每个 tile 若干 S2MM + MM2S |
| Lock 数量 | 每个 tile 有限（~16） |
| Packet ID | 全局唯一，跨列不可重复 |
| XRT buffer 参数 | 单次 kernel 最多 5 个 BO |
| BD block release | 一个 BD block 最多一个 release |

## 附录 C：Pattern 速查表

| # | 问题类型 | 通用解法 |
|---|----------|----------|
| 1 | 算子太大, 单 tile 放不下 | tile-local 工作单元 |
| 2 | 低 batch matvec memory-bound | stream weight, reuse activation |
| 3 | 多 tile 小输出汇聚 | record → column compact → global compact |
| 4 | producer/consumer 速率不同 | ping-pong + credit lock |
| 5 | channel 稀缺且流量多 | channel ownership + packet ID |
| 6 | 逻辑 layout 不适合 DMA | hardware-first 物理 layout |
| 7 | 中间矩阵太大 | block carrier + online merge |
| 8 | 小追加 + 大扫描 | append 和 scan 分阶段 |
| 9 | runtime 参数变化 | RTP / descriptor patch |
| 10 | 中间激活回主存太贵 | 稳定 ABI 间片上交接 |
| 11 | 硬件路径难定位 | 集成边界验证 |

## 附录 D：推荐阅读路径

**入门路线**（第一周）：
1. 通读第一部分（第 1-7 章），建立硬件心智模型
2. 跑通第七章的最小例子

**动手路线**（第二周）：
3. 读第二部分（第 8-11 章），理解各层编程方式
4. 自己写一个 2-tile 的 ping-pong 传输

**设计路线**（第三周）：
5. 读第三部分 Pattern 1-6（第 12-17 章），理解拆分和同步设计
6. 动手做第二十四章的量化 Matvec 实战

**融合路线**（第四周）：
7. 读 Pattern 7-11（第 18-22 章），理解融合和验证
8. 做第二十五章的融合进阶
9. 遇到 bug 时翻第二十三章
