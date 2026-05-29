# XDNA NPU 编程入门：从零开始理解数据流机器

面向已有 CPU/GPU 编程经验但从未接触过 NPU 数据流编程的读者。

全书分三部分：
- **第一部分（硬件基础）**：建立心智模型，能看懂一个最小例子
- **第二部分（设计 Pattern）**：11 个通用规律，拿到新算子知道怎么拆
- **第三部分（工程实践）**：编译、调试、端到端实战

---

# 第一部分：硬件基础

## 第一章：为什么需要另一种计算模型

- CPU 的冯诺依曼模型：取指令→解码→执行→写回，程序决定一切
- GPU 的 SIMT 模型：成千上万线程跑同一段程序，吞吐换延迟
- 问题：单 token LLM decode 时，计算量小、带宽是瓶颈，GPU 大量核心空转
- NPU 数据流模型的核心思路：不让 CPU 每步调度，把整个计算图预配置成一台硬件机器，数据"流过"就出结果
- 类比：CPU 像人工车间（每步看图纸），GPU 像千人抄写班（同时抄同一段），NPU 像自动流水线（接通管道，开阀门就跑）

## 第二章：硬件地图——Tile 阵列长什么样

- 三层结构：Shim（大门）→ Memtile（中转站）→ Compute Tile（车间）
- 每层的物理资源：本地内存大小、DMA 引擎数量、lock 数量
- 坐标系统：`c{列}r{行}`，如何用网格定位一个 tile
- 物理连线：tile 之间有固定方向的 stream 连接，不是任意互联
- 一个具体例子：8 列 × 6 行 = 48 格的 NPU 分区

## 第三章：数据怎么搬——DMA 和 Buffer Descriptor

- DMA 的角色：tile 的"自动搬运工"，给任务单就持续搬，不需要 CPU 过问
- BD（Buffer Descriptor）是什么：一张任务单，写着从哪搬、搬多少、搬完干什么
- BD 的关键字段：buffer 地址、长度、stride（跨步）、next BD（链式）、packet ID（路由标签）
- 硬约束：Shim 只有 16 个 BD slot，用完就没有了
- 2D stride：不是逐字节连续搬，而是按行跳跃，适合矩阵切片
- BD ring：多个 BD 首尾相连形成循环，DMA 自动轮转

## 第四章：同步怎么做——Lock 和信用机制

- 问题：producer 写了一半 consumer 就来读，或 consumer 没读完 producer 就覆盖
- Lock 的本质：一个整数计数器 + acquire/release 语义
- 最小例子：单 buffer 的 empty/full lock
- Ping-pong 双缓冲：两个 buffer 交替使用，传输和计算重叠
- Counting lock：一个 producer 多个 consumer，初值 = consumer 数量
- 死锁的常见原因：acquire 没配 release、lock 初值错、BD bank 规则违反
- 调试信号：NPU timeout ≈ 几乎总是 lock 等不到

## 第五章：数据怎么路由——Stream 和 Packet

- Stream：tile 之间的固定物理连线，编译时确定路径
- 物理线很少，逻辑数据流很多，怎么办？
- Packet 路由：同一根线上给每个数据包贴 ID，接收端按 ID 筛选
- packet_flow vs circuit_flow：分时复用 vs 独占
- 硬约束：packet ID 全局唯一，重复 = 跨列死锁

## 第六章：从 CPU 思维转换到数据流思维

- "调用" → "数据到达 + lock 就绪"
- "返回值" → "写入下游 buffer + release lock"
- "循环" → "BD ring 自动轮转"
- "条件分支" → "packet ID 路由"
- "全局变量" → "RTP buffer（Runtime Parameter）"
- 什么东西不要带进来：函数调用栈、动态内存分配、任意寻址

## 第七章：第一个完整例子——让两个 Tile 传一个向量

- 目标：Tile A 产生 256 个 bf16，通过 stream 传给 Tile B
- 步骤：声明 tile、buffer、lock、BD、stream flow、core 程序
- 运行结果：B 读到 A 产出的向量
- 常见错误清单：BD 长度不匹配、lock 初值反了、channel 方向写错

---

# 第二部分：设计 Pattern

> 这些 pattern 不是某个模型的特殊做法，而是在 XDNA 上做任何算子都会反复遇到的通用规律。
> 每个 pattern 的结构：问题类型 → 通用解法 → 适用条件 → 失败信号 → 实例。

## 第八章：Pattern 1——大算子拆成 tile-local 工作单元

- 问题：输入/权重/输出太大，单 tile 放不下完整张量
- 解法：每个 tile 只负责固定输出切片 + 固定输入窗口 + 本地 accumulator
- tile 不需要理解整个模型，只需要知道"这次来了哪块、我负责哪几行、算完怎么写出去"
- 适用条件：输出可切分、reduce 可分块累加、local memory 能放下当前 chunk
- 失败信号：tile 里出现大量全局 if/switch、一个 tile 保存太多阶段状态
- 实例：16 个 tile 各自处理 32 行 × 256 列窗口的投影 MAC

## 第九章：Pattern 2——低 batch matvec 优先优化数据流

- 问题：单 token decode 下 matvec 是 memory-bound，不是 compute-bound
- 解法：权重流进来一次就用完、activation 便宜就 replay、量化权重在线反量化不展开
- 两种调度策略：
  - block-major：对一个输出 block 读完整输入 chunks（activation replay 便宜时）
  - chunk-major：一个 activation chunk 同时服务多个输出 block（activation 来自片上 packet 时）
- 适用条件：reduce 是线性累加、权重流量 > 激活流量
- 失败信号：反量化结果被物化成完整矩阵、activation 被重复搬太多次
- 实例：Q/K/V 用 block-major，O/down 用 chunk-major multi-block

## 第十章：Pattern 3——多 tile 输出先变成稳定 record，再汇聚

- 问题：多 tile 并发产出小块结果，下游需要一个大逻辑张量
- 解法：定义稳定的小 record ABI（header + payload），做层级汇聚
  - tile record → column compact → global compact → downstream
- 为什么不直接让每个 tile 写大 tensor 任意位置：地址复杂、BD 膨胀、debug 困难
- 适用条件：tile 输出粒度固定、下游消费顺序固定、metadata 很小
- 失败信号：header 越来越大、record 顺序靠运行时条件修正
- 实例：17 dword tile record → 65 dword column compact → 257 dword global compact

## 第十一章：Pattern 4——producer/consumer 用 ping-pong + credit lock 表达

- 问题：producer 和 consumer 速度不同，没有信用机制就会覆盖或读脏
- 解法：双缓冲 + empty/full lock 对
  ```
  producer: acquire empty → 写 → release full
  consumer: acquire full → 读 → release empty
  ```
- 一对多 fanout 用 counting lock（初值 = consumer 数）
- 适用条件：固定数据流、每个 slot 容量和消费次数可预先确定
- 失败信号：BD 只有 acquire 没有 release、lock 初值靠猜、多 producer 共用 channel 靠顺序碰运气
- 实例：权重 stream 的 ping/pong patch buffer、shared activation bridge

## 第十二章：Pattern 5——scarce channel 先分所有权，再用 packet 复用逻辑流

- 问题：物理 DMA channel 很少，逻辑数据流很多，容易冲突
- 解法：分两层——
  - channel ownership：规定每个物理 channel 属于哪类流量（硬约束）
  - packet ID：在已拥有的物理路径上区分逻辑数据（软标签）
- 先固定 channel ownership，再分配 packet ID，两者不能混为一谈
- 适用条件：物理路径稳定、接收端只需筛选少数 packet 类型
- 失败信号：以为换 packet ID 就能解决 channel 冲突、同一 channel 承担两个无关 producer
- 实例：row1 S2MM0-3=compact gather, S2MM4-5=weight ingress, MM2S0-3=weight fanout, MM2S5=compact output

## 第十三章：Pattern 6——layout 是算子设计的一部分

- 问题：逻辑 tensor layout（人类友好）和 DMA 需求（连续地址、可表达 stride、少量 BD）往往不一致
- 解法：从 DMA 消费单位反推物理 layout
  - 下游一次读多少？能否用一个 BD 描述？需要 2D stride 吗？值得提前 pack 吗？
- 热路径数据 pack 成硬件友好布局，pack 成本只付一次，后面换来更少 BD 和更少 copy
- 适用条件：tensor 被多次消费或在关键路径上很大
- 失败信号：为保持逻辑 layout 用了很多小 BD、stride 接近硬件范围、scan 需要复杂 gather
- 实例：KV cache block-major 布局、current K/V 先 even 后 odd、Q4NX 权重预 pack

## 第十四章：Pattern 7——大中间矩阵改成 block carrier + online merge

- 问题：某些算子有巨大中间矩阵（如 attention score matrix），落地会压垮内存或带宽
- 解法：按 block 流式处理——
  - 每个 block 计算局部结果 → 提取最小充分统计量 carrier → 交给 merge 端
  - merge 端保存 running state，用 online 公式合并每个 block
- carrier 应远小于完整中间矩阵，只包含跨 block 合并必须的信息
- 适用条件：跨 block 合并有稳定数学公式（如 online softmax）、running state 放得下
- 失败信号：carrier 越做越大接近原始矩阵、merge 需要回看过去 block 完整数据
- 实例：Shape-A 产出 80-dword carrier（权重 + max/sum），Shape-B 做 online softmax merge

## 第十五章：Pattern 8——小追加和大扫描要分成两个数据流

- 问题：状态型算子同时有两种访问——append（少量写入，地址动态）和 scan（大量顺序读）
- 解法：拆成两个阶段——
  1. append/update 当前状态（少数 2D/scatter BD）
  2. sync 确认写入完成
  3. bulk scan 历史状态（顺序 BD / iterated BD / queue repeat）
- 适用条件：当前写入必须被本次 scan 看到、scan 布局可 block 化
- 失败信号：scan 偶尔读到旧值、每 block 分配一个 BD 超硬件上限、只有 iteration 没有 repeat
- 实例：current K/V 用 linked BD 写回 → npu.sync → KV scan 用 iteration + repeat 复用 BD

## 第十六章：Pattern 9——动态参数走 RTP/descriptor patch

- 问题：拓扑不变但每次运行有少量参数变化（token position、block count、tail mask）
- 解法：把动态性分层——
  - tile 读的值 → RTP buffer + runtime-start lock
  - DMA 需要的值 → runtime descriptor writebd/address_patch
  - instruction stream 常数 → patch design.bin
  - topology/channel 改变 → 重新编译
- 适用条件：动态参数只改变次数/offset/tail，不改变物理图
- 失败信号：tile 在 RTP 写入前启动、patch 后没有和直接编译结果比对
- 实例：current-token RTP、scan iteration word、queue repeat count、KV write offset

## 第十七章：Pattern 10——融合不是一个大 kernel，而是稳定 ABI 之间的片上交接

- 问题：多个算子连续执行，中间激活大，每个算子写回主存再读回太浪费
- 解法：按稳定 ABI 融合——
  - 算子 A 输出稳定 packet/record
  - 算子 B 直接从片上接收
  - 中间不回主存，host 只在图边界配置和提交
- 融合边界应该是数据 ABI，不是函数调用边界
- 适用条件：中间数据大且下游马上消费、producer/consumer 能用固定 packet/record 对接
- 失败信号：一个 tile 变成全能 station、ABI 变得含混、不同 layout 被硬捏在一起
- 实例：整层 hidden→Q/K/V→attention→O→up/gate→SwiGLU→down 通过 packet0/1/2 片上交接

## 第十八章：Pattern 11——用集成边界验证数据流

- 问题：XDNA 错误跨越 Python 生成器 / MLIR / BD-lock-packet / C++ kernel / host BO 多层
- 解法：验证稳定的集成边界——
  - 一条完整物理路径 + 真实 DMA/packet/lock 闭环
  - host reference 覆盖同样物理 layout
  - 必要时读回中间 BO 定位失败点
  - poisoned data 区分路径失败和数学失败
- 结构检查（contract/dataflow/MLIR marker）和真机运行检查都要有
- 适用条件：边界相对稳定不因小重构变化、reference 能准确模拟物理 layout
- 失败信号：大量测试只验证 Python helper 没有验证生成 MLIR、只看最终输出不读中间状态
- 实例：check_contract.py 结构检查、run_npu.py --check-only/--build-only、poisoned current slots

---

# 第三部分：工程实践

## 第十九章：Compute Tile 编程——在小盒子里写 C++

- 每个 tile 的约束：几十 KB 本地内存，看不到其他 tile
- 编程模型：收一块输入 → 算 → 发一块输出，周而复始
- 和 DMA 的协作：core 只管 local buffer，DMA 在后台搬运
- AIE API 基础：向量运算、类型转换、SIMD
- 两种角色的 kernel：
  - 同构 kernel（16 个 tile 跑一样的程序，如 projection）
  - 异构 kernel（每个 tile 不同角色，如 postprocess/swiglu/attention）
- 常见陷阱：buffer 超 local memory、忘记清零 accumulator、输出格式和 BD 长度不匹配

## 第二十章：Memtile 编程——纯 DMA 编排

- Memtile 不跑 C++ 程序，只做数据搬运编排
- 典型用途：拆分/缓冲/扇出/汇聚
- Channel 数量和 BD bank 规则：偶通道用 BD 0-23，奇通道用 BD 24-47
- 实际例子：权重从 Shim → Memtile 双缓冲 → 分发给 4 行 Compute Tile

## 第二十一章：Host 侧——怎么把任务提交给 NPU

- BO（Buffer Object）：host 和 NPU 共享的内存区域
- Runtime sequence：npu.writebd / npu.push_queue / npu.sync
- XRT API 流程：加载 xclbin → 分配 BO → 填数据 → 提交 → 等待 → 读回
- 5 个 BO 参数上限及其应对

## 第二十二章：编译流水线——从 Python 到 xclbin

- 为什么用 Python 生成 MLIR 而不是手写
- MLIR-AIE 的核心概念：tile 声明、buffer、lock、dma_start、flow、core
- 编译步骤：Python generator → .mlir → aiecc → kernel ELF + route → xclbin
- 关键产物：design.mlir、design.xclbin、design.bin（instruction stream）
- 结构检查：在编译之前用代码验证约束（contract/physical_contract 思路）

## 第二十三章：调试——NPU 出错时怎么定位

- NPU 的错误信号很有限：timeout 或数据不对
- timeout 定位：哪个 lock acquire 永远等不到
- 数据错误定位：
  - 读回中间 BO，缩小范围
  - poisoned data（填特征值），区分"没搬到"和"搬错了"
  - host reference 用同样物理 layout 模拟
- 常见 bug 分类：lock 配对错、BD bank 违规、packet ID 冲突、channel 所有权冲突、RTP 时序错

## 第二十四章：实战——一个量化 Matvec 的完整实现

- 问题：4096 维输入 × Q4NX 权重 → 4096 维输出（单 token decode）
- 应用 Pattern 1：16 tile 各负责 32 行输出，权重按 256 列 chunk 流入，本地累加
- 应用 Pattern 2：权重流入一次用完，activation replay 广播
- 应用 Pattern 3：每 tile 产出 17-dword record → Memtile 层级汇聚 → 257-dword 全局输出
- 应用 Pattern 4：权重 ping-pong + credit lock
- 应用 Pattern 5：Memtile channel 所有权划分——weight ingress 和 compact gather 分离
- 应用 Pattern 6：Q4NX 权重预 pack 成 tile-friendly 的 chunk 布局
- 应用 Pattern 11：host reference 用相同物理 layout 做 numpy 对比
- 验证：真机跑通、reference 对齐

## 第二十五章：实战进阶——给 Matvec 加上融合和动态参数

- 应用 Pattern 10：两个 projection 片上交接（不回主存）
- 应用 Pattern 9：不同输出维度用 descriptor patch 而非重编译
- 应用 Pattern 8：如果有 cache scan 类需求，append 和 scan 分阶段
- 应用 Pattern 7：如果有大中间矩阵，block carrier + online merge
- 端到端：从单个算子到多算子融合的渐进路径

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

**进阶路线**（第二周）：
3. 读第二部分 Pattern 1-6（第 8-13 章），理解拆分和同步设计
4. 动手做第二十四章的量化 Matvec 实战

**融合路线**（第三周）：
5. 读 Pattern 7-11（第 14-18 章），理解融合和验证
6. 做第二十五章的融合进阶
7. 遇到 bug 时翻第二十三章
