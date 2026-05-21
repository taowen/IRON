 核心结构：每个答案都自然引出下一个 why，读者的直觉不断被硬件现实修正。

  ---
  1. 为什么 NPU 需要一个专门的编程框架？
  
  GPU 有统一虚拟地址、cache 层次、线程调度器——程序员写 kernel，硬件负责搬数据。

  NPU (AIE) 没有这些。每个 compute tile 只有 64KB 本地存储，不能 load/store 到主存，没有 OS。数据必须由独立的 DMA 引擎搬进搬出，而 DMA 引擎需要编译时就写好的搬运指令（Buffer Descriptor）。

  所以问题不是"怎么写计算"，而是"怎么描述数据在哪、什么时候搬、搬到哪"。IRON 就是这个描述语言。

  ---
  2. 为什么数据搬运要在编译期决定？
  
  因为 DMA 引擎不是 CPU——它没有分支、没有条件跳转。它只会按照 BD 里写好的 (offset, size, stride) 序列机械执行。

  这意味着：你在编译时画好了 DMA 的完整路径图。一旦 xclbin 生成，搬运行为就固化了。

  IRON 里叫 TensorAccessPattern：你以为你在写一个高级切片描述，实际你在给 DMA 硬件编程。

  ---
  3. 那 compute tile 怎么知道数据到了？
  
  tile 不轮询，不中断。IRON 用 ObjectFifo 实现 producer-consumer 协议：

  - DMA 填满一个 slot → tile 可以 acquire
  - tile 处理完 → release 归还 slot → DMA 可以填下一个

  这不是可选的设计模式，而是唯一的通信机制。tile 上的代码必须严格按 acquire/release 节奏写，否则要么挂死（acquire 了空 slot），要么数据覆盖（release 前 DMA 写入）。

  ---
  4. 为什么这套机制能跑通一个 28 层模型？
  
  因为 28 层的计算模式是重复的：每层做相同的操作，只是权重不同。

  IRON 的 persistent 模式让 Worker 启动后不退出，循环 28 次。每次循环，DMA 按预设的 TAP offset 序列取下一层的权重。只要层结构一致，一份编译产物就能跑完所有层。

  关键词是模式可预测：层数固定、维度固定、搬运路径固定。编译器能在编译期完全展开整个执行计划。

  ---
  5. 那为什么换一个 decode 位置就不行了？
  
  这里暴露了编译期固化的根本代价。

  position 26 → position 27 时，变了什么？

  - KV cache 的写入位置从 offset 26 变到 27 → DMA 路径变了 → runtime .bin 变了
  - RoPE 的角度从 θ×26 变到 θ×27 → core ELF 里的立即数变了 → .text 段变了

  不是一个东西变了，是两层东西同时变了。而且这两层分别属于不同的编译产物（runtime .bin vs core ELF），没有统一的 patch 点。

  所以每生成一个新 token，就需要一套新的编译产物。这就是为什么当前方案是 "exact-position precompile"——提前为每个可能的位置编译好，运行时选用。能用，但 artifact 数量线性增长。

  ---
  6. 为什么不能用运行时参数（RTP）绕过这个问题？
  
  IRON 确实有 Buffer(use_write_rtp=True)——host 可以在运行时写入一个标量，Worker 可以读它。GEMM 用这个机制让 host 告诉 Worker "这次 K 维度循环多少次"。

  但 RTP 活在 tile 内部。它能影响 Worker 的分支和循环，但不能改变 shim DMA 的 BD offset。

  位置问题有两个面：
  - DMA 搬运路径（在 runtime .bin 里）→ RTP 碰不到
  - core 内部立即数（在 ELF 里）→ RTP 理论上能替代，但需要重写 kernel 逻辑，且尚未证明跨 invocation 稳定
  
  所以 RTP 不是"解决方案"，而是"可能解决一半问题的候选机制"，需要实验验证。

  ---
  7. 为什么不能 patch 编译产物？
  
  看起来最直接：runtime .bin 里就那几个 offset word 变了，直接改二进制？

  两个原因：
  1. core ELF 也变了。AIE 指令编码没有公开 relocation 表，不知道哪些字节是位置相关的立即数
  2. 即使只改 runtime .bin，你怎么确认改完的 BD 不会越界读写、不会破坏其他 DMA channel 的 interleave？
  
  缺乏验证机制的二进制 patch 是不可证伪的——出错时不报错，只是算出垃圾。

  ---
  8. 为什么不能用更多 tile 来掩盖单 token 编译开销？
  
  换个角度：如果每个 token 必须重编译，那把每个 token 的计算时间压到极短，编译开销就显得不那么大？

  直觉：GEMV 是 embarrassingly parallel，把 1024 行的矩阵分给 32 个 tile，每个只算 32 行。

  但硬件不允许。每个 tile 需要独立的 DMA 通道（ObjectFifo），每个通道占用 shim 上的一个端点。32 个输入 + 32 个输出 = 64 个端点。而 shim DMA 的端点数是有限的硬件资源。

  这不是软件配额，是物理上的连线数。超了就是放不下。resolve_program() 有时不报错（它只做逻辑放置），直到 aiecc 阶段才在 BD 分配时失败。

  ---
  9. 那 GEMM 是怎么用多 tile 的？为什么 GEMV 不能照搬？
  
  GEMM 的诀窍：L3→L2→L1 层级。一个 L2 tile 从 L3 接数据，再 fan-out 给多个 L1 compute tile。多个 compute tile 共享一个 L3 端点，端点数从 O(tiles) 降到 O(columns)。

  GEMV 的问题：M=1（只有一行输入向量）。GEMM 的 split 策略是按 M 维切块分发，M=1 时没有 M 维可切。需要设计不同的 split/join 拓扑，而这个新拓扑的资源开销和正确性都没有现成证明。

  教训：一个已验证的拓扑对另一个形状不自动成立。"GEMM 跑得好" 不能直接推导出 "同样的模式对 GEMV 也行"。

  ---
  10. 为什么不能让同一批 tile 先做 attention 再做 MLP？
  
  这是利用率问题的另一条路：不增加 tile 总数，而是让每个 tile 做更多工作。

  28 层中每层有 attention + MLP 两个阶段。如果同一组 Worker 能在一次 dispatch 中先跑 attention phase 再跑 MLP phase，就不需要静态划分"attention 列"和"MLP 列"。

  但 IRON 当前的 task_group 语义是：一批同构 DMA 操作的同步屏障。它证明了"一组 fill/drain 一起完成"，没有证明"host 在 Worker 运行中途更新状态，Worker 据此切换行为"。

  两者的差距：
  - 已证明：Worker 醒来 → 读一次 RTP → 跑一个循环 → 结束
  - 需要证明：Worker 醒来 → 跑 phase A → 等待 host 信号 → 跑 phase B → ... → 28 层后结束
  
  这个"运行中等待"需要 ObjectFifo 在 phase 间保持 acquire/release 平衡。如果 Worker 写了 acquire(input_B) 但 host 没有对 phase B 发 fill（因为想跳过），Worker 就永远挂在那里。跳不跳 phase 不是 Worker 内部的 if 能决定的——它是 dataflow 协议层的问题。

  ---
  11. 为什么 attention 不能留到最后再解决？
  
  如果 MLP/GEMV 占了 token time 的大头，先优化 MLP 不行吗？

  可以先做 MLP 实验，但不能假设 attention "到时候自然能塞进去"。原因：

  - attention 是位置依赖最密集的组件（RoPE 角度、KV cache 偏移、mask 长度全是位置函数）
  - 之前唯一一次尝试通过 ObjectFifo 元数据传递位置信息，就是在 attention 上失败的（timeout / NaN）
  - 如果最终证明 RTP 或 phase 调度机制无法处理 attention 的位置依赖，整个"动态复用 artifact"的方向就不成立

  attention 不是"最后 5% 的收尾工作"，它是方案可行性的试金石。

  ---
  12. 那当前到底知道什么、不知道什么？
  
  这是最重要的区分：

  ┌──────────────────────────────────────────┬────────────────────────────┐
  │         已证明（可以在上面构建）         │   未证明（不能假设成立）   │
  ├──────────────────────────────────────────┼────────────────────────────┤
  │ 28 层 persistent 循环跑通                │ 同一 xclbin 跑不同位置     │
  ├──────────────────────────────────────────┼────────────────────────────┤
  │ exact-position precompile 消除热循环编译 │ RTP 跨 invocation 改变行为 │
  ├──────────────────────────────────────────┼────────────────────────────┤
  │ GEMM 的 L2 split/join 在 GEMM 形状下合法 │ GEMV 形状下同样合法        │
  ├──────────────────────────────────────────┼────────────────────────────┤
  │ task_group 做一批 DMA 同步               │ task_group 做多 phase 调度 │
  ├──────────────────────────────────────────┼────────────────────────────┤
  │ RTP 控制 GEMM 的 K 循环次数              │ RTP 替代位置相关立即数     │
  ├──────────────────────────────────────────┼────────────────────────────┤
  │ segment-major packing 对 2-column 拓扑   │ 对 full-array 新拓扑       │
  └──────────────────────────────────────────┴────────────────────────────┘

  每一个"未证明"项都需要一个最小实验去证实或证伪，而不是直接在 Qwen3 全模型上试——因为全模型失败时你不知道是哪个假设出了问题。

  ---
  13. 为什么实验顺序不能随便选？
  
  因为假设之间有依赖：

  RTP 能跨 invocation 改变 Worker 行为吗？
      ↓ 如果能
  同一 Worker 能在一次 dispatch 中执行多个 phase 吗？
      ↓ 如果能
  attention 的位置依赖能通过 RTP + phase 协议解决吗？
      ↓ 如果能
  整个方案可行，开始集成到 Qwen3 layer graph

  如果第一个问题的答案是 No，后面的问题都不需要回答——方向本身需要重选。

  这就是为什么正确的工作方式是：从最小的、能独立判定的机制证明开始，而不是从最终目标倒推然后一次性实现。
