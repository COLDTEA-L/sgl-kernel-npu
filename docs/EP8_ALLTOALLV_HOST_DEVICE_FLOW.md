# 原生 HCCL 与 EP8 EID 多路径 AllToAllV：Host / Device 数据流对照

更新日期：2026-10-30。

图的格式是 **Markdown 中的 `text` 代码块，即等宽纯文本流程图**，不依赖 Mermaid 插件或图片文件。这里 Host 指 CPU，Device 指 NPU；Device 与 NPU 不是两个不同的执行位置。

EP8 实现说明以 `ce130f2` 为基线。更完整的调用链、选路算法和失败分析见 [EP8_CCU_EID_MULTIPATH_IMPLEMENTATION.md](EP8_CCU_EID_MULTIPATH_IMPLEMENTATION.md)。

## 1. 先区分两类数据

| 类型 | 内容 | 所在位置与流动方式 |
|---|---|---|
| Payload | packed token 的 FP8/BF16 hidden states | 源 NPU HBM → 目的 NPU HBM，不经 CPU 中转 |
| 控制信息 | 行计数、地址、偏移、长度、内存 token、选路结果 | Host 处理，必要时由 NPU 获取计数，再通过运行时提交给通信引擎 |

两种方案都不要求把 hidden states 先 D2H，再由 CPU H2D。可能引入 D2H 的主要是计数；Host 提交启动参数也不等于搬运整个 payload。

以下 router → packing → 通信的流程描述模型接入关系。router、packing/unpacking、专家计算和 top-k 权重合并由模型完成，不属于这个 AllToAllV 算子。

## 2. 当前 EP8 CCU + URMA EID 多路径

### 2.1 数据流

```text
                  Host / CPU                         Device / NPU
                      │                                   │
                      │                           router 计算专家选择
                      │                                   ↓
                      │                          token packing + 计数
                      │                                   │
                      │        若只有各卡本地计数：          │
                      │                          交换得到全局 C[8,8]
                      │                                   │
               C[8,8] ←────────── D2H：仅计数 ─────────────┘
                      │
               C++ 贪心规划
               · 哪条流分流？
               · relay 选谁？
               · 转发多少行？
                      │
               计算源/目的偏移
               获取地址、内存 token
               构造 46 个启动参数
                      │
                      └──────── 提交 CCU kernel ─────────→ CCU
                                                            │
                      ┌─────────────────────────────────────┤
                      │                                     │
                  direct 前缀                           relay 后缀
                      │                                     │
                源卡 HBM                              源卡 HBM
                      │                                     │
                 direct 通道                       EID 指定的转发路径
                      │                                     │
                      │                              经过组内 relay
                      │                              不落应用层中转 HBM
                      │                                     │
                      └──────────→ 目的卡 HBM ←──────────────┘
                                  两段写入不同区间
                                         │
                                  完成通知 / stream 顺序
                                         │
                                  专家计算或 unpack
```

图中计数交换和 D2H 是条件步骤：模型若已经持有一致的全局 CPU `C[8,8]`，可以直接调用 host。当前 EID host 不强制在内部执行计数 gather；测试脚本在 host 调用之外完成这一过程。

### 2.2 Host 做什么

1. 接收本次真实 packed 行数矩阵 `C[s,d]`。
2. 在静态 EID 候选目录内做贪心规划，每条原始流最多选择一个组内 relay。
3. 生成 direct 前缀与 relay 后缀的长度及源/目的偏移。
4. 校验 Tensor 布局，获取本次缓冲区的内存 token，提交 46 个 CCU 参数。

Host 不把完整 C 交给 CCU 再选路，而是将其转换为本 rank 的执行参数。参数包括地址、token、self-copy 信息、active mask，以及各 peer 的偏移、字节数和 relay rank。

### 2.3 Device 做什么

CCU 通过 active channels 交换最终接收地址和 token，提交 direct/relay 的非阻塞 Write，处理 self copy，并进行完成握手。

同一条原始流的两段读取原始 send 的不同区间，直接写入最终 recv 的对应区间，无需为了分流先复制出两份 payload。

relay 通道的逻辑对端仍是最终目的 rank；EID 配置指定中间转发路径。relay 卡不需要先把数据收进应用 Tensor，再运行第二跳发送算子。源、目的仍访问 HBM；这里不对互联硬件内部缓存作额外假设。

direct 与 relay 在同一 kernel 内具有并发执行机会，实际重叠程度及收益依赖硬件与负载，不能只根据流程图推算加速比。

## 3. 普通原生 HCCL AllToAllV

### 3.1 数据流

```text
                 Host / CPU                         Device / NPU
                     │                                   │
                     │                           router 计算专家选择
                     │                                   ↓
                     │                          token packing + 计数
                     │                                   │
                     │                         必要时交换收发计数
                     │                                   │
              本卡收发计数 ←──── D2H：仅计数 ──────────────┘
                     │
              send_counts[8]
              recv_counts[8]
              计算/提供各段偏移
                     │
              调用原生 HCCL AllToAllV
              传入：
              · send / recv 的设备地址
              · 各 peer 收发长度与偏移
              · dtype、通信域、stream
                     │
                     └────────── 提交通信任务 ───────────→ HCCL 执行引擎
                                                         （例如 CCU）
                                                              │
                                                       按收发描述搬运
                                                              │
                    源卡 HBM                                  │
               ┌────────────────┐                             │
               │ 发给 rank 0    │ ──────────────────→ rank 0 HBM
               │ 发给 rank 1    │ ──────────────────→ rank 1 HBM
               │       …        │
               │ 发给 rank 7    │ ──────────────────→ rank 7 HBM
               └────────────────┘
                                                              │
                                                    完成 / stream 顺序
                                                              │
                                                    专家计算或 unpack
```

原生接口需要本 rank 的收发长度，不要求应用提供完整的全局 C 矩阵。计数的来源取决于模型通信前处理；模型已有 CPU 计数或可以复用计数时，无需额外 D2H。不能把“交换计数”理解为原生 AllToAllV 必然替调用者完成的步骤。

图中是 API 层的逻辑流程。PyTorch 的 `all_to_all_single` 通过 split sizes 表达分段，底层再转换为通信所需描述；当前测试还使用 byte view 搬运 payload，以保持原始比特。

### 3.2 接收缓冲区如何组织

以 rank 2 为例，接收区按源 rank 排列，各源发送量可以不同：

```text
rank 0 send 中“发给 2”的一段 ──→ rank 2 recv 中“来自 0”的一段
rank 1 send 中“发给 2”的一段 ──→ rank 2 recv 中“来自 1”的一段
rank 2 send 中的 self 段     ──→ rank 2 recv 中的 self 段
                         …
rank 7 send 中“发给 2”的一段 ──→ rank 2 recv 中“来自 7”的一段
```

这里的箭头表示逻辑传递，不保证每条流只走一条物理直连边。原生 HCCL 的内部算法、执行引擎和物理路径由它自己的实现及运行配置决定；“native HCCL”不能直接等同于“强制 direct-only”。

## 4. 两种方案的关键区别

| 环节 | 原生 HCCL AllToAllV | 当前 EP8 EID 多路径 |
|---|---|---|
| 应用调用所需计数 | 本 rank 的发送、接收计数 | 全局一致的 `C[8,8]` |
| 应用侧 relay 规划 | 无本项目的贪心规划 | C++ 选择 relay 和转发行数 |
| 应用描述的分段 | 每个 peer 的发送/接收段 | 每个流可再拆为 direct 前缀、relay 后缀 |
| 通信控制 | HCCL 内部算法管理 | 已注册的 EID channels 和自定义 CCU kernel |
| Payload 流动 | NPU HBM → NPU HBM | NPU HBM → NPU HBM |
| 是否经 CPU 中转 payload | 否 | 否 |
| 结果是否自动 D2H | 否 | 否 |
| 路径收益保证 | 需测量 | 同样需测量，局部负载改善不保证端到端加速 |

对于本项目的 EID 后端，初始化时建立并缓存 communicator 相关的 channel 和 kernel，后续调用复用。每次调用仍有布局校验、内存 token 查询、参数构造和提交；“注册一次”不表示后续 host 开销为零。

## 5. 数据量与计时口径

使用 int64 计数时：

- 本 rank 的 8 个发送计数：64 B。
- 本 rank 的发送和接收计数合计 16 项：128 B。
- 全局 `C[8,8]` 共 64 项：512 B。

这些是计数本身的大小，不是整次 collective 在网络上的总字节数。小体积 D2H 也可能因为等待前序 NPU 任务完成而产生显著同步开销，不能按 512 B 的传输带宽直接估算时延。

计数与 payload 的关系为：

```text
rank r 发送 payload = sum_d C[r,d] × row_bytes
rank r 接收 payload = sum_s C[s,r] × row_bytes
全组跨卡有效 payload = sum_(s != d) C[s,d] × row_bytes
```

FP8、宽度 5120 时 `row_bytes=5120 B`；BF16 同宽时 `row_bytes=10240 B`。self 行通过本地复制处理，不应计入跨卡有效 payload。

prepared 复用已有选路决策，但每次仍提交通信；dynamic 使用本次 C 重新规划。若测试把计数交换/D2H 纳入 dynamic，应让 native 与 multipath 使用相同的公共前处理口径，并分别报告规划等额外开销。

当前修复后的 EID 后端已通过开发容器编译和 CPU 回归；这些图描述实现流程，不代表目标 CANN 9.3 八卡已完成数据正确性或性能验证。复测步骤见 [操作文档第 11 节](A5_CCU_92_93_ALLTOALL.md)。
