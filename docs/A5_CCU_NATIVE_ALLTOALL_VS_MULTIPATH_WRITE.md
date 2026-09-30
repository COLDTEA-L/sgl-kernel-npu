# A5 原生 HCCL CCU AllToAll 与早期多路径 Write 对比

## 1. 文档目的

本文对比两套容易混淆、但资源模型并不相同的实现：

1. HCCL 原生 CCU AllToAll；
2. `a5_ccu_urma_route_probe` 中早期的多路径 `WriteNb` 实验。

重点回答：为什么早期 Write 已能通过多个显式 Channel 搬运数据，却不能只增加一个 `op_def`、
`tiling.cpp` 和 AIV kernel 就直接变成标准 AllToAll 算子。

结论是：

> 多路径 Write 的选路、分片和并发下发逻辑可以复用；它的 host 直调、资源索引、同步协议、
> stream cache 和数据语义不能原样进入 HCCL 标准算子。标准化需要把这些能力翻译到 HCCL 的
> selector、executor、`CalcRes`、`TemplateResource`、`KernelRun/FastLaunch` 生命周期中。

## 2. 两者不是同一个算子语义

### 2.1 原生 AllToAll

对两卡、每个 peer 数据量为 `B` 的场景，rank `r` 的输入为：

```text
send[0]：发给 rank0 的 B Bytes
send[1]：发给 rank1 的 B Bytes
```

输出为：

```text
recv[0]：来自 rank0 的 B Bytes
recv[1]：来自 rank1 的 B Bytes
```

每个 rank 的输入和输出总量都是 `2B`。网络只搬运发给另一张卡的一个 slice，本 rank slice
在本地完成复制。

### 2.2 早期 MultiRouteWrite

`HcclCcuUrmaMultiRouteWrite(sendBuf, recvBuf, sendCount, ...)` 的输入只有一份 `B` 字节数据，
接收区大小为 `rankSize * B`。每个 rank：

```text
本地：send[0:B] -> recv[rank * B : (rank + 1) * B]
远端：send[0:B] -> peer.recv[rank * B : (rank + 1) * B]
```

因此它更接近两卡 all-gather/write 穿刺，而不是完整 AllToAll。`--remote-only` 还会跳过
host 侧 `aclrtMemcpyAsync` self copy，只保留远端 Write，用于隔离网络路径。

早期 Write 证明的是：

```text
同一 src/dst rank
  + 多个 CommLink/EID pair
  + 多个 ChannelHandle
  + 每条 Channel 搬运不重叠的数据片段
```

可以形成显式多路径数据面；它没有证明该 host API 已满足标准 AllToAll 的算子协议。

## 3. 原生 HCCL CCU AllToAll 调用链

### 3.1 两种上层入口

PyTorch 原生基线：

```text
torch.distributed.all_to_all_single
  -> ProcessGroupHCCL
  -> HcclAlltoall
  -> HCCL operator dispatch
```

标准 MC2/Ascend C 通信算子入口：

```text
Python / ACLNN
  -> 标准 OpDef + tiling
  -> AIV/MC2 kernel
  -> Hccl<HCCL_SERVER_TYPE_CCU>::InitV2
  -> Hccl::AlltoAll
  -> Hccl::Wait
  -> Hccl::Finalize
```

两条入口最终都必须向 HCCL 提交合法的 `HCCL_CMD_ALLTOALL`、数据描述、通信域和 stream 信息。

### 3.2 HCCL 控制面

```text
HCCL AllToAll request
  -> AlltoAllAutoSelector::SelectCcuScheduleAlgo
  -> 按拓扑、rank 数、数据量选择算法
       CcuAlltoAllMesh1D
       CcuAlltoAllMesh1DMultiJetty
       CcuAllToAllMesh1DConcurrent
       CcuAllToAllMesh1D2Die / Mesh2Die
  -> InsV2AlltoAllVSoleExecutor
  -> topology match / hierarchy info
  -> template::CalcRes
  -> AlgResourceRequest
       thread request
       channel descriptors
       CCU kernel info
  -> HCCL resource allocator
  -> TemplateResource
       threads
       acquired channels
       registered CCU kernels
```

原生模板通过 `CalcChannelRequestMesh1DWithPriorityTopo` 生成 Channel 请求，Channel、thread、
kernel 的创建和销毁由 HCCL communicator/resource context 统一管理。

### 3.3 HCCL 执行面

首次执行：

```text
template::KernelRun
  -> 读取 input/output、token、send/recv counts、displacements
  -> 构造 CCU task arguments
  -> HcommCcuKernelLaunch
  -> 保存 CcuKernelSubmitInfo
```

图复用或 FastLaunch：

```text
template::FastLaunch
  -> 使用 TemplateFastLaunchCtx
  -> 修正本次 input/output 地址
  -> 复用已注册 kernel/channel/thread
  -> HcommCcuKernelLaunch
```

### 3.4 原生 CCU kernel

以 `CcuAllToAllMesh1DMultiJettyKernel` 为例：

```text
InitResource
  -> channel resource 0/1/2 = input/output/token

LoadArgs
  -> buffer、slice、stride、offset、jetty slice

PreSync
  -> 每个 Channel 发布 output/token
  -> NotifyWait(CKE 0, output bit | token bit)

DoAllToAll
  -> 为所有 peer 生成 remote Write
  -> GroupCopy 本 rank slice
  -> 所有 Write/GroupCopy 下发后统一 EventWait

PostSync
  -> 每个 Channel NotifyRecord(CKE 1, 1 << 3)
  -> 每个 Channel NotifyWait(CKE 1, 1 << 3)
```

当前源码中 `STUB_JETTY_NUM=1`。MultiJetty kernel 已包含 jetty slice、event mask 和循环框架，
但默认并不等价于“一个 peer 自动使用多条用户指定 relay”。原生 AllToAll 的 Channel 主要按 peer
和 HCCL 拓扑策略建立；显式多 relay 仍需要额外 Channel catalog。

## 4. 早期多路径 Write 调用链

### 4.1 Python 到独立 route-probe SO

```text
test_a5_ccu_urma_multiroute_write.py
  -> deep_ep.Buffer.ccu_urma_multiroute_write(send)
  -> DeepEP C++ dlsym
  -> liba5_ccu_urma_route_probe.so
  -> HcclCcuUrmaMultiRouteWrite
```

它不进入 HCCL AllToAll selector/executor，也没有 `HCCL_CMD_ALLTOALL` 对应的标准资源上下文。

### 4.2 路径枚举与资源创建

```text
GetRouteResources
  -> HcclRankGraphGetLayers
  -> HcclRankGraphGetLinks
  -> SelectRoute / 显式 EID provider
  -> 为每条路径构造 HcclChannelDesc
  -> HcclThreadAcquireWithStream
  -> HcclChannelAcquire
  -> HcclCcuKernelRegister
  -> HcclCcuKernelRegisterFinish
  -> thread-local cache(comm, stream, routeKey)
```

这些副作用发生在普通 host 函数第一次调用时。资源由 route-probe 自己缓存，不属于 HCCL 标准
executor 的 `AlgResourceRequest`/`TemplateResource`。

### 4.3 Host 分片与 launch

```text
HcclCcuUrmaMultiRouteWrite
  -> host self copy（可由 --remote-only 跳过）
  -> GetTokenInfo(send/recv)
  -> 按路径权重和 256B 对齐切分 B Bytes
  -> RouteTaskArg
       input/output
       inputToken/outputToken
       sourceOffset[i]
       remoteOffset[i]
       pathBytes[i]
  -> 必要时同步 main thread 与 die1 route thread
  -> HcclCcuKernelLaunch
```

### 4.4 RouteKernel

```text
Load(input/output/inputToken/outputToken)
  -> 每条 Channel 创建 remote output/token variable
  -> 每条 Channel 发布并等待 output/token
  -> 对所有 Channel 依次 WriteNb
  -> 所有 WriteNb 提交后统一 WaitEvent
  -> 每条 Channel completion NotifyRecord/NotifyWait
```

所有 `WriteNb` 都位于第一个 `WaitEvent` 之前，这是多个 Channel 能同时 in-flight 的关键。

## 5. 关键差异

| 项目 | 原生 HCCL CCU AllToAll | 早期多路径 Write |
|---|---|---|
| 算子语义 | 标准 AllToAll | 单输入 slice 的 remote write/all-gather 穿刺 |
| 标准 op type | `HCCL_CMD_ALLTOALL` | 无 |
| 算法选择 | HCCL selector | 环境变量/manifest 指定路径 |
| topology/hierarchy | executor 统一计算 | route-probe 自己枚举 RankGraph |
| 资源申请 | `CalcRes -> AlgResourceRequest` | host 首次调用直接 Acquire/Register |
| 资源所有者 | communicator/resource context | 独立 SO 的 thread-local/global cache |
| stream 生命周期 | HCCL graph/fast-launch 管理 | cache key 中手工绑定 stream |
| Channel 资源槽 | 原生 input=0、output=1、token=2 | 私有 output=0、token=1 |
| pre-sync | CKE 0 上的 output/token bitmask | 两个独立 notify index |
| self slice | CCU `GroupCopy` | host `aclrtMemcpyAsync`，或跳过 |
| remote copy | `ccu::Write` | object API `WriteNb` |
| completion | 原生 event mask + CKE 1 bit 3 | `CompletedEvent` + 私有 notify index 2 |
| FastLaunch | 原生支持并缓存 submit info | 原始 Write API 不具备标准 FastLaunch ctx |
| dtype/rank | HCCL 算法定义 | 穿刺阶段仅 FP32、两卡 |
| 错误/销毁/DFX | HCCL 统一管理 | 独立 SO 自行处理 |

`ccu::Write` 和 `WriteNb` 最终都描述 CCU 单边写，差别不在“能不能搬数据”，而在它们所属的
资源协议和指令生成 API。不能因为两者都叫 Write，就互换 Channel 资源索引、notify 约定或 event。

## 6. 为什么不能直接给原来的 Write 套一个标准算子外壳

### 6.1 tiling 不能承担通信资源初始化

标准 Ascend C tiling 的职责是计算 block、workspace 和 tiling data。它可能在编译、capture、shape
推导或图优化阶段执行，不能安全地在其中：

```text
HcclThreadAcquireWithStream
HcclChannelAcquire
HcclCcuKernelRegister
```

这些接口依赖真实 communicator、执行 stream 和明确的销毁时机；host handle 也不能作为普通
tiling data 序列化给 device kernel。

### 6.2 AIV kernel 不能替代 host 资源管理

标准 AIV kernel 可以读写 GM、生成 MC2/HCCL 请求，但不能直接执行 host 侧 ChannelAcquire 或
注册 CCU kernel。仅增加一个 AIV shell 不会让 route-probe 的 global/thread-local cache 进入
HCCL resource context。

### 6.3 图 capture 使用的 stream 可能不同

早期 prepared-plan 实验已经观察到，npugraphs/ACLGraph warmup 和 capture 可能使用独立 stream。
原始 Write 按 `(comm, stream, routeKey)` 缓存 thread/channel/kernel。直接捕获 host API 容易：

- 在默认 stream 上创建资源，却在 graph stream 上复用；
- capture 时再次创建 Channel；
- replay 使用已经失效或未序列化的 handle；
- communicator 销毁后 cache 仍持有资源。

HCCL `TemplateResource` 和 `TemplateFastLaunchCtx` 正是用来管理这类生命周期。

### 6.4 私有 Channel 协议与原生协议不兼容

早期 route-probe 约定：

```text
output=0, token=1
output notify index=0
token notify index=1
completion notify index=2, mask=1
```

原生 AllToAll 约定：

```text
input=0, output=1, token=2
pre-sync CKE=0, mask=(1<<1)|(1<<2)
post-sync CKE=1, mask=1<<3
```

把旧 RouteKernel 直接放入原生 Channel 后，会把 input 地址当作 remote output、把 output 地址当作
token，或者等待错误的通知位。ABI 4 到 ABI 7 的有卡调试已经证明，这类错误通常表现为：

```text
Channel 和 kernel launch 成功
  -> 首次 warmup 卡在 torch.npu.synchronize()
```

所以问题不是缺少一个标准 `op_def`，而是旧 kernel 与标准 Channel 的协议不相同。

### 6.5 原始 Write 没有完整 AllToAll 元数据

标准 AllToAll 需要将 send/recv counts、displacements、dtype、rank size、buffer offset 和 self slice
纳入统一语义。旧 Write 只有一份 sendCount 和手工计算的 remote offset，不能直接覆盖更多 rank、
非均匀 AllToAllV、不同 dtype 或 inplace 检查。

### 6.6 HCCL 对旧资源不可见

独立 SO 自行创建的 Channel/thread/kernel 不在 HCCL 算子的 `AlgResourceRequest` 中。HCCL 无法：

- 为图执行序列化这些资源；
- 生成正确的 FastLaunch cache；
- 参与统一超时和异常恢复；
- 在 communicator 销毁时可靠回收；
- 将资源计入标准 profiling/DFX。

因此，直接把旧 host API 放进 ACLNN 包装，最多得到“可在 eager host 调用、可能被某些图捕获”的
扩展，不能得到资源生命周期完整的标准 HCCL 通信算子。

## 7. 哪些部分可以复用

| 早期 Write 能力 | 标准化后的去向 |
|---|---|
| relay physical ID -> EID pair 解析 | communicator 初始化/manifest resolver |
| 一个 peer 的多个 ChannelDesc | HCCL template `CalcRes` 的 Channel catalog |
| 256B 对齐的权重分片 | `FillPathArgs`/动态 `pathPolicy` |
| 先提交全部 Write、再统一等待 | 原生 CCU primitive kernel |
| serial/concurrent 对照 | 性能与并发回归测试 |
| HCCN 端口计数 | 物理 relay 因果验证 |
| correctness/profiling 脚本 | 标准算子测试矩阵 |

以下部分不能原样复用：

```text
独立 dlsym host entry
thread-local RouteResources cache
host aclrtMemcpyAsync self copy
route-probe 0/1 variable layout
私有 notify index 0/1/2 协议
FP32-only、两卡 write 数据布局
```

## 8. 已完成的尝试及其边界

本节按时间顺序记录已经验证过的方向。这里的“不可行”只表示该实现路径不能同时满足当前的
标准算子、显式路径和图执行要求，不表示 A5 硬件没有 relay 或多路径能力。

### 8.1 早期独立多路径 Write

做法：在 `liba5_ccu_urma_route_probe.so` 中直接调用旧版对象接口：

```text
HcclRankGraphGetLayers/GetLinks
  -> HcclChannelAcquire
  -> HcclCcuKernelRegister/Finish
  -> HcclCcuKernelLaunch
  -> CcuRep::WriteNb
```

结果：route0、route2 以及多条显式 relay Channel 均能正确搬运数据；多个 `WriteNb` 在第一次
`WaitEvent` 前提交时能够并发。后续通过拓扑 JSON 和 EID inventory 合成的 CommLink 还证明：

```text
指定 relay physical ID
  -> 求 src->relay 与 relay->dst 的拓扑边
  -> 选出相应 destination EID
  -> 构造 HcclChannelDesc
  -> HcclChannelAcquire 成功
  -> 数据正确
```

保留价值：它验证了显式 relay 的控制点是 CommLink/Endpoint EID pair，而不是必须修改 UBUS
全局路由表；其路径解析、分片和并发下发逻辑都可以复用。

不能直接作为最终方案的原因：它是独立 host API、私有 cache 和私有同步协议，不进入标准
AllToAll selector/executor、`CalcRes`、`TemplateResource` 和 FastLaunch 生命周期；语义也只是
两卡 remote-write/all-gather 穿刺。

### 8.2 UBUS route table FORCE/ADD

黑箱分析确认 UBUS route table 的核心含义是：

```text
destination CNA -> egress-port bitmap
```

理论上把源 entity 的目的 CNA 映射到 relay first-hop port，可以强制 `src -> relay -> dst`；增加
多个 port bit 还可用于验证底层 bitmap 是否是现有 multipath selector。

未采用原因：该表是整机共享硬件状态，不是 per-process、per-communicator 或 per-Channel 状态。
加载修改后的 `ubus.ko` 或覆盖 route entry 会影响公共服务器上的其他任务，也不能满足每个算子
独立选择路径的要求。这个方向适合专机上的硬件机制验证，不适合作为算子控制面。

### 8.3 普通用户态 URMA EID 和 TP selector

尝试过：

- 用 `urma_admin`/`liburma` 枚举和绑定 EID；
- 扫描 EID pair 并结合 HCCN counter 推测 relay；
- 比较 `flow_label`、`spray_en`、`port_id`、`route_addr_idx` 等 TP 属性；
- hook `urma_cmd_get_tp_list()`、`urma_get_tp_attr()` 等公开边界。

结果：HCCL route endpoint 中的完整 `df12` EID 不一定出现在普通用户态可直接查询和绑定的本地
EID 清单中；route0/route2 的 `GET_TP_LIST` 返回资源集合和公开 TP 属性也没有稳定的路径差异。
后续追踪表明，不同 candidate 会在 `HcclChannelAcquire -> URMA import` 阶段激活不同 TP/TPN，
但 path matcher 和 TPN 到物理路径的映射位于 HCCP/MUE/driver 私有控制面。

未采用原因：公开结构中“存在字段”不等于用户能用它强制 HCCL Channel 走某个物理出口；普通
URMA TP 的行为也不能直接等价于 HCCL/MUE provision 的 TP。继续只追 `GET_TP_LIST` 无法得到可编程
的 relay 创建接口。

### 8.4 ChannelDesc/EndpointDesc 因果替换

针对 candidate0 和 candidate2 做了以下交叉实验：

```text
只换 local endpoint       -> Acquire 失败
只换 remote endpoint      -> Acquire 失败
同时换 endpoint pair      -> Acquire 成功
同时换 CommAddr pair      -> Acquire 成功
只换 location             -> 不能解释路径切换
```

结果说明完整 CommAddr/Endpoint pair 是匹配一个已 provision path object 的 key；单端混合不是合法
路径。该实验定位了“既有路径如何被选择”，但不能凭空生成任意 relay path。

保留价值：它指导后续不再修改 TP handle，而是在 communicator/RankGraph 资源准备阶段构造完整、
一致的 EID pair。合成 CommLink 实验最终证明：只要 EID pair 与实际拓扑边一致，candidate 不必先由
`HcclRankGraphGetLinks` 返回，也能被 `HcclChannelAcquire` 接受。

### 8.5 HCCN counter、profiling 与显式多 relay

通过 topology JSON、端口/EID inventory 和 HCCN counter，已经验证了 `2 -> 4 -> 3` 等显式 relay
链；进一步构造两条、三条和四条 relay Channel 并发时，性能从约 86 us 降至 74 us、69 us，证明
显式增加可用路径能够带来收益。

HCCN counter 能证明指定物理端口在同一 workload 窗口内承载流量，CCU kernel 中“所有 WriteNb
先提交、之后统一等待”的结构提供并发因果条件；counter 本身不提供 cycle-level overlap。

### 8.6 prepared-plan `torch.ops` 方案

做法：图外创建 CommLink/Channel/CCU kernel，返回 `plan_handle`；图内 `torch.ops` 只引用已经准备
好的 plan。npugraphs 和 ACLGraph 已完成首次 capture 与重复 replay，证明 host 资源只创建一次、
图 replay 可以复用准备好的显式路径。

未作为最终标准算子的原因：该接口仍是 DeepEP 自定义 `torch.ops` 加独立 route-probe 资源注册，
不是完整的标准 Ascend C/HCCL AllToAll；HCCL communicator 看不到这些资源，无法统一管理
FastLaunch、异常恢复、profiling 和销毁。

保留价值：当标准 HCCL 改造尚未完成时，它是功能正确、可入图的回退实现，也证明“图外准备、
图内复用”这一生命周期设计是成立的。

### 8.7 AIV -> HBM CommandBlock -> 常驻 CCU worker

这个方向试图用标准 Ascend C AIV kernel 写 HBM command，再由常驻 CCU worker 轮询或 doorbell
唤醒，从而同时获得标准算子外壳和 device-time 动态 path mask。

未采用原因：当前软件栈没有公开、可验证的 AIV->CCU doorbell；自定义 HBM 地址虽可分配，但
不能仅凭地址证明 CCU worker 能稳定读取、被唤醒并与图 stream 建立完成语义。轮询会长期占用 CCU
资源，也缺少标准的退出和异常恢复协议。原生 HCCL 内部可能有类似机制，但没有暴露成可供外部
自定义算子复用的接口。

### 8.8 标准 Ascend C 外壳直接申请 HCCL/MC2 资源

做法：实现 OpDef、tiling、AIV kernel 和 ACLNN，并让 MC2/HCCL 按 `HCCL_CMD_ALLTOALL + CCU_SCHED`
为自定义模板分配资源。

实验中曾出现：

```text
HcclAllocComResourceByTiling / tiling function 失败
CcuInstructions isEmpty
运行时返回 NOT_SUPPORT/ret=5
```

根因不是 OpDef 或 tiling 形式不标准，而是自定义 CCU 模板最终依赖目标 HCOMM 的 CCU 开发 ABI。
`libmc2_client.so` 只能消费 HCOMM 已提供的资源能力，不能替代缺失的底层接口。

### 8.9 基于新版 `HcommCcu*`/`Ccu*` 接口修改 HCCL

为接入 HCCL 原生生命周期，曾实现自定义 AllToAll selector/template/kernel，并使用：

```text
HcommCcuKernelRegisterStart/Register/End
HcommCcuKernelLaunch
CcuVariableCreateByChannel
CcuNotifyRecord/Wait
CcuWriteVariableWithNotify
CcuWriteMemToMem
CcuEventWait
```

ABI 4~7 是本项目用于标记 selector、资源槽、Channel 生命周期和 completion 修正的版本号，不是
厂商 HCOMM ABI。源码能编译，是因为 HCCL compatibility/dlsym 层允许弱符号构建；编译成功并不
代表目标 runtime 提供这些函数。

在目标有卡环境 CANN `9.1.T560` 上直接检查 `libhcomm.so`：除 `HcclChannelAcquire` 外，上述新版
接口均未导出。相同检查在开发 Docker 的 beta/T500 环境也得到一致结果；compat support flag 若为
0，则可最终确认运行时不支持。因此 ABI7 的卡死不能继续只按 resource slot 或 notify bug 调试：
当前软件包没有承载这套新模板所需的底层执行接口。

这否定的是“依赖新版 HCOMM CCU ABI 的自定义模板”，不是“修改 HCCL”本身。目标 CANN 当前仍
提供早期 Write 和原生 CCU AllToAll 已实际使用的旧对象接口：

```text
HcclCcuKernelRegister
HcclCcuKernelRegisterFinish
HcclCcuKernelLaunch
hcomm::CcuRep / WriteNb
```

后续应改造已经能在 T560 运行的原生 CCU AllToAll，而不是继续要求不存在的新版 ABI。

## 9. 原生 HCCL CCU 增量改造（ABI 10）

前述控制面设计保留不变：标准 Ascend C/MC2 仍以 `HCCL_CMD_ALLTOALL + CCU_SCHED` 进入
HCCL selector/template，路径仍由 EID pair/ChannelDesc 在资源准备阶段决定。ABI 8 首先替换
T560 无法执行的数据面后端：

```text
标准 ExplicitMultipathAll2AllCcu
  -> OpDef / infer / tiling / ACLNN
  -> MC2 以 HCCL_CMD_ALLTOALL + CCU_SCHED 调用 HCCL
  -> AlltoAll selector 选择 CcuExplicitMultipathAllToAll2Rank
  -> template::CalcRes
       native direct ChannelDesc
       + manifest relay ChannelDesc
  -> CalcRes 保存 direct/relay ChannelDesc catalog
  -> 原生资源层按 notifyNumOnMainThread=0 创建 main thread
  -> KernelRun 首次执行
       复用 TemplateResource.threads[0]
       HcclChannelAcquire
       HcclCcuKernelRegister / Finish
       拒绝重复 endpoint pair / 重复 Channel handle
       按 (comm, stream, plan_id, manifest 内容摘要) 缓存资源
       保存 HCCL input/output base offset
  -> HcclCcuKernelLaunch
  -> hcomm::CcuKernel / CcuRep
  -> direct/relay WriteNb 全部提交后统一 WaitEvent
  -> executor 保存 zero-submitInfo engine context
  -> FastLaunch 复用同一资源并更新 buffer/policy task args
```

ABI 9 修复了第一次有卡 ABI 8 单 case 暴露的 main-thread 生命周期问题：ABI 8 先以 0 notify
创建 native main thread，又在 KernelRun 对同一 stream 二次 Acquire 3-notify thread，首次 launch
后卡在 device synchronize。ABI 9 改为由 `CalcRes` 声明 3 个 notify，并直接复用原生资源层返回的
thread。ABI 9 的第一次有卡结果进一步证明两个 rank 都能完成 communicator/Buffer 初始化，
`run_op()` 也能异步返回，但都卡在紧随其后的 `torch.npu.synchronize()`。静态核对发现 ABI 9
把 output/token/completion 三个 **Channel notify** 错当成了 main-thread notify，并令
`notifyNumOnMainThread=3`。原生一 die CCU AllToAll 在此处使用 0，且本项目 CCU kernel 没有产生
任何 thread-level notify。ABI 10 因此保留原生 thread 复用，但把 thread notify 数量改回 0。

该实现已经在开发容器完整编译，并确认 `libhccl.so` 对
`HcclCcuKernelRegister/Finish/Launch` 形成强动态引用；它不再要求 T560 缺失的
`HcommCcuKernelRegisterStart/Register/End`。项目扩展版本提升为 ABI 10。

当前尚不能宣称有卡完成。第一版主动限制 `localDie == 0`；资源缓存清理、失败恢复、标准算子首次
执行、ACLGraph FastLaunch/replay 和 HCCN 物理路径仍需有卡验证。当前已补上旧对象模板专用的
zero-submitInfo engine context，使 replay 可以进入模板 `FastLaunch()`；但这一行为仍需有卡验证。
`CalcRes` 不向通用 allocator
提交 `ccuKernelInfos`，因为该 allocator 正是缺失新接口的调用方；旧对象资源在 `KernelRun` 首次
执行时创建。这是对目标 T560 的兼容接入点，不应描述成已经完全等同于上游原生
`TemplateResource` 生命周期。

`plan_id` 和路径目录固定在一次 communicator/capture 生命周期内；缓存键已经加入 manifest
内容摘要，覆盖文件不会再静默复用旧 catalog，而会 fail-fast。图内 `pathPolicy` 仍可在
replay 前改变已准备路径的启用状态与权重，但不能在 replay 中创建新 relay。下一步不是继续逆向
MUE/UVS，而是先在有卡 T560 验证 ABI 10 的 zero-notify 旧对象资源生命周期与 FastLaunch。

为避免错误计划演化为 device wait，当前 HCCL 还会在 launch 前拒绝：重复的 endpoint pair、
空或重复的 acquired Channel handle，以及超过 15 条路径的 policy。SGL 产品接口保持更保守的
1 direct + 12 relay 上限。两个 rank 仍必须使用同一 manifest 顺序和同一动态 policy。

## 10. 相关源码

HCCL 原生 CCU AllToAll：

```text
hccl/src/ops/all_to_all_v/selector/alltoall_auto_selector.cc
hccl/src/ops/all_to_all_v/executor/ins_v2_all_to_all_v_sole_executor.cc
hccl/src/ops/all_to_all_v/template/ccu/ccu_temp_all_to_all_mesh1d_multi_jetty.cc
hccl/src/ops/all_to_all_v/template/ccu/kernel/ccu_kernel_all_to_all_mesh1d_multi_jetty.cc
```

早期多路径 Write：

```text
examples/a5_ccu_urma_route_probe/op_host/route_probe.cc
examples/a5_ccu_urma_route_probe/op_host/utils.cc
examples/a5_ccu_urma_route_probe/op_kernel_ccu/route_kernel.cc
tests/python/deepep/test_a5_ccu_urma_multiroute_write.py
```

此前基于新版 HCOMM ABI 的标准显式多路径 AllToAll 尝试：

```text
csrc/deepep/ops/op_host/explicit_multipath_all2all_ccu_def.cpp
csrc/deepep/ops/op_host/explicit_multipath_all2all_ccu_tiling.cpp
csrc/deepep/ops/op_kernel/explicit_multipath_all2_all_ccu.cpp
hccl/src/ops/all_to_all_v/template/ccu/ccu_temp_explicit_multipath_alltoall.cc
hccl/src/ops/all_to_all_v/template/ccu/kernel/ccu_kernel_explicit_multipath_alltoall.cc
```
