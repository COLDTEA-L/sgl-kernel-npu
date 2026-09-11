# A5 通信算子中的 CCU 与 AIV：调用链、API 与执行流程

本文总结 `sgl-kernel-npu` 中两类通信实现：

- **CCU 数据面**：原生 HCCL/MC2 CCU AllToAll、CCU dispatch/combine，以及当前的 CCU+URMA 多路径 AllToAll；
- **AIV/MTE 数据面**：dispatch/combine、NotifyDispatch 和显式 HBM 绕路 AllToAll。

重点不是只比较两个缩写，而是区分通信算子的三个独立维度。

## 1. 先区分三个维度

| 维度 | 可选项 | 回答的问题 |
|---|---|---|
| 算子入口/下发者 | Host、AICPU、AIV kernel | 谁准备参数并触发任务？ |
| 通信执行引擎 | CCU、AIV/MTE | 谁真正产生搬运指令和管理通信进度？ |
| 传输与路径 | HCCS、UBC_CTP/URMA、RoCE，hop-1/hop-2 | 数据最终从哪条链路到达对端？ |

因此，下列描述可以同时成立：

```text
Host 下发 ACLNN 算子
  -> 启动一个 __aicore__ AIV kernel
  -> 仅 AIV block 0 调用 Hccl<HCCL_SERVER_TYPE_CCU>
  -> 真正的数据传输由 CCU 执行
  -> CCU channel 底层使用 UBC_CTP/URMA route
```

这仍然是 **CCU 通信**，不能因为入口函数是 `__aicore__` 就称为 AIV 搬运。

判断数据面最可靠的依据是：

- 看 tiling 是否选择 `A5_CCU_ENGINE` 或 `AIV_ENGINE`；
- 看 device 侧是否调用 `Hccl<HCCL_SERVER_TYPE_CCU>` / CCU kernel；
- 或者是否由 AIV 直接对 `windowsIn` 对应的 GM 地址执行 `CpGM2GM`、`DataCopy` 等操作。

## 2. 两类实现的总调用链

### 2.1 原生 HCCL/MC2 CCU 调用链

仓内基线实现：

```text
Python / DeepEP
  -> deep_ep_cpp
  -> aclnnHcclAll2AllCcuGetWorkspaceSize
  -> ACLNN executor + tiling
       Mc2CcTilingConfig(...)
       SetCommEngine(A5_CCU_ENGINE)
       NnopbaseSetHcclServerType(..., CCU)
  -> aclnnHcclAll2AllCcu
  -> AIV kernel hccl_all2_all_ccu
       GetHcclContext<0>()
       Hccl<HCCL_SERVER_TYPE_CCU>::InitV2()
       SetCcTilingV2()
       block 0: Hccl::AlltoAll()
       Hccl::Wait()
       Hccl::Finalize()
  -> HCCL 根据拓扑和算法配置选择 executor/template/channel
  -> CCU 执行网络写、同步和完成通知
```

相关代码：

- `csrc/deepep/ops/op_host/op_api/aclnn_hccl_all2_all_ccu.cpp`
- `csrc/deepep/ops/op_host/hccl_all2_all_ccu_tiling.cpp`
- `csrc/deepep/ops/op_kernel/hccl_all2_all_ccu.cpp`
- `csrc/deepep/ops/op_kernel/moe_distribute_dispatch_v2_ccu.h`
- `csrc/deepep/ops/op_kernel/moe_distribute_combine_v2_ccu.h`

这里的 AIV kernel 是 MC2 的入口壳。它负责取得 context、发起 CCU collective 和等待完成；数据并不是由该
AIV block 用 GM/UB 搬运循环逐块复制。

### 2.2 直驱 HCOMM/CCU 多路径调用链

当前两卡多路径 AllToAll 没有把 route 选择交给 HCCL collective，而是显式取得多个 route/channel：

```text
Python Buffer.ccu_urma_multiroute_alltoall_out()
  -> deep_ep_cpp
  -> dlopen/dlsym 自定义 route library
  -> HcclCcuUrmaMultiRouteAllToAll()
  -> GetRouteResources()
       HcclRankGraphGetLayers()/GetLinks()
       按 --route-indices 选 route
       HcclThreadAcquireWithStream(COMM_ENGINE_CCU)
       HcclChannelAcquire()
       HcclCcuKernelRegister()
       HcclCcuKernelRegisterFinish()
  -> HcclCcuKernelLaunch()
  -> AllToAllMultiRouteKernel::Algorithm()
       NotifyRecord/NotifyWait：交换远端 output 地址和 token
       LocalCopyNb：self block
       WriteNb(route 0)：direct 分片
       WriteNb(route 2)：hop-2 分片
       WaitEvent：等待各非阻塞操作完成
       NotifyRecord/NotifyWait：peer 级完成握手
```

相关代码：

- `examples/a5_ccu_urma_route_probe/op_host/all_to_all_multiroute.cc`
- `examples/a5_ccu_urma_route_probe/op_kernel_ccu/all_to_all_multiroute_kernel.h`
- `examples/a5_ccu_urma_route_probe/op_kernel_ccu/all_to_all_multiroute_kernel.cc`
- `examples/a5_ccu_urma_route_probe/op_host/route_resources.cc`
- `tests/python/deepep/test_a5_ccu_urma_multiroute_all2all.py`

这个路径直接使用 HCOMM/CCU object API，所以可以指定 RankGraph 中已经发布的 route。它能控制的是
`route/channel`，而不是在算子里凭空创建一个底层不存在的 hop-2 路径。

### 2.3 AIV/MTE 窗口搬运调用链

以 NotifyDispatch、dispatch/combine 以及旧的显式 HBM 绕路实验为例：

```text
Python / DeepEP
  -> deep_ep_cpp
  -> ACLNN GetWorkspaceSize/execute
  -> tiling
       Mc2CcTilingConfig(...)
       SetCommEngine(AIV_ENGINE)
       NnopbaseSetHcclServerType(..., MTE)
  -> AIV kernel
       GetHcclContext<group>()
       取得 rank/rankSize 和通信窗口
       windowsIn[rank]
         或 remoteRes[rank].nextDevicePtr->windowsIn
       GlobalTensor::SetGlobalBuffer(remote_address)
       AIV block 切分 token/peer/path
       CpGM2GM / CpGM2GMPingPong / DataCopy
       flag 写入、轮询或同步
       接收侧读取、解包、combine/reduce
```

相关代码：

- `csrc/deepep/ops/op_kernel/notify_dispatch_a5.h`
- `csrc/deepep/ops/op_kernel/cam_moe_dispatch_normal_a5.h`
- `csrc/deepep/ops/op_kernel/cam_moe_combine_normal.h`
- `csrc/deepep/ops/op_kernel/data_copy.h`
- `csrc/deepep/ops/op_kernel/all2_all_detour_io_die.cpp`
- `csrc/deepep/ops/op_kernel/sync_collectives.h`

该模式下，AIV 自己计算远端地址、数据分片和同步位置，并通过 MTE/DataCopy 指令搬运。底层可能仍经 IO Die 和
UB/URMA 网络到达远端，但 AIV kernel 通常看到的是映射后的 GM/window 地址，而不是直接操作 EID 和 channel。

## 3. CCU 侧主要 API

### 3.1 Host/tiling 层

| API | 作用 |
|---|---|
| `Mc2CcTilingConfig(group, opType, algConfig)` | 为 MC2 collective 生成初始化和通信 tiling |
| `SetCommEngine(A5_CCU_ENGINE)` | 指定通信数据面使用 A5 CCU |
| `NnopbaseSetHcclServerType(executor, ...CCU)` | 告诉 ACLNN executor 拉起 CCU HCCL server/context |
| `HcclRankGraphGetLayers/GetLinks` | 枚举 HCCL/RankGraph 已发布的候选 route |
| `HcclThreadAcquireWithStream` | 获取与 ACL stream 有序的 CCU thread |
| `HcclChannelAcquire` | 针对一个 route/link 获取 CCU channel |
| `HcclCcuKernelRegister/Finish` | 把 CCU 表达式注册并完成编译/准备 |
| `HcclCcuKernelLaunch` | 将本次地址、token、offset、bytes 参数提交给 CCU kernel |

### 3.2 MC2 device wrapper

| API | 作用 |
|---|---|
| `GetHcclContext<group>()` | 取得 HCCL 为当前通信组准备的 context |
| `Hccl<HCCL_SERVER_TYPE_CCU>::InitV2` | 将 context 和 tiling 绑定到 CCU collective wrapper |
| `SetCcTilingV2` | 指定 collective 通信 tiling 在自定义 tiling 中的偏移 |
| `AlltoAll/AlltoAllV/BatchWrite` | 生成或提交对应的 HCCL collective |
| `Wait(handle)` | 等待该 collective 的完成条件 |
| `Finalize()` | 结束本次 HCCL/CCU 生命周期；应由提交任务的 block 调用 |

### 3.3 直驱 CCU kernel 原语

| API | 作用 |
|---|---|
| `CreateVariable(channel, index, ...)` | 获取通过 channel 交换得到的远端变量 |
| `NotifyRecord` | 向 peer 发布变量或完成通知 |
| `NotifyWait` | 等待 peer 发布的变量或通知掩码 |
| `LocalCopyNb` | CCU 非阻塞本地复制 |
| `WriteNb` / `ReadNb` | CCU 经指定 channel 执行非阻塞单边写/读 |
| `WaitEvent` | 等待某个非阻塞 copy/read/write 完成 |

HCCL 仓内部还会出现 `ccu::Write`、`ccu::Read`、`ccu::LocalCopy`、`GroupCopy`、event bitmask 等更底层
表达式。当前自定义包使用的是对外可构建的 HCOMM object API；二者底层都由 CCU 执行，但 API 层次、事件模型和
资源所有权不同。

## 4. AIV/MTE 侧主要 API

| API/对象 | 作用 |
|---|---|
| `SetCommEngine(AIV_ENGINE)` | 选择由 AIV/MTE 完成通信搬运 |
| `NnopbaseSetHcclServerType(...MTE)` | 为 AIV/MTE 模式准备通信窗口/context |
| `GetHcclContext<group>()` | 取得通信组 rank 信息和窗口元数据 |
| `windowsIn[rank]` | 当前 rank 可访问的某个通信窗口首地址 |
| `remoteRes[rank]...windowsIn` | 另一版 context 布局中的远端窗口首地址 |
| `GlobalTensor::SetGlobalBuffer` | 把本地或远端 GM 地址包装为 AscendC GlobalTensor |
| `CpGM2GM` | 使用分块/ping-pong 实现 GM 到 GM 搬运 |
| `CpGM2GMPingPong` | 显式 GM→UB→GM 流水搬运 |
| `DataCopy` / `CpGM2UB` / `CpUB2GM` | 更细粒度的 MTE 搬运原语 |
| `SetFlag/WaitFlag`、`pipe_barrier` | AIV pipeline 内部同步 |
| 窗口 flag + 轮询 | 跨 rank 的生产者/消费者同步 |

`windowsIn` 是此模式的前提之一。若 context 类型用错、server type 不匹配、窗口没有注册，或者 A5 当前通信模式
没有填充对应字段，读到的首地址可能是 0。CCU 模式不依赖 AIV 手动解引用这些窗口；它通过 channel、远端变量和
token 管理可访问地址。

## 5. 两类数据流的具体流程

### 5.1 CCU 原生 collective

1. Host 确定 group、rank、shape、dtype 和 collective 类型。
2. Tiling 选择 CCU engine，并把 HCCL 初始化/通信 tiling 写入算子 tiling。
3. ACLNN executor 创建 CCU server/context。
4. AIV 入口 kernel 的 block 0 调用 `Hccl::AlltoAll`。
5. HCCL 根据拓扑和算法配置选择 channel、分片及调度模板。
6. CCU 执行传输，设备侧 `Wait` 等待完成。
7. 提交 block 调用 `Finalize`，stream 后续任务才可安全消费输出。

### 5.2 CCU 多路径直驱

1. Host 从 RankGraph 枚举 route，读取 `layer/link/hop/die/address/BW_COEFF`。
2. 根据用户选择的 route indices 创建并缓存一个或多个 channel。
3. 为每个 channel 建立远端 output 地址/token 变量。
4. 按权重切分 peer payload，例如 route0:route2 = 2:1。
5. 在同一个 CCU kernel 中先后发出多次非阻塞 `WriteNb`；全部发出后再统一 `WaitEvent`。
6. 所有 route 完成后只做一次 peer 级完成握手。
7. 返回到用户 stream；输出布局仍保持普通两卡 AllToAll 语义。

多次 `WriteNb` 在表达式上是“先提交、后等待”，允许硬件重叠，但是否真正并行取决于 channel 是否落在独立可并发
资源上。一个 MindStudio `CCU Launch` 可以包含全部 route，不能仅凭只有一条泳道判断串行。

### 5.3 AIV/MTE dispatch/combine

1. Host 计算 token/expert/rank 布局及窗口大小，生成 AIV tiling。
2. AIV kernel 从 HCCL context 取得可访问的本地/远端窗口地址。
3. 各 AIV block 按 token、expert、peer 或 path 切片。
4. dispatch 可在 AIV 上完成计数、量化、排列和打包。
5. 通过 `CpGM2GM`/`DataCopy` 把数据写入远端窗口，随后发布 flag。
6. 接收侧等待 flag，再读取并解包。
7. combine 可把读取、反量化、累加/规约融合在同一 AIV kernel 中。

### 5.4 AIV 显式 HBM relay

当前旧绕路实验对每条路径分配一个 AIV block：

```text
source AIV
  -> CpGM2GM 写 relay rank 的 windowsIn/relay HBM
  -> 写 flag
destination AIV
  -> 等待 relay window flag
  -> CpGM2GM 从 relay HBM 读回 recv
```

这里可以明确选择哪张 rank 作为 relay，但 relay HBM 是可见的中间落点，路径包含一次写和一次读。它不同于
RankGraph 已配置的透明 hop-2：后者对 CCU 看起来仍是一次 `WriteNb`，中间 IO Die 转发不暴露为第二次算子读写。

## 6. 相同点

CCU 和 AIV/MTE 实现都需要：

- 同一通信组、正确的 rank 映射和一致的输入/输出语义；
- Host API、op definition、tiling、device 实现和 Python binding 的完整调用链；
- 可访问的本地/远端内存以及相应注册信息；
- 数据分片、offset、对齐和边界检查；
- 跨 rank 的同步与明确的完成条件；
- 与用户 stream 的顺序关系；
- warmup、批量迭代和设备侧 profiling 才能进行可信性能比较；
- 底层链路、路由和端点处于可用状态。

它们最终都在搬运 HBM 中的输入/输出数据，也都可能经过 IO Die。区别主要在于谁看到并控制“地址、channel、
同步、分片与进度”。

## 7. 不同点

| 对比项 | CCU 数据面 | AIV/MTE 数据面 |
|---|---|---|
| 真正执行通信 | CCU | AIV 发出 MTE/DataCopy 指令 |
| 远端资源表示 | channel + remote variable + token | 映射的 `windowsIn`/GM 地址 |
| 标准 collective | HCCL 自动选择算法和拓扑 | 算子开发者手写数据布局与搬运 |
| 显式 route | 直驱 HCOMM 时可选 RankGraph 已发布 route | 通常按远端/relay window 地址选择落点，不直接选 EID route |
| 多路径实现 | 多 channel 分片 `WriteNb` | 多 AIV block 对不同 window/relay 分片搬运 |
| 中间不落 HBM | 已配置的透明 hop-2 可做到 | 显式 relay 算法通常需要 relay HBM 写+读 |
| 计算通信融合 | 较弱，偏纯通信调度 | 强，适合 token pack、quant、unpack、reduce |
| 并行单位 | CCU thread/channel/event | AIV block + MTE pipeline |
| 同步 | notify、CCU event、collective handle | GM flag、原子量、pipe/block 同步 |
| 资源生命周期 | HCCL executor 或显式 thread/channel/kernel cache | HCCL window/context + AIV kernel 生命周期 |
| 性能特点 | 固定开销小、通信调度效率高 | 灵活，但易受 block 数、UB 流水、轮询和额外落盘影响 |
| Profiling | 常显示为一个 `CCU Launch` | 可看到 AIV kernel、block/MTE 活动 |
| 常见失败 | channel acquire 超时、route 不可用、token/notify 不匹配 | `windowsIn=0`、地址/offset 错、flag 永久等待、block 切分错误 |
| 可移植性 | 原生 HCCL 高；直驱内部 CCU API 低 | AscendC 算子可控，但依赖 context/window ABI |

## 8. 如何选择

优先使用原生 HCCL/CCU，当需求是：

- 标准 AllToAll/AllReduce 等 collective；
- 希望 HCCL 自动处理拓扑、channel、算法和资源生命周期；
- 追求较低固定开销和更稳定的通信性能。

使用直驱 HCOMM/CCU，当需求是：

- 必须显式选择 RankGraph 已发布的 route；
- 需要把同一个 peer payload 分到多个 CCU channel；
- 接受对 HCOMM/CCU API 和 CANN 版本更强的耦合。

使用 AIV/MTE，当需求是：

- 通信前后有大量 token/expert 排列、量化、反量化或规约，适合融合；
- 需要由不同 AIV block 精确处理不同 token/path；
- 已确认通信窗口首地址有效；
- 可以接受手工设计同步、分片和流水。

如果目标是“明确指定某些卡中转且不落中转卡 HBM”，仅把某个 relay rank 写入 AIV 算子参数并不够。需要底层
RankGraph/UVS/HIXL 已经发布对应透明 hop route，然后由 CCU/HCOMM 选择该 channel；否则只能实现显式 HBM
relay，或者扩展底层路由控制面。

## 9. 阅读代码的推荐顺序

### CCU 原生路径

1. `csrc/deepep/ops/op_host/hccl_all2_all_ccu_tiling.cpp`
2. `csrc/deepep/ops/op_host/op_api/aclnn_hccl_all2_all_ccu.cpp`
3. `csrc/deepep/ops/op_kernel/hccl_all2_all_ccu.cpp`
4. `csrc/deepep/ops/op_kernel/moe_distribute_dispatch_v2_ccu.h`
5. `csrc/deepep/ops/op_kernel/moe_distribute_combine_v2_ccu.h`

### CCU 多路径直驱

1. `examples/a5_ccu_urma_route_probe/op_host/route_resources.cc`
2. `examples/a5_ccu_urma_route_probe/op_host/all_to_all_multiroute.cc`
3. `examples/a5_ccu_urma_route_probe/op_kernel_ccu/all_to_all_multiroute_kernel.h`
4. `examples/a5_ccu_urma_route_probe/op_kernel_ccu/all_to_all_multiroute_kernel.cc`
5. `tests/python/deepep/test_a5_ccu_urma_multiroute_all2all.py`

### AIV/MTE 路径

1. `csrc/deepep/ops/op_host/notify_dispatch_tiling.cc`
2. `csrc/deepep/ops/op_kernel/notify_dispatch_a5.h`
3. `csrc/deepep/ops/op_kernel/cam_moe_dispatch_normal_a5.h`
4. `csrc/deepep/ops/op_kernel/cam_moe_combine_normal.h`
5. `csrc/deepep/ops/op_kernel/data_copy.h`
6. `csrc/deepep/ops/op_kernel/all2_all_detour_io_die.cpp`

原生 CCU AllToAll 与当前多路径 CCU AllToAll 的逐项对比，另见：

```text
docs/A5_CCU_NATIVE_VS_MULTIROUTE_ALLTOALL.md
```
