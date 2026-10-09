# A5 CCU+URMA prepared-plan 显式多路径 AllToAll 实现说明

## 1. 结论

当前实现采用“图外资源准备 + 图内缓存 launch”，不修改系统 `libhccl.so`：

```text
Prepare(comm, stream, plan_id, direct, relays)
  -> 显式构造 CommLink
  -> Channel Acquire
  -> Thread Acquire with stream
  -> CCU kernel register
  -> plan_handle

Execute(send, recv, plan_handle, path_weights)
  -> 以当前 stream 查找已绑定资源
  -> 生成本次 buffer/offset/weight task args
  -> launch 已注册 CCU kernel
  -> direct/relay Channel 并发 WriteNb
```

这样避开了此前两个已确认失败点：

1. 不让 AIV `hccl.AllToAll()` 进入 T560 闭源 MC2 server 后再期待 host HCCL patch
   生效；
2. 不在 ACLGraph capture/replay 内创建 Channel、Thread 或 CCU kernel。

## 2. 模块边界

### 2.1 系统 HCCL

只负责建立正常 communicator。系统 HCCL 不感知显式 relay plan，也不需要替换。

### 2.2 route runtime

`liba5_ccu_urma_route_probe.so` 是薄资源管理层，复用 T560 已验证可用的 HCOMM/CCU
对象接口：

```text
HcclRankGraphGetLayers/GetLinks       取得 direct candidate
synthetic relay manifest             取得显式 relay endpoint pair
HcclChannelAcquire                   建 Channel
HcclThreadAcquireWithStream          将资源绑定执行 stream
HcclCcuKernelRegister/Finish         注册 CCU kernel
HcclCcuKernelLaunch                  执行缓存 kernel
```

它不是另一套 HCCL：不实现 collective rendezvous、rank 管理、拓扑发现或容错，只保存
显式路径计划与已经建立的通信资源。

### 2.3 DeepEP bridge

DeepEP 通过 `dlopen/dlsym` 加载 route runtime，提供 Python/torch dispatcher 边界：

```text
Buffer.prepare_ccu_urma_explicit_multipath_plan(...)
Buffer.bind_ccu_urma_explicit_multipath_plan(handle)
Buffer.ccu_urma_prepared_multipath_alltoall_policy_out(...)

torch.ops.deep_ep.ccu_urma_prepared_multipath_alltoall_policy(
    send, plan_handle, path_weights)
```

Meta kernel使 torch dispatcher/图前端能够推导输出形状与 dtype。

DeepEP ABI 10 在直接 ACL/HCOMM 调用前使用 `getCurrentNPUStream().stream()`。
torch_npu 的 `stream(false)` 只返回原始流，不能保证之前的 `fill/stack/add_` 已从
host queue 提交；原始 ACL 调用可能抢先读取输入。`stream()` 清空 host 提交队列，
让 tensor producer、D2D self-copy、CCU launch 按流顺序提交；它不等待 device 完成。
这是 PyTorch bridge 的要求，单独 C++ probe 没有这层 torch_npu host queue。

当前 route ABI 为 12、DeepEP ABI 为 11，保留上述提交顺序修复。bridge 增加 tensor DeviceGuard、
同 device/不重叠检查和 caching allocator `recordStream`。后者保证异步访问期间的存储
生命周期，不代替生产流与执行流的依赖；跨流输入需要调用者先 `wait_stream`。

route ABI 10 的本地 slice 使用 D2D copy，peer slice 使用 RouteKernel 并发 WriteNb。
两 rank 的地址映射为 `send[peer] -> remote_recv[rank]`、`send[rank] -> recv[rank]`。
整行缺失只说明某个 slice 未得到预期值，不能单凭它认定硬件路径或 kernel 有故障。

## 3. plan 数据模型

### 3.1 不可变目录

plan 的逻辑 identity 是：

```text
(HcclComm, plan_id)
```

以下字段创建后不可变：

- direct route ordinal；
- relay manifest 路径及其内容；
- 路径数量和顺序；
- 默认权重。

同一 `plan_id` 若传入不同目录会返回参数错误，避免缓存命中错误路径。
plan 同时记录创建 device，新 stream 绑定会重新验证 manifest 内容。创建和初始绑定
共同受 build mutex 保护，失败撤销 host registry 中的新 plan，而不宣称可以完全回滚
底层部分创建的资源。

### 3.2 每 stream 资源

同一个 plan 可以绑定多个 stream，每个 stream 有自己的 host 资源记录：

```text
RouteResources
  rank / peer
  HcclThread
  vector<HcclChannel>
  registered HcclCcuKernel
```

映射键为：

```text
(plan_handle, stream address)
```

所有资源构建经过全局 build mutex 串行化，避免编译线程为相同 plan/stream 重复
Acquire Channel 或注册 kernel。

这不证明底层 Channel 独占：`HcclChannelAcquire` 可能复用相同 Channel handle。
同一 plan/共享 Channel 的不同 stream **只允许有明确完成依赖的顺序使用**，并发使用
尚未验证。提交 mutex 仅防止 host notify 序列交错，不能阻止设备在不同流同时执行，
也不拦截 graph replay。stream 必须保持有效，不支持销毁后用复用的裸指针命中旧缓存。

## 4. C ABI

route runtime ABI 5 新增显式 stream bind 和每次 launch policy：

```c
HcclResult HcclCcuUrmaExplicitMultipathPlanCreate(
    HcclComm comm,
    aclrtStream stream,
    const char *planId,
    const char *relayManifest,
    uint32_t directRoute,
    const uint32_t *pathWeights,
    uint32_t pathCount,
    uint64_t *planHandle);

HcclResult HcclCcuUrmaExplicitMultipathPlanBindStream(
    uint64_t planHandle,
    HcclComm comm,
    aclrtStream stream);

HcclResult HcclCcuUrmaExplicitMultipathPlanExecuteV2(
    void *sendBuf,
    void *recvBuf,
    uint64_t elementsPerPeer,
    HcclDataType dataType,
    HcclComm comm,
    aclrtStream stream,
    uint64_t planHandle,
    const uint32_t *launchWeights,
    uint32_t pathCount);
```

`Create` 同时绑定调用时 stream；`BindStream` 用于编译器/ACLGraph 的另一个 capture
stream；`ExecuteV2` 不允许创建资源，只允许查表和 launch。

旧 `PlanExecute` 保留兼容，使用 plan 默认权重。

## 5. 路径构造

direct 路径从 HCCL 已发现 candidate 中按 ordinal 选择。relay 路径来自拓扑解析脚本
生成的 manifest。每一行已经包含两端真正使用的 CommAddr/EID，而不是把 relay 卡号
直接当成 endpoint：

```text
src physical device
  -> 与指定 relay 相连的 source endpoint
  -> UB forwarding
  -> 与 destination 相连的 destination endpoint
  -> dst physical device
```

route runtime 将 direct 与所有 manifest 行组成有序 `RoutePlanRequest`，逐条执行
`HcclChannelAcquire`。路径顺序决定 `path_weights[i]` 的含义：

```text
index 0 : direct
index 1 : relay manifest row 0
index 2 : relay manifest row 1
...
```

当前最多 13 条路径（1 direct + 12 relay），不是物理系统只能有 12 relay。CCU task
argument 上限为 48；当前 RouteKernel 固定参数 4 个，每条路径 3 个参数，因此
`4 + 3 * 13 = 43`。legacy 组合 kernel 固定参数 7 个，总数 46，也保留数量门禁。

## 6. CCU 数据面

注册阶段生成一个 peer-write RouteKernel。本地 slice 在 caller stream 上通过 D2D copy
完成，peer slice 切成连续 chunk，CCU kernel 中执行：

```text
for each path:
    WriteNb(path.channel, send_chunk, remote_recv_chunk)

for each path:
    Wait(path.completion)
```

所有 `WriteNb` 先提交，再等待完成，因此不是 host 上逐条串行调用。默认比例为：

```text
direct : 每一条 relay = 2 : 1
```

V2 允许同一个 plan 在 eager 模式下使用另一组同长度正权重，不会重建 Channel/kernel。
零权重暂时拒绝：尚未证明所有 T560 provider 对 zero-byte `WriteNb` 都有一致且无副作用的
语义。

公共 `BuildTwoRankPathLayout` 在任何复制/launch 前验证完整布局：FP32、两 rank、正权重、
总字节及地址无溢出、各路径非零、256-byte 分片对齐（末片保留余数）及完整覆盖。
权重乘积使用 `__uint128_t`。注册签名包含实际 Channels 和带分隔符的 ordinals，不能仅
用 route 编号决定 kernel 身份；运行时权重/地址仍是 task 参数，不触发重新注册。

route ABI 12 的公共布局函数允许单路径，用于 native0/native2 候选基线；prepared-plan
入口仍单独要求至少两条路径。ABI 11 把公共函数误设为最少两条，导致单路径 baseline
在注册成功后、launch 前返回参数错误。此修复不改变路径、权重分配或 prepared 数据面。

### 6.1 路径切片、CCU 搬运、WQE 与网络分包（2026-10-09 核实）

这四个层次不能混为一谈：

```text
一个远端 peer 的数据
  -> 按路径权重切出连续数据块（算子控制）
  -> 每条路径生成 WriteNb 搬运操作（CCU kernel 控制）
  -> CCU/底层生成传输任务与 WQE（当前未完整解码）
  -> 协议/硬件拆分为线上报文（当前未确认分段粒度）
```

**当前不是在算子里每 4 KiB 手工组一个 WQE。** 两卡版本的
`common/path_layout.h::BuildTwoRankPathLayout` 按权重计算 `pathBytes`；非末片向下
按 256 字节对齐，末片接收余数，保证所有路径合计覆盖完整 peer slice。
四卡兼容分支的 `common/peer_plan.h::BuildPeerPathLayout` 对每个远端 peer 分别做
相同切片，不把所有 peer 的数据混成一个切片池。

例如，一个 peer 的数据为 4 MiB，direct 与两条 relay 权重为 `[2,1,1]`：

| 路径 | 数据量 | 算子生成的逻辑搬运 |
|---|---:|---|
| direct | 2 MiB | WriteNb(channel_direct, ..., 2 MiB, event) |
| relay0 | 1 MiB | WriteNb(channel_relay0, ..., 1 MiB, event) |
| relay1 | 1 MiB | WriteNb(channel_relay1, ..., 1 MiB, event) |

`op_kernel_ccu/route_kernel.cc::Algorithm` 对每个 Channel 生成一次大块 `WriteNb`，
再统一 `WaitEvent`；`all_to_all_multiroute_kernel.cc` 的并发分支同样先生成全部远端
路径的 WriteNb 再等待。这里的 C++ 循环是在注册阶段构建 CCU 指令表示，执行阶段
由 CCU 运行，并不是 host 每次执行都循环调用普通 `urma_post_send`。
不能据此推导“一次 WriteNb = 一个 WQE = 一个网络包”。

### 6.2 当前使用 CTP，但不能推导 WriteNb 只能写 4 KiB

本实现 `op_host/utils.cc::SelectRoute` 明确筛选 `COMM_PROTOCOL_UBC_CTP`，显式 relay
Channel 使用对应 CTP 模板。内存到内存调用使用
`WriteNb(ChannelHandle, RemoteAddr, LocalAddr, Variable len, CompletedEvent)`，
不是以内部 `CcuBuf` 为源的另一种重载。

已核实的事实：

- 本实现直接传入 MiB 级 `pathBytes`，没有按 4096 字节循环调用 WriteNb；已有有卡
  实验完成了这些大块搬运和数据正确性检查。
- 开发容器的 `ccu_microcode_v1.h` 定义 `CCU_MS_SIZE = 4096`，这是 CCU 内部 MS
  缓冲区粒度，不能当作上述内存到内存 WriteNb 的长度上限，也不能当作 CTP MTU。
- HCCL 源码 `src/ops/op_common/inc/alg_param.h` 定义
  `UB_MAX_DATA_SIZE = 256 * 1024 * 1024`。这是该源码版本的传输大小常量，说明
  HCCL 不将所有 UB 搬运统一限制为 4 KiB；不将它直接宣称为 T560 每一种 WriteNb
  或硬件配置的实测最大值。
- 开发容器 CANN 9.1.0-beta.1 的 HCOMM 反汇编显示：内存到内存 WriteNb 构建
  `CcuRepWrite`，其 Translate 生成带长度寄存器的 `TransLocMemToRmtMemInstr`，
  未看到该软件层按 4 KiB循环。**该库不是有卡机器的 T560**，结论仅限所检查版本，
  且不覆盖硬件内部的 WQE/报文切分。

因此，当前准确结论为：**应用层一次 WriteNb 可以描述大块搬运；底层是否按
4 KiB 分段、一个大块产生多少 WQE、CTP 报文最大载荷分别是多少，尚未确认。**
验证这些问题需要匹配 T560 的 CCU/UDMA/固件实现或权威协议资料；不能用
GET_TP_LIST、Channel 数量或 WriteNb 次数代替分包证据。

### 6.3 原生 HCCL CCU 与 URMA/CTP 的关系

| 名称 | 所在层次 |
|---|---|
| HCCL AllToAll | collective 算法、资源管理和执行入口 |
| CCU | 执行通信搬运与同步操作的硬件引擎 |
| URMA | 通信资源及操作接口体系 |
| CTP | 底层传输类型之一 |

检查 HCCL 的 `src/ops/op_common/executor/channel/channel.cc::GetProtocolByEngine`，
在 `CANN_VERSION_NUM >= 9.1.0` 分支，`COMM_ENGINE_CCU` 允许
`COMM_PROTOCOL_UBC_CTP` 和 `COMM_PROTOCOL_UBC_TP`。native multi-jetty AllToAll
还会按 jetty 数及对齐要求计算 slice，再调用 CCU Write，并不是在算法层统一按
4 KiB循环写。

所以，本算子可明确描述为“CCU 执行 + UBC_CTP Channel”；**不能把所有原生
HCCL CCU 算子都等同于单一 URMA CTP 协议**。具体 baseline 的协议仍需核对其
实际选中 CommLink/Channel 及版本，不能只凭 profiling 的 `Ccu` 任务名判断。

参考官方源码：
[Channel 协议选择](https://gitcode.com/cann/hccl/blob/master/src/ops/op_common/executor/channel/channel.cc)、
[UB 传输大小常量](https://gitcode.com/cann/hccl/blob/master/src/ops/op_common/inc/alg_param.h)。

## 7. ACLGraph 语义

正确顺序：

```text
创建 capture stream
  -> 等待输入生产 stream
  -> 在该 stream 上 PlanCreate/PlanBindStream
  -> eager warmup
  -> capture ExecuteV2
  -> replay 已记录 launch
```

capture/replay 中若查不到 `(plan_handle, current_stream)`，正式路径直接失败，并提示在
图外 bind。只有设置 `A5_CCU_PREPARED_ALLOW_LAZY_STREAM=1` 才允许兼容性懒绑定；该开关
不能用于正式图执行。

当前 torch schema 的 `int[] path_weights` 会被 capture 为图常量。需要另一组权重时：

- eager：直接再次调用 V2；
- ACLGraph：为另一组 policy capture 另一张 graph；
- 未来控制器动态 policy：需要稳定 device policy block 或厂商 host-task/doorbell 接口，
  不能假装当前 list 参数已经支持 replay 中动态变化。

## 8. 与“原生标准 Ascend C 算子”的关系

当前完成的是 prepared-plan runtime 和图可见 torch custom op。它可以 capture/replay，
但不能据此宣称已经是原生 GE/Ascend C 算子。

原生 Ascend C 外壳的问题不是 tiling 写不出来，而是 AIV device kernel 不能调用 host
侧 `HcclCcuKernelLaunch`。此前尝试：

```text
Ascend C AIV -> hccl.AllToAll -> MC2 device server
```

实际进入闭源 MC2/HCOMM 资源链，host patch 没有进入预期 selector/template，并在首次
device synchronize 超时。HBM mailbox 常驻 worker 又缺少可靠 AIV->CCU doorbell/可见性
契约。

因此要变成真正原生标准算子，还需要以下任一官方能力：

1. GE/ACL 可注册的 host task，自图中调用本 runtime；
2. AIV 可调用的公开 CCU launch/doorbell；
3. 官方支持的 device command block 与 CCU worker 协议。

在这些接口缺失前，prepared op 是可验证且不伪造底层能力的实现边界。

## 9. 生命周期和并发

- plan 资源按 communicator/process 生命周期保存；
- 不循环创建一次性 plan_id；稳定路径目录复用一个 plan；
- 同一 plan 的 build 串行，execute 只读取不可变目录和 stream 资源；
- 当前公开接口没有完整、已验证的 Channel/thread/kernel 对称释放链，因此没有实现
  不安全的“半释放”；
- communicator 销毁后不能继续使用旧 handle。
- 没有 `PlanDestroy`，未声称每轮释放 CCU event；event 在注册 kernel 生命周期内复用；
- 目前 forward-only，没有 autograd 公式；两 rank 必须保持相同路径/权重/调用顺序。

### 9.1 无卡验证与有卡验证的区别

`tests/cpp/test_a5_multipath_layout.cpp` 覆盖两 rank、1..12 relay、余数、乘积溢出、
错误 policy、buffer 重叠和缓存签名碰撞。开发 Docker 编译 route package/wheel、验证
route ABI 12 / DeepEP ABI 11 和 Meta/fullgraph capture。它们不证明 T560 有卡完成同步正确；必须再运行操作
指南第 9.2 节逐轮数据诊断和第 6 节真实 ACLGraph replay。
Python 无卡回归入口为 `tests/python/deepep/test_a5_prepared_multipath_meta.py`。

## 10. 代码位置

```text
examples/a5_ccu_urma_route_probe/inc/a5_ccu_urma_route_probe.h
examples/a5_ccu_urma_route_probe/op_host/all_to_all_multiroute.cc
examples/a5_ccu_urma_route_probe/op_kernel_ccu/route_kernel.cc

csrc/deepep/deep_ep.cpp
csrc/deepep/deep_ep.hpp
csrc/deepep/pybind_extension.cpp
python/deep_ep/deep_ep/buffer.py

tests/python/deepep/test_a5_ccu_urma_multiroute_all2all.py
scripts/run_a5_ccu_explicit_multipath_alltoall_perf.sh
scripts/prepare_a5_ccu_explicit_multipath_plans.py
scripts/resolve_a5_explicit_multirelay_eids.py
```
