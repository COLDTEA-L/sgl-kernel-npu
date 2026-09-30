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

## 3. plan 数据模型

### 3.1 不可变目录

plan 的逻辑 identity 是：

```text
(HcclComm, plan_id)
```

以下字段创建后不可变：

- direct route ordinal；
- relay manifest；
- 路径数量和顺序；
- 默认权重。

同一 `plan_id` 若传入不同目录会返回参数错误，避免缓存命中错误路径。

### 3.2 每 stream 资源

同一个 plan 可以绑定多个 stream，但每个 stream 都有独立的：

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

当前最多 13 条路径。CCU task argument 上限为 48；kernel 固定参数 7 个，每条路径
占 3 个参数，因此 `7 + 3 * 13 = 46`，仍在上限内。

## 6. CCU 数据面

注册阶段生成一个并发 AllToAll kernel。每次执行根据权重把每个 peer 的 payload 按比例
切成连续 chunk，然后：

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

## 7. ACLGraph 语义

正确顺序：

```text
创建 capture stream
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
