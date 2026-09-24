# A5 CCU+URMA 显式多路径 AllToAll：方案 2 实现说明

## 1. 为什么改为两阶段模型

旧标准 Ascend C 方案把路径属性放进 host tiling，再调用 MC2 申请通信资源：

```text
aclnn custom op
  -> host tiling / Mc2CcTilingConfig
  -> HcclAllocComResourceByTiling
  -> libmc2_client.so
  -> system-private mc2_ops_hccl
```

实际运行表明系统 `libmc2_client.so` 并不消费本分支修改的开源 HCCL selector，因此在进入显式
CommLink/Channel 构造前就返回 `HCCL_E_NOT_SUPPORT`。继续修改 `libhccl.so` 无法越过这个边界。

方案 2 保留“标准图节点”这个目标，但把不可图捕获的资源初始化前移：

```text
control plane (once)                   graph/data plane (every iteration)
----------------------------------     -----------------------------------
resolve explicit EID pairs             torch.ops.deep_ep...
HcclChannelAcquire                     validate opaque plan_handle
HcclThreadAcquireWithStream            HcclCcuKernelLaunch
HcclCcuKernelRegister/Finish           CCU WriteNb on prepared Channels
store resources under plan_handle      return recv tensor
```

这与 CUDA/NPU 图常见的“图外创建 communicator/handle，图内只执行算子”一致。

## 2. 总体调用链

### 2.1 控制面准备

```text
Python host/controller
  Buffer.prepare_ccu_urma_explicit_multipath_plan(...)
    -> deep_ep_cpp pybind
      -> dlsym(HcclCcuUrmaExplicitMultipathPlanCreate)
        -> ParseExplicitPlan
          -> HcclRankGraphGetLayers/GetLinks（取得 direct candidate）
          -> 读取 controller 提供的 relay manifest
          -> 构造有序 HcclChannelDesc 列表
        -> GetRouteResources
          -> HcclThreadAcquireWithStream
          -> HcclChannelAcquire(each path)
          -> HcclCcuKernelRegister
          -> HcclCcuKernelRegisterFinish
        -> PreparedPlanRegistry[(comm, stream, plan_id)]
        -> uint64 plan_handle
```

relay manifest 中只保存已经由拓扑解析器算出的两端 EID/CommAddr，不写死 `P67/P62` 等端口编号。

### 2.2 图内执行

```text
torch.ops.deep_ep.ccu_urma_prepared_multipath_alltoall
  -> PrivateUse1 implementation
    -> Buffer.ccu_urma_prepared_multipath_alltoall_out
      -> HcclCcuUrmaExplicitMultipathPlanExecute(plan_handle, ...)
        -> PreparedPlanRegistry lookup
        -> LaunchPreparedAllToAll
          -> 计算每条 path 的 byte range
          -> HcclCcuKernelLaunch
            -> CCU kernel
              -> NotifyWait(start)
              -> WriteNb(channel_0, direct chunk)
              -> WriteNb(channel_1, relay chunk)
              -> ...
              -> WaitEvent(all writes)
              -> completion notify
```

并发 schedule 的关键是先向所有 Channel 发出 `WriteNb`，最后统一等待；它不是 host 侧依次调用多个完整
通信算子。

## 3. API

### 3.1 route runtime C ABI

头文件：`examples/a5_ccu_urma_route_probe/inc/a5_ccu_urma_route_probe.h`

```c
int A5CcuUrmaPreparedPlanAbiVersion(void);

HcclResult HcclCcuUrmaExplicitMultipathPlanCreate(
    HcclComm comm, aclrtStream stream, const char *planId,
    uint32_t directRoute, const char *relayManifest,
    const uint32_t *pathWeights, uint32_t pathCount,
    uint64_t *planHandle);

HcclResult HcclCcuUrmaExplicitMultipathPlanExecute(
    HcclComm comm, aclrtStream stream, uint64_t planHandle,
    void *send, void *recv, uint64_t bytesPerPeer);
```

`PlanCreate` 是 host-only 控制面 API；`PlanExecute` 是固定语义的数据面入口。

### 3.2 Python Buffer API

```python
plan_handle = buffer.prepare_ccu_urma_explicit_multipath_plan(
    plan_id, direct_route, relay_manifest, path_weights,
)

buffer.ccu_urma_prepared_multipath_alltoall_out(
    send, recv, plan_handle,
)
```

### 3.3 图算子 API

```python
recv = torch.ops.deep_ep.ccu_urma_prepared_multipath_alltoall(
    send, recv, plan_handle,
)
```

注册 schema：

```text
ccu_urma_prepared_multipath_alltoall(
    Tensor send, Tensor(a!) recv, int plan_handle
) -> Tensor(a!)
```

实现包含 `PrivateUse1` 与 `Meta` dispatch；编译器能看到一个有固定输入/输出 alias 关系的算子节点。
`plan_handle` 是不透明整数，图内不解析字符串或文件。

## 4. 路径计划和流量分片

路径顺序固定为：

```text
path 0: HCCL-discovered direct candidate
path 1..N: controller 显式给出的 relay EID pair
```

分片采用最大余数法按 `path_weights` 对齐到 256 bytes。为了满足 direct 总量与 relay 总量 `2:1`：

| plan | path weights |
|---|---|
| direct + 2 relay | `4,1,1` |
| direct + 4 relay | `8,1,1,1,1` |
| direct + 6 relay | `12,1,1,1,1,1,1` |

relay 是显式 EID path，不是 UBUS 全局 route override；中转卡不运行 rank 进程，数据在 IO Die/UB 转发，
不落中转卡 HBM。

## 5. 资源身份和生命周期

注册表 key 为：

```text
(communicator identity, aclrtStream, plan_id)
```

同一 key、同一 fingerprint 重复 prepare 会返回原 handle；同一 key 但 plan 内容不同会报错，防止控制器
悄悄改变已捕获图的语义。

plan fingerprint 包含 direct candidate、relay manifest 和 weights。当前没有公开销毁接口，因为底层 Channel/
CCU kernel release 语义仍与 communicator 生命周期绑定；进程退出时统一释放。

限制：

- handle 不能跨进程；每个 torchrun rank 都要各自 prepare；
- handle 不能跨 stream；换 stream 必须重新 prepare；
- communicator 被销毁后不能继续 execute；
- 当前实现只处理两 rank AllToAll；未来多 rank 应把 peer/path matrix 放进 controller plan，而不是扩展图内参数。

## 6. 为什么不需要 patched HCCL 或 MC2 源码

方案 2 直接动态调用已安装 route runtime 中的 prepared APIs。该 runtime 使用系统已经暴露的：

- `HcclRankGraphGetLayers/GetLinks`；
- `HcclChannelAcquire`；
- `HcclThreadAcquireWithStream`；
- `HcclCcuKernelRegister/Finish/Launch`。

它不要求修改系统 `libmc2_client.so`，也不把新增 selector 接入系统 `mc2_ops_hccl`。HCCL 源码仓仅作为
编译 route runtime 时的头文件/兼容源码来源，不需要把其 `libhccl.so` 注入运行进程。

## 7. 与旧实现的区别

| 项目 | 旧 standard/MC2 路径 | 方案 2 prepared 路径 |
|---|---|---|
| path plan 入口 | Ascend C attrs/tiling | host controller prepare |
| Channel 建链 | 期望 MC2 转发到 patched HCCL | route runtime 直接建链 |
| 热路径建链 | 理论上缓存，但实际未进入 selector | 明确禁止，资源已准备 |
| 图内文件读取 | 不应有 | 没有 |
| 图节点 | ACLNN custom op | `torch.ops.deep_ep` PrivateUse1 op |
| patched HCCL | 需要且仍未解决 MC2 边界 | 不需要 |
| `libmc2_client` 源码 | 成为 blocker | 不需要 |

原 Ascend C `ExplicitMultipathAll2AllCcu` 源码保留为实验记录，但性能矩阵的正式显式 case 使用
`implementation=prepared`，不再使用 `implementation=standard`。

## 8. 当前验证状态

已完成无卡编译/装载级验证：

- route package 编译、安装通过；
- DeepEP wheel 编译、强制安装通过；
- DeepEP attr ABI `4`；
- prepared-plan ABI `1`；
- plan create/execute 动态符号存在；
- PyTorch graph op 注册与 Meta dispatch 通过。
- `torch.compile(..., fullgraph=True)` 的 Meta capture 通过。

仍需在有卡环境验证：

1. 每个 rank prepare 一次并获得 handle；
2. warmup/iters 中只调用 execute；
3. 数据正确性；
4. direct+2/4/6 relay 的性能与 HCCN 物理路径；
5. 有卡 PrivateUse1 执行下的 graph capture/replay（无卡 Meta capture 已通过）。
