# 两卡显式多路径 AllToAll：当前实现设计

本文描述 `feature/a5-ccu-explicit-multipath-alltoall` 的 prepared-plan 实现（DeepEP ABI 11、route runtime ABI 12），不是早期修改 libhccl 的标准 Ascend C 实验。四卡扩展在 `feature/a5-ccu-peer-plan-alltoall-4rank` 上新增独立接口，保留这里的接口。

## 1. 输入输出与数据量

输入、输出是连续 FP32 NPU tensor，形状 `[2, elements_per_peer]`。`send[dst_rank]` 发送至目的卡的 `recv[src_rank]`。每个 rank 的本地分片复制到自己的接收分片；远端分片按路径权重拆开。测试参数 `--bytes` 是每个目的 rank 的分片字节数，不是整个 tensor 大小。

例如每 peer 4 MiB，direct + 两条 relay，权重 `[2,1,1]`：direct 2 MiB，每条 relay 1 MiB；send/recv 各 8 MiB。比例是 **direct : 每一条 relay = 2:1**，不是 direct : relay 总和 = 2:1。对齐粒度 256 bytes，最后一条路径承接余数；所有分片恰好覆盖远端数据，不重叠、不遗漏。

## 2. 实际调用链

```text
通信域初始化（所有 rank 按一致顺序，图外）
  torch.distributed / Buffer
  -> Buffer.prepare_ccu_urma_explicit_multipath_plan(...)
  -> deep_ep.cpp：ResolveComm + 当前 NPU stream().stream()
  -> HcclCcuUrmaExplicitMultipathPlanCreate
  -> GetRouteResources(RouteKernelKind::ROUTE_WRITE)
       -> HcclRankGraphGetLayers / HcclRankGraphGetLinks
       -> direct CommLink -> HcclChannelDesc
       -> 拓扑/EID manifest -> 合成 relay ChannelDesc
       -> HcclThreadAcquireWithStream（先准备线程，再 Acquire Channel）
       -> HcclChannelAcquire / HcommChannelGetStatus
       -> HcclCcuKernelRegister(CreateRouteKernel)
       -> HcclCcuKernelRegisterFinish
  -> 保存 communicator + device + stream 的资源，返回 plan_handle

执行（每次独立调用，可在预绑定 stream 上 capture）
  Buffer.ccu_urma_prepared_multipath_alltoall[_policy][_out]
  或 torch.ops.deep_ep.ccu_urma_prepared_multipath_alltoall[_policy]
  -> DeviceGuard / recordStream / 刷新 torch_npu host 提交队列
  -> HcclCcuUrmaExplicitMultipathPlanExecute[ V2 ]
  -> LookupPreparedPlan（不重新枚举、建链或注册）
  -> BuildTwoRankPathLayout，校验大小与输入输出不重叠
  -> aclrtMemcpyAsync：本地分片，当前执行 stream
  -> GetTokenInfo：输入、输出整个 tensor 的访问 token
  -> RouteTaskArg：地址、token、每路径 offset/bytes
  -> HcclCcuKernelLaunch：已注册的 RouteKernel
  -> 当前 stream 上的完成依赖
```

**生产 prepared-plan 使用 `RouteKernel`，不是 `AllToAllMultiRouteKernel`。** 后者是旧的合并本地复制实验。`all_to_all_multiroute.cc` 的 `LaunchPreparedRouteAllToAll()` 明确用 stream-ordered D2D copy 处理 self slice，再用已验证的多路径 write kernel 处理 peer slice。

## 3. relay CommLink / ChannelDesc 如何构造

`resolve_a5_explicit_multirelay_eids.py` 将本地 EID 清单与驱动拓扑 JSON 联接。对于 S→R→D，找到 S↔R 和 D↔R 两条边，要求它们接入 relay 上同一兼容 IO die，然后按真实端口匹配 **S、D 两张卡自己的 EID**。不把目的地址改成 relay 卡的 EID，不硬编码 port，不解析 `udmac` 名称猜 physical device。

`utils.cc` 先从 direct candidate 取得公开 endpoint/location/protocol 字段，再通过 `ApplySyntheticRouteSpec()` 改成 manifest 中对应的端点 EID。在反向 rank 上交换 EID 和 endpoint die。`remoteRank` 始终是最终通信对端，不是 relay。随后通过系统 HCCL/HCOMM `HcclChannelAcquire` 建链。

relay 卡不加入两 rank 通信域，不运行 relay 用户进程，不分配 relay HBM 中转 buffer。路径是否实际经过指定卡，要通过拓扑边与 HCCN 对应端口计数验证，不能只凭 Channel 创建成功命名物理路径。

`route0/route2` 是 HCCL candidate ordinal；不是 `route_addr_idx` 或硬件路由表 index。当前实现不修改 ubus.ko、全局 UB route table、MUE/UVS。

## 4. API 与参数

| API | 时机 | 主要参数/行为 |
|---|---|---|
| `Buffer.prepare_ccu_urma_explicit_multipath_plan` | 图外 | `plan_id, relay_manifest, direct_route, path_weights`，返回整数 handle |
| `Buffer.bind_ccu_urma_explicit_multipath_plan` | 图外 | `plan_handle`，绑定当前 NPU stream；capture stream 也必须提前绑定 |
| `Buffer.ccu_urma_prepared_multipath_alltoall_out` | 执行 | `send, recv, plan_handle`，复用预分配输出 |
| `Buffer.ccu_urma_prepared_multipath_alltoall_policy_out` | 执行 | 再加 `path_weights`，改变同一路径集合的分片比例 |
| `torch.ops.deep_ep.ccu_urma_prepared_multipath_alltoall_policy` | 图内/执行 | `send, plan_handle, int[] weights`，返回新 output；有 PrivateUse1/Meta 注册 |
| `HcclCcuUrmaExplicitMultipathPlanCreate` | 图外 C ABI | `comm, stream, planId, relayManifest, directRoute, uint32 weights[], pathCount, handle*` |
| `HcclCcuUrmaExplicitMultipathPlanBindStream` | 图外 C ABI | `handle, comm, stream` |
| `HcclCcuUrmaExplicitMultipathPlanExecute/V2` | 数据面 C ABI | 指针、每 peer 元素数、FP32、comm、stream、handle；V2 接受新权重 |

原两卡 prepared API 要求至少 direct + 一条 relay（2..13 paths），不能靠空 manifest 切成 direct-only。新多 peer API 会支持 direct-only，旧 API 保持原有契约。原生 baseline 应调用 `dist.all_to_all_single`；旧脚本的 `native0/native2` 是自定义 discovered-path 对照，不是原生 collective。

## 5. CCU 数据面与同步

`RouteKernel::Algorithm()` 注册时生成指令：

```text
Load input/output 地址、token、每路径 source_offset / remote_offset / bytes
-> 在所有 Channel 上发送 output 地址和 token（独立 notify 槽 0、1）
-> 等待所有 Channel 的对端地址/token
-> 对所有 Channel 提交 WriteNb（每路径独立 CompletedEvent）
-> 等待所有 CompletedEvent
-> 在每个 Channel 上发送 completion notify（槽 2）
-> 等待每个 Channel 的对端 completion
```

先提交全部 WriteNb 再等待，并不表示底层出口一定完全独立。完成等待不可以只保留第一条 Channel，否则重复调用可能复用尚未完成的输出/通知状态。

CompletedEvent 是注册阶段生成的 CCU representation，不是每次执行新申请一批。相关 vector 提前 reserve，避免移动对象导致指令引用失效。通知槽、访问 token、完成事件不是同一种资源。

die1 使用 route thread/slave stream 时，执行 stream 和 route thread 之间通过双向通知形成依赖，不在 host 上插入设备同步。tensor 的 allocator lifetime 通过 `recordStream` 保护；`NPUStream.stream()` 刷新 host 提交队列，避免通信抢在 tensor producer 前执行。

## 6. 资源生命周期、入图与控制器边界

plan 属于进程和 communicator/device；同 `plan_id` 相同配置幂等，不同配置拒绝。绑定资源按 stream 缓存。当前旧 API 没有显式 PlanDestroy，不能无限创建 plan/stream；需保持 Buffer、communicator、执行 stream 存活至工作完成，再退出进程/销毁通信域。

执行不解析 manifest、不 Acquire Channel、不注册 kernel。新 stream 必须图外 bind；图内不应依赖 lazy stream binding。graph replay 固定 capture 时的 host 权重，改变 Python list 不会自动修改已捕获图。真正设备动态策略/控制器 mailbox 尚未在这个版本实现。

这是一套 PyTorch 自定义通信 op + 自有 host runtime，不是具有 tiling.cpp 的标准 Ascend C ACLNN 算子；已经通过的 capture/replay 不应被表述为任意模型或任意新资源组合已经验证。

## 7. 性能测量

先 prepare/bind，warmup 100 次，再正式执行 20 次，120 次都是独立调用。初始化、拓扑解析、建链、注册不计入稳定执行时间。eager 性能测试不必 capture；capture/replay 是独立正确性验证。比较 profiling 必须取完整 collective 对应的数据面时间范围，不能使用 CCU 条目里 `0xFFFFFFFF` 等无效 size 推算带宽，也不能把不同同步口径的 host_batch_avg_us 直接比较。

## 8. 代码位置

- `csrc/deepep/deep_ep.cpp`：PyTorch tensor 校验、stream/lifetime、runtime 动态加载。
- `csrc/deepep/pybind_extension.cpp`：Python API 与 torch.ops 注册。
- `examples/a5_ccu_urma_route_probe/op_host/all_to_all_multiroute.cc`：plan registry、绑定与执行。
- `examples/a5_ccu_urma_route_probe/op_host/utils.cc`：ChannelDesc、线程、Channel、kernel 资源准备。
- `examples/a5_ccu_urma_route_probe/common/path_layout.h`：两卡偏移与精确分片。
- `examples/a5_ccu_urma_route_probe/op_kernel_ccu/route_kernel.cc`：生产 prepared 数据面。
- `scripts/resolve_a5_explicit_multirelay_eids.py`：拓扑/EID 联接；不是路径控制器。

四卡扩展应使用 `peer -> paths` 模型：从 send[peer] 写入目的卡 recv[rank]，不能将两卡的 `1-rank` 算法直接增加两个进程。
