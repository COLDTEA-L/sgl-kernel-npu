# A5 原生 CCU AllToAll 与多路径 CCU AllToAll 实现对比

## 1. 文档范围

本文对比两条实现链路：

1. `sgl-kernel-npu` 通过 MC2 调用 HCCL 原生 CCU AllToAll；
2. `feature/a5-ccu-urma-multirelay-forwarding` 中两卡多路径 CCU+URMA AllToAll。

二者的数据语义相同：每个 rank 的输入按目标 rank 排列，输出按来源 rank 排列。当前多路径实现只支持
`world_size=2`、FP32，以及位于同一个 IO Die 的一组 RankGraph route。

## 2. 接口对比

| 层次 | 原生 CCU AllToAll | 多路径 CCU AllToAll |
|---|---|---|
| Python | `Buffer.hccl_all2_all_ccu(send)` | `Buffer.ccu_urma_multiroute_alltoall(send)` |
| 无分配 Python | 暂无独立 out 接口 | `Buffer.ccu_urma_multiroute_alltoall_out(send, recv)` |
| 输入布局 | 一维连续 tensor，均分为 `rank_size` 块 | `[2, elements_per_peer]` |
| C++ Buffer | `Buffer::hccl_all2_all_ccu` | `Buffer::ccu_urma_multiroute_alltoall[_out]` |
| Host ABI | `aclnnHcclAll2AllCcu*` | `HcclCcuUrmaMultiRouteAllToAll` |
| 通信入口 | MC2 `Hccl::AlltoAll` | `HcclCcuKernelLaunch` |
| 算法选择 | HCCL selector 自动选择 | 环境变量显式指定 route |
| Channel 选择 | HCCL 根据拓扑和优先级选择 | 从 RankGraph 枚举结果按序号选择 |
| CCU 数据原语 | HCCL 内部 `ccu::Write`、`GroupCopy` | 公开 object API 的 `WriteNb`、`LocalCopyNb` |
| 多路径范围 | 主要面向多个 peer、Mesh/Clos、2Die | 同一 peer 的多个 route |
| 数据比例 | HCCL 根据算法和拓扑切分 | route0 权重 2，其他 route 权重 1 |
| profiling | 标准 HCCL/MC2 通信算子 | `deep_ep::ccu_urma_multiroute_alltoall` + CCU Launch |

## 3. 数据布局

两卡时，rank `r` 的输入为：

```text
send[0]：发给 rank0
send[1]：发给 rank1
```

rank `r` 的输出为：

```text
recv[0]：来自 rank0
recv[1]：来自 rank1
```

设每个 peer 的数据量为 `B` 字节。每个 rank 的输入、输出总量均为 `2B`：

```text
self source = send[rank]
peer source = send[1-rank]
self destination = recv[rank]
remote destination = peer.recv[rank]
```

多路径只切分 `peer source`。self slice 不经过网络。

## 4. 原生 CCU AllToAll 调用链

### 4.1 sgl-kernel-npu 到 MC2

```text
Python
  Buffer.hccl_all2_all_ccu(send)
    ↓
C++ DeepEP
  Buffer::hccl_all2_all_ccu
    ↓
aclnn 自定义算子
  aclnnHcclAll2AllCcuGetWorkspaceSize
  aclnnHcclAll2AllCcu
    ↓
AscendC/MC2 wrapper
  hccl_all2_all_ccu(...)
    ↓
  Hccl<HCCL_SERVER_TYPE_CCU>::InitV2
  Hccl::AlltoAll<true>(...)
  Hccl::Wait(handle)
  Hccl::Finalize()
```

关键文件：

```text
python/deep_ep/deep_ep/buffer.py
csrc/deepep/deep_ep.cpp
csrc/deepep/ops/op_host/op_api/aclnn_hccl_all2_all_ccu.cpp
csrc/deepep/ops/op_host/hccl_all2_all_ccu_tiling.cpp
csrc/deepep/ops/op_kernel/hccl_all2_all_ccu.cpp
```

tiling 使用 `HCCL_CMD_ALLTOALL`，并通过 `SetCommEngine(A5_CCU_ENGINE)` 请求 CCU。AscendC kernel 本身不实现
数据搬运算法，只负责把通信请求交给 HCCL runtime。

### 4.2 HCCL runtime 内部

```text
Hccl::AlltoAll
  ↓
HCCL AllToAll selector
  ↓
CCU executor/template
  ├── CcuAlltoAllMesh1D
  ├── CcuAlltoAllMesh1DMultiJetty
  ├── CcuAllToAllMesh1DConcurrent
  └── 2Die template
  ↓
计算 channel/resource/thread/kernel
  ↓
HcommCcuKernelLaunch
  ↓
原生 CCU kernel
```

HCCL 关键文件：

```text
/home/liuyuanwen/hccl/src/ops/all_to_all_v/selector/alltoall_auto_selector.cc
/home/liuyuanwen/hccl/src/ops/all_to_all_v/executor/ins_v2_all_to_all_v_sole_executor.cc
/home/liuyuanwen/hccl/src/ops/all_to_all_v/executor/ins_v2_all_to_all_concurrent_executor.cc
/home/liuyuanwen/hccl/src/ops/all_to_all_v/template/ccu/ccu_temp_all_to_all_mesh1d_multi_jetty.cc
/home/liuyuanwen/hccl/src/ops/all_to_all_v/template/ccu/ccu_temp_all_to_all_mesh1d_2Die.cc
/home/liuyuanwen/hccl/src/ops/all_to_all_v/template/ccu/kernel/ccu_kernel_all_to_all_mesh1d.cc
/home/liuyuanwen/hccl/src/ops/all_to_all_v/template/ccu/kernel/ccu_kernel_all_to_all_mesh1d_multi_jetty.cc
/home/liuyuanwen/hccl/src/ops/all_to_all_v/template/ccu/kernel/ccu_kernel_all_to_all_mesh2die.cc
```

原生 Mesh1D kernel 对所有非本 rank 的 channel 依次生成 `ccu::Write`，所有 Write 都提交后再调用一次
`EventWait`。因此循环生成指令不等于串行完成传输。Multi-Jetty 版本还会将一个 peer 的 slice 拆成多个
Jetty slice，然后统一等待。

2Die 模板会查询 endpoint 的 `DIE_ID`，将 channel 分配到两个 Die，并申请主、从 CCU thread；每个 Die
启动一个 kernel。原生实现因此同时覆盖“同一 kernel 内多个 channel 在途”和“两个 Die/kernel/thread 并发”。

## 5. 多路径 CCU AllToAll 调用链

```text
Python
  Buffer.ccu_urma_multiroute_alltoall_out(send, recv)
    ↓
C++ DeepEP
  Buffer::ccu_urma_multiroute_alltoall_out
    ↓ dlsym
HcclCcuUrmaMultiRouteAllToAll
    ↓
GetRouteResources
  ├── HcclRankGraphGetLayers
  ├── HcclRankGraphGetLinks
  ├── SelectRoute
  ├── HcclThreadAcquireWithStream(CCU)
  ├── HcclChannelAcquire
  ├── HcclCcuKernelRegister
  └── HcclCcuKernelRegisterFinish
    ↓
计算 self/peer offset 和各 route 字节数
    ↓
HcclCcuKernelLaunch
    ↓
AllToAllMultiRouteKernel::Algorithm
  ├── 交换远端 output/token
  ├── LocalCopyNb(self slice)
  ├── WriteNb(route0 slice)
  ├── WriteNb(route2 slice)
  ├── WaitEvent(all)
  └── peer-level post sync
```

关键文件：

```text
python/deep_ep/deep_ep/buffer.py
csrc/deepep/deep_ep.cpp
examples/a5_ccu_urma_route_probe/op_host/utils.cc
examples/a5_ccu_urma_route_probe/op_host/all_to_all_multiroute.cc
examples/a5_ccu_urma_route_probe/op_kernel_ccu/all_to_all_multiroute_kernel.cc
tests/python/deepep/test_a5_ccu_urma_multiroute_all2all.py
```

## 6. 多路径是怎样实现的

### 6.1 多路径不是在 CCU kernel 中计算出来的

UVS/HIXL/HCCL 控制面先把可用链路发布到 RankGraph。对于同一个 `src rank -> dst rank`，RankGraph 可能返回：

```text
route0: layer0, hop1, endpoint pair A
route1: layer1, hop2, endpoint pair B
route2: layer1, hop2, endpoint pair C
```

多路径算子不修改硬件路由表，也不凭空创建 hop。它根据：

```bash
A5_CCU_ROUTE_INDEX=0
A5_CCU_ROUTE_INDICES=0,2
```

选择 RankGraph 已存在的 `CommLink`，把每条 link 的源、目的 endpoint、协议和物理位置填入一个独立
`HcclChannelDesc`，然后通过一次 `HcclChannelAcquire` 得到多个 `ChannelHandle`。

因此，多路径的第一层含义是：

```text
同一 peer + 多个 CommLink/EID endpoint pair + 多个 ChannelHandle
```

如果 route2 的 RankGraph 属性为 `hop=2`，其透明两跳由底层 UVS/HIXL/IO Die 路由完成；CCU kernel 只看到
route2 对应的 channel，并不知道中间物理卡号。

### 6.2 对端数据按 route 分片

Host 按权重切分长度。当前默认：

```text
route0 weight = 2
route2 weight = 1
```

对于 `B=2,097,152 Bytes`、`--route-indices 0,2`：

```text
route0：1,398,016 Bytes
route2：  699,136 Bytes
```

首条路径按 256 Bytes 向下对齐，最后一条路径接收余数，保证总长度严格等于 `B`。每条路径都获得独立的：

```text
sourceOffset
remoteOffset
pathBytes
ChannelHandle
CompletedEvent
```

### 6.3 CCU 中先提交全部 Write，再等待

CCU kernel 的关键顺序是：

```cpp
LocalCopyNb(selfDestination, selfSource, selfBytes, localEvent);

for (route : routes) {
    WriteNb(route.channel, route.remoteDestination,
            route.localSource, route.pathBytes, route.event);
}

WaitEvent(localEvent);
for (route : routes) {
    WaitEvent(route.event);
}
```

route0、route2 的 `WriteNb` 都在第一个 `WaitEvent` 之前提交。这使多个 channel 可以同时处于 in-flight 状态。
是否最终落到不同端口、是否在数据面真正重叠，取决于所选 RankGraph link、channel 建链、IO Die 资源和底层仲裁。

所以，多路径的完整实现方式是：

```text
显式选择多个 RankGraph link
  + 每条 link 建立独立 HCOMM channel
  + 将同一 peer slice 切成互不重叠的片段
  + 在同一 CCU kernel 中对多个 channel 下发 WriteNb
  + 全部下发后统一等待
```

## 7. 同步实现差异

| 阶段 | 原生 HCCL | 多路径实现 | 原因 |
|---|---|---|---|
| 地址/token 发布 | 每个 peer channel 发布 | 每条 route channel 发布 | remote variable 属于 channel 资源 |
| pre-sync wait | output/token bitmask 一次等待 | 同一 notify index，`mask=3` 一次等待 | 避免两个独立 wait |
| self copy | `GroupCopy`/内部 LocalCopy | `LocalCopyNb` | 使用公开 object API |
| remote write | `ccu::Write` + event bit | `WriteNb` + CompletedEvent | API 层次不同，都是 CCU 单边写 |
| 数据等待 | 共享 event bitmask | 每个 route 的 CompletedEvent | object API 未公开 event bitmask 聚合接口 |
| post-sync | 每个 peer channel | 多 route 共用第一条 channel做一次 peer 同步 | 多 route 的远端 peer 相同 |
| host thread 同步 | executor 管理主从 thread | die1 时显式 thread notify | 保证用户 stream 与 slave CCU thread 有序 |

远端 output 地址和 token 不能直接删除。Python out 接口虽然复用本地 `recv`，但对端 CCU 仍需要获得该地址及其
访问 token；并且每条 channel 的 remote variable 是独立资源。完成事件也不能删除，否则 host 返回时无法证明
远端数据已经可见。

## 8. 资源与缓存差异

HCCL 原生 executor 会统一管理：

- 算法选择和拓扑优先级；
- channel、CCU thread、kernel 资源；
- Mesh/Clos 和 2Die 切分；
- Multi-Jetty；
- FastLaunch 上下文及参数缓存；
- 标准 profiling/DFX。

多路径实现的 `GetRouteResources` 使用 thread-local cache，以 `(comm, stream, routeKey)` 为键缓存 channel、thread
和两个已注册 kernel。稳定运行期间不会每次重新枚举 route、创建 channel 或注册 kernel；每次调用仍需要传入
本次 buffer 地址、token、offset 和长度，并执行必要的跨 rank 同步。

## 9. 性能比较时的注意事项

两种算子必须按相同的网络字节数比较。两卡、每 peer 为 `B` 时：

```text
每个 rank self copy：B
每个 rank 网络发送：B
每个 rank 网络接收：B
```

多路径测试使用 `_out` 接口预分配输出，关闭 `A5_CCU_DEBUG`，warmup 后连续提交全部迭代，最后只做一次
`torch.npu.synchronize()`。`host_batch_avg_us` 是 host 批量摊销值；MindStudio 中的 CCU Launch 是设备执行区间。

一个 CCU Launch 中包含多条 Write，并不意味着它们串行。判断并发应综合：

1. route0、route2、route0+2 的 CCU Launch 时间；
2. route 顺序 `0,2` 与 `2,0`；
3. 各路径分片长度；
4. 可用的 per-port/URMA 计数器；
5. route 的 layer、hop、die、endpoint 和 `BW_COEFF`。

## 10. 当前实现边界

- 只支持两个通信 rank；
- 只支持 FP32；
- `--route-indices 0,2` 的 `0/2` 是当前 RankGraph 枚举序号，不是永久硬件编号；
- 多 route 必须属于同一 IO Die，否则当前资源层返回 `HCCL_E_NOT_SUPPORT`；
- route0:route2 暂定 2:1，尚未根据实测带宽自适应；
- `hop=2` 能证明 RankGraph 将该 link 描述为两跳，但公开接口不能直接返回中转物理卡 ID；
- 当前实现验证的是“同一 peer 多 channel 分片写”，还没有覆盖多 peer、多 Die 和动态拥塞控制。

构建、安装、运行和 profiling 命令见：

```text
docs/A5_CCU_URMA_ROUTE_VALIDATION.md
```

CCU 与 AIV/MTE 通信模式的调用链、API、执行流程及选型对比见：

```text
docs/A5_CCU_AND_AIV_COMMUNICATION_GUIDE.md
```
