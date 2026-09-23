# A5 标准显式多路径 AllToAll：CommLink、Channel 与 CCU 数据面

## 1. 先澄清：当前没有向 RankGraph 新增 `CommLink`

当前实现中，只有 direct 路径来自 RankGraph 的原生 `CommLink`。显式 relay 路径并没有调用
`HcclRankGraphAddLink`，也没有修改 communicator 的 RankGraph catalog；代码实际做的是：

```text
RankGraph 原生 direct CommLink
  -> 转成 direct HcclChannelDesc
  -> 克隆 direct HcclChannelDesc
  -> 只替换 local/remote CommAddr 中的 128-bit EID
  -> 得到 synthetic relay HcclChannelDesc
  -> HcclChannelAcquire() 按 endpoint pair 匹配已 provision 的底层 path/TP
```

因此，工程里常说的“合成 CommLink”是便于交流的简称。准确名称应是：

> **由拓扑解析得到 EID pair，并据此合成的 relay `HcclChannelDesc`。**

这也解释了为什么不需要让 relay 卡运行进程：relay 卡只承担 UB/IO Die transit forwarding；通信端仍然是
源 rank 和目的 rank。

## 2. 完整调用链

```text
用户指定 src/dst/relay cards
  |
  | resolve_a5_explicit_multirelay_eids.py
  |  - 读取 driver topology JSON
  |  - 读取 device/UDMAC/EID inventory
  |  - 对每张 relay 拼接 src<->relay 与 relay<->dst 两条物理边
  |  - 输出源端 EID、目的端 EID、die/port/edge 元数据
  v
relay manifest.tsv
  |
  | prepare_a5_ccu_explicit_multipath_plans.py
  |  - 生成 direct+2/direct+4/direct+6 relay plan
  |  - 生成 direct:relay aggregate = 2:1 的 weights
  v
A5_CCU_EXPLICIT_MULTIPATH_{MANIFEST,PLAN_ID,WEIGHTS}
  |
  v
Python Buffer.explicit_multipath_all2all_ccu()
  -> DeepEP C++ Buffer::explicit_multipath_all2all_ccu()
  -> aclnnExplicitMultipathAll2AllCcu()
  -> 标准 Ascend OpDef + tiling
  -> AIV 控制 kernel: Hccl<CCU>.Alltoall()
  -> AlltoAllAutoSelector::SelectCcuScheduleAlgo()
  -> CcuExplicitMultipathAllToAll2Rank
  -> CcuTempExplicitMultipathAllToAll::CalcRes()
  -> BuildExplicitChannels()
       |- direct: CalcChannelRequestMesh1DWithPriorityTopo()
       |    -> ProcessLinksForChannel()
       |    -> HcclRankGraphGetLayers()
       |    -> HcclRankGraphGetLinks()
       |    -> CommLink.src/dstEndpointDesc -> HcclChannelDesc
       |
       `- relay: LoadPlan()
            -> clone direct HcclChannelDesc
            -> replace localEndpoint.commAddr.eid
            -> replace remoteEndpoint.commAddr.eid
  -> CcuKernelInfo.channels = [direct, relay0, relay1, ...]
  -> AlgResourceRequest.ccuKernelInfos
  -> HcclGetChannelForCcu()
       -> AddExchangeInfo()
       -> HcclChannelAcquire(desc[], channelNum, handles[])
       -> handles[] 写入 CcuKernelArg.channels[]
  -> HcclGetCcuKernel()
       -> HcommCcuKernelRegisterStart()
       -> HcommCcuKernelRegister(CcuExplicitMultipathAllToAllKernel)
       -> HcommCcuKernelRegisterEnd()
  -> HcommCcuKernelLaunch()
  -> CCU LocalCopy(self) + 多 Channel Write + EventWait
  -> UB/IO Die forwarding
  -> peer GM
```

## 3. relay EID pair 如何生成

入口脚本为：

```text
scripts/resolve_a5_explicit_multirelay_eids.py
```

对每个显式指定的 `relay_phy`，`resolve_relay()` 执行：

1. `directed_links(edges, src_phy, relay_phy, net_layer)` 找源卡到 relay 卡的物理边；
2. `directed_links(edges, dst_phy, relay_phy, net_layer)` 找目的卡到同一 relay 卡的物理边；
3. 要求两条边在 relay 侧落到同一个 `relay_die`；
4. `endpoint_eids(entries, src_phy, src_die, src_port)` 找源卡该端口的本地 EID；
5. `endpoint_eids(entries, dst_phy, dst_die, dst_port)` 找目的卡该端口的本地 EID；
6. 输出一行 manifest。

关键点：manifest 中写入的是通信两端的 EID，不是 relay 卡自己的 EID：

```text
src_eid = 源卡上与指定 relay 物理边对应的 endpoint EID
dst_eid = 目的卡上与指定 relay 物理边对应的 endpoint EID
```

`relay_phy`、`src_edge`、`dst_edge`、port/die 等字段用于证明和审计“这对 EID 是如何由指定 relay 推导出来的”；
真正交给 `HcclChannelAcquire` 参与匹配的是两端 `EndpointDesc/CommAddr` 中的 EID。

当前实现要求每张 relay 解析结果唯一，而且一个 launch 中所有路径在两个端点分别使用一致的 local die。否则脚本或
HCCL template 会拒绝执行，避免把不兼容的 CCU/IO Die 资源塞进同一个 kernel。

## 4. direct `CommLink` 如何变成 `HcclChannelDesc`

`BuildExplicitChannels()` 首先调用：

```cpp
CalcChannelRequestMesh1DWithPriorityTopo(
    comm, param, topoInfo, subCommRanks, base,
    CommTopo::COMM_TOPO_1DMESH);
```

其内部 `ProcessLinksForChannel()` 的关键步骤是：

```cpp
HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum);
HcclRankGraphGetLinks(comm, netLayer, myRank, peerRank,
                      &linkList, &listSize);
```

选择优先拓扑的一个 `CommLink` 后，原生代码执行等价于：

```cpp
HcclChannelDescInit(&channelDesc, 1);
channelDesc.remoteRank = peerRank;
channelDesc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
channelDesc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
channelDesc.localEndpoint.loc = link.srcEndpointDesc.loc;
channelDesc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
channelDesc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
channelDesc.remoteEndpoint.loc = link.dstEndpointDesc.loc;
channelDesc.channelProtocol = link.srcEndpointDesc.protocol;
channelDesc.notifyNum = NORMAL_NOTIFY_NUM;
```

两卡场景预期只产生一个 direct descriptor，放入：

```text
channels[0] = base[0]
```

## 5. relay `HcclChannelDesc` 如何合成

`LoadPlan()` 读取以下环境变量：

```text
A5_CCU_EXPLICIT_MULTIPATH_MANIFEST
A5_CCU_EXPLICIT_MULTIPATH_PLAN_ID
A5_CCU_EXPLICIT_MULTIPATH_WEIGHTS
```

manifest 的最小必需列为：

```text
route_id  relay_phy  src_die  dst_die  src_eid  dst_eid
```

每条 relay 的构造逻辑是：

```cpp
HcclChannelDesc desc = base[0];

const uint8_t *local  = rank0 ? srcEid : dstEid;
const uint8_t *remote = rank0 ? dstEid : srcEid;

copy(local,  local  + 16, desc.localEndpoint.commAddr.eid);
copy(remote, remote + 16, desc.remoteEndpoint.commAddr.eid);
channels.push_back(desc);
```

所以两个 rank 看到的是互为镜像的 endpoint pair：

```text
rank0: local=src_eid, remote=dst_eid
rank1: local=dst_eid, remote=src_eid
```

当前版本只覆盖 EID。下面这些字段沿用 direct descriptor：

- `remoteRank`；
- endpoint `protocol`；
- endpoint `loc`；
- `channelProtocol`；
- `notifyNum`。

`route_id` 和 `relay_phy` 当前是计划标识与日志元数据，不会直接写入硬件 route index。`src_die/dst_die` 当前用于
同 die 一致性检查，也不会作为 path ID 传下去。真正造成 relay path 区分的是替换后的 EID pair 被
`HcclChannelAcquire` 消费。

## 6. `ChannelAcquire` 与 CCU kernel 注册

template 不直接逐条调用 `HcclChannelAcquire`。它把所有 descriptor 放进：

```cpp
CcuKernelInfo info;
info.channels = channels;
resourceRequest.ccuKernelInfos.push_back(info);
```

HCCL 通用资源层 `HcclGetChannelForCcu()` 再一次性执行：

```cpp
AddExchangeInfo(comm, param);
HcclChannelAcquire(comm, param.engine,
                   kernelChannelRequest.data(), channelNum,
                   kernelChannels.data());
```

成功返回后：

```cpp
kernelArgBase->channels[i] = kernelChannels[i];
kernelArgBase->channelCount = channelNum;
```

随后 `HcclGetCcuKernel()` 通过：

```text
HcclCommQueryCcuIns
HcommCcuKernelRegisterStart
HcommCcuKernelRegister
HcommCcuKernelRegisterEnd
```

把 `CcuExplicitMultipathAllToAllKernel` 与上述 Channel handles 绑定到 communicator resource。CCU thread、kernel、
Channel 的实际句柄由 HCCL 资源上下文缓存；标准算子 replay 不会重复读取 manifest 或重新 Acquire Channel。

## 7. 数据如何分片并并发发送

`KernelRun()` 根据 `weights` 将每个 rank 发给 peer 的数据按 256 Byte 对齐切片。参数排列为：

```text
[input, output, token,
 selfSrcOffset, selfDstOffset, selfBytes,
 path0SrcOffset, path0DstOffset, path0Bytes,
 path1SrcOffset, path1DstOffset, path1Bytes,
 ...]
```

`CcuExplicitMultipathAllToAllKernel` 的顺序是：

1. `WriteVariableWithNotify`/`NotifyWait` 交换每条 Channel 的 output address 和 token；
2. `LocalCopy` 搬运本 rank 的 self slice；
3. 对 direct 与全部 relay Channel 逐一提交非阻塞 `ccu::Write`；
4. 所有 `Write` 都提交后，才开始 `EventWait`；
5. 所有 path completion event 完成后，在 Channel 0 做最终完成握手。

因此并发的因果保证是：

```text
Write(channel0)
Write(channel1)
...
Write(channelN)
EventWait(channel0..N)
```

而不是：

```text
Write(channel0) -> Wait -> Write(channel1) -> Wait
```

## 8. selector、图节点与 replay

`AlltoAllAutoSelector::SelectCcuScheduleAlgo()` 检测到 manifest 与 plan ID 后选择：

```text
CcuExplicitMultipathAllToAll2Rank
```

executor 注册关系为：

```text
HCCL_CMD_ALLTOALL
  -> InsV2AlltoAllVSoleExecutor
  -> CcuTempExplicitMultipathAllToAll
```

首次 launch 把 task args、arg count、input/output base offset 存入 `CcuKernelSubmitInfo::cachedArgs`。
`FastLaunch()` 只替换新 tensor 地址并再次 `HcommCcuKernelLaunch()`，不会重新建 Channel。

## 9. 主要 API 与职责

| API/结构 | 位置 | 职责 |
|---|---|---|
| `HcclRankGraphGetLayers` | HCCL RankGraph | 枚举原生网络层 |
| `HcclRankGraphGetLinks` | HCCL RankGraph | 取得 direct `CommLink` catalog |
| `CalcChannelRequestMesh1DWithPriorityTopo` | HCCL channel helper | 选择原生 direct link |
| `HcclChannelDesc` | HCCL/HCOMM | 描述一对 local/remote endpoint 与协议 |
| `BuildExplicitChannels` | 本分支 HCCL template | 保留 direct，并用 EID pair 合成 relay descriptors |
| `HcclChannelAcquire` | HCOMM/HCCL resource | 将 descriptors 匹配/建立成 Channel handles |
| `CcuKernelInfo.channels` | HCCL resource request | 把 Channel 描述绑定到 CCU kernel 请求 |
| `HcommCcuKernelRegister` | HCOMM CCU runtime | 注册自定义 CCU kernel 与静态参数 |
| `HcommCcuKernelLaunch` | HCOMM CCU runtime | 启动首次执行和 replay |
| `ccu::Write` | CCU 数据面 | 通过指定 Channel 发起 GM 到远端 GM 的搬运 |

## 10. 能证明什么，不能证明什么

当前实现和既有实验已经证明：

- 不需要把 synthetic relay 写回 RankGraph，也能把 EID pair 交给 `HcclChannelAcquire` 建成可用 Channel；
- 只替换目的端 EID 的实验可以使 HCCN footprint 切换到指定 relay 链；
- 多条显式 relay Channel 可以在同一个 CCU kernel 中提交并取得带宽收益。

但仅从 manifest 或 Channel 成功不能证明物理 relay；严格闭环仍需检查对应端口的 HCCN before/after counter。
另外，当前做法复用 direct descriptor 的 protocol/loc，只覆盖 EID，因此只支持已经由系统 provision、并与该 direct
descriptor 兼容的 endpoint/path；它不是任意创建新的底层 TP 或修改 UBUS 路由表。

## 11. 主要文件

DeepEP/算子仓：

- `scripts/resolve_a5_explicit_multirelay_eids.py`
- `scripts/prepare_a5_ccu_explicit_multipath_plans.py`
- `csrc/deepep/ops/op_host/explicit_multipath_all2all_ccu_def.cpp`
- `csrc/deepep/ops/op_host/explicit_multipath_all2all_ccu_tiling.cpp`
- `csrc/deepep/ops/op_kernel/explicit_multipath_all2_all_ccu.cpp`
- `csrc/deepep/deep_ep.cpp`
- `tests/python/deepep/test_a5_ccu_urma_multiroute_all2all.py`

HCCL 仓：

- `src/ops/all_to_all_v/selector/alltoall_auto_selector.cc`
- `src/ops/all_to_all_v/executor/ins_v2_all_to_all_v_sole_executor.cc`
- `src/ops/all_to_all_v/template/ccu/ccu_temp_explicit_multipath_alltoall.cc`
- `src/ops/all_to_all_v/template/ccu/kernel/ccu_kernel_explicit_multipath_alltoall.cc`
- `src/ops/op_common/executor/channel/channel.cc`
- `src/ops/op_common/op_common.cc`

## 12. 多卡与动态控制扩展点

多卡 plan 应升级为 `peerPlans[srcRank][dstRank]`，每项持有 direct、relay `PathSpec`、weights 和 die/thread group。
host 控制器可在 communicator 创建前生成内存 plan，并让 plan ID 参与 resource cache key。切换路径集合应创建新
resource/graph，不能修改正在 replay 的 Channel。
