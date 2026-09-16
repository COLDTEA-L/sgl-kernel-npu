# 为什么 HCCL `route2` 可以通信，而直接选择 EID 不能指定 relay

## 1. 文档目的与结论

本文对应分支：`feature/a5-urma-explicit-relay-provider`。

本文解释两个看似矛盾的实验现象：

1. `HcclRankGraphGetLinks` 枚举出的 `route2` 可以被 `HcclChannelAcquire` 建成 channel，并由 CCU `WriteNb` 完成通信；
2. 从 `route2` 日志取出完整 EID 后，用户态 URMA 无法通过“选择该 EID”复现同一路径，更无法指定一张物理卡作为 relay。

核心结论如下。

- `route2` 是本测试程序为 RankGraph 已发布的第三条 `CommLink` 分配的本地序号，并不是一个可任意编程的硬件 route ID。
- `CommLink` 不只有 EID。它还包含源、目的 endpoint 描述、协议、hop、layer/link 位置等由 HCCL/HCOMM 控制面发布的信息。
- `HcclChannelAcquire` 接收完整的 `HcclChannelDesc`，并使用底层已经存在的 RankGraph、UVS、UDMA 和固件路由状态创建 channel。算子没有在这里指定 relay。
- CCU `WriteNb` 只使用已经创建成功的 `ChannelHandle`。它不会解析 EID，也不会决定 next-hop。
- 用户态公开 URMA API 可以查询 EID、查找 device、创建 context，但没有公开的 `install_source_route`、`set_next_hop`、`set_relay_phy`、`write_route_addr_idx` 或 DAM/TPM 表项写接口。
- 因此，“选择已有 `route2`”与“用 EID 创建一条指定 relay 的新路径”不是同一层能力。前者在当前软件栈可用，后者缺少路由控制面 API。

## 2. 三种标识不能混为一谈

| 名称 | 本仓中的含义 | 是否能指定 relay |
|---|---|---|
| `routeIndex` / `route2` | 测试程序对 `HcclRankGraphGetLinks` 返回的 UBC CTP `CommLink` 按遍历顺序编号 | 只能选择已有 `CommLink`，不能修改它的 next-hop |
| EID | URMA/UB endpoint identity，标识通信端点 | 公开 API 下只能选择/绑定端点，不能把 EID 当作 hop-list |
| `ChannelHandle` | `HcclChannelAcquire` 根据完整 channel descriptor 创建出的可执行通信资源 | CCU 可以用它执行 `WriteNb`，但 relay 已在更低层决定 |

特别要注意：本测试里的 `route2` 只是 ordinal 2。换物理卡、拓扑、RankGraph 或 CANN/HCOMM 版本后，ordinal 2 不保证仍代表相同路径。

## 3. `route2` 从 RankGraph 变成 Channel 的六步溯源

这一节按照真实执行顺序展开。最重要的边界是：`route2` 这个名字只存在于本仓 probe 的候选数组中；进入 HCOMM
以后，继续向下传递的是 endpoint、协议和 channel descriptor，而不是整数 `2`。

### 第1步：从 RankGraph 枚举已经存在的 CommLink

实现位置：

- `examples/a5_ccu_urma_route_probe/op_host/utils.cc`
- 函数：`SelectRoute`

主要调用为：

```text
HcclRankGraphGetLayers(comm, ...)
  -> HcclRankGraphGetLinks(comm, layer, rank, peer, ...)
  -> 过滤 COMM_PROTOCOL_UBC_CTP
  -> 按遍历顺序生成 route0 / route1 / route2
  -> 选择 candidates[routeIndex]
```

代码打印的这些字段来自 `CommLink`，而不是测试程序自己构造的拓扑：

```text
layer
link
protocol
hop
srcEndpointDesc
dstEndpointDesc
src_addr / dst_addr
src_phy / dst_phy
```

因此，看到 `route2 hop=2` 说明 HCCL RankGraph 已经向上层发布了一条 hop-2 link。它不说明测试代码知道中间
relay 卡是谁，也不说明 route2 的全部数据一定只经过单一 relay；这些信息没有出现在公开的 `CommLink` 字段中。

这里还有两个容易误解的点：

1. `route2` 是 `candidates[2]`，不是写入硬件的 `route_addr_idx=2`；
2. 换卡、换 RankGraph 或换软件版本后，第三条候选 link 不保证仍代表同一底层路径。

### 第2步：把选中的 CommLink 转成 HcclChannelDesc

`SelectRoute` 对选中的 link 执行：

```cpp
HcclChannelDescInit(desc, 1);
desc->remoteRank = peer;
desc->notifyNum = CHANNEL_NOTIFY_NUM;
desc->channelProtocol = selectedLink.linkAttr.linkProtocol;
desc->localEndpoint = selectedLink.srcEndpointDesc;
desc->remoteEndpoint = selectedLink.dstEndpointDesc;
```

这里复制的是完整 endpoint descriptor，而不只是 EID。公开的 `HcclChannelDesc` 填充过程没有 `routeIndex`、
`relayPhyId`、`nextHop` 或 hop-list 字段。因此从这一步开始，HCOMM并不知道上层把这条link叫作route2；它只知道：

```text
通信域 + remoteRank + engine=CCU + protocol=UB_CTP
+ localEndpoint + remoteEndpoint
```

这也是后续源码追踪时不能继续搜索变量名 `route2` 的原因。应当使用完整的本端/远端 endpoint pair 作为关联键。

### 第3步：HcclChannelAcquire进入HCOMM资源管理

实现仍在 `utils.cc`：

```text
HcclThreadAcquireWithStream(..., COMM_ENGINE_CCU, ...)
  -> HcclChannelAcquire(comm, COMM_ENGINE_CCU, descs, count, channels)
  -> HcommChannelGetStatus(channels, ...)
```

`HcclChannelAcquire` 的公开语义是：在通信域中获取对应 channel；没有时创建。HCCL层把
`HcclChannelDesc` 交给HCOMM资源管理层，HCOMM根据engine和protocol选择具体channel实现：

```text
COMM_ENGINE_CCU + COMM_PROTOCOL_UB_CTP
  -> CcuUrmaChannel
```

`HcclChannelAcquire` 成功意味着：底层能够把 descriptor 中的 endpoint pair 映射到一条可用 channel。对于
route2，这个映射依赖已经存在的 RankGraph/UVS/UDMA/固件配置。

channel对象能否复用、需要新建还是进入异步建链，由HCOMM资源管理层判断；上层probe不直接创建URMA TP。

### 第4步：CcuUrmaChannel根据endpoint pair推进URMA建链

HCOMM公开源码所展示的建链模型可以概括为：

```text
HcommChannelCreate/Acquire
  -> 创建或取得 CcuUrmaChannel 对象
  -> 预分配 device 侧 channel/context 资源
  -> HcommChannelGetStatus
  -> ConnectChannelsOnce
  -> 根据 localEndpoint/remoteEndpoint 构造 LinkData/连接资源
  -> 通过 HCCP/URMA 创建或取得 TP、jetty 等底层资源
  -> FillDevEntities
  -> Channel READY
```

我们的probe在`HcclChannelAcquire`后调用：

```cpp
HcommChannelGetStatus(created.channels.data(),
                      static_cast<uint32_t>(created.channels.size()),
                      channelStates.data());
```

这一调用不只是“读一个布尔状态”。在当前HCOMM异步建链模型中，它可能推进`ConnectChannelsOnce`并在建链完成后
填充device侧通信实体。只有状态READY，CCU数据面才能安全使用这个channel。

此处真正值得继续向下追踪的关联信息是：

```text
localEndpoint完整EID
remoteEndpoint完整EID
Channel对象/ChannelHandle
LinkData
TP或jetty handle
```

而不是已经消失的 `routeIndex`。

### 第5步：底层控制面把endpoint pair映射到已有路由资源

在 `CcuUrmaChannel -> HCCP/URMA -> UDMA/固件` 这一段，endpoint pair最终必须落到可发送的TP和硬件路径上。
结合openEuler UDMA公开结构，可以确认TP context中存在`route_addr_idx`、`sr_en`、`route_type`、
`switch_mp_en`等路由相关字段，也存在destination-address table和TPM table资源。

但当前公开API看不到底层究竟如何：

从当前公开接口看不到底层究竟如何：

- 查找 destination-address table；
- 选择或分配 `route_addr_idx`；
- 读取 TPM entry；
- 解析 next-hop；
- 选择具体 relay 卡或内部 ECMP 分支。

所以这里是“根据endpoint pair消费已有路由”，不是“由probe创建一条新路由”。route2能走通，说明其descriptor
最终能命中一套可用的底层路由状态；并不表示`HcclChannelAcquire`调用参数中显式携带了relay卡。

### 第6步：CCU kernel只消费建好的ChannelHandle

实现位置：

- `examples/a5_ccu_urma_route_probe/op_kernel_ccu/route_kernel.cc`
- 函数：`RouteKernel::Algorithm`

数据面流程为：

```text
NotifyRecord / NotifyWait 交换远端地址和 token
  -> WriteNb(channel, remoteAddr, localAddr, bytes, event)
  -> WaitEvent(event)
  -> completion NotifyRecord / NotifyWait
```

关键调用为：

```cpp
WriteNb(channels_[i], destination, source, pathBytes[i], events.back());
```

到这一阶段，CCU只得到`ChannelHandle`。route2的layer、hop、EID以及可能的relay已经在channel创建阶段固化，
`WriteNb`没有relay参数，也不会重新读取RankGraph。

完整链路可压缩为：

```text
HcclRankGraphGetLayers
  -> HcclRankGraphGetLinks
  -> candidates[2]（probe本地命名route2）
  -> 复制srcEndpointDesc/dstEndpointDesc
  -> HcclChannelDesc
  -> HcclChannelAcquire(COMM_ENGINE_CCU)
  -> CcuUrmaChannel
  -> HCCP/URMA建立或取得TP和连接资源
  -> 底层已有路由状态/route entry
  -> HcommChannelGetStatus READY
  -> ChannelHandle
  -> CCU WriteNb
```

## 4. 为什么直接选择完整 EID 仍然失败

验证实现位置：

- `examples/a5_urma_full_eid_probe/resolve_full_eid.c`
- `scripts/run_a5_urma_full_eid_route_validation.sh`

脚本先从 route probe 日志提取 route0/1/2 的完整 `src_addr` 和 `dst_addr`，然后依次调用公开 URMA API：

```text
urma_str_to_eid
  -> urma_get_device_by_eid
  -> urma_get_eid_list
  -> urma_create_context(device, eid_index)
```

实机结果为：

```text
EID_RESULT status=NOT_VISIBLE ...
SCHEME_A_GATE=FAIL
```

这说明 HCCL RankGraph 中使用的完整 endpoint EID 没有作为可精确绑定的用户态 URMA EID 暴露出来。只依靠 `udmacX/eid_idx` 也不等价，因为不同 device 下的局部 `eid_idx` 不能代替 RankGraph 中的完整 EID。

即使未来某个完整 EID 能通过 `urma_get_device_by_eid` 和 `urma_create_context` 成功绑定，也只能证明用户态可以选择该 endpoint identity，仍不能证明能够指定 relay。原因是这些 API 没有下列输入：

```text
relay physical device
next-hop EID
source-route hop list
route_addr_idx
DAM/TPM entry
forwarding-only / no-relay-HBM policy
```

EID回答的是“端点是谁”，而显式relay需要回答“包按什么路径到达端点”。下面按五层进一步拆解。

### 4.1 第一层：EID解析不等于路径创建

```cpp
urma_str_to_eid(text, &eid);
```

只把字符串转换为`urma_eid_t`，不访问网络控制面，也不会安装路径。就像把IP地址解析成二进制地址，并不等于
创建了一条经过指定路由器的source route。

### 4.2 第二层：查到device/context也只证明本地端点可绑定

```cpp
urma_get_device_by_eid(eid, URMA_TRANSPORT_UB);
urma_get_eid_list(device, &count);
urma_create_context(device, eid_index);
```

这组API解决的是：

```text
该EID是否由用户态URMA可见？
它属于哪个URMA device？
能否用这个本地EID创建context？
```

它们没有输入destination，更没有relay。因此即使三步全部成功，也只能得到“本地通信身份”，不能得到
`src -> relay -> dst`路径。

当前实机更早就被阻断：从HCCL `CommLink`提取的完整EID在`urma_get_device_by_eid`阶段返回`NOT_VISIBLE`。
这表明RankGraph/HCCL使用的endpoint命名空间或内部注册状态没有等价暴露给普通用户态URMA context。

### 4.3 第三层：目的EID只描述终点，不描述next-hop

假设目的卡7存在某个EID，它表达的是“最终endpoint是这个EID”。它不天然表达：

```text
先从卡6哪个port发出
先到卡2还是卡3
经过一个还是多个hop
是否允许ECMP/spray
是否禁止落relay HBM
```

同一个目的endpoint可以由多条底层路径到达。具体使用哪条路径，需要路由/TP状态，而不是仅由目的EID值决定。

因此不能根据EID字符串里看似包含的device/port数字，推断“把dst设成这个EID就一定经过对应卡”。这些位可能是
endpoint身份编码，但不是公开的source-route指令格式。

### 4.4 第四层：自主构造HcclChannelDesc仍然缺少relay语义

即使应用自己填充：

```cpp
desc.localEndpoint = ...;
desc.remoteEndpoint = ...;
desc.channelProtocol = COMM_PROTOCOL_UB_CTP;
```

公开`HcclChannelDesc`也没有：

```text
relayPhyId
nextHopEid
sourceRouteHopList
routeAddrIndex
forwardingOnly
noRelayHbm
```

所以自主构造descriptor只能请求“在这对endpoint之间建channel”。如果控制面已有可用路径，底层可能选择默认路径；
如果没有匹配路径，`HcclChannelAcquire`就会失败或超时。它不会因为应用希望relay=2而自动创建这条规则。

### 4.5 第五层：真正缺少的是建链前的路由安装操作

要让EID选择进一步变成显式relay，必须在`HcclChannelAcquire`之前完成类似操作：

```text
InstallRoute(
    srcEndpoint,
    dstEndpoint,
    relayPhyId=2,
    dieId,
    forwardingOnly=true,
    noRelayHbm=true)
  -> 返回lease/route handle
  -> 返回或发布可被HCOMM使用的HcclChannelDesc/CommLink
```

其内部至少要完成以下一种工作：

- 安装UVS/HCCP source-route或next-hop规则；
- 创建/更新TP，并为它选择正确`route_addr_idx`；
- 写入DAM/TPM或等价destination/forwarding表项；
- 将新路径发布进RankGraph，或直接返回HCOMM可获取的descriptor；
- 在使用结束后查询和删除这条route lease。

目前公开URMA API没有这类安装接口。因此直接选EID最多能够选择endpoint或命中一条已经存在的默认路径，不能创造
“6经2到7”这一新路径。

### 4.6 为什么route2没有受到同样限制

route2的顺序恰好相反：

```text
底层控制面已经配置路径
  -> RankGraph发布CommLink
  -> probe再选择这条CommLink
```

直接EID方案试图做的是：

```text
probe先选择EID
  -> 希望底层自动推导并安装指定relay路径
```

第二条链缺少“安装指定relay路径”的控制面调用。因此两者不能因为都出现EID字段就被视作等价。

## 5. API 能力对比

| 阶段 | route2 现有路径可用 API | 直接 EID/显式 relay 所需能力 | 当前状态 |
|---|---|---|---|
| 路径发现 | `HcclRankGraphGetLayers`、`HcclRankGraphGetLinks` | 枚举任意 relay/source-route | 只能看已发布 `CommLink` |
| 路径选择 | 选择 `CommLink` ordinal，例如 `routeIndex=2` | 输入 `relayPhyId` 或 hop-list | 公开 HCCL API 不支持 |
| channel 描述 | `HcclChannelDescInit`，填 local/remote endpoint 和 UBC CTP 协议 | 在 descriptor 中携带 next-hop/source-route | 没有对应字段 |
| channel 创建 | `HcclChannelAcquire` | 安装新的 UVS/UDMA forwarding rule 后再 acquire | 只能消费底层可建立的 channel |
| channel 状态 | `HcommChannelGetStatus` | 查询实际 relay/route entry | 只返回 channel state，不返回 relay |
| CCU 数据传输 | `WriteNb`、`WaitEvent` | 数据面动态选择 relay | `WriteNb` 只接收 `ChannelHandle` |
| EID 解析 | `urma_str_to_eid`、`urma_get_device_by_eid` | 把 RankGraph 内部 EID精确绑定到用户 context | 当前 route EID 返回 `NOT_VISIBLE` |
| URMA context | `urma_get_eid_list`、`urma_create_context` | 配置 next-hop/source-route | 只能绑定端点，没有路径参数 |
| 路由表写入 | 无公开接口 | 写 `route_addr_idx`、DAM/TPM、next-hop | 当前缺失 |

## 6. 两条路径的本质区别

### 6.1 选择 route2

```text
控制面之前已建立 hop-2 路由
  -> HCCL RankGraph 发布完整 CommLink
  -> 测试程序选择第三条 CommLink（route2）
  -> HcclChannelAcquire 消费这条 CommLink
  -> 得到 ChannelHandle
  -> CCU WriteNb 执行数据传输
```

这里可见的是 `hop=2`、endpoint pair 和 channel 是否成功。relay 卡的显式身份及底层表项不可见。

### 6.2 直接选择 EID并指定 relay

我们实际需要的是：

```text
用户给出 src=6、relay=2、dst=7
  -> 控制面验证拓扑与两个方向
  -> 安装 forwarding-only、no-relay-HBM 的正反向 route
  -> 写入/更新 DAM、TPM 或等价路由表
  -> 获得能够表示新路径的 endpoint/channel descriptor
  -> HcclChannelAcquire
  -> CCU WriteNb
  -> 使用完后查询并删除 route lease
```

当前公开 URMA API只能覆盖“创建 endpoint/context”，不能完成中间的路由安装步骤。于是直接选择 EID最多命中已有默认路径，不能把默认路径改成指定 relay。

## 7. 为什么现在必须研究或扩展路由表控制面

目标不是让 CCU 多传一个参数，而是在 `HcclChannelAcquire` 之前，使底层存在一条满足以下约束的合法 CommLink：

- source 和 destination 是指定端点；
- next-hop 是指定物理 relay 卡；
- 转发发生在 IO Die；
- relay 计算任务不参与、不被打断；
- 数据不落 relay HBM；
- 正反向路径和资源生命周期可查询、可回收。

这些约束需要路由控制面写入或选择 destination-address/TPM/next-hop 表项。只修改 CCU kernel 的 `WriteNb` 或更换 EID无法补齐这一层。

openEuler UDMA 公开结构中能看到 `route_addr_idx`、`sr_en`、`route_type`、`switch_mp_en` 等 TP context 字段，也能看到 destination-address table、TPM table 的资源数量，但当前公开代码没有提供其表项格式及可写用户 ioctl。因此不能安全地猜测位域并直接写硬件。

## 8. 本分支为缺失控制面预留的接口

本分支没有伪造“EID 等于显式 relay”，而是增加了 fail-closed provider contract：

- `examples/a5_ccu_urma_route_probe/inc/a5_uvs_source_route_provider.h`
- `examples/a5_ccu_urma_route_probe/inc/a5_uvs_explicit_route_backend.h`
- `examples/a5_ccu_urma_route_probe/op_host/explicit_route_provider.cc`

预期平台 backend 必须实现：

```text
A5UvsBackendGetCapabilities
A5UvsBackendInstallRoute
A5UvsBackendQueryRoute
A5UvsBackendRemoveRoute
```

`InstallRoute` 不能只返回普通目的 EID；它必须真正安装路由，并返回：

- 可被 `HcclChannelAcquire` 使用的 `HcclChannelDesc`；
- `leaseId`；
- 实际安装的 source、destination、relay、die；
- `forwardingOnly=1`；
- no-relay-HBM 能力。

若没有 backend 或能力不完整，provider 明确失败，不会静默回退到 route0/route2。这样可避免把“使用已有透明 hop-2 路径”误报成“成功指定 relay 卡”。

## 9. 可以从现有实验得出什么、不能得出什么

可以得出：

1. RankGraph 为测试卡对发布了多条 UBC CTP `CommLink`；
2. route2 的公开属性是 `hop=2`；
3. route2 对应 descriptor 能被 `HcclChannelAcquire` 建成 channel；
4. CCU `WriteNb` 可以经该 channel 完成通信；
5. 结合隔离环境下的 HCCN 计数，可以为 route2 使用额外转发端口或多路径提供实验性证据；
6. route2 的完整 endpoint EID 当前不能被用户态 URMA 精确绑定。

不能得出：

1. `route2` 永远对应某一固定物理 relay 卡；
2. EID 的某段数字就是可写的 relay 编码；
3. `eid_idx=2` 就代表 physical device 2；
4. route2 的 hop-2 路径可以由应用任意重写；
5. 仅凭 `WriteNb` 或 HCCN 总计数就能恢复完整逐包 hop 顺序；
6. 选择目的 EID 等价于安装 source-route。

## 10. 后续实现显式 relay 所缺的最小能力

要把当前 provider contract 变成真实实现，至少需要从 A5/HIXL/HCCP/UVS/驱动或固件中获得一种受支持的能力：

1. route add/query/delete API，输入 source、destination、relay/next-hop、die 和策略；或
2. 可写 `route_addr_idx` 及其 destination-address/TPM entry 格式；或
3. source-route/hop-list 的驱动 ioctl；或
4. 能返回指定 relay 所对应 `HcclChannelDesc` 的平台内部 API。

在拿到其中一种能力前，正确的工程结论是：

> route2 证明 A5 软件栈能够消费并执行一条预先配置的透明 hop-2 路径；它不证明公开 URMA API能够通过选择 EID创建或修改一条显式 relay 路径。
