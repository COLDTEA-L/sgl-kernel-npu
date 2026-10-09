# 两卡显式多路径 AllToAll：当前实现设计

本文描述 `feature/a5-ccu-explicit-multipath-alltoall` 的 prepared-plan 实现（DeepEP ABI 11、route runtime ABI 12），不是早期修改 libhccl 的标准 Ascend C 实验。四卡扩展在 `feature/a5-ccu-peer-plan-alltoall-4rank` 上新增独立接口，保留这里的接口。

**第一次看代码，建议先读第 9 节的逐步讲解，再回来看第 1～8 节的技术摘要。**
第 10 节是可用于口头介绍的讲稿和常见问题。代码依据此两卡分支核对，不混用四卡 peer-plan API。

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

## 9. 从零读代码：一次 AllToAll 到底发生了什么

### 9.0 先记住这句话和六个名词

我们实现的是：**提前准备好多条通信通道，执行时把同一份远端数据切成互不重叠的块，交给 CCU 沿这些通道搬运，最后确认全部完成。**

| 名词 | 在本实现中的含义 | 不要混淆成 |
|---|---|---|
| rank | 通信域内的编号，当前只有 0、1 | 物理卡号 |
| EID | 128-bit 的通信端点标识，写入 EndpointDesc 的 CommAddr | 简单的卡号或我们创建的 route ID |
| CommLink / ChannelDesc | 前者是 RankGraph 给出的链路描述；后者是传给建链 API 的具体要求 | 已经搬了数据的通道 |
| ChannelHandle | 系统接受 ChannelDesc 后返回的已建通道句柄 | 用户选择的 relay 卡号 |
| stream | 按顺序提交设备任务的执行队列 | 某一条物理网络链路 |
| plan_handle | 自有 host runtime 查找准备好资源的整数编号 | 硬件地址、TPN、全局跨进程句柄 |

`host` 是运行 Python/C++ 控制逻辑的 CPU；`device` 是 NPU。HBM 是 NPU 的内存。
CCU 负责通信执行。URMA/HCOMM 提供通信资源和访问机制，不是 Python 自己逐字节搬运。
下面的 CCU C++ `Algorithm()` 是在 host 注册阶段**生成 CCU 指令**，不是每次运行时让 CPU
执行 C++ for 循环把 tensor 搬过去。

贯穿本文的例子：物理卡 2、3 通信，显式借用物理卡 0、1 转发。
启动器令 `ASCEND_RT_VISIBLE_DEVICES=2,3`，因此逻辑 rank0 对应物理卡2，rank1 对应物理卡3。
relay 卡0、1没有 rank 进程，也不加入这个两 rank 通信域。

```text
                    direct Channel
物理卡2 / rank0 ------------------------------> 物理卡3 / rank1
           \                                     ^
            \ relay0 Channel：经物理卡0 ----------|
             \ relay1 Channel：经物理卡1 ---------|

反方向同时也有：卡3 -> 卡2、卡3 -> 卡0 -> 卡2、卡3 -> 卡1 -> 卡2。
这是硬件转发路径，不是 relay 卡上的 Python 收包再发包。
```

### 9.1 第一步：先明确 AllToAll 的数据布局

代码位置：[path_layout.h](../examples/a5_ccu_urma_route_probe/common/path_layout.h)，函数 `BuildTwoRankPathLayout()`。

```text
rank0 的 send： [发给 rank0 的 4 MiB | 发给 rank1 的 4 MiB]
rank1 的 send： [发给 rank0 的 4 MiB | 发给 rank1 的 4 MiB]

rank0 的 recv： [来自 rank0 的 4 MiB | 来自 rank1 的 4 MiB]
rank1 的 recv： [来自 rank0 的 4 MiB | 来自 rank1 的 4 MiB]
```

**send 的行按目的 rank 编号；recv 的行按来源 rank 编号。**
所以 rank0 必须读自己的 `send[1]`，写对端的 `recv[0]`；rank1 则读 `send[0]`，写对端 `recv[1]`。
自己的分片不需要上网，直接本地复制。

源代码中的大小计算：

```cpp
layout.peerBytes = elementsPerPeer * 4;
layout.totalBytes = layout.peerBytes * 2;
layout.selfOffset = rank * layout.peerBytes;
```

这里 `4` 是 FP32 每元素4 bytes；`2` 是两 rank。
`elementsPerPeer=1048576` 时，每 peer 为 `4194304 bytes = 4 MiB`，整个 send/recv 各为8 MiB。
这不是总共4 MiB，也不是每条路径各发送4 MiB。

### 9.2 第二步：把“我要借哪张卡”翻译成端点 EID

代码位置：[resolve_a5_explicit_multirelay_eids.py](../scripts/resolve_a5_explicit_multirelay_eids.py)，函数 `resolve_relay()`。
它调用 [resolve_a5_synthetic_relay_eids.py](../scripts/resolve_a5_synthetic_relay_eids.py) 中的拓扑/EID辅助函数。

输入不是猜出来的端口号，而是：

- 用户指定的源卡、目的卡、relay 卡。
- 驱动的真实拓扑 JSON：`/usr/local/Ascend/driver/topo/950/atlas_950_1.json`。
- 已保存的 physical device / die / port / udmac / EID 对应清单。

下面是 `resolve_relay()` 的关键代码摘录；省略了错误检查和结果去重：

```python
src_links = directed_links(edges, src_phy, relay_phy, net_layer)
dst_links = directed_links(edges, dst_phy, relay_phy, net_layer)

for src_link in src_links:
    for dst_link in dst_links:
        src_die, src_port = src_link["endpoint_port"]
        dst_die, dst_port = dst_link["endpoint_port"]
        relay_src_die, relay_src_port = src_link["relay_port"]
        relay_dst_die, relay_dst_port = dst_link["relay_port"]
        if relay_src_die != relay_dst_die or relay_src_die not in relay_dies:
            continue
        for src_eid in endpoint_eids(entries, src_phy, src_die, src_port):
            for dst_eid in endpoint_eids(entries, dst_phy, dst_die, dst_port):
                # 保存 src_eid["eid"]、dst_eid["eid"] 和两跳的真实端口信息
                ...
```

按顺序理解：

1. 找“源卡↔relay”和“目的卡↔relay”两条拓扑边。
2. 确认两条边在 relay 上接入同一个兼容 IO die。
3. 确认源、目的卡各自使用哪一个 die/port。
4. 查清单，把这些端口翻译成**源卡自己的 EID 和目的卡自己的 EID**。
5. 形成 TSV manifest，每条 relay 一行。候选不唯一就报错，要求显式选 plane，不偷偷猜一个。

例如2→0→3，manifest 的 `dst_eid` 是**目的卡3连接relay卡0的那个端点 EID**，
不是relay卡0的 EID。`relay_phy=0` 是我们的计划/校验元数据；真正传给建链 API 的选择信息
是端点描述及其 EID。relay 卡上的端口用于证明拓扑连接和核对 HCCN，不作为一个 rank 的地址传进去。

最终计划权重由 [prepare_a5_ccu_explicit_multipath_plans.py](../scripts/prepare_a5_ccu_explicit_multipath_plans.py) 生成：

```python
plans["all"] = {
    "manifest": str(all_path.resolve()),
    "relays": relays,
    "direct_route": args.direct_route,
    "weights": [direct_ratio, *([relay_ratio] * len(rows))],
}
```

两条 relay、比例2:1的结果是 `[2,1,1]`：第一个权重对应 direct，后面按 manifest 行顺序对应 relay。
更换 relay 要更换 manifest；更改权重只是改变已有路径的流量份额，不会更换物理 relay。

### 9.3 第三步：Python 创建通信域，然后 prepare 一次

代码位置：[test_a5_ccu_urma_multiroute_all2all.py](../tests/python/deepep/test_a5_ccu_urma_multiroute_all2all.py) 的 `main()`，
以及 [buffer.py](../python/deep_ep/deep_ep/buffer.py) 的 `Buffer.__init__()` / `prepare_ccu_urma_explicit_multipath_plan()`。

下面是现有测试流程的教学版，不是另一个可直接复制运行的完整测试脚本。
启动器已经设置两 rank、设备可见性和通信域 rendezvous；`relay_manifest` 是前一步生成的实际 TSV 绝对路径。

```python
torch.npu.set_device(local_rank)  # 当前进程用可见设备0或1，不直接写物理卡号
dist.init_process_group("hccl")
buffer = deep_ep.Buffer(dist.group.WORLD, num_nvl_bytes=0, num_rdma_bytes=0)

weights = [2, 1, 1]             # direct、relay0、relay1
plan_handle = buffer.prepare_ccu_urma_explicit_multipath_plan(
    "explicit-2relay", relay_manifest, 0, weights
)

send = torch.empty((2, 1048576), device="npu", dtype=torch.float32)
recv = torch.empty_like(send)
# 真实测试还会填充 send，并计算 expected 验证输出。

buffer.ccu_urma_prepared_multipath_alltoall_policy_out(
    send, recv, plan_handle, weights
)
torch.npu.synchronize()         # 等本次设备任务完成，再读/校验 recv
```

这里有两种编号：`"explicit-2relay"` 是用户指定的 **plan_id**；返回的整数是 **plan_handle**。
前者用于幂等创建，后者用于后续快速查找资源。两个进程恰好都拿到 handle=1也不代表共享同一个对象；
每个 rank 进程都有自己的 registry。

`Buffer` 把 ProcessGroup 的 rank、size、HCCL group name 交给 C++：

```python
backend = group._get_backend(torch.device("npu"))
moe_all_to_all_group_name = backend.get_hccl_comm_name(self.rank)
self.runtime = deep_ep_cpp.Buffer(
    self.rank, self.group_size, num_nvl_bytes, num_rdma_bytes,
    low_latency_mode, moe_all_to_all_group_name,
)
```

这里的 Buffer 是通信上下文封装，不是 send/recv tensor。
`num_nvl_bytes=0`、`num_rdma_bytes=0` 不意味着没数据；send/recv HBM 内存由 PyTorch tensor 分配。
两 rank 必须以兼容的顺序准备相同路径计划；不能只有一边 prepare，另一边直接执行。

### 9.4 第四步：Python API 穿过 C++ 扩展，进入我们自己的 runtime

代码位置：[deep_ep.cpp](../csrc/deepep/deep_ep.cpp)，`Buffer::prepare_ccu_urma_explicit_multipath_plan()`。
关键原代码摘录：

```cpp
HcclComm comm = ResolveComm(moe_all_to_all_group_name, ep_comm);
auto stream = c10_npu::getCurrentNPUStream().stream();
uint64_t plan_handle = 0;
HCCL_CHECK(GetCcuUrmaExplicitMultipathPlanCreate()(
    comm, stream, plan_id.c_str(), relay_manifest.c_str(),
    static_cast<uint32_t>(direct_route), weights.data(),
    static_cast<uint32_t>(weights.size()), &plan_handle));
return static_cast<int64_t>(plan_handle);
```

它做三件事：取出当前 HCCL 通信域、取出当前 NPU 执行 stream、调用我们的 `PlanCreate`。
其中 `.stream()` 会刷新 Torch NPU 的 host 提交队列；它不是 `torch.npu.synchronize()`，
不要求此处等待所有设备工作完成。

`GetCcuUrmaExplicitMultipathPlanCreate()` 不是让系统 HCCL 新增一个官方 API。
它从**我们安装的** `liba5_ccu_urma_route_probe.so` 加载自定义函数：

```cpp
void *handle = dlopen(library, RTLD_NOW | RTLD_LOCAL);
void *symbol = dlsym(handle, "HcclCcuUrmaExplicitMultipathPlanCreate");
```

因此库的分工是：

```text
Python deep_ep.Buffer
  -> deep_ep_cpp.so                  我们的 tensor/API 桥接
  -> liba5_ccu_urma_route_probe.so    我们的资源计划、分片、launch runtime
  -> 系统 HCCL/HCOMM                 已有 Channel/Thread/CCU 开发接口
```

**这一版本不要求修改系统 libhccl.so。** 不能把“用了 HCCL 接口”理解成“重写了 HCCL”。
ABI 11/12 是我们自己的扩展版本标记，不是硬件有11或12种选路方式。

### 9.5 第五步：合成 ChannelDesc；我们在这里显式描述所选路径

代码位置：[utils.cc](../examples/a5_ccu_urma_route_probe/op_host/utils.cc)，
`EnumeratePaths()` → `SelectRoute()` → `ApplySyntheticRouteSpec()` → `GetRouteResources()`。

先由系统 RankGraph 找到可用 direct candidate，作为合法的 endpoint/protocol/location 模板：

```cpp
HcclRankGraphGetLayers(comm, &layers, &layerNum);
HcclRankGraphGetLinks(comm, layers[layerIndex], rank, peer, &linkList, &listSize);
// 只保留 COMM_PROTOCOL_UBC_CTP。

HcclChannelDescInit(desc, 1);
desc->remoteRank = peer;
desc->notifyNum = CHANNEL_NOTIFY_NUM;
desc->channelProtocol = selectedLink.linkAttr.linkProtocol;
desc->localEndpoint = selectedLink.srcEndpointDesc;
desc->remoteEndpoint = selectedLink.dstEndpointDesc;
```

这里的 `direct_route=0` 是当前枚举列表的 candidate 编号。使用前必须确认该 candidate
在目标卡对上是预期 direct；它不是硬件路由表的第0项，也不能假设跨环境编号永远一致。

对于 relay 描述，`SelectRoute(..., explicitPlan)` 的公开字段重建分支先复制
EndpointDesc 的 `protocol/commAddr/loc`，再用 manifest 替换 EID。
**没有向系统 RankGraph 插入一个新 CommLink**；准确说是在自己 plan 中合成 ChannelDesc，直接提交系统建链。

`ApplySyntheticRouteSpec()` 中真正替换地址的原代码：

```cpp
ParseEidText("src_eid", spec.rank0LocalEid.c_str(), rank0Local);
ParseEidText("dst_eid", spec.rank0RemoteEid.c_str(), rank0Remote);
const uint8_t *local = rank == 0 ? rank0Local : rank0Remote;
const uint8_t *remote = rank == 0 ? rank0Remote : rank0Local;
std::copy(local, local + EID_BYTE_NUM, desc->localEndpoint.commAddr.eid);
std::copy(remote, remote + EID_BYTE_NUM, desc->remoteEndpoint.commAddr.eid);
*dieId = rank == 0 ? spec.rank0LocalDie : spec.rank0RemoteDie;
```

rank0 使用 manifest 的 src→dst；rank1 自动反过来用 dst→src。
注意 `remoteRank` **始终是最终对端 rank**。rank0 的 `remoteRank=1`，并没有改成 relay 卡的编号。
本 runtime 使用完整端点对来对称建链；此前 dst-only 的穿刺结论不等于此生产版本只修改了 dst 字段。

`GetRouteResources()` 把 direct 放到第0个 Channel，逐行添加 relay 描述（省略错误检查）：

```cpp
SelectRoute(comm, rank, 1U - rank, baseRoute,
            &selectedDescs[0], &directDie, &directUid);
for (size_t i = 0; i < specs.size(); ++i) {
    const size_t channelIndex = i + directCount;  // 本例 directCount=1
    SelectRoute(comm, rank, 1U - rank, baseRoute,
                &selectedDescs[channelIndex], &dieId, &ignoredUid, explicitPlan);
    ApplySyntheticRouteSpec(comm, rank, 1U - rank, specs[i],
                            &selectedDescs[channelIndex], &dieId, &pathUid);
}
```

形成 `[direct_desc, relay0_desc, relay1_desc]`。
实际路径选择的系统实现仍在 HCCL/HCOMM/驱动内部；我们利用已验证的合法端点描述，不手工指定 TP handle，
不声称已经逆向了 firmware 的全部匹配规则。物理 relay 的证明仍需结合真实拓扑与端口计数。

### 9.6 第六步：把描述变成可用 Channel，并注册 CCU 程序

代码位置：仍在 `utils.cc` 的 `GetRouteResources()`。
以下均为真实调用，来自函数内不同位置；省略了错误检查、die1分支和调试日志：

```cpp
// 1. 先准备执行线程资源，绑定当前 stream。
HcclThreadAcquireWithStream(comm, COMM_ENGINE_CCU, stream,
                           THREAD_NOTIFY_NUM, &created.mainThread);

// 2. 再按完整 ChannelDesc 建通道。
created.channels.resize(selectedDescs.size());
HcclChannelAcquire(comm, COMM_ENGINE_CCU, selectedDescs.data(),
                   static_cast<uint32_t>(selectedDescs.size()), created.channels.data());
HcommChannelGetStatus(created.channels.data(),
    static_cast<uint32_t>(created.channels.size()), channelStates.data());

// 3. 把这批 Channel 绑定进要注册的 CCU 程序。
RouteKernelArg kernelArg(created.channels, routeIndices);
hcomm::KernelCreator creator = CreateRouteKernel;
HcclCcuKernelRegister(comm, &kernel, &creator, &kernelArg);
HcclCcuKernelRegisterFinish(comm);
created.kernel = kernel;
```

可以把 ChannelDesc 理解为“通道订单”，ChannelHandle 理解为“办好的通道编号”。
`HcclChannelAcquire` 成功后，代码还检查 Channel 状态为0；没有准备好就拒绝执行。
`HcclThreadAcquireWithStream` 的 Thread 是通信执行资源，不是我们新启动一个 relay Python 线程。

注册阶段只固定“有哪几个 Channel、CCU 指令如何操作它们”。send/recv 地址和本次流量大小
是执行时参数，不需要每次重新注册程序。`GetKernelSignature()` 把实际 Channels 编入 signature，
避免相同 candidate 编号、不同 Channels 错误复用同一程序。

### 9.7 第七步：把资源保存起来，返回 plan_handle

代码位置：[all_to_all_multiroute.cc](../examples/a5_ccu_urma_route_probe/op_host/all_to_all_multiroute.cc)，
`PreparedPlan`、`HcclCcuUrmaExplicitMultipathPlanCreate()`、`BindPreparedPlanStreamLocked()`。

核心数据结构摘录：

```cpp
struct PreparedPlan {
    HcclComm comm = nullptr;
    std::string planId;
    std::string fingerprint;
    std::string relayManifest;
    std::string manifestContent;
    int32_t deviceId = -1;
    uint32_t directRoute = 0;
    std::vector<uint32_t> weights;
    std::unordered_map<uintptr_t, RouteResources> resourcesByStream;
};
std::unordered_map<uint64_t, PreparedPlan> g_plans;
```

可以把 `g_plans[handle]` 理解为一张“预先办好的通信资源卡”：里面有 communicator、默认权重，
以及各个已绑定 stream 的 Channels、Thread、kernel handle。
创建函数先保存配置，然后执行 `BindPreparedPlanStreamLocked()`，完成初始 stream 的资源准备，才返回 handle。

同一个 handle 在同一个 stream 上再次 bind，直接复用：

```cpp
const auto existing = prepared.resourcesByStream.find(streamKey);
if (existing != prepared.resourcesByStream.end()) {
    if (boundResources != nullptr) *boundResources = existing->second;
    return HCCL_SUCCESS;
}
// 新 stream 才调用 GetRouteResources(..., RouteKernelKind::ROUTE_WRITE, ...)。
```

相同 communicator 内，相同 plan_id、相同配置再次 prepare 是幂等的；相同 plan_id 却换 manifest/
direct/默认权重会被 fingerprint 校验拒绝。换路径集合应生成新计划，而非悄悄篡改旧 handle。
当前没有独立 PlanDestroy API；不能在每次算子执行前新建 plan，把资源越积越多。

### 9.8 第八步：每次执行只查计划、检查 tensor，然后算分片

Python 测性能用 `_policy_out`，原因是复用同一个预分配 recv，避免把 output 分配开销算进去。
代码位置：`deep_ep.cpp` 中 `Buffer::ccu_urma_prepared_multipath_alltoall_policy_out()`。

```cpp
c10::DeviceGuard deviceGuard(send_data.device());
// 前面还检查 contiguous、FP32、[2,n]、recv 形状、NPU device、handle 等。
const std::vector<uint32_t> weights = NormalizePreparedPathWeights(path_weights);
HcclComm comm = ResolveComm(moe_all_to_all_group_name, ep_comm);
auto stream = OrderedCcuTensorStream(send_data, recv_data);
HCCL_CHECK(GetCcuUrmaExplicitMultipathPlanExecuteV2()(
    send_data.data_ptr(), recv_data.data_ptr(),
    static_cast<uint64_t>(send_data.size(1)), HCCL_DATA_TYPE_FP32,
    comm, stream, static_cast<uint64_t>(plan_handle), weights.data(),
    static_cast<uint32_t>(weights.size())));
return recv_data;
```

传到 runtime 的是 tensor 的 HBM 首地址、**每 peer 的元素数**、通信域、stream、handle 和权重。
`OrderedCcuTensorStream()` 还处理很重要的安全条件：

```cpp
TORCH_CHECK(send.device() == recv.device(), "CCU send/recv must use the same NPU");
at::assert_no_overlap(send, recv);
const auto npuStream = c10_npu::getCurrentNPUStream();
const aclrtStream stream = npuStream.stream();
c10_npu::NPUCachingAllocator::recordStream(send.storage().data_ptr(), npuStream);
c10_npu::NPUCachingAllocator::recordStream(recv.storage().data_ptr(), npuStream);
```

`.stream()` 保证之前填充 send 的 Torch NPU host 任务先提交，避免 CCU 抢先读取旧数据。
`recordStream()` 告诉内存分配器：这些 tensor 存储还有当前 stream 的异步工作在用，不能提前回收。
它不是等待通信完成；返回 recv 时设备工作可能还没结束，由 stream 的依赖或 synchronize 保证使用顺序。

runtime 的 `ExecutePreparedPlan()` 主要做：

```cpp
LookupPreparedPlan(planHandle, comm, stream,
                   &preparedComm, &resources, &defaultWeights);
// 使用默认权重，或者校验本次新权重与 Channel 数相等。
return LaunchPreparedRouteAllToAll(sendBuf, recvBuf, elementsPerPeer, dataType,
                                  preparedComm, stream, resources, weights);
```

**这段执行路径不读拓扑、不开新 Channel、不重新注册 kernel。**
新 stream 没有 bind 则报错。不要在 capture 中打开 lazy bind 来绕过资源准备限制。

分片由 `BuildTwoRankPathLayout()` 完成，关键算式是：

```cpp
// 除最后一条之外，按权重计算，然后向下对齐至256 bytes。
bytes = static_cast<uint64_t>(
    static_cast<__uint128_t>(layout.peerBytes) * weights[i] / totalWeight);
bytes = bytes / PREPARED_PATH_ALIGNMENT * PREPARED_PATH_ALIGNMENT;

// assigned 是前面路径已经分配的字节数。
layout.sourceOffsets.push_back((1U - rank) * layout.peerBytes + assigned);
layout.remoteOffsets.push_back(rank * layout.peerBytes + assigned);
layout.pathBytes.push_back(bytes);
```

最后一条不再按商取整，拿走 `peerBytes-assigned` 的全部剩余数据；代码检查总和等于 peerBytes。
本例 `[2,1,1]`、4 MiB 恰好对齐，分片及**相对各自 tensor 首地址的 byte offset**如下：

| 发送 rank | 路径 | 本地 send offset | 对端 recv offset | bytes |
|---|---|---:|---:|---:|
| 0 | direct | 4 MiB | 0 MiB | 2 MiB |
| 0 | relay0 | 6 MiB | 2 MiB | 1 MiB |
| 0 | relay1 | 7 MiB | 3 MiB | 1 MiB |
| 1 | direct | 0 MiB | 4 MiB | 2 MiB |
| 1 | relay0 | 2 MiB | 6 MiB | 1 MiB |
| 1 | relay1 | 3 MiB | 7 MiB | 1 MiB |

这就是为什么需要分别传 `sourceOffsets` 和 `remoteOffsets`：send 的行按目的编号，recv 的行按来源编号。
不同路径搬的是**不同的数据块**，不是把4 MiB完整复制三遍。

### 9.9 第九步：本地复制 + 参数打包 + launch

代码位置：`all_to_all_multiroute.cc` 中 `LaunchPreparedRouteAllToAll()`。
首先用当前 stream 做 self 分片复制（这里 `bytes=4 MiB`）：

```cpp
aclrtMemcpyAsync(
    output + static_cast<uint64_t>(resources.rank) * bytes, bytes,
    input + static_cast<uint64_t>(resources.rank) * bytes, bytes,
    ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
```

rank0 在自己 tensor 的 offset0 复制，rank1 在自己 tensor 的 offset4 MiB 复制。
然后为整块 send/recv 地址获取访问 token，打包本次任务：

```cpp
const uint64_t inputToken = hcomm::CcuRep::GetTokenInfo(
    reinterpret_cast<uint64_t>(sendBuf), totalBytes);
const uint64_t outputToken = hcomm::CcuRep::GetTokenInfo(
    reinterpret_cast<uint64_t>(recvBuf), totalBytes);
RouteTaskArg taskArg(
    reinterpret_cast<uint64_t>(sendBuf), reinterpret_cast<uint64_t>(recvBuf),
    inputToken, outputToken, layout.sourceOffsets, layout.remoteOffsets, layout.pathBytes);

HcclCcuKernelLaunch(comm, resources.routeThread, resources.kernel, &taskArg);
```

token 是通信引擎访问内存所需的访问标识，不是 EID，也不是路径选择权重。
`resources.kernel` 来自注册阶段；`taskArg` 才是本次调用的新地址、token、分片参数。
因此下一次 send/recv 地址或权重变化，仍可使用同一个已注册 kernel。

若 endpoint 用 die1，runtime 会准备 slave stream / routeThread，并在主执行线程与 routeThread
之间插入通知依赖。不要把这种线程间通知误认为 relay 卡上执行了一个计算 kernel。

### 9.10 第十步：CCU 先知道“往哪写”，再并发搬数据

代码位置：[route_kernel.cc](../examples/a5_ccu_urma_route_probe/op_kernel_ccu/route_kernel.cc)，
`RouteKernel::GeneArgs()` 和 `RouteKernel::Algorithm()`。

`GeneArgs()` 把本次参数变成64-bit参数列表：

```cpp
std::vector<uint64_t> args = {taskArg->inputAddr, taskArg->outputAddr,
                              taskArg->inputToken, taskArg->outputToken};
for (size_t i = 0; i < taskArg->pathBytes.size(); ++i) {
    args.push_back(taskArg->sourceOffsets[i]);
    args.push_back(taskArg->remoteOffsets[i]);
    args.push_back(taskArg->pathBytes[i]);
}
```

`Algorithm()` 用相同顺序的 `Load()` 接收这些参数。
注册时生成的指令模板知道如何用这些位置；每次 launch 用本次值填进去。

第一阶段是地址/token交换：不能只知道自己的 recv，也必须知道对端此次 recv 的地址和访问 token。
下面是生成指令的关键代码摘录：

```cpp
for (size_t i = 0; i < channels_.size(); ++i) {
    // 前面还通过 Load() 取得 localOutput / localOutputToken。
    NotifyRecord(channels_[i], OUTPUT_NOTIFY_INDEX, OUTPUT_VAR_INDEX,
                 localOutput, NOTIFY_MASK);
    NotifyRecord(channels_[i], TOKEN_NOTIFY_INDEX, TOKEN_VAR_INDEX,
                 localOutputToken, NOTIFY_MASK);
}
for (size_t i = 0; i < channels_.size(); ++i) {
    NotifyWait(channels_[i], OUTPUT_NOTIFY_INDEX, NOTIFY_MASK);
    NotifyWait(channels_[i], TOKEN_NOTIFY_INDEX, NOTIFY_MASK);
}
```

远端变量由 `CreateVariable(channel, variable_index, ...)` 关联，代码里对应
`remoteOutputs[i]`、`remoteTokens[i]`。通知槽0传 output 地址、槽1传 output token。
双方都先发布，再等待，所以不是“双方都先等对方发送”的死锁写法。

第二阶段是分片 write。关键的两个循环保持原代码顺序：

```cpp
std::vector<hcomm::CcuRep::CompletedEvent> events;
events.reserve(channels_.size());
for (size_t i = 0; i < channels_.size(); ++i) {
    hcomm::CcuRep::LocalAddr source = CreateLocalAddr();
    source.addr = localInput;
    source.addr += sourceOffsets[i];
    source.token = localInputToken;
    hcomm::CcuRep::RemoteAddr destination = CreateRemoteAddr();
    destination.addr = remoteOutputs[i];
    destination.addr += remoteOffsets[i];
    destination.token = remoteTokens[i];
    events.push_back(CreateCompletedEvent());
    WriteNb(channels_[i], destination, source, pathBytes[i], events.back());
}
for (auto &event : events) {
    WaitEvent(event);
}
```

`Nb` 是 non-blocking：提交后不立即等待该路径传完。
生成的执行顺序是 `Write direct → Write relay0 → Write relay1 → wait所有路径`，
而不是 `Write direct → wait direct → Write relay0 → wait relay0 ...`。
所以多条通道的传输可以重叠；不是同时复制多份数据，也不是保证所有硬件链路从第一拍起完全并行。

`CompletedEvent` 的创建在**kernel注册/生成指令时**，不是120次测试各申请一批事件。
`reserve()` 提前保留 vector 容量，避免追加元素时重分配/移动，使生成指令引用的事件或变量失效。

第三阶段是双向完成握手：

```cpp
for (const ChannelHandle channel : channels_) {
    NotifyRecord(channel, COMPLETION_NOTIFY_INDEX, NOTIFY_MASK);
}
for (const ChannelHandle channel : channels_) {
    NotifyWait(channel, COMPLETION_NOTIFY_INDEX, NOTIFY_MASK);
}
```

通知槽2表示本次本地发出的数据都完成了，随后等待对端也完成。
这样接收方不会在对端还写着 recv 时就进入下一轮复用缓冲区。
**每条 Channel 都要完成等待，不能只等 direct 或第一条 relay。**

### 9.11 第十一步：为什么它能以 torch custom op 的形式入图

代码位置：[pybind_extension.cpp](../csrc/deepep/pybind_extension.cpp)，
以及 `deep_ep.cpp` 中 `ccu_urma_prepared_multipath_alltoall_policy_op()` / `_meta()`。

注册代码摘录：

```cpp
TORCH_LIBRARY_FRAGMENT(deep_ep, m) {
    m.def("ccu_urma_prepared_multipath_alltoall_policy(Tensor send_data, int plan_handle, int[] path_weights) -> Tensor");
}
TORCH_LIBRARY_IMPL(deep_ep, PrivateUse1, m) {
    m.impl("ccu_urma_prepared_multipath_alltoall_policy",
           TORCH_FN(deep_ep::ccu_urma_prepared_multipath_alltoall_policy_op));
}
TORCH_LIBRARY_IMPL(deep_ep, Meta, m) {
    m.impl("ccu_urma_prepared_multipath_alltoall_policy",
           TORCH_FN(deep_ep::ccu_urma_prepared_multipath_alltoall_policy_meta));
}
```

三层含义：

1. schema 声明“tensor + handle + 权重列表 → tensor”。
2. `PrivateUse1` 是 Torch NPU 的实际执行后端，进入我们前面讲的 ExecuteV2。
3. `Meta` 只检查形状/类型并返回形状相同的 meta output，供编译器推导；不会搬数据。

functional op 内部会 `torch::empty_like(send_data)` 创建 output；测试 eager 性能的 `_out` 版本则复用 output。
不能把 `_out` 的可变输出接口与 functional 图接口当成完全相同的 Python 调用。

图外先在 capture stream prepare/bind。图内只调用：

```python
y = torch.ops.deep_ep.ccu_urma_prepared_multipath_alltoall_policy(
    send, plan_handle, [2, 1, 1]
)
```

路径集合不变，host eager 调用可以传新正权重改变比例；capture 之后 graph replay 的 host 权重已经固化，
不是 replay 前改一下 Python list 就能让图内改路。真正 device-side controller/flag/动态策略还未实现。
也不能从 Meta PASS 推出硬件 capture/replay PASS，后者需要独立有卡实验。

### 9.12 第十二步：正确性、计时和 profiling 分别看什么

代码位置：两卡测试中的 `run_op()`、`run_repeated()` 与 warmup/timing 部分。
真实 prepared 分支调用：

```python
buffer.ccu_urma_prepared_multipath_alltoall_policy_out(
    send, recv, plan_handle, explicit_weights
)
```

当实现为 prepared 时，`run_repeated()` 每轮 `run_op()` 之后调用 `torch.npu.synchronize()`。
因此100次 warmup、20次正式测试是120次**完成后再进行下一次**的独立调用；
不是同时排队120个共用 recv/通知资源的任务。
但每一次 AllToAll 内部的 direct/relay 仍先发全部 WriteNb，所以轮次之间独立不等于路径之间串行。

区分三种结论：

| 观察 | 能说明 | 不能单独说明 |
|---|---|---|
| send/recv 与 expected 相同 | AllToAll 数据语义正确 | 确实经过某个物理 relay |
| 拓扑对应两跳端口 HCCN tx/rx 增量匹配 | 所选 relay 的物理链路有匹配流量，需排除其他作业噪声 | 每条链路周期级重叠的精确时刻 |
| MindStudio CCU 时间、吞吐收益 | 测量口径下的数据面开销/收益 | 仅凭一个无效 size 字段反推真实带宽 |

初始化/拓扑解析/建链/注册不计入稳定执行时间。host 平均耗时包含提交与同步，
MindStudio CCU task 更接近设备通信任务耗时，两者不是一个指标。
不要把 CCU 记录中的 `4294967295` 当 payload；payload 以真实 tensor 和 per-peer bytes 为准。

## 10. 下午讲解可以怎么说

### 10.1 三分钟版本

> 我们做了一个两卡 AllToAll 的自定义通信算子。每张卡的 tensor 分成两行：一行给自己，
> 一行给对端。给自己的用本地 D2D copy，给对端的可以同时走 direct 和用户指定的多条 relay。
>
> 实现分为准备和执行两部分。准备时，根据用户选择的 relay 卡，从驱动拓扑找到两跳的真实端口，
> 再查出源、目的卡对应的完整 EID。我们用 RankGraph 的 direct 描述作模板，合成 relay 的
> ChannelDesc，调用系统 HCCL/HCOMM 建 Channel，并注册一个 CCU 程序。准备好的资源存成 plan，返回 handle。
>
> 后续每次执行只给这个 handle、tensor 和权重，不再建链。比如每 peer 4 MiB、权重2:1:1，
> direct搬2 MiB，两条relay各搬1 MiB。CCU先交换对端的输出地址和访问token，
> 再把三条路径的非阻塞write全部提交，最后统一等待每条完成，并做对端完成握手。
>
> 因而我们没有修改全局路由表，也没有让relay卡加入通信域或运行中转进程。
> 选路是图外端点/Channel准备，流量比例是执行参数；PyTorch custom op负责可入图的执行入口。
> 当前版本不是带tiling.cpp的标准Ascend C算子，device-side动态控制器也是后续扩展，而不是已经实现的功能。

### 10.2 最容易被问到的问题

**“我们改了 RankGraph 吗？”** 没有。查询 direct 模板，但自己合成 relay ChannelDesc，
不调用一个不存在的 RankGraph 插入 API。多条描述在自有 plan 中维护。

**“显式在哪？”** 用户输入 relay 物理卡集合；resolver 解析该卡的真实拓扑边与端点 EID，
不随机选 relay，也不简单使用已有 candidate2 冒充所有用户指定路径。

**“只是把 dst 改成 relay 卡吗？”** 不是。dst 仍是最终目的卡的端点 EID，
选的是它连接所需 relay 的那个端点；remoteRank仍是最终对端。

**“为什么有多个 kernel 文件，读哪个？”** 当前 prepared 生产路径看 `route_kernel.cc`。
`all_to_all_multiroute_kernel.cc` 是旧合并本地复制实验，不是这里的生产数据面；
`command_block_worker_kernel.cc` 是另一条控制链穿刺，也不在此 prepared 执行链里。

**“有没有每次创建 Channel？”** 没有。当前 stream 的资源在 prepare/bind缓存，执行时 Lookup。
新 stream需要图外 bind，不能无限创建 plan/stream；目前没有独立销毁 API。

**“能不能随时改比例/禁用某条路径？”** eager 的 policy API 支持同一Channel集合的新正权重，
但原两卡接口不接受0权重；更换路径集合需要新plan。capture的host权重不会在replay时自动改变。

**“为什么两卡可以直接 rank=1-rank？”** 因为只有两个rank，另一个只能是1-rank。
四卡要按peer遍历并分别计算offset，不能只把进程数从2改到4。

**“最多多少条？”** 此两卡 prepared API 当前2..13条总路径，包含direct。
一机8卡、通信2卡且relay不在域内，物理候选最多6张外部relay。
resolver脚本允许输出更多候选，不等于单次CCU launch无限支持；真实限制以runtime/硬件资源为准。

### 10.3 不熟悉代码时的推荐阅读顺序

1. `buffer.py` 的prepare/执行方法：看用户如何调用。
2. `deep_ep.cpp` 的同名方法：看通信域、stream、指针如何传下去。
3. `utils.cc` 的 `ApplySyntheticRouteSpec()`、`GetRouteResources()`：看选择端点、建链、注册。
4. `all_to_all_multiroute.cc` 的 `PreparedPlan`、`ExecutePreparedPlan()`、`LaunchPreparedRouteAllToAll()`：看复用与launch。
5. `path_layout.h`：用第9.8节的offset表核对分片。
6. `route_kernel.cc` 的 `Algorithm()`：重点看“两个write/wait循环分开”以及最后的完成握手。

准备阶段读的是配置和通信资源；执行阶段读的是tensor地址和分片参数。
只要讲清这条分界，就不会把“可选路径”“并发搬运”“图捕获”“动态控制器”混成同一件事。
