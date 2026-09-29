# A5 CCU+URMA 显式多路径 AllToAll 实现说明

## 1. 目标和结论

本实现同时满足三项要求：

1. 图中是标准 Ascend C/ACLNN 算子，而不是 Python 组合通信原语；
2. direct 和 relay 的物理路径在 communicator 初始化时显式 provision；
3. ACLGraph replay 时可以通过设备 Tensor 动态改变 path mask/权重，不重新建 Channel。

最终结构不是常驻 CCU worker/mailbox，而是短生命周期标准算子：

```text
标准 AIV kernel
  -> 读取设备 pathPolicy
  -> Hccl::AlltoAll（MC2 消息）
  -> 配套 HCCL 显式多路径 template
  -> 单次 CCU kernel
  -> 多个已准备 Channel 并发 Write
```

因此不需要自定义 HBM CommandBlock 或长期占用 CCU，但必须使用配套 HCCL 扩展。系统 stock HCCL
不知道如何根据显式 relay manifest 构造这些 Channel，也不知道如何解释动态 PathPolicy。

## 2. 完整调用链

```text
Python/controller
  Buffer.explicit_multipath_all2all_ccu(send, path_policy, plan_id)
    |
    v
DeepEP C++
  Buffer::explicit_multipath_all2all_ccu
  - 校验 2 ranks、dtype、policy=[1]int64/NPU
  - 校验 HCCL extension ABI >= 7
  - 校验 plan_id/manifest 在 communicator 初始化前已设置
    |
    v
ACLNN
  aclnnExplicitMultipathAll2AllCcuGetWorkspaceSize
  aclnnExplicitMultipathAll2AllCcu
    |
    v
标准 Ascend C host
  ExplicitMultipathAll2AllCcu op_def
  ExplicitMultipathAll2AllCcuTiling
  - rank_size=2
  - 输入/输出 dtype 和 shape 校验
  - pathPolicy=[1]int64
  - 生成 CCU MC2 tiling
    |
    v
标准 Ascend C AIV kernel
  explicit_multipath_all2_all_ccu
  - 从 GM pathPolicy Tensor 读取当前 policyWord
  - Hccl<CCU>.AlltoAll(..., strideCount=policyWord)
    |
    v
MC2/HCCL resource path
  HCCL_CMD_ALLTOALL
  OpExecuteConfig = CCU_SCHED (6)
  OpParam.DataDes.strideCount = policyWord
    |
    v
HCCL selector
  AutoSelectorBase::Select
  -> AlltoAllAutoSelector::SelectCcuScheduleAlgo
  - 检测 ALLTOALL + CCU_SCHED + manifest + plan_id + 2 ranks
  - 选择 CcuExplicitMultipathAllToAll2Rank
    |
    v
HCCL CalcRes（communicator/resource 初始化）
  CcuTempExplicitMultipathAllToAll::CalcRes
  - 取得一条 native direct ChannelDesc
  - 从 relay manifest 读取每条 src/dst EID
  - 复制 direct desc，并替换两端 CommAddr EID
  - 为 direct + relay 注册一个 CCU kernel/channel catalog
    |
    v
HCCL KernelRun / FastLaunch（每次执行）
  - DecodePathPolicy(strideCount)
  - 按权重和 256-byte 对齐计算每条 path 的 offset/bytes
  - HcommCcuKernelLaunch
    |
    v
CCU kernel
  PreSync(active Channels)
  LocalCopy(self slice)
  Write(channel_0, direct chunk)
  Write(channel_1, relay-0 chunk)
  ...
  EventWait(all active writes)
  PostSync(channel_0)
```

并发的因果保证是：所有 active Channel 的 `Write` 都先提交，再开始任何远端 `EventWait`。

这里不把显式路径塞进原生 MultiJetty。MultiJetty 证明固定 AllToAll 的合法资源入口是
`ALLTOALL + CCU_SCHED`，但它没有公开 `jetty index -> EID/relay` 映射；因此最终数据面仍是
direct/每条 relay 各自一个 `HcclChannelDesc`，由自定义 CCU kernel 并发提交。

自定义 kernel 必须遵守 HCCL 原生 AllToAll Channel 的资源布局：`INPUT_XN_ID=0`、
`OUTPUT_XN_ID=1`、`TOKEN_XN_ID=2`。route-probe 自建 Channel 曾使用 0/1 作为 output/token，
该私有布局不能复制到 HCCL provision 的 Channel。

## 3. 标准算子接口

### 3.1 OpDef

```text
ExplicitMultipathAll2AllCcu(
    sendData: Tensor,
    pathPolicy: Tensor[int64, shape=(1,)],
    group: str,
    rank_size: int,
    rank_id: int,
    plan_id: str,
) -> recvData
```

支持的 payload dtype：`float16`、`bfloat16`、`float32`、`int32`。对应生成 4 个 Ascend950
kernel；它们共享一份泛型 kernel 源码，由算子编译器按 dtype 实例化。

### 3.2 DeepEP API

```python
recv = buffer.explicit_multipath_all2all_ccu(
    send_data,
    path_policy,
    plan_id,
)
```

`path_policy` 必须是 NPU Tensor，地址在 graph capture/replay 间保持不变；控制器通过原地更新其内容
改变下一次 replay 的路径策略。

## 4. 路径资源如何构造

路径目录固定为：

```text
catalog[0]    = HCCL native direct candidate
catalog[1..N] = controller 解析拓扑后写入 manifest 的显式 relay EID pair
```

manifest 每行包含 `route_id`、`relay_phy`、`src_die`、`dst_die`、`src_eid`、`dst_eid`。HCCL 不使用
硬编码的 `P67/P62`，而是把 EID 写入 `HcclChannelDesc.localEndpoint/remoteEndpoint.commAddr` 后走原生
ChannelAcquire/resource 流程。relay 卡不启动用户进程，转发发生在 IO Die/UB 数据面，不落 relay HBM。

manifest 和 `plan_id` 必须在 communicator 初始化前设置。它们决定 Channel catalog，是静态控制面；
图内 `pathPolicy` 只决定本次执行如何使用这个 catalog，是动态数据面。

## 5. PathPolicy ABI

一个 64-bit word 的编码为：

```text
bits 60..63 = ABI 1
bits 0..3   = catalog[0] 权重
bits 4..7   = catalog[1] 权重
...
```

每个权重为 0..15；0 跳过该路径的变量交换、Write 和 EventWait。若 policy word 为 0，HCCL 为迁移期
兼容使用 `2,1,1,...`；正式调用始终编码 ABI 1。

当前最多 13 条总路径，来源是 HCCL `CCU_MAX_TASK_ARG_NUM=48`：固定参数 6 个，每条路径 3 个参数，
FastLaunch cache 额外保存 3 个元数据，故 `6 + 3 * 13 + 3 = 48`。

## 6. 动态策略如何进入 ACLGraph replay

host tiling 不读取 policy 值，只检查它是 `[1]int64`。AIV kernel 每次执行从 GM 读取该值，因此：

```text
capture: policy=[2,1,1]
  -> direct + relay0 + relay1

host/controller 原地写 policy=[2,0,1]

replay:
  -> 同一图、同一 tensor 地址
  -> direct + relay1
  -> 不重建 communicator/channel/kernel catalog
```

HCCL `KernelRun` 和 `FastLaunch` 都调用同一个 `FillPathArgs`，避免首次执行可动态而 graph fast-launch
退回静态分片。

## 7. 修改位置

### sgl-kernel-npu

- `csrc/deepep/ops/op_host/explicit_multipath_all2all_ccu_def.cpp`
- `csrc/deepep/ops/op_host/explicit_multipath_all2all_ccu_tiling.cpp`
- `csrc/deepep/ops/op_kernel/explicit_multipath_all2_all_ccu.cpp`
- `csrc/deepep/ops/op_host/op_api/aclnn_explicit_multipath_all2all_ccu.*`
- `csrc/deepep/deep_ep.cpp` / `deep_ep.hpp`
- `csrc/deepep/pybind_extension.cpp`
- `python/deep_ep/deep_ep/buffer.py`
- `tests/python/deepep/test_a5_ccu_urma_multiroute_all2all.py`

### hccl

- `src/ops/op_common/selector/auto_selector_base.cc`
- `src/ops/all_to_all_v/selector/alltoall_auto_selector.cc`
- `src/ops/all_to_all_v/executor/ins_v2_all_to_all_v_sole_executor.cc`
- `src/ops/all_to_all_v/template/ccu/ccu_temp_explicit_multipath_alltoall.*`
- `src/ops/all_to_all_v/template/ccu/kernel/ccu_kernel_explicit_multipath_alltoall.*`

其中 `auto_selector_base.cc` 不再包含显式多路径的 `HALF_ALLTOALLV` 特判；路径选择只位于
`AlltoallAutoSelector::SelectCcuScheduleAlgo`。executor 也只在 `HCCL_CMD_ALLTOALL` 下注册该模板，
避免一个固定 AllToAll 模板被两个不兼容的资源协议同时命中。

## 8. 生命周期与扩展性

- communicator 初始化：创建完整 direct/relay Channel catalog；
- graph capture/replay：只传 policy 和 buffer 地址，复用 catalog；
- communicator 销毁：跟随 HCCL 资源生命周期回收；
- 新增 relay：需要在下一次 communicator 初始化前更新 manifest；
- 动态禁用 relay、切换子集、改变比例：只更新 `pathPolicy`；
- 扩展到 4 ranks：应把当前“一对 rank 的 path catalog”提升为 peer-pair/path matrix，不能简单复用两 rank
  offset 公式。

## 9. 已验证与待验证边界

已在 `cam_lyw_dev_91` 完成：

- HCCL host/CCU 源码完整编译、链接和 run-package 打包；
- HCCL extension ABI 7 装载验证；
- 标准 opbuild、4 个 dtype kernel、custom OPP 打包；
- DeepEP C++ extension 和 wheel 编译；
- wheel 在容器内 force-reinstall，DeepEP ABI 8 装载成功；
- 安装后容器仍保持运行。

有卡环境已经完成、并直接推动本次修正的验证：

- ABI 3 的 `HALF_ALLTOALLV + CCU_MS` 在 `GetTilingAccelerator` 阶段返回
  `HCCL_E_NOT_SUPPORT (5)`，尚未进入 selector；
- relay plan 本身完整，`src/dst/relay die` 与 EID 均已解析，不能用 plan 问题解释上述失败。

ABI 4 已在有卡环境跨过资源分配，但因沿用 route-probe 的 0/1 resource index 卡在首次
warmup completion。ABI 5 改用 HCCL 原生 1/2 output/token index；ABI 6 对齐
原生 AllToAll/MultiJetty 的 Channel 生命周期和 per-channel token。ABI 6 有卡运行仍挂起，
最终定位到 completion mask 沿用了 route-probe 的 `1<<5`，而原生 AllToAll 固定使用
`POST_SYNC_ID=3`。ABI 7 已改为 `1<<3`。

ABI 7 仍需有卡环境完成：

1. 标准算子单次正确性；
2. ACLGraph capture/replay 正确性；
3. replay 修改 policy 后的 HCCN 端口变化；
4. 1/2/4/6 relay 性能和稳定性。

## 10. 底层接口边界与替代架构

### 10.1 本项目 ABI 不是厂商底层 ABI

`A5HcclExplicitMultipathExtensionVersion()` 和 DeepEP attr ABI 是本项目自己的兼容性标记，
作用是拒绝旧 HCCL、旧 wheel、旧 OPP 与新算子混装。ABI 4..7 分别记录 selector、资源槽、
Channel 生命周期和完成通知的修正，不表示 CANN/HCOMM 额外暴露了 4..7 套接口。

当前模板只依赖以下既有能力：

| 层次 | 当前使用的接口或对象 | 是否由本项目新增 |
|---|---|---|
| HCCL 模板层 | `CcuKernelInfo`、`AlgResourceRequest`、`HcclChannelDesc` | 新增模板逻辑，不新增底层 ABI |
| HCOMM Channel | `HcclChannelAcquire` | 否 |
| HCOMM CCU 注册/执行 | `HcommCcuKernelRegister*`、`HcommCcuKernelLaunch` | 否 |
| CCU primitive | `GetResByChannel`、`Write`、`EventWait`、`NotifyRecord/Wait` | 否 |
| TP/MUE/固件 | TPN、`route_addr_idx`、私有 path object | 未直接访问，也没有公开修改接口 |

ABI 7 中的 `POST_SYNC_ID=3` 是 notify 内部的 mask bit，真正使用的 notify 槽是
`POST_SYNC_CKE=1`。原生 Channel 的 `NORMAL_NOTIFY_NUM=3` 已覆盖该槽，因此 ABI 7 没有要求
底层新增“第 4 个 notify”。

HCCL 的 9.1 兼容层以 weak symbol + `dlsym` 解析 HCOMM CCU 接口。这意味着：

- 目标 `9.1.T560` 有这些符号时，修改开源 HCCL 模板是可行的；
- 另一个 9.1 预览包可能缺少相同符号，所以必须锁定 CANN/HCOMM 版本；
- 仅编译成功不能证明运行时支持；操作指南第 4.1、4.2 节的 preflight 才是运行前 gate；
- 如果 preflight 已通过，而程序卡在 device synchronize，阻塞点是 CCU 指令/资源语义，
  不是“底层 ABI 数量不足”。

### 10.2 不修改核心 HCCL 的首选替代方案

如果目标包不允许当前 HCCL 模板继续运行，首选迁移为独立 HCOMM 自定义通信算子插件：

```text
communicator/resource initialization
  -> 根据 direct/relay EID 构造 HcclChannelDesc
  -> HcclChannelAcquire
  -> HcommCcuKernelRegister*
  -> 缓存 plan/channel/kernel handles

GE/ACLNN 可识别的通信算子
  -> 输入 plan handle、path mask、weights 和 buffer
  -> 通过 HCOMM 自定义通信算子资源上下文取得已准备资源
  -> HcommCcuKernelLaunch

CCU kernel
  -> 先提交所有启用路径的 Write
  -> 再统一 EventWait
  -> 完成 NotifyRecord/NotifyWait
```

该结构不修改核心 `libhccl.so`，也不需要 HBM mailbox 或常驻 CCU worker；但仍依赖目标版本
公开/可解析的 HCOMM CCU 接口，因此必须做相同的版本预检。

### 10.3 如果 CCU 接口确实不可用

严格的“标准算子 + CCU 数据面 + 显式 relay”无法只靠普通 Ascend C tiling/AIV kernel完成：
tiling 阶段不能代替运行时创建 Channel，AIV kernel 也不能调用 host 侧 `HcclChannelAcquire`。
此时只有以下降级路线：

1. 标准 AICPU_TS/HCOMM 通信算子：继续显式构造 Channel，使用
   `HcommWriteOnThread`/notify 原语，性能可能低于 CCU；
2. 标准 AIV+URMA/UMDK CAM 算子：可保留显式 EID 多路径，但数据面不再是 CCU；
3. 已验证的 prepared-plan + ACLGraph 方案：保留 CCU 和显式路径，但它不是完整标准通信算子。

不再推荐 HBM mailbox + 常驻 CCU worker：现有穿刺没有确认公开、可靠的 AIV 到 CCU doorbell；
也不推荐直接设置 TPN、`route_addr_idx` 或私有 path object，因为当前没有稳定公开接口。
