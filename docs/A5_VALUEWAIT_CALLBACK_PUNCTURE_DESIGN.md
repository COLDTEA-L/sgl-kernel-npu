# A5 ValueWait/ValueWrite + Host Callback 多路径穿刺实验设计

## 1. 文档目标

本文设计一个两 rank、1 条 direct + 2 条 relay 的控制面穿刺实验，验证：

1. 多路径资源可以在图捕获前完成创建和 CCU kernel 注册；
2. 算子到达执行点后，host 调度器可以在图外选择本次启用路径和数据比例；
3. `aclrtValueWrite` 可以通过一个 64-bit HBM policy 放行图内 `aclrtValueWait`；
4. 算子执行完成后，stream-ordered host callback 可以通知 host；
5. “完成”不是指任务已经下发，而是两个通信 rank 上 local copy、所有启用路径的
   `WriteNb`、双方 completion handshake 和 AIV `hccl.Wait` 均已结束；
6. 物理卡 0、1 是否真正参与转发，通过对应 HCCN 端口 counter 独立验证。

本穿刺不包含正式选路算法，也不维护资源矩阵。调度器只在预设策略中选择，并维护
`ARRIVED/RELEASED/DONE` 状态和计数。

## 2. 实验拓扑和路径编号

默认拓扑：

```text
communication rank 0                         communication rank 1
physical card 2                              physical card 3
       |                                             |
       |---------------- direct path ----------------|
       |                                             |
       |---- physical relay card 0 forwarding ------|
       |                                             |
       |---- physical relay card 1 forwarding ------|
```

默认命令行参数：

```text
--src-phy 2
--dst-phy 3
--relay-phys 0,1
```

relay 卡不是 HCCL rank，不运行本实验的 Python 进程和 AIV kernel。它们是 Channel
所选物理路径上的转发节点。图捕获前建立以下稳定 catalog：

| PathPolicy 下标 | Channel | 稳定 route 标签 | 含义 |
|---:|---|---:|---|
| 0 | `channels[0]` | 0 | HCCL native direct |
| 1 | `channels[1]` | 1000 | manifest 第 1 行，经 physical relay 0 |
| 2 | `channels[2]` | 1001 | manifest 第 2 行，经 physical relay 1 |

`route_id=1000+relay_phy` 只是 CCU kernel signature 的稳定标签，不是硬件 route index。

## 3. 完成语义

### 3.1 单 rank 完成条件

对 rank `r`，定义启用路径集合：

```text
P(r) = {p | pathBytes[r][p] > 0}
```

本实验只有在以下条件全部成立时，才认为该 rank 的算子完成：

```text
local copy CompletedEvent 已完成
AND
对所有 p ∈ P(r)，WriteNb CompletedEvent[p] 已完成
AND
direct channel 上的 peer completion NotifyRecord/NotifyWait 已完成
AND
AIV 侧 hccl.Wait(handle) 已返回
AND
graph 中 operator 后面的 audit copy 和 READY reset 已完成
AND
done HostFunc 已开始执行
```

CCU kernel 对所有启用路径先提交 `WriteNb`，然后逐个 `WaitEvent`：

```cpp
for (path : channels) {
    CCU_IF(pathBytes[path] != 0) {
        WriteNb(channels[path], ..., pathBytes[path], remoteEvents[path]);
    }
}
WaitEvent(localEvent);
for (path : channels) {
    CCU_IF(pathBytes[path] != 0) {
        WaitEvent(remoteEvents[path]);
    }
}
NotifyRecord(channels.front(), COMPLETION_NOTIFY_INDEX, NOTIFY_MASK);
NotifyWait(channels.front(), COMPLETION_NOTIFY_INDEX, NOTIFY_MASK);
```

因此 done callback 不会只等待 direct，也不会只等待某一条 relay；它位于整个 graph
之后，而 graph 中的 operator 只有在全部启用路径完成后才能退出。

### 3.2 两 rank 全局完成条件

单个 done callback 只直接证明当前 rank 的 stream 完成。实验最终完成条件是：

```text
rank0.local_done == true
AND
rank1.local_done == true
```

当前穿刺由两个进程分别执行 `scheduler.wait_done()`，完成数据校验后进入
`dist.barrier()`；只有两个 rank 都到达 barrier，rank 0 才打印最终 `PASS`。

未来如果把它接入单一 host 调度器，应将这一条件实现为：

```text
completed_rank_mask(op_id) == 0b11
```

但本穿刺不增加 `op_id`、全局资源矩阵或 reservation 生命周期。

### 3.3 relay 卡完成的保证边界

relay 0、1 没有独立 callback。它们的完成通过对应 Channel 的传输完成事件覆盖：

```text
rank endpoint WriteNb(channel[relay])
    -> provider/URMA transport traverses relay
    -> remote write completion event
    -> peer completion handshake
```

所以“逻辑完成”的依据是 HCOMM Channel completion contract，而不是从 relay 卡再返回一个
软件 ACK。物理 relay 是否确实承载流量属于另一项验证，使用 HCCN counter；不能只根据
`aclrtValueWrite([2,1,1])` 得出物理路径结论。

## 4. 总体架构

```text
                         graph capture 之前

  --relay-phys 0,1
          |
          v
  topology resolver: physical relay -> endpoint EID pair
          |
          v
  direct_plus_2relay.tsv
          |
          v
  HCCL LoadPlan / BuildExplicitChannels
          |
          +--> channels[0] = direct
          +--> channels[1] = relay 0
          `--> channels[2] = relay 1
          |
          v
  HcclChannelAcquire + HcclCcuKernelRegister
          |
          v
  cached CCU resources bound to capture stream

                         每次 graph replay

 operator stream                         host scheduler/control stream
       |                                             |
       | arrival aclrtLaunchHostFunc                 |
       |-------------------------------------------->|
       |                                             | choose fixed policy
       | graph: aclrtValueWait(READY)                 |
       |                                             | aclrtValueWrite(policy+READY)
       |<--------------------------------------------|
       | ValueWait satisfied                         |
       | AIV reads PathPolicy from HBM               |
       | hccl.AlltoAll(... policyWord ...)           |
       | CCU WriteNb for every enabled Channel       |
       | CCU WaitEvent for every enabled Channel     |
       | peer completion notify                      |
       | AIV hccl.Wait returns                       |
       | copy policy to audit tensor                 |
       | clear READY                                 |
       | done aclrtLaunchHostFunc                    |
       |-------------------------------------------->|
       |                                             | state = DONE
       | host wait_done() returns                    |
```

`aclrtValueWrite` 和 `aclrtValueWait` 位于不同 stream，不要求 ValueWrite 先开始：

- 如果 ValueWrite 先到，ValueWait 立即通过；
- 如果 ValueWait 先到，它阻塞到 READY 被写入；
- 两种情况下 operator 都只能在 policy 可见之后执行。

## 5. 静态路径准备调用链

### 5.1 CLI 到 manifest

```text
scripts/run_a5_valuewait_callback_puncture.sh
  --relay-phys 0,1
      |
      `-- resolve_a5_explicit_multirelay_eids.py
            |-- csv_ints()
            |-- directed_links(src, relay)
            |-- directed_links(dst, relay)
            |-- endpoint_eids(src endpoint)
            |-- endpoint_eids(dst endpoint)
            `-- resolve_relay()
                    -> one ordered TSV row per relay
      |
      `-- prepare_a5_ccu_explicit_multipath_plans.py
                    -> plans/direct_plus_2relay.tsv
```

`resolve_relay()` 要求：

- `src -> relay` 和 `relay -> dst` 拓扑边都存在；
- 两段经过同一个 relay die；
- src/dst endpoint EID 唯一；
- 所有选中路径在一个 rank 上使用同一 local IO die。

### 5.2 manifest 到 Channel

测试在 `dist.init_process_group("hccl")` 之前设置：

```text
A5_CCU_EXPLICIT_MULTIPATH_MANIFEST=/.../direct_plus_2relay.tsv
A5_CCU_EXPLICIT_MULTIPATH_PLAN_ID=valuewait-direct-plus-2relay
```

首次 warmup 调用链：

```text
Buffer::explicit_multipath_all2all_ccu()
  -> EXEC_NPU_CMD(aclnnExplicitMultipathAll2AllCcu, ...)
  -> aclnnExplicitMultipathAll2AllCcuGetWorkspaceSize()
  -> NnopbaseSetHcclServerType(CCU)
  -> ExplicitMultipathAll2AllCcuTiling()
  -> Mc2CcTilingConfig::SetCommEngine(A5_CCU_SCHED_ENGINE)
  -> AlltoAllAutoSelector::SelectCcuScheduleAlgo()
  -> CcuTempExplicitMultipathAllToAll::CalcRes()
  -> BuildExplicitChannels()
       |-- CalcChannelRequestMesh1DWithPriorityTopo() -> direct desc
       |-- LoadPlan() -> ordered relay EIDs
       `-- construct [direct, relay0, relay1] desc catalog
  -> CcuTempExplicitMultipathAllToAll::KernelRun()
  -> GetOrCreateLegacyResources()
       |-- HcclChannelAcquire()
       |-- RegisterLegacyExplicitMultipathKernel()
       |-- HcclCcuKernelRegister()
       `-- HcclCcuKernelRegisterFinish()
```

资源 cache key 包含 communicator、stream、plan ID、manifest 路径和 manifest 内容 hash。
graph replay 只复用资源，不再解析 `relay-phys`。

## 6. 动态调度调用链

### 6.1 PathPolicy 编码

PathPolicy 是一个 NPU HBM 中的单元素 `int64` tensor：

```text
bit 60..63 : ABI，当前为 1
bit 59     : READY
bit 0..3   : direct weight
bit 4..7   : relay 0 weight
bit 8..11  : relay 1 weight
```

本实验使用：

| 阶段 | weights | ready word | 预期分配 |
|---|---|---|---|
| direct-only | `[1,0,0]` | `0x1800000000000001` | direct 全量 |
| relay0-only 诊断 | `[0,1,0]` | `0x1800000000000010` | relay 0 全量 |
| relay1-only 诊断 | `[0,0,1]` | `0x1800000000000100` | relay 1 全量 |
| combined | `[2,1,1]` | `0x1800000000000112` | direct 1/2，两条 relay 各约 1/4 |

默认穿刺执行 direct-only 和 combined；严格物理路径诊断应额外执行两个 relay-only 阶段。

### 6.2 replay 调用链

```text
replay_stage(policy)
  |-- scheduler.arm(policy)
  |-- aclrtLaunchHostFunc(operator_stream, ARRIVED)
  |-- graph.replay()
  |     |-- aclrtValueWait(path_policy, READY_MASK, AND, operator_stream)
  |     |-- explicit_multipath_all2all_ccu()
  |     |-- observed_policy.copy_(path_policy)
  |     `-- aclrtValueWrite(path_policy, direct_without_READY, operator_stream)
  |-- aclrtLaunchHostFunc(operator_stream, DONE)
  `-- scheduler.wait_done(timeout)
```

ARRIVED callback 只做原子状态切换和唤醒。调度线程被唤醒后，在独立 control stream 上调用：

```cpp
aclrtValueWrite(path_policy.data_ptr(), policyWord, 0, control_stream);
```

callback 内不调用 Runtime API，避免违反 `aclrtLaunchHostFunc` callback 限制。

## 7. Device 和 HCCL 执行链

AIV kernel 读取 policy：

```cpp
GlobalTensor<int64_t> policy;
policy.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(pathPolicy), 1);
uint64_t policyWord = policy.GetValue(0);
HcclHandle handle = hccl.AlltoAll<true>(
    sendData, recvData, perRankBytes,
    HCCL_DATA_TYPE_INT8, policyWord, 1);
hccl.Wait(handle);
```

patched HCCL 将 `policyWord` 放在 `param.DataDes.strideCount`，执行：

```text
CcuTempExplicitMultipathAllToAll::KernelRun()
  -> LaunchLegacyExplicitMultipath(policyWord)
  -> DecodePathPolicy()
  -> FillPathArgs()
       -> sourceOffsets[path]
       -> remoteOffsets[path]
       -> pathBytes[path]
  -> LaunchLegacyExplicitMultipathKernel()
  -> HcclCcuKernelLaunch()
```

如果 peer payload 为 `B`：

```text
[1,0,0] -> [B, 0, 0]
[0,1,0] -> [0, B, 0]
[0,0,1] -> [0, 0, B]
[2,1,1] -> [B/2, B/4, remaining B/4]
```

中间分片按 256 Byte 对齐，最后一条启用路径承担余数。

## 8. 关键 API

| API/函数 | 所在线程/处理器 | 功能 |
|---|---|---|
| `aclrtLaunchHostFunc` | host/runtime | 在指定 stream 中插入有序 host callback |
| `aclrtValueWait` | operator stream | 等待 HBM 64-bit policy 满足 READY 条件 |
| `aclrtValueWrite` | control 或 operator stream | 发布 policy，或在图尾清除 READY |
| `HcclChannelAcquire` | HCCL host resource phase | 把 direct/relay descriptors 实例化为 CCU ChannelHandle |
| `HcclCcuKernelRegister` | HCCL host resource phase | 注册绑定有序 Channel catalog 的 CCU kernel |
| `HcclCcuKernelLaunch` | HCCL runtime | 用本次 offset/bytes 参数启动已注册 CCU kernel |
| `WriteNb` | CCU | 向一个启用 Channel 提交 non-blocking remote write |
| `WaitEvent` | CCU | 等待 local copy 或指定 Channel remote write 完成 |
| `NotifyRecord/NotifyWait` | CCU Channel | 两个通信 endpoint 的 completion handshake |
| `hccl.Wait` | AIV | 等待本次 HCCL handle 完成，阻止 operator 提前返回 |
| `dist.barrier` | 两个 host rank | 汇聚两个 rank 的本地完成，用于最终实验 PASS |
| `hccn_tool -g -stat` | host diagnostic | 采集物理端口业务 flit，证明指定 relay 链参与 |

## 9. 状态机

```text
IDLE/DONE
   |
   | arm(policy)
   v
ARMED
   |
   | arrival HostFunc
   v
ARRIVED
   |
   | scheduler aclrtValueWrite(policy+READY)
   v
RELEASED
   |
   | graph operator and all enabled paths finish
   | done HostFunc
   v
DONE
```

任何 Runtime API 失败进入 `FAILED`。`wait_done()` 有超时；如果 CCU path、peer handshake
或 operator stream 卡住，done callback 不会执行，实验必须失败，不能把 enqueue 成功当作完成。

## 10. 物理 relay 验证

### 10.1 每条 relay 的端口链

manifest 已包含：

```text
src_phy/src_die/src_port
relay_phy/relay_die/relay_port_from_src/relay_port_to_dst
dst_phy/dst_die/dst_port
```

对每条 relay 采集：

```text
forward:
src TX -> relay ingress RX -> relay egress TX -> dst RX

reverse:
dst TX -> relay egress RX -> relay ingress TX -> src RX
```

使用：

```bash
hccn_tool -g -dev_info -i DEVICE
hccn_tool -g -stat -i DEVICE -u UDIE -p PORT
```

主要 counter：

```text
tx_busi_flit_num
rx_busi_flit_num
```

### 10.2 非空闲机器上的因果对照

因为 relay 0、1 可能有背景流量，不能只做 combined 前后两次快照。严格实验使用相同 payload
和迭代次数，执行多轮 ABBA 或随机顺序的四个阶段：

```text
A direct-only [1,0,0]
B relay0-only [0,1,0]
C relay1-only [0,0,1]
D combined [2,1,1]
```

预期指纹：

| policy | relay 0 链增量 | relay 1 链增量 |
|---|---:|---:|
| `[1,0,0]` | 背景基线 | 背景基线 |
| `[0,1,0]` | 显著高于基线 | 接近基线 |
| `[0,0,1]` | 接近基线 | 显著高于基线 |
| `[2,1,1]` | 显著高于基线 | 显著高于基线 |

每个阶段应执行足够多 replay，使预期增量远高于 counter 噪声。现有
`scripts/analyze_a5_ccu_explicit_multirelay.py` 可检查每条正反向四段链是否 active，
以及链内最大/最小 counter 差是否不超过 10%。

HCCN counter 证明在同一 workload 时间窗内两条物理链都有流量；它不能单独证明 cycle-level
并发。CCU kernel 的提交顺序提供并发逻辑证据：先为所有 non-zero path 提交 `WriteNb`，再等待
任意一个 remote event。

## 11. 实验步骤

1. 从本分支构建并安装 DeepEP wheel；
2. 使用 `A5HcclExplicitMultipathExtensionVersion >= 10` 的 patched HCCL；
3. 根据实际空闲卡设置通信卡和 relay 卡；
4. 运行 topology resolver，人工检查两行 manifest 的物理端口；
5. 启动两 rank 测试；
6. warmup 在 capture stream 上完成 Channel acquire 和 CCU kernel register；
7. capture `ValueWait -> operator -> audit -> reset`；
8. replay direct-only，检查两个 rank 的 callback、policy audit 和数据正确性；
9. replay combined，执行同样检查；
10. 严格物理验证时，增加 relay0-only/relay1-only，多轮采集 HCCN counter；
11. 两个 rank 都 `wait_done()`、数据校验通过并进入 `dist.barrier()` 后，rank 0 打印最终 PASS。

默认运行：

```bash
export A5_EXPLICIT_HCCL_LIB_DIR=/path/to/patched/hccl/lib
bash scripts/run_a5_valuewait_callback_puncture.sh
```

显式覆盖卡位：

```bash
bash scripts/run_a5_valuewait_callback_puncture.sh \
  --src-phy 4 --dst-phy 5 \
  --relay-phys 6,7 \
  --hccl-lib-dir /path/to/patched/hccl/lib
```

## 12. 成功判据

### 12.1 控制面和图兼容

- graph capture 成功；
- arrival/done HostFunc 位于 graph 外；
- `aclrtValueWait`、operator、audit 和 reset 位于 graph 内；
- 每次 replay 的 `observed_policy` 等于本次 scheduler release word；
- 第二次 replay 不会复用第一次 READY。

### 12.2 单 rank 全路径完成

- 所有 non-zero `pathBytes` 都对应一个 `WriteNb` 和 `WaitEvent`；
- local copy event 已等待；
- peer completion notify 已结束；
- `hccl.Wait` 已返回；
- done callback 已执行；
- 输出与预期完全一致。

### 12.3 两 rank 全局完成

- rank 0 和 rank 1 都打印本阶段 `CASE_STAGE ... PASS`；
- 两个 rank 的 arrival/release/done count 一致；
- 两个 rank 都通过最终 `dist.barrier()`；
- 只有此时允许打印实验最终 `PASS`。

### 12.4 物理路径

- relay0-only 只显著提升 relay 0 的正反向端口链；
- relay1-only 只显著提升 relay 1 的正反向端口链；
- combined 同时显著提升两条链；
- direct-only 提供同时间尺度背景基线；
- 每条链四段 counter 均为正且基本平衡。

## 13. 失败模式

| 现象 | 判定 |
|---|---|
| manifest/EID 解析失败 | 路径 catalog 未建立，禁止进入 graph |
| patched HCCL ABI 不匹配 | fail closed，不回退普通 AllToAll |
| arrival callback 未执行 | scheduler 保持 ARMED，最终超时 |
| ValueWrite 未执行 | ValueWait 阻塞，done callback 不执行 |
| 任意启用 path 的 WaitEvent 不返回 | CCU kernel 不退出，done callback 不执行 |
| peer completion handshake 不返回 | operator 不退出，done callback 不执行 |
| 只有一个 rank done | 最终 `dist.barrier()` 不通过 |
| audit policy 不等于 release word | policy 可见性或 ordering 失败 |
| 数据不正确 | path split、offset 或 Channel 错误 |
| policy 正确但 HCCN relay counter 无增量 | 只能证明逻辑调度，不能确认物理 relay |
| HCCN counter 有背景增量但单路径指纹不成立 | 结果不确定，不能声明 relay confirmed |

## 14. 代码位置

| 内容 | 文件 |
|---|---|
| 实验 runner | `scripts/run_a5_valuewait_callback_puncture.sh` |
| relay 物理卡到 EID | `scripts/resolve_a5_explicit_multirelay_eids.py` |
| manifest/plan 文件 | `scripts/prepare_a5_ccu_explicit_multipath_plans.py` |
| 两 rank graph 测试 | `tests/python/deepep/test_a5_valuewait_callback_puncture.py` |
| callback、ValueWait/Write、host scheduler | `csrc/deepep/a5_value_wait_callback.cpp` |
| DeepEP 标准多路径入口 | `csrc/deepep/deep_ep.cpp` |
| AIV 读取 policy 和 `hccl.Wait` | `csrc/deepep/ops/op_kernel/explicit_multipath_all2_all_ccu.cpp` |
| HCCL manifest、Channel catalog、policy split | HCCL 仓 `src/ops/all_to_all_v/template/ccu/ccu_temp_explicit_multipath_alltoall.cc` |
| CCU `WriteNb/WaitEvent/Notify` | HCCL 仓 `src/ops/all_to_all_v/template/ccu/kernel/ccu_kernel_explicit_multipath_alltoall.cc` |
| HCCN 多 relay 分析器 | `scripts/analyze_a5_ccu_explicit_multirelay.py` |

## 15. 本穿刺边界

- 只支持两个通信 rank；
- 只在算子执行前调度，不在通信执行中修改路径；
- 默认验证 1 direct + 2 relay；
- 不维护资源矩阵；
- 不实现多算子公平性、抢占、回滚或调度算法；
- callback 表示 stream-ordered completion，不是 relay 卡独立上报；
- 物理路径结论必须来自 HCCN counter 因果对照，不能只来自 policy word。
