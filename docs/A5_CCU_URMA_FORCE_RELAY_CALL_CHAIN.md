# A5 CCU+URMA FORCE Relay 单向 Write 调用链

## 1. 控制面调用链

```text
run_a5_ccu_urma_force_relay.sh
  -> a5_ub_route_ctl.py resolve
       -> 读取 /sys/bus/ub/devices/*/{uent_num,primary_cna,direct_link}
       -> phy map: physical device -> ub_entity.uent_num
       -> src.ports[i].r_uent == relay，得到真实 first-hop port
       -> relay.ports[j].r_uent == dst，验证原生 second hop
  -> a5_ub_route_ctl.py force
       -> 写 src entity 的 route_override sysfs 属性
          "force <relay_uent_num> <dst_uent_num>"
       -> ub_route_override_store()
       -> ub_validate_route_override()
       -> ub_mark_cna_for_sync(src, dst->cna)
       -> ub_route_sync_dev(src)
       -> 根据 native cna_maps 生成 bitmap
       -> ub_apply_route_override()
       -> 动态再次扫描 src port.r_uent == relay
       -> FORCE: bitmap = BIT(resolved_port.index)
       -> ub_set_route_table_entry(src, dst_cna, bitmap)
       -> ub_cfg_write_dword() 写硬件 route table
```

`cna_list`、distance、refcount、BFS 与 `cna_maps` 均保持原生状态。CLEAR 删除独立 override 后重新执行原生 sync，恢复 native bitmap。

## 2. Host 与 CCU 算子调用链

```text
run_a5_ccu_urma_route_probe.sh --one-way-src-rank 0
  -> testcase/main.cc（两个进程/rank）
  -> HcclCommInitRootInfo()
  -> HcclCcuUrmaOneWayWrite()
  -> GetRouteResources(... ONE_WAY_WRITE, sourceRank=0)
  -> HcclRankGraphGetLayers()
  -> HcclRankGraphGetLinks()
  -> 选择普通 rank0<->rank1 UBC_CTP CommLink（route-index 0）
  -> HcclThreadAcquireWithStream(COMM_ENGINE_CCU)
  -> HcclChannelDescInit()
  -> HcclChannelAcquire(COMM_ENGINE_CCU)
  -> HcclCcuKernelRegister()
       rank0: OneWayWriteKernel(source=true)
       rank1: OneWayWriteKernel(source=false)
  -> HcclCcuKernelRegisterFinish()
  -> HcclCcuKernelLaunch()
```

这里 route-index 0 只负责建立正常的源/目的 CCU+URMA Channel。真正 first-hop 已在创建 communicator/channel 前由 UBUS FORCE 改写，因此不依赖 HCCL 暴露 relay 字段。

## 3. CCU kernel 数据面

目的 rank 的静态 CCU 程序：

```text
NotifyRecord(OUTPUT_VAR_INDEX, recvBuf)
NotifyRecord(TOKEN_VAR_INDEX, recvToken)
NotifyWait(COMPLETION_NOTIFY_INDEX)
```

源 rank 的静态 CCU 程序：

```text
NotifyWait(OUTPUT_NOTIFY_INDEX)
NotifyWait(TOKEN_NOTIFY_INDEX)
CreateVariable(remote output/token)
WriteNb(channel, remoteOutput, localInput, bytes, event)
WaitEvent(event)
NotifyRecord(COMPLETION_NOTIFY_INDEX)
```

只有源 rank 发出数据 Write。目的 rank 的 CCU 只发布远端地址/token并参与完成同步。

## 4. 最终数据路径

以 `src=6, relay=2, dst=7` 为例：

```text
Device 6 HBM sendBuf
  -> CCU WriteNb
  -> HCOMM CcuUrmaChannel
  -> URMA/UDMA packet，destination 仍是 Device 7
  -> Device 6 UBUS route lookup(CNA7)
  -> FORCE bitmap 动态选择 port whose r_uent == Device 2
  -> Device 2 IO Die 收包
  -> Device 2 原生 route lookup(CNA7)
  -> port whose r_uent == Device 7
  -> Device 7 IO Die/UDMA
  -> Device 7 HBM recvBuf
```

Device 2 不改变 packet destination，不执行 host/AIV relay，不把 payload 落入 Device 2 HBM。

## 5. 关键 API 对照

| 层次 | API/结构 | 作用 |
|---|---|---|
| 用户控制 | `a5_ub_route_ctl.py` | 将物理卡映射为 UBUS entity，动态解析邻接端口 |
| UBUS sysfs | `route_override` | 接收 `force/add/clear` 命令 |
| UBUS 验证 | `ub_validate_route_override()` | 验证 first hop 和 relay 原生 second hop |
| UBUS 同步 | `ub_route_sync_dev()` | 生成最终 destination-CNA egress bitmap |
| UBUS override | `ub_apply_route_override()` | FORCE 替换、ADD 合并 egress bitmap |
| 硬件写入 | `ub_set_route_table_entry()` / `ub_cfg_write_dword()` | 写 UBUS route table |
| HCCL 拓扑 | `HcclRankGraphGetLayers/GetLinks` | 获取普通源/目的 UBC_CTP CommLink |
| HCOMM 通道 | `HcclChannelAcquire()` | 创建 CCU 可使用的通信 Channel |
| CCU 注册 | `HcclCcuKernelRegister/Finish()` | 注册源端和目的端不同的静态 CCU 程序 |
| CCU 数据搬运 | `WriteNb()` | 源端发起非阻塞 URMA 写 |
| CCU 完成 | `WaitEvent()` | 等待本次 Write 完成 |

## 6. FORCE 与后续 ADD 的关系

FORCE 只验证显式 `src -> relay -> dst`。后续 direct+relay 实验复用同一控制面，把模式改为 ADD：

```text
native bitmap = BIT(port_to_dst)
ADD bitmap    = native bitmap | BIT(port_to_relay)
```

届时 Channel 只能命名为 `channel_A/channel_B`，在端口计数证明映射前不能预设某个 Channel 就是 direct 或 relay。
