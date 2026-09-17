# A5 CCU discovered-path AllToAll 调用链与接口

## 总览

```text
Python test
  -> deep_ep.Buffer.ccu_urma_multiroute_alltoall_out
  -> pybind Buffer::ccu_urma_multiroute_alltoall_out
  -> dlsym HcclCcuUrmaMultiRouteAllToAll
  -> HcclRankGraphGetLayers/GetLinks
  -> PATH_CATALOG + path_uid 解析
  -> HcclChannelDescInit + CommLink endpoint copy
  -> HcclThreadAcquireWithStream(COMM_ENGINE_CCU)
  -> HcclChannelAcquire
  -> HcclCcuKernelRegister/Finish
  -> HcclCcuKernelLaunch
  -> CCU LocalCopyNb(self slice)
  -> CCU WriteNb(channel_A/B, disjoint peer chunks)
  -> WaitEvent + peer completion Notify
```

## 逐层说明

| 层次 | 文件/接口 | 作用 |
|---|---|---|
| Python | `tests/python/deepep/test_a5_ccu_urma_multiroute_all2all.py` | 建立两 rank HCCL 通信域，设置 `A5_CCU_PATH_UIDS/WEIGHTS`，构造输入并校验结果 |
| Python wrapper | `python/deep_ep/deep_ep/buffer.py` | 调用 pybind runtime 方法 |
| PyBind | `csrc/deepep/pybind_extension.cpp` | 暴露 `ccu_urma_multiroute_alltoall_out` |
| C++ bridge | `csrc/deepep/deep_ep.cpp` | `dlopen/dlsym` 自定义库，传入 buffer、HcclComm 和 aclrtStream |
| Host API | `examples/a5_ccu_urma_route_probe/op_host/all_to_all_multiroute.cc` | 计算 token、按权重切分 peer slice、构造 CCU task 参数并 launch |
| Path catalog | `examples/a5_ccu_urma_route_probe/op_host/utils.cc` | 调用 RankGraph API 枚举 CommLink，生成稳定 UID，解析选择并创建 Channel |
| CCU kernel | `examples/a5_ccu_urma_route_probe/op_kernel_ccu/all_to_all_multiroute_kernel.cc` | local copy self slice；对每条 Channel 发不重叠的 `WriteNb`；并发或受控串行等待 |
| HCCL/HCOMM | `HcclChannelAcquire`, `HcclCcuKernelRegister`, `WriteNb` | 把已发现 CommLink endpoint 交给 MUE/HCOMM 建立/获取 TP，并执行 CCU 数据搬运 |

## 建链诊断接口

`A5_CCU_TRACE_LINK=1` 时，Host probe 会在 `HcclChannelAcquire` 前后输出：

```text
COMMLINK_TRACE
CHANNEL_DESC_TRACE phase=before_acquire
CHANNEL_HANDLE_TRACE phase=after_acquire
```

raw bytes 与 `object_size` 一起记录，目的是在缺少全部私有字段定义时仍能精确比较两个 ABI 对象；raw offset 不能在没有对应版本头文件时自行命名。

`examples/a5_urma_tp_trace/liba5_urma_tp_trace.so` 是可选的 `LD_PRELOAD` 诊断库，拦截公开动态符号：

```text
urma_get_tp_list / urma_cmd_get_tp_list
urma_get_tp_attr / urma_cmd_get_tp_attr
urma_set_tp_attr / urma_cmd_set_tp_attr
urma_modify_tp / urma_cmd_modify_tp
urma_cmd_exchange_tp_info
```

它记录请求、返回 TP handle 和公开 attr，不修改建链参数或返回值。为确定不同返回 handle 的最终网络属性，GET_TP_LIST 成功后还会立即对每个 handle 执行一次只读 `urma_get_tp_attr()`，并记录 status、bitmap、SIP/DIP、MAC、VLAN、DSCP、SL 和 TTL。若 provider 返回 not support，只表示该只读查询能力未开放。GET_TP_LIST 事件同时保存共享库级调用栈，供后续把公开 liburma 边界定位回实际 HCOMM/HCCP/MUE 调用模块；地址以 object-relative offset 表示，不能跨版本直接比较绝对地址。

当前已经确认的边界是：

```text
candidate 0/2
  -> 不同 CommLink EndpointDesc
  -> 不同 HcclChannelDesc 参数
  -> 相同 HcclChannelAcquire API 行为
  -> 公开 GET_TP_LIST 的 flag/trans_mode/EID pair 可相同
  -> 返回 TP handle 集合存在差异
```

因此不能表述为“Acquire 的行为和传参都相同”。准确说法是：调用方式相同、ChannelDesc 输入不同；到公开 GET_TP_LIST 边界时可见配置相同，但隐藏的 Channel/path 上下文使其返回不同 TP 资源。

HCOMM 若绕过这些公开动态符号，trace 文件会为空；此时必须把追踪点下移至 HIXL/HCCP/MUE，不能认为 Channel 没有使用 URMA TP。

`A5_CCU_CHANNEL_HOLD_SECONDS` 让 channel-only probe 在 Channel 存活期间暂停，脚本会同期执行 `urma_admin list_res` 尝试读取 TP/TPG。驱动明确不支持时，原始错误会被保存为“不可见证据”，不会伪造 TPN、TPG 或 path ID。

## path_uid 的含义

`path_uid` 是从以下信息生成的 64-bit FNV-1a 标识：

```text
layer | protocol | hop | min(endpoint_A, endpoint_B) | max(endpoint_A, endpoint_B)
```

endpoint 顺序被规范化，因此两个 rank 能得到相同 UID。UID 的目的只是避免把易变的枚举 ordinal 当成路径身份；它不等于任何硬件 route index。

选择过程：

```text
A5_CCU_PATH_UIDS
  -> ResolvePathUids()
  -> 本会话 PATH_CATALOG 中查找 UID
  -> 对应 CommLink
  -> HcclChannelDesc.localEndpoint/remoteEndpoint
  -> HcclChannelAcquire()
```

这里显式选择的是 HCCL 已经发现的 CommLink，不是自行创建任意 source route。是否经过某张 relay 卡必须由 HCCN counter footprint 判断。

## 数据切分

两 rank 时每个 rank 的输入布局为：

```text
[发给本 rank 的 elementsPerPeer][发给 peer 的 elementsPerPeer]
```

Host 端根据权重计算互不重叠的 `sourceOffsets`、`remoteOffsets` 和 `pathBytes`。例如 peer slice 为 2 MiB、权重 2:1：

```text
channel_A: 前 2/3（按 256 B 对齐）
channel_B: 剩余 1/3
```

CCU kernel 先提交 self `LocalCopyNb`，再为所有 path 创建 `WriteNb`。concurrent 版本在所有 Write 提交后统一 `WaitEvent`；serial 对照版本每次 Write 后立即等待。两者使用同一组 Channel 和同一数据布局，差异只在等待位置。

## 与原生 HCCL AllToAll 的关系

相同点：都从 RankGraph 的 CommLink 创建 `HcclChannelDesc`，按 endpoint/die 获取 CCU Thread 和 Channel，并通过 CCU kernel 执行非阻塞搬运与事件同步。

差异：原生算法由 HCCL executor 自动决定 Channel 集合、die 分组和数据调度；本实验由用户提供 `path_uid` 和权重，仅支持两 rank，且当前要求所选 path 位于同一 die。它是验证 discovered-path 可控复用的穿刺，不是完整替代 HCCL AllToAll。

## 不能从代码中推出的结论

- `route0/route2` 与 `route_addr_idx` 没有已知对应关系。
- `hop=2` 不等于某张指定 relay 卡。
- 两个 `WriteNb` 在 CCU 指令图中并发提交，不自动证明两个物理出口同时工作。
- 普通用户态 URMA TP 的 `flow_label/port_id/spray_en` 字段，不能直接等价为 HCCL/MUE Channel 的可控 selector。

因此完整验证链必须是：

```text
CommLink UID -> Channel -> 单 path 打流 -> HCCN physical footprint
             -> A/B serial vs concurrent -> correctness/time/counter 联合判断
```
