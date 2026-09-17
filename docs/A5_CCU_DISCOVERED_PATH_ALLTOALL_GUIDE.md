# A5 CCU discovered-path AllToAll 实验指导

## 目标与边界

本分支把 HCCL RankGraph 已发现的 `CommLink` 当作可选择的 path object，创建对应 CCU Channel，并把对端 slice 按权重切给多个 Channel 并发 `WriteNb`。

它不会修改 `ubus.ko` 或全局 UB route table，也不宣称能指定物理 relay 卡。`path_uid` 只标识本次 HCCL 通信会话中已经发现的 CommLink；最终物理出口必须由 HCCN counter 实验确认。

注意：旧参数 `route-index=0/2` 只是枚举序号，既不是 `route_addr_idx`，也不是 UBUS route table index 或 TP path ID。

## 1. 拉取分支

```bash
cd /home/l00934901/sgl-kernel-npu
git fetch origin
git switch -C feature/a5-ccu-discovered-path-alltoall \
  --track origin/feature/a5-ccu-discovered-path-alltoall
git rev-parse --short HEAD
```

## 2. 编译、安装并刷新 DeepEP

```bash
cd /home/l00934901/sgl-kernel-npu
source /usr/local/Ascend/cann-9.1.T560/set_env.sh

bash scripts/build_a5_ccu_urma_route_probe.sh --install

cd python/deep_ep
python3 setup.py bdist_wheel
LATEST_WHEEL=$(ls -1t dist/*.whl | head -n 1)
python3 -m pip install --force-reinstall --no-deps "${LATEST_WHEEL}"
cd ../..

source python/deep_ep/deep_ep/vendors/hwcomputing/bin/set_env.bash
python3 - <<'PY'
import deep_ep, deep_ep_cpp
print(deep_ep.__file__)
print(deep_ep_cpp.__file__)
PY
```

每次 C++ binding 或自定义 CCU 库变化后都要重新编译安装。不要用目录中任意一个旧 wheel；命令必须选择修改时间最新的包。

## 3. 枚举本次会话的 path catalog

以下示例使用物理卡 4、5；换卡只需修改 `--devices` 以及 `ASCEND_RT_VISIBLE_DEVICES`。

```bash
mkdir -p /home/l00934901/profiling

bash scripts/run_a5_ccu_urma_route_probe.sh \
  --devices 4,5 \
  --route-index 0 \
  --bytes 2097152 \
  --warmup 1 \
  --iters 1 \
  --channel-only 2>&1 | tee /home/l00934901/profiling/path_catalog_4_5.log

python3 scripts/parse_a5_ccu_path_catalog.py \
  /home/l00934901/profiling/path_catalog_4_5.log \
  --output /home/l00934901/profiling/path_catalog_4_5.tsv
```

日志中的记录格式为：

```text
PATH_CATALOG rank=0 peer=1 path_uid=... ordinal=... layer=... hop=... src_addr=... dst_addr=...
```

只有同时出现在 rank0、rank1 的 UID 才会进入 TSV。UID 由规范化后的两端地址、协议、hop 和 layer 计算；它用于本次拓扑/软件版本下的选择，升级 CANN、改变卡对或重建拓扑后应重新枚举。

## 4. 建立 CommLink 到物理出口 footprint

对 TSV 中每个 UID 对应的 `rank0_ordinal` 单独打流。当前 HCCN 采集脚本仍用 ordinal，是因为它和同一次 catalog 日志一一对应；最终 AllToAll 使用 UID。

```bash
bash scripts/run_a5_ccu_urma_route_probe.sh \
  --devices 4,5 \
  --route-index 0 \
  --bytes 4194304 \
  --warmup 100 \
  --iters 100 \
  --remote-only \
  --hccn-stat \
  --hccn-devices 0,1,2,3,4,5,6,7 \
  --hccn-stat-root /home/l00934901/profiling
```

对其他 ordinal 重复。需要同时保存空载 counter，避免把后台业务流量误判为 relay。建议至少测 1/2/4 MiB、每个大小重复 3 次，并比较随 payload 增长的 counter 斜率：

- 只在端点直连出口显著增长：direct footprint；
- 中间卡对应 ingress/egress 同时、稳定且按 payload 增长：forwarding footprint；
- 直连与 forwarding 端口同时增长：aggregate footprint；
- 只有全机背景波动：UNKNOWN，不能据此命名 relay。

## 5. CommLink → Channel → TP/path object 建链追踪

第 4 步只确认物理 footprint。本步骤比较 pure-direct candidate 与 direct+relay aggregate candidate 在建链调用链中的首次差异，不重复进行大流量 HCCN 实验。

先编译并安装包含诊断输出的最新自定义库，然后执行：

```bash
cd /home/l00934901/sgl-kernel-npu

bash scripts/run_a5_ccu_channel_trace.sh \
  --devices 4,5 \
  --routes 0,2 \
  --hold-seconds 20 \
  --urma-include /usr/include/ub/umdk/urma \
  --output-root /home/l00934901/profiling
```

如果已经确认本机所有 `urma_admin list_res` 都返回 `not support query tp/dev stats`，后续复测可跳过这组无效查询：

```bash
bash scripts/run_a5_ccu_channel_trace.sh \
  --devices 6,7 \
  --routes 0,2 \
  --hold-seconds 5 \
  --urma-include /usr/include/ub/umdk/urma \
  --skip-resource-snapshot \
  --output-root /home/l00934901/profiling
```

脚本完成以下两次独立实验：

- 第 5.1 步：candidate 0 执行 `--channel-only`，保存 direct 建链诊断；
- 第 5.2 步：candidate 2 执行 `--channel-only`，保存 aggregate 建链诊断；
- 第 5.3 步：比较两组 `CommLink`、`HcclChannelDesc`、Channel handle、可观察的 URMA TP 事件和 TP/TPG resource snapshot。

结果目录包含：

```text
candidate_0/channel.log
candidate_0/urma_tp.pid*.jsonl
candidate_0/{before,active,after}_resources.txt
candidate_2/...
channel_trace_comparison.md
channel_trace_comparison.json
```

诊断代码输出：

- `COMMLINK_TRACE`：完整对象大小、两端 CommAddr、EndpointDesc raw bytes 和 CommLink raw bytes；
- `CHANNEL_DESC_TRACE`：传给 `HcclChannelAcquire` 前的完整 ChannelDesc raw bytes及关键字段；
- `CHANNEL_HANDLE_TRACE`：建链后的 handle、状态和 path UID；
- `GET_TP_LIST`：公开 liburma 的 TP 获取边界，记录配置和返回 handle；
- 每条 `GET_TP_LIST` 还记录 `caller_frames`，包含调用栈中的共享库、object offset，以及可解析时的动态符号名；它用于定位 HCOMM/HCCP/MUE 到 liburma 的最后一个可见调用点；
- `GET_TP_ATTR`：除记录 HCOMM 自身调用外，追踪器还会在每次 GET_TP_LIST 成功后主动查询每个返回 handle，记录查询状态、bitmap、SIP/DIP、MAC、VLAN、DSCP、SL、TTL；
- `SET_TP_ATTR/MODIFY_TP`：TP 被选中后写入的属性，包括可见的 `peer_tpn`、`spray_en`、`local_net_addr_idx`、UDP、flow label 和 `port_id`；
- `EXCHANGE_TP_INFO`：本端/对端 TP handle 与 PSN 的配对边界。
- `TP_ACTIVATION`：拦截 `urma_import_jfr[_ex]`、`urma_import_jetty[_ex]`、`urma_bind_jetty[_ex]`；其中 ex 接口直接记录 `active_tp_handle/active_peer_tp_handle`，普通接口记录返回 target 的 TPN、target EID 和 ID。

如果 `urma_tp.pid*.jsonl` 不存在，不能解释为“没有 TP”。它表示当前 HCOMM/HCCP/MUE 建链没有经过可拦截的公开 liburma 符号，可能使用隐藏符号、静态绑定或内部控制面接口。

如果 TP 与 DEV stats 全部返回 `not support query`，只说明当前驱动关闭了公共资源查询面；它不否定已经成功的 `HcclChannelAcquire`，也不能作为“没有 TP”或“没有 relay”的证据。已经得到一次该结果后，不必反复执行 resource snapshot，使用 `--skip-resource-snapshot` 即可。

比较报告会分别统计五类 URMA 事件。判断顺序是：

1. 若 candidate 0/2 的 `SET_TP_ATTR`、`MODIFY_TP` 或 `EXCHANGE_TP_INFO` 出现稳定差异，沿差异字段继续定位 TP/path selector；
2. 若 GET_TP_LIST 的可见配置相同但 handle 集合不同，说明 candidate 已影响到底层 TP 资源选择；继续比较报告中的逐 EID-pair handle 集合和主动 TP 属性查询，不能把 handle 尾号直接命名为 direct/relay；
3. 若主动 `GET_TP_ATTR` 成功，比较只在差异 handle 上出现的 SIP/DIP、MAC、VLAN/DSCP/SL/TTL，建立 `candidate → handle → network attributes` 映射；
4. 若主动查询也返回 not support，则当前公开 API 最远只能证明“返回了不同 TP handle”。下一步需追踪 GET_TP_LIST ioctl/UDMA/HCCP/MUE 如何把不同 `HcclChannelDesc` 映射到这些 handle；
5. 若后续三类事件均未出现，或出现但无 candidate 相关差异，则路径绑定位于公开 liburma API 之外。下一步应追踪 HCCP/MUE 消费 `HcclChannelDesc/EndpointDesc` 的内部入口，不能根据 handle 数值猜测 `route_addr_idx`。

UMDK 兼容控制面源码显示，RM/UM 的典型流程为：

```text
GET_TP_LIST
  -> active_tp_cfg.tp_handle
  -> optional EXCHANGE_TP_INFO
  -> import_jfr_ex / import_jetty_ex / bind_jetty_ex
  -> returned target_tpn
```

因此判断 path→TP 绑定时，优先看报告的“import/bind 最终激活的 TP”表，而不是仅凭 GET_TP_LIST 返回池。如果该表为空，表示 HCCP 使用 provider function pointer、隐藏符号或私有 `ascend_urma_*` 路径绕过了公开入口；此时应根据 caller stack 定位实际 provider SO，再对真实导出符号做版本匹配后的 hook。

特别注意：candidate 0/2 的 `HcclChannelAcquire` 行为相同，但传入参数并不相同；两者的 `HcclChannelDesc` 包含不同的 CommLink endpoint。当前重复实验显示，在随后可见的 GET_TP_LIST 中 `flag/trans_mode/local_eid/peer_eid` 相同，TP handle 集合也可以完全相同；个别运行出现的 handle 尾号互换并不稳定，不能当作路径标识。这更支持路径身份由 Acquire 之前的 endpoint/path 上下文及内部 Channel object 持有，而非公开 TP handle/attr 持有。

比较器会忽略 PID、context 指针和 backtrace 等进程局部字段，再判断语义事件是否不同。只有重复实验中稳定存在的 EID/配置/handle 集合差异才可继续解释；单次 handle 尾号互换不是路径证据。若 handle 集合与属性最终相同，则不能再假设“final TP 不同”，应根据 `caller_frames` 找到内部 Channel/path object 的消费位置。

本步骤不需要再次执行第 4 步的 100 次 HCCN 打流；`channel-only` 建链成功即可。只有某些 TP 资源在首次数据搬运后才出现时，才将 probe 改成 `--warmup 1 --iters 3` 的少量打流诊断。

### 5.4 同一通信域内按 path 标记并顺序建链

前面的 candidate 0/2 若分别由两次进程启动完成，TP handle/TPN 的差异可能只是 communicator 重建或资源分配顺序造成。使用下面的受控实验，在每个 rank 的**同一个进程、同一个 communicator** 中依次单独 acquire 两个 CommLink，并用 `trace_label` 标记 acquire 窗口；脚本还会自动交换顺序并重复三次：

```bash
cd /home/l00934901/sgl-kernel-npu

# 本功能修改了 route-probe 的 host C++，需重新构建并安装一次。
bash scripts/build_a5_ccu_urma_route_probe.sh --install

bash scripts/run_a5_ccu_same_process_tp_binding.sh \
  --devices 6,7 \
  --routes 0,2 \
  --repeats 3 \
  --urma-include /usr/include/ub/umdk/urma \
  --output-root /home/l00934901/profiling
```

每个顺序用例实际执行：

```text
初始化一次 HCCL communicator
  -> trace_label(path_uid=A, ordinal=A, acquire_order=0)
  -> HcclChannelAcquire(A, channelNum=1)
  -> 保留 channel A，不销毁 communicator
  -> trace_label(path_uid=B, ordinal=B, acquire_order=1)
  -> HcclChannelAcquire(B, channelNum=1)
  -> 同时保留 channel A/B 至进程退出
```

输出目录形如：

```text
/home/l00934901/profiling/a5_ccu_same_process_tp_6_7_YYYYMMDD_HHMMSS/
  order_0_2_r1/
  order_2_0_r1/
  ...
  tp_binding_by_path.tsv
```

重点查看 `tp_binding_by_path.tsv` 的：

- `path_uid/ordinal`：当前 acquire 的 CommLink；
- `acquire_order`：本次是第一个还是第二个 acquire；
- `active_tp_handles/target_tpns/target_ids`：该标记窗口内 import/bind 最终激活的资源；
- `remote_eid`：激活对应的对端 endpoint。

判断规则：若同一个 `path_uid` 在 `0,2` 与 `2,0` 两种顺序、多个 repeat 中仍稳定对应同一类 TP/TPN，而不是稳定对应“第一个/第二个 acquire”，才支持 `CommLink -> active TP` 的稳定绑定结论。若 TP/TPN 随 acquire 顺序变化，则先按资源分配噪声处理。该实验仍不能单独给出 `TPN -> physical port/relay`，后续还必须把各 channel 的单独打流与 HCCN footprint 对齐。

## 6. 运行显式 path UID AllToAll

从 `path_catalog_4_5.tsv` 选择两个 UID：

```bash
PATH_A=<第一个path_uid>
PATH_B=<第二个path_uid>

bash scripts/run_a5_ccu_discovered_path_alltoall.sh \
  --devices 4,5 \
  --path-uids "${PATH_A},${PATH_B}" \
  --path-weights 2,1 \
  --bytes 2097152 \
  --warmup 100 \
  --iters 100 \
  --output-root /home/l00934901/profiling
```

脚本在同一配置下依次运行：native HCCL、A-only、B-only、A+B serial、A+B concurrent，并保存 `summary.json`。`2,1` 表示 peer slice 的 2/3 给 A、1/3 给 B；不是给每条路径各发送完整 2 MiB。

并发实验的最低判据：

1. correctness PASS；
2. `T_concurrent < T_serial`；
3. aggregate bandwidth 优于最佳单路径；
4. 同一测量窗口内两个预期 footprint 都有显著 counter 增量。

`T_concurrent ≈ max(T_A_chunk,T_B_chunk)` 只是理想参考，不作为硬性 PASS。若只有前两项而没有物理 counter 证据，只能说 CCU 命令并发提交，不能说物理链路并发。

## 7. 单独调用 Python 接口

```bash
export ASCEND_RT_VISIBLE_DEVICES=4,5
export HCCL_OP_EXPANSION_MODE=CCU_SCHED
export HCCL_BUFFSIZE=2300

python3 -m torch.distributed.run --standalone --nproc-per-node=2 \
  tests/python/deepep/test_a5_ccu_urma_multiroute_all2all.py \
  --implementation multiroute \
  --path-uids "${PATH_A},${PATH_B}" \
  --path-weights 2,1 \
  --schedule concurrent \
  --bytes 2097152 --warmup 100 --iters 100
```

故障判断：`path_uid is not in this session catalog` 表示 UID 已过期或卡对不一致；`selected routes span die` 表示当前实现不能把跨 die path 放进同一个 CCU kernel，需要按 die 分组，不能强行运行。
