# A5 CCU Channel 路径选择因果实验指导

## 1. 实验目标

本实验不再把 `GET_TP_LIST` 当作主 selector。它通过 route0/route2 的 `HcclChannelDesc` 单变量交叉替换回答：

```text
哪个 EndpointDesc 字段足以让 HcclChannelAcquire
绑定 direct 或 direct+relay 的既有 path object？
```

实验同时保存 communicator 初始化、RankGraph、CommLink、ChannelDesc 和 Acquire/ChannelHandle；可见 URMA 边界追踪改为显式可选，
可选系统调用/调用图以及 HCCN footprint，最终生成因果比较表。

边界：该实验选择和拆解已有 candidate，不会创建新的 relay path，不修改 `ubus.ko` 或全局路由表。

## 2. 拉取分支

```bash
cd /home/l00934901/sgl-kernel-npu
git fetch origin
git switch -C feature/a5-ccu-discovered-path-alltoall \
  --track origin/feature/a5-ccu-discovered-path-alltoall
git rev-parse --short HEAD
```

## 3. 重新编译并安装

本次修改包含 route-probe host C++，必须重新编译并安装自定义库；只运行新脚本而不重装会看不到 descriptor mutation。

```bash
cd /home/l00934901/sgl-kernel-npu
source /usr/local/Ascend/cann-9.1.T560/set_env.sh

bash scripts/build_a5_ccu_urma_route_probe.sh --install

make -C examples/a5_ccu_urma_route_probe/testcase clean all
```

确认安装库时间：

```bash
ls -l /usr/local/Ascend/cann-9.1.T560/opp/vendors/cust/lib64/\
liba5_ccu_urma_route_probe.so
```

此实验使用 C++ testcase，不要求重新安装 DeepEP wheel。

## 4. 先做一次快速无 HCCN 建链扫描

先确认 12 个 case 哪些能够 READY，避免一开始就执行全部端口采样：

```bash
unset LD_PRELOAD
unset A5_URMA_TP_TRACE_PREFIX

mkdir -p /home/l00934901/profiling

bash scripts/run_a5_ccu_channel_path_selector_forensics.sh \
  --devices 6,7 \
  --candidates 0,2 \
  --repeats 1 \
  --timeout-seconds 180 \
  --skip-footprint \
  --output-root /home/l00934901/profiling
```

快速扫描默认**不加载**仓库自带的 `liba5_urma_tp_trace.so`。该 `LD_PRELOAD` tracer 曾在
root-info/communicator 初始化阶段干扰基线进程；本步骤只验证 ChannelDesc 字段变化是否改变
`HcclChannelAcquire`，不依赖 URMA API 追踪。脚本还会清除调用环境里遗留的 `LD_PRELOAD`，并用
行缓冲写入每个 case 的 `channel.log`。

正常情况下，至少以下两个基线必须为 `channel_status=0`：

```text
c00_base_a
c01_base_b
```

若基线仍为 `124`，先看：

```bash
tail -n 120 "${RUN_DIR}/cases/c00_base_a_r1/channel.log"
tail -n 120 "${RUN_DIR}/cases/c01_base_b_r1/channel.log"
```

不要把 `124` 解释为字段穿刺失败；它表示整个 case 被外层 timeout 终止。

换成其他空闲卡时只修改 `--devices`。但必须先确认在当前卡对上 candidate 0 是 direct、candidate 2 是
direct+relay；ordinal 会随卡对、拓扑和软件版本变化。

结果路径示例：

```text
/home/l00934901/profiling/a5_ccu_channel_selector_6_7_YYYYMMDD_HHMMSS/
```

脚本结束后，直接定位本次最新实验目录并查看状态、因果表和自动生成的 Markdown 报告：

```bash
RUN_DIR=$(ls -dt \
  /home/l00934901/profiling/a5_ccu_channel_selector_* \
  | head -1)

echo "RUN_DIR=${RUN_DIR}"

column -s $'\t' -t "${RUN_DIR}/case_status.tsv"
column -s $'\t' -t "${RUN_DIR}/parameter_causality.tsv" | less -S
sed -n '1,240p' "${RUN_DIR}/channel_path_selector_report.md"
```

先确认 `c00_base_a` 和 `c01_base_b` 的 `channel_status` 均为 `0`。只有两个未修改基线成功后，
c02--c11 的结果才具有字段因果意义。变体 case 的 `channel_status=124` 表示它被 `timeout` 终止，
可能说明该字段组合无法命中合法建链资源；如果基线也是 `124`，则是实验框架或环境异常，不能归因于字段穿刺。

## 5. 完整因果与 footprint 实验

快速扫描正常后执行完整实验：

```bash
bash scripts/run_a5_ccu_channel_path_selector_forensics.sh \
  --devices 6,7 \
  --candidates 0,2 \
  --bytes 4194304 \
  --warmup 3 \
  --iters 20 \
  --repeats 3 \
  --timeout-seconds 180 \
  --hccn-devices 0,1,2,3,4,5,6,7 \
  --output-root /home/l00934901/profiling
```

脚本对每个 Channel READY 的 case 执行数据正确性和 HCCN before/after 采样。为降低共享机器背景流量影响，完整
实验应尽量在低负载窗口执行，并至少重复 3 次。

## 6. 12 个 case 分别测试什么

设 candidate A=0、candidate B=2：

| case | base | donor | mutation | 问题 |
|---|---:|---:|---|---|
| `c00_base_a` | A | 无 | 无 | direct 基线 |
| `c01_base_b` | B | 无 | 无 | direct+relay 基线 |
| `c02_b_local_endpoint_from_a` | B | A | local endpoint | 本端 endpoint 是否控制路径 |
| `c03_b_remote_endpoint_from_a` | B | A | remote endpoint | 对端 endpoint 是否控制路径 |
| `c04_b_both_endpoints_from_a` | B | A | 两端 endpoint | B 是否退化为 A |
| `c05_a_local_endpoint_from_b` | A | B | local endpoint | 反向验证本端 selector |
| `c06_a_remote_endpoint_from_b` | A | B | remote endpoint | 反向验证对端 selector |
| `c07_a_both_endpoints_from_b` | A | B | 两端 endpoint | endpoint pair 是否足以复现 B |
| `c08_b_local_comm_addr_from_a` | B | A | local CommAddr | 区分 CommAddr 与 EndpointDesc 其他字段 |
| `c09_b_remote_comm_addr_from_a` | B | A | remote CommAddr | 检验 remote path-specific EID |
| `c10_b_both_comm_addrs_from_a` | B | A | 两端 CommAddr | 判断仅 EID pair 是否足够 |
| `c11_b_locations_from_a` | B | A | 两端 location | 判断 die/device location 是否参与匹配 |

每个 case 在独立两进程通信域运行，防止前一个 case 的 Channel cache 污染后一个 case。

## 7. 结果文件

```text
case_status.tsv
parameter_causality.tsv
channel_desc_variant_raw.tsv
channel_path_selector_report.md
summary.json
inventory/
cases/<case>_r<repeat>/
  case_meta.tsv
  channel.log
  channel_status.txt
  urma_tp.pid*.jsonl
  data.log
  data_status.txt
  hccn_routes_*/hccn_counter_deltas.tsv
```

关键日志标记：

```text
COMM_INIT_SCOPE
PATH_CATALOG
COMMLINK_TRACE
CHANNEL_DESC_TRACE
CHANNEL_DESC_VARIANT_TRACE
CHANNEL_ACQUIRE_SCOPE
CHANNEL_HANDLE_TRACE
```

`CHANNEL_DESC_VARIANT_TRACE` 同时保存 base raw、donor raw 和 mutation 后 raw，确保分析时知道实际传给 Acquire 的
对象，而不是仅凭 case 名推测。

## 8. 如何判断哪个参数控制路径

严格按以下顺序判断：

1. `channel_status=0`：混合 descriptor 能建立 Channel；否则该组合不是合法 path key。
2. `data_status=0`：数据正确；错误数据不进入路径判断。
3. 比较 `closer_footprint`：归一化 HCCN 计数更接近 candidate A 还是 B。
4. 检查至少 3 次 repeat 是否一致。
5. 查看非端点卡 counter 是否随 payload/iterations 近似线性增长，排除背景业务。

重要判据：

```text
c07: base A + B 的完整 endpoint pair
```

如果它稳定表现为 candidate B footprint，则 path-specific endpoint pair 很可能足以选择既有 path object。下一步应
定位 endpoint pair 的生成和注册接口。

如果 `c07` 仍保持 A、失败或结果不稳定，则路径还依赖：

- communicator side table；
- RankGraph 内部 path identity；
- Channel cache key；
- 未复制到公开字段的 pathHandle；
- HCCP/MUE 私有上下文。

`closer_footprint` 只是相对基线距离，不自动证明经过某张 relay 卡。最终 relay 身份仍需结合端口拓扑和隔离 HCCN
计数确认。

## 9. 可选 Acquire 窗口调用图与 ioctl 追踪

快速实验可以增加：

```bash
bash scripts/run_a5_ccu_channel_path_selector_forensics.sh \
  --devices 6,7 \
  --candidates 0,2 \
  --repeats 1 \
  --skip-footprint \
  --perf-callgraph \
  --syscall-trace \
  --output-root /home/l00934901/profiling
```

只有在快速扫描得到稳定结果、并且只需要对少量关键 case 补充公开 URMA 边界事件时，才增加：

```bash
--urma-trace
```

不要在 shell 中全局 `export LD_PRELOAD=liba5_urma_tp_trace.so`。该开关仍属于辅助黑箱手段，
不是第四步字段因果扫描的必要条件；若它再次导致 root-info 初始化失败，以无 tracer 的
ChannelDesc/Acquire 结果为准。

需要系统存在 `perf`、`strace`，并允许跟踪子进程。输出：

```text
cases/<case>/channel.perf.data
cases/<case>/channel.perf.txt
cases/<case>/channel.strace.*
```

这些文件用于比较 route0/route2 及最小因果 mutation 的调用图和 ioctl/connect 差异。它们不是自动解析出的
`path ID`；需要结合 Build ID 和 object-relative offset 分析。

## 10. 从实验结果继续定位 Acquire 内部

选出“能够稳定改变 footprint 的最小 mutation”，然后只对基线 A、基线 B 和该 mutation 做聚焦追踪：

```text
HcclChannelDesc 中发生变化的 endpoint bytes
  -> 第一次 read/copy/hash
  -> Channel cache key
  -> LinkData/HCCP private request
  -> internal path object
  -> handle registry
  -> ChannelHandle
```

优先检查：

1. 当前版本 `HcclChannelAcquire` 的真实 SO、Build ID 和内部调用图；
2. 第一个读取不同 endpoint 字节的内部函数；
3. Channel cache lookup 的输入、hit/miss 和返回对象；
4. `HcommChannelGetStatus(handle)` 如何从 handle registry 找回对象；
5. `GET_TP_LIST` 之前的 HCCP/HDC/MUE 私有请求；
6. 内部对象中的 endpoint key、remote ID、UASID、TPG/jetty/path resource ID。

如果 endpoint pair 足以复现路径，最终显式 relay 的改造点是 path-specific endpoint 的创建/注册控制面；如果不
足够，则应给 HCOMM 增加明确的 `pathHandle` 绑定，而不是继续猜 TP handle 或 `route_addr_idx` 数值。

## 11. 失败处理

- 某个 mutation Channel 超时：保留 `channel.log`，继续其他 case；总脚本不会因单 case 失败停止。
- HCCN 不可用：Channel 因果实验仍有效，但不能给出物理路径结论。
- `perf` 权限不足：去掉 `--perf-callgraph`，不影响 descriptor mutation 主实验。
- `strace` 不存在或 ptrace 被禁：去掉 `--syscall-trace`。
- 所有 mutation 都失败：说明 endpoint pair 很可能是原子注册对象，应直接追 endpoint registration/path catalog。
- 换 CANN 包：必须重新保存 inventory，旧 object offset 和 Build ID 不再适用。
