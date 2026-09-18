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
  --devices 2,3 \
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

本次第四步已经得到：

- `c00/c01` 两个原始 candidate 均可建链；
- `c02/c03/c05/c06/c08/c09` 的单边 endpoint/CommAddr 混搭均返回 HCCL status 9；
- `c04/c07/c10` 的成对 endpoint/CommAddr 替换均可建链；
- `c11` 只替换 location 仍可建链。

这说明建链 matcher 要求 local/remote path-specific 地址成对匹配，单边地址不构成合法 path；同时 location
并不是区分 candidate 0/2 的必要 selector。第五步的目标是确认成功变体的**物理端口 footprint 是否随地址对一起切换**：

- `c04`、`c10` 是否从 candidate B 的 footprint 切到 A；
- `c07` 是否从 candidate A 的 footprint 切到 B；
- `c11` 是否仍保持 B 的 footprint。

只运行两个基线和四个成功变体，避免重复执行已经确定必败的单边混搭：

```bash
unset LD_PRELOAD
unset A5_URMA_TP_TRACE_PREFIX

bash scripts/run_a5_ccu_channel_path_selector_forensics.sh \
  --devices 2,3 \
  --candidates 0,2 \
  --cases c00_base_a,c01_base_b,c04_b_both_endpoints_from_a,c07_a_both_endpoints_from_b,c10_b_both_comm_addrs_from_a,c11_b_locations_from_a \
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

完成后查看：

```bash
RUN_DIR=$(ls -dt \
  /home/l00934901/profiling/a5_ccu_channel_selector_* \
  | head -1)

echo "RUN_DIR=${RUN_DIR}"
column -s $'\t' -t "${RUN_DIR}/case_status.tsv"
column -s $'\t' -t "${RUN_DIR}/parameter_causality.tsv" | less -S
sed -n '1,280p' "${RUN_DIR}/channel_path_selector_report.md"
```

第五步有效的最低条件是六个 case 的 `channel_status=0` 且 `data_status=0`。如果 channel 成功但
`data_status` 非零，先看相应 `cases/<case>_r<repeat>/data.log`，不要用不完整的 HCCN delta 判断路径。

当前实验窗口使用物理卡 2、3，因此第五步命令写为 `--devices 2,3`。换卡后必须重新执行本步骤，
不能复用 6、7 卡的 endpoint 或 footprint；并应先从 `PATH_CATALOG` 确认 candidate 0/2 在新卡对上的含义。

旧版本 C++ probe 使用了不适用于 A5/950 的无端口 `hccn_tool -i DEV -stat -g`，会出现 workload
成功但报告全部为 `NO_HCCN`。当前版本先执行 `hccn_tool -g -dev_info -i DEV` 枚举每张卡的 UP
端口，再对每个 `(device, udie, port)` 执行：

```text
hccn_tool -g -stat -i DEV -u UDIE -p PORT
```

若 workload 成功但未生成 `hccn_counter_deltas.tsv`，取证脚本会将 `data_status` 置为 `126`，
不再把缺失 footprint 误报成完整成功。

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

## 7. 从既有 path matching 前移到 path provisioning

第四、第五步只证明成对 `CommAddr` 能选择一个**已经存在**的 path object，不能创建新的
`src -> 指定 relay -> dst`。下一步必须分离两个时间窗：

```text
窗口 A：HcclCommInitRootInfo
        -> communicator / topology / path catalog provisioning

窗口 B：HcclRankGraphGetLinks
        -> HcclChannelAcquire
        -> endpoint pair 匹配既有 path object
```

新增脚本运行以下对照：

| case | 操作 | 用途 |
|---|---|---|
| `c00_comm_init_only` | 只初始化/销毁 communicator | 隔离初始化期 provisioning |
| `c1_candidate_0` | Acquire direct candidate | direct 正对照 |
| `c1_candidate_1` | Acquire hop-2 candidate 1 | 可见但不可用时作为负对照 |
| `c1_candidate_2` | Acquire working forwarding candidate | forwarding 正对照 |
| `c20_base2_addrpair_from0` | base=2，换 candidate 0 CommAddr pair | request 是否随地址对翻转 |
| `c21_base0_addrpair_from2` | base=0，换 candidate 2 CommAddr pair | 反向因果验证 |

### 7.1 拉取与执行

本阶段新增 testcase 的 `--comm-init-only`，统一脚本会自动重新编译 testcase；不需要重新安装 DeepEP wheel
或自定义算子包。

```bash
cd /home/l00934901/sgl-kernel-npu
git pull origin feature/a5-ccu-discovered-path-alltoall

source /usr/local/Ascend/cann-9.1.T560/set_env.sh
unset LD_PRELOAD
unset A5_URMA_TP_TRACE_PREFIX

bash scripts/run_a5_ccu_path_provisioning_forensics.sh \
  --devices 2,3 \
  --candidates 0,1,2 \
  --timeout-seconds 180 \
  --output-root /home/l00934901/profiling
```

默认使用 `strace -ff` 记录子进程的 `ioctl/connect/send/recv/read/write`，并保存 HCCL/HCOMM/HAL/HCCP/URMA
相关 SO 的 Build ID、动态符号和字符串。若容器没有 `strace`，可用 `--no-strace` 验证 case，但不能定位
私有 request code。

### 7.2 查看结果

```bash
RUN_DIR=$(ls -dt \
  /home/l00934901/profiling/a5_ccu_path_provisioning_* \
  | head -1)

echo "RUN_DIR=${RUN_DIR}"
column -s $'\t' -t "${RUN_DIR}/case_status.tsv"
column -s $'\t' -t "${RUN_DIR}/provisioning_cases.tsv" | less -S
column -s $'\t' -t "${RUN_DIR}/candidate_ioctl_differences.tsv" | less -S
column -s $'\t' -t "${RUN_DIR}/commaddr_causal_ioctl.tsv" | less -S
sed -n '1,320p' "${RUN_DIR}/path_provisioning_report.md"
```

主要文件：

```text
path_provisioning_report.md
provisioning_cases.tsv
ioctl_request_matrix.tsv
candidate_ioctl_differences.tsv
commaddr_causal_ioctl.tsv
case_status.tsv
inventory/
cases/<case>/run.log
cases/<case>/syscall.strace.<pid>
```

## 8. 如何从结果定位可注入 first-hop/relay 的请求

### 8.1 先确认边界有效

最低要求：

```text
c00_comm_init_only status=0
c1_candidate_0    status=0
c1_candidate_2    status=0
```

`c1_candidate_1` 可以失败；若它仍是 RankGraph 可见的 hop-2 link，它正好作为“可见但没有可用 Channel”的
provisioning 负对照。

### 8.2 筛选 request code

`candidate_ioctl_differences.tsv` 保留 Channel 窗口相对 comm-init-only 新增、或 candidate 间次数不同的 ioctl。
优先查看同时满足以下模式的 request：

```text
candidate 0 != candidate 2
candidate 1 与成功 candidate 2 不同
c20/c21 随 CommAddr pair 翻转
caller SO 属于 HCOMM/HCCP/HAL/UDMA
```

调用次数差异只能定位 request，不能把它直接命名为 path ID。

新版本使用 `strace -X raw -e raw=ioctl`，因此 request 应显示为真实十六进制值，而不再出现误导性的
`ZFS_IOC_*`、`VT_GETMODE` 等跨子系统宏名。`commaddr_causal_ioctl.tsv` 中优先查看：

```text
commaddr_pair_score << base_ordinal_score
top_caller 属于 HCOMM/HCCP/HAL/UDMA
```

这表示 request 的行为随 CommAddr pair 交叉替换，而不是随原始 candidate ordinal。

### 8.3 从 request 定位 producer 与 payload

使用 `syscall.strace.<pid>` 的时间、fd 和 `-k` 调用栈，结合 `inventory/*.inventory.txt` 找到第一个提交该
request 的 SO。随后只针对筛出的 Build ID + request code 增加版本绑定的只读 hook：

```text
request entry
  -> 有界复制顶层 command struct
  -> 按已验证 length 字段复制 input/output blob
  -> candidate 0/1/2/c20/c21 raw diff
  -> 找 host 写入且随 forwarding candidate 稳定变化的字段
```

不得盲目解引用未知指针，也不能把某个偏移直接叫 `route_addr_idx/path_id`。

### 8.4 可注入 relay 参数的最终判据

字段必须同时满足：

- host 在 provisioning/channel 创建请求中写入；
- candidate 0 与 working forwarding candidate 2 稳定不同；
- candidate 1 提供合理负对照；
- 修改后创建新的合法 path object/CommAddr pair，而非只匹配已有 candidate；
- HCCN footprint 随指定 first-hop/relay 改变且数据正确；
- 不修改全局 UBUS route table。

若 host 请求 payload 没有 candidate 差异，selector 就位于 MUE/固件内部；此时需要新增最小 HCCP/MUE
opcode/ABI，而不是继续猜 TPN、handle 尾号或普通 URMA `flow_label`。

### 8.5 ChannelAcquire 定向 ioctl payload 黑箱

request 编号和次数未随 CommAddr pair 稳定翻转后，执行下面的 payload 因果实验：

```bash
cd /home/l00934901/sgl-kernel-npu
source /usr/local/Ascend/cann-9.1.T560/set_env.sh
unset LD_PRELOAD A5_URMA_TP_TRACE_PREFIX

bash scripts/run_a5_ccu_ioctl_payload_forensics.sh \
  --devices 2,3 \
  --repeats 3 \
  --timeout-seconds 180 \
  --max-per-request 64 \
  --output-root /home/l00934901/profiling
```

脚本首先运行 `preload_smoke`：只预加载最小 `liba5_ioctl_payload_trace.so`，但不开启 Acquire 标签和 payload
采集；它必须正常完成 root-info、communicator 初始化和 candidate0 ChannelAcquire。冒烟失败时脚本立即终止，
不会继续执行整批实验。

冒烟成功后才运行 candidate0、candidate2、`base2+addr0` 和 `base0+addr2` 四组对照。`LD_PRELOAD` 只注入最终
rank worker；只在 Acquire 标签窗口中读取 `_IOC_SIZE` 声明的顶层 payload，不递归读取未知指针。最小 SO 不再
包含旧 tracer 的 `urma_get_tp_list/import/bind/modify` 等 hook，避免在 root-info 初始化期间介入 URMA 控制面。

随后脚本额外运行一次 `comm_init_inventory`，在 `HcclCommInitRootInfo` 周围设置独立标签，并临时取消 request
白名单，以收集 communicator/path catalog provisioning 阶段所有带 `_IOC_SIZE` 的控制请求。正式四组实验继续只
采集已经筛选过的 request，避免高频无关调用淹没结果。

默认每个 worker 最多记录 2048 次 ioctl、每种 request 最多64次，并且每种 request 只解析一次调用栈。
终端会输出每个 case 的 `BEGIN/END`；单个成功 case 不应停留数分钟。

```bash
RUN_DIR=$(ls -dt \
  /home/l00934901/profiling/a5_ccu_ioctl_payload_* \
  | head -1)

column -s $'\t' -t "${RUN_DIR}/case_status.tsv"
column -s $'\t' -t "${RUN_DIR}/ioctl_payload_inventory.tsv" | less -S
column -s $'\t' -t "${RUN_DIR}/control_window_inventory.tsv" | less -S
column -s $'\t' -t "${RUN_DIR}/commaddr_causal_payload_offsets.tsv" | less -S
column -s $'\t' -t "${RUN_DIR}/commaddr_causal_nested_offsets.tsv" | less -S
column -s $'\t' -t "${RUN_DIR}/endpoint_token_hit_summary.tsv" | less -S
column -s $'\t' -t "${RUN_DIR}/udmac_ioctl_layout.tsv" | less -S
sed -n '1,260p' "${RUN_DIR}/ioctl_payload_report.md"
```

候选偏移必须满足：

```text
candidate0 == c20_addr0
candidate2 == c21_addr2
candidate0 != candidate2
```

并优先检查 `min_modal_confidence >= 0.90` 的记录。它们仍只是私有 ABI 候选，不能直接命名为 path ID。
`ioctl_nested_snapshots.tsv` 只对顶层 payload 中按指针宽度对齐、且能被 `process_vm_readv` 安全读取的地址保存
最多128字节；每条 payload 最多8个二级对象。它不是全进程内存扫描。若唯一 request 仍是
`anon_inode:[jfce]` 且二级内容没有四组因果差异，应将它判为完成事件路径，转向 HCOMM/HCCP matcher。

分析器还会从每个 case 的 `PATH_CATALOG` 提取 candidate 0/2 的完整128-bit src/dst endpoint，并检查它的
原始字节、16字节整体倒序、每个u64倒序和两个u64互换四种表示是否出现在 communicator 初始化或 Acquire
payload 中：

- `endpoint_token_hit_summary.tsv` 非空：先按 `window=comm_init`、`/dev/uburma/*` 过滤，定位 endpoint
  首次跨越的 request、顶层/二级结构和偏移；
- 只有 `channel_acquire + anon_inode:[jfce]` 命中：不能把它当 selector，仍应回到初始化期；
- 完全无命中：说明 endpoint 在 ioctl 前已经被转换成内部 object ID/handle/index。下一步追 `/dev/uburma/*`
  request 的用户态 builder 输入，而不是扩大 JFCE 采集。

`udmac_ioctl_layout.tsv` 将 `/dev/uburma/*` 顶层 payload 同时打印为 little-endian u64 words，便于把固定字段、
指针和疑似 handle 与反汇编中的结构访问偏移对齐。它只是布局证据，不能单独把某个 word 解释为 path ID。

如果已经完成了一轮实验，不需要重新占卡即可使用新版分析器重新生成这些结果：

```bash
python3 scripts/analyze_a5_ccu_ioctl_payload.py --run-dir "${RUN_DIR}"
```

## 9. 失败处理

### 9.1 payload 实验连续出现 `status=1`

`preload_smoke` 和四组 payload case 的成功状态都应为 `0`。如果实验尚未结束，但已完成的 case 全部是
`status=1`，应立即
`Ctrl+C` 停止；这通常表示 tracer 系统性干扰 worker，继续运行不会产生可用的因果数据。

先查看最新运行和第一个基线 case：

```bash
RUN_DIR=$(ls -dt \
  /home/l00934901/profiling/a5_ccu_ioctl_payload_* \
  | head -1)

column -s $'\t' -t "${RUN_DIR}/case_status.tsv"

sed -n '1,300p' \
  "${RUN_DIR}/cases/candidate0_r1/run.log"

ls -lh \
  "${RUN_DIR}/cases/candidate0_r1"
```

再统一搜索建链、动态加载及进程错误：

```bash
grep -RHniE \
'Segmentation|failed|error|undefined symbol|HcclChannelAcquire|rank workers|status=|LD_PRELOAD' \
"${RUN_DIR}/cases"/*/run.log
```

不要使用含 `status=1` 的 case 运行 payload 因果分析。若 candidate0 基线也失败，应先修 tracer 的透明转发；
重点检查最小 tracer 的通用 `ioctl()` 可变参数 ABI、无第三参数的 ioctl，以及 `LD_PRELOAD` 是否只注入最终
worker。不要退回包含所有 URMA hooks 的 `liba5_urma_tp_trace.so`。

- 某个 mutation Channel 超时：保留 `channel.log`，继续其他 case；总脚本不会因单 case 失败停止。
- HCCN 不可用：Channel 因果实验仍有效，但不能给出物理路径结论。
- `perf` 权限不足：去掉 `--perf-callgraph`，不影响 descriptor mutation 主实验。
- `strace` 不存在或 ptrace 被禁：去掉 `--syscall-trace`。
- 所有 mutation 都失败：说明 endpoint pair 很可能是原子注册对象，应直接追 endpoint registration/path catalog。
- 换 CANN 包：必须重新保存 inventory，旧 object offset 和 Build ID 不再适用。
