# A5 CCU synthetic CommLink 显式 relay 穿刺指南

## 1. 目的与边界

本实验验证：不修改 `ubus.ko` 和全局 route table，利用驱动整机拓扑中两条真实物理边，为通信两端构造一对
指向同一中转卡 IO Die 的 EID，再用该 EID pair 构造 `HcclChannelDesc` 并建立 forwarding Channel。

当前目标为 `2 -> relay 4 -> 3`。解析器必须完成两次联接，不能把物理卡号4直接当成EID中的port字段：

```text
/usr/local/Ascend/driver/topo/950/atlas_950_1.json
  edge(2,4) -> 2侧 die/port + 4侧 ingress die/port
  edge(4,3) -> 4侧 egress die/port + 3侧 die/port

docs/topology/a5_hccn_device_topology_raw.txt
  (device2, 2侧 die/port) -> rank0 local EID
  (device3, 3侧 die/port) -> rank0 remote EID
```

只有relay4的ingress与egress位于同一IO Die，才生成IO-die-only候选。源2与目的3自身的die允许不同；运行时
分别为两个rank选择对应per-die CCU thread/channel。

旧实现把`relay_phy=4`直接解释成两端EID第三字段`0004`，产生过
`0004...4502 -> 0004...6503`。结合驱动JSON后已确认：第三字段是**本地port key**，不是物理relay ID；
该旧EID pair只是“device2 Port4到device3 Port4”，不能证明经过物理卡4，现已废弃。

`2/3/4`仅是本轮实验参数，不是代码常量。通信卡或relay变化时只修改
`--src-phy/--dst-phy/--relay-phy`；解析器会从当前机器的JSON重新查询真实边和端口，并拒绝三者重复。

## 2. 拉取、编译和安装

```bash
cd /home/l00934901/sgl-kernel-npu
git fetch origin
git switch feature/a5-ccu-discovered-path-alltoall
git pull --ff-only origin feature/a5-ccu-discovered-path-alltoall

source /usr/local/Ascend/cann-9.1.T560/set_env.sh
bash scripts/build_a5_ccu_urma_route_probe.sh \
  --install \
  --install-path /usr/local/Ascend/cann-9.1.T560
```

本次修改位于 host 侧 ChannelDesc 构造逻辑，因此需要重新编译并安装自定义 HCCL 算子包；不需要安装或重载
内核模块。

## 3. 先验证三条已发现 CommLink

以当前可用物理卡2、3为例：

```bash
for route in 0 1 2; do
  timeout --signal=TERM --kill-after=5 180 \
    bash scripts/run_a5_ccu_urma_route_probe.sh \
      --devices 2,3 --route-index "${route}" \
      --bytes 4096 --warmup 1 --iters 1 --channel-only \
    2>&1 | tee "/home/l00934901/profiling/discovered_route${route}.log"
done
```

保存 candidate 1 的 `PATH_CATALOG`、`SYNTHETIC_COMMLINK_TRACE` 以外的常规日志以及
`HcclChannelAcquire end`。candidate 1 已被 RankGraph 发布却不能建链，是判断“catalog entry 是否等于可用
provision”的负对照。

2026-09-18在物理卡2、3上的实测结果为：

| route | layer/hop/die | endpoint protocol | ChannelAcquire | 结论 |
|---|---|---|---|---|
| 0 | layer0 / hop1 / die0 | link/src/dst均为UBC_CTP(4) | status 0 | 可用直连基线 |
| 1 | layer1 / hop2 / die1 | link/src/dst均为UBC_CTP(4) | status 9，约120秒 | 已发布但未形成可用Channel |
| 2 | layer1 / hop2 / die0 | link/src/dst均为UBC_CTP(4) | status 0 | 可用forwarding基线 |

所以 route1 的问题不在“RankGraph漏生成CommLink”：它已经具有完整EID、protocol和hop信息。失败点位于
`HcclChannelAcquire` 的endpoint-pair provision/匹配阶段。后续生成新的 synthetic CommLink 时必须同时检查
“被枚举”和“可Acquire”，不能只统计RankGraph中的link数量。

### 3.1 native candidate 2 已确认到什么程度

物理卡2、3上，RankGraph公开的candidate 2为一条`layer=1/hop=2/die0/protocol=UBC_CTP` CommLink。rank0方向
公开EID pair为：

```text
device2 local : 0000:0000:003f:0200:0010:0000:df12:4b01
device3 remote: 0000:0000:003f:0200:0010:0000:df12:6b01
```

这只是candidate 2的**聚合CommLink端点**，不是六组独立relay EID pair。应用侧行为是：

```text
candidate 2
  -> 一次 HcclChannelAcquire
  -> 一个 ChannelHandle
  -> CCU kernel中一次 WriteNb(channel, wholeChunk)
  -> HCCP/MUE/UDMA内部聚合path/TP对象再做物理分流
```

在已完成的干净HCCN对照中，native0只有两个通信端点出现同量级流量；native2则在六个非端点设备
`0,1,4,5,6,7`上出现近似均衡的计数增长。六个非端点RX合计约占目的端接收量的86%，与`6/7`吻合。因此当前
最强证据是：**native2内部约等比分成7条物理路径，即1条direct加6条single-relay path并发**。

需要区分两层并发：

- native2单独运行：应用/CCU层只有一个Channel和一次`WriteNb`，七路并发发生在私有聚合path对象以下；
- 显式选择多个CommLink运行：`all_to_all_multiroute_kernel.cc`先对每个Channel连续提交`WriteNb`，随后统一
  `WaitEvent`，因此不同Channel在应用可见层是非阻塞并发。

公开接口尚未给出native2内部六条relay path各自的EID pair或relay顺序。上述“1+6”来自端口footprint，不能把
candidate 2的单个公开EID pair拆成六组未经观测的EID。

在当前机器上重新确认公开EID pair：

```bash
timeout --signal=TERM --kill-after=5 180 \
  bash scripts/run_a5_ccu_urma_route_probe.sh \
    --devices 2,3 --route-index 2 \
    --bytes 4096 --warmup 1 --iters 1 --channel-only \
  2>&1 | tee /home/l00934901/profiling/native2_catalog.log

grep -E 'route=2|acquiring route=2|HcclChannelAcquire end' \
  /home/l00934901/profiling/native2_catalog.log
```

若要重新确认“1 direct + 6 relay”的footprint，去掉`--channel-only`，加入
`--remote-only --hccn-stat --hccn-devices 0,1,2,3,4,5,6,7`。公开日志仍只会显示一组CommLink EID；六个
非端点设备的平衡计数才是内部relay数量证据。

## 4. 公共字段重建对照

下面不复制完整 `EndpointDesc` 对象，而是清零后仅重新填写公开的 `protocol/commAddr/loc`：

```bash
timeout --signal=TERM --kill-after=5 180 \
  bash scripts/run_a5_ccu_urma_route_probe.sh \
    --devices 2,3 --route-index 2 --rebuild-public \
    --bytes 4096 --warmup 1 --iters 3 --channel-only
```

- 成功：现有 forwarding candidate 不依赖 `EndpointDesc` 未公开尾部字节。
- 失败：停止 synthetic 实验；说明仍存在隐藏字段或对象身份，必须先定位该依赖。

去掉 `--channel-only` 再执行一次，要求数据校验 `PASS`。

## 5. 查看将要使用的 EID pair

```bash
python3 scripts/resolve_a5_synthetic_relay_eids.py \
  --topology docs/topology/a5_hccn_device_topology_raw.txt \
  --topology-json /usr/local/Ascend/driver/topo/950/atlas_950_1.json \
  --src-phy 2 --dst-phy 3 --relay-phy 4 --plane all
```

先检查输出，再运行数据实验：

```bash
python3 scripts/resolve_a5_synthetic_relay_eids.py \
  --topology docs/topology/a5_hccn_device_topology_raw.txt \
  --topology-json /usr/local/Ascend/driver/topo/950/atlas_950_1.json \
  --src-phy 2 --dst-phy 3 --relay-phy 4 --plane all \
  | column -s $'\t' -t
```

每行必须同时满足：

1. `src_edge`在JSON中确实连接physical device 2和4；
2. `dst_edge`确实连接physical device 4和3；
3. `relay_port_from_src`与`relay_port_to_dst`属于同一`relay_die`；
4. `src_eid`由device2的`src_die/src_port`解析；
5. `dst_eid`由device3的`dst_die/dst_port`解析。

`plane`现在表示relay卡的IO Die，不再表示两个通信端点必须使用同一个die。如果两条物理边在relay4上落入不同
die，解析器会拒绝该组合，而不是伪造跨die forwarding。换卡时只改三个physical-device参数，不要手工改EID。

## 6. 自动执行 synthetic CommLink 穿刺

先只建 Channel：

```bash
bash scripts/run_a5_ccu_synthetic_relay_probe.sh \
  --src-phy 2 --dst-phy 3 --relay-phy 4 \
  --topology-json /usr/local/Ascend/driver/topo/950/atlas_950_1.json \
  --plane all --base-route 0 \
  --bytes 4096 --warmup 1 --iters 1 \
  --channel-only --timeout-seconds 180 \
  --output-root /home/l00934901/profiling
```

先从`resolved_eid_pairs.tsv`读取当前JSON实际返回的`plane`。不要沿用旧实验中“plane0可用”的结论；旧实验使用
了错误的Port4/Port4 EID pair。若只有一行，就将下面`--plane`替换为该行的`relay_die`：

```bash
bash scripts/run_a5_ccu_synthetic_relay_probe.sh \
  --src-phy 2 --dst-phy 3 --relay-phy 4 \
  --topology-json /usr/local/Ascend/driver/topo/950/atlas_950_1.json \
  --plane all --base-route 0 \
  --bytes 4194304 --warmup 10 --iters 100 \
  --hccn-stat --hccn-devices 0,1,2,3,4,5,6,7 \
  --timeout-seconds 300 \
  --output-root /home/l00934901/profiling

echo "exit_status=$?"
```

这里必须区分两个参数：

- `--plane all|0|1`：筛选relay4的ingress/egress所在IO Die；
- `--base-route 0`：只表示使用原生route0的protocol、rank location等公共字段作为
  `HcclChannelDesc`模板，不表示数据仍走原生direct route0。

2026-09-20旧版Port4/Port4 pair虽能搬运数据，但它不是拓扑JSON证明的`2->4->3`，不能作为本轮结论复用。
新版只有在manifest中的两条edge、relay-side端口和HCCN footprint同时吻合时才确认该物理路径。

脚本自动生成：

```text
resolved_eid_pairs.tsv
case_status.tsv
synthetic_relay_summary.tsv
synthetic_relay_report.md
plane*_*.log
hccn/
```

自动脚本会为每条 synthetic candidate 强制加入 `--rebuild-public`，避免 plane1 的EID错误继承base route中
可能存在的die0私有尾部字节。`--base-route`只提供remote rank、protocol和两端device location模板，不代表
synthetic流量仍走该base route。

查看：

```bash
RUN_DIR=$(ls -dt \
  /home/l00934901/profiling/a5_ccu_synthetic_relay_* \
  | head -1)

column -s $'\t' -t "${RUN_DIR}/resolved_eid_pairs.tsv"
column -s $'\t' -t "${RUN_DIR}/synthetic_relay_summary.tsv" | less -S
sed -n '1,240p' "${RUN_DIR}/synthetic_relay_report.md"
grep -RHnE 'SYNTHETIC_COMMLINK_TRACE|HcclChannelAcquire end|PASS|failure' "${RUN_DIR}"
```

### 6.1 定位本轮HCCN差分表

```bash
find "${RUN_DIR}/hccn" \
  -name hccn_counter_deltas.tsv \
  -print

TSV=$(find "${RUN_DIR}/hccn" \
  -name hccn_counter_deltas.tsv \
  | head -1)

test -n "${TSV}" && test -f "${TSV}"
echo "HCCN_TSV=${TSV}"
```

如果没有找到TSV，先查看本轮日志中的HCCN告警；业务数据PASS仍然有效，但不能据此命名物理relay。

### 6.2 查看所有卡上流量最大的端口

```bash
{
  head -1 "${TSV}"
  awk -F $'\t' '
    NR > 1 &&
    ($4 == "tx_busi_flit_num" || $4 == "rx_busi_flit_num") {
      print
    }
  ' "${TSV}" | sort -t $'\t' -k7,7nr | head -40
} | column -s $'\t' -t
```

输出列依次为physical device、UDie、port、counter、before、after和delta。重点比较delta，不要直接比较
不同端口的after绝对值。

### 6.3 按JSON精确查看 `2 -> 4 -> 3` 的四个物理端口

本轮`resolved_eid_pairs.tsv`给出的路径是：

```text
Device2 die0/port6
  -> Device4 die1/port0
  -> Device4 die1/port3
  -> Device3 die0/port3
```

只提取这四个端口的TX/RX计数：

```bash
{
  head -1 "${TSV}"
  awk -F $'\t' '
    NR > 1 {
      is_counter = ($4 == "tx_busi_flit_num" || $4 == "rx_busi_flit_num")
      is_path_port = ($1 == 2 && $2 == 0 && $3 == 6) || ($1 == 4 && $2 == 1 && ($3 == 0 || $3 == 3)) || ($1 == 3 && $2 == 0 && $3 == 3)
      if (is_counter && is_path_port) print
    }
  ' "${TSV}"
} | column -s $'\t' -t
```

上述端口号来自本轮driver JSON，换通信卡或relay后不能照抄。必须先读取新的`resolved_eid_pairs.tsv`，将
`src_die/src_port`、`relay_die/relay_port_from_src`、`relay_die/relay_port_to_dst`和`dst_die/dst_port`
代入过滤条件。

由于当前测试是双向allgather，四个端口可能同时出现TX和RX。判断正向`2 -> 4 -> 3`时重点看：

1. Device2 die0/port6的TX；
2. Device4 die1/port0的RX；
3. Device4 die1/port3的TX；
4. Device3 die0/port3的RX。

四项应在同一实验窗口内出现与业务量同阶的增量。然后结合第6.2节的全局Top40，确认其他非端点卡没有同量级
流量；否则不能把背景业务或其他relay误判为本次指定路径。

如需按设备粗筛通信端点2、3和relay4，可使用：

```bash
{
  head -1 "${TSV}"
  awk -F $'\t' '
    NR > 1 &&
    ($4 == "tx_busi_flit_num" || $4 == "rx_busi_flit_num") &&
    ($1 == 2 || $1 == 3 || $1 == 4) {
      print
    }
  ' "${TSV}"
} | column -s $'\t' -t
```

### 6.4 一次完成base route与synthetic EID因果对照

单次HCCN结果容易受到共享服务器背景业务干扰，也无法区分`HcclChannelAcquire`究竟采用了synthetic EID，
还是仍然匹配base CommLink。使用下面的批量脚本，在固定同一组synthetic EID的前提下自动运行四组实验：

| case | EID | base CommLink |
|---|---|---:|
| `native0` | 原生 | route0 |
| `synthetic0` | relay4 synthetic pair | route0 |
| `native2` | 原生 | route2 |
| `synthetic2` | 与synthetic0完全相同 | route2 |

```bash
RELAY_DIE=1  # 按本轮resolved_eid_pairs.tsv修改为0或1

bash scripts/run_a5_ccu_synthetic_relay_causal_compare.sh \
  --src-phy 2 --dst-phy 3 --relay-phy 4 \
  --topology-json /usr/local/Ascend/driver/topo/950/atlas_950_1.json \
  --plane "${RELAY_DIE}" \
  --bytes 4194304 --warmup 10 --iters 100 \
  --repeats 3 \
  --hccn-devices 0,1,2,3,4,5,6,7 \
  --timeout-seconds 300 \
  --output-root /home/l00934901/profiling
```

脚本使用`--remote-only`，四组的网络字节数完全一致；奇数轮和偶数轮采用相反执行顺序，减小共享环境中
随时间变化的背景流量偏差。每个case均单独保存before/after HCCN快照，失败不会中止其他case。

查看自动分析：

```bash
RUN_DIR=$(ls -dt \
  /home/l00934901/profiling/a5_ccu_synthetic_causal_* \
  | head -1)

column -s $'\t' -t "${RUN_DIR}/case_status.tsv"
sed -n '1,260p' "${RUN_DIR}/causal_comparison_report.md"
column -s $'\t' -t "${RUN_DIR}/footprint_similarity.tsv"
column -s $'\t' -t "${RUN_DIR}/device_traffic_summary.tsv"
```

完整输出包括：

```text
resolved_eid_pair.tsv
case_status.tsv
footprint_long.tsv
median_port_footprints.tsv
device_traffic_summary.tsv
footprint_similarity.tsv
causal_comparison_report.md
cases/<case>_r<repeat>/run.log
cases/<case>_r<repeat>/hccn/.../hccn_counter_deltas.tsv
```

报告结论含义：

- `BASE_PATH_BOUND`：`synthetic0`更像`native0`且`synthetic2`更像`native2`，路径仍由base
  CommLink/path object控制，覆盖EID不是有效selector；
- `SYNTHETIC_EID_BOUND_CANDIDATE`：两个synthetic case相互更接近，并同时区别于两个原生基线；还必须检查
  relay4的balanced RX/TX是否稳定高于对应原生基线，才能确认relay4；
- `INCONCLUSIVE`：原生route0/route2本身无法区分、HCCN缺失或背景流量过大，需要换空闲窗口重测。

自动分类只用于缩小方向，不能覆盖物理判定规则。最终仍需在`median_port_footprints.tsv`中确认指定relay的
一进一出端口增量与业务量同阶，并排除其他relay上的同阶增量。

### 6.5 只改变每个发送方向的destination EID

第6.3节已经证明完整synthetic pair双向走`2 <-> 4 <-> 3`。本节进一步验证路径是否只由每个发送方向的
remote/destination EID决定。使用两个非对称EID pair做交叉实验：

```text
Device2 native/direct EID : 0000:0000:0007:0300:0010:0000:df12:4802
Device3 native/direct EID : 0000:0000:0000:0300:0010:0000:df12:6103
Device2 relay4-facing EID : 0000:0000:0006:0300:0010:0000:df12:4702
Device3 relay4-facing EID : 0000:0000:0003:0300:0010:0000:df12:6403
```

这些值只适用于当前physical device 2、3、4。换卡后必须分别从native route0日志和
`resolved_eid_pairs.tsv`重新读取，不能照抄。

#### 6.5.1 实验A：`2 -> 3`的destination选择relay，反向保持direct

```bash
mkdir -p /home/l00934901/profiling/dst_only_a
LOG_A=/home/l00934901/profiling/dst_only_a/run.log
status_a=0

bash scripts/run_a5_ccu_urma_route_probe.sh \
  --devices 2,3 \
  --route-index 0 \
  --rebuild-public \
  --synthetic-rank0-local-eid \
    0000:0000:0007:0300:0010:0000:df12:4802 \
  --synthetic-rank0-remote-eid \
    0000:0000:0003:0300:0010:0000:df12:6403 \
  --synthetic-rank0-local-die 0 \
  --synthetic-rank0-remote-die 0 \
  --synthetic-hop 2 \
  --bytes 4194304 --warmup 10 --iters 100 \
  --remote-only \
  --hccn-stat --hccn-devices 0,1,2,3,4,5,6,7 \
  --hccn-stat-root /home/l00934901/profiling/dst_only_a \
  >"${LOG_A}" 2>&1 || status_a=$?

echo "experiment_A_exit_status=${status_a}"
grep -E 'HcclChannelAcquire end|PASS engine|HCCL failure|Probe failed' "${LOG_A}" || true
```

完整调试输出保存在`${LOG_A}`；XShell只显示ChannelAcquire、PASS或错误摘要。

预期：

```text
2 -> 3: 2 d0/p6 TX -> 4 d1/p0 RX -> 4 d1/p3 TX -> 3 d0/p3 RX
3 -> 2: 3 d0/p0 TX --------------------------------> 2 d0/p7 RX
```

#### 6.5.2 实验B：`2 -> 3`保持direct，反向destination选择relay

```bash
mkdir -p /home/l00934901/profiling/dst_only_b
LOG_B=/home/l00934901/profiling/dst_only_b/run.log
status_b=0

bash scripts/run_a5_ccu_urma_route_probe.sh \
  --devices 2,3 \
  --route-index 0 \
  --rebuild-public \
  --synthetic-rank0-local-eid \
    0000:0000:0006:0300:0010:0000:df12:4702 \
  --synthetic-rank0-remote-eid \
    0000:0000:0000:0300:0010:0000:df12:6103 \
  --synthetic-rank0-local-die 0 \
  --synthetic-rank0-remote-die 0 \
  --synthetic-hop 2 \
  --bytes 4194304 --warmup 10 --iters 100 \
  --remote-only \
  --hccn-stat --hccn-devices 0,1,2,3,4,5,6,7 \
  --hccn-stat-root /home/l00934901/profiling/dst_only_b \
  >"${LOG_B}" 2>&1 || status_b=$?

echo "experiment_B_exit_status=${status_b}"
grep -E 'HcclChannelAcquire end|PASS engine|HCCL failure|Probe failed' "${LOG_B}" || true
```

完整调试输出保存在`${LOG_B}`；不要用`tee`，否则仍会把全部CCU launch日志输出到XShell。

预期：

```text
2 -> 3: 2 d0/p7 TX --------------------------------> 3 d0/p0 RX
3 -> 2: 3 d0/p3 TX -> 4 d1/p3 RX -> 4 d1/p0 TX -> 2 d0/p6 RX
```

#### 6.5.3 提取两组实验的关键端口

```bash
TSV_A=$(find /home/l00934901/profiling/dst_only_a \
  -name hccn_counter_deltas.tsv -type f | head -1)
TSV_B=$(find /home/l00934901/profiling/dst_only_b \
  -name hccn_counter_deltas.tsv -type f | head -1)

show_dst_selector_ports() {
  local label=$1
  local input_tsv=$2
  echo "===== ${label}: ${input_tsv} ====="
  {
    head -1 "${input_tsv}"
    awk -F $'\t' '
      NR > 1 {
        is_counter = ($4 == "tx_busi_flit_num" || $4 == "rx_busi_flit_num")
        is_path_port = ($1 == 2 && $2 == 0 && ($3 == 6 || $3 == 7)) || ($1 == 3 && $2 == 0 && ($3 == 0 || $3 == 3)) || ($1 == 4 && $2 == 1 && ($3 == 0 || $3 == 3))
        if (is_counter && is_path_port) print
      }
    ' "${input_tsv}"
  } | column -s $'\t' -t
}

test -n "${TSV_A}" && test -f "${TSV_A}"
test -n "${TSV_B}" && test -f "${TSV_B}"
show_dst_selector_ports A "${TSV_A}"
show_dst_selector_ports B "${TSV_B}"
```

严格PASS条件：

| 实验 | `2 -> 3` | `3 -> 2` |
|---|---|---|
| A | relay4四段计数闭环 | direct两端计数闭环，relay方向无同量级增量 |
| B | direct两端计数闭环，relay方向无同量级增量 | relay4四段计数闭环 |

如果A、B均满足该交叉关系，就证明每个发送方向的物理路径跟随其remote/destination EID；local/source EID无需
指向relay。若两方向仍一起切换，则EID pair作为整体匹配path，不能得出“只由dst EID控制”的结论。

## 7. 判定规则

仅 `HcclChannelAcquire status=0` 证明 EID pair 被控制面接受，尚不能证明经过指定 relay。

显式 relay 的完整 PASS 条件：

1. `resolved_eid_pairs.tsv`显示JSON中的`2<->4`、`4<->3`两条真实edge；
2. 两条edge在relay4侧落到同一`relay_die`，且EID与端点侧`die/port`一致；
3. rank0/rank1 的 synthetic endpoint 查询和 ChannelAcquire 成功；
4. 4MiB 数据校验通过；
5. HCCN同一窗口中，JSON给出的`2->4`及`4->3`物理端口计数明显增加；
6. 非指定relay没有同量级增量；
7. relay卡没有用户进程，也没有中转HBM buffer；
8. 至少三次重复结果稳定。

终端中的`PASS engine=CCU`证明端到端Channel、CCU kernel、WriteNb及数据校验成功；它本身不证明指定
relay。只有relay4相关端口出现与业务量同量级的RX/TX增量，且其他relay没有同量级增量，才可以写出
`2 -> relay4 -> 3`这一物理路径结论。

如果 EID 查询成功但 Acquire 为status 9，说明“EID存在”仍不等于“该EID pair已 provision”。下一修改点是
communicator 初始化期的 endpoint-pair/path provision，而不是 CCU `WriteNb`、TP handle 尾号或全局 UBUS route。

## 8. 调用链

```text
driver atlas_950_1.json
  -> edge(src, relay): src die/port + relay ingress die/port
  -> edge(relay, dst): relay egress die/port + dst die/port
  -> require relay ingress die == relay egress die
  -> join a5_hccn_device_topology_raw.txt
  -> rank0 local EID + local die
  -> rank0 remote EID + remote die
  -> SelectRoute(base discovered CommLink)
  -> rebuild HcclChannelDesc
  -> replace both CommAddr.eid[16]
  -> each rank selects its own local die CCU thread
  -> HcclChannelAcquire
  -> CCU kernel WriteNb
  -> destination EID driven hardware forwarding
  -> HCCN physical footprint verification
```

RankGraph 的公开API仍是只读枚举接口。本穿刺先用应用侧 synthetic CommLink catalog 验证 EID pair 是否足以建链；
若成功，再把相同生成逻辑接入多路径 AllToAll。若失败，才需要修改 RankGraph/path 的上游 provision builder。
