# A5 CCU + URMA 显式多 relay AllToAll 指南

## 1. 本实现保证什么

本实现要求调用者显式列出所有中转物理卡。例如：

```text
physical device 2 -> relay 4 -> physical device 3
physical device 2 -> relay 5 -> physical device 3
```

命令行必须写成`--relay-phys 4,5`。解析器不会从其余卡中自动补一条路径，也不会在relay4失败时退化到
relay5、native route2或direct route。任意指定relay无法从driver topology和本地EID inventory唯一解析时，实验
在建链前失败。

这不是修改`ubus.ko`或全局route table。每条路径使用已经验证过的语义：对某一发送方向，把remote/destination
EID设为目的卡面向指定relay的EID，硬件透明经过该relay的IO Die。由于AllToAll双向通信，两端在相反方向都会
成为destination，因此manifest为每个relay同时保存通信两端的relay-facing EID。

当前限制：同一次CCU kernel launch中的所有显式路径，在每个通信端点上必须落到同一个local IO Die。解析器会
检查`src_die`集合和`dst_die`集合；跨die组合明确失败，后续需要拆成两个per-die CCU kernel。

## 2. 调用链和接口

```text
run_a5_ccu_explicit_multirelay_alltoall.sh
  |
  +-- resolve_a5_explicit_multirelay_eids.py
  |     |-- atlas_950_1.json: src<->relay、relay<->dst真实edge/port
  |     `-- a5_hccn_device_topology_raw.txt: endpoint(die,port)->128-bit EID
  |           -> 每个显式relay恰好一行manifest
  |
  +-- test_a5_ccu_urma_multiroute_all2all.py
  |     `-- deep_ep.Buffer.ccu_urma_multiroute_alltoall_out()
  |           `-- HcclCcuUrmaMultiRouteAllToAll()
  |                 `-- GetRouteResources()
  |                       |-- 读取A5_CCU_SYNTHETIC_ROUTE_MANIFEST
  |                       |-- 以native route0只重建公开protocol/location字段
  |                       |-- 每行覆盖一组rank0 local/remote EID与endpoint die
  |                       `-- HcclChannelAcquire(desc[N]) -> Channel[N]
  |                 `-- 按A5_CCU_PATH_WEIGHTS切分peer slice
  |                 `-- AllToAllMultiRouteKernel::Algorithm()
  |                       |-- WriteNb(channel[0], chunk[0])
  |                       |-- WriteNb(channel[1], chunk[1])
  |                       `-- concurrent模式在全部提交后才WaitEvent
  |
  `-- analyze_a5_ccu_explicit_multirelay.py
        |-- correctness和serial/concurrent/reversed-order时延
        `-- 每张指定relay的双向四段HCCN counter链
```

关键接口：

| 层 | 接口/变量 | 作用 |
|---|---|---|
| topology | `resolve_a5_explicit_multirelay_eids.py` | 根据显式物理relay逐条求EID pair，不做自动relay选择 |
| host | `A5_CCU_SYNTHETIC_ROUTE_MANIFEST` | 将多条显式路径传入host资源构造层 |
| HCOMM | `HcclChannelDescInit`、`HcclChannelAcquire` | 每个manifest row建立一个CCU Channel |
| split | `A5_CCU_PATH_WEIGHTS` | 控制每条显式relay承载的peer数据比例 |
| CCU | `WriteNb`、`WaitEvent` | 并发版先提交全部non-blocking write，再等待完成 |
| physical proof | `hccn_tool -g -stat` | 验证src->relay ingress->relay egress->dst链 |

`route_id=1000+relay_phy`只是本实验给CCU kernel signature使用的稳定标签，不是硬件route index。

## 3. 拉取、编译和安装

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

host侧manifest解析和Channel构造发生了变化，因此必须重新编译安装算子SO；不需要安装KO。

运行脚本会在source DeepEP vendor `set_env.bash`时临时关闭`set -u`。这是因为该vendor脚本会直接读取可能尚未
定义的`ASCEND_CUSTOM_OPP_PATH`；修复后的运行脚本会随即恢复`set -u`，用户无需提前手工export该变量。

若Python wheel中的`deep_ep_cpp`尚未包含`ccu_urma_multiroute_alltoall_out`，按本分支原有DeepEP构建流程重新
安装wheel。只更新自定义HCCL包而wheel已经是本分支版本时，无需重复安装wheel。

## 4. 先只解析两张显式relay卡

下面示例以通信卡2、3和显式relay 4、5为例。relay卡必须按实际可用拓扑修改：

```bash
python3 scripts/resolve_a5_explicit_multirelay_eids.py \
  --topology docs/topology/a5_hccn_device_topology_raw.txt \
  --topology-json /usr/local/Ascend/driver/topo/950/atlas_950_1.json \
  --src-phy 2 --dst-phy 3 \
  --relay-phys 4,5 \
  --weights 1,1 \
  | column -s $'\t' -t
```

`--relay-phys` 中的物理卡号必须互不重复；`--weights` 是各显式 relay 的切分权重，
允许重复，因此 `--weights 1,1` 表示两条路径等分数据。

输出必须恰好两行，`relay_phy`分别为4和5。每行中的：

- `src_edge`必须连接`src_phy`与该行`relay_phy`；
- `dst_edge`必须连接该行`relay_phy`与`dst_phy`；
- `relay_port_from_src`和`relay_port_to_dst`必须同属该行`relay_die`；
- 两行`src_die`必须一致，两行`dst_die`也必须一致。

如果某张relay存在多个候选，使用`--relay-planes 1,0`按relay列表顺序显式限制每张relay的IO Die。不要删除
失败relay后继续运行，因为那会改变实验定义。

## 5. 完整多路径实验

```bash
bash scripts/run_a5_ccu_explicit_multirelay_alltoall.sh \
  --src-phy 2 --dst-phy 3 \
  --relay-phys 4,5 \
  --weights 1,1 \
  --bytes 4194304 \
  --warmup 10 --iters 100 --repeats 3 \
  --hccn-devices 0,1,2,3,4,5,6,7 \
  --timeout-seconds 300 \
  --output-root /home/l00934901/profiling
```

脚本依次执行：

1. 每张显式relay单独运行，确认各Channel、数据正确性和单路时延；
2. 全部relay的`serial`控制组：每条`WriteNb`后立即`WaitEvent`；
3. 全部relay的`concurrent`组：先提交所有`WriteNb`再等待；
4. 反转Channel顺序的`concurrent_reverse`，排除“第一条Channel特殊处理”；
5. 一次带HCCN before/after快照的并发multi-route write，验证每张物理relay的双向链路。

整个实验不包含direct路径；两条都是显式指定的relay。如需以后加入direct，应把direct也作为一个明确的
manifest row，而不能用“剩余流量自动走direct”的隐式规则。

## 6. 查看结果

```bash
RUN_DIR=$(ls -dt \
  /home/l00934901/profiling/a5_ccu_explicit_multirelay_* \
  | head -1)

column -s $'\t' -t "${RUN_DIR}/resolved_explicit_relays.tsv"
column -s $'\t' -t "${RUN_DIR}/case_status.tsv"
column -s $'\t' -t "${RUN_DIR}/explicit_relay_path_validation.tsv" | less -S
sed -n '1,300p' "${RUN_DIR}/explicit_multirelay_report.md"
cat "${RUN_DIR}/explicit_multirelay_summary.json"
```

主要判据：

- 所有`single_relayN`、`serial`、`concurrent`、`concurrent_reverse`均输出数据正确性PASS；
- 每张指定relay都有正向四段链：src TX、relay ingress RX、relay egress TX、dst RX；
- 每张指定relay都有反向四段链，四段计数max/min偏差不超过10%；
- `concurrent < serial`用于判断并发收益；共享注入资源可能使它无法接近理想`max(T0,T1)`；
- `concurrent`与`concurrent_reverse`相差不超过20%，用于排除固定Channel顺序偏置。

报告`EXPLICIT_MULTIRELAY_CONFIRMED`表示数据正确且所有显式物理relay链均被HCCN确认。HCCN before/after只能
确认它们出现在同一工作负载窗口，不能单独提供cycle级重叠证据；并发因果来自CCU kernel中“全部`WriteNb`
先提交、随后统一`WaitEvent`”以及serial控制组。

## 7. 输出目录

```text
resolved_explicit_relays.tsv        # 拓扑解析的原始显式relay列表
manifests/all.tsv                   # 实际交给host层的全部路径
manifests/relay_<phy>.tsv           # 单路径控制组
manifests/all_reversed.tsv          # 反序控制组
case_status.tsv
cases/*.log
footprint/run.log
footprint/**/hccn_counter_deltas.tsv
explicit_relay_path_validation.tsv
explicit_multirelay_summary.json
explicit_multirelay_report.md
```

任何manifest中的relay都是用户命令显式给出的物理卡；脚本不会扫描并挑选“最快”的relay。
