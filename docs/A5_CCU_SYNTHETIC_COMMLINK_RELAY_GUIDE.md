# A5 CCU synthetic CommLink 显式 relay 穿刺指南

## 1. 目的与边界

本实验验证：不修改 `ubus.ko` 和全局 route table，仅根据整机拓扑为通信两端选择一对具有相同 relay route-key
的 EID，构造 `HcclChannelDesc`，能否让 `HcclChannelAcquire` 建立一条新的 forwarding Channel。

例如物理卡 `6 -> relay 2 -> 7`，拓扑记录中可解析出两组候选：

```text
plane0: 0002...c301 -> 0002...e301, die0
plane1: 0042...d301 -> 0042...f301, die1
```

EID 不在代码中硬编码；解析器按 `src/dst/relay/plane` 从拓扑文件动态选择两端同一 `udmac`、同一 die、同一
route-key 的 EID。这里的 route-key 是实验性拓扑编码解释，最终 relay 必须由 HCCN 端口计数确认。

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

以当前可用物理卡6、7为例：

```bash
for route in 0 1 2; do
  timeout --signal=TERM --kill-after=5 180 \
    bash scripts/run_a5_ccu_urma_route_probe.sh \
      --devices 6,7 --route-index "${route}" \
      --bytes 4096 --warmup 1 --iters 1 --channel-only \
    2>&1 | tee "/home/l00934901/profiling/discovered_route${route}.log"
done
```

保存 candidate 1 的 `PATH_CATALOG`、`SYNTHETIC_COMMLINK_TRACE` 以外的常规日志以及
`HcclChannelAcquire end`。candidate 1 已被 RankGraph 发布却不能建链，是判断“catalog entry 是否等于可用
provision”的负对照。

## 4. 公共字段重建对照

下面不复制完整 `EndpointDesc` 对象，而是清零后仅重新填写公开的 `protocol/commAddr/loc`：

```bash
timeout --signal=TERM --kill-after=5 180 \
  bash scripts/run_a5_ccu_urma_route_probe.sh \
    --devices 6,7 --route-index 2 --rebuild-public \
    --bytes 4096 --warmup 1 --iters 3 --channel-only
```

- 成功：现有 forwarding candidate 不依赖 `EndpointDesc` 未公开尾部字节。
- 失败：停止 synthetic 实验；说明仍存在隐藏字段或对象身份，必须先定位该依赖。

去掉 `--channel-only` 再执行一次，要求数据校验 `PASS`。

## 5. 查看将要使用的 EID pair

```bash
python3 scripts/resolve_a5_synthetic_relay_eids.py \
  --topology docs/topology/a5_hccn_device_topology_raw.txt \
  --src-phy 6 --dst-phy 7 --relay-phy 2 --plane all
```

输出中的 `udmac`、die、两端 EID 必须成对一致。换成物理卡2、3或4、5时，只修改三个 physical-device 参数，
不要手工修改 EID。

## 6. 自动执行 synthetic CommLink 穿刺

先只建 Channel：

```bash
bash scripts/run_a5_ccu_synthetic_relay_probe.sh \
  --src-phy 6 --dst-phy 7 --relay-phy 2 \
  --plane all --base-route 0 \
  --bytes 4096 --warmup 1 --iters 1 \
  --channel-only --timeout-seconds 180 \
  --output-root /home/l00934901/profiling
```

再进行数据正确性和端口 footprint 验证：

```bash
bash scripts/run_a5_ccu_synthetic_relay_probe.sh \
  --src-phy 6 --dst-phy 7 --relay-phy 2 \
  --plane all --base-route 0 \
  --bytes 4194304 --warmup 10 --iters 100 \
  --hccn-stat --hccn-devices 0,1,2,3,4,5,6,7 \
  --timeout-seconds 300 \
  --output-root /home/l00934901/profiling
```

脚本自动生成：

```text
resolved_eid_pairs.tsv
case_status.tsv
synthetic_relay_summary.tsv
synthetic_relay_report.md
plane*_*.log
hccn/
```

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

## 7. 判定规则

仅 `HcclChannelAcquire status=0` 证明 EID pair 被控制面接受，尚不能证明经过指定 relay。

显式 relay 的完整 PASS 条件：

1. rank0/rank1 的 synthetic endpoint 查询和 ChannelAcquire 成功；
2. 4MiB 数据校验通过；
3. HCCN 同一实验窗口中，源到指定 relay、指定 relay 到目的的对应端口计数明显增加；
4. 非指定 relay 没有同量级增量；
5. relay 卡没有用户进程，也没有中转 HBM buffer；
6. 至少三次重复结果稳定。

如果 EID 查询成功但 Acquire 为status 9，说明“EID存在”仍不等于“该EID pair已 provision”。下一修改点是
communicator 初始化期的 endpoint-pair/path provision，而不是 CCU `WriteNb`、TP handle 尾号或全局 UBUS route。

## 8. 调用链

```text
topology raw inventory
  -> resolve_a5_synthetic_relay_eids.py
  -> rank0 local/remote EID pair + die
  -> SelectRoute(base discovered CommLink)
  -> rebuild HcclChannelDesc
  -> replace both CommAddr.eid[16]
  -> HcclChannelAcquire
  -> CCU kernel WriteNb
  -> destination EID driven hardware forwarding
  -> HCCN physical footprint verification
```

RankGraph 的公开API仍是只读枚举接口。本穿刺先用应用侧 synthetic CommLink catalog 验证 EID pair 是否足以建链；
若成功，再把相同生成逻辑接入多路径 AllToAll。若失败，才需要修改 RankGraph/path 的上游 provision builder。
