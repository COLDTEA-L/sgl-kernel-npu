# A5 URMA EID 对与显式 Relay 映射验证

## 1. 验证目标

本实验验证“方案 A”：不修改 UVS/UMDK 路由表，只选择源端和目的端的 URMA device、`eid_idx`，观察某个 EID 对是否稳定映射到指定物理中转卡。

脚本执行以下流程：

1. 枚举 `src_eid_idx × dst_eid_idx`。
2. 每个 EID 对用 `urma_perftest write_bw --ctp` 从源卡向目的卡发送数据。
3. 通信前后采集 0～7 卡的 HCCN 端口计数。
4. 分析非端点卡是否同时出现显著 RX 和 TX 增量。
5. 将结果分类为直连、稳定单 relay、多 relay/ECMP 或无效。

该实验不依赖 `sgl-kernel-npu` 自定义算子，因此不需要重新编译或安装算子包。

## 2. 拉取分支

```bash
cd /home/l00934901/sgl-kernel-npu
git fetch origin feature/a5-urma-explicit-relay-provider
git switch feature/a5-urma-explicit-relay-provider
git pull --ff-only origin feature/a5-urma-explicit-relay-provider
```

确认脚本存在：

```bash
ls scripts/run_a5_urma_eid_pair_scan.sh \
   scripts/analyze_a5_urma_eid_pairs.py
```

## 3. 环境检查

```bash
command -v urma_perftest
command -v urma_admin
test -x /usr/local/Ascend/driver/tools/hccn_tool
urma_admin show | tee /home/l00934901/profiling/urma_admin_show.txt
```

必须注意：`--eid_idx` 是某一个 `ubep_dev` 内部的 EID 序号，不是 `urma_admin show` 第一列的全局 `num`，而 `udmac...eN` 中的 `N` 也不是物理卡号。脚本启动时会用 `urma_admin show` 自动校验 device/EID 是否真实存在。例如：

```text
udmac0d1e6  ... eid0 ...
udmac0d1e6  ... eid1 ...
```

对应 `--src-dev udmac0d1e6 --src-eids 0,1`。同一端点如果有 `udmac0...` 和 `udmac1...` 两个 IO Die device，需要分别运行扫描，不能把两个 device 的 EID 序号混为一组。

当前机器的清单中没有 `udmac0d1e7`。根据 EID 原始地址，物理卡 `6 → 7` 的首轮候选应在同一个 UDMA device 内验证：

```text
die0: udmac0d1e6/eid0 -> udmac0d1e6/eid2
die1: udmac1d1e6/eid0 -> udmac1d1e6/eid2
```

这里 `eid2` 地址中的 `0007:0600` 或 `0047:0600` 只是“通向物理卡 7”的候选线索，最终仍以 URMA 建链结果和 HCCN RX/TX 增量为准。

实验期间应保证端点卡和候选 relay 卡尽量空闲，否则其他业务流量会污染 HCCN 计数。

## 4. 扫描 EID 对

以下以物理卡 `6 → 7` 的 die0 为例。先只验证候选 `eid0 → eid2`，不要一开始执行 81 个组合：

```bash
cd /home/l00934901/sgl-kernel-npu
mkdir -p /home/l00934901/profiling

bash scripts/run_a5_urma_eid_pair_scan.sh \
  --src-phy 6 \
  --dst-phy 7 \
  --src-dev udmac0d1e6 \
  --dst-dev udmac0d1e6 \
  --src-eids 0 \
  --dst-eids 2 \
  --bytes 4194304 \
  --iters 100 \
  --repeats 3 \
  --hccn-devices 0,1,2,3,4,5,6,7 \
  --output-root /home/l00934901/profiling
```

这里每次 URMA Write 为 4 MiB，执行 100 次并重复 3 轮。die1 需要单独执行：

```bash
bash scripts/run_a5_urma_eid_pair_scan.sh \
  --src-phy 6 --dst-phy 7 \
  --src-dev udmac1d1e6 --dst-dev udmac1d1e6 \
  --src-eids 0 --dst-eids 2 \
  --bytes 4194304 --iters 100 --repeats 3 \
  --hccn-devices 0,1,2,3,4,5,6,7 \
  --output-root /home/l00934901/profiling
```

两个候选都能建链后，才扩大到同一 device 内的合法 EID 组合。例如扫描 die0 的 `eid0～eid8` 时，`src-dev` 和 `dst-dev` 都应为 `udmac0d1e6`。脚本最后会打印本次 `RUN_DIR`。目录中包含每个 EID 对的 client/server 日志、HCCN 原始快照、计数差值和汇总结果。

若某一对失败，直接查看：

```bash
cat "${RUN_DIR}/src0_dst2_r1/server.log"
cat "${RUN_DIR}/src0_dst2_r1/client.log"
cat "${RUN_DIR}/src0_dst2_r1/hccn_analysis.log"
```

若实际工具不在 `PATH`，增加：

```bash
--perftest /实际路径/urma_perftest \
--hccn-tool /usr/local/Ascend/driver/tools/hccn_tool
```

## 5. 单独重新分析

```bash
RUN_DIR=/home/l00934901/profiling/a5_urma_eid_scan_6_to_7_YYYYMMDD_HHMMSS

python3 scripts/analyze_a5_urma_eid_pairs.py \
  --scan-dir "${RUN_DIR}" \
  --endpoint-devices 6,7 \
  --output "${RUN_DIR}/eid_pair_relay_map.tsv"
```

默认优先使用 `nic_*_all_oct_num`、`*_busi_flit_num` 或 `ub_mem_pkt_cnt_*`。若自动选择的计数器不合适，可以显式指定本机确实存在的计数器：

```bash
python3 scripts/analyze_a5_urma_eid_pairs.py \
  --scan-dir "${RUN_DIR}" \
  --endpoint-devices 6,7 \
  --tx-counter tx_busi_flit_num \
  --rx-counter rx_busi_flit_num \
  --output "${RUN_DIR}/eid_pair_relay_map.tsv"
```

背景流量较高时，可以收紧判定阈值：

```bash
python3 scripts/analyze_a5_urma_eid_pairs.py \
  --scan-dir "${RUN_DIR}" \
  --endpoint-devices 6,7 \
  --min-ratio 0.05 \
  --dominance-ratio 5 \
  --output "${RUN_DIR}/eid_pair_relay_map.tsv"
```

## 6. 输出与判定

重点查看：

```bash
column -ts $'\t' "${RUN_DIR}/stable_eid_pair_relay_map.tsv" | less -S
cat "${RUN_DIR}/eid_pair_relay_summary.json"
```

`stable_eid_pair_relay_map.tsv` 中：

- `stable_single_relay=1, relay_phy=2`：该 EID 对在所有成功重复中都由物理卡 2 表现出主导 RX+TX 信号，是“EID 对稳定映射 relay 2”的候选证据。
- `DIRECT_ONLY_OR_NO_RELAY_SIGNAL`：没有非端点卡出现足够强的双向端口计数，可能为直连，也可能是所选计数器无法观测该流量。
- `MULTI_RELAY_OR_ECMP`：多个非端点卡均出现明显信号，可能是多路径、ECMP 或背景流量污染。
- EID 对没有结果或 `pairs.tsv` 标记 `FAIL`：URMA channel/连接未建立，不可作为有效映射。

确认一个 EID 对可用于“显式 relay”至少需要同时满足：

1. 三次重复均成功，且 `relay_phy` 完全一致。
2. 候选 relay 卡同时存在显著 RX 和 TX 增量，其他非端点卡低于阈值。
3. 将负载从 1 MiB 提升到 4 MiB 后，relay 计数增量随总发送量增长。
4. 换一个 EID 对后，relay 卡也按预期发生变化；否则只能证明存在固定底层路径，不能证明 EID 可控选路。
5. 反向 `7 → 6` 单独扫描并验证。路由不能假设双向对称。

建议对候选 EID 对再做两组确认：

```bash
# 1 MiB
--bytes 1048576 --iters 100 --repeats 3

# 4 MiB
--bytes 4194304 --iters 100 --repeats 3
```

## 7. 结论边界

HCCN 计数能够证明某张物理卡的 IO Die/端口参与了转发，但不能单独证明每个包的完整 hop 顺序。只有当 EID 对、候选 relay、负载缩放和多次重复同时稳定对应时，才能把它作为方案 A 的工程验证结果。

如果所有 EID 对都只能得到固定 route、多个 relay 或不稳定结果，就说明当前 UVS 配置没有提供“通过端点 EID 选择任意 relay 卡”的能力。此时方案 A 不成立，下一阶段才需要扩展 UVS/UMDK 的 route/source-hop 配置；不能仅靠算子侧更换 EID 实现任意 `6 → relay 2 → 7`。
