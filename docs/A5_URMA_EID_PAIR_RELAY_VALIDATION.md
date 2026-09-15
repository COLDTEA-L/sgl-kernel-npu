# A5 URMA EID 对与显式 Relay 映射验证

## 1. 验证目标与重要修正

本实验验证“方案 A”：不修改 UVS/UMDK 路由表，选择完整的源端/目的端 EID，观察它们是否能被用户态 URMA 精确绑定，并进一步判断该 EID 对是否稳定映射到指定物理中转卡。

旧脚本仅使用 `udmac... + eid_idx`，而 `hccn_tool -g -dev_info -i DEVICE` 已证明同名 `udmac` 会出现在不同物理卡的局部视图中。因此它不能证明进程绑定到了物理卡 6/7，旧结果只保留为探索数据，不能作为方案 A 的可行性结论。

新实验先做完整 EID 可见性门禁：

1. 用 `urma_str_to_eid()` 解析 HCCL route 日志中的完整 128-bit EID。
2. 用 `urma_get_device_by_eid()` 精确解析 URMA device。
3. 用 `urma_get_eid_list()` 找到实际 `eid_index`。
4. 用 `urma_create_context()` 建 context，并逐字节核对 `context->eid`。
5. 任一步失败即停止，绝不回退到同名 device 或猜测的 `eid_idx`。

脚本执行以下流程：

1. 根据 `--src-phy/--dst-phy` 从 `urma_admin show` 自动解析两张物理卡在各 IO Die 上的 EID。
2. 枚举所有匹配到的 `(src_dev, src_eid) × (dst_dev, dst_eid)`。
3. 每个 EID 对用 `urma_perftest write_bw --ctp` 从源卡向目的卡发送数据。
4. 先通过 `hccn_tool -g -dev_info` 枚举每张卡的 UP 端口，再在通信前后逐端口采集 HCCN 计数。
5. 分析非端点卡是否同时出现显著 RX 和 TX 增量。
6. 将结果分类为直连、稳定单 relay、多 relay/ECMP 或无效。

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
   scripts/analyze_a5_urma_eid_pairs.py \
   scripts/run_a5_urma_full_eid_route_validation.sh \
   examples/a5_urma_full_eid_probe/resolve_full_eid.c
```

## 3. 首先执行：完整 EID 可见性门禁

有卡容器需要能访问 URMA 头文件和已安装的 `liburma.so`。当前环境的头文件位于 `/usr/include/ub/umdk/urma/urma_api.h`，库位于 `/lib64/liburma.so`。脚本会优先自动检查该目录，随后检查 `/usr/include/urma`、`/usr/local/include/urma` 和 UMDK 源码目录；系统已经安装开发头文件时，不要求存在 `/home/l00934901/umdk`。

```bash
cd /home/l00934901/sgl-kernel-npu

bash scripts/run_a5_urma_full_eid_route_validation.sh \
  --src-phy 6 \
  --dst-phy 7 \
  --urma-lib-dir /lib64 \
  --output-root /home/l00934901/profiling
```

当前有卡环境也可以显式写出已确认的路径：

```bash
bash scripts/run_a5_urma_full_eid_route_validation.sh \
  --src-phy 4 --dst-phy 5 \
  --urma-include /usr/include/ub/umdk/urma \
  --urma-lib-dir /lib64 \
  --output-root /home/l00934901/profiling
```

你当前机器的 `ldconfig` 输出显示 `liburma.so` 位于 `/lib64`。如果其他环境位置不同，先定位：

```bash
ldconfig -p | grep liburma
```

如果仍报告 `urma_api.h not found`，定位头文件并显式传入其所在目录：

```bash
find /usr/include /usr/local/include /home/l00934901 -name urma_api.h 2>/dev/null

bash scripts/run_a5_urma_full_eid_route_validation.sh \
  --src-phy 4 --dst-phy 5 \
  --urma-include /实际包含urma_api.h的目录 \
  --urma-lib-dir /lib64 \
  --output-root /home/l00934901/profiling
```

结果保存在：

```text
/home/l00934901/profiling/a5_urma_full_eid_6_to_7_YYYYMMDD_HHMMSS/full_eid_visibility.tsv
```

判定规则：

- `SCHEME_A_GATE=PASS`：route0/1/2 两端的完整 EID 均能精确解析，并且创建出的 context EID 完全一致；可以进入 WRITE + HCCN 计数验证。
- `status=NOT_VISIBLE`：该 HCCL/HCCN 路由 EID没有注册到用户态 URMA。此时仅凭公开 URMA API不能选择这条 route，方案 A 在当前软件栈上不成立。
- `CONTEXT_FAILED` 或 `CONTEXT_MISMATCH`：EID能枚举但不能可靠建 context，同样不能用于显式选路。

脚本支持物理 Device 0～7 中任意两张不同的卡。只需要修改 `--src-phy` 和 `--dst-phy`，脚本会先运行 HCCL route probe，解析当前卡对实际枚举出的完整 route EID，再生成本次 manifest。不会把 6/7 的 EID复用到其他卡。

例如切换为 4→5：

```bash
bash scripts/run_a5_urma_full_eid_route_validation.sh \
  --src-phy 4 --dst-phy 5 \
  --urma-lib-dir /lib64 \
  --output-root /home/l00934901/profiling
```

切换为 2→3：

```bash
bash scripts/run_a5_urma_full_eid_route_validation.sh \
  --src-phy 2 --dst-phy 3 \
  --urma-lib-dir /lib64 \
  --output-root /home/l00934901/profiling
```

自动发现会调用现有 `run_a5_ccu_urma_route_probe.sh`。route probe 在 channel acquire 之前已经打印全部 route，外层默认等待 20 秒；即便随后因某条 hop-2 channel 不可用而超时，只要 route 行已经打印，EID解析仍然有效。原始输出保存在本次结果目录的 `hccl_route_discovery.log`。

如果现场已有某卡对的 route-probe 日志，可以避免再次建 channel：

```bash
bash scripts/run_a5_urma_full_eid_route_validation.sh \
  --src-phy 4 --dst-phy 5 \
  --route-log /path/to/existing_route_probe.log \
  --urma-lib-dir /lib64 \
  --output-root /home/l00934901/profiling
```

6→7 的已知结果示例如下，但它不再硬编码进执行逻辑：

| route | hop | Device 6 source EID | Device 7 destination EID |
|---|---:|---|---|
| route0 | 1 | `0046:0300:...:d706` | `0046:0300:...:f707` |
| route1 | 2 | `003f:0200:...:cb01` | `003f:0200:...:eb01` |
| route2 | 2 | `007f:0200:...:db01` | `007f:0200:...:fb01` |

每次动态生成的完整清单位于 `route_eid_manifest.tsv`；脚本不再由 EID位字段推断物理卡。

## 4. 环境检查

```bash
command -v urma_perftest
command -v urma_admin
test -x /usr/local/Ascend/driver/tools/hccn_tool
urma_admin show | tee /home/l00934901/profiling/urma_admin_show.txt
```

必须注意：`--eid_idx` 是某一个 `ubep_dev` 内部的 EID 序号，不是 `urma_admin show` 第一列的全局 `num`，而 `udmac...eN` 中的 `N` 也不是物理卡号。默认不再手工填写 `--src-dev/--dst-dev/--src-eids/--dst-eids`。

```text
udmac0d1e6  ... eid0 ...
udmac0d1e6  ... eid1 ...
```

手工诊断时可用 `--src-dev udmac0d1e6 --src-eids 0,1` 指定端点；正常验证应省略这些参数，由脚本同时发现 `udmac0...` 和 `udmac1...` 下属于目标物理卡的EID，并分别保留完整的 `device+eid_idx` 身份。

旧扫描脚本曾根据 EID 位字段推断物理 Device；该推断已被完整拓扑证明不可靠。以下自动解析输出不能用于证明物理卡归属：

```text
die0: udmac0d1e6/eid3 (0006:0600) -> udmac0d1e6/eid2 (0007:0600)
die1: udmac1d1e6/eid3 (0046:0600) -> udmac1d1e6/eid2 (0047:0600)
```

自动解析结果写入每次运行目录的 `resolved_endpoints.tsv` 并在测试前打印。若没有解析到指定物理卡，脚本直接停止，不会用猜测的EID继续测试。

实验期间应保证端点卡和候选 relay 卡尽量空闲，否则其他业务流量会污染 HCCN 计数。

A5/950 不支持旧格式 `hccn_tool -i DEV -stat -g`。脚本采用以下流程：

```bash
hccn_tool -g -dev_info -i DEV
hccn_tool -g -stat -i DEV -u UDIE -p PORT
```

第二条命令中的 `UDIE/PORT` 由第一条命令中 `Link Status=UP` 的端口自动生成，不需要手工指定。

## 5. 旧版探索性 EID-index 扫描

只有完整 EID 门禁通过后，才可将以下命令作为补充探索。它本身不能证明精确物理卡或 route 绑定：

```bash
cd /home/l00934901/sgl-kernel-npu
mkdir -p /home/l00934901/profiling

bash scripts/run_a5_urma_eid_pair_scan.sh \
  --src-phy 6 \
  --dst-phy 7 \
  --bytes 4194304 \
  --iters 1000 \
  --repeats 3 \
  --hccn-devices 0,1,2,3,4,5,6,7 \
  --output-root /home/l00934901/profiling
```

脚本会测试同die和跨die组合；无法建链的组合记录为 `urma_status=FAIL`，其余组合继续执行。脚本最后打印本次 `RUN_DIR`。目录中包含 `resolved_endpoints.tsv`、每个EID对的client/server日志、逐UDie/Port的HCCN快照、计数差值和汇总结果。`pairs.tsv` 分开记录 `urma_status` 与 `hccn_status`。

若某一对失败，直接查看：

```bash
find "${RUN_DIR}" -name server.log -o -name client.log -o -name hccn_analysis.log
```

若实际工具不在 `PATH`，增加：

```bash
--perftest /实际路径/urma_perftest \
--hccn-tool /usr/local/Ascend/driver/tools/hccn_tool
```

## 6. 单独重新分析

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

## 7. 输出与判定

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

## 8. 结论边界

HCCN 计数能够证明某张物理卡的 IO Die/端口参与了转发，但不能单独证明每个包的完整 hop 顺序。只有当 EID 对、候选 relay、负载缩放和多次重复同时稳定对应时，才能把它作为方案 A 的工程验证结果。

如果完整 route EID 对用户态不可见，或者精确 EID 对只能得到固定 route、多个 relay 或不稳定结果，就说明当前公开 URMA 配置没有提供“通过端点 EID 选择任意 relay 卡”的能力。此时方案 A 不成立，下一阶段才需要扩展 UVS/UMDK 的 route/source-hop 配置；不能仅靠算子侧更换 EID 实现任意 `6 → relay 2 → 7`。
