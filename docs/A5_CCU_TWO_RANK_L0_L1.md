# 两卡 CCU+URMA：L0、L1、L0+L1 分流验证

本用例独立于 EP8。分支 `feature/a5-ccu-two-rank-l0-l1-probe` 基于 `9a378d4`，只新增探针、运行脚本、CPU 测试与本文档；不修改 EP8 算子、planner、route 包或模型接入。

目标是验证当前机器的两张 NPU 能否通过 **L0 fullmesh**、**L1 CLOS 网络**分别通信，以及同时使用两者是否增加吞吐。拓扑有 L1 端口、端口 UP、SDK 返回 L1 链路、通道建链、数据正确、物理流量走 L1、性能增加，是不同层次的证据。

## 0. 单端口版本：默认仅选择单端口 L1 候选

旧版 `4ee4ca9` 现场选择的一个 L1 逻辑通道实际使用了 p1/p2 两个端口，因此 32MiB 下 L1 约90GB/s、L0约47GB/s并不矛盾。新版默认 `--l1-port-policy single`：查询每条 L1 链路两端的 `ENDPOINT_ATTR_BW_COEFF`，仅在查询均成功且两端系数都等于1时允许选中。Ascend950 的公开 rank graph 实现用接口 `GetPorts().size()` 返回这个系数；它仍是拓扑声明，最终需要端口计数验证。

**这是选择已存在单端口端点的版本，不是把聚合 EID 强行拆成单端口的接口。** 当前9.2 SDK的 CCU/UB_CTP channel descriptor 没有 port-id/port-mask 字段；`ubMemAttr.pathMode` 只用于 UB_MEM，不能拿来控制 UB_CTP。本用例不改 EID 字节、不关物理口、不改系统 rootinfo/Atlas，不影响其他用户。

默认 single 模式的 `--l1-index` 指过滤后的 `single_index`；`--l1-port-policy any` 才使用原来的完整 catalog `index`，仅供复现旧聚合实验。新旧结果会记录不同 policy，不能当成相同路径比较。L0筛选和 EP8 均未改动。

拉取更新后先重编独立二进制，再运行：

```bash
bash scripts/build_a5_ccu_l0_l1_probe.sh "$A5_CANN_PATH" \
  /home/l00934901/profiling/ccu-l0-l1-build
bash scripts/run_a5_ccu_l0_l1_probe.sh --suite discover
# 查看两个 rank 的 L1_SINGLE_PORT_DISCOVERY、L0L1_CATALOG。
# 只有两端都有匹配的 single_index 才继续：
bash scripts/run_a5_ccu_l0_l1_probe.sh --suite correctness --mode all --bytes 8388608
bash scripts/run_a5_ccu_l0_l1_probe.sh --suite benchmark --mode l1 \
  --bytes 33554432 --rounds 2 --capture-ports
```

`L0L1_SELECTED` 的 L1 必须显示 `l1_port_policy=single bw_coeff=1 remote_bw_coeff=1`，端口计数还必须显示每张卡只有一个 L1 数据端口承担主体载荷；不能只看逻辑通道数或速度降到50GB/s。

如果 `L1_SINGLE_PORT_DISCOVERY candidates=0`，说明**当前运行时没有暴露满足条件的 L1 链路**；discovery 可以正常退出，但 create/execute 会在建链前以 `L1_SINGLE_PORT_UNAVAILABLE` 停止，不回退到双端口。已知现场 rootinfo 的 L1 组为2口和6口，因此确实可能遇到这个结果。请发回两 rank 的 CANDIDATE/CATALOG/DISCOVERY 行；届时需要该版本 SDK/驱动支持的单端口 EID 暴露或绑定方案，不能靠减半字节数、改比例或缩小 rootinfo 的 ports 数组伪造单端口通信。未取得该能力前，不宣称已经实现该机器上的物理端口强制绑定。

## 1. 用例与执行边界

每张卡发送 B 字节给另一张卡，同时接收 B 字节；没有 self 数据、没有第三张卡、没有 relay HBM。

| 模式 | L0 每方向字节 | L1 每方向字节 | 总量 |
|---|---:|---:|---:|
| `l0` | B | 0 | B |
| `l1` | 0 | B | B |
| `split` | floor(B × L0比例 / 256) × 256 | B − L0部分 | B |

默认比例 50%，用 `--l0-percent 25/50/75` 调整。`--bytes` 是**每卡发往对端的总字节数**，不是每条路径大小；双向网络载荷总计 2B。

```text
Host：创建两卡通信域 → GetLayers/GetLinks → 校验两端路径一致
      → 申请 stream/thread、注册输入输出内存、建立 channel
      → 在各 endpoint 所属 die 注册 CCU kernel（仅一次）
      → H2D 输入与 guard → 同时提交所需的路径 → 等全部 stream 完成

L0 stream：CCU 发布/等待对端输出地址及 token → UB_CTP Write → 完成握手
L1 stream：CCU 发布/等待对端输出地址及 token → UB_CTP Write → 完成握手
             两个 stream 不互相 wait；分流时写入互不重叠的输出切片

Device：源 HBM → L0 直连 / L1 网络 → 目的 HBM
Host：计时外 D2H 校验 → 完成后销毁 comm → 释放 buffer/stream
```

L0 必须来自 layer=0、hop=1、UB_CTP；L1 必须来自 layer=1、CLOS、UB_CTP。只使用 `HcclRankGraphGetLinks` 返回的端点，按运行时 `ENDPOINT_ATTR_DIE_ID` 注册，不猜 EID、die 或端口映射。两个端点的 rank0 定向 key 排序，两个 rank 以文件握手确认同一对端点、hop、die；默认各取 index=0，可用 `--l0-index` / `--l1-index` 选择其他候选。

L0/L1 endpoint pair 相同或返回同一个 channel handle，会拒绝继续；**不同 handle 仍不能证明物理资源独立**。路径不可用时报 `PATH_UNAVAILABLE`，不回退 L0，不创建 synthetic EID。

每条路径使用一个独立 stream/thread/kernel，kernel 的 die 来自端点。`split` 先提交两个 kernel，再等待，避免“先等 L0 结束，再提交 L1”。没有 CCU 比较指令、动态 mask 或 EP8 资源依赖。

## 2. 拉取与编译（新机器容器内）

用户 push 后，在有卡机器容器内：

```bash
cd /home/l00934901/sgl-kernel-npu
git fetch origin
git switch --track origin/feature/a5-ccu-two-rank-l0-l1-probe
# 已有本地同名分支时：git switch feature/a5-ccu-two-rank-l0-l1-probe
# 然后 git pull --ff-only

export A5_CANN_PATH=/usr/local/Ascend/cann-9.3.0
export A5_L0L1_BINARY=/home/l00934901/profiling/ccu-l0-l1-build/a5_ccu_l0_l1_probe
bash scripts/build_a5_ccu_l0_l1_probe.sh "$A5_CANN_PATH" \
  /home/l00934901/profiling/ccu-l0-l1-build
```

要求出现 `L0L1_ABI_PASS` 和 `L0L1_BUILD_PASS`。输出 `L0L1_LIBRARY` 必须是选定 CANN 的 `libhcomm.so`。**无需重编/安装 route，无需重打 wheel，也不会覆盖已有 EP8 动态库。**

已在开发服务器现有 `cam_lyw_92`、CANN `/usr/local/Ascend/cann-9.2.0` 内编译和 ABI 加载成功；日志在开发服务器 `/home/liuyuanwen/profiling/ccu-l0-l1-build/{configure,build,abi}.log`。这不是新机器 CANN 9.3.0 的上卡验证，下面步骤仍需执行。

可选 CPU 回归（不占 NPU）：

```bash
python3 tests/cpu/test_a5_l0_l1_summary.py
g++ -std=c++14 -Wall -Wextra -Werror tests/cpu/test_a5_l0_l1_layout.cc \
  -o /home/l00934901/profiling/test_a5_l0_l1_layout
/home/l00934901/profiling/test_a5_l0_l1_layout
```

## 3. 先确认运行时暴露了 L1

只在选定两卡空闲时运行。先用物理 0、1；后续可以换 0、4 验证跨 die 端点组合，但必须重新 discovery，不复用路径 index 的假定。

```bash
export A5_DEVICES=0,1
export HCCL_BUFFSIZE=64
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0
bash scripts/run_a5_ccu_l0_l1_probe.sh --suite discover
```

每次结果自动保存到 `/home/l00934901/profiling/a5-l0-l1-XXXXXX`，终端打印 `OUTPUT_DIR`。含原始两 rank 日志、参数、git 版本、环境、rootinfo/Atlas 快照、退出状态。

检查两端的 `L0L1_LAYER`、`L0L1_CATALOG`：

* layer=0 必须至少一条 eligible=1 的 hop=1 链路。
* layer=1 必须至少一条 eligible=1 的 CLOS UB_CTP 链路。
* `discover` 的 PASS 只表示查询完成、正常退出；**没有 L1 候选也可能完成 discovery**。缺 L1 时保存输出并停止，不用继续测性能。

当前机器 rootinfo 的物理 0..7 对应 Atlas local_id=48..55，不要把两种 ID 混用。L0 PEER2NET 和 L1 PEER2NET 中重复列出的端口不是两份带宽。SDK 没有暴露 L1 时，本用例不修改系统拓扑或尝试凭空构造 EID。

## 4. 建链、注册、正确性

```bash
bash scripts/run_a5_ccu_l0_l1_probe.sh --suite create --mode all
bash scripts/run_a5_ccu_l0_l1_probe.sh --suite correctness --mode all --bytes 65536
bash scripts/run_a5_ccu_l0_l1_probe.sh --suite correctness --mode all --bytes 8388608
```

均需 `L0L1_ALL_RANKS_AND_EXIT_PASS`、两个 worker exit=0。create 不发 payload；correctness 对每个模式执行 3 个不同 epoch，逐个 uint32 校验来源、epoch、位置及前后 guard。它测试字节搬运，不限定模型 dtype。

如果联合建链失败，分开定位：

```bash
bash scripts/run_a5_ccu_l0_l1_probe.sh --suite create --mode l0
bash scripts/run_a5_ccu_l0_l1_probe.sh --suite create --mode l1
```

看最后 `L0L1_PHASE` 与 `L0L1_FAILED`：discover / query_instance / acquire_channels / register_start / register_kernel_Lx / register_end / correctness / teardown。保留两个 rank 的完整日志；不能把 Create PASS 当成数据可用。

超时默认 600 秒，`--timeout` 可调整。launcher 仅终止本次两个 worker 的进程组；不清理或重置其他用户的卡。失败无 PASS，不自动重试或静默回退。

## 5. 证明物理路径（先于性能结论）

分别运行单路径并采集前后只读计数：

```bash
for A5_MODE in l0 l1 split; do
  bash scripts/run_a5_ccu_l0_l1_probe.sh --suite benchmark --mode "$A5_MODE" \
    --bytes 33554432 --warmup 20 --iters 50 --rounds 2 --capture-ports
done
```

`ports_before.log` / `ports_after.log` 保存两卡两个 UDie 的端口 0..8 的 `hccn_tool -g -stat` 与 `-g -dev_info` 原文；不存在的端口/命令错误如实记录，不能当成 DOWN 或零流量。可用 `--hccn-tool PATH` 指定工具。抓取只在 worker 启动前和退出后执行，不在计时区间；包括初始化、正确性和 warmup 的流量，**不用于精确算 payload 带宽**。

按本机 rootinfo 把计数器端口归类：物理0的 L0 端口是 `0/0,0/3,0/4,0/5,0/6,0/7,0/8`，L1 端口是 `0/1,0/2,1/0,1/1,1/2,1/3,1/5,1/6`；另一张卡必须读自己的配置。检查 l0 仅对应直连数据端口显著增长、l1 对应网络端口显著增长、split 两类均增长。控制包/后台流量不应被解释为 payload；有其他网络负载时应重测。

这一步能排除“逻辑 layer=1，实际仍用了相同物理通路”的误读。没有可用端口计数时只能报告 SDK 路径选择与正确性，不能宣布物理分流已证实。

## 6. 同数据量的性能比较

首轮建议从 64KiB 到 64MiB 扫描。每次 `--mode all` 复用同一 comm、两条 channel、两对资源和同一 send/recv buffer，只改变启用路径/切片。模式顺序按 6 种排列轮换；每块单独 warmup、同步；计时外做数据校验和 rank 对齐。

```bash
for A5_BYTES in 65536 131072 1048576 8388608 33554432 67108864; do
  bash scripts/run_a5_ccu_l0_l1_probe.sh --suite benchmark --mode all \
    --bytes "$A5_BYTES" --l0-percent 50 --warmup 20 --iters 50 --rounds 6
done
```

默认 `--timing block`：50 次连续提交后等待所有使用的 stream 完成，wall/50；每轮取两 rank 最大，再跨轮取中位。`enqueue_or_serial_us` 在 block 模式仅含主机提交；它不是设备执行时间。JSON `effective_GBps_per_direction` 是 B/t，双向合计为 2B/t。

同样大小扫分配比例，不能假定 8 个 L1 端口就应分配 8/9：

```bash
for A5_SHARE in 25 50 75; do
  bash scripts/run_a5_ccu_l0_l1_probe.sh --suite benchmark --mode all \
    --bytes 8388608 --l0-percent "$A5_SHARE" --warmup 20 --iters 50 --rounds 6
done
```

对找到的候选比例，重新启动独立进程至少 3 次，再补逐次完成口径：

```bash
for A5_REP in 1 2 3; do
  bash scripts/run_a5_ccu_l0_l1_probe.sh --suite benchmark --mode all \
    --bytes 8388608 --l0-percent 50 --warmup 20 --iters 50 --rounds 6
done
bash scripts/run_a5_ccu_l0_l1_probe.sh --suite benchmark --mode all \
  --bytes 8388608 --l0-percent 50 --timing serial --warmup 20 --iters 50 --rounds 6
```

用实测最优比例替换示例50。serial 每次两个路径都提交后同步全部 stream；该口径的 `enqueue_or_serial_us` 包含等待。不要和 block 的主机 enqueue 混用。50 次连续提交组成一个样本，不是50个独立样本。

汇总会输出 `l0_over_split`、`l1_over_split`。大于1表示 split 更快。**split 需同时快于 L0 和 L1，才能证明相对最佳单路径有收益**。收益须超过运行间漂移，并有物理计数支持。独立单路径进程资源更少，不要把那组数字直接混入同 handle 的三模式比较。

## 7. 回报给作者

发回 commit、CANN、物理卡号、HCCL_BUFFSIZE、`L0L1_LIBRARY`、两 rank 的 `L0L1_CATALOG` / `L0L1_SELECTED`、每次结果目录及 `summary.json`、端口计数差分。失败发最后 phase 和完整 `L0L1_FAILED`，保留 plog 的同 PID 首个异常。

已有目录可只读重新汇总，不占卡：

```bash
python3 scripts/a5_ccu_l0_l1_probe.py --summarize /home/l00934901/profiling/a5-l0-l1-实际目录
```

当前仅完成开发容器编译/ABI和CPU回归；不承诺本机 L1 一定可建链，也不把两卡吞吐提升外推为 EP8/MoE 加速。待路径与收益验证完成后，再设计 EP8 改造。

## 8. 显式单口 EID 实验：不在 rank graph 中，还能否建 CCU 通道？

若现场发现 UB 地址 type=IPv6、远端属性查询返回6，请先阅读并执行 §9 的兼容回归。

### 8.1 本轮回答的问题与边界

新增 `--explicit-eids <JSON>`，让两端使用指定的完整 EID。新增 `--suite endpoint`，只测试
`HcommEndpointCreate/Destroy`，不申请通道、不注册 kernel、不发送 payload。

后续 `create/correctness/benchmark` 仍使用 **`HcclChannelAcquire + COMM_ENGINE_CCU`**：
从选中的 L1 graph 路径复制 EndpointDesc 的物理位置和协议，仅覆盖 EID；本地/对端 die
由经过核实的 JSON 指定，必须与所选模板相同。对新 EID 的 graph 查询失败会记录，
但不会在应用侧提前拒绝建链。若查询成功而 die 不一致，立即拒绝。

这是分层实验，不是宣称已实现完全无通信域的 `HcommChannelCreate` 后端。
独立 endpoint 创建成功不等于通道可用，更不等于 CCU 能传输。如果 managed acquire
仍因 graph 查询失败而失败，下一步才核对现场 SDK 的 domainless socket/内存交换/CCU
资源契约；不能拿普通 host URMA 或 UB_MEM 传输替代 CCU 对照，也不能据此宣布硬件不支持。

不会修改 rootinfo、Atlas、驱动 EID、全局端口状态，不会自动 fallback 聚合 EID。
原 EP8、route 包、wheel 均不改。所有结果在 profiling，每个命令使用独立双 worker。

### 8.2 编译与发现（先做）

```bash
cd /home/l00934901/sgl-kernel-npu
git switch feature/a5-ccu-two-rank-l0-l1-probe
git pull --ff-only
export A5_CANN_PATH=/usr/local/Ascend/cann-9.3.0
export A5_DEVICES=0,1
export HCCL_BUFFSIZE=64
export A5_EXPLICIT_ROOT=/home/l00934901/profiling/l0-l1-explicit
mkdir -p "$A5_EXPLICIT_ROOT"
bash scripts/build_a5_ccu_l0_l1_probe.sh "$A5_CANN_PATH" \
  /home/l00934901/profiling/ccu-l0-l1-build
python3 tests/cpu/test_a5_l0_l1_summary.py
bash scripts/run_a5_ccu_l0_l1_probe.sh --suite discover --mode l1 \
  --l1-port-policy any
```

构建应出现 `L0L1_EXPLICIT_EID_ABI version=1`、`L0L1_BUILD_PASS`。无需重新打 wheel/route。
`discover` 新增可选的 SDK 枚举 `HcommEndpointGetDescNum/GetDescs`，通过 dlsym 查询，
旧 SDK 没有符号或返回失败会明确记录，不影响原探针。

保留两 rank 的 `L0L1_DRIVER_ENDPOINT`（包含 EID/protocol/物理位置）、
`L0L1_CATALOG`、`L0L1_ENDPOINT_INVENTORY`。它们帮助把候选 EID 映射到选中的 NPU。
不要把 URMA 设备名的 c/d/e 编号直接当作物理 NPU ID；不要仅凭 0401/0441 等地址片段
猜完整 EID；不要把 FE=2 和 FE=6 当成相同端点。驱动清单也不直接证明 L1 归属。

### 8.3 先建聚合 EID 手填对照，再准备单口候选

从上一步输出的目录选一个 L1 catalog `index`（这是 `--l1-port-policy any` 的 index）。
下面只读提取两个 rank 实际发现的 EID/die，不编造地址。

```bash
export A5_DISCOVERY_DIR=/home/l00934901/profiling/a5-l0-l1-替换为刚才目录
export A5_TEMPLATE_INDEX=0  # 依实际 catalog 选择，后续命令保持一致
python3 - <<'PY'
import json, os, re
from pathlib import Path
root = Path(os.environ['A5_DISCOVERY_DIR'])
idx = int(os.environ['A5_TEMPLATE_INDEX'])
env = json.loads((root/'environment.json').read_text())
devices = env['ASCEND_RT_VISIBLE_DEVICES'].split(',')
records = []
for rank in range(2):
    lines = (root/f'rank{rank}.log').read_text(errors='replace').splitlines()
    hits = [s for s in lines if re.search(rf'L0L1_CATALOG rank={rank} layer=1 index={idx}\b', s)]
    if len(hits) != 1: raise SystemExit(f'rank{rank}: expected exactly one catalog entry')
    die = int(re.search(r'\bdie=(\d+)', hits[0]).group(1))
    key = re.search(r'\bkey=(\S+)', hits[0]).group(1)
    eids = re.findall(r'\d+:\d+:([0-9a-f]{32})', key)
    if len(eids) != 2: raise SystemExit('invalid canonical EID pair')
    records.append((key, dict(rank=rank, device=int(devices[rank]), die=die, eid=eids[rank])))
if records[0][0] != records[1][0]: raise SystemExit('rank catalog mismatch')
out = Path(os.environ['A5_EXPLICIT_ROOT'])/'aggregate.json'
out.write_text(json.dumps(dict(mapping_note=f'graph catalog {root}, index={idx}',
                                endpoints=[r[1] for r in records]), indent=2))
print(out.read_text())
PY
```

依次测试 graph 选择、同一个 EID 手填 endpoint、同一个 EID 手填 managed create：

```bash
bash scripts/run_a5_ccu_l0_l1_probe.sh --suite create --mode l1 \
  --l1-port-policy any --l1-index "$A5_TEMPLATE_INDEX" --bytes 65536
for A5_SUITE in endpoint create; do
  bash scripts/run_a5_ccu_l0_l1_probe.sh --suite "$A5_SUITE" --mode l1 \
    --l1-port-policy any --l1-index "$A5_TEMPLATE_INDEX" --bytes 65536 \
    --explicit-eids "$A5_EXPLICIT_ROOT/aggregate.json" || break
done
```

三项全通过再测候选。若聚合 endpoint 单独创建不支持而 managed create 支持，记录这一差别，
它说明独立端点入口存在限制；不要把它归因到单口候选。

复制 aggregate.json 为 `single.json`，只将 `eid` 换成两张选定物理 NPU 上**实际枚举到、
且物理映射已确认**的单口 EID；`die` 必须真实对应该端口，并选择同 die 的 L1 模板。
`mapping_note` 写明证据文件、URMA 设备名、源/目标端口及其 L1 归属。保留完整128位地址。

JSON 示例结构（下列说明文字不是可执行 EID，禁止照填）：

```json
{
  "mapping_note": "物理NPU与URMA设备的对应证据；两个端口的L1归属证据",
  "endpoints": [
    {"rank": 0, "device": 0, "die": 0, "eid": "实际源卡完整32位十六进制EID"},
    {"rank": 1, "device": 1, "die": 0, "eid": "实际目的卡完整32位十六进制EID"}
  ]
}
```

`device` 必须对应 `A5_DEVICES` 的顺序；die 示例不是固定要求。程序校验 rank/device/die、
完整地址、两端不同和非空 mapping_note，但 **mapping_note 是操作者提供的证据说明，
程序不能据此证明线路归属**。本轮不自动合成任何新 EID。

### 8.4 单口候选：端点 → 建链/注册 → 正确性

逐条执行，检查上一条结果后再继续。`endpoint` 与 `create` 是独立实验，前者失败不自动
证明后者失败；失败保留目录，不让脚本自动批量重试。

```bash
bash scripts/run_a5_ccu_l0_l1_probe.sh --suite endpoint --mode l1 \
  --l1-port-policy any --l1-index "$A5_TEMPLATE_INDEX" --bytes 65536 \
  --explicit-eids "$A5_EXPLICIT_ROOT/single.json"

bash scripts/run_a5_ccu_l0_l1_probe.sh --suite create --mode l1 \
  --l1-port-policy any --l1-index "$A5_TEMPLATE_INDEX" --bytes 65536 \
  --explicit-eids "$A5_EXPLICIT_ROOT/single.json"
```

`create` 两 rank clean PASS 后才执行：

```bash
bash scripts/run_a5_ccu_l0_l1_probe.sh --suite correctness --mode l1 \
  --l1-port-policy any --l1-index "$A5_TEMPLATE_INDEX" --bytes 65536 \
  --explicit-eids "$A5_EXPLICIT_ROOT/single.json"
```

使用 `any` 是因为候选端点可能不在 graph，无法用 BW_COEFF=1 筛选；
**不是允许 fallback，也不是声明它已物理单口化**。日志必须显示 `endpoint_source=explicit`、
`L0L1_EXPLICIT_PAIR actual=...` 与 single.json 完全一致。
独立 endpoint suite 不会被汇总器误报成数据验证通过。

### 8.5 只有正确性通过后，验证物理端口及 L0+单口 L1 性能

先独立占用测试窗口，确认没有其他作业在这两个 NPU/相关共享链路上产生流量。

```bash
bash scripts/run_a5_ccu_l0_l1_probe.sh --suite benchmark --mode l1 \
  --l1-port-policy any --l1-index "$A5_TEMPLATE_INDEX" --bytes 33554432 \
  --explicit-eids "$A5_EXPLICIT_ROOT/single.json" \
  --warmup 20 --iters 50 --rounds 2 --capture-ports
```

对照 §5 的端口计数，确认 payload 主要落在**两端各一个已核对的 L1 物理口**，而非依旧
p1/p2 各半。端点名称、EID 数量、`port_count=1`、带宽约50GB/s均不能替代物理计数证据。

物理验证通过后再跑三种模式：

```bash
for A5_REP in 1 2 3; do
  bash scripts/run_a5_ccu_l0_l1_probe.sh --suite benchmark --mode all \
    --l1-port-policy any --l1-index "$A5_TEMPLATE_INDEX" --bytes 33554432 \
    --explicit-eids "$A5_EXPLICIT_ROOT/single.json" \
    --l0-percent 50 --warmup 20 --iters 50 --rounds 6 || break
done
```

L0 保持原 graph 选路，只有 L1 使用显式 EID。每方向总量32MiB，split 各16MiB，
无中间 relay 卡、无 relay HBM；两路径仍是独立 stream、先全部提交再等完成。
不保证50/50最优，更不能只凭两类端口都有流量推断时间上真正并行。

### 8.6 如何判读，回报什么

| 结果 | 能支持的结论 | 不能推出的结论 |
|---|---|---|
| 聚合手填 create 失败，graph create 成功 | 先查手填参数、模板、版本和实验入口 | 不能判单口不可用 |
| 候选 endpoint_create 成功，graph 查询失败 | 独立端点入口认识该地址，当前 graph 不认识 | 不能判 CCU 通道/传输可用 |
| 候选 channel_acquire 失败 | 当前 managed CCU 入口未建立该通道；看首个底层错误 | 不能只凭状态码判定一定依赖 graph，或硬件不支持 |
| acquire 成功，注册失败 | 问题推进到 CCU 资源/注册环节 | 不是端点枚举问题 |
| create 成功，执行或数据校验失败 | 注册成功不等于可用，继续查同步/数据路径 | 不能跑性能后宣称通过 |
| graph 查询失败，但正确性和单口计数均通过 | 此版本可用显式 EID 在当前 managed 路径建立单口 CCU 通道 | 不能外推所有 FE/端口/SDK 版本 |
| 正确性通过，多个物理口有 payload | 显式地址没有达到预期的物理单口效果 | 不能称为单口 L1 性能 |

请发回每个实验的目录、命令、`explicit_eids.json`、`environment.json`、`git.txt`、
两 rank 的 `L0L1_ENDPOINT_INVENTORY` / `L0L1_DRIVER_ENDPOINT` / `L0L1_EXPLICIT_QUERY` /
`L0L1_EXPLICIT_PAIR` / `L0L1_STAGE` / `L0L1_SELECTED`，以及 `summary.json`。
失败提供 `failure_summary.json`、`worker_exit.json` 和同 PID plog 首个错误；
不要只给最后的 status=4/7。阶段有 begin、status、elapsed_us，可区分未进入、返回失败和超时。
候选不在 graph 时产生的查询错误是预期诊断，需与后续 acquire/register 的错误区分。

本轮编译验证在 cam_lyw_92；现场9.3建链和物理端口验证尚待以上实验。

## 9. CANN 9.3 现场兼容：UB 地址类型与远端属性查询

### 9.1 问题与修复语义

用户在物理卡2/3的现场验证发现：GetLinks 返回 UB_CTP 的正确16字节地址，但 type 为
`COMM_ADDR_TYPE_IP_V6`；GetDescs 对相同地址返回 `COMM_ADDR_TYPE_EID`。此外，
本 rank 的 DIE_ID/BW_COEFF 查询成功，对端 rank 的查询返回 NOT_FOUND(6)。
2a09314 同时要求 EID type 和两端直接查询成功，因而将这些真实链路全部筛掉。
这份证据指向探针的兼容条件，不能用换容器/更换拓扑文件代替修复，也不意味着所有9.3安装都有同样行为。

本修复：

- 地址 helper 只接受 UB_CTP +（EID 或 IPv6 标记）+ 非零16字节，不接受任意 type/协议。
- SDK 入参中的 EndpointDesc、地址字节和 type 保留；仅用于去重/匹配的身份 key 将上述
  两种 type 归一，避免同一个地址由于 API 标签差异不能配对。日志保留 src/dst 原始 type。
- discover 暂存合格地址，不因为远端查询失败提前删除。每端把**自己本地**查询的
  EID→die/status/BW/status 写入独立运行目录，通过原有文件握手交换。
- 按完整地址与协议匹配对端报告；不从 EID 位段猜 die，不从端口个数猜 BW。若成功的远端
  graph 查询与对端成功的本地查询不一致，报 `ENDPOINT_GRAPH_PEER_CONFLICT`；
  缺少对应地址报 `PEER_ENDPOINT_MISSING`。不把未核实的默认0当成 die0或单口。
- graph 路径必须有两端各自成功查询到的有效 die 才进入最终 catalog。`L0L1_CANDIDATE`
  是查询原始结果，带 `resolution=pending_peer`；最终看 `L0L1_RESOLVED eligible=1`
  和 `L0L1_CATALOG`，不要以原始 `remote_die_status=6` 判失败。
- 显式 EID 覆盖后，再用单独一轮文件握手交换最终端点属性。graph 不认识显式端点时，
  die 沿用双方已经确认一致的显式配置并标为 `declared_explicit`，BW 标为 unverified。
  graph 若返回成功但与声明 die 不同，依旧立即拒绝。
- `single` 仍要求两端可验证的 BW_COEFF 都为1；未知/零值输出 `BW_COEFF_UNVERIFIABLE`。
  `any + --explicit-eids` 可继续探测，但必须用端口计数确认物理单口。BW=1也不是物理计数的替代品。

发现属性文件 `discovery_endpoint_attrs.{0,1}`、显式覆盖后的
`selected_endpoint_attrs.{0,1}` 都保留在结果目录。文件握手只在初始化执行，不在计时区间内；
CCU kernel、数据分配比例、提交顺序不变。无需改 EP8、wheel、route 或系统 rootinfo。

现场此前成功的数据来自用户的本地修补副本：单口 L1约47 GB/s、L0+单口L1约88–96 GB/s，
并有端口计数。它们证明修复方向可行，**不是本提交在现场跑过的结果**。本提交仍需以下短回归。

### 9.2 DeepSeek 执行步骤

先保留现场补丁与原始结果。拉取本分支并按 §8.2 重编独立二进制；不要让旧的
`A5_L0L1_BINARY` 指向现场副本或旧程序。以下命令在已配好 CANN 环境的同一 shell 执行：

```bash
cd /home/l00934901/sgl-kernel-npu
git switch feature/a5-ccu-two-rank-l0-l1-probe
git pull --ff-only
export A5_CANN_PATH=/usr/local/Ascend/cann-9.3.0
export A5_DEVICES=2,3
export HCCL_BUFFSIZE=64
export A5_L0L1_BINARY=/home/l00934901/profiling/ccu-l0-l1-build/a5_ccu_l0_l1_probe
bash scripts/build_a5_ccu_l0_l1_probe.sh "$A5_CANN_PATH" \
  /home/l00934901/profiling/ccu-l0-l1-build
python3 tests/cpu/test_a5_l0_l1_summary.py
g++ -std=c++14 -Wall -Wextra -Werror tests/cpu/test_a5_l0_l1_layout.cc \
  -o /home/l00934901/profiling/ccu-l0-l1-build/layout_cpu_test
/home/l00934901/profiling/ccu-l0-l1-build/layout_cpu_test
bash scripts/run_a5_ccu_l0_l1_probe.sh --suite discover --mode l1 --l1-port-policy any
```

检查构建标记 `L0L1_EXPLICIT_EID_ABI version=1 graph_compat=1`。新 discover 应显示对端
直接查询失败仍可通过 `L0L1_PEER_ATTR ... peer_local_die_status=0` 解析，最终出现可用 catalog。
如果对端本地也失败，保留证据，不应把失败直接忽略后建链。

依据新 catalog 重新选择 `A5_TEMPLATE_INDEX`，按 §8.3 导出聚合对照。然后复用现场已经
验证的 single.json，但先确认 `device=2/3`、完整 EID、die、mapping_note 均正确，
不能拿旧物理0/1的配置直接跑2/3。下列两个变量要填实际值：

```bash
export A5_TEMPLATE_INDEX=0  # 替换成新 catalog 中与候选 EID 同 die 的 L1 index
export A5_SINGLE_PAIR=/home/l00934901/profiling/l0-l1-explicit/single.json
bash scripts/run_a5_ccu_l0_l1_probe.sh --suite create --mode all \
  --l1-port-policy any --l1-index "$A5_TEMPLATE_INDEX" \
  --explicit-eids "$A5_SINGLE_PAIR" --bytes 65536
```

create 两 rank clean PASS 后执行正确性：

```bash
for A5_BYTES in 65536 8388608; do
  bash scripts/run_a5_ccu_l0_l1_probe.sh --suite correctness --mode all \
    --l1-port-policy any --l1-index "$A5_TEMPLATE_INDEX" \
    --explicit-eids "$A5_SINGLE_PAIR" --bytes "$A5_BYTES" || break
done
```

每个大小应两 rank 各9次 `L0L1_CORRECT`，包含 L0/L1/split ×3 epoch，且 clean PASS。
正确性全过后，只需先做一次32MiB短复测，不必重复所有扫描：

```bash
bash scripts/run_a5_ccu_l0_l1_probe.sh --suite benchmark --mode all \
  --l1-port-policy any --l1-index "$A5_TEMPLATE_INDEX" \
  --explicit-eids "$A5_SINGLE_PAIR" --bytes 33554432 \
  --warmup 20 --iters 50 --rounds 6
```

若要重新证明单口，按 §8.5 另跑 `--mode l1 --capture-ports`；`mode all` 的总窗口包含三种
模式，不能把全窗口端口增量当成单独 L1 或单独 split 的流量。
报告区分 `median_us` 与 `effective_GBps_per_direction`：约47→90 GB/s 是带宽提升，
对应32MiB耗时约714→373 µs，不是耗时翻倍。

### 9.3 回报材料

发 commit、二进制构建标记、结果目录和 `summary.json`，以及两 rank 的
`L0L1_CANDIDATE`、`L0L1_PEER_ATTR`、`L0L1_RESOLVED`、`L0L1_CATALOG`、
`L0L1_EXPLICIT_QUERY`、`BW_COEFF_UNVERIFIABLE`（若有）。
失败追加两轮 endpoint_attrs 文件与 `failure_summary.json`。
注意原始远端查询 status=6 可以继续出现；判据是对端上报值、最终 catalog、实际建链和正确性，
不是让所有诊断查询都强行显示 status=0。

## 10. 计算图捕获与重放（显式 L1 EID）

见 [CCU + URMA 计算图验证指南](A5_CCU_ACL_GRAPH_VALIDATION.md)。DeepSeek 先读该文 §1–3，
按说明新增独立 graph 诊断入口，再执行 L0、显式单口 L1、双流 split 捕获/重放。
现有 eager correctness/benchmark 不能证明计算图支持；现有 launcher 尚无 `--suite graph`。
两卡通过后才继续该文 §4 的 EP8 READY/DONE 桥和 §5 的动态参数验证。

现场已完成 G0/G1、G2 失败后的下一步见该文 **§7–11**：external 事件不能证明跨流支持；
先 P0 核对事件创建 API 和两流 modelRI，再 P1 纯本地双流、P2 同 die split，必要时 P3 双图隔离。
