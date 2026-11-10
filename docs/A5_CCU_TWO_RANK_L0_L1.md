# 两卡 CCU+URMA：L0、L1、L0+L1 分流验证

本用例独立于 EP8。分支 `feature/a5-ccu-two-rank-l0-l1-probe` 基于 `9a378d4`，只新增探针、运行脚本、CPU 测试与本文档；不修改 EP8 算子、planner、route 包或模型接入。

目标是验证当前机器的两张 NPU 能否通过 **L0 fullmesh**、**L1 CLOS 网络**分别通信，以及同时使用两者是否增加吞吐。拓扑有 L1 端口、端口 UP、SDK 返回 L1 链路、通道建链、数据正确、物理流量走 L1、性能增加，是不同层次的证据。

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
