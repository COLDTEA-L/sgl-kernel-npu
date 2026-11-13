# A5：物理链路冲突下的 host 串行调度

分支：`feature/a5-ccu-valuewait-callback-puncture-92-93`。

本次修正替代 `57ddd97` 的“共享 relay 卡就加锁”实验。旧例中 2→0→3 和 4→0→5 没有共同物理链路，不能用来验证物理链路冲突。本版使用两个并发的两 rank 通信 job，显式构造共同的 UB_CTP 链路，对照 uncontrolled 与 serialize。

## 设计与范围

默认卡位建议如下，最终必须通过本机拓扑解析，不预设具体 die/port：

```text
job A：通信卡 2、3，relay 0、1
       2 ↔ 0 ↔ 3
       2 ↔ 1 ↔ 3

job B：通信卡 0、1，relay 2、3
       0 ↔ 2 ↔ 1
       0 ↔ 3 ↔ 1

目标共同链路：2↔0、3↔0、2↔1、3↔1 中至少一条
必须是相同的卡/die/port 两端，不只是相同的两张卡。
```

通信是双向的，因此使用同一物理链路的两个方向都会有数据流量；不能把单向 A→B 与 B→A 自动视为同向带宽冲突。本版两 rank 双向交换会同时覆盖两方向。

每张卡只有一个 endpoint worker；它可以同时充当另一个 job 的 relay。这是有意构造共同物理 hop，需要在目标设备验证同时建链与执行是否可用。
单机管理域仍为八卡，最小冲突实验实际使用其中四张，卡号可显式修改。
两 job 各一张图，每张图有两个按序执行的通信节点；并发来自跨 job。

为了明确物理路径，本版 benchmark 只创建显式 relay catalog，不创建自动发现的 direct channel。
默认 mixed 是两条 relay 各一半 payload，仍是多路径通信；支持固定 relay0/relay1，暂不在物理 benchmark 中使用 direct/cycle。
旧的独立 gate 实验不设置 `A5_GATE_RELAY_ONLY` 时保持 direct+relay 行为。

## 物理资源校验和控制流程

`prepare_a5_physical_conflict.py`：

1. 查询本机 rootinfo、Atlas 和实时 hccn EID/端口状态。
2. 枚举候选 relay 路径，要求同一 relay die、每个 endpoint 的 catalog 使用同一 IO die。
3. 只接受 EID 唯一绑定到预期物理端口的路径；EID 对应多个端口时拒绝将它视为精确端口控制。
4. 为两个 job 选择至少一条共同 payload 链路；无解就失败，不通过手填资源名制造冲突。
5. 输出 manifest、config、拓扑快照、provenance 和共同链路列表。

链路资源名格式：

```text
link:<物理卡>/<die>/<port><-><物理卡>/<die>/<port>
例如 link:0/0/7<->2/0/0（仅格式示例，不是实际端口断言）
```

两端排序归一化；一项代表双向物理链路，因为每个 job 本身双向收发。
launcher 再校验每条 manifest hop 存在于 Atlas、EID 绑定及端口状态，且 provenance 的拓扑 hash 一致。
共享 relay 卡、不同物理端口的两条路径不再产生同一个锁。
共享模式要求非零 payload 路径真正有公共链路，仅 metadata 重叠不算 payload 冲突。
所有 catalog hop 仍参与预约，因为零 payload 的 channel 仍有 metadata；payload 链路列表单独输出。

```text
图内 ARRIVED → ValueWait(gate) → 读 policy → 全部路径搬运/等待/完成握手 → DONE

host 普通线程：ARRIVED → Unix socket → 收齐参与 rank
scheduler：检查物理链路锁 → 原子预约 → GO(ticket, split)
host control stream：写 policy → READY
scheduler：收齐同一 ticket 的全部 rank DONE → 释放物理链路 → 下一组 GO
```

callback 只更新原子通知；socket 和 Runtime 提交由普通 host 线程执行。
WAITING 不部分占锁。共享资源 FIFO，不相交任务可并行。超时、旧 ticket、重复 DONE、worker 失败均使实验失败，不以预计结束时间释放资源。
policy/audit 地址仍逐路径由 host 传入，任务 ABI 仍为 `4 + 2*paths`。
relay-only 时 paths 为 relay 数量；完成握手使用 catalog 的第一个 relay channel。

拓扑与 EID 校验提供“所配置路径具有共同物理 hop”的证据。实际转发是否按这些端口执行、并发是否导致带宽下降，还需要 HCCN 和时间测量，不能仅凭准备脚本或 CPU PASS 下结论。

## 1. 构建与 CPU 检查

在仓库根目录执行，路径按实际 CANN 版本修改：

```bash
unset LD_PRELOAD
source /usr/local/Ascend/cann-9.3.0/set_env.sh
bash scripts/build_a5_ccu_urma_route_probe.sh \
  --cann-path /usr/local/Ascend/cann-9.3.0 --build-dir /tmp/a5-scheduler-build
/tmp/a5-scheduler-build/a5_graph_gate --check-abi
python3 -m unittest discover -s tests/python/deepep -p 'test_a5_path_scheduler_cpu.py' -v
A5_SCHEDULER_CPU_CLIENT=/tmp/a5-scheduler-build/a5_gate_scheduler_client_cpu \
  python3 -m unittest discover -s tests/python/deepep -p 'test_a5_path_scheduler_cpu.py' -v
python3 -m unittest discover -s tests/python/deepep -p 'test_a5_relay_graph_gate_cpu.py' -v
```

CPU fixture 会模拟成功标记，仅用于协议测试；所有上卡命令必须使用 `a5_graph_gate`。
新 CPU 测试覆盖仅共享 relay 卡拒绝、metadata-only 重叠拒绝、错误端口/EID 绑定拒绝、拓扑 hash 校验、真实 socket、调度资源生命周期。
开发用 9.2 容器无 NPU，编译与 CPU 通过不能替代这次新物理配置的上卡验证。
本次已通过 9.2 SDK 的 -Werror 编译、严格链接、ABI 加载、原 host ctest 1/1、原 launcher 4/4，以及新物理调度测试 21/21（含实际 C++ socket client）。新路径角色交叉和物理流量尚待上卡验证。

## 2. 生成有共同物理链路的计划

使用一个新的输出目录。卡号和文件路径按现场修改：

```bash
python3 scripts/prepare_a5_physical_conflict.py \
  --root-info /etc/hccl_rootinfo.json \
  --topology-json /usr/local/Ascend/driver/topo/950/atlas_950_1.json \
  --managed-devices 0,1,2,3,4,5,6,7 \
  --scenario shared \
  --a-devices 2,3 --a-relays 0,1 \
  --b-devices 0,1 --b-relays 2,3 \
  --output-dir /tmp/a5-physical-shared
```

检查 `selection.json` 的 `shared_payload_links` 非空，列出共同链路两端卡/die/port。
脚本输出 `CONFIG=/tmp/a5-physical-shared/config.json`。
计划只选择满足目标的物理路径，不自动猜“同一 relay 必然冲突”。没有可用组合时调整实际卡位或 relay 数，不能跳过校验。
本步是只读硬件查询，不提交 NPU 任务；需重新准备本机计划，避免使用其他机器或过期 EID。

## 3. 正确性：共同链路被占用时等待

```bash
python3 scripts/run_a5_serial_scheduler.py \
  --binary /tmp/a5-scheduler-build/a5_graph_gate \
  --config /tmp/a5-physical-shared/config.json \
  --schedule serialize --mode GRAPH --notify CALLBACK \
  --bytes 4194304 --iters 4 --warmup-iters 0 \
  --cohort-barrier --grant-delay-ms 200 --require-blocked \
  --timeout 300 --output-root /tmp/a5-physical-correctness
```

`cohort-barrier` 对齐两 job 同一节点的到达；200ms 是预约后、READY 前的调试窗口，仅用于观察排队，不能计为通信性能。
两 job 共四个 worker，每个 worker 应有 8 个 GATE_OP_PASS，audit/payload 正确，零退出，launcher 输出 SERIAL_SCHEDULER_ALL_WORKERS_PASS。
events.jsonl 应出现 BLOCKED；后一个共享链路任务的 GRANT 必须在前者两个 rank 的 DONE 和 RELEASE 之后。
逐节点 gate 不得提前执行。依次验证 EAGER/GRAPH × CALLBACK/D2H 四组合，不自动 fallback。

## 4. 无共同链路的对照

```bash
python3 scripts/prepare_a5_physical_conflict.py \
  --root-info /etc/hccl_rootinfo.json \
  --topology-json /usr/local/Ascend/driver/topo/950/atlas_950_1.json \
  --managed-devices 0,1,2,3,4,5,6,7 --scenario disjoint \
  --a-devices 2,3 --a-relays 0,1 \
  --b-devices 4,5 --b-relays 6,7 \
  --output-dir /tmp/a5-physical-disjoint
```

将第3步 config 换为此 config，移除 require-blocked。预期在首个 RELEASE 前已有两项 GRANT，没有物理链路锁冲突，数据校验通过。
也可用原来的共享 relay 卡配置 A2/3 via0/1、B4/5 via0/1，指定 scenario=disjoint：只有拓扑证明其各 hop 不相交时才接受。这验证“同卡不同链路”不会被卡级锁误串行。
物理链路不相交仍可能有 die/HBM 等其他争用，本实验只验证链路冲突。

## 5. 性能：并发使用同一链路 vs scheduler 串行

使用完全相同的 shared config，先将 grant-delay 设为0：

```bash
for schedule in uncontrolled serialize; do
  python3 scripts/run_a5_serial_scheduler.py \
    --binary /tmp/a5-scheduler-build/a5_graph_gate \
    --config /tmp/a5-physical-shared/config.json \
    --schedule "$schedule" --mode GRAPH --notify CALLBACK \
    --bytes 67108864 --iters 50 --warmup-iters 5 \
    --cohort-barrier --grant-delay-ms 0 --poll-us 50 \
    --timeout 600 --output-root /tmp/a5-physical-perf
done
```

再测 4/16/256MiB，交替顺序各重复至少三次。uncontrolled 保留相同通知、IPC 和 policy 写入，只允许物理链路重叠预约。
单独测 A、B 的通信作为无竞争耗时参考：在同一个 shared config 和上述性能命令中分别加 `--only-job a`、`--only-job b`，保留相同 bytes、iters、policy 和通知设置，移除 require-blocked。
完整物理配置仍会先验证，再只启动选中的一个 job；catalog 和 payload split 保持一致。不要拿旧 direct+relay 模式作单算子基线。

HCCN 验证必须覆盖 `physical_plan.json` 的共同链路两端：

1. 单独 A、单独 B 都应使同一对物理端口的 payload 计数显著增加，证明共同 hop 真被使用。
2. uncontrolled 必须确认有实际搬运重叠，不能只看 GO/预约区间重叠；使用适当消息大小、重复轮数和端口采样。
3. serialize 应显示分时使用，并结合全部 DONE 后放行的时间线。
4. 全程只有共同 relay 卡而端口不一致，不满足实验要求；没有实际重叠时不能声称测到冲突膨胀。

结果保存 `physical_plan.json`（path→物理 hop、共同 payload 链路）、拓扑/provenance 快照、各 job manifest、events.jsonl、metrics.json、summary.json 和 worker 日志。
时间线使用同一 host monotonic_ns；release→done 包含 Runtime 和 polling，不是纯 device 时间；iteration span 包含节点间验证与 IPC，不是模型迭代时间。
给出 resource_wait、reservation、ready→done、iteration span 的均值/分位数及重复实验波动；性能变差也如实记录。

## 失败归因

依次区分拓扑/EID准备、HCCL初始化、建链、CCU注册、图捕获/replay、调度IPC、gate放行、通信完成、audit/payload。
一个卡兼任另一个 job 的 relay 可能暴露额外的建链/执行限制，失败时保留具体 API、返回码和 plog，不换成无共同链路配置后宣称完成物理冲突实验。
9.2 既有 jetty 创建失败与 policy/audit 地址派生注册失败是不同问题；保持逐路径地址 ABI。
最终报告必须分别说明：共同物理 hop 的拓扑证据、实际 HCCN 证据、串行控制正确性、是否测到时间膨胀和串行收益。
