# A5 单机 8 卡：路径占用即等待的 host scheduler v1

分支：`feature/a5-ccu-valuewait-callback-puncture-92-93`。

本版只实现一个决策：**所需资源全部空闲才放行，否则保持 gate 关闭，等待占用者的全部 rank DONE**。
没有耗时膨胀预测、自动改走 direct、抢占、运行中换路或资源超时自动释放。
资源不相交的通信允许并行，不是整机全局串行。

## 范围与验证状态

- 一个 Python host scheduler 管理显式配置的单机八卡资源域；各 rank 是独立 C++ worker。
- 复用已有通过上卡测试的 `a5_graph_gate` 两 rank 执行入口。每个 job 是独立 communicator、独立图，图中有两个有数据依赖的通信节点。
- 多个 job 可以并发到达，因此能制造真实的跨图资源冲突。原先同一 stream 内串行的两个节点本身不能制造并发冲突。
- shared 示例有三组通信：2/3、4/5、6/7，共享 relay 0/1，实际涉及八卡。所有卡号均由 JSON 显式配置，manifest 必须与之匹配。
- **这是八卡资源域中多个两 rank 通信的调度，不是一个八 rank AllToAllV 的 device 实现。** scheduler 核心支持任意参与 rank 数，CPU 测试覆盖八 rank 到齐及完成聚合；八 rank AllToAllV/DeepEP/NPUGraph adapter 尚未接入。
- CANN 9.2 容器已完成新 C++ worker `-Wall -Wextra -Werror` 编译、严格链接、ABI 加载；原 host ctest 1/1、原 launcher CPU 测试 4/4、新 scheduler CPU 测试 14/14 通过。新 CPU 测试含真实 Unix socket 和多进程假 worker，不是 NPU 性能证据。
- 当前容器无 NPU 映射，**本版调度器尚未上卡，不声明性能提升**。已有 9.3 gate/relay 通过报告不能代替新多 job 调度实验。

## 资源和调度语义

`JobSpec.resources` 默认是该 job 两张 endpoint 和所有 relay 的 `card:<physical_id>` 集合。
同一卡在 endpoint/relay 两种角色下使用同一个资源名。资源粒度故意保守：共享 relay 卡就串行，即使可能走不同端口。
不是从卡号自动推断真实端口拓扑，也不是卡与卡邻接矩阵。

当前 `PolicyKernel` 会在所有 catalog channels 上交换 metadata，零字节路径只跳过 payload。
因此 **direct policy 仍预约整个已建链 catalog，包括 relay 卡**，不能将 direct policy 当作完全不使用 relay 的资源基线。
以后修改 metadata 协议或增加独立 direct catalog 后，才能安全缩减锁集合。

`extra_resources` 可给 job 增加共享资源名，例如两个 job 都设置 `["fabric-port-X"]`，会附加相同的 `configured:fabric-port-X` 锁。
它们只能增加默认锁，不能删掉默认锁。不同卡仍可能共享某个交换/链路资源，需用实际拓扑补充；不把默认卡集合宣称为完整物理冲突模型。
disjoint 示例的含义是“配置的资源集合不相交”，是否物理完全无争用还需 HCCN/拓扑证据。

只记录当前 launcher 管理的 workload；其他用户、进程的通信不在账上，建议测试时记录背景流量。
v1 不允许多个 worker job 复用 endpoint 卡，以免引入未经验证的同卡多进程 runtime 行为；relay 可以共享。

状态流：

```text
ARRIVING：聚合同一 (job, iteration, op) 的 ARRIVED，不占用资源
   ↓ 所有 rank 到齐
WAITING：所需资源有冲突则留在队列
   ↓ 一次性预约全部资源；生成唯一 ticket；发送各 rank 的 GO
RUNNING：预约期间包含发送 GO、写 policy、READY 和通信执行
   ↓ 聚合全部 rank 的 DONE（匹配同一个 ticket）
DONE：释放全部资源，重新调度等待队列
```

按就绪顺序扫描。后来的不相交任务可以越过阻塞任务；与较早阻塞任务共享资源的任务不能插队。
等待者不部分占锁，避免多资源“先占一半再等另一半”的死锁。
重复 ARRIVED、重复/旧 ticket DONE、越序消息、提前断连或进程失败均使本次实验失败，不自动解锁继续运行。
只有实际全部 DONE 才释放；没有基于预计耗时或超时的资源释放。

## 调用链与数据流

```text
run_a5_serial_scheduler.py
  load_config → JobSpec[] → Scheduler → ControlServer(AF_UNIX)
  启动各 job 的 rank0 / rank1（各自 visible_devices、manifest、bootstrap）

graph_gate.cc / GateMain
  GetRouteResources → 注册 PolicyKernel → capture 两个图内通信节点
  GateSchedulerClient：HELLO job rank → OK
  ExecuteAsync(model)
    图内：arrival callback / D2H notice
    host：Await(arrival) → started==0 / done 未出现
          scheduler.Arrive(iteration, op) → 阻塞等待 GO
    server：on_arrived → 收齐 rank → 检查资源 → reserve → GO ticket split[]
    host：再次确认 started==0 / done 未出现
          control stream: ValueWrite(policy[p], split[p]) → ValueWrite(gate, 1)
    图内：ValueWait(gate==1) → started=1 → PolicyKernel
          逐路径 LoadArg(policyAddr/auditAddr) → Load policy → Store audit
          非零路径 Write → 全部 EventWait → 双端完成握手 → done notice
    host：Await(done) → DONE iteration op ticket release_to_done_ns → ACK
    server：收齐全部 DONE → release → 下一个等待者 GO
    host：audit/payload 校验 → 两 rank 验证 barrier
    图内：gate=0 → D2D recv0→send1 → 第二个节点
  每轮尾部 drain exec/control 后才能复位通知槽和重放
```

callback 只更新原子通知计数；socket、调度、Runtime API 都在普通 host 线程执行。
放行使用独立 control stream，policy 和 READY 按顺序写入；没有把 READY 排在已被 ValueWait 阻塞的 stream 上。
任务 ABI 保持 `4 + 2*paths`，固定的是 device 地址，policy 值由 scheduler 的 GO 每次给出。
各 rank 根据同一个 job 的 policy 得到相同分块，支持 `mixed/direct/relay0/relay1/cycle`；本版不根据冲突改变这些分块。

DONE 前已有全部本地路径的 EventWait 和两端完成握手；调度器还会聚合两个 rank 的 DONE。
DONE 上报在 host audit/payload 验证前，减少验证拷贝对通信资源释放的影响。若随后校验失败，整次实验仍失败并终止其他 worker。
完整图结束和通信 DONE 是不同层次；本版每个 graph slot 同时仅允许一轮在途。

## API 和文件

| 文件 / 接口 | 职责 |
|---|---|
| `scripts/a5_path_scheduler.py` | 无 CANN 依赖的调度状态机、资源预约、FIFO、公平性、统计 |
| `scripts/run_a5_serial_scheduler.py` | 配置校验、Unix socket server、worker 生命周期、超时、日志和 summary |
| `testcase/graph_gate_scheduler.h` | C++ host client，HELLO / ARRIVED / GO / DONE / ACK |
| `testcase/graph_gate.cc` | 图内通知、gate、policy、CCU 通信与结果校验；无 scheduler 环境变量时保持原实验模式 |
| `aclrtLaunchHostFunc` / D2H notice | 图内到达、完成通知 |
| `aclrtValueWait` | 图内通信主体前等待 |
| `aclrtValueWrite` | 图外 control stream 写 policy、READY |
| `HcommCcuKernelLaunch` | 启动已注册的 policy kernel |

## 1. 构建与 CPU 检查

在此分支的仓库根目录运行。CANN 9.3 替换为实际安装路径，不加载旧 patched HCCL：

```bash
unset LD_PRELOAD
source /usr/local/Ascend/cann-9.3.0/set_env.sh
bash scripts/build_a5_ccu_urma_route_probe.sh \
  --cann-path /usr/local/Ascend/cann-9.3.0 \
  --build-dir /tmp/a5-scheduler-build
/tmp/a5-scheduler-build/a5_graph_gate --check-abi
python3 -m unittest discover -s tests/python/deepep -p 'test_a5_path_scheduler_cpu.py' -v
python3 -m unittest discover -s tests/python/deepep -p 'test_a5_relay_graph_gate_cpu.py' -v
# 使用实际 C++ GateSchedulerClient 替换 socket 集成测试中的 Python 假 worker：
A5_SCHEDULER_CPU_CLIENT=/tmp/a5-scheduler-build/a5_gate_scheduler_client_cpu \
  python3 -m unittest discover -s tests/python/deepep -p 'test_a5_path_scheduler_cpu.py' -v
```

`a5_gate_scheduler_client_cpu` 仅为协议测试夹具，会模拟 worker 成功标记，不能用于上卡/性能验收。
所有上卡命令必须使用 `a5_graph_gate`。

## 2. 准备卡号与真实 manifest

以下示例使用 0..7，不代表这些卡目前空闲。更换卡号时同时修改 JSON 和 manifest 生成参数。
manifest 相对路径相对于 JSON 所在目录解析。

```bash
mkdir -p /tmp/a5-scheduler-plans
cp examples/a5_ccu_urma_route_probe/testcase/scheduler_shared.example.json \
  /tmp/a5-scheduler-plans/shared.json
cp examples/a5_ccu_urma_route_probe/testcase/scheduler_disjoint.example.json \
  /tmp/a5-scheduler-plans/disjoint.json

for pair in 2,3 4,5 6,7; do
  src=${pair%,*}
  dst=${pair#*,}
  python3 scripts/resolve_a5_explicit_multirelay_eids.py \
    --root-info /etc/hccl_rootinfo.json \
    --topology-json /usr/local/Ascend/driver/topo/950/atlas_950_1.json \
    --src-phy "$src" --dst-phy "$dst" --relay-phys 0,1 \
    > "/tmp/a5-scheduler-plans/relays_${src}_${dst}.tsv"
done

python3 scripts/resolve_a5_explicit_multirelay_eids.py \
  --root-info /etc/hccl_rootinfo.json \
  --topology-json /usr/local/Ascend/driver/topo/950/atlas_950_1.json \
  --src-phy 4 --dst-phy 5 --relay-phys 6,7 \
  > /tmp/a5-scheduler-plans/relays_4_5_via_6_7.tsv
```

resolver 发现候选歧义时显式选择 plane，不猜 EID。JSON/manifest 的卡序必须一致，所有 endpoint/relay local die 约束沿用原实验。
多 communicator 同时初始化可能暴露 jetty/channel 限制：若 HcclCommInitRootInfo/建链失败，报告具体 API 和 plog，不归为 scheduler gate 失败。

## 3. 先验证“冲突时确实等待”

```bash
python3 scripts/run_a5_serial_scheduler.py \
  --binary /tmp/a5-scheduler-build/a5_graph_gate \
  --config /tmp/a5-scheduler-plans/shared.json \
  --schedule serialize --mode GRAPH --notify CALLBACK \
  --bytes 4194304 --iters 4 --warmup-iters 0 \
  --cohort-barrier --grant-delay-ms 200 --require-blocked \
  --timeout 300 --output-root /tmp/a5-scheduler-results
```

`--cohort-barrier` 是合成 workload 的到达对齐开关：同一 iteration/op 的全部 job 到齐后才调度；不是算法的必要条件。
`--grant-delay-ms 200` 仅用于正确性：拿到预约的 worker 在写 READY 前等待 200ms，让竞争者的等待清晰可见。它属于预约占用，不能当通信耗时或性能收益。

通过标准：

1. launcher 输出 `SERIAL_SCHEDULER_ALL_WORKERS_PASS`，所有 worker 零退出。
2. 每个 worker 都有 `2*iters` 个 `GATE_OP_PASS`，audit 和 payload 正确。
3. `events.jsonl` 有 BLOCKED，且任意共享资源的两次 GRANT 之间，前者必须出现所有 rank DONE 和 RELEASE。
4. worker 在收到 GO 前后的检查均为 started=0 / done 未出现。
5. 同一任务只有一个 ticket；所有 rank 使用相同 split；没有旧 replay 的通知混入。

可将 GRAPH 改 EAGER、CALLBACK 改 D2H 做四组合检查，不允许自动 fallback。

## 4. 验证“不冲突的任务仍并行”

使用同一命令换成 `disjoint.json`，移除 `--require-blocked`，仍可保留 200ms debug 窗口。
预期在第一项 RELEASE 前能看到两项 GRANT，且没有资源 BLOCKED。
这证明没有错误地串行整机。这里只证明配置资源域的并发，物理路径仍建议 HCCN 验证。

## 5. 性能对照

正确性通过后必须把 debug delay 设为 0。相同配置、字节数、policy、通知方式、poll 间隔、cohort 开关下比较：

```bash
for schedule in uncontrolled serialize; do
  python3 scripts/run_a5_serial_scheduler.py \
    --binary /tmp/a5-scheduler-build/a5_graph_gate \
    --config /tmp/a5-scheduler-plans/shared.json \
    --schedule "$schedule" --mode GRAPH --notify CALLBACK \
    --bytes 67108864 --iters 50 --warmup-iters 5 \
    --cohort-barrier --grant-delay-ms 0 --poll-us 50 \
    --timeout 600 --output-root /tmp/a5-scheduler-perf
done
```

建议再用 4MiB、16MiB、256MiB；去掉 cohort barrier 重测非同步到达；交替测试顺序、各重复至少三次，并记录背景通信。
`uncontrolled` 仍经过同一套通知、IPC 和 policy 下发，只是不限制资源重叠，因此可以比较“冲突串行策略”的效果。
另用原 `run_a5_relay_graph_gate.py --hold-ms 0` 测独立 job，估计集中控制额外开销；单 job 的结果不能直接当多 job 总吞吐。

结果目录：

- `arguments.json`、`config.json`：实际参数、卡号、manifest 和资源集合。
- `events.jsonl`：单调时钟下 ARRIVED / READY / BLOCKED / GRANT / DONE / RELEASE 时间线。
- `<job>/rank{0,1}.log`：真实 worker 的 CCU、gate、audit、payload 结果。
- `metrics.json`：逐 collective 的各 rank 到达差、cohort 等待、资源等待、预约占用、ready→done，以及每 rank release→done。
- `summary.json`：去 warmup 后的均值/p50/p95、阻塞次数和整个合成 iteration 的 span。
- `failure.json`：出错时仍在预约、排队的任务；不会通过超时清账伪造 PASS。

时间含义：

- 所有 scheduler 时间戳是同一 host 的 `monotonic_ns`，不依赖设备时钟同步。
- 资源等待从满足 cohort 条件开始计，cohort 等待单列；预约占用从 GRANT 记到最后一个 DONE。
- rank release→done 从 host 调用 READY 前记到发现 DONE，包含 Runtime 下发与轮询延迟，不是纯 device kernel 时间。
- iteration span 从当轮最早 ARRIVED 到最后 DONE，包含两个节点间的验证、D2D、IPC 等开销；未含图前输入初始化，跨 job 非同步重放时各轮 span 可重叠。
- Python、socket、50us host poll 和逐节点验证对小消息可能占主导。它们用于第一版功能与相对性能，不能将 span 直接宣称为真实模型迭代耗时，也不能用其倒数声称模型吞吐。
- 建议结合 HCCN 观察 uncontrolled 的 relay 重叠与 serialize 的时分使用，但正确性最终依赖 audit/payload、完整 rank 完成和资源时间线。

后续接八 rank AllToAllV 时，应保留 Scheduler 状态机，新增能生成完整 peer/path policy、聚合八 rank 到达/完成的 adapter；不要仅把当前两 rank worker 的 communicator size 改成 8。
