# 显式多路径 AllToAll：本地 copy 与远端 CCU 并行调度

分支 `feature/a5-ccu-copy-comm-overlap`，基于多类型分支 `2bb77ec`。
支持两卡 prepared-plan、两/四卡 peer-plan、FP16/BF16/FP32/INT8/INT32。
默认 `--self-copy-mode overlap`；`serial` 保留原本本地copy后再CCU的对照。
新增 `--self-copy-mode ccu`：同一个已注册CCU kernel内提交本地LocalCopyNb和远端WriteNb，
不再创建D2D copy stream/Event；执行和验收指令见**第9节**。
不是直接恢复旧组合CCU kernel，不修改系统HCCL、不加载ko、不改变UB路由表。

## 1. 数据区域与 fork/join

rank r 的本地复制只处理 `send[r] -> recv[r]`。
CCU仍使用已验证的RouteKernel，读取 `send[d] (d != r)`，接收数据写入
`recv[s] (s != r)`，自复制和远端通信的读写分片不重叠。
DeepEP拒绝send/recv别名、非连续tensor、类型不同等请求；host布局计算继续检查
整个buffer范围、整数溢出、peer覆盖和元素边界。不同调用使用相同buffer也要完整排序。

```text
caller stream：输入生产任务 -> record input_ready
                                  |
                  +---------------+----------------+
                  |                                |
                  v                                v
             CCU远端通信                       copy stream
          WriteNb / peer同步                  wait input_ready
                  |                           本地D2D copy
                  |                           record copy_done
                  +-------------+------------------+
                                |
                 caller stream wait copy_done
                                |
                  下游消费 / 下一次输入覆写
```

host按这个顺序提交：
1. 在调用流记录input_ready（记录点在CCU之前）。
2. 提交CCU及原有必要的die/thread完成依赖。
3. copy流等待第1步的输入进度，提交本地copy并记录copy_done。
4. 调用流在CCU之后等待copy_done，再允许下游或下一次调用。

**优先提交CCU**：避免4us的copy先结束，CPU却还在准备CCU launch。
copy等待的不是“CCU完成”，只是CCU提交之前的输入生产任务；没有双方互等的环。
IO die1仍沿用原有main/route thread通知，最后在调用流汇合，不删掉原有CCU完成握手。
这允许两条设备分支重叠，不能保证任意负载实际发生重叠，更不能提前承诺性能收益。
输入在其他stream生产时，调用者仍必须建立该stream到调用stream的正确依赖。

## 2. 事件复用、资源生命周期与错误

每个(plan, execution stream)在图外绑定时创建一个copy stream、两个逻辑Event。
只使用 `aclrtCreateEventExWithFlag(..., ACL_EVENT_SYNC)`；SDK支持重复Record，
Wait绑定最近一次Record的进度，不会被后续Record改写。
依据：[官方RecordEvent语义](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910beta1/API/runtimeapi/aclcppdevg_03_0083.html)、
[Ex Event约束](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha001/API/appdevgapi/aclcppdevg_03_0081.html)。
不能替换成旧Event然后省略Reset；Ex Event本身也不应调用Reset。
逻辑handle只创建一次；底层Record可能使用临时SDK资源，其回收由SDK负责，不宣称
无限排队不受任何资源限制。测试的正常路径不增加逐call CPU synchronize。

输入、输出的allocator寿命在调用流记录；调用流的最终join覆盖copy流的访问。
如果CCU提交失败，不再启动本地copy；Event/copy/join失败会poison lane，拒绝继续使用。
已经提交copy后的错误路径尝试同步copy流，避免悄悄留下访问已释放tensor的任务。
发生driver/runtime故障时同步本身也可能失败，因此不能承诺故障后设备仍可安全继续；
保留错误、停止实验，不自动reset公共设备。

新增图外销毁接口（不要求新DeepEP wheel）：

```c
HcclCcuUrmaPeerPlanDestroy(uint64_t handle);
HcclCcuUrmaExplicitMultipathPlanDestroy(uint64_t handle);
```

停止该plan的host提交，等待已有执行完成，释放所有引用该plan的图，再调用Destroy。
它清理自有copy stream/Event及host plan记录；原有HCCL Channel缓存仍由通信runtime
管理，不宣称这个接口释放了所有底层Channel/TP。
测试脚本在最终同步、各rank报告barrier后，先reset ACLGraph，再Destroy，最后
destroy_process_group。模型集成同样遵守这个顺序；不要在capture或仍会replay时销毁。
旧两卡脚本的其他 torch.compile 后端可能由框架缓存持有图，不能擅自 Destroy；
其进程退出回收不能替代长期服务的显式图生命周期管理。本指南先验证 none/aclgraph。
晚期static析构不调用可能已关闭的ACL runtime，未显式释放的资源只能由进程退出回收；
长期服务必须调用Destroy，不能靠进程退出管理资源。

## 3. 拉取与构建

```bash
(
set -eo pipefail
cd /home/l00934901/sgl-kernel-npu
git stash push -m "backup-before-copy-overlap-$(date +%Y%m%d_%H%M%S)"
BRANCH=feature/a5-ccu-copy-comm-overlap
git fetch origin "${BRANCH}"
git switch "${BRANCH}"
git merge --ff-only "origin/${BRANCH}"
git branch --show-current
git rev-parse --short HEAD
# 必须包含LocalCopyNb+WriteNb实现，不能只拉到旧的overlap版本。
git merge-base --is-ancestor 613ce68 HEAD
source /usr/local/Ascend/cann-9.1.T560/set_env.sh
unset LD_PRELOAD A5_CCU_PREPARED_ALLOW_LAZY_STREAM A5_URMA_TP_TRACE_PREFIX

HCCL_REPO=/home/l00934901/hccl \
bash scripts/build_a5_ccu_urma_route_probe.sh \
  --install --install-path /usr/local/Ascend/cann-9.1.T560

python3 tests/python/deepep/test_a5_ccu_peer_plan_alltoall.py --runtime-check --self-copy-mode ccu
)
```

需要重编安装route package。若已装多类型版DeepEP(peer ABI>=2、attr ABI>=12)，
**无需重编DeepEP/wheel**；如果仍是旧FP32/旧四卡wheel，按多类型指南先升级wheel。
不编译/替换系统libhccl。新版route产物：prepared ABI=15、peer ABI=4、
local+remote ABI=1。旧serial/overlap最低要求仍为prepared ABI>=14、peer ABI>=3、
copy-overlap ABI>=1。Supported()=1只说明找到Ex Event符号，不表示硬件/capture已验证。
缺少该接口时overlap模式明确失败，不默默换不安全event；可选serial模式作对照。
安装前停止本用户使用旧库的进程；共享CANN目录仍有公共文件覆盖风险。

## 4. 正确性短测：先direct-only，再显式relay

先确认计算卡可用。下面用2、3卡测FP32，参数可换成其他可用的两/四张卡。

```bash
bash scripts/run_a5_ccu_peer_plan_alltoall.sh \
  --devices 2,3 --dtype float32 --self-copy-mode overlap \
  --sync-mode per-call --graph-backend none \
  --bytes 4194304 --warmup 1 --iters 3 --repeats 1 \
  --output-root /home/l00934901/profiling
```

四卡relay配置先用现有JSON，只有确认0..3及外部relay资源允许使用时执行：

```bash
bash scripts/run_a5_ccu_peer_plan_alltoall.sh \
  --devices 0,1,2,3 --relay-map docs/topology/a5_peer_plan_4rank_example.json \
  --dtype bfloat16 --self-copy-mode overlap \
  --sync-mode per-call --graph-backend none \
  --bytes 4194304 --warmup 1 --iters 3 --repeats 1 \
  --output-root /home/l00934901/profiling
```

报告须complete=True，所有rank正确且正常退出；检查资源创建及清理阶段：

```bash
RUN_DIR=$(ls -dt /home/l00934901/profiling/a5_ccu_peer_plan_* 2>/dev/null | head -1)
if test -n "${RUN_DIR}" && test -f "${RUN_DIR}/cases/peer_plan_r1.log"; then
  grep -nE 'SELF_COPY_LANE|COPY_OVERLAP_ERROR|release_copy_plan|RESULT_JSON|PEER_CASE_FAILURE' \
    "${RUN_DIR}/cases/peer_plan_r1.log"
  sed -n '1,180p' "${RUN_DIR}/peer_plan_report.md"
fi
```

应看到mode=overlap、不同main/copy stream、每绑定两个逻辑Event、各rank
release_copy_plan_done。必须分别验证五种dtype；数据图外gate使用可精确表示的小整数，
有位置/source/destination/epoch编码和poison，不靠宽松误差容忍。

## 5. 串行/并行A/B与profiling

### 5.1 两卡 direct + 2 / 4 / 6 条显式 relay

下面以通信卡2、3为例，直接用命令行 `--relay-phys` 选择relay，不需要修改JSON。
三组分别是：

| 命令行参数 | 显式 relay 卡 | 总 Channel 数 | 权重（direct在前） |
|---|---|---:|---|
| `--relay-phys 0,1` | 0、1 | 3 | 2,1,1 |
| `--relay-phys 0,1,4,5` | 0、1、4、5 | 5 | 2,1,1,1,1 |
| `--relay-phys 0,1,4,5,6,7` | 0、1、4、5、6、7 | 7 | 2,1,1,1,1,1,1 |

只需修改 `--devices` 和 `--relay-phys` 的卡号即可适应资源变化。`--relay-phys`
用于两卡唯一的通信对；保留列表顺序，不自动增减/替换relay。1、3、5条同样支持，
不限制必须为2、4、6条。脚本将参数交给同一个拓扑/EID解析及manifest构建逻辑，
不会根据卡号硬编码port/EID，也不修改CCU数据面。
relay不能是通信域内的卡，且全部卡必须在 `--available-phys` 的集合中（默认0..7）。
重复、负数、空列表、通信域内或不在available集合的relay会在加载NPU runtime之前
被拒绝。资源不允许时减少relay，或省略 `--relay-phys` 和 `--relay-map` 测direct-only。
不要借用未获准使用的卡。两参数互斥，不接受同时传入。

四卡有多个通信对，继续用 `--relay-map JSON` 指定逐对策略；不能把一份relay列表
自动复制给所有peer。原来的两卡JSON接口及2/4/6relay示例文件也保留兼容。

可先只检查参数，不加载runtime、不读拓扑、不写文件：

```bash
python3 scripts/prepare_a5_ccu_peer_plan.py \
  --devices 2,3 --relay-phys 0,1 --validate-only
```

`PEER_POLICY_VALIDATED` 只表示配置合法，不证明这些物理路径已经可以建链。

三组都采用 **direct : 每一条relay = 2:1**。每peer总数据4MiB不变，因此每条路径
分到的数据会随路径数改变，不是额外增加总流量。`--dtype` 可替换为其他支持的类型。

先按第4节完成短测。下面每条命令只跑一轮，warmup100次、正式20次，profiling
只采正式20次；没有graph capture，没有逐次device synchronize。

两卡 direct + 2 条relay（0、1）：

```bash
bash scripts/run_a5_ccu_peer_plan_alltoall.sh \
  --devices 2,3 --relay-phys 0,1 \
  --direct-route 0 --dtype float32 --self-copy-mode overlap \
  --sync-mode batch --graph-backend none \
  --bytes 4194304 --warmup 100 --iters 20 --repeats 1 --profile \
  --output-root /home/l00934901/profiling
```

两卡 direct + 4 条relay（0、1、4、5）：

```bash
bash scripts/run_a5_ccu_peer_plan_alltoall.sh \
  --devices 2,3 --relay-phys 0,1,4,5 \
  --direct-route 0 --dtype float32 --self-copy-mode overlap \
  --sync-mode batch --graph-backend none \
  --bytes 4194304 --warmup 100 --iters 20 --repeats 1 --profile \
  --output-root /home/l00934901/profiling
```

两卡 direct + 6 条relay（0、1、4、5、6、7）：

```bash
bash scripts/run_a5_ccu_peer_plan_alltoall.sh \
  --devices 2,3 --relay-phys 0,1,4,5,6,7 \
  --direct-route 0 --dtype float32 --self-copy-mode overlap \
  --sync-mode batch --graph-backend none \
  --bytes 4194304 --warmup 100 --iters 20 --repeats 1 --profile \
  --output-root /home/l00934901/profiling
```

每条命令结束后记下它打印的 `Result directory`，三个目录使用相同前缀，
不要仅取latest目录就当成全部三组结果。每组prof在其结果目录的
`r1/profiling/rank0/` 和 `r1/profiling/rank1/` 下，MindStudio读取里面的
`*_ascend_pt` 目录。`plans/peer_plan.tsv` 和 `plans/peer_plan.json` 可核对实际relay、
EID和权重；`peer_plan_report.md` 应complete=True且所有rank正确、正常退出。

### 5.2 本地copy串行 / 并行A/B

以下以6条relay为例。测2条或4条时，把 `--relay-phys` 换成第5.1节对应列表即可。
两次都使用同一组路径，唯一改变的是本地copy与CCU的调度方式：

```bash
(
set -e
for mode in serial overlap; do
  bash scripts/run_a5_ccu_peer_plan_alltoall.sh \
    --devices 2,3 --relay-phys 0,1,4,5,6,7 \
    --dtype float32 --self-copy-mode "${mode}" \
    --sync-mode batch --graph-backend none \
    --bytes 4194304 --warmup 100 --iters 20 --repeats 1 --profile \
    --output-root /home/l00934901/profiling
done
)
```

每组：图外资源 → serial changed-payload检查 → 两轮batch gate（换输入、poison、
每次snapshot）→ warmup100 → 正式20 → 正确性/资源清理。
只采正式20；保留同一张卡、dtype、bytes、路径及权重才能比较两个mode。
四卡换devices和relay-map即可，当前每peer最多1外部relay的约束不变。
旧两卡性能脚本亦新增 `--self-copy-mode`，对prepared实现有效；native和legacy组合
probe不由这个开关改写，因此不要把它当remote路径串/并行的schedule开关。

### 5.3 如何判断重叠与完整耗时

MindStudio中重叠要用**同一调用**的设备Ccu与D2D copy时间区间判断：

```text
overlap_us = max(0, min(copy_end, ccu_end) - max(copy_start, ccu_start))
data_task_span = max(copy_end, ccu_end) - min(copy_start, ccu_start)
```

overlap_us>0才是物理时间重叠证据，mode=overlap不等于该值大于0。
两段耗时不能相加；data_task_span也未包含所有host准备/Event提交开销。
端到端同时报告batch host总时间/20，不能只看CCU时间就声称收益。
HBM竞争、Event/launch提交开销及peer等待可能使收益小甚至无收益，需实际数据。

## 6. ACLGraph单独验证

首先上述eager、batch正确性通过，再测graph。多个stream的capture支持取决于目标
T560 runtime，编译或Ex符号存在不能替代这一步；任何失败都停止，不伪造PASS或
把copy移回串行来掩盖失败。

```bash
bash scripts/run_a5_ccu_peer_plan_alltoall.sh \
  --devices 2,3 --relay-phys 0,1,4,5,6,7 \
  --dtype bfloat16 --self-copy-mode overlap \
  --sync-mode per-call --graph-backend aclgraph \
  --bytes 4194304 --warmup 1 --iters 3 --repeats 1 \
  --output-root /home/l00934901/profiling
```

四次capture/replay验证每次改变输入并poison graph输出，检查包括self row，能捕获
“CCU replay但本地copy只在capture时发生”的错误。该测试是有界验证，不证明任意
并发/图生命周期安全。reset graph后再Destroy；已捕获图不能仍引用已销毁plan。

## 7. 代码与开发验证

`op_host/self_copy_lane.{h,cc}`：资源、输入fork、copy提交、join、失败poison与Close。
`peer_plan_runtime.cc` / `all_to_all_multiroute.cc`：接入Guard并保持原远端CCU协议。
`RouteKernel`不变；没有修改CompletedEvent容器或恢复LocalCopyNb组合kernel。
Tensor allocator不新增依赖库，调用流join让原有recordStream覆盖两个分支。

CPU模型 `tests/cpp/test_a5_ccu_self_copy_lane.cpp` 使用假的ACL队列：验证2/4rank、
1/2/4byte元素、50次无逐call sync的不同输入和输出canary、record进度快照、
copy与CCU无互相依赖、consumer同时等待二者、部分创建失败与join失败、资源关闭。
它不是实际设备并行测试。测试需CANN头文件，编译时不链接真实ACL库：

```bash
(
set -e
TMP_TEST=$(mktemp -d /tmp/a5_copy_model.XXXXXX)
g++ -std=c++14 -Wall -Wextra -Werror -rdynamic \
  -Iexamples/a5_ccu_urma_route_probe/op_host \
  -I/usr/local/Ascend/cann-9.1.T560/include \
  tests/cpp/test_a5_ccu_self_copy_lane.cpp \
  examples/a5_ccu_urma_route_probe/op_host/self_copy_lane.cc \
  -ldl -o "${TMP_TEST}/copy_model"
"${TMP_TEST}/copy_model"
python3 tests/python/deepep/test_a5_ccu_copy_overlap_cpu.py
)
```

开发容器cam_lyw_dev_91/CANN9.1.0-beta.1已编译route package、通过上述CPU队列模型。
没有NPU；目标T560的实际正确性、物理重叠、性能和ACLGraph replay尚需按4~6节验证。

## 8. 独立 CCU 本地 copy 穿刺（先证实，再考虑组合 kernel）

普通四卡分支的串行copy较稳定，新分支 `--self-copy-mode serial` 对照也较稳定，
因此现在优先隔离 CCU 本地copy能力，而不是修改relay或删掉输出完成依赖。
本节测试不恢复组合kernel，不修改生产算子，不安装新的SO/wheel，不编译系统HCCL。
**拉取本次代码后直接执行下面脚本即可**；脚本自动把独立C++程序编译到结果目录。
需有目标CANN的开发头文件（包括pkg_inc）和g++，不是只有运行时SO的环境。

### 8.1 有卡执行命令

```bash
cd /home/l00934901/sgl-kernel-npu
source /usr/local/Ascend/cann-9.1.T560/set_env.sh
unset LD_PRELOAD A5_URMA_TP_TRACE_PREFIX

bash scripts/run_a5_ccu_local_copy_probe.sh \
  --devices 2,3 \
  --sizes 4096,4194304 --copy-mode both \
  --changed-iters 3 --queued-iters 20 \
  --timeout-seconds 300 \
  --cann-root /usr/local/Ascend/cann-9.1.T560 \
  --output-root /home/l00934901/profiling
```

修改 `--devices` 即可换计算卡；接受一张或两张卡，默认两卡使用已知通信域初始化
流程，但每个rank只在自己的HBM上copy。单卡后端若在初始化/注册阶段拒绝请求，
不能据此判断本地copy不支持，可用上述两卡上下文继续隔离。
没有relay参数，不使用其他六张卡做数据转发。通信域初始化可能有控制面通信。
这是新CCU kernel的设备验证，不能承诺零风险；仅在获准测试的卡上执行，遇到
驱动错误/超时立即停止，不reset公共设备。

默认依次测试：

| case | CCU本地数据面 |
|---|---|
| direct_4096 | 单次 `LocalCopyNb(LocalAddr, LocalAddr, 4096) + WaitEvent` |
| chunked_4096 | 4KB分块对照（一个块） |
| chunked_4194304 | CCU内部循环，1024个4KB块，每块copy后Wait，累积复制4MB |
| direct_4194304 | 单次 `LocalCopyNb(LocalAddr, LocalAddr, 4194304) + WaitEvent` |

先跑分块大尺寸，再跑单次大尺寸，保留对长度限制的区分。没有引入CCU buffer/
GroupCopy的另一套资源，不宣称该分块控制就是原生GroupCopy的实现。
若失败或超时，脚本停止后续硬件case，不自动reset设备。每case每rank独立进程，
外层timeout覆盖初始化、执行及退出；不能用程序打印过数据PASS就忽略退出失败。

### 8.2 测试如何排除假阳性

独立入口：`examples/a5_ccu_urma_route_probe/testcase/local_copy_probe.cc`。

```text
aclInit / SetDevice / HcclCommInitRootInfo
  -> HcclThreadAcquireWithStream(COMM_ENGINE_CCU)
  -> HcclCcuKernelRegister / RegisterFinish（图外各一次，channels为空）
  -> H2D输入 + H2D污染输出（同一执行stream）
  -> HcclCcuKernelLaunch：只有LocalCopyNb + WaitEvent
  -> D2H输出快照
  -> 同步后逐字节检查 + guard检查
  -> 清理通信域、stream、device/pinned host缓冲区
```

不调用 `HcclChannelAcquire`、远端 `WriteNb` 或 `ACL_MEMCPY_DEVICE_TO_DEVICE`。
H2D只准备输入/污染输出，D2H只读取结果，因此不存在ACL D2D代替CCU把输入搬到输出。
输入/输出有各自的token、独立allocation，copy只覆盖payload；前后各4KB guard。
污染值为对应期望字节取反，任何未写字节都不能碰巧通过。input也检查未被修改。

先3次改变输入、每次同步校验，用于确认首次/独立调用；再20次改变输入并连续排队，
复用device input/output，每次保存独立D2H快照，**循环外同步后检查全部20个快照**。
Pinned host源/目的区间保留到同步完成，避免异步H2D读取被CPU提前覆盖。
资源只注册一次；不能靠每轮重新注册或host同步掩盖连续复用问题。
这是正确性诊断，不是性能benchmark：不要把混合H2D/D2H的耗时当本地copy带宽。

### 8.3 PASS、失败与分析文件

全部rank、全部请求case的数据、guard、连续快照及进程清理均成功，最后才打印：

```text
PASS: LocalCopyNb + WaitEvent completed on device; changed payload, every queued snapshot, guards and cleanup verified.
```

各case也单独打印 `direct_4096: PASS` 等。只有小尺寸或分块PASS、单次4MB失败时，
说明已有对应case的本地copy实测证据；不应推断“CCU完全不能本地copy”，也不能
把单次4MB模式用于生产。若卡在kernel_register等阶段，结论仅为该上下文/接口未
验证通过，而不是硬件复制失败。注册返回0、构图通过都不算数据PASS。

```bash
RUN_DIR=$(ls -dt /home/l00934901/profiling/a5_ccu_local_copy_* 2>/dev/null | head -1)
if test -n "${RUN_DIR}" && test -d "${RUN_DIR}"; then
  python3 scripts/analyze_a5_ccu_local_copy_probe.py --run-dir "${RUN_DIR}"
  column -s $'\t' -t "${RUN_DIR}/local_copy_summary.tsv"
  sed -n '1,220p' "${RUN_DIR}/local_copy_report.md"
  grep -RHnE 'LOCAL_COPY_PHASE|LOCAL_COPY_LIBRARY|LOCAL_COPY_FAILURE|LOCAL_COPY_WORKER_PASS' \
    "${RUN_DIR}/cases" --include='rank*.log'
fi
```

分析脚本要求worker和case退出状态均为0，逐rank严格校验JSON与cleanup完成标记；
缺文件、半轮、损坏JSON、超时或纯builder结果均不能生成整体PASS。
编译失败看 `build.log`；执行失败看 `cases/<case>/rank0.log`、`rank1.log` 的最后phase。

### 8.4 无卡开发验证（不是硬件证明）

```bash
bash scripts/run_a5_ccu_local_copy_probe.sh --builder-only \
  --cann-root /usr/local/Ascend/cann-9.1.T560 \
  --output-root /tmp/a5_local_copy_builder
```

只编译并调用 `CcuKernel::Init()` 构建direct/chunked指令表示，不创建通信域或launch。
输出 `BUILD_ONLY_PASS` / `hardware_copy_executed=0`，**不等同第8.3节的硬件PASS**。
开发容器没有NPU；独立程序的目标SDK编译、构图和CPU校验只能证明软件链路，
实际结论以8.1的有卡数据为准。

补充纠正：SDK的CompletedEvent按值传递，内部持有资源表示。旧版本增加reserve
是防御性修改，不能仅凭vector扩容就声称已证实悬挂引用；过去组合失败还涉及bridge
生产任务提交顺序、通知布局和逐Channel结束握手，尚无LocalCopyNb本身失败的定论。

## 9. 已通过 local-copy 穿刺后的组合 AllToAll 算子

### 9.1 实现与边界

用户反馈第8节local-copy有卡矩阵全部PASS，包括单次4MB和连续快照。基于这个
前提，本版本在**现有prepared-plan算子**中增加CCU本地+远端数据面，而不是仅增加
另一个独立测试程序。开发容器编译/CPU回归不代替目标机器上的组合验收。
`ccu`仍为显式选择项，没有静默回退D2D；旧默认overlap保持不变。
本版本仍是已有Buffer/torch.ops prepared-plan封装，不宣称新增了Ascend C tiling标准外壳。

调用入口不变：

```text
图外：prepare_ccu_urma_peer_plan / bind_ccu_urma_peer_plan
  -> topology/EID -> explicit ChannelDesc -> ChannelAcquire
  -> RouteKernelArg(localCopy=true) -> Register -> RegisterFinish -> handle

每次：buffer.ccu_urma_peer_plan_alltoall_out(input, output, handle, policy)
  -> HcclCcuUrmaPeerPlanExecute
  -> 推断dtype并计算字节布局/完整token
  -> HcclCcuKernelLaunch（缓存kernel，不重新注册）
  -> caller stream上完成；必要的IO die线程汇合保留
```

旧两卡 `ccu_urma_prepared_multipath_alltoall_policy_out` prepared-plan入口也支持ccu模式；
推荐下面统一的peer-plan测试，两卡/四卡共用同一份环境和计时逻辑。

组合CCU指令次序：

```text
Load input/output/tokens/selfOffset/selfBytes/path offsets
  -> 每条Channel发布本rank的output/token
  -> LocalCopyNb(recv[r], send[r], peerBytes, localEvent)  # dst在前，提交不等待
  -> 等待各对端output/token
  -> WriteNb(channel0, chunk0, remoteEvent0)             # 提交，不等待
  -> WriteNb(channel1, chunk1, remoteEvent1)
  -> ...所有远端分片均提交...
  -> WaitEvent(localEvent)
  -> WaitEvent(remoteEvent0/1/...)
  -> 每条Channel completion NotifyRecord + NotifyWait
  -> 下游使用output / 下一次覆写input
```

LocalCopyNb使用`LocalAddr -> LocalAddr`，与第8节direct模式相同，不在这里按4KB
串行分块。本地复制与远端write**允许并行**，不保证任意硬件资源下时间完全重叠。
本地任务可能在对端等待期间已完成，因此不能只凭API名宣称性能必然提升。

正确性约束：

- 自复制只写`recv[r]`；来自rank s的数据只写`recv[s]`，s!=r。
- 每个对端的direct/relay chunk覆盖该peer行，互不重叠；dtype元素边界保持。
- self输入/输出使用完整buffer token，不用某个Channel的remote token。
- 一个独立local CompletedEvent，远端每条Channel各自独立event；不占用Channel通知槽。
- **所有WriteNb都在第一次WaitEvent之前提交**；逐Channel后置握手防止前一轮接收未完
  就返回、下一轮覆写buffer。保留bridge的输入任务flush、DeviceGuard、recordStream。
- 内核签名包含local模式和实际Channel集合；旧remote-only注册缓存不可能冒充组合kernel。
- 共6+3×Channel数个uint64 launch参数，仍受48个参数限制（最多14 Channels）。
  两卡direct+6relay为7条，四卡每peer direct+1relay为6条；不是无限资源承诺。
- 模式在图外绑定时固定；想换serial/overlap/ccu请建立新plan或用独立进程，不在同一
  plan执行中途改环境变量。不同stream须图外绑定，各rank策略匹配。

源码：`op_kernel_ccu/route_kernel.{h,cc}`、`op_host/peer_plan_runtime.cc`、
`op_host/all_to_all_multiroute.cc`、`op_host/utils.cc`（路径前缀均为
`examples/a5_ccu_urma_route_probe/`）。本版本没有恢复历史AllToAllMultiRouteKernel。

### 9.2 更新、安装与ABI预检

按第3节拉取当前overlap分支并**重编/安装route自定义包**。DeepEP若已有多类型
peer ABI>=2 / attr ABI>=12，不需要重新编译wheel；不改/安装系统libhccl或ko。
共享CANN安装目录存在覆盖旧custom SO的风险，先停止使用该库的本用户进程，并遵守
机器管理权限；本版本运行时不修改UB路由表。

```bash
cd /home/l00934901/sgl-kernel-npu
source /usr/local/Ascend/cann-9.1.T560/set_env.sh
unset LD_PRELOAD A5_URMA_TP_TRACE_PREFIX A5_CCU_PREPARED_ALLOW_LAZY_STREAM
HCCL_REPO=/home/l00934901/hccl \
bash scripts/build_a5_ccu_urma_route_probe.sh \
  --install --install-path /usr/local/Ascend/cann-9.1.T560

python3 tests/python/deepep/test_a5_ccu_peer_plan_alltoall.py \
  --runtime-check --self-copy-mode ccu
```

要求route peer ABI>=4且`A5CcuLocalRemoteAbiVersion()>=1`；旧两卡入口要求prepared
ABI>=15。检查通过只证明正确库已加载，不证明有卡数据正确。此预检会拒绝旧库。

### 9.3 先验证direct-only，再验证显式relay

```bash
# 两卡direct-only：不加relay参数。
bash scripts/run_a5_ccu_peer_plan_alltoall.sh \
  --devices 2,3 --self-copy-mode ccu \
  --sync-mode batch --graph-backend none \
  --bytes 4194304 --warmup 1 --iters 3 --repeats 1 \
  --output-root /home/l00934901/profiling

# 同两张计算卡，relay 0/1由命令行指定。
bash scripts/run_a5_ccu_peer_plan_alltoall.sh \
  --devices 2,3 --relay-phys 0,1 --self-copy-mode ccu \
  --sync-mode batch --graph-backend none \
  --bytes 4194304 --warmup 1 --iters 3 --repeats 1 \
  --output-root /home/l00934901/profiling

# 四卡：逐peer显式分配通信域外的relay，沿用已有JSON。
bash scripts/run_a5_ccu_peer_plan_alltoall.sh \
  --devices 0,1,2,3 --relay-map docs/topology/a5_peer_plan_4rank_example.json \
  --self-copy-mode ccu --sync-mode batch --graph-backend none \
  --bytes 4194304 --warmup 1 --iters 3 --repeats 1 \
  --output-root /home/l00934901/profiling
```

四卡direct-only只需省略relay-map。relay数量/实际port/EID仍由原拓扑解析处理，
没有写死relay0/1或固定端口。五种dtype用`--dtype float16|bfloat16|float32|int8|int32`。
每目的rank的数据4MB，tensor总大小k×4MB，含self行；direct:每条relay=2:1。

batch模式在warmup前自动做两轮改变输入/毒化输出的排队验证，保存并检查**每次**
结果，包括本地行和所有对端行。不能仅检查最后一轮，不能用固定输入掩盖重复写旧值。
如果组合数据有任何错误/资源超时，停止性能测试，不自动fallback、不reset公共卡。

### 9.4 两卡2/4/6 relay profiling：一轮warmup100+正式20

下面每个relay列表为一个独立进程组，只采正式20次；warmup及正确性gate不入profiler。
warmup100循环后同步一次，正式20循环后同步一次。第9.3节先PASS后再执行。

```bash
for RELAYS in 0,1 0,1,4,5 0,1,4,5,6,7; do
  bash scripts/run_a5_ccu_peer_plan_alltoall.sh \
    --devices 2,3 --relay-phys "${RELAYS}" --self-copy-mode ccu \
    --sync-mode batch --graph-backend none \
    --bytes 4194304 --warmup 100 --iters 20 --repeats 1 --profile \
    --output-root /home/l00934901/profiling || break
done
```

serial对照用完全相同卡/路径/大小/计时模式，只改`--self-copy-mode serial`。
组合模式正式区域预期主要是一个CCU任务，不再有算子self行的ACL D2D MemcpyAsync；
输入构造、正确性快照、profiler自身仍可能产生其他memcpy，勿据名称一概判错。
CCU任务含本地复制、远端通信及peer等待；它的时长不是纯链路write时间。系统仍可能
出现rank起始偏差或资源争用，不能承诺全部消除历史波动。

### 9.4.1 四卡 direct+relay：完整 profiling 指令

先完成第9.2节新版安装/预检及第9.3节四卡短测，再执行：

```bash
cd /home/l00934901/sgl-kernel-npu
source /usr/local/Ascend/cann-9.1.T560/set_env.sh
unset LD_PRELOAD A5_URMA_TP_TRACE_PREFIX A5_CCU_PREPARED_ALLOW_LAZY_STREAM

bash scripts/run_a5_ccu_peer_plan_alltoall.sh \
  --devices 0,1,2,3 \
  --relay-map docs/topology/a5_peer_plan_4rank_example.json \
  --self-copy-mode ccu \
  --sync-mode batch --graph-backend none \
  --dtype float32 --bytes 4194304 \
  --warmup 100 --iters 20 --repeats 1 --profile \
  --timeout-seconds 600 \
  --cann-root /usr/local/Ascend/cann-9.1.T560 \
  --output-root /home/l00934901/profiling
```

示例JSON显式指定：0↔1、2↔3用relay4；0↔2、1↔3用relay5；0↔3、1↔2用relay6。
这些relay均在通信域0～3之外，不启动relay rank进程；每rank对应3个peer、共6条
远端Channel（每peer一个direct和一个relay），另有本地self copy。
每peer direct:relay=2:1；`--bytes`是每目的rank 4MiB，tensor总大小16MiB，含self行。
如果卡组或relay不可用，先更换对应的逐peer JSON，不要只改devices沿用不匹配的map。

初始化/资源注册一次；batch正确性gate、warmup100不采profiler，**只采正式20次**。
warmup循环外同步，正式循环外同步，跑一轮。MindStudio输出位于本次目录
`r1/profiling/rank0/`～`rank3/`；结果/PASS检查使用第9.5节命令。
想测四卡direct-only，省略`--relay-map`即可；想做旧D2D串行对照，只改
`--self-copy-mode serial`，其余卡组、数据量和计时参数保持一致。

### 9.5 组合PASS提示与分析

```bash
RUN_DIR=$(ls -dt /home/l00934901/profiling/a5_ccu_peer_plan_* 2>/dev/null | head -1)
if test -n "${RUN_DIR}" && test -d "${RUN_DIR}"; then
  python3 scripts/analyze_a5_ccu_peer_plan.py --run-dir "${RUN_DIR}"
  sed -n '1,220p' "${RUN_DIR}/peer_plan_report.md"
  grep -HnE 'CCU_LOCAL_REMOTE_PLAN|PEER_CASE_FAILURE|PASS|RESULT_JSON' \
    "${RUN_DIR}"/cases/*.log
fi
```

所有rank结果、改变输入batch gate、case退出清理均正确，且每个rank明确注册了
LocalCopyNb组合模式，分析脚本才输出：

```text
PASS: combined LocalCopyNb + peer WriteNb AllToAll; data, queued snapshots and lifecycle verified.
```

缺rank、缺组合标记、退出失败、数据错误不会给整体PASS。这个PASS验证本次算子功能，
不是周期级重叠/带宽收益/指定relay硬件转发的新证明；后者结合profiler与HCCN。

图验证需另测，不能由旧D2D模式的capture PASS直接推出：

```bash
bash scripts/run_a5_ccu_peer_plan_alltoall.sh \
  --devices 2,3 --relay-phys 0,1 --self-copy-mode ccu \
  --sync-mode per-call --graph-backend aclgraph \
  --bytes 4194304 --warmup 1 --iters 3 --repeats 1 \
  --output-root /home/l00934901/profiling
```

图模式是功能验收，不与batch性能混比。Destroy仍在同步并释放图之后调用。

### 9.6 开发回归

```bash
python3 tests/python/deepep/test_a5_ccu_local_remote_cpu.py
```

此测试用CPU表示记录器编译**实际生产RouteKernel源码**，验证所有WriteNb在完成等待
前提交、local/remote独立事件、逐Channel结束握手、参数Load/GeneArgs匹配，并模拟
2/4卡、1/2/4字节、不同path数、40次新输入复用/乱序copy完成、source和guard保护。
它不是设备并行模拟器，不验证目标固件资源、DMA/CCU性能或真实graph capture。
