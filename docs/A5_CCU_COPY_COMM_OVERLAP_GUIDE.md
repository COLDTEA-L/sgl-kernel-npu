# 显式多路径 AllToAll：本地 copy 与远端 CCU 并行调度

分支 `feature/a5-ccu-copy-comm-overlap`，基于多类型分支 `2bb77ec`。
支持两卡 prepared-plan、两/四卡 peer-plan、FP16/BF16/FP32/INT8/INT32。
默认 `--self-copy-mode overlap`；`serial` 保留原本本地copy后再CCU的对照。
`--self-copy-mode ccu` 现为原生风格 GroupCopy（8路×32KiB、HBM→MS→HBM）与远端WriteNb；
旧的整块LocalCopyNb保留为 `ccu-direct` 对照，不再作为高性能实现。
不创建D2D copy stream/Event；新版本验证请从**第11节**开始，第9节组合测试也采用新版GroupCopy。
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
不编译/替换系统libhccl。新版route产物：prepared ABI=16、peer ABI=5、
local+remote ABI=2（原生风格GroupCopy）。旧serial/overlap最低要求仍为prepared ABI>=14、peer ABI>=3、
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

新版 `ccu` 按原生GroupCopy结构执行；旧整块方案为 `ccu-direct` 对照。
**以下第9.1节的指令图仅描述ccu-direct，不是新版ccu。** 新版见第11.2节。

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
- ccu-direct共6+3×Channel数个uint64 launch参数，最多14 Channels；新版ccu为
  9+3×Channel数，最多13 Channels。两者受旧SDK48参数限制。
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

要求route peer ABI>=5且`A5CcuLocalRemoteAbiVersion()>=2`；旧两卡入口要求prepared
ABI>=16。检查通过只证明正确库已加载，不证明有卡数据正确。此预检会拒绝旧库。

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
GroupCopy8x32K组合模式，分析脚本才输出：

```text
PASS: combined grouped MS copy + peer WriteNb AllToAll; data, queued snapshots and lifecycle verified.
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
GroupCopy使用明确的CPU语义替身模拟字节位置，不模拟MS/Executor硬件；另有分批和尾块数学回归。
它不是设备并行模拟器，不验证目标固件资源、DMA/CCU性能或真实graph capture。

## 10. 单独测量 CCU LocalCopyNb 的设备时间

### 10.1 目的、更新与执行

用于定位组合模式2/4/6 relay均约80us：先测**单独4MiB本地复制**，而不是根据
“local-copy正确性PASS”或整卡HBM理论峰值推断它一定很快。
这只升级独立诊断程序/脚本，**拉取即可，不重编安装route包、不装DeepEP wheel、
不改系统HCCL或ko**。脚本现场编译独立程序到本次结果目录，使用现有CANN SDK。

```bash
(
set -e
cd /home/l00934901/sgl-kernel-npu
git stash push -m "backup-before-local-copy-perf-$(date +%Y%m%d_%H%M%S)"
BRANCH=feature/a5-ccu-copy-comm-overlap
git fetch origin "${BRANCH}"
git switch "${BRANCH}"
git merge --ff-only "origin/${BRANCH}"
source /usr/local/Ascend/cann-9.1.T560/set_env.sh
unset LD_PRELOAD A5_URMA_TP_TRACE_PREFIX ASCEND_LAUNCH_BLOCKING

bash scripts/run_a5_ccu_local_copy_perf.sh \
  --devices 2,3 \
  --sizes 4194304 --copy-mode direct \
  --warmup 100 --iters 20 \
  --changed-iters 3 --queued-iters 20 \
  --timeout-seconds 300 \
  --cann-root /usr/local/Ascend/cann-9.1.T560 \
  --output-root /home/l00934901/profiling
)
```

2、3分别执行自己的local-copy kernel，不互相发送payload。没有ChannelAcquire或
远端WriteNb；通信域初始化可能有控制面通信，但它在计时/profiling之前完成。
不需要relay卡。本测试默认有profiling，不要再在外面套msprof采集或另一种profiler。
源码为`examples/a5_ccu_urma_route_probe/testcase/local_copy_probe.cc`，沿用第8节
已通过的同一LocalCopyNb+WaitEvent kernel，不是用ACL D2D替代CCU复制。

### 10.2 明确的测量边界

```text
图外准备一次：comm -> thread -> kernel register/finish -> device buffers/tokens
  -> 改变输入/guard/全部queued快照正确性gate
  -> 准备本次输入（H2D）
  -> 100次独立CCU LocalCopyNb+WaitEvent，循环外同步
  -> 再毒化output（H2D），同步                    # 防止warmup结果冒充正式结果
  -> aclprofStart
  -> 20次独立CCU LocalCopyNb+WaitEvent，循环外同步 # 正式只有CCU launch
  -> aclprofStop/Finalize
  -> D2H回读input/output，检查数据和guard
  -> 清理资源并退出
```

输入/输出地址相同，重复注册次数为1；正式阶段没有malloc、H2D、D2H、ACL D2D、
输出毒化或每次host synchronize。correctness gate中有H2D/D2H快照，但不进入正式
计时/profiler。warmup与正式是100+20个独立kernel调用，不是一个kernel内循环120次。
固定输入仅用于稳态计时，前面的变化输入队列gate负责排除旧值/丢调用等错误。

只采当前rank设备：通过当前device调用标准aclprof API。每个rank有独立PROF目录。
采集配置为`ACL_PROF_TASK_TIME | ACL_PROF_ACL_API`，只在正式阶段开启；依据
[官方profiling配置API](https://www.hiascend.com/doc_center/source/en/canncommercial/800/apiref/appdevgapi/aclcppdevg_03_1610.html)。
硬件是否实际上报完整CCU事件仍由本次导出内容确认，构图/编译成功不算设备计时证据。

### 10.3 看结果与MindStudio文件

脚本自动调用SDK的`msprof --export=on`导出，然后生成性能报告。
这里是原生aclprof采集，**不是torch_npu的`*_ascend_pt`目录格式**：

```text
a5_ccu_local_copy_2_3_<时间>.<随机后缀>/
├── local_copy_report.md                  # 变化输入/guard/cleanup正确性
├── local_copy_perf_report.md             # host与device计时分别报告
├── local_copy_perf_summary.tsv
├── local_copy_device_samples.tsv         # 20个设备CCU duration及来源文件
└── cases/direct_4194304/
    ├── rank0.log / rank1.log
    └── profiling/
        ├── rank0/PROF_*/mindstudio_profiler_output/msprof_*.json
        └── rank1/PROF_*/mindstudio_profiler_output/msprof_*.json
```

将导出的`msprof_*.json`导入MindStudio timeline，看**设备层的Ccu任务**，不是
`HcclCcuKernelLaunch`/`rtCcuLaunch`的host API事件。具体导出层级以SDK输出为准。

```bash
RUN_DIR=$(ls -dt /home/l00934901/profiling/a5_ccu_local_copy_* 2>/dev/null | head -1)
if test -n "${RUN_DIR}" && test -d "${RUN_DIR}"; then
  python3 scripts/analyze_a5_ccu_local_copy_perf.py --run-dir "${RUN_DIR}"
  column -s $'\t' -t "${RUN_DIR}/local_copy_perf_summary.tsv"
  sed -n '1,220p' "${RUN_DIR}/local_copy_perf_report.md"
  find "${RUN_DIR}/cases" -type f \( -name 'msprof*.json' -o -name 'task_time*.csv' -o -name 'export.log' \) -print
fi
```

分析器优先读取明确us单位的`task_time*.csv`中CCU硬件任务，或完整timeline的Ccu
事件；排除host API、去重重复导出。**每rank必须恰好20个CCU任务且没有正式memcpy
任务**，否则不接受设备时间：不猜字段单位、不截取“最后20个”掩盖采集边界错误。
也要求正确性gate、正式后回读及进程清理通过。

报告`DEVICE_TIMING_CAPTURED`表示计时数据完整，不是“性能满足理论”的PASS。
`INCOMPLETE_OR_FAIL`且正确性PASS时，先检查各rank的`export.log`、原始PROF目录和
任务字段/数量，不能把host均值作为设备复制时间。如果没有msprof工具或用了
`--no-export`，保留原始PROF，可在Toolkit环境按
[官方导出命令](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/devaids/Profiling/atlasprofiling_16_0021.html)
解析，再重新分析。也可用`--no-profile`只保留host批时间，但这不能验证设备80us猜测。

### 10.4 判读及可选尺寸对照

重点给出每rank的**CCU min/median/mean/max us**，同时注明大小和copy_mode：

- 单独4MiB CCU local-copy也约80us：支持本地复制成为固定瓶颈的解释，再研究并行
  GroupCopy/CcuBuffer流水化。仍不能凭这个测试就断言组合时没有资源竞争。
- 单独local-copy很快，而组合约80us：重点转向local/remote资源争用、提交顺序和peer
  等待。空Channel独立测试不是“与组合相同活跃资源条件”的性能保证。
- 只有host批均值大，设备CCU任务很短：优先检查host提交/排队，不归因于复制硬件。

整卡HBM带宽是聚合能力，单条CCU本地复制未必用到全部通道/并行资源。报告payload
GB/s以真实配置的bytes计算；“read+write GB/s”只是2×bytes的名义流量换算，不是
采集到的HBM计数。**忽略profiling中`size(Byte)=4294967295`之类的哨兵值。**

可选测1/4/8MiB，减少队列正确性快照数量以限制pinned内存，各尺寸仍各自100+20：

```bash
bash scripts/run_a5_ccu_local_copy_perf.sh \
  --devices 2,3 --sizes 1048576,4194304,8388608 --copy-mode direct \
  --queued-iters 3 --warmup 100 --iters 20 \
  --timeout-seconds 300 --output-root /home/l00934901/profiling
```

`chunked`仍是每4KiB后Wait的串行控制，不等于原生GroupCopy，不能当作高性能优化。
本次无卡开发验证只包含独立程序编译/构图、计时边界及合成CSV/timeline解析回归，
**没有测得或预设真实NPU复制耗时**。

## 11. 按原生 HCCL 改造 self copy：GroupCopy

### 11.1 为什么改，具体借鉴了哪里

已反馈4MiB整块LocalCopyNb约65–66us，ACL D2D约6–7us。按单份payload换算，
分别约64GB/s与599–699GB/s。这说明在该配置下两种实现的有效吞吐相差约一个数量级，
**不是卡内复制天然就要65us**；但单凭耗时不能断定是某条固件规则或某个内存通道造成。
CCU local-memory指令与ACL D2D的搬运方式不同，也不能直接拿整卡HBM聚合带宽作为其保证。

原生HCCL并不是用我们原来的单条大长度LocalCopyNb做self copy。参考：

- `hccl/src/ops/all_to_all_v/template/ccu/kernel/ccu_kernel_all_to_all_mesh1d_multi_jetty.cc`：
  先提交RemoteWrite，再GroupCopy，之后等待远端事件并PostSync。
- `hccl/src/ops/all_to_all_v/template/ccu/kernel/ccu_kernel_alg_base.cc`：
  `CreateMultiOpCopy / GroupCopy` 用LoopGroup、MS缓冲和多执行器并行复制。
- `hccl/src/ops/all_to_all_v/template/ccu/kernel/ccu_kernel_utils.cc`：
  完整批、尾块与Loop/Parallel/Offset参数编码。

当前仓的原生代码含较新接口，不能直接照搬其缺失的C primitives。
我们用目标CANN已具备的旧对象式SDK重建同一结构；也核对了开发容器旧版
libhccl中GroupCopy的调用结构和三个参数编码函数。**不继承私有HCCL实现类、
不链接新C primitives、不修改libhccl/MUE/UBUS路由表。**

### 11.2 新数据面顺序及正确性

```text
注册一次：64个连续MS（8×8）+ 8个完成事件 + 8个执行器 + 两个LoopBlock

每次CCU执行：交换各Channel远端地址/token
    -> 所有显式direct/relay Channel提交WriteNb
    -> GroupCopy本地self row
         8路LoopGroup，每路每轮32KiB
         HBM -> MS -> WaitEvent -> MS -> HBM -> WaitEvent
         完整256KiB批 + residual-first尾块
    -> 等待各远端WriteNb事件
    -> 每Channel completion record/wait
    -> kernel完成，后续才允许消费/覆写tensor
```

4MiB恰好是16批×8路×32KiB。它不是每4KiB host发一次，也不是每4KiB同步一次。
LoopGroup在CCU图内运行，host仍只launch一次；组内读写Wait确保MS数据就绪和复用安全。
资源使用连续块，MS/event按lane偏移，不能以不连续的单个资源代替。
在原生顺序下远端写已提交，组内等待不会把远端提交拖到本地复制之后。
能否实际并行及其性能仍要设备测量，不用指令顺序替代硬件证据。

`send[r] -> recv[r]`与所有remote行互不重叠；源不被修改。保留原有输入生产依赖、
非别名检查、元素对齐分片、token、per-channel尾握手、图和tensor寿命检查。
完整批/尾块使用不同kernel参数，但不重新建Channel或注册资源。

实现位置：

| 内容 | sgl-kernel-npu文件 |
|---|---|
| 分批与尾块、编码 | `examples/a5_ccu_urma_route_probe/common/group_copy_plan.h` |
| 旧SDK GroupCopy图 | `examples/a5_ccu_urma_route_probe/op_kernel_ccu/grouped_local_copy_kernel.cc` |
| 与所有remote WriteNb组合 | `examples/a5_ccu_urma_route_probe/op_kernel_ccu/route_kernel.cc` |
| peer-plan准备/调用 | `examples/a5_ccu_urma_route_probe/op_host/peer_plan_runtime.cc` |
| 两卡prepared-plan准备/调用 | `examples/a5_ccu_urma_route_probe/op_host/all_to_all_multiroute.cc` |
| 孤立复制正确性/prof | `examples/a5_ccu_urma_route_probe/testcase/local_copy_probe.cc` |

当前旧接口launch最多48个参数。group前缀9项，每Channel3项，所以当前实现最多
13个Channel（两卡direct+6relay为7个；四卡每peer direct+1relay为6个，均满足）。
self row最多2GiB，是此版循环编码边界，不是“硬件只能复制2GiB”。非法容量在launch前拒绝；
未来更多peer/路径要扩展参数布局，不能静默丢路径。mode在图外bind时固定，切mode必须重建plan。

### 11.3 先测孤立GroupCopy，不需要安装wheel或route SO

先按第3节拉取本分支。该独立程序自行编译，仅使用当前系统CANN；不会覆盖生产SO。
先做变化输入、guard、queued快照正确性测试，覆盖整批和尾块：

```bash
cd /home/l00934901/sgl-kernel-npu
source /usr/local/Ascend/cann-9.1.T560/set_env.sh
unset LD_PRELOAD A5_URMA_TP_TRACE_PREFIX

bash scripts/run_a5_ccu_local_copy_probe.sh \
  --devices 2,3 --copy-mode group \
  --sizes 4096,32768,33024,262144,262400,4194304 \
  --changed-iters 3 --queued-iters 20 \
  --timeout-seconds 300 --output-root /home/l00934901/profiling
```

所有case正确性PASS、guard/source未改变、正常退出才继续。`--builder-only`只是构图，
不是设备PASS。整体通过时输出`PASS: GroupCopy (HBM-MS-HBM, all requested slices) completed on device`。
再只测4MiB，100次warmup、20次正式，只采正式20：

```bash
bash scripts/run_a5_ccu_local_copy_perf.sh \
  --devices 2,3 --copy-mode group --sizes 4194304 \
  --warmup 100 --iters 20 --timeout-seconds 300 \
  --output-root /home/l00934901/profiling
```

查看第10.3节的`local_copy_perf_summary.tsv`和MindStudio文件。本次case名为
`group_s32k_4194304`（旧结果目录为`group_4194304`）。每rank必须有20个设备CCU任务、无正式memcpy，比较min/median/mean/max，
不要用host均值或profiler size哨兵当作复制性能。旧控制仍可用`--copy-mode direct`重测。
新版group不会回退到ACL D2D或整块LocalCopyNb；失败就保留日志停止，不能报性能成功。

### 11.4 组合算子：构建安装及快速正确性

第3节完整命令已包含route package编译安装与runtime-check；**组合算子必须安装新版**：
prepared ABI16、peer ABI5、local+remote ABI2。已安装多dtype DeepEP wheel无需重装。
不需要拉取/编译修改版libhccl，只使用HCCL仓作自定义package构建框架。

```bash
python3 tests/python/deepep/test_a5_ccu_peer_plan_alltoall.py \
  --runtime-check --self-copy-mode ccu

bash scripts/run_a5_ccu_peer_plan_alltoall.sh \
  --devices 2,3 --relay-phys 0,1 --dtype float32 --self-copy-mode ccu \
  --sync-mode batch --graph-backend none \
  --bytes 4194304 --warmup 1 --iters 3 --repeats 1 \
  --output-root /home/l00934901/profiling

bash scripts/run_a5_ccu_peer_plan_alltoall.sh \
  --devices 0,1,2,3 --relay-map docs/topology/a5_peer_plan_4rank_example.json \
  --dtype float32 --self-copy-mode ccu --sync-mode batch --graph-backend none \
  --bytes 4194304 --warmup 1 --iters 3 --repeats 1 \
  --output-root /home/l00934901/profiling
```

报告须complete=True；每rank日志应有`CCU_LOCAL_REMOTE_PLAN ... self_copy=GroupCopy8x32K`。
`PASS: combined grouped MS copy + peer WriteNb AllToAll`表示数据、队列快照和生命周期通过，
**不是性能PASS**。分析器拒绝以旧`LocalCopyNb`marker冒充新版group结果。
尚未证明新kernel的ACLGraph时，不能沿用旧kernel的capture/replay PASS。
可单独用`--sync-mode per-call --graph-backend aclgraph`做功能验证，性能仍用batch/none。
五种dtype、direct-only也应按第9节矩阵分别检查。

### 11.5 两卡2/4/6relay与四卡1relay的性能命令

仅在通信卡和明确指定的外部relay允许使用时执行。所有命令只跑一轮，
warmup100、正式20、同步在循环外、profiling只采正式阶段；direct:每条relay=2:1。

```bash
for relays in 0,1 0,1,4,5 0,1,4,5,6,7; do
  bash scripts/run_a5_ccu_peer_plan_alltoall.sh \
    --devices 2,3 --relay-phys "${relays}" --dtype float32 --self-copy-mode ccu \
    --sync-mode batch --graph-backend none \
    --bytes 4194304 --warmup 100 --iters 20 --repeats 1 --profile \
    --output-root /home/l00934901/profiling || break
done

bash scripts/run_a5_ccu_peer_plan_alltoall.sh \
  --devices 0,1,2,3 --relay-map docs/topology/a5_peer_plan_4rank_example.json \
  --dtype float32 --self-copy-mode ccu --sync-mode batch --graph-backend none \
  --bytes 4194304 --warmup 100 --iters 20 --repeats 1 --profile \
  --output-root /home/l00934901/profiling
```

同一组卡/路径/大小做对照，仅改self-copy-mode：

| mode | 本地复制实现 | 用途 |
|---|---|---|
| `ccu` | GroupCopy8x32K，同CCU kernel内 | 本次原生风格优化 |
| `ccu-direct` | 整块LocalCopyNb+WaitEvent，同CCU kernel内 | 65–66us旧实现对照 |
| `serial` | ACL D2D后远端CCU，caller stream | 6–7us D2D及总时间对照 |
| `overlap` | 独立D2D stream与远端CCU fork/join | 原重叠方案，可能有peer等待偏斜 |

`ccu`与`ccu-direct`的CCU任务包含本地复制；`serial/overlap`的CCU任务不包含D2D。
应分别比较本地诊断、设备CCU任务及完整算子时间，不能只比不等价的CCU duration。
四卡self4MiB、发送12MiB、接收12MiB，与两卡资源争用不同；若group孤立快但组合仍慢，
再查活跃remote竞争/peer等待，而不是直接认定新增relay没有作用。

开发验证：无卡容器完成自定义package编译（ABI16/5）、direct/chunked/group构图，
旧libhccl编码函数数值比对、58项CPU回归（含2/4rank、4MiB/尾块/guard/容量）。
CPU模型不模拟MS硬件调度，**新group的设备正确性和吞吐尚待上述有卡实验确认**；
目前不承诺达到D2D的6–7us或恢复任何理论带宽。

## 12. 独立 local-copy：更大的 slice 性能对照

本节只改变独立诊断程序的GroupCopy参数：**8路固定，slice可选32/64/128KiB**。
生产两/四卡算子的`--self-copy-mode ccu`仍默认8×32KiB；先获得独立性能结果再决定是否推广。
没有改变任何direct/relay EID、权重、通知协议或系统HCCL。

| slice | 每路MS数/偏移 | 总MS数 | 4MiB每路循环次数 |
|---:|---:|---:|---:|
| 32KiB（对照） | 8 | 64 | 16 |
| 64KiB | 16 | 128 | 8 |
| 128KiB | 32 | 256 | 4 |

更大的slice可能减少循环/等待成本，但也需要更多连续MS；不保证更快。
代码同步调整MS块容量、lane偏移、内存步长、bulk/tail参数及kernel signature，
不是只把length改大而让相邻lane共享缓冲。每档仍有独立读/写Wait，不能删掉它们。
若目标SDK/设备拒绝该资源配置，报告FAIL，不回退32KiB，不扩大为任意资源请求。

### 12.1 拉取代码；本实验无需编译安装算子包或wheel

推送后在有卡容器中执行：

```bash
(
set -eo pipefail
cd /home/l00934901/sgl-kernel-npu
BRANCH=feature/a5-ccu-copy-comm-overlap
git fetch origin "${BRANCH}"
git switch "${BRANCH}"
git merge --ff-only "origin/${BRANCH}"
git log -1 --oneline
)
```

若本地修改阻止更新，按第3节备份处理，不要删除整个仓。
**这里不用执行第3节的route package构建安装，也不用装DeepEP wheel或改libhccl。**
第12.2节脚本会自行编译standalone可执行文件到本次结果目录，需要当前CANN头文件、
编译器及旧对象式CCU SDK；构建失败会显示`build.log`末尾，不会安装/覆盖生产SO。
子shell错误不会主动退出外层docker交互shell。

### 12.2 一条命令测32/64/128KiB：warmup100，正式20，只采正式阶段

```bash
cd /home/l00934901/sgl-kernel-npu
source /usr/local/Ascend/cann-9.1.T560/set_env.sh
unset LD_PRELOAD A5_URMA_TP_TRACE_PREFIX

bash scripts/run_a5_ccu_local_copy_perf.sh \
  --devices 2,3 --copy-mode group --group-slices-kib 32,64,128 \
  --sizes 4194304 --warmup 100 --iters 20 \
  --timeout-seconds 300 --output-root /home/l00934901/profiling
```

每档各自初始化一次、注册该geometry的kernel、变化输入/guard及20次排队快照正确性gate，
再warmup100，随后只采正式20。两侧是各自本地复制，无remote WriteNb或relay进程。
warmup和正式循环都只在循环外同步；正式区域无H2D/D2H/D2D/malloc。
默认依次跑三档，不跑三轮；每档两rank各20个设备CCU任务。
输出`PASS: GroupCopy ... all requested slices`说明所有请求档位的数据与清理通过，
性能报告`DEVICE_TIMING_CAPTURED`说明采集完整；它们都不自动代表性能提升。

若只想先测64KiB，将`--group-slices-kib`改为`64`；只测128KiB则改为`128`。
32KiB应同一次对照测量，避免把其他时间段/负载的30us当作唯一基准。

### 12.3 查看分析和MindStudio文件

```bash
RUN_DIR=$(ls -dt /home/l00934901/profiling/a5_ccu_local_copy_* 2>/dev/null | head -1)
if test -n "${RUN_DIR}" && test -d "${RUN_DIR}"; then
  column -s $'\t' -t "${RUN_DIR}/local_copy_summary.tsv"
  column -s $'\t' -t "${RUN_DIR}/local_copy_perf_summary.tsv"
  sed -n '1,240p' "${RUN_DIR}/local_copy_perf_report.md"
  grep -HnE 'LOCAL_COPY_GEOMETRY|LOCAL_COPY_FAILURE' "${RUN_DIR}"/cases/*/rank*.log
  find "${RUN_DIR}/cases" -type f -name 'msprof*.json' -print
fi
```

三个case分别是`group_s32k_4194304`、`group_s64k_4194304`、`group_s128k_4194304`。
MindStudio导入每个case下`profiling/rank*/PROF_*/.../msprof_*.json`，不是torch的ascend_pt格式。
每rank比较设备CCU的min/median/mean/max，重点看64/128KiB是否稳定低于同次32KiB。
报告有`group_slice_bytes`/`group_lanes`，并核对worker JSON实际参数；旧32KiB日志缺少
新geometry字段时不能冒充64/128KiB采集。不要用host平均值替代设备复制耗时。

失败时保留本case的`rank*.log`和`build.log`。资源注册失败仅说明该配置不可用，
不等于“CCU不能本地复制”。某档FAIL时仍保留其他档的报告，不reset公共设备。

### 12.4 推广到组合算子前，再补尾块正确性

若某档4MiB明显更快，先用相同geometry测非整批/非整slice：

```bash
bash scripts/run_a5_ccu_local_copy_probe.sh \
  --devices 2,3 --copy-mode group --group-slices-kib 64,128 \
  --sizes 65792,131328,1048832 \
  --changed-iters 3 --queued-iters 20 --timeout-seconds 300 \
  --output-root /home/l00934901/profiling
```

只有独立测试正确且更快后，才考虑升级生产kernel和组合算子资源配置。
本次不改生产默认，不需要更新installed route ABI；尚未证明64/128KiB在remote活跃时
也能更快，或保证更多MS不与通信资源竞争。

开发验证：无卡容器完成三档standalone编译/构图、60项CPU回归；包括各geometry的
bulk/tail覆盖、2GiB边界、默认32KiB生产布局和错误geometry拒绝。
**构图PASS不是设备正确性或性能PASS**，最终以本节有卡结果为准。
