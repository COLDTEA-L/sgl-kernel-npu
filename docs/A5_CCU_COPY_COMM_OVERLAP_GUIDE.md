# 显式多路径 AllToAll：本地 copy 与远端 CCU 并行调度

分支 `feature/a5-ccu-copy-comm-overlap`，基于多类型分支 `2bb77ec`。
支持两卡 prepared-plan、两/四卡 peer-plan、FP16/BF16/FP32/INT8/INT32。
默认 `--self-copy-mode overlap`；`serial` 保留原本本地copy后再CCU的对照。
不是恢复旧组合CCU kernel，不修改系统HCCL、不加载ko、不改变UB路由表。

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
source /usr/local/Ascend/cann-9.1.T560/set_env.sh
unset LD_PRELOAD A5_CCU_PREPARED_ALLOW_LAZY_STREAM A5_URMA_TP_TRACE_PREFIX

HCCL_REPO=/home/l00934901/hccl \
bash scripts/build_a5_ccu_urma_route_probe.sh \
  --install --install-path /usr/local/Ascend/cann-9.1.T560

python3 tests/python/deepep/test_a5_ccu_peer_plan_alltoall.py --runtime-check
)
```

需要重编安装route package。若已装多类型版DeepEP(peer ABI>=2、attr ABI>=12)，
**无需重编DeepEP/wheel**；如果仍是旧FP32/旧四卡wheel，按多类型指南先升级wheel。
不编译/替换系统libhccl。当前route版本要求：prepared ABI>=14、peer ABI>=3、
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
