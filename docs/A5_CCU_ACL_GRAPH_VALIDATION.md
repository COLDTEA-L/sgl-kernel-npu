# CCU + URMA 显式 L1 EID：计算图捕获与重放验证

适用分支：`feature/a5-ccu-single-port-l1-alltoall`，生产基线 `86acba6`；两卡探针也可单独移植本诊断。目标环境 CANN 9.3.0 / Ascend950DT。

本文的“图”是 **ACL Graph 计算图**，不是 HCCL rank graph，也不是 CCU kernel 指令构图。
现场已经验证 rank graph 未列出的真实单口 EID 可以建链、注册、eager 通信；最新现场 G1 已验证其单流固定参数捕获与重放，G2 多流仍待定位，见 §7–11。
之前模型采用 `--enforce-eager`，不能据此宣称已支持计算图。

**交付范围：本文是给现场 DeepSeek 的诊断实施与测试说明，仅更新文档。仓库基线 launcher 没有 `--suite graph`；现场已在 2dd863a 上增加未提交诊断实现，须保存该补丁。§2–3 说明该入口的实现契约；§7–11 是本轮后续步骤。不要把现场未提交功能当作拉取后已自带的功能。**
不修改生产选路算法、EID、拓扑或 EP8 提交行为来绕过失败。

## 1. 先固定环境和 API

只使用本人获准的容器与空闲卡。两卡优先使用已经实测的物理 2/3 和原来的显式单口配置，保持两 rank 的执行顺序完全一致。保存 commit、诊断补丁、二进制 SHA、CANN/驱动版本、完整 EID、端口及 rootinfo 摘要。

```bash
export A5_CANN_PATH=/usr/local/Ascend/cann-9.3.0
export A5_DEVICES=2,3
export HCCL_BUFFSIZE=64
export A5_SINGLE_PAIR=/实际路径/已验证的single.json
export A5_TEMPLATE_INDEX=0  # 改为当前机器已验证的同 die graph 模板 index

python3 - <<'PY'
import os, re
from pathlib import Path
p = Path(os.environ['A5_CANN_PATH']) / 'include/acl/acl_rt.h'
s = p.read_text()
for name in ('aclmdlRICaptureBegin', 'aclmdlRICaptureEnd',
             'aclmdlRICaptureGetInfo', 'aclmdlRIExecuteAsync', 'aclmdlRIDestroy',
             'aclmdlRIDebugJsonPrint', 'aclmdlRICaptureTaskUpdateBegin',
             'aclmdlRICaptureTaskUpdateEnd'):
    m = re.search(r'[^;{}]*\b' + name + r'\s*\([^;]*;', s)
    print(name, ':', m.group(0).strip() if m else 'NOT FOUND in acl_rt.h')
PY
```

开发机 9.2 头文件存在 Begin/End/ExecuteAsync/Destroy，但这只证明 API 声明存在，不证明目标 9.3 的 CCU launch 可以被捕获。以现场头文件及库为准；缺符号则停止，不改成名字相近的旧接口。官方概念参考：[aclmdlRICaptureBegin](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/API/runtimeapi/aclcppdevg_03_1782.html)；该参考为 9.1 文档，不代替现场 9.3 支持性验证。

## 2. DeepSeek 先增加独立诊断入口

在 `examples/a5_ccu_l0_l1_probe/main.cc` 的现有 `Run()` 中复用已成功的发现、握手、建链、注册、`launch(mode)`、`fill(epoch)` 和 `verify(epoch)`。在初始化完成后分支进入 graph 测试。不要再写一套端点选择或 payload 布局。

修改 `scripts/a5_ccu_l0_l1_probe.py` 的参数校验、传参、汇总器，新增 `--suite graph`，继续使用已有 `--mode l0/l1/split`、`--explicit-eids`、`--bytes`、`--iters` 等参数。这是诊断新增接口，要求：

- 所有 channel、thread、kernel、固定 send/recv、mem token、事件在捕获前建立；保留现有输入输出 guard。
- 先跑原来的 eager 正确性与 warmup，然后完成同步；确认 communicator 没有在途工作，再开始捕获。
- 捕获范围只包含本次传输的提交及必要的设备事件依赖。文件握手 `Agree`、日志等待、H2D 填数、D2H 验数、设备同步、内存分配、建链和注册都放在范围外。
- Begin、launch、End、Execute、最终同步逐项记录返回码和 rank；失败标记落盘后退出该轮，launcher 有超时并终止本次测试的另一 rank。不要在失败通信域内接着试下一个模式。
- 每个进程的捕获图只属于自己的 device/context；两个 rank 捕获同一种模式、重放相同次数。捕获不应被当作一次有效 payload 执行，第一次验数必须在 Execute + sync 后。
- 保留原有 eager suite；graph suite 使用独立完成标记与严格汇总，不得让原来的 `L0L1_RANK_PASS` 代替 graph PASS。

建议标记：`L0L1_GRAPH_STAGE rank=... mode=... stage=... status=...`，stage 至少包含 `capture_begin/capture_launch/capture_end/replay_submit/replay_sync/verify/graph_destroy`。
单 rank 最终输出 `L0L1_GRAPH_RANK_PASS`；两个 rank 均完成所有 epoch、guard 校验及 clean exit 才输出 `L0L1_GRAPH_ALL_RANKS_PASS`。

### 2.1 G0：本地计算图基本能力

先独立做一个无通信的固定 device buffer 异步拷贝捕获/重放，验证 runtime、capture stream 和现场 API。更换输入内容后重放并逐元素验数。
G0 失败属于捕获环境或诊断程序问题，不能归因于 EID；G0 的实现不混入 CCU 的计时结果。

### 2.2 G1：分别捕获 L0 和显式单口 L1

分别独立进程运行 `mode=l0` 和 `mode=l1`。捕获流必须是对应 lane 的原 stream，因为 `HcclThreadAcquireWithStream` 已绑定它；不要临时新建另一个流仅包住 launch。

以下是应接入原 `Run()` 的核心伪代码，不是独立可编译程序：

```cpp
// 初始化、fill、Agree、warmup、sync 均在图外。
aclmdlRI graph = nullptr;
auto capture_stream = selected_lane.stream;
CHECK(aclmdlRICaptureBegin(capture_stream, ACL_MODEL_RI_CAPTURE_MODE_GLOBAL));
launch(mode);  // 只记录一轮该 mode 的 CCU 提交；里面无 sync
CHECK(aclmdlRICaptureEnd(capture_stream, &graph));
// 检查 graph 非空；如现场支持，导出图节点/任务信息。
for (unsigned epoch : {101U, 137U, 211U}) {
    fill(epoch);                 // 原地址上改数据，并清空 recv；图外
    Agree(o, unique_epoch_tag);   // 双方输入准备完成；图外
    CHECK(aclmdlRIExecuteAsync(graph, capture_stream));
    CHECK(aclrtSynchronizeStream(capture_stream));
    verify(epoch);               // payload 与 guard；图外
}
CHECK(aclmdlRIDestroy(graph));   // 完成所有 replay 后、销毁 comm/buffer/stream 前
```

按现场签名实现错误处理。捕获失败后走现场支持的终止捕获/清理流程；不能继续使用失效 graph，也不能把创建或销毁失败吞掉后报 PASS。
初测保持全局 capture mode；若失败，先查调用和 SDK 支持性，不要用 relaxed 模式掩盖违规调用。

更换 epoch、每次清空 recv 是必要条件：只重放相同输入，可能把 eager 预热遗留的正确输出误当作计算图执行结果。记录 capture 时的 launch 次数、replay 时的 host launch 次数；后者应为 0，只调用 graph Execute。

### 2.3 G2：split 必须捕获两条流

原探针 split 在 L0 stream 和 L1 stream 各提交一次。**只捕获 L0，然后在捕获区间调用原 `launch("split")`，不足以证明 L1 被捕获。** 要显式用设备事件使第二条流加入捕获，并在结束前汇回主流。

事件在图外预创建，现场确认所用事件类型支持捕获。逻辑顺序如下（把返回码检查和 stage 日志补全）：

```text
主流 = L0 lane.stream，辅流 = L1 lane.stream
CaptureBegin(主流)
  主流 record(fork_event)
  辅流 wait(fork_event)          # 辅流加入同一捕获
  launch("split")              # 两条流各提交自己的分片，不在这里同步
  辅流 record(join_event)
  主流 wait(join_event)          # 所有分支汇回捕获起始流
CaptureEnd(主流) -> 一张 graph
ExecuteAsync(graph, 主流)
同步主流 -> 验完整 payload 和两端 guard
```

在 fork 后用 `aclmdlRICaptureGetInfo` 检查两条流是否处于有效捕获、属于同一个 modelRI；若现场接口不支持查询，保留错误并用图任务导出等证据补足。两条流的事件交叉捕获也是待验证项，不能只因 End 成功就略过。

初测可以另做“两条 lane 分别捕获、分别 replay”的隔离实验，但它只能证明两条单路径各自可捕获，**不能代替 G2 的一张图和完整完成依赖**。

G1/G2 验数后各做一次端口计数差分：计数窗口只覆盖 replay，排除建链、捕获和 eager warmup。单口 L1 应命中原先实测端口；split 两类口各承担配置的字节份额。声明是否未在 graph 枚举到，依据保存的完整端点清单，不凭 EID 前缀猜测。

## 3. 两卡执行顺序与命令

**完成 §2 新增入口和汇总器后才执行本节 graph 命令。** 先检查 `--help` 已有 graph，确认没有误用旧二进制。仅诊断二进制需要重编，不需要重打 EP8 wheel 或 route。

```bash
export A5_L0L1_BINARY=/home/l00934901/profiling/ccu-l0-l1-graph-build/a5_ccu_l0_l1_probe
bash scripts/build_a5_ccu_l0_l1_probe.sh "$A5_CANN_PATH" \
  /home/l00934901/profiling/ccu-l0-l1-graph-build
python3 scripts/a5_ccu_l0_l1_probe.py --help

# 先原有 eager 正确性：两 rank clean PASS 后才继续。
bash scripts/run_a5_ccu_l0_l1_probe.sh --suite correctness --mode all \
  --l1-port-policy any --l1-index "$A5_TEMPLATE_INDEX" \
  --explicit-eids "$A5_SINGLE_PAIR" --bytes 65536

# G0 本地图验证后，G1/G2 各用新进程；任何失败立即停，不继续性能测量。
for A5_BYTES in 65536 8388608; do
  for A5_MODE in l0 l1 split; do
    bash scripts/run_a5_ccu_l0_l1_probe.sh --suite graph --mode "$A5_MODE" \
      --l1-port-policy any --l1-index "$A5_TEMPLATE_INDEX" \
      --explicit-eids "$A5_SINGLE_PAIR" --bytes "$A5_BYTES" \
      --warmup 20 --iters 100 --rounds 1 || exit 1
  done
done
```

新增 graph suite 应先执行 §2 的三个变数据 epoch，再执行 `iters` 次 replay 压测；每次 correctness replay 后同步验数。压测以固定参数连续重放，最后同步验数，另报计时，不把连续 100 次当 100 个独立样本。

64 KiB / 8 MiB 全过后，补 32 MiB 和 3 次独立进程对照（eager vs graph）。比较同 mode、同路径、同 bytes 的总 wall 和 enqueue；不把捕获构建时间计入稳态 replay，但单独报告捕获开销。

“失败立即停”指停止本次失败通信域上的后续压测和性能推断，不禁止按 §7–11 在新进程中做隔离诊断。

## 4. G3：再验证 EP8 的真实 host + READY/DONE 桥

两卡通过只说明裸 `HcommCcuKernelLaunch` 路径可捕获。EP8 还经过 tensor/host、token 查询、DFX、READY/DONE 事件桥，必须单独验证。

在现有 `tests/python/deepep/test_a5_ccu_eid_npu.py` 复用 harness 的同一套 manifest/topology/planner、对齐的 buffer 和 prepared 对象，另加诊断入口，**不是直接把现有 benchmark 当作图测试**。

1. 保持 caller stream 明确且固定；在该流上下文中创建 host、准备 plan、预热 execute，再同步。当前 `NpuEidRuntime::Check` 要求调用流等于创建时的流。不要让 torch_npu 的图上下文自动选择另一流后直接归因于 EID。
2. 按现场 torch_npu 版本核对 `NPUGraph` / `graph` API，明确指定同一 caller stream，捕获一次已有的 `host.execute(prepared, send, recv)`。不要在捕获内部调用 CPU planner、创建 tensor、D2H 或 HCCL barrier。
3. route 的 stream、READY/DONE 事件在捕获前已创建且存活。验证事件依赖确实把 CCU route stream 纳入同一张图，并在 caller stream 读输出之前完成；不能删除 READY/DONE 来“让捕获通过”。
4. 依次验证 D、实际有 L1 moved 的 M，先固定 C 和 FP8 行宽；然后单独为 `C.T + BF16` 建图验证。更换输入 epoch 后每次 replay 完成再验数、检查 guard。记录实际 L1 selector/moved，避免 M 实际没分流。
5. 图销毁在所有重放完成之后，host/handle/comm/stream/buffer 清理之前；不靠 Python 析构先后碰运气。

若两卡单路径通过而 EP8 失败，按 `host stream check -> token/DFX -> READY -> CCU launch -> DONE -> CaptureEnd -> replay` 定位首次失败；这时还不能说 rank graph 外的 EID 不能入图。

## 5. G4：动态 MoE 参数，单独列为第二阶段

固定图通过，不代表 dynamic MoE 可以直接重放。捕获时的 CPU 规划只运行一次，`g.replay()` 不会重新运行任意 Python/C++ host 代码。
改变 host 侧 counts、moved 或 task args 数组，也不等于已更新设备图里的任务。不得重放旧计划、跳过数值校验后称为 dynamic PASS。

第一阶段只改变原地址上的 payload 内容，保持 C、offset、bytes、selectors 不变。第二阶段再做：

- 准备 C0/C1 两份不同但容量合法的矩阵，用现有 planner 分别规划；记录预期 splits/offsets/分流变化。
- **先用分别捕获的两张固定图 G(C0)、G(C1)** 交替重放，验证固定图各自的语义。共用预分配容量时，每张图仍要按自己的布局验数。这是可控的图缓存实验，不代表任意 C 都有缓存命中。
- 如果要在同一图上更新 C，先确认当前 SDK 的任务更新 API 能更新 CCU launch 的参数和相关任务；实现更新并逐次检验 C0/C1/零流/不同 row_bytes。不能仅凭 API 名字存在宣布可用。
- 若没有可验证的参数更新方案，明确结论为“固定 prepared 计算图支持；动态规划留图外或按完整参数建图”，不要直接去掉模型 `--enforce-eager`。

端点/通道变更不属于普通 task args 更新。更换 EID、policy generation 或 communicator 时，先销毁旧图并按正常生命周期重建资源。

## 6. 如何判读与回报

| 结果 | 能说明什么 | 下一步 |
|---|---|---|
| G0 失败 | 本地捕获环境/API/诊断程序有问题 | 先修最小本地图，不查 EID |
| G0 通过，L0 与 L1 都失败 | CCU 公共提交或捕获链存在问题 | 查第一条 SDK 错误、线程流绑定和捕获限制 |
| L0 通过，显式单口 L1 失败 | 路径相关问题候选，尚不能证明是 graph 缺项 | 同卡同 die 补 rank graph 中聚合 L1 捕获对照，保存端点/任务信息 |
| L0/L1 单独通过，split 失败 | 多流加入、汇回或并发执行问题候选 | 检查两流 capture status/modelRI 和 fork/join |
| 两卡全过，EP8 失败 | EP8 wrapper/事件桥/提交链问题候选 | 按 §4 分段，保留最早失败位置 |
| End 成功，replay 不返回或输出旧数据 | 尚未证明任务被正确捕获执行 | 查图任务、completion、变 epoch 验数；不得报 PASS |
| 固定图全过，参数变化失败 | 动态参数更新不正确或不支持 | 按 §5 分离固定图与动态语义 |

回报最少包含：版本与诊断补丁；完整 launch 命令；两 rank 的路径/stream/thread/die/EID；每个 capture/replay 阶段返回码；三个 epoch 的逐元素与 guard 校验；重放次数；清理结果；图任务导出（支持时）；replay 专属端口增量。失败保留首次错误附近 SDK 日志及两 rank 最后阶段，不只贴最终 watchdog。

成功结论限定为：“在该 SDK/驱动、该显式端点及该 stream/任务结构下，固定参数计算图捕获和重放通过”。不能扩展为所有 rank graph 外 EID、所有 die、动态 MoE 或整个 vLLM 模型都已支持计算图。

## 7. G0–G2 现场结果与结论修正

本轮结果来自用户现场报告，非开发机代跑：CANN 9.3.0、物理 6/7、`2dd863a + 未提交 main.cc/launcher 诊断补丁`。
两端显式 L1 分别为 `000000000401020000100000df76c256` / `000000000401020000100000df76e256`，声明 die=0、port=1。

| 项目 | 已取得证据 | 目前能下的结论 |
|---|---|---|
| G0 本地单流 D2D | 三个变数据 epoch 验数通过 | 本地单流图可用 |
| G1 L0 / 显式 L1 | 64 KiB、8 MiB、32 MiB payload/guard 通过 | 两条路径分别可捕获和重放，显式 EID 非单流入图障碍 |
| G2 普通事件 / progress 等尝试 | Record 返回 207000 | 特定 API/事件对象/捕获状态组合不支持；尚缺首次失败的精确调用及事件创建函数 |
| G2 external / 无 fork-join global | API 成功但 L1 切片首字 word=8256 错 | split 整体未通过；不能用 API 返回 0 代替任务归属和数据校验 |

32 MiB eager 约 715 µs、单图 replay 约 679 µs；8 MiB 约 212–215 → 178 µs。这里只记现场观察，未完成严格配对重复，不承诺模型收益。

需修正三个判断：

1. **`ACL_EVENT_EXTERNAL` 不是把辅流纳入图的办法。** 官方说明这种事件用于图任务更新，不能用于跨流任务捕获；external 路径失败不是“所有跨流捕获都不支持”的证据。[官方任务更新说明](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/programug/acldevg/runtime_doc_dev_0032.html)
2. `GLOBAL` 是捕获期间的线程/API 限制模式，不会自动收集进程里所有 stream。记录 `status=1, modelRI 非空` 还不够，必须证明**两条流同时 ACTIVE 且 modelRI 相同**。官方给出的跨流方法仍是事件建立依赖并汇回主流。[官方跨流捕获说明](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/others/acldevg/runtime_doc_dev_0031.html)
3. 本轮物理 6/7 的已知 L0 拓扑位于 die=1，显式 L1 选的是 die=0。因此本轮 G2 很可能是**跨流又跨 IO die**，需要以本次 `L0L1_SELECTED` 的两 lane die 再确认。它不能直接否定生产 L1 分支限制的“同 IO die”配置。

上面官方参考为公开 9.0/9.1 资料，实际 9.3 支持性仍以现场测试及厂商确认为准。当前准确结论：**现有 G2 实现未通过；首要排查事件对象种类、辅流归属和同/跨 die，尚未证明 runtime 无法实现单图双流。**

## 8. P0：不占卡的代码与 API 核查

先保存现场未提交修改，再补诊断；不要为了拉文档 reset 或丢掉补丁。本轮不要求更换容器、SDK、拓扑或模型配置。

```bash
mkdir -p /home/l00934901/profiling/ccu-graph-followup
git rev-parse HEAD
git diff -- examples/a5_ccu_l0_l1_probe/main.cc scripts/a5_ccu_l0_l1_probe.py \
  > /home/l00934901/profiling/ccu-graph-followup/graph_diag_before_followup.patch

grep -nE 'CreateEvent|RecordEvent|StreamWaitEvent|CaptureGetInfo|CaptureBegin|CaptureEnd' \
  examples/a5_ccu_l0_l1_probe/main.cc
grep -nE 'CreateEventWithFlag|CreateEventExWithFlag|CaptureToModelRIBegin|CaptureGetInfo' \
  "$A5_CANN_PATH/include/acl/acl_rt.h"
```

为每个事件记录 **准确创建函数**（`aclrtCreateEvent`、`aclrtCreateEventWithFlag` 或 `aclrtCreateEventExWithFlag`）、flag 十六进制值、handle；为每次 Record/Wait 记录 fork/join、调用 API、stream、返回码、失败后的 `aclGetRecentErrMsg()`。不能把三种创建函数统称为“普通事件”，也不能把 fork 的失败写成 join 失败。

特别检查：

- 在独立诊断中优先验证 `aclrtCreateEventWithFlag(&event, ACL_EVENT_SYNC)` + 普通 Record/Wait；另用 `aclrtCreateEventExWithFlag(..., ACL_EVENT_SYNC)` 做一组仅改变创建函数的对照。当前 EP8 `op_host/thread_event_bridge.h` 使用后者，不能假设二者捕获行为相同，也不能未经验证就认定换前者能修复。
- 单独 `ACL_EVENT_CAPTURE_STREAM_PROGRESS` 不是带 SYNC 的跨流同步事件。官方 API 对 Wait 有 flag 限制，不把“仅 progress 创建”当作等价正对照。[事件 API 说明](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910beta2/API/runtimeapi/aclcppdevg_03_0080.html)
- 不再将 `ACL_EVENT_EXTERNAL` / `ACL_EVENT_RECORD_EXTERNAL` 当作自动加入同一图的修复。它们保留为已有失败记录即可，不继续扫 flag 组合。
- 确认本次实际加载的 ACL/runtime 库路径与 CANN 版本；可在程序内用 `dladdr` 记录上述 API 来源，保存进程 maps。不要只看编译头文件。

捕获状态统一输出：

```text
GRAPH_MEMBERSHIP point=after_fork rank=0
 main_stream=... main_status=... main_ri=...
 aux_stream=... aux_status=... aux_ri=... same_model=0/1
```

至少在 Begin 后、fork Wait 后、CCU launch 后、join 后查询两流。只有 `GetInfo` 返回成功时其输出才有效；同一进程内比较 RI，不能比较两 rank 的指针值。查询 capture 元数据不替换成同步流或查询完成事件。

## 9. P1–P3：最少的占卡隔离实验（按顺序）

### 9.1 P1：纯本地双流复制，先完全拿掉 CCU

这是 **G0 双流扩展**，只需要一张获准空闲卡，不初始化 HCCL、不获取 channel。固定四块 device buffer A/B/C/D，图外填 A=epoch pattern，清 B/C/D。预创建 main/aux 两条流和 fork/join 两个事件。

```text
CaptureBegin(main)
  main: async D2D A -> B
  main: Record(fork)
  aux : Wait(fork)
  查询 main/aux 的 capture status 与 RI，相同且 ACTIVE 才继续
  aux : async D2D B -> C
  aux : Record(join)
  main: Wait(join)
  main: async D2D C -> D
CaptureEnd(main)
```

按 §8 的 **WithFlag+SYNC / ExWithFlag+SYNC** 各用一个新进程；若之前已按完全相同 API、flag、依赖链测过，直接提供代码和逐点日志，不重复跑。
用三个不同 epoch 改 A、清 B/C/D 后 replay，只同步主流再校验 D 和所有 guard。这个串联复制设计同时检查辅流任务及 join 的真实完成依赖。

判读：

- 两个版本都在第一条 Record 就报 207000：已复现与 CCU/EID 无关的最小失败，先反馈 runtime；暂不跑八卡。
- WithFlag 通过、Ex 失败：事件创建接口是直接候选，先只改诊断 G2，不能直接批量替换生产事件桥。
- 两者都通过：跨流计算图基本可用，转查 CCU thread/stream、die 和提交实现。
- `same_model=0`：辅流未正确加入。停止该轮，不继续提交通信任务，以免 eager 辅流等待被捕获但未执行的主流造成挂起。

### 9.2 P2：同 die 的 L0 + 显式 L1 split

P1 成功后，只用其通过的事件配置，复用现场已有 G2 测试，先 64 KiB、三个 epoch，不先跑性能。

| 对照 | 卡和路径选择 | 用途 |
|---|---|---|
| P2-same | 6/7 的 L0 die=1 + 从真实库存选择的 L1 die=1 单口；或空闲时用之前验证的 2/3 两路 die=0 | 去掉跨 die 变量 |
| P2-cross | 保留本轮 6/7、L0 die=1、L1 die=0 原配置 | 与现象直接对照 |

每轮以 `L0L1_SELECTED` 确认实际两端 die；为新的 L1 配置重选同 die 的模板。**不能把现有 0401 的字节改成另一个前缀来造 EID**，必须查当前机器驱动库存、rootinfo plane、完整地址并先 eager 验证。没有可验证候选就报告缺项。

执行复用现场已实现的命令，输入必须替换为该次配对的实际配置：

```bash
# 先导出已验证的 A5_DEVICES、A5_TEMPLATE_INDEX、A5_SINGLE_PAIR 和新诊断二进制路径。
bash scripts/run_a5_ccu_l0_l1_probe.sh --suite correctness --mode all \
  --l1-port-policy any --l1-index "$A5_TEMPLATE_INDEX" \
  --explicit-eids "$A5_SINGLE_PAIR" --bytes 65536
# 仅 eager PASS 才执行；事件版本及 same/cross die 标签写进结果元数据。
bash scripts/run_a5_ccu_l0_l1_probe.sh --suite graph --mode split \
  --l1-port-policy any --l1-index "$A5_TEMPLATE_INDEX" \
  --explicit-eids "$A5_SINGLE_PAIR" --bytes 65536 \
  --warmup 3 --iters 3 --rounds 1
```

以上不引入新的 CLI flag；新增事件版本和 P1 可以用单独诊断二进制区分，必须保存 SHA/补丁。不要原样运行未补 P0 日志的旧 G2 再只返回“失败”。

若 same 通过而 cross 失败，结论限定为本配置跨 die 的捕获/提交问题；不能写成所有双流不支持，也不扩展生产 EP8 的同 die 约束。若两者都失败而 P1 通过，记录首次从同 RI 变成不同 RI 或首次 SDK 错误发生在 CCU launch 前还是后。

### 9.3 P3：分开验证“同时执行”和“一张图”

仅在 P2 仍失败时做隔离，不作为生产替代的默认选择：

1. **两张独立图**：L0、L1 各捕获自己的 split 分片（不是各自发送全量），重放时先提交 G_L0 和 G_L1，再等待两者完成，校验完整输出与 guard。初始化/预热也要双方按同样顺序完成。通过只能证明双图组合可用，不证明单图支持，也不默认等于 vLLM 的一张图。
2. **同一流顺序提交**（可选）：只在 same-die 且 SDK 明确允许的情况下，初始化时把所需 CCU thread 绑定到同一 stream，捕获两次顺序 launch；不能只改 capture stream 而沿用原来的 thread。此对照改变并行性，只用于判断是多流成员关系还是多任务本身失败，不能把串行通过当作 split 性能修复。

每项都是新进程、先 eager、再三 epoch graph 正确性。任何通信挂起后结束本次进程组，不在原上下文续测。P1–P3 未正确前停止 8/32 MiB 扫描和长压测。

## 10. HCCL 实现线索与 EP8 后续边界

本地 HCCL 参考源码（不保证与现场 9.3 二进制同版）提供了一条重要线索：

```text
src/legacy/ascend950/framework/resource_manager/stream/stream_manager.cpp
  StreamManager::CaptureSlaveStream
    -> 查询主流是否 capture
    -> AddStreamToModel(slave, model)
src/legacy/ascend950/common/stream_utils.cc
  AddStreamToModel -> rtStreamAddToModel
```

说明 HCCL 某些路径会**显式登记辅流到捕获模型**，而不只是把 eager 的事件桥原样搬进图。当前 EP8 `ThreadEventBridge` 只有 Ex+SYNC 的 Record/Wait，没有这段捕获专用登记。这是后续调查方向，不是已证实根因。

若 P1 成功而 P2 仍失败，将上面线索和最小复现交给 CANN 确认：9.3 CCU thread 绑定流是否需要额外登记、对应受支持的公开 API 是什么、完成依赖如何建立。可只读查找现场声明，但**不能自行 dlsym 猜私有 ABI、硬写函数原型调用内部 rt 接口**。
也不要把 `aclmdlRIBindStream` 的显式 build 流程或 `aclmdlRICaptureToModelRIBegin` 按名字当成可互换替代；须先核实该版本语义与支持范围。

G3 不是宣布失败，而是先暂缓八卡。只有同 die 的局部多流实验能捕获并完成、或厂商确认另一条受支持路径后，才按 §4 验证真实 EP8 caller/route stream。即使发现事件接口修复有效，也先在诊断构建验证，再评估生产改动。G4 动态参数继续后置。

## 11. DeepSeek 本轮执行清单与回报表

本轮优先顺序：**P0 代码/加载库/事件记录 → P1 纯本地双流 → P2 同 die CCU split → 必要时 P2-cross / P3**。不需要重复已通过的三档 G1 性能，不需要模型长测。

回报表每行一个新进程实验：

```text
test, commit+patch_sha, binary_sha, device,
L0_die, L1_die, event_create_api, event_flag,
first_failed_api, stage, status, recent_error,
main/aux capture_status 与 modelRI（各采样点），
epoch_101/137/211_correct, guards_ok, clean_exit, result_dir
```

验数失败追加：word index、字节偏移、期望值、实际值、是否仍为清空 sentinel、所属 L0/L1 分片、图中任务/stream 导出。`word=8256` 要按实际元素宽度和 guard 算偏移，不能仅凭编号断言必定没捕获（也可能是错地址、错参数或完成等待缺失）。

**本轮通过标准是同一张图中两条流任务都存在、三个新 epoch 数据和 guard 正确、只等主流即可保证消费者可读。** 单独 `CaptureEnd=0`、某一条流 `ACTIVE`、或两张单图各自 PASS 都不能替代这个标准。
