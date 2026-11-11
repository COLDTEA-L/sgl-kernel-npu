# CCU + URMA 显式 L1 EID：计算图捕获与重放验证

适用分支：`feature/a5-ccu-single-port-l1-alltoall`，生产基线 `86acba6`；两卡探针也可单独移植本诊断。目标环境 CANN 9.3.0 / Ascend950DT。

本文的“图”是 **ACL Graph 计算图**，不是 HCCL rank graph，也不是 CCU kernel 指令构图。
现场已经验证 rank graph 未列出的真实单口 EID 可以建链、注册、eager 通信。2026-11-11 更新：现场回报 G1/G2/G3 通过，G4 两张固定图交替通过；同一张图更新动态 C 尚未通过，见 §12–15。§7–11 保留为早期 G2 失败的排查记录，不再是当前阻塞结论。
之前模型采用 `--enforce-eager`，不能据此宣称已支持计算图。

**交付范围：本文是给现场 DeepSeek 的诊断实施与测试说明，仅更新文档。仓库基线 launcher 没有 `--suite graph`；现场已在 2dd863a 上增加未提交诊断实现，须保存该补丁。§2–3 说明该入口的实现契约；当前后续步骤在 §14–15。不要把现场未提交功能当作拉取后已自带的功能。**
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

## 12. 2026-11-11 更新：G3 计时修正与 G4 能力边界

以现场最新回报为准：G3 的 single_hot/uniform/competing、BF16/FP8、direct/multipath、C 转置等六组固定 prepared 捕获、变 epoch 重放均通过。现场还回报 G2 已通过，须保留最终事件实现补丁、流/die 配置，不能只保存最早失败的版本。

此前“graph 比 eager 慢三倍”的结论撤回。旧测试没有 replay 预热、计时前没有跨 rank 对齐；rank0 的验数结束比其他 rank 晚约 27 ms，其他 rank 的计时包含等待。修正为：

```text
图外 setup → replay warmup → device sync → 所有 rank barrier → device sync
→ t0 → 连续 50 次 replay → t_enqueue → device sync → t_end
```

修正后的 single_hot/BF16（row_bytes=8192）graph 各 rank 为 359.3–360.7 µs，另次 eager 为 347.3 µs；约 3.7% 差异仍须同进程边界、交替顺序、独立进程重复确认，不能认定为内核退化。该行宽也不能替代模型 BF16 的 10240。计时前 barrier 在图外，不能每次 replay 插入 barrier 改变测量口径。

G4-2a 同 host、非默认 caller stream、两张分别按 C0/C1 捕获的固定图交替通过，只证明“分别捕获的两种布局可重放”。G4-2b 给 `NPUGraph.update` 传回调失败，只说明该调用不符合接口契约；**不证明所有 task update 都不可用，也不证明已有动态 CCU 更新能力。**

## 13. DS4.1 原生 prefill / CPU splits / 计算图：源码核对

### 13.1 版本与结论范围

本次实际读取服务器 `/home/liuyuanwen/vllm-ascend-dsv41`（另一个 `/home/liuyuanwen/vllm-ascend` 同 HEAD），提交 **`56debc194a42d473f8c456d4850369442d694c8f`**。下面行号均针对该提交；未改这个模型仓库。现场曾报告 `f171c530` 加接入补丁，该对象不在本次可读仓库中，所以不能把当前源码当成现场已安装版本的逐行证明。先执行 §14.1，确认实际加载代码。

**查到的原生 AllToAllV 路径没有“捕获 C0 后自动更新 CPU splits”的实现。** DS4.1 模型代码明确把 A5 packaged cache 的支持范围写为 eager prefill 和 full-graph decode；`requires_uncompiled_fallback` 让 runtime NONE 绕过 compiled model。这不等于任何配置下 prefill 永远不能编译，而是当前已有实现与验证边界。

[官方 DeepSeek-V4.1-Flash 部署文档](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/DeepSeek-V4.1-Flash.html)（2026-11-11 查询，950DT DP8 的 P/D 示例）也把 P 端配置为 eager、D 端为 FULL_DECODE_ONLY。该在线文档可能继续更新，不代替现场版本核对。

| 源码定位（56debc1） | 实际行为 | 对当前算子的含义 |
|---|---|---|
| `models/deepseek_v41/model.py:1612`，模型构造函数 | A5 eager prefill / full-graph decode 说明，设置 `requires_uncompiled_fallback` | “模型支持图”不等于 prefill 的 CPU splits 分支在整图内 |
| `ops/fused_moe/token_dispatcher.py:615`，All2AllV `_preprocess` | 当前 topk_ids → histogram；本地聚合得到 input_splits；gather 专家计数后聚合得到 output_splits；两者 `.to(cpu, non_blocking=True).numpy()` | 当前路由决定本次 CPU splits；没有跨 replay 更新机制 |
| 同文件 `:518` / `:566` | dispatch 调用 `async_all_to_all` 后 `handle.wait()`；combine 交换使用 dispatch 保留的两份 splits | combine 不重新 routing；不能沿用上一个 batch 的 splits |
| `ops/fused_moe/moe_utils.py:63` | 按 `sum(output_split_sizes)` 分配输出，调用 `dist.all_to_all_single(..., input_split_sizes=..., output_split_sizes=..., async_op=True)` | CPU sizes 是本次 host 调用参数，动态输出分配也是 host 工作 |
| `compilation/acl_graph.py:155` | runtime NONE/模式不匹配时调用 runnable；图按 batch_descriptor 缓存 | 同一 batch shape 不代表同一 C；图键不能充当 routing 布局校验 |
| `device/hardware_profile.py:338`、`ascend_forward_context.py:336` | A5 使用 CAPACITY_AND_WORLD_SIZE 选择策略；还受 fused 配置、token capacity、EP 等影响 | 必须记录实际 MoECommType，不能从“开启图”推断正在跑 AllToAllV |
| `token_dispatcher.py:109`、`:232`、`:331` | MC2 dispatcher 调 `npu_moe_distribute_dispatch_v2` / `combine_v2`，传 expert_ids 等 tensor，保留 ep_recv_counts/assist_info 供 combine | 这是另一条通信接口，不是 CPU splits AllToAllV 自动支持动态 routing |

在 rank r 上，`input_splits[j]=C[r,j]`，`output_splits[i]=C[i,r]`；它们是 C 的一行、一列，**不是全局行和/列和向量**。不能仅凭本 rank 的两个向量恢复全局 C。现有“一份 CPU C 派生模型 splits、combine 复用 Cᵀ”的接入仍然有意义。

原生这段源码有 `non_blocking=True`，不应写成“`.numpy()` 一定自动等待 D2H”。应核对现场 CPU 首次读值前的实际完成保证。`num_local_experts>1` 分支还有动态 `repeat_interleave`，另一分支显式 synchronize；不能仅凭函数名推断所有版本的同步行为。这里也不能把 `Work.wait()` 的 host 返回时间直接解释为 device 完成时间。

### 13.2 原生究竟怎样处理动态路由

```text
原生 prefill AllToAllV（本次查到的 eager 分支）
  本次 hidden_states → router/topk_ids（device）
  → 专家计数 + gather（device）
  → 本次 CPU input/output splits
  → host 分配输出、提交本次 AllToAllV
  → 原有 wait/流依赖 → 专家计算 → 用该次交换后的 splits 做 combine

模型采用计算图时
  先根据运行时模式选择 eager / 分段图 / 整图和 MoE 后端
  若是 MC2：传当前路由 tensor 给对应设备算子（不是 CPU splits 接口）
  若是分段执行：CPU 代码只有留在图外并真正每次执行，才会读取新 splits
  若整图重放：不会重新执行捕获时的 Python _preprocess/CPU planner
```

分段图一行是可采用的设计原则，**不是已证明现场 DS4.1 A5 自动把 AllToAllV 切出图**。本次源码反而对 runtime NONE 设置了 uncompiled fallback，不能直接照其他模型的 PIECEWISE 设置改上线。

因此我们现在可直接沿用的是原生 eager prefill 的调用时机、同源 counts/splits、combine 元数据复用和依赖顺序。加入 L1 没有改变 C 的数据依赖。固定图需要动态参数这件事，旧 L0 relay 同样存在；此前 eager 能接模型不等于它已经支持动态整图重放。

### 13.3 找到了 task update 的真实例子，但目前用于 attention

`compilation/updatable_graph.py` 提供了 G4 当时缺的 handle 来源：

```text
捕获 register_task（约 :125）
  ExternalEvent.wait/reset
  graph_task_group_begin(current_stream)
  operation(**capture_kwargs)
  handle = graph_task_group_end(current_stream)
  保存 operation / kwargs / provider / handle / event

本轮更新 GraphUpdateTask.apply（约 :72）
  graph_task_update_begin(update_stream, handle)
  operation(**本轮 kwargs)
  graph_task_update_end(update_stream)
  event.record(update_stream)
```

`attention/attention_v1.py:687` 等处注册 attention op；`compilation/acl_graph.py:306` 从 **attention metadata** 解析本次参数。根据 enable_enpu，代码有 update→replay 或 replay→update 两种顺序，靠 external event 协调；不能只抄后两行而删除 event，也不能把任意 stream/CCU kernel handle 当作 task-group handle。

这说明可借鉴其“捕获任务组—保存句柄—更新任务—事件放行”框架；**没有证明 `dist.all_to_all_single` 或 `HcommCcuKernelLaunch` 是该更新机制支持的任务类型**。本次检查的 All2AllV dispatcher 没有注册这样的 update task。也不能把新版本整个 UpdatableGraph 类直接复制进现场旧版 torch_npu，须先核对接口及版本。

更关键的是：attention 的许多 host metadata 在 forward 前已知；**第 l 层的 C 依赖该次 forward 内、第 l 层 router 的输出，不能在重放整张模型图前凭上一轮 C 准备好。** 即使 CCU 参数更新探针成功，整模型仍须解决：router 完成 → counts D2H → CPU 规划 → 只放行对应通信任务；其余在途任务不能读旧计划或复用未完成的 buffer。这是与“图外已知 C 的裸算子探针”不同的第二道验证。

## 14. DeepSeek 下一步：先核对现场原生路径，再决定是否需要动态图

### 14.1 V0：无需占卡，固定实际加载源文件和版本

在服务使用的同一个容器、Python、工作目录、PYTHONPATH 下运行，保留输出。不要换成另一个 checkout 的解释器：

```bash
python3 - <<'PY'
import hashlib, importlib.util, importlib.metadata, pathlib, subprocess
for name in ('vllm', 'vllm-ascend', 'torch', 'torch-npu', 'deep-ep'):
    try:
        print('VERSION', name, importlib.metadata.version(name))
    except importlib.metadata.PackageNotFoundError:
        print('VERSION', name, 'not found under this distribution name')
spec = importlib.util.find_spec('vllm_ascend')
root = pathlib.Path(next(iter(spec.submodule_search_locations))).resolve()
print('LOADED_PACKAGE_ROOT', root)
p = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                   text=True, capture_output=True)
print('SOURCE_COMMIT', p.stdout.strip() if p.returncode == 0 else 'not a git checkout')
files = (
    'models/deepseek_v41/model.py', 'ascend_forward_context.py',
    'ops/fused_moe/token_dispatcher.py', 'ops/fused_moe/moe_utils.py',
    'compilation/acl_graph.py', 'compilation/updatable_graph.py',
)
keys = ('requires_uncompiled_fallback', 'class TokenDispatcher',
        'def _preprocess', 'def async_all_to_all', 'all_to_all_single(',
        'graph_task_', 'def _updatable_graph_replay', 'cudagraph_runtime_mode',
        'npu_moe_distribute_dispatch', 'npu_moe_distribute_combine')
for rel in files:
    f = root / rel
    if not f.exists():
        print('MISSING', f)
        continue
    data = f.read_bytes()
    print('FILE', f, 'SHA256', hashlib.sha256(data).hexdigest())
    for n, line in enumerate(data.decode(errors='replace').splitlines(), 1):
        if any(k in line for k in keys):
            print(f'{n}: {line.strip()}')
PY
```

同时保存现场 vLLM/vllm-ascend 的 `git diff`、G2/G3/G4 诊断补丁、实际 wheel/route SHA 和完整启动命令。输出只是索引；请贴命中函数的完整实现，尤其现场的 graph wrapper 和后端选择。若没有 updatable_graph.py，不代表所有旧版 update 不存在，继续检索 `graph_task_group` / `graph_task_update` 在 attention backend 的旧实现。

### 14.2 V1：有卡后只做一轮短 native 路径核对

先维持已通过的启动配置，不为测量直接删除 `--enforce-eager`。在 warmup/profile 后，只记录少量真实 prefill step；诊断不新增逐算子 synchronize。

必须从**两个位置**记录，避免 graph replay 不执行 Python 而漏报：

1. runner/graph wrapper：生效后的 compilation/cudagraph 配置、V1/V2 runner、每步 runtime_mode、batch_descriptor、实际 capture/replay/eager 分支；区分 prefill/decode、target/draft、profile/dummy。
2. MoE dispatcher/后端选择：层号、step、实际 MoECommType、dispatcher 类、有效/padded tokens、CPU splits 首次读值位置、D2H 提交与已有完成保证；记录 `_preprocess` 在真实多步中是否每步调用。

回报判定：

| 实际结果 | 下一步 |
|---|---|
| Prefill runtime NONE + All2AllV | 沿用 eager 动态 C；本次 L1 接入无需等待 G4-2b，继续原有正确性/端到端对照 |
| Decode full graph + MC2/FUSED_MC2 | 不是原生 CPU splits 先例；不能用其 PASS 证明我们的 host planner 入图 |
| Prefill 分段图且 All2AllV 在图外 | 保存切分点及调用次数；评估按相同边界放多路径，检查 torch.compile 自定义算子/输出 shape 契约 |
| Prefill full graph 且确实使用 CPU splits AllToAllV | 提供本次 routing 到 splits 的更新代码、task handle 来源；执行 V2，不能只以服务启动成功为证据 |

不存在通用 CLI 能自动打印以上全部字段。请按现场代码添加只读诊断，保存补丁；不要虚构已经存在的 graph flag。

### 14.3 V2：若要证明原生可更新 splits，做原生最小对照

仅在 V1 有相应实现证据，或明确要验证 SDK 能力时执行；不需要先跑整个大模型。复用当前已通过的八卡 harness 初始化/分配/验数，分别测 native 和 custom，固定通信域、caller stream、缓冲地址、dtype、容量。

- 构造 C0/C1：每个 rank 的发送总量和接收总量保持一致，但多条 peer counts 不同（可用合法的 2×2 增减保持行/列和）。这样 tensor shape 相同仍能检测 splits/offsets 是否真的更新；不能只换 epoch 而 C 不变。
- 正对照：eager 按 C0/C1 交替每次提交、固定图 G(C0)/G(C1) 分别捕获后交替，均需先 PASS。
- 一张图：按现场受支持的 task update 实现变更 C0/C1 的 splits/参数；每轮填新 epoch、清输出，再 replay、同步、按本轮布局逐元素验数及 guard。所有 rank 按同一顺序。
- “只改 CPU list，不调用更新”不是实现方案；若作负对照，只用上述总容量不变矩阵，单独有超时进程运行，预期不能据此得到新布局，不能在失败 communicator 内继续测。
- 记录捕获/更新/重放时 host wrapper 调用次数，防止实际 eager 执行了一遍而误报图更新 PASS。保留首次 API 返回码和任务组来源。

相同行/列和的 PASS 后，再单独测总接收量变化但不超预分配容量、零流、Cᵀ/BF16；row_bytes 改变使用独立图，未验证前不能在同图直接改 dtype。原生动态 shape 更新是否允许也要独立核实，不能用“大 buffer 容量足够”代替运行时 shape 契约。

### 14.4 V3：借鉴 task-group update，只做 CCU 最小能力探针

这是候选实验，不是现成生产 API。先核对现场 torch_npu 的 group/update API 签名及其支持任务类型；缺正式契约时向 CANN 提供 G1/G3 PASS 与最小复现询证，不猜私有 ABI。

按 §13.3 在捕获时从 `graph_task_group_end` 取得 handle；用现场支持的 external-event 协议更新同一任务组。先从单流、固定 buffer/行宽/通道/kernel 的裸 CCU launch 做 C0/C1 参数更新验证，再考虑完整 `host.execute`。后者含 caller/route 跨流桥，不能默认跨流任务都属于同一个可更新组。host.prepare/规划放在捕获和 task-update 区间之外。

需要 CANN/实测明确：CCU task args 是否复制、参数内存存活到何时、可变字段/大小限制、更新是否允许改变分流 selector/字节数/offset、跨流组如何归属、何时可以安全更新与重放。若某一步失败，保留具体错误，结论限定为该任务/版本/字段，不宣称整个 ACL Graph 不支持动态参数。

V3 正确后仍须验证 §13.3 的**模型内 routing 依赖**：不能用图外预存 C 冒充本次 router 输出。最小模型式链路是“设备生成两种不同 routing → 计数完成 → CPU 可读 → 规划/更新 → 对应通信 → 消费者验数”，必须有明确边界或受支持的等待/放行机制，不能让 CPU 在整图未结束时等整图完成而死锁。

## 15. 本轮交付与执行优先级

1. **现在做 V0**：确认现场版本、原生 AllToAllV 与 MC2、图更新代码是否与本次审阅一致。
2. **卡空后做 V1 短验证**：优先回答“P 节点实际哪一段入图、哪一段还在 eager”。若仍为 eager prefill，不因 G4 未通阻塞 L1 多路径接入性能测试。
3. **只有目标确实是动态图时做 V2/V3**：先验证原生接口/CCU 更新契约，再评估模型内 CPU 规划边界；不直接去掉 enforce-eager 长测。
4. 图性能另作配对：replay 预热和计时前对齐、同 host/plan/地址/行宽、eager/graph 交替、至少三次独立进程，报告全 rank 最大 wall、enqueue、正确性和流依赖；不要重用旧三倍结论。

回报最少字段：`source_commit+patch_sha / loaded_files_sha / wheel+route_sha / configured_mode / runtime_mode / actual_moe_backend / layer+step / C0_C1_hash / splits来源与完成保证 / capture_update_replay次数 / task_handle来源 / first_failed_API / payload+guard / clean_exit`。

本轮只更新文档与源码结论；没有修改模型仓库、生产 CCU 算子或现场诊断脚本，没有在 NPU 上新增验证。**可直接复用的是原生当前 eager 数据流；可研究复用的是 task-group 参数更新框架；尚未证实的是动态 CPU routing 驱动整张模型图内 CCU 任务。**
