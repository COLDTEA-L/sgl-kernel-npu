# Rank graph L1 聚合端点：先接通模型的执行手册

分支：`feature/a5-ccu-single-port-l1-alltoall`。本轮目标是验证：模型使用 **HCCL 实际 rank graph 暴露的 L1 CommLink** 是否能绕开 graph 外显式单口 EID 的建链问题。不是已确认的卡死修复，不承诺性能提升。继续 prefill eager，不改系统 rootinfo/Atlas，不占用其他人的卡。

## 1. 改了什么，哪些边界没变

- `A5_L1_ENDPOINT_MODE=rank_graph` 是新的默认值。`l1_graph` manifest 行必须匹配 **实际通信域** 的 layer=1 UB_CTP 链路，正反向分别由各 rank 检查。匹配 EID 的完整 16 字节、物理卡、端点协议/位置，并验证本地 die 与 L0 一致。
- 直接复制 rank graph 返回的完整 `EndpointDesc`；不再从 L0 描述符替换 EID，不按 EID 前缀推测单口/聚合，不把地址类型强制改成 EID。兼容 SDK 对 UB EID 使用 IPv6 类型标签的情况，但必须出现在 UB_CTP graph 链路中。
- 远端 `GetEndpointInfo ret=6` 仍可打印诊断，不据此单方面拒绝；对端进程验证自己的 local die。本地 die 查询失败或与 L0 不同则明确失败。
- 没有匹配路径报 `L1_GRAPH_PATH_UNAVAILABLE`，绝不回退成 graph 外 EID。旧单口目录在默认模式报 `L1_MODE_MISMATCH`；要恢复旧实验必须显式 `A5_L1_ENDPOINT_MODE=explicit_single`，并使用旧单口 policy/catalog。
- graph 枚举和匹配只在 handle 创建/新 stream 绑定阶段进行，不在每次 dynamic 规划或提交时查询；counts 仍为64项。
- 新 ABI：route `A5CcuRankGraphL1AbiVersion()=1`。CCU kernel/task 参数 ABI 不变；新 Python 配置逻辑需部署，模型建议重编 wheel 保持版本一致。
- 每条流仍至多一个替代路径，selector=8 表示源→L1网络→目的，不经过第三张 NPU 的 HBM。L0 与 L1 继续同一个执行器，混合 IO die 不支持。

**候选与验证分两步，不混称“发现成功”：**

1. CPU 工具读取当前机器 rootinfo 中已有的 L1 聚合声明，以及已验证 L0 拓扑的 rank/die 映射，生成候选。输出 `runtime_graph_match_required=1`，此时尚未证明 HCCL 接受它。
2. `prepare_ccu_urma_eid_alltoallv` 在真实 comm 上逐条匹配 `HcclRankGraphGetLinks`。只有目录全部匹配、acquire/register 成功后，才可建立 host 并执行 planner 计划。CPU planner 本身不访问 SDK，必须使用**该 handle 对应同一次生成的 topology.json**，不能手写 l1_resources 或复用旧目录。

若一个候选不在 graph 中，首版直接失败，不擅自删除局部路径：这样避免八个 rank 各自删出不同目录。需要根据 `PATH_CATALOG` 重新统一生成 policy，不能去掉运行时检查。

## 2. 聚合资源如何计费

首版每 rank 只选一个共同 plane/instance、与 L0 同 die 的 endpoint group，默认优先物理成员口数较少的公共组；可用 `--plane` 明确限定。**不是按 043f/047f 选端口组。** 一个 rank 对七个 peer 的七条 L1 channel 共享同一个资源槽，TX/RX 分别累计。

资源 ID 沿用 `rank*18 + die*9 + min(member_ports)`，但这里最小端口只是**整组的标识**。`resource_groups` 保存完整成员口；它不表示所有 payload 实际只走最小端口。

容量先保守设为一条 L0 边的 1 倍（`capacity_permille=1000`），不会把七条 channel 当作七份独立带宽，也不会仅凭两口/六口声明假定 2x/6x。多个物理口之间的分配由 SDK 决定。聚合资源成员重叠、同 rank 使用不同 endpoint group 均拒绝。控制器自定义预算首版不开放；真实吞吐测清楚后再扩展容量标定。

因此这是“先验证可用”的规划，不保证最优分流比例。两卡聚合 EID 比单口快的旧结果，不能直接等同于八卡 MoE 的端到端收益。

## 3. 不占卡：拉取、构建、CPU 回归

在目标容器按现场路径调整；保留未提交的适配层/诊断补丁，不要 reset。不要修改共享宿主的 `/etc/hccl_rootinfo.json`。

```bash
cd /home/l00934901/sgl-kernel-npu-alltoall
export A5_REPO=$PWD
git status --short
git pull --ff-only origin feature/a5-ccu-single-port-l1-alltoall
export A5_CANN_PATH=/usr/local/Ascend/cann-9.3.0
source "$A5_CANN_PATH/set_env.sh"
export A5_SESSION=$(mktemp -d /home/l00934901/profiling/graph-l1-model.XXXXXX)
git rev-parse HEAD | tee "$A5_SESSION/commit.txt"

python3 tests/python/deepep/test_a5_ccu_graph_l1_cpu.py
python3 tests/python/deepep/test_a5_ccu_l1_switch_cpu.py
bash scripts/build_a5_ccu_urma_route_probe.sh \
  --cann-path "$A5_CANN_PATH" --build-dir "$A5_SESSION/route-build" \
  --install --install-path "$A5_CANN_PATH"
export A5_CCU_ROUTE_PROBE_LIB="$A5_CANN_PATH/opp/vendors/cust/lib64/liba5_ccu_urma_route_probe.so"
export LD_LIBRARY_PATH="$(dirname "$A5_CCU_ROUTE_PROBE_LIB"):${LD_LIBRARY_PATH:-}"
python3 - <<'PY'
import ctypes, os
lib=ctypes.CDLL(os.environ['A5_CCU_ROUTE_PROBE_LIB'])
assert lib.A5CcuRankGraphL1AbiVersion()==1
print('GRAPH_L1_ROUTE_ABI_PASS')
PY
```

构建输出要求 `graph_l1_match`/`public_ccu_host` 测试通过、`PUBLIC_ROUTE_LOAD_PASS`、`PUBLIC_ROUTE_BUILD_PASS`。ABI 检查必须针对**模型实际加载的库路径**。

wheel 用主教程 §3.2 的目标容器构建/安装流程：

```bash
DEEPEP_SINGLE_OP=explicit_multipath_all2all_ccu bash build.sh -a deepep Ascend950
```

不要把别的 Python/CANN 环境的 wheel 装进服务。新代码本地验证不会替代现场 CANN 9.3 的构建与 NPU 正确性测试。

## 4. 生成全新目录（读取当前机器，不生成 EID）

先生成 L0-only 基底，再从 rootinfo 编译 graph L1 候选。拓扑工具会只读查询 hccn 端口状态；在允许读取现场设备状态时执行，不启动 NPU payload。

```bash
export A5_DEVICES=0,1,2,3,4,5,6,7
export A5_ROOT_INFO=/etc/hccl_rootinfo.json
export A5_ATLAS_JSON=/usr/local/Ascend/driver/topo/950/atlas_950_1.json
export A5_HCCN_TOOL=/usr/local/Ascend/driver/tools/hccn_tool
export A5_L1_ENDPOINT_MODE=rank_graph
export A5_EID_MAX_RELAYS_PER_PEER=0

A5_ENABLE_L1=0 python3 scripts/prepare_a5_ccu_ep8_topology.py \
  --executor eid --max-relays-per-peer 0 --devices "$A5_DEVICES" \
  --root-info "$A5_ROOT_INFO" --topology-json "$A5_ATLAS_JSON" \
  --hccn-tool "$A5_HCCN_TOOL" --output-dir "$A5_SESSION/l0-base"

python3 scripts/a5_ccu_graph_l1_policy.py \
  --root-info "$A5_ROOT_INFO" --base-topology "$A5_SESSION/l0-base/topology.json" \
  --output "$A5_SESSION/graph-policy/l1_policy.json"

export A5_ENABLE_L1=1
export A5_L1_CONFIG="$A5_SESSION/graph-policy/l1_policy.json"
python3 scripts/prepare_a5_ccu_ep8_topology.py \
  --executor eid --max-relays-per-peer 0 --devices "$A5_DEVICES" \
  --root-info "$A5_ROOT_INFO" --topology-json "$A5_ATLAS_JSON" \
  --hccn-tool "$A5_HCCN_TOOL" --l1-config "$A5_L1_CONFIG" \
  --output-dir "$A5_SESSION/model-plans"
```

预期 policy schema `a5.rank-graph-l1.v1`、28 个无向 pair；manifest 28 行 direct + 28 行 l1_graph；14 channel/rank。`allowed_relays` 全空是预期（关闭组内 L0 relay），不等于关闭 L1。topology 同时有 `l1_resources`、`l1_budget`、`l1_policy.resource_groups`。

如果没有共同同-die plane，停止并回报候选/rootinfo；不拼跨 plane 的端点，不按前缀构造 EID。

## 5. 有空闲卡：单算子 create → 正确性 → 性能

这些命令会独占本测试的八张卡。保持 harness 使用 64，**不能把此值照搬进模型**（已有模型 native MoE WinSize 失败证据）。

```bash
export HCCL_BUFFSIZE=64
export HCCL_OP_EXPANSION_MODE=CCU_SCHED
export A5_CCU_EP8_THREAD_BRIDGE=event
bash scripts/run_a5_ccu_eid_alltoallv.sh --suite create \
  --planner l1_bottleneck --register-stage full

bash scripts/run_a5_ccu_eid_alltoallv.sh --suite correctness \
  --planner l1_bottleneck --mode all --dtype float8_e4m3fn --width 5120 \
  --row-scale 1 --warmup 3 --iters 10 --rounds 1 \
  --dynamic-iters 10 --bidirectional-iters 10
```

launcher 会用同一个 `A5_L1_CONFIG` 现场重新生成目录；注意结果目录与持久 model-plans 不是同一路径。

create 验收：8 rank create PASS、无 fallback，每个 rank 7 个不同 peer 的 `L1_GRAPH_MATCH`（去重后一共56对），`PEER_PATH_DESC kind=l1_graph`，acquire/register 成功。只看到候选文件或服务端口就绪都不算通过。

若报 `L1_GRAPH_PATH_UNAVAILABLE`，贴该 rank/peer 的全部 `PATH_CATALOG` 和 policy；不要继续模型启动。若依然卡 `acquire_channels`，本实验说明“graph 外单口 EID”不是充分解释，保留 C++ 栈/请求跟踪，不再盲扫 planner/buffer。

短正确性通过后再测同一真实矩阵：

```bash
# 替换为现场真实矩阵，counts 为64个行数，三种模式读取同一文件。
export A5_MATRIX=/home/l00934901/profiling/sec6_dispatch.json
bash scripts/run_a5_ccu_eid_alltoallv.sh --suite benchmark \
  --planner l1_bottleneck --mode all --matrix-json "$A5_MATRIX" \
  --dtype float8_e4m3fn --width 5120 --warmup 20 --iters 50 --rounds 6 --dynamic-iters 100
```

先记录五口径 N/D/M、分流量、路径/资源组，不急于调容量。端口计数确认 payload 确实走对应 L1 成员口，且不把后台流量计入；聚合模式不要求只亮一个物理口。

## 6. 模型适配：保持原首次 MoE 位置，只换路径来源

本仓库不包含现场 vllm-ascend 适配层；以下由 DeepSeek 在现场修改并记录 diff。**不要同时继续 early-init 前移实验**。当前三个提前位置在 comm init 就失败，尚未回答 acquire 行为。

进程启动配置：

```bash
export A5_L1_ENDPOINT_MODE=rank_graph
export A5_ENABLE_L1=1
export A5_L1_CONFIG="$A5_SESSION/graph-policy/l1_policy.json"
export VLLM_ASCEND_EP8_PLAN_DIR="$A5_SESSION/model-plans"
export VLLM_ASCEND_EP8_EARLY_INIT=0
export HCCL_BUFFSIZE=1024
export HCCL_OP_EXPANSION_MODE=CCU_SCHED
export VLLM_ASCEND_EP8_MULTIPATH=1
export VLLM_ASCEND_EP8_COMBINE=0
# A5_CCU_ROUTE_PROBE_LIB 和 LD_LIBRARY_PATH 必须传入真正的 worker。
```

沿用目前已经建好的 custom ep8_custom 域、显式 PG buffer=200、全局1024；原生 MoE 继续原 EP 域。不要在首次 MoE 里临时 new_group。Buffer/handle 继续在原来能完成 comm init 的首次 MoE 位置初始化。

适配层 `ensure_engine` 必须按顺序：

1. 读取同一 model-plans 的 manifest/topology，核对 schema/mode 和 route 新 ABI；打印实际 so 路径、SHA、组名、组内 rank/physical device 映射。
2. `Buffer(custom_group)` → 用同一 Buffer 创建 handle。等待全目录 `L1_GRAPH_MATCH`、acquire/register 成功，**再**创建 rb=5120/10240 两个 host。
3. 两个 planner 都读取该 topology 的 `allowed_relays`、`l1_resources`、`l1_budget`。不再从单口 policy 的 `src_port/dst_port` 手算资源（新 policy 是成员口列表，整个组共享一个槽）。
4. 创建 host 时沿用这个 Buffer 的 runtime 和 handle、custom group 的 rank顺序。不能把原 EP runtime/旧 handle 混入。
5. 首次失败必须抛出并输出原始 stage/traceback；诊断期间关闭适配层静默原生回退，启动“就绪”不能替代成功证据。

配置检查示例（使用目标 wheel 中的新模块；实际代码按已有 import 布局调整）：

```python
from deep_ep.ccu_l1_config import validate_l1_schema
assert validate_l1_schema(topology['l1_policy']['schema']) == 'rank_graph'
assert route.A5CcuRankGraphL1AbiVersion() == 1
planner = RelayPlanner(
    row_bytes, topology['allowed_relays'], native_module=native,
    strategy='l1_bottleneck', max_actions=56,
    l1_resources=topology['l1_resources'], l1_budget=topology['l1_budget'])
```

第一轮只验证启动和一个真实请求，不测端到端性能。记录首次 dispatch 的 moved/selector 和实际 custom 提交，不能把“0个组内relay”当成“没有L1”。推荐依次做三个开关对照，均使用完整 L1 handle：

- **D**：planner `max_actions=0`，仅 L0 payload；确认建链后模型输入/输出布局正确。
- **M-dispatch**：恢复 `max_actions=56`，只替换 dispatch。若该请求计划确实 moved>0，才算测试到 L1 payload。
- **M-both**：前者通过再开 COMBINE=1；combine 用本层 dispatch CPU C 的转置 contiguous int64，rb=10240，单独规划。不能复用 FP8 moved 字节/offset。

D 与 M 不要换 handle/catalog；派生 CPU splits 方案维持现场已验证实现。两个方向顺序共享一个 handle；不要并发调用、重复销毁。关闭时先完成工作、关闭 hosts，再销毁共享 handle一次。

## 7. 回报与停止条件

回报：commit/wheel/route SHA+ABI、实际 worker 环境与 PG 配置、policy/manifest/topology SHA、选定 plane及每rank成员口、56个唯一 `L1_GRAPH_MATCH`、create/正确性PASS、D/M-dispatch/M-both 是否执行且无fallback。

失败按首阶段报告：

- comm init error4：尚未进入本改动的 graph 匹配，不能声称聚合路径失败。
- MODE_MISMATCH：旧目录与新模式混用，重新生成，不关闭检查。
- GRAPH_PATH_UNAVAILABLE/DIE_INVALID：图候选或IOdie不兼容，贴对应graph行。
- acquire卡住：记录实际GET_SOCKET进入/返回与全线程栈，不推断是planner问题。
- execute失败：首条 `ALLTOALLV_STAGE_FAILED`、counts、rb、capacity、tensor形状/stride/地址对齐及组/rank映射。

仅 D 正常而 M 失败时，才进入 L1 payload 诊断。功能全过后，另做 DIAG=0 的原生/多路径交替多轮端到端对照，保留模型原生所需 buffer，不把本轮“能接入”写成“已获性能收益”。

## 8. 本次开发侧验证范围

在本人 `cam_lyw_92` / CANN 9.2 中已完成 route 编译、动态加载与 ABI 检查；`graph_l1_match` 和 `public_ccu_host` 均通过。新增 graph policy CPU 回归、L1 开关、原有 L1 planner/资源预算、peer-plan、机器拓扑回归通过。修正了旧 peer-plan 测试把 `L1_SWITCH` 日志误当 JSON 的解析方式，生产输出不变。

未在现场 CANN 9.3/NPU 上验证模型启动、数据正确性和性能；这些正是 §5–§7 要完成的内容。没有修改现场 vllm-ascend 文件，也没有修改宿主 rootinfo/Atlas。
