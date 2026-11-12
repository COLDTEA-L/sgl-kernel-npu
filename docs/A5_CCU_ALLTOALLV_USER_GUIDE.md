# A5 CCU+URMA 多路径 AllToAllV 使用、性能测试与模型接入教程

面向有卡服务器上的 DeepSeek 执行者。适用分支：`feature/a5-ccu-single-port-l1-alltoall`；接口基线为 `dc66ebb`（L1 资源预算）和 `4c2f742`（L1 启动开关）。本教程按该版本源码核对，不代表下列现场测试已经全部执行。

**第一次使用依次执行 §2–§5；测单算子看 §6；接 DeepSeek-V4.1 prefill 看 §7–§8；回报格式在 §10；service64 dynamic 专项定位看 §11。** 已安装旧 wheel 的机器仍需更新 wheel：只更新 route 不会更新 Python 开关和 planner binding。

> **模型 L1 建链兼容路线更新：** 新默认 `A5_L1_ENDPOINT_MODE=rank_graph`，完整流程见 [Rank graph L1 模型接入](A5_CCU_RANK_GRAPH_L1_MODEL.md)。这次由 DeepSeek 先执行该文档 §3–§5，通过后 §6，按 §7 回报。需要重编 route、新 ABI=1，并部署新 Python 代码。下面旧的显式单口流程必须显式设置 `A5_L1_ENDPOINT_MODE=explicit_single`；不要把旧 policy 混入新默认模式。

## 1. 算子做什么，当前支持到哪里

- 2/4 卡固定 AllToAll：显式策略配置 L0/L1 分流权重，入口见 §9。
- MoE EP8 AllToAllV：输入全组一致的 8×8 行数矩阵 `C`，`C[i][j]` 是源 rank i 发往目的 rank j 的 **packed payload 行数**，不是原始 HTTP 请求数或未展开的 token 数。
- 路径包括 L0 direct、组内 NPU 的 L0 fullmesh relay、显式单口 L1 网络路径。都使用 CCU+URMA；relay 不落中间 NPU HBM，源/目的 payload 本身仍在 NPU HBM。
- 每条流至多一个备用路径：direct + 一个 L0 relay，或 direct + 一条 L1；不会同时给同一条流配置 L0 relay 和 L1。selector `8` 表示 L1，不是第九张卡。
- L1 只接受实际枚举、与当前机器拓扑匹配的单口 EID，且与 L0 同 IO die。不能保证每张卡八个 L1 物理口都能被本执行器同时使用。这条是旧 explicit_single 模式的限制；新的 rank_graph 模式接受图中聚合端点，见上方专用手册。
- L1 物理口按源 TX、目的 RX 分别累计负载；多个 peer 使用同一个口不能各自假设独占带宽。`C` 仍为 8×8，控制器可用冻结预算描述外部竞争；本版不是多节点 EP16 实现。
- 模型接入先采用 **prefill eager**。固定 prepared 图验证成功不等于本次动态 routing 可以直接重放上次图；本教程不要求取消 `--enforce-eager`。

L1 开关与目录配置：

| 设置 | 生成/创建行为 |
|---|---|
| `A5_ENABLE_L1=0` | 禁用 L1，忽略传入的 L1 policy；Python `RelayPlanner` 将 `l1_bottleneck` 降为 `bottleneck`，保留 L0 relay |
| `A5_ENABLE_L1=1` | 生成拓扑必须提供有效 L1 policy；使用 L1 算法需显式选择 `l1_bottleneck` |
| 未设置 | 兼容旧的显式 config/strategy 选择；建议新部署明确设置 0 或 1 |
| `A5_EID_MAX_RELAYS_PER_PEER=0` | 不建组内 L0 relay 候选；**不代表关闭 L1** |
| `A5_EID_MAX_RELAYS_PER_PEER=6` | 每 peer 最多六个组内 L0 relay 候选，可与 L1 同时存在 |

开关只接受 `0/1`，在拓扑和 planner 创建时读取。改变开关、policy 或预算后，需重新生成拓扑并重建 handle/host；不能让旧 prepared 计划配新目录。初测建议 L1=1、组内 relay=0，先隔离 L1 效果，再验证混合目录。

## 2. 拉代码、环境和版本记录

所有命令在**现场目标容器**中用 Bash 执行。路径按现场调整；只在获准使用的空闲卡上测试，不停止其他人的进程。构建与模型服务必须使用同一套 Python/Torch/torch_npu/CANN 环境。

```bash
export A5_REPO=/home/l00934901/sgl-kernel-npu
cd "$A5_REPO"
git status --short
# 有未提交的现场诊断补丁时先保存并人工合并；不要 reset --hard。
git fetch origin feature/a5-ccu-single-port-l1-alltoall
# 本地分支不存在时：git switch --track origin/feature/a5-ccu-single-port-l1-alltoall
# 本地分支已存在时：
git switch feature/a5-ccu-single-port-l1-alltoall
git pull --ff-only origin feature/a5-ccu-single-port-l1-alltoall

export A5_CANN_PATH=/usr/local/Ascend/cann-9.3.0
export A5_OUTPUT_ROOT=/home/l00934901/profiling
mkdir -p "$A5_OUTPUT_ROOT"
export A5_SESSION=$(mktemp -d "$A5_OUTPUT_ROOT/a5-user-guide.XXXXXX")
source "$A5_CANN_PATH/set_env.sh"
source scripts/a5_ccu_test_env.sh
a5_ccu_prepare_test_env "$A5_CANN_PATH" "$A5_REPO"
export HCCL_BUFFSIZE=64
export HCCL_OP_EXPANSION_MODE=CCU_SCHED
export A5_CCU_EP8_THREAD_BRIDGE=event
export A5_DEVICES=0,1,2,3,4,5,6,7
export A5_ROOT_INFO=/etc/hccl_rootinfo.json
export A5_ATLAS_JSON=/usr/local/Ascend/driver/topo/950/atlas_950_1.json
export A5_HCCN_TOOL=/usr/local/Ascend/driver/tools/hccn_tool
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64:${LD_LIBRARY_PATH:-}
git rev-parse HEAD | tee "$A5_SESSION/commit.txt"
git diff --stat | tee "$A5_SESSION/local-changes.txt"
```

`HCCL_BUFFSIZE=64` 是现场复现实验的固定值，不是 L1 端口带宽参数，也不是所有模型的最佳值。改变它要重启进程，并保持 A/B 两组一致。模型启动时须继承环境；在宿主 shell export 不等于已有容器/服务进程已获得设置。

## 3. 构建 route、wheel，核对实际加载版本

### 3.1 route

```bash
bash scripts/build_a5_ccu_urma_route_probe.sh \
  --cann-path "$A5_CANN_PATH" --build-dir "$A5_SESSION/route-build" \
  --install --install-path "$A5_CANN_PATH"
export A5_CCU_ROUTE_PROBE_LIB="$A5_CANN_PATH/opp/vendors/cust/lib64/liba5_ccu_urma_route_probe.so"
export LD_LIBRARY_PATH="$(dirname "$A5_CCU_ROUTE_PROBE_LIB"):${LD_LIBRARY_PATH:-}"
sha256sum "$A5_CCU_ROUTE_PROBE_LIB" | tee "$A5_SESSION/route.sha256"
```

必须出现 `PUBLIC_ROUTE_BUILD_PASS`、`PUBLIC_ROUTE_LOAD_PASS` 和单口 L1 ABI=1。已有同版本 route 可复用；第一次从旧 EP8 后端升级到 L1 分支必须重编。`4c2f742` 的开关本身没有新增 route ABI。

### 3.2 wheel

```bash
(
set -Ee -o pipefail
cd "$A5_REPO"
source scripts/a5_ccu_test_env.sh
a5_ccu_prepare_test_env "$A5_CANN_PATH" "$A5_REPO"
A5_WHEEL_BUILD=$(mktemp -d "$A5_SESSION/wheel.XXXXXX")
mkdir -p "$A5_WHEEL_BUILD/tmp"
export TMPDIR="$A5_WHEEL_BUILD/tmp"
touch "$A5_WHEEL_BUILD/started"
DEEPEP_SINGLE_OP=explicit_multipath_all2all_ccu bash build.sh -a deepep Ascend950 \
  > "$A5_WHEEL_BUILD/build.log" 2>&1 || {
  tail -100 "$A5_WHEEL_BUILD/build.log"; exit 1;
}
mapfile -t A5_WHEELS < <(find "$A5_REPO/output" -maxdepth 1 \
  -name 'deep_ep-*.whl' -newer "$A5_WHEEL_BUILD/started" -print)
[[ ${#A5_WHEELS[@]} == 1 ]] || { echo '请检查新生成的 wheel，禁止误装旧包'; exit 1; }
python3 -m pip install --force-reinstall --no-deps "${A5_WHEELS[0]}"
)
```

在模型使用的 Python 中验证，不要让另一份源码目录的 `PYTHONPATH` 意外遮盖安装包：

```bash
python3 - <<'PY'
import ctypes, os
import torch, torch_npu, deep_ep
from deep_ep import deep_ep_cpp as cpp
import deep_ep.ccu_l1_config as l1cfg
print('torch', torch.__version__, 'torch_npu', torch_npu.__version__)
print('package', deep_ep.__file__)
print('extension', cpp.__file__)
print('l1_switch', l1cfg.__file__)
for name, expected in {
    'ep8_eid_host_abi': 1,
    'ep8_eid_host_counts_tensor_abi': 1,
    'ep8_single_port_l1_abi': 1,
    'ep8_l1_resource_planner_abi': 1,
}.items():
    actual = getattr(cpp, name, None)
    print(name, actual)
    assert actual == expected, (name, actual)
route = ctypes.CDLL(os.environ['A5_CCU_ROUTE_PROBE_LIB'])
for name in ('A5CcuSinglePortL1AbiVersion', 'A5CcuEidSubmitFastAbiVersion',
             'A5CcuEidV1CompatAbiVersion'):
    fn = getattr(route, name); fn.argtypes = []; fn.restype = ctypes.c_int
    actual = fn(); print(name, actual); assert actual == 1
PY
```

建议 CPU 回归分进程执行，避免 pybind 同名类型重复注册：

```bash
for A5_TEST in test_a5_ccu_l1_switch_cpu.py test_a5_ccu_l1_cpu.py test_a5_ccu_bottleneck_cpu.py; do
  python3 -m unittest discover -s tests/python/deepep -p "$A5_TEST" -v || exit 1
done
```

ABI 通过只证明接口存在，不代替数值校验和物理路径验证。

## 4. 用当前机器生成路径

### 4.1 有可用 L1：发现单口 EID

```bash
export A5_ENABLE_L1=1
python3 scripts/discover_a5_single_port_l1.py \
  --devices "$A5_DEVICES" --root-info "$A5_ROOT_INFO" \
  --output-dir "$A5_SESSION/policy-8"
export A5_L1_CONFIG="$A5_SESSION/policy-8/l1_policy.json"
export A5_EID_MAX_RELAYS_PER_PEER=0
```

输出目录必须是新目录。检查 `inventory.json`、`candidates.json`、`l1_policy.json`：物理卡顺序、完整 EID、die/port、plane/instance、rootinfo/inventory 摘要须对应当前机器。不要复制旧机器的 EID 或把聚合地址改几个字节当单口地址。没有有效候选时停止 L1 验证，改测 L0 模式。

EID 被枚举且建链成功仍不足以证明只走一个物理口。首次上机按[单口 L1 指南 §7](A5_CCU_SINGLE_PORT_L1_ALLTOALL.md#7-性能判据与物理路径验证)做两卡端口计数差分；与空窗口背景比较，核对选中端口承担 payload、相邻未选口没有分摊 payload。

### 4.2 没有 L1 或只测 L0

```bash
export A5_ENABLE_L1=0
unset A5_L1_CONFIG
export A5_EID_MAX_RELAYS_PER_PEER=6
```

以下统一传 `--planner l1_bottleneck`，禁用时 wrapper 会转成 `bottleneck`。须在日志中确认实际 strategy；这不表示 `A5_ENABLE_L1=1` 会自动把显式指定的 `greedy` 改成 L1 策略。

### 4.3 EP8 持久拓扑（模型也用这一份）

```bash
A5_L1_ARGS=()
if [[ "$A5_ENABLE_L1" == 1 ]]; then
  A5_L1_ARGS+=(--l1-config "$A5_L1_CONFIG")
fi
export A5_MODEL_PLANS="$A5_SESSION/model-plans"
python3 scripts/prepare_a5_ccu_ep8_topology.py --executor eid \
  --devices "$A5_DEVICES" --max-relays-per-peer "$A5_EID_MAX_RELAYS_PER_PEER" \
  --root-info "$A5_ROOT_INFO" --topology-json "$A5_ATLAS_JSON" \
  --hccn-tool "$A5_HCCN_TOOL" "${A5_L1_ARGS[@]}" \
  --output-dir "$A5_MODEL_PLANS"
```

预期 `EP8_TOPOLOGY_PASS`、executor=`participant_eid_ccu_urma`。模型必须从这个目录**配套读取** `peer_plan.tsv` 和 `topology.json`。旧 `participant_hbm_two_stage_ccu` 文件即使时间戳一致也不能复用。

官方单算子脚本每次会现场生成独立拓扑，不直接读取 `A5_MODEL_PLANS`；它读取同一组环境变量及 policy。路径/预算修改后重跑生成，不能只改模型 planner、不重建实际通道。

资源预算及控制器扩展见[单口 L1 指南 §11](A5_CCU_SINGLE_PORT_L1_ALLTOALL.md#11-资源预算版-planner单节点现在可用多节点预留接口)。默认不声明外部背景流量，不能据此断言网络空闲。预算按规划窗口中的字节数和容量千分比描述，不能直接填累计 flit 计数。

## 5. 先验证正确性，再测性能

```bash
bash scripts/run_a5_ccu_eid_alltoallv.sh \
  --suite create --register-stage full --planner l1_bottleneck

bash scripts/run_a5_ccu_eid_alltoallv.sh \
  --suite correctness --mode all --planner l1_bottleneck --max-actions 56 \
  --dtype float8_e4m3fn --width 5120 --row-scale 1 \
  --warmup 3 --iters 10 --rounds 1 --dynamic-iters 100 --bidirectional-iters 100

bash scripts/run_a5_ccu_eid_alltoallv.sh \
  --suite correctness --mode all --planner l1_bottleneck --max-actions 56 \
  --dtype bfloat16 --width 5120 --row-scale 1 \
  --warmup 3 --iters 10 --rounds 1 --dynamic-iters 100
```

不带外部矩阵时 correctness 会覆盖 zero/self/single_hot/competing/asymmetric/uniform/cut；双向循环另测 FP8/BF16 切换。验收必须同时满足：

1. create 出现 `EP8_EID_CREATE_ALL_RANKS_AND_EXIT_PASS`；它仅证明建链和注册。
2. correctness 出现 `EP8_EID_ALL_RANKS_AND_EXIT_PASS`，8 rank 干净退出，结果目录 `exit.status=0`。
3. L1 验证至少有实际 `l1_rows>0` 的 case 数值正确。全走直连不能记为 L1 PASS。
4. 启用六个 L0 relay 候选时，重新执行 create 和 correctness，不能用 L1-only 目录的结果代替混合目录验证。

`--max-actions 56` 是算法允许的上限，不是每次都发 56 个动作；原 L0 bottleneck 仍有内部最多八动作限制。均匀矩阵若要所有跨卡边同时削峰，可能需要 56 个动作，设为 8/16 可能合法退回 direct。

## 6. 单算子性能：怎么测、怎么解释

### 6.1 输入真实矩阵，生成 combine 转置

当前 harness 的 JSON 字段是 **`counts`**，支持长度 64 或 8×8；可附 `devices`，顺序必须匹配拓扑。不要只提供名为 `C` 的字段。

将真实 dispatch 矩阵保存为 `$A5_SESSION/dispatch.json`，例如结构 `{"devices":[0,1,2,3,4,5,6,7],"counts":[[...八行...]]}`；这里省略号只是格式说明，实际必须 64 个非负整数。然后执行：

```bash
export A5_DISPATCH_JSON="$A5_SESSION/dispatch.json"
export A5_COMBINE_JSON="$A5_SESSION/combine.json"
python3 - <<'PY'
import json, os
from pathlib import Path
src = json.loads(Path(os.environ['A5_DISPATCH_JSON']).read_text())
c = src['counts']
flat = [x for row in c for x in row] if len(c) == 8 and all(isinstance(r, list) for r in c) else c
assert len(flat) == 64 and all(type(x) is int and x >= 0 for x in flat)
out = {'counts': [[flat[j*8+i] for j in range(8)] for i in range(8)]}
if 'devices' in src: out['devices'] = src['devices']
Path(os.environ['A5_COMBINE_JSON']).write_text(json.dumps(out, indent=2) + '\n')
PY
```

dispatch FP8：width=5120，row_bytes=5120；combine BF16：同 width，row_bytes=10240，**使用 C.T**。只换 dtype、不转置 C 是另外一个实验，不是该模型的 combine。

先对外部矩阵再跑一次 correctness（用上节命令添加 `--matrix-json`，相应选择 FP8/C 或 BF16/C.T），通过后做三次独立进程复测：

```bash
(
set -Ee -o pipefail
for A5_REP in 1 2 3; do
  for A5_DIR in dispatch combine; do
    if [[ "$A5_DIR" == dispatch ]]; then
      A5_TYPE=float8_e4m3fn; A5_MATRIX="$A5_DISPATCH_JSON"
    else
      A5_TYPE=bfloat16; A5_MATRIX="$A5_COMBINE_JSON"
    fi
    bash scripts/run_a5_ccu_eid_alltoallv.sh \
      --suite benchmark --mode all --planner l1_bottleneck --max-actions 56 \
      --matrix-json "$A5_MATRIX" --dtype "$A5_TYPE" --width 5120 \
      --counts-path legacy --warmup 20 --iters 50 --rounds 6 --dynamic-iters 100 \
      | tee "$A5_SESSION/${A5_DIR}-r${A5_REP}.launch.log"
  done
done
)
```

脚本输出每次独立结果目录；保留目录中的 `commit.txt`、`arguments.txt`、`environment.log`、`topology.log`、`run.log`、`exit.status`、`plans/`、`results/rank*.json`。不要只抄终端单个 median。

### 6.2 三种模式与五种口径

| 模式 | 含义 |
|---|---|
| native / N | 实际 HCCL AllToAllV 基线 |
| direct / D | 同一 custom host 执行体系、相同 handle/通道目录，规划不分流 |
| multipath / M | 启用实际 planner，可能走 L0 relay、L1，也可能合法选择 direct |

`D−M` 衡量分流带来的净收益；`N−M` 衡量替换原生实现的收益。L1-only 和 L0+L1 的两次独立目录对比还包含目录差异，不能直接称为纯路径收益。

| 口径 | 当前脚本实际执行的工作 |
|---|---|
| prepared | 计时外 prepare；计时内执行固定计划及 payload 传输 |
| host_counts | CPU counts 已就绪；custom 每次 `host.run()`，含重新规划和 payload 传输 |
| host_counts_serial | 上一项加每次调用完成同步，减少连续调用重叠 |
| counts_only | 只获取/交换 counts，不做 payload AllToAllV |
| dynamic | 每次获取 counts，再执行本次规划和 payload 传输 |

**host_counts 不是“只测计数”或“只测 host”；prepared 也不是纯设备链路时间。** 当前 harness 在计时前预热、同步、跨 rank 对齐；每轮多次调用后最终同步，按每次耗时做全 rank MAX，再汇总 rounds。50 次连续提交组成一个块样本，不是 50 次独立实验。输出 buffer 已在计时外分配，因此这里的 dynamic 不是模型适配层全程。

真实模型已提供 CPU C 时，`host_counts` / `host_counts_serial` 更接近算子入口；不能把微基准额外 counts collective 成本再次算到模型。要分段诊断，可给同一命令加 `--combination-audit`，但主机 action_us 不等于设备传输时间，各段中位数也不能直接相加当精确总耗时。

报告两个不同的百分比：加速比=`N/M`；时延降低率=`(N−M)/N`。三轮方向不一致或量级接近噪声时写“未检出稳定收益”。不要用逻辑峰值降低 9% 直接推导整个算子快 9%。

## 7. 模型接入：数据契约与调用示例

### 7.1 修改位置与职责

现场 vllm-ascend 的 `vllm_ascend/ops/fused_moe/token_dispatcher.py` 负责 routing、counts、split 和前后置排列；`ep8_multipath_a2a.py` 适配层负责持久 engine、输出分配和一次 C++ 调用。以下是接口示例，不是对任意 vllm-ascend 提交直接可套的补丁；先核对现场这两个文件。

- 用模型现有 EP group，`rank` 是组内 rank，不能直接用物理 device ID。manifest 的物理卡顺序必须对应该 EP group 的 rank 顺序。
- packed send 按目的 rank 连续排列；recv 按源 rank 连续排列。FP8 payload 与 BF16 payload 原样搬运，量化 scale、专家排序、权重及 combine reduction 仍由原模型处理。
- 所有 rank 本次使用同一个 C。`input_splits[j]=C[rank,j]`，`output_splits[i]=C[i,rank]`，它们是本 rank 的一行/一列，不是全局矩阵的行和/列和。
- 只有本 rank 的两个 split 无法恢复完整 C。若模型已有设备上的全局 expert counts，按专家归属聚合为 C；复用其 collective，不额外跑一次全组 counts gather。
- C 只 D2H 一次；由这份 CPU C 派生模型 splits 和 planner 输入。若 CPU C 已存在，直接复用。异步 D2H 必须在读取 `.numpy()`、splits 或调用 C++ 前确认完成，不能假设 `.numpy()` 自动等待拷贝。
- combine 使用对应本次 dispatch 的 `C.T.contiguous()`，不再 D2H；这是行数转置。FP8/BF16 行宽不同，要按 combine 行宽重新规划，不能直接重放 dispatch 的 action。

### 7.2 engine 在初始化阶段创建一次

下例由每个 EP rank 在正确 device 和将用于调用的 caller stream 上执行。`ep_group` 由模型提供；`A5_MODEL_PLANS` 是 §4.3 生成的目录。不要在每层调用时重建通信组/Buffer/handle。

```python
import ctypes
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
import deep_ep
from deep_ep import deep_ep_cpp as cpp
from deep_ep.heuristic_alltoallv import RelayPlanner

plans_dir = Path(os.environ['A5_MODEL_PLANS'])
topology = json.loads((plans_dir / 'topology.json').read_text())
assert topology['executor'] == 'participant_eid_ccu_urma'
assert dist.get_world_size(ep_group) == 8
rank = dist.get_rank(ep_group)
# 部署时还须核对 topology['devices'] 与 group rank -> 物理卡映射。
l1_on = os.environ.get('A5_ENABLE_L1') == '1'
assert os.environ.get('A5_ENABLE_L1') in ('0', '1')
assert l1_on == (topology.get('l1_resources') is not None)

planner = RelayPlanner(
    5120, topology['allowed_relays'], native_module=cpp,
    strategy='l1_bottleneck' if l1_on else 'bottleneck',
    l1_resources=topology.get('l1_resources'),
    l1_budget=topology.get('l1_budget'), max_actions=56)
buffer = deep_ep.Buffer(ep_group, num_nvl_bytes=0, num_rdma_bytes=0)
handle = buffer.prepare_ccu_urma_eid_alltoallv(
    'ep8-model-eid', str(plans_dir / 'peer_plan.tsv'))
host = cpp.ep8_eid_create(planner._native, rank, buffer.runtime, handle)
# engine 强引用保存 planner/buffer/handle/host/EP group 和 caller stream。
```

本路径不需要旧 HBM 执行器的 `PipelineContext`、三 lane、`attach_lane` 或 first/second-hop 通信组。host 绑定创建时的 device/caller stream；不要换流调用或并发复用同一个 host。

### 7.3 每次只规划一次

```python
def run_with_cpu_counts(host, counts_cpu, send, recv_rows):
    # counts_cpu: 已完成 D2H 的 CPU int64 contiguous [8,8] 或 [64]
    # send: contiguous NPU payload，第一维是 packed rows
    # recv_rows: 直接复用模型已算好的 sum(output_splits)
    recv = send.new_empty((recv_rows, *send.shape[1:]))
    host.run_counts_tensor(counts_cpu, send, recv)
    return recv
```

`run_counts_tensor()` 返回内部计时信息，不是输出 tensor；输出写入传入的 recv。不要在它前面再调用 `planner.plan()` 或 `host.prepare()`，否则重复规划。生产热路径不用 `.tolist()` 仅为把同一份 counts 再传回 C++；诊断指纹仅在诊断打开时构造。

同一个 host 可按顺序处理 dispatch FP8 和 combine BF16，不同 row_bytes 会按本次 payload 处理。示意：

```python
# C_cpu 属于本次 batch、本层 dispatch，不能被另一在途请求覆盖。
dispatch_recv = run_with_cpu_counts(host, C_cpu, dispatch_send, dispatch_recv_rows)
# 原模型完成专家计算及 combine 前的排列后：
combine_C_cpu = C_cpu.reshape(8, 8).transpose(0, 1).contiguous()
combine_recv = run_with_cpu_counts(host, combine_C_cpu, combine_send, combine_recv_rows)
```

必须保留模型原来的流依赖和输出消费顺序。共享输出 buffer 只能在前一消费者已完成时复用；不能因为形状一样就覆盖在途 payload。不要逐调用新增整 device synchronize 来“保证安全”，应先沿用已验证的 host 流契约。多 batch/层交错时，以调用上下文保存 C 和对应 combine，不能只保存一个会被覆盖的全局矩阵。

准备固定计划的接口仅用于真实固定 C 的场合：`prepared = host.prepare_counts_tensor(C_cpu, send)`，随后 `host.execute(prepared, send, recv)`。C、方向、dtype/row_bytes、owner host 或拓扑改变时，旧 prepared 不应复用。

### 7.4 关闭顺序与失败处理

停止发起新调用后，在 NPU context、caller stream、buffer 和 EP group 仍有效时：

```python
host.close()  # 等待相关工作并释放 host 缓存；不销毁外部 route handle
route = ctypes.CDLL(os.environ['A5_CCU_ROUTE_PROBE_LIB'])
destroy = route.HcclCcuUrmaPeerPlanDestroy
destroy.argtypes = [ctypes.c_uint64]
destroy.restype = ctypes.c_int
status = destroy(handle)  # 唯一 owner 只销毁一次
if status != 0:
    raise RuntimeError(f'EID plan destroy failed: {status}')
```

多个 host 共享 handle 时，先关闭全部 host，再由唯一 owner destroy。之后由模型框架正常销毁共享 EP group；不要在单层适配器中 destroy 模型通信组。不能依赖 Python 进程退出后析构，也不要在 stream 已销毁后才 close。route plan destroy 不等于独立释放通信域持有的所有资源，通信域仍需按框架生命周期结束。

自定义执行已经提交或某 rank 出错后，不要只在该 rank 捕获异常并直接原生回退；其他 rank 可能仍等待 custom 通信。只允许在**全 rank 一致、尚未提交**的准入判断处选择 native/custom；执行期错误应整体停止并保留第一条失败阶段。

## 8. 模型启动与端到端性能验证

沿用现场已经成功的 DeepSeek-V4.1 prefill 启动脚本，保持 DP8/TP1/EP8、token/sequence 预算、量化、请求集及 eager 设置一致。先完成 §5，再接模型。以下变量是已有现场适配层的约定，**算子仓本身不会自动替你修改 vllm-ascend**；确认适配层读取了它们：

```bash
export A5_MODEL_PLANS="$A5_SESSION/model-plans"
export VLLM_ASCEND_EP8_PLAN_DIR="$A5_MODEL_PLANS"
export VLLM_ASCEND_EP8_MULTIPATH=1
export VLLM_ASCEND_EP8_COMBINE=1
export VLLM_ASCEND_EP8_DIAG_EVERY=0
# A5_ENABLE_L1、route 路径、HCCL_BUFFSIZE 等同样传入每个服务进程。
# 然后执行现场已验证的 prefill 启动脚本，继续使用 --enforce-eager。
```

`A5_L1_CONFIG` 用于生成拓扑；服务中真正配对使用的是生成的 manifest/topology。只设置 policy 环境变量、适配层仍读取旧 HBM 目录，是错误接入。启动记录 requested/effective strategy、L1 开关、目录摘要、实际模块路径和两方向 row_bytes。

建议分两步验收：

1. **短功能验证，诊断开。** 跳过 profile/dummy 数据的收益统计，但不能随意破坏它们原有通信流程；核对真实调用的 C、local splits、combine 转置、每次一次 host.run、输出形状和完整请求成功。记录 L1/L0 relay 实际行数；不把“启用开关”当成“发生分流”。
2. **性能验证，诊断关。** 固定真实请求文本、tokenizer、发送方式、并发数；baseline 原生与 multipath 交替独立启动，至少三组。只关闭现场适配开关来测 baseline，不更改 eager、预算或 HCCL_BUFFSIZE。记录成功率、输入 token 分布、P50/P90/P99、吞吐和各 rank 实际 scheduled tokens。

不能把 custom 的 plan/alloc/submit 全程与 native 的异步提交区间直接相减。native 也有分配、提交与完成依赖；`py_run` 可能包住 `cpp_plan/cpp_submit`，这些嵌套时间不能相加。分别记录 CPU counts 就绪、分配、host 调用、内部 plan/submit、原流程已有 wait；不额外同步改变生产重叠。最终收益以 DIAG=0 的端到端 A/B 为准。

收益筛选应作为独立实验：先验证无筛选接入，再用冻结规则决定是否启用 custom。规则从全组一致的 C 和配置推导，各 rank 决策一致，且放在昂贵规划/输出分配之前。此前 `peak_share`/接收压力规则只在特定 L0 样本上验证过，不能直接宣称适用于新增 L1；需用未参与调参的真实矩阵验证误判和收益。L1 可用与“值得分流”是两个不同条件。

## 9. 2/4 卡固定 AllToAll 的入口

它们不是 EP8 MoE host 接口。按实际 devices 另生成 policy，不复用八卡 policy：

```bash
export A5_ENABLE_L1=1
python3 scripts/discover_a5_single_port_l1.py --devices 2,3 \
  --root-info "$A5_ROOT_INFO" --output-dir "$A5_SESSION/policy-2"
export A5_L1_CONFIG="$A5_SESSION/policy-2/l1_policy.json"
bash scripts/run_a5_ccu_peer_plan_alltoall.sh \
  --devices 2,3 --available-phys 2,3 --cann-root "$A5_CANN_PATH" \
  --self-copy-mode ccu --bytes 65536 --warmup 3 --iters 10 --repeats 1
# 正确性通过后，固定每 peer 32 MiB，连续提交口径：
bash scripts/run_a5_ccu_peer_plan_alltoall.sh \
  --devices 2,3 --available-phys 2,3 --cann-root "$A5_CANN_PATH" \
  --self-copy-mode ccu --sync-mode batch --bytes 33554432 \
  --warmup 20 --iters 50 --repeats 3
```

只列获准使用的物理卡作为 `available-phys`，避免生成涉及他人卡的 relay。四卡改成四个实际 device，重新发现 policy。L0-only 对照用 `A5_ENABLE_L1=0 bash scripts/run_a5_ccu_peer_plan_alltoall.sh ...` 并保持其余参数相同。静态 `l0_weight:l1_weight` 控制该 pair 分流比例；EP8 的 L1 资源预算并不适用于这个静态权重算法。返回 EP8 前恢复八卡 `A5_L1_CONFIG`，避免拿两卡 policy 跑八卡。

## 10. DeepSeek 执行与回报清单

按顺序完成，失败先定位当前阶段，不直接继续长性能测试：

| 阶段 | 必须回报 |
|---|---|
| 安装 | commit + 本地补丁、wheel/extension 实际路径、route SHA、ABI、Python/Torch/CANN |
| 拓扑 | devices/rank 顺序、rootinfo 摘要、L1 开关、policy generation、manifest/topology 配套路径、实际端口验证 |
| 正确性 | create/数值 PASS、8 rank exit.status、FP8/BF16、实际 L1/L0 relay 行数 |
| 单算子 | C 与 C.T、row_bytes、配置、三次独立结果目录、N/D/M 五口径逐轮数据、actions/moved/端口预算 |
| 模型 | 适配层 diff、每次规划/分配次数、counts/splits/转置验证、两方向耗时边界、DIAG=0 三组端到端数据 |

常见问题与第一检查点：

- `config_allgather` 未完成：还没到算子创建；核对本进程 HCCL_BUFFSIZE、设备占用、首个 runtime 错误，不先归因 planner。
- create/register 失败：保留 `CCU_CREATE_STAGE_*`，确认同一 route/SDK/拓扑版本；create 成功并不说明 payload 正确。
- `validate_plan_device_layout`：核对 C、容量、payload 地址/对齐、行宽、manifest 与 planner 路径是否一致。不要把所有 status=1 都叫 HCCL 内核错误。
- `l1_rows=0`：先看实际 strategy、policy 是否装入、max_actions 与资源预算；可能是合法 direct，不是一定未生效。
- `.close()` 的 stream invalid：检查是否在 Torch/NPU 或共享通信域已销毁后才释放 host。
- 基准很快、模型很慢：先对齐 CPU counts 就绪边界、重复规划、分配和等待；不要把微基准 prepared 收益直接乘 80 当端到端收益。

需要进一步定位时阅读：[单口 L1 设计与预算](A5_CCU_SINGLE_PORT_L1_ALLTOALL.md)、[EP8 bottleneck 与计时诊断](A5_CCU_EP8_BOTTLENECK.md)、[ACL Graph 验证边界](A5_CCU_ACL_GRAPH_VALIDATION.md)。本教程完成前，不需要为了 prefill eager 接入额外解决动态入图。

## 11. service64 dynamic 专项定位：从总差落实到具体阶段

### 11.1 为什么需要这个入口

现场三次同配置运行的 native counts_only 为 1889.9 / 312.2 / 1550.6 µs；同一段计数代码差约六倍。相对稳定的第二次运行中，M 的 host_counts_serial 比 D 快 11.7 µs，但 dynamic 比 N 慢 13.2 µs、比 D 慢 4.4 µs。不能据此宣布 L1 planner 是根因，也不能把不同时段的中位数相减当精确分解。

新增 **`--dynamic-stage-audit`**，只用于 benchmark + 外部矩阵 + legacy counts。它自动启用 combination audit，并只跑五个诊断口径：

| 口径 | 用途 |
|---|---|
| audit_counts_only | 同样的计数获取，无 payload/规划；N/D/M 标签下是同一段操作 |
| audit_dynamic_prepared | counts + 固定 prepared 执行；剥离本次规划 |
| audit_dynamic | counts + 本次规划 + 提交 |
| audit_dynamic_serial | 上一项加每次完成同步；保留原有诊断对照 |
| audit_counts_prepare_only | counts + 构造本次计划，不提交 payload |

诊断改为每轮交错上述口径，轮换模式和口径顺序；普通测试的顺序不变。所有块仍有预热、原有 barrier、块尾同步。计数路径仍是原来的 synchronous all_gather API → stack → CPU → tolist/normalize，**不新增 collective、事件或计数阶段同步**。唯一逐次同步仍只在名字明确为 serial 的对照中。

### 11.2 新增分段和含义

每次调用的 `rank_local_audit[].samples[].stages` 保存：

- `counts_alloc`：八个临时接收 tensor 分配。
- `counts_contiguous`：本地计数输入整理。
- `counts_collective`：原 `dist.all_gather` 调用的主机边界，不代表设备已经完成。
- `counts_stack`：打包为 8×8 device tensor 的主机边界。
- `counts_d2h`：`.cpu()` 的主机边界，**可能等待前置 collective/设备任务**，不能称为 512 字节拷贝的纯带宽耗时。
- `counts_convert`：CPU tolist 和 normalize。
- native：`native_splits`、`native_views`、`native_enqueue`、`native_work_wait`。prepared 对照复用 splits，所以没有本次 `native_splits` 字段。
- custom dynamic：`custom_run`，以及已有 C++ Run 返回的 `cpp_plan_validate_us`、`cpp_submit_us`。前者含 payload 选择、规划和校验；后者是主机 Submit 边界，都不是设备时间。
- custom prepared：`custom_execute`；prepare-only：`prepare_only`。

上述 span 各有 `_wall_us`、`_cpu_us`；CPU 时间只统计当前 Python 调用线程，不含 SDK 工作线程。wall 大、CPU 小提示等待/被抢占的可能，**不能单凭这对数字区分主机调度和设备等待**。C++ 时间是 custom_run 的子区间，不要重复相加。

汇总器对每个块选择一个最慢 rank，所有分段都取这个 rank，保留每轮记录、单次调用 P50/P95/max；不删除极端值。另输出同一次调用的八 rank `start_host_s` 差值（`start_spread_*`），用于判断到达不齐。该字段只对本实验单宿主、同一单调时钟域有效，不是跨机器时钟同步方案；到达差异可能是前一调用延迟的结果，不能直接当成根因。

计时器会引入额外开销，这些数据用于定位，不作为新的性能成绩。修复后仍需关闭诊断回测 §6。

### 11.3 DeepSeek 现场执行

只拉当前分支的诊断脚本即可，**本次不改生产 C++/route/Python 包接口，不需要重编 wheel 或 route**。若现场脚本有未提交补丁，先保留再合并，不用 reset 覆盖。

```bash
cd "$A5_REPO"
git pull --ff-only origin feature/a5-ccu-single-port-l1-alltoall
python3 -m unittest discover -s tests/python/deepep -p test_a5_ccu_dynamic_stage_audit.py -v
python3 -m unittest discover -s tests/python/deepep -p test_a5_ccu_combination_audit.py -v
```

沿用已经成功的八卡单口 L1 policy；不重新选端口，不改流量矩阵，不用 profiler：

```bash
export A5_ENABLE_L1=1
export A5_EID_MAX_RELAYS_PER_PEER=0
export HCCL_BUFFSIZE=64
export A5_DIAG_MATRIX=/home/l00934901/profiling/sec6_dispatch.json
# A5_L1_CONFIG 保持上一轮已验证 policy 的完整路径。
: "${A5_L1_CONFIG:?请设置上一轮已验证的八卡单口L1 policy路径}"
# 一次短验证：确认八 rank 数值、诊断字段和汇总都通过。
bash scripts/run_a5_ccu_eid_alltoallv.sh \
  --suite benchmark --mode all --planner l1_bottleneck --max-actions 56 \
  --matrix-json "$A5_DIAG_MATRIX" --dtype float8_e4m3fn --width 5120 \
  --counts-path legacy --dynamic-stage-audit \
  --warmup 3 --iters 10 --rounds 1 --dynamic-iters 10
```

取脚本打印的**本次**结果目录，不要用可能指向别人任务的“最新目录”：

```bash
A5_DIAG_RUN=/替换为本次输出的/a5-ep8-eid.XXXXXX
python3 scripts/summarize_a5_ccu_combination_audit.py "$A5_DIAG_RUN" \
  --output-json "$A5_DIAG_RUN/dynamic-stage-summary.json" \
  > "$A5_DIAG_RUN/dynamic-stage-summary.log"
tail -1 "$A5_DIAG_RUN/dynamic-stage-summary.log"
```

短验证须同时有 `EP8_EID_ALL_RANKS_AND_EXIT_PASS`、`EP8_COMBINATION_AUDIT_SUMMARY_PASS` 和 `exit.status=0`。随后**只测 service64 dispatch 两次独立进程**，每次将上面的运行参数改为：

```text
--warmup 20 --iters 50 --rounds 3 --dynamic-iters 40
```

每次运行后都单独执行汇总。当前诊断各口径循环次数采用 dynamic-iters，iters 只保留普通 prepared 的参数语义。无需重跑其他八个矩阵或所有计数实现。卡忙则等空闲，不强杀他人任务；若再次出现毫秒级异常，保留，不挑一个“正常块”替换它。

回传两个结果目录中的：`arguments.txt`、`environment.log`、`plans/topology.json`、`dynamic-stage-summary.json`，以及八 rank 的 `results/rank*.json`（需要时再取逐调用数据）。只抄总 N/M 不够定位。

### 11.4 据证据选择修复，不先猜答案

| 观测 | 下一步 |
|---|---|
| counts_alloc/convert 的 CPU 时间显著上升 | 定位临时分配/转换；之后用已有 reuse 计数路径做单项对照，不同时改 planner |
| counts_collective 或 counts_d2h 的 wall 主导，CPU 很小 | 检查到达差、CPU 绑定/限流/共享负载和 HCCL 等待；D2H 包含依赖等待，不直接改成异步就宣称修复 |
| counts_only 已有 300/1500 µs 两档，N/D/M 同时变 | 单独做纯计数、创建 L0 目录后计数、创建 L0+L1 目录后计数的常驻资源对照；当前 harness 即使 N 标签也已创建 custom 目录，不能充当纯 native 进程 |
| fixed prepared 的 action 差距还在，而 cpp_plan_validate 很小 | 优先查 native work.wait 与 custom 提交/READY-DONE 的等待边界；不归因 planner |
| 只有 dynamic 变慢，cpp_plan_validate 稳定增加 | 再优化 C++ 规划或输入转换；检查 custom_run 与其 C++ 子区间之差 |
| serial 把 counts/action 差异转移到 fence，总差缩小 | 说明等待归属变化；不要将某一段减少当真实加速 |
| CPU C 就绪的对照继续有收益 | 模型优先复用现有 CPU C，确认没有重复 gather/D2H；仍测 DIAG=0 端到端，不把微基准额外 collective 当模型必付开销 |

这个入口是诊断补丁，不是“dynamic 已修复”。先定位具体慢在哪个区间，再做单项修改、关闭诊断复测，才算解决。

## 12. 模型卡在 acquire_channels：先查通信域，另查均匀矩阵性能

### 12.1 已知事实与不能据此下的结论

2026-11-12 现场反馈：同 policy 的单算子正常，模型八 rank 均停在 `HcclChannelAcquire`，每 rank 14 条通道、7 条 L1。此时尚未执行 payload，不能用均匀矩阵的拥塞或 planner 动作数解释启动卡死。`L1_ENDPOINT_BIND graph_status=6` 在显式 EID 路径是允许的缺失查询，不等于 ChannelAcquire 返回错误。

`prepare_ccu_urma_eid_alltoallv()` 先建 handle，之后才创建两个 dtype host。两个 host 共享一个 handle 不等于建了两份 L1 通道；只把 combine planner 改为 L0，没有改变该 handle 在创建阶段申请的目录，不能用此实验排除/证明通道竞争。

[HcclChannelAcquire 文档](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/commlib/commopdev/docs/en/api_ref/comm_opdev/control_plane_api/comms_domain_resource_mgmt/HcclChannelAcquire.md)确实描述了通信域/引擎/远端/protocol 的复用标识，但不能仅据这一概述断言现场 9.3 实现必然忽略端点差异。当前 harness 在同一 comm 中同时申请 L0 和显式 L1，并校验通道句柄互不重复；现场成功说明这种组合至少在该创建条件下可行。模型此前已经占用“7 direct+42 relay”需要实际创建日志证明，不能由旧版本配置推断。

### 12.2 创建配置必须按真实 API 语义核对

对模型与 harness 记录：实际 wheel/route SHA、已加载 libhccl/libhcomm 路径、通信组名称、group rank 到物理卡顺序、创建时 pg_options.hccl_config、环境变量、第一次 native collective/route prepare 的顺序。comm 指针只能比较同一进程内的身份，不用于跨 rank 判断是同一通信域。

当前 harness 默认 `A5_CCU_PG_MODE=env`，通过创建前的 `HCCL_OP_EXPANSION_MODE=CCU_SCHED` 配置，并非必需显式传 config6。模型可能用通信域级配置覆盖环境。[HcclCommConfig 文档](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910beta1/API/hcclug/hcclcpp_07_0047.html)中创建字段 `hcclOpExpansionMode=6` 表示 CCU_SCHED；查询枚举 `HcclOpExpansionMode` 不是同一个数值编码，不能把查询 raw=5/6 与创建参数直接比较。以现场 9.3 头文件和实际 torch_npu 配置传递实现核对，不能盲改模型全局通信模式。

本次模型命令是 `HCCL_BUFFSIZE=1024`，此前许多 harness 使用 64，二者不是完全相同的资源条件。也要核对通信域是否显式设置 buffer size 覆盖了环境。2026-11-12 模型对照已发现：全局改为 64 会在进入 custom 路径前触发原生 MoE 的 `Get WinSize failed`。因此不要把单算子用的 64 直接套到完整模型；该失败不能用于判断 L1 acquire 的 buffer 敏感性。后续保留模型已能运行的全局设置，只在确认的 EP group 创建配置处做单变量对照，详见 §12.6。

### 12.3 DeepSeek 的最小排查顺序

先收集现有日志，不重跑完整端到端：

1. 给每次 Buffer/handle 创建记录 engine 实例编号、进程、EP group 名称、组内 rank/size、manifest/topology 摘要和调用位置。确认是否单次创建，是否混入旧 engine，所有 rank 是否使用同 generation。
2. 找到**真正创建 EP group**的代码，贴其 pg_options；仅看适配层或 shell 环境不够。确认模型工作是否仍在初始化其他 collective 时进入 route prepare。
3. 核对 `PEER_PATH_DESC`：每条 local/remote EID、物理卡和 die 必须与对端反向一致；policy 的 28 条无向记录展开成 56 条有向路径后统计端口，不能只统计 src 字段。
4. 若有一次仍存活的卡死现场，抓各 worker 的原生线程栈及对应时段 plog（必要时两次快照）。需要区分 SDK 等待对端、资源/通知等待、进程内锁；不要仅从“无返回”推断复用键冲突。

有卡后在新的进程中逐项改变一个条件，只测初始化/小数据正确性：

| 对照 | 保持/改变 | 能说明什么 |
|---|---|---|
| 模型原 EP group、L0-only 目录 | 同模型配置；L1=0 且重新生成七条 direct 的目录 | 模型的公共 CCU 建链是否可用；只改 planner 不算此对照 |
| 模型原 EP group、L0+L1、显式 EP buffer 200/512/1024 | 保持全局 HCCL_BUFFSIZE=1024 及其他组配置不变；各档先确认原生 MoE 能通过 | 是否受该 EP 域 buffer/常驻资源条件影响；只改 shell 不能证明改变了该域 |
| 模型初始化阶段，专用 custom EP8 通信域、完整 L0+L1 目录 | 同物理卡/rank 顺序、同 SDK 配置；原模型 EP group 不变 | 若成功，问题与原通信域配置/状态/初始化顺序有关；仍不能单独证明缓存键冲突 |
| 最小程序同一 comm，先执行模型用过的 native collective，再 prepare L0+L1 | 与全新 comm 直接 prepare 对照 | 是否由同域的先前通信/懒初始化触发 |

专用通信域应在全部参与进程一致的初始化位置、按框架要求的创建顺序建立，不能放在某 rank 的首次 MoE 调用里。诊断时让 **custom 的 L0+L1 整体**使用此域，不是只把 L1 channel 挪到另一个 comm；后者需要修改执行器资源契约，不是适配层改一行。独立域会增加资源，需要独立 close/destroy 生命周期，不能保证必然解决资源不足。一次 create 卡死后停止本次测试，不在同一失败域内反复申请。

先定位启动阻塞，再测 uniform。跨 die、均匀门槛、动作数修改不作为本轮启动修复。

### 12.4 uniform 的正确字节账和端口模型

`U[i][j]=3072`，FP8 row_bytes=5120，self 不计入网络：

| 项目 | 字节数 |
|---|---:|
| 原跨卡 payload，56×3072×5120 | 880,803,840 |
| 改走 L1，32704×5120 | 167,444,480 |
| 留在 L0 | 713,359,360 |
| 分流后的 L0+L1 payload 合计 | 880,803,840 |

旧汇总器把 `plan.relay_bytes` 显示成 `extra_bytes`，对 L1 有误导。本次改为 `moved_bytes`，并独立打印 `extra_network_bytes=transmitted_network_bytes-original_network_bytes`。仅 L1 时此增量为零；组内 L0 relay 的额外一跳才增加此处的网络字节账。这些字段不是交换网络内部物理 hop-byte 计数，也不表示 L1 不产生控制/协议代价。

因此，不能用 167 MB 除以多出的 49.4 µs 证明“额外字节按 UB 线速发送”。prepared 慢约 10.8% 是实测事实，但原因仍可能是更多数据操作、共享注入资源、端口/交换路径竞争、控制/完成等待等。host_counts 小于 prepared 也不能证明规划为零，两者重叠和计时边界不同。

planner 已以 `(rank, die, port)` 为资源累计 TX/RX，并非把每个逻辑 channel 当独占端口。例如 rank0 两口分配 4/3 条流，每流搬 584 行：L1 两口分别为 2336/1752 行，L0 各边为 2488 行；当前模型已计入这层共享。没有计入的可能是两个口共同的更上层注入瓶颈及协议成本，需要实测而不是再次重复计端口共享。

“同 IO die”是当前执行器的支持边界，不是已证明的硬件永远限制；放开校验不等于支持跨 die。跨 die 需要验证各 die 的 CCU 实例、线程、通道、kernel、内存访问以及完成依赖。未来可用八个口时，也应按真实源 TX/目的 RX 资源分配，不强制“一口只准一条逻辑路径”。

两个 dtype host 顺序执行时，使用同一份物理资源预算，不因 host 数量为二就把每口容量除二；并发执行则需调度器分配不重叠预算或串行化。适配层要同时传 `topology.get('l1_resources')` 和 `topology.get('l1_budget')`，不能只传前者丢掉控制器预算。

### 12.5 收益准入暂不硬编码旧 F2 规则

`hot_share>=0.28 && recv_pressure<=2.0` 是旧样本/路径条件下的经验筛选，不是 L1 的必要条件。可以作为独立保守 A/B 策略，但不能同时改规则、通信域和跨 die，再把结果归给某一项。

uniform 当前路径配置下没有收益，可暂时不在模型启用这类计划；生产通用代价模型应考虑动作数、移动字节数、端口累计负载、共享注入上限和固定完成成本。不能直接按“L1 每搬一行额外多发一行”收费。若要确认 uniform 的性能原因，使用同 handle 的 D/M、逐轮配对、实际端口计数与设备时间线，保持计划和端口策略固定。


### 12.6 2026-11-12 排查回报后的执行顺序

本节更新 §12.3，不要求重复已经完成的静态核查。

**已确认：** 每进程单次 Buffer/handle 创建；两个 dtype host 均在 handle 创建之后。28 对 L1 端点反向匹配零差异，每 rank 两个端口分配为 4/3。原模型 EP group 配合重新生成的 L0-only 七通道目录能启动；L0+L1 十四通道目录此前停在 acquire。故阻塞与增加 L1 目录相关，但尚未区分端点处理、资源、通信域状态或初始化时序。尚未进入 dispatch/combine 执行，不应归因两方向 payload 冲突。

**配置证据：** 模型创建路径给 EP group 传入 `hccl_buffer_size=200, group_name=ep`；脚本环境为 `HCCL_BUFFSIZE=1024`。前者是已查到的创建输入，实际 SDK 生效值仍应由现场创建日志/配置传递核对。全局 64 那次在原生 `aclnnMoeDistributeDispatchV4` 的 window 检查失败，没有到达 custom prepare；不能据此断定 EP buffer 200 被覆盖，也不能证明 200 不可用，因为环境可能影响其他通信域。

**已有 harness 能力的边界：** harness 在 config all_gather 之后才申请 L0+L1，证明“此前运行过任意 native collective 就必然失败”不成立。它没有复现模型完整 EP group 配置、原生 MoE 初始化及其他通信域的常驻占用，不能替代模型内专用 custom 通信域实验。

按以下顺序开展，只测初始化及短正确性，不跑端到端性能：

1. **先做 harness 配置对照。** 保持成功样本的 policy、设备顺序、wheel/route、环境和 counts 不变，记录默认创建配置，然后仅在实际 `ProcessGroupHCCL.Options().hccl_config` 中加入 `hccl_buffer_size=200`。在所有 rank 创建进程组前生效；不要用 `HCCL_BUFFSIZE=200` 代替。确认解析该 key 的现场 torch_npu 实现，并记录传入配置。默认/200 使用独立进程配对；若都过，再单独核对 `group_name` 等剩余模型配置，不一次改多个字段。失败则在仍存活时收集各 rank 原生栈和 plog；成功只排除“200 单独足以致错”，不排除模型资源组合。
2. **复现一次模型原配置并保留现场。** 全局仍为 1024、EP 配置仍为 200，使用原 L0+L1 目录。记录各 rank acquire 前后的时间和创建序号；卡住时在清理前采集所有 worker 的原生线程栈及同期 plog，两次快照间隔约 5–10 秒。py-spy 可补 Python 调用位置，但不能代替 C++ 栈。工具若使用暂停式 attach，记录暂停时间，取栈后恢复进程，避免把诊断干预误认为原始等待原因。
3. **模型内做专用域对照。** 在所有 rank 一致的模型初始化位置创建一份专用 custom EP8 域，保持 rank 顺序和明确的创建配置；custom 的完整 L0+L1 handle 使用该域，原生 MoE 继续使用原 EP 域。其余配置不改。若通过，优先调查原 EP 域既有状态/资源复用/初始化时序，不直接宣布通道缓存键冲突；若仍挂，继续查模型整体资源和调用时序。专用域没有自动消除全设备资源竞争。
4. **按证据再做 EP buffer 扫描。** 保持模型全局 1024 及其他域不变，仅改变被测 EP 域显式 buffer 200/512/1024；每档用新进程，先确认能越过原生 MoE 前置阶段。未到达 acquire 的档位标记为前置失败，不参与 acquire 成败对比。不要同时改域、policy、planner 或 combine 开关。

每组回报：实际创建配置、环境、库/manifest/policy 摘要、每 rank 最后阶段、acquire 耗时或采样等待栈、是否完成小数据校验和清理。先取得第一组有区别的对照，再缩小实验范围，无需机械跑完所有档位。

适配层补传 `l1_budget` 是独立的契约完善：当前文件值为 None 时不改变默认预算，也不可能修复发生在 host 创建之前的 acquire 阻塞。后续带非空预算的拓扑必须原样传入。d46cd86 的文档/汇总器更新未改变生产 C++，现场 ac8da373 wheel 未因这一更新重编，本身不是版本错误证据；仍须固定实际加载库的 SHA。

## 13. Rank graph L1 聚合端点接入

当前以尽快接通模型为目标，执行 [A5_CCU_RANK_GRAPH_L1_MODEL.md](A5_CCU_RANK_GRAPH_L1_MODEL.md) §3–§7。不要继续把首次 MoE 前的 comm init error4 当作 acquire_channels 的同一失败；本轮恢复首次 MoE 初始化位置，保持专用域与buffer设置，只更换 L1端点来源。

新 route 默认只接受 l1_graph。旧显式单口实验须设置 A5_L1_ENDPOINT_MODE=explicit_single；该开关不改变 HCCL 的全局 rank graph、不修改系统拓扑文件，也不是运行中热切换。
