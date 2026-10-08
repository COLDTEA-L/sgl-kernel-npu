# 显式 peer-plan CCU AllToAll：两卡 / 四卡

分支：`feature/a5-ccu-peer-plan-alltoall-4rank`。本版在已验证的两卡 prepared-plan
数据面上扩展 peer 维度，不改 HCCL、不改 UBUS route table，不加载新 ko。
两卡原接口保留；四卡和 direct-only 使用新增 peer-plan API。

## 1. 范围、数据流与约束

输入、输出均为 contiguous FP32 `[k, elements_per_peer]`，`k=2` 或 `4`。
源 rank `s` 的 `send[d]` 写入目的 rank `d` 的 `recv[s]`；self 分片在本地 D2D copy。
`--bytes` 是**每个目的 rank**的数据量，包含 self；四卡 4 MiB/peer 对应 16 MiB tensor，
每 rank 网络发送总量是 12 MiB，而非把 4 MiB 平分给三个 peer。

对每个远端 peer，一条 direct 加零条或多条显式 relay。relay 不得在通信域内。
上限 `L=(N-k)//(k-1)` 是**每对通信 peer 的 relay 数**；一机八卡，四卡 L=1，
两卡 L=6。这里接受用户给出的分配，不实现自动选卡/控制器。
`--available-phys` 默认一机 0..7，可传实际允许借用的物理卡集合；脚本不自动判断卡是否空闲。
同一个源 rank 的不同 peer 不复用同一 relay；不相交的卡对可以共享 relay。

四卡手工分配示例（只是可修改的示例，不是算法写死）：

| 卡对 | direct | relay |
|---|---|---|
| 0 ↔ 1 | 0 ↔ 1 | 0 ↔ 4 ↔ 1 |
| 2 ↔ 3 | 2 ↔ 3 | 2 ↔ 4 ↔ 3 |
| 0 ↔ 2 | 0 ↔ 2 | 0 ↔ 5 ↔ 2 |
| 1 ↔ 3 | 1 ↔ 3 | 1 ↔ 5 ↔ 3 |
| 0 ↔ 3 | 0 ↔ 3 | 0 ↔ 6 ↔ 3 |
| 1 ↔ 2 | 1 ↔ 2 | 1 ↔ 6 ↔ 2 |

每个 peer 的默认分片权重为 direct `2`、**每条** relay `1`。
每个 peer 独立分配完整 byte budget，按 256-byte 边界取整，最后一条吃余数。
direct-only 每个 peer 只有一条 Channel，传完整分片。

初版限制：每 rank 的所有 Channel 必须在同一个本地 IO die 上；拓扑可能使某些
用户分配不满足这个条件，runtime 会在 Acquire 前明确拒绝，不会偷偷换 relay。
relay 的 ingress/egress 必须同一个 relay die，但该 die 可以不同于通信端点的 die。
一 kernel 当前最多 14 条 Channel（48 个 launch 参数限制）；四卡 direct+1 relay
每 rank 6 条、两卡 direct+6 relay 每 rank 7 条，均在上限内。未来大集群需增加
multi-die/grouped launch 调度，不能仅改变 `k` 或声称没有任何硬件资源上限。

## 2. 拉取与构建

如果本地指导文档差异阻止切分支，先运行以下代码块（备份 tracked 差异，保留
untracked 文件；不需要在内网机器 push，不要恢复旧 stash）。括号中的 `set -e`
只影响子 shell，避免失败时退出当前 Docker 交互 shell。每行末尾不要额外加反斜杠。

```bash
(
set -e
cd /home/l00934901/sgl-kernel-npu
git stash push -m "backup-before-4rank-update-$(date +%Y%m%d_%H%M%S)"
BRANCH=feature/a5-ccu-peer-plan-alltoall-4rank
git fetch origin "${BRANCH}"
git switch "${BRANCH}"
git merge --ff-only "origin/${BRANCH}"
git rev-parse HEAD
git rev-parse "origin/${BRANCH}"
git status --short
)
```

两个提交号应相同。没有本地差异时也可以直接使用下面的拉取命令。

```bash
cd /home/l00934901/sgl-kernel-npu
git fetch origin feature/a5-ccu-peer-plan-alltoall-4rank
git switch feature/a5-ccu-peer-plan-alltoall-4rank
git pull --ff-only origin feature/a5-ccu-peer-plan-alltoall-4rank
source /usr/local/Ascend/cann-9.1.T560/set_env.sh
unset LD_PRELOAD A5_CCU_PREPARED_ALLOW_LAZY_STREAM A5_URMA_TP_TRACE_PREFIX

# HCCL_REPO 只用作旧对象式 CCU custom-op 的构建框架，不需要拉取/编译 patched libhccl。
HCCL_REPO=/home/l00934901/hccl \
bash scripts/build_a5_ccu_urma_route_probe.sh \
  --install --install-path /usr/local/Ascend/cann-9.1.T560

DEEPEP_SINGLE_OP=explicit_multipath_all2all_ccu bash build.sh -a deepep Ascend950
WHEEL=$(ls -t output/deep_ep-*.whl | head -1)
test -f "${WHEEL}"
python3 -m pip install --force-reinstall --no-cache-dir --no-deps "${WHEEL}"
```

若旧仓有未提交修改，先备份/stash，不要直接 reset 或删 untracked。
构建脚本的 offline cann-cmake 与 UTF-8 源码预检保持原有行为。
需要重新装 route package 和 wheel；**不需要重编 HCCL**。

**2026-10-08 worker 入口修复只改 Python 测试与文档**：若第 3 节独立 peer-plan
ABI 已通过，本次仅拉取更新即可，不需要重复编译/安装 route package 或 wheel。
先做无需 NPU 的入口回归，再执行第 4 节：

```bash
python3 tests/python/deepep/test_a5_ccu_peer_entrypoint_cpu.py
python3 tests/python/deepep/test_a5_ccu_peer_plan_alltoall.py --help
```

应显示 3 个测试 `OK`，help 包含 `--manifest`、`--available-cards`、`--run-dir`、
`--graph-backend`，不能显示旧两卡 benchmark 的 `--implementation`。

若之前已经被迫退出 Docker，在宿主机使用 `docker exec -it sglang_yuanwen_old bash`
重新进入（容器名按实际环境修改）。Git 报错不会主动停止容器；编译脚本用
`bash scripts/...` 执行，不要 `source` 编译脚本或将带 `exit` 的恢复代码直接粘贴进交互 shell。

## 3. 验证新增 API（独立 ABI，不能只检查旧 ABI 11/12）

```bash
export A5_CCU_ROUTE_PROBE_LIB=/usr/local/Ascend/cann-9.1.T560/opp/vendors/cust/lib64/liba5_ccu_urma_route_probe.so
python3 - <<'PY'
import ctypes, os
from pathlib import Path
import torch
import deep_ep.deep_ep_cpp as ext
from deep_ep import Buffer

route = ctypes.CDLL(os.environ['A5_CCU_ROUTE_PROBE_LIB'], mode=ctypes.RTLD_GLOBAL)
deep = ctypes.CDLL(str(Path(ext.__file__).resolve()))
for library, name in [(route, 'A5CcuPeerPlanAbiVersion'), (deep, 'A5DeepEpPeerPlanAbiVersion')]:
    fn = getattr(library, name); fn.restype = ctypes.c_int
    print(name, fn()); assert fn() >= 1
assert hasattr(Buffer, 'prepare_ccu_urma_peer_plan')
for k in (2, 4):
    x = torch.empty((k, 1024), device='meta', dtype=torch.float32)
    y = torch.ops.deep_ep.ccu_urma_peer_plan_alltoall(x, 1, [])
    assert y.shape == x.shape and y.dtype == x.dtype
print('peer-plan Meta PASS; not a hardware correctness test')
PY
```

## 4. 先测四卡 direct-only

仅通信域卡需要运行 rank 进程，relay 上没有用户进程。
确认通信卡可用后执行；构建/预检不自动抢卡或改共享路由。

```bash
bash scripts/run_a5_ccu_peer_plan_alltoall.sh \
  --devices 0,1,2,3 \
  --bytes 4194304 --warmup 1 --iters 3 --repeats 1 \
  --timeout-seconds 600 \
  --output-root /home/l00934901/profiling
```

不传 `--relay-map` 就是 direct-only，也不依赖 hccn EID 文件。
逻辑 rank 与 `--devices` 的顺序对应，可使用其他四张卡或任意两张卡。

### 4.1 `unrecognized arguments` / 错误进入旧两卡脚本

若日志出现 `test_a5_ccu_urma_multiroute_all2all.py: error: unrecognized arguments:
--manifest ... --available-cards ... --run-dir ... --graph-backend ...`，且最终 status=137，
这不是四卡 Channel/CCU 失败。旧版 peer 测试导入了两卡 benchmark 的辅助函数；
两卡 benchmark 在模块初始化时 `prepare_runtime()` 修改环境并执行
`os.execvpe(..., __file__, ...)`，把四卡 worker 替换成了两卡入口。新参数因而被
旧 parser 拒绝，分布式启动器随后未能正常结束，外层 timeout TERM/KILL。
仅凭 137 不能断言 OOM，本次日志中明确出现了 timeout 的 Killed。

修复版将 barrier/profiler 移到无启动副作用的 `a5_ccu_peer_test_support.py`，
peer 入口先解析自己的参数再导入 NPU runtime，不再导入旧 benchmark。
拉取后执行第 2 节入口回归，再重跑本节同一条 direct-only 命令即可。
检查新日志中的阶段标记：

```bash
RUN_DIR=$(ls -dt /home/l00934901/profiling/a5_ccu_peer_plan_* 2>/dev/null | head -1)
if test -n "${RUN_DIR}" && test -f "${RUN_DIR}/cases/peer_plan_r1.log"; then
  grep -nE 'PEER_CASE_PHASE|PEER_CASE_FAILURE|RESULT_JSON|unrecognized arguments' "${RUN_DIR}/cases/peer_plan_r1.log"
fi
```

至少应先看到 `import_runtime`，随后是 `communicator_init`、`prepare_plan`；
只有进入这些阶段后才开始判读通信域/建链/数据面错误。四个 rank 都输出 correctness
PASS 的 `RESULT_JSON` 才算此 case 完整通过。

## 5. 四卡显式 direct+1 relay

下面文件已随仓提供，可自行编辑卡对/relay/plane；key 为升序**物理**卡对，
不是逻辑 rank。省略的卡对只走 direct。每张卡必须使用三张互不相同的外部 relay。

```bash
cat docs/topology/a5_peer_plan_4rank_example.json

# 先解析真正的 topology edge/port/EID，尚不打流。
python3 scripts/prepare_a5_ccu_peer_plan.py \
  --devices 0,1,2,3 \
  --relay-map docs/topology/a5_peer_plan_4rank_example.json \
  --topology docs/topology/a5_hccn_device_topology_raw.txt \
  --topology-json /usr/local/Ascend/driver/topo/950/atlas_950_1.json \
  --output-dir /home/l00934901/profiling/peer_plan_preview
column -s $'\t' -t /home/l00934901/profiling/peer_plan_preview/peer_plan.tsv

bash scripts/run_a5_ccu_peer_plan_alltoall.sh \
  --devices 0,1,2,3 \
  --relay-map docs/topology/a5_peer_plan_4rank_example.json \
  --bytes 4194304 --warmup 1 --iters 3 --repeats 1 \
  --timeout-seconds 600 \
  --output-root /home/l00934901/profiling
```

若存在多个合法 topology/EID 候选，resolver **拒绝猜测**。把 relay 改成对象，例如
`{"physical_device":4,"plane":1}`，明确 relay IO die。plane 不等于源卡 IO die。
若指定边不通或跨本地 die，保留失败结果，改用户分配或只走 direct；不要自动用 native2 冒充所选 relay。

## 6. 两卡兼容与任意合法 relay 数

新接口同样支持 `2,3`，示例明确指定 relay `0,1`：

```bash
bash scripts/run_a5_ccu_peer_plan_alltoall.sh \
  --devices 2,3 --relay-map docs/topology/a5_peer_plan_2rank_example.json \
  --bytes 4194304 --warmup 1 --iters 3 --repeats 1 \
  --output-root /home/l00934901/profiling
```

`{"2,3":[0]}`、`{"2,3":[0,1,4]}`、五条或六条均允许，不要求偶数。
两卡 direct-only 只需 `--devices 2,3`，不传 map。
原来的 `run_a5_ccu_explicit_multipath_alltoall_perf.sh` 与旧 APIs 未替换，仍按旧两卡行为运行。

## 7. 性能 / profiling / graph 分开验证

性能使用初始化一次、100 次独立 warmup、20 次独立执行；只给正式 20 次开 profiler。
每次执行后 synchronize，不含资源准备、warmup 或 capture。正式 warmup 前还有三次
变化 payload 的正确性检查，它们不计时；统计整段 HCCN 数据量时需要计入这些额外调用。

```bash
bash scripts/run_a5_ccu_peer_plan_alltoall.sh \
  --devices 0,1,2,3 \
  --relay-map docs/topology/a5_peer_plan_4rank_example.json \
  --bytes 4194304 --warmup 100 --iters 20 --repeats 3 --profile \
  --output-root /home/l00934901/profiling
```

prof 在结果目录的 `r1/profiling/rank0/` 到 `rank3/`，每个 repeat 独立目录。
报告中的 host-call us 含提交与同步开销；MindStudio CCU task 时长才是 device timing。
不把 host timing 除 payload 当作 link bandwidth。

单独验证新接口 graph capture/replay：

```bash
bash scripts/run_a5_ccu_peer_plan_alltoall.sh \
  --devices 0,1,2,3 \
  --relay-map docs/topology/a5_peer_plan_4rank_example.json \
  --graph-backend aclgraph \
  --bytes 4194304 --warmup 1 --iters 3 --repeats 1 \
  --output-root /home/l00934901/profiling
```

新四卡图支持仍需此有卡测试确认；旧两卡 capture PASS 不自动等于四卡 PASS。
capture stream 的 bind/warmup 在 capture 外进行，replay 修改输入验证非旧输出。
每个 plan 的执行须串行使用；不能在多 stream 同时使用共享 Channel 的 notify 槽。

## 8. 结果检查、中断后分析、物理转发确认

```bash
RUN_DIR=$(ls -dt /home/l00934901/profiling/a5_ccu_peer_plan_* 2>/dev/null | head -1)
test -n "${RUN_DIR}" && test -d "${RUN_DIR}"
python3 scripts/analyze_a5_ccu_peer_plan.py --run-dir "${RUN_DIR}"
column -s $'\t' -t "${RUN_DIR}/peer_plan_results.tsv"
sed -n '1,240p' "${RUN_DIR}/peer_plan_report.md"
grep -HnE 'PEER_CASE_PHASE|PEER_PATH_DESC|PEER_PATH_CHANNEL|PEER_PLAN_BOUND|PEER_CASE_FAILURE|RESULT_JSON|PEER_GRAPH_' \
  "${RUN_DIR}"/cases/*.log
```

分析必须看到所有 k 个 rank 的 RESULT_JSON 和 status=0 才算正确性 PASS。
Ctrl-C 中断时，无 status 文件的 case 标记 INTERRUPTED，已有完整 case 仍可分析。

`plans/peer_plan.json` 的 `relay_edges` 保存 src/dst/relay 真实 die/port、拓扑边及完整 EID。
物理 relay 验证沿用已有 HCCN before/after 采样，在每条预期四端口链上比较 tx/rx flit；
本新脚本**不自动采 HCCN**。需要以这些端口和工作负载关联，而非某张卡总流量。
仍须排除公共机器其他用户噪声，不能把本脚本数据正确性 PASS 单独称为物理转发证明。

## 9. API、调用链与控制器扩展点

```text
用户明确的 physical-pair -> relay list JSON
  -> prepare_a5_ccu_peer_plan.py
  -> topology edge join + endpoint EID inventory
  -> 完整对称 peer_plan.tsv（direct 在前，relay 在后）
  -> Buffer.prepare_ccu_urma_peer_plan()（图外）
  -> HcclCcuUrmaPeerPlanCreate
  -> ValidatePeerPlan + GetPeerPlanResources
  -> RankGraph direct CommLink -> 显式 relay endpoint pair
  -> 必要 Thread Acquire -> 一批 ChannelAcquire -> ChannelGetStatus
  -> RouteKernel Register + RegisterFinish -> plan_handle

Buffer.ccu_urma_peer_plan_alltoall[_out](send, ..., handle)
  / torch.ops.deep_ep.ccu_urma_peer_plan_alltoall(send, handle, weights)
  -> 当前 stream host-queue flush + tensor recordStream
  -> HcclCcuUrmaPeerPlanExecute（只查已绑定资源，不建链、不读文件）
  -> 每 peer 独立 weighted offsets/bytes -> self D2D
  -> CCU launch -> 交换每 Channel 的 recv base/token
  -> 所有 peer/path WriteNb -> 所有 completion wait -> peer completion handshake
```

| API | 输入 | 行为 |
|---|---|---|
| prepare_ccu_urma_peer_plan | plan_id, TSV 绝对路径, N | 完整、对称计划；进程内 handle |
| bind_ccu_urma_peer_plan | handle | 图外绑定当前执行/capture stream |
| ccu_urma_peer_plan_alltoall | send, handle, path_weights=[] | functional torch custom op，分配 output |
| ccu_urma_peer_plan_alltoall_out | send, recv, handle, path_weights=[] | 复用 output；测试性能用此入口 |
| HcclCcuUrmaPeerPlanExecute | buffers, elements/peer, rankSize, dtype, comm, stream, handle, weights/count | validates owner/shape/stream before enqueue |

非空 weights 是**本 rank**按 peer 升序、各 peer 的 direct+relay 顺序展开，必须与 channelCount 一致。
默认空 list 直接用 manifest 权重，不是 path_mask。初版不接受 0 权重以禁用路径，
direct-only 用独立 plan；policy 只能重分已有路径，不能凭空增加 Channel。
host weights 在 capture 时固化，replay 期间修改 Python list 不会动态改图内分片。
后续控制器可以输出同一 JSON/TSV，图外准备新 plan，再选择 plan/weights；自动计算分配、
device-side 动态 policy 和 multi-die 调度均是独立扩展，不在本版伪装完成。

计划/Channel/kernel 与 communicator 及 stream 绑定，直到进程退出保持资源；没有单独
Destroy API，因此不要无限创建不同 plan_id/stream。保持 Buffer/communicator 存活，所有
异步执行及 profiler 完成后再销毁 communicator。此版不是带 tiling.cpp 的标准 Ascend C
算子；它延续此前 prepared-plan torch custom op 路线。

两卡实现细节见 [两卡设计](A5_CCU_TWO_RANK_PREPARED_PLAN_DESIGN.md)。新增资源/数据面在
`examples/a5_ccu_urma_route_probe/op_host/peer_plan_runtime.cc`、`utils.cc`、
`common/peer_plan.h`；沿用 `op_kernel_ccu/route_kernel.cc` 的事件与 notify 机制。

## 10. 无卡回归

```bash
c++ -std=c++14 -Wall -Wextra -Werror \
  -Iexamples/a5_ccu_urma_route_probe/common \
  tests/python/deepep/test_a5_ccu_peer_layout.cpp -o /tmp/a5_peer_layout_test
/tmp/a5_peer_layout_test
python3 tests/python/deepep/test_a5_ccu_peer_plan_cpu.py
python3 tests/python/deepep/test_a5_ccu_peer_meta.py
```

开发容器已验证旧对象式 CCU runtime 与 DeepEP 编译、布局和 policy 校验、Meta/fullgraph
形状检查；没有 NPU 的容器无法证明四卡 Acquire、实际转发或 graph replay。
