# A5 CCU+URMA prepared-plan 显式多路径 AllToAll 操作指南

## 1. 当前实现和边界

当前可运行主线不再修改 `libhccl.so`，也不再让标准 Ascend C AIV kernel 进入
T560 未开放的 MC2 资源入口。实现分为两层：

```text
图外控制面
  显式 direct/relay CommLink
    -> HcclChannelAcquire
    -> HcclThreadAcquireWithStream
    -> HcclCcuKernelRegister/Finish
    -> (comm, plan_id, stream) 对应的已注册资源
    -> plan_handle

图内执行面
  plan_handle + tensor + 本次 path weights
    -> deep_ep prepared torch op
    -> 自有薄 host runtime
    -> 在当前 stream 上 HcclCcuKernelLaunch
    -> 多个 Channel WriteNb
```

它已经具备：

- 1 条 direct 加 1..12 条显式 relay；
- relay 卡号由命令行/manifest 指定，没有写死；
- direct 与每条 relay 默认按 `2:1` 分片；
- Channel 和 CCU kernel 只在 plan 绑定 stream 时创建，执行阶段只 launch；
- eager 和 ACLGraph capture/replay；
- 不安装 `ubus.ko`，不修改全局 route table；
- 使用系统 HCCL communicator，不需要实验版 HCCL runtime。

准确边界：它目前是 PyTorch 图可见的 prepared custom op，不是最终的原生
GE/Ascend C `op_def + tiling.cpp + AIV kernel` 形态。T560 AIV kernel 不能直接调用
host runtime；在缺少官方 host-task/doorbell 接口时，不能把这一层伪装成已经完成的
原生 Ascend C 算子。

ABI 要求：route runtime ABI >= 5，DeepEP ABI >= 9。

## 2. 只拉取 sgl-kernel-npu（不要拉取 HCCL）

正常干净仓库：

```bash
cd /home/l00934901/sgl-kernel-npu
BRANCH=feature/a5-ccu-explicit-multipath-alltoall

git fetch origin "${BRANCH}"
git switch "${BRANCH}"
git merge --ff-only "origin/${BRANCH}"

git rev-parse --short HEAD
git status --short

# prepared-plan ABI5/DeepEP ABI9 的最低源码门禁
git merge-base --is-ancestor fa86838 HEAD
grep -n 'prepared-plan cases support' \
  scripts/run_a5_ccu_explicit_multipath_alltoall_perf.sh
```

`??` 未跟踪文件不会阻止 fast-forward，除非它将覆盖远端同名文件。不要在公共机器上
使用 `git clean -fd`。若存在 tracked 修改，先保存补丁再处理：

```bash
STAMP=$(date +%Y%m%d_%H%M%S)
git diff > "/tmp/sgl-kernel-npu_worktree.${STAMP}.patch"
git diff --cached > "/tmp/sgl-kernel-npu_index.${STAMP}.patch"
git status --short
```

本方案不要求 fetch、switch、编译或安装实验 HCCL 分支。已有且健康的
`/home/l00934901/hccl` 仅被自定义 CCU package 构建脚本用作 CMake/头文件工具树；
运行时仍使用当前 CANN 的系统 HCCL。

## 3. 编译并安装 route runtime

```bash
cd /home/l00934901/sgl-kernel-npu
source /usr/local/Ascend/cann-9.1.T560/set_env.sh
unset LD_PRELOAD

HCCL_REPO=/home/l00934901/hccl \
bash scripts/build_a5_ccu_urma_route_probe.sh \
  --install \
  --install-path /usr/local/Ascend/cann-9.1.T560
```

脚本会强制检查三个 prepared-plan 入口以及 ABI 5：

```text
HcclCcuUrmaExplicitMultipathPlanCreate
HcclCcuUrmaExplicitMultipathPlanBindStream
HcclCcuUrmaExplicitMultipathPlanExecuteV2
A5CcuUrmaPreparedPlanAbiVersion() >= 5
```

也可以手工检查已安装文件：

```bash
ROUTE_SO=/usr/local/Ascend/cann-9.1.T560/opp/vendors/cust/lib64/liba5_ccu_urma_route_probe.so

nm -D "${ROUTE_SO}" | grep -E \
'HcclCcuUrmaExplicitMultipathPlan(Create|BindStream|ExecuteV2)|A5CcuUrmaPreparedPlanAbiVersion'

ROUTE_SO="${ROUTE_SO}" python3 - <<'PY'
import ctypes
import os

lib = ctypes.CDLL(os.environ["ROUTE_SO"], mode=ctypes.RTLD_GLOBAL)
version = lib.A5CcuUrmaPreparedPlanAbiVersion
version.restype = ctypes.c_int
print("prepared-plan route ABI:", version())
assert version() >= 5
PY
```

## 4. 编译并安装 DeepEP wheel

```bash
cd /home/l00934901/sgl-kernel-npu
source /usr/local/Ascend/cann-9.1.T560/set_env.sh
unset LD_PRELOAD

DEEPEP_SINGLE_OP=explicit_multipath_all2all_ccu \
bash build.sh -a deepep Ascend950

WHEEL=$(ls -t output/deep_ep-*.whl | head -1)
python3 -m pip install --force-reinstall --no-cache-dir --no-deps "${WHEEL}"
```

在开发 Docker 中先进入包含 torch/torch_npu 的环境：

```bash
source /opt/conda/bin/activate cam_py311_pt28
```

验证 ABI、Python wrapper 和 Meta dispatch：

```bash
python3 - <<'PY'
import ctypes
from pathlib import Path
import torch
import deep_ep.deep_ep_cpp as ext
from deep_ep import Buffer

path = Path(ext.__file__).resolve()
lib = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
version = lib.A5DeepEpExplicitMultipathAttrAbiVersion
version.restype = ctypes.c_int

print("loaded deep_ep_cpp:", path)
print("prepared-plan DeepEP ABI:", version())
print("bind API:", hasattr(Buffer, "bind_ccu_urma_explicit_multipath_plan"))
print("policy API:", hasattr(Buffer, "ccu_urma_prepared_multipath_alltoall_policy_out"))

x = torch.empty((2, 16), device="meta", dtype=torch.float32)
y = torch.ops.deep_ep.ccu_urma_prepared_multipath_alltoall_policy(
    x, 1, [2, 1]
)
print("Meta:", y.device, tuple(y.shape), y.dtype)

assert version() >= 9
assert tuple(y.shape) == tuple(x.shape)
PY
```

## 5. 快速单 case

下面使用物理卡 2、3 通信，显式指定 0、1 为 relay：

```bash
cd /home/l00934901/sgl-kernel-npu
source /usr/local/Ascend/cann-9.1.T560/set_env.sh
unset LD_PRELOAD A5_CCU_PREPARED_ALLOW_LAZY_STREAM

bash scripts/run_a5_ccu_explicit_multipath_alltoall_perf.sh \
  --src-phy 2 --dst-phy 3 \
  --relay-phys 0,1 \
  --direct-route 0 \
  --cases direct_plus_relays \
  --bytes 4194304 \
  --warmup 1 --iters 1 --repeats 1 \
  --timeout-seconds 300 \
  --cann-root /usr/local/Ascend/cann-9.1.T560 \
  --output-root /home/l00934901/profiling
```

不要再传 `--hccl-lib-dir`。即使传入，脚本也会忽略它，避免重新注入失败的实验
`libhccl.so`。

成功日志应包含：

```text
Verified prepared-plan route runtime: ... (ABI=5)
Verified requested APIs: ... prepared=True; DeepEP ABI=9
PREPARED_MULTIPATH_STREAM phase=bind_ready ... paths=3
PREPARED_MULTIPATH_PLAN ... paths=3
PASS: implementation=prepared ...
```

检查最新结果：

```bash
RUN_DIR=$(ls -dt \
  /home/l00934901/profiling/a5_ccu_explicit_multipath_2_3_* | head -1)
LOG="${RUN_DIR}/cases/direct_plus_relays_r1.log"

column -s $'\t' -t "${RUN_DIR}/case_status.tsv"
grep -nE \
'explicit multipath direct|explicit relay|PATH_CHANNEL|PREPARED_MULTIPATH_(PLAN|STREAM)|CASE_FAILURE|PASS:' \
"${LOG}"
```

## 6. ACLGraph capture/replay

plan 必须在 capture stream 上图外准备。测试程序已经这样做；capture/replay 中不会
Acquire Channel 或注册 kernel：

```bash
bash scripts/run_a5_ccu_explicit_multipath_alltoall_perf.sh \
  --src-phy 2 --dst-phy 3 \
  --relay-phys 0,1 \
  --direct-route 0 \
  --cases direct_plus_relays \
  --graph-backend aclgraph \
  --bytes 4194304 \
  --warmup 1 --iters 3 --repeats 1 \
  --timeout-seconds 600 \
  --cann-root /usr/local/Ascend/cann-9.1.T560 \
  --output-root /home/l00934901/profiling
```

检查：

```bash
RUN_DIR=$(ls -dt \
  /home/l00934901/profiling/a5_ccu_explicit_multipath_2_3_* | head -1)
LOG="${RUN_DIR}/cases/direct_plus_relays_r1.log"

grep -nE \
'CASE_(PREPARED_PLAN|ACLGRAPH_CAPTURE|ACLGRAPH_REPLAY)|PREPARED_MULTIPATH_(PLAN|STREAM)|HcclChannelAcquire|PASS:' \
"${LOG}"
```

判据：

- `bind_ready` 在 capture 前只出现一次；
- capture 与 replay 都 PASS；
- replay 窗口内不能再次出现 Channel Acquire/kernel register；
- send tensor 在 replay 前被修改，结果校验通过，因此不是复用旧输出。

当前 `path_weights` 是图常量。eager 调用可以对同一 plan 传另一组正权重；ACLGraph
需要为另一组权重 capture 另一张图。`--replay-path-weights` 对 prepared case 会直接报错，
避免把未生效的策略更新误判为动态选路。权重 0 暂不开放，因为 T560 对零字节
`WriteNb` 的 provider 行为没有可靠契约。

## 7. relay 数量和性能矩阵

`direct_plus_relays` 使用 `--relay-phys` 中的全部 relay，支持 1..12 张，不要求偶数：

```bash
# 1 条 relay
--relay-phys 0 --cases direct_plus_relays

# 3 条 relay
--relay-phys 0,1,4 --cases direct_plus_relays

# 固定性能矩阵别名
--relay-phys 4,5 --cases direct_plus_2relay
--relay-phys 4,5,1,0 --cases direct_plus_4relay
--relay-phys 4,5,1,0,6,7 --cases direct_plus_6relay
```

默认权重为：

```text
direct : 每一条 relay = 2 : 1
```

即两条 relay 为 `2,1,1`，不是 direct 与 relay 总量之比 `2:1`。

## 8. profiling

只采显式多路径 case：

```bash
bash scripts/run_a5_ccu_explicit_multipath_alltoall_perf.sh \
  --src-phy 2 --dst-phy 3 \
  --relay-phys 0,1 \
  --direct-route 0 \
  --cases direct_plus_relays \
  --bytes 4194304 \
  --warmup 100 --iters 20 --repeats 1 \
  --profile --profile-iters 20 \
  --timeout-seconds 600 \
  --cann-root /usr/local/Ascend/cann-9.1.T560 \
  --output-root /home/l00934901/profiling
```

`msprof`/MindStudio 产物位于本次 `RUN_DIR/profiling`。先用普通单 case 验证正确性，
再采 profiling，避免把建链失败误判为 profiler 问题。

profiling harness 会在每次 prepared-plan 调用后执行一次 NPU synchronize，并打印：

```text
CASE_PROFILE_LAUNCH_MODE ... mode=complete_each_prepared_invocation
```

这是为了让 100 次 warmup 表示 100 个已经完成的样本，而不是一次性排队 100 个复用
同一输出 buffer、Channel notify 和 completion event 的未完成 launch。该同步只隔离不同的
AllToAll 调用，不会把单次 AllToAll 内的 direct/relay 路径串行化；CCU kernel 内仍然是所有
`WriteNb` 提交后才统一等待。

正式计时的 20 次同时就是 profiler 采集的 20 次，不会先计时 20 次、再额外运行 20 次。
因此上述命令在图外 plan 初始化完成后总共执行：

```text
100 次独立且完成的 warmup
+ 20 次独立且完成的正式计时/profiling
= 120 次 AllToAll
```

启用 `--profile` 时，`--iters` 必须与 `--profile-iters` 相等；不一致会在启动前报参数错误，
防止性能报告与 profiling 样本数不一致。

如果日志在 profiler 启动前就于 `warmup_correctness` 报 100% 数据不一致，且没有上述
`CASE_PROFILE_LAUNCH_MODE`，说明测试脚本仍是旧版本；这不是 `msprof` 初始化错误。先拉取包含
该门禁的最新分支，再重新运行第 8 节。

## 9. 常见故障

### 9.1 route runtime ABI 小于 5

重新执行第 3 节并确认安装路径。构建脚本会比较 packaged 和 installed `.so`，避免
custom package installer 留下旧文件。

### 9.2 DeepEP ABI 小于 9

重新构建并 `--force-reinstall` wheel，然后用第 4 节确认实际加载路径。不要只看源码
分支。

### 9.3 plan 未绑定当前 stream

错误会明确提示 `call PlanBindStream outside graph capture`。不能在 capture/replay 内懒建
资源。测试用 `A5_CCU_PREPARED_ALLOW_LAZY_STREAM=1` 仅用于兼容性排查，不是正式路径。

### 9.4 更改已有 plan_id 的路径目录

plan `(communicator, plan_id)` 是不可变路径目录。更换 relay manifest 或路径数量必须用
新的 `plan_id`；每次执行只允许改变同长度的正权重。

### 9.5 资源释放

当前公开 T560 接口没有提供完整、可证明安全的 Channel/thread/kernel 对称释放链。
prepared plan 因此按 communicator/process 生命周期持有资源。不要循环创建大量一次性
plan_id；每个稳定路径目录复用一个 plan。
