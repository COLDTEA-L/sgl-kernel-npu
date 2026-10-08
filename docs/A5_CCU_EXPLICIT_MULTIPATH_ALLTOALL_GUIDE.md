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

本次 ABI 要求：route runtime ABI >= 11，DeepEP ABI >= 11。必须重新执行第 3、4 节，
安装 route package 和 wheel 两部分；不需要编译或替换系统 HCCL。
当前 prepared 数据面仅支持两 rank、连续 FP32 tensor、互不重叠的输入/输出。

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

# 检查本次源码包含新的布局校验，不把旧提交号当成最新版本门禁
test -f examples/a5_ccu_urma_route_probe/common/path_layout.h
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

脚本会强制检查三个 prepared-plan 入口以及 route ABI 11：

```text
HcclCcuUrmaExplicitMultipathPlanCreate
HcclCcuUrmaExplicitMultipathPlanBindStream
HcclCcuUrmaExplicitMultipathPlanExecuteV2
A5CcuUrmaPreparedPlanAbiVersion() >= 11
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
assert version() >= 11
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

assert version() >= 11
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
  --graph-backend none \
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
Verified prepared-plan route runtime: ... (ABI=11)
Verified requested APIs: ... prepared=True; DeepEP ABI=11
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
Acquire Channel 或注册 kernel。本节用于验证“可入图”，不作为裸算子性能结论：

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

capture 和第一次校验 replay 都位于性能计时窗口之前。如果需要测图模式稳态性能，
本命令后续的 `warmup/iters` 测到的是 `aclgraph_replay`，而不是 capture 时间。

当前 `path_weights` 是图常量。eager 调用可以对同一 plan 传另一组正权重；ACLGraph
需要为另一组权重 capture 另一张图。`--replay-path-weights` 对 prepared case 会直接报错，
避免把未生效的策略更新误判为动态选路。权重 0 暂不开放，因为 T560 对零字节
`WriteNb` 的 provider 行为没有可靠契约。

## 7. relay 数量和性能矩阵

裸算子通信性能统一使用 `--graph-backend none`。这不会执行 ACLGraph capture，
结果中会标记 `measurement_mode=prepared_eager`。每一次 prepared AllToAll 完成后才会
进入下一次，避免把异步批量排队吞吐误当成单次通信延迟。

`direct_plus_relays` 使用 `--relay-phys` 中的全部 relay，支持 1..12 张，不要求偶数：

```bash
# 1 条 relay
--relay-phys 0 --cases direct_plus_relays --graph-backend none

# 3 条 relay
--relay-phys 0,1,4 --cases direct_plus_relays --graph-backend none

# 固定性能矩阵别名
--relay-phys 4,5 --cases direct_plus_2relay --graph-backend none
--relay-phys 4,5,1,0 --cases direct_plus_4relay --graph-backend none
--relay-phys 4,5,1,0,6,7 --cases direct_plus_6relay --graph-backend none
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
  --graph-backend none \
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
CASE_LAUNCH_COMPLETION ... mode=complete_each_prepared_invocation
CASE_MEASUREMENT_MODE ... mode=prepared_eager capture_in_timing=0
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
`CASE_LAUNCH_COMPLETION`，说明测试脚本仍是旧版本；这不是 `msprof` 初始化错误。先拉取包含
该门禁的最新分支，再重新运行第 8 节。

## 9. 常见故障

### 9.1 route runtime ABI 小于 11

重新执行第 3 节并确认安装路径。构建脚本会比较 packaged 和 installed `.so`，避免
custom package installer 留下旧文件。

### 9.2 warmup 正确性出现 50% 或 100% 不一致

先确认日志中的 direct/relay `PATH_CHANNEL` 都为 `state=0`。此前先出现缺 row 0，
route ABI 10 后又出现缺 row 1。缺失整行只能定位数据范围，不能证明是某条 relay 或
某个 CCU 指令坏了；此前据此归因组合 kernel 的结论过强。`CompletedEvent` 容器扩容
以及替换 kernel 均未解决首次调用错误。

这次源码检查发现 DeepEP bridge 在直接调用 ACL/HCOMM 前使用 `stream(false)`。
torch_npu 中它只返回流指针，跳过 host task queue 清空；`fill/stack/add_` 可能尚未提交，
通信就先读了输入。执行后 `torch.npu.synchronize()` 只能等待已提交工作，不能修复这种
先读后写。DeepEP ABI 10 改用 `stream()`，先提交完之前的 torch_npu host 任务，再提交
D2D 和 CCU launch；不增加每次 device synchronize。这个缺陷已从源码确认，它是否解释
本次整行缺失仍须用下面的逐轮有卡实验确认。

本次进一步审查修复见第 9.7 节，必须重新执行第 3、4 节，确认 **route ABI >= 11、
DeepEP ABI >= 11**；不用修改 HCCL。

语义依据：[torch_npu NPUStream 源码](https://github.com/Ascend/pytorch/blob/master/torch_npu/csrc/core/npu/NPUStream.cpp)
中的 `NPUStream::stream(bool)` 和 `NPUStream::stream()`；同时已核对开发容器
`libtorch_npu.so` 存在这两个重载。

route ABI 10 不再让 prepared-plan 使用该实验性组合 graph。它恢复已经在显式多 relay Write
实验中验证过的数据面：caller stream 上用 D2D copy 完成本地 slice，已验证的 `RouteKernel`
只负责 peer slice，并把 peer slice 按权重拆给 direct/relay Channels。实验性组合 graph 仅保留
在 legacy probe 入口中，不再服务 production prepared-plan。

ABI 6 的逐 Channel post-sync 和 ABI 8 的 T560 独立 output/token/completion slot 仍然保留。
ABI 7 曾照搬新版 primitive API 的 bitmask 布局，但 T560 object API 首次 launch 返回全零，
因此不能使用该布局。

升级后先用第 5 节的 `--warmup 2 --iters 2 --repeats 1` 验证连续四次独立调用。日志应同时
出现 route `ABI=11`、`DeepEP ABI=11`、`CASE_LAUNCH_COMPLETION` 和最终 `PASS`。未通过前不要进行性能或 profiling
实验。

如果短序列通过而 100 次 warmup 失败，用下面的诊断模式定位第一轮错误；该选项每轮读取并
检查整个输出，只用于正确性排查，不能用于性能计时：

```bash
bash scripts/run_a5_ccu_explicit_multipath_alltoall_perf.sh \
  --src-phy 2 --dst-phy 3 \
  --relay-phys 0,1 \
  --direct-route 0 \
  --cases direct_plus_relays \
  --graph-backend none \
  --bytes 4194304 \
  --warmup 4 --iters 2 --repeats 1 \
  --validate-every-iteration \
  --timeout-seconds 600 \
  --cann-root /usr/local/Ascend/cann-9.1.T560 \
  --output-root /home/l00934901/profiling
```

该诊断每轮先把输入和期望值加 17，并把输出填为 -999，然后调用算子；Python 不在这些
写入与算子之间加 synchronize，以检验 bridge 的提交顺序。每个 rank 应有六次
`CASE_ITERATION_PASS`。失败时查看 `CASE_ITERATION_FAILURE` 的 `stage`、`iteration`、
`mismatch_count`、`recv_heads`、`expected_heads` 和 `send_heads`。
通过后可把 warmup 改为 100 再检验重复执行。正式 profiling 不带该诊断选项。
即使未启用诊断选项，正确性失败也会打印 `CASE_DATA_DIAGNOSTIC`，包括每行错误数、
输出首尾值、输入值和期望值，避免仅根据 assert 的最大差值推断实际输出。

### 9.3 DeepEP ABI 小于 11

重新构建并 `--force-reinstall` wheel，然后用第 4 节确认实际加载路径。不要只看源码
分支。

### 9.4 plan 未绑定当前 stream

错误会明确提示 `call PlanBindStream outside graph capture`。不能在 capture/replay 内懒建
资源。测试用 `A5_CCU_PREPARED_ALLOW_LAZY_STREAM=1` 仅用于兼容性排查，不是正式路径。

### 9.5 更改已有 plan_id 的路径目录

plan `(communicator, plan_id)` 是不可变路径目录。更换 relay manifest 或路径数量必须用
新的 `plan_id`；每次执行只允许改变同长度的正权重。ABI 11 同时比对 manifest 内容，
不再只比较文件路径。修改同名文件不能悄悄改变已存在的 plan；新 stream 绑定时也会检查。

### 9.6 资源释放

当前公开 T560 接口没有提供完整、可证明安全的 Channel/thread/kernel 对称释放链。
prepared plan 因此按 communicator/process 生命周期持有资源。不要循环创建大量一次性
plan_id；每个稳定路径目录复用一个 plan。`CompletedEvent` 是注册 kernel 中复用的资源，
不是每轮新申请再释放。没有实现 `PlanDestroy`，不能宣称每次调用后释放了全部 CCU 资源。
通信域及其执行 stream 必须覆盖 plan 和 graph 的整个使用期，销毁前等待全部执行完成。

### 9.7 本次全面审查修复与使用约束

- kernel 缓存签名包含实际 Channel handles 和明确分隔符，避免只有 route ordinal 时
  不同显式 relay 命中同一 kernel，或 `[1,23]` 与 `[12,3]` 拼接碰撞。
- 权重分片使用 128-bit 中间乘积；先校验全部地址、长度、对齐及覆盖，再提交本地复制。
  极小 payload 造成某路径零长度时会明确拒绝，不能先复制一半再报错。
- bridge 选择 tensor 所在 device，拒绝跨 device 和输入/输出重叠；登记 allocator
  `recordStream`，避免异步执行时 tensor 存储被过早回收。另一流生产的 tensor 仍须由
  调用者执行 `execution_stream.wait_stream(producer_stream)`；生命周期登记不等于依赖同步。
- ACLGraph 测试在 capture stream warmup 前等待输入生产流；没有增加逐次 device 同步。
- plan 绑定 device，创建失败不留下可查到的半初始化 host plan；显式 Channel 非 ready
  状态直接报错。底层部分创建失败是否全部释放，仍受第 9.6 节的接口限制。
- host 提交锁防止单次主/从流 notify 序列被多 host 线程穿插；它**不保证不同 stream
  的 device 执行互斥**。同一 plan/共享 Channel 的跨流并发未经验证，不应这样使用。
  graph replay 期间不能同时在另一流用相同通信资源执行 eager 或另一张 graph。
- 两端必须采用一致的路径顺序、权重和调用顺序；不能一个 rank 换 policy、另一个不换。
  目前没有 autograd 实现；训练反向不能直接依赖本算子自动生成。

开发 Docker 已完成 route/wheel 编译和安装检查，以及布局、溢出、地址重叠、签名、Meta
测试。这些不是有卡正确性证明；升级后按第 9.2 节逐轮诊断通过，再执行第 6 节 capture/replay，
最后采性能。

可单独重复无卡回归（Python 使用已安装 ABI 11 wheel）：

```bash
cd /home/l00934901/sgl-kernel-npu
g++ -std=c++14 -Wall -Wextra -Werror -I. \
  tests/cpp/test_a5_multipath_layout.cpp -o /tmp/a5_multipath_layout_test
/tmp/a5_multipath_layout_test
python3 tests/python/deepep/test_a5_prepared_multipath_meta.py
```
