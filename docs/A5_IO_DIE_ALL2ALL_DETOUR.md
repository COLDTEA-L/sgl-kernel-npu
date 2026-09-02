# A5 IO Die All2All 绕路算子

## 目标与实现差异

`All2AllDetourIoDie` 用于 A5（Ascend 950）上的通信 rank 子集 All2All。

旧实现依赖 `GetHcclContext()` 中的 `windowsIn[peerRank]`，由通信卡直接读写绕路卡 HBM。在 A5 环境中这些远端 window 地址可能为 0，因此无法作为跨卡目标地址。

本实现不读取 `windowsIn[]`，也不把远端地址暴露给 AIV：

1. host 侧将 executor 的 HCCL server type 设置为 `CCU`；
2. host tiling 使用与 CANN 9.1 device 实现匹配的 `HCCL_CMD_HALFALLTOALLV + CCU_SCHED engine(6)`；
3. kernel 调用 `Hccl<HCCL_SERVER_TYPE_CCU>::AlltoAllvWrite`，由 CCU/IO Die 把数据写入每个目标 rank 的本地 `windowsOut[0]`；
4. `commRankIds` 之外的 rank 使用 0 send size，但仍参与同一通信域中的 collective；
5. 通信完成后，目标 rank 从自己的本地 window 拷贝到 `recvData`。整个过程不访问 `windowsIn[]`。

输入 `sendData` 的第 `i` 个等长数据块发往 `commRankIds[i]`；输出 `recvData` 的第 `i` 个数据块来自 `commRankIds[i]`。所有 rank 必须传入完全相同、升序且不重复的 `commRankIds`。

## 已验证环境

- 分支：`feature/a5-io-die-all2all-detour`
- Docker：`cam_lyw_dev_91`
- CANN：`9.1.0-beta.1`
- SoC 编译目标：`Ascend950` / `ascend950`
- 编译结果：FP16、BF16、FP32、INT32 四个 device binary 均成功生成

无卡服务器只能完成编译验证；正确性测试需要在 A5 有卡服务器执行。

## 在 sglang Docker 中准备环境

以下命令均在已有 sglang Docker 内执行，不使用 conda。先加载 CANN 环境：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || \
source /usr/local/Ascend/cann/set_env.sh 2>/dev/null || true

python3 -c 'import torch, torch_npu, pybind11; print(torch.__version__); print(torch_npu.__version__)'
```

如果缺少 `pybind11` 或构建工具：

```bash
python3 -m pip install pybind11 wheel setuptools
```

`torch` 和 `torch_npu` 应继续使用当前能够启动 sglang 服务的版本，不要为了编译算子单独替换。

## 单算子编译

先更新到远端分支的最新提交，并确认当前提交号：

```bash
cd /home/l00934901/sgl-kernel-npu
git fetch origin
git switch feature/a5-io-die-all2all-detour
git pull --ff-only origin feature/a5-io-die-all2all-detour
git log -1 --oneline
```

再执行单算子编译：

```bash
bash scripts/build_a5_io_die_all2all_detour.sh
```

脚本设置 `DEEPEP_SINGLE_OP=all2_all_detour_io_die`，跳过 Catlass 与其他 device kernel。成功时末尾应出现：

```text
All2AllDetourIoDie device binaries: 4
Built: output/deep_ep-....whl
```

安装到当前 sglang Python：

```bash
latest_wheel="$(ls -1t output/deep_ep-*.whl | head -n 1)"
test -n "${latest_wheel}" || { echo "没有找到 deep_ep wheel" >&2; exit 1; }
echo "Installing ${latest_wheel}"
python3 -m pip install --force-reinstall --no-deps "${latest_wheel}"
```

必须给 wheel 路径加双引号。不要直接把 `output/deep_ep-*.whl` 传给 `pip install`：如果目录中保留了多次编译生成的 wheel，shell 会把通配符展开为多个包，导致 pip 同时安装多个版本并报冲突。`ls -1t` 按修改时间倒序排列，上述命令只选取第一项，即本次最新生成的 wheel。

测试脚本会优先加载当前源码目录下由编译步骤刷新的 `deep_ep_cpp` 和 `libcust_opapi.so`，启动时会打印两者的绝对路径。运行测试前可同时核对提交和产物时间：

```bash
git log -1 --oneline
ls -l python/deep_ep/deep_ep/deep_ep_cpp*.so
ls -l python/deep_ep/deep_ep/vendors/hwcomputing/op_api/lib/libcust_opapi.so
```

## 选择测试卡

`torchrun --nproc-per-node=N` 使用 `ASCEND_RT_VISIBLE_DEVICES` 中从左到右的 N 张卡。例如选择物理卡 2、5：

```bash
export ASCEND_RT_VISIBLE_DEVICES=2,5
```

仅设置四张可见卡但使用 `--nproc-per-node=2` 时，实际只会使用可见列表中的前两张卡。

## 两卡全通信测试

```bash
cd /home/l00934901/sgl-kernel-npu
export HCCL_BUFFSIZE=2300
export HCCL_OP_EXPANSION_MODE=CCU_SCHED
export ASCEND_RT_VISIBLE_DEVICES=0,1

python3 -m torch.distributed.run \
  --standalone --nproc-per-node=2 \
  tests/python/deepep/test_a5_io_die_all2all_detour.py
```

两卡场景可验证 CCU 调用、数据布局和正确性，但没有非通信 rank。

## 四卡通信子集 / IO Die 路由测试

下面让 rank 0、2 交换数据，rank 1、3 以 0 count 参与 collective：

```bash
export HCCL_BUFFSIZE=2300
export HCCL_OP_EXPANSION_MODE=CCU_SCHED
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3

python3 -m torch.distributed.run \
  --standalone --nproc-per-node=4 \
  tests/python/deepep/test_a5_io_die_all2all_detour.py \
  --comm-ranks 0,2
```

成功时 rank 0 输出：

```text
PASS: A5 All2AllDetourIoDie (subset/IO-Die routing)
```

## 八卡全通信测试

确认 8 张卡都映射进容器后，可直接跑完整通信域：

```bash
export HCCL_BUFFSIZE=2300
export HCCL_OP_EXPANSION_MODE=CCU_SCHED
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python3 -m torch.distributed.run \
  --standalone --nproc-per-node=8 \
  tests/python/deepep/test_a5_io_die_all2all_detour.py
```

如果只想让偶数逻辑 rank 传数据、其余 rank 仅参与 collective，以验证子集转发：

```bash
python3 -m torch.distributed.run \
  --standalone --nproc-per-node=8 \
  tests/python/deepep/test_a5_io_die_all2all_detour.py \
  --comm-ranks 0,2,4,6
```

## 约束

- `rank_size <= 32`，与当前 CANN 9.1 CCU AlltoAllV 参数上限一致；
- `commRankIds` 必须是 NPU 上连续的 INT32 一维 tensor；
- `sendData.numel()` 必须能被 `commRankIds.numel()` 整除；
- 支持 FP16、BF16、FP32、INT32；底层按字节调用 CCU，因此不会做类型转换；
- 通信域中每个 rank 都必须进入算子，否则 collective 会等待；
- 当前版本针对单机 A5 通信域验证，跨机拓扑需要在目标集群另行做正确性和性能验证。

## `HcclAllocComResourceByTiling ret = 5`

HCCL 返回码 `5` 是 `HCCL_E_NOT_SUPPORT`，错误发生在 Host 侧通信资源分配阶段，此时 device kernel 还没有启动。

本算子 host tiling 使用 `HCCL_CMD_HALFALLTOALLV + CCU_SCHED engine(6)`。这里的两项不能混用：CANN 9.1 的 v310 `Hccl::AlltoAllvWrite` device 实现内部提交的是 `HCCL_CMD_HALF_ALLTOALLV`，所以 host 必须按 HalfAllToAllV 类型申请 task 资源；通信引擎仍须使用值 `6`（`CCU_SCHED`）。值 `5` 是 `CCU_MS`，不适用于此 AlltoAllvWrite 路径。

另外，创建 HCCL 通信域之前必须选择 950 的 CCU 调度展开模式：

```bash
export HCCL_OP_EXPANSION_MODE=CCU_SCHED
```

环境变量必须在 `torch.distributed.init_process_group("hccl")` 之前生效。当前测试脚本会在导入 `torch` 和创建通信域前自动设置它；如果外部已经设置成其他值（例如 `AIV` 或 `AI_CPU`），脚本会直接报出冲突，避免等到异步执行阶段才失败。

如果更新、重编译后仍返回 `5`，先保存以下诊断信息，不要继续修改 device kernel；此时 kernel 尚未启动：

```bash
git rev-parse --short HEAD
grep -n "HCCL_CMD_.*ALLTOALLV\|SetCommEngine" \
  csrc/deepep/ops/op_host/all2_all_detour_io_die_tiling.cpp

python3 -c 'import torch, torch_npu; print("torch", torch.__version__); print("torch_npu", torch_npu.__version__)'
find /usr/local/Ascend -maxdepth 5 -type f -name version.info -print
ldconfig -p 2>/dev/null | grep -E "libhccl|libopapi|libascendcl"
ldd python/deep_ep/deep_ep/vendors/hwcomputing/op_api/lib/libcust_opapi.so

find /usr/slog -type f -mmin -10 -print 2>/dev/null
grep -RInE "HcclAllocComResourceByTiling|All2AllDetourIoDie|HCCL_E_NOT_SUPPORT" \
  /usr/slog 2>/dev/null | tail -n 200
```

## 设置 `CCU_SCHED` 后进程报 `SIGSEGV`

Ascend 950 CCU 使用双 Die。`AlltoAllvWrite` 的 `sendSizes` 和 `sendOffsets` 不是普通的 `rankDim` 长度数组，而是各包含 `2 * rankDim` 项：前半段描述 Die0，后半段描述 Die1。每个 peer 的数据也必须拆分成两个连续片段。旧版本只提供了单 Die 参数，CCU 读取第二组参数时会越界，表现为两个 worker 同时收到 `SIGSEGV`。请更新到包含双 Die 参数布局修复的版本并重新编译、安装 wheel。

## `HcclGetCcuTaskInfo ret = 4`

如果日志在进入 AICore kernel 前报以下错误：

```text
HcclGetCcuTaskInfo ... ret = 4
```

请确认 host tiling 使用的是 `HCCL_CMD_HALFALLTOALLV + CCU_SCHED engine(6)`。旧版本使用 `AIV_ENGINE(3)` 时会在 Host 侧生成 CCU task info 阶段失败；使用 `CCU_MS engine(5)` 时则会在资源分配阶段返回不支持。更新分支后必须重新编译并重装 wheel，不能只更新 Python 测试脚本：

```bash
git fetch origin
git switch feature/a5-io-die-all2all-detour
git pull --ff-only origin feature/a5-io-die-all2all-detour
bash scripts/build_a5_io_die_all2all_detour.sh
latest_wheel="$(ls -1t output/deep_ep-*.whl | head -n 1)"
python3 -m pip install --force-reinstall --no-deps "${latest_wheel}"
```

重新运行前可确认源码与已安装自定义库的时间：

```bash
git log -1 --oneline
ls -l python/deep_ep/deep_ep/vendors/hwcomputing/op_api/lib/libcust_opapi.so
```

## `Can not find kernel ... tilingKey=1`

如果 HCCL 资源分配已经成功，但启动算子时报：

```text
Can not find kernel by function[0x0], tilingKey=1
rtFusionLaunch execution failed, reason=kernel pointer null
```

说明 host 下发的 tiling key 与编译进 wheel 的 device kernel 不一致。当前 device kernel 使用
`REGISTER_TILING_DEFAULT(All2AllDetourIoDieTilingData)`，对应默认 key `0`；host tiling 也必须执行
`context->SetTilingKey(0UL)`。旧版本错误地设置成了 `1`，但单算子编译没有生成 key 1 变体，
因此运行时找不到 kernel 函数。

更新源码、重编并安装最新 wheel：

```bash
cd /home/l00934901/sgl-kernel-npu
git fetch origin
git switch feature/a5-io-die-all2all-detour
git pull --ff-only origin feature/a5-io-die-all2all-detour

grep -n "SetTilingKey" \
  csrc/deepep/ops/op_host/all2_all_detour_io_die_tiling.cpp
# 预期：context->SetTilingKey(0UL)

bash scripts/build_a5_io_die_all2all_detour.sh
latest_wheel="$(ls -1t output/deep_ep-*.whl | head -n 1)"
test -n "${latest_wheel}" || { echo "没有找到 deep_ep wheel" >&2; exit 1; }
echo "Installing ${latest_wheel}"
python3 -m pip install --force-reinstall --no-deps "${latest_wheel}"
```

这类错误与选了哪两张物理卡无关，也不是 `HCCL_BUFFSIZE` 不足；在 kernel 真正启动前就已经失败。

## 启动后无报错但一直不退出

如果已经打印 `Using HCCL runtime`，没有新的错误，但 `torchrun` 一直不退出，先按 `Ctrl+C`
终止进程。这说明算子已经进入 device kernel，通常卡在 CCU collective 或核间同步，不应继续无限等待。

测试脚本会为每个 rank 打印阶段和耗时，并在 60 秒后自动输出 Python 线程栈。例如：

```text
[rank0] All2AllDetourIoDie enqueue begin
[rank0] All2AllDetourIoDie enqueue done; device synchronize begin
```

最后一条阶段日志可以区分 HCCL 建域、普通 HCCL barrier、Buffer 初始化、Host 算子下发和 device
kernel 执行。线程栈只用于诊断，不会主动结束进程；仍需用 `Ctrl+C` 终止卡住的测试。可调整等待时间：

```bash
python3 -m torch.distributed.run \
  --standalone --nproc-per-node=2 \
  tests/python/deepep/test_a5_io_die_all2all_detour.py \
  --traceback-timeout 30
```

请保留两个 rank 的最后一条阶段日志及自动输出的线程栈。若普通 `preflight HCCL barrier` 都无法完成，
问题在算子之外的 HCCL 通信域或卡状态；若卡在 `device synchronize begin`，才是 device kernel 内部等待。

若两个 rank 都停在 `All2AllDetourIoDie enqueue begin`，开启 ACLNN wrapper 跟踪：

```bash
export A5_DETOUR_TRACE=1
echo "ASCEND_LAUNCH_BLOCKING=${ASCEND_LAUNCH_BLOCKING:-unset}"
```

日志会进一步显示：

```text
[A5 detour][rank=0] GetWorkspaceSize begin
[A5 detour][rank=0] GetWorkspaceSize end: status=0, workspace=...
[A5 detour][rank=0] HCCL server type set to CCU
[A5 detour][rank=0] execute begin
```

- 只有 `GetWorkspaceSize begin`：Host tiling 或 HCCL 资源申请阻塞；
- 打印 `execute begin` 但没有 `execute end`：执行接口正在等待 device kernel；
- `execute end` 已打印而 Python 停在 `device synchronize begin`：算子已异步下发，device kernel 未完成。

如果两个 rank 均显示 `GetWorkspaceSize status=0`、`execute status=0`，但调用仍不返回，说明 Host 下发已经成功，阻塞位于 device kernel 的 CCU 完成等待。一个关键检查项是 host/device task type 必须一致：

```text
device AlltoAllvWrite -> HCCL_CMD_HALF_ALLTOALLV
host Mc2CcTilingConfig -> HCCL_CMD_HALFALLTOALLV
engine                -> CCU_SCHED (6)
```

曾经使用的 `HCCL_CMD_ALLTOALLV + engine(6)` 能通过 Host 资源分配和下发，但与 device API 实际提交的 HalfAllToAllV task 不一致，可能表现为 `execute status=0` 后一直等待完成。更新到修复该组合的提交后，需要重新编译、安装最新 wheel；仅设置环境变量无法修复已安装库中的 host tiling。

`ASCEND_LAUNCH_BLOCKING=1` 会把 device 等待放进 `execute` 调用中；unset 时通常表现为 enqueue 返回、
随后阻塞在 `torch.npu.synchronize()`。两者只是错误定位位置不同，不会消除通信死锁。

`AlltoAllvWrite + CCU_SCHED` 要求所有 AIV 核以相同顺序执行：

```text
AlltoAllvWrite -> Wait -> SyncAll -> Finalize
```

旧版本仅让 block 0 调用 `AlltoAllvWrite/Wait`，其他 AIV 核直接进入 `SyncAll`，会形成互等。
当前版本已按照 CCU_SCHED 调度模型处理：

- 用 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)` 声明 kernel task type；
- block 0 生成双 Die 的 `sendSizes/sendOffsets`，然后所有 AIV 核同步；
- 所有 AIV 核均调用 `AlltoAllvWrite` 和 `Wait`；
- 数据拷贝完成后，所有 AIV 核再次同步并分别调用 `Finalize`。

更新、编译、安装最新 wheel 后再测试。调试时可以设置：

```bash
export ASCEND_LAUNCH_BLOCKING=1
```

测试结束后应恢复异步模式：

```bash
unset ASCEND_LAUNCH_BLOCKING
```

## `Invalid device ID`（四进程测试）

`--nproc-per-node=4` 要求容器内至少有 4 张可见卡。若此前执行过：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1
```

则 local rank 2、3 必然报 `Expected value: [0, 2)`。四卡测试前应改为：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
# 或者使用容器已经映射好的全部卡
unset ASCEND_RT_VISIBLE_DEVICES
```

然后再执行 `--nproc-per-node=4`。物理卡号由 `ASCEND_RT_VISIBLE_DEVICES` 决定；测试程序里的 rank 0～3 是重映射后的逻辑卡号。
