# A5 IO Die All2All CCU 算子

## 当前目标

`All2AllDetourIoDie` 当前首先验证 A5（Ascend 950）上的完整 CCU AllToAllV 通信是否能够建立、下发并得到正确结果。

它不再访问 `GetHcclContext()` 中为 0 的 `windowsIn[peerRank]`，也不让 AIV 直接读写远端卡的 HBM。数据由 HCCL 的 CCU 通信任务从 `sendData` 直接传到目标 rank 的 `recvData`。

需要特别区分两个目标：

- 当前版本验证“完整通信域上的 CCU AllToAllV”以及“部分 rank 的 count 为 0”；
- CCU/HCCL 根据拓扑选择 IO Die 路径，当前公开接口不能指定必须经过哪张中间卡。因此 `--comm-ranks 0,2` 不等价于强制 rank 1、3 做转发卡。

显式指定 2～7 卡共同承担 0→1 的转发，是后续路由策略问题，不能由本测试的 `commRankIds` 参数保证。

## 为什么从 HalfAllToAllV 改为完整 AllToAllV

仓库中的 A5 dispatch/combine CCU 实现使用：

```text
Hccl<HCCL_SERVER_TYPE_CCU>
InitV2 + SetCcTilingV2
AlltoAllvWrite
HCCL_CMD_HALF_ALLTOALLV
```

`AlltoAllvWrite` 是 dispatch/combine 使用的半程 window 协议。此前版本照搬该接口后，目标环境在 device kernel 启动前返回：

```text
HcclAllocComResourceByTiling ... ret = 5
```

`ret=5` 是 `HCCL_E_NOT_SUPPORT`。这说明该通信域没有成功建立 HalfAllToAllV 资源，不是 AIV kernel 内部死锁。

`/home/liuyuanwen/hccl` 的完整 AllToAllV 路径有 A5 CCU selector、executor 和 CCU kernel 注册，包括 `CcuAlltoAllVMesh1D` 与双 Die 的 `CcuAllToAllVMesh2Die`。因此当前版本改为一致的完整协议：

```text
host:   HCCL_CMD_ALLTOALLV + A5_CCU_ENGINE(5)
device: Hccl::AlltoAllV<true>(sendData, ..., recvData, ...)
```

只有 AIV0 写 HCCL message area、提交任务并等待 handle；所有 AIV 在结束前同步并调用 `Finalize`。收发 count 和 displacement 使用 `uint64_t[rankSize]`，数据类型传 `HCCL_DATA_TYPE_INT8`，因此单位是字节。

两卡全通信时，每个 rank 的输入包含两个等长块：输入块 `i` 发给 rank `i`；输出块 `i` 来自 rank `i`。测试脚本据此逐元素校验。

## 已验证的编译环境

- 分支：`feature/a5-io-die-all2all-detour`
- Docker：`cam_lyw_dev_91`
- CANN：`9.1.0-beta.1`
- 编译目标：`Ascend950` / `ascend950`
- FP16、BF16、FP32、INT32 四个 device binary 均可生成

无卡服务器只能验证编译；通信与正确性必须在 A5 有卡服务器验证。

## 有卡 sglang Docker 环境

以下命令均在已有 sglang Docker 内执行，不使用 conda：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || \
source /usr/local/Ascend/cann/set_env.sh 2>/dev/null || true

python3 -c 'import torch, torch_npu, pybind11; print(torch.__version__); print(torch_npu.__version__)'
```

若只缺构建工具：

```bash
python3 -m pip install pybind11 wheel setuptools
```

不要替换当前能够启动 sglang 的 `torch` 和 `torch_npu`。

## 拉取、编译、安装最新包

```bash
cd /home/l00934901/sgl-kernel-npu
git fetch origin
git switch feature/a5-io-die-all2all-detour
git pull --ff-only origin feature/a5-io-die-all2all-detour
git log -1 --oneline

bash scripts/build_a5_io_die_all2all_detour.sh

latest_wheel="$(ls -1t output/deep_ep-*.whl | head -n 1)"
test -n "${latest_wheel}" || { echo "没有找到 deep_ep wheel" >&2; exit 1; }
echo "Installing ${latest_wheel}"
python3 -m pip install --force-reinstall --no-deps "${latest_wheel}"

source /home/l00934901/sgl-kernel-npu/python/deep_ep/deep_ep/vendors/hwcomputing/bin/set_env.bash
```

不要直接执行 `pip install output/deep_ep-*.whl`：目录中有多个历史包时，通配符会展开为多个 wheel。上面的 `ls -1t ... | head -n 1` 只选择最新包。

`set_env.bash` 只对当前 shell 生效。每次进入 Docker、新开 shell 或重装 wheel 后，都要重新 `source`。

可核对源码和实际加载产物：

```bash
git rev-parse --short HEAD
grep -n "HCCL_CMD_.*ALLTOALLV\|SetCommEngine" \
  csrc/deepep/ops/op_host/all2_all_detour_io_die_tiling.cpp
ls -l python/deep_ep/deep_ep/deep_ep_cpp*.so
ls -l python/deep_ep/deep_ep/vendors/hwcomputing/op_api/lib/libcust_opapi.so
```

当前源码应显示 `HCCL_CMD_ALLTOALLV`，不能再是 `HCCL_CMD_HALFALLTOALLV`。

## 两卡全量 CCU 测试

以下示例选择物理卡 4、5；两个 worker 内的逻辑 device 0、1 分别映射到物理卡 4、5：

```bash
cd /home/l00934901/sgl-kernel-npu
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || \
source /usr/local/Ascend/cann/set_env.sh 2>/dev/null || true
source /home/l00934901/sgl-kernel-npu/python/deep_ep/deep_ep/vendors/hwcomputing/bin/set_env.bash

export HCCL_BUFFSIZE=2300
export HCCL_OP_EXPANSION_MODE=CCU_SCHED
export ASCEND_RT_VISIBLE_DEVICES=4,5
unset ASCEND_LAUNCH_BLOCKING

python3 -m torch.distributed.run \
  --standalone --nproc-per-node=2 \
  tests/python/deepep/test_a5_io_die_all2all_detour.py
```

成功时应输出：

```text
PASS: A5 All2AllDetourIoDie (full group)
world_size=2, comm_ranks=[0, 1], elements_per_peer=256
```

`torchrun --nproc-per-node=N` 使用 `ASCEND_RT_VISIBLE_DEVICES` 中从左到右的 N 张卡。换卡时只修改该列表。

## 四卡和八卡测试

四卡全量通信：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
python3 -m torch.distributed.run \
  --standalone --nproc-per-node=4 \
  tests/python/deepep/test_a5_io_die_all2all_detour.py
```

四卡通信域中仅 rank 0、2 使用非零 count：

```bash
python3 -m torch.distributed.run \
  --standalone --nproc-per-node=4 \
  tests/python/deepep/test_a5_io_die_all2all_detour.py \
  --comm-ranks 0,2
```

八卡全量通信：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3 -m torch.distributed.run \
  --standalone --nproc-per-node=8 \
  tests/python/deepep/test_a5_io_die_all2all_detour.py
```

即使某些 rank 的 count 为 0，通信域中的所有 rank 也必须进入算子，否则 collective 会等待。

## 故障定位

### `HcclAllocComResourceByTiling ret = 5`

若仍报 `ret=5`，先看 wrapper 输出的 workspace：

- 新的完整 AllToAllV 版本应为 `16777216`（16 MiB）；
- 旧 HalfAllToAllV 版本为 `16778240`（16 MiB + 1024），说明仍加载了旧 wheel。

确认源码、重新编译、安装最新 wheel 并重新 `source set_env.bash`。如果源码已是 `HCCL_CMD_ALLTOALLV` 且仍返回 5，收集运行时库和 HCCL 日志：

```bash
python3 -c 'import torch, torch_npu; print("torch", torch.__version__); print("torch_npu", torch_npu.__version__)'
find /usr/local/Ascend -maxdepth 5 -type f -name version.info -print
ldconfig -p 2>/dev/null | grep -E "libhccl|libhcomm|libopapi|libascendcl"
ldd python/deep_ep/deep_ep/vendors/hwcomputing/op_api/lib/libcust_opapi.so

find /usr/slog -type f -mmin -10 -print 2>/dev/null
grep -RInE "HcclAllocComResourceByTiling|All2AllDetourIoDie|HCCL_E_NOT_SUPPORT" \
  /usr/slog 2>/dev/null | tail -n 300
```

普通 HCCL barrier 成功只说明通信域可用，不代表 MC2 的指定 CCU task 资源一定可用。

### 卡住或 `status=561000`

测试脚本逐 rank 输出阶段，并在默认 60 秒后打印 Python 栈。需要缩短时间时：

```bash
python3 -m torch.distributed.run \
  --standalone --nproc-per-node=2 \
  tests/python/deepep/test_a5_io_die_all2all_detour.py \
  --traceback-timeout 30
```

开启 wrapper 跟踪：

```bash
export A5_DETOUR_TRACE=1
```

判断方式：

- 卡在 `preflight HCCL barrier`：问题在算子外部的 HCCL 建域、卡状态或拓扑；
- `GetWorkspaceSize` 直接失败：host tiling 或通信资源配置错误；
- `execute end: status=0` 后卡在同步：device CCU task 没有完成；
- `execute end: status=561000, detail=...`：以 `detail` 中第一条 HCCL/ACL 错误为根因，Python 的 SIGSEGV 只是后续异常。

失败后立即收集：

```bash
find /usr/slog -type f -mmin -5 -print 2>/dev/null
grep -RInE "All2AllDetourIoDie|561000|HcclAllocComResourceByTiling|Kernel Run failed|rtFusionLaunch" \
  /usr/slog 2>/dev/null | tail -n 300
```

### `Can not find kernel ... tilingKey=1`

device 使用 `REGISTER_TILING_DEFAULT`，host 必须使用：

```text
context->SetTilingKey(0UL)
```

出现 key 1 表示安装的是旧包，需要更新、重编并重装最新 wheel。

### `Invalid device ID`

`--nproc-per-node=4` 要求 `ASCEND_RT_VISIBLE_DEVICES` 至少列出四张卡。如果只设置 `0,1`，local rank 2、3 必然报错。

## 当前约束

- `rank_size <= 32`；
- `commRankIds` 必须是 NPU 上连续的 INT32 一维 tensor；
- `sendData.numel()` 必须能被 `commRankIds.numel()` 整除；
- 支持 FP16、BF16、FP32、INT32，底层按字节通信，不做类型转换；
- 当前只验证单机 A5；跨机拓扑需单独测试；
- 当前不提供用户指定 IO Die 中转 rank 的能力。
