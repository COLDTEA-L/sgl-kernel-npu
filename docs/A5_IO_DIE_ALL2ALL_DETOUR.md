# A5 AllToAll：CCU 基线与 AIV+URMA 绕路验证

本分支包含两个彼此独立的算子和测试。必须先跑阶段一，再跑阶段二，这样可以把 CCU/HCCL 环境问题与自定义绕路算法问题分开。

| 阶段 | Python 接口 | 设备实现 | 用途 |
|---|---|---|---|
| 1 | `Buffer.hccl_all2_all_ccu` | `Hccl::AlltoAll` + `HCCL_CMD_ALLTOALL` + A5 CCU | 证明当前环境可以运行 HCCL 普通 AllToAll |
| 2 | `Buffer.all2_all_detour_io_die` | AIV + MTE/URMA 映射窗口 + `CpGM2GM` 双缓冲 | 通信卡 Write 绕路卡，通信卡再从绕路卡 Read |

阶段一没有复制 HCCL 的私有 CCU kernel。自定义算子只是 MC2 薄封装，真正执行的是 CANN/HCCL 已注册的 `CcuAlltoAllMesh1D`、`CcuAllToAllMesh2Die` 或对应拓扑实现。

阶段二不是 CCU，也不是透明 IO Die cut-through。非通信卡提供 HCCL/URMA 注册的 HBM 窗口，数据在绕路卡 HBM 中转。

## 1. 获取分支

在有卡服务器的 SGLang Docker 中执行：

```bash
cd /home/l00934901/sgl-kernel-npu

git fetch origin feature/a5-detour-cpgm2gm
git switch feature/a5-detour-cpgm2gm
git pull --ff-only origin feature/a5-detour-cpgm2gm

git branch --show-current
git log -1 --oneline
```

如果远端名不是 `origin`：

```bash
git remote -v
```

然后把上面命令中的 `origin` 换成实际远端名。

## 2. 准备 CANN 9.1 和 Python 环境

不使用 Conda。使用 SGLang Docker 当前的 Python：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || \
source /usr/local/Ascend/cann/set_env.sh

python3 - <<'PY'
import torch
import torch_npu
import pybind11
print("python:", __import__("sys").executable)
print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
PY
```

如果 `pybind11` 缺失：

```bash
python3 -m pip install pybind11
```

查看 CANN 版本：

```bash
cat /usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/ascend_toolkit_install.info 2>/dev/null || \
cat /usr/local/Ascend/ascend-toolkit/latest/ascend_toolkit_install.info 2>/dev/null || \
cat /etc/ascend_install.info
```

## 3. 编译两个验证算子

```bash
cd /home/l00934901/sgl-kernel-npu
bash scripts/build_a5_io_die_all2all_detour.sh
```

脚本设置：

```bash
DEEPEP_SINGLE_OP=a5_all2all_validation
```

同一个 wheel 中应同时存在：

```text
hccl_all2_all_ccu
all2_all_detour_io_die
```

这两个 kernel 已在无卡服务器的 CANN 9.1 `cam_lyw_dev_91` 环境完成过编译验证；无卡环境只能验证编译和出包，不能验证通信结果。本分支新增的 `CpGM2GM` 版本也已用下面的单算子命令完成设备kernel编译：

```bash
docker exec cam_lyw_dev_91 bash -lc '
set -e
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || \
source /usr/local/Ascend/cann/set_env.sh
cd /home/liuyuanwen/sgl-kernel-npu/csrc/deepep/ops
export DEEPEP_SINGLE_OP=all2_all_detour_io_die
export ASCEND_COMPUTE_UNIT=ascend950
bash build.sh binary
'
```

该命令只编译自定义算子包中的 `All2AllDetourIoDie` device kernel，不依赖容器中安装PyTorch，也不生成可供有卡环境安装的完整DeepEP wheel。有卡SGLang容器仍应执行本节开头的 `scripts/build_a5_io_die_all2all_detour.sh` 来生成wheel。

## 4. 始终选择最新 wheel 安装

`output/` 中可能保留多个 wheel，不要使用未经排序的通配符直接安装。

```bash
cd /home/l00934901/sgl-kernel-npu

latest_wheel="$({
  find output -maxdepth 1 -type f -name 'deep_ep*.whl' -printf '%T@ %p\n' 2>/dev/null || true
} | sort -nr | awk 'NR==1 {print $2}')"

test -n "${latest_wheel}" || { echo "没有找到 deep_ep wheel"; exit 1; }
echo "installing: ${latest_wheel}"
python3 -m pip install --force-reinstall --no-deps "${latest_wheel}"
```

安装后必须加载自定义算子环境：

```bash
cd /home/l00934901/sgl-kernel-npu
source python/deep_ep/deep_ep/vendors/hwcomputing/bin/set_env.bash
```

如果 wheel 安装位置不在源码目录，也可以定位安装后的脚本：

```bash
source "$(python3 - <<'PY'
import deep_ep
from pathlib import Path
print(Path(deep_ep.__file__).resolve().parent / "vendors/hwcomputing/bin/set_env.bash")
PY
)"
```

## 5. 阶段一：普通 HCCL AllToAll 的 CCU 基线

后续验证环境固定使用物理卡2～5。阶段一先使用物理卡2、3，它们映射成逻辑 rank 0、1：

```bash
cd /home/l00934901/sgl-kernel-npu
source python/deep_ep/deep_ep/vendors/hwcomputing/bin/set_env.bash

export ASCEND_RT_VISIBLE_DEVICES=2,3
export HCCL_BUFFSIZE=2300
export HCCL_OP_EXPANSION_MODE=CCU_SCHED
export ASCEND_LAUNCH_BLOCKING=1

python3 -m torch.distributed.run \
  --standalone --nproc-per-node=2 \
  tests/python/deepep/test_a5_hccl_ccu_all2all.py
```

成功标志：

```text
PASS: fixed HCCL AllToAll executed through A5 CCU
```

若仍在 `HcclAllocComResourceByTiling` 返回 `ret=5`，说明问题位于 CANN/HCCL CCU MC2 运行环境，而不是绕路算法。此时不要继续阶段二排查 CCU。

## 6. 阶段二：AIV+URMA 显式绕路

阶段二使用物理卡2～5：逻辑 rank 0、1（物理卡2、3）负责通信，逻辑 rank 2、3（物理卡4、5）作为绕路卡。必须启动4个进程，让4张卡都进入同一个 HCCL world group：

```bash
cd /home/l00934901/sgl-kernel-npu
source python/deep_ep/deep_ep/vendors/hwcomputing/bin/set_env.bash

export ASCEND_RT_VISIBLE_DEVICES=2,3,4,5
export HCCL_BUFFSIZE=2300
export DEEP_USE_MODE=default
unset HCCL_OP_EXPANSION_MODE
export ASCEND_LAUNCH_BLOCKING=1

python3 -m torch.distributed.run \
  --standalone --nproc-per-node=4 \
  tests/python/deepep/test_a5_aiv_urma_all2all_detour.py \
  --comm-ranks 0,1 \
  --elements-per-peer 1536 \
  --warmup 10 \
  --iters 100
```

成功标志：

```text
PASS: A5 AIV+URMA AllToAll detour (communication-rank Write + Read)
comm_ranks=[0, 1], relay_ranks=[2, 3]
```

注意：`comm-ranks` 是可见设备重新编号后的逻辑 rank，不是物理卡号。这里的逻辑通信 rank 0、1分别对应物理卡2、3，逻辑绕路 rank 2、3分别对应物理卡4、5。

当前 kernel 只支持恰好两个通信 rank，但允许零个或多个 relay rank。每个通信 rank 启动的 AIV block 分工如下：

```text
block 0：本 rank 数据复制
block 1：直连路径
block 2：relay rank 2
block 3：relay rank 3
```

这些 block 独立执行，不存在“先直连、再 relay 1、再 relay 2”的串行循环。直连路径分配两个虚拟 lane，每条 relay 路径分配一个虚拟 lane，因此直连数据量约为每条 relay 的2倍。所有 lane 均按32字节切分，余数按 lane 顺序分配。

本分支没有增加 block 数量，也没有改变上述切分方法。变化仅发生在每个 block 内部：原来的单 UB buffer `CopyBytes` 会等待一块数据完成 GM→UB→GM 后再处理下一块；现在改用 `CpGM2GM<uint8_t>`，使用两个约95 KiB的 UB buffer 交替搬运，使一块数据的 MTE3（UB→GM）可以和下一块数据的 MTE2（GM→UB）重叠。block 0的self copy、写通信窗口、读通信窗口均使用同一实现。

UB的前96字节保留给同步flag，两个数据buffer从 `UB_HEAD_OFFSET` 和 `UB_MID_OFFSET` 开始，避免flag与ping-pong数据区重叠。`CpGM2GM` 返回前会等待两块buffer上的MTE3全部完成，所以仍然保证“先写数据、再发布flag”。

默认 `1536 * sizeof(int32) = 6144` 字节，切分结果为：直连3072字节、relay 2为1536字节、relay 3为1536字节。

如果希望每个 peer 总共传输2 MiB，使用：

```bash
--elements-per-peer 524288
```

此时直连为1 MiB，每条 relay 为512 KiB。若使用 `524304`，总量为2097216字节，实际切分为直连1048640字节、每条 relay 524288字节；它控制的是每个 peer 的总量，不是单条路径的数据量。

需要检查 HCCL window 地址和各 block 分片时，只跑一次并打开调试输出：

```bash
export A5_DETOUR_DEBUG_WINDOWS=1
python3 -m torch.distributed.run \
  --standalone --nproc-per-node=4 \
  tests/python/deepep/test_a5_aiv_urma_all2all_detour.py \
  --comm-ranks 0,1 \
  --elements-per-peer 1536 \
  --warmup 0 \
  --iters 1
unset A5_DETOUR_DEBUG_WINDOWS
```

性能测试时不要设置 `A5_DETOUR_DEBUG_WINDOWS`，避免设备侧打印干扰计时。

### 6.1 两卡直连 baseline（同一算子）

当 `WORLD_SIZE` 与通信 rank 数相同，即两张卡都在 `--comm-ranks 0,1` 中时，算子自动退化为普通直连模式：`block 0` 复制本 rank 数据，`block 1` 搬运发往对端的全部数据，不创建 relay 路径。这样可以在不更换算子实现和计时方法的情况下与绕路模式比较。

```bash
export ASCEND_RT_VISIBLE_DEVICES=2,3
export HCCL_BUFFSIZE=2300
export DEEP_USE_MODE=default
unset HCCL_OP_EXPANSION_MODE
export ASCEND_LAUNCH_BLOCKING=1

python3 -m torch.distributed.run \
  --standalone --nproc-per-node=2 \
  tests/python/deepep/test_a5_aiv_urma_all2all_detour.py \
  --comm-ranks 0,1 \
  --elements-per-peer 524288 \
  --warmup 10 \
  --iters 100
```

预期路径输出为：

```text
used_aiv_blocks: 2 (self + direct + relays)
direct_bytes   : 2097152
relay_bytes    : []
```

随后使用第6节的4进程命令测试两条 relay。两次都保持 `--elements-per-peer 524288`，即可对比同一 AIV+URMA 算子的两卡直连与四卡绕路模式。

### 6.2 原生 torch.distributed/HCCL AllToAll

使用 `torch.distributed.all_to_all_single` 可以在 SGLang Docker 中直接测试原生 HCCL AllToAll，不经过本仓库的自定义 AllToAll kernel。测试脚本为：

```text
tests/python/deepep/test_a5_native_hccl_alltoall.py
```

当前脚本使用 `int32`。`--elements-per-peer 11804800` 表示每个目标peer的数据量为47219200字节；两卡时每个rank的输入包含一份self数据和一份对端数据，总计94438400字节。

原生CCU模式：

```bash
cd /home/l00934901/sgl-kernel-npu
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || \
source /usr/local/Ascend/cann/set_env.sh

export ASCEND_RT_VISIBLE_DEVICES=2,3
export HCCL_BUFFSIZE=2300
export HCCL_OP_EXPANSION_MODE=CCU_SCHED
unset ASCEND_LAUNCH_BLOCKING

python3 -m torch.distributed.run \
  --standalone --nproc-per-node=2 \
  tests/python/deepep/test_a5_native_hccl_alltoall.py \
  --elements-per-peer 11804800 \
  --warmup 10 \
  --iters 100
```

原生AIV模式需要重新启动进程：

```bash
export HCCL_OP_EXPANSION_MODE=AIV

python3 -m torch.distributed.run \
  --standalone --nproc-per-node=2 \
  tests/python/deepep/test_a5_native_hccl_alltoall.py \
  --elements-per-peer 11804800 \
  --warmup 10 \
  --iters 100
```

脚本会输出平均值、P50、P95、最小值、最大值，以及按每个rank实际对外发送字节数计算的有效带宽。CCU与AIV模式必须在不同的 `torchrun` 进程中测试，因为HCCL在通信域初始化时读取展开模式。

## 7. AIV+URMA 算法

以当前4卡 world、通信 rank `[0,1]` 为例，0发往1的数据按2:1:1分成3路：

```text
一路（2份）：rank0 Write rank1窗口，rank1 Read rank1窗口
两路（各1份）：rank0分别 Write rank2、rank3窗口，rank1再分别 Read rank2、rank3窗口
```

反方向同时使用独立的 `(srcRank,dstRank)` window cell：

```text
rank1 Write rank0、rank2、rank3窗口，rank0 Read相同窗口
```

每个 cell 包含同步 flag 和数据区。发送通信卡先写数据、再写带 generation 的 flag；接收通信卡等待对应 flag 后读数据。直连与每条 relay 由不同 AIV block 并发执行。非通信 rank 只提供远端映射窗口，不执行第二跳 kernel。

### 7.1 `CpGM2GM` 数据搬运流水

`CpGM2GM` 不是硬件直接支持的 GM→GM 单指令，实际仍由 MTE 完成 GM→UB→GM。它的优化点是把 UB 分成ping和pong两半：

```text
迭代0：MTE2 GM→UB ping；MTE3 UB ping→GM
迭代1：MTE2 GM→UB pong；MTE3 UB pong→GM
迭代2：等待ping的上一次MTE3完成后复用ping
```

不同UB半区之间允许MTE2和MTE3形成流水。该实现参考本仓库 `NotifyDispatchA5::CpGM2GMPingPong` 以及HCCL `AivCommBase::CpGM2GM` 的处理方式，但没有引入HCCL私有的 `AivCommBase` 和 `TQue` 依赖。

需要确认有卡环境确实加载了本分支新kernel时，可临时打开：

```bash
export A5_DETOUR_DEBUG_WINDOWS=1
```

设备日志应包含：

```text
[A5 copy debug] CpGM2GM ping-pong
```

性能测试前必须取消该变量：

```bash
unset A5_DETOUR_DEBUG_WINDOWS
```

## 8. 使用 msprof 采集算子性能

不需要修改 device kernel 添加打点。仓库提供脚本，通过 CANN 9.1 自带的 `msprof` 采集 AICore、MTE、Runtime 和 HCCL 数据。每次采集会在 `/home/l00934901/profiling` 下创建独立的时间戳目录。`torchrun` 的不同rank/device可能分别生成一个 `PROF_*` 目录，脚本会发现本次运行生成的全部 `PROF_*`，逐个执行 `msprof --export=on`，不会只导出时间最新的一张卡。

脚本位置：

```text
scripts/profile_a5_all2all_detour.sh
```

### 8.1 两卡直连

当前测试数据为 `int32`。`11804800 * sizeof(int32) = 47219200 Byte`，表示两卡之间的对端数据量为47219200字节：

```bash
cd /home/l00934901/sgl-kernel-npu

bash scripts/profile_a5_all2all_detour.sh \
  --visible-devices 2,3 \
  --comm-ranks 0,1 \
  --elements-per-peer 11804800 \
  --warmup 10 \
  --iters 100 \
  --metrics PipeUtilization
```

结果目录类似：

```text
/home/l00934901/profiling/a5_all2all_2card_PipeUtilization_YYYYmmdd_HHMMSS/
```

脚本会自动设置 `ASCEND_RT_VISIBLE_DEVICES`，并根据设备列表自动得到 `--nproc-per-node=2`。profiling时会取消 `ASCEND_LAUNCH_BLOCKING` 和设备侧调试打印，避免改变正常的异步调度并减少干扰。

如果要对原生 HCCL AllToAll 做同样的 `msprof` 采集，增加 `--target native`。原生算法模式由运行脚本前设置的 `HCCL_OP_EXPANSION_MODE` 决定：

```bash
export HCCL_OP_EXPANSION_MODE=CCU_SCHED

bash scripts/profile_a5_all2all_detour.sh \
  --target native \
  --visible-devices 2,3 \
  --elements-per-peer 11804800 \
  --metrics PipeUtilization
```

自定义绕路算子保持默认 `--target custom`。该模式会在脚本内部取消 `HCCL_OP_EXPANSION_MODE`。

### 8.2 八卡运行、六卡绕路

八卡全部可用时，逻辑rank 0、1通信，逻辑rank 2～7作为六张relay卡：

```bash
bash scripts/profile_a5_all2all_detour.sh \
  --visible-devices 0,1,2,3,4,5,6,7 \
  --comm-ranks 0,1 \
  --elements-per-peer 11804800 \
  --warmup 10 \
  --iters 100 \
  --metrics PipeUtilization
```

### 8.3 采集 Memory 指标

`msprof`一次采集一组 AIC metrics。检查 MTE2/MTE3 流水先使用 `PipeUtilization`；检查 GM/HBM 访问再单独采集一次 `Memory`：

```bash
bash scripts/profile_a5_all2all_detour.sh \
  --visible-devices 2,3 \
  --comm-ranks 0,1 \
  --elements-per-peer 11804800 \
  --warmup 10 \
  --iters 100 \
  --metrics Memory
```

也可以覆盖默认输出根目录：

```bash
bash scripts/profile_a5_all2all_detour.sh \
  --output-root /home/l00934901/profiling \
  --visible-devices 2,3
```

查看所有导出的 CSV：

```bash
find /home/l00934901/profiling -type f -name '*.csv' | sort
```

确认一次多卡采集实际生成了多少份原始profile：

```bash
latest_run="$(find /home/l00934901/profiling -mindepth 1 -maxdepth 1 -type d \
  -name 'a5_*_all2all_*' -printf '%T@ %p\n' | sort -nr | awk 'NR==1 {print $2}')"
echo "${latest_run}"
find "${latest_run}" -type d -name 'PROF_*' | sort
```

正常情况下，脚本结束时会先输出发现的profile目录数量，再列出每个rank/device导出的CSV。如果这里只找到一个 `PROF_*`，问题发生在msprof对子进程的采集阶段，而不是导出阶段；此时需要再改为每个rank分别启动msprof，不能把缺失数据误判为导出脚本遗漏。

查找当前算子记录：

```bash
grep -Rin 'All2AllDetourIoDie' \
  /home/l00934901/profiling/*/PROF_*/mindstudio_profiler_output 2>/dev/null
```

重点查看 `op_summary*.csv` 中的算子时长、Block Dim、MTE2/MTE3利用率和GM访问指标。脚本默认先warmup 10次，再正式profiling 100次，以便与常规benchmark的迭代配置保持一致；相应的采集文件会比较大。

注意：`/home/l00934901/profiling` 必须在有卡Docker内可写。当前Docker已挂载 `/home/l00934901` 时，profiling结果会直接保存在宿主机目录中，容器删除后仍然保留。

## 9. 代码位置

普通 CCU AllToAll：

```text
csrc/deepep/ops/op_host/hccl_all2_all_ccu_def.cpp
csrc/deepep/ops/op_host/hccl_all2_all_ccu_tiling.cpp
csrc/deepep/ops/op_host/op_api/aclnn_hccl_all2_all_ccu.{h,cpp}
csrc/deepep/ops/op_kernel/hccl_all2_all_ccu.cpp
tests/python/deepep/test_a5_hccl_ccu_all2all.py
tests/python/deepep/test_a5_native_hccl_alltoall.py
```

AIV+URMA 绕路：

```text
csrc/deepep/ops/op_host/all2_all_detour_io_die_def.cpp
csrc/deepep/ops/op_host/all2_all_detour_io_die_tiling.cpp
csrc/deepep/ops/op_host/op_api/aclnn_all2_all_detour_io_die.{h,cpp}
csrc/deepep/ops/op_kernel/data_copy.h
csrc/deepep/ops/op_kernel/all2_all_detour_io_die.cpp
tests/python/deepep/test_a5_aiv_urma_all2all_detour.py
scripts/profile_a5_all2all_detour.sh
```
