# A5 AllToAll：CCU 基线与 AIV+URMA 绕路验证

本分支包含两个彼此独立的算子和测试。必须先跑阶段一，再跑阶段二，这样可以把 CCU/HCCL 环境问题与自定义绕路算法问题分开。

| 阶段 | Python 接口 | 设备实现 | 用途 |
|---|---|---|---|
| 1 | `Buffer.hccl_all2_all_ccu` | `Hccl::AlltoAll` + `HCCL_CMD_ALLTOALL` + A5 CCU | 证明当前环境可以运行 HCCL 普通 AllToAll |
| 2 | `Buffer.all2_all_detour_io_die` | AIV + MTE/URMA 映射窗口 | 通信卡 Write 绕路卡，通信卡再从绕路卡 Read |

阶段一没有复制 HCCL 的私有 CCU kernel。自定义算子只是 MC2 薄封装，真正执行的是 CANN/HCCL 已注册的 `CcuAlltoAllMesh1D`、`CcuAllToAllMesh2Die` 或对应拓扑实现。

阶段二不是 CCU，也不是透明 IO Die cut-through。非通信卡提供 HCCL/URMA 注册的 HBM 窗口，数据在绕路卡 HBM 中转。

## 1. 获取分支

在有卡服务器的 SGLang Docker 中执行：

```bash
cd /home/l00934901/sgl-kernel-npu

git fetch origin feature/a5-io-die-all2all-detour
git switch feature/a5-io-die-all2all-detour
git pull --ff-only origin feature/a5-io-die-all2all-detour

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

这两个 kernel 已在无卡服务器的 CANN 9.1 `cam_lyw_dev_91` 环境完成编译验证；无卡环境只能验证出包，不能验证通信结果。

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
  --elements-per-peer 1536
```

成功标志：

```text
PASS: A5 AIV+URMA AllToAll detour (communication-rank Write + Read)
comm_ranks=[0, 1], relay_ranks=[2, 3]
```

注意：`comm-ranks` 是可见设备重新编号后的逻辑 rank，不是物理卡号。这里的逻辑通信 rank 0、1分别对应物理卡2、3，逻辑绕路 rank 2、3分别对应物理卡4、5。`1536 * sizeof(int32) = 6144` 字节，平均分到直达路径和两条绕路路径后每路为2048字节，满足当前 MTE 拷贝的32字节对齐要求。

## 7. AIV+URMA 算法

以当前4卡 world、通信 rank `[0,1]` 为例，0发往1的数据平均分成3路：

```text
一路：rank0 Write rank1窗口，rank1 Read rank1窗口
两路：rank0分别 Write rank2、rank3窗口，rank1再分别 Read rank2、rank3窗口
```

反方向同时使用独立的 `(srcRank,dstRank)` window cell：

```text
rank1 Write rank0、rank2、rank3窗口，rank0 Read相同窗口
```

每个 cell 包含同步 flag 和数据区。发送通信卡先写数据、再写带 generation 的 flag；接收通信卡等待对应 flag 后读数据。非通信 rank 只提供远端映射窗口，不执行第二跳 kernel。

## 8. 代码位置

普通 CCU AllToAll：

```text
csrc/deepep/ops/op_host/hccl_all2_all_ccu_def.cpp
csrc/deepep/ops/op_host/hccl_all2_all_ccu_tiling.cpp
csrc/deepep/ops/op_host/op_api/aclnn_hccl_all2_all_ccu.{h,cpp}
csrc/deepep/ops/op_kernel/hccl_all2_all_ccu.cpp
tests/python/deepep/test_a5_hccl_ccu_all2all.py
```

AIV+URMA 绕路：

```text
csrc/deepep/ops/op_host/all2_all_detour_io_die_def.cpp
csrc/deepep/ops/op_host/all2_all_detour_io_die_tiling.cpp
csrc/deepep/ops/op_host/op_api/aclnn_all2_all_detour_io_die.{h,cpp}
csrc/deepep/ops/op_kernel/all2_all_detour_io_die.cpp
tests/python/deepep/test_a5_aiv_urma_all2all_detour.py
```
