# A5 CCU+URMA 显式多路径 AllToAll 操作指南

## 1. 当前实现

当前正式实现是标准 Ascend C 算子 `ExplicitMultipathAll2AllCcu`，不是 prepared-plan
`torch.ops` 包装。它具备标准 `op_def`、host tiling、AIV kernel、ACLNN API 和二进制 kernel。

路径资源在 communicator 初始化时由配套 HCCL 扩展建立；每次图执行时，AIV 从 NPU 上的
`pathPolicy` Tensor 读取路径权重并通过 MC2 消息交给 HCCL/CCU。ACLGraph replay 前可以原地更新
这个 Tensor，从而动态启用、禁用或重新分配已经 provision 的 direct/relay Channel。

边界如下：

- 当前仅支持同机两 rank；
- 路径目录为 1 条 direct 加 1..12 条显式 relay；
- 动态 policy 只能选择已建好的 Channel，不能在 replay 中新建 relay；
- 不安装 `ubus.ko`，不修改全局 UB route table；
- 需要本分支配套的 HCCL 扩展 ABI 3 和 DeepEP ABI 8。

## 2. 在有卡环境准备干净的 HCCL 仓库

当前有卡环境的旧 HCCL 工作区曾用于手工穿刺，并且已经出现损坏的 loose object。此时不能再用
`stash/reset/pull` 修补原仓库；Git 不会因为普通 fetch 自动替换它误认为已经存在的损坏对象。
保留整个旧仓库，然后重新 clone 干净分支：

```bash
cd /home/l00934901

STAMP=$(date +%Y%m%d_%H%M%S)
OLD_HCCL="/home/l00934901/hccl_corrupt_${STAMP}"

# 同一文件系统内只是重命名，旧源码、stash 和构建依赖仍可恢复。
mv /home/l00934901/hccl "${OLD_HCCL}"
echo "old HCCL: ${OLD_HCCL}"

env -u GIT_ASKPASS -u SSH_ASKPASS \
git clone \
  --single-branch \
  --branch feature/a5-ccu-explicit-multipath-alltoall \
  https://gitcode.com/yuanwenliu/hccl.git \
  /home/l00934901/hccl
```

检查新仓库：

```bash
cd /home/l00934901/hccl

git rev-parse --short HEAD
git status --short
git fsck --full
```

预期 HEAD 至少包含 `c6e51b2`，且 `git fsck --full` 不报告损坏对象。不要从旧仓库执行
`stash pop`，也不要复制旧 `.git`、源码或构建目录。

如果离线编译确实需要旧仓库中的未跟踪 `third_party`，只复用这个依赖目录：

```bash
if test -d "${OLD_HCCL}/third_party"; then
  ln -s "${OLD_HCCL}/third_party" /home/l00934901/hccl/third_party
fi
```

旧仓库先保留；新仓库编译、运行通过后再人工清理。

## 3. 拉取 sgl-kernel-npu

```bash
cd /home/l00934901/sgl-kernel-npu
git fetch origin
git switch feature/a5-ccu-explicit-multipath-alltoall
git pull --ff-only origin feature/a5-ccu-explicit-multipath-alltoall
```

## 4. 编译配套 HCCL

```bash
cd /home/l00934901/hccl
source /usr/local/Ascend/cann-9.1.T560/set_env.sh

./build.sh --pkg -j4 -p /usr/local/Ascend/cann-9.1.T560

HCCL_RUNTIME=$(find /home/l00934901/hccl/build/_CPack_Packages \
  -type f -name libhccl.so -printf '%T@ %h\n' 2>/dev/null | \
  sort -nr | awk 'NR == 1 { print $2 }')

echo "HCCL_RUNTIME=${HCCL_RUNTIME}"
test -f "${HCCL_RUNTIME}/libhccl.so"
test -f "${HCCL_RUNTIME}/libhccl_compat.so"
```

验证扩展 ABI。必须先以 global 模式加载同一目录的 `libhccl_compat.so`，否则直接 `ctypes.CDLL`
可能报告 `IsHcommDefaultTimeoutSupported` 等未解析符号：

```bash
LD_LIBRARY_PATH="${HCCL_RUNTIME}:${LD_LIBRARY_PATH:-}" \
python3 - "${HCCL_RUNTIME}/libhccl_compat.so" \
          "${HCCL_RUNTIME}/libhccl.so" <<'PY'
import ctypes
import sys

ctypes.CDLL(sys.argv[1], mode=ctypes.RTLD_GLOBAL)
lib = ctypes.CDLL(sys.argv[2], mode=ctypes.RTLD_GLOBAL)
version = lib.A5HcclExplicitMultipathExtensionVersion
version.restype = ctypes.c_int
print("HCCL explicit-multipath ABI:", version())
assert version() >= 3
PY
```

不要把这套实验库覆盖到公共 CANN 目录。测试脚本通过 `--hccl-lib-dir` 只给本次子进程注入它。

## 5. 编译标准算子和 DeepEP wheel

```bash
cd /home/l00934901/sgl-kernel-npu
source /usr/local/Ascend/cann-9.1.T560/set_env.sh

DEEPEP_SINGLE_OP=explicit_multipath_all2all_ccu \
bash build.sh -a deepep Ascend950

WHEEL=$(ls -t output/deep_ep-*.whl | head -1)
python3 -m pip install --force-reinstall --no-cache-dir --no-deps "${WHEEL}"
```

如果在开发 Docker `cam_lyw_dev_91` 中构建，先进入包含 torch/torch_npu 的 Python 环境：

```bash
source /opt/conda/bin/activate cam_py311_pt28
```

安装后不会退出 Docker。验证 wheel：

```bash
python3 - <<'PY'
import ctypes
from pathlib import Path
import deep_ep.deep_ep_cpp as ext

path = Path(ext.__file__).resolve()
lib = ctypes.CDLL(str(path))
version = lib.A5DeepEpExplicitMultipathAttrAbiVersion
version.restype = ctypes.c_int
print("loaded deep_ep_cpp:", path)
print("dynamic PathPolicy ABI:", version())
assert version() >= 8
assert hasattr(ext.Buffer, "explicit_multipath_all2all_ccu")
PY
```

## 6. PathPolicy 格式

`pathPolicy` 是设备上的 `[1]`、`int64` Tensor：

```text
bits 60..63 : ABI，当前为 1
nibble 0    : direct 权重
nibble 1    : relay 0 权重
nibble 2    : relay 1 权重
...
```

每个权重为 `0..15`；0 表示本次执行不在该路径发 payload。至少一条路径必须非零。
路径目录最多 13 条，因为当前 CCU task argument cache 最多容纳 1 direct + 12 relay。

默认性能约定是：

```text
direct 数据量 : 每一条 relay 数据量 = 2 : 1
```

因此 direct + 2 relay 使用 `2,1,1`，不是 `4,1,1`。

## 7. 快速单 case

下面以物理卡 `2,3` 为通信端点、`0,1` 为 relay：

```bash
cd /home/l00934901/sgl-kernel-npu
source /usr/local/Ascend/cann-9.1.T560/set_env.sh
unset LD_PRELOAD

bash scripts/run_a5_ccu_explicit_multipath_alltoall_perf.sh \
  --src-phy 2 --dst-phy 3 \
  --relay-phys 0,1 \
  --direct-route 0 \
  --cases direct_plus_relays \
  --bytes 4194304 \
  --warmup 1 --iters 1 --repeats 1 \
  --timeout-seconds 300 \
  --hccl-lib-dir "${HCCL_RUNTIME}" \
  --cann-root /usr/local/Ascend/cann-9.1.T560 \
  --output-root /home/l00934901/profiling
```

`direct_plus_relays` 使用 `--relay-phys` 中的全部卡，可传 1..12 张 relay；卡号没有写死。
日志应包含：

```text
Verified patched HCCL: ... extension=3
Verified requested APIs: ... standard=True; DeepEP ABI=8
PASS: implementation=standard ...
```

## 8. ACLGraph capture/replay 与动态选路

该实验第一次执行使用 `2,1,1`，replay 前把稳定地址的 policy Tensor 原地改为 `2,0,1`，即关闭
relay 0、保留 direct 和 relay 1：

```bash
bash scripts/run_a5_ccu_explicit_multipath_alltoall_perf.sh \
  --src-phy 2 --dst-phy 3 \
  --relay-phys 0,1 \
  --direct-route 0 \
  --cases direct_plus_relays \
  --graph-backend aclgraph \
  --replay-path-weights 2,0,1 \
  --bytes 4194304 \
  --warmup 1 --iters 3 --repeats 1 \
  --timeout-seconds 600 \
  --hccl-lib-dir "${HCCL_RUNTIME}" \
  --cann-root /usr/local/Ascend/cann-9.1.T560 \
  --output-root /home/l00934901/profiling
```

检查：

```bash
RUN_DIR=$(ls -dt \
  /home/l00934901/profiling/a5_ccu_explicit_multipath_2_3_* | head -1)
LOG="${RUN_DIR}/cases/direct_plus_relays_r1.log"

grep -nE \
'CASE_ACLGRAPH_CAPTURE|CASE_DYNAMIC_PATH_POLICY|CASE_ACLGRAPH_REPLAY|CASE_FAILURE|PASS:' \
"${LOG}"
```

必须看到两个 rank 的 capture/replay PASS，并看到：

```text
CASE_DYNAMIC_PATH_POLICY ... weights=[2, 0, 1]
```

这证明 policy 是 replay 时从设备 Tensor 读取，而非固化在 host tiling。数据正确性只能证明动态图执行
正确；若要证明被关闭的 relay 确实无 payload，需同时采 HCCN 端口计数。

## 9. 性能矩阵

```bash
bash scripts/run_a5_ccu_explicit_multipath_alltoall_perf.sh \
  --src-phy 2 --dst-phy 3 \
  --relay-phys 4,5,1,0,6,7 \
  --direct-route 0 \
  --cases native0,native2,direct_plus_2relay,direct_plus_4relay,direct_plus_6relay \
  --bytes 4194304 \
  --warmup 100 --iters 20 --repeats 3 \
  --timeout-seconds 600 \
  --hccl-lib-dir "${HCCL_RUNTIME}" \
  --cann-root /usr/local/Ascend/cann-9.1.T560 \
  --output-root /home/l00934901/profiling
```

`native0/native2` 仍由 route-probe baseline 执行；显式 case 使用标准算子。查看结果：

```bash
RUN_DIR=$(ls -dt \
  /home/l00934901/profiling/a5_ccu_explicit_multipath_2_3_* | head -1)
column -s $'\t' -t "${RUN_DIR}/case_status.tsv"
sed -n '1,240p' "${RUN_DIR}/explicit_multipath_perf_report.md"
grep -RHnE 'PASS:|CASE_FAILURE|ExplicitMultipath' "${RUN_DIR}/cases"
```

## 10. 只采多路径 profiling

```bash
bash scripts/run_a5_ccu_explicit_multipath_alltoall_perf.sh \
  --src-phy 2 --dst-phy 3 \
  --relay-phys 4,5,1,0,6,7 \
  --direct-route 0 \
  --cases direct_plus_6relay \
  --bytes 4194304 \
  --warmup 100 --iters 20 --repeats 1 \
  --profile --profile-iters 20 \
  --timeout-seconds 600 \
  --hccl-lib-dir "${HCCL_RUNTIME}" \
  --cann-root /usr/local/Ascend/cann-9.1.T560 \
  --output-root /home/l00934901/profiling
```

MindStudio 中一个 AIV/CCU launch 内部可以包含多个 Channel 的并发 Write，不能用 launch 数量推断
物理路径数量；relay 身份仍以 manifest 和 HCCN 端口计数为准。

## 11. 常见故障

| 现象 | 处理 |
|---|---|
| HCCL extension `< 3` 或 symbol missing | 重新编译第 4 节分支，并传正确的 `--hccl-lib-dir` |
| DeepEP ABI `< 8` | 重新编译并 force-reinstall 第 5 节 wheel |
| opbuild 报 input dtype size 0 | `pathPolicy` 必须为 sendData 的 4 个 dtype 组合分别声明 `DT_INT64` |
| `IsHcommDefaultTimeoutSupported` undefined | 先加载同目录 `libhccl_compat.so`，再加载 `libhccl.so` |
| `ChannelAcquire status=9` | 指定 EID path 当前未 provision；检查 relay manifest、plane 和拓扑 |
| standard case 提示 route-probe stale | 脚本过旧；标准 case 不依赖 prepared-plan route API |
| `--graph-backend npugraphs` 被拒绝 | 当前标准动态 policy 验证使用 `aclgraph`；不要套用旧 prepared backend |
| `status=137`，栈停在 `triton/tools/get_ascend_devices.py` 的 `npu-smi` | 旧脚本让 `npu-smi` 继承了实验 HCCL 的 `LD_PRELOAD`；更新脚本后会通过 `scripts/wrappers/npu-smi` 自动隔离。该 137 是外层 timeout 强杀，不是算子 OOM |
| replay 正确但端口没有变化 | 先确认 `CASE_DYNAMIC_PATH_POLICY`，再用 HCCN 在同一时间窗验证物理路径 |
