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
- 需要本分支配套的 HCCL 扩展 ABI 5 和 DeepEP ABI 8。ABI 4 首先让标准算子通过
  `HCCL_CMD_ALLTOALL + CCU_SCHED` 进入 HCCL，而不再伪装成
  `HALF_ALLTOALLV + CCU_MS`；ABI 5 进一步修正 HCCL 原生 AllToAll Channel 的资源槽位为
  `INPUT=0、OUTPUT=1、TOKEN=2`。

## 2. 在有卡环境准备干净的 HCCL 仓库

如果 `/home/l00934901/hccl` 是健康仓库，直接拉取配套分支：

```bash
cd /home/l00934901/hccl

# 确认没有遗留的 Git 锁或损坏对象。
test ! -e .git/HEAD.lock
git fsck --full

git fetch origin feature/a5-ccu-explicit-multipath-alltoall
git switch feature/a5-ccu-explicit-multipath-alltoall
git pull --ff-only origin feature/a5-ccu-explicit-multipath-alltoall

git rev-parse --short HEAD
git status --short
```

预期 HEAD 至少包含 `a9eb52a`（ABI 5 / 原生 AllToAll Channel 资源布局）。如果 `git status --short`
列出源码修改，不要直接 reset；先确认它们是否为需要保留的本地工作。

只有当 `git fsck --full` 报告 corrupt loose object，或者正常 fetch/pull 因对象损坏失败时，才使用下面的
重新 clone 流程。Git 不会因为普通 fetch 自动替换它误认为已经存在的损坏对象。保留整个旧仓库，
然后重新 clone 干净分支：

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

预期 HEAD 至少包含 `a9eb52a`（ABI 5 / 原生 AllToAll Channel 资源布局），且 `git fsck --full` 不报告
损坏对象。不要从旧仓库执行
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
assert version() >= 5
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
Verified patched HCCL: ... extension=5
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

## 11. 标准算子进入 HCCL 的正确路径

ABI 4/5 修正后的控制链是：

```text
ExplicitMultipathAll2AllCcu host tiling
  -> opType = HCCL_CMD_ALLTOALL
  -> commEngine = CCU_SCHED (6)
  -> HcclAllocComResourceByTilingV2
  -> AlltoAllAutoSelector::SelectCcuScheduleAlgo
  -> CcuExplicitMultipathAllToAll2Rank
  -> CcuTempExplicitMultipathAllToAll::CalcRes
  -> direct + 显式 relay Channel catalog
  -> custom CCU kernel
```

旧 ABI 3 使用 `HALF_ALLTOALLV + CCU_MS (5)`。有卡日志已经证明它会在 selector 之前被
`communicator_impl.cc:GetTilingAccelerator` 拒绝：

```text
Tiling hcclAccelerator not support, hcclAccelerator[HcclAccelerator::CCU_MS]
HcclAllocComResourceByTilingV2 ret=5
```

这不是 relay EID、Channel 或 CCU kernel 的错误。ABI 4 删除了通用 selector 中针对
`HALF_ALLTOALLV` 的提前劫持，也删除了该 op type 下的重复 template 注册。

ABI 4 首次有卡运行已跨过上述 gate，但两个 rank 卡在 warmup 的 `torch.npu.synchronize()`。
与原生 `ccu_kernel_all_to_all_mesh1d.cc` 对比后确认，HCCL provision 的 Channel 使用：

```text
XN 0 = peer input
XN 1 = peer output
XN 2 = peer token
```

旧自定义 kernel 沿用了 route-probe 的私有 `OUTPUT=0、TOKEN=1` 布局，导致把 peer input 当输出、
把 peer output 当 token，并卡在 `PreSync/Write/EventWait`。ABI 5 改为原生 `OUTPUT=1、TOKEN=2`；
运行脚本和 DeepEP C++ 都会拒绝 ABI 4，防止误用旧库。

Native `CcuAlltoAllMesh1DMultiJetty` 仅作为 `CCU_SCHED` 入口与资源协议的参考；显式 relay
仍采用“一条物理路径一个 Channel”，因为原生 MultiJetty 没有公开 `jetty -> EID/relay` 绑定接口。

### 11.1 首次最小验证

完成第 4、5 节的重新编译后，先跑第 7 节单 case，然后检查：

```bash
RUN_DIR=$(ls -dt \
  /home/l00934901/profiling/a5_ccu_explicit_multipath_* | head -1)
LOG="${RUN_DIR}/cases/direct_plus_relays_r1.log"

grep -nE \
'selector MATCH|CalcRes begin|CalcRes ready|provisioned|KernelRun begin|CASE_FAILURE|PASS:' \
"${LOG}"
```

若普通 case log 没包含 HCCL INFO 日志，再按 11.2 收集对应 plog。成功路径至少应依次看到：

```text
[ExplicitMultipath] selector MATCH ... opType[10] executeConfig[6]
[ExplicitMultipath] CalcRes begin ...
[ExplicitMultipath] provisioned 3 channels (direct + 2 relays)
[ExplicitMultipath] CalcRes ready ... paths[3]
[ExplicitMultipath] KernelRun begin ...
PASS: implementation=standard ...
```

其中 `opType[10]` 对应本仓枚举里的 `HCCL_CMD_ALLTOALL`；以符号和日志语义为准，不要把
数字 10 与 route index 混淆。

### 11.2 失败时收集 HCCL/MC2 边界日志

如果标准算子在第一次 warmup 失败，且日志包含：

```text
Nnopbase fails to invoke the HcclAllocComResourceByTiling function
ret = 5
```

说明失败发生在 MC2/HCCL 通信资源分配阶段，还没有进入 CCU kernel 和
`WriteNb`。当前 HCCL 错误码中 `5` 为 `HCCL_E_NOT_SUPPORT`。不要先调整数据切分、
path weight 或性能参数，先执行下面两组检查。

下面的命令会自动选择最新的显式多路径运行目录；不要把
`YYYYMMDD_HHMMSS` 当成实际目录名执行。如需检查更早的运行，再将第一行手工改成该目录的完整路径。

```bash
(
set -e

RUN_DIR=$(
  ls -dt /home/l00934901/profiling/a5_ccu_explicit_multipath_* \
    2>/dev/null | head -1
)

if [[ -z "${RUN_DIR}" || ! -d "${RUN_DIR}" ]]; then
  echo "ERROR: no explicit-multipath result directory found" >&2
  false
fi

echo "RUN_DIR=${RUN_DIR}"
LOG="${RUN_DIR}/cases/direct_plus_relays_r1.log"

if [[ ! -f "${LOG}" ]]; then
  echo "ERROR: missing case log: ${LOG}" >&2
  false
fi

PIDS=$(
  grep -oE 'PID: ?[0-9]+' "${LOG}" |
  grep -oE '[0-9]+' |
  sort -u
)

echo "PIDS=${PIDS}"

PLOG_LIST="${RUN_DIR}/hccl_plogs.txt"
: > "${PLOG_LIST}"

for pid in ${PIDS}; do
  find /root/ascend/log -type f \
    -name "plog-${pid}_*.log" -print 2>/dev/null
done | sort -u > "${PLOG_LIST}"

cat "${PLOG_LIST}"

TRACE_OUT="${RUN_DIR}/hccl_resource_failure_trace.txt"
: > "${TRACE_OUT}"

while IFS= read -r file; do
  grep -HnE \
'ExplicitMultipath|CcuExplicit|ALLTOALL|CCU_MS|CCU_SCHED|GetTilingAccelerator|Select|CalcRes|BuildExplicitChannels|exactly two ranks|same local die|provisioned|HcclAllocComResourceByTiling|CcuInstructions|isEmpty|NOT_SUPPORT|not support|ret.?=.?5' \
  "${file}" >> "${TRACE_OUT}" 2>/dev/null || true
done < "${PLOG_LIST}"

sed -n '1,300p' "${TRACE_OUT}"
)
```

按下表判读：

| 日志特征 | 结论 |
|---|---|
| `CCU_MS` + `GetTilingAccelerator ... not support` | 仍加载了旧 OPP/tiling 或 HCCL ABI `< 5`；重新执行第 4、5 节 |
| ABI 4、warmup 后卡在 `torch.npu.synchronize()` | 旧 CCU kernel 使用错误的 `OUTPUT=0/TOKEN=1`；更新到 ABI 5 并重编 HCCL |
| `selector MATCH ... executeConfig[6]` | 已正确进入 AllToAll 的 CCU_SCHED selector |
| `exactly two ranks and one native direct channel are required` | `CalcChannelRequest...` 返回的原生 Channel 数量不符合当前 template 假设 |
| `all channels in one launch must use the same local die` | relay plan 中本地 die 不一致 |
| ABI 为 4 但完全没有 `[ExplicitMultipath]` | manifest/plan 环境变量未传入 worker，或标准算子仍未发出 `HCCL_CMD_ALLTOALL` |
| 已有 `provisioned N channels`，随后仍 `ret=5` | HCCL 已构造 Channel，失败在 MC2/libmc2_client 资源序列化或 instruction 生成 |
| `CcuInstructions isEmpty` | 闭源 MC2 未识别新的 CCU instruction，与早期标准算子尝试的失败相同 |
| `param.varMemSize ... invalid` | 固定 AllToAll 与 `HALFALLTOALLV` 资源协议不匹配 |

### 11.3 核对实际传入 HCCL 的 relay plan

这一段可以脱离 11.1 单独复制执行，因此会重新自动确定 `RUN_DIR`：

```bash
(
set -e

RUN_DIR=$(
  ls -dt /home/l00934901/profiling/a5_ccu_explicit_multipath_* \
    2>/dev/null | head -1
)

if [[ -z "${RUN_DIR}" || ! -d "${RUN_DIR}" ]]; then
  echo "ERROR: no explicit-multipath result directory found" >&2
  false
fi

echo "RUN_DIR=${RUN_DIR}"
find "${RUN_DIR}/plans" -maxdepth 2 -type f -print

for file in "${RUN_DIR}"/plans/*.tsv; do
  test -f "${file}" || continue
  echo "===== ${file} ====="
  column -s $'\t' -t "${file}"
done

sed -n '1,240p' "${RUN_DIR}/plans/multipath_plans.json"
)
```

重点核对：

- `src_phy/dst_phy/relay_phy` 是否为本次实际卡号；
- 每条 relay 的 `src_eid/dst_eid` 是否完整；
- 同一次 CCU launch 中各 relay 的本地 die 是否一致；
- plan id 是否与用例日志中传入的 plan id 一致；
- 不要将 HCCL extension 版本检查通过解释为 MC2 已接受自定义 CCU template。

如果已经出现 `selector MATCH` 而未出现 `CalcRes begin`，问题位于 selector 到 executor/template
注册之间；如果出现 `CalcRes ready` 后失败，才进入 Channel 资源序列化或 CCU instruction 生成范围。
不要再回退到 `HALF_ALLTOALLV + CCU_MS`。

## 12. 常见故障

| 现象 | 处理 |
|---|---|
| HCCL extension `< 5` 或 symbol missing | 重新编译第 4 节分支，并传正确的 `--hccl-lib-dir` |
| DeepEP ABI `< 8` | 重新编译并 force-reinstall 第 5 节 wheel |
| opbuild 报 input dtype size 0 | `pathPolicy` 必须为 sendData 的 4 个 dtype 组合分别声明 `DT_INT64` |
| `IsHcommDefaultTimeoutSupported` undefined | 先加载同目录 `libhccl_compat.so`，再加载 `libhccl.so` |
| `ChannelAcquire status=9` | 指定 EID path 当前未 provision；检查 relay manifest、plane 和拓扑 |
| standard case 提示 route-probe stale | 脚本过旧；标准 case 不依赖 prepared-plan route API |
| `--graph-backend npugraphs` 被拒绝 | 当前标准动态 policy 验证使用 `aclgraph`；不要套用旧 prepared backend |
| `status=137`，栈停在 `triton/tools/get_ascend_devices.py` 的 `npu-smi` | 旧脚本让 `npu-smi` 继承了实验 HCCL 的 `LD_PRELOAD`；更新脚本后会通过 `scripts/wrappers/npu-smi` 自动隔离。该 137 是外层 timeout 强杀，不是算子 OOM |
| replay 正确但端口没有变化 | 先确认 `CASE_DYNAMIC_PATH_POLICY`，再用 HCCN 在同一时间窗验证物理路径 |
