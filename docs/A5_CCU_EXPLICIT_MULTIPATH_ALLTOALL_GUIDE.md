# A5 CCU+URMA 显式多路径 AllToAll：方案 2 操作指南

## 1. 方案结论

本分支不再把显式路径计划塞进 Ascend C/MC2 tiling，也不要求修改或替换系统 HCCL。

原因是标准 NNOP 路径中的通信资源申请实际由系统 `libmc2_client.so` 完成；当前没有该组件源码，定制
`libhccl.so` 无法让这条调用链识别我们扩展的 plan。方案 2 将流程拆为：

1. 图外控制面 `prepare`：解析 direct/relay EID plan，执行 `HcclChannelAcquire`，注册 CCU kernel；
2. 图内数据面 `execute`：通过不透明的 `plan_handle` 启动已经准备好的 CCU+URMA 资源。

图内入口为：

```python
torch.ops.deep_ep.ccu_urma_prepared_multipath_alltoall(send, recv, plan_handle)
```

因此热路径不读取文件、不枚举 RankGraph、不重新建 Channel，也不经过
`HcclAllocComResourceByTiling -> libmc2_client.so`。

当前边界：仅支持同机两 rank；plan 与创建它的进程、communicator 和 NPU stream 绑定；一个进程退出前
plan 一直有效。

## 2. 拉取代码

只需更新 `sgl-kernel-npu`，不再需要切换或编译定制 HCCL 分支：

```bash
cd /home/l00934901/sgl-kernel-npu
git fetch origin
git switch feature/a5-ccu-explicit-multipath-alltoall
git pull --ff-only origin feature/a5-ccu-explicit-multipath-alltoall
```

保留 `/home/l00934901/hccl` 仅供 route package 构建脚本取得已有 HCOMM/HCCL 头文件；测试运行时使用
CANN 自带 HCCL。

## 3. 编译并安装显式路径运行库

```bash
cd /home/l00934901/sgl-kernel-npu
source /usr/local/Ascend/cann-9.1.T560/set_env.sh

HCCL_REPO=/home/l00934901/hccl \
bash scripts/build_a5_ccu_urma_route_probe.sh \
  --install \
  --install-path /usr/local/Ascend/cann-9.1.T560
```

脚本必须同时满足两个约束：HCCL 的 `add_subdirectory` 要求 custom-op 目录位于 HCCL 仓内；部分有卡
环境又会在跨工作区复制文件时令目标侧读到密文。为此脚本会在 HCCL 仓内建立临时**硬链接目录**，
不会复制源码字节。正常日志应包含：

```text
Custom ops source : /home/l00934901/sgl-kernel-npu/examples/a5_ccu_urma_route_probe
Hard-link staging : /home/l00934901/hccl/.a5_ccu_urma_route_probe.*
```

若环境不允许硬链接或两个仓不在同一文件系统，脚本会在进入 CMake 前明确失败；不要退回 `cp -a`。
若仍看到 `CMakeLists.txt:1 Parse error` 和乱码，先确认拉到了包含硬链接修复的最新提交，再检查：

```bash
cd /home/l00934901/sgl-kernel-npu
git status --short
grep -n 'cp -al' scripts/build_a5_ccu_urma_route_probe.sh
head -n 3 examples/a5_ccu_urma_route_probe/CMakeLists.txt
stat -c '%d %i %n' examples/a5_ccu_urma_route_probe/CMakeLists.txt
```

### 3.1 源 CMakeLists.txt 已显示为 data

若脚本在创建 hard-link staging 之前报告：

```text
Custom-op CMakeLists.txt is not readable text: .../CMakeLists.txt
.../CMakeLists.txt: data
```

说明损坏的是 `sgl-kernel-npu` 工作区中的源文件，不是本次 staging，也不是 CCU 编译错误。先比较当前
工作树内容和当前提交中的 Git blob：

```bash
cd /home/l00934901/sgl-kernel-npu

REL=examples/a5_ccu_urma_route_probe/CMakeLists.txt

echo "expected blob: $(git rev-parse HEAD:${REL})"
echo "working blob : $(git hash-object "${REL}")"
git status --short -- "${REL}"

# 该输出应该是以 # 开头的 CMake 文本。
git show "HEAD:${REL}" | head -n 5
```

只有确认 Git blob 是正常文本、工作文件确实损坏后，才恢复这个文件：

```bash
git restore --source=HEAD --worktree -- "${REL}"

file "${REL}"
head -n 5 "${REL}"
test "$(LC_ALL=C head -c 1 "${REL}")" = "#"
```

继续校验整个 route-probe 目录。命令无输出表示所有已跟踪文件均与当前提交一致：

```bash
while IFS= read -r f; do
  expected=$(git rev-parse "HEAD:${f}")
  actual=$(git hash-object "${f}")
  if [[ "${expected}" != "${actual}" ]]; then
    echo "MISMATCH ${f}"
  fi
done < <(git ls-files examples/a5_ccu_urma_route_probe)
```

如果有多个 `MISMATCH`，并且确认该目录没有需要保留的本地修改，可以只恢复这个 probe 目录：

```bash
git restore --source=HEAD --worktree -- \
  examples/a5_ccu_urma_route_probe
```

不要逐个文件恢复后立即重试构建。例如顶层 `CMakeLists.txt` 修好后又在
`op_kernel_ccu/CMakeLists.txt:1` 报乱码，就已经证明目录中存在多个损坏文件，应执行上面的整目录恢复。
当前构建脚本会在调用 CMake 前扫描该目录的全部源文件，并一次列出所有非文本文件。

另外，root shell 可能配置了 `alias cp='cp -i'`；此时 `cp -f` 仍可能进入交互询问，并没有真正覆盖目标。
首选 `git restore`。必须从已验证的 Git blob 手工恢复时，应明确调用 `/bin/cp`：

```bash
git show "HEAD:${REL}" > /tmp/CMakeLists.txt.good
file /tmp/CMakeLists.txt.good
/bin/cp -f /tmp/CMakeLists.txt.good "${REL}"
```

恢复后重新执行本节开头的构建命令。如果 `git restore` 后 `file CMakeLists.txt` 仍显示 `data`，停止
构建并保留上述 blob、`git status` 和 `git check-attr -a -- "${REL}"` 输出；这表明文件系统或检出过滤
策略仍在改写工作文件，不能再通过普通复制或 hard-link 绕过。

确认 prepared-plan ABI：

```bash
ROUTE_SO=/usr/local/Ascend/cann-9.1.T560/opp/vendors/cust/lib64/liba5_ccu_urma_route_probe.so

nm -D "${ROUTE_SO}" | grep -E \
'HcclCcuUrmaExplicitMultipathPlan(Create|Execute)|A5CcuUrmaPreparedPlanAbiVersion'

python3 - "${ROUTE_SO}" <<'PY'
import ctypes
import pathlib
import sys

path = pathlib.Path(sys.argv[1]).resolve()
lib = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
version = lib.A5CcuUrmaPreparedPlanAbiVersion
version.restype = ctypes.c_int
print("route library:", path)
print("prepared-plan ABI:", version())
assert version() >= 3
PY
```

这里不安装新的 `ubus.ko`，也不改全局 UB route table。

## 4. 编译并安装 DeepEP wheel

在当前 Python/torch_npu 环境中执行：

```bash
cd /home/l00934901/sgl-kernel-npu
source /usr/local/Ascend/cann-9.1.T560/set_env.sh

DEEPEP_SINGLE_OP=explicit_multipath_all2all_ccu \
bash build.sh -a deepep Ascend950

WHEEL=$(ls -t output/deep_ep-*.whl | head -1)
python3 -m pip install --force-reinstall --no-cache-dir --no-deps "${WHEEL}"
```

安装不会替换 shell，也不应退出 Docker。不要把 build/install 命令包在会退出父 shell 的脚本中。

### 4.1 检查 wheel 和图算子

```bash
python3 - <<'PY'
import ctypes
from pathlib import Path
import torch
import deep_ep.deep_ep_cpp as ext

path = Path(ext.__file__).resolve()
lib = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
version = lib.A5DeepEpExplicitMultipathAttrAbiVersion
version.restype = ctypes.c_int

print("loaded deep_ep_cpp:", path)
print("explicit multipath attr ABI:", version())
print("prepared graph op:", torch.ops.deep_ep.ccu_urma_prepared_multipath_alltoall)

assert version() >= 5
assert hasattr(ext.Buffer, "prepare_ccu_urma_explicit_multipath_plan")
assert hasattr(ext.Buffer, "ccu_urma_prepared_multipath_alltoall_out")
assert hasattr(torch.ops.deep_ep, "ccu_urma_prepared_multipath_alltoall")

send = torch.empty((2, 8), device="meta", dtype=torch.float32)
recv = torch.empty_like(send)
out = torch.ops.deep_ep.ccu_urma_prepared_multipath_alltoall(send, recv, 1)
assert out is recv
print("Meta dispatch: PASS")

def graph_fn(send, recv):
    return torch.ops.deep_ep.ccu_urma_prepared_multipath_alltoall(send, recv, 1)

compiled = torch.compile(graph_fn, backend="eager", fullgraph=True)
assert compiled(send, recv) is recv
print("torch.compile fullgraph Meta capture: PASS")
PY
```

预期 DeepEP ABI 至少为 `5`、prepared-plan ABI 至少为 `3`，并输出 `Meta dispatch: PASS` 和
`torch.compile fullgraph Meta capture: PASS`。

## 5. 快速单 case 验证

单 case 只要求提供实际使用的 relay，不要求 relay 数量为偶数。例如当前用物理卡 `2,3` 作为通信
端点，显式指定物理卡 `0,1` 为两条 relay：

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
  --cann-root /usr/local/Ascend/cann-9.1.T560 \
  --output-root /home/l00934901/profiling
```

`direct_plus_relays` 会使用 `--relay-phys` 中的全部卡。可传 1、2、3、5、6 或更多 relay，例如：

```text
--relay-phys 0                 # direct + 1 relay
--relay-phys 0,1,4             # direct + 3 relay
--relay-phys 0,1,4,5,6         # direct + 5 relay
```

规则如下：

- `direct_plus_relays` 是通用单 case，使用任意 `1..63` 张 relay；
- `direct_plus_2relay`、`direct_plus_4relay`、`direct_plus_6relay` 只是便于横向比较的性能预置，不是
  算子语义约束；
- 一个 prepared plan 当前最多 64 条总路径，即 1 条 direct 加最多 63 条 relay，覆盖 64-device
  集群；更大规模需要把一个 plan 分批为多个 CCU launch，不能静默截断；
- relay 顺序就是 manifest/path 顺序，不能包含 `src-phy` 或 `dst-phy`，卡号不会在脚本中写死；
- relay 卡不启动 rank，但其 IO Die/端口会承载转发流量，不能把他人正在使用的卡视为完全无影响。

不要再传 `--hccl-lib-dir`，也不要设置 patched HCCL 的 `LD_PRELOAD`。

日志应依次出现：

```text
CASE_PREPARED_PLAN_ABI ... version=1
PREPARED_MULTIPATH_PLAN ...
PASS: ... implementation=prepared
```

`PREPARED_MULTIPATH_PLAN` 每个 rank 只应在 warmup 前出现一次。若反复出现，说明控制面准备错误地进入了
热循环。

### 5.1 验证是否真正入图

先用 `eager` backend 验证 Dynamo 能以 `fullgraph=True` 捕获完整节点：

```bash
bash scripts/run_a5_ccu_explicit_multipath_alltoall_perf.sh \
  --src-phy 2 --dst-phy 3 \
  --relay-phys 0,1 \
  --direct-route 0 \
  --cases direct_plus_relays \
  --graph-backend eager \
  --bytes 4194304 \
  --warmup 1 --iters 3 --repeats 1 \
  --timeout-seconds 600 \
  --cann-root /usr/local/Ascend/cann-9.1.T560 \
  --output-root /home/l00934901/profiling
```

再用 `npugraphs` 验证 ACLGraph 首次 capture 和重复 replay：

```bash
bash scripts/run_a5_ccu_explicit_multipath_alltoall_perf.sh \
  --src-phy 2 --dst-phy 3 \
  --relay-phys 0,1 \
  --direct-route 0 \
  --cases direct_plus_relays \
  --graph-backend npugraphs \
  --bytes 4194304 \
  --warmup 1 --iters 3 --repeats 1 \
  --timeout-seconds 600 \
  --cann-root /usr/local/Ascend/cann-9.1.T560 \
  --output-root /home/l00934901/profiling
```

PASS 必须同时满足：

```text
CASE_GRAPH_CAPTURE ... fullgraph=1
CASE_GRAPH_FIRST_EXECUTE ... PASS
CASE_GRAPH_REPLAY ... PASS
PASS: implementation=prepared ...
```

并且每个 rank 只有一条 `PREPARED_MULTIPATH_PLAN`。`eager` 只证明 FX/Dynamo 完整捕获；只有
`npugraphs` 的 first execute 与 replay 都正确，才能说明这条路径能进入当前环境的 NPU graph。

## 6. 完整性能矩阵

推荐稳定参数为 warmup 100、计时 20 次、重复 3 轮：

```bash
bash scripts/run_a5_ccu_explicit_multipath_alltoall_perf.sh \
  --src-phy 2 --dst-phy 3 \
  --relay-phys 4,5,1,0,6,7 \
  --direct-route 0 \
  --cases native0,native2,direct_plus_2relay,direct_plus_4relay,direct_plus_6relay \
  --bytes 4194304 \
  --warmup 100 --iters 20 --repeats 3 \
  --timeout-seconds 300 \
  --cann-root /usr/local/Ascend/cann-9.1.T560 \
  --output-root /home/l00934901/profiling
```

显式 case 的总流量分配保持：

```text
direct 总字节 : 所有 relay 总字节 = 2 : 1
```

因此脚本使用：

- direct + 2 relay：`4,1,1`；
- direct + 4 relay：`8,1,1,1,1`；
- direct + 6 relay：`12,1,1,1,1,1,1`。

查看结果：

```bash
RUN_DIR=$(ls -dt \
  /home/l00934901/profiling/a5_ccu_explicit_multipath_2_3_* | head -1)

column -s $'\t' -t "${RUN_DIR}/case_status.tsv"
sed -n '1,240p' "${RUN_DIR}/explicit_multipath_perf_report.md"
grep -RHnE 'PREPARED_MULTIPATH_PLAN|PASS:|CASE_FAILURE' "${RUN_DIR}/cases"
```

## 7. 只采显式多路径 profiling

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
  --cann-root /usr/local/Ascend/cann-9.1.T560 \
  --output-root /home/l00934901/profiling
```

结果位于运行目录的 `profiling/` 下。MindStudio 中看到一个 prepared CCU launch 是正常的；一条 launch 内
包含多个 Channel 的 `WriteNb`，不能仅凭泳道数量判断物理路径。路径归属仍应配合 HCCN 端口计数验证。

## 8. 常见故障

### 8.1 正确性失败时检查 rank 与路径计划

如果 case 已完成 plan/Channel 准备，但在 `warmup_correctness` 报 `Tensor-likes are not close`，先分析
现有日志，不要立即重跑。例如：

```bash
RUN_DIR=/home/l00934901/profiling/a5_ccu_explicit_multipath_2_3_20260928_104753
LOG="${RUN_DIR}/cases/direct_plus_relays_r1.log"

grep -nE \
'PREPARED_MULTIPATH_PLAN|explicit multipath direct|explicit relay|PATH_CHANNEL|launch kernel' \
"${LOG}"
```

应分别看到 `PREPARED_MULTIPATH_PLAN ... rank=0 ...` 和
`PREPARED_MULTIPATH_PLAN ... rank=1 ...`：

- 如果两个进程都显示同一个 rank，优先检查 DeepEP communicator 到 route runtime 的逻辑 rank 传递；
- 如果 rank 分别为 0 和 1，再检查 `AllToAllMultiRouteTaskArg -> GeneArgs -> CCU Load` 的
  `selfDestinationOffset`、`sourceOffsets` 和 `remoteOffsets`；
- `explicit multipath direct`、`explicit relay` 和 `PATH_CHANNEL` 用于确认计划中的 direct/relay 数量、
  顺序、权重及 Channel 是否与请求一致。

将 `RUN_DIR` 替换为实际失败目录。上述命令只读已有日志，不会重新建链或再次运行算子。

| 现象 | 原因/处理 |
|---|---|
| DeepEP ABI `< 5` | 加载了仍限制为 8 条总路径的旧 wheel；重新构建并 `--force-reinstall` |
| prepared-plan ABI 缺失 | route package 是旧版本；重建并安装第 3 节 |
| 只跑 prepared case，却提示缺少 `HcclCcuUrmaMultiRouteAllToAll` | 旧性能脚本误查了 legacy API；更新到最新分支。prepared case 只校验 PlanCreate/PlanExecute/ABI marker |
| `nm -D` 手工能看到符号，脚本却报告 missing | 旧脚本在 `set -o pipefail` 下使用 `grep -q`，`grep` 提前退出令 `nm` 收到 SIGPIPE，形成假阴性；更新脚本 |
| 安装后 SO 仍是旧符号表 | 第 3 节的新构建脚本会比较 packaged/installed SO；安装器未覆盖时会用已验证产物刷新并再次校验 |
| `--relay-phys must ... six` | 旧脚本把性能矩阵约束错误施加给算子；更新后用 `direct_plus_relays` 传任意 1..63 张 relay |
| 找不到 `torch.ops.deep_ep...` | wheel 未包含新的 `TORCH_LIBRARY` 注册 |
| `npugraphs` backend 不存在 | 当前 torch_npu 版本未提供 ACLGraph backend；先用 `python3 -c 'import torch,torch_npu; print(torch._dynamo.list_backends())'` 检查 |
| fullgraph graph break | 保存完整 `CASE_FAILURE`；不能把 eager PASS 当作设备图 PASS |
| `path_weights must contain...` | wheel/测试脚本版本不一致；先做 4.1 |
| `HcclAllocComResourceByTiling ret=5` | 仍在执行旧 `standard`/MC2 路径；正式 case 必须显示 `implementation=prepared` |
| `plan handle not found` | plan 在另一进程创建，或进程已重启；每个 rank 必须各自 prepare |
| `plan stream mismatch` | prepare 与 execute 不在同一 NPU stream；按 stream 分别准备 plan |
| ChannelAcquire status=9 | 所选 EID pair 未被当前底层 provision；检查 relay manifest/拓扑 |
| 正确性固定缺失一个 rank 的整块数据 | 确认 route ABI 至少为 3；旧 AllToAll kernel 将 output/token 合并在同一个 notify index，必须重建第 3 节 route package，使用独立 output/token/completion 通知 |

方案 2 不需要 `libmc2_client.so` 源码，也不需要 patched `libhccl.so`。它仍依赖系统 HCOMM/HCCL 的公开
Channel/CCU primitive 来准备资源。
