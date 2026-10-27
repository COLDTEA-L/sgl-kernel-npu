# AllToAllV 分支：从拉取代码到选路和正确性短测

分支：`feature/a5-ccu-8rank-direct-alltoallv`。
适用当前实验容器 `sglang_yuanwen_old`，仓库 `/home/l00934901/sgl-kernel-npu`，
CANN `/usr/local/Ascend/cann-9.1.T560`。全部使用容器现有 `python3`。

**尚未拉取本分支时，从第 1 步开始。** 不需要重新导入镜像或重建已有容器。
若目前在宿主机，先执行 `docker exec -it sglang_yuanwen_old bash`。
以下编号步骤均在实验容器执行；路径与实际不符时先替换。
每个多行执行块使用子 shell，失败不会退出外层容器交互会话。

本分支提供：2/4 卡等长多路径 AllToAll，以及 8 卡 direct-only AllToAllV。
8 卡动态 relay 调度尚未实现。此前开发容器编译成功不等于新机运行验证成功。

## 1. 拉取并切换分支

已有 `/home/l00934901/sgl-kernel-npu` 时执行：

```bash
(
set -Ee -o pipefail
trap 'echo "第1步失败，行号 ${LINENO}；保留当前文件，按错误处理后重试。" >&2' ERR
cd /home/l00934901/sgl-kernel-npu
echo '第1步开始：检查工作区并更新分支'
git status --short
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo '有已跟踪文件的本地修改，请先备份；本步骤不会覆盖它们。' >&2
    exit 1
fi
A5_BRANCH=feature/a5-ccu-8rank-direct-alltoallv
git fetch origin "refs/heads/${A5_BRANCH}:refs/remotes/origin/${A5_BRANCH}"
if git show-ref --verify --quiet "refs/heads/${A5_BRANCH}"; then
    git switch "${A5_BRANCH}"
else
    git switch -c "${A5_BRANCH}" "refs/remotes/origin/${A5_BRANCH}"
fi
git merge --ff-only "origin/${A5_BRANCH}"
git log -3 --oneline
git status --short
echo '第1步完成'
)
```

若旧仓是 single-branch clone，`git fetch origin 分支名` 可能只写 FETCH_HEAD，
不会创建 `origin/分支名`，随后报 `invalid reference`。上面的显式 refspec 会直接
创建对应 remote-tracking ref，不依赖旧仓的 fetch 范围；新建本地分支也不要求旧配置
能自动识别 upstream。后续更新仍用本步骤的 fetch 和 `merge --ff-only`。

有本地修改时先检查，再可用 `git stash push -m backup-before-alltoallv-branch`
保存 tracked 修改；此命令不收走 untracked 文件。切分支若提示未跟踪文件冲突，
先另存冲突文件。不要用 reset/clean，也不要直接恢复其他分支的旧 stash。

只有仓库目录尚不存在时，改用下面这一块，然后继续第 2 步：

```bash
(
set -Ee -o pipefail
trap 'echo "首次拉取失败，行号 ${LINENO}" >&2' ERR
cd /home/l00934901
git clone --branch feature/a5-ccu-8rank-direct-alltoallv \
  https://github.com/COLDTEA-L/sgl-kernel-npu.git sgl-kernel-npu
echo '首次拉取完成'
)
```

## 2. 核对容器环境和 CANN 别名

```bash
(
set -Ee -o pipefail
trap 'echo "第2步失败，行号 ${LINENO}" >&2' ERR
echo '第2步开始：环境核对'
source /usr/local/Ascend/cann-9.1.T560/set_env.sh
python3 --version
python3 -m pip --version
python3 - <<'PY'
import sys
import torch, torch_npu
print('python:', sys.executable)
print('torch:', torch.__version__)
print('torch_npu:', torch_npu.__version__)
PY
# route builder 会优先读取这些别名；避免覆盖为另一套 CANN。
python3 - <<'PY'
from pathlib import Path
expected = Path('/usr/local/Ascend/cann-9.1.T560').resolve()
for name in ('cann', 'ascend-toolkit'):
    alias = Path('/usr/local/Ascend') / name
    if (alias / 'set_env.sh').is_file():
        print('builder CANN candidate:', alias, '->', alias.resolve())
        if alias.resolve() != expected:
            raise SystemExit('CANN 别名不是 T560；先确认环境，不要混用两套 SDK')
        break
PY
command -v g++
command -v cmake
command -v make
python3 -m pip show pybind11 setuptools wheel
echo '第2步完成'
)
```

不需要根据 `torch` 版本中的 `+cpu` 字样单独判定失败，也不需要为了 profiler
去安装名为 msprof 的 Python 包。已有匹配的软件不重复升级。

## 3. 不使用 NPU 的 CPU 回归

```bash
(
set -Ee -o pipefail
trap 'echo "第3步失败，行号 ${LINENO}" >&2' ERR
cd /home/l00934901/sgl-kernel-npu
echo '第3步开始：CPU 回归'
mkdir -p build/ep8_validation
python3 -m unittest discover -s tests/python/deepep -p 'test_a5_ccu_*cpu.py' -v \
  2>&1 | tee build/ep8_validation/new_machine_cpu.log
echo '第3步完成；应看到 unittest 最终 OK'
)
```

包含 rootinfo/跨端点 plane 兼容修复的提交预期是 90 项通过。部分负向分析测试会输出 FAIL_OR_INCOMPLETE，
以 unittest 最终汇总为准；CPU 模型不代表 NPU 正确性。

## 4. 新机器 rootinfo/Atlas/实时 EID 选路预检

这一步只读文件并查询 `hccn_tool`，不创建 HCCL 通信域、不运行通信算子。
不用把长 JSON 粘贴到聊天或复制到开发服务器，也不用手存 EID TXT。

先分别在宿主机和容器运行以下只读命令，比较实际文件与 SHA256：

```bash
find /etc -maxdepth 1 -iname 'hccl_rootinfo.json' -exec sha256sum {} \;
find /usr/local/Ascend/driver/topo/950 -maxdepth 1 \
  \( -name 'atlas_950.json' -o -name 'atlas_950_1.json' \) -exec sha256sum {} \;
```

确认容器看到的是本机配置。仅“文件存在”不足以证明一致；镜像可能遗留旧机文件。
若不一致，先纠正容器的配置来源，不能拿旧文件覆盖新宿主机。

当前已定位到该实验容器漏挂 rootinfo、仍读旧机 EID。
先按 [容器 rootinfo 修复指南第 2 节](A5_CCU_ROOTINFO_CONTAINER_FIX.md#2-更新代码后在宿主机运行修复脚本)
在宿主机执行检查/修复，再继续本步骤。脚本先校验宿主副本，成功才备份并更新容器文件；
运行中的容器无需立即重建。持久只读挂载参数见该指南第 4 节。

当前用户提供的 Atlas 中，计算卡 2、3 经 relay 4 的两段都在 relay IO die 1。
下面据此使用 die 1；换机器或卡号时按实际选择替换：

```bash
(
set -Ee -o pipefail
trap 'echo "第4步失败，行号 ${LINENO}；先处理选路错误，再继续。" >&2' ERR
cd /home/l00934901/sgl-kernel-npu
echo '第4步开始：实时选路预检'
A5_CHECK_DIR=$(mktemp -d /tmp/a5-route-check.XXXXXX)
echo "选路输出目录：${A5_CHECK_DIR}"
python3 scripts/resolve_a5_explicit_multirelay_eids.py \
  --src-phy 2 --dst-phy 3 --relay-phys 4 --relay-planes 1 \
  --diagnostics-json "${A5_CHECK_DIR}/machine_topology.json" \
  > "${A5_CHECK_DIR}/relays.tsv"
cat "${A5_CHECK_DIR}/relays.tsv"
echo '第4步完成：应有表头及一行 relay 路径'
)
```

`rank_id` 和 `local_id` 不作为选路必填项；若看到裸 `error: 'rank_id'`，说明仍是修复前脚本，先更新本分支。

自动发现两个常见大小写的 rootinfo 路径和两种 Atlas 文件名；冲突或缺失时，
在上面的 Python 命令中显式增加 `--root-info 实际路径 --topology-json 实际路径`。
`--hccn-tool` 默认为 `/usr/local/Ascend/driver/tools/hccn_tool`。
网络 plane 用 `--plane-id` 选择；它不等于 `--relay-planes` 的 IO die。
其他字段及报错解释见 [机器选路说明](A5_CCU_MACHINE_ROUTE_DISCOVERY.md)。

若出现 EI0014/get_hccl_comm_name/owned_comm 错误，那是后续通信初始化问题，
不能由本次选路预检通过推断已经解决。

## 5. 在实验环境构建并安装 route 与 DeepEP

首次切换到含 EP8 的分支，需要构建两份组件。不是安装完整实验版 libhccl。
本步骤会安装自定义 route 包和当前 Python 的 DeepEP wheel；确保没有运行中的测试使用它们。

HCCL 构建框架应位于 `/home/l00934901/hccl`，分支
`feature/a5-ccu-explicit-multipath-alltoall`。已有正确源码和 third_party 缓存则直接继续。
若该目录完全不存在，可先执行：

```bash
git clone --branch feature/a5-ccu-explicit-multipath-alltoall \
  https://gitcode.com/yuanwenliu/hccl.git /home/l00934901/hccl
```

route 构建还需要 cann-cmake 和 makeself 的配套依赖/缓存；离线环境仅 clone
两个源码仓不一定足够。两源码目录应位于同一次父目录挂载下，避免 cp -al 跨文件系统失败。

### 5.1 route 自定义包

```bash
(
set -Ee -o pipefail
trap 'echo "第5.1步失败，行号 ${LINENO}" >&2' ERR
source /usr/local/Ascend/cann-9.1.T560/set_env.sh
cd /home/l00934901/sgl-kernel-npu
echo '第5.1步开始：route 构建和安装'
mkdir -p build/ep8_validation
if [[ ! -d /home/l00934901/hccl ]]; then
    echo '缺少 /home/l00934901/hccl 构建框架，请先准备源码。' >&2
    exit 1
fi
HCCL_REPO=/home/l00934901/hccl \
  bash scripts/build_a5_ccu_urma_route_probe.sh \
  --install --install-path /usr/local/Ascend/cann-9.1.T560 \
  2>&1 | tee build/ep8_validation/new_machine_route_build.log
echo '第5.1步完成'
)
```

### 5.2 DeepEP wheel

```bash
(
set -Ee -o pipefail
trap 'echo "第5.2步失败，行号 ${LINENO}" >&2' ERR
source /usr/local/Ascend/cann-9.1.T560/set_env.sh
cd /home/l00934901/sgl-kernel-npu
echo '第5.2步开始：DeepEP 构建和安装'
mkdir -p build/ep8_validation
A5_WHEEL_MARKER=$(mktemp /tmp/a5-wheel-build.XXXXXX)
DEEPEP_SINGLE_OP=explicit_multipath_all2all_ccu bash build.sh -a deepep Ascend950 \
  2>&1 | tee build/ep8_validation/new_machine_deepep_build.log
python3 - "${A5_WHEEL_MARKER}" <<'PY'
from pathlib import Path
import subprocess, sys
start = Path(sys.argv[1]).stat().st_mtime
wheels = [p for p in Path('output').glob('deep_ep-*.whl') if p.stat().st_mtime >= start]
if len(wheels) != 1:
    raise SystemExit(f'需要唯一的本次新 wheel，找到 {wheels}；不要安装旧 dist/output 包')
wheel = wheels[0].resolve()
print('安装本次 wheel:', wheel, flush=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '--force-reinstall',
                '--no-cache-dir', '--no-deps', str(wheel)], check=True)
PY
rm -f "${A5_WHEEL_MARKER}"
echo '第5.2步完成'
)
```

## 6. 已安装库的 ABI 与 Meta 预检

```bash
(
set -Ee -o pipefail
trap 'echo "第6步失败，行号 ${LINENO}" >&2' ERR
cd /home/l00934901/sgl-kernel-npu
echo '第6步开始：已安装产物检查'
source scripts/a5_ccu_test_env.sh
a5_ccu_prepare_test_env /usr/local/Ascend/cann-9.1.T560 "$PWD"
export A5_CCU_ROUTE_PROBE_LIB=/usr/local/Ascend/cann-9.1.T560/opp/vendors/cust/lib64/liba5_ccu_urma_route_probe.so
python3 tests/python/deepep/test_a5_ccu_peer_plan_alltoall.py \
  --runtime-check --self-copy-mode ccu --channel-sync-mode baseline
python3 - <<'PY'
import ctypes, importlib.util, os, subprocess, sys
from pathlib import Path
route = ctypes.CDLL(os.environ['A5_CCU_ROUTE_PROBE_LIB'], mode=ctypes.RTLD_GLOBAL)
for symbol, minimum in [('A5CcuUrmaPreparedPlanAbiVersion', 16), ('A5CcuPeerPlanAbiVersion', 5),
                        ('A5CcuChannelSyncAbiVersion', 1), ('A5CcuAllToAllVAbiVersion', 1)]:
    marker = getattr(route, symbol)
    marker.restype = ctypes.c_int
    value = marker()
    print(symbol, value, flush=True)
    if value < minimum:
        raise SystemExit('route ABI 版本不符')
spec = importlib.util.find_spec('deep_ep')
if spec is None or spec.origin is None:
    raise SystemExit('当前 python3 找不到安装的 deep_ep')
extensions = list(Path(spec.origin).parent.glob('deep_ep_cpp*.so'))
if len(extensions) != 1:
    raise SystemExit(f'无法唯一定位已安装 DeepEP 扩展: {extensions}')
subprocess.run([sys.executable, 'tests/python/deepep/test_a5_ccu_alltoallv_meta.py',
                '--extension', str(extensions[0].resolve())], check=True)
PY
echo '第6步完成：应看到 ALLTOALLV_META_PASS valid=160 invalid=12 legacy=10'
)
```

本步骤不建立设备通信域。ABI/Meta 通过后，再选择下面的设备短测。

## 7. 两卡兼容性短测：先 direct-only，再 relay

以下开始使用 NPU；选择实际允许使用的计算卡和中转卡。先跑 direct-only，若 HCCL
初始化失败就停在这里，处理配置，不继续多路径或性能测试。

```bash
(
set -Ee -o pipefail
trap 'echo "第7步 direct-only 失败，行号 ${LINENO}" >&2' ERR
source /usr/local/Ascend/cann-9.1.T560/set_env.sh
cd /home/l00934901/sgl-kernel-npu
bash scripts/run_a5_ccu_peer_plan_alltoall.sh \
  --devices 2,3 --bytes 4096 --dtype float32 \
  --self-copy-mode ccu --channel-sync-mode baseline \
  --warmup 1 --iters 3 --repeats 1 --timeout-seconds 180 \
  --output-root /home/l00934901/profiling/a5-alltoallv-branch
)
```

direct-only 正常退出且结果通过后，用和第 4 步相同的卡与 IO die 测 relay：

```bash
(
set -Ee -o pipefail
trap 'echo "第7步 relay 失败，行号 ${LINENO}" >&2' ERR
source /usr/local/Ascend/cann-9.1.T560/set_env.sh
cd /home/l00934901/sgl-kernel-npu
A5_POLICY_DIR=$(mktemp -d /tmp/a5-two-card-policy.XXXXXX)
cat > "${A5_POLICY_DIR}/policy.json" <<'JSON'
{"2,3": [{"physical_device": 4, "plane": "1"}]}
JSON
bash scripts/run_a5_ccu_peer_plan_alltoall.sh \
  --devices 2,3 --relay-map "${A5_POLICY_DIR}/policy.json" \
  --bytes 4096 --dtype float32 --self-copy-mode ccu --channel-sync-mode baseline \
  --warmup 1 --iters 3 --repeats 1 --timeout-seconds 180 \
  --output-root /home/l00934901/profiling/a5-alltoallv-branch
)
```

若第 4 步使用了显式 rootinfo/Atlas 路径、net-instance 或 plane-id，这里也传相同选项。
`--direct-route 0` 是默认 HCCL 候选序号，不是通用物理路径编号；runtime 会验证 hop=1。
四卡多路径使用同一个 runner，把 `--devices` 改为四卡列表并传逐对 relay-map；
完整策略示例见 [机器选路说明](A5_CCU_MACHINE_ROUTE_DISCOVERY.md)。
2/4 卡 runner 仅用于等长 AllToAll，不能拿它启动 EP8 AllToAllV。

## 8. 八卡 AllToAllV 正确性短测

只有八卡都可用于测试时执行。此测试是 eager correctness gate，不是性能或入图测试。

```bash
(
set -Ee -o pipefail
trap 'echo "第8步失败，行号 ${LINENO}" >&2' ERR
cd /home/l00934901/sgl-kernel-npu
source scripts/a5_ccu_test_env.sh
a5_ccu_prepare_test_env /usr/local/Ascend/cann-9.1.T560 "$PWD"
export A5_CCU_ROUTE_PROBE_LIB=/usr/local/Ascend/cann-9.1.T560/opp/vendors/cust/lib64/liba5_ccu_urma_route_probe.so
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export A5_CCU_CHANNEL_SYNC_MODE=baseline
A5_EP8_RUN=$(mktemp -d /tmp/a5-ep8-correctness.XXXXXX)
echo "八卡结果目录：${A5_EP8_RUN}"
python3 scripts/prepare_a5_ccu_peer_plan.py \
  --devices "$ASCEND_RT_VISIBLE_DEVICES" --available-phys 0,1,2,3,4,5,6,7 \
  --direct-route 0 --output-dir "${A5_EP8_RUN}/plans"
if timeout --signal=TERM --kill-after=5 300 \
  python3 -m torch.distributed.run --standalone --nproc-per-node=8 \
  tests/python/deepep/test_a5_ccu_alltoallv_npu.py \
  --manifest "${A5_EP8_RUN}/plans/peer_plan.tsv" \
  --run-dir "${A5_EP8_RUN}/results" --dtype float32 \
  > "${A5_EP8_RUN}/run.log" 2>&1; then
    echo 0 > "${A5_EP8_RUN}/exit.status"
else
    A5_EP8_STATUS=$?
    echo "$A5_EP8_STATUS" > "${A5_EP8_RUN}/exit.status"
    tail -80 "${A5_EP8_RUN}/run.log"
    exit "$A5_EP8_STATUS"
fi
python3 - "${A5_EP8_RUN}" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
assert (root / 'exit.status').read_text().strip() == '0'
for rank in range(8):
    result = json.loads((root / 'results' / f'rank{rank}.json').read_text())
    assert result['rank'] == rank and result['ranks'] == 8
    assert result['correctness'] == 'PASS' and result['cases'] == 12
    assert result['native_reference'] is True
print('EP8_ALL_RANKS_AND_EXIT_PASS')
PY
echo '第8步完成'
)
```

更改物理卡顺序时，manifest 的 `--devices` 必须与 `ASCEND_RT_VISIBLE_DEVICES` 一致。
每次使用新结果目录；八个 rank 结果和成功退出缺一不可。
其他 dtype 可在 float32 gate 通过后分别测试；API 与布局细节见
[EP8 AllToAllV 说明](A5_CCU_8RANK_ALLTOALLV.md)。
