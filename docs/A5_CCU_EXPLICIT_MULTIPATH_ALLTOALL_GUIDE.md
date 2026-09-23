# A5 CCU+URMA 两卡显式多路径 AllToAll 指导

CommLink、synthetic relay `HcclChannelDesc`、`HcclChannelAcquire`、CCU kernel 注册及数据面调用链的逐步说明见：

```text
docs/A5_CCU_EXPLICIT_MULTIPATH_ALLTOALL_IMPLEMENTATION.md
```

## 1. 实现范围

本分支实现标准 Ascend 自定义算子 `ExplicitMultipathAll2AllCcu`。当前版本限定两个 rank，路径集合为一条
HCCL direct CommLink 加调用者显式指定的 1～7 条 relay EID pair；总路径数不超过 8。

数据面由 CCU `WriteNb` 通过 URMA Channel 搬运 GM→GM。AIV 只负责图内启动 `Hccl<CCU>.Alltoall()`，
不执行 payload 的 GM/UB DataCopy。relay 卡不运行用户进程，也不落其 HBM。

路径计划在 communicator/resource 初始化阶段转换成 `HcclChannelDesc[]` 并缓存。图执行和 replay 只启动缓存的
CCU kernel，因此不会在图内读取文件、解析拓扑或重新建链。当前用环境变量把 host 控制计划交给 HCCL；未来 host
控制器可替换这一配置入口，不需要修改 AIV/CCU 数据面。

## 2. 两个配套分支

DeepEP/自定义算子仓：

```bash
cd /home/l00934901/sgl-kernel-npu
git fetch origin
git switch -c feature/a5-ccu-explicit-multipath-alltoall \
  --track origin/feature/a5-ccu-explicit-multipath-alltoall
```

HCCL 仓也必须使用同名分支：

```bash
cd /home/l00934901/hccl
git fetch origin
git switch -c feature/a5-ccu-explicit-multipath-alltoall \
  --track origin/feature/a5-ccu-explicit-multipath-alltoall
```

已有本地分支时改用 `git switch <branch>` 和 `git pull --ff-only`。不需要安装或替换 `ubus.ko`。

## 3. 在 Docker 内编译 HCCL

先用交互方式进入容器；不要 `source build.sh`：

```bash
docker exec -it sglang_yuanwen_old bash
```

容器内执行：

```bash
cd /home/l00934901/hccl
source /usr/local/Ascend/cann-9.1.T560/set_env.sh
bash build.sh -j8

HCCL_RUNTIME=/home/l00934901/hccl/build/_CPack_Packages/makeself_staging/aarch64-linux/lib64
HCCL_SO=$(readlink -f "${HCCL_RUNTIME}/libhccl.so")
HCCL_COMPAT_SO=$(readlink -f "${HCCL_RUNTIME}/libhccl_compat.so")
test -f "${HCCL_SO}"
test -f "${HCCL_COMPAT_SO}"
nm -D "${HCCL_SO}" | \
  grep ' A5HcclExplicitMultipathExtensionVersion$'

# 使用 -S 禁止 sitecustomize/torch_npu 提前加载系统 CANN 的 libhccl，
# 并按绝对路径打开本次构建产物，不能写成 CDLL("libhccl.so")。
LD_LIBRARY_PATH="${HCCL_RUNTIME}:${LD_LIBRARY_PATH:-}" \
python3 -S - "${HCCL_SO}" <<'PY'
import ctypes, pathlib, sys
path = pathlib.Path(sys.argv[1]).resolve()
lib = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
version = lib.A5HcclExplicitMultipathExtensionVersion
version.restype = ctypes.c_int
assert version() >= 1
assert pathlib.Path(lib._name).resolve() == path
print("loaded HCCL:", path)
print("HCCL explicit multipath extension:", version())
PY
```

必须使用打包暂存目录中的完整 `lib64`，不能只把 `build/src/libhccl.so` 加入路径；后者缺少同一次构建产生的
`libhccl_compat.so`，单独加载会出现未解析符号。这里不覆盖系统 CANN，公共机器上也不需要 root 安装。

如果异常里的库路径仍是 `/usr/local/Ascend/cann-9.1.T560/lib64/libhccl.so`，说明验证程序加载了系统 HCCL，
不是编译失败。先以 `nm -D "${HCCL_SO}"` 的结果判断定制二进制是否正确，再使用上面的绝对路径命令验证。
如果 `nm` 本身找不到 marker，则检查 HCCL 分支并重新构建：

```bash
cd /home/l00934901/hccl
git branch --show-current
grep -Rsn 'A5HcclExplicitMultipathExtensionVersion' \
  src/ops/all_to_all_v/template/ccu
bash build.sh -j8
```

### 3.1 按四层定位“undefined symbol”

不要只看最后一个 Python 异常。按下面顺序检查，哪一层失败就停在哪一层处理。

第一层检查分支和源码：

```bash
cd /home/l00934901/hccl

git branch --show-current
git log -3 --oneline

grep -Rsn 'A5HcclExplicitMultipathExtensionVersion' \
  src/ops/all_to_all_v/template/ccu
```

预期分支是 `feature/a5-ccu-explicit-multipath-alltoall`，且源码能搜到 marker。搜不到表示拉取的 HCCL 分支不对，
此时不要继续检查动态库。

第二层检查构建产物：

```bash
HCCL_RUNTIME=/home/l00934901/hccl/build/_CPack_Packages/makeself_staging/aarch64-linux/lib64
HCCL_SO=$(readlink -f "${HCCL_RUNTIME}/libhccl.so")
HCCL_COMPAT_SO=$(readlink -f "${HCCL_RUNTIME}/libhccl_compat.so")

ls -lh "${HCCL_SO}" "${HCCL_COMPAT_SO}"
nm -D "${HCCL_SO}" | \
  grep ' A5HcclExplicitMultipathExtensionVersion$'
```

源码存在但 `nm` 没有输出，表示 build/package 仍是旧产物，需要重新执行 `bash build.sh -j8`；这和系统
`/usr/local/Ascend/.../libhccl.so` 无关。

第三层按绝对路径加载定制 HCCL：

```bash
LD_LIBRARY_PATH="${HCCL_RUNTIME}:${LD_LIBRARY_PATH:-}" \
python3 -S - "${HCCL_SO}" <<'PY'
import ctypes
import pathlib
import sys

expected = pathlib.Path(sys.argv[1]).resolve()
lib = ctypes.CDLL(str(expected), mode=ctypes.RTLD_GLOBAL)
version = lib.A5HcclExplicitMultipathExtensionVersion
version.restype = ctypes.c_int
print("loaded:", pathlib.Path(lib._name).resolve())
print("extension:", version())
assert pathlib.Path(lib._name).resolve() == expected
assert version() >= 1
PY
```

第四层检查带 `torch_npu` 的真实进程最终映射到哪个 HCCL：

```bash
HCCL_PRELOAD="${HCCL_COMPAT_SO}:${HCCL_SO}"

env \
  A5_EXPECTED_HCCL="${HCCL_SO}" \
  LD_LIBRARY_PATH="${HCCL_RUNTIME}:${LD_LIBRARY_PATH:-}" \
  LD_PRELOAD="${HCCL_PRELOAD}${LD_PRELOAD:+:${LD_PRELOAD}}" \
  python3 - <<'PY'
import ctypes
import os
from pathlib import Path

import torch_npu

# 让本进程显式解析 HCCL SONAME，然后检查实际内存映射。
ctypes.CDLL("libhccl.so", mode=ctypes.RTLD_GLOBAL)
expected = Path(os.environ["A5_EXPECTED_HCCL"]).resolve()
loaded = set()
for line in Path("/proc/self/maps").read_text().splitlines():
    path = line.split()[-1]
    if path.endswith("/libhccl.so"):
        loaded.add(Path(path).resolve())
print("expected:", expected)
print("mapped HCCL:", sorted(str(path) for path in loaded))
assert expected in loaded, "patched HCCL is not mapped into the torch_npu process"

process = ctypes.CDLL(None)
version = process.A5HcclExplicitMultipathExtensionVersion
version.restype = ctypes.c_int
assert version() >= 1
print("torch_npu process extension:", version())
PY
```

四层结果的含义：

| 失败位置 | 含义 | 处理 |
|---|---|---|
| 源码搜不到 marker | HCCL 分支/提交不对 | 切换并拉取配套 HCCL 分支 |
| `nm` 搜不到 marker | 构建或 package 是旧产物 | 重新执行 HCCL build/package |
| 绝对路径加载失败 | 定制 `libhccl.so` 与配套依赖不一致 | 使用同一 package 的完整 `lib64` |
| `/proc/self/maps` 只有系统 HCCL | 进程启动时动态库选择错误 | 使用更新后的脚本限定 `LD_PRELOAD`/`LD_LIBRARY_PATH` |

不要在交互 shell 中长期 `export LD_PRELOAD`。性能脚本只对子 `torchrun` 进程注入定制 HCCL，避免影响编译器、
`git` 和其他系统命令。

## 4. 编译并安装标准算子 wheel

推荐单算子构建，避免重编所有 MoE kernel：

```bash
cd /home/l00934901/sgl-kernel-npu
source /usr/local/Ascend/cann-9.1.T560/set_env.sh

(
  set -euo pipefail
  rm -f output/deep_ep*.whl
  DEEPEP_SINGLE_OP=explicit_multipath_all2all_ccu \
    bash build.sh -a deepep Ascend950
  WHEEL=$(find output -maxdepth 1 -name 'deep_ep*.whl' -type f \
    -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
  test -n "${WHEEL}"
  python3 -m pip install --force-reinstall --no-cache-dir --no-deps "${WHEEL}"
)

echo "still in container: $(hostname), shell_pid=$$"
```

子 shell 中的 `set -e` 只终止构建组，不会退出 `docker exec -it` 的交互 shell。验证安装：

```bash
python3 - <<'PY'
from pathlib import Path
import deep_ep.deep_ep_cpp as ext
print(Path(ext.__file__).resolve())
assert hasattr(ext.Buffer, "explicit_multipath_all2all_ccu")
print("standard explicit multipath API: PASS")
PY

python3 -m py_compile \
  tests/python/deepep/test_a5_ccu_urma_multiroute_all2all.py
bash -n scripts/run_a5_ccu_explicit_multipath_alltoall_perf.sh
```

## 5. 路径与数据切分

manifest 每行描述一条显式 relay 的双向 endpoint，至少包含：

```text
route_id  relay_phy  src_die  dst_die  src_eid  dst_eid
```

Channel 0 是 HCCL direct，Channel 1..N 按 manifest 行序构造。等权 relay 且要求
`direct 总量 : relay 总量 = 2:1` 时：

| relay 数 | `path_weights` |
|---:|---|
| 2 | `4,1,1` |
| 4 | `8,1,1,1,1` |
| 6 | `12,1,1,1,1,1,1` |

每段按 256 Byte 对齐，最后一条路径接收余数。CCU kernel 先提交全部远端 `WriteNb`，再统一等待事件；因此是
因果上的并发提交，而不是 host 循环串行调用多个算子。

## 6. 完整性能矩阵

下面以通信卡 2、3 和六张显式 relay `4,5,1,0,6,7` 为例：

```bash
cd /home/l00934901/sgl-kernel-npu

HCCL_RUNTIME=/home/l00934901/hccl/build/_CPack_Packages/makeself_staging/aarch64-linux/lib64

bash scripts/run_a5_ccu_explicit_multipath_alltoall_perf.sh \
  --src-phy 2 --dst-phy 3 \
  --relay-phys 4,5,1,0,6,7 \
  --direct-route 0 \
  --bytes 4194304 \
  --warmup 100 --iters 20 --repeats 3 \
  --timeout-seconds 300 \
  --hccl-lib-dir "${HCCL_RUNTIME}" \
  --cann-root /usr/local/Ascend/cann-9.1.T560 \
  --output-root /home/l00934901/profiling
```

脚本输出五组：`native0`、`native2`、标准算子的 direct+2 relay、direct+4 relay、direct+6 relay。
`native0/native2` 是旧 probe 的 RankGraph candidate 基线，不是 `route_addr_idx`。三个显式用例均调用新
`Buffer.explicit_multipath_all2all_ccu()` 图算子；缺少定制 HCCL 标记时会立即失败，不会静默退化。

查看结果：

```bash
RUN_DIR=$(ls -dt \
  /home/l00934901/profiling/a5_ccu_explicit_multipath_2_3_* | head -1)
column -s $'\t' -t "${RUN_DIR}/case_status.tsv"
sed -n '1,220p' "${RUN_DIR}/explicit_multipath_perf_report.md"
cat "${RUN_DIR}/explicit_multipath_perf_summary.json"
```

## 7. MindStudio profiling

在第 6 节命令末尾增加：

```bash
  --profile --profile-iters 20
```

每个 case 的两个进程分别输出到：

```text
RUN_DIR/profiling/a5_ccu_alltoall_<case>_<run-id>/rank0/
RUN_DIR/profiling/a5_ccu_alltoall_<case>_<run-id>/rank1/
```

性能结论以无 profiling 的三轮中位数为主；MindStudio 用来确认标准图节点、CCU Launch 时间线和异常长尾。
严格证明指定物理 relay 仍应另用 HCCN before/after counter 做闭环。

## 8. 结果判定与限制

- 五个 case 均须 correctness PASS；
- explicit case 必须加载定制 `libhccl.so`；
- plan ID、weights 和 manifest 必须在 communicator/resource 初始化前一致；
- 当前一个 launch 内的所有 Channel 必须落在兼容的本地 IO Die/CCU thread 资源上；
- 当前仅支持两个 rank；多卡扩展应使用 `peerPlans[src][dst]`，不能复用 `peer=1-rank`；
- host 控制器以后可改变 relay 集合和权重，但计划变化需要新的 communicator/resource identity，不能在已捕获图的
  replay 中原地改路径。
