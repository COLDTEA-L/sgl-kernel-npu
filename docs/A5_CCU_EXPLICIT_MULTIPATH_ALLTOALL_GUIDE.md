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
git pull --ff-only origin feature/a5-ccu-explicit-multipath-alltoall
```

HCCL 仓也必须使用同名分支：

```bash
cd /home/l00934901/hccl
git fetch origin
git switch -c feature/a5-ccu-explicit-multipath-alltoall \
  --track origin/feature/a5-ccu-explicit-multipath-alltoall
git pull --ff-only origin feature/a5-ccu-explicit-multipath-alltoall
```

上面的 `switch -c` 仅用于本地尚无该分支的第一次拉取。已有本地分支时，必须分别在两个仓库显式更新，不能只更新
`sgl-kernel-npu`：

```bash
cd /home/l00934901/sgl-kernel-npu
git fetch origin
git switch feature/a5-ccu-explicit-multipath-alltoall
git pull --ff-only origin feature/a5-ccu-explicit-multipath-alltoall
git log -2 --oneline

cd /home/l00934901/hccl
git fetch origin
git switch feature/a5-ccu-explicit-multipath-alltoall
git pull --ff-only origin feature/a5-ccu-explicit-multipath-alltoall
git log -2 --oneline

# 必须包含 T560 缺少公开 CCU C/C++ 头文件时的完整兼容修复。
# 9644f7c 已包含其父提交 1c566f8。
git merge-base --is-ancestor 9644f7c HEAD && \
  echo "HCCL compatibility fix: PASS"
```

若最后没有输出 `PASS`，不要开始编译；先比较 `git remote -v`，确认 HCCL 的 `origin` 是
`https://gitcode.com/yuanwenliu/hccl.git`。不需要安装或替换 `ubus.ko`。

## 3. 在 Docker 内编译 HCCL

先用交互方式进入容器；不要 `source build.sh`，也不要在宿主机直接编译：

```bash
docker exec -it sglang_yuanwen_old bash
```

容器内执行：

```bash
cd /home/l00934901/hccl
source /usr/local/Ascend/cann-9.1.T560/set_env.sh

# 这里只负责构建。-j8 已在 cam_lyw_dev_91 中完整验证通过。
bash build.sh -j8

# 不硬编码 CPack 子目录；不同 CANN/HCCL 版本的 staging 层级可能不同。
# 选择最近生成且同时包含 libhccl.so/libhccl_compat.so 的目录。
HCCL_RUNTIME=$(
  find /home/l00934901/hccl \
    -name libhccl.so -printf '%T@ %h\n' 2>/dev/null | \
  sort -nr | \
  while read -r _ candidate; do
    if [[ -f "${candidate}/libhccl_compat.so" ]]; then
      echo "${candidate}"
      break
    fi
  done
)

if [[ -z "${HCCL_RUNTIME}" ]]; then
  echo "ERROR: no packaged directory contains matching libhccl.so and libhccl_compat.so" >&2
  find /home/l00934901/hccl \
    -type f \( -name libhccl.so -o -name libhccl_compat.so \) -print 2>/dev/null
  echo "Check the end of build.sh output; do not continue to Python verification." >&2
else
  echo "HCCL_RUNTIME=${HCCL_RUNTIME}"
  HCCL_SO=$(readlink -f "${HCCL_RUNTIME}/libhccl.so")
  HCCL_COMPAT_SO=$(readlink -f "${HCCL_RUNTIME}/libhccl_compat.so")
  test -f "${HCCL_SO}" -a -f "${HCCL_COMPAT_SO}"
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
fi
```

预期最后两行类似：

```text
HCCL_RUNTIME=<本次构建实际产生的配套 lib64 目录>
loaded HCCL: <HCCL_RUNTIME>/libhccl.so
HCCL explicit multipath extension: 1
```

必须使用打包暂存目录中的完整 `lib64`。不能只加载 `build/src/libhccl.so`，因为它还依赖同一次构建产生的
`libhccl_compat.so`；混用系统依赖会出现 `IsHcommDefaultTimeoutSupported` 等未解析符号。以上步骤不覆盖系统 CANN，
公共机器上也不需要 root 安装。

如果异常里的库路径仍是 `/usr/local/Ascend/cann-9.1.T560/lib64/libhccl.so`，说明验证程序加载了系统 HCCL，
不是 `-j8` 编译失败。禁止使用下面这种按 SONAME 搜索的校验：

```python
# 错误：set_env.sh 通常会让它命中系统 CANN 的 libhccl.so。
ctypes.CDLL("libhccl.so")
```

只以 `nm -D "${HCCL_SO}"` 和上面的绝对路径加载结果判断本次构建产物。

### 3.1 仅在校验失败时执行

正常构建无需执行额外的 `torch_npu`、`/proc/self/maps` 或系统 HCCL 校验。为避免终端一次粘贴过长发生错行，
下面三小节应分段执行；前一小节失败便停止。

#### 3.1.1 确认分支和源码

```bash
cd /home/l00934901/hccl

git branch --show-current
git log -3 --oneline
grep -Rsn 'A5HcclExplicitMultipathExtensionVersion' \
  src/ops/all_to_all_v/template/ccu
```

源码不存在 marker 时不要构建，先切换并拉取配套 HCCL 分支。

#### 3.1.2 无打包产物时重新构建并保存日志

当 `HCCL_RUNTIME` 为空，或整个仓库找不到配套的 `libhccl.so`、`libhccl_compat.so` 时，单独执行：

如果此前日志包含下面任一错误：

```text
fatal error: ccu_types.h: No such file or directory
fatal error: ccu_primitives.hpp: No such file or directory
fatal error: ccu_variable.hpp: No such file or directory
fatal error: ccu_control_flow_macro.h: No such file or directory
```

说明 T560 报告了 9.1 版本号、但安装包没有完整交付新版公开 CCU C/C++ 开发头。`1c566f8` 只处理了
`ccu_types.h` 及部分 C 接口；`9644f7c` 进一步覆盖 `ccu_primitives.hpp` 和同组 CCU C++ 包装头。
配套 HCCL 分支现在按头文件实际存在性整体选择公共实现或仓内兼容实现。先显式更新 HCCL：

```bash
cd /home/l00934901/hccl
git fetch origin
git switch feature/a5-ccu-explicit-multipath-alltoall
git pull --ff-only origin feature/a5-ccu-explicit-multipath-alltoall
git log -2 --oneline

git merge-base --is-ancestor 9644f7c HEAD && \
  echo "HCCL CCU header compatibility: PASS"
```

确认最后输出 `PASS` 后再构建。若没有输出，当前 HCCL 仍是旧提交；不要通过复制某一个缺失头文件绕过错误：

```bash
cd /home/l00934901/hccl
source /usr/local/Ascend/cann-9.1.T560/set_env.sh
unset LD_PRELOAD

bash build.sh -j8 > /tmp/hccl_build_j8.log 2>&1
BUILD_RC=$?

echo "BUILD_RC=${BUILD_RC}"
tail -n 100 /tmp/hccl_build_j8.log
```

只有 `BUILD_RC=0` 才继续。非零时不要执行 `nm` 或 Python，先提取真正的编译/打包错误：

```bash
grep -nEi \
  'error:|fatal:|undefined reference|failed|no such file' \
  /tmp/hccl_build_j8.log | \
tail -n 100
```

`BUILD_RC=0` 后查看实际生成位置：

```bash
find /home/l00934901/hccl \
  \( -name libhccl.so -o -name libhccl_compat.so \) \
  -type f -print
```

#### 3.1.3 构建长时间停在某个 `100% Built target` 时定位

类似下面的输出只表示一个并行子目标完成，不表示整个 `build.sh` 已经完成：

```text
[100%] Built target hccl_aiv_reduce_scatter_op_910_95
```

真正完成时还应看到主目标链接、CPack 打包成功，并返回 shell 提示符或 `BUILD_RC=0`。如果该行之后长时间无输出，
先不要中断；在第二个终端进入同一个容器，查明 `build.sh` 正在等待哪个子进程：

```bash
BUILD_PID=$(pgrep -o -f '[b]ash build.sh')
echo "BUILD_PID=${BUILD_PID}"

if [[ -n "${BUILD_PID}" ]]; then
  pstree -ap "${BUILD_PID}"
fi

ps -eo pid,ppid,etime,stat,%cpu,%mem,wchan:32,cmd | \
  grep -E 'build\.sh|cmake --build|gmake|make|ccec|c\+\+|collect2|(^|/)ld |cpack|makeself|gen_esb' | \
  grep -v grep
```

判读原则：

- `ccec`、`c++`、`ld`、`cpack` 或 `makeself` 仍有持续 CPU/I/O：任务仍在工作；
- `STAT=D`：进程阻塞在文件系统或内核 I/O；
- 只有 `cmake/gmake` 停在 `futex_wait`，却没有编译、链接或打包子进程：可能是并行 jobserver 等待；
- 某个 `ccec` 已运行很久且 CPU、I/O 均不再变化：可能卡在该 device kernel 编译。

用下面的命令保存完整现场。该命令只读取 `/proc` 和构建目录，不改变构建状态：

```bash
DIAG=/tmp/hccl_hang_diag_$(date +%Y%m%d_%H%M%S).txt

{
  date
  echo "===== process tree ====="
  BUILD_PID=$(pgrep -o -f '[b]ash build.sh')
  echo "BUILD_PID=${BUILD_PID}"
  [[ -n "${BUILD_PID}" ]] && pstree -ap "${BUILD_PID}"

  echo "===== relevant processes ====="
  ps -eo pid,ppid,etime,stat,%cpu,%mem,wchan:40,cmd | \
    grep -E 'build\.sh|cmake|gmake|make|ccec|c\+\+|collect2|ld|cpack|makeself|gen_esb' | \
    grep -v grep

  echo "===== process details ====="
  for pid in $(pgrep -f 'build\.sh|cmake|gmake|make|ccec|collect2|cpack|makeself|gen_esb'); do
    [[ -r "/proc/${pid}/status" ]] || continue
    echo "--- PID ${pid} ---"
    grep -E '^(Name|State|Pid|PPid|Threads|VmRSS):' "/proc/${pid}/status"
    printf 'wchan='
    cat "/proc/${pid}/wchan" 2>/dev/null || true
    printf '\ncmd='
    tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null || true
    printf '\ncwd='
    readlink "/proc/${pid}/cwd" 2>/dev/null || true
    printf '\n'
  done

  echo "===== recently modified files ====="
  find /home/l00934901/hccl/build \
    -type f -mmin -10 -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %p\n' 2>/dev/null | \
    sort | tail -80
} | tee "${DIAG}"

echo "DIAG=${DIAG}"
```

确认相关编译、链接、打包子进程均无进展后，先保存上述现场，再在原构建终端按一次 `Ctrl+C`。此时不要重新运行
`build.sh`，因为它会清理构建目录；直接复用现有 `build/`，用单线程 verbose 模式继续，最后一条完整命令就是实际卡点：

```bash
cd /home/l00934901/hccl
source /usr/local/Ascend/cann-9.1.T560/set_env.sh
unset LD_PRELOAD

cmake --build build --parallel 1 --verbose \
  2>&1 | tee /tmp/hccl_build_j1_verbose.log

BUILD_RC=${PIPESTATUS[0]}
echo "BUILD_RC=${BUILD_RC}"
```

如果编译阶段成功，再单独串行定位打包阶段：

```bash
cmake --build build --target package --parallel 1 --verbose \
  2>&1 | tee /tmp/hccl_package_j1_verbose.log

PACKAGE_RC=${PIPESTATUS[0]}
echo "PACKAGE_RC=${PACKAGE_RC}"
```

若串行模式仍卡住，先从进程列表找到最后一个活动的叶子进程；系统存在 `strace` 时，可进行 15 秒只读观察：

```bash
ps -eo pid,ppid,etime,stat,%cpu,wchan:32,cmd | \
  grep -E 'ccec|c\+\+|collect2|(^|/)ld |cpack|makeself|gen_esb' | \
  grep -v grep

LEAF_PID=<上面找到的PID>
timeout 15 strace -ff -tt -T -s 256 \
  -p "${LEAF_PID}" \
  -o /tmp/hccl_hang_strace

tail -n 100 /tmp/hccl_hang_strace*
```

重复 `futex(...FUTEX_WAIT...)` 通常表示锁/并行等待；`wait4()` 表示还要继续检查它等待的子 PID；持续
`connect()` 表示网络等待；进程处于 `D` 状态则优先检查磁盘、NFS 或容器挂载。反馈问题时至少保留
`DIAG`、`/tmp/hccl_build_j1_verbose.log` 最后 100 行以及 `BUILD_RC`。

#### 3.1.4 定位配套运行目录并加载验证

不要假设固定的 CPack 目录层级；自动选择最近生成、且同时包含两个库的目录：

```bash
cd /home/l00934901/hccl

HCCL_RUNTIME=$(
  find /home/l00934901/hccl \
    -name libhccl.so -printf '%T@ %h\n' 2>/dev/null | \
  sort -nr | \
  while read -r _ candidate; do
    [[ -f "${candidate}/libhccl_compat.so" ]] && { echo "${candidate}"; break; }
  done
)
if [[ -z "${HCCL_RUNTIME}" ]]; then
  echo "ERROR: build/package did not produce a matching HCCL runtime directory" >&2
  find /home/l00934901/hccl \
    -type f \( -name libhccl.so -o -name libhccl_compat.so \) -print 2>/dev/null
else
  echo "HCCL_RUNTIME=${HCCL_RUNTIME}"
  HCCL_SO=$(readlink -f "${HCCL_RUNTIME}/libhccl.so")
  HCCL_COMPAT_SO=$(readlink -f "${HCCL_RUNTIME}/libhccl_compat.so")
  ls -lh "${HCCL_SO}" "${HCCL_COMPAT_SO}"
  nm -D "${HCCL_SO}" | \
    grep ' A5HcclExplicitMultipathExtensionVersion$'

  # 使用配套依赖和绝对路径加载。
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
fi
```

| 失败位置 | 含义 | 处理 |
|---|---|---|
| 源码搜不到 marker | HCCL 分支/提交不对 | 切换并拉取配套 HCCL 分支 |
| 缺少 `ccu_types.h` | T560 未交付公开 CCU C 头，但旧兼容判断只看版本号 | 拉取包含 `9644f7c`（含 `1c566f8`）的 HCCL 分支后重建 |
| 缺少 `ccu_primitives.hpp` 或其他 `ccu_*.hpp` | T560 未完整交付公开 CCU C++ 包装头 | 拉取包含 `9644f7c` 的 HCCL 分支后重建；不要手工复制单个头文件 |
| `BUILD_RC` 非零 | 编译或打包真实失败 | 查看 `/tmp/hccl_build_j8.log`，不要继续动态库校验 |
| `BUILD_RC=0` 但 `HCCL_RUNTIME` 为空 | 未生成一对配套运行库 | 检查 build 日志结尾及上面的 `find` 输出 |
| `nm` 搜不到 marker | 构建或 package 是旧产物 | 重新执行 HCCL build/package |
| 报错路径为 `/usr/local/Ascend/.../libhccl.so` | 加载了系统 HCCL | 改用打包产物的绝对路径，不能使用 `CDLL("libhccl.so")` |
| 绝对路径加载仍缺符号 | 定制 HCCL 与依赖混用 | 将同一 package 的完整 `lib64` 放在 `LD_LIBRARY_PATH` 最前面 |

性能脚本会对子 `torchrun` 进程同时限定 `LD_LIBRARY_PATH` 和匹配的 `libhccl_compat.so:libhccl.so`，不需要手工在
交互 shell 中长期 `export LD_PRELOAD`；长期设置反而可能影响编译器、`git` 和其他系统命令。

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

HCCL_RUNTIME=$(
  find /home/l00934901/hccl \
    -name libhccl.so -printf '%T@ %h\n' 2>/dev/null | \
  sort -nr | \
  while read -r _ candidate; do
    [[ -f "${candidate}/libhccl_compat.so" ]] && { echo "${candidate}"; break; }
  done
)
test -n "${HCCL_RUNTIME}" || { echo "matching HCCL runtime not found" >&2; false; }
echo "HCCL_RUNTIME=${HCCL_RUNTIME}"

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
