# 显式 peer-plan CCU AllToAll：两卡 / 四卡

分支：`feature/a5-ccu-peer-plan-alltoall-4rank`。本版在已验证的两卡 prepared-plan
数据面上扩展 peer 维度，不改 HCCL、不改 UBUS route table，不加载新 ko。
两卡原接口保留；四卡和 direct-only 使用新增 peer-plan API。

## 1. 范围、数据流与约束

输入、输出均为 contiguous FP32 `[k, elements_per_peer]`，`k=2` 或 `4`。
源 rank `s` 的 `send[d]` 写入目的 rank `d` 的 `recv[s]`；self 分片在本地 D2D copy。
`--bytes` 是**每个目的 rank**的数据量，包含 self；四卡 4 MiB/peer 对应 16 MiB tensor，
每 rank 网络发送总量是 12 MiB，而非把 4 MiB 平分给三个 peer。

对每个远端 peer，一条 direct 加零条或多条显式 relay。relay 不得在通信域内。
上限 `L=(N-k)//(k-1)` 是**每对通信 peer 的 relay 数**；一机八卡，四卡 L=1，
两卡 L=6。这里接受用户给出的分配，不实现自动选卡/控制器。
`--available-phys` 默认一机 0..7，可传实际允许借用的物理卡集合；脚本不自动判断卡是否空闲。
同一个源 rank 的不同 peer 不复用同一 relay；不相交的卡对可以共享 relay。

四卡手工分配示例（只是可修改的示例，不是算法写死）：

| 卡对 | direct | relay |
|---|---|---|
| 0 ↔ 1 | 0 ↔ 1 | 0 ↔ 4 ↔ 1 |
| 2 ↔ 3 | 2 ↔ 3 | 2 ↔ 4 ↔ 3 |
| 0 ↔ 2 | 0 ↔ 2 | 0 ↔ 5 ↔ 2 |
| 1 ↔ 3 | 1 ↔ 3 | 1 ↔ 5 ↔ 3 |
| 0 ↔ 3 | 0 ↔ 3 | 0 ↔ 6 ↔ 3 |
| 1 ↔ 2 | 1 ↔ 2 | 1 ↔ 6 ↔ 2 |

每个 peer 的默认分片权重为 direct `2`、**每条** relay `1`。
每个 peer 独立分配完整 byte budget，按 256-byte 边界取整，最后一条吃余数。
direct-only 每个 peer 只有一条 Channel，传完整分片。

初版限制：每 rank 的所有 Channel 必须在同一个本地 IO die 上；拓扑可能使某些
用户分配不满足这个条件，runtime 会在 Acquire 前明确拒绝，不会偷偷换 relay。
relay 的 ingress/egress 必须同一个 relay die，但该 die 可以不同于通信端点的 die。
一 kernel 当前最多 14 条 Channel（48 个 launch 参数限制）；四卡 direct+1 relay
每 rank 6 条、两卡 direct+6 relay 每 rank 7 条，均在上限内。未来大集群需增加
multi-die/grouped launch 调度，不能仅改变 `k` 或声称没有任何硬件资源上限。

## 2. 拉取与构建

如果本地指导文档差异阻止切分支，先运行以下代码块（备份 tracked 差异，保留
untracked 文件；不需要在内网机器 push，不要恢复旧 stash）。括号中的 `set -e`
只影响子 shell，避免失败时退出当前 Docker 交互 shell。每行末尾不要额外加反斜杠。

```bash
(
set -e
cd /home/l00934901/sgl-kernel-npu
git stash push -m "backup-before-4rank-update-$(date +%Y%m%d_%H%M%S)"
BRANCH=feature/a5-ccu-peer-plan-alltoall-4rank
git fetch origin "${BRANCH}"
git switch "${BRANCH}"
git merge --ff-only "origin/${BRANCH}"
git rev-parse HEAD
git rev-parse "origin/${BRANCH}"
git status --short
)
```

两个提交号应相同。没有本地差异时也可以直接使用下面的拉取命令。

```bash
cd /home/l00934901/sgl-kernel-npu
git fetch origin feature/a5-ccu-peer-plan-alltoall-4rank
git switch feature/a5-ccu-peer-plan-alltoall-4rank
git pull --ff-only origin feature/a5-ccu-peer-plan-alltoall-4rank
source /usr/local/Ascend/cann-9.1.T560/set_env.sh
unset LD_PRELOAD A5_CCU_PREPARED_ALLOW_LAZY_STREAM A5_URMA_TP_TRACE_PREFIX

# HCCL_REPO 只用作旧对象式 CCU custom-op 的构建框架，不需要拉取/编译 patched libhccl。
HCCL_REPO=/home/l00934901/hccl \
bash scripts/build_a5_ccu_urma_route_probe.sh \
  --install --install-path /usr/local/Ascend/cann-9.1.T560

DEEPEP_SINGLE_OP=explicit_multipath_all2all_ccu bash build.sh -a deepep Ascend950
WHEEL=$(ls -t output/deep_ep-*.whl | head -1)
test -f "${WHEEL}"
python3 -m pip install --force-reinstall --no-cache-dir --no-deps "${WHEEL}"
```

若旧仓有未提交修改，先备份/stash，不要直接 reset 或删 untracked。
构建脚本的 offline cann-cmake 与 UTF-8 源码预检保持原有行为。
需要重新装 route package 和 wheel；**不需要重编 HCCL**。

**2026-10-09 测试统一更新仅修改 Python、shell 与文档**：若第 3 节独立 peer-plan
ABI 已通过，本次仅拉取更新即可，不需要重复编译/安装 route package 或 wheel。
先做无需 NPU 的入口回归，再执行第 4 节：

```bash
python3 tests/python/deepep/test_a5_ccu_peer_entrypoint_cpu.py
python3 tests/python/deepep/test_a5_ccu_test_runtime_cpu.py
python3 tests/python/deepep/test_a5_ccu_peer_plan_alltoall.py --help
```

两个测试文件均应显示 `OK`，help 包含 `--manifest`、`--available-cards`、`--run-dir`、
`--graph-backend`，不能显示旧两卡 benchmark 的 `--implementation`。

若之前已经被迫退出 Docker，在宿主机使用 `docker exec -it sglang_yuanwen_old bash`
重新进入（容器名按实际环境修改）。Git 报错不会主动停止容器；编译脚本用
`bash scripts/...` 执行，不要 `source` 编译脚本或将带 `exit` 的恢复代码直接粘贴进交互 shell。

## 3. 验证新增 API（独立 ABI，不能只检查旧 ABI 11/12）

```bash
export A5_CCU_ROUTE_PROBE_LIB=/usr/local/Ascend/cann-9.1.T560/opp/vendors/cust/lib64/liba5_ccu_urma_route_probe.so
source scripts/a5_ccu_test_env.sh
a5_ccu_prepare_test_env /usr/local/Ascend/cann-9.1.T560 "${PWD}"
# 和正式 worker 完全相同的加载/ABI 预检；不初始化卡、不创建通信域。
timeout --signal=TERM --kill-after=5 180 \
  python3 tests/python/deepep/test_a5_ccu_peer_plan_alltoall.py --runtime-check
python3 - <<'PY'
import ctypes, os
from pathlib import Path
# 与正式 worker 一样，先全局加载 route library，再导入 torch。
route = ctypes.CDLL(os.environ['A5_CCU_ROUTE_PROBE_LIB'], mode=ctypes.RTLD_GLOBAL)
import torch
import deep_ep.deep_ep_cpp as ext
from deep_ep import Buffer

deep = ctypes.CDLL(str(Path(ext.__file__).resolve()))
for library, name in [(route, 'A5CcuPeerPlanAbiVersion'), (deep, 'A5DeepEpPeerPlanAbiVersion')]:
    fn = getattr(library, name); fn.restype = ctypes.c_int
    print(name, fn()); assert fn() >= 1
assert hasattr(Buffer, 'prepare_ccu_urma_peer_plan')
for k in (2, 4):
    x = torch.empty((k, 1024), device='meta', dtype=torch.float32)
    y = torch.ops.deep_ep.ccu_urma_peer_plan_alltoall(x, 1, [])
    assert y.shape == x.shape and y.dtype == x.dtype
print('peer-plan Meta PASS; not a hardware correctness test')
PY
```

## 4. 先测四卡 direct-only

仅通信域卡需要运行 rank 进程，relay 上没有用户进程。
确认通信卡可用后执行；构建/预检不自动抢卡或改共享路由。

```bash
bash scripts/run_a5_ccu_peer_plan_alltoall.sh \
  --devices 0,1,2,3 \
  --bytes 4194304 --warmup 1 --iters 3 --repeats 1 \
  --timeout-seconds 600 \
  --output-root /home/l00934901/profiling
```

不传 `--relay-map` 就是 direct-only，也不依赖 hccn EID 文件。
逻辑 rank 与 `--devices` 的顺序对应，可使用其他四张卡或任意两张卡。

### 4.1 `unrecognized arguments` / 错误进入旧两卡脚本

若日志出现 `test_a5_ccu_urma_multiroute_all2all.py: error: unrecognized arguments:
--manifest ... --available-cards ... --run-dir ... --graph-backend ...`，且最终 status=137，
这不是四卡 Channel/CCU 失败。旧版 peer 测试导入了两卡 benchmark 的辅助函数；
两卡 benchmark 在模块初始化时 `prepare_runtime()` 修改环境并执行
`os.execvpe(..., __file__, ...)`，把四卡 worker 替换成了两卡入口。新参数因而被
旧 parser 拒绝，分布式启动器随后未能正常结束，外层 timeout TERM/KILL。
仅凭 137 不能断言 OOM，本次日志中明确出现了 timeout 的 Killed。

修复版将 barrier/profiler 移到无启动副作用的 `a5_ccu_peer_test_support.py`，
peer 入口先解析自己的参数再导入 NPU runtime，不再导入旧 benchmark。
拉取后执行第 2 节入口回归，再重跑本节同一条 direct-only 命令即可。
检查新日志中的阶段标记：

```bash
RUN_DIR=$(ls -dt /home/l00934901/profiling/a5_ccu_peer_plan_* 2>/dev/null | head -1)
if test -n "${RUN_DIR}" && test -f "${RUN_DIR}/cases/peer_plan_r1.log"; then
  grep -nE 'PEER_CASE_PHASE|PEER_CASE_FAILURE|RESULT_JSON|unrecognized arguments' "${RUN_DIR}/cases/peer_plan_r1.log"
fi
```

至少应先看到 `runtime_bootstrap`、`CASE_TEST_RUNTIME`、`import_runtime`，随后是
`PEER_RUNTIME_VERIFIED`、`communicator_init`、`prepare_plan`；
只有进入这些阶段后才开始判读通信域/建链/数据面错误。四个 rank 都输出 correctness
PASS 的 `RESULT_JSON` 且进程正常退出 status=0，才算此 case 完整通过。

### 4.2 四个 rank 都已 PASS，但 status=137：退出阶段诊断

2026-10-08 的四卡 direct-only 日志已观察到所有 rank 的 Channel Acquire 成功、
Channel state=0、plan bound，随后三次 changed-payload 检查及正式执行完成，
四个 rank 均输出 correctness=PASS。这证明该次四卡 direct 数据面通过了测试，
但外层 timeout 最终杀掉进程，不能算完整运行通过，更不能定位为 relay 失败。

原版 RESULT_JSON 后还有最后一次 NPU synchronize、文件 barrier、
`dist.destroy_process_group()`、局部对象析构及 Python/native 退出。旧日志没有
这些位置的标记，不能直接断言哪个 API 死锁。统一环境后仍需有卡复测，不能声称已修复退出。
不要通过 `os._exit(0)` 绕过清理来把结果伪装为正常 PASS。

本次修改仅 Python/报告，无需重编算子或重装 wheel。推送并拉取后，重跑第 4 节
同一命令；可保留 timeout=600。最后 synchronize/barrier/ProcessGroup 清理期间若卡住，
默认每 30 秒打印 Python 线程堆栈；正常清理完成后取消 watchdog，输出
`cleanup_watchdog_cancelled`。随后 native/interpreter 退出由第 4.3 节的外部观察器负责，
不留下持续运行的 Python watchdog。只打印堆栈，不修改路由、驱动或共享硬件配置：

```bash
export A5_CCU_PEER_CLEANUP_WATCHDOG_SECONDS=30
bash scripts/run_a5_ccu_peer_plan_alltoall.sh \
  --devices 0,1,2,3 \
  --bytes 4194304 --warmup 1 --iters 3 --repeats 1 \
  --timeout-seconds 600 \
  --output-root /home/l00934901/profiling
```

可在第二个终端读取日志，不必等满十分钟再看：

```bash
RUN_DIR=$(ls -dt /home/l00934901/profiling/a5_ccu_peer_plan_* 2>/dev/null | head -1)
if test -n "${RUN_DIR}" && test -f "${RUN_DIR}/cases/peer_plan_r1.log"; then
  LOG="${RUN_DIR}/cases/peer_plan_r1.log"
  grep -nE 'RESULT_JSON|PEER_CASE_PHASE|PEER_CASE_FAILURE|Timeout|destroy_process_group' "${LOG}"
  tail -n 160 "${LOG}"
fi
```

每 rank 最后一个 begin/done 标记界定位置：

| 最后阶段 | 当前收敛范围 |
|---|---|
| `final_synchronize_begin` 无 done | 最终 NPU stream 同步 |
| `reported_barrier_begin` 无 done | 文件 barrier；检查哪个 rank 没到达 |
| `destroy_process_group_begin` 无 done | ProcessGroupHCCL 销毁/底层等待 |
| `main_return_begin` 无 done | main 局部对象释放 |
| `main_return_done` / `python_atexit` | Python 退出或 native/static 资源析构；需结合堆栈 |

`python_atexit` 只说明这个 Python 回调已执行，不代表所有底层析构完成。
如果全部 RESULT_JSON 已通过而退出仍失败，新报告保留 data correctness=PASS，
lifecycle=`FAILED_OR_INTERRUPTED_AFTER_DATA_PASS`，complete=False；计时仅供诊断，
不纳入有效性能样本。历史日志也可以重新运行第 8 节分析命令重新分类。

### 4.3 `main_return_done` 和 `python_atexit` 都出现但仍超时

最新四卡 direct-only 实验已在全部 rank 上观察到 final synchronize、reported barrier、
destroy_process_group、main return 均完成，且均输出 python_atexit。此前怀疑的
`dist.destroy_process_group()` 内部等待已排除。当前范围是 Python 后续退出/native
清理、进程退出中的驱动等待，或 rank 已退出但 torchrun 启动器未收尾；不能据此
断言 HCCL 模块某个析构函数，也不能把“atexit 回调执行”当作进程已经退出。

更新版 runner 在每个 repeat 启动一个独立观察器。四个 rank 全部报告数据 PASS
后仍未退出超过 30 秒，自动在 `r1/shutdown_trace/` 保存三次 /proc 快照，识别
本次 case 的 rank/启动器/timeout 进程。匹配依据是完整测试脚本名和精确 `--run-dir`
参数，不扫描或 attach 其他用户的 Python 作业。默认只是只读采集。
阶段标记还增加了 PID，便于对应 rank。

若容器已经有 gdb，推荐本次额外加 `--shutdown-native-backtrace`。GDB 在测量完成后
短暂停顿本次匹配进程并逐个采集调用栈；没有 gdb 或 ptrace 权限时记录不可用，
不安装工具，不修改宿主机 ptrace 配置，不重启容器或驱动。

```bash
bash scripts/run_a5_ccu_peer_plan_alltoall.sh \
  --devices 0,1,2,3 \
  --bytes 4194304 --warmup 1 --iters 3 --repeats 1 \
  --timeout-seconds 600 \
  --shutdown-native-backtrace \
  --output-root /home/l00934901/profiling
```

不加这个新参数也会生成只读 /proc 快照。无需重编算子或安装 wheel。
先验证观察器回归可在无卡机器执行：

```bash
python3 tests/python/deepep/test_a5_ccu_peer_shutdown_cpu.py
```

在第二个终端或运行结束后检查：

```bash
RUN_DIR=$(ls -dt /home/l00934901/profiling/a5_ccu_peer_plan_* 2>/dev/null | head -1)
if test -n "${RUN_DIR}" && test -d "${RUN_DIR}/r1"; then
  cat "${RUN_DIR}/r1/shutdown_monitor.log"
  TRACE_DIR="${RUN_DIR}/r1/shutdown_trace"
  if test -d "${TRACE_DIR}"; then
    cat "${TRACE_DIR}/scope.json"
    sed -n '1,220p' "${TRACE_DIR}/proc_snapshot0.json"
    for file in "${TRACE_DIR}"/native_pid*.txt; do
      test -f "${file}" || continue
      echo "===== ${file} ====="
      sed -n '1,220p' "${file}"
    done
  fi
fi
```

优先提供 `scope.json` 和 `native_pid*.txt`。若 rank 进程都不在而只剩 torchrun，
转查 elastic/launcher 清理；若 rank 仍在，按用户态栈查实际阻塞函数。
`futex`/`do_wait` 只说明锁/条件变量或子进程等待，不能从 /proc wchan 直接命名
HCOMM destructor。`UNAVAILABLE`/`Operation not permitted` 是观测权限缺口，不代表
数据面失败。若没有 native 栈，保留这个边界，不要继续猜测 TP/路由参数导致退出。

### 4.4 两卡与四卡共用环境准备，不再维护两套初始化环境

旧两卡与新 peer-plan launcher 均调用 `scripts/a5_ccu_test_env.sh`：source 相同 CANN，
设置 `CCU_SCHED`，保留用户配置的 `HCCL_BUFFSIZE`（未配置时沿用两卡默认 2300），
清除旧 tracer/调试环境。vendor 路径从当前 checkout 推导，不 source 含开发服务器绝对路径
的生成脚本。worker 均使用 `tests/python/deepep/a5_ccu_test_runtime.py`，优先安装的 wheel，
在导入 Torch **之前**以 `RTLD_GLOBAL` 加载 route library，并输出实际 .so 路径。
需要刷新动态加载路径时只 re-exec 当前 worker，自身参数保持不变，不跳转到另一测试入口。

这是对旧两卡已验证加载顺序的对齐，不能据此推断 native 退出卡顿一定由加载顺序造成。
通信域仍由 `dist.init_process_group("hccl")` 创建、Buffer 持有；两卡/四卡只有 rank 数、
peer-plan/旧 prepared-plan API 和路径计划不同，不需要另一个 HCCL 环境或 patched libhccl。
launcher 的预检也直接调用正式 worker 的 `--runtime-check`，避免预检成功但 worker 加载另一套库。

图模式直接在 capture stream 准备 plan，数据校验与图外 warmup 也使用该 stream。
不再先在默认 stream 建一套资源，再在 capture stream 重建一套。
进入 capture 前的 bind 是对同一 stream 的幂等检查，不是第二次创建 Channels。
当前没有独立 plan/Channel Destroy API，不能靠无限新建 stream 或 plan 规避资源问题。

这次还加强了校验：发送数据含 source/destination 和分片内位置；接收缓冲区预填无效值，
三次变化输入及 warmup 后都核对完整输出，避免常数填充掩盖分片重叠/缺失或旧结果。
重跑第 4 节 direct-only；只有所有 rank 数据 PASS **且 status=0** 后，再运行 relay/graph/profiling。

## 5. 四卡显式 direct+1 relay

下面文件已随仓提供，可自行编辑卡对/relay/plane；key 为升序**物理**卡对，
不是逻辑 rank。省略的卡对只走 direct。若三个 peer 均配置 relay，则每张卡使用三张
互不相同的外部 relay；也允许只为部分 peer 配置 relay。

```bash
cat docs/topology/a5_peer_plan_4rank_example.json

# 先解析真正的 topology edge/port/EID，尚不打流。
python3 scripts/prepare_a5_ccu_peer_plan.py \
  --devices 0,1,2,3 \
  --relay-map docs/topology/a5_peer_plan_4rank_example.json \
  --topology docs/topology/a5_hccn_device_topology_raw.txt \
  --topology-json /usr/local/Ascend/driver/topo/950/atlas_950_1.json \
  --output-dir /home/l00934901/profiling/peer_plan_preview
column -s $'\t' -t /home/l00934901/profiling/peer_plan_preview/peer_plan.tsv

bash scripts/run_a5_ccu_peer_plan_alltoall.sh \
  --devices 0,1,2,3 \
  --relay-map docs/topology/a5_peer_plan_4rank_example.json \
  --bytes 4194304 --warmup 1 --iters 3 --repeats 1 \
  --timeout-seconds 600 \
  --output-root /home/l00934901/profiling
```

若存在多个合法 topology/EID 候选，resolver **拒绝猜测**。把 relay 改成对象，例如
`{"physical_device":4,"plane":1}`，明确 relay IO die。plane 不等于源卡 IO die。
若指定边不通或跨本地 die，保留失败结果，改用户分配或只走 direct；不要自动用 native2 冒充所选 relay。

## 6. 两卡兼容与任意合法 relay 数

新接口同样支持 `2,3`，示例明确指定 relay `0,1`：

```bash
bash scripts/run_a5_ccu_peer_plan_alltoall.sh \
  --devices 2,3 --relay-map docs/topology/a5_peer_plan_2rank_example.json \
  --bytes 4194304 --warmup 1 --iters 3 --repeats 1 \
  --output-root /home/l00934901/profiling
```

`{"2,3":[0]}`、`{"2,3":[0,1,4]}`、五条或六条均允许，不要求偶数。
两卡 direct-only 只需 `--devices 2,3`，不传 map。
原来的 `run_a5_ccu_explicit_multipath_alltoall_perf.sh` 与旧 APIs 未替换，仍按旧两卡行为运行。

## 7. 性能 / profiling / graph 分开验证

性能使用初始化一次、100 次独立 warmup、20 次独立执行；只给正式 20 次开 profiler。
每次执行后 synchronize，不含资源准备、warmup 或 capture。正式 warmup 前还有三次
变化 payload 的正确性检查，它们不计时；统计整段 HCCN 数据量时需要计入这些额外调用。
测量或起跑 rendezvous 抛出异常时也通过 finally 停止已启动的 profiler，不继续生成 PASS。

```bash
bash scripts/run_a5_ccu_peer_plan_alltoall.sh \
  --devices 0,1,2,3 \
  --relay-map docs/topology/a5_peer_plan_4rank_example.json \
  --bytes 4194304 --warmup 100 --iters 20 --repeats 1 --profile \
  --output-root /home/l00934901/profiling
```

prof 在结果目录的 `r1/profiling/rank0/` 到 `rank3/`，每个 repeat 独立目录。
报告中的 host-call us 含提交与同步开销；MindStudio CCU task 时长才是 device timing。
不把 host timing 除 payload 当作 link bandwidth。

两卡/四卡共用文件 rendezvous，所有 rank 在 profiler 启动后、计时前等待同一个
单机 monotonic 时间点；减少旧 10 ms 文件轮询造成的首调用等待。此等待不计入样本，
`measurement_start_lateness_us` 记录 host 调度迟到量。它不是硬件时钟同步，也不保证
所有 rank 周期级同时发起；首样本若仍较慢，应检查调度迟到和 profiler，而非直接推断路径瓶颈。
此对齐只适用于当前一机测试，多机需另做时钟/同步设计。host 每次计时仍包含该调用等待
其他 rank 的时间；平均值与 device-only profiling 必须分别报告。

单独验证新接口 graph capture/replay：

```bash
bash scripts/run_a5_ccu_peer_plan_alltoall.sh \
  --devices 0,1,2,3 \
  --relay-map docs/topology/a5_peer_plan_4rank_example.json \
  --graph-backend aclgraph \
  --bytes 4194304 --warmup 1 --iters 3 --repeats 1 \
  --output-root /home/l00934901/profiling
```

新四卡图支持仍需此有卡测试确认；旧两卡 capture PASS 不自动等于四卡 PASS。
capture stream 的 prepare/bind/warmup 在 capture 外进行，replay 修改输入验证非旧输出。
每个 plan 的执行须串行使用；不能在多 stream 同时使用共享 Channel 的 notify 槽。

### 7.0 多路径 batch 验证与性能复测（新增，显式开启）

第7节旧命令默认 `--sync-mode per-call`，每次调用完成后才重新下发下一次。
这可能把四个rank的host提交错位放大为CCU内部等待，warmup100次也不能排除。
CCU时长包含地址/token通知、WriteNb完成和peer完成通知的等待，并非纯搬运时间。
现在增加 `--sync-mode batch`，用于与第7.1节原生batch基线采用相同显式同步口径。
batch仍是验证模式，不声称已经在所有尺寸、权重、流或设备上证明可连续排队。

**先自动做安全门槛，再采性能。** batch每次运行先做两轮图外排队校验，每轮
`min(max(iters,2),20)`次：复用同一send/recv，每次输入不同、接收缓冲区被清空；
在同一执行stream上排入生产数据、CCU调用、该次输出的独立snapshot拷贝。
仅在该轮末尾同步，然后逐一精确校验全部snapshot和最终输出，而非只看最后一次。
这用于检测生产者/CCU/消费者顺序、丢失调用、旧输出和通知复用问题。
每一轮全部rank校验通过才进入下一阶段；失败时不进入warmup或profiler，不生成性能PASS。
校验需要额外HBM（四卡4MiB/peer、20次时输入与snapshot合计约640MiB/rank）；
这些校验调用不计时、不采profiling，但统计覆盖整个进程的HCCN流量时必须计入。

```bash
# 可先执行CPU回归，无需NPU。
python3 tests/python/deepep/test_a5_ccu_peer_entrypoint_cpu.py
python3 tests/python/deepep/test_a5_ccu_peer_plan_cpu.py

# 多路径batch：自动校验 -> warmup100 -> 正式20；只采正式20，跑一轮。
bash scripts/run_a5_ccu_peer_plan_alltoall.sh \
  --devices 0,1,2,3 \
  --relay-map docs/topology/a5_peer_plan_4rank_example.json \
  --sync-mode batch --graph-backend none \
  --bytes 4194304 --warmup 100 --iters 20 --repeats 1 --profile \
  --output-root /home/l00934901/profiling
```

两卡采用同一入口，将设备改为`2,3`、map改为第6节的两卡map即可。
warmup连续提交后同步一次，所有rank预热完成后开启profiler；正式计时前做一次
设备同步和起跑rendezvous，正式20次连续提交，最后同步确认完成后停止profiler。
batch host均值是整批耗时/20，不伪造逐次host延迟；设备单次时长仍看MindStudio。
与原生比较必须保持相同卡组、per-peer字节、warmup、iters、profiler和sync-mode。
批量提交仍可能受host供给、peer等待和链路争用影响，不承诺消除所有波动。

```bash
(
set -e
RUN_DIR=$(ls -dt /home/l00934901/profiling/a5_ccu_peer_plan_* 2>/dev/null | head -1)
test -n "${RUN_DIR}" && test -d "${RUN_DIR}"
echo "RUN_DIR=${RUN_DIR}"
grep -HnE 'PEER_BATCH_VALIDATION|PEER_MEASUREMENT|PEER_CASE_FAILURE' "${RUN_DIR}"/cases/*.log
python3 scripts/analyze_a5_ccu_peer_plan.py --run-dir "${RUN_DIR}"
sed -n '1,200p' "${RUN_DIR}/peer_plan_report.md"
find "${RUN_DIR}" -path '*/results/rank*.json' -o -type d -name '*_ascend_pt'
)
```

应看到全部rank `PEER_BATCH_VALIDATION ... PASS`、`sync_mode=batch`和完整数据/退出PASS。
新结果保存为`r*/results/rank*.json`，分析和退出观察器以独立文件为准，避免stdout交错漏判；
仍兼容旧stdout-only结果。分析器拒绝缺少batch门槛记录、模式不一致或错误的批均值。
需要A/B对照时把`--sync-mode batch`改为`per-call`，其余参数保持不变。
batch目前不与aclgraph组合；图capture/replay继续使用per-call单独验证。
只修改Python/shell/分析脚本和文档，无HCCL或CCU二进制改动，**拉取即可，不用重新编译安装**。

### 7.1 四卡原生 HCCL AllToAll baseline（不是自定义 direct-only）

使用新增 `scripts/run_a5_ccu_native_alltoall_baseline.sh`。实际调用是
`dist.all_to_all_single(recv, send)`，由系统 HCCL 自己选择算法与路径。
**第 4 节的 direct-only 仍是我们的 peer-plan 算子，不能当作原生 baseline。**
这里不选择 CommLink candidate0/2，也不合成 EID、创建自定义 plan、加载 route .so，
不要求安装 DeepEP wheel或 patched libhccl。不要加旧两卡实验的 `--route-index`。
原生 HCCL 内部可能使用多路径，因此此 baseline 不意味着强制纯 direct。

本次只新增 Python/shell/分析脚本及文档，已有四卡环境**拉取即可，无需重新编译安装**。
设备必须和自定义算子对比实验一致；下面用通信卡0、1、2、3，没有 relay rank 进程。

```bash
cd /home/l00934901/sgl-kernel-npu
source /usr/local/Ascend/cann-9.1.T560/set_env.sh
unset LD_PRELOAD A5_URMA_TP_TRACE_PREFIX

# 可先做不依赖 NPU 的脚本回归。
python3 tests/python/deepep/test_a5_ccu_native_baseline_cpu.py

# 快速正确性与退出检查；先通过，再采性能。
bash scripts/run_a5_ccu_native_alltoall_baseline.sh \
  --devices 0,1,2,3 \
  --bytes 4194304 --warmup 1 --iters 3 --repeats 1 \
  --timeout-seconds 600 \
  --output-root /home/l00934901/profiling

# 正式 baseline：只跑一轮，100次warmup + 20次正式执行。
# batch模式连续提交，最后同步一次；profiler只采正式20次，不采warmup。
bash scripts/run_a5_ccu_native_alltoall_baseline.sh \
  --devices 0,1,2,3 \
  --bytes 4194304 --warmup 100 --iters 20 --repeats 1 --profile \
  --sync-mode batch \
  --timeout-seconds 600 \
  --output-root /home/l00934901/profiling
```

可改 `--devices` 为其他四张可用卡或两张卡，顺序决定逻辑 rank。
脚本与 peer-plan 复用 CANN 环境准备、文件 barrier、测量起跑 rendezvous和 profiler helper；
区别是 baseline 不调用 `Buffer.prepare/bind`，而是原生 AllToAll 自己准备通信资源。
`HCCL_OP_EXPANSION_MODE=CCU_SCHED` 是请求原生 CCU 调度，最终实际实现要在 profiling
中确认，不能仅凭这个环境变量声称所有通信任务都是 CCU。

每 rank 的 send/recv 都为 `[4,1048576]` FP32，即 **16 MiB tensor**；每个目的 rank
分片 **4 MiB**，其中 self 4 MiB，网络发送给另外三卡共 **12 MiB/rank**。
数据模式和 peer-plan 一致。三次变化输入的正确性检查、首次资源创建、100次 warmup
均不计时、不开 profiler。默认 `--sync-mode batch`：连续提交100次 warmup 后同步；
全部 rank 完成预热后开启 profiler，计时前先同步并做起跑 rendezvous；连续提交正式20次
AllToAll，最后同步一次，确认完成后关闭 profiler。循环中不打印、不做数据校验，
校验在 profiler 关闭后完成。总计100次预热与20次正式调用（另有3次图外正确性检查）。
脚本去掉的是显式逐次设备同步；`dist.all_to_all_single` 本身仍遵循目标后端的等待语义，
不保证底层一定形成多个同时在途的任务，也不保证完全消除host提交或peer等待波动。
profiling可能包含测量阶段的同步/运行时任务；“20次”指20次AllToAll调用，
不保证只出现20个CCU任务，因为一次AllToAll可能展开为多个任务。不强制 graph capture。
脚本默认值为 `--warmup 100 --iters 20 --repeats 1 --sync-mode batch`；无需连续运行三轮。

**两卡与四卡必须采用同一同步口径。** 两卡复测只需将上面命令改为 `--devices 2,3`。
此前旧两卡 native 测试没有逐次显式同步，新四卡版本却逐次同步，二者不能直接比较。
需要因果对照时，仅将 `--sync-mode batch` 改为 `--sync-mode per-call`，其他参数不变。
后者每次调用后同步，可能把重新下发的rank错位放大成CCU内部的peer等待；
仅增加warmup不能排除这种等待。默认batch模式测的是连续调用的稳态性能。

原生默认batch；自定义peer-plan默认仍逐次完成，也可按第7.0节显式开启有门槛检查的batch。
没有修改HCCL、Channel、EID或多路径CCU kernel。拉取脚本更新即可，无需编译安装。

结果目录如下，当前命令只启动一轮四 rank 进程：

```text
a5_ccu_native_alltoall_0_1_2_3_<time>.<suffix>/
├── run_settings.json                  # 数据量、版本环境、HCCL配置
├── cases/native_alltoall_r1.log
├── cases/native_alltoall_r1.status
├── r1/results/rank0.json ... rank3.json
├── r1/profiling/rank0/..._ascend_pt/    # --profile时生成，rank1..3同样
├── native_alltoall_results.tsv
├── native_alltoall_summary.json
└── native_alltoall_report.md
```

分析文档自动生成。也可中断后手动分析；未跑的 repeat 标记不完整，已完成 repeat 的
样本仍保留。每 rank 结果独立、原子写 JSON，避免多进程终端输出拼接导致漏分析。

```bash
(
set -e
RUN_DIR=$(ls -dt /home/l00934901/profiling/a5_ccu_native_alltoall_* 2>/dev/null | head -1)
test -n "${RUN_DIR}" && test -d "${RUN_DIR}"
echo "RUN_DIR=${RUN_DIR}"
python3 scripts/analyze_a5_ccu_native_alltoall.py --run-dir "${RUN_DIR}"
column -s $'\t' -t "${RUN_DIR}/native_alltoall_results.tsv"
sed -n '1,240p' "${RUN_DIR}/native_alltoall_report.md"
grep -HnE 'NATIVE_CASE_PHASE|NATIVE_CASE_FAILURE|NATIVE_RUNTIME|RESULT_JSON' "${RUN_DIR}"/cases/*.log
find "${RUN_DIR}" -type d -name '*_ascend_pt' -print
)
```

将对应 `r*/profiling/rank*/..._ascend_pt/` 导入 MindStudio。baseline 和自定义 peer-plan
要用**相同卡组、每 peer 数据量、warmup、iterations、同步模式、是否开启 profiler**进行比较。
host统计按每轮最慢 rank 的平均耗时汇总；它包含提交与同步，不等于设备CCU时长。
比较设备性能时查看正式执行的 CCU task，若一个 AllToAll 展开成多个任务，需看完整
调用的相关通信任务跨度，不能只挑最快的一条。profiling中的无效 size/rank字段不能用于
反推带宽。所有 rank 数据正确且进程 status=0，才计为完整 baseline。

## 8. 结果检查、中断后分析、物理转发确认

```bash
RUN_DIR=$(ls -dt /home/l00934901/profiling/a5_ccu_peer_plan_* 2>/dev/null | head -1)
test -n "${RUN_DIR}" && test -d "${RUN_DIR}"
python3 scripts/analyze_a5_ccu_peer_plan.py --run-dir "${RUN_DIR}"
column -s $'\t' -t "${RUN_DIR}/peer_plan_results.tsv"
sed -n '1,240p' "${RUN_DIR}/peer_plan_report.md"
grep -HnE 'PEER_CASE_PHASE|PEER_PATH_DESC|PEER_PATH_CHANNEL|PEER_PLAN_BOUND|PEER_CASE_FAILURE|RESULT_JSON|PEER_GRAPH_' \
  "${RUN_DIR}"/cases/*.log
```

分析要求所有 k 个 rank 的完整、唯一 RESULT_JSON，采样数量匹配 iterations、时间为有限
非负值且平均值与样本一致。数据 PASS 与退出 lifecycle 分开；status=0 才算 complete。
Ctrl-C 中断时，无 status 文件的 case 标记 INTERRUPTED，已有完整 case 仍可分析。

`plans/peer_plan.json` 的 `relay_edges` 保存 src/dst/relay 真实 die/port、拓扑边及完整 EID。
物理 relay 验证沿用已有 HCCN before/after 采样，在每条预期四端口链上比较 tx/rx flit；
本新脚本**不自动采 HCCN**。需要以这些端口和工作负载关联，而非某张卡总流量。
仍须排除公共机器其他用户噪声，不能把本脚本数据正确性 PASS 单独称为物理转发证明。

## 9. API、调用链与控制器扩展点

```text
用户明确的 physical-pair -> relay list JSON
  -> prepare_a5_ccu_peer_plan.py
  -> topology edge join + endpoint EID inventory
  -> 完整对称 peer_plan.tsv（direct 在前，relay 在后）
  -> Buffer.prepare_ccu_urma_peer_plan()（图外）
  -> HcclCcuUrmaPeerPlanCreate
  -> ValidatePeerPlan + GetPeerPlanResources
  -> RankGraph direct CommLink -> 显式 relay endpoint pair
  -> 必要 Thread Acquire -> 一批 ChannelAcquire -> ChannelGetStatus
  -> RouteKernel Register + RegisterFinish -> plan_handle

Buffer.ccu_urma_peer_plan_alltoall[_out](send, ..., handle)
  / torch.ops.deep_ep.ccu_urma_peer_plan_alltoall(send, handle, weights)
  -> 当前 stream host-queue flush + tensor recordStream
  -> HcclCcuUrmaPeerPlanExecute（只查已绑定资源，不建链、不读文件）
  -> 每 peer 独立 weighted offsets/bytes -> self D2D
  -> CCU launch -> 交换每 Channel 的 recv base/token
  -> 所有 peer/path WriteNb -> 所有 completion wait -> peer completion handshake
```

| API | 输入 | 行为 |
|---|---|---|
| prepare_ccu_urma_peer_plan | plan_id, TSV 绝对路径, N | 完整、对称计划；进程内 handle |
| bind_ccu_urma_peer_plan | handle | 图外绑定当前执行/capture stream |
| ccu_urma_peer_plan_alltoall | send, handle, path_weights=[] | functional torch custom op，分配 output |
| ccu_urma_peer_plan_alltoall_out | send, recv, handle, path_weights=[] | 复用 output；测试性能用此入口 |
| HcclCcuUrmaPeerPlanExecute | buffers, elements/peer, rankSize, dtype, comm, stream, handle, weights/count | validates owner/shape/stream before enqueue |

非空 weights 是**本 rank**按 peer 升序、各 peer 的 direct+relay 顺序展开，必须与 channelCount 一致。
默认空 list 直接用 manifest 权重，不是 path_mask。初版不接受 0 权重以禁用路径，
direct-only 用独立 plan；policy 只能重分已有路径，不能凭空增加 Channel。
host weights 在 capture 时固化，replay 期间修改 Python list 不会动态改图内分片。
后续控制器可以输出同一 JSON/TSV，图外准备新 plan，再选择 plan/weights；自动计算分配、
device-side 动态 policy 和 multi-die 调度均是独立扩展，不在本版伪装完成。

计划/Channel/kernel 与 communicator 及 stream 绑定，直到进程退出保持资源；没有单独
Destroy API，因此不要无限创建不同 plan_id/stream。保持 Buffer/communicator 存活，所有
异步执行及 profiler 完成后再销毁 communicator。此版不是带 tiling.cpp 的标准 Ascend C
算子；它延续此前 prepared-plan torch custom op 路线。

两卡实现细节见 [两卡设计](A5_CCU_TWO_RANK_PREPARED_PLAN_DESIGN.md)。新增资源/数据面在
`examples/a5_ccu_urma_route_probe/op_host/peer_plan_runtime.cc`、`utils.cc`、
`common/peer_plan.h`；沿用 `op_kernel_ccu/route_kernel.cc` 的事件与 notify 机制。

## 10. 无卡回归

```bash
c++ -std=c++14 -Wall -Wextra -Werror \
  -Iexamples/a5_ccu_urma_route_probe/common \
  tests/python/deepep/test_a5_ccu_peer_layout.cpp -o /tmp/a5_peer_layout_test
/tmp/a5_peer_layout_test
python3 tests/python/deepep/test_a5_ccu_peer_plan_cpu.py
python3 tests/python/deepep/test_a5_ccu_peer_entrypoint_cpu.py
python3 tests/python/deepep/test_a5_ccu_test_runtime_cpu.py
python3 tests/python/deepep/test_a5_ccu_peer_shutdown_cpu.py
python3 tests/python/deepep/test_a5_ccu_peer_meta.py
```

开发容器已验证旧对象式 CCU runtime 与 DeepEP 编译、布局和 policy 校验、Meta/fullgraph
形状检查；没有 NPU 的容器无法证明四卡 Acquire、实际转发或 graph replay。

2026-10-09 在 `cam_lyw_dev_91` 的 `cam_py311_pt28` Python 环境通过了入口、共享
bootstrap/rendezvous、计划/报告、退出观察器 CPU 回归，以及两卡/四卡 Meta/fullgraph
和 C++ 分片布局回归。真实动态加载预检使用该容器之前编译的
`/tmp/a5-peer-plan-install/lib/liba5_ccu_urma_route_probe.so`（peer ABI=1）及安装的 wheel，
两卡 prepared 入口亦可正常加载。这不是有卡传输测试。
容器 CANN 默认安装位置仍是旧 route package，预检能正确拒绝缺失 peer ABI 的库；
没有为了这次 Python 修改覆盖安装包。用户机器第 3 节 peer ABI 已通过则不用重新安装。

若预检报 `No module named torch`，先核对 `command -v python3` 与
`python3 -c 'import sys, torch; print(sys.executable, torch.__version__)'`，使用已经安装
Torch/torch_npu/DeepEP 的 Python 环境；不应通过更换 HCCL、驱动或源代码来修解释器环境。
