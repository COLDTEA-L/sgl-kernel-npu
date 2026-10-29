# EP8 单 relay 贪心 AllToAllV：接口、正确性和性能对比

分支：`feature/a5-ccu-8rank-heuristic-alltoallv`，基于已通过新机测试的
`feature/a5-ccu-8rank-direct-alltoallv` / `d47d7b6`。
新算法仅用于八卡；relay 只能是同一通信组内、且不同于源和目的的 rank。
2/4 卡等长多路径 API、组外 relay 规则及原执行路径保留，不使用这个规划器。

本版默认 `--executor pipeline`，执行 **direct 独立推进 + relay 分块双缓冲流水线**。
候选 relay 仍由原来的单 relay 贪心算法选择，输入仍是本批实际 packed 行数 C[8][8]。
新增实现使用现有 CCU direct AllToAllV API，不改 2/4 卡接口，不用组外 relay。

本次默认 `--planner native`：C++ 固定数组贪心选路和分块布局，静态拓扑校验一次。
`--planner python` 保留原参考实现。两者选择相同 relay、分流行数和偏移；原生扫描顺序不同，
诊断用候选评估次数可能不同。每轮 C 都会重新处理，没有固定路由缓存，也没有自动关闭 relay。

```text
调用者 stream：输入就绪 ── direct CCU（原偏移直接读写） ─────────── 等待 relay 完成 → 消费
                             （与下面两条通道无完成依赖）
第一跳 stream：块 0 → 块 1 → [槽 0 可复用后] 块 2 → 块 3 ...
                   ↘ ready 0 ↘ ready 1        ↘ ready 2
第二跳 stream：     转发块 0 → 转发块 1 → 转发块 2 ... → done
```

三条通道使用三个不同 HCCL communicator：原 direct communicator，加两个相同 EP8 成员的
relay communicator，隔离 ADDRESS/TOKEN/DONE 通知和 CCU 资源。两条 relay stream 和通信组
在初始化时创建一次，dynamic 不会逐批重建。初始化成功不代表硬件已经验证可重叠执行。

每个 chunk 是“每条原始 relay 流至多 chunk_rows 行”的一个批次，默认每流 8 MiB（向下按行
对齐，至少一行），不是整个机器合计 8 MiB。同一 relay 的多条流可能拼成更大的单次传输。
**首版的 ready 粒度是一个 chunk 批次的 CCU AllToAllV 完成握手，仍等待该批次所有 peer；
不是某一个 packet 到达就直接触发，也不是单次融合 CCU kernel。** 但是不再等待 direct，
也不等待整条消息的所有第一跳完成；第二跳块 j 与第一跳块 j+1 没有依赖，可以重叠。
两组槽位复用由第二跳完成（含 unpack）的 event 保护，防止覆盖仍在读取的数据。
最终调用者 stream 等待第二跳末块，所以紧跟的 Torch consumer 能看到完整输出。

`--executor serial` 保留之前的两阶段串行参考实现；其 `--executor-sync device` 保留四次
全设备同步，`stream` 依靠同一 stream 的排序。pipeline 只允许 `stream + force`，不会套用
串行成本模型或悄悄回退。没有 relay 的矩阵仍只执行一次 direct。
执行期没有全设备同步；计时末尾、正确性检查和显式释放资源时会等待完成。

所有新机测试输出默认放在 `/home/l00934901/profiling/a5-ep8-heuristic.*`，
包括机器拓扑、计划、日志、八个 rank 的 JSON、汇总、比较和 TMPDIR；不采集 profiler trace。
驱动自身的系统日志目录仍由驱动管理。测试前保证参与的八卡可用；旧 native CCU 和
八卡 direct 成功只说明之前环境/用例可用，不代替本分支 NPU 验证。

## 1. 获取新分支

开发者尚未 push 时，新机器不能 fetch 到这个分支。开发机提交完成后，由用户执行 push：

```bash
# 8.146.225.22 上
cd /home/liuyuanwen/sgl-kernel-npu-discovered-path
git push origin feature/a5-ccu-8rank-heuristic-alltoallv
```

以下命令均在新机器容器 `sglang_yuanwen_old` 内执行；已有仓库和第九步安装产物可以保留：

```bash
(
set -Ee -o pipefail
cd /home/l00934901/sgl-kernel-npu
git status --short
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo '请先保存本地 tracked 修改；本步骤不会覆盖它们。' >&2
  exit 1
fi
A5_BRANCH=feature/a5-ccu-8rank-heuristic-alltoallv
git fetch origin "refs/heads/${A5_BRANCH}:refs/remotes/origin/${A5_BRANCH}"
if git show-ref --verify --quiet "refs/heads/${A5_BRANCH}"; then
  git switch "$A5_BRANCH"
else
  git switch -c "$A5_BRANCH" "origin/$A5_BRANCH"
fi
git merge --ff-only "origin/$A5_BRANCH"
)
```

## 2. CPU 回归和构建

**这次必须重新构建并安装 DeepEP wheel**：增加 C++ `Ep8RelayPlanner` 和 Python 扩展属性
`ep8_relay_planner_abi=1`。默认 native worker 会在建组前检查，旧 wheel 不会悄悄回退到 Python。
已经安装 `b3a5a5b` 或更新 route 包（`A5CcuEp8DfxAbiVersion=1`）时，route 不需要重新构建；
下方保留完整双包流程，可跳过 route 的构建命令但保留 ABI 检查。
从 `973aafb` 或更早版本升级时，DeepEP 和 route 都按完整流程构建安装。
先执行本节、再执行第 3 节、4.8 节和 4.4 节长跑。2/4 卡接口不变。

本次增加 EP8 die-1 的持久 event 通知桥，以及按原始段偏移传输 direct 前缀的接口。
worker 要求 `A5DeepEpPipelineWorkspaceAbiVersion=2`，route 要求
`A5CcuAllToAllVLayoutAbiVersion=1`、`A5CcuEp8ThreadBridgeAbiVersion=1`；缺失时在建组前退出，
不能只拉 Python 代码后继续使用旧 wheel。2/4 卡 API 与原有 ABI 版本不变。

CPU 测试包含：精确单流分配、组间下界、早停、容量/预算/阈值、非法输入、拓扑关联、
150 组随机流量独立 oracle、结果汇总拒绝缺失 rank，以及 Torch CPU 下八线程执行器验证。
Torch CPU 测试不申请 NPU；没有 Torch 时该项会明确显示 skipped。

```bash
cd /home/l00934901/sgl-kernel-npu
mkdir -p /home/l00934901/profiling/a5-ep8-cpu/tmp
export TMPDIR=/home/l00934901/profiling/a5-ep8-cpu/tmp
python3 -m unittest discover -s tests/python/deepep -p 'test_a5_ccu_heuristic_cpu.py' -v
python3 -m unittest discover -s tests/python/deepep -p 'test_a5_ccu_pipeline_cpu.py' -v
python3 -m unittest discover -s tests/python/deepep -p 'test_a5_ccu_alltoallv_cpu.py' -v
python3 -m unittest discover -s tests/python/deepep -p 'test_a5_ccu_error_check_cpu.py' -v
python3 -m unittest discover -s tests/python/deepep -p 'test_a5_ccu_thread_bridge_cpu.py' -v
python3 -m unittest discover -s tests/python/deepep -p 'test_a5_ccu_dfx_cpu.py' -v
python3 -m unittest discover -s tests/python/deepep -p 'test_a5_ccu_native_planner_cpu.py' -v
```

首次升级到包含 direct EID/die 绑定校验的版本，需重编并安装 route 自定义包；
已安装并通过下面全部 route ABI 检查时，本次仅需重编 DeepEP wheel。
旧 manifest 的 `src_eid=-,dst_eid=-` 保持原行为，新生成的 manifest 使用真实绑定，
HCCL 实际 hop=1 端点与绑定不一致时直接报 `PEER_PLAN_BOUND_DIRECT_MISMATCH`，不会猜路径。
新 worker 强制检查 `A5CcuBoundDirectAbiVersion=1`，避免旧 SO 忽略绑定。

```bash
(
set -Ee -o pipefail
cd /home/l00934901/sgl-kernel-npu
source scripts/a5_ccu_test_env.sh
a5_ccu_prepare_test_env /usr/local/Ascend/cann-9.1.T560 "$PWD"
# 与之前构建成功的环境一致；确认 cann 别名仍指向 T560。
readlink -f /usr/local/Ascend/cann
export HCCL_REPO=/home/l00934901/hccl
mkdir -p /home/l00934901/profiling
A5_BUILD_DIR=$(mktemp -d /home/l00934901/profiling/a5-heuristic-build.XXXXXX)
mkdir -p "$A5_BUILD_DIR/tmp"
export TMPDIR="$A5_BUILD_DIR/tmp"
bash scripts/build_a5_ccu_urma_route_probe.sh --install --install-path /usr/local/Ascend/cann-9.1.T560 > "$A5_BUILD_DIR/build.log" 2>&1 || {
  tail -100 "$A5_BUILD_DIR/build.log"; exit 1;
}
export A5_CCU_ROUTE_PROBE_LIB=/usr/local/Ascend/cann-9.1.T560/opp/vendors/cust/lib64/liba5_ccu_urma_route_probe.so
python3 - <<'PY'
import ctypes, os
lib = ctypes.CDLL(os.environ['A5_CCU_ROUTE_PROBE_LIB'], mode=ctypes.RTLD_GLOBAL)
for name in ('A5CcuBoundDirectAbiVersion', 'A5CcuAllToAllVAbiVersion',
             'A5CcuAllToAllVLayoutAbiVersion', 'A5CcuEp8ThreadBridgeAbiVersion',
             'A5CcuEp8DfxAbiVersion'):
    fn = getattr(lib, name)
    fn.restype = ctypes.c_int
    assert fn() == 1, name
print('BOUND_DIRECT_ALLTOALLV_LAYOUT_BRIDGE_AND_DFX_ABI_PASS')
PY
# 必须重新构建/安装 DeepEP wheel；日志和临时目录放个人 profiling 目录。
touch "$A5_BUILD_DIR/wheel-build-started"
DEEPEP_SINGLE_OP=explicit_multipath_all2all_ccu bash build.sh -a deepep Ascend950 \
  > "$A5_BUILD_DIR/deepep-build.log" 2>&1 || {
  tail -100 "$A5_BUILD_DIR/deepep-build.log"; exit 1;
}
mapfile -t A5_WHEELS < <(find "$PWD/output" -maxdepth 1 -name 'deep_ep-*.whl' \
  -newer "$A5_BUILD_DIR/wheel-build-started" -print)
if [ "${#A5_WHEELS[@]}" -ne 1 ]; then
  echo "没有唯一的新 wheel，请检查 $A5_BUILD_DIR/deepep-build.log" >&2
  exit 1
fi
python3 -m pip install --force-reinstall --no-cache-dir --no-deps "${A5_WHEELS[0]}"
echo "构建日志：$A5_BUILD_DIR"
)
```

DeepEP C++ 调用接口保持兼容，本次增加 host-only 规划绑定，必须使用新编译的扩展。
实验 worker 从 checkout 加载新 Python 模块；模型环境通过 `deep_ep.heuristic_alltoallv` 导入时，
也必须安装包含本分支 Python 模块的 wheel。
旧分支第 6 步的 ABI/Meta 预检、第 7 步的 2/4 卡旧测试仍适用。

## 3. 八卡正确性

新 runner 每次只读查询 rootinfo、Atlas 和实时 hccn EID/端口，生成 manifest 和
`plans/topology.json`；不读取手写 TXT，不要求用户编辑路径表。
目前要求每对卡只有一条可验证的 UB_CTP 物理边、每 rank 单个 local IO die、不同 peer
不复用同一物理端口；歧义/缺链/共享端口会报错。该约束对应当前独立等带宽链路模型。
实际 die 带宽、HBM 等共享瓶颈尚未纳入预测，不把理论负载当作耗时保证。

```bash
cd /home/l00934901/sgl-kernel-npu
bash scripts/run_a5_ccu_heuristic_alltoallv.sh \
  --executor pipeline --suite correctness --dtype float32 --width 257 --row-scale 1 \
  --warmup 1 --iters 2 --rounds 1 --dynamic-iters 1
```

一次依次测试 zero、self、single_hot、competing、asymmetric、uniform、cut 七个矩阵，
复用固定的三个 lane/handle，同时变化矩阵、buffer 和 payload。每个 case 校验 direct、
multipath、原生 HCCL 与独立 CPU oracle，检查输入不被修改和前后 guard。
包含空发送 rank、空接收 rank、零长度方向及独立 self 偏移。
CPU 测试还覆盖禁止路径、阈值、relay 容量和一次分流后负载变化。
小矩阵会额外自动选择更小的 chunk，尽可能形成 5 个批次，覆盖双缓冲槽多次复用；
结果记录 `pipeline_correctness_chunk_counts`。连续三次改变 payload、通信后立即 clone
输出，中间不做设备同步，检查 producer→跨 stream relay→consumer 的依赖。
CPU 异步 DAG 模拟还故意延迟 direct，让第二跳先于 direct 完成，检测错误屏障、
缺失 ready/reuse/join，以及同一个 context 连续处理变化矩阵与张量存储。

成功必须看到 `EP8_HEURISTIC_ALL_RANKS_AND_EXIT_PASS`：八个 rank 均 PASS、输出完整、
plan fingerprint 相同且 launcher 正常退出。只看到建链成功或某一个 rank PASS 不算通过。
失败保留本次目录，先看 `topology.log` 或 `run.log`，不要继续放大数据。

若新机文件路径不同，可覆盖以下环境变量；物理卡顺序也必须与输入矩阵一致：

```bash
export A5_ROOT_INFO=/etc/hccl_rootinfo.json
export A5_ATLAS_JSON=/usr/local/Ascend/driver/topo/950/atlas_950_1.json
export A5_HCCN_TOOL=/usr/local/Ascend/driver/tools/hccn_tool
export A5_DEVICES=0,1,2,3,4,5,6,7
```

## 4. 原生 HCCL baseline、custom direct 对照和多路径比较

默认 `single_hot`：`C[0,4]=32*128` 行，另有 rank 0 和 3 的 self 行。
FP16、hidden width=4096，每行 8192 B；跨卡总量 32 MiB，含 self 共 36 MiB。
默认贪心把热点流分成 16 MiB direct + 16 MiB 经一个组内 relay。
原始跨卡有效载荷仍是 32 MiB，但物理发送总量增至 48 MiB。

**主要性能 baseline 改为原生 HCCL AllToAllV。** `native` 调用
`dist.all_to_all_single(..., async_op=True)` 后 `Work.wait()`，对齐 vLLM Ascend
AllToAllV 分支的 payload API。`direct` 是我们的自定义 CCU direct-only，仅用于隔离分流收益。
原生 HCCL 的物理选路由库决定，不能把 native 等同为“所有字节都走直连”。

分别运行 native 和 multipath；参数相同，输出都放个人 profiling 目录：

```bash
bash scripts/run_a5_ccu_heuristic_alltoallv.sh --mode native --case single_hot --chunk-bytes 8388608
bash scripts/run_a5_ccu_heuristic_alltoallv.sh --mode multipath --case single_hot --chunk-bytes 8388608
# 用两次实际输出目录替换 NATIVE_RUN / MULTIPATH_RUN
python3 scripts/summarize_a5_ccu_heuristic.py --baseline native --compare \
  /home/l00934901/profiling/a5-ep8-heuristic.NATIVE_RUN \
  /home/l00934901/profiling/a5-ep8-heuristic.MULTIPATH_RUN
```

同进程三路测量用 `--mode all`，建议6轮以覆盖全部六种测量顺序：

```bash
bash scripts/run_a5_ccu_heuristic_alltoallv.sh --mode all --case single_hot --rounds 6 --chunk-bytes 8388608
```

输出同时包含 `native_over_multipath` 和 `direct_over_multipath`。
只有超过 native 才说明此通信负载下有替换收益；超过 custom direct 不足以证明。
`--mode both` 保留原来的 direct/multipath 两路比较；`--compare` 默认仍为direct以兼容旧命令，
主baseline比较必须指定 `--baseline native`。

**边界**：所有模式预分配输出，测量包含提交/完成等待，排除模型wrapper输出分配和router、
专家计算、模型permute/unpermute。此runner固定 `HCCL_OP_EXPANSION_MODE=CCU_SCHED` 并清除
`HCCL_ALGO`；结果记录这两个值、HCCL_BUFFSIZE及Torch版本。它是同配置下的原生HCCL
通信层baseline；若模型部署使用不同engine/算法配置，还必须另做部署配置下的原生对照，
不能将此结果直接称为DeepSeek 4.1端到端加速。

以下保留 custom direct 消融实验。两次使用完全相同的参数：

```bash
bash scripts/run_a5_ccu_heuristic_alltoallv.sh --mode direct --case single_hot
bash scripts/run_a5_ccu_heuristic_alltoallv.sh --mode multipath --case single_hot
```

`--mode` 只决定计时对象；每次仍会先验证 direct/multipath/native，防止跳过正确性。
prepared 和 dynamic 分别预热各自完整路径 3 次，每轮各调用 10 次，共 4 轮。
`both` 模式采用 AB/BA/AB/BA 顺序，dynamic 不再固定先 direct、后 multipath。
prepared 每批末等待完成；dynamic 每次等待完成。pipeline 工作区和 event 由 context 持有，
同一容量内重复准备只更新布局；容量增长时先等待旧工作完成再换池。
`--warmup/--iters/--dynamic-iters/--rounds` 可调整；性能比较建议偶数轮。
下面把占位路径替换为上面真实输出的目录，不要把名字原样执行：

```bash
python3 scripts/summarize_a5_ccu_heuristic.py --compare \
  /home/l00934901/profiling/a5-ep8-heuristic.DIRECT_RUN \
  /home/l00934901/profiling/a5-ep8-heuristic.MULTIPATH_RUN
```

比较工具先检查八个 rank、矩阵/plan fingerprint、dtype、width、测量设置等，再计算
`direct_over_multipath = direct_us / multipath_us`，写入第二个目录 `comparison.json`。
大于 1 是多路径更快，小于 1 则没有加速；不同输入不能直接比较。

为了降低跨运行波动，推荐再在同一个 worker 中交替顺序测量：

```bash
bash scripts/run_a5_ccu_heuristic_alltoallv.sh --mode both --case single_hot
bash scripts/run_a5_ccu_heuristic_alltoallv.sh --mode both --case asymmetric
bash scripts/run_a5_ccu_heuristic_alltoallv.sh --mode both --case competing
bash scripts/run_a5_ccu_heuristic_alltoallv.sh --mode both --case cut
```

asymmetric 对应之前第九步矩阵：默认八卡含 self 244 MiB、跨卡 214 MiB、最大原始流 10 MiB。
cut 对应 A={0,1,2,3} 向 B={4,5,6,7} 每对发送 7*128 行，验证组间瓶颈不会凭空消失。
uniform/cut/zero/self 等无 relay case 的 prepared 两个标签执行同一个 direct 调用，作为 A/A 控制。
dynamic 的 multipath 标签仍包括规划/布局开销；其比值不能解释为多路径加速。
汇总输出 `NO_RELAY_EXECUTED` 和 `execution.executed_relay=false`，不根据比值误报收益。
可用 `--row-scale 1/16/128/256` 扫描数据量，`--width` 对应每行 hidden 元素数。

汇总字段的含义：

| 字段 | 解释 |
|---|---|
| `plan.direct_peak_bytes` | 全部直连时的最大有向链路字节数 |
| `plan.planned_link_peak_bytes` | 两阶段流量累计到物理边后得到的峰值，不是运行时间 |
| `plan.cut_lower_bound_bytes` | 等带宽 full mesh、完整行切分的全分组下界 |
| `schedule.phase_peak_sum_bytes` | 仅 serial：两阶段各自峰值之和 |
| `schedule.chunks` / `chunk_bytes_per_flow` | pipeline 的批次数、每条 relay 流的分块大小 |
| `schedule.ccu_calls_per_execute` | pipeline 真分流时为 1+2×chunks；无分流为 1 |
| `plan.relay_bytes` | 需要第二跳转发的数据量 |
| `schedule.scratch_bytes_per_rank` | 当前布局的逻辑工作区；pipeline 实际池容量见各 rank 的 `memory_samples[].workspace.reserved_bytes`，包含零行 backing 和容量余量 |
| `planner_max_rank_us` | 单次 CPU 贪心规划的最慢 rank 耗时，不含计数交换/布局/分配 |
| `prepared` | 已准备计划/工作区；包含 relay pack/unpack、全部 CCU 调用、event 依赖和完成等待；pipeline direct 不再打包/回拷 |
| `dynamic` | device 上行数 all-gather、转 CPU 同步、规划、布局、工作区准备、执行与完成等待；pipeline 在容量内复用池，serial 仍逐次分配 |

prepared、dynamic 均逐轮取八 rank 最大值，再对轮求均值，同时保留每轮值、median/min/max。
`dynamic.slowest_rank_components_round_us` 记录当轮总耗时最慢 rank 的三个分项：
`counts_gather_sync_us`、`plan_workspace_us`、`execute_completion_us`。
分项不是分别取各 rank 最大值；这种做法会拼接出不存在的总耗时。host 调度/对象清理等
小量开销包含在总耗时中，因此分项和不必严格等于总时间。
测量协议为 6（原生HCCL计时及环境指纹），比较器拒绝与旧协议、不同runtime配置或不同sync/policy的结果混合做性能比。
都是 host wall-clock；不是 CCU 内核 event 时间。不包括 router、模型 pack/unpack、专家计算；
这里的 pack/unpack 专指新增 relay 执行器内部的分片整理。native已经参与性能计时，
custom direct仅作内部消融对照。native/direct动态分支没有贪心规划，但三路都测量同一套
合成8×8计数all-gather+D2H；这部分是微基准约定，不是对模型额外通信量的断言。

动态测试每次重新规划同一个输入矩阵，不缓存路由决策；只复用工作区和 event 容量；验证“变化矩阵”的正确性由第 3 步覆盖。
prepared 与 dynamic 的同步方式不同，只在同一口径内比较 direct/multipath。

旧 serial 实现特别是单热点流，尽管累计链路峰值从 32 降到 16 MiB，两阶段分别需要传 16 MiB，
加上 pack/unpack 和额外 launch，这个串行原型可能比 direct 更慢。这是有效的实验结论，
不能只根据 32→16 推导 2 倍加速。若 dynamic 无收益，真实模型应继续 direct，
流水线是否解决该瓶颈以本版复测为准，不能仅凭减少理论峰值声称性能收益。

### 4.1 旧串行劣化定位及参考对照

实测旧版 single_hot prepared 约 664→905 us，competing 820→1106 us，
asymmetric 255→1065 us；cut 无 relay 却出现 dynamic 3051→1490 us。
代码确认四次全设备同步及两阶段串行；dynamic 缺完整预热、固定测量顺序。
不能从这些数据单独量化每个因素的贡献，新版通过以下对照区分：

```bash
# 先完成第 2 节构建和第 3 节正确性。
# 同一输入，对比去掉设备同步前后；两次仍各自包含 direct baseline。
bash scripts/run_a5_ccu_heuristic_alltoallv.sh --executor serial --mode both --case single_hot --executor-sync device
bash scripts/run_a5_ccu_heuristic_alltoallv.sh --executor serial --mode both --case single_hot --executor-sync stream
bash scripts/run_a5_ccu_heuristic_alltoallv.sh --executor serial --mode both --case asymmetric --executor-sync stream
# 无分流 A/A 控制：prepared 应接近，dynamic 不用于声称多路径收益。
bash scripts/run_a5_ccu_heuristic_alltoallv.sh --executor serial --mode both --case cut --executor-sync stream
```

默认 `--multipath-policy force` 真正执行贪心产生的分流，便于观察真实性能，绝不把
热点 case 偷换成 direct 再宣称加速。serial 参考实现另提供 `--multipath-policy auto`：
根据执行结构选择；执行决策与候选路由同时输出。
当前串行执行器对任何分流都有 `max(stage1) + max(stage2) >= max(original)`：
取最大原始流 n，其直连 n-x 必须经过阶段一，转发 x 必须经过阶段二，因此两阶段峰值
之和至少为 n。此等带宽模型尚未计额外拷贝、launch 或 HBM/IO die 共享资源。
**所以 serial 的 auto 会拒绝所有候选分流、显式回退 direct。这个回退不是多路径加速。pipeline 不使用此模型。**
只有换成真正重叠的流水线/融合执行器，才应替换这一成本模型。去掉同步可以降低
额外开销，却不会消除串行两跳本身的限制；新机是否变快必须重测。

```bash
bash scripts/run_a5_ccu_heuristic_alltoallv.sh --executor serial --mode both --case single_hot --multipath-policy auto
```

即使选择 auto，正确性阶段也额外强制验证真实 relay 路径，避免回退掩盖实现错误。

### 4.2 流水线稳定性与性能复测（本次重点）

先完成第 2 节重编安装和第 3 节正确性。确认 `EP8_THREAD_BRIDGE` 中跨流 rank 为
`mode=event events=2`（同流 rank 为 `same_thread`）。首先用原失败参数复测，保留 2 MiB、
4 轮、每轮 3 次动态预热和 10 次动态测量，共 52 次动态执行；不要靠缩短迭代绕过失败：

```bash
bash scripts/run_a5_ccu_heuristic_alltoallv.sh \
  --executor pipeline --thread-bridge event --mode multipath --case single_hot --chunk-bytes 2097152 \
  --rounds 4 --warmup 3 --dynamic-iters 10
```

成功必须包含 `EP8_HEURISTIC_ALL_RANKS_AND_EXIT_PASS`。检查 `run.log` 的 `PIPELINE_MEMORY`：
固定输入的 `before_dynamic` 到每轮末，`workspace.generation`、`tensor_allocations`、
`event_allocations`、`addresses` 应不变。RSS/HBM 的实际值也被记录，观察预热后是否趋稳；
池计数稳定并不保证底层 HCCL 没有其他资源增长。若仍崩溃，优先找本次新增的
`EP8_EVENT_BRIDGE_FAILED`、`ALLTOALLV_STAGE_FAILED`、`ALLTOALLV_EXECUTE_FAILED` 和 `PIPELINE_TRANSFER_FAILED`，
它们保留第一次返回码及失败 lane/chunk。`EP8_PHASE` 带 rank、PID 和时间戳，区分建组、
correctness、prepared、dynamic 与销毁阶段，便于对齐 plog。
单独的 `GetEndpointInfo -> 6` 也可能来自可选属性查询，不能只按“最早 ERROR”认定致命原因。

通过后可进一步长跑，并验证有竞争流和无 relay 对照：

```bash
bash scripts/run_a5_ccu_heuristic_alltoallv.sh --executor pipeline --mode multipath --case single_hot --chunk-bytes 2097152 --dynamic-iters 100
bash scripts/run_a5_ccu_heuristic_alltoallv.sh --executor pipeline --mode both --case competing --chunk-bytes 8388608
bash scripts/run_a5_ccu_heuristic_alltoallv.sh --executor pipeline --mode both --case asymmetric --chunk-bytes 8388608
bash scripts/run_a5_ccu_heuristic_alltoallv.sh --executor pipeline --mode both --case cut
```

性能比较使用 8 MiB 作为起始实验配置，分别运行 baseline/direct 与真实分流，两次参数完全一致：

```bash
bash scripts/run_a5_ccu_heuristic_alltoallv.sh --executor pipeline --mode direct --case single_hot --chunk-bytes 8388608
bash scripts/run_a5_ccu_heuristic_alltoallv.sh --executor pipeline --mode multipath --case single_hot --chunk-bytes 8388608
# 用第 4 节 --compare 命令比较两个实际输出目录，均位于 /home/l00934901/profiling。
# 再做同进程交替测量：
bash scripts/run_a5_ccu_heuristic_alltoallv.sh --executor pipeline --mode both --case single_hot --chunk-bytes 8388608
```

用户提供的旧版低迭代测量中，single_hot 的 2/4/8 MiB 分块分别约 2199/1210/690 us，
direct 约 665 us。**这些数据尚未证明多路径加速。** 默认从 2 改成 8 MiB 是减少批次启动开销，
不是性能保证；2 MiB 保留作稳定性回归与负收益对照。可用 2/4/8/16 MiB 扫描，每个配置都
用相同参数分别跑 direct/multipath，不减少迭代数掩盖不稳定。

single_hot 中 direct 16 MiB，relay 16 MiB；8 MiB 分块为 2 批次、5 次 CCU 调用，
2 MiB 为 8 批次、17 次调用。当前每批都有 collective 握手、host launch、pack/unpack，
允许 direct 与两跳流水线推进，但尚不是单次融合 CCU kernel。估算出的每批开销包含
多种因素，不能单独归因某个 kernel。物理流量从 32 增到 48 MiB 也不能单独证明抢同一链路：
full mesh 的 i→j、i→k、k→j 可以是不同边，是否争用 IO die、HBM 等共享资源还需设备证据。

输出应有 `PIPELINE_LANES`（三个不同 group）和 `participant_hbm_pipelined_ccu`。
三个 group/handle 只在初始化创建一次，dynamic 不调用 `dist.new_group`。
context 持有 8 个 relay 工作区张量（双缓冲各四个），按各槽容量高水位复用；
direct 直接读写原始布局的前缀，移除原来的两个 direct 工作区及其 pack/unpack。
零行 rank 也有持久 backing，避免每次 CCU 调用创建临时占位张量。
`workspace.event_allocations` 只计 Python 分块事件；C++ 跨流桥的两个事件另由每个 bound plan 持有，
可在 `EP8_THREAD_BRIDGE events=2` 中确认，同样不随 execute 次数增长。
容量按 2 的幂增长，若超过 scratch 限额则用精确容量；所有 rank 根据同一个 C 做全局检查。
仅容量增长时同步调用者 stream、释放旧 view 并扩容；旧 prepared 再执行时绑定新池。
每个 context 固定 dtype 和行形状，容量不足不能静默改变计划或独立降级。

设备上是否真正重叠、加速多少仍以复测为准。默认不打 profiler；比较器拒绝不同协议、
executor/chunk 设置的独立结果做同配置比较。若 dynamic 无收益，模型仍应使用 direct baseline。

### 4.3 9cd5091 复测后的定位与本版变化

用户的新机报告中，2 MiB 失败点已确认是 die-1 路径的
`HcommThreadNotifyRecordOnThread(routeThread, mainThread, 0)`，日志名为 `record_route_done`，
返回 2。它是本地线程间通知，不是路由登记表，也不是 payload 指针检查。
前面的 `WaitOnThread` 成功只表示等待命令提交成功，不能证明设备传输已经完成。
失败发生在 dynamic round 1；round 0、correctness、prepared 已经过，不是首次 slot 复用就必然失败。
第一跳零行的 rank 2、3 不走这个跨流分支；因此不能仅因 rank 4–7 零行而认定占位空间不足。
底层 Hcomm 为何在重复调用后返回 2 尚无足够内部证据，不能称为已证实的登记表耗尽。

本版 EP8 默认 `--thread-bridge event`：为每个跨流 bound plan 创建两个 ACL Ex event，
caller record ready → route wait ready → CCU launch → route record done → caller wait done。
每次复用 event，提交完成 record 成功后才提交 caller wait；失败后拒绝重用该 bridge。
这些依赖保证 producer/consumer 顺序，不在执行热路径增加设备或 host 同步。
显式销毁时先等待完成再释放 event。2/4 卡仍保留原通知路径。
所有 rank 继续参加每个 CCU collective，未跳过零行 rank 的地址/完成握手。
`--thread-bridge hcomm` 保留旧路径用于定位对照，已知可能复现原错误，不是推荐测量配置。
比较工具要求两次独立运行的 thread-bridge 相同。

性能修改包括：

- direct 使用原始 C 的 source/receive 偏移，只传每个 peer 段的 direct 前缀；relay 写不重叠的 suffix。
  移除整段 direct 的 pack/unpack 及两块工作区，不通过回退 direct 掩盖多路径代价。
- 每个 chunk 的各 rank 尺寸、双缓冲最大尺寸只计算一次；不再生成并丢弃一份串行布局。
- 贪心按当前边负载降序检查：已经找到可行候选后，更低负载不可能胜出，停止该轮扫描。
  同负载仍完整检查，以保留原 `(load,gain,src,dst,relay)` 的确定性优先规则。
  没有缓存流量矩阵，真实 C 每次变化都重新规划；整数参数检查保留非法输入拒绝规则。

`9cd5091` 干净环境下 single_hot 的 prepared 662.3→582.9 us 是该配置的实测收益，
但 dynamic 1166.3→2487.9 us 仍负收益；competing/asymmetric 的 prepared 也仍负收益。
本版必须重测。不能把旧 prepared 改善全部归因 dynamic 少分配：prepared 本来已经复用工作区。
“累计物理字节/总耗时”是聚合吞吐指标，不能当作某条链路利用率或直接证明硬件执行重叠。

稳定性通过后，可以扫描分块，所有输出仍在个人 profiling 目录：

```bash
for A5_CHUNK in 1048576 2097152 4194304 8388608 16777216; do
  bash scripts/run_a5_ccu_heuristic_alltoallv.sh \
    --executor pipeline --thread-bridge event --mode both --case single_hot \
    --chunk-bytes "$A5_CHUNK"
done
```

16 MiB 对 single_hot 是一个 relay 批次：两跳之间无分块重叠，但 direct 仍可与 relay 同时推进，
不等价于整个 serial 执行器。正在被其他任务使用的卡上采到的异常/耗时不能用于对照结论。
若新 bridge 仍失败，保留当前 run.log 和 plog，优先看 `EP8_EVENT_BRIDGE_FAILED` 的原始 ACL 返回码；
若仍看到 `record_route_done`，先确认此次实际选择是否为旧的 `--thread-bridge hcomm`。

### 4.4 973aafb 长跑失败：缺少 DFX 算子注册（2026-10-29）

新机报告确认 `973aafb` 的七个正确性 case、2 MiB / 52 次 dynamic 三次重复通过；
`--dynamic-iters 100` 在 round 1 失败。真正的返回链为：

```text
ReportHcclTaskDetails: taskInfo.dfxOpInfo_->comm_ is nullptr
  -> AddTaskInfoCallback -> SaveDfxTaskInfo
  -> HcclReportCcuProfilingInfo -> HcclCcuKernelLaunch: status=2
```

本仓库此前直接调用 CCU launch，漏掉了原生算子执行前的
`HcclDfxRegOpInfoByCommId`。两个 relay 通信组/handle 在 context 初始化时创建一次，
dynamic `prepare()` 仅重新构建 Python 布局，不销毁/重建通信域。
公开 HCOMM 源码显示注册接口会创建本次 DFX 信息并填入对应 `collComm`，
缺少注册时 mirror task manager 中的默认信息可以一直带有空 `comm_`。
任务上报可能延后至队列回调，解释了“前面调用通过、之后集中失败”。

参考源码快照 `9c43030e1a02c00ce82f9f5b66845cfe8f9f91bc`（用于核对机制，非声称与 T560 二进制完全一致）：

- [注册接口填入 comm_](https://github.com/hicann/hcomm/blob/9c43030e1a02c00ce82f9f5b66845cfe8f9f91bc/src/base_comm/primitives/api_c_adpt/cpu_primitives_c_adpt.cc#L740)
- [任务队列满时触发回调](https://github.com/hicann/hcomm/blob/9c43030e1a02c00ce82f9f5b66845cfe8f9f91bc/src/legacy/ascend950/framework/dfx/common/mirror_task_manager.cc#L43)
- [检查 comm_ 早于 profiling 开关判断](https://github.com/hicann/hcomm/blob/9c43030e1a02c00ce82f9f5b66845cfe8f9f91bc/src/legacy/ascend950/framework/dfx/profiling/profiling_handler.cc#L150)

该快照 circular queue 上限确为 2048，但队列按 stream 管理，且一个 CCU API 调用可能
产生多个底层 task；**不能用三个 lane 的合计调用数直接断言精确失败阈值为 2048**。
关闭 profiler trace 不会跳过上面的空指针检查，不提供未经核实的 DFX_DISABLE 环境变量。
开发机 9.1.0-beta.1 的 `libhcomm.so` 反汇编显示 `rtCCULaunch` 在 profiling 上报前调用。
因此上报失败不能证明 kernel 尚未提交，更不能吞掉错误或重试同一传输。

本版对每次 EP8 调用（包括 direct baseline 和零行 rank）注册有效的本次 DFX 信息，
使用安装 SDK 的 `HcclDfxOpInfo`，通过真实 communicator 名称调用官方注册接口。
注册失败则不提交本次 stream/event/kernel；成功后只提交一次。每次重新注册，以免
同一 communicator 上原生 collective 改写当前 op 信息。无需每轮重建通信域或扩大队列。
`EP8_DFX ... stage=registered status=0` 应在每个 lane 首次执行时出现。
失败时打印该 plan 的尝试序号、group、handle、stage 和首次错误；序号不是底层 task 数。
`event_route_done` 以前还会重复打印 launch 的错误，本版消除这个重复阶段误导。

DFX修复提交 `b3a5a5b` 的 benchmark protocol 为5；native性能baseline版本为6，
当前C++规划版为 **7**。必须重新获取比较数据，比较工具拒绝跨protocol或不同planner的结果。
DFX 注册也有 host 成本，本修复不承诺性能提升。

复测顺序：第 2 节安装新 route、第 3 节正确性，然后执行第 4.2 节的 2 MiB 短测及下面长测：

```bash
cd /home/l00934901/sgl-kernel-npu
bash scripts/run_a5_ccu_heuristic_alltoallv.sh \
  --executor pipeline --thread-bridge event --mode multipath --case single_hot \
  --chunk-bytes 2097152 --rounds 4 --warmup 3 --dynamic-iters 100
```

这是 4×(3+100)=412 次 dynamic execute，不是总计 100 次；还包括正确性/prepared 调用。
应完整通过所有轮次并出现 `EP8_HEURISTIC_ALL_RANKS_AND_EXIT_PASS`，同时检查 plog 中
不再出现 `dfxOpInfo_->comm_ is nullptr`（含退出时的清理阶段）。之后再按第 4.2 节重测
8 MiB direct/multipath、both 和其他 case；结果仍写 `/home/l00934901/profiling`。
新机运行结果尚待回传，CPU 模型及编译通过不等于长跑已修复。

### 4.5 8 MiB 单热点收益预期与已测边界

32 MiB 单热点对半分流时，direct 为 16 MiB，relay 为两个 8 MiB 块。
在三条有向边独立且带宽均为 B、没有提交/拷贝/握手开销的假设下，
direct 用时 16/B，两跳流水用时 (2+1)×8/B=24/B，整体为 24/B；纯直传为 32/B。
因此该方案理想耗时降低 **25%**（加速比 1.33×）；20% 可以作为目标，不是实测保证。
以 direct 666 µs 为例，1.20× 对应 555 µs；耗时降低20%对应533 µs。
不能把666 µs全部当作链路传输时间来精确预测最终耗时。

`973aafb` 的8 MiB多次实测 multipath约566–705 µs，direct约664–670 µs：
中位数593 µs对应约11%耗时降低、1.13×加速；有退化到不加速的运行。
competing约0.97×、asymmetric约0.35×，dynamic仍全面负收益。
这些是历史实测，不应写成新版本稳定1.2×或推荐模型直接启用。
目前每次8 MiB执行需5次CCU collective，并保留relay pack/forward/unpack和全peer块完成握手。
后续优化应分解这些开销及跨rank提交偏斜；总物理字节增加50%本身不证明独立链路不能获益。

### 4.6 100次测试及资源复用的准确含义

`--dynamic-iters 100` 使用默认 `--rounds 4 --warmup 3`，表示每个被测模式执行
4×(3+100)=412次dynamic调用；此前还执行正确性，以及4×(3+10)=52次prepared调用。
它不是总共100次，更不是建100次通信域。`--rounds 1 --dynamic-iters 100` 才是一轮
100次计时调用（另有3次预热、正确性和prepared阶段）。`--mode all` 对三个模式分别计数。

| 对象 | 生命周期 |
|---|---|
| direct及两个relay HCCL通信域 | context初始化时创建，测试结束时销毁 |
| channel/commlink、CCU thread和kernel handle | 按通信域及绑定stream准备一次，计时循环复用 |
| relay工作区和event | context高水位池复用，仅容量增长时扩容 |
| Python路由plan与offset布局 | prepared复用；dynamic按本批C重算，不重新注册CCU kernel |
| `HcclDfxRegOpInfoByCommId` | 每次CCU调用更新本次op元数据，不是建链或kernel编译/注册 |

DFX注册不仅填comm，也产生本次op信息；原生collective可能替换同通信域当前信息，
且dynamic的字节数/地址可以变化，因此当前遵循原生HCCL每次执行前注册的契约。
不能用“一次非空comm”替代每次算子元数据的生命周期。若未来测得该调用成本显著，
应验证SDK支持的预绑定/fast-launch接口后再优化，不能直接移除刚补的注册。

### 4.7 DeepSeek 4.1目标调用链（源码核对，尚未接入模型）

核对仓库 `/home/liuyuanwen/vllm-ascend-dsv41`，提交
`56debc194a42d473f8c456d4850369442d694c8f`：

```text
vllm_ascend/models/deepseek_v41/model.py:236 FusedMoEFactory
  -> 已选中的 ALLTOALL 通信实现
  -> ops/fused_moe/moe_comm_method.py:263 AlltoAllCommImpl
  -> ops/fused_moe/token_dispatcher.py:442 TokenDispatcherWithAll2AllV
  -> ops/fused_moe/moe_utils.py:66 async_all_to_all
  -> dist.all_to_all_single(..., input_split_sizes, output_split_sizes, async_op=True)
  -> Work.wait()
```

Dispatch发送排好序的hidden rows；combine交换input/output splits，返回源rank。
量化dispatch还会用同样splits交换dynamic_scale。新算子应替换这段payload通信，
保留原token排序、专家计算和加权还原；实际row_bytes要取量化后的dtype/形状，不能固定8192。
combine的计数方向为dispatch的转置；scale与activation的row_bytes不同，不能无条件复用切片计划。

A5采用 `CAPACITY_AND_WORLD_SIZE` 选择策略（hardware_profile.py:346）：
启用且支持的FUSED_MC2可优先走MegaMoE；小token批次可能走MC2；较大批次若
`world_size_across_dp <= num_experts_per_tok` 会选ALLGATHER，否则才是ALLTOALL。
还有EP关闭、LoRA等分支。因此“DeepSeek4.1+prefill”本身不能证明选中了AllToAllV，
最终需核对部署TP/DP/EP、top-k、fused配置与实际MoECommType。当前不修改模型分支选择。

`TokenDispatcherWithAll2AllV._preprocess`（token_dispatcher.py:616）已经all-gather
各源rank的expert计数，得到E[source, expert]。按现有连续expert分组，EP8可直接计算：

```python
# 已有全局expert计数，不另发一次all-gather；L为每rank的local expert数
C = E.reshape(8, 8, L).sum(dim=2)
input_splits = C[rank, :]
output_splits = C[:, rank]
```

未来对接应复用这份计数及已有D2H边界，再传给`make_plan`；当前dynamic微基准独立生成
device行计数并all-gather，不能把它的全部开销当成模型必然新增开销。
router已知不等于通信计划可永久复用：每批C变化时仍要更新长度/offset/分流方案，
通信域、channel、kernel和工作区则继续复用。

### 4.8 C++ 规划/布局优化与 dynamic 分项复测（2026-10-29）

用户的七场景扫描中，single_hot prepared 为 native 666.1 µs / multipath 598.3 µs，
dynamic 则为1068.7 / 1861.6 µs；asymmetric dynamic为674.1 / 2809.3 µs。
不能把dynamic总耗时称作规划耗时，也不能把双方都有的计数gather/D2H全算作多路径额外成本。
本版的变更为：

1. `RelayPlanner` 在初始化时校验拓扑和选路选项，C++以64个relay位掩码保存；每批仅输入C。
2. C++固定数组扫描代替Python每轮列表构建、排序、生成器和候选tuple；严格保留
   `(当前边负载, 局部峰值收益, -源rank, -目的rank, -relay)` 的决策与整数行对齐。
3. 在C++计算direct/两跳偏移、分块pieces、八rank行数和双缓冲容量；Python只包装返回值。
4. 工作区容量充足时跳过高水位容量列表的重复构造；通信域/channel/kernel仍只初始化一次。
5. 默认不缓存C、不裁剪relay动作、不改变min_gain阈值，保持真实变化路由语义；
   `--planner python` 是参考实现。无relay、容量、非对称长度、self偏移、零长度规则保留。

**这次不改变数据面**：pack/forward/unpack、每块collective及event依赖仍然存在。
它主要针对dynamic规划开销，不声称解决asymmetric prepared劣化，也不保证小流量分流有收益。

完成第2节wheel重装和第3节正确性后，可先跑不占卡的CPU对照：

```bash
cd /home/l00934901/sgl-kernel-npu
python3 scripts/benchmark_a5_ep8_planner.py \
  --output /home/l00934901/profiling/a5-ep8-planner.json --iters 500
```

该脚本编译与生产扩展相同的host-only绑定，不加载NPU后端。交替输入两个C，
测试贪心、布局和CPU张量视图准备；zero自然不变。`layout_us`已包含于`prepare_cpu_rank0_us`，
不能再相加。结果不能当成NPU端到端加速比。

然后以相同参数执行两种规划器的全场景扫描（默认8MiB分块）：

```bash
cd /home/l00934901/sgl-kernel-npu
for A5_PLANNER in python native; do
  for A5_CASE in single_hot competing asymmetric cut uniform self zero; do
    bash scripts/run_a5_ccu_heuristic_alltoallv.sh \
      --executor pipeline --planner "$A5_PLANNER" --mode all --case "$A5_CASE" \
      --rounds 6 --chunk-bytes 8388608
  done
done
```

每个native规划用例会在计时前逐rank打印`EP8_PLANNER_EQUIVALENCE_PASS`，
核对现场拓扑上的C++与Python动作及负载完全一致，然后执行原有三种通信实现的正确性检查。
最终必须有`EP8_HEURISTIC_ALL_RANKS_AND_EXIT_PASS`。之后执行4.4节100次长跑。

在`results/rank*.json`和summary的dynamic分项中读取：

- `counts_gather_sync_us`：双方都有的计数获取和D2H；
- `planner_us`：当前C的选路；native/direct模式为0；
- `layout_workspace_us`：选路之后的布局、参数/视图准备；
- `plan_workspace_us = planner_us + layout_workspace_us`，是父项，勿重复累计；
- `execute_completion_us`：提交并等待完成，包含跨rank到达偏斜，不是纯链路时间。

summary保留每轮总耗时最慢rank的各分项，不把不同rank的独立最大值相加。
不同planner的扫描用于分项分析；`--compare`仍只接受相同planner/protocol。
正式native HCCL与multipath比较，两次独立运行均使用`--planner native`，按第4节
`--baseline native --compare NATIVE_DIR MULTIPATH_DIR`执行。所有结果在个人profiling目录。

DP8/TP1的N是每个DP实例当前进入该MoE的token数，不再除以8；它随chunked prefill、
调度负载、prefix cache命中而变。模型配置H/K尚未确认时，只能作条件估算：
H=4096、K=6、N=1024、BF16对应每源rank打包48MiB，均匀时每个目的rank6MiB，
其中跨卡42MiB；W8A8 activation dispatch减半，另交换scale。6MiB不是整组或每卡总量。
模型通信分支也需确认确实为ALLTOALL，而不是MC2/FUSED_MC2/ALLGATHER。

## 5. 输入真实路由矩阵

外部矩阵文件格式如下，`counts` 必须是实际 dispatch 打包后的行数；也支持扁平 64 项：

```json
{
  "devices": [0,1,2,3,4,5,6,7],
  "counts": [
    [0,0,0,0,4096,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0]
  ]
}
```

```bash
bash scripts/run_a5_ccu_heuristic_alltoallv.sh --mode both \
  --matrix-json /home/l00934901/profiling/router_counts.json \
  --dtype bfloat16 --width 4096
```

外部 counts 不会再乘 `--row-scale`，设备顺序、row_bytes、dtype 和实际打包必须一致。
计数矩阵只有 64 个整数；不能把 token payload、历史专家频率或 topk_weights 当作 counts。
同一 token 的多个专家位于同一 EP rank 时，必须采用当前 dispatch 实际去重/复制规则，
不是无条件对 top-k 索引做 histogram。

未来模型调用的接口（伪调用环境需要已有 EP8 Buffer、绑定 manifest 和张量）：

```python
import deep_ep
from deep_ep.heuristic_alltoallv import RelayPlanner, gather_routing_counts
from deep_ep.pipelined_alltoallv import PipelineContext, PreparedPipelinedAllToAllV

# 模型初始化：同一 EP8 的成员以相同顺序建立两组 relay communicator。
# ep_world_ranks 是 WORLD rank，必须与原 EP group 顺序一致，不是物理设备 ID。
context = PipelineContext.create(buffer, handle, manifest,
                                 group_ranks=ep_world_ranks)
planner = RelayPlanner(row_bytes, topology["allowed_relays"], max_actions=56,
                       native_module=deep_ep.deep_ep_cpp)

# 每批：local_counts 是实际 packed 行数，device int64[8]。
# 若已持有一致的全局 host C，可直接传入，避免重复交换。
C = gather_routing_counts(local_counts, group=ep_group)
plan = planner.plan(C)
prepared = PreparedPipelinedAllToAllV(context, plan, packed_send,
                                     chunk_bytes=8 << 20, max_scratch_bytes=1 << 30)
prepared.execute(packed_send, packed_recv)
# packed_recv 的后续使用必须在 context 的调用者 stream 上，或显式建立 event 依赖。
# context 必须存活到执行完成；下一批复用工作区容量，C 变化仍需重新准备布局。

```

`planner.plan`和参考`make_plan`只接受host整数，显式拒绝隐式读取GPU tensor。
planner只保存静态拓扑、row_bytes和选路选项；模型逐批传入真实C，不复用旧C对应的路由。
`gather_routing_counts` 是同步参考适配器，包含 all-gather 和 D2H 等待；其开销已在 dynamic 计时中体现。
全部 rank 必须使用相同矩阵、候选路径、chunk 大小/上限和参数；模型在进入通信前必须建立这一一致性。
测试 worker 有配置一致性检查，模型不能依靠某个 rank 自己更改阈值或自动回退。
输入按目的 rank 排列，输出按原始源 rank 排列，relay suffix 最终写回该源对应区间；
self 单独从原输入偏移复制到正确接收偏移。算子不修改专家权重和 token 恢复映射。

计划和 workspace 对应当前矩阵，矩阵变化必须重新准备；固定计划可重复执行变化的 payload。
context/对象只允许同一调用者 stream 有序复用，不支持多个 host 调用者并发；销毁/换工作区前保证先前执行完成。
现阶段是 eager 接口，未支持图捕获或动态 token 数的图重放，也未修改 vLLM-Ascend。

## 6. 贪心选择和停止条件

负载使用整数行。对候选 i→k→j，a=L[i,j]，h=max(L[i,k],L[k,j])：
`x=min(C[i,j], max(0,(a-h)//2), relay剩余容量行数)`。
只有完整行分片达到 min_chunk、局部峰值严格下降且达到 min_gain 才进入候选。
每轮优先当前直连边负载最大者，再按改善量最大者，最后按 src/dst/relay rank 打破平局。
选中后同时更新三条边的总负载，并标记原始流已分流；绝不二次转发已转发片段。
没有候选就提前退出，最多 56 个动作。最坏 9,576 次候选评估，不做联合 exhaustive search。
全分组 cut 下界只作报告，不参与反复目标搜索。

默认 min_chunk=1 字节实际向上对齐一行，min_gain=0 但要求严格正收益，适合正确性与初始实验。
性能实验可以增加 `--min-chunk-bytes`、`--min-gain-bytes`，或设置 `--max-actions 4/8/16/56`；
`--relay-capacity-mib` 限制每张 relay 承担的转发有效载荷，`--max-scratch-mib` 限制工作区。
超过工作区上限会在数据通信前报错，不允许各 rank 独立选择不同方案。

## 7. 开发侧验证记录（2026-10-28）

以下保留各历史版本记录；本次提交对应末尾“本版开发验证：通知桥与 direct 原偏移”。

在 8.146.225.22 的开发 checkout 和 `cam_lyw_dev_91` 完成：

- 112 项 CPU 回归通过，宿主机缺 Torch 而跳过的八线程执行器测试在容器内单独通过；
  新增测试文件共 9 项，包括 150 组随机矩阵和真实执行器的 CPU tensor 并发模拟。
- route 自定义包与 deep_ep_cpp 编译成功。开发 CANN 是 9.1.0-beta.1，
  不把它等同于新机器 T560 上的运行验证；新增 bound-direct ABI 符号已导出。
- `ALLTOALLV_META_PASS valid=160 invalid=12 legacy=10`；Python/Shell 语法和 diff 检查通过。
- 未在新机器执行本分支 NPU 测试，尚无多路径加速结论，未 push 到 origin。

开发构建/回归日志：`/home/liuyuanwen/profiling/a5-heuristic-build/`（开发机账号目录）。
新机器账号目录是 `/home/l00934901`，runner 默认不会使用开发机路径或 `/tmp` 存结果。


### 历史记录：串行性能修正

本次修正只涉及 Python 执行器、CPU/NPU 测试、统计和文档；2/4 卡及 C++ 算子接口不变。
新增随机矩阵串行成本下界验证、相邻拷贝边界、v2 逐轮最慢 rank 汇总，以及 stream/device、
force/auto 两组开关的 CPU 八线程数据正确性回归。新机 NPU 排序/性能结果仍待第 3、4.1 节复测。

修正后开发侧：115 项 CPU 回归无失败（宿主机缺 Torch 跳过 1 项）；容器内 12 项新增模块回归全部通过，补齐实际执行器八线程测试。Python 语法和 diff 检查通过。未 push；无新机修正后的 NPU 耗时数据。


### 历史记录：4ebb7e3 首版流水线

本次新增 `python/deep_ep/deep_ep/pipelined_alltoallv.py`；worker 默认 pipeline，旧 serial 可选。
保留原始 C 输入接口、组内单 relay 贪心以及 2/4 卡旧接口。没有改动 C++ ABI。
验证包括布局独立 oracle、100 组随机矩阵、非整除分块、零行、自拷贝、容量限制，
以及真实 Torch CPU 张量的异步 DAG 执行（延迟 direct、交错两跳、复用槽位、反复修改
payload、变化矩阵与 storage）。CPU 模拟不替代目标机 CCU 的多 communicator/stream 验证。
新机尚未执行本次流水线版本，不预先报告 NPU 正确性通过或性能收益。

开发侧本次验证：119 项 CPU 回归无失败（3 项依赖 Torch 的测试在宿主机跳过）；
`cam_lyw_dev_91` 内流水线 4 项、原贪心执行器 12 项全部通过，补齐这 3 项。
日志：开发机 `/home/liuyuanwen/profiling/a5-pipeline/`；新机输出仍为 `/home/l00934901/profiling`。

### 历史记录：9cd5091 工作区复用（2026-10-28）

目标机 `4ebb7e3` 的第 3 节 7×8 rank 正确性已由用户验证；第 4.2 节真实 relay 的 dynamic
在默认参数下崩溃，不能因小规模 correctness 通过就声称长期稳定。
本次确认错误宏在失败时重复求值，可能重复提交 execute 并把错误码误报为 0；已改为只调用一次。
工作区/event 改由 context 复用，并修复零行 view 的 bridge backing 使用；这些措施减少地址创建与
资源生命周期抖动，但第一次底层失败的原因仍需新日志验证，不能预先声称已证实为资源耗尽。

新增 CPU 回归会编译实际错误宏，验证“第一次返回 6、第二次成功”时只调用一次且保留 6；
另用 64 次变化矩阵和 payload 的八 rank 异步执行，检查输出正确、零行 backing 存在，且工作区
地址、张量分配数和 event 数在容量内不增长。容量增长和原双缓冲依赖也继续检查。
开发容器完整 CPU 回归 122 项全部通过、无跳过，包括实际 Torch CPU 执行器和宏编译测试。
日志见 `/home/liuyuanwen/profiling/a5-pipeline-stability/container-cpu.log`。
route 自定义包与 deep_ep_cpp 已在开发容器 CANN 9.1.0-beta.1 编译成功；目标机 T560 上
默认 52 次/长跑仍须按第 4.2 节复测，不能把开发编译等同于硬件修复已验证。
开发日志：`/home/liuyuanwen/profiling/a5-pipeline-stability/`。未 push。

### 历史记录：973aafb 通知桥与 direct 原偏移

开发容器 CPU 全套回归 124 项通过，无跳过；route 和 DeepEP 扩展编译成功。
最终整数校验快路径另通过 13 项规划/执行回归；编译产物 ABI 验证通过，
`ALLTOALLV_META_PASS valid=160 invalid=12 legacy=10` 保持通过。
实际 CCU 图的 CPU 调度模型验证原始布局下部分前缀传输、未传 suffix 不改写、非对称/self 偏移与 guard。
新增异步 ACL 模型编译运行生产通知桥，连续排队 1024 次只创建两个 event，并验证失败 record
不会留下未发布通知的 caller wait；100 组矩阵与完整候选扫描 oracle 得到相同选路和分流量。
以上不替代目标机 T560 的长跑和性能验证。

开发机纯 CPU 中位耗时（各 20 次预热、100 次测量；不含 NPU 通信/D2H）：

| case | 规划 old→new（us） | 布局/工作区准备 old→new（us） |
|---|---:|---:|
| single_hot | 474→198 | 776→331 |
| competing | 557→236 | 1011→447 |
| asymmetric | 4792→1128 | 912→388 |
| cut | 527→257 | 193→97 |

这些是开发机 CPU 微基准，不能直接作为新机 dynamic 延迟或加速比。
本版仍含 device counts 收集、D2H 和 Python 规划，尚未解决模型接入的全部规划开销。
日志：开发机 `/home/liuyuanwen/profiling/a5-pipeline-v4/`；新机输出继续在 `/home/l00934901/profiling`。

### 历史记录：b3a5a5b EP8 DFX 注册（2026-10-29）

- `cam_lyw_dev_91` CPU 回归：125 项通过，无 skipped。
- 新增测试直接编译生产 `ep8_dfx.h`：模拟旧路径到延迟上报时失败；修复路径三个 lane
  各执行9000次、跨过多次2048项模型队列回调，检查元数据被外部调用替换后重新注册、
  零payload、注册失败不提交、提交失败仅调用一次，以及新旧两种SDK公开结构体字段。
  该测试只验证调用协议，不是T560二进制或NPU长测替代品。
- route 包在开发机 CANN 9.1.0-beta.1 构建成功，动态加载并校验
  `A5CcuEp8DfxAbiVersion=1`、layout=1、thread-bridge=1。
- Python语法、指导文档17个bash代码块语法及 `git diff --check` 通过。
- 没有修改DeepEP二进制代码；workspace ABI仍为2。本次没有NPU稳定性/性能实测。
- 开发机日志：`/home/liuyuanwen/profiling/a5-pipeline-v5/{cpu-tests.log,route-build.log}`。
  新机输出仍为 `/home/l00934901/profiling`。

### 历史开发验证：原生 HCCL baseline（2026-10-29）

- 只修改测试、汇总与文档，未修改vLLM Ascend模型仓库、route或DeepEP二进制。
- 开发容器14项planner/执行器/汇总CPU回归通过；新增三路对比测试确保“胜过custom direct、
  但慢于native”不被报告为替换收益，并拒绝缺失native计时和runtime配置不一致的输入。
- worker参数入口、Python语法、文档19个bash代码块及 `git diff --check` 通过。
- 新的native/all测量尚未在目标NPU运行，未产生新的性能或长跑稳定性结论。
- 开发日志：`/home/liuyuanwen/profiling/a5-pipeline-v5/native-baseline-tests.log`。

### 本版开发验证：C++ 规划与布局（2026-10-29）

- 开发容器实际名称是`cam_lyw_dev_91`，CANN 9.1.0-beta.1，Python 3.11 / Torch 2.8。
- 131项CPU回归全部通过，包括1000组随机矩阵、候选子集、容量、阈值、预算、零/self、
  整数上界、逐项layout对比；同一个planner连续处理200个变化C；native布局通过
  流水线依赖、排队payload、64次变化矩阵工作区复用及扩容后旧对象重绑定测试。
- DeepEP实际扩展`cmake --build build --target deep_ep_cpp -j4`编译成功；动态加载该SO，
  检查`ep8_relay_planner_abi=1`并对100个变化C及布局与Python核对通过。
- host-only对照使用生产绑定头文件、500次迭代、交替C、全mesh六候选relay、8MiB分块。
  两个实现共享本版工作区快速检查；不缓存路由。下表为开发机CPU，非新机NPU计时。

| case | Python选路 µs | C++选路（含Python边界）µs | Python选路+CPU准备 µs | C++选路+CPU准备 µs |
|---|---:|---:|---:|---:|
| zero | 186.2 | 17.4 | 275.0 | 44.7 |
| single_hot | 197.1 | 17.8 | 476.9 | 129.4 |
| competing | 228.2 | 19.1 | 598.6 | 161.3 |
| asymmetric | 1068.7 | 26.7 | 1342.8 | 163.5 |
| uniform | 453.2 | 19.2 | 552.8 | 61.4 |

CPU准备包括layout、池检查、rank0张量视图构造，不包括数据传输。
原生规划的候选评估顺序与Python不同，asymmetric为1002 vs 630次，但31个动作、
每个分流量和所有偏移完全相同。无需为减少诊断计数重新引入Python排序。
剩余Python视图/拷贝描述构建仍有开销，不能据此保证dynamic比native HCCL快。
新机NPU全场景和100次长测尚未执行；route包没有改动，2/4卡接口不变。

开发机日志和CPU数据：`/home/l00934901/profiling/a5-ep8-planner-dev/`
下的`cpu-tests.log`、`deepep-build.log`、`timing.json`、`timing.log`。
