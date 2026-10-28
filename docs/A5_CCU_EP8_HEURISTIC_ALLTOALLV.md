# EP8 单 relay 贪心 AllToAllV：接口、正确性和性能对比

分支：`feature/a5-ccu-8rank-heuristic-alltoallv`，基于已通过新机测试的
`feature/a5-ccu-8rank-direct-alltoallv` / `d47d7b6`。
新算法仅用于八卡；relay 只能是同一通信组内、且不同于源和目的的 rank。
2/4 卡等长多路径 API、组外 relay 规则及原执行路径保留，不使用这个规划器。

本版默认 `--executor pipeline`，执行 **direct 独立推进 + relay 分块双缓冲流水线**。
候选 relay 仍由原来的单 relay 贪心算法选择，输入仍是本批实际 packed 行数 C[8][8]。
新增实现使用现有 CCU direct AllToAllV API，不改 2/4 卡接口，不用组外 relay。

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

**从 `9cd5091` 升级也必须重编并安装 DeepEP 扩展和 route 包，再执行第 3、4 节。**
本次增加 EP8 die-1 的持久 event 通知桥，以及按原始段偏移传输 direct 前缀的接口。
worker 要求 `A5DeepEpPipelineWorkspaceAbiVersion=2`，route 要求
`A5CcuAllToAllVLayoutAbiVersion=1`、`A5CcuEp8ThreadBridgeAbiVersion=1`；缺失时在建组前退出，
不能只拉 Python 代码后继续使用旧 wheel。2/4 卡 API 与原有 ABI 版本不变。

CPU 测试包含：精确单流分配、组间下界、早停、容量/预算/阈值、非法输入、拓扑关联、
150 组随机流量独立 oracle、结果汇总拒绝缺失 rank，以及 Torch CPU 下八线程执行器验证。
Torch CPU 测试不申请 NPU；没有 Torch 时该项会明确显示 skipped。

```bash
cd /home/l00934901/sgl-kernel-npu
python3 -m unittest discover -s tests/python/deepep -p 'test_a5_ccu_heuristic_cpu.py' -v
python3 -m unittest discover -s tests/python/deepep -p 'test_a5_ccu_pipeline_cpu.py' -v
python3 -m unittest discover -s tests/python/deepep -p 'test_a5_ccu_alltoallv_cpu.py' -v
python3 -m unittest discover -s tests/python/deepep -p 'test_a5_ccu_error_check_cpu.py' -v
python3 -m unittest discover -s tests/python/deepep -p 'test_a5_ccu_thread_bridge_cpu.py' -v
```

本分支新增可选的 direct EID/die 绑定校验，**需要重编并安装 route 自定义包**。
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
             'A5CcuAllToAllVLayoutAbiVersion', 'A5CcuEp8ThreadBridgeAbiVersion'):
    fn = getattr(lib, name)
    fn.restype = ctypes.c_int
    assert fn() == 1, name
print('BOUND_DIRECT_ALLTOALLV_LAYOUT_AND_BRIDGE_ABI_PASS')
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

DeepEP C++ 调用接口保持兼容，但本次增加了按原始布局传输的接口，必须使用新编译的扩展。
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

## 4. direct 和多路径分别计时，再比较

默认 `single_hot`：`C[0,4]=32*128` 行，另有 rank 0 和 3 的 self 行。
FP16、hidden width=4096，每行 8192 B；跨卡总量 32 MiB，含 self 共 36 MiB。
默认贪心把热点流分成 16 MiB direct + 16 MiB 经一个组内 relay。
原始跨卡有效载荷仍是 32 MiB，但物理发送总量增至 48 MiB。

分别执行，两次使用完全相同的参数。保存终端打印的两个结果目录：

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
测量协议升级为 4（原偏移 direct、通知桥选择与池化工作区），比较器拒绝与旧协议或不同 sync/policy 的结果混合做性能比。
都是 host wall-clock；不是 CCU 内核 event 时间。不包括 router、模型 pack/unpack、专家计算；
这里的 pack/unpack 专指新增 relay 执行器内部的分片整理。原生 HCCL 只作正确性参考；
主要性能 baseline 是同一个 CCU direct-only AllToAllV。

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
from deep_ep.heuristic_alltoallv import make_plan, gather_routing_counts
from deep_ep.pipelined_alltoallv import PipelineContext, PreparedPipelinedAllToAllV

# 模型初始化：同一 EP8 的成员以相同顺序建立两组 relay communicator。
# ep_world_ranks 是 WORLD rank，必须与原 EP group 顺序一致，不是物理设备 ID。
context = PipelineContext.create(buffer, handle, manifest,
                                 group_ranks=ep_world_ranks)

# 每批：local_counts 是实际 packed 行数，device int64[8]。
# 若已持有一致的全局 host C，可直接传入，避免重复交换。
C = gather_routing_counts(local_counts, group=ep_group)
plan = make_plan(C, row_bytes, topology["allowed_relays"], max_actions=56)
prepared = PreparedPipelinedAllToAllV(context, plan, packed_send,
                                     chunk_bytes=8 << 20, max_scratch_bytes=1 << 30)
prepared.execute(packed_send, packed_recv)
# packed_recv 的后续使用必须在 context 的调用者 stream 上，或显式建立 event 依赖。
# context 必须存活到执行完成；下一批复用工作区容量，C 变化仍需重新准备布局。

```

`make_plan` 只接受 host 整数，显式拒绝隐式读取 GPU tensor。
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

### 本版开发验证：通知桥与 direct 原偏移

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
