# EP8 单 relay 贪心 AllToAllV：接口、正确性和性能对比

分支：`feature/a5-ccu-8rank-heuristic-alltoallv`，基于已通过新机测试的
`feature/a5-ccu-8rank-direct-alltoallv` / `d47d7b6`。
新算法仅用于八卡；relay 只能是同一通信组内、且不同于源和目的的 rank。
2/4 卡等长多路径 API、组外 relay 规则及原执行路径保留，不使用这个规划器。

本版是 **组内 HBM 转发、两阶段 CCU AllToAllV 的 eager 实验实现**：
第一阶段发送直连片段与 relay 片段；第二阶段由 relay 转发；最后恢复原来的
source-major 接收布局。组内转发不复用旧 2/4 卡的组外 UB source-route 技巧。
不是单次融合 CCU launch，也没有重叠两阶段、消除 pack/unpack 或接入 vLLM。
每次计时包含实际分片整理、两次 CCU 通信和结果恢复，不把通信子阶段冒充完整耗时。
为避免依赖未经验证的 Torch task queue 与 CCU host API 混合提交顺序，本版在每阶段
提交前、完成后显式 device synchronize（共四个阶段边界），全部计入多路径耗时。
这是保守 eager 正确性版本；后续优化可验证 stream/event 依赖后去掉这些全设备同步。

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

CPU 测试包含：精确单流分配、组间下界、早停、容量/预算/阈值、非法输入、拓扑关联、
150 组随机流量独立 oracle、结果汇总拒绝缺失 rank，以及 Torch CPU 下八线程执行器验证。
Torch CPU 测试不申请 NPU；没有 Torch 时该项会明确显示 skipped。

```bash
cd /home/l00934901/sgl-kernel-npu
python3 -m unittest discover -s tests/python/deepep -p 'test_a5_ccu_heuristic_cpu.py' -v
python3 -m unittest discover -s tests/python/deepep -p 'test_a5_ccu_alltoallv_cpu.py' -v
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
bash scripts/build_a5_ccu_urma_route_probe.sh --install > "$A5_BUILD_DIR/build.log" 2>&1 || {
  tail -100 "$A5_BUILD_DIR/build.log"; exit 1;
}
export A5_CCU_ROUTE_PROBE_LIB=/usr/local/Ascend/cann-9.1.T560/opp/vendors/cust/lib64/liba5_ccu_urma_route_probe.so
python3 - <<'PY'
import ctypes, os
lib = ctypes.CDLL(os.environ['A5_CCU_ROUTE_PROBE_LIB'], mode=ctypes.RTLD_GLOBAL)
for name in ('A5CcuBoundDirectAbiVersion', 'A5CcuAllToAllVAbiVersion'):
    fn = getattr(lib, name)
    fn.restype = ctypes.c_int
    assert fn() == 1, name
print('BOUND_DIRECT_AND_ALLTOALLV_ABI_PASS')
PY
)
```

DeepEP C++ API 没变，实验 worker 从本次 checkout 加载新 Python 模块，原有已安装
DeepEP 扩展可继续使用。未来在模型环境通过 `deep_ep.heuristic_alltoallv` 导入时，
需要把本分支 Python 包一同构建/安装，不能只更新 route SO。
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
  --suite correctness --dtype float32 --width 257 --row-scale 1 \
  --warmup 1 --iters 2 --rounds 1 --dynamic-iters 1
```

一次依次测试 zero、self、single_hot、competing、asymmetric、uniform、cut 七个矩阵，
复用同一个 CCU handle，同时变化矩阵、buffer 和 payload。每个 case 校验 direct、
multipath、原生 HCCL 与独立 CPU oracle，检查输入不被修改和前后 guard。
包含空发送 rank、空接收 rank、零长度方向及独立 self 偏移。
CPU 测试还覆盖禁止路径、阈值、relay 容量和一次分流后负载变化。

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
每个计时对象预热 3 次，每轮批量调用 10 次、等待完成，共 3 轮；另做 3 次动态调用。
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
uniform/cut/zero/self 等无收益 case 会走原有单次 direct 调用，没有两阶段额外开销。
可用 `--row-scale 1/16/128/256` 扫描数据量，`--width` 对应每行 hidden 元素数。

汇总字段的含义：

| 字段 | 解释 |
|---|---|
| `plan.direct_peak_bytes` | 全部直连时的最大有向链路字节数 |
| `plan.planned_link_peak_bytes` | 两阶段流量累计到物理边后得到的峰值，不是运行时间 |
| `plan.cut_lower_bound_bytes` | 等带宽 full mesh、完整行切分的全分组下界 |
| `schedule.phase_peak_sum_bytes` | 两阶段各自峰值之和，反映阶段串行的限制 |
| `plan.relay_bytes` | 需要第二跳转发的数据量 |
| `schedule.scratch_bytes_per_rank` | 四个临时张量的有效载荷大小；不含最终输入输出 |
| `planner_max_rank_us` | 单次 CPU 贪心规划的最慢 rank 耗时，不含计数交换/布局/分配 |
| `prepared` | 已准备计划/工作区，包含 pack、两次 CCU、unpack 和完成等待的平均每次时间 |
| `dynamic` | device 上行数 all-gather、转 CPU 同步、规划、布局、临时分配、执行与完成等待 |

prepared 的每轮先取八个 rank 中最大值，再对轮数求均值；dynamic 取各 rank 平均值中的最大值。
都是 host wall-clock；不是 CCU 内核 event 时间。不包括 router、模型 pack/unpack、专家计算；
这里的 pack/unpack 专指新增 relay 执行器内部的分片整理。原生 HCCL 只作正确性参考；
主要性能 baseline 是同一个 CCU direct-only AllToAllV。

动态测试每次重新规划同一个输入矩阵，不使用缓存；验证“变化矩阵”的正确性由第 3 步覆盖。
prepared 与 dynamic 的同步方式不同，只在同一口径内比较 direct/multipath。

特别是单热点流，尽管累计链路峰值从 32 降到 16 MiB，两阶段分别需要传 16 MiB，
加上 pack/unpack 和额外 launch，这个原型可能比 direct 更慢。这是有效的实验结论，
不能只根据 32→16 推导 2 倍加速。若 dynamic 无收益，真实模型应继续 direct，
再决定是否值得实现融合/流水线版本；本版不会声称已获得性能收益。

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
from deep_ep.heuristic_alltoallv import make_plan, PreparedAllToAllV, gather_routing_counts

# local_counts: 本 rank 的实际 packed 行数，device int64[8]。
# 若模型已持有一致的全局 host C，可直接传 C，避免重复交换。
C = gather_routing_counts(local_counts, group=ep_group)
plan = make_plan(C, row_bytes, topology["allowed_relays"],
                 min_chunk_bytes=1, min_gain_bytes=0, max_actions=56)
prepared = PreparedAllToAllV(buffer, handle, ep_rank, plan, packed_send,
                            max_scratch_bytes=1 << 30)
prepared.execute(packed_send, packed_recv)
```

`make_plan` 只接受 host 整数，显式拒绝隐式读取 GPU tensor。
`gather_routing_counts` 是同步参考适配器，包含 all-gather 和 D2H 等待；其开销已在 dynamic 计时中体现。
全部 rank 必须使用相同矩阵、候选路径和参数；模型在进入通信前必须建立这一一致性。
测试 worker 有配置一致性检查，模型不能依靠某个 rank 自己更改阈值或自动回退。
输入按目的 rank 排列，输出按原始源 rank 排列，relay suffix 最终写回该源对应区间；
self 单独从原输入偏移复制到正确接收偏移。算子不修改专家权重和 token 恢复映射。

计划和 workspace 对应当前矩阵，矩阵变化必须重新准备；固定计划可重复执行变化的 payload。
对象只允许单 stream、单在途调用，不支持并发复用；销毁/换工作区前保证先前执行完成。
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

在 8.146.225.22 的开发 checkout 和 `cam_lyw_dev_91` 完成：

- 112 项 CPU 回归通过，宿主机缺 Torch 而跳过的八线程执行器测试在容器内单独通过；
  新增测试文件共 9 项，包括 150 组随机矩阵和真实执行器的 CPU tensor 并发模拟。
- route 自定义包与 deep_ep_cpp 编译成功。开发 CANN 是 9.1.0-beta.1，
  不把它等同于新机器 T560 上的运行验证；新增 bound-direct ABI 符号已导出。
- `ALLTOALLV_META_PASS valid=160 invalid=12 legacy=10`；Python/Shell 语法和 diff 检查通过。
- 未在新机器执行本分支 NPU 测试，尚无多路径加速结论，未 push 到 origin。

开发构建/回归日志：`/home/liuyuanwen/profiling/a5-heuristic-build/`（开发机账号目录）。
新机器账号目录是 `/home/l00934901`，runner 默认不会使用开发机路径或 `/tmp` 存结果。
