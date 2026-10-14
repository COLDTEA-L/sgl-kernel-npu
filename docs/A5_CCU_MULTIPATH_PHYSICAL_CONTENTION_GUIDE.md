# A5 多路径算子物理方向冲突实验

本实验只面向有卡实验机器。它比较两个 prepared direct+relay AllToAll 作业的单独执行
baseline 与同步并发执行，并统计发生实际执行区间重叠的次数及时间膨胀。

## 1. 冲突构造

| 共享物理方向 | 作业 P：端点2、3，relay4 | 作业 Q：端点4、5，relay2 |
|---|---|---|
| `2→4` | `2→4→3` 的第一跳 | `5→2→4` 的第二跳 |
| `4→2` | `3→4→2` 的第二跳 | `4→2→5` 的第一跳 |

每个作业同时使用 direct 和一条显式 relay 路径，默认权重为 `1,1`。因此每次调用都有
正数据量经过 relay，P/Q 的执行区间发生重叠时，就形成对上述物理方向的冲突尝试。
分析器不仅比较卡号，还强制比较两份manifest中的UDie、port和topology edge；EID允许不同，
但物理端口不相同会直接判定实验无效。

尺寸参数表示每个 peer slice 的字节数，默认取：

```text
16KiB, 64KiB, 256KiB, 1MiB, 4MiB, 16MiB, 64MiB, 256MiB
```

两 rank AllToAll 的单 rank send tensor 是 `2 × bytes_per_peer`。测试还持有 recv 和
expected tensor；256MiB 点每个 rank 约需1.5GiB基础tensor空间，四个endpoint进程约6GiB，
不含通信runtime额外占用。

## 2. 拉取与入口检查

实验机只拉取和运行。若tracked工作区不干净则停止，不自动清理或覆盖。

```bash
(
set -e
cd /home/l00934901/sgl-kernel-npu
git status --short
git diff --quiet
git diff --cached --quiet
BRANCH=feature/a5-multipath-physical-contention
git fetch origin "${BRANCH}"
git switch "${BRANCH}"
git merge --ff-only "origin/${BRANCH}"
python3 tests/python/deepep/test_a5_ccu_path_contention_cpu.py
python3 tests/python/deepep/test_a5_ccu_path_contention_analyzer_cpu.py
bash scripts/run_a5_ccu_multipath_contention.sh --help
)
```

这是Python、Shell和分析器改动，不含新的C++内核。实验机需要已经安装与基线分支
`feature/a5-ccu-copy-comm-overlap`匹配的DeepEP扩展及route library。

## 3. 首次短测

先确认物理卡2、3、4、5均空闲。短测仅使用16KiB，不启用MindStudio profiler：

```bash
cd /home/l00934901/sgl-kernel-npu
bash scripts/run_a5_ccu_multipath_contention.sh \
  --sizes 16384 \
  --warmup 1 --iters 3 --repeats 1 \
  --output-root /home/l00934901/profiling
```

拓扑解析器会在运行前生成：

```text
manifests/P.tsv   # 2→4→3
manifests/Q.tsv   # 4→2→5
```

如果拓扑存在多个同卡relay候选，解析器会拒绝猜测。根据报错使用
`--p-relay-plane 0|1`、`--q-relay-plane 0|1`明确指定，不要随意选择。

## 4. 正式时间实验

默认完整扫描：

```bash
cd /home/l00934901/sgl-kernel-npu
bash scripts/run_a5_ccu_multipath_contention.sh \
  --warmup 2 --iters 10 --repeats 3 \
  --output-root /home/l00934901/profiling
```

对每个尺寸和repeat，顺序如下：

1. P单独连续执行，形成P baseline；
2. Q单独连续执行，形成Q baseline；
3. P/Q四个rank等待同一个`CLOCK_MONOTONIC`目标时间后同时开始，分别连续执行10次；
4. 每次prepared调用都等待本次调用完成，再进入下一次，因此每条记录对应一次完整算子。

baseline 使用同一作业、同一尺寸全部repeat的单次耗时中位数。实验组每次调用的
`time_inflation`定义为：

```text
并发时该次调用耗时 / 同作业同尺寸baseline中位数
```

## 5. MindStudio时间线复核

Profiler会改变小尺寸的绝对时间，因此不要把开启profiler的结果与无profiler baseline混用。
建议单独跑一个代表尺寸和较少迭代，例如1MiB：

```bash
cd /home/l00934901/sgl-kernel-npu
bash scripts/run_a5_ccu_multipath_contention.sh \
  --sizes 1048576 \
  --warmup 1 --iters 5 --repeats 1 \
  --mindstudio-profile \
  --output-root /home/l00934901/profiling
```

并发P/Q的每次调用都有CPU标记：

```text
A5_PATH_CONTENTION_P_ITER_0001
A5_PATH_CONTENTION_Q_ITER_0001
```

Profiler输出位于结果目录的`profiles/`，baseline和concurrent都会采集，以免在同一份
报告中混合有profiler和无profiler的耗时。将各rank生成的`*_ascend_pt`目录或其中的DB
复制到装有MindStudio Insight的机器中导入，在Timeline中检查P/Q标记、CCU任务和通信
任务的重叠。正式冲突计数仍使用`traces/`中的同机单调时钟区间，避免依赖两个独立
profiler文件的时间轴展示精度。

## 6. 输出与判读

```text
run_settings.json                 # 完整实验参数
manifests/P.tsv、Q.tsv            # 显式物理relay清单
case_status.tsv                   # 每个子作业是否通过
logs/                             # torchrun完整日志
traces/                           # 每rank每次调用起止时间
profiles/                         # 可选MindStudio数据
path_contention_baselines.csv     # baseline原始单次耗时
path_contention_iterations.csv    # 每次并发调用、冲突命中和膨胀
path_contention_summary.csv       # Excel可直接打开的尺寸级汇总
path_contention_summary.json      # 机器可读汇总
path_contention_report.md         # 人读报告
```

`path_contention_iterations.csv`中的关键列：

- `conflict=YES`：该作业本次调用与另一作业至少一次调用有非零区间重叠；
- `peer_iterations`：与它重叠的另一作业迭代编号；
- `overlap_fraction`：本次collective窗口被另一作业覆盖的比例；
- `time_inflation`：相对solo baseline的倍数；
- `time_inflation_percent`：相对solo baseline增加的百分比。

汇总中的`overlapping_pairs`按P迭代与Q迭代的重叠配对计数；
`conflict_iterations`则按每个作业自身被冲突覆盖的迭代计数。一个较慢的P迭代可能跨过
两个Q迭代，所以两种计数不一定相等。

## 7. 结论边界

- manifest中的卡号、UDie、port和edge共同证明路径共享同一物理链路；EID不要求相同；
- 正权重保证每次调用有relay数据；
- 单调时钟区间证明两个完整算子在时间上重叠；
- 相对baseline的膨胀量化冲突影响；
- 这些证据不能单独证明端口内部逐周期仲裁，最终归因应结合MindStudio时间线和HCCN端口计数；
- 同机其他作业也可能造成膨胀，正式实验应选择空闲窗口并至少重复三轮；
- `2→4`与`4→2`分别统计，不假设双向链路互相竞争。
