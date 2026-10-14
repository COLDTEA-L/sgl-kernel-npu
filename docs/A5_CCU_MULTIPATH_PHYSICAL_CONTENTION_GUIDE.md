# A5 多路径算子物理方向冲突实验

本实验只面向有卡实验机器。它比较两个 prepared direct+relay AllToAll 作业的单独执行
baseline 与独立instance并发执行。脚本自动计算整批平均时间膨胀；逐次冲突概率只从
MindStudio device timeline统计，不用host提交区间代替device执行区间。

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
不含通信runtime额外占用。send和expected直接按最终shape分配并原地填充，不再通过
`torch.stack`额外产生约512MiB/rank的临时峰值；100次正式调用始终复用同一组tensor。

runner不再对全部尺寸固定设置`HCCL_BUFFSIZE=2300`。默认`--hccl-buffsize-mb auto`按
下式为每个尺寸、每个通信域单独设置：

```text
max(200MB, ceil(2 × bytes_per_peer / MiB) + 64MB)
```

因此16KiB至64MiB使用200MB，256MiB使用576MB。这里`2 × bytes_per_peer`是单rank完整
payload，64MB是初始化余量；prepared算子直接读写用户send/recv tensor，这个HCCL buffer
不是256MiB数据的64份副本。需要复现实验机的固定配置时，可显式传
`--hccl-buffsize-mb N`，实际取值保存在`run_settings.json`并写入每个case日志。

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

仓库源码中不要求存在
`python/deep_ep/deep_ep/vendors/hwcomputing/bin/set_env.bash`。它是构建或安装阶段的生成文件，
不属于Git checkout。runner只加载CANN环境；每个worker会通过`a5_ccu_test_runtime.py`从当前
Python解释器实际安装的`deep_ep`包定位扩展、custom OPP和route library，并在导入torch前
补齐环境。若wheel或route library没有安装，worker会直接报告缺少的组件，不会误用源码目录。

## 3. 首次短测

先确认物理卡2、3、4、5均空闲。短测仅使用16KiB，并显式关闭MindStudio profiler：

```bash
cd /home/l00934901/sgl-kernel-npu
bash scripts/run_a5_ccu_multipath_contention.sh \
  --sizes 16384 \
  --warmup 2 --iters 5 --repeats 1 \
  --no-mindstudio-profile \
  --output-root /home/l00934901/profiling
```

拓扑解析器会在运行前生成：

```text
manifests/P.tsv   # 2→4→3
manifests/Q.tsv   # 4→2→5
```

如果拓扑存在多个同卡relay候选，解析器会拒绝猜测。根据报错使用
`--p-relay-plane 0|1`、`--q-relay-plane 0|1`明确指定，不要随意选择。

## 4. 正式实验

默认完整扫描已经固定为：每个case先warmup 10次，再连续正式执行100次，repeat为1，
并启用MindStudio profiler。直接运行：

```bash
cd /home/l00934901/sgl-kernel-npu
bash scripts/run_a5_ccu_multipath_contention.sh \
  --output-root /home/l00934901/profiling
```

对每个尺寸，顺序如下：

1. P单独warmup 10次，warmup完成后连续执行100次，形成P baseline；
2. Q单独warmup 10次，warmup完成后连续执行100次，形成Q baseline；
3. 并发P/Q各自先完成warmup 10次；四个rank随后等待同一个`CLOCK_MONOTONIC`
   目标时间并同时开始，P和Q分别连续执行100次；
4. P、Q各自在自己的进程和stream中连续提交100次，循环内部不调用
   `torch.npu.synchronize()`；各instance只在自己的循环结束后同步一次；
5. P和Q只对齐第一个正式iteration的起跑点，此后没有逐iteration barrier，也不会互相
   synchronize。

Profiler在每个case的warmup和warmup正确性检查全部完成后才启动。也就是说，每份
MindStudio profile包含该case正式阶段的100个标记
`A5_PATH_CONTENTION_<P|Q>_ITER_0001..0100`，不包含前10次warmup。Profiler启动后到正式
调用开始前仍会有一次本地stream清空和四进程首轮起跑屏障，100次提交完成后有一次各自
instance的本地同步。因此“只取100次”是指其中只有100次被测算子，不是说profile文件
完全没有CPU同步事件。

这里一个尺寸包含三个实验case：baseline P、baseline Q和concurrent PQ。concurrent PQ
同时运行P、Q并分别保存两份采集结果，所以总共得到baseline P、baseline Q、并发中的P、
并发中的Q四份100次记录，而不是四份合计100次。若通过`--repeats N`增加重复轮数，
每轮的每份记录仍各自包含100次。

其中`concurrent P`和`concurrent Q`不是先后运行的两个实验。runner会同时启动P、Q两个
torchrun进程组，四个rank在同一个host单调时钟目标点释放。因为P和Q属于不同进程组，
它们必须各自生成trace和profiler文件；目录统一放在同一次
`concurrent_PQ/P`、`concurrent_PQ/Q`下面。host trace只记录提交时刻和整批完成窗口，
不把提交区间解释成device冲突区间。

baseline和concurrent都采用“连续提交100次、循环外同步”的同一方式。每轮整批耗时除以
100得到batch平均单次耗时；`time_inflation`定义为：

```text
并发batch平均单次耗时 / 同作业同尺寸baseline batch平均耗时中位数
```

## 5. MindStudio时间线复核

正式实验默认启用Profiler，baseline和concurrent都会在相同条件下采集，避免在同一报告中
混合有profiler和无profiler的耗时。如需只测host计时，可额外使用
`--no-mindstudio-profile`另起一次实验；不要把两次实验的绝对时间直接混用。

若只想先复核1MiB的完整100次时间线：

```bash
cd /home/l00934901/sgl-kernel-npu
bash scripts/run_a5_ccu_multipath_contention.sh \
  --sizes 1048576 \
  --output-root /home/l00934901/profiling
```

并发P/Q的每次调用都有CPU标记：

```text
A5_PATH_CONTENTION_P_ITER_0001
A5_PATH_CONTENTION_Q_ITER_0001
```

Profiler输出按数据量、repeat、phase和job分别保存在`profiles/`。将各rank生成的
`*_ascend_pt`目录或其中的DB
复制到装有MindStudio Insight的机器中导入，在Timeline中检查P/Q标记、CCU任务和通信
任务的重叠。逐次冲突概率必须以device task区间为准：对P的100次device调用，统计其中
与Q任意一次device调用在共享物理方向上发生重叠的次数，再除以100；Q同理。CPU marker
只用于关联iteration编号，其自身宽度是host提交耗时，不是device算子耗时。

## 6. 输出与判读

```text
run_settings.json                 # 完整实验参数
manifests/P.tsv、Q.tsv            # 显式物理relay清单
case_status.tsv                   # 每个子作业是否通过
logs/                             # torchrun完整日志
traces/                           # host提交记录和整批完成窗口，不是device执行区间
profiles/                         # 可选MindStudio数据
results/size_<KB或MB>/            # 每个数据量单独保存的一套结果
path_contention_batches.csv       # baseline/concurrent整批平均时间和膨胀
path_contention_submissions.csv   # 每次host提交记录，仅用于编号和诊断
path_contention_summary.csv       # Excel可直接打开的尺寸级汇总
path_contention_summary.json      # 机器可读汇总
path_contention_report.md         # 人读报告
```

例如16KiB的独立结果位于：

```text
results/size_16KB/path_contention_batches.csv
results/size_16KB/path_contention_submissions.csv
results/size_16KB/path_contention_summary.csv
results/size_16KB/path_contention_summary.json
results/size_16KB/path_contention_report.md
```

目录名使用`16KB`、`64KB`、`256KB`、`1MB`等可读标签；这里KB和MB分别按1024和
1024²字节换算。CSV/JSON内部仍记录精确的`bytes_per_peer`整数。

`traces/size_<KB或MB>/`和`profiles/size_<KB或MB>/`也按数据量隔离。结果目录根部的同名CSV、
JSON和Markdown是跨全部数据量的总表，便于一次导入Excel比较；`results/size_<KB或MB>/`
中的文件仅包含该数据量，方便后续逐个留存和查看。

以16KB、repeat 1为例，并发目录为：

```text
profiles/size_16KB/repeat_1/concurrent_PQ/P/   # P在同一次P+Q并发运行中的profile
profiles/size_16KB/repeat_1/concurrent_PQ/Q/   # Q在同一次P+Q并发运行中的profile
traces/size_16KB/repeat_1/concurrent_PQ/P/     # P的100次host提交和batch窗口
traces/size_16KB/repeat_1/concurrent_PQ/Q/     # Q的100次host提交和batch窗口
```

`path_contention_batches.csv`中的关键列：

- `batch_average_us`：整批完成时间除以正式调用次数；
- `time_inflation`：并发batch平均值相对solo baseline中位数的倍数；
- `host_batch_overlap_fraction`：P/Q整批host等待窗口是否确实重叠，只用于实验有效性检查；
- `conflict_probability=MINDSTUDIO_TIMELINE_REQUIRED`：明确禁止用host记录自动推断逐次冲突。

`path_contention_submissions.csv`中的`enqueue_duration_us`只是Python/C++入口提交耗时。
它不能用于算子时延、P/Q重叠或冲突概率计算。

## 7. 结论边界

- manifest中的卡号、UDie、port和edge共同证明路径共享同一物理链路；EID不要求相同；
- 正权重保证每次调用有relay数据；
- host整批窗口只证明两组实验在总体时间上同时运行；
- 相对baseline的batch平均膨胀量化总体影响，逐次膨胀和冲突概率以Timeline为准；
- 这些证据不能单独证明端口内部逐周期仲裁，最终归因应结合MindStudio时间线和HCCN端口计数；
- 同机其他作业也可能造成膨胀；默认先完整采一轮，需要评估轮间波动时再使用
  `--repeats 3`另跑三轮；
- `2→4`与`4→2`分别统计，不假设双向链路互相竞争。
