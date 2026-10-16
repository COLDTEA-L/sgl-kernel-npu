# A5 多路径算子物理方向冲突实验

本实验只面向有卡实验机器。它比较两个 prepared direct+relay AllToAll 作业的单独执行
baseline 与独立instance并发执行。脚本自动计算整批平均时间膨胀；逐次冲突概率只从
MindStudio device timeline统计，不用host提交区间代替device执行区间。

新增`--experiment baseline-check`，优先判断另一套闲置instance是否影响baseline；该模式
不运行concurrent。默认`--experiment contention`仍运行常驻baseline和concurrent。
两种模式的正式循环都只调用预先绑定参数的`launch()`，不做逐次Python计时、字典记录或
`record_function`标记；仅在循环两侧记录整批时间，并在循环外同步。
这里的纯launch指Python提交循环；C++算子原有的die1 main/route stream通知仍保留，
本次没有修改CCU kernel或Channel协议。

数据面固定使用overlap分支已经验证的全CCU组合：远端为本仓的CCU+URMA direct/relay
多路径WriteNb，本地self slice为`GroupCopy8x32K`，Channel元数据协议为`baseline`。
runner会强制设置`A5_CCU_SELF_COPY_MODE=ccu`和
`A5_CCU_CHANNEL_SYNC_MODE=baseline`；不允许退回ACL D2D overlap，也不使用存在性能劣化的
`packed`或`packed-ready`协议。

## 1. 冲突构造

| 共享物理方向 | 作业 P：端点2、3，relay4 | 作业 Q：端点4、5，relay2 |
|---|---|---|
| `2→4` | `2→4→3` 的第一跳 | `5→2→4` 的第二跳 |
| `4→2` | `3→4→2` 的第二跳 | `4→2→5` 的第一跳 |

每个作业同时使用 direct 和一条显式 relay 路径，默认权重为 `1,1`。因此每次调用都有
正数据量经过 relay，P/Q 的执行区间发生重叠时，就形成对上述物理方向的冲突尝试。
分析器不仅比较卡号，还强制比较两份manifest中的UDie、port和topology edge；EID允许不同，
但物理端口不相同会直接判定实验无效。

表中卡号是默认配置，不是runner硬编码。可以通过`--p-devices A,B --p-relay C`和
`--q-devices C,D --q-relay A`选择任意四张物理卡。两个作业的端点集合必须互不重叠，
并且P的relay必须是Q的第一个端点、Q的relay必须是P的第一个端点；这样共享方向自动记为
`A→C`和`C→A`。例如卡2、3、6、7使用：

```text
P：端点2、3，relay6
Q：端点6、7，relay2
共享方向：2→6、6→2
```

对应参数为：

```bash
--p-devices 2,3 --p-relay 6 --q-devices 6,7 --q-relay 2
```

尺寸参数表示每个 peer slice 的字节数，默认取：

```text
16KiB, 64KiB, 256KiB, 1MiB, 4MiB, 16MiB, 64MiB, 256MiB
```

两 rank AllToAll 的单 rank send tensor 是 `2 × bytes_per_peer`。测试还持有 recv 和
expected tensor；256MiB 点每个 rank 约需1.5GiB基础tensor空间，四个endpoint进程约6GiB，
不含通信runtime额外占用。每个active阶段warmup前做两次改变输入的排队校验，额外保存
两份recv快照；256MiB时校验阶段每rank额外约1GiB，校验后释放，正式阶段不保留快照。
send和expected直接按最终shape分配并原地填充，不再通过
`torch.stack`额外产生约512MiB/rank的临时峰值；100次正式调用始终复用同一组tensor。

runner不再对全部尺寸固定设置`HCCL_BUFFSIZE=2300`。默认`--hccl-buffsize-mb auto`按
下式计算每个尺寸的最低需求：

```text
max(200MB, ceil(2 × bytes_per_peer / MiB) + 64MB)
```

常驻通信域初始化后不能按尺寸修改该环境变量，因此一次实验取所有请求尺寸需求的最大值：
扫描到128MiB时统一使用320MB，扫描到256MiB时统一使用576MB。这里
`2 × bytes_per_peer`是单rank完整payload，64MB是初始化余量；prepared算子直接读写用户
send/recv tensor，这个HCCL buffer不是256MiB数据的64份副本。需要复现实验机的固定配置时，
可显式传`--hccl-buffsize-mb N`。最低需求、整次实验实际值都保存在
`run_settings.json`，启动时shell也会打印`PERSISTENT_ENVIRONMENT`。

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
python3 tests/python/deepep/test_a5_ccu_test_runtime_cpu.py
bash scripts/run_a5_ccu_multipath_contention.sh --help
)
```

这是Python、Shell和分析器改动，不含新的C++内核。实验机需要已经安装与基线分支
`feature/a5-ccu-copy-comm-overlap`匹配的DeepEP扩展及route library。

正式runner还会在解析拓扑、初始化设备和创建通信域之前自动执行一次真实runtime预检。
成功时会打印`CASE_TEST_EXTENSION`和`MULTIPATH_RUNTIME_VERIFIED`，确认当前`python3`选中的
DeepEP包、`deep_ep_cpp`二进制及route library来自同一套可用环境。预检失败不会占卡。
本实验要求prepared-plan ABI至少为16、local+remote ABI至少为2、Channel sync ABI至少为1；
旧route library会在预检阶段直接失败，不能静默改用D2D或旧Channel协议。

仓库源码中不要求存在
`python/deep_ep/deep_ep/vendors/hwcomputing/bin/set_env.bash`。它是构建或安装阶段的生成文件，
不属于Git checkout。runner只加载CANN环境；每个worker会通过`a5_ccu_test_runtime.py`从当前
Python解释器实际安装的`deep_ep`包定位扩展、custom OPP和route library，并在导入torch前
补齐环境。bootstrap同时搜索系统site-packages、user site、当前解释器的`sys.path`以及本仓
源码构建目录；加载时将包内extension注册为`deep_ep_cpp`和`deep_ep.deep_ep_cpp`同一个模块，
兼容旧wheel的顶层导入和新版相对导入。若仍失败，错误会列出实际Python解释器和全部搜索
目录，而不是只显示模糊的module-not-found。

## 3. 首次短测

先确认所选的四张物理卡均空闲。短测仅使用16KiB，并显式关闭MindStudio profiler：

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

以上注释对应默认卡号；自定义卡号时manifest会按命令行端点和relay动态生成。

如果拓扑存在多个同卡relay候选，解析器会拒绝猜测。根据报错使用
`--p-relay-plane 0|1`、`--q-relay-plane 0|1`明确指定，不要随意选择。

每个case日志必须同时出现以下标记，否则runner把该case记为失败：

```text
CASE_DATA_PLANE_POLICY ... self_copy=ccu channel_sync=baseline
CCU_LOCAL_REMOTE_PLAN ... self_copy=GroupCopy8x32K
```

日志中出现`SELF_COPY_LANE mode=overlap`同样会导致失败。这保证正式100次调用的本地和
卡间数据搬运都在同一个CCU图内，不会混入独立D2D copy stream的`MemcpyAsync`及其
`aclrtStreamWaitEvent`。

## 4. 先验证baseline，再做正式冲突实验

### 4.1 判断闲置instance的影响

先用64KiB和4MiB、warmup100、正式100次，对同一路径做以下对照：

1. `solo_baseline_P`：只启动P的两个rank，Q没有worker、通信域或plan。
2. 等P的进程完全退出，再启动`solo_baseline_Q`；此时P没有worker或资源。
3. 两个solo环境都退出后，启动P和Q两套常驻资源。测`baseline_P`时Q只等待host屏障，
   测`baseline_Q`时P只等待host屏障；两者都不运行concurrent。

每套环境只初始化一次，跨所有请求尺寸复用；solo P、solo Q、resident合计三批环境初始化。
三组使用同一份manifest、相同权重、dtype、HCCL buffer预算、warmup和正式次数。
例如使用2、3、6、7卡：

```bash
cd /home/l00934901/sgl-kernel-npu
bash scripts/run_a5_ccu_multipath_contention.sh \
  --experiment baseline-check \
  --p-devices 2,3 --p-relay 6 --q-devices 6,7 --q-relay 2 \
  --sizes 65536,4194304 \
  --warmup 100 --iters 100 --repeats 1 \
  --output-root /home/l00934901/profiling_baseline_check
```

这个命令默认采profiling。若使用2、3、4、5，则改为：

```bash
--p-devices 2,3 --p-relay 4 --q-devices 4,5 --q-relay 2
```

每完成一个solo尺寸，shell输出`BASELINE_COMPLETE mode=solo job=P|Q size=...`。
常驻组每完成一个尺寸，输出`SIZE_COMPLETE`及四种baseline的host整批平均时间。

结果根目录和`results/size_64KB/`、`results/size_4MB/`分别保存：

```text
baseline_resource_summary.csv   # 每个job/rank的solo、resident时间与比值
baseline_resource_cases.csv     # 每轮CCU数量、min/median/max及host时间
baseline_resource_summary.json
baseline_resource_report.md
```

按同一rank比较`solo_ccu_median_us`与`resident_ccu_median_us`；
`resident_over_solo_ccu_ratio`是两者的比值。每份profile必须有恰好100个正式CCU任务，
且不含Memcpy，否则设备时间标记为不完整，不拿host时间补齐。host平均值包含提交空隙和
最后同步，不能当作纯CCU设备时间。这个对照测的是整个闲置instance环境的影响，包括
worker和runtime后台线程、通信域、plan/channel等；单次差异不能直接归因到某一种资源。

profile路径按尺寸分别保存：

```text
profiles/size_64KB/repeat_1/solo_baseline_P/rank0/
profiles/size_64KB/repeat_1/solo_baseline_P/rank1/
profiles/size_64KB/repeat_1/solo_baseline_Q/rank0/
profiles/size_64KB/repeat_1/solo_baseline_Q/rank1/
profiles/size_64KB/repeat_1/baseline_P/rank0/       # Q资源常驻但不发流量
profiles/size_64KB/repeat_1/baseline_Q/rank0/       # P资源常驻但不发流量
```

先检查solo中两张卡的CCU时长及相邻Launch空隙是否恢复正常，再比较resident。
若solo仍不均衡，不能把问题归到另一套闲置资源。若solo均衡、resident变差，再继续隔离
闲置instance的影响；同机其他用户的流量及不同时间段环境变化也需结合时间线判断。
目前路径仍为direct+1relay、默认权重`1,1`，不能与direct+2relay、`2,1,1`直接比较绝对时延。

### 4.2 baseline确认后，继续baseline与concurrent

默认完整扫描已经固定为：每个case先warmup 100次，再连续正式执行100次，repeat为1，
并启用MindStudio profiler。直接运行：

```bash
cd /home/l00934901/sgl-kernel-npu
bash scripts/run_a5_ccu_multipath_contention.sh \
  --experiment contention \
  --output-root /home/l00934901/profiling
```

对每个尺寸，顺序如下：

1. P单独warmup 100次，warmup完成后连续执行100次，形成P baseline（Q资源常驻）；
2. Q单独warmup 100次，warmup完成后连续执行100次，形成Q baseline（P资源常驻）；
3. 并发P/Q各自先完成warmup 100次；四个rank随后等待同一个`CLOCK_MONOTONIC`
   目标时间并同时开始，P和Q分别连续执行100次；
4. P、Q各自在自己的进程和stream中连续提交100次，循环内部不调用
   `torch.npu.synchronize()`；各instance只在自己的循环结束后同步一次；
5. P和Q只对齐第一个正式iteration的起跑点，此后没有逐iteration barrier，也不会互相
   synchronize。

P和Q两组两rank worker在整次尺寸扫描开始时各启动一次。每个rank只创建一次HCCL通信域，
只执行一次prepared plan创建及`HcclCcuKernelRegister/Finish`，随后同一个plan和默认stream
依次服务全部数据量、baseline和concurrent阶段。尺寸切换只重新分配对应大小的send/recv/
expected tensor，不重启`torchrun`，也不销毁重建通信域。全部尺寸结束后才统一销毁plan和
通信域。因此14个尺寸不再触发56次独立worker初始化。

两个instance始终是独立HCCL通信域。baseline P时Q进程只等待host文件屏障且不向device提交
算子；baseline Q时P同理。并发阶段才同时向各自stream提交任务。host文件屏障只控制阶段边界，
不产生P/Q之间的device Event或Notify依赖。

每个`size × repeat`完成全部四份记录后，shell立即输出一行，例如：

```text
SIZE_COMPLETE size=16KB bytes_per_peer=16384 repeat=1 baseline_P_us=... baseline_Q_us=... concurrent_P_us=... concurrent_Q_us=... artifacts=...
```

这行出现时，该组的四份trace、profile和状态JSON均已落盘，可以直接看到进度；最终跨尺寸CSV、
JSON和Markdown仍在全部数据量完成后由分析器统一生成。

每次正式调用应只有一个包含GroupCopy和多路径WriteNb的CCU launch。P、Q起跑屏障只使用
host文件和`CLOCK_MONOTONIC`，不会建立P与Q之间的device Event/Notify依赖。

每个active阶段先做两次改变输入、毒化recv并保存完整snapshot的batch校验，全部snapshot
正确后才进入warmup。校验和warmup都不入profile。
Profiler在每个case的warmup和warmup正确性检查全部完成后才启动。每份
MindStudio profile包含该case正式阶段的100次CCU launch，不再生成逐次Python标记，
不包含前100次warmup。Profiler启动后到正式
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
`concurrent_PQ/P`、`concurrent_PQ/Q`下面。host trace只记录整批完成窗口和调用次数，
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

并发P/Q的正式循环只提交launch，不添加逐次CPU标记。按各rank执行stream中的100个CCU
launch的先后顺序关联iteration；CPU/CANN原有的算子/API记录仍由profiler采集。

Profiler输出按数据量、repeat、phase和job分别保存在`profiles/`。检查并发case时，将完整的
`profiles/size_<尺寸>/repeat_<轮次>/concurrent_PQ/`父目录复制到装有MindStudio Insight的
机器并一次性导入，使P、Q各rank处于同一cluster Timeline；不要分别只导入P或Q的单个DB。
在Timeline中按job、rank、stream检查CCU任务和通信
任务的重叠。逐次冲突概率必须以device task区间为准：对P的100次device调用，统计其中
与Q任意一次device调用在共享物理方向上发生重叠的次数，再除以100；Q同理。

## 6. 输出与判读

```text
run_settings.json                 # 完整实验参数
manifests/P.tsv、Q.tsv            # 显式物理relay清单
case_status.tsv                   # 每个子作业是否通过
logs/                             # torchrun完整日志
logs/persistent_P.log             # P常驻instance的全部尺寸/阶段日志
logs/persistent_Q.log             # Q常驻instance的全部尺寸/阶段日志
status/                           # 每个尺寸/阶段/rank的完成状态与batch时间
progress/                         # shell逐尺寸完成通知
traces/                           # 调用次数和整批完成窗口，不是device执行区间
profiles/                         # 可选MindStudio数据
results/size_<KB或MB>/            # 每个数据量单独保存的一套结果
path_contention_batches.csv       # baseline/concurrent整批平均时间和膨胀
path_contention_submissions.csv   # 兼容旧结果；新纯launch实验仅有表头，不伪造逐次时间
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
traces/size_16KB/repeat_1/concurrent_PQ/P/     # P的调用次数和batch窗口
traces/size_16KB/repeat_1/concurrent_PQ/Q/     # Q的调用次数和batch窗口
```

`path_contention_batches.csv`中的关键列：

- `batch_average_us`：整批完成时间除以正式调用次数；
- `time_inflation`：并发batch平均值相对同作业resident baseline中位数的倍数；
- `host_batch_overlap_fraction`：P/Q整批host等待窗口是否确实重叠，只用于实验有效性检查；
- `conflict_probability=MINDSTUDIO_TIMELINE_REQUIRED`：明确禁止用host记录自动推断逐次冲突。

旧结果`path_contention_submissions.csv`中的`enqueue_duration_us`只是Python/C++入口提交耗时。
新结果不再采集该值；逐次时延与重叠从device timeline读取。

## 7. 结论边界

- manifest中的卡号、UDie、port和edge共同证明路径共享同一物理链路；EID不要求相同；
- 正权重保证每次调用有relay数据；
- host整批窗口只证明两组实验在总体时间上同时运行；
- 相对baseline的batch平均膨胀量化总体影响，逐次膨胀和冲突概率以Timeline为准；
- 这些证据不能单独证明端口内部逐周期仲裁，最终归因应结合MindStudio时间线和HCCN端口计数；
- 同机其他作业也可能造成膨胀；默认先完整采一轮，需要评估轮间波动时再使用
  `--repeats 3`另跑三轮；
- `2→4`与`4→2`分别统计，不假设双向链路互相竞争。
