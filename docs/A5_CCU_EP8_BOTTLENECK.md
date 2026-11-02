# EP8 EID：只对瓶颈分流的规划器与新机性能测试

分支：`feature/a5-ccu-ep8-bottleneck-alltoallv`，基于 `feature/a5-ccu-92-93-alltoall` 的 `121e8f4`。新策略名 `bottleneck`，旧策略 `greedy` 保留。只修改EP8主机侧规划和测试，不修改route/kernel、2/4卡接口。仍为通信组内CCU+URMA EID转发，不落relay HBM，每次一次CCU kernel launch，不分chunk。

## 1. 算法

旧greedy反复寻找能改善三条边局部峰值的动作，可能在全网瓶颈已经几乎不可改善时继续分流其他小流。

新策略：

1. 验证本次真实8×8行数C，去掉self得到L。计算原峰值P。
2. 利用8卡全部254个非平凡有向割计算下界，并加入单条流最多拆成直连和一个relay路径的 `ceil(P/2)` 下界。割求和用子集递推，约256×8的固定整数运算；下界不代表一定可达。
3. 最多尝试4个目标：`LB`、`LB+ceil((P-LB)/7)`、`LB+ceil((P-LB)/3)`、`LB+ceil((P-LB)/2)`，跳过重复或不低于P的目标。不是凸优化，也不依赖贪心可行性的单调性。
4. 原始流只排序一次，按负载降序、索引升序处理。只为当前负载超过T的边选relay，默认只搬 `L[i,j]-T` 行；若配置了更大的最小分流量，则按该最小量搬运。
5. k必须是拓扑允许的组内rank。要求两条借道边加上x后都不超过T、容量足够、x不超过该原始流行数。优先选择两条借道边更新后峰值更小的k，同值优先已承担转发量更少的k，再按rank固定打破平局。
6. 每条流最多一个relay，每个候选计划最多 `min(max_actions,8)` 次动作。第一个完整可行计划被采用；不能处理完全部超T边时丢弃该次全部动作，尝试下一目标。所有目标失败则保持custom direct，不自动切换native。

允许把一组并列热点一起削峰，但绝不发布未完成的局部计划。接受的计划保证在该等带宽边负载模型下全网峰值严格下降；`min_gain_bytes` 对新策略约束全网峰值减少量。旧greedy的阈值语义不变。每次使用最新C，不缓存profile或上一请求的分流决定。`WithRowBytes()`保留策略，dispatch/combine根据真实dtype分别规划。

规划候选评估有固定上限 `4×56×6`；动作上限8。达到目标后立即结束，不再追求其他非瓶颈边的均衡。小目标可能被贪心遗漏，不能把返回直连解释为数学上无可行relay。

### 673-token真实样本

随仓库提供 `tests/data/ep8_prefill_673.json`。每rank4038行，FP8 width5120。全部合法relay可用时：

| 策略 | 动作数 | 总relay行数 | 全网跨卡峰值 |
|---|---:|---:|---:|
| 旧greedy（报告） | 27 | 1314 | 1371→1281 |
| 新bottleneck | 4 | 241 | 1371→1281 |

具体helper可能随现场合法候选变化；不要绕过拓扑限制来匹配表格。rank4跨卡接收8861行，对应入边下界1266行，说明该样本本来就只有很小的削峰空间。原网络流量28319行，1314行额外流量是4.64%，新241行是0.85%。这不是NPU性能结果，不承诺时延改善。

## 2. 新机更新和安装

以下操作在目标容器 `yuanwen_dsk` 内，目录 `/home/l00934901/sgl-kernel-npu`。先保存自己已有改动，禁止用reset/clean覆盖模型接入代码。开发者push后执行：

```bash
cd /home/l00934901/sgl-kernel-npu
git status --short
git fetch origin feature/a5-ccu-ep8-bottleneck-alltoallv
# 首次本地创建：
git switch --track origin/feature/a5-ccu-ep8-bottleneck-alltoallv
# 本地分支已存在时，以上switch改为：
# git switch feature/a5-ccu-ep8-bottleneck-alltoallv
# git merge --ff-only origin/feature/a5-ccu-ep8-bottleneck-alltoallv
```

**已安装并验证通过121e8f4 EID route的机器，不必重编route。本次必须按 `docs/A5_CCU_92_93_ALLTOALL.md` 第4节重编安装DeepEP wheel。** 不切回旧分支，保留本分支完成构建。若仍是HBM/READY实验route或首次部署，先按同文档第3节构建本分支route，再执行第4节wheel构建。不能只更新Python文件而复用旧planner二进制。

新增扩展能力：`ep8_bottleneck_planner_abi=1`，原 `ep8_relay_planner_abi=2` 保留。bottleneck构造会检查新能力，旧wheel会明确拒绝。重启使用旧wheel的模型进程后才会加载新扩展。

```bash
export A5_CANN_PATH=/usr/local/Ascend/cann-9.3.0
export A5_CCU_ROUTE_PROBE_LIB="$A5_CANN_PATH/opp/vendors/cust/lib64/liba5_ccu_urma_route_probe.so"
export A5_OUTPUT_ROOT=/home/l00934901/profiling
export A5_DEVICES=0,1,2,3,4,5,6,7
export A5_ROOT_INFO=/etc/hccl_rootinfo.json
export A5_ATLAS_JSON=/usr/local/Ascend/driver/950/atlas_950_1.json
export A5_HCCN_TOOL=/usr/local/Ascend/driver/tools/hccn_tool
export HCCL_OP_EXPANSION_MODE=CCU_SCHED HCCL_BUFFSIZE=64
export A5_EID_MAX_RELAYS_PER_PEER=6 A5_CCU_RESOURCE_DIAGNOSTICS=0
unset A5_EID_RELAY_SCHEDULE
```

Atlas路径按现场实际文件替换。旧HBM `participant_hbm_two_stage_ccu` manifest不能用于EID：模型必须使用同一次 `--executor eid` 生成的 `peer_plan.tsv` 和 `topology.json`，后者version=2、executor=`participant_eid_ccu_urma`。独立测试脚本每次自动重新生成这对文件。

## 3. 先验证正确性

八卡可用时执行：

```bash
bash scripts/run_a5_ccu_eid_alltoallv.sh \
  --suite create --register-stage full --planner bottleneck

bash scripts/run_a5_ccu_eid_alltoallv.sh \
  --suite correctness --mode all --planner bottleneck \
  --dtype float8_e4m3fn --width 5120 --row-scale 1 \
  --warmup 3 --iters 10 --rounds 1 --dynamic-iters 100 \
  --bidirectional-iters 100
```

依次要求 `EP8_EID_CREATE_ALL_RANKS_AND_EXIT_PASS`、`EP8_EID_ALL_RANKS_AND_EXIT_PASS`。原7个合成用例加双向交替验证；真实样本按下一节单独运行，不把注册成功当作数据正确性成功。

## 4. DeepSeek在机器上执行的性能对照

2026-11-02补充：分析dynamic劣化前，先读第8节。已有结果优先按8.2提取，缺项或需要确认稳定性时才按8.3补测。

**使用同一个新wheel，分别选择greedy和bottleneck。不要在两个不同提交的旧测试脚本间比较。** 两种策略均与native HCCL和custom direct同进程交替测量。

先测试随仓库保存的真实样本，两种dtype：

```bash
mkdir -p /home/l00934901/profiling
A5_SWEEP=$(mktemp -d /home/l00934901/profiling/ep8-bottleneck-sweep.XXXXXX)
for A5_POLICY in greedy bottleneck; do
  for A5_DTYPE in float8_e4m3fn bfloat16; do
    bash scripts/run_a5_ccu_eid_alltoallv.sh \
      --suite benchmark --mode all --planner "$A5_POLICY" \
      --matrix-json tests/data/ep8_prefill_673.json \
      --dtype "$A5_DTYPE" --width 5120 \
      --warmup 3 --iters 10 --rounds 6 --dynamic-iters 100 \
      > "$A5_SWEEP/${A5_POLICY}_${A5_DTYPE}.log" 2>&1 || {
        tail -100 "$A5_SWEEP/${A5_POLICY}_${A5_DTYPE}.log"; exit 1;
      }
    tail -20 "$A5_SWEEP/${A5_POLICY}_${A5_DTYPE}.log"
  done
done
echo "扫描摘要：$A5_SWEEP"
```

同一C换BF16这里只测试字节宽度，**不代表真实combine矩阵**。随后把matrix路径换为模型profile结束后采集的真实dispatch和combine文件，各按FP8/BF16再测。格式同样为64个packed行数，按源rank行、目的rank列展开，并保持物理卡顺序一致。不要将旧profile矩阵混入结果。

以下是三种核心时间口径，全部含提交至完成的host wall-clock，不是纯CCU device event时间；第9节版本另外保留host_counts_serial和counts_only两个诊断口径：

| 指标 | 输入计数 | 每次是否规划 | 用途 |
|---|---|---|---|
| prepared | 已准备 | 否 | 同一计划的执行差异 |
| host_counts | 完整C已在CPU | 是（native无relay规划） | 隔离获取计数之外的CPU转换、规划、校验、提交开销 |
| dynamic | 每轮device本地8计数→gather→D2H | 是 | 与旧微基准一致的全路径对照 |

`host_counts`不是省略真实调用中必要的计数开销；它用于归因。模型已具备全局专家计数时应复用，不应重复测试中的gather。prepared可连续提交，dynamic存在计数同步，三者差值不能严格拆成各个独立阶段时间。不要再把HBM pipeline的workspace/chunk耗时套到EID后端。

每轮取八rank最大耗时再跨轮取中位数。汇总器输出 `native/multipath`，大于1表示多路径更快。每个结果JSON新增 `plan`：actions、candidate_evaluations、峰值前后、relay字节、总物理字节、下界和stop_reason。`relay_rows`是全网总量。`ccu_launches_per_call=1`、`relay_hbm_bytes=0`不变。

先看：

1. 新策略是否真正分流；无relay时是custom direct，不是native回退。
2. 此样本是否用显著更少relay字节达到同样峰值；拓扑完整时应为4次/241行/1281峰值。
3. prepared是否改善；若没有，表明当前链路负载模型或固定通信开销仍限制收益。
4. host_counts是否改善；若仍明显慢于native，继续定位counts转换/校验/route提交，不能全归咎于搜索。
5. dynamic与端到端是否改善；prepared胜出不能替代dynamic和模型验证。

同一策略的独立进程native/multipath对照可以各跑 `--mode native`、`--mode multipath`，其他参数一致，再执行：

```bash
python3 scripts/summarize_a5_ccu_eid.py \
  /home/l00934901/profiling/a5-ep8-eid.MULTIPATH \
  --compare /home/l00934901/profiling/a5-ep8-eid.NATIVE
```

替换为实际目录。`--compare`用于同策略native基线，不用于greedy与bottleneck互比；两种策略直接对照各自同进程摘要，防止不同矩阵/拓扑/计时口径混比。

## 5. 模型接入开关

为保留旧调用行为，普通 `RelayPlanner(...)` 仍默认greedy。**模型必须显式加 `strategy="bottleneck"` 才启用此次优化。** 独立EID测试脚本的 `--planner` 默认是bottleneck。

```python
planner = RelayPlanner(
    row_bytes, allowed_relays,
    native_module=cpp_ext,
    strategy="bottleneck",
)
handle = buffer.prepare_ccu_urma_eid_alltoallv("moe-eid-bottleneck", manifest)
host = cpp_ext.ep8_eid_create(planner._native, rank, buffer.runtime, handle)
# 每步真实CPU C；同host按实际dtype处理FP8 dispatch / BF16 combine。
host.run(dispatch_counts, fp8_send, fp8_recv)
host.run(combine_counts, bf16_send, bf16_recv)
```

不要在每步先调用 `planner.plan()` 再 `host.run()`，后者已经内部规划一次。日志和离线描述可在采样时单独调用plan，不放进性能区间。通信域、handle和host仍只初始化一次并复用，close所有权契约不变。

模型对照使用同一请求集合、DP路由、并发、batch上限、prefix-cache状态和相同的成功验证条件，分别测native、旧greedy、新bottleneck。确认真实请求期间 `in_profile_run=false`，不能只看HTTP200（异常后回退native也会200）；记录disabled/fallback次数、实际relay样本，并重复端到端数值一致性检查。

## 6. 开发验证边界

开发容器 `cam_lyw_92`，CANN9.2；目标机器CANN9.3上的数值和性能必须按第3–5节验证。日志放开发机 `/home/liuyuanwen/profiling/a5-ep8-bottleneck`；用户机器所有结果放 `/home/l00934901/profiling`。

验证覆盖真实样本、转置矩阵、随机拓扑和容量、全割瓶颈、全零/self/uniform、单热点、int64边界、失败候选回滚、策略确定性；同时保留旧greedy回归，并让实际ATen EID host用两种策略分别执行FP8/BF16交替100轮。完整DeepEP扩展按指定SDK重新编译及严格链接。编译验证不是目标NPU性能结论。

本次开发验证结果：

- 14组CPU测试共88项通过，无跳过。完整日志：`/home/liuyuanwen/profiling/a5-ep8-bottleneck/a5-public-cpu.FueurG/`。
- `cam_lyw_92` / CANN9.2：完整DeepEP扩展8个编译单元及 `--no-undefined` 严格链接成功，日志 `extension.log`。
- 实际完整扩展导入成功，FP8/BF16行宽分别得到4次动作、241行relay、峰值1281行、54次候选评估，日志 `import.log`。这一步只调用CPU规划，不启动NPU通信。
- 上述构建和导入日志均在 `/home/liuyuanwen/profiling/a5-ep8-bottleneck/`。尚未完成新机CANN9.3八卡正确性及性能验证；目标机器仍需重编安装匹配其Python/CANN环境的wheel。

## 7. 前置配置同步失败与BF16排查

现场 `a5-ep8-eid.1AWrFE` 的8个rank均到达 `config_allgather_wait_complete`，随后在 `config_allgather_sync_begin` 后失败；尚无catalog或case标记。该次失败发生在前置配置同步，不能归因为bottleneck/BF16计划非法。`Work.wait()`返回并不等于设备工作已完成。与历史失败位置相同，也不证明底层原因相同；恢复运行不能单独证明某个修复有效。

测试未设置 `relay_capacity_bytes`，不会因BF16行宽翻倍而触发一个配置的relay容量上限。显式设置容量时，host通过 `WithRowBytes()` 按实际行宽重新换算。`--suite create --dtype bfloat16`仍只是注册验证，不执行BF16数据，无法替代BF16正确性测试。

新增诊断均在计时区间外，不新增collective或设备同步：

- `results/rankN.phases.jsonl`：每rank独立阶段记录，避免多进程/SDK日志粘连；区分验证提交、验证同步、预热、测量前barrier、测量执行完成和计时all_reduce。
- `results/rankN.external.plan.json`（其他case替换external）：首次提交前保存实际行数、relay选择、行宽、指纹、字节偏移及buffer大小。已有run.log中的阶段标记继续保留。
- 原始失败日志仍应优先保留 `environment.log`、`arguments.txt`、`commit.txt` 和首个底层错误。默认日志级别不会屏蔽Python阶段标记；外层终端末尾没有标记，不代表完整run.log没有。

CPU回归新增真实673-token矩阵的生产CCU指令图回放，检查FP8/BF16交替时的地址范围、完整数据与guard；另检查同一host在固定327680字节relay容量下按两种行宽分别规划。CPU回放不模拟硬件EID转发、传输时序或HCCL启动，不能作为新机挂起已修复的证据。选路、route/kernel与all_gather后端均未因这次报告改动。

## 8. 交给DeepSeek：纠正dynamic归因并补齐性能证据（2026-11-02）

### 8.1 必须纠正的结论

当前目标是 `participant_eid_ccu_urma`，不是旧HBM pipeline。先阅读本节，再解读第4节的性能表。

| 原总结 | 应改为 |
|---|---|
| 每次dynamic执行 `planner.plan()`，约30 µs | 每次dynamic调用的是C++ `host.run()`；Python `planner.plan()`用于计时外检查和报告。30 µs不是当前链路的已测分段结果。模型侧也不得先plan再run而重复规划。 |
| 每次layout/chunk/workspace约55 µs | 当前没有chunk切分或relay HBM workspace。route仍计算字节偏移、复制资源和构造参数，但其耗时未被单独测出。旧HBM执行器耗时不可移植到EID。 |
| Python开销约15 µs | 这是总差减去其他估计值的余额，不是实测Python耗时。 |
| native dynamic−prepared=205.1 µs，所以计数获取恰好205.1 µs | 差值混合了计数、D2H、分配及同步边界改变后的重叠效果。两种实现调用同一计数函数，实际等待时间也未必相同。 |
| multipath多100 µs，全部属于规划+布局 | 需补齐custom direct和host_counts。route参数构造、token获取、事件桥和CCU提交在prepared与dynamic中都存在。 |
| prepared 1.009x证明稳定获益 | 约1.6 µs、0.9%的差异需独立进程重复及逐轮数据确认。prepared包含主机提交和完成等待，不是纯链路或纯kernel时间。 |
| 峰值降低6.6%，所以必然节省约11 µs | 仅在时间完全由该边等带宽负载决定时可这样估算；固定开销、共享资源和转发时序会改变收益。 |
| 241行已接近全局最优 | 对该样本、目标1281行，4条原始超限边至少需移走241行，当前方案达到这个目标所需的行数下限；不证明最优目标、最优时延或全局最优路由。 |
| config_allgather挂起证明BF16计划非法，或证明仍是HCCL_BUFFSIZE导致 | 该次尚未进入规划/建链；故障位置明确，底层原因未确认，后来恢复也不能单独证明根因。 |

开发机当前C++规划微基准（`cam_lyw_92`、`-O2`、真实样本、复用Plan、5组×10000次、每组均值取中位数）：direct约0.09 µs、greedy约4.38 µs、bottleneck约2.69 µs。它仅测 `Planner::MakeInto()`，不含Python入口、NPU、HCCL或route，且为缓存较热的重复矩阵；不能当作目标机器或真实请求的端到端耗时。

此次只增加文档。固定参数存储、减少资源/layout复制、连续CPU计数入口是下一轮优化候选，尚未实现，不能写成当前版本的已完成项。

### 8.2 第一优先：从已有结果提取，不占NPU

在新机容器的仓库根目录运行。查找真实673-token样本、`mode=all`、8rank完整成功的结果；每个策略/dtype取最近一份成功记录，打印路径、提交、三口径九个耗时和逐轮值。不会启动Torch或占卡。较新的失败运行不会覆盖旧成功结果，务必保留实际路径与提交，不把旧成功结果称作刚复跑。

```bash
cd /home/l00934901/sgl-kernel-npu
python3 - <<'PY'
from pathlib import Path
import json, sys
sys.path.insert(0, 'scripts')
from summarize_a5_ccu_eid import load
root = Path('/home/l00934901/profiling')
fixture = json.loads(Path('tests/data/ep8_prefill_673.json').read_text())['counts']
latest = {}
for path in sorted(root.glob('a5-ep8-eid.*/results/rank0.json'), key=lambda p: p.stat().st_mtime):
    run = path.parents[1]
    try:
        raw = json.loads(path.read_text()); opt = raw['options']
        if opt.get('mode') != 'all' or opt.get('suite') == 'create' or opt.get('width') != 5120: continue
        if not any(r.get('counts') == fixture for r in raw['reports']): continue
        result, topology = load(run)  # 校验exit.status、8rank报告和PASS标记
    except (OSError, ValueError, KeyError, TypeError):
        continue
    latest[(opt.get('planner', 'greedy'), opt['dtype'])] = (run, result)
for key in [('greedy','float8_e4m3fn'), ('bottleneck','float8_e4m3fn'),
            ('greedy','bfloat16'), ('bottleneck','bfloat16')]:
    if key not in latest:
        print('MISSING_RUN', key); continue
    run, result = latest[key]; opt = result['options']
    print('\nRUN', run, 'policy/dtype=', key)
    print('COMMIT', (run/'commit.txt').read_text().strip() if (run/'commit.txt').exists() else 'unknown')
    print('OPTIONS', {k: opt.get(k) for k in ('mode','width','warmup','iters','rounds','dynamic_iters')})
    if (run/'environment.log').exists(): print((run/'environment.log').read_text().strip())
    for report in result['reports']:
        if report['counts'] != fixture: continue
        plan = report.get('plan', {})
        print('PLAN', 'relay_rows=', report['relay_rows'], 'executed_relay=', report['executed_relay'],
              'actions=', len(plan['actions']) if 'actions' in plan else 'unknown',
              'evaluations=', plan.get('candidate_evaluations'),
              'peak_bytes=', (plan.get('direct_peak_bytes'), plan.get('planned_link_peak_bytes')))
        for metric in ('prepared', 'host_counts', 'dynamic'):
            values = report['median_us'].get(metric, {})
            if any(m not in values for m in ('native','direct','multipath')):
                print('MISSING_METRIC', metric); continue
            print(metric, {m: round(values[m],3) for m in ('native','direct','multipath')})
            samples = report.get('samples_us', {}).get(metric, {})
            print('ROUND_US', {m: [round(v,3) for v in samples.get(m, [])]
                               for m in ('native','direct','multipath')})
PY
```

优先回传bottleneck的FP8、BF16两份完整输出。最缺的是 `host_counts native/direct/multipath` 和 `dynamic direct`；同轮prepared三项、逐轮样本、relay动作/行数、结果目录也要保留。不要只贴比值。旧greedy结果可参考，但核对wheel、参数和拓扑之前，不作为严格策略A/B结论。

### 8.3 缺项或需要验证稳定性时，才补跑NPU

若之前遇到dtype重复参数错误，先按8.6更新并短验证，再运行本节。

使用第2节同一套环境和已验证的wheel/route。文档更新不需要重编；旧wheel缺 `ep8_bottleneck_planner_abi=1` 才需按第2节重编wheel。无需重复注册探针或完整7case扫描，benchmark本身先做数据正确性检查。

本轮只测bottleneck：进程内轮换native/direct/multipath，独立启动3轮并交替FP8/BF16顺序。不打profiler、不启动模型，不与其他八卡任务重叠。已有greedy结果无需再补一遍。

```bash
cd /home/l00934901/sgl-kernel-npu
(
set -Ee -o pipefail
export A5_OUTPUT_ROOT=/home/l00934901/profiling
export HCCL_BUFFSIZE=64
export A5_EID_LOG_LEVEL=3 A5_EID_LOG_STDOUT=0
export A5_CCU_RESOURCE_DIAGNOSTICS=0
unset A5_EID_RELAY_SCHEDULE
mkdir -p "$A5_OUTPUT_ROOT"
A5_AUDIT=$(mktemp -d "$A5_OUTPUT_ROOT/ep8-dynamic-audit.XXXXXX")
for A5_REPEAT in 1 2 3; do
  if (( A5_REPEAT % 2 )); then
    A5_DTYPES=(float8_e4m3fn bfloat16)
  else
    A5_DTYPES=(bfloat16 float8_e4m3fn)
  fi
  for A5_DTYPE in "${A5_DTYPES[@]}"; do
    A5_LOG="$A5_AUDIT/repeat${A5_REPEAT}_${A5_DTYPE}.log"
    if bash scripts/run_a5_ccu_eid_alltoallv.sh \
      --suite benchmark --mode all --planner bottleneck \
      --matrix-json tests/data/ep8_prefill_673.json \
      --dtype "$A5_DTYPE" --width 5120 \
      --warmup 3 --iters 10 --rounds 6 --dynamic-iters 100 \
      > "$A5_LOG" 2>&1; then
      tail -25 "$A5_LOG"
    else
      A5_RC=$?
      echo "FAILED rc=$A5_RC log=$A5_LOG"
      tail -100 "$A5_LOG"
      exit "$A5_RC"
    fi
  done
done
echo "AUDIT_DIR=$A5_AUDIT"
)
```

失败立即停止并保留目录，不靠循环重试挑成功样本。64只是固定对照配置，不保证规避任何同步故障；若再停在config_allgather，按第7节回报，不写成算子执行失败。`--mode`只选择计时模式，计时前正确性检查仍执行三个模式，不能把单独 `--mode native` 当作完全不执行custom的隔离探针。

实际结果目录在外层log第一行“结果目录”中。回传六份log末尾的完整汇总及各自结果目录，不只回传8.2选出的最近一轮。三轮FP8或三轮BF16分别比较，不跨dtype混合取中位数。所有结果位于 `/home/l00934901/profiling`。

### 8.4 能判断什么，暂时不能判断什么

| 对照 | 用途 | 不能声称 |
|---|---|---|
| host_counts multipath vs direct | 不含计数交换时，启用规划和relay后的整体差异 | 差值全是planner CPU时间，两者传输计划也不同 |
| host_counts direct vs native | custom调用/提交/执行整体与原生的差异 | 差值全是route布局或event桥 |
| 同模式host_counts vs prepared | 新C输入、规划/校验等动态host路径的影响 | 精确拆分Python、规划、校验各多少µs |
| 同模式dynamic vs host_counts | 加上计数交换和D2H后的实际影响 | 差值等于独立D2H耗时 |
| 三次独立进程的逐轮数据 | 0.9%变化是否可重复、是否受启动次序影响 | 已证明模型端到端获益 |

每个模式的逐轮数值已经取八rank最大值，各模式的最慢rank可能不同；相减只作工作负载对照，不是精确CPU阶段剖析。

本轮不要求profiler或修改SDK。先拿齐九项耗时，再决定是否增加route内部细分计时。脚本尚未单独输出token查询、资源复制、布局、事件桥、DFX、kernel提交六项耗时，不能把推算值当实测。`host.run()`返回 `[规划和host校验耗时, 0, 提交耗时]`，不含进入C++前的Python计数转换，提交时间也不是设备完成时间；中间的0不能命名为EID布局耗时。

先不重启大模型。归因完成后，再按第5节使用profile结束后的真实dispatch C、combine C分别测试。把同一个dispatch C换成BF16是在隔离行宽变量，不能称为真实combine结果；每层真实C变化时也不能沿用旧计划。

### 8.5 回报模板

```text
分支/提交：
wheel与route路径/版本（同一套）：
HCCL_BUFFSIZE、devices、拓扑/manifest目录：
实际结果目录（每次独立启动一项）：
case、dtype、row_bytes、warmup/iters/rounds/dynamic_iters：
每轮是否全部通过；若失败，最后阶段和首个底层错误：

metric       native_us   direct_us   multipath_us
prepared
host_counts
dynamic

每轮原始samples_us：
relay_rows / actions / candidate_evaluations / peak_before→after：
三次独立启动的差异：
说明：仅真实样本通信微基准；不把估算阶段成本写成实测，不将BF16同C测试称为真实combine。
```

### 8.6 dtype日志冲突修复后，DeepSeek按此顺序恢复测试

**先读本节，再执行8.3。无需stash或回退9102cc3。** `ede46d4`新增日志时，`dict(dtype=args.dtype, **fields)`与配置探针传入的 `fields['dtype']='torch.int32'`重复，触发Python运行时TypeError；`10bed4b`继承了这段脚本。此前“9102cc3到10bed4b只差文档”的表述不准确，中间包含诊断脚本变更。wheel/route的算子实现则没有因这次日志修复变化。

修复把阶段日志提取成纯Python的 `make_phase_logger()`，先创建默认字段，再显式update阶段字段：

- `dtype`表示阶段传入的dtype，未传入时用payload默认值。
- 新增 `payload_dtype`，始终保留配置的FP8/BF16数据类型。配置探针可同时记录 `dtype=torch.int32` 和 `payload_dtype=bfloat16`，不再冲突。
- 保留逐rank日志和原始失败信息，不捕获吞掉通信异常、不自动重试、不跳过配置检查。

**步骤1：更新测试代码，无需重编。** 开发服务器push后，新机容器执行：

```bash
cd /home/l00934901/sgl-kernel-npu
git status --short
git fetch origin feature/a5-ccu-ep8-bottleneck-alltoallv
git switch feature/a5-ccu-ep8-bottleneck-alltoallv
git merge --ff-only origin/feature/a5-ccu-ep8-bottleneck-alltoallv
git log -1 --oneline
```

保留自己的接入改动；若fast-forward失败，先处理分支差异，不用reset或stash回退旧脚本。继续使用已通过验证、带bottleneck能力的现有wheel和EID route。

**步骤2：先跑不占NPU的日志回归。** 下面2项测试直接组合正式日志函数与配置同步探针，用模拟通信检查正常/异常路径。FP8和BF16都覆盖，不导入Torch，不启动进程组。

```bash
PYTHONPATH="tests/python/deepep:python/deep_ep/deep_ep${PYTHONPATH:+:$PYTHONPATH}" \
python3 -m unittest \
  test_a5_ccu_eid_cpu.EidTests.test_phase_logger_collective_dtype_override \
  test_a5_ccu_eid_cpu.EidTests.test_phase_logger_preserves_collective_failure
```

要求 `Ran 2 tests`、`OK`。开发容器已通过这2项及原有的2项配置探针回归，共4项；这是Python日志回归，不是目标机器HCCL通过的证明。

**步骤3：八卡空闲时，仅做一轮BF16短验证。** 沿用第2节环境，在仓库根目录执行：

```bash
export A5_OUTPUT_ROOT=/home/l00934901/profiling
export HCCL_BUFFSIZE=64
export A5_EID_LOG_LEVEL=3 A5_EID_LOG_STDOUT=0
export A5_CCU_RESOURCE_DIAGNOSTICS=0
unset A5_EID_RELAY_SCHEDULE
bash scripts/run_a5_ccu_eid_alltoallv.sh \
  --suite benchmark --mode all --planner bottleneck \
  --matrix-json tests/data/ep8_prefill_673.json \
  --dtype bfloat16 --width 5120 \
  --warmup 1 --iters 2 --rounds 1 --dynamic-iters 2
```

要求最终 `EP8_EID_ALL_RANKS_AND_EXIT_PASS`。只看config_allgather完成或注册成功还不够。这轮用来快速检查脚本和完整数据路径，不用于性能结论，不与正式6轮×100次结果混合；8.2自动提取若选中了它，会显示不同OPTIONS，应据此排除。

**步骤4：短验证成功后，原样执行8.3，再按8.5回报。** FP8/BF16各3次独立启动，共6份成功记录；不重新启动DeepSeek模型，不额外开profiler。目录 `ep8-dynamic-audit.153535`中的旧失败记录保留，不覆盖、也不算性能样本。新命令使用mktemp建立新目录。

8.3已有失败即停止的逻辑，按整个命令块运行；若第一份失败，不要继续手工补跑其余5份。回报实际结果目录、最后阶段及首个异常；若是设备同步错误，要与本次Python dtype重复参数错误区分。

## 9. Dynamic 计数路径优化：复用缓冲区与显式 begin/finish

### 9.1 新证据与本次改动的边界

新审计的 `host_counts_serial` 中 native/multipath 基本持平，而 dynamic 仍有差距。这不支持“仅逐次同步就能解释 custom/native 差距”；应继续检查计数获取与后续提交的组合路径。它**尚未证明**特定 stream、event 或 HCCL 内部资源是根因。`counts_only` 约300µs不能直接与其他口径相加减；`host_counts` 包含完整传输，绝不是“只测计数”。

本次保持当前层真实C、原all_gather协议、EP通信组、CCU+URMA EID转发和一调用一launch不变。增加：

- `RoutingCountsExchange`：复用一个device int64[8,8]、8个行view、一个pinned CPU int64[8,8]及其flat view。省去调用侧每次的8个张量分配、`stack`和CPU张量分配；HCCL内部是否仍分配临时空间不在保证范围内。
- `finish()` 在HCCL Work等待之后提交非阻塞D2H，以完成event同步后才把CPU计数交给规划器；没有跳过数据依赖。
- `host.run_counts_tensor(cpu_counts, send, recv)`：直接检查并复制64个CPU int64，避免`.tolist()`、Python normalize和逐个PyLong转换。原 `run(list_or_tuple, ...)` 接口保留，原host ABI不变，新增能力 `ep8_eid_host_counts_tensor_abi=1`。
- 可选独立计数流：`begin()`捕获真实producer流的完成依赖；`finish()`在A2A前取回当前层C。模型可在二者间提交不依赖全局C的当前层permute工作。**不是预测下一层C，不复用上一次计划，不增加进程组。** HCCL自身的流实现及模型依赖仍可能限制重叠。

提供三种 `--counts-path`，默认保留 `legacy`，不未经目标NPU验证就替换模型：

| 选项 | count交换/内存 | custom入口 |
|---|---|---|
| legacy | 原gather + stack + cpu + normalize | run(tuple) |
| reuse | 复用缓冲区，原caller流 | run_counts_tensor |
| stream | 复用缓冲区，专用计数流，保留producer依赖 | run_counts_tensor |

三种模式的native使用同样的计数交换路径，之后转换为HCCL split接口所需的列表；不能只优化multipath的计数路径、却拿legacy native比较得出加速结论。`--compare`拒绝不同counts_path的独立结果配对；跨path应逐格对照，并报告同path native/multipath。

### 9.2 更新、构建和不占卡的检查

1. 按第8.6步骤1更新本分支。现场自改过五口径脚本时先保留diff并合并；新版本已包含 `host_counts_serial` 和 `counts_only`，不要覆盖回旧文件。
2. **按 `docs/A5_CCU_92_93_ALLTOALL.md` 第4节重编安装DeepEP wheel**，本次新增C++接口，不能沿用edb42f5测试时的旧wheel。已通过的EID route无需重编。本次没有改SDK注册、CCU构图或转发协议。
3. 沿用第2节环境，执行CPU计数调度/错误处理回归：

```bash
PYTHONPATH="tests/python/deepep:python/deep_ep/deep_ep${PYTHONPATH:+:$PYTHONPATH}" \
python3 -m unittest test_a5_routing_counts -v
```

要求5项通过。它们模拟通信调度和数据，检查100次变化C、缓冲区复用、等待顺序、失败保留引用和关闭路径；不能代替NPU测试。NPU脚本在reuse/stream时会明确检查新wheel能力，缺少时在建组前报错。

### 9.3 先跑短验证，再测三条路径

八卡空闲时在目标容器执行。保持原有拓扑环境，结果全部写到个人profiling目录；失败立即停止，不用连续重试挑成功样本。

```bash
(
set -Ee -o pipefail
cd /home/l00934901/sgl-kernel-npu
export A5_OUTPUT_ROOT=/home/l00934901/profiling
export HCCL_BUFFSIZE=64
export A5_EID_LOG_LEVEL=3 A5_EID_LOG_STDOUT=0
export A5_CCU_RESOURCE_DIAGNOSTICS=0
unset A5_EID_RELAY_SCHEDULE
A5_COUNTS_SMOKE=$(mktemp -d "$A5_OUTPUT_ROOT/ep8-counts-smoke.XXXXXX")
for A5_COUNTS_PATH in reuse stream; do
  for A5_COUNTS_DTYPE in float8_e4m3fn bfloat16; do
    bash scripts/run_a5_ccu_eid_alltoallv.sh \
      --suite correctness --mode all --planner bottleneck \
      --counts-path "$A5_COUNTS_PATH" --dtype "$A5_COUNTS_DTYPE" \
      --width 5120 --row-scale 1 --warmup 1 --iters 2 --rounds 1 --dynamic-iters 2 \
      > "$A5_COUNTS_SMOKE/${A5_COUNTS_PATH}_${A5_COUNTS_DTYPE}.log" 2>&1 || {
        tail -100 "$A5_COUNTS_SMOKE/${A5_COUNTS_PATH}_${A5_COUNTS_DTYPE}.log"; exit 1;
      }
    tail -20 "$A5_COUNTS_SMOKE/${A5_COUNTS_PATH}_${A5_COUNTS_DTYPE}.log"
  done
done
echo "$A5_COUNTS_SMOKE"
)
```

要求4次均 `EP8_EID_ALL_RANKS_AND_EXIT_PASS`。每个case计时前还有3个不同C的 `counts_exchange_validate_complete`，用于检查计数缓冲区未缓存旧值、rank顺序和跨流依赖。原输入/输出guard及queued通信验证保留。

短验证完成后，先各做一次六轮正式测试，比较同一新wheel的legacy/reuse/stream：

```bash
(
set -Ee -o pipefail
cd /home/l00934901/sgl-kernel-npu
export A5_OUTPUT_ROOT=/home/l00934901/profiling
export HCCL_BUFFSIZE=64 A5_EID_LOG_LEVEL=3 A5_EID_LOG_STDOUT=0
export A5_CCU_RESOURCE_DIAGNOSTICS=0
unset A5_EID_RELAY_SCHEDULE
A5_COUNTS_AUDIT=$(mktemp -d "$A5_OUTPUT_ROOT/ep8-counts-audit.XXXXXX")
for A5_COUNTS_DTYPE in float8_e4m3fn bfloat16; do
  for A5_COUNTS_PATH in legacy reuse stream; do
    bash scripts/run_a5_ccu_eid_alltoallv.sh \
      --suite benchmark --mode all --planner bottleneck \
      --counts-path "$A5_COUNTS_PATH" \
      --matrix-json tests/data/ep8_prefill_673.json \
      --dtype "$A5_COUNTS_DTYPE" --width 5120 \
      --warmup 3 --iters 10 --rounds 6 --dynamic-iters 100 \
      > "$A5_COUNTS_AUDIT/${A5_COUNTS_PATH}_${A5_COUNTS_DTYPE}.log" 2>&1 || {
        tail -100 "$A5_COUNTS_AUDIT/${A5_COUNTS_PATH}_${A5_COUNTS_DTYPE}.log"; exit 1;
      }
    tail -35 "$A5_COUNTS_AUDIT/${A5_COUNTS_PATH}_${A5_COUNTS_DTYPE}.log"
  done
done
echo "$A5_COUNTS_AUDIT"
)
```

保留全部5口径×3实现的 `samples_us` 和 `median_us`、新wheel版本、counts_path及每次结果目录。`counts_only`的mode标签只表示轮换位置，不调用该模式的A2A，不应宣称“counts_only多路径加速”。它计时外校验C，不检查未写入的payload。

先看reuse/stream能否降低**同path下multipath−native的dynamic差距**，以及各自相对legacy的绝对耗时。若某路径有可重复改善，再把该路径和legacy以交替顺序各补2次独立启动；没有改善时回报全部结果，不直接接模型。此次微基准begin后立即finish，没有permute可重叠；它测复用和流路径，不代表模型重叠收益。

### 9.4 模型接入契约（NPU短验证通过后）

```python
from deep_ep.routing_counts import RoutingCountsExchange

# 每个模型通信组创建一次；两种payload dtype不影响int64计数。
exchange = RoutingCountsExchange(local_counts_example, group=ep_group,
                                 dedicated_stream=True)

# 当前层router完成，已知真实packed token行数 local_counts[8]。
# begin必须在生成local_counts的stream上下文中调用。
exchange.begin(local_counts)
# 仅可在此执行不需要完整C的当前层工作，例如满足自身依赖的pack/permute。
# 不在同一进程组中插入乱序collective；所有rank必须维持相同collective顺序。
cpu_counts = exchange.finish()  # 已完成D2H；返回借用的CPU int64[64]
host.run_counts_tensor(cpu_counts, packed_send, output)

# 停止提交后、通信组和NPU runtime销毁前：
exchange.close()
host.close()
# 再按原契约销毁plan/Buffer/通信组。
```

`cpu_counts`只在下次begin前有效，若保存历史应clone；C++入口当场复制64个值，不保留tensor指针。begin至finish之间不得修改input计数，且一个exchange只允许一个未完成任务。出错后不继续复用。不要依赖Python析构在NPU runtime销毁后清理。

不使用独立stream时选择 `dedicated_stream=False`，begin要求原caller流。两个方向必须顺序使用exchange；若应用确实并发调用，分别建计数对象并明确集合通信排序。host的原caller流要求仍然有效，不能把EID执行搬到计数流上。若模型已经持有同一调用的完整CPU C，可直接使用 `run_counts_tensor`，不重复交换；若仅有本地splits则仍需要获取全局C。

说明：保持跨流tensor引用直到完成，且D2H完成前不读取CPU缓冲区，遵循[PyTorch跨流内存生命周期说明](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html)；NPU具体Event接口已对照开发容器的torch_npu SDK。性能和HCCL流行为仍需9.3.0目标机实测。

### 9.5 开发侧验证记录

在专用 `cam_lyw_92`（CANN9.2）完成：

- 19项CPU回归通过，包括5项计数交换调度/生命周期模拟测试、生产CCU图解释测试及实际ATen C++ host测试。后者覆盖变化C、FP8/BF16各100次交替、CPU tensor计数复制与非法输入拒绝。
- 完整8个C++编译单元编译通过，`--no-undefined`严格链接通过。完整扩展导入通过，新增counts tensor ABI和原接口同时存在。
- 文档命令通过bash语法检查。

开发日志：`/home/liuyuanwen/profiling/a5-ep8-bottleneck/counts-cpu.log`、`counts-extension.log`、`counts-import.log`。这不是9.3.0八卡NPU性能结果；独立流的HCCL依赖、pinned D2H和实际收益仍按9.3节验收。目标机须重打自己的cp311 wheel，不能使用开发侧cp312编译产物。
