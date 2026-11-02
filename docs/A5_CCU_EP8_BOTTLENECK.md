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

新测试输出三种时间口径，全部含提交至完成的host wall-clock，不是纯CCU device event时间：

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
