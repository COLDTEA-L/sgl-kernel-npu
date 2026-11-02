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
