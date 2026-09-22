# A5 CCU+URMA 两卡显式多路径 AllToAll

## 1. 范围和结论边界

本分支实现一个正式的两卡 AllToAll 数据面：一条 HCCL RankGraph 已发现的 direct CommLink，加上调用者显式
指定的若干 single-relay CommLink。当前支持最多8条总路径，因此两卡模式可覆盖`1 direct + 6 relay`。

当前版本明确限制`WORLD_SIZE=2`。四卡及更多卡的peer矩阵、每peer路径集合、并发顺序和完成协议不同，后续通过
扩展plan维度实现，不在两卡代码中隐式假设。

“显式”表示relay物理卡、对应EID pair和权重全部来自调用者的路径计划；算子不会从剩余卡中自动挑路，也不会
退化到native candidate2。当前命令行负责生成plan；正式API把plan作为函数参数传入，不再依赖
`A5_CCU_SYNTHETIC_ROUTE_MANIFEST`等进程全局环境变量。未来host控制器可以生成同一标准plan并调用相同API。

正式显式路径算子固定采用并发提交，不读取旧probe使用的`A5_CCU_ROUTE_SCHEDULE`。串行调度仅保留在旧
`multiroute` probe中作为实验对照，避免生产算子被遗留环境变量意外切换为串行。

## 2. 数据切分

输入和输出均为`[2, elements_per_peer]`的FP32 tensor：

- 本rank对应的slice通过`LocalCopyNb`复制；
- 发往peer的slice由direct和所有relay共同搬运；
- 默认要求`direct总数据 : 全部relay总数据 = 2:1`。

等权relay情况下使用：

| relay数 | Channel总数 | path weights（direct在第0项） |
|---:|---:|---|
| 2 | 3 | `4,1,1` |
| 4 | 5 | `8,1,1,1,1` |
| 6 | 7 | `12,1,1,1,1,1,1` |

例如六条relay时，direct权重12、relay总权重6，比例正好为2:1。每段按256 Byte对齐，最后一条路径接收除法和
对齐产生的余数。

## 3. 调用链

```text
run_a5_ccu_explicit_multipath_alltoall_perf.sh
  |
  +-- resolve_a5_explicit_multirelay_eids.py
  |     `-- 显式物理relay -> 双向EID pair
  |
  +-- prepare_a5_ccu_explicit_multipath_plans.py
  |     |-- direct_plus_2relay.tsv
  |     |-- direct_plus_4relay.tsv
  |     `-- direct_plus_6relay.tsv
  |
  `-- test_a5_ccu_urma_multiroute_all2all.py
        `-- Buffer.ccu_urma_explicit_multipath_alltoall_out()
              `-- deep_ep_cpp
                    `-- HcclCcuUrmaExplicitMultipathAllToAll()
                          `-- RoutePlanRequest
                                |-- SelectRoute(direct candidate)
                                |-- ApplySyntheticRouteSpec(relay EID pair[N])
                                `-- HcclChannelAcquire(desc[N+1])
                                      `-- Channel[direct, relay0..relayN]
                          `-- AllToAllMultiRouteKernel
                                |-- LocalCopyNb(self slice)
                                |-- WriteNb(direct, direct chunk)
                                |-- WriteNb(relay0, relay chunk0)
                                |-- ...
                                |-- WriteNb(relayN, relay chunkN)
                                `-- 所有WriteNb提交后统一WaitEvent
```

关键控制面API：

```c
HcclCcuUrmaExplicitMultipathAllToAll(
    send, recv, elementsPerPeer, dtype, comm, stream,
    relayManifest, directRoute, pathWeights, pathCount);
```

`GetRouteResources()`按plan内容生成cache key。第一次调用负责解析plan、建Channel并注册CCU kernel；相同
communicator、stream和plan的后续调用命中缓存，热路径不会重新解析拓扑或重新建链。

## 4. 为后续host控制器和多卡扩展保留的边界

当前`RoutePlanRequest`已经把路径控制从环境变量中分离，包含：

- discovered direct candidate；
- 有序relay manifest；
- 每条路径权重；
- plan name/cache identity。

当前relay描述仍由文件承载，便于复用已经验证的拓扑解析器。host控制器接入时可以先原样生成manifest；随后可将
`relayManifest`替换为内存中的`PathSpec[]`，而CCU kernel仍只消费已经建立的`Channel[] + weights[]`。

四卡及更多卡应把plan扩展为：

```text
Plan
  `-- peerPlans[srcRank][dstRank]
        |-- direct path
        |-- relay paths[]
        |-- weights[]
        `-- local IO die / thread group
```

不能简单把两卡的`peer=1-rank`推广到多卡，也不能让不同IO Die的Channel进入同一个现有CCU launch。

## 5. 拉取分支

```bash
cd /home/l00934901/sgl-kernel-npu
git fetch origin
git switch -c feature/a5-ccu-explicit-multipath-alltoall \
  --track origin/feature/a5-ccu-explicit-multipath-alltoall
```

若本地分支已经存在：

```bash
git switch feature/a5-ccu-explicit-multipath-alltoall
git pull --ff-only origin feature/a5-ccu-explicit-multipath-alltoall
```

## 6. 编译和安装

本次同时修改自定义CCU SO和DeepEP C++扩展，两者都必须重新安装；不需要安装KO。

```bash
cd /home/l00934901/sgl-kernel-npu
source /usr/local/Ascend/cann-9.1.T560/set_env.sh

bash scripts/build_a5_ccu_urma_route_probe.sh \
  --install \
  --install-path /usr/local/Ascend/cann-9.1.T560

bash build.sh -a deepep Ascend950
LATEST_WHEEL=$(find output -maxdepth 1 -type f -name 'deep_ep*.whl' \
  -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
test -n "${LATEST_WHEEL}"
python3 -m pip install --force-reinstall --no-deps "${LATEST_WHEEL}"
```

检查新接口：

```bash
nm -D /usr/local/Ascend/cann-9.1.T560/opp/vendors/cust/lib64/\
liba5_ccu_urma_route_probe.so | grep HcclCcuUrmaExplicitMultipathAllToAll

python3 - <<'PY'
import deep_ep.deep_ep_cpp as ext
assert hasattr(ext.Buffer, "ccu_urma_explicit_multipath_alltoall_out")
print(ext.__file__)
PY
```

## 7. 一次运行完整性能矩阵

下面以通信卡2、3为例，显式relay顺序为`4,5,1,0,6,7`。2/4 relay case分别使用此前缀，6 relay case使用
全部列表：

```bash
cd /home/l00934901/sgl-kernel-npu

bash scripts/run_a5_ccu_explicit_multipath_alltoall_perf.sh \
  --src-phy 2 --dst-phy 3 \
  --relay-phys 4,5,1,0,6,7 \
  --direct-route 0 \
  --bytes 4194304 \
  --warmup 100 --iters 20 --repeats 3 \
  --timeout-seconds 300 \
  --output-root /home/l00934901/profiling
```

每轮依次输出五组：

1. `native0`：单独使用HCCL RankGraph candidate0；
2. `native2`：单独使用HCCL RankGraph candidate2；
3. `direct_plus_2relay`：direct + relay4、5；
4. `direct_plus_4relay`：direct + relay4、5、1、0；
5. `direct_plus_6relay`：direct + relay4、5、1、0、6、7。

这里的`native0/native2`是RankGraph公开CommLink ordinal，不是UDMA `route_addr_idx`，也不是标准
`torch.distributed.all_to_all_single`算法编号。五组都使用同一个两卡CCU AllToAll数据面，因此可隔离比较路径集合。

结果：

```bash
RUN_DIR=$(ls -dt \
  /home/l00934901/profiling/a5_ccu_explicit_multipath_2_3_* | head -1)

column -s $'\t' -t "${RUN_DIR}/case_status.tsv"
sed -n '1,220p' "${RUN_DIR}/explicit_multipath_perf_report.md"
cat "${RUN_DIR}/explicit_multipath_perf_summary.json"
```

## 8. 同时采集MindStudio profiling

在上一条命令末尾增加：

```bash
  --profile \
  --profile-iters 20
```

每个case的两个rank分别写入：

```text
RUN_DIR/profiling/a5_ccu_alltoall_<case>_<run-id>/rank0/
RUN_DIR/profiling/a5_ccu_alltoall_<case>_<run-id>/rank1/
```

profiling会增加固定开销；终端性能比较使用未开启`--profile`的结果，MindStudio主要查看稳定CCU Launch的P50/P95
和不同路径数对应的设备时间线。

## 9. 正确性和性能判读

- 每个case必须输出`PASS`，否则时延无效；
- 2/4/6 relay plan必须分别包含3/5/7个Channel；
- 所有路径在通信端点必须落入同一local IO Die，否则当前版本明确返回not support；
- direct+relay的收益应同时参考终端median和MindStudio CCU Launch；
- 时延下降证明额外路径带来收益，但严格证明每张显式relay承担流量仍需另跑HCCN counter闭环；
- 若慢relay造成拖尾，可由host控制器修改`path_weights`，无需改CCU kernel。
