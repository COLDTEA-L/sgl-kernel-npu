# EP8 稠密矩阵：轻载卡 READY 后 EID 转发

分支：`feature/a5-ccu-ep8-dense-ready-alltoallv`，基于 `feature/a5-ccu-92-93-alltoall` 的 `121e8f4`。原 2/4 卡接口和 EP8 默认 greedy 策略保留，原分支不修改。

## 1. 算法和边界

新策略 `dense_ready` 的规划、布局和执行入口均在 C++。不估计微秒、不拟合带宽、不做迭代搜索：扫描 64 个 counts，最多检查 6 个组内合法 relay。本版每次通信**全局至多选一条流、一个 relay**，先验证小比例分流和通知开销；不是多热点最优算法。

排除 self 后，H 为唯一最大流 C[i,j]，S 为其他流的最大值，B[k] 为卡 k 所有跨卡发送和接收流的最大值：

```
x = floor((H - B[k]) / divisor)
after = max(S, H-x, B[k]+x)
```

只考虑拓扑允许且 k!=i,j、B[k]<H 的 relay。x 受容量和最小行数约束；after<H 才接受。选择降幅最大者，平局优先更小 x，再按 rank 固定打破平局。最大流并列时保持直连：本版仅拆一条流，无法降低剩余并列峰值。默认 divisor=8，即原 /2 分流的四分之一，可扫 2/4/8/16，允许范围 2–1024。按行计算再乘 row_bytes，等价于字节公式按整行向下取整。

B[k]+x 是等带宽、独立链路下的流量代理，**不是实际完成时间保证**。端口争用和 READY 开销仍需实测。无 relay 时走自定义 EID direct，不自动回退 native；端到端仍可能比 native 慢。

### 真实采样与合成用例

`observed_dense` 精确使用以下行重复 8 次：

```
[3048,3165,3067,3017,3071,3028,3046,3134]
```

共 196608 行，FP8 width=5120 时含 self 为 960 MiB，跨卡 840 MiB，单流平均 15 MiB。rank1 的 7 条入链路均为 3165 行，入方向割下界也是 3165，等于直连峰值；所有候选卡 B[k] 也不小于 3165。**预期 executed_relay=false**，不能强行把它当多路径收益用例。该下界解释以等容量独立有向链路为前提。

| case | 定义 | divisor=8 的预期 |
|---|---|---|
| dense_near（合成） | 所有元素3017，仅0→1为3165 | 拓扑允许时分18行：FP8为90 KiB，BF16为180 KiB |
| dense_hot（合成） | 所有元素3017，仅0→1为4096 | 拓扑允许时分134行 |

这三个用例的 counts **不随 --row-scale 缩放**。继续保留 `--matrix-json` 真实64元素 counts 输入（packed 行数），不从历史热度推断。

## 2. READY 协议：仍然不落 relay HBM

1. 所有 rank 先提交全部直连写，包括热点流保留的直连部分。
2. 非 helper rank 仅等待自己发往 helper 的直连写完成，再发 DIRECT_TO_HELPER_DONE；不等待自己发往其他卡的热点流。
3. helper 等自身所有跨卡直连发送完成及另外7卡的上述通知，广播 READY。
4. 热点源收到 READY，通过预建 EID channel 提交 x 行；热点直连仍可能在进行。
5. 等数据操作完成，再执行原最终完成握手。

helper 的“自身完成”指跨卡收发，self-copy 不作为 READY 条件。全局仅一条 relay 流且 helper 不是该流端点，不存在互等 relay 的环。源 HBM → CCU+URMA/EID 物理转发 → 目的 HBM；无 relay HBM、pack/unpack、新通信域、CPU完成轮询或分块提交。

复用 direct channel notify slot2：mask1最终完成、mask2发往helper的直连完成、mask4 READY。仍使用14个写完成event，已消费event不重复wait。task args仍46个，slot10在READY模式传helper rank，8代表无relay。只用生产已有EQ/NE，不用历史失败的GE动态mask解码。无relay时跳过READY握手。

## 3. 拉取并重编 route 和 wheel

首次拉取：

```bash
cd /home/l00934901/sgl-kernel-npu
git fetch origin feature/a5-ccu-ep8-dense-ready-alltoallv
git switch --track origin/feature/a5-ccu-ep8-dense-ready-alltoallv
```

这次**必须重编 route 和 DeepEP wheel**，不能复用 f1859d2 wheel。执行 [9.2/9.3指导文档](A5_CCU_92_93_ALLTOALL.md) §3构建命令和§4 wheel命令；**跳过§3开头切回旧分支的 fetch/switch**，始终留在本新分支。

新增能力标记：route `A5CcuEidReadyRelayAbiVersion=1`，DeepEP `ep8_dense_ready_planner_abi=1`。测试及host构造检查能力与handle模式，避免新planner误配旧immediate kernel。

在目标容器设置（Atlas路径按实际文件调整）：

```bash
export A5_CANN_PATH=/usr/local/Ascend/cann-9.3.0
export A5_CCU_ROUTE_PROBE_LIB="$A5_CANN_PATH/opp/vendors/cust/lib64/liba5_ccu_urma_route_probe.so"
export A5_OUTPUT_ROOT=/home/l00934901/profiling
export A5_DEVICES=0,1,2,3,4,5,6,7
export A5_ROOT_INFO=/etc/hccl_rootinfo.json
export A5_ATLAS_JSON=/usr/local/Ascend/driver/950/atlas_950_1.json
export HCCL_OP_EXPANSION_MODE=CCU_SCHED HCCL_BUFFSIZE=64
export A5_EID_MAX_RELAYS_PER_PEER=6 A5_CCU_RESOURCE_DIAGNOSTICS=0
```

选路继续由rootinfo、Atlas、实时hccn状态联合生成，不手写EID txt。

## 4. 先注册，再正确性和长跑

```bash
bash scripts/run_a5_ccu_eid_alltoallv.sh \
  --suite create --register-stage full --planner dense_ready
```

必须看到 `EP8_EID_CREATE_ALL_RANKS_AND_EXIT_PASS`。这是新READY图，不能用旧full PASS替代。

```bash
bash scripts/run_a5_ccu_eid_alltoallv.sh \
  --suite correctness --mode all --planner dense_ready --dense-split-divisor 8 \
  --dtype float8_e4m3fn --width 5120 --row-scale 1 \
  --rounds 1 --iters 10 --dynamic-iters 100 --bidirectional-iters 100
```

应有 `EP8_EID_ALL_RANKS_AND_EXIT_PASS`，原7用例加3稠密用例共10个case×8rank。双向验证同host交替FP8/BF16。核对dense_near/dense_hot实际发生relay，observed_dense无relay；拓扑无候选时先处理拓扑，不绕过校验。

## 5. 同参数性能对照与分流扫描

```bash
for A5_CASE in observed_dense dense_near dense_hot; do
  for A5_DIV in 2 4 8 16; do
    bash scripts/run_a5_ccu_eid_alltoallv.sh \
      --planner dense_ready --dense-split-divisor "$A5_DIV" \
      --mode all --case "$A5_CASE" --dtype float8_e4m3fn --width 5120 \
      --warmup 3 --iters 10 --rounds 6 --dynamic-iters 100
  done
done
```

自动写入 `/home/l00934901/profiling/a5-ep8-eid.*`，不打profiler。另以 `--dtype bfloat16 --width 5120` 同参数测combine字节量。divisor是分流比例，不是chunk大小。

分别比较prepared/dynamic的native、direct、multipath。dynamic包含共同的计数交换及各自调用开销；prepared不含每次规划。查看relay_rows、executed_relay、candidate_evaluations和正常退出。无relay的direct/multipath是A/A；native才是目标通信baseline，均不包含router、permute、专家计算。

独立进程对照：同参数分别运行 `--mode native` 和 `--mode multipath`，将真实结果目录代入：

```bash
python3 scripts/summarize_a5_ccu_eid.py \
  /home/l00934901/profiling/a5-ep8-eid.MULTIPATH \
  --compare /home/l00934901/profiling/a5-ep8-eid.NATIVE
```

两次planner/divisor、矩阵、dtype、width、轮数及拓扑须相同。不能用不同divisor送入compare。没有预设10%/20%收益，小流量差异可能被通知成本抵消。

## 6. 模型接口

创建handle之前设置 `A5_EID_RELAY_SCHEDULE=ready`，所有rank一致。模式在create时固定，改env不能切换已有handle。每个通信组/stream仍只准备一次：

```python
os.environ['A5_EID_RELAY_SCHEDULE'] = 'ready'
handle = buffer.prepare_ccu_urma_eid_alltoallv('ep8-dense-ready', manifest)
planner = RelayPlanner(row_bytes, allowed_relays, native_module=cpp_ext,
                       dense_split_divisor=8)
host = cpp_ext.ep8_eid_create(planner._native, rank, buffer.runtime, handle)
host.run(counts, send, recv)
```

counts为本轮全局8×8 packed行数。同host可交替FP8/BF16，WithRowBytes保留dense策略及按实际字节应用的阈值。没有消除输入counts的获取成本：已有CPU counts就复用，仅有设备本地counts则仍需现有交换/D2H；本次不新增第二轮计数交换。

关闭契约不变：stream/设备/通信域存活时先host.close，再由handle唯一所有者销毁plan，最后销毁通信域。不要重复销毁共享handle。

## 7. 开发验证边界

日志在开发机 `/home/liuyuanwen/profiling/a5-ep8-dense-ready`。CANN9.2 route编译、SDK host-test和动态加载通过。生产图离线解释器覆盖immediate/READY的7/14/49通道、重复通知、零流、延迟完成event；实际ATen host检查FP8/BF16交替、字节正确性、保护区；C++ planner检查真实和随机稠密矩阵的拓扑、容量、分流边界。

本轮13组86项CPU回归全部通过；READY图额外检查拒绝多条relay流的task参数。完整DeepEP C++翻译单元使用官方torch_npu 2.10头文件/库编译并通过 `--no-undefined` 链接，扩展导入和新planner能力检查通过。导入使用SDK devlib，仅验证加载和CPU规划，不代表NPU可执行；未制作目标机cp311 wheel。日志为 `route.log`、`extension.log`、`extension-import.log`、`cpu-suite.log`、`eid-cpu.log`。

这些不替代9.3八卡READY注册、实际通知语义、长期稳定性或性能测试。尚无本算法真机收益数字，按§4、§5继续验证。
