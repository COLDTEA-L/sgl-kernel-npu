# A5 ValueWait/ValueWrite + Host Callback 穿刺实验

## 目标

在不把 host 调度逻辑捕获进图的前提下，验证一条完整控制链：

1. 两个通信 rank 默认只走 direct 路径；
2. 算子到达 device 执行点时，stream-ordered host callback 通知 host 调度器；
3. 调度线程通过独立 control stream 执行 `aclrtValueWrite`，发布路径策略并设置 READY 位；
4. 图内的 `aclrtValueWait` 看到 READY 后放行多路径算子；
5. 算子完成后，第二个 host callback 更新 host 侧完成计数；
6. 第二次 replay 把策略从 `1,0,0` 切换为 `2,1,1`，启用两条 relay。

默认物理卡为：通信卡 `2,3`，relay 卡 `0,1`。三者都可以在命令行修改。

## 时序

```text
host submit thread       captured device stream          host scheduler thread
       |                           |                               |
       | arm(policy)               |                               |
       | enqueue arrival callback  |                               |
       |-------------------------->| callback: ARRIVED             |
       |                           |------------------------------>|
       | replay graph              | aclrtValueWait(READY)          |
       |-------------------------->| blocks                         |
       |                           |       aclrtValueWrite(policy)  |
       |                           |<-------------------------------|
       |                           | multipath AllToAll             |
       |                           | copy policy to audit tensor    |
       |                           | aclrtValueWrite(direct,no READY)|
       | enqueue done callback     | callback: DONE                 |
       |                           |------------------------------>|
       | wait_done()               |                               |
       |<----------------------------------------------------------|
```

callback 只做原子状态切换和唤醒，不调用 Runtime API。CANN 明确禁止在
`aclrtLaunchHostFunc` callback 中直接或间接调用任务下发、同步、资源申请/释放等
Runtime API，因此 `aclrtValueWrite` 由单独调度线程执行。

## Policy ABI

复用现有 standard explicit-multipath 算子的 64-bit PathPolicy：

- bit 60..63：ABI，当前为 1；
- bit 0..51：最多 13 个 path 的 4-bit 权重；
- bit 59：本实验保留的 READY 位；
- `aclrtValueWait(..., 1ULL << 59, ACL_STREAM_WAIT_VALUE_AND, stream)` 等待 READY；
- HCCL 只读取 ABI 和每条 path 权重，忽略 bit 59。

图尾把 policy 写回 `ABI=1, weights=1,0,0, READY=0`，因此下一次 replay
一定重新等待 host 调度决策，不能误用上次的 READY。

## 运行

先从本分支构建并安装 DeepEP wheel，并准备包含
`A5HcclExplicitMultipathExtensionVersion >= 10` 的 patched HCCL。

默认卡位：

```bash
export A5_EXPLICIT_HCCL_LIB_DIR=/path/to/patched/hccl/lib
bash scripts/run_a5_valuewait_callback_puncture.sh
```

显式修改通信卡和 relay 卡：

```bash
bash scripts/run_a5_valuewait_callback_puncture.sh \
  --src-phy 4 --dst-phy 5 \
  --relay-phys 6,7 \
  --hccl-lib-dir /path/to/patched/hccl/lib
```

也可以修改两阶段策略：

```bash
bash scripts/run_a5_valuewait_callback_puncture.sh \
  --src-phy 2 --dst-phy 3 --relay-phys 0,1 \
  --direct-weights 1,0,0 \
  --scheduled-weights 2,1,1 \
  --hccl-lib-dir /path/to/patched/hccl/lib
```

## 成功判据

日志必须同时包含：

- `CASE_GRAPH_CAPTURE ... PASS`；
- `CASE_STAGE ... name=direct_only weights=[1, 0, 0] ... PASS`；
- `CASE_STAGE ... name=direct_plus_relay weights=[2, 1, 1] ... PASS`；
- 每个 rank 的 `arrival_count=2 release_count=2 done_count=2`；
- 每阶段 `observed_policy` 与 scheduler 的 release word 完全相等，证明图内算子窗口
  确实看到了本次 host 写入，而不是仅验证 host 侧意图；
- 最终 `PASS: ValueWait gate stayed in graph ...`。

同时检查 patched HCCL 日志中的：

```text
[ExplicitMultipath] KernelRun ... stridePolicy[...]
```

两个 replay 的 `stridePolicy` 应分别包含 `1,0,0` 和 `2,1,1`。内核已经用
`CCU_IF(pathBytes[i] != 0)` 跳过禁用路径，因此 direct-only 阶段不会提交
0-byte relay WriteNb。

## 实验边界

- 这是两 rank、1 direct + 2 relay 的控制面穿刺，不包含选路算法；
- host 只在算子执行逻辑前决策，算子开始通信后不再接受控制；
- 当前复用历史 standard dynamic-policy 算子，因此必须使用对应 patched HCCL；
- `aclrtLaunchHostFunc` 不进入图，进入图的是 `ValueWait -> operator -> reset`；
- 物理卡 0/1/2/3 不假定空闲，runner 要求使用者按机器实际空闲情况显式覆盖。
