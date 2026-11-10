# CANN 9.2/9.3：两卡 direct 控制实验基线

分支：`feature/a5-ccu-valuewait-callback-puncture-92-93`。

本分支以原穿刺分支 `dcd8d40` 为第一父系，合入
`feature/a5-ccu-92-93-alltoall` 的 `121e8f4`。保留原 callback 调度封装，
引入公开 CCU 接口实现和独立两卡 direct smoke。原 9.1 分支保持不变。

## 状态与范围

这是 92/93 实验开发基线，**不是两卡控制实验已实现或已通过的声明**。
原 `A5ValueWaitCallbackScheduler` 仍在 arrival 后自动放行，需要增加手动/延迟 release。
原 Python 穿刺仍依赖 relay manifest 和 patched HCCL，不能用于此次两卡目标；
旧 `run_a5_valuewait_callback_puncture.sh` 在本分支提前拒绝执行，避免混用库。

实施任务和完整验收见 [Issue #13](https://github.com/COLDTEA-L/sgl-kernel-npu/issues/13)。
仅用两个通信卡、不用 relay、不维护资源矩阵。

## 版本变化

| 内容 | 9.1 历史路径 | 本分支 9.2/9.3 路径 |
|---|---|---|
| CCU 构图 | `hcomm::CcuRep` | `AscendC::ccu` |
| 注册 | `HcclCcuKernelRegister` | `HcommCcuKernelRegisterStart/Register/End` |
| 启动 | `HcclCcuKernelLaunch` | `HcommCcuKernelLaunch` |
| direct 写入/完成 | `WriteNb/WaitEvent` | smoke 的 `ccu::Write/ccu::EventWait` |
| gate/callback | 原 Runtime 封装 | 在目标 SDK 重编译并验证相同 Runtime API |

不要跨版本复用旧 wheel、route 库、patched HCCL 或 `LD_PRELOAD`。
9.2 的编译/加载结果不能代替 9.3 验证；9.3 需记录具体补丁版本及驱动、固件。

## 先检查目标 Runtime ABI（不占卡）

```bash
bash scripts/check_a5_valuewait_runtime_abi.sh \
  /usr/local/Ascend/cann-9.3.0 /tmp/a5-valuewait-93-abi
```

9.2 使用对应的实际安装路径。检查真实头文件签名、链接、加载，打印三个符号
`aclrtValueWait/aclrtValueWrite/aclrtLaunchHostFunc` 的库路径。
`VALUEWAIT_RUNTIME_ABI_PASS` 只证明声明/链接/加载，不证明上卡或图捕获支持。
检查输出路径是否属于选定 SDK；遇到旧库路径应先修正环境。

## 两卡 direct 的起点

使用 `examples/a5_ccu_public_api_smoke/main.cc`，而非原 patched-HCCL 多路径入口：

1. `SelectDirect()` 从 rank graph 选择一跳 UB_CTP。
2. `HcclChannelAcquire()` 每 rank 建立一个 direct Channel。
3. `HcommCcuKernelRegisterStart/Register/End` 在 warmup/准备阶段注册一次。
4. `HcommCcuKernelLaunch(routeThread, ...)` 提交通信。
5. `ExchangeKernel()` 执行 `Write → EventWait → NotifyRecord/NotifyWait`。
6. 原 smoke 用 stream 同步后读回校验；新实验需追加有序 DONE callback。

该程序在部分 die 情况下会另建 `routeStream`。gate、STARTED、通信、FINISHED、
reset、DONE 必须关联到实际绑定 `routeThread` 的执行流；如果使用 caller stream，
必须验证双向 event 依赖，防止通信绕过 gate 或 DONE 提前。
`ThreadEventBridge` 可参考，但其捕获兼容性仍需测试。

现有直接通信基础检查（不是控制实验）：

```bash
bash scripts/build_a5_ccu_public_api_smoke.sh \
  /usr/local/Ascend/cann-9.3.0 /tmp/a5-direct-93-build
(
  set +u
  unset LD_PRELOAD
  source /usr/local/Ascend/cann-9.3.0/set_env.sh
  /tmp/a5-direct-93-build/a5_ccu_public_api_smoke --check-abi
  python3 scripts/run_a5_ccu_public_api_smoke.py \
    --binary /tmp/a5-direct-93-build/a5_ccu_public_api_smoke \
    --devices 2,3 --bytes 4096 --iters 100 --output-root /tmp/a5-direct-93-results
)
```

两张物理卡必须可由命令行替换。先检查基线 direct，再加 eager gate，最后验证
graph capture/replay；捕获失败要单独报告，不以 eager PASS 替代。

## 新实验待实现的执行链

```text
S_exec (=实际 routeStream):
  ARRIVED callback [图外]
  ┌ graph: ValueWait(gate==1) → STARTED → direct launch
  │        → 通信真实完成 → FINISHED → gate=0
  └
  DONE callback [图外]

S_ctrl: 独立 host 线程在延迟观察后 ValueWrite(gate=1)
Host: 两 rank 本轮 DONE + 两端数据校验 → GLOBAL_DONE
```

gate 为独立 64-bit 的 0/1 变量，不需要路径权重 ABI。
callback 只更新 host 状态并唤醒线程，不调用 Runtime API。
必须验证保持期间 STARTED/FINISHED/输出未改变，且 20 次同图 replay 无旧信号穿透。
初始化、观察读回与放行必须有显式顺序；不能用全设备同步读取被阻塞图的状态。

先完成独立 C++ 控制实验，再接 DeepEP/torch_npu；后者须用当前容器的
Python/Torch/torch_npu/CANN 重编译，不能把独立 C++ graph 结果等同 NPUGraph 结果。
重建步骤参考 [92/93 迁移说明](A5_CCU_92_93_ALLTOALL.md)。
