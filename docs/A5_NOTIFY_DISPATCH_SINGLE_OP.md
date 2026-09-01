# A5 NotifyDispatch 单算子编译与通信窗口验证

本方案只编译 DeepEP 的 `NotifyDispatch`，用于验证 A5 上的 HCCL 通信域和通信窗口地址获取链路，不编译完整 dispatch/combine/FusedDeepMoe。

## 为什么选择 NotifyDispatch

`NotifyDispatchA5::InitSmallFullMesh` 执行以下操作：

```cpp
winContext_[0] =
    (__gm__ HcclOpParam *)AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
shareAddrs[rank] = GetWindAddrByRankId(rank, 0) + winDataOffset;
```

在 A5 上，`HcclOpParam` 是 `HcclCombinOpParam`，通信窗口首地址来自：

```cpp
winContext->windowsIn[rankId]
```

Host 定义通过 `MC2().HcclGroup("comm_group")` 把 EP 通信域注入设备 kernel。因此测试通过意味着 CANN 自定义算子、HCCL group、设备侧 `GetHcclContext` 和通信窗口访问链路均已工作。

测试脚本不会用 `get_dispatch_layout()` 冒充通信验证；该接口只计算本地布局。脚本手工构造布局输入后调用 `Buffer.notify_verify()`，后者会直接执行 `aclnnNotifyDispatch`。最后将 kernel 返回的本地专家接收计数与 HCCL `all_reduce` 参考值比较。

## 有卡 SGLang Docker 环境

使用 CANN 9.1 development toolkit。不要替换能运行 SGLang 的 torch/torch_npu，只补构建依赖：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
npu-smi info -t board -i 0

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  build-essential cmake git ca-certificates
python3 -m pip install --no-cache-dir pybind11 wheel pyyaml decorator scipy attrs psutil
```

确认至少两张 A5 对当前 Docker 可见：

```bash
python3 - <<'PY'
import torch
import torch_npu
print(torch.__version__, torch_npu.__version__)
print("available:", torch.npu.is_available())
print("count:", torch.npu.device_count())
PY
```

## 编译

```bash
cd /home/l00934901/sgl-kernel-npu
git checkout feature/a5-notify-dispatch-single-op
bash scripts/build_a5_notify_dispatch.sh
```

若 SGLang 使用特定 conda 环境：

```bash
bash scripts/build_a5_notify_dispatch.sh --python-env /path/to/sglang-conda-env
```

产物是只携带 `NotifyDispatch` A5 设备二进制的 `output/deep_ep*.whl`。脚本会检查 wheel，发现混入其他 A5 设备 kernel 时直接失败。

## 安装和双卡测试

安装时禁止 pip 改动现有 torch 依赖：

```bash
wheel="$(ls -1t output/deep_ep*.whl | head -1)"
python3 -m pip install --no-deps --force-reinstall "${wheel}"
```

这个 wheel 只携带 `NotifyDispatch`，不能替代生产服务需要的完整 DeepEP。请在单独的开发容器/环境中安装和测试，或在测试后恢复原来的完整 `deep_ep` wheel；不要让正在提供请求的 SGLang worker 使用它。

在一个新的 Python 进程中用两张卡测试：

```bash
export DEEP_USE_MODE=default
export HCCL_BUFFSIZE=2300
torchrun --standalone --nproc-per-node=2 \
  tests/python/deepep/test_a5_notify_dispatch_window.py
```

成功输出：

```text
PASS: A5 NotifyDispatch acquired and used the HCCL communication window
```

单卡只能取得本 rank 窗口，不能证明远端 rank 地址可访问，所以测试强制要求至少两个进程/两张 NPU。
