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

使用 CANN 9.1 development toolkit。本流程直接使用 SGLang Docker 内的系统 Python，不需要 conda，也不要替换能运行 SGLang 的 torch/torch_npu。

先确认 CANN、Python 和两张 A5：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
readlink -f /usr/local/Ascend/ascend-toolkit/latest
npu-smi info

DEEPEP_PYTHON_HOME=/usr/local/python3.11.10
"${DEEPEP_PYTHON_HOME}/bin/python3" --version
"${DEEPEP_PYTHON_HOME}/bin/python3" - <<'PY'
import torch
import torch_npu
print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("available:", torch.npu.is_available())
print("count:", torch.npu.device_count())
PY
```

只补充构建依赖：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
DEEPEP_PYTHON_HOME=/usr/local/python3.11.10

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  build-essential cmake git ca-certificates
"${DEEPEP_PYTHON_HOME}/bin/python3" -m pip install --no-cache-dir \
  pybind11 wheel pyyaml decorator scipy attrs psutil
```

## 更新开发分支

不依赖本地 remote 别名，直接从 GitHub 获取分支：

```bash
cd /home/l00934901/sgl-kernel-npu

git fetch https://github.com/COLDTEA-L/sgl-kernel-npu.git \
  feature/a5-notify-dispatch-single-op

git switch feature/a5-notify-dispatch-single-op
git merge --ff-only FETCH_HEAD

git branch --show-current
git rev-parse --short HEAD
```

第一次拉取、尚无本地分支时，使用：

```bash
git switch -c feature/a5-notify-dispatch-single-op FETCH_HEAD
```

如果 fast-forward 失败或提示存在本地修改，先执行 `git status --short`，不要使用 `git reset --hard` 覆盖本地代码。

## 使用系统 Python 编译

编译和测试必须使用同一个 Python/torch_npu 环境：

```bash
cd /home/l00934901/sgl-kernel-npu
source /usr/local/Ascend/ascend-toolkit/set_env.sh

DEEPEP_PYTHON_HOME=/usr/local/python3.11.10
bash scripts/build_a5_notify_dispatch.sh \
  --python-env "${DEEPEP_PYTHON_HOME}"
```

脚本会生成只携带一个 `NotifyDispatch` A5 设备二进制的 `output/deep_ep*.whl`，并在发现混入其他 A5 设备 kernel 时失败。确认宿主扩展和自定义 Op API 都存在：

```bash
find python/deep_ep/deep_ep output/lib -maxdepth 1 \
  -name 'deep_ep_cpp*.so' -print

test -f python/deep_ep/deep_ep/vendors/hwcomputing/op_api/lib/libcust_opapi.so
```

## 安装和双卡测试

测试脚本会优先从当前代码仓的 `python/deep_ep/deep_ep/deep_ep_cpp*.so` 自动加载刚编译的扩展，因此只做算子验证时不需要安装 wheel。

如果确实需要安装，必须禁止 pip 改动现有 torch 依赖：

```bash
DEEPEP_PYTHON_HOME=/usr/local/python3.11.10
wheel="$(ls -1t output/deep_ep*.whl | head -1)"
"${DEEPEP_PYTHON_HOME}/bin/python3" -m pip install \
  --no-deps --force-reinstall "${wheel}"
```

这个 wheel 只携带 `NotifyDispatch`，不能替代生产服务需要的完整 DeepEP。请在单独的开发容器/环境中安装和测试，或在测试后恢复原来的完整 `deep_ep` wheel；不要让正在提供请求的 SGLang worker 使用它。

先用 `npu-smi info` 选择两张空闲卡。例如测试容器内逻辑卡 0、1：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export DEEP_USE_MODE=default
export HCCL_BUFFSIZE=2300

ASCEND_RT_VISIBLE_DEVICES=0,1 \
/usr/local/python3.11.10/bin/torchrun \
  --standalone --nproc-per-node=2 \
  tests/python/deepep/test_a5_notify_dispatch_window.py
```

`LOCAL_RANK=0/1` 分别映射到 `ASCEND_RT_VISIBLE_DEVICES` 中第一、第二张卡。若 Docker 启动时已限制设备，以容器内 `npu-smi info` 显示的逻辑编号为准。

测试脚本会在导入 torch/DeepEP 前自动完成以下工作：

1. 查找当前仓库或系统 Python 中的 `deep_ep_cpp*.so`。
2. 把 `vendors/hwcomputing` 加入 `ASCEND_CUSTOM_OPP_PATH`。
3. 把 `libcust_opapi.so` 目录加入 `LD_LIBRARY_PATH`。
4. 重新执行当前 Python 进程，使动态库路径从进程启动时生效。
5. 显式加载 `libcust_opapi.so`，再调用 `aclnnNotifyDispatch`。

启动时应看到：

```text
Using deep_ep_cpp: .../deep_ep_cpp.cpython-311-aarch64-linux-gnu.so
Using custom op API: .../op_api/lib/libcust_opapi.so
```

成功输出：

```text
PASS: A5 NotifyDispatch acquired and used the HCCL communication window
```

单卡只能取得本 rank 窗口，不能证明远端 rank 地址可访问，所以测试强制要求至少两个进程/两张 NPU。

## 常见问题

### `No module named deep_ep_cpp`

先确认分支和测试脚本版本：

```bash
git rev-parse --short HEAD
nl -ba tests/python/deepep/test_a5_notify_dispatch_window.py | head -20
```

新脚本顶部不会直接 `import deep_ep`。然后检查 `.so`；若不存在，用同一个系统 Python 重新编译：

```bash
find python/deep_ep/deep_ep output/lib -maxdepth 1 \
  -name 'deep_ep_cpp*.so' -print

DEEPEP_PYTHON_HOME=/usr/local/python3.11.10
bash scripts/build_a5_notify_dispatch.sh \
  --python-env "${DEEPEP_PYTHON_HOME}"
```

### `aclnnNotifyDispatch ... not in libopapi.so`

最新版测试脚本会自动准备运行环境。如果需要手工排查：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

DEEPEP_VENDOR_DIR=/home/l00934901/sgl-kernel-npu/python/deep_ep/deep_ep/vendors/hwcomputing
export ASCEND_CUSTOM_OPP_PATH="${DEEPEP_VENDOR_DIR}:${ASCEND_CUSTOM_OPP_PATH:-}"
export LD_LIBRARY_PATH="${DEEPEP_VENDOR_DIR}/op_api/lib:${LD_LIBRARY_PATH:-}"

ldd "${DEEPEP_VENDOR_DIR}/op_api/lib/libcust_opapi.so" | grep 'not found' || true
nm -D "${DEEPEP_VENDOR_DIR}/op_api/lib/libcust_opapi.so" | \
  grep -E 'aclnnNotifyDispatch(GetWorkspaceSize)?$'
```

`ldd` 不应出现 `not found`；`nm` 应同时输出 `aclnnNotifyDispatch` 和 `aclnnNotifyDispatchGetWorkspaceSize`。

### `allow_internal_format=False`

该信息是创建基础格式 tensor 时的 warning，不是算子测试失败原因，可以忽略。
