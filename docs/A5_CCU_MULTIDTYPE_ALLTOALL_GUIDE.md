# CCU 显式多路径 AllToAll：自动 dtype（两卡 / 四卡）

分支：`feature/a5-ccu-multidtype-alltoall`，从四卡分支 `bed7925` 创建。
不修改 HCCL、不加载 ko、不修改共享 UB 路由表。本分支保留两卡 prepared-plan，
以及兼容两卡/四卡的 peer-plan；测试统一推荐 peer-plan 入口。

## 1. 接口与类型

支持 `torch.float16 / bfloat16 / float32 / int8 / int32`，读取输入 tensor 的 dtype，
不需要手动传 HCCL type，不转换数据、不做数值运算。输出必须与输入同 dtype。
FP64、bool、complex、FP8 和打包的低比特格式不在本版支持范围，明确拒绝。

```python
# 图外：按显式拓扑路径准备资源，和 FP32 版相同。
handle = buffer.prepare_ccu_urma_peer_plan("my-plan", manifest, available_cards)
send = torch.empty((k, elements_per_peer), device="npu", dtype=torch.bfloat16)
recv = torch.empty_like(send)
# 注意这里 send 只是示意；真实调用前须由模型/测试填好输入。
buffer.ccu_urma_peer_plan_alltoall_out(send, recv, handle)
# 图内 functional 入口，返回同 dtype/shape 的新输出：
recv = torch.ops.deep_ep.ccu_urma_peer_plan_alltoall(send, handle, [])
```

旧两卡接口 `buffer.ccu_urma_prepared_multipath_alltoall[_policy]_out(...)`
及对应 functional torch.ops 同样自动读取 dtype。图外绑定正确执行/capture stream
仍是必要条件；本次不改变资源生命周期、stream 顺序或图捕获方式。

`--dtype` 是测试脚本生成 tensor 的选项，不是算子 API 的新参数。
所有 rank 必须使用一致 dtype、每 peer 元素数及路径权重；本版不增加逐执行
host 元数据协商，不能让某个 rank 单独改变 dtype/长度。

## 2. 字节数据面与调用链

```text
输入 tensor.scalar_type()
  -> DeepEP CcuPayloadType：映射 HcclDataType，检查收发类型相同
  -> HcclCcuUrmaPeerPlanExecute / ExplicitMultipathPlanExecute[V2]
  -> PayloadElementBytes：FP16/BF16=2，FP32/INT32=4，INT8=1
  -> peerBytes = elementsPerPeer * elementBytes（检查乘法溢出）
  -> 每 peer 按路径权重划分完整字节预算
  -> sourceOffsets / remoteOffsets / pathBytes（单位全部是 byte）
  -> self 分片 D2D copy，remote 分片沿已准备 Channel 执行 WriteNb
  -> completion 同步
```

共用 `RouteKernel`；不为每种 dtype 生成新的 CCU 搬运算法。
非最后分片保持 256-byte 对齐，最后分片承接余数；支持的元素大小 1/2/4 均整除256，
因此分片边界不会切开元素。原有过小负载导致某条路径分到0字节的请求仍拒绝，
不会提交零长度 WriteNb。实际设备是否接受特定小负载/尾数，需要有卡验证。
输入/输出连续且不重叠，模型仍需保证生命周期、stream 与 plan 匹配。

资源 plan 不绑定 dtype；字节布局在执行时计算，因此可在正确排序的执行中复用
Channel/kernel。已经 capture 的图参数是该次 tensor/shape/dtype，不能宣称修改
dtype 后仍能原样 replay；需针对新 dtype 重新捕获/编译。

类型变更不改变 EID、relay 选卡、2:1 权重或 N/k relay 限制。
`--bytes` 始终是每个目的 rank 的**字节数**：四卡4MiB/peer总tensor16MiB。
同样4MiB：FP32/INT32每peer1048576元素，FP16/BF16为2097152，INT8为4194304。
历史 ACLNN/MC2 标准算子及实验 HBM mailbox 不在本次扩展范围，仍保留其类型约束。

## 3. 拉取新分支、构建安装

在有卡容器执行。备份 tracked 本地差异，保留 untracked；括号避免失败时退出
当前 Docker shell。不要把 `set -e` 或带 `exit` 的恢复片段直接 source 进交互 shell。

```bash
(
set -eo pipefail
cd /home/l00934901/sgl-kernel-npu
git stash push -m "backup-before-multidtype-$(date +%Y%m%d_%H%M%S)"
BRANCH=feature/a5-ccu-multidtype-alltoall
git fetch origin "${BRANCH}"
git switch "${BRANCH}"
git merge --ff-only "origin/${BRANCH}"
git branch --show-current
git rev-parse --short HEAD

source /usr/local/Ascend/cann-9.1.T560/set_env.sh
unset LD_PRELOAD A5_CCU_PREPARED_ALLOW_LAZY_STREAM A5_URMA_TP_TRACE_PREFIX
HCCL_REPO=/home/l00934901/hccl \
bash scripts/build_a5_ccu_urma_route_probe.sh \
  --install --install-path /usr/local/Ascend/cann-9.1.T560

DEEPEP_SINGLE_OP=explicit_multipath_all2all_ccu bash build.sh -a deepep Ascend950
BUILT_SO=$(find output/lib -maxdepth 1 -type f -name 'deep_ep_cpp*.so' -print -quit)
test -n "${BUILT_SO}"
nm -D --defined-only "${BUILT_SO}" | grep ' A5DeepEpPeerPlanAbiVersion$'
WHEEL=$(ls -t output/deep_ep-*.whl | head -1)
test -f "${WHEEL}"
echo "Installing wheel: ${WHEEL}"
python3 -m pip install --force-reinstall --no-cache-dir --no-deps "${WHEEL}"
python3 tests/python/deepep/test_a5_ccu_peer_plan_alltoall.py --runtime-check
)
```

首次升级本分支**必须重编安装两份库**，不是之前的脚本-only更新；无需拉取/编译
patched HCCL。HCCL_REPO只提供旧对象式custom-op构建框架。
编译打印的 CANN home 应与实际使用的 CANN 版本一致。
安装前停止本用户使用旧库的实验；共享挂载上的 CANN/Python 文件更新会影响其他
使用该目录的环境。这里不涉及驱动/ko，也不保证容器文件系统与宿主机完全隔离。

版本要求：

| 库 | 符号 | 最低值 |
|---|---|---:|
| route | A5CcuPeerPlanAbiVersion | 2 |
| DeepEP | A5DeepEpPeerPlanAbiVersion | 2 |
| route（旧两卡接口） | A5CcuUrmaPreparedPlanAbiVersion | 13 |
| DeepEP（旧两卡接口） | A5DeepEpExplicitMultipathAttrAbiVersion | 12 |

build route脚本现在检查 packaged 与 installed 两份库的peer-plan符号和版本，
不再仅用旧两卡ABI通过就宣称四卡安装成功。预检出现undefined symbol或旧版本时
停止，检查实际库路径/编译日志，不跳过ABI检查。

## 4. 先短测单个 dtype（两卡或四卡，direct-only）

不要一次性抢占不存在的资源，确认通信卡可用；relay不运行rank进程。
下面先用用户常用2、3卡，无 relay，测 BF16。改 `--devices 0,1,2,3` 即四卡。

```bash
bash scripts/run_a5_ccu_peer_plan_alltoall.sh \
  --devices 2,3 --dtype bfloat16 \
  --bytes 4194304 --warmup 1 --iters 3 --repeats 1 \
  --sync-mode per-call --graph-backend none \
  --output-root /home/l00934901/profiling
```

逐种类型验证（只用确认可用的卡；失败停止，不把部分成功当全部通过）：

```bash
(
set -e
for dtype in float16 bfloat16 float32 int8 int32; do
  bash scripts/run_a5_ccu_peer_plan_alltoall.sh \
    --devices 2,3 --dtype "${dtype}" \
    --bytes 4194304 --warmup 1 --iters 3 --repeats 1 \
    --sync-mode per-call --graph-backend none \
    --output-root /home/l00934901/profiling
done
)
```

每次必须所有rank数据正确且清理正常退出，报告complete=True。结果JSON和报告
记录dtype/element_bytes；跨rank混合dtype或与run_settings不匹配的结果不接受。
校验数据是有位置、source、destination、epoch编码的 [-63,63] 整数，五种格式
均可精确表示，poison=-128，不再使用会使FP16溢出的百万级marker。

## 5. 四卡显式relay、batch性能与图验证

保持现有显式relay-map，示例外部relay使用4、5、6，执行前确认资源可用。
先使用上一节短测参数，再按需进入下列性能命令：

```bash
bash scripts/run_a5_ccu_peer_plan_alltoall.sh \
  --devices 0,1,2,3 \
  --relay-map docs/topology/a5_peer_plan_4rank_example.json \
  --dtype bfloat16 --sync-mode batch --graph-backend none \
  --bytes 4194304 --warmup 100 --iters 20 --repeats 1 --profile \
  --output-root /home/l00934901/profiling
```

测试流程沿用：资源一次准备 → 三次changed-payload → 两轮batch gate → warmup100
→ 正式20 → 检查/清理，profiling不包括warmup与gate。
gate每次换输入、poison输出、检查所有snapshot，不仅验证最后一次。
CPU batch总时间/20不是单次latency，device耗时看MindStudio；CCU任务含peer等待。

图验证单独运行，不能把batch和aclgraph混在同一实验里：

```bash
bash scripts/run_a5_ccu_peer_plan_alltoall.sh \
  --devices 0,1,2,3 \
  --relay-map docs/topology/a5_peer_plan_4rank_example.json \
  --dtype bfloat16 --sync-mode per-call --graph-backend aclgraph \
  --bytes 4194304 --warmup 1 --iters 3 --repeats 1 \
  --output-root /home/l00934901/profiling
```

将dtype依次改为五种格式，并分别在2/4卡、direct-only/relay配置复测。
没有这些有卡结果前，不宣称新类型的数据面或ACLGraph replay已通过。
两卡也可直接用peer-plan入口；原两卡性能脚本
`run_a5_ccu_explicit_multipath_alltoall_perf.sh` 同样新增 `--dtype`，其prepared实现
自动读取tensor类型，旧standard/MC2实验仍限制FP32。

## 6. 无卡验证与本次开发验证边界

```bash
python3 tests/python/deepep/test_a5_ccu_multidtype_cpu.py
python3 tests/python/deepep/test_a5_ccu_peer_entrypoint_cpu.py
python3 tests/python/deepep/test_a5_ccu_peer_plan_cpu.py
python3 tests/python/deepep/test_a5_ccu_test_runtime_cpu.py
```

C++ CPU布局测试覆盖1/2/4字节、2/4rank、direct/relay、元素尾数和uint64溢出：

```bash
(
set -e
TMP_TEST=$(mktemp -d /tmp/a5_dtype_tests.XXXXXX)
g++ -std=c++14 -Wall -Wextra -Werror -I. \
  tests/cpp/test_a5_multipath_layout.cpp -o "${TMP_TEST}/two_layout"
"${TMP_TEST}/two_layout"
g++ -std=c++14 -Wall -Wextra -Werror \
  -Iexamples/a5_ccu_urma_route_probe/common \
  tests/python/deepep/test_a5_ccu_peer_layout.cpp -o "${TMP_TEST}/peer_layout"
"${TMP_TEST}/peer_layout"
)
```

有CANN/Torch但无卡时，可在编译后运行Meta测试（用实际.so路径）：

```bash
EXTENSION=$(find output/lib -maxdepth 1 -type f -name 'deep_ep_cpp*.so' -print -quit)
test -n "${EXTENSION}"
python3 tests/python/deepep/test_a5_ccu_multidtype_meta.py --extension "${EXTENSION}"
```

2026-10-12 开发容器cam_lyw_dev_91、CANN9.1.0-beta.1环境已编译DeepEP与route
package，验证route prepared ABI13/peer ABI2、五种dtype的两/四卡Meta/fullgraph输出及
旧两卡prepared Meta。未改驱动、未安装到公共有卡机器。
CPU/Meta与开发版CANN编译不能替代用户T560有卡正确性、转发端口和图replay验证。
