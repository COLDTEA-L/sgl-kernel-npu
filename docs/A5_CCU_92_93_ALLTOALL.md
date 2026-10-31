# CANN 9.2 / 9.3 完整多路径 AllToAll 分支

分支：`feature/a5-ccu-92-93-alltoall`。从原 `feature/a5-ccu-8rank-heuristic-alltoallv` 的 `370a14d` 建立，原 9.1 分支和旧容器安装产物不修改。本分支要求新公开 CCU API，不用于 9.1。

## 1. 迁移范围与验证边界

| 路径 | 保留功能 |
| --- | --- |
| 2 卡 | 等长多路径 AllToAll、prepared plan、动态权重、显式组外物理 relay、serial/concurrent、self-copy 模式 |
| 4 卡 | 原 peer-plan、组外物理 relay、baseline/packed/packed-ready 通知协议 |
| 8 卡 | EP8 AllToAllV，非对称 C[8][8]、零长度、独立 self 偏移、storage-count 偏移接口 |
| EP8 多路径 | §11 新增组内显式 EID + CCU/URMA 端到端转发；原 HBM pipeline 保留作对照 |
| 计时 | 原 native HCCL / custom direct / multipath，prepared 和 dynamic 两种口径 |

2/4 卡不套用 EP8 贪心算法；8 卡 AllToAllV 继续专供 EP8，relay 仅在通信组成员中选择。**当前验证新的 EP8 EID 后端请执行 §3、§4 重编译，然后从 §11 开始；§6/§7/§10 的 HBM pipeline 命令保留用于历史对照，不会自动切换到 EID。**

服务器已在独立 `cam_lyw_92` 中编译、严格链接、加载整个 route 库及 local-copy 诊断程序；新 host mock 检查固定注册、100 次复用、设备/通信域归属、注册失败不重试；EP8 CPU 图模型执行 100 轮、800 次调度；2/4 卡布局、通知、自拷贝和规划器 CPU 回归通过。

首次迁移时该容器没有 Torch/torch_npu；后续独立 venv 的 CPU Torch 用于 §9/§10 的真实 ATen C++ 编译。§11 又在任务 profiling 目录解包了 torch_npu 2.10.0 的真实头文件/库，完成整个 DeepEP C++ 扩展的编译、严格链接和导入。未挂载 NPU，尚未打包验证完整 device-op wheel 或上卡。9.3 接口名称/参数形式来自用户提供的检查结果，仍需在 9.3 本机重编译和上卡。CPU 模型不验证固件、实际 MS/loop 资源或性能；9.2 二进制也不能作为 9.3 的已验证产物。

## 2. 实现与 ABI

项目自己的 `common/public_ccu.h`、`op_host/public_ccu.cc` 使用 namespace `a5ccu`，不导入旧 `hcomm::CcuKernel/CcuRep` C++ ABI。构图转为 `AscendC::ccu`，注册转为 `HcommCcuKernelRegisterStart/Register/End`，token 转为 `HcommCcuGetMemToken`，启动转为 `HcommCcuKernelLaunch`。

每次准备实际 stream/kernel 绑定时 Query 通信域自有实例，显式传本地 die 注册，并检查 SDK task-argument 数量与 LoadArg/GeneArgs 约定一致。Query 返回借用句柄，不 Assign、不自行 Destroy。构造参数和运行参数分别传递。注册事务失败后不重试该通信域。

execute 复用通信域、thread、channel 和 kernel，不逐次 Query/Register。self-copy 转为公开 Array/Func/Loop/LoopGroup，保留连续资源、生产 8×32KiB 配置和四字 go 参数，需新机验证实际执行。新 API 上报走 SDK thread callback 路径，不再注入 9.1 私有 `HcclDfxRegOpInfoByCommId`。

保留 client ABI：prepared-plan 16、peer-plan 5、alltoallv/layout/bound-direct/thread-bridge 1。新增 `A5CcuPublicApiAbiVersion=1`；`A5CcuEp8DfxAbiVersion=2` 表示 SDK reporting 路径，不表示已经证明真实长跑稳定。

## 3. 获取代码、构建 route

开发者 push 后，在目标容器（例如 `yuanwen_dsk`）执行；先保存本地修改。若本地已有同名分支，用 `git switch feature/a5-ccu-92-93-alltoall` 和 `git merge --ff-only origin/feature/a5-ccu-92-93-alltoall` 更新。

```bash
cd /home/l00934901/sgl-kernel-npu
git fetch origin feature/a5-ccu-92-93-alltoall
git switch --track origin/feature/a5-ccu-92-93-alltoall
export A5_CANN_PATH=/usr/local/Ascend/cann-9.3.0
mkdir -p /home/l00934901/profiling/a5-public-930/tmp
export TMPDIR=/home/l00934901/profiling/a5-public-930/tmp
bash scripts/build_a5_ccu_urma_route_probe.sh \
  --cann-path "$A5_CANN_PATH" \
  --build-dir /home/l00934901/profiling/a5-public-930/route-build \
  --install --install-path "$A5_CANN_PATH"
export A5_CCU_ROUTE_PROBE_LIB="$A5_CANN_PATH/opp/vendors/cust/lib64/liba5_ccu_urma_route_probe.so"
```

构建只用指定 SDK，不需要 HCCL 源码或 `HCCL_REPO`，不依赖 `cann` 别名，不生成旧 `.run` 安装器。`--install` 直接安装库和头文件到指定 CANN 的 `opp/vendors/cust`。预期 `public_ccu_host Passed`、`PUBLIC_ROUTE_LOAD_PASS`、`PUBLIC_ROUTE_BUILD_PASS`；构建目录保留配置、编译、host-test 日志和 `undefined.txt`。

开发服务器容器中仓库为 `/workspace`，CANN 为 `/usr/local/Ascend/cann-9.2.0`，结果为 `/home/liuyuanwen/profiling/cann92-ccu-abi-wJLnOX/route-build`。旧临时 `a5-lyw-public-api-compile` 已自动删除，其他容器不动。

## 4. 重建目标 DeepEP wheel

使用目标 Python、Torch、torch_npu、CANN 重编译；不要直接安装旧 9.1 wheel。调用接口保留不等于 wheel 内其他 device op 二进制兼容。第三方源码依赖仍按仓库构建要求准备。

```bash
(
set -Ee -o pipefail
cd /home/l00934901/sgl-kernel-npu
source scripts/a5_ccu_test_env.sh
a5_ccu_prepare_test_env /usr/local/Ascend/cann-9.3.0 "$PWD"
A5_WHEEL_BUILD=$(mktemp -d /home/l00934901/profiling/a5-public-wheel.XXXXXX)
mkdir -p "$A5_WHEEL_BUILD/tmp"
export TMPDIR="$A5_WHEEL_BUILD/tmp"
touch "$A5_WHEEL_BUILD/started"
DEEPEP_SINGLE_OP=explicit_multipath_all2all_ccu bash build.sh -a deepep Ascend950 \
  > "$A5_WHEEL_BUILD/build.log" 2>&1 || {
  tail -100 "$A5_WHEEL_BUILD/build.log"; exit 1;
}
mapfile -t A5_WHEELS < <(find "$PWD/output" -maxdepth 1 -name 'deep_ep-*.whl' \
  -newer "$A5_WHEEL_BUILD/started" -print)
[[ ${#A5_WHEELS[@]} == 1 ]] || { echo "没有唯一的新 wheel：$A5_WHEEL_BUILD"; exit 1; }
python3 -m pip install --force-reinstall --no-deps "${A5_WHEELS[0]}"
echo "Wheel 日志：$A5_WHEEL_BUILD"
)
```

## 5. 2/4 卡回归

参与卡及组外物理 relay 卡都应可用。以下两卡沿用先前 2/3 经 relay 4 的配置；每台机器仍需验证实际 rootinfo、Atlas 和实时 EID/端口，示例不是对所有拓扑的保证。

```bash
cd /home/l00934901/sgl-kernel-npu
export A5_CCU_ROUTE_PROBE_LIB=/usr/local/Ascend/cann-9.3.0/opp/vendors/cust/lib64/liba5_ccu_urma_route_probe.so
bash scripts/run_a5_ccu_peer_plan_alltoall.sh \
  --cann-root /usr/local/Ascend/cann-9.3.0 \
  --devices 2,3 --available-phys 0,1,2,3,4,5,6,7 --relay-phys 4 \
  --root-info /etc/hccl_rootinfo.json \
  --topology-json /usr/local/Ascend/driver/topo/950/atlas_950_1.json \
  --bytes 4096 --dtype float32 --warmup 3 --iters 100 --repeats 1 \
  --output-root /home/l00934901/profiling
```

四卡示例只给 0↔1 配物理 relay 4，其他 peer 直连：

```bash
bash scripts/run_a5_ccu_peer_plan_alltoall.sh \
  --cann-root /usr/local/Ascend/cann-9.3.0 \
  --devices 0,1,2,3 --available-phys 0,1,2,3,4,5,6,7 \
  --relay-map '{"0,1":[4]}' \
  --root-info /etc/hccl_rootinfo.json \
  --topology-json /usr/local/Ascend/driver/topo/950/atlas_950_1.json \
  --bytes 4096 --dtype float32 --warmup 3 --iters 100 --repeats 1 \
  --output-root /home/l00934901/profiling
```

先用默认 overlap self-copy、baseline 通知；通过后按相同物理策略补 `--self-copy-mode ccu`，再补 `--channel-sync-mode packed` / `packed-ready`、需要的 dtype/graph 模式。检查全部 rank 校验和退出。解析失败先处理系统拓扑，不绕过 EID/die 校验。

## 6. EP8 正确性和 100 次 dynamic

```bash
cd /home/l00934901/sgl-kernel-npu
export A5_CANN_PATH=/usr/local/Ascend/cann-9.3.0
export A5_OUTPUT_ROOT=/home/l00934901/profiling
export A5_ROOT_INFO=/etc/hccl_rootinfo.json
export A5_ATLAS_JSON=/usr/local/Ascend/driver/topo/950/atlas_950_1.json
bash scripts/run_a5_ccu_heuristic_alltoallv.sh \
  --routing-policy greedy --executor pipeline --pipeline-host cpp \
  --suite correctness --mode all --dtype float32 --width 257 --row-scale 1 \
  --warmup 1 --iters 2 --rounds 1 --dynamic-iters 1
```

应有 `EP8_HEURISTIC_ALL_RANKS_AND_EXIT_PASS`：7 case、全部 8 rank 校验及 launcher 正常退出。原 Python host 和 serial 执行器仍可对照。通过后执行：

```bash
bash scripts/run_a5_ccu_heuristic_alltoallv.sh \
  --routing-policy greedy --executor pipeline --pipeline-host cpp \
  --mode all --case single_hot --chunk-bytes 8388608 \
  --warmup 3 --iters 10 --rounds 4 --dynamic-iters 100
```

`PUBLIC_CCU_REGISTER` 只应在 lane/stream 首次准备时出现，dynamic 不逐次建链/注册。长跑还须检查 worker 退出、plog、上报层错误、内存增长；host mock 的复用检查不能代替设备长跑。失败保留 `PUBLIC_CCU_REGISTER`、`ALLTOALLV_STAGE_FAILED`、`PIPELINE_TRANSFER_FAILED` 及底层日志，不降低迭代绕过问题。

## 7. 性能对比

```bash
for A5_CASE in single_hot competing asymmetric cut uniform self zero; do
  bash scripts/run_a5_ccu_heuristic_alltoallv.sh \
    --routing-policy greedy --executor pipeline --pipeline-host cpp \
    --mode all --case "$A5_CASE" --chunk-bytes 8388608 \
    --warmup 3 --iters 10 --rounds 6 --dynamic-iters 10
done
```

同一次结果保留 native/direct/multipath 的 prepared/dynamic 耗时、relay 字节及实际策略。比较相同 case、dtype、width、chunk 和轮数，不把 9.1/9.3 整体软件栈差异都归因于选路。`greedy` 保持原决策，`fast` 允许 native 回退，是另一组实验。无 relay 时 direct/multipath 为 A/A 对照。

8MiB 是每条被 relay 的原始流的 chunk 上限，不是每卡总发送量。本次接口迁移不声称消除了 competing/asymmetric 的既有劣化，也没有测出新硬件收益。

## 8. CPU 检查入口

也可一次运行全部下列核心套件及 runtime bootstrap 检查，日志、临时文件都写入个人结果目录：

```bash
bash scripts/check_a5_ccu_public_api_cpu.sh /home/l00934901/profiling
```

```bash
python3 -m unittest discover -s tests/python/deepep -p 'test_a5_ccu_alltoallv_cpu.py'
python3 -m unittest discover -s tests/python/deepep -p 'test_a5_ccu_peer_plan_cpu.py'
python3 -m unittest discover -s tests/python/deepep -p 'test_a5_ccu_local_remote_cpu.py'
python3 -m unittest discover -s tests/python/deepep -p 'test_a5_ccu_channel_sync_cpu.py'
python3 -m unittest discover -s tests/python/deepep -p 'test_a5_ccu_heuristic_cpu.py'
python3 -m unittest discover -s tests/python/deepep -p 'test_a5_ccu_pipeline_cpu.py'
python3 -m unittest discover -s tests/python/deepep -p 'test_a5_ccu_thread_bridge_cpu.py'
python3 -m unittest discover -s tests/python/deepep -p 'test_a5_ccu_dfx_cpu.py'
```

缺少 Torch 的用例会明确 skipped。local-copy 诊断也已迁移公开接口；旧 `--builder-only` 不再提供脱离实例的构图证明，无 NPU host 生命周期检查使用构建目录的 `a5_ccu_public_host_test`。

开发机本轮 CPU 套件共 59 项：54 通过、5 项因缺少 Torch 跳过；另有 1 项基于真实 SDK 头文件的 host mock ctest 通过。结果在 `/home/liuyuanwen/profiling/cann92-ccu-abi-wJLnOX/a5-public-cpu.8SPBR4`，route 编译与 ctest 日志在同级 `route-build`。没有新硬件吞吐或延迟数据。

## 9. FP8 payload 支持

EP8 direct、serial 参考执行器、Python/C++ pipeline 支持 `float8_e4m3fn`、`float8_e5m2`、`float8_e4m3fnuz`、`float8_e5m2fnuz`。接口输入/输出保持原始 dtype，按每元素 1 字节计算 row_bytes。C++ CCU 入口映射到同宽 INT8 传输类型；relay pack/forward/unpack 和原生 HCCL baseline 使用无拷贝 uint8 view，保持所有位模式，不进行数值转换、量化或反量化。没有新增每次注册或建链。

这是通信 payload 兼容；模型权重 FP8 不代表该处激活必定 FP8，接入时以实际输入 tensor 为准。独立 scale tensor 不会被隐式打包，调用方仍须按对应路由传递 scale；本次不实现 payload/scale 融合通信。2/4 卡既有调用接口和测试类型不变。

拉取新提交后，按 **§4 重建并安装 DeepEP wheel**。旧 wheel 会被新增的 `A5DeepEpFp8PayloadAbiVersion=1` 检查拒绝。route 本次未改动，已安装本分支公开 API route 的环境无需仅为 FP8 重建 route。

先沿用 §6 的环境变量，执行两种主要格式的七类正确性测试：

```bash
for A5_FP8 in float8_e4m3fn float8_e5m2; do
  bash scripts/run_a5_ccu_heuristic_alltoallv.sh \
    --routing-policy greedy --executor pipeline --pipeline-host cpp \
    --suite correctness --mode all --dtype "$A5_FP8" --width 257 --row-scale 1 \
    --warmup 1 --iters 2 --rounds 1 --dynamic-iters 1
done
```

每次均要求 `EP8_HEURISTIC_ALL_RANKS_AND_EXIT_PASS`。检查原始 256 种字节模式（含 NaN、正负零）、输入及输出保护区、非零 storage offset、独立 self 偏移、零长度和多批次 slot 复用。FP8 校验按字节比较，避免 NaN 浮点比较或转换舍入掩盖错误。其他两种 fnuz 格式也可使用相同命令。

再对实际目标格式执行长跑和性能比较：

```bash
bash scripts/run_a5_ccu_heuristic_alltoallv.sh \
  --routing-policy greedy --executor pipeline --pipeline-host cpp \
  --mode all --case single_hot --dtype float8_e4m3fn \
  --width 4096 --row-scale 256 --chunk-bytes 8388608 \
  --warmup 3 --iters 10 --rounds 4 --dynamic-iters 100
```

此配置 0→4 原始流为 `32 × 256 × 4096 × 1 = 32 MiB`，与旧 FP16 缺省案例的跨卡字节数相同。FP8 若仍使用 row-scale 128，则该流为 16 MiB；比较时必须区分相同行数与相同字节数。每行 FP8 payload 字节数为 width，8MiB 参数仍是每条 relay 流的 chunk 上限。

CPU 检查：

```bash
python3 -m unittest discover -s tests/python/deepep -p 'test_a5_ccu_fp8_cpu.py'
```

需要支持上述 FP8 类型的 Torch 和 g++；其中一个用例直接编译真实 ATen + 生产 C++ copy/pipeline host，检查原始字节、保护区、错误 span 不提前写入，以及 400 次 host 动态调用时 workspace 不新增分配。另一个用例用八 rank 异步 CPU DAG 检查四种格式、变动矩阵和 relay 双缓冲复用。它们不证明 NPU/驱动上的 dtype view 或通信正确性；仍需上述目标机测试。

本次 FP8 开发验证在 `cam_lyw_92` 的独立 CPU Torch 2.10 venv 中完成，未修改其他容器或 9.1 安装。10 组核心 CPU 套件共 **62 项全部通过**，包含 FP8 的真实 ATen C++ 编译/链接和四格式异步 DAG 测试；源文件语法检查通过。日志：`/home/liuyuanwen/profiling/a5-fp8-validation/a5-public-cpu.i4NJm7`。完整 DeepEP wheel 仍须在带 torch_npu 的目标 9.3 环境按 §4 编译；尚无 FP8 NPU 实测结果。

## 10. 同一个 C++ host 支持 FP8 dispatch / BF16 combine

本节对应 `ep8_pipeline_host_abi=3`。**已有本分支 route 安装不必重建；更新代码后从 §4 重建并安装 DeepEP wheel，再执行本节。** 原 9.1 分支和 2/4 卡接口不变。

对照开发服务器 `/home/liuyuanwen/vllm-ascend-dsv41` 的 `56debc194`：
`vllm_ascend/ops/fused_moe/token_dispatcher.py` 中 `TokenDispatcherWithAll2AllV.token_dispatch`
对量化 payload 调用 `async_all_to_all(..., output_splits, input_splits, ...)`，
`token_combine` 对专家输出调用 `async_all_to_all(..., input_splits, output_splits, ...)`。
开启量化时，dynamic scale 还有一次独立通信。`moe_utils.py` 的 wrapper 按实际 input dtype 分配 output。
因此 dispatch/combine 要分别采用实际张量的 dtype 和行形状，不能假设两个方向的 row_bytes 相同。

### 10.1 输入与复用约定

| 方向 | 输入/输出 dtype | 行形状 | 每行字节 |
| --- | --- | --- | --- |
| dispatch | FP8 | `[5120]` | 5120 |
| combine | BF16 | `[5120]` | 10240 |

9072 行对应 46,448,640 B / 92,897,280 B；这是这 9072 行的 payload 总量，不包含 scale、self 排除或 relay 增量。

`ep8_pipeline_create` 仍只创建一次，`attach_lane` 仍只调用三次。第一次见到 `(device, dtype, row_shape)` 时创建一份内部 payload 配置，后续同签名复用。行数、counts 不属于签名，可逐次变化；同一 row_bytes 但 dtype 或行形状不同也不能混用。

- 每份配置分别保存 planner、chunk 行数、布局、copy span、零行占位和双缓冲 workspace；按张量重新计算实际 row_bytes。
- relay 门槛和容量保留原始**字节**值，再换算为各 dtype 的行数，避免整数舍入后反推字节。
- 三条 lane 的通信域、stream、channel、CCU kernel handle、event pool 共享；切换 dtype 不重新建链或注册。首次分配或容量增长仍可能同步；预热后切换不增加 host/device 同步。
- 前一个调用在 caller stream 上的完成依赖继续约束下一个方向；接口仍是异步提交，保持原 caller stream，不允许多个 CPU 线程同时提交。
- `prepare(C, example)` 返回的对象固定属于该签名，新增 `prepared.row_bytes`；FP8 prepared 不能拿来执行 BF16。两个方向各 prepare 一份即可，二者长期并存。
- `run(C, send, recv)` 自动选择签名并重规划实际 counts。单次调用 send/recv 仍必须 dtype、device、行形状相同；不会做量化、反量化或自动转换。
- 新增可选 `max_payloads=4`，覆盖 FP8、BF16、scale 等签名。`max_scratch_bytes` 是所有签名**合计**的每 rank workspace 上限，不是每个签名都分一份上限。零行占位单独报告。超过签名/容量上限明确报错，不回收仍可能在途的 workspace。
- `payload_stats()` 返回各签名的 dtype、row_shape、row_bytes、reserved_bytes、tensor_allocations、generation。`stats()` 是合计；event 数量按共享池计一次。任何签名提交失败会阻止整个 host 继续执行。

接入示意（沿用已创建的三条 `pipeline.lanes` 和原 `RelayPlanner`，counts 是实际 packed 行数矩阵）：

```python
# 创建一次：planner._native 的初始 row_bytes 不再限制后续 payload。
cpp = deep_ep_cpp.ep8_pipeline_create(
    planner._native, rank, chunk_bytes=8388608,
    max_scratch_bytes=1 << 30, max_payloads=4,
)
for li, lane in enumerate(pipeline.lanes):
    with torch.npu.stream(lane.stream):
        cpp.attach_lane(li, lane.buffer.runtime, lane.handle)

# 每层/每批的实际输入；同一个 cpp，同一套通信资源。
cpp.run(C_dispatch, dispatch_fp8, dispatched_fp8)   # [..., 5120], 5120 B/行
# 专家计算、对应的模型 permute/unpermute ...
cpp.run(C_combine, expert_bf16, combined_bf16)       # [..., 5120], 10240 B/行
print(cpp.payload_stats())
```

在上述标准往返路由中 `C_combine = transpose(C_dispatch)`；仍由调用方传入真实矩阵，不在算子内部隐式转置。若模型 padding、去重或额外过滤改变 packed 行数，应以实际 splits 为准。scale 也必须独立传递正确 counts；本改动不融合 scale，不改动模型本身。此多签名能力针对 C++ `Ep8PipelineHost`；Python 参考 `PipelineContext` 仍使用其原有固定签名约束。

### 10.2 新机双向正确性和复用长跑

先按 §4 安装新 wheel，并沿用 §6 的环境变量。用较小 chunk 专门覆盖两个 dtype 的不同分块数、真实 relay、双缓冲复用；**40960 B 是正确性压力参数，不是性能推荐值**：

```bash
cd /home/l00934901/sgl-kernel-npu
export A5_CANN_PATH=/usr/local/Ascend/cann-9.3.0
export A5_OUTPUT_ROOT=/home/l00934901/profiling
export A5_ROOT_INFO=/etc/hccl_rootinfo.json
export A5_ATLAS_JSON=/usr/local/Ascend/driver/topo/950/atlas_950_1.json
bash scripts/run_a5_ccu_heuristic_alltoallv.sh \
  --routing-policy greedy --executor pipeline --pipeline-host cpp \
  --suite correctness --mode all --dtype float8_e4m3fn \
  --width 5120 --row-scale 1 --chunk-bytes 40960 \
  --bidirectional-iters 100 \
  --warmup 1 --iters 2 --rounds 1 --dynamic-iters 1
```

应看到 8 个 rank 各自的 `EP8_BIDIRECTIONAL_PASS`，以及正常退出后的 `EP8_HEURISTIC_ALL_RANKS_AND_EXIT_PASS`。
每个 case 先各执行一次 prepared dispatch/combine，再用同一 host 交替动态执行 100 对 FP8/BF16 调用。
两方向之间不插入 device synchronize；每对完成后校验数据、保护区和原生基线（基线在初次 prepared 校验），确认预热后 workspace/event 分配计数不再增长。每轮更新测试数据，避免把旧结果误认为本轮成功。
`results/rank*.json` 新增 `bidirectional` 字段，记录两个 row_bytes、chunk 数、每签名与合计 workspace；测试不属于性能计时。

性能仍按 §7，使用 8MiB chunk 和真实矩阵分别测 dispatch/combine；本改动解决双 dtype 兼容和资源复用，不代表已经获得模型端加速。

### 10.3 开发验证

```bash
python3 -m unittest discover -s tests/python/deepep -p 'test_a5_ccu_bidirectional_cpu.py'
```

该测试编译并链接真实 ATen + 生产 C++ planner/pipeline host，使用八 rank 延迟执行的 CPU transport 检查依赖：宽 5120，FP8/BF16 交替 100 对，覆盖 staged/免拷贝、非对称与转置矩阵、零行、self、原始字节和保护区、旧 prepared 复用、同字节数不同签名、总工作区上限、错误传播。预热后不新增 workspace 分配或同步；不能代替 NPU 验证。

本轮 `cam_lyw_92` 中 11 组 CPU 套件 **63 项全部通过**，包含上述真实 C++ 编译/链接；CANN 9.2 route 全量构建、严格链接、host ctest 与加载通过。结果：`/home/liuyuanwen/profiling/a5-bidirectional-validation/`（CPU 日志子目录 `a5-public-cpu.figBa1`）。完整 DeepEP wheel 和真实 NPU 双向执行仍须在目标环境按 §4、§10.2 验证。

## 11. EP8 组内显式 EID 转发（新后端）

实现调用链、单 relay 贪心算法、资源生命周期和历次失败的证据说明，见 [EP8_CCU_EID_MULTIPATH_IMPLEMENTATION.md](EP8_CCU_EID_MULTIPATH_IMPLEMENTATION.md)。

### 11.1 数据路径与适用条件

与原 2/4 卡一样，以源端通向 relay 的 EID、目的端通向同一个 relay die/plane/net-instance 的 EID 建立端到端 Channel；Channel 的 remote rank 和写入地址仍是最终目的卡。relay 必须是本通信组的其他成员。应用不在 relay 上接收至 HBM、不 pack/unpack、不启动第二跳 AllToAllV。直连前缀与 relay 后缀在一个 CCU kernel 中通过非阻塞 Write 下发，再统一等待完成。

启动时注册一个 communicator/stream 的候选 Channel 和 kernel，每 peer 最多一条 direct 加六条 relay，每 rank 最多 49 条 Channel。动态参数压缩为 46 个 task args：每次只更新 C、relay 选择、偏移和长度；零流、receive-only、两个方向选不同 relay 使用固定直连peer控制握手（§11.12）；数据可走不同relay，生产图不再解码active mask。FP8 dispatch 与 BF16 combine 可复用同一个 host/handle，分别按真实 tensor 的 row_bytes 规划。

候选目录由 rootinfo、Atlas 和实时 hccn EID/端口联合生成；保持每 rank 单 IO die 的约束。缺失、歧义、不同 plane/net-instance/die 的 relay 不纳入候选。不接受手写 TXT，也不以 8 卡 full mesh 推断某条 EID 一定可转发。脚本默认 `--executor eid`；旧 HBM 启动脚本明确使用 `--executor hbm`。

新增 route ABI `A5CcuEidAllToAllVAbiVersion=1` 和扩展 `ep8_eid_host_abi=1`。**本次必须同时重建 route 和 DeepEP wheel（§3、§4），不能仅更新 Python。** 旧 2/4 卡接口和旧 direct-only EP8 接口保留；旧 9.1 分支不修改。

### 11.2 新机正确性和长跑

在目标容器中执行；所有八张卡需要可用。先按 §3/§4 编译安装当前环境产物，再设置。若此前在 `f1859d2` 遇到 `Unsupported repType[11]`，先执行 §11.6 的修复和只注册检查：

```bash
cd /home/l00934901/sgl-kernel-npu
export A5_CANN_PATH=/usr/local/Ascend/cann-9.3.0
export A5_CCU_ROUTE_PROBE_LIB="$A5_CANN_PATH/opp/vendors/cust/lib64/liba5_ccu_urma_route_probe.so"
export A5_DEVICES=0,1,2,3,4,5,6,7
export A5_ROOT_INFO=/etc/hccl_rootinfo.json
export A5_ATLAS_JSON=/usr/local/Ascend/driver/topo/950/atlas_950_1.json
export A5_HCCN_TOOL=/usr/local/Ascend/driver/tools/hccn_tool
export A5_OUTPUT_ROOT=/home/l00934901/profiling
bash scripts/run_a5_ccu_eid_alltoallv.sh \
  --suite correctness --mode all --dtype float8_e4m3fn --width 5120 \
  --row-scale 1 --rounds 1 --iters 10 --dynamic-iters 100 \
  --bidirectional-iters 100
```

每次生成独立 `a5-ep8-eid.XXXXXX` 目录，含 `plans/topology.json`、`plans/peer_plan.tsv`、`topology.log`、`run.log`、八份 `results/rank*.json` 和 `exit.status`。预期：每 rank 出现 `EP8_EID_CATALOG ... participant_relays=1 hbm_relay=0`；7 case 的数据、输入/输出 guard、异步 producer/consumer 检查通过；FP8/BF16 双向交替 100 次通过；最终 `EP8_EID_ALL_RANKS_AND_EXIT_PASS`。

`single_hot` 默认必须真正选中 relay；无候选则直接报错，防止把 direct-only 当成多路径正确性 PASS。只有专门诊断时使用 `--allow-no-relay`。`topology.json` 的 `relay_routes`/`excluded_relays` 给出纳入和排除证据。八卡参与者同时充当 EID 转发节点、49 条候选 Channel 的实际 SDK/固件资源容量，都需新机验证；编译和 CPU 模型不代表已验证这两项。

如果 Channel/kernel 资源不足，可显式 `export A5_EID_MAX_RELAYS_PER_PEER=1` 缩小候选目录后定位建链问题（默认 6，范围 0–6）；这会限制选路，不是等价性能配置，不自动重试或回退 HBM。每次重新启动进程，以免复用失败注册后的通信域。

### 11.3 对照性能

先比较同进程交替测量，六轮覆盖 native/direct/multipath 顺序排列：

```bash
bash scripts/run_a5_ccu_eid_alltoallv.sh \
  --mode all --case single_hot --dtype bfloat16 --width 4096 \
  --row-scale 128 --rounds 6 --iters 100 --dynamic-iters 100
```

此例每行 8192 B，热点流 4096 行 = **32 MiB 单向跨卡**，不是“每卡 8 MiB”；另外还有 self 行。EID 路径不分 HBM chunk，因此没有 `--chunk-bytes`。随后用相同参数分别测 `competing`、`asymmetric`、`cut`、`uniform`、`self`、`zero`。不能沿用旧流水线的时间或假定收益比例。

需要独立两次运行时：

```bash
bash scripts/run_a5_ccu_eid_alltoallv.sh --mode native \
  --case single_hot --dtype bfloat16 --width 4096 --row-scale 128 \
  --rounds 6 --iters 100 --dynamic-iters 100
bash scripts/run_a5_ccu_eid_alltoallv.sh --mode multipath \
  --case single_hot --dtype bfloat16 --width 4096 --row-scale 128 \
  --rounds 6 --iters 100 --dynamic-iters 100
# 用上面输出的实际目录替换两处路径：
python3 scripts/summarize_a5_ccu_eid.py \
  /home/l00934901/profiling/a5-ep8-eid.MULTIPATH \
  --compare /home/l00934901/profiling/a5-ep8-eid.NATIVE
```

每轮取八 rank 最大耗时，再跨轮取中位数。`native/multipath > 1` 才表示多路径更快。prepared 复用已规划结果；dynamic 三种模式都包含相同计数 gather/D2H，随后 C++ 规划/布局/提交（native 无 relay 规划）。两种口径均为 host wall-clock 至完成，不是 CCU event 时间。三种模式都预分配输出，启动建链/注册和正确性预检不进入计时。无 relay case 是 custom direct 的 A/A 对照；新后端不会因预测开销静默改成 native/HBM。

### 11.4 模型输入接口

C 为当前批次实际 packed 行计数，行列顺序与 communicator 的 rank 一致。所有 rank 必须拿到一致的 C 和候选目录。模型已经持有完整 CPU C 时可直接传入，测试脚本的 gather 不属于 host 的强制步骤。启动一次：

```python
# buffer 是既有 EP8 DeepEP Buffer；allowed 来自 topology.json。
planner = RelayPlanner(row_bytes, allowed, native_module=deep_ep_cpp)
handle = buffer.prepare_ccu_urma_eid_alltoallv("moe-eid", manifest)
host = deep_ep_cpp.ep8_eid_create(planner._native, rank, buffer.runtime, handle)
# 每次更新真实矩阵；同一个 host 支持不同方向 dtype/row_bytes。
host.run(dispatch_counts, fp8_send, fp8_recv)
host.run(combine_counts, bf16_send, bf16_recv)
# 全部提交结束后，在销毁 handle/通信域前：
host.close()
```

模型自身 dispatch/combine 的 packing 顺序和 counts 应由接入处提供，不在算子内猜测转置关系。不进行 dtype 数值转换，FP8 原始比特保持。核心 `EidHost::Run`、规划和布局均在 C++；Python 入口仅包装。`run` 必须在创建时的原始设备/stream 上调用，调用方保持 Buffer/handle/comm 生命周期；handle 销毁前先 close host。固定 payload signature 预热后只保留两块一行零长度 backing，不再分配 relay workspace；失败后该 host 不继续提交。

### 11.5 开发机验证记录

2026-10-30，独立 `cam_lyw_92`、CANN 9.2.0、Torch 2.10.0 CPU/aarch64，结果目录 `/home/liuyuanwen/profiling/a5-ep8-eid-validation`：

- 整个 route 库编译、严格链接、SDK host-test、实际加载及 EID ABI 1 检查通过，见 `route.log` / `route-build/`。
- 实际生产 EID CCU 指令图 CPU 模型：100 epochs × 8 ranks、49 channels/rank、46 task args；非对称方向、零长度、自拷贝独立偏移、通知清除、guard 和不同 relay 选择通过。
- 实际 ATen C++ `EidHost`：FP8 5120 B/row 与 BF16 10240 B/row 交替 100 次；每次单提交、无 relay HBM、固定 sentinel 分配；错误后停止提交通过。
- 12 组 67 项 CPU 回归全部通过，涵盖原 2/4 卡、direct AllToAllV、HBM pipeline、DFX、thread bridge；见 `cpu-suite.log`。
- 使用官方 torch_npu 2.10.0 cp312/aarch64 wheel 的真实头文件和库，编译全部 DeepEP C++ 翻译单元，`--no-undefined` 严格链接；导入成功且 `ep8_eid_host_abi=1`。见 `extension-build/commands.json`、`extension-build/link.log`、`extension-import.log`。仅解包为构建依赖，不更改其他容器；构建脚本保留在该 profiling 目录的 `a5-build-extension.py`。9.2 严格链接显式加入 `hccl_fwk`、`nnopbase`；profiling timestamp 特性按实际头文件关闭，与仓库 CMake 特性检测一致。

这些结果证明编译和离线协议/布局检查通过，**未证明组内 EID 物理转发、实际 CCU 资源容量、八卡长期稳定性或性能收益**。新机按 §11.2/§11.3 继续验证，不能引用旧 HBM pipeline 的 PASS 作为新后端硬件结论。

还直接构建了仓库 `csrc/deepep/CMakeLists.txt` 的完整扩展目标，严格链接并导入通过，见 `host-cmake.log` / `host-cmake-import.log`。生产 CMake 现会在指定 CANN 中发现并显式链接 `hccl_fwk`、`nnopbase`，减少对其他模块预先全局加载库的依赖。测试用 torch_npu 2.10.0 发布包携带的 HCCL 枚举头与 9.2 SDK 存在 LTO 定义差异警告；不据此宣称目标模型的软件栈兼容，目标环境仍按 §4 使用配套 Torch/torch_npu/CANN 重建。

### 11.6 修复 `PUBLIC_CCU_BUILD_FAILED: [GetInstrCount] Unsupported repType[11]`

现场证据为 `PEER_PLAN_ACQUIRE channels=49 status=0`、49 条 `PEER_PATH_CHANNEL state=0` 后出现构图异常。此次失败在 kernel 注册，不在 manifest 解析或 ChannelAcquire，也发生在 payload/FP8 运算前。

开发容器 CANN 9.2 `libhcomm.so` 的 `CcuRepSub::SetCommonInfo` 将表示类型设为 11；V1 生成器的 Sub 和 And 翻译返回不支持。`f1859d2` 的 EID kernel 新增了 `direct = total - relay` 和 `live = active & bit`，公开头文件声明这些操作不代表实际选用的 CCU 生成器支持它们。现场 9.3 的错误与前者匹配；不应推广为“整个 CANN 9.3 不支持 CCU/EID”。

修复将 `total - relay` 放在 CPU `GeneArgs` 中，复用原有参数槽；active mask 从高位到低位，用无符号比较和加上预计算的负常量解码，不产生设备 Sub/And。仍为 46 个 task args、最多 49 channels、每次一条 kernel 提交、无 relay HBM，外部布局/handle ABI 不变，2/4 卡 kernel 不变。新的 `A5CcuEidV1CompatAbiVersion=1` 用于阻止测试误加载旧 route 库。

**更新代码后，必须按 §3 用目标 CANN 重新编译并安装 route 包，重启测试进程。** 若已安装 `f1859d2` 对应的 DeepEP wheel，本修复不改扩展 ABI，无须仅为此重新打 wheel；若 wheel 更旧，仍按 §4 更新。构建输出必须包含 `PUBLIC_ROUTE_ABI A5CcuEidV1CompatAbiVersion=1`。

保留 §11.2 的机器环境变量，先跑：

```bash
cd /home/l00934901/sgl-kernel-npu
bash scripts/run_a5_ccu_eid_alltoallv.sh --suite create
```

它仍使用八卡进行通信组初始化、配置一致性检查、EID 建链和 kernel 注册，但不创建 payload host、不执行 AllToAllV、不跑性能循环。八个 rank 正常释放并退出后输出 `EP8_EID_CREATE_ALL_RANKS_AND_EXIT_PASS`，**这不是数据正确性 PASS**。通过后继续 §11.2 原有 FP8/BF16 正确性和长跑命令，再测 §11.3 性能。

新日志会在每次启动的 `run.log` 中输出 `CCU_CREATE_STAGE_BEGIN/OK/FAILED/EXCEPTION`，区分实例查询、线程申请、ChannelAcquire、注册 Start/Register/End 和 task args 查询；保存 SDK 原始状态且不自动重试。`catalog_register_begin` 同时记录 manifest 绝对路径及 SHA256，不能只靠没看到旧路径来判断使用了哪个文件。

如仍失败，先贴本次 `run.log` 的 `PUBLIC_CCU_BUILD_FAILED` 和 `CCU_CREATE_STAGE_FAILED/EXCEPTION` 附近日志。需要 HCCL/HCCP INFO 日志时，在独立诊断进程中运行：

```bash
A5_EID_LOG_LEVEL=1 A5_EID_LOG_STDOUT=1 \
  bash scripts/run_a5_ccu_eid_alltoallv.sh --suite create
```

仅当阶段日志指向通道/注册资源容量时，再以 `A5_EID_MAX_RELAYS_PER_PEER=0`（7 通道）或 `1`（最多 14 通道）作定位对照；这些不是完整候选目录的等价性能配置。所有结果仍写 `/home/l00934901/profiling`，每次用新进程，不复用失败后的注册实例。

本修复在 `cam_lyw_92` / CANN 9.2.0 完成 route 编译、严格链接、SDK host-test 和实际动态加载（含新能力标记）。12 组 69 项 CPU 回归通过。受限指令模型主动拒绝 Sub/And：旧 kernel 在 Sub 处失败；修复后 202 种 active mask（含第 48 位）、100 epochs × 8 ranks 指令图和 FP8/BF16 host 交替检查通过。结果位于开发机 `/home/liuyuanwen/profiling/a5-ep8-eid-v1-fix`，见 `route-final.log`、`cpu-suite.log`、`before-v1.log`、`sdk-instruction-evidence.txt`。这是编译和离线验证，目标 9.3 八卡仍需先跑上述 `--suite create`，再做数据正确性验证。

### 11.7 `register_kernel status=7`：减少 EID kernel 资源并查询 SDK 需求

现场 `ce130f2`：49 通道状态正常，`register_start` 成功，`register_kernel` 返回 `HCCL_E_UNAVAIL(7)`；与之前的 `repType[11]` 异常不同。当前证据把失败定位在注册阶段，但**仅凭 7 不能确定耗尽的是 event、变量、指令还是其他资源**。容器重启不排除固有配额或设备共享资源压力；`instance=2` 是不透明句柄，不表示容量。

核对 9.2 公开 SDK 后，发现原图有确定的多余申请：

- `vector<Event>(49)` 默认构造就申请 49 个 event，随后逐项赋值 `CreateCompletedEvent()` 又申请 49 个；move 赋值仅更换句柄，不释放第一次构图申请。因此仅远端写完成事件就申请了 **98 个**。
- `remoteOutput/remoteToken` 默认构造先申请本地变量，随后绑定 channel 借用变量，产生不必要申请。
- 按 49 条候选路径分别构造地址和长度，但策略限制每 peer 最多同时执行 direct 和一个 relay，两者都写到同一 peer 的最终 recv buffer。

修复保持全候选目录与真实 EID 转发：远端写 event 改为 **14 个**，地址描述符改为 **14 对**，按 peer 共享最终 recv 地址/token，移除冗余 direct/每路径长度变量。以上数字只统计远端 payload 部分，不包含 channel、self-copy 和 SDK 自身资源。不同 peer 不共享完成事件；同一 peer 的所有 relay 候选互斥，才允许共用一个 relay 完成事件。incoming-only rank 的 metadata/done 握手保留。46 个 task args 不变，不增加 kernel launch，不落 relay HBM。

**先按 §3 重新编译安装 route，保留 §11.2 环境变量，用新进程测试。** 已有 `f1859d2` wheel 时无须为本修复重打 wheel；构建输出新增 `PUBLIC_ROUTE_ABI A5CcuEidPeerResourceAbiVersion=1`，测试会校验，防止加载旧图。

```bash
cd /home/l00934901/sgl-kernel-npu
# 完整候选目录，先只建链、注册及退出；不传 payload。
A5_EID_MAX_RELAYS_PER_PEER=6 \
  bash scripts/run_a5_ccu_eid_alltoallv.sh --suite create
```

若仍失败，或需要查明现场资源需求，在**新进程**开启诊断：

```bash
A5_CCU_RESOURCE_DIAGNOSTICS=1 \
A5_EID_MAX_RELAYS_PER_PEER=6 \
A5_EID_LOG_LEVEL=1 A5_EID_LOG_STDOUT=1 \
  bash scripts/run_a5_ccu_eid_alltoallv.sh --suite create
```

诊断仅发生在注册前，不进入执行/性能热路径。它用独立 kernel 对象调用 `HcommCcuKernelQueryResReq`，然后正式注册使用另一个对象，避免混用 DSL 句柄。查询失败保留原始状态；不自动换实例或重试失败注册。每类输出包含 loop、CCU buffer、variable、address、event、CCU thread、instruction：

| `PUBLIC_CCU_RESOURCE kind=` | 含义 |
| --- | --- |
| `kernel_required` | SDK 对当前 kernel 的资源需求统计；仅 status=0 时有效 |
| `instance_owned` | 通信域实例在该 die 上拥有/占用的资源，**不是实例内剩余量** |
| `device_max_contiguous_remaining` | 设备各类资源的最大连续剩余块，**不是总空闲量，也不是实例内剩余量** |

不能把后两行直接相减当作实例可用配额。若 query 自身返回错误，也保留 `PUBLIC_CCU_BUILD_FAILED` 和底层 plog；公开 query 不保证每个 SDK 版本都能统计这种借用 channel 的图。

`HcclCommQueryCcuIns` 已返回通信域拥有的实例；本库借用它，不进行额外 `HcclCommAssignCcuIns` 或 destroy。RegisterStart 成功仅代表会话开启，不能证明后面的 kernel 资源能分配成功。若仍失败，请贴上述三类资源行、`CCU_CREATE_STAGE_FAILED` 和同时间 plog 的第一处资源申请错误，再决定是否需调整实例/SDK 配置。

只有看到 `EP8_EID_CREATE_ALL_RANKS_AND_EXIT_PASS` 后才继续 §11.2 正确性和长跑，再进行 §11.3 性能测试。资源优化本身不构成目标 9.3 注册成功或性能提升的证明。

开启诊断时还输出 `PUBLIC_CCU_BUILD_STATUS status=... task_args=...`，区分 DSL callback 正常返回/非零返回与后续注册阶段；原 `PUBLIC_CCU_BUILD_FAILED` 只捕获异常，缺少该行并不能证明 callback 一定成功。

开发验证：`cam_lyw_92` / CANN 9.2.0，route 完整编译、严格链接、SDK host-test 和动态加载通过；12 组 69 项 CPU 回归通过，含真实 EID 指令图 100 epochs × 8 ranks、202 种 mask、FP8/BF16 交替 100 次。新增检查强制远端写 event 只构造 14 次，并验证每次启动不会有两条 Write 共用同一完成事件。结果目录 `/home/liuyuanwen/profiling/a5-ep8-eid-peer-resources`（`route.log`、`cpu-suite.log`）。开发容器无 NPU；目标 9.3 的 create、正确性、长跑均待复测。

### 11.8 `1e33bca` 注册通过、`register_end status=4`

现场 `a5-ep8-eid.IyV5jk` 已从 `register_kernel=7` 推进到 `register_kernel=0`、`register_end=4`。目前贴出的 device 4–7 / die=1 均失败；device 0–3 的完整结果仍需核对。**本阶段没有证明 EID 已可执行，勿开始 payload 或模型测试。**

`PUBLIC_CCU_RESOURCE ... status=0` 表示查询成功，不等于所有资源足够。`device_max_contiguous_remaining event=352` 是设备该 die 的最大连续空闲块，不是当前 instance 的 event 可用配额。新图注册成功说明它通过了本次 Register 的检查，但既不能倒推旧图的 status=7 与资源无关，也不能说明 End 一定成功。`PUBLIC_CCU_BUILD_STATUS status=0 task_args=46` 说明 callback 完成并记录了 46 个参数槽，不是硬件执行或微码翻译已通过。

公开源码及开发容器的 CANN 9.2 `libhcomm.so` 调用链：

```text
本 rank：RegisterStart → Register(callback 构图) → RegisterEnd
                                                    ├─ instance.EndRegister()
                                                    └─ kernelMgr.Translate(untranslatedKernels)
                                                         └─ 翻译及设备指令下发
```

这是本进程实例的注册事务；该调用链没有其他 rank 的 barrier。各 rank 必须在执行通信前完成自身注册，但无需为了 RegisterEnd 加一轮跨 rank 同步，也不能跳过 End。公开文档将 End 的内部错误列为翻译、设备内存拷贝或调用顺序问题；具体是哪条分支，仍需目标 9.3 的内部错误。

旧 `scope=peer_resources stage=register_kernel` 实际包裹了 `a5ccu::RegisterKernel` 的整个查询及 Start/Register/End 流程，其 213–235 ms 包含资源诊断及 90–105 ms 的 Register。它不是只测 End 的 1 ms，也不证明等待其他 rank 超时。现已将外层名改为 `register_transaction`，另输出 `resource_diagnostics` 耗时；SDK Register 的内层阶段名不变。

还修复了日志配置：旧 launcher 默认用 `A5_EID_LOG_LEVEL=3` 覆盖已设置的 `ASCEND_GLOBAL_LOG_LEVEL=1`，所以仅设后者未必得到 INFO。现在 A5 专用设置优先，否则保留用户已有 SDK 设置；最终值写入结果目录 `environment.log`。资源行增加 device/PID，并一次写出所有字段，避免八进程交错而把数值误归到其他 rank。

下一次先**关闭资源查询**，保持相同的49通道，单独检查正式注册流程。使用新进程，不复用失败实例：

```bash
cd /home/l00934901/sgl-kernel-npu
A5_CCU_RESOURCE_DIAGNOSTICS=0 \
A5_EID_MAX_RELAYS_PER_PEER=6 \
A5_EID_LOG_LEVEL=0 A5_EID_LOG_STDOUT=1 \
  bash scripts/run_a5_ccu_eid_alltoallv.sh --suite create
```

这条命令在 `1e33bca` 也可执行。若已更新本节日志修复，先按 §3 重编 route；wheel ABI 未变。不要增加 barrier、重试 RegisterEnd 或换通信域来掩盖错误。若关闭查询后通过，再在**另一个新进程**把 `A5_CCU_RESOURCE_DIAGNOSTICS` 改为1作对照；若仍失败，保留完整 `run.log` 中第一处 SDK 错误及八个 device 的 End 结果，而不只保留我们打印的 `CCU_CREATE_STAGE_FAILED`。结果继续存放 `/home/l00934901/profiling`。

依据：[HcommCcuKernelRegisterEnd 文档](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/commlib/commopdev/docs/en/api_ref/comm_opdev/control_plane_api/ccu_kernel_launch_execution/HcommCcuKernelRegisterEnd.md)、[hcomm 公开注册实现](https://gitcode.com/cann/hcomm/blob/master/src/base_comm/primitives/api_c_adpt/ccu/ccu_launch.cc)。公开版本及本地9.2只用于核对调用语义，不等同于现场9.3二进制。本轮不改 EID 数据路径，尚未确认 End=4 的具体根因。

### 11.9 create 前置 AICPU `0x2a` 与 End=4 分开定位

现场 `1e33bca` 的三组报告：诊断开时八卡 End=4；诊断关的两次运行没有 End 记录，Runtime 报 AICPU `retCode=0x2a`。这不能证明同一个 EID Register 在随机失败，也不能排除查询影响：两组可能尚未走到相同调用阶段。

`A5_CCU_RESOURCE_DIAGNOSTICS=0` 只跳过实例/设备资源查询和额外的 `KernelQueryResReq` 查询图；正式 Start/Register/End 均在开关判断外。开关=1会多构一次查询图，所以不是纯打印开关。不开诊断时成功的 `PUBLIC_CCU_BUILD_STATUS` 也会被抑制，不能用缺少该行判断没构图。

create 测试的真实前置流程如下（无 EID payload，但并非没有设备任务）：

```text
device init → HCCL process group init → digest 张量创建
→ 原生 HCCL all_gather（检查8个rank配置一致）→ CPU比对
→ DeepEP Buffer初始化 → EID实例/线程/建链 → 可选资源查询
→ RegisterStart → Register → RegisterEnd → 释放及退出
```

`0x2a` 是 Runtime 报告的 AICPU 执行异常，不是 `HcclResult=42`，也不直接表示 CCU 配额不足。还需任务名、底层 `errcode` 和对应调用栈；`sqe_type=5(aicpu_d)`只是本条Runtime日志标出的任务类别。官方故障处理示例同样要求结合 `fault op_name` / kernel name / 内层 errcode 定位：[AI CPU算子执行报错](https://www.hiascend.com/document/detail/zh/canncommercial/80RC2/developmentguide/maintenref/troubleshooting/troubleshooting_0151.html)。

新增启动阶段日志：`device_init_*`、`process_group_*`、`config_digest_*`、`config_allgather_*`、`config_check_*`、`buffer_init_*`。digest创建、原生all_gather、Buffer初始化后分别同步，防止其异步错误拖到后续 EID 阶段才暴露。配置一致性检查保留，不更换通信后端、不增加跨rank barrier、不改变性能计时路径。

优先读取已失败B/C目录的最后 `EP8_EID_PHASE`、`CCU_CREATE_STAGE` 和首个Python traceback：若没有 `catalog_register_begin`，失败在测试前置阶段，不能归因于EID注册时的诊断开关；若已进入，则按最末成功子阶段定位。End=4这一路仍需A组的SDK翻译/下发错误，不能用B/C的0x2a替代其根因。

本节只补足阶段边界，不宣称修复 AICPU 或 End=4。尚未push的提交必须由开发机push后，新机才能pull；不要把本地commit号当作远端已发布版本。

### 11.10 all_gather 等待诊断、最小 create 与缓冲区缩减

本节保留HCCL配置all_gather，不使用FileStore/Gloo。当前顺序是配置检查→Buffer→EID建链注册，所以停在config_allgather时**本次还没有申请49条EID通道**。完整测试复用一个EP进程组（8个rank各持本地句柄），不能把它解释成额外创建8个独立通信组。之前运行的资源残留尚需现场确认；资源查询status=0仅表示查询成功，不表示资源需求一定满足。

**降低初始化缓冲区。** 仅 `--suite create` 在调用者未设置HCCL_BUFFSIZE时，默认从旧脚本2300改为64（MB）。显式设置优先，correctness/benchmark仍保持原默认2300。这是减少HCCL共享缓冲占用，不是增加CCU实例配额，也不保证修复IndOp或RegisterEnd。官方说明：[HCCL_BUFFSIZE](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/commlib/hcclug/docs/en/user_guide/hccl_env/HCCL_BUFFSIZE.md)。实际值写入environment.log和startup_resources阶段；对照实验必须显式使用相同值。

**等待定位。** 配置检查改用 `async_op=True` 加显式Work.wait（180秒）及原有设备同步。日志依次为：

```text
config_allgather_begin       dtype/shape/output_shapes/async_op/input_bytes/output_bytes/caller_stream
config_allgather_submitted   submit_us
config_allgather_wait_begin
config_allgather_wait_complete wait_us
config_allgather_sync_begin
config_allgather_sync_complete sync_us
config_allgather_complete    total_us
```

Work.wait返回不一定等同host已等到设备完成，所以保留后续同步。caller_stream是调用者流，不冒充HCCL内部流。异常会打印 `config_allgather_failed stage=submit|wait|device_sync`（JSON字段）和 `EP8_EID_FAILURE`，继续抛原始异常，不重试、不吞错。配置输入int32[32]共128B，8个输出共1024B。

**步骤1：相同启动条件的最小变体。** 仍保留完整测试相同的导入、route ABI校验和通信组初始化，配置检查后直接销毁进程组；不创建Buffer、不申请EID channels、不注册kernel、不额外做退出barrier。

```bash
HCCL_BUFFSIZE=64 A5_CCU_RESOURCE_DIAGNOSTICS=0 \
A5_EID_LOG_LEVEL=0 A5_EID_LOG_STDOUT=1 \
bash scripts/run_a5_ccu_eid_alltoallv.sh --suite create --minimal
```

只有8rank干净退出才显示 `EP8_EID_MINIMAL_ALL_RANKS_AND_EXIT_PASS`。它不等同EID注册/数据正确性通过，也不能用于性能比较。最小与完整模式在配置检查前没有EID通道规模差异，因此不能用它们一次通过/失败认定49通道导致前置检查失败。

**步骤2：同一缓冲配置执行完整create。**

```bash
HCCL_BUFFSIZE=64 A5_CCU_RESOURCE_DIAGNOSTICS=0 \
A5_EID_LOG_LEVEL=0 A5_EID_LOG_STDOUT=1 \
A5_EID_MAX_RELAYS_PER_PEER=0 \
bash scripts/run_a5_ccu_eid_alltoallv.sh --suite create
```

仅当走过config_check_complete/buffer_init_complete/catalog_register_begin后，0/1/6个relay候选的注册结果才可用于分析通道规模。完整create PASS仍为 `EP8_EID_CREATE_ALL_RANKS_AND_EXIT_PASS`。

**步骤3：可选预热对照。** 在上述任一命令末尾加 `--config-prewarm 1`，会先用同一组digest/输出张量执行并完成一次小all_gather，然后再做正式配置检查；日志前缀 `config_prewarm_0`。默认0，预热失败立即退出，不自动重试，不另建通信域或stream。它可能使延迟初始化提前发生，也会增加一次通信，不应当作清理资源或修复问题的保证。减少测试buffer是否改变现象，可用显式HCCL_BUFFSIZE=2300与64作对照，其他条件保持相同。

**资源回收审计。**

- 本测试正常路径使用Buffer获取的已有EP组名；C++ Buffer只在组名为空的fallback分支新建通信域，本次config检查前尚未构造Buffer。
- 当前HcclCcuUrmaPeerPlanDestroy关闭bridge/self-copy并删除plan登记，**不逐条调用HcclChannelDestroy，不释放HcclThreadAcquire借用的线程，也不销毁通信域或实例**。不能把plan_destroy_complete解释为所有SDK资源已释放。
- create新增plan_destroy、create_exit_barrier、process_group_destroy的begin/complete日志。正常路径由测试拥有者调用dist.destroy_process_group。最小模式不创建plan，直接销毁组。
- SDK公开了HcclChannelDestroy，但通道可能被通信域缓存/其他plan复用；不能未经所有权确认直接在通用Destroy里批量释放。当前公开kernel wrapper还保留注册对象至进程结束，route slave stream也未在plan destroy中显式销毁，因此同一进程反复create/destroy不能视为零资源累积。此轮不改这些共享资源生命周期，避免模型侧已有handle悬空。
- 异步设备故障或SIGKILL后不保证Python清理执行；不能通过杀其他用户进程、重置设备来代替定位。当前仅能确认正常退出调用到哪一步，不能证明驱动侧残留已回收。

本轮只改测试控制及启动参数，wheel/route ABI不变；已有§3更新route的环境不需重编。本次可先拉取后执行步骤1，再执行步骤2。结果仍写入 `/home/l00934901/profiling`。不要把新增诊断或缓冲区缩减表述为已修复真实9.3设备故障。

### 11.11 RegisterEnd=4：逐段注册图，禁止执行诊断内核

> 本节记录截至d6c1caa的旧累积分段。最新实现已移除生产mask解码，执行顺序与新增探针见§11.12。

现场HCCL_BUFFSIZE=64时前置检查通过，relay上限0/1/6均在End失败。这排除了“只有relay通道才触发”的解释，但relay=0仍使用EID图（46参数、掩码解析、通知、GroupCopy等），不能推出所有内核的公共注册过程都失败。资源查询返回0仍不等于配额足够。

本节使用同一个EP组、同样的建链及Start/Register/End封装，只逐步增加注册图内容。每次启动只注册一个阶段，失败后不在同一实例上重试。partial阶段不调用CCU kernel launch，因此只验证注册，绝不表示传输正确或模型可用。

| --register-stage | 相对上一阶段加入的内容 |
|---|---|
| minimal | 一个Variable和立即数赋值；0个runtime task args，不引用通道变量 |
| args | 正式EID的46个LoadArg；尚无通道引用/同步/读写 |
| mask | live掩码解析、条件分支、初始化peer变量；尚无通道引用 |
| channel_vars | 借用已建通道的output/token变量；尚无notify/wait |
| sync | 地址/token发布、NotifyWait及peer变量拷贝 |
| direct | 14个完成事件、源/目的地址计算、直连Write；尚无relay写/本地拷贝/最终完成握手 |
| relay | 加入relay Write选择分支（需候选relay>0才有新增写路径） |
| self | 加入GroupedLocalCopy |
| full | 完成事件等待及最终通知握手，生产完整图 |

以上阶段仍经过同一EID建链流程；例如max-relays=0时每rank仍建7条直连通道，minimal只是不在kernel里引用它们。mask阶段包含现有peer变量初始化，direct阶段同时加入地址/事件分配，不声称是单条指令的A/B比较。首次失败阶段确定后，再细分该阶段。

先按§3重新编译安装route，必须看到 `A5CcuEidRegisterProbeAbiVersion=1`。**wheel接口未变，不用重打。** 新测试拒绝未包含该标记的旧route，防止参数被旧实现忽略后误测完整图。

先单独执行最小注册图（区别于§11.10的 `--minimal`，后者完全不注册）：

```bash
cd /home/l00934901/sgl-kernel-npu
HCCL_BUFFSIZE=64 A5_CCU_RESOURCE_DIAGNOSTICS=0 \
A5_EID_LOG_LEVEL=0 A5_EID_LOG_STDOUT=1 \
A5_EID_MAX_RELAYS_PER_PEER=0 \
bash scripts/run_a5_ccu_eid_alltoallv.sh --suite create --register-stage minimal
```

通过后每次只把stage改成下一项：`args → mask → channel_vars → sync → direct → relay → self → full`。先固定max-relays=0，relay阶段此时不会新增relay写，可直接用于验证self/full；若直连完整图可注册，再固定max-relays=1从头对照，最后才测6。第一个失败处停止并保留该阶段以及前一个成功阶段的日志。不要在循环里忽略失败继续测试。

partial阶段通过输出 `EP8_EID_REGISTER_PROBE_ALL_RANKS_AND_EXIT_PASS stage=...`；full通过仍是 `EP8_EID_CREATE_ALL_RANKS_AND_EXIT_PASS`。每个rank打印 `EP8_EID_GRAPH_BUILD ... stage=...`，End失败追加raw_ccu_status、mapped_hccl_status、instance/die、kernel、task_args。所有结果留在 `/home/l00934901/profiling`。C++ plan记录创建时的stage，partial plan即使之后修改环境变量，也会在Execute入口报 `registration_probe_cannot_execute`；GeneArgs另有拒绝保护。缓存匹配包含stage，不允许复用不同阶段的plan。

**End内部信息的实际能力边界。** 本机9.2头文件及公开接口签名均为 `CcuResult HcommCcuKernelRegisterEnd(CcuInsHandle)`，没有detail/sub-error输出参数；这是Hcomm接口，外层把CcuResult映射成HcclResult。我们不能直接给现场已编译的libhcomm.so内部加日志，也不调用私有Translate/实例方法重复提交。公开源码的End调用实例EndRegister，再调用kernel manager Translate；此源码版本不等于现场9.3二进制，不能仅凭4区分资源绑定、翻译和指令下发。参考：[公开注册实现](https://gitcode.com/cann/hcomm/blob/master/src/base_comm/primitives/api_c_adpt/ccu/ccu_launch.cc)。

新增 `PUBLIC_CCU_SDK` 用dladdr打印Start/Register/End实际来自哪个动态库。End失败时可选读取ACL recent error，但明确标记 `recent_error_unattributed`：它可能为空、陈旧或属于其他调用，不能冒充CCU子错误。`internal_phase=unavailable`明确表示外层无法观测内部分支。若最小图也失败，保留这组最小复现交由匹配9.3 SDK源码/调试包定位EndRegister和Translate内部返回点。

**minimal汇总。** 081dd25已存在minimal专用分支；本轮进一步兼容历史多进程print造成的PASS行粘连/SDK前缀，新的PASS使用单次os.write。仍要求exit.status=0、8份一致的rank结果及8个相应标记，不把诊断PASS当作正确性PASS。缺标记时打印expected_marker、实际rank集合、minimal/probe模式及汇总脚本绝对路径。无需重跑已有minimal：

```bash
python3 scripts/summarize_a5_ccu_eid.py \
  /home/l00934901/profiling/a5-ep8-eid.ycYc6U
```

如果仍失败，请保留新的完整异常及rank0.json；不能仅凭“missing clean rank completion”认定未适配minimal。


#### 11.11.1 minimal 已有日志的粘连确认

现场rank0.json确认 `suite=create, minimal=true, reports=[]`，这符合minimal设计，汇总器并不要求reports非空。实际漏判来自 `EP8_EID_MINIMAL_RANK_PASS rank=7[INFO] ASCENDCL(...)`：严格行尾匹配漏掉rank7，acc34ec允许空白/下一条EP8标记的版本也尚未覆盖紧邻的`[`。现已补上SDK方括号前缀边界，并验证缺rank及rank=70不会被误认成rank7。

本补丁只修改汇总Python、测试与文档。无需重编route/wheel，不用重新跑卡；用更新后的 `scripts/summarize_a5_ccu_eid.py` 重新汇总上述ycYc6U目录即可。新的worker已使用单次os.write输出PASS，不改回多次写入的print/sys.stdout.write。

### 11.12 d6c1caa 现场结果：移除生产图中的动态 mask 解码

现场已确认 `minimal` 和 `args` 的 RegisterEnd 均8/8通过，`mask` 首次8/8返回4。失败范围已缩到该阶段新增的变量初始化、立即数、比较及加法组合；尚未定位SDK内部的具体拒绝条件。后续阶段继承mask，不能据此证明它们单独兼容。资源查询status=0也仅表示查询成功。

两项报告口径修正：create不调用运行期GeneArgs，所以CPU上的 `totalBytes-relayBytes` 不参与本次注册失败。AICPU初始化日志的 `argsSize=8` 是另一个任务的参数大小，与本图46个uint64参数槽无关。HCCL_BUFFSIZE=64在现场恢复了前置通信，仍不据此断言具体驱动分配冲突机制。

**生产路径变更。** 每个peer固定使用一条直连通道交换接收地址/token，并在所有本rank数据写完成后，用这些直连通道完成对端握手。数据仍按每次规划走直连前缀和一个可选EID relay后缀，relay必须是EP8成员，不在relay HBM中转。不解码active mask、不为mask产生负常量/GE/加法，不按49条路径逐一交换控制信息。控制通道固定7条；数据通道仍最多49条。零行和仅接收rank也参加固定握手，保证双方通知匹配。

kernel与通道仍只初始化一次；每次传入新的偏移、长度和relay编号。46参数槽保持不变，第10槽仍加载但生产图忽略，旧wheel无需重打。新增日志 `EP8_EID_TASK_LAYOUT` 列出全部槽位。新route能力标记为 `A5CcuEidDirectControlAbiVersion=1`，测试与构建脚本必须检查到它。2/4卡实现不涉及本次修改。

这是移除已知失败构图的实现调整，不是已经证明9.3设备修复。固定7peer握手可能增加全零/稀疏负载控制成本；CPU图测试无法验证真实EID链路、硬件顺序、SDK翻译或性能。

**新机执行顺序：**

1. 拉取后按§3重新编译、安装route，确认BUILD/LOAD PASS和上述新标记；不用重打原wheel。不要只更新Python而仍加载旧route。
2. 沿用§11.2已配置的rootinfo、Atlas和CANN路径，再设置：

```bash
cd /home/l00934901/sgl-kernel-npu
export HCCL_BUFFSIZE=64 A5_CCU_RESOURCE_DIAGNOSTICS=0
export A5_EID_LOG_LEVEL=0 A5_EID_LOG_STDOUT=1
export A5_EID_MAX_RELAYS_PER_PEER=0
bash scripts/run_a5_ccu_eid_alltoallv.sh --suite create --register-stage full
```

若失败，保持relay=0，分别跑 `channel_vars → sync → direct → self → full`，在首个失败处停止并保留前后日志。**当前生产阶段不再包含mask**，不沿用§11.11旧版 `args → mask → channel_vars` 的累积含义。成功后再把relay上限改为1、6，各执行一次full create。

3. relay=6的full create通过后，执行正确性及连续调用；本段命令显式保留64MB，避免correctness默认值改变前置条件：

```bash
export A5_EID_MAX_RELAYS_PER_PEER=6
bash scripts/run_a5_ccu_eid_alltoallv.sh --suite correctness --mode all \
  --dtype float8_e4m3fn --width 5120
bash scripts/run_a5_ccu_eid_alltoallv.sh --suite correctness --mode all \
  --dtype bfloat16 --width 5120
```

需看到 `EP8_EID_ALL_RANKS_AND_EXIT_PASS` 才继续§11的双向、长跑及性能测试；create PASS不表示传输正确。结果统一位于 `/home/l00934901/profiling`。

**可选：继续定位旧mask拒绝条件。** 以下阶段只注册、不允许payload执行。它们各自从相同46参数加载开始，互为独立对照；不是必须全部通过才能用新生产路径。

| stage | 注册图内容 |
|---|---|
| mask_init | 原mask变量规模，peer变量与live置0；不比较、不赋负常量、不相加 |
| mask_static | 与init相同，额外把live赋为常量1 |
| mask_negative | 与init相同，增加逐位 `0-bit` 负常量赋值；常量在host计算 |
| mask_compare | 与init相同，仅 `active >= bit` 条件下把live置1，不修改active |
| mask_add | 与init相同，负常量赋值后无条件相加，不做比较 |
| mask | 原完整decoder：负常量、比较、条件加法组合；预期可复现旧故障 |

例如保持其他环境相同，单独执行 `bash scripts/run_a5_ccu_eid_alltoallv.sh --suite create --register-stage mask_compare`。不能把这些probe当作可执行数据内核；不引入V1已不支持的减法/位移表达式。若单项都通过而组合失败，仍需SDK侧查看组合翻译、资源需求及生成指令，不能仅从外层4认定申报数不一致。

RegisterEnd的公开签名没有detail输出参数。外层可打印任务槽与失败阶段，但无法直接给现场libhcomm.so内部增加阶段日志，也没有证据确认本次失败来自“申报指令数与生成数不一致”的检查。

开发侧验证：`cam_lyw_92` / CANN 9.2.0 的route编译、SDK host test、动态加载及新能力标记检查通过。实际生产C++图的离线解释测试覆盖relay上限0/1/6各100轮×8rank，含零行、非对称收发、每行1/2/257/5120/10240字节及连续更换relay；另校验202种旧mask值不再改变控制通道集合、14个注册阶段的payload隔离及FP8/BF16 host交替。构建及CPU回归日志在开发机 `/home/liuyuanwen/profiling/a5-ep8-direct-control/`。这不是CANN 9.3真机注册或性能结果。

### 11.13 比较探针的解释：mask 不等于 full

613abc9现场：args/static/add注册及退出通过；negative八卡End通过但进程失败；compare与mask八卡End=4；init未到注册。**`--register-stage mask`是故意保留的旧故障探针，生产路径是`full`**。生产仍使用`bytes != 0`、`relay_id == k`的条件块和self-copy条件，不使用mask中的`active >= bit`。不能从mask失败推出613abc9生产路径仍执行旧decoder，也不能从GE失败推出所有EQ/NE都不可用。

首先用现有613abc9按§11.12执行`--suite create --register-stage full`。若已运行，先读该次结果，不必重新跑旧mask。新版本的 `EP8_EID_GRAPH_BUILD` 明确打印 `graph=legacy_mask_probe` 或 `graph=direct_peer_control`。

如果需要继续区分比较类型、常量范围和重复条件块，新增四个仅注册探针。需按§3重编route，出现 `A5CcuEidCompareProbeAbiVersion=1`；wheel不变。其他环境保持§11.12一致。

| stage | 相对mask_compare的改动 | 用途 |
|---|---|---|
| mask_equal | 同样的bit常量和块数，仅GE改EQ | 检查相等条件块 |
| mask_not_equal | 同样的bit常量和块数，仅GE改NE | 检查不等条件块 |
| mask_compare_zero | 同样块数，GE右侧改为0 | 区分立即数范围因素 |
| mask_compare_single | 同样变量规模，只生成一个`active >= 1`块 | 区分单块与重复块/规模因素 |

例如：

```bash
bash scripts/run_a5_ccu_eid_alltoallv.sh --suite create --register-stage mask_equal
```

按需要单独运行其余探针，保留各自结果。EQ/NE通过不等于真实Write、嵌套条件、完成通知已通过，仍须full create和正确性测试。若单个GE也失败，可进一步提交最小复现给SDK维护者；不要直接替换成V1此前不支持的AND/Sub/shift。

`CCU_IF`不只是数值比较，它还生成跳转、标签和条件块收尾。开发机9.2头文件声明EQ/NE/GE，但声明并不证明现场9.3特定图能翻译；公开参考实现与安装SDK存在差异，不能套用旧源码中仅EQ/NE的限制作为9.3结论。当前未证实SDK内部拒绝的是谓词、标签还是翻译资源。

**两个rc=1需独立处理。** `mask_init`应该走完整注册过程，不是`--minimal`；没有注册记录表示尚不能评价初始化图，需查看最末startup阶段/traceback。`mask_negative`的End=0不表示整个Create或退出已成功：后续还有get_task_args、plan绑定/登记、destroy、退出barrier、通信域销毁和结果汇总。汇总器对所有partial stage使用同一规则，允许reports=[]；不能仅凭rc=1修改成PASS。已扩展离线汇总测试验证init/negative和新增比较探针均可正常汇总，同时非零exit仍必须失败。新的`EP8_EID_FAILURE`包含`last_phase`及register_stage，辅助区分注册前和退出期异常。

无需重跑即可先读取两次异常目录的run.log末尾60行、终端traceback和exit.status。若缺少这些日志，不能认定是summarizer，也不能把End=0抹成注册失败。

本次开发验证：cam_lyw_92的CANN9.2 route编译、SDK host-test和动态加载通过；12项EID CPU测试通过，包含18个注册阶段的payload隔离、各新增条件块数量、7/14/49通道三种图共2400次离线rank执行及init/negative的汇总判定。日志：开发机 `/home/liuyuanwen/profiling/a5-ep8-compare-probes/route.log`、`eid-cpu.log`。没有9.3真机验证结果，本次不改变full生产算法。

### 11.14 full注册已通过；修复测量阶段FP64归约

现场613abc9的full复测已出现 `EP8_EID_CREATE_ALL_RANKS_AND_EXIT_PASS`：八卡Register/End及退出通过。这确认生产路径移除mask decoder的改动生效，不等于“生产不再有比较”：数据长度NE和relay编号EQ条件仍存在。旧mask/GE探针失败不能替代full结果。

正确性测试随后报 `HCCL allreduce: Unsupported data type at::kDouble`。已定位到EID测试的最慢rank耗时归约：把host测得的微秒值创建成NPU FP64张量，再执行all_reduce(MAX)。现改为FP32；计时、校验、归约先后顺序及MAX口径不变，归约仍在测量区间之外。FP8/BF16 payload与算子C++代码均不变。同目录A5测试扫描未发现其他FP64统计归约；multidtype meta中的FP64是拒绝非法输入的负向测试，保留。

**本修复只需更新Python，不需重编route或wheel。** 测试仍要求613abc9已有的 `A5CcuEidDirectControlAbiVersion=1`；c015f22新增的 `A5CcuEidCompareProbeAbiVersion=1` 现在只在四个新增比较探针运行时检查，full/create/correctness可继续用已装的613abc9 route。若要跑新比较探针，再按§3重编。

更新后沿用§11.2的CANN、rootinfo、Atlas配置，从本次失败的正确性命令重跑即可，不必重复旧mask扫描：

```bash
cd /home/l00934901/sgl-kernel-npu
export HCCL_BUFFSIZE=64
export A5_EID_MAX_RELAYS_PER_PEER=6
bash scripts/run_a5_ccu_eid_alltoallv.sh --suite correctness --mode all \
  --dtype float8_e4m3fn --width 5120 --row-scale 1 --rounds 1 \
  --iters 10 --dynamic-iters 100 --bidirectional-iters 100
```

结果仍写入 `/home/l00934901/profiling`。部分rank的 `EP8_EID_BIDIRECTIONAL_PASS` 与 `measure_begin` 是阶段进展；需最终 `EP8_EID_ALL_RANKS_AND_EXIT_PASS` 才能宣告本轮完整验证通过。若再失败，保留首个 `EP8_EID_FAILURE`（含last_phase）及traceback。

mask_init未到注册与mask_negative End成功但rc=1，仍不能据缺少PASS判定为汇总器假阴性：mask_init应注册kernel，只有`--minimal`才不注册。现有partial专用PASS允许reports为空，但必须等正常清理完成才能写出；不新增跳过失败的PASS出口。它们的退出原因仍待失败目录的原始日志。

## 12. 稠密矩阵与轻载卡 READY 分流（独立新分支）

`feature/a5-ccu-ep8-dense-ready-alltoallv` 新增 C++ `dense_ready` 策略、可调更小分流量与 CCU READY 依赖，仍使用组内 EID/URMA 转发、不落 relay HBM。需要重新构建 route **和** DeepEP wheel。旧默认策略和2/4卡接口保留。

完整拉取、构建、注册、10用例正确性和2/4/8/16比例性能扫描见 [EP8稠密READY指导](A5_CCU_EP8_DENSE_READY.md)。真实8行相同的采样预期不分流；不能把该A/A结果当成多路径加速。
