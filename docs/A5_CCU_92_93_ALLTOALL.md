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

启动时注册一个 communicator/stream 的候选 Channel 和 kernel，每 peer 最多一条 direct 加六条 relay，每 rank 最多 49 条 Channel。动态参数压缩为 46 个 task args：每次只更新 C、relay 选择、偏移和长度；零流、receive-only、两个方向选不同 relay 使用双向 active mask 协议。FP8 dispatch 与 BF16 combine 可复用同一个 host/handle，分别按真实 tensor 的 row_bytes 规划。

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
