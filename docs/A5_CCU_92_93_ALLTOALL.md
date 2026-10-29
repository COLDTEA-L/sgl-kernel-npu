# CANN 9.2 / 9.3 完整多路径 AllToAll 分支

分支：`feature/a5-ccu-92-93-alltoall`。从原 `feature/a5-ccu-8rank-heuristic-alltoallv` 的 `370a14d` 建立，原 9.1 分支和旧容器安装产物不修改。本分支要求新公开 CCU API，不用于 9.1。

## 1. 迁移范围与验证边界

| 路径 | 保留功能 |
| --- | --- |
| 2 卡 | 等长多路径 AllToAll、prepared plan、动态权重、显式组外物理 relay、serial/concurrent、self-copy 模式 |
| 4 卡 | 原 peer-plan、组外物理 relay、baseline/packed/packed-ready 通知协议 |
| 8 卡 | EP8 AllToAllV，非对称 C[8][8]、零长度、独立 self 偏移、storage-count 偏移接口 |
| EP8 多路径 | 原 greedy/fast 规划、组内 HBM relay、direct 与双缓冲 relay 流水线、C++ 布局和提交路径 |
| 计时 | 原 native HCCL / custom direct / multipath，prepared 和 dynamic 两种口径 |

2/4 卡不套用 EP8 贪心算法；8 卡 AllToAllV 继续专供 EP8，relay 仅在通信组成员中选择。模型输入接口、规划算法和 C++ pipeline host 未在本次迁移中重写。

服务器已在独立 `cam_lyw_92` 中编译、严格链接、加载整个 route 库及 local-copy 诊断程序；新 host mock 检查固定注册、100 次复用、设备/通信域归属、注册失败不重试；EP8 CPU 图模型执行 100 轮、800 次调度；2/4 卡布局、通知、自拷贝和规划器 CPU 回归通过。

该容器没有 Torch/torch_npu，也未挂载 NPU，因此未在这里编译完整 DeepEP wheel、运行全部 Torch CPU 用例或上卡测试。9.3 接口名称/参数形式来自用户提供的检查结果，仍需在 9.3 本机重编译和上卡。CPU 模型不验证固件、实际 MS/loop 资源或性能；9.2 二进制也不能作为 9.3 的已验证产物。

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
