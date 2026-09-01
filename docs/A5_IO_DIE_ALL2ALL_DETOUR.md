# A5 IO Die All2All 绕路算子

## 目标与实现差异

`All2AllDetourIoDie` 用于 A5（Ascend 950）上的通信 rank 子集 All2All。

旧实现依赖 `GetHcclContext()` 中的 `windowsIn[peerRank]`，由通信卡直接读写绕路卡 HBM。在 A5 环境中这些远端 window 地址可能为 0，因此无法作为跨卡目标地址。

本实现不读取 `windowsIn[]`，也不把远端地址暴露给 AIV：

1. host 侧将 executor 的 HCCL server type 设置为 `CCU`；
2. kernel 调用 `Hccl<HCCL_SERVER_TYPE_CCU>::AlltoAllV`；
3. `commRankIds` 之外的 rank 使用 0 send/recv count，但仍参与同一通信域中的 collective；
4. 数据转发和链路选择由 A5 CCU/IO Die 完成，接收数据直接写入 `recvData`。

输入 `sendData` 的第 `i` 个等长数据块发往 `commRankIds[i]`；输出 `recvData` 的第 `i` 个数据块来自 `commRankIds[i]`。所有 rank 必须传入完全相同、升序且不重复的 `commRankIds`。

## 已验证环境

- 分支：`feature/a5-io-die-all2all-detour`
- Docker：`cam_lyw_dev_91`
- CANN：`9.1.0-beta.1`
- SoC 编译目标：`Ascend950` / `ascend950`
- 编译结果：FP16、BF16、FP32、INT32 四个 device binary 均成功生成

无卡服务器只能完成编译验证；正确性测试需要在 A5 有卡服务器执行。

## 在 sglang Docker 中准备环境

以下命令均在已有 sglang Docker 内执行，不使用 conda。先加载 CANN 环境：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || \
source /usr/local/Ascend/cann/set_env.sh 2>/dev/null || true

python3 -c 'import torch, torch_npu, pybind11; print(torch.__version__); print(torch_npu.__version__)'
```

如果缺少 `pybind11` 或构建工具：

```bash
python3 -m pip install pybind11 wheel setuptools
```

`torch` 和 `torch_npu` 应继续使用当前能够启动 sglang 服务的版本，不要为了编译算子单独替换。

## 单算子编译

```bash
cd /home/l00934901/sgl-kernel-npu
git switch feature/a5-io-die-all2all-detour
bash scripts/build_a5_io_die_all2all_detour.sh
```

脚本设置 `DEEPEP_SINGLE_OP=all2_all_detour_io_die`，跳过 Catlass 与其他 device kernel。成功时末尾应出现：

```text
All2AllDetourIoDie device binaries: 4
Built: output/deep_ep-....whl
```

安装到当前 sglang Python：

```bash
python3 -m pip install --force-reinstall --no-deps output/deep_ep-*.whl
```

## 选择测试卡

`torchrun --nproc-per-node=N` 使用 `ASCEND_RT_VISIBLE_DEVICES` 中从左到右的 N 张卡。例如选择物理卡 2、5：

```bash
export ASCEND_RT_VISIBLE_DEVICES=2,5
```

## 两卡全通信测试

```bash
cd /home/l00934901/sgl-kernel-npu
export HCCL_BUFFSIZE=2300
export ASCEND_RT_VISIBLE_DEVICES=0,1

python3 -m torch.distributed.run \
  --standalone --nproc-per-node=2 \
  tests/python/deepep/test_a5_io_die_all2all_detour.py
```

两卡场景可验证 CCU 调用、数据布局和正确性，但没有非通信 rank。

## 四卡通信子集 / IO Die 路由测试

下面让 rank 0、2 交换数据，rank 1、3 以 0 count 参与 collective：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3

python3 -m torch.distributed.run \
  --standalone --nproc-per-node=4 \
  tests/python/deepep/test_a5_io_die_all2all_detour.py \
  --comm-ranks 0,2
```

成功时 rank 0 输出：

```text
PASS: A5 All2AllDetourIoDie (subset/IO-Die routing)
```

## 约束

- `rank_size <= 32`，与当前 CANN 9.1 CCU AlltoAllV 参数上限一致；
- `commRankIds` 必须是 NPU 上连续的 INT32 一维 tensor；
- `sendData.numel()` 必须能被 `commRankIds.numel()` 整除；
- 支持 FP16、BF16、FP32、INT32；底层按字节调用 CCU，因此不会做类型转换；
- 通信域中每个 rank 都必须进入算子，否则 collective 会等待；
- 当前版本针对单机 A5 通信域验证，跨机拓扑需要在目标集群另行做正确性和性能验证。
