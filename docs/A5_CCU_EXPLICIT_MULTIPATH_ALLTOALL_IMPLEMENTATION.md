# A5 标准显式多路径 AllToAll：实现与调用链

## 1. 分层结构

```text
Python Buffer.explicit_multipath_all2all_ccu(send, plan_id, weights)
  -> DeepEP C++ / aclnnExplicitMultipathAll2AllCcu
  -> 标准 Ascend OpDef + tiling
  -> AIV 控制 kernel: Hccl<CCU>.Alltoall()
  -> HCCL selector: CcuExplicitMultipathAllToAll2Rank
  -> HCCL CalcResRequest
  -> direct CommLink + manifest EID pairs -> HcclChannelDesc[]
  -> HcclChannelAcquire(desc[i]) -> Channel[i]
  -> 自定义 CCU kernel
  -> LocalCopy(self slice) + WriteNb(weighted peer chunks) + WaitEvent
  -> UB/IO Die forwarding -> peer GM
```

## 2. host 与图边界

`plan_id` 是稳定控制面 key；manifest 和权重在 HCCL communicator/resource 初始化期间解析。resource cache 保存
Channel、CCU thread、registered kernel 和 cached launch args。图内 AIV 不访问文件系统，也不建 Channel。

DeepEP 在 ACLNN 前检查 HCCL 扩展标记、plan ID、weights 和 manifest，防止标准 AllToAll selector 静默接管。

## 3. 路径构造

HCCL template 用原生 `CalcChannelRequestMesh1DWithPriorityTopo` 得到 direct descriptor，再逐行读取 relay
manifest。rank0 使用 `src_eid -> dst_eid`，rank1 反向使用；每行通过同一 `HcclChannelAcquire` 建独立 Channel。

它不修改全局 UBUS route table。目的 EID 选择已经 provision 的 endpoint/path，relay 卡只执行 IO Die transit。

## 4. CCU 并发与完成协议

kernel 参数为 self copy 三元组及每条路径的 `[peerSrcOffset, peerDstOffset, bytes]`。执行顺序：获取远端地址与
token、全 Channel 启动同步、local copy、提交全部非阻塞 write、再等待全部 event、最后 completion notify。
所以并发由设备端提交顺序保证，而不是 host 串行组合多个通信原语。

## 5. 图回放

首次 launch 把 task args、arg count 和 input/output base offset 存进 `CcuKernelSubmitInfo::cachedArgs`。
FastLaunch 只重定位新 tensor 地址，不能依赖重建 template 的临时 `weights_`，从而保持 path count 和 chunk layout。

## 6. 主要文件

DeepEP 仓：

- `csrc/deepep/ops/op_host/explicit_multipath_all2all_ccu_def.cpp`
- `csrc/deepep/ops/op_host/explicit_multipath_all2all_ccu_tiling.cpp`
- `csrc/deepep/ops/op_host/op_api/aclnn_explicit_multipath_all2all_ccu.cpp`
- `csrc/deepep/ops/op_kernel/explicit_multipath_all2_all_ccu.cpp`
- `csrc/deepep/deep_ep.cpp`
- `tests/python/deepep/test_a5_ccu_urma_multiroute_all2all.py`

HCCL 仓：

- `src/ops/all_to_all_v/selector/alltoall_auto_selector.cc`
- `src/ops/all_to_all_v/executor/ins_v2_all_to_all_v_sole_executor.cc`
- `src/ops/all_to_all_v/template/ccu/ccu_temp_explicit_multipath_alltoall.cc`
- `src/ops/all_to_all_v/template/ccu/kernel/ccu_kernel_explicit_multipath_alltoall.cc`

## 7. 多卡与动态控制扩展点

多卡 plan 应升级为 `peerPlans[srcRank][dstRank]`，每项持有 direct、relay PathSpec、weights 和 die/thread group。
host 控制器可在 communicator 创建前生成内存 plan，并让 plan ID 参与 resource cache key。切换路径集合应创建新
resource/graph，不能修改正在 replay 的 Channel。
