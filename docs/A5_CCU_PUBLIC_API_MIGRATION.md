> 本文保留独立两卡 smoke 的说明。完整 2/4 卡及 EP8 迁移现位于 `feature/a5-ccu-92-93-alltoall`，部署按 [A5_CCU_92_93_ALLTOALL.md](A5_CCU_92_93_ALLTOALL.md) 执行；验证边界以该文档为准。

# CCU 公开接口迁移：9.2 编译验证与 9.3 两卡入口

分支：`feature/a5-ccu-public-api-smoke`，基于 `370a14d`。

本阶段交付独立两卡程序，不修改现有 9.1 route 库或 DeepEP。先在目标 9.3 上验证新 API 的实例、通道、注册和重复启动，再迁移现有 EP8 AllToAllV 执行器。编译通过不等于两卡运行通过，也不代表跨版本二进制兼容。目标 9.3 必须用自己的头文件和库重编译。

## 1. 已核对的 9.2 / 9.3 接口

9.2 证据来自服务器现有 `deep-ep-ascend-a5-build` 的安装文件和 `libhcomm.so`；9.3 证据来自用户提供的接口检查结果。

| 能力 | 9.2 实际声明 | 与用户提供的 9.3 检查结果 |
| --- | --- | --- |
| 实例查询、绑定 | `include/hccl/hccl_ccu_res.h`：`HcclCommQueryCcuIns`、`HcclCommAssignCcuIns` | 名称和参数形式相同 |
| 创建默认实例、查询 kernel 资源 | `include/hcomm/ccu/ccu_res.h`：`HcommCcuInsCreateDefault`、`HcommCcuKernelQueryResReq` | 名称和参数形式相同 |
| 获取 thread、thread 对应 stream | `include/hccl/hccl_res.h`：`HcclThreadAcquire`、`HcclThreadAcquireWithStream`、`HcclThreadResGetInfo` | 名称和参数形式相同 |
| 建链、远端内存信息 | `include/hccl/hccl_channel.h`：`HcclChannelAcquire`、`HcclChannelGetRemoteMems` | 名称和参数形式相同 |
| 内存注册 | `include/hccl/hccl_res.h`：`HcclCommMemReg` | 名称和参数形式相同 |
| 注册 | `HcommCcuKernelRegisterStart/Register/End` | 与之前提供的 9.3 公开接口形式一致 |
| 启动 | `HcommCcuKernelLaunch(thread, kernel, taskArgs, argNum)` | 与之前提供的 9.3 公开接口形式一致 |
| 变量、token | `AscendC::ccu::Variable`、`HcommCcuGetMemToken` | 与之前提供的 9.3 公开接口形式一致 |

9.2 的上述关键实例管理函数已核对动态导出。9.2 的 `HcommThreadAllocWithStream` 虽有动态符号，但当前公开头文件未声明，本程序不使用它。旧 `CcuRep` 的部分符号仍存在，但旧 `HcclCcuKernelRegister/Launch` 入口不在该 `libhcomm.so` 导出表中，所以不能按旧 ABI 继续链接。

### 实例生命周期

根据 9.2 实际头文件：

- `QueryCcuIns` 返回通信域自有的借用实例；CCU 模式下可以按需创建。本程序使用此路径，不销毁、不再次 Assign 该实例。
- `CreateDefault + AssignCcuIns` 是显式创建并转交所有权的另一条路径。Assign 成功后，销毁责任归通信域；重复 Assign 会失败。
- `KernelQueryResReq` 用于资源需求查询，不是所有 kernel 每次启动的必经步骤。
- 注册需要 `insHandle`、`dieId` 和构造参数表；启动仅传 thread、kernel 和运行参数表。构造参数与运行参数的生命周期、用途不同，不能只机械改名。

## 2. 9.3 容器内编译与加载检查

以下均在 `yuanwen_dsk` 内执行。根据实际仓库位置修改 `cd`；profiling 统一放个人目录。

```bash
cd /home/l00934901/sgl-kernel-npu
bash scripts/build_a5_ccu_public_api_smoke.sh \
  /usr/local/Ascend/cann-9.3.0 \
  /home/l00934901/profiling/ccu-public-930-build

(
  set +u
  source /usr/local/Ascend/cann-9.3.0/set_env.sh
  /home/l00934901/profiling/ccu-public-930-build/a5_ccu_public_api_smoke --check-abi
)
```

预期：`PUBLIC_CCU_BUILD_PASS`、`PUBLIC_CCU_ABI_PASS`。加载检查会打印 `HcommCcuKernelLaunch`、`HcclCommInitRootInfo` 的实际提供库路径，不初始化 ACL、不申请 NPU。服务器 9.2 实测这两个符号均由该安装的 `libhcomm.so` 提供。构建脚本显式使用选定 CANN，不会再次加载 `/usr/local/Ascend/cann` 别名。不要把 9.1 的库预加载进这个进程。

## 3. 两卡正确性与 100 次复用

确认所选两张卡可用后执行；示例物理卡为 2、3，可自行调整。

```bash
cd /home/l00934901/sgl-kernel-npu
(
  set +u
  source /usr/local/Ascend/cann-9.3.0/set_env.sh
  python3 scripts/run_a5_ccu_public_api_smoke.py \
    --binary /home/l00934901/profiling/ccu-public-930-build/a5_ccu_public_api_smoke \
    --devices 2,3 --bytes 4096 --iters 100 \
    --output-root /home/l00934901/profiling
)
```

预期每 rank `PUBLIC_CCU_RANK_PASS ... iterations=100 registrations=1 channels=1`，launcher 最后输出 `PUBLIC_CCU_ALL_RANKS_AND_EXIT_PASS`。必须同时满足两 rank 校验通过和正常退出。

`--bytes 4096` 是每 rank 缓冲区容量，单位字节，不是 hidden size。每四次启动依次测试：双向全长；0→1 全长、1→0 半长；0→1 零长、1→0 全长；双向零长。每次改变数据并校验接收内容和未写区域，验证固定注册、多次启动时不会读到旧结果。每轮的长度由运行参数传入，不重新注册 kernel 或创建通道。

路径从当前 HCCL rank graph 选择一跳 UB_CTP，并查询本地 die；不写死 EID，不使用自制路由 txt。设备的系统 rootinfo / 拓扑配置仍需正确。两 rank 利用 channel 交换接收地址与 token；token 不打印。bootstrap 和校验同步文件保存在独立结果目录，属于控制数据，不决定路由。

这是正确性程序，包含每轮 H2D/D2H、完整数据校验及文件同步，不提供算子性能结论。注册参数带通道句柄；运行参数为输入/输出地址、token、字节数。运行失败会保留 rank 日志和阶段标记，launcher 在单 rank 失败或超时后回收其余 worker。

## 4. 验证边界与后续迁移

本次服务器验证使用独立临时 9.2 镜像容器，仅做编译和加载，不启动停止的他人容器、不挂载 NPU。已经取得 `PUBLIC_CCU_BUILD_PASS` 和 `PUBLIC_CCU_ABI_PASS`，动态依赖检查未发现旧 `hcomm::CcuRep` / `HcclCcuKernel*` 接口。构建结果保存在服务器 `/home/liuyuanwen/profiling/cann92-ccu-abi-wJLnOX/build`。

两卡运行验证通过后，再将新接口用于生产路径：先迁移 EP8 direct AllToAllV 的注册、参数打包、token 和 CCU kernel；通过非对称、自偏移、零长度后，再迁移 direct/relay pipeline，并验证 100 次复用。2/4 卡旧接口继续保留。尚未完成的两卡上卡验证、EP8 迁移和模型接入不能由本阶段的编译通过替代。
