# A5 CCU + URMA 显式路由验证

## 1. 目标和当前边界

本分支 `feature/a5-ccu-urma-route-probe` 增加一个两 rank、AllGather 语义的最小探针。它使用：

- `HcclThreadAcquireWithStream(..., COMM_ENGINE_CCU, ...)` 获取 CCU engine thread；
- `COMM_PROTOCOL_UBC_CTP` 建立 HCOMM channel；
- `HcclRankGraphGetLayers/GetLinks` 枚举 HCCL/HIXL 已经发布的链路；
- `A5_CCU_ROUTE_INDEX` 从候选链路中选定一条，并打印源/目的 EID、物理卡号和 `hop`。
- `HcclCcuKernelRegister/HcclCcuKernelRegisterFinish` 注册真正的 CCU kernel；
- `HcclCcuKernelLaunch` 把用户 buffer 地址、token 和长度送入 CCU；
- CCU kernel 使用 `NotifyRecord/NotifyWait` 交换远端地址和 token，并用 `WriteNb/WaitEvent` 完成传输。

这一步验证“CCU 能否通过指定的 RankGraph UBC_CTP 链路正确通信”。它不会凭空构造 RankGraph 中不存在的 `1 -> 3 -> 8` 路由，也还不是最终的绕路 AllToAll。若 RankGraph 只返回一条 UBC_CTP 链路，则只能选择 route 0；后续要继续检查 UVS/HIXL 路由配置，而不是仅修改 CCU kernel。

## 2. 文件位置

```text
examples/a5_ccu_urma_route_probe/       CCU 自定义通信算子和两卡测试程序
scripts/build_a5_ccu_urma_route_probe.sh 编译自定义算子包
scripts/run_a5_ccu_urma_route_probe.sh   运行、选路和 msprof 采集
docs/A5_CCU_URMA_ROUTE_VALIDATION.md     本文档
```

## 3. 在有卡服务器拉取分支

仓库存在未提交修改时先用 `git status --short` 检查并自行提交或暂存，避免切分支覆盖工作。

首次拉取该分支：

```bash
cd /home/l00934901/sgl-kernel-npu
git status --short
git fetch origin feature/a5-ccu-urma-route-probe
git switch --track origin/feature/a5-ccu-urma-route-probe
```

本地已经有同名分支时：

```bash
cd /home/l00934901/sgl-kernel-npu
git switch feature/a5-ccu-urma-route-probe
git pull --ff-only origin feature/a5-ccu-urma-route-probe
git rev-parse --short HEAD
```

如果 GitHub 仓库对应的 remote 不叫 `origin`，先用 `git remote -v` 找到正确名称，并替换以上命令中的 `origin`。

## 4. 准备环境（不使用 conda）

进入原来可运行 sglang 的 Docker 后执行：

```bash
cd /home/l00934901/sgl-kernel-npu

source /usr/local/Ascend/cann/set_env.sh 2>/dev/null || \
source /usr/local/Ascend/ascend-toolkit/set_env.sh

export HCCL_OP_EXPANSION_MODE=CCU_SCHED
```

本探针不要在安装前 source `python/deep_ep/deep_ep/vendors/hwcomputing/bin/set_env.bash`。该脚本会把
`ASCEND_CUSTOM_OPP_PATH` 指向 DeepEP vendor 目录；未指定 `--install-path` 的 HCCL 安装包会因此被重定向到
`python/deep_ep/deep_ep/vendors/hwcomputing`，而不是 CANN 的 `opp/vendors/cust`。

编译和运行脚本会自动 source CANN 环境。测试脚本默认选择物理卡 `2,5`，并通过
`ASCEND_RT_VISIBLE_DEVICES=2,5` 映射成逻辑 rank 0、1。

确认 CANN 和 HCOMM：

```bash
echo "ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
ls -l "${ASCEND_HOME_PATH}/include/hcomm/hcomm_res.h"
ls -l "${ASCEND_HOME_PATH}/lib64/libhcomm.so"
urma_admin show
```

## 5. 编译自定义 CCU 算子包

编译依赖 HCCL 源码仓。无卡编译服务器默认使用 `/home/liuyuanwen/hccl`：

```bash
docker exec -it cam_lyw_dev_91 bash
cd /home/liuyuanwen/sgl-kernel-npu
bash scripts/build_a5_ccu_urma_route_probe.sh
```

有卡服务器会自动优先使用 `/home/l00934901/hccl`，直接编译并安装到当前 CANN：

```bash
cd /home/l00934901/sgl-kernel-npu
bash scripts/build_a5_ccu_urma_route_probe.sh \
  --install \
  --install-path /usr/local/Ascend/cann-9.1.T560
```

如果 HCCL 或 CANN 位于其他目录，再分别设置 `HCCL_REPO` 或修改 `--install-path`。

脚本会输出本次生成的最新 `.run` 包。不要直接用通配符安装多个历史包；按修改时间只选最新包：

HCCL 8.5/9.1 这版构建入口要求 `--custom_ops_path` 位于 HCCL 仓内部，而且会解析软链接。
脚本会在 HCCL 仓内创建临时构建副本，构建结束自动删除；算子源码仍只维护在本仓。

```bash
LATEST_PACKAGE=$(
  HCCL_REPO=${HCCL_REPO:-/home/l00934901/hccl}
  find "${HCCL_REPO}/build_out" -maxdepth 1 -type f \
    -name '*ccu_urma_route_probe*.run' -printf '%T@ %p\n' |
  sort -nr | head -n 1 | cut -d' ' -f2-
)

test -n "${LATEST_PACKAGE}"
echo "installing ${LATEST_PACKAGE}"
CANN_ROOT=/usr/local/Ascend/cann-9.1.T560
env -u ASCEND_CUSTOM_OPP_PATH -u ASCEND_OPP_PATH \
  "${LATEST_PACKAGE}" --quiet --install "--install-path=${CANN_ROOT}"
```

安装后确认文件：

```bash
ls -l /usr/local/Ascend/cann-9.1.T560/opp/vendors/cust/include/a5_ccu_urma_route_probe.h
ls -l /usr/local/Ascend/cann-9.1.T560/opp/vendors/cust/lib64/liba5_ccu_urma_route_probe.so
```

脚本执行 `--install` 时会自动清除安装子进程中的 `ASCEND_CUSTOM_OPP_PATH` 和 `ASCEND_OPP_PATH`，并强制传入
CANN 根目录。因此，即使当前 shell 以前 source 过 DeepEP vendor 环境，也不会再安装到代码仓内部。
`--quiet` 用于更新已安装的同名探针包，避免安装器等待交互输入。

## 6. 两卡运行和候选路由枚举

测试程序是单进程、两线程、两设备程序，不使用 `torchrun`。先在物理卡 2 和 5 上运行 route 0：

```bash
cd /home/l00934901/sgl-kernel-npu

bash scripts/run_a5_ccu_urma_route_probe.sh \
  --devices 2,5 \
  --route-index 0 \
  --bytes 2097152 \
  --warmup 10 \
  --iters 100
```

`--bytes 2097152` 表示每个 rank 输入 2 MiB。每个 rank 的接收区为 4 MiB，因为探针当前是两 rank AllGather 语义。

首次调用会为选中的 channel 注册 CCU kernel。每轮调用中，本 rank 数据通过异步 D2D copy 写入自己的输出切片，
对端数据由 CCU kernel 直接写入对端用户输出 buffer。两端在 kernel 内交换远端输出地址和 token，并在写入前后
通过 channel notify 同步；host 侧不再调用 `HcommWriteOnThread`，也不再经过 CCL buffer、读回阶段或 host barrier。
输出的 `avg_us` 包含一次 CCU launch、一次本地 self copy 和一次 stream synchronize，用于路线正确性及初步对比，
仍不能直接代表最终多 route AllToAll 的性能。

当前无卡编译镜像的 CANN 9.1 使用公开的 object-style CCU API（`hcomm::CcuKernel`、`WriteNb`）。它与 HCCL
原生 CCU collective 中“注册 CCU kernel，再由 kernel 发起 write”的执行模型一致；不要把 host HCOMM primitive
提交到 `COMM_ENGINE_CCU` thread 上。

首次创建 channel 时会打印类似：

```text
[A5 CCU URMA][rank=0 peer=1] route=0 layer=0 link=0 protocol=4 hop=... src_phy=2 dst_phy=5 src_addr=type=...,raw=... dst_addr=type=...,raw=... SELECTED
[A5 CCU URMA][rank=0 peer=1] HcclChannelAcquire end: status=0 channel=...
[A5 CCU URMA][rank=0] HcclCcuKernelRegister end: status=0 kernel=...
[A5 CCU URMA][rank=0] HcclCcuKernelRegisterFinish end: status=0
[A5 CCU URMA][rank=0] HcclCcuKernelLaunch end: status=0
[rank=0] PASS engine=CCU protocol=UBC_CTP route=0 ...
```

输出中 `protocol=4` 即 `COMM_PROTOCOL_UBC_CTP`。`src_addr/dst_addr` 同时打印 `CommAddr.type` 和前16字节原始地址；
部分 CANN 9.1 内部 RankGraph 不把地址标成公开枚举 `COMM_ADDR_TYPE_EID`，但原始字节仍会保留，不能据此把
`not-an-eid` 误判成“没有使用 URMA”。可将原始字节、物理卡号以及 `urma_admin show` 联合对照。
`hop` 是 RankGraph 对该 link 发布的跳数信息。

如果输出存在 route 1、route 2 等候选，可逐条运行：

```bash
for route in 0 1 2 3; do
  echo "===== route ${route} ====="
  bash scripts/run_a5_ccu_urma_route_probe.sh \
    --devices 2,5 \
    --route-index "${route}" \
    --bytes 2097152 \
    --warmup 10 \
    --iters 100 || break
done
```

超出候选数量会明确报 `route index ... is out of range`。两端 rank 必须选择相同序号且形成相互匹配的 EID 对，否则 channel 建链会失败或通信超时。

## 7. 使用 msprof

运行脚本的 `--profile` 会把结果写入 `/home/l00934901/profiling`，并导出本次发现的全部 `PROF_*` 目录：

```bash
cd /home/l00934901/sgl-kernel-npu

bash scripts/run_a5_ccu_urma_route_probe.sh \
  --devices 2,5 \
  --route-index 0 \
  --bytes 2097152 \
  --warmup 10 \
  --iters 100 \
  --profile \
  --profile-root /home/l00934901/profiling
```

探针在一个进程的两个线程中使用两张 NPU；因此一个 `PROF_*` 目录也可能同时包含两张设备的数据。应检查导出的 HCCL、Runtime 和 task CSV 中的 device 字段，而不能用 `PROF_*` 目录数量判断采集了几张卡。

## 8. 判定标准和下一步

第一阶段通过需要同时满足：

1. 两个 rank 均打印 `PASS`，表明数据正确；
2. 日志明确显示 `engine=CCU`、`protocol=UBC_CTP`；
3. 选中的源/目的 EID 能在 `urma_admin show` 中找到；
4. 更换 `--route-index` 后日志中的 EID/hop 确实改变，而且通信仍正确；
5. 用 msprof 或平台链路计数器观察到所选 UDMA/IO Die 路径的流量变化。

其中 1～3 只能证明 CCU+URMA 通信可用；4～5 才能证明“选路”真正改变了数据路径。仅看到目的 EID 不足以证明经过了指定中转卡。

若 RankGraph 已暴露多条满足要求的链路，下一步把相同的 channel 选择逻辑接入 AllToAll，并按多条 route 并发切片。若只有一条链路，则先在 UVS/HIXL 拓扑配置中创建并发布新的 EID 对/路由，再继续算子实现。

## 9. 常见失败与阶段性结论

第一版直接注册测试程序的 `sendBuf/recvBuf`，并调用 `HcommWriteWithNotifyOnThread`。即使将
`HcclCommMemReg` 返回的 handle 填入下面字段、解决 `remote memory ... not found; memNum=0`：

```cpp
channelDesc.memHandles = &memHandle;
channelDesc.memHandleNum = 1;
```

实机仍在 `HcommWriteWithNotifyOnThread` 返回 status 5。第二版改用 `HcclGetHcclBuffer` 和
`HcclChannelGetHcclBuffer`，但在 CCU thread 上调用 host primitive `HcommWriteOnThread` 时发生 SIGSEGV。
日志已经证明 CCL buffer 地址相互匹配，所以根因不是 remote buffer 查找，而是执行模型错误。

当前版本删除 host primitive 和 CCL staging，改为 `HcclCcuKernelRegister` + `HcclCcuKernelLaunch`；用户
buffer 的 token 由 `hcomm::CcuRep::GetTokenInfo` 生成，真正的 remote write 在 CCU kernel 内执行。这正是本次
需要验证的修复点。

排障时按日志中的第一个失败点判断：

- `HcclChannelAcquire end: status=9`：所选 route 建链超时，优先核对两端 route/EID 是否互相匹配；
- `HcclCcuKernelRegister` 或 `RegisterFinish` 非 0：优先核对 CANN/HCCL 版本、CCU kernel API 和 notify 数量；
- `HcclCcuKernelLaunch` 非 0：检查两 rank 是否选择了对称 route，以及 task 参数和 user-buffer token；
- launch 返回 0 但 stream synchronize 超时：通常是 CCU kernel 内双方 notify 未匹配，先比较两端第一个错误和
  所选 channel；
- 两端 PASS 但更换 route 后链路计数器不变：只能证明功能正确，尚不能证明指定 route 改变了实际转发路径。

失败路径使用 `std::_Exit`，避免部分初始化的 HCCL channel 在 C++/ACL 清理阶段再次触发 SIGSEGV；因此应以
退出前打印的首个 HCCL/ACL 错误为根因。
