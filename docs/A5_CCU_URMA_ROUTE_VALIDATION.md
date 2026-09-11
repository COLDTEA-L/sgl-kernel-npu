# A5 CCU + URMA 显式路由验证

## 1. 目标和当前边界

本分支 `feature/a5-ccu-urma-multirelay-forwarding` 增加两 rank 的 CCU+URMA 路由验证能力。底层探针采用
AllGather 语义；同时提供 DeepEP Python 接口，便于直接用 `torch_npu.profiler` 采集 CCU 路径。它使用：

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
git fetch origin feature/a5-ccu-urma-multirelay-forwarding
git switch -c feature/a5-ccu-urma-multirelay-forwarding \
  --track origin/feature/a5-ccu-urma-multirelay-forwarding
```

本地已经有同名分支时：

```bash
cd /home/l00934901/sgl-kernel-npu
git switch feature/a5-ccu-urma-multirelay-forwarding
git pull --ff-only origin feature/a5-ccu-urma-multirelay-forwarding
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

编译和运行脚本会自动 source CANN 环境。当前测试默认选择物理卡 `2,3`，并通过
`ASCEND_RT_VISIBLE_DEVICES=2,3` 映射成逻辑 rank 0、1。

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

测试程序采用一张卡一个独立 rank 进程，不使用 `torchrun`。运行脚本会创建临时 root-info 文件，rank0 写入
`HcclGetRootInfo` 的结果，rank1 读取同一份 root info 后分别调用 `HcclCommInitRootInfo`。先在物理卡 2 和 3
上运行 route 0：

```bash
cd /home/l00934901/sgl-kernel-npu

bash scripts/run_a5_ccu_urma_route_probe.sh \
  --devices 2,3 \
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
    --devices 2,3 \
    --route-index "${route}" \
    --bytes 2097152 \
    --warmup 10 \
    --iters 100 || break
done
```

超出候选数量会明确报 `route index ... is out of range`。两端 rank 必须选择相同序号且形成相互匹配的 EID 对，否则 channel 建链会失败或通信超时。

对于同一 network layer 中的多条链路（A5 的 hop-2/2Die 场景通常如此），探针会先通过
`ENDPOINT_ATTR_DIE_ID` 查询选中 link 所属的 IO Die，再仿照 HCCL 原生 2Die AllToAll，将该 link 作为对应
Die kernel 的单独 channel 请求传给 `HcclChannelAcquire`。它不会再把两个 Die 的 link 合并为一次 acquire。
日志会显示 `die_id=0/1`；die 0 使用主 CCU thread，die 1 使用独立 slave stream/CCU thread，并通过
`HcommThreadNotifyRecordOnThread/HcommThreadNotifyWaitOnThread` 与主 thread 做前后同步。
资源申请顺序与 HCCL 原生 CCU 算子保持一致：先枚举 route 并得到 `die_id`，再申请所需的 main/slave
CCU thread，最后才调用 `HcclChannelAcquire`。因此 die 1 测试日志必须先出现
`acquire die1 slave CCU thread end: status=0`，之后才应出现 `acquiring route=... die_id=1`。

重点验证 route 1：

```bash
bash scripts/run_a5_ccu_urma_route_probe.sh \
  --devices 2,3 \
  --route-index 1 \
  --bytes 2097152 \
  --warmup 10 \
  --iters 100
```

如果 route 0 成功，而 route 1 在这种 per-die acquire 模式下仍然稳定返回 `status=9`，说明失败发生在
HCOMM/UVS 的 hop-2 channel 建链阶段，CCU kernel 和 `WriteNb` 尚未执行。

### 6.1 查询 IO Die 与 endpoint 的对应关系

探针会为每条候选 route 打印 `src_die`、`dst_die`、物理卡号以及原始 EID。先把 EID 与 UDMA 设备对应起来：

```bash
urma_admin show
```

A5 当前命名中通常可按设备名直接判断 IO Die：

```text
udmac0... -> IO Die 0
udmac1... -> IO Die 1
```

本机输出中 `003f:0200` 一组对应 `udmac0d1e2`，`007f:0200` 一组对应 `udmac1d1e2`。应以当前机器
`urma_admin show` 的实际 EID 为准，不要只依赖这一段前缀。

当前环境不能导出整机拓扑，`urma_admin show -a -w` 只能给出本机 endpoint 能力。重点记录：

```text
state / active_width / active_speed / active_mtu
各 priority 的 tp_type
max_write_size
```

当前 `udmac0d1e2` 和 `udmac1d1e2` 都是 `ACTIVE + LINK_X4 + SP_400G + MTU_4096`，并支持 CTP；
因此 route 1 建链失败不能简单归因于本地 die 1 端口关闭或不支持 CTP。没有整机拓扑时，使用不同卡对的
route 枚举结果和端口计数器做黑盒反推。

需要注意：RankGraph 公开接口只提供 endpoint 的 `DIE_ID`、`LOCATION` 和 `BW_COEFF`，没有
`relay_rank` 属性。因此上述信息可以确认使用哪个 IO Die/端口和相邻节点，但仅凭 probe 日志不一定能直接得出
“经过物理 NPU 几号卡”。最终应在 route 1/2 分别运行时对比相关 IO Die 端口计数器，并确认中间卡 HBM
读写量没有随 payload 增加。

### 6.2 仅验证 channel 建链

`--channel-only` 在 `HcclChannelAcquire` 和 `HcommChannelGetStatus` 成功后立即返回，不注册或启动 CCU kernel：

```bash
bash scripts/run_a5_ccu_urma_route_probe.sh \
  --devices 2,3 \
  --route-index 1 \
  --bytes 2097152 \
  --warmup 1 \
  --iters 3 \
  --channel-only
```

如果仍在 `HcclChannelAcquire status=9` 失败，即可确认与 CCU kernel、用户 buffer token 和 `WriteNb` 无关。

### 6.3 纯远端传输与数据量扫描

默认 `allgather` 模式每轮包含本 rank 的 D2D self copy。`--remote-only` 会在计时前一次性初始化本地输出切片，
正式迭代只测 CCU 对端 write、notify 和 stream 同步：

```bash
bash scripts/run_a5_ccu_urma_route_probe.sh \
  --devices 2,3 \
  --route-index 2 \
  --bytes 2097152 \
  --warmup 10 \
  --iters 100 \
  --remote-only
```

输出增加 `min_us/p50_us/p95_us/effective_gbps`。`effective_gbps` 按每个 rank 单方向发送的 payload 计算，
不把双向流量相加。

使用固定数据量集合扫描 route 0 或 route 2：

```bash
bash scripts/run_a5_ccu_urma_route_probe.sh \
  --devices 2,3 \
  --route-index 0 \
  --warmup 10 \
  --iters 100 \
  --remote-only \
  --sweep

bash scripts/run_a5_ccu_urma_route_probe.sh \
  --devices 2,3 \
  --route-index 2 \
  --warmup 10 \
  --iters 100 \
  --remote-only \
  --sweep
```

扫描大小依次为 64 KiB、256 KiB、1 MiB、2 MiB、8 MiB 和 32 MiB。对比时间曲线的截距与斜率：截距主要
反映固定 launch/notify 开销，斜率主要反映实际传输带宽。为减少频率和后台流量影响，建议按
`route0 -> route2 -> route0 -> route2` 的顺序重复至少两轮。

## 7. 使用 msprof

运行脚本的 `--profile` 会把结果写入 `/home/l00934901/profiling`，并导出本次发现的全部 `PROF_*` 目录：

```bash
cd /home/l00934901/sgl-kernel-npu

bash scripts/run_a5_ccu_urma_route_probe.sh \
  --devices 2,3 \
  --route-index 0 \
  --bytes 2097152 \
  --warmup 10 \
  --iters 100 \
  --profile \
  --profile-root /home/l00934901/profiling
```

脚本会分别启动 rank0、rank1 两个 worker 进程，并为每个进程独立启动一次 `msprof`：

```text
a5_ccu_urma_routes_*/
├── rank0/PROF_*/
└── rank1/PROF_*/
```

因此应当得到两个独立的 `PROF_*` 采集目录。rank0 和 rank1 的退出码会分别检查，任意一个 worker 失败都会使
脚本返回非 0。官方 HCCL 文档的一进程一卡样例同样通过广播/文件共享 `HcclRootInfo`，再由每个 rank 独立初始化
通信域；官方 msprof 文档也说明，多 Device 单采集进程只生成一个 `PROF_*`，多采集进程才生成多个目录。

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

## 10. 多 route CCU 并发

本分支支持把同一 IO Die 上的多条 HCOMM channel 一次交给 CCU kernel。kernel 会先对所有 channel 下发
非阻塞 `WriteNb`，然后统一等待完成，避免 `write/wait/write/wait` 造成路径串行。

首次在有卡服务器拉取本分支（本地尚不存在该分支）：

```bash
cd /home/l00934901/sgl-kernel-npu
git fetch origin
git switch -c feature/a5-ccu-urma-multirelay-forwarding \
  --track origin/feature/a5-ccu-urma-multirelay-forwarding
git rev-parse --short HEAD
```

如果本地已经存在该分支，则执行：

```bash
cd /home/l00934901/sgl-kernel-npu
git fetch origin
git switch feature/a5-ccu-urma-multirelay-forwarding
git pull --ff-only origin feature/a5-ccu-urma-multirelay-forwarding
git rev-parse --short HEAD
```

当前提交应至少包含 `1d00863 feat: add CCU URMA multiroute forwarding framework`。拉取完成后需要重新编译并安装最新算子包，再运行测试。

例如 6、7 卡当前的 route0 和 route2 都位于 die1，可直接验证双路径：

```bash
cd /home/l00934901/sgl-kernel-npu

bash scripts/run_a5_ccu_urma_route_probe.sh \
  --devices 6,7 \
  --route-indices 0,2 \
  --bytes 2097152 \
  --warmup 10 \
  --iters 100 \
  --remote-only
```

数据按权重切分：route0（direct）权重为 2，其余 route 权重为 1。选择 direct 加六条 relay 时总权重为 8，
direct 搬运总数据的 1/4，每条 relay 搬运 1/8。切片按 256 字节对齐，余数由最后一条路径处理。

当前一个 CCU kernel 只能包含同一 die 的 channel；跨 die 路径会明确返回 `HCCL_E_NOT_SUPPORT`，需要拆成
die0/die1 两个 kernel 后再并发 launch，不能把跨 die channel 强行交给一个 CCU thread。

## 11. RankGraph 未暴露路径时的 source-route provider

公开 UMDK 的 `uvs_route_t.hops` 当前注明只支持 direct route，URMA WRITE WR 也只有目标 tjetty/segment，
没有 relay rank 或 hop-list。因此本仓不能仅靠修改算子安装 IO Die forwarding 规则。

本分支新增 `a5_uvs_source_route_provider.h` ABI。平台 UVS/HIXL provider 必须完成两件事：

1. 根据 manifest 在 UVS/驱动/IO Die 中安装正反向 source-route/next-hop 规则；
2. 返回能够被 `HcclChannelAcquire` 使用的显式 `HcclChannelDesc`、die、relay 物理卡和权重。

provider 不能只替换目标 EID；没有实际安装 forwarding rule 的 descriptor 不满足接口约定。

运行形式：

```bash
bash scripts/run_a5_ccu_urma_route_probe.sh \
  --devices 6,7 \
  --source-route-manifest /home/l00934901/routes/6_to_7.yaml \
  --source-route-provider /usr/local/lib64/liba5_uvs_source_route_provider.so \
  --bytes 2097152 \
  --warmup 10 \
  --iters 100 \
  --remote-only
```

若 provider 未安装或不支持底层 source-route，程序会明确失败，不会静默退化成 route0。这样可以防止把普通
直连误判为“0～5 卡均参与了 relay”。

编译并安装最新包：

```bash
cd /home/l00934901/sgl-kernel-npu
bash scripts/build_a5_ccu_urma_route_probe.sh

latest_pkg=$(find /home/l00934901/hccl/build_out -maxdepth 1 -type f \
  -name 'cann-hccl_custom_ccu_urma_route_probe_linux-aarch64.run' \
  -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
test -n "${latest_pkg}"
bash "${latest_pkg}" --install
```

验收“指定 0～5 转发”必须同时看到 provider 返回六个不同 `relayPhyId`，并通过 IO Die 路径计数器确认对应
字节增长。只有 CCU kernel PASS 不能证明具体中转卡。

## 12. 使用 hccn_tool 对比 route0 和 route2 的报文增量

华为文档给出的网口统计命令是：

```bash
/usr/local/Ascend/driver/tools/hccn_tool -i DEVICE_ID -stat -g
```

其中 `-i` 后面是宿主机的**逻辑 Device ID**（以宿主机 `npu-smi info` 的 NPU ID 为准）。脚本调用
`hccn_tool` 时会临时取消 `ASCEND_RT_VISIBLE_DEVICES/ASCEND_VISIBLE_DEVICES`，避免 workload 的可见卡映射
把 `4,5` 重映射成进程内的 `0,1`。`nic_tx_all_pkg_num`、`nic_tx_all_oct_num`、
`nic_rx_all_pkg_num`、`nic_rx_all_oct_num` 分别表示 NIC 累计发送/接收报文数和字节数；
`roce_new_pkt_rty_num` 可用于观察重传。它们是累计值，因此不能只查看一次绝对值。

运行脚本增加了以下参数：

- `--hccn-stat`：在 workload 前后采集 `hccn_tool -stat -g`，并计算增量；
- `--hccn-devices 0,1,...`：需要观察的物理卡，默认 `0,1,2,3,4,5,6,7`；
- `--hccn-tool PATH`：工具不在 `PATH` 或默认驱动目录时显式指定；
- `--hccn-stat-root PATH`：原始快照和增量表的根目录，默认沿用 `--profile-root`，即
  `/home/l00934901/profiling`。

先测试 direct route0：

```bash
cd /home/l00934901/sgl-kernel-npu

bash scripts/run_a5_ccu_urma_route_probe.sh \
  --devices 6,7 \
  --route-index 0 \
  --bytes 2097152 \
  --warmup 10 \
  --iters 100 \
  --remote-only \
  --hccn-stat \
  --hccn-devices 0,1,2,3,4,5,6,7 \
  --hccn-stat-root /home/l00934901/profiling
```

确认上一次进程完全退出后，再用完全相同的数据量和迭代次数测试 hop-2 route2：

```bash
bash scripts/run_a5_ccu_urma_route_probe.sh \
  --devices 6,7 \
  --route-index 2 \
  --bytes 2097152 \
  --warmup 10 \
  --iters 100 \
  --remote-only \
  --hccn-stat \
  --hccn-devices 0,1,2,3,4,5,6,7 \
  --hccn-stat-root /home/l00934901/profiling
```

脚本不会清零全局计数器，而是保存 `before_deviceN.txt`、`after_deviceN.txt`，并生成：

```text
/home/l00934901/profiling/hccn_routes_ROUTE_TIMESTAMP/hccn_counter_deltas.tsv
```

终端也会打印每张物理卡的 TX/RX 报文和字节增量。判读时注意：

1. 在没有其他通信任务的独占窗口执行；否则这些设备级累计计数会混入其他业务流量。
2. 6、7 卡是通信端点，route0 和 route2 下出现收发增量都是正常的。
3. 如果 0～5 中某张卡只在 route2 测试中出现与 payload/迭代次数成比例的明显增量，而 route0
   没有，可作为该卡参与转发的强证据。
4. `hccn_tool -i DEVICE_ID -stat -g` 是设备/NIC 级统计，公开命令没有 RankGraph `route-index` 或内部
   IO Die `link` 参数。透明 IO Die forwarding 未必记入中间 NPU 的 host 可见 NIC 计数。因此“中间卡有增量”
   能增强 relay 判断，但“中间卡无增量”不能单独否定 hop-2 转发；最终仍需结合 RankGraph 的 `hop/src_addr/dst_addr`
   和平台侧 UDMA/IO Die per-port 计数器。
5. 单独选择 `--route-index 2` 表示全部 2 MiB 走 route2 这一条 RankGraph route，并不是脚本同时运行
   route0+route2。要测试 direct 与 relay 并发，使用 `--route-indices 0,2`，此时才按权重切片。

部分 A5/950 端口不向 `hccn_tool -stat` 暴露传统 NIC 统计。新版脚本遇到单卡查询失败时会把该命令的真实
回显直接打印到终端、记录原始文件，并继续查询其余卡；不会再因为 device 0 不支持而终止 workload。增量表
只包含前后两次都查询成功的设备。如果所有设备均失败，算子仍会继续运行并明确打印
`HCCN statistics unavailable`。这表示当前产品/驱动没有通过该接口提供所需计数，不能把“查询失败”解释成
“该卡没有流量”。

## 13. Python 直接调用 CCU 多路径 write 并采集 profiling

前面的 C++ 探针仍适合枚举 route、做 `channel-only` 和采集 `hccn_tool`。为了让 CCU 通信进入 Python
测试进程，本分支另外导出 `HcclCcuUrmaMultiRouteWrite`，并依次接入：

```text
examples/a5_ccu_urma_route_probe/op_host/route_probe.cc
    CCU 多 route 切片、HcclCcuKernelLaunch
csrc/deepep/deep_ep.cpp
    解析 PyTorch HCCL group、加载探针动态库、取得当前 NPU stream
csrc/deepep/pybind_extension.cpp
python/deep_ep/deep_ep/buffer.py
    Python 接口 Buffer.ccu_urma_multiroute_write
tests/python/deepep/test_a5_ccu_urma_multiroute_write.py
    两卡正确性、warmup/计时和 torch_npu.profiler
```

该接口当前是两 rank、FP32 的 AllGather-like 验证原语：每个 rank 输入 `--bytes` 字节，CCU 把这一整段数据
按选中的 route 权重切片并写入对端输出；返回张量形状为 `[2, elements]`。它不是 AIV 算子，也不启动两个
外部 C++ worker。`torchrun` 的两个 Python rank 各自直接调用 CCU library。

### 13.1 拉取并重新编译两个组件

先拉取分支：

```bash
cd /home/l00934901/sgl-kernel-npu
git fetch origin
git switch feature/a5-ccu-urma-multirelay-forwarding
git pull --ff-only origin feature/a5-ccu-urma-multirelay-forwarding
git rev-parse --short HEAD
```

本次同时修改了 CCU route 动态库和 `deep_ep_cpp`，因此二者都必须重编译。先编译并安装 CCU 包：

```bash
source /usr/local/Ascend/cann-9.1.T560/set_env.sh
cd /home/l00934901/sgl-kernel-npu
bash scripts/build_a5_ccu_urma_route_probe.sh \
  --install \
  --install-path /usr/local/Ascend/cann-9.1.T560
```

再编译并安装 DeepEP wheel。只安装最新生成的一个 wheel，避免通配符同时命中历史包：

```bash
cd /home/l00934901/sgl-kernel-npu
bash build.sh -a deepep Ascend950

latest_wheel=$(find output -maxdepth 1 -type f -name 'deep_ep*.whl' \
  -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
test -n "${latest_wheel}"
python3 -m pip install --force-reinstall --no-deps "${latest_wheel}"
```

安装后确认两个新符号均存在：

```bash
nm -D /usr/local/Ascend/cann-9.1.T560/opp/vendors/cust/lib64/liba5_ccu_urma_route_probe.so | \
  grep HcclCcuUrmaMultiRouteWrite

python3 - <<'PY'
import deep_ep_cpp
assert hasattr(deep_ep_cpp.Buffer, "ccu_urma_multiroute_write")
print("deep_ep_cpp CCU Python binding: OK")
PY
```

### 13.2 功能与性能测试

先 source CANN 和 DeepEP vendor 环境，并用物理卡 4、5 映射为进程内 device 0、1：

```bash
cd /home/l00934901/sgl-kernel-npu
source /usr/local/Ascend/cann-9.1.T560/set_env.sh
source python/deep_ep/deep_ep/vendors/hwcomputing/bin/set_env.bash
export ASCEND_RT_VISIBLE_DEVICES=4,5
export HCCL_OP_EXPANSION_MODE=CCU_SCHED
export HCCL_BUFFSIZE=2300
```

单 route0：

```bash
python3 -m torch.distributed.run \
  --standalone --nproc-per-node=2 \
  tests/python/deepep/test_a5_ccu_urma_multiroute_write.py \
  --route-index 0 \
  --bytes 2097152 \
  --warmup 10 \
  --iters 100 \
  --remote-only
```

route0 与 route2 并发：

```bash
python3 -m torch.distributed.run \
  --standalone --nproc-per-node=2 \
  tests/python/deepep/test_a5_ccu_urma_multiroute_write.py \
  --route-indices 0,2 \
  --bytes 2097152 \
  --warmup 10 \
  --iters 100 \
  --remote-only
```

`--remote-only` 只跳过本 rank 到本 rank 的 D2D self copy；它不会减少 CCU 发往对端的 2 MiB，也不会把
多 route 测试退化为单 route。

### 13.3 在 Python 进程内采集 CCU profiling

```bash
python3 -m torch.distributed.run \
  --standalone --nproc-per-node=2 \
  tests/python/deepep/test_a5_ccu_urma_multiroute_write.py \
  --route-indices 0,2 \
  --bytes 2097152 \
  --warmup 10 \
  --iters 100 \
  --remote-only \
  --profile \
  --profile-root /home/l00934901/profiling
```

结果按 rank 分开保存：

```text
/home/l00934901/profiling/a5_ccu_urma_write_RUN_ID/
├── rank0/
└── rank1/
```

采集配置使用 `ProfilerLevel.Level1`，活动包含 CPU 和 NPU，并开启 Text（环境支持时同时开启 Db）导出。
时间线应能看到 `deep_ep::ccu_urma_multiroute_write` 以及其下发的 CCU task。这个接口绕过标准
`HcclAlltoAll` 算法入口，因此仅设置 profiler 详细度不会自动生成标准 collective 的带宽、通信矩阵或
`communication.json`；若后续需要这些字段，必须给自定义 CCU kernel 补 HCCL DFX/通信 profiling 上报，不能
把 AIV 算子的 profiling 当作 CCU 结果。
