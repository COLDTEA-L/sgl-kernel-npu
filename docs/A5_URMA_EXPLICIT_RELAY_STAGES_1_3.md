# A5 URMA 显式中转：阶段 1～3

## 1. 目标与强制边界

本分支 `feature/a5-urma-explicit-relay-provider` 验证端点卡能否显式指定一张 NPU 作为 URMA 中转卡，同时满足：

- 中转卡上不创建 rank、stream、AIV/CCU kernel 或 HBM staging buffer；
- 报文只在中转卡 IO Die/UDMA forwarding pipeline 中转发；
- 路由不存在时直接失败，不把普通目的 EID 或已有 RankGraph route 冒充显式中转路径。

源卡和最终目的卡的输入、输出位于各自 HBM，这是端点存储，不属于“中转落 HBM”。IO Die forwarding
不会抢占中转卡计算核，但仍会占用端口和 IO Die 带宽；如要求中转卡业务性能也不受影响，平台控制面还必须提供
QoS、限速或准入控制。

## 2. 代码结论与三阶段实现

对 `/home/liuyuanwen/umdk`、`hccl`、`hcomm` 和 `hixl` 的检查结论如下：

- UMDK 公开 `uvs_get_route_list()`，但它是查询接口；`uvs_route_t.hops` 的源码注释明确为
  `Only supports direct routes, currently 0`。
- 当前公开头文件没有 `uvs_add/install/remove_source_route` 或 next-hop API。
- URMA `hop_limit` 只是跳数上限，不携带中转卡或下一跳列表。
- HCCL/HCOMM 可以消费平台已经建立好的 `HcclChannelDesc`，CCU kernel 可以在该 channel 上执行
  `WriteNb`，但它们不会自行安装指定中转卡的 UVS forwarding rule。

因此前三阶段采用 fail-closed 的插件结构：

1. `probe_a5_urma_source_route_capability.py` 检查 UMDK 源码、运行库和平台 backend 能力。
2. `liba5_uvs_manifest_route_provider.so` 解析显式路由 manifest，调用平台 UVS/HIXL backend 安装、查询和删除
   forwarding lease，并向 HCOMM 返回可获取的 `HcclChannelDesc`。
3. `run_a5_explicit_relay_stage3.sh` 先做能力闸门，再执行 channel-only 或单路 4 MiB Write，并可同时采集端点及
   指定中转卡 HCCN 计数。

平台 backend 必须实现：

```text
A5UvsBackendGetCapabilities
A5UvsBackendInstallRoute
A5UvsBackendQueryRoute
A5UvsBackendRemoveRoute
```

ABI 位于：

```text
examples/a5_ccu_urma_route_probe/inc/a5_uvs_explicit_route_backend.h
```

它必须声明并兑现五项能力：安装、查询、删除 source route、IO Die forwarding、禁止 relay HBM。缺少任意一项，
provider 会拒绝运行。当前开源 UMDK 本身没有该 backend；需要平台 UVS/HIXL 服务实现该 ABI，或先扩展其
控制面接口。仅有 `liburma`/`libuvs` 和 `uvs_get_route_list` 时，阶段 1 的预期结果是
`BLOCKED_NO_ROUTE_PROGRAMMING_BACKEND`。

## 3. 拉取分支

首次在有卡服务器拉取：

```bash
cd /home/l00934901/sgl-kernel-npu
git status --short
git fetch origin feature/a5-urma-explicit-relay-provider
git switch -c feature/a5-urma-explicit-relay-provider \
  --track origin/feature/a5-urma-explicit-relay-provider
git rev-parse --short HEAD
```

已经有本地分支时：

```bash
cd /home/l00934901/sgl-kernel-npu
git switch feature/a5-urma-explicit-relay-provider
git pull --ff-only origin feature/a5-urma-explicit-relay-provider
```

## 4. 编译安装

```bash
cd /home/l00934901/sgl-kernel-npu
source /usr/local/Ascend/cann-9.1.T560/set_env.sh

bash scripts/build_a5_ccu_urma_route_probe.sh \
  --install \
  --install-path /usr/local/Ascend/cann-9.1.T560
```

确认两个库和 ABI 头文件均已安装：

```bash
ls -l /usr/local/Ascend/cann-9.1.T560/opp/vendors/cust/lib64/\
liba5_ccu_urma_route_probe.so
ls -l /usr/local/Ascend/cann-9.1.T560/opp/vendors/cust/lib64/\
liba5_uvs_manifest_route_provider.so
ls -l /usr/local/Ascend/cann-9.1.T560/opp/vendors/cust/include/\
a5_uvs_explicit_route_backend.h
```

在无卡 CANN 容器中可用 mock backend 验证 manifest 解析、能力协商、安装、查询、回滚/释放调用链。mock 只用于
host 逻辑测试，不能用于有卡转发实验：

```bash
cd /home/liuyuanwen/sgl-kernel-npu
bash scripts/test_a5_explicit_route_provider.sh
```

## 5. 阶段 1：能力闸门

没有平台 backend 时，可以在无卡服务器生成报告但允许返回缺失：

```bash
cd /home/liuyuanwen/sgl-kernel-npu
python3 scripts/probe_a5_urma_source_route_capability.py \
  --umdk-root /home/liuyuanwen/umdk \
  --allow-missing-backend \
  --output /tmp/a5_explicit_relay_capability.json
```

有卡环境获得平台 backend 后必须严格检查，不能加 `--allow-missing-backend`：

```bash
export A5_UVS_EXPLICIT_ROUTE_BACKEND=/opt/platform/lib/liba5_uvs_route_backend.so

python3 scripts/probe_a5_urma_source_route_capability.py \
  --umdk-root /home/l00934901/umdk \
  --backend "${A5_UVS_EXPLICIT_ROUTE_BACKEND}" \
  --output /home/l00934901/profiling/a5_explicit_relay_capability.json
```

只有输出 `PASS_EXPLICIT_RELAY_BACKEND` 才能进入阶段 2/3。

## 6. 阶段 2：配置显式路由

示例 manifest：

```text
examples/a5_ccu_urma_route_probe/config/explicit_relay_example.csv
```

每个方向必须单独配置。例如端点卡 6、7，指定卡 2 中转：

```csv
route_id,local_rank,peer_rank,src_phy,dst_phy,relay_phy,die_id,weight
1002,0,1,6,7,2,0,1
1002,1,0,7,6,2,0,1
```

`die_id` 是源端用于该路径的 IO Die，不是中转卡 rank。backend 必须原子安装 forwarding rule，查询确认
`installedRelayPhyId` 与 manifest 一致，再返回最终目的端的 HCOMM channel 描述。安装部分成功、后续失败时
provider 会逆序回滚 lease；进程退出时会调用删除接口清理剩余 lease。

## 7. 阶段 3：先建链，再传输

先只验证指定 relay 的 channel 能否建立：

```bash
cd /home/l00934901/sgl-kernel-npu
source /usr/local/Ascend/cann-9.1.T560/set_env.sh
export A5_UVS_EXPLICIT_ROUTE_BACKEND=/opt/platform/lib/liba5_uvs_route_backend.so

bash scripts/run_a5_explicit_relay_stage3.sh \
  --devices 6,7 \
  --relay-device 2 \
  --manifest examples/a5_ccu_urma_route_probe/config/explicit_relay_example.csv \
  --backend "${A5_UVS_EXPLICIT_ROUTE_BACKEND}" \
  --channel-only
```

建链通过后执行单路 4 MiB Write，并采集端点 6、7 和中转卡 2 的端口计数：

```bash
bash scripts/run_a5_explicit_relay_stage3.sh \
  --devices 6,7 \
  --relay-device 2 \
  --manifest examples/a5_ccu_urma_route_probe/config/explicit_relay_example.csv \
  --backend "${A5_UVS_EXPLICIT_ROUTE_BACKEND}" \
  --bytes 4194304 \
  --warmup 10 \
  --iters 100 \
  --hccn-stat \
  --hccn-stat-root /home/l00934901/profiling
```

成功必须同时满足：

1. 两端打印的 route ID、`relay_phy`、die 和 lease 与 manifest 一致；
2. `HcclChannelAcquire` 成功，4 MiB 数据校验通过；
3. 指定中转卡对应 IO Die 端口计数随 payload/iters 成比例增长；
4. 更换 `relay_phy` 后流量移动到新中转卡，旧中转卡增量消失；
5. 中转卡没有新增 rank、stream、AIV/CCU kernel 或 HBM allocation。

如果阶段 1 失败，不应继续阶段 3。此时缺失的是 UVS/HIXL 平台路由控制能力，不是 CCU `WriteNb` 算法问题。
