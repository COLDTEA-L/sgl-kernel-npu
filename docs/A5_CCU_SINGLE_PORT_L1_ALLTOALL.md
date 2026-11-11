# 2/4 卡与 EP8：单口 L1 路径接入和现场测试

首次部署或交接给现场执行者，请先阅读 [完整使用、单算子性能与模型接入教程](A5_CCU_ALLTOALLV_USER_GUIDE.md)。本文保留单口 L1 的详细设计、策略格式和资源预算说明。

分支：`feature/a5-ccu-single-port-l1-alltoall`。基于 EP8 提交优化 `9a378d4` 和两卡 L1 探针 `2dd863a`；不修改旧 9.1 分支。后端仍为 CCU + URMA，L1 网络路径不经过中间 NPU 的 HBM。

## 1. 本版行为与边界

- 2/4 卡：每个 peer pair 可显式配置一条 L1 单口路径，用 `l0_weight:l1_weight` 将该 pair 的 payload 切成互不重叠的区间。默认生成 1:1，不是双份发送。原有外部卡 relay 可以保留，其权重也参与分流归一化，且必须满足现有通道/参数预算。
- EP8：保留 L0 direct、组内 NPU relay，新增 L1 网络备用路径。每条流至多选择一条备用路径；selector `-1` 表示直连，`0..7` 表示组内 relay rank，`8` 表示 L1，**8 不是第九张卡**。
- L1 通过 `source → L1 network → destination`，无需中间 NPU 执行转发或等待 READY。控制消息仍走每 peer 的 L0 direct 通道，数据完成后沿原有完成通知返回。
- 只允许与该 rank L0 通道**同 IO die**的单口 L1 EID。异 die 明确拒绝；本版未实现跨 die 多 kernel 调度。不能把本版说成已经同时使用一张卡的全部八个 L1 口。
- 每 pair 最多一个 L1 endpoint pair。EP8 最多 `7 × (1 direct + 6 participant relay + 1 L1) = 56` 个缓存通道，task args 仍为 46。初测建议 `A5_EID_MAX_RELAYS_PER_PEER=0`，只建 14 条通道/rank，隔离 L1 效果。
- 旧配置默认不启用 L1；不增加逐调用建链、注册、EID 查询、通信组创建或 HBM workspace。

单口分类用 Ascend950 已验证的 EID 编码：第 5 个零基字节的高两位是 die、低六位是 port；`port=0x3f` 是聚合组。本版只接受 die 0/1、port 0..8。**只分类枚举得到的完整 EID，不通过改前缀构造地址**。分类之后还要把端口与当前 rootinfo 的 L1 CLOS plane/instance 相交，并保持 rootinfo 声明的 FE 命名空间；同一物理端口的其他 FE 地址不会自动入选。`043f/047f` 不是固定的 2 口/6 口带宽标签。

## 2. 显式配置与控制器接口

`scripts/discover_a5_single_port_l1.py` 生成以下文件：

- `inventory.json`：按实际物理 device ID 查询的完整 EID 清单，绑定 rootinfo SHA256。
- `candidates.json`：与 L1 rootinfo 端口集合相交后的单口候选。
- `l1_policy.json`：可供人或控制器编辑的冻结策略，schema 为 `a5.single-port-l1.v1`。

策略格式（字段示意；EID 必须取自 inventory，禁止照示例造地址）：

```json
{
  "schema": "a5.single-port-l1.v1",
  "generation": "controller-generation-001",
  "devices": [2, 3],
  "rootinfo": "/etc/hccl_rootinfo.json",
  "rootinfo_sha256": "实际摘要",
  "inventory": "inventory.json",
  "inventory_sha256": "实际摘要",
  "paths": [{
    "id": "pair-2-3-l1",
    "src_phy": 2, "dst_phy": 3,
    "src_eid": "枚举出的完整地址", "dst_eid": "枚举出的完整地址",
    "src_die": 0, "dst_die": 0, "src_port": 1, "dst_port": 1,
    "plane_id": "实际plane", "net_instance_id": "实际instance",
    "l0_weight": 1, "l1_weight": 1
  }]
}
```

`devices` 是通信组的物理卡顺序，不是假设 rank 等于物理卡号。pair 顺序按该列表中的 rank 顺序规范化。同一对卡只能有一个 L1 配置；源/目的必须属于同一 L1 plane/instance。不同 FE 的 EID 即使地址不同，指向同一个 `(rank, die, port)` 时也共用规划预算。

控制器后续可直接生成这个 JSON，复用校验和 manifest 编译入口。新增路径必须来自新的实际 inventory；拓扑/rootinfo 改变需重新发现。不要在线改已绑定 handle 的策略：先等在途调用完成、按现有 close 契约释放旧实例，再为新 generation 建立 handle，所有 rank 使用同一份配置。`topology.json` 保存 generation、config/inventory/rootinfo 摘要和实际 path IDs。

库存枚举及端口分类只是候选验证。输出中的 `physical_single_port_verified=false` 是刻意保留的；直到完成端口计数差分，才能宣布实际使用了指定单口。

## 3. EP8 的选路算法

启用 `--planner l1_bottleneck`，从同一份生成的 topology 读取 `allowed_relays` 与 `l1_resources`。后者是 64 项，每项为空或 `[source_port_resource, destination_port_resource]`；资源索引为 `rank*18 + die*9 + port`。

算法在 C++ 中运行；各有向割的 L1 可用端口容量在 planner 构造期缓存，不逐调用枚举 EID 或重复扫描拓扑：

1. 从 C 去掉 self 得到初始 L0 边负载。L1 TX/RX 默认零背景流量，也可读取控制器冻结的端口背景字节数和可用带宽预算（§11）。不自动推测外部业务占用；流量 C 仍为 8×8。
2. 计算目标下界：原有有向割加上去重后 L1 源/目的物理口的可用带宽容量；并保留单流最多两路的 `ceil(peak/2)` 下界。默认 L0/L1 单口等带宽，允许 L1 按千分比降额；背景流量形成不可削减的负载下界。尚未建模交换网络内部争用、端点聚合带宽和 HBM 带宽，不能将削峰率当时延收益。
3. 最多尝试原来的四个目标 `LB`、`LB+ceil(gap/7)`、`/3`、`/2`。只处理超过目标的边，每流移动最少的超出行数，同时遵守最小 chunk、最小收益及显式 `max_actions`（0..56）的约束。仅 L1 策略解除旧的内部 8 动作上限；原 L0 bottleneck 保持最多 8 个动作。
4. L0 relay 候选检查两条借道边；L1 候选检查源 TX 物理口和目的 RX 物理口的**本次累计字节 + 背景字节 / 可用带宽**。多个逻辑通道共享同一端口时，共用该预算；反向 RX/TX 分开，按全双工建模。
5. 选加入这部分流量后候选资源峰值更低的路径，同值优先 L1。每条流只选一个备用路径。若不能让所有相关资源都压到目标下，丢弃这一整次尝试；全失败保持 direct。
6. C++ host 按本次 counts 和 row_bytes 更新参数并执行。dispatch FP8 与 combine BF16 都支持；combine 用 C 的转置，**重新按转置方向和当前行宽规划**，不能直接交换旧 relay action。

可观测字段：`route_selector_8`、`l1_rows`、`participant_relay_rows`、`l1_tx_port_rows`、`l1_rx_port_rows`、`resource_lower_bound_rows`、`resource_peak_before_equivalent_rows`、`resource_peak_after_equivalent_rows`、`l1_budget_generation`、`planned_link_peak_bytes`、`candidate_evaluations`。带 `equivalent_rows` 的字段是按一个 L0 口归一化的资源负载，包含背景字节和带宽降额；`planned_link_peak_bytes` 仍是本算子原始字节负载，二者不能混用。L1 的源出口和目的入口不是两份有效载荷，`transmitted_network_bytes` 对 L1 部分只计一次；它也不等于交换机内部 hop-byte 计数。

## 4. 构建与 CPU 回归

目标容器中先拉取新分支。不要安装旧 wheel：本版有新 planner binding。

```bash
cd /home/l00934901/sgl-kernel-npu
git fetch origin feature/a5-ccu-single-port-l1-alltoall
git switch --track origin/feature/a5-ccu-single-port-l1-alltoall
export A5_CANN_PATH=/usr/local/Ascend/cann-9.3.0
export A5_OUTPUT_ROOT=/home/l00934901/profiling
export HCCL_BUFFSIZE=64
mkdir -p "$A5_OUTPUT_ROOT/l1-alltoall"
bash scripts/build_a5_ccu_urma_route_probe.sh \
  --cann-path "$A5_CANN_PATH" \
  --build-dir "$A5_OUTPUT_ROOT/l1-alltoall/route-build" \
  --install --install-path "$A5_CANN_PATH"
export A5_CCU_ROUTE_PROBE_LIB="$A5_CANN_PATH/opp/vendors/cust/lib64/liba5_ccu_urma_route_probe.so"
```

预期 `PUBLIC_ROUTE_ABI A5CcuSinglePortL1AbiVersion=1`、`PUBLIC_ROUTE_LOAD_PASS`、`PUBLIC_ROUTE_BUILD_PASS`。route 构建同时生成可选的 `a5_ccu_l1_inventory` SDK 库存查询工具。

按 [9.2/9.3 构建指南 §4](A5_CCU_92_93_ALLTOALL.md#4-重建目标-deepep-wheel) 重建并安装当前分支 wheel，确认 `deep_ep_cpp.ep8_single_port_l1_abi == 1` 和 `deep_ep_cpp.ep8_l1_resource_planner_abi == 1`。该指南的旧分支名只用于历史说明，本次不要切回旧分支。

CPU 回归分别运行，避免测试模块重复注册同名 pybind 类：

```bash
export PYTHONPATH=tests/python/deepep:scripts
python3 -m unittest test_a5_ccu_l1_cpu -v
python3 -m unittest test_a5_ccu_eid_cpu -v
python3 -m unittest test_a5_ccu_bottleneck_cpu -v
python3 -m unittest test_a5_ccu_peer_plan_cpu -v
```

## 5. 当前机器发现，先测 2/4 卡

默认通过 DCMI 的物理卡映射和 URMA EID 枚举获取库存，不从 `udmac...` 名称猜物理卡，不要求 per-port EID 出现在 rank graph。使用的驱动接口和结构体与 HCCL `topo_addr_info/hal.c`、`hal.h` 一致；不写驱动配置。如驱动不导出该接口，明确报错，不回退到猜 EID。

```bash
source "$A5_CANN_PATH/set_env.sh"
# 确保驱动的 libdcmi.so 可被动态加载；使用当前机器的实际驱动库目录。
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64:${LD_LIBRARY_PATH:-}
python3 scripts/discover_a5_single_port_l1.py \
  --devices 2,3 --root-info /etc/hccl_rootinfo.json \
  --output-dir "$A5_OUTPUT_ROOT/l1-alltoall/policy-2"
export A5_L1_CONFIG="$A5_OUTPUT_ROOT/l1-alltoall/policy-2/l1_policy.json"
```

输出目录必须是新的。若希望对照 SDK 枚举，可另用 `--inventory-source sdk --inventory-binary "$A5_OUTPUT_ROOT/l1-alltoall/route-build/a5_ccu_l1_inventory"`。SDK 清单可能没有 per-port EID；没有候选就停止，不修改 rootinfo、不造地址。

检查 `candidates.json` 和 `l1_policy.json` 的 device/EID/die/port/plane。发现脚本按同 die、同 plane 配对，尽量分散端口；这不是性能调优结果。可在现有 inventory 内改选 endpoint pair，或改静态权重，然后更新 generation。

```bash
bash scripts/run_a5_ccu_peer_plan_alltoall.sh \
  --devices 2,3 --available-phys 0,1,2,3,4,5,6,7 \
  --cann-root "$A5_CANN_PATH" --self-copy-mode ccu \
  --bytes 65536 --warmup 3 --iters 10 --repeats 1
# 正确性通过后，再测 32 MiB/peer；包含 self 的 tensor 大小另按 rank 数计算。
bash scripts/run_a5_ccu_peer_plan_alltoall.sh \
  --devices 2,3 --available-phys 0,1,2,3,4,5,6,7 \
  --cann-root "$A5_CANN_PATH" --self-copy-mode ccu --sync-mode batch \
  --bytes 33554432 --warmup 20 --iters 50 --repeats 3
```

2 卡先验证已证明可用的一对 L0+单口 L1 EID。4 卡重新发现，不能复用 2 卡 devices 配置：

```bash
python3 scripts/discover_a5_single_port_l1.py --devices 0,1,2,3 \
  --root-info /etc/hccl_rootinfo.json --output-dir "$A5_OUTPUT_ROOT/l1-alltoall/policy-4"
export A5_L1_CONFIG="$A5_OUTPUT_ROOT/l1-alltoall/policy-4/l1_policy.json"
bash scripts/run_a5_ccu_peer_plan_alltoall.sh --devices 0,1,2,3 \
  --available-phys 0,1,2,3,4,5,6,7 --cann-root "$A5_CANN_PATH" \
  --self-copy-mode ccu --bytes 65536 --warmup 3 --iters 10 --repeats 1
```

对照 L0-only 时用 `A5_ENABLE_L1=0 bash ...`，保持 bytes、dtype、时序口径不变。不要将不同通道目录之间的差全部归于路径带宽。4 卡共享 L1 端口的静态分流未做动态流量优化，不承诺线性叠加带宽。

## 6. EP8：先 L1-only 备用目录，再加入组内 relay

```bash
python3 scripts/discover_a5_single_port_l1.py --devices 0,1,2,3,4,5,6,7 \
  --root-info /etc/hccl_rootinfo.json --output-dir "$A5_OUTPUT_ROOT/l1-alltoall/policy-8"
export A5_L1_CONFIG="$A5_OUTPUT_ROOT/l1-alltoall/policy-8/l1_policy.json"
export A5_DEVICES=0,1,2,3,4,5,6,7
export A5_EID_MAX_RELAYS_PER_PEER=0
bash scripts/run_a5_ccu_eid_alltoallv.sh \
  --suite create --register-stage full --planner l1_bottleneck
```

create PASS 仅证明建链/注册；然后运行小数据量 correctness：

```bash
bash scripts/run_a5_ccu_eid_alltoallv.sh \
  --suite correctness --mode all --planner l1_bottleneck \
  --dtype float8_e4m3fn --width 5120 --row-scale 1 \
  --warmup 3 --iters 10 --rounds 1 --dynamic-iters 100 --bidirectional-iters 100
```

必须看到 `EP8_EID_ALL_RANKS_AND_EXIT_PASS`、8 rank 干净退出；检查 plan 的 `l1_rows>0`，并确认至少一个启用 L1 的 case 逐元素正确。没有走 L1 不能当作 L1 验证通过。

再用 6 个组内 relay 候选建混合目录，重复 create 和 correctness：

```bash
export A5_EID_MAX_RELAYS_PER_PEER=6
# 重复上面的 create + correctness；每 rank 最多 56 个通道。
```

若 SDK 资源不能承载混合目录，保留错误阶段和资源记录；不要重试同一失败通信域。可以先维持 0 个组内 relay 的 L1-only 目录，不据此宣称混合目录已验证。

## 7. 性能判据与物理路径验证

每次都在独占、稳定的卡上测；L1 网络可能与其他机器共享，记录网络背景流量。先比较同一目录下 `native/direct/multipath`：direct 与 multipath 使用相同 handle、通道目录；direct 不搬 payload 到 L1。目标 baseline 仍是 native HCCL。

```bash
# 先保持 A5_EID_MAX_RELAYS_PER_PEER=0，便于归因。
for A5_TYPE in float8_e4m3fn bfloat16; do
  for A5_REP in 1 2 3; do
    bash scripts/run_a5_ccu_eid_alltoallv.sh \
      --suite benchmark --mode all --planner l1_bottleneck \
      --matrix-json /实际路径/真实矩阵.json --dtype "$A5_TYPE" --width 5120 \
      --warmup 20 --iters 50 --rounds 6 --dynamic-iters 100
  done
done
```

BF16 combine 要另外保存 `C.T` 的 JSON 后测试，不能只改 dtype 就称为 combine。先测真实矩阵，再测单热点和均匀矩阵。报告各模式 prepared、host_counts_serial、dynamic；不把连续提交的 50 次当 50 个独立样本。记录每个 plan 的 L1 搬移量、各物理端口 TX/RX 行数、总峰值和三次独立运行差异。

端口验证沿用 [两卡 L0/L1 探针 §5](A5_CCU_TWO_RANK_L0_L1.md)：测试前后读取当前机相关 die/port 的 flit 计数，扣除空窗口背景流量。对 L1-only 备用目录，实际 payload 应只增加配置中的 L0 和单口 L1 端口；相邻未选 L1 口不应分摊 payload。端口计数有背景噪声时不以“UP”或带宽推断单口。先两卡验证，再进行八卡；八卡每个物理口承载多 peer 时，按汇总端口负载核对，不能按单逻辑 channel 分摊猜测。

## 8. 模型接入

拓扑必须用本分支、当前机器、同一 L1 policy 现场生成：

```bash
python3 scripts/prepare_a5_ccu_ep8_topology.py --executor eid \
  --devices "$A5_DEVICES" --max-relays-per-peer 0 --l1-config "$A5_L1_CONFIG" \
  --root-info /etc/hccl_rootinfo.json \
  --topology-json /usr/local/Ascend/driver/topo/950/atlas_950_1.json \
  --hccn-tool /usr/local/Ascend/driver/tools/hccn_tool \
  --output-dir "$A5_OUTPUT_ROOT/l1-alltoall/model-plans"
```

适配层从同一个 `model-plans` 目录读取 manifest 和 topology。创建 planner 时：

```python
planner = RelayPlanner(
    row_bytes, topology['allowed_relays'], native_module=cpp_ext,
    strategy='l1_bottleneck', l1_resources=topology['l1_resources'],
    l1_budget=topology.get('l1_budget'), max_actions=56)
# 后续 ep8_eid_create / run_counts_tensor 沿用现有接口。
```

不创建旧 HBM first/second-hop lanes。仍复用模型的 CPU C/splits；combine 复用 dispatch C 的转置和该次正确的行宽。先关闭模型侧性能诊断测正确性，再按现有独立交替 A/B 口径测端到端。当前提交不修改 vllm-ascend 仓库，也不宣称有端到端收益。

## 9. 已完成与现场待验证

开发容器 `cam_lyw_92` / CANN 9.2：route 和 SDK 库存工具编译，route 加载/ABI 检查；新 planner 的实际 pybind 扩展编译和 CPU 执行；端口预算与配置校验；CCU 指令图的 CPU 数据回放。编译不需要占用 NPU。

该容器没有 Torch/torch_npu，完整 DeepEP wheel 和 Torch host 回归留给目标环境重建/执行；CPU graph 模拟不模拟真实 EID 转发网络。尚未验证新生产 2/4/8 卡 L1 目录的 9.3 建链、设备正确性、端口落点与性能。用户先完成 §4–6，再做 §7；通过后才按 §8 接模型。

## 10. 计算图支持性验证

见 [CCU + URMA 计算图验证指南](A5_CCU_ACL_GRAPH_VALIDATION.md)。这项验证针对 ACL Graph 捕获/重放，
不是 CCU 指令构图或 rank graph 枚举。先完成指南 §1–3 的两卡裸提交，再做 §4 的实际 EP8 host/事件桥；
§5 单独验证动态参数。当前 fixed prepared eager PASS 不代表已支持计算图，不能直接以移除
`--enforce-eager` 代替最小验证。该指南包含待现场实现的诊断入口，未新增生产 graph API。

2026-11-11 现场已回报 G1/G2/G3 固定图通过、G4 两张固定图交替通过；同图动态 C 更新未验证。
DeepSeek 本轮重点读指南 **§12–15**：§13 是 DS4.1 原生 prefill、CPU splits 和 attention task update
的源码核对，§14.1 先不占卡确认现场版本，§14.2 再做短 native 路径验证，§14.3–14.4 是条件性动态图探针。
可读模型源为 `56debc1`，不等于现场旧版加补丁；不能把 decode MC2 入图当成 prefill AllToAllV 更新 splits 的证明。


## 11. 资源预算版 planner：单节点现在可用，多节点预留接口

### 11.1 改动与不变项

本次改的是 host planner、配置编译、库存默认选口和测试入口；CCU kernel、route ABI、selector=8、task args=46 不变。**需要重编 wheel**。已有本分支 route ABI=1 可继续使用；不必因 planner 改动单独重编 route。

- 8×8 C 只表示 EP8 的需求；端口不是新的 rank，不扩张 C。
- 每个 flow 仍是 L0 + 最多一个备用路径。多个 peer 可以使用不同单口 L1 路径，但不会把一条 flow 拆到八个 L1 EID。
- 保留现有单 IO die 限制；库存中另一 die 的单口不能因“有八个 L1 口”直接加入本 kernel。实际可用路径以现场生成 policy 为准。
- 默认库存选口改为优先使用已经分配 peer 最少的源/目的物理口；不同 EID 若对应同一物理口仍共用配额。不改已有 policy，需重新发现并在新目录生成才能使用这个默认分配；显式 policy 仍优先。
- L1 策略最多四个 target 尝试、每流最多一个动作、最多 56 个动作，不做无界全局求解。不保证找到可行最优解，也不保证模型削峰等于时延收益。

算法的候选约束（target 单位为 L0 等效行数）：

```text
可增加行数 = floor(max(0,
    target × row_bytes × capacity_permille / 1000
    − background_bytes − 本次已分配字节) / row_bytes)
```

源 TX 和目的 RX 都够才允许 L1 分流；TX/RX 独立记账。端口 capacity=0 表示不可新增，background 必须为零（否则配置自相矛盾）。成功结果再次检查所有资源不超过 target，失败尝试整份回滚。

纯 CPU 资源模型用例（跨卡每流 3072 行，self 不参与）：

| 已配置资源 | 新 planner 可得结果 | 动作数 |
|---|---|---|
| 每卡一个共享等带宽 L1 口 | 每条 L0 2688；每流 L1 384；每卡 L1 TX/RX 各 2688 | 56 |
| 每 peer 独立等带宽 L1 口 | 每条 L0 1536；对应 L1 1536 | 56 |
| 上一行的 L1 每口仅 50% 可用带宽 | 每条 L0 2048；对应 L1 1024，等效负载 2048 | 56 |
| 动作预算只有 8/16、均匀矩阵所有边都超过 target | 整份尝试失败，保持 direct | 0 |

第二行是资源模型验证，不是声称当前机器同 IO die 有七个独立 L1 口。是否能取得该拓扑、是否有共享内部瓶颈，须按实际 policy/端口计数验证。

### 11.2 控制器预算格式（可选，当前无外部占用可不填）

在 EP8 的 `l1_policy.json` 顶层加 `resource_budget`；下面字段示例的 device/die/port 必须替换为该 policy 已配置的端口：

```json
{
  "resource_budget": {
    "generation": "budget-window-001",
    "ports": [{
      "device_id": 0, "die": 0, "port": 1,
      "tx_capacity_permille": 500, "rx_capacity_permille": 1000,
      "tx_background_bytes": 1048576, "rx_background_bytes": 0
    }]
  }
}
```

capacity 是相对一个 L0 单口的可用带宽，整数 0..1000；不填的端口默认 1000。background 是**同一规划窗口内需要竞争该口的其他字节量**，默认零，不能填端口历史累计计数或 bytes/s；若用带宽降额代表某项外部竞争，就不要又把同一项重复计入 background。

背景字节在 FP8/BF16 两方向保持字节单位，planner 用本次 row_bytes 重算预算，不把 FP8 的行数套给 BF16。必须基于同一个窗口更新所有 rank 的配置，预算是一份不可变快照，不会每调用查询控制器或读取 JSON。更新预算需要在无在途操作时重新创建 planner/host 并丢弃旧 prepared 计划；不要只改 Python planner 却让 host 留用旧副本。

`prepare_a5_ccu_ep8_topology.py` 将物理 device ID 映射到本通信组顺序的 `rank*18 + die*9 + port`，输出 `topology.l1_budget`，字段为 schema/generation 与四个 144 元素数组。harness 已自动传给 planner，且 topology 包含在跨 rank 配置摘要检查内；模型适配层按 §8 同时传 `l1_resources` 和可选 `l1_budget`，并在建 host 前确认所有 rank 的配置摘要一致。

这个接口预留给跨节点/多组共享资源的控制器：它可以给每个 EP8 分配可用份额、估算竞争工作量。**它不是跨机 EP16 实现，不会自动发现交换机内部拥塞，也不执行带宽隔离。** 将来一个 EP16 覆盖两节点时，还需为跨机原始流建立基准路径与更上层资源协调；本次仅验证“已有 L1 负载不能当作空闲”这一局部契约。

2/4 卡仍使用原显式权重执行接口，不读取 EP8 资源预算；给 2/4 policy 添加 `resource_budget` 会拒绝，避免静默忽略。

### 11.3 DeepSeek 更新和验证步骤

1. 拉取本分支，按 §4 重编/安装 wheel。检查 `ep8_single_port_l1_abi=1` 与 **`ep8_l1_resource_planner_abi=1`**；新 Python 配旧 wheel 会明确报错。保留现场未提交诊断补丁，先检查 `git diff`，不要强制 reset。
2. 在空闲卡上按 §5/§6 重新生成一个新 policy 目录。比较新旧 `paths` 的实际物理口分布；如果继续旧 policy，只会更新 planner，不会自动换端口。
3. 先 CPU 回归，再 create、短正确性；最后用同一机器 policy 测真实矩阵和 uniform。若预算限制导致无 relay，要如实记录，不能伪报 relay PASS。

```bash
# 在已安装新 wheel、配置好 CANN/route/rootinfo 等环境后运行。
export A5_L1_CONFIG=/实际路径/新生成或已审核的policy-8/l1_policy.json
export HCCL_BUFFSIZE=64
export A5_EID_MAX_RELAYS_PER_PEER=0  # 首轮隔离 L1，之后再测 6
python3 -m unittest discover -s tests/python/deepep -p test_a5_ccu_l1_cpu.py -v
bash scripts/run_a5_ccu_eid_alltoallv.sh --suite create --planner l1_bottleneck
bash scripts/run_a5_ccu_eid_alltoallv.sh --suite correctness --planner l1_bottleneck \
  --case uniform --mode all --dtype float8_e4m3fn --width 5120 \
  --max-actions 56 --warmup 3 --iters 3 --rounds 1 --dynamic-iters 3 \
  --bidirectional-iters 3 --allow-no-relay

# 真实矩阵；同目录 policy 固定，比较动作预算，不同时更换路径。
export A5_MATRIX=/实际路径/real_counts.json
for A5_ACTIONS in 8 16 56; do
  bash scripts/run_a5_ccu_eid_alltoallv.sh --suite benchmark --planner l1_bottleneck \
    --matrix-json "$A5_MATRIX" --mode all --dtype float8_e4m3fn --width 5120 \
    --max-actions "$A5_ACTIONS" --warmup 20 --iters 50 --rounds 6 \
    --dynamic-iters 100 --allow-no-relay
done
```

将真实矩阵命令换成 `--case uniform` 再做预算对照；BF16 使用 `--dtype bfloat16 --width 5120`，combine 语义对照用真实矩阵的 Cᵀ。先短正确性，任何错误停止该轮。最后才接模型 eager，保持 native/multipath 相同配置；不需要为本次 planner 修改继续动态图验证。

独立验证背景预算时，可在新 policy 副本里把所有已配置 L1 口的 TX/RX capacity 设为 0、background 设为 0；在 `MAX_RELAYS_PER_PEER=0` 下，应得全 direct。用原可用预算重新创建 planner 后才恢复 L1。policy 副本须与 inventory 同目录或调整正确的相对路径，不能破坏 SHA 绑定。

回报：policy generation/SHA、budget generation、实际每卡端口数、actions、L1/participant moved、候选评估数、资源峰值前后、每口累计行数；prepared/host_counts_serial/dynamic 各自报告，做至少三次独立进程并交替模式顺序。模型等效峰值不是真实微秒数；最多 56 动作是否值得，需要这组设备结果决定。

### 11.4 开发侧已验证与限制

在本人 `cam_lyw_92` / CANN 9.2 完成实际 pybind planner 编译与 CPU 回归；资源模型测试包含 1000 组矩阵×两行宽、16000 份 executor 布局对照，另加 200 组带背景/带宽降额矩阵×两行宽；覆盖均匀分流、共享端口、关闭端口、整数边界、回滚及 FP8/BF16 字节单位。原 bottleneck 回归与机器拓扑回归通过。route 编译、host ctest、`PUBLIC_ROUTE_LOAD_PASS` / `PUBLIC_ROUTE_BUILD_PASS` 通过（未安装到运行环境）。

纯 CPU 规划微基准已加入 `tests/cpp/bench_a5_ccu_l1_planner.cpp`，5 批×10000 次，取中位；不含 Python、D2H、通信，不是设备时延。该开发容器不具备 Torch 完整 wheel 构建环境，现场仍须重编 wheel 并完成 NPU 正确性/吞吐测试。

```bash
c++ -O3 -std=c++17 -I csrc/deepep tests/cpp/bench_a5_ccu_l1_planner.cpp -o /tmp/l1-plan-bench
/tmp/l1-plan-bench
```

同开发机本轮参考：matrix_few 共享口约 2.6 µs；独立 peer 口、14 动作约 2.7 µs；uniform 的 56 动作约 3.4 µs。旧版对应约 2.0/2.3/2.5 µs，但 uniform 旧版因内部 8 动作上限没有分流。此处增加的是约亚微秒至 1 µs 量级的 CPU 规划工作；不是 56 次 host launch，CCU 提交接口不变。结果受 CPU/编译器影响，应在目标机复测。


## 12. L1 总开关：不同机器按启动配置启用

```bash
# 无 L1 连通条件的机器，或只测 L0 fullmesh：
export A5_ENABLE_L1=0
# 有已验证单口 L1 的机器：
export A5_ENABLE_L1=1
export A5_L1_CONFIG=/实际路径/本机policy-8/l1_policy.json
```

| 设置 | 拓扑生成（2/4/8 卡） | EP8 planner |
|---|---|---|
| `0` | 忽略 --l1-config/A5_L1_CONFIG，不读取对应 L1 policy/inventory，不生成 L1 数据通道 | l1_bottleneck 转成 bottleneck，丢弃 L1 resources/budget，保留 L0 direct/fullmesh relay |
| `1` | 必须有显式 L1 config，执行本机单口 EID/拓扑校验；失败报错，不静默改走另一条路径 | 允许 l1_bottleneck；仍须选择该策略且提供 l1_resources，显式 greedy/bottleneck 不自动改策略 |
| 未设置 | 兼容原行为：有显式 L1 config 才生成 L1 | 兼容原行为：显式 l1_bottleneck 才用 L1 |

只接受字符串 `0`/`1`，空字符串、true、auto、2 都报错。建议所有生产启动脚本明确设置 0 或 1。当前单口限制不变，开关不会启用聚合 EID，也不自动证明网络连通性。

**开关须在生成 manifest、创建 handle/host 之前设置，所有 rank 一致。** 关闭后重新生成本机 L0-only manifest/topology 并重启对应算子资源；不能仅改环境变量后继续用旧的 L1 handle/prepared plan。仅新建一个禁用 L1 的 planner 可以阻止它选择 selector=8，但不会销毁已经注册的 L1 通道。

官方 harness 已统一处理：

```bash
# 即使环境中保留旧 A5_L1_CONFIG，关闭也不会读取那个文件。
A5_ENABLE_L1=0 bash scripts/run_a5_ccu_eid_alltoallv.sh \
  --suite correctness --planner l1_bottleneck --mode all \
  --dtype float8_e4m3fn --width 5120 --warmup 3 --iters 3 --rounds 1 \
  --dynamic-iters 3 --allow-no-relay

# 新目录生成本机 L1 拓扑；不能复用别的机器的 policy。
A5_ENABLE_L1=1 bash scripts/run_a5_ccu_eid_alltoallv.sh \
  --suite correctness --planner l1_bottleneck --mode all \
  --dtype float8_e4m3fn --width 5120 --warmup 3 --iters 3 --rounds 1 \
  --dynamic-iters 3 --allow-no-relay
```

关闭时打印 `L1_SWITCH enabled=0`、`L1_PLANNER_SWITCH requested=l1_bottleneck effective=bottleneck`。直接运行 NPU 测试并传入旧的含 L1 topology 时会拒绝，提示重新生成，避免仍创建 L1 通道后误称“已关闭”。如 `A5_EID_MAX_RELAYS_PER_PEER=0` 同时关闭 L1，则只剩 direct；要保留 fullmesh relay，使用原来的非零组内 relay 目录配置。

模型接入：启动进程继承同一个 A5_ENABLE_L1；用相同设置运行 prepare 脚本生成配置，再按 §8 创建 RelayPlanner/host。`planner.requested_strategy`、`planner.strategy`、`planner.l1_enabled` 可记录实际选择。不要用代码直接调用底层 `native.Ep8RelayPlanner.with_l1_routes()` 绕过 Python 开关，也不要复用关闭前的 prepared plan。当前未自动修改现场 vllm-ascend 适配文件。

本次开关是 Python/配置层改动，C++ 与 route ABI 不变。安装方式若从 wheel 读取 Python 文件，需要更新 wheel 中的 `heuristic_alltoallv.py` 与新模块 `ccu_l1_config.py`（建议按原步骤重新构建安装）；仅更新仓库不一定会更新服务实际加载的包。

无卡回归：

```bash
python3 -m unittest discover -s tests/python/deepep -p test_a5_ccu_l1_switch_cpu.py -v
python3 -m unittest discover -s tests/python/deepep -p test_a5_ccu_l1_cpu.py -v
```

覆盖：关闭时不读不存在的 L1 文件、生成 L0-only EP8 拓扑、非法开关值拒绝、显式开启缺配置拒绝、关闭后实际 native planner 仍能选择 L0 relay 且不会选择 L1。
