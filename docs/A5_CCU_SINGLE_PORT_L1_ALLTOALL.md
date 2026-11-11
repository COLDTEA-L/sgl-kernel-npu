# 2/4 卡与 EP8：单口 L1 路径接入和现场测试

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

1. 从 C 去掉 self 得到初始 L0 边负载。L1 TX/RX 负载从零开始；只统计本通信组本次调用，不推测外部业务占用。
2. 计算目标下界：原有有向割加上可用 L1 源/目的物理口数量所给的乐观容量；并保留单流最多两路的 `ceil(peak/2)` 下界。这里假设 L0/L1 单口等带宽，尚未建模交换网络内部争用、端点聚合带宽和 HBM 带宽，不能将削峰率当时延收益。
3. 最多尝试原来的四个目标 `LB`、`LB+ceil(gap/7)`、`/3`、`/2`。只处理超过目标的边，每流移动最少的超出行数，同时遵守最小 chunk、最小收益及最多 8 个动作的约束。
4. L0 relay 候选检查两条借道边；L1 候选检查源 TX 物理口和目的 RX 物理口的**累计负载**。多个逻辑通道共享同一端口时，共用该预算；反向 RX/TX 分开，按全双工建模。
5. 选加入这部分流量后候选资源峰值更低的路径，同值优先 L1。每条流只选一个备用路径。若不能让所有相关资源都压到目标下，丢弃这一整次尝试；全失败保持 direct。
6. C++ host 按本次 counts 和 row_bytes 更新参数并执行。dispatch FP8 与 combine BF16 都支持；combine 用 C 的转置，**重新按转置方向和当前行宽规划**，不能直接交换旧 relay action。

可观测字段：`route_selector_8`、`l1_rows`、`participant_relay_rows`、`l1_tx_port_rows`、`l1_rx_port_rows`、`resource_lower_bound_rows`、`planned_link_peak_bytes`、`candidate_evaluations`。L1 的源出口和目的入口不是两份有效载荷，`transmitted_network_bytes` 对 L1 部分只计一次；它也不等于交换机内部 hop-byte 计数。

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

按 [9.2/9.3 构建指南 §4](A5_CCU_92_93_ALLTOALL.md#4-重建目标-deepep-wheel) 重建并安装当前分支 wheel，确认 `deep_ep_cpp.ep8_single_port_l1_abi == 1`。该指南的旧分支名只用于历史说明，本次不要切回旧分支。

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

对照 L0-only 时用 `env -u A5_L1_CONFIG bash ...`，保持 bytes、dtype、时序口径不变。不要将不同通道目录之间的差全部归于路径带宽。4 卡共享 L1 端口的静态分流未做动态流量优化，不承诺线性叠加带宽。

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
    strategy='l1_bottleneck', l1_resources=topology['l1_resources'])
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
