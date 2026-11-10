# 当前机器的 CCU 多路径选路：rootinfo + Atlas + hccn_tool

> 尚未拉取本分支：从 [分支操作指南第 1 步](A5_CCU_ALLTOALLV_BRANCH_QUICKSTART.md#1-拉取并切换分支) 开始，按序完成环境、编译、选路和短测。

适用分支：`feature/a5-ccu-8rank-direct-alltoallv`。

本分支同时保留旧 2/4 卡等长多路径 AllToAll 和新的 8 卡 direct-only
AllToAllV。两者的 C/Python 接口、kernel、ABI、数据布局和通知协议不变。
这次修改在资源准备脚本中替换 relay 地址的来源，不实现 8 卡 relay。

## 地址如何生成

1. 从当前机器 rootinfo 2.0 的 `rank_list[].device_id` 读取物理卡号，不要求
   资源文件包含 collective 的 `rank_id`；`local_id` 有则记录，无则留空。
   物理卡身份必须显式来自 `device_id`，不从 rank_id/local_id 或行号猜测。
2. 在指定 `net_layer` / `net_instance_id` 中读取 `rank_addr_list` 的
   `addr_type=EID`、`addr`、`ports`、`plane_id`。端口必须是明确的 `DieID/PortID`。
3. 依据 Atlas JSON 的 `peer_list[].local_id`（物理卡号）和
   `edge_list[].local_a/local_b`、两端 ports，关联两段 `UB_CTP / PEER2PEER`
   DEVICE 链路。端口列表按位置配对；列表长度不相同则报错。
4. 在资源准备时，对涉及的每张物理卡执行一次：
   `/usr/local/Ascend/driver/tools/hccn_tool -g -dev_info -i PHYSICAL_ID`。
   校验明确绑定的 EID 确实出现在该卡 EID 表中，端口状态为 `UB / UP`。
   源、目的、relay 入口和出口四个端点都检查。
5. 要求 relay 两段在同一 IO die，relay 本地 plane 与 instance 相容，路径唯一。
   有歧义则报错，不取第一项、不取模、不通过 EID 位段推测端口。
6. 输出原格式 TSV；peer-plan JSON 自动包含配置文件路径、SHA256、查询命令、
   查询输出 SHA256 和地址验证结果。单独 resolver 可以用 `--diagnostics-json`
   保存相同来源记录。

支持紧凑 32 位十六进制、完整及压缩冒号格式的 EID，统一输出完整冒号格式。
一个 EID 绑定多个明确 ports 时逐个关联；IPV4/IPV6 host 地址不会被重新解释成 EID。
UDMAC 名称和 EID Index 仅用于记录，不参与设备、端口或路径推断。

**不再读取或维护 `docs/topology/a5_hccn_device_topology_raw.txt`。**
旧 `--topology TXT` 参数在实际生成时明确报错并给出迁移提示。
历史文档中的旧 TXT 命令应改成下面的新选项。

## 配置文件发现

不传路径时，在当前执行环境中检查：

- rootinfo：`/etc/hccl_rootinfo.json`、`/etc/hccl_rootInfo.json`。
- Atlas：`/usr/local/Ascend/driver/topo/950/atlas_950.json`、
  `/usr/local/Ascend/driver/topo/950/atlas_950_1.json`。
- hccn_tool：`/usr/local/Ascend/driver/tools/hccn_tool`。

同类自动候选内容不同时，要求显式指定，不猜哪个有效。没有完整 2.0 EID/ports
映射时明确失败。可用参数：

```text
--root-info PATH        # --rootinfo 是同义参数
--topology-json PATH
--hccn-tool PATH        # 查询工具实际名 hccn_tool；也接受 --hccl-tool 同义参数
--query-timeout 15      # 每张卡查询上限秒数
--net-layer 0
--net-instance-id NAME  # rootinfo 涉及多个机器时明确选择本机 instance
--plane-id LABEL        # rootinfo 网络 plane 标签，不是 IO die
--diagnostics-json PATH
```

保留旧 `--plane` / `--relay-planes` / relay-map `plane` 的含义：选择 **relay IO die**。
它们和新增 `--plane-id` 是不同维度。跨卡物理 hop 由 Atlas 的明确端口配对决定，
不能要求两个设备的本地 plane 标签字符串相同。例如 `2:0/6 plane_0` 与
`4:1/0 plane_1` 可以是 Atlas 明确连接的两个端点。
relay 自身的入口/出口仍保守要求同 IO die、相同本地 plane 标签；
`--plane-id` 是对所有地址的显式字面过滤，只有确定所有所需端点都具备该标签时才使用。

Docker 中必须使用当前宿主机的实际 rootinfo、Atlas 与 driver 信息。
先核对宿主机与容器内容，不能从旧镜像复制配置覆盖新宿主机。
本脚本不改系统文件，不初始化 HCCL，不修改全局路由，也不会修复 HCCL 的 EI0014。
选路校验成功和 HCCL 通信域初始化成功是两个独立条件。

## 只生成计划，不占卡运行通信

两卡示例（IDs 和 IO die 按实际拓扑选择）：

```bash
python3 scripts/prepare_a5_ccu_peer_plan.py \
  --devices 2,3 --relay-phys 0,1 \
  --root-info /etc/hccl_rootInfo.json \
  --topology-json /usr/local/Ascend/driver/topo/950/atlas_950_1.json \
  --output-dir /tmp/a5-two-card-plan
```

若返回多条候选，可用 relay-map 明确指定 relay IO die，而不是重新手写 EID：

```json
{"2,3": [{"physical_device": 0, "plane": "0"}, {"physical_device": 1, "plane": "0"}]}
```

使用 `--relay-map /path/policy.json` 代替 `--relay-phys`。
此 JSON 是用户选择的卡对/relay 策略，不是静态机器 EID 库。

四卡策略仍使用 `--relay-map`，例如：

```json
{
  "0,1": [{"physical_device": 4, "plane": "0"}],
  "2,3": [{"physical_device": 4, "plane": "0"}],
  "0,2": [{"physical_device": 5, "plane": "0"}],
  "1,3": [{"physical_device": 5, "plane": "0"}],
  "0,3": [{"physical_device": 6, "plane": "0"}],
  "1,2": [{"physical_device": 6, "plane": "0"}]
}
```

```bash
python3 scripts/prepare_a5_ccu_peer_plan.py \
  --devices 0,1,2,3 --relay-map /path/four-card-policy.json \
  --output-dir /tmp/a5-four-card-plan
```

两卡旧 explicit-multirelay 接口也使用同一查询模块，输出列和 route_id 保持兼容：

```bash
python3 scripts/resolve_a5_explicit_multirelay_eids.py \
  --src-phy 2 --dst-phy 3 --relay-phys 0,1 --relay-planes 0,0 \
  --diagnostics-json /tmp/a5-route-source.json > /tmp/a5-relays.tsv
```

peer-plan、explicit-multirelay、explicit-multipath 性能测试、channel-sync、
synthetic probe/causal compare 和 HBM command runner 均转发这些新选项。
查询和关联在准备阶段一次完成；warmup/正式调用复用 prepared plan。

直连仍由 HCCL 从当前 communicator 发现并由 runtime 检查 hop=1、物理卡身份和
本地 IO die。没有 synthetic relay 的 2/4/8 卡 direct-only 计划不需要运行 hccn 查询，
也不需要这三个来源来人为构造 EID。`--direct-route` 继续表示 HCCL 候选序号，
不是 Atlas 物理边编号；机器变化后应核对对应的 PATH_CATALOG。

## 验证范围

CPU 回归：

```bash
python3 -m unittest discover -s tests/python/deepep -p 'test_a5_ccu_*cpu.py' -v
```

新测试使用显式 rootinfo/Atlas fixtures 和独立的 fake hccn 子进程；覆盖新编码、
实际 hccn 表格形状、一次/卡查询、2/4 卡 TSV 和 8 卡 direct-only，以及过期地址、
端口 DOWN、缺失字段、网络 plane 不一致、跨 die 和多候选失败。
它们不证明新机器真实配置、建链、NPU 正确性或性能已经通过。

字段依据：[CANN 9.1 Atlas 950 rank table 与拓扑配置](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/commlib/hcclug/docs/en/user_guide/cluster_info_config/rank_table_config_a5.md)。

### 本次开发验证（2026-10-27）

- 开发服务器 CPU suite：87 项通过，包含原有 2/4 卡及 EP8 CPU kernel 模型。
- `cam_lyw_dev_91`：route 自定义包重新构建通过，DeepEP build target 通过。
  prepared=16、peer=5、channel-sync=1、AllToAllV=1 ABI 验证通过。
- 编译产物 Meta gate：160 个有效 EP8 用例、12 个非法输入、10 个旧接口用例通过。
- 用户提供的新机物理 2 卡 hccn 输出可解析为 21 个 EID、17 个端口；这是离线解析。
- 未安装系统包，未运行 NPU 通信。尚未取得新机完整 rootinfo/Atlas 配对样本，
  因而未确认新机配置兼容性、HCCL 初始化、实际 ChannelAcquire 或设备数据正确性。
- 开发服务器日志：`build/ep8_validation/rootinfo_cpu.log`、`rootinfo_build.log`。

### 资源文件缺少 rank_id 时

2026-10-27 修正：旧解析器误把 collective rank table 的必填 `rank_id` 校验
套用到本地资源 rootinfo，可能出现 `error: 'rank_id'`。选路实际上不用该字段，
现已移除依赖，`local_id` 也仅作为可选元数据。无需修改系统 rootinfo 或给它补假 rank。
仍要求明确的 `device_id` 和地址—端口绑定；其他必要字段缺失时，会报告字段完整路径
和该对象已有的键，便于核对实际 schema。这次是 Python 准备脚本修复，无需重编二进制。

### 新机 2→4→3 诊断（2026-10-27）

已见 Atlas 配对为 `2:0/6 ↔ 4:1/0`、`3:0/3 ↔ 4:1/3`，因此 relay die 为 1。
不同设备的 plane_0/plane_1 标签不同，不应单凭字符串不等过滤这条物理边；已修正。
用户输出同时显示上述四个 rootinfo EID 的 `live_eid_present=False`，端口为 UB/UP。
这项地址不一致尚未定位，仍会阻止建计划；不能因端口 UP 就跳过 EID 校验。
需进一步比较 rootinfo 中的实际 EID、实时 hccn EID，以及宿主机/容器配置来源。

### 宿主机与容器来源已确认不一致

用户进一步确认宿主/容器 rootinfo hashes 不同，显式挂载缺少 rootinfo；容器地址
逐条命中旧机快照。当前容器修复及后续只读挂载见
[容器 rootinfo 修复指南](A5_CCU_ROOTINFO_CONTAINER_FIX.md)。
检查宿主副本是否匹配实时设备后再安装，不能直接把“宿主文件”视作已经校验。
