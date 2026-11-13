# CANN 9.2/9.3：direct + relay 与图内逐算子通知穿刺

分支：`feature/a5-ccu-valuewait-callback-puncture-92-93`。目标是让 DeepSeek
在目标卡上验证同一张图内的两个通信节点可被 host 分别放行，并能按本轮决策改变
direct/relay 数据分配。程序不依赖 Torch，不代表 DeepEP/NPUGraph 已验证。

## 拓扑和实现

- 两个通信 rank；支持 1 条 direct + 1 或 2 条显式 relay。
- 默认示例通信卡 2、3，relay 0、1：实际共涉及 4 张物理卡。
  只有两张物理卡时不能验证独立 relay；请等 relay 卡可用后执行完整实验。
- 所有卡号由 launcher 参数及 manifest 显式给出，校验二者顺序一致且卡号不重叠。
- 复用生产 `GetRouteResources()`：direct 为 Channel 0，后续 relay Channel 按
  resolver manifest 的行顺序建链。保留同一 local die 的约束，不自动降级 direct。
- `PolicyKernel()` 在 ValueWait 后从 HBM policy 数组读取每条路径的字节数，
  根据前缀和计算源和目的偏移。每次先提交所有非零 Write，再等全部 EventWait，
  最后通过 direct Channel 完成两端 NotifyRecord/NotifyWait。
- CCU 将实际读到的各路径字节数 Store 到 audit，host 对照决策并逐字校验 payload。
  零字节路径不提交 payload，但仍参与 metadata 交换。

```text
一张图，一次 capture，多次 replay，exec 为实际 routeThread 所绑定的 stream

  ARRIVED0 → ValueWait(gate0) → STARTED0 → 通信0 → DONE0 → gate0=0
       → D2D copy(recv0 → send1)
       → ARRIVED1 → ValueWait(gate1) → STARTED1 → 通信1 → DONE1 → gate1=0

  host: 到达0 → 保持窗口检查 → 写policy0 → gate0=1 → 完成0/两rank校验
        到达1 → 保持窗口检查 → 写policy1 → gate1=1 → 完成1/两rank校验
```

D2D copy 代表图中的数据依赖节点，不是实际 MoE 计算。通信0将本rank输入送至对端；
通信1把收到的数据送回原rank，因此两个输出分别校验为对端输入、本rank输入。
图尾还要排空 exec/control stream，不能提前复位 host 通知槽或开始下一轮。

## 图内通知：两个独立实验，均未上卡验证

`--notify CALLBACK`：两个 ARRIVED 和两个 DONE 均以 `aclrtLaunchHostFunc`
放在捕获范围内。callback 参数在所有 replay 期间保持有效，只更新原子计数。
捕获期间不应执行 callback；replay 每轮每个 callback 必须恰好执行一次。
API 拒绝捕获、End失败、通知超时均返回失败，不将图外 callback 结果当成图内成功。

`--notify D2H`：图内执行 `ValueWrite(device_notice=1)`，随后把 8 字节通知
复制到预分配的 `aclrtMallocHost` pinned 槽。host 轮询 pinned 槽，无需启动等待
kernel，也不要求 HostFunc 可捕获。此模式仍有 host polling 和小 D2H 传输开销。
DMA 到 CPU 的及时可见性、捕获/replay 可用性也是待测项；volatile 读取仅防止
编译器省略访问，不是跨设备一致性证明。每轮等图完全排空后才将 host 槽归零。

独立 notification probe 不创建 HCCL communicator，也不建 relay Channel；
它用两个 gate 和四个通知测试同图逐节点 capture/replay，重复三轮。
用于区分通知机制问题与通信建链问题。运行它需要一张实际 NPU。

官方捕获语义参考：
https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/API/runtimeapi/aclcppdevg_03_1782.html
对应 9.2 安装已核对 CaptureBegin/End/ExecuteAsync、HostFunc 和 ValueWait 的真实声明，
声明和链接可用不能替代执行支持证明。

## 1. 编译（92 容器已验证）

在目标容器中，使用新分支 checkout；9.3 请换成其实际 SDK 路径：

```bash
unset LD_PRELOAD
source /usr/local/Ascend/cann-9.2.0/set_env.sh
bash scripts/build_a5_ccu_urma_route_probe.sh \
  --cann-path /usr/local/Ascend/cann-9.2.0 \
  --build-dir /tmp/a5-relay-gate-build
/tmp/a5-relay-gate-build/a5_graph_gate --check-abi
```

新增 CMake target `a5_graph_gate` 使用真实公开 CCU/ACL 头文件，
`-Wall -Wextra -Werror`、`-Wl,-z,defs`。route builder 不加 --install，
不覆盖原 SDK。二进制加载构建目录的 route 库，不要预加载旧 9.1 patched HCCL。

开发服务器本次编译目录：
`/home/liuyuanwen/profiling/a5-relay-graph-validation.6xu62w/`。
`cam_lyw_92` 没有 NPU 映射，本次只能确认编译、链接、ABI 加载和已有 host ctest；
不能确认 callback graph、D2H graph 或实际 relay 已通过。

## 2. 先隔离验证图内通知（需要单卡）

每个命令用外部 timeout，卡死时终止独立进程，保留 plog。逻辑 device 0 映射到示例物理卡2：

```bash
ASCEND_RT_VISIBLE_DEVICES=2 timeout --kill-after=5 60 \
  /tmp/a5-relay-gate-build/a5_graph_gate --notification-probe 0 CALLBACK
ASCEND_RT_VISIBLE_DEVICES=2 timeout --kill-after=5 60 \
  /tmp/a5-relay-gate-build/a5_graph_gate --notification-probe 0 D2H
```

成功必须同时出现 `NOTIFICATION_CAPTURE_PASS` 和 `NOTIFICATION_REPLAY_PASS`。
如 CALLBACK 不支持，记录失败 API、返回码、SDK/driver/firmware，D2H 另起进程测试。
不能只看到 capture 成功就声明回调在 replay 时按节点触发。

## 3. 准备实际 relay manifest

使用本机 rootinfo/Atlas JSON 与实时 EID 查询；不使用旧机器静态 EID 或 TXT 清单。
以下为默认卡位示例；更换卡位时 resolver 和 launcher 同时修改：

```bash
python3 scripts/resolve_a5_explicit_multirelay_eids.py \
  --root-info /etc/hccl_rootinfo.json \
  --topology-json /usr/local/Ascend/driver/topo/950/atlas_950_1.json \
  --src-phy 2 --dst-phy 3 --relay-phys 0,1 \
  > /tmp/a5-gate-relays.tsv
```

如果候选不唯一，用 `--relay-planes` 等明确选择，不自动猜测。可以只指定一个 relay，
最少需要两个 endpoint 加一个独立 relay；relay 可以有背景业务，但不能据此跳过物理对照。
manifest 中 direct 与所有 relay 的 endpoint local die 必须兼容，否则明确失败。

## 4. 完整双节点通信测试

先用 EAGER 验证控制协议和传输，再在同一配置下运行 GRAPH。两种通知均独立测试：

```bash
for MODE in EAGER GRAPH; do
  for NOTIFY in CALLBACK D2H; do
    python3 scripts/run_a5_relay_graph_gate.py \
      --binary /tmp/a5-relay-gate-build/a5_graph_gate \
      --devices 2,3 --relay-phys 0,1 --relay-manifest /tmp/a5-gate-relays.tsv \
      --mode "$MODE" --notify "$NOTIFY" --policy cycle \
      --bytes 4194304 --iters 20 --hold-ms 100 --timeout 300 \
      --output-root /tmp/a5-gate-results
  done
done
```

如 GRAPH/CALLBACK 不支持，单独保留其失败，不影响另起进程验证 GRAPH/D2H。
默认 cycle 每轮、每节点在 direct-only、relay0-only、relay1-only、mixed 之间切换。
两 relay 的 mixed 分配为 `[B/2, B/4, B/4]`；单 relay 为 `[B/2, B/2]`。
同一张图反复 replay 不重新捕获/注册，不从 host launch 参数冻结路径字节数。

必须检查：

1. node0 放行前，node1 ARRIVED 不得出现；证明通知位于图内各自节点。
2. 每个 node ARRIVED 后保持窗口内 STARTED=0、DONE未到达。
3. 每次将 policy 字节数先写入控制流，再在该流写 gate=1。
4. 每个 node 的 device audit 等于本轮发布值，payload 完全正确。
5. 每节点完成后两rank通过独立文件 barrier，才作为全局节点完成；运行目录每次唯一。
6. 每rank每轮恰好2个 `GATE_OP_PASS`；20轮为40个。callback无重复或丢失。
7. 两个rank成功退出，launcher 输出 `GRAPH_GATE_ALL_RANKS_AND_EXIT_PASS`。
8. 改 `--hold-ms 2000 --iters 4` 做更明显的等待对照；运行超时相应增加。

## 5. 证明指定 relay 真正在转发

逻辑 audit 只证明 CCU 选择了对应 Channel。使用 `--policy direct`、`relay0`、
`relay1`、`mixed` 固定策略，分别启动相同字节量/轮数测试；前后采集 manifest
对应端口的 HCCN `tx_busi_flit_num/rx_busi_flit_num`。命令参数示例：

```bash
hccn_tool -g -stat -i DEVICE -u UDIE -p PORT
```

正向检查 src TX → relay ingress RX → relay egress TX → dst RX；反向同理。
预期 relay0-only 只显著提高 relay0 链，relay1-only 对应 relay1，mixed 两链均提高。
有背景业务时使用多轮随机顺序或 ABBA 对照和相同时间窗。未获得因果指纹时，只能
报告逻辑 Channel 调度通过，不能报告 physical relay confirmed。
可参考现有 `scripts/analyze_a5_ccu_explicit_multirelay.py` 的快照格式与链平衡检查。

## 交付给 DeepSeek 的结果要求

保留源码commit、完整命令、manifest及生成依据（不输出内存token）、rank日志、plog和
端口counter。分别报告：92/93编译、通知capture、通知replay、通信eager、通信graph、
物理relay证据。NPUGraph尚未覆盖，不以独立C++模型测试代替框架实际入图。
