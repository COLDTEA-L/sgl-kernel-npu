# A5 AIV → HBM CommandBlock → CCU worker 穿刺实验

## 1. 目的和判定边界

本实验用来验证不修改系统 `libhccl.so` 的标准算子架构：

```text
图外 prepare
  → 显式构造 direct/relay CommLink
  → HcclChannelAcquire
  → 启动常驻 CCU worker
  → 返回 worker_handle

图内标准 Ascend C AIV 算子
  → 将 send/recv 地址、分片、path mask 写入 HBM CommandBlock
  → flush/fence
  → 发布 EXECUTE
  → 轮询 completion/status

常驻 CCU worker
  → 轮询 CommandBlock
  → 读取这一次的控制参数
  → 对已准备的 Channels 并发 WriteNb
  → 写回 completion/status
```

实验分三层：

1. `HANDSHAKE`：`path_mask=0`，不搬运数据。通过只能证明 AIV → HBM → CCU 命令可见，以及 CCU → HBM → AIV 完成可见。
2. `TRANSFER`：同一 worker 读取 AIV 写入的地址和分片，通过显式 direct/relay Channels 完成真实两 rank AllToAll 并校验数据。
3. `ACLGRAPH_CAPTURE/REPLAY`：将 AIV 算子 capture 后改变输入并 replay，证明图中重放使用的仍是已准备的 worker/Channels，且数据不是首次执行的旧结果。

本穿刺不证明最终性能，也不以 HCCN counter 代替数据正确性。

## 2. 实现位置

| 层次 | 文件 | 作用 |
|---|---|---|
| CCU kernel | `examples/a5_ccu_urma_route_probe/op_kernel_ccu/command_block_worker_kernel.cc` | 轮询 mailbox，执行 `WriteNb`，写回 completion |
| 图外控制面 | `examples/a5_ccu_urma_route_probe/op_host/all_to_all_multiroute.cc` | 构造 Channels，启动/停止 worker |
| 标准 op def/tiling | `csrc/deepep/ops/op_host/ccu_hbm_command_puncture_*.cpp` | 定义 Ascend C 算子、shape/attr 校验和 tiling |
| AIV kernel | `csrc/deepep/ops/op_kernel/ccu_hbm_command_puncture.cpp` | 写 CommandBlock，flush，等待 completion |
| Python/C++ 绑定 | `csrc/deepep/deep_ep.cpp` | prepare/execute/stop 入口 |
| 两 rank 用例 | `tests/python/deepep/test_a5_ccu_hbm_command_puncture.py` | 三阶段验证和数据校验 |
| 一键脚本 | `scripts/run_a5_ccu_hbm_command_puncture.sh` | 解析 relay EID、生成 manifest、启动 torchrun |

## 3. CommandBlock ABI

CommandBlock 是一块至少 38 个 `uint64_t` word 的 NPU HBM：

| word | 字段 |
|---:|---|
| 0 | magic |
| 1 | command: `0=IDLE, 1=EXECUTE, 2=STOP` |
| 2 | completion |
| 3 | worker status |
| 4,5 | send/recv GM address |
| 6,7 | send/recv token，由图外 prepare 填入 |
| 8 | bytes per peer |
| 9,10,11 | self-copy source offset / destination offset / bytes |
| 12,13 | path count / path mask |
| 14..21 | 每条 path 的 source offset |
| 22..29 | 每条 path 的 remote offset |
| 30..37 | 每条 path 的 bytes |

发布顺序不能更改：AIV 先清 completion、写完 payload 并 flush，最后单独写/flush command。CCU 完成后先写 status、将 command 恢复 IDLE，最后才发布 completion。`completion=1` 是该 epoch 的 release 标志；如果在 completion 之后再清 command，有可能覆盖下一次 ACLGraph replay 刚提交的命令。

## 4. 拉取代码

```bash
cd /home/l00934901/sgl-kernel-npu
git fetch origin
git switch feature/a5-ccu-explicit-multipath-alltoall
git pull --ff-only origin feature/a5-ccu-explicit-multipath-alltoall
```

## 5. 编译并安装 route/CCU package

```bash
cd /home/l00934901/sgl-kernel-npu
source /usr/local/Ascend/cann-9.1.T560/set_env.sh

HCCL_REPO=/home/l00934901/hccl \
bash scripts/build_a5_ccu_urma_route_probe.sh \
  --install \
  --install-path /usr/local/Ascend/cann-9.1.T560
```

验证实际装入 CANN 的 SO：

```bash
ROUTE_SO=/usr/local/Ascend/cann-9.1.T560/opp/vendors/cust/lib64/liba5_ccu_urma_route_probe.so

nm -D "${ROUTE_SO}" | grep -E \
'HcclCcuUrmaCommandBlockWorker(Create|Stop)|A5CcuHbmCommandPunctureAbiVersion'

python3 - "${ROUTE_SO}" <<'PY'
import ctypes, pathlib, sys
p = pathlib.Path(sys.argv[1]).resolve()
lib = ctypes.CDLL(str(p), mode=ctypes.RTLD_GLOBAL)
fn = lib.A5CcuHbmCommandPunctureAbiVersion
fn.restype = ctypes.c_int
print("route library:", p)
print("command puncture ABI:", fn())
assert fn() >= 4
PY
```

## 6. 编译标准 AIV 算子并重装 wheel

```bash
cd /home/l00934901/sgl-kernel-npu
source /usr/local/Ascend/cann-9.1.T560/set_env.sh

DEEPEP_SINGLE_OP=ccu_hbm_command_puncture \
bash build.sh -a deepep Ascend950

WHEEL=$(ls -t output/deep_ep-*.whl | head -1)
python3 -m pip install --force-reinstall --no-cache-dir --no-deps "${WHEEL}"
```

某些旧构建脚本在单个 Ascend C kernel 编译失败后仍继续打 wheel，因此不能只用“生成了 wheel”作为成功
判据。构建结束后还必须确认本算子的二进制 JSON 已生成，且日志没有该算子的编译失败：

```bash
find csrc/deepep/ops/build_out/op_kernel/ascendc_kernels/binary/ascend950 \
  -path '*ccu_hbm_command_puncture*' -name '*.json' -print

! grep -RniE \
'CcuHbmCommandPuncture.*compile failed|Kernel Compilation Error: OpType CcuHbmCommandPuncture' \
  csrc/deepep/ops/build_out 2>/dev/null
```

验证加载的 wheel 不是旧版：

```bash
python3 - <<'PY'
import ctypes
from pathlib import Path
import deep_ep.deep_ep_cpp as ext

p = Path(ext.__file__).resolve()
lib = ctypes.CDLL(str(p), mode=ctypes.RTLD_GLOBAL)
abi = lib.A5DeepEpExplicitMultipathAttrAbiVersion
abi.restype = ctypes.c_int
print("loaded deep_ep_cpp:", p)
print("command puncture ABI carrier:", abi())
assert abi() >= 7
for name in (
    "prepare_ccu_hbm_command_worker",
    "ccu_hbm_command_puncture",
    "stop_ccu_hbm_command_worker",
):
    assert hasattr(ext.Buffer, name), name
print("command puncture Python APIs: PASS")
PY
```

## 7. 先运行 CCU 注册能力矩阵

如果完整 worker 在 `HcclCcuKernelRegister` 返回 `status=4`，不要继续猜测
HBM cache、EID 或 relay。先用注册矩阵把失败拆成四类：

| mode | 只注册的指令组 | 能回答的问题 |
|---|---|---|
| `hbm_once` | 一次原始 HBM `LoadVariable + StoreVariable` | 注册器是否接受 CCU 直接读写这块 HBM |
| `loop_only` | 不访问 HBM 的 `CCU_WHILE` | 注册器是否接受 repeat/while 控制流 |
| `loop_hbm` | `CCU_WHILE` 内只做 HBM load | 循环与 HBM 访问的组合是否可注册 |
| `full` | 完整 mailbox + direct/relay `WriteNb` worker | 完整 worker 是否可注册 |

这些模式都使用真实的 direct/relay Channel，但设置
`A5_CCU_WORKER_REGISTER_ONLY=1`，只执行 register/finalize，不 launch CCU kernel，
因此不会产生数据流量，也不会进入 AIV handshake。

> ABI 3 的四项测试曾全部在 `algorithm_begin` 后返回 4，但该结果不能用于判定
> HBM/WHILE 均不受支持：当时 `GeneArgs()` 返回了 `commandBlockAddr`，Algorithm
> 却没有用 `Load()` 消费该 task argument。ABI 4 已按原生 CCU kernel 的约定补齐
> 这一项；只有 ABI 4 的矩阵结果才具有能力判定意义。日志中应在各模式看到
> `phase=task_arg_ready`。

端点为物理卡 2、3，relay 为 0、1 时执行：

```bash
cd /home/l00934901/sgl-kernel-npu
source /usr/local/Ascend/cann-9.1.T560/set_env.sh
unset LD_PRELOAD A5_URMA_TP_TRACE_PREFIX

bash scripts/run_a5_ccu_hbm_command_register_matrix.sh \
  --src-phy 2 --dst-phy 3 \
  --relay-phys 0,1 \
  --direct-route 0 \
  --bytes 4194304 \
  --timeout-seconds 300 \
  --output-root /home/l00934901/profiling
```

结果位于：

```bash
MATRIX_DIR=$(ls -dt \
  /home/l00934901/profiling/a5_ccu_hbm_command_register_matrix_* | head -1)

column -s $'\t' -t "${MATRIX_DIR}/register_matrix.tsv"
grep -m1 '^Route puncture:' "${MATRIX_DIR}/hbm_once.log"
grep -RHnE \
'COMMAND_BLOCK_REGISTER_TRACE|COMMAND_BLOCK_REGISTER_PROBE_RESULT|register command_block_worker end' \
"${MATRIX_DIR}"
```

矩阵脚本会从第一个子测试日志回显一次 `Route puncture: ... ABI= 4`。旧版脚本
因为将子测试 stdout 重定向到 `hbm_once.log`，终端只显示 PASS/FAIL；这不表示 ABI
没有检查。上面的 `grep` 可以直接读取实际加载结果。

判读规则：

- `hbm_once=FAIL`：当前 CCU 注册边界不接受原始地址形式的 HBM Load/Store；先换成公开 remote-variable/GM 机制，不能进入 mailbox 调试。
- `hbm_once=PASS, loop_only=FAIL`：问题是 `CCU_WHILE`/repeat 控制流，需改为 host/AIV 每次 launch 一次短 CCU kernel，或寻找官方常驻 worker 原语。
- 前两项 PASS、`loop_hbm=FAIL`：单项均支持，但循环内 HBM 访问组合不支持；同样不能采用当前 busy-poll worker。
- 三个最小项 PASS、`full=FAIL`：基础控制链成立，失败来自完整分支、Notify/WriteNb 指令规模或资源上限；再对 full body 二分。
- 四项均 PASS：注册问题已解决，才继续下一节的真实 handshake/transfer。

## 8. 运行完整穿刺

例如端点使用物理卡 2、3，relay 显式指定为 0、1：

```bash
cd /home/l00934901/sgl-kernel-npu
source /usr/local/Ascend/cann-9.1.T560/set_env.sh
unset LD_PRELOAD A5_URMA_TP_TRACE_PREFIX

bash scripts/run_a5_ccu_hbm_command_puncture.sh \
  --src-phy 2 --dst-phy 3 \
  --relay-phys 0,1 \
  --direct-route 0 \
  --bytes 4194304 \
  --timeout-seconds 600 \
  --output-root /home/l00934901/profiling
```

relay 不限于两张；穿刺版最多支持 7 张 relay（加一条 direct 后共 8 paths）。例如只有卡 0、1 可作端点时，可改为：

```bash
bash scripts/run_a5_ccu_hbm_command_puncture.sh \
  --src-phy 0 --dst-phy 1 \
  --relay-phys 2 \
  --direct-route 0 \
  --bytes 4194304 \
  --timeout-seconds 600 \
  --output-root /home/l00934901/profiling
```

先只验证非图路径时增加 `--no-graph`。

## 9. 查看结果

```bash
RUN_DIR=$(ls -dt \
  /home/l00934901/profiling/a5_ccu_hbm_command_puncture_* | head -1)

cat "${RUN_DIR}/status.tsv"
grep -nE \
'COMMAND_BLOCK_WORKER|PUNCTURE_(WORKER|HANDSHAKE|TRANSFER|GRAPH_WARMUP|ACLGRAPH_CAPTURE|ACLGRAPH_REPLAY|STOP|RESULT)' \
"${RUN_DIR}/puncture.log"
```

完整通过应同时看到两个 rank 的：

```text
PUNCTURE_HANDSHAKE ... PASS
PUNCTURE_TRANSFER ... PASS
PUNCTURE_ACLGRAPH_CAPTURE ... PASS
PUNCTURE_ACLGRAPH_REPLAY ... PASS
PUNCTURE_STOP ... PASS
PUNCTURE_RESULT PASS
```

`COMMAND_BLOCK_WORKER phase=ready` 应只在 prepare 阶段出现，不应在 capture/replay 间重新创建 Channel。

## 10. 失败定位

1. 缺少 worker symbol：安装的 route SO 过旧，重做第 5 节。
2. DeepEP ABI `< 7`：导入了旧 wheel，重做第 6 节并核对 `ext.__file__`。
3. `HANDSHAKE` 超时：优先检查 AIV/CCU 对 CommandBlock 的 cache flush/可见性和 worker 是否真正启动，此时与 URMA 路径无关。
4. `HANDSHAKE` 通过但 `TRANSFER` 失败：查 relay manifest、Channel acquire、token 和分片；不要归因为 mailbox 不可见。
5. 普通 transfer 通过但 capture/replay 失败：检查是否在 capture 内重新 prepare/stop，以及 command/completion 是否正确清零。
6. 外层 timeout 中止进程时，`finally` 可能来不及发 STOP；进程退出后资源由 runtime 回收，但不要在同一进程中遗留 worker 后继续其他测试。
7. 当前公开 HCOMM 边界没有与 `HcclThreadAcquireWithStream`/`HcclChannelAcquire` 配对的 release API。穿刺程序会发 STOP 并等待 worker 退出，但不在存活进程内销毁仍被 HCCL thread/channel 引用的专用 stream。因此该版本用于一次性穿刺，不应在长寿命进程中反复 prepare/stop。
8. `register command_block_worker end: status=4` 表示失败在 CCU kernel 注册，尚未进入 HBM handshake。ABI 3 提供第 7 节的四级注册能力矩阵；先跑矩阵，再使用下列命令区分 primitive 生成失败和 register finalize 失败：

```bash
grep -nE \
'COMMAND_BLOCK_REGISTER_TRACE|register command_block_worker|HcclCcuKernelRegister' \
"${RUN_DIR}/puncture.log"
```

`primitive_failed` 会给出失败源码行；如果已出现 `algorithm_ready` 但 Register 仍返回 4，则问题在 HCOMM 对整个 instruction group 的 finalize/verify，不在 relay Channel 或 CommandBlock 可见性。

## 11. 通过后的下一步

三层全部通过后，才将该协议收敛成最终标准算子：

- CommandBlock 改为可版本化结构，增加 epoch 避免 ABA；
- `path_mask`、weights 和每路 bytes 改为图内动态控制输入；
- 增加错误码、超时、并发调用与 worker 生命周期管理；
- 再用 HCCN counter 验证每条显式 relay 的物理 footprint。
