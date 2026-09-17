# A5 CCU+URMA 显式 Relay 单向 Write 实验手册

## 1. 实验边界

本实验只修改源卡的 UBUS first-hop 路由。例如 `6 -> relay 2 -> 7`：

```text
Device 6: route[CNA7] = port_to(Device 2)   # FORCE 穿刺
Device 2: route[CNA7] = port_to(Device 7)   # 原生路由，不修改
Device 7: 不修改
```

Relay 卡不启动用户进程、不运行 AIV、不分配 relay HBM。代码不硬编码
`P67/P62/P27`；实际端口由 `ub_port.r_uent` 动态解析。遇到零个或多个候选端口时直接失败。

这是实验性内核穿刺。路由 override 是 source-entity/destination-CNA 级全局状态，测试期间不要运行相同源、目的的其他通信任务。

## 2. 拉取分支

```bash
cd /home/l00934901/sgl-kernel-npu
git fetch origin
git switch -C feature/a5-ccu-urma-path-puncture \
  --track origin/feature/a5-ccu-urma-path-puncture
git rev-parse --short HEAD
```

## 3. 在有卡容器中编译并安装单向 CCU Write `.so`

这一部分只安装用户态自定义通信库，不安装 DeepEP wheel，也不会生成或替换
`ubus.ko`。普通 route0 单向 Write 可以先用它独立验证；显式 FORCE relay 仍需第 4 节的
UBUS 内核补丁。

确认当前 CANN 与 HCCL 源码：

```bash
cd /home/l00934901/sgl-kernel-npu
source /usr/local/Ascend/cann-9.1.T560/set_env.sh
export HCCL_REPO=/home/l00934901/hccl

test -x "${HCCL_REPO}/build.sh"
test -d /usr/local/Ascend/cann-9.1.T560/opp/vendors/cust
echo "ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
```

编译并安装到正在使用的 CANN 9.1 目录：

```bash
cd /home/l00934901/sgl-kernel-npu

bash scripts/build_a5_ccu_urma_route_probe.sh \
  --install \
  --install-path /usr/local/Ascend/cann-9.1.T560
```

脚本会在 `/home/l00934901/hccl/build_out` 中选择时间最新的
`*ccu_urma_route_probe*.run` 包并执行安装。不要使用通配符直接执行多个 `.run` 文件，
也不要把该包安装到 `python/deep_ep/.../vendors/hwcomputing`。

确认头文件、动态库和导出符号均已安装：

```bash
A5_CANN_ROOT=/usr/local/Ascend/cann-9.1.T560

test -f "${A5_CANN_ROOT}/opp/vendors/cust/include/a5_ccu_urma_route_probe.h"
test -f "${A5_CANN_ROOT}/opp/vendors/cust/lib64/liba5_ccu_urma_route_probe.so"

grep -n HcclCcuUrmaOneWayWrite \
  "${A5_CANN_ROOT}/opp/vendors/cust/include/a5_ccu_urma_route_probe.h"
nm -D "${A5_CANN_ROOT}/opp/vendors/cust/lib64/liba5_ccu_urma_route_probe.so" \
  | grep HcclCcuUrmaOneWayWrite
ldd "${A5_CANN_ROOT}/opp/vendors/cust/lib64/liba5_ccu_urma_route_probe.so" \
  | grep 'not found' && echo 'ERROR: missing dependency' || true
```

运行时必须优先加载刚安装的库。DeepEP 的 vendor 环境可能把
`ASCEND_CUSTOM_OPP_PATH` 指向仓库内部目录，因此这个独立 probe 执行前要清除该重定向：

```bash
source /usr/local/Ascend/cann-9.1.T560/set_env.sh
unset ASCEND_CUSTOM_OPP_PATH
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.1.T560/opp/vendors/cust/lib64:${LD_LIBRARY_PATH:-}

ldd examples/a5_ccu_urma_route_probe/testcase/a5_ccu_urma_route_probe_test \
  | grep liba5_ccu_urma_route_probe || true
```

先运行不依赖 UBUS override 的 route0 单向 Write 冒烟测试，例如使用物理卡 4、5：

```bash
bash scripts/run_a5_ccu_urma_route_probe.sh \
  --devices 4,5 \
  --route-index 0 \
  --one-way-src-rank 0 \
  --bytes 4194304 \
  --warmup 1 \
  --iters 3
```

看到 `PASS` 才说明新 `.so` 已成功加载且单向 API 可用。这个 C++ probe 不依赖
`deep_ep_cpp`，因此不需要重装 DeepEP wheel。若报 `undefined symbol`，优先检查：

```bash
echo "${LD_LIBRARY_PATH}"
find /usr/local/Ascend -name liba5_ccu_urma_route_probe.so -type f -print
ldd examples/a5_ccu_urma_route_probe/testcase/a5_ccu_urma_route_probe_test
```

## 4. 准备带 FORCE override 的 UBUS 内核模块

补丁位置：

```text
kernel_patches/openeuler-6.6/0001-a5-ubus-route-override.patch
```

先只检查，不改源码：

```bash
bash scripts/apply_a5_ubus_route_override_patch.sh \
  --kernel-root /home/l00934901/openeuler-kernel
```

检查通过后再应用：

```bash
bash scripts/apply_a5_ubus_route_override_patch.sh \
  --kernel-root /home/l00934901/openeuler-kernel \
  --apply
```

必须使用与运行内核匹配的配置和构建环境。先核对：

```bash
uname -r
modinfo ubus | grep -E 'filename|srcversion|vermagic'
git -C /home/l00934901/openeuler-kernel status --short
```

当前有卡容器的已知状态是：

```text
运行内核:            6.6.0-159.4.13.167.oe2403sp4.aarch64
CONFIG_MODVERSIONS:   y
CONFIG_MODULE_SIG:    y
容器内匹配 build:    不存在
容器内匹配 source:   不存在
现有 Module.symvers: 6.6.0-159.4.9.163（与运行内核不匹配）
```

虽然 `lsmod` 能看到 `ubus` 已加载，但 `modinfo ubus` 报 `Module ubus not found`，说明
容器能观察宿主机模块状态，却没有挂载宿主机对应的
`/lib/modules/6.6.0-159.4.13.167.oe2403sp4.aarch64` 模块树。这不影响第 3 节安装和运行
用户态 `.so`，但意味着不能在该容器内直接生成或安装可加载的 patched `ubus.ko`。

尤其不能使用现有 `159.4.9.163` 的 `Module.symvers` 为 `159.4.13.167` 编译模块；在
`CONFIG_MODVERSIONS=y` 下，即使源码能编译，也很可能因符号 CRC 不匹配而拒绝加载。
必须另行取得与 `159.4.13.167` 完全匹配的源码、`.config`、完整
`Module.symvers` 和模块签名/安装条件，再构建并通过宿主机测试内核部署。

模块构建示例：

```bash
# Ubuntu/aarch64 构建容器缺少工具时先安装：
apt-get update
apt-get install -y flex bison bc libelf-dev

cd /home/l00934901/openeuler-kernel
cp /boot/config-$(uname -r) .config
make olddefconfig
make -j"$(nproc)" modules_prepare
make -j"$(nproc)" M=drivers/ub/ubus modules
```

只执行 `modules_prepare + M=...` 而没有完整内核的 `Module.symvers` 时，源码对象可以编译完成，最终 modpost 会因内核符号表缺失而失败。用于部署的 `ubus.ko` 必须在完整、版本匹配的内核构建目录中生成，不能忽略 modpost 错误。

不要在承载业务的机器上直接卸载 `ubus`：它被 `ubcore/udma/uburma` 等模块依赖。
推荐把匹配版本的模块安装到测试内核并重启；具体安装方式由机器的内核包、签名和启动策略决定。

启动新模块后确认属性存在：

```bash
find /sys/bus/ub/devices -name route_override -print
```

## 5. 建立物理卡到 UBUS entity 的映射

先导出完整邻接关系：

```bash
python3 scripts/a5_ub_route_ctl.py topology | \
  tee /home/l00934901/profiling/a5_ubus_topology.txt
```

结合 `npu-smi info`、设备父节点/BDF 和现场拓扑，为八张卡建立一次映射：

```csv
phy_id,uent_num
0,0x...
1,0x...
2,0x...
3,0x...
4,0x...
5,0x...
6,0x...
7,0x...
```

建议保存为：

```text
/home/l00934901/a5_ub_phy_entity_map.csv
```

该文件只映射物理卡到 entity；端口始终由当前 `direct_link/r_uent` 动态寻找，不写死。

## 6. Dry-run：解析 6 -> 2 -> 7

```bash
python3 scripts/a5_ub_route_ctl.py resolve \
  --phy-map /home/l00934901/a5_ub_phy_entity_map.csv \
  --src-phy 6 --relay-phy 2 --dst-phy 7
```

必须看到三个唯一实体和三条动态解析结果：

```text
resolved direct port
resolved relay first hop
verified relay native second hop
```

## 7. 自动 FORCE、单向 Write、自动 CLEAR

```bash
cd /home/l00934901/sgl-kernel-npu
source /usr/local/Ascend/cann-9.1.T560/set_env.sh
export HCCL_OP_EXPANSION_MODE=CCU_SCHED
export HCCL_BUFFSIZE=2300

bash scripts/run_a5_ccu_urma_force_relay.sh \
  --src-phy 6 \
  --relay-phy 2 \
  --dst-phy 7 \
  --phy-map /home/l00934901/a5_ub_phy_entity_map.csv \
  --bytes 4194304 \
  --warmup 10 \
  --iters 100 \
  --hccn-stat
```

可以换成任意三张互异物理卡，例如 `4 -> 2 -> 5`，只需修改三个 `--*-phy` 参数。

脚本用 `trap` 在成功、失败或 Ctrl-C 时执行 CLEAR。测试后仍应检查：

```bash
python3 scripts/a5_ub_route_ctl.py status \
  --phy-map /home/l00934901/a5_ub_phy_entity_map.csv \
  --src-phy 6 --relay-phy 2 --dst-phy 7
```

正常应为空。如不为空，手动恢复：

```bash
python3 scripts/a5_ub_route_ctl.py clear \
  --phy-map /home/l00934901/a5_ub_phy_entity_map.csv \
  --src-phy 6 --relay-phy 2 --dst-phy 7
```

## 8. 成功判据

- rank0（源卡）调用 `WriteNb`，rank1 只发布地址/token并等待；
- 目的卡数据校验 PASS；
- 源卡 direct port 不随 payload 线性增长；
- 源卡到 relay 的端口 TX 增长；
- relay 入端口 RX 与 relay 到目的端口 TX 增长；
- 目的卡来自 relay 的 RX 增长；
- relay 卡没有用户进程、AIV kernel 和 relay HBM buffer；
- CLEAR 后 direct-only 基线恢复。

注意：HCCN 计数必须比较 before/after，并用不同 payload/iterations 验证增量线性关系，不能只凭单次背景流量判断。
