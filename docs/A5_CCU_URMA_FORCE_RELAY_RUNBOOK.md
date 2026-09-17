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

### 4.1 必须在宿主机确认模块来源

以下命令要退出 `sglang_yuanwen` 容器后，在有卡服务器宿主机执行。容器内的
`modinfo ubus` 失败不能说明宿主机没有模块，只说明容器没有挂载宿主机模块目录。

```bash
A5_KERNEL_RELEASE=$(uname -r)

uname -r
modinfo -n ubus
modinfo -F vermagic ubus
modinfo -F srcversion ubus
modinfo -F signer ubus
rpm -qf "$(modinfo -n ubus)" || true
rpm -q --qf '%{NAME} %{VERSION}-%{RELEASE} %{SOURCERPM}\n' \
  kernel kernel-core kernel-modules kernel-devel 2>/dev/null || true

ls -l "/lib/modules/${A5_KERNEL_RELEASE}/build" \
      "/lib/modules/${A5_KERNEL_RELEASE}/source" 2>/dev/null || true
find "/usr/src/kernels/${A5_KERNEL_RELEASE}" \
  -maxdepth 1 -name Module.symvers -type f -ls 2>/dev/null

cat /sys/module/module/parameters/sig_enforce 2>/dev/null || true
cat /proc/sys/kernel/tainted
```

预期宿主机上的 `modinfo -n ubus` 会返回类似：

```text
/lib/modules/6.6.0-159.4.13.167.oe2403sp4.aarch64/kernel/drivers/ub/ubus/ubus.ko.xz
```

如果宿主机同样找不到它，则先检查模块是否来自 initramfs 或被删除：

```bash
lsinitrd "/boot/initramfs-${A5_KERNEL_RELEASE}.img" | grep '/ubus\.ko' || true
find "/lib/modules/${A5_KERNEL_RELEASE}" -type f \
  \( -name 'ubus.ko' -o -name 'ubus.ko.xz' -o -name 'ubus.ko.zst' \) -print
```

### 4.2 获取精确匹配的源码和符号表

有卡机不必预先存在 `openeuler-kernel` Git 仓库，但构建机器必须具备：

- `6.6.0-159.4.13.167.oe2403sp4.aarch64` 对应的完整内核源码；
- 该二进制内核构建生成的 `.config` 和 `Module.symvers`；
- 对应的 aarch64 编译器及模块签名条件。

先在宿主机查询软件源是否提供精确版本，不要直接安装较旧版本：

```bash
A5_KERNEL_RELEASE=$(uname -r)

dnf list --showduplicates kernel-devel 2>/dev/null | grep "${A5_KERNEL_RELEASE%.*}" || true
dnf repoquery --installed --qf '%{name}-%{evr}.%{arch} %{sourcerpm}' \
  kernel kernel-core kernel-modules 2>/dev/null || true
dnf repoquery --available --qf '%{name}-%{evr}.%{arch}' kernel-devel 2>/dev/null \
  | grep '159\.4\.13\.167' || true
```

只有查询到完全一致的包时，才可以安装：

```bash
dnf install "kernel-devel-$(uname -r)"
```

`kernel-devel` 通常能提供 prepared build tree 和 `Module.symvers`，但未必包含需要修改的
`drivers/ub/ubus/*.c`。仍需根据上一步输出的 `SOURCERPM` 获取完全对应的 source RPM，
或者从内核维护方取得对应 Git commit。若内部版本 `159.4.13.167` 不在公开软件源中，
必须向系统镜像/内核提供方索取 source RPM 与 matching `kernel-devel`；不能用
`159.4.9.163` 替代。

### 4.3 在独立构建目录生成 patched `ubus.ko`

构建可以放在无卡服务器或独立 aarch64 构建容器，不要求放在有卡机上。下面的
`A5_KERNEL_SRC` 必须指向精确匹配的完整源码树：

```bash
A5_KERNEL_SRC=/path/to/exact-6.6.0-159.4.13.167-source
A5_KERNEL_BUILD=/path/to/exact-6.6.0-159.4.13.167-build
SGL_KERNEL_NPU=/home/liuyuanwen/sgl-kernel-npu-explicit-relay

# Git 工作树使用仓库辅助脚本：
bash "${SGL_KERNEL_NPU}/scripts/apply_a5_ubus_route_override_patch.sh" \
  --kernel-root "${A5_KERNEL_SRC}"
bash "${SGL_KERNEL_NPU}/scripts/apply_a5_ubus_route_override_patch.sh" \
  --kernel-root "${A5_KERNEL_SRC}" --apply

make -C "${A5_KERNEL_SRC}" O="${A5_KERNEL_BUILD}" \
  M=drivers/ub/ubus -j"$(nproc)" modules

modinfo "${A5_KERNEL_BUILD}/drivers/ub/ubus/ubus.ko" | \
  grep -E 'filename|vermagic|srcversion|signer'
```

如果源码来自解包后的 source RPM 而不是 Git 工作树，辅助脚本中的 `git apply` 不适用，
改为在源码根目录执行：

```bash
cd "${A5_KERNEL_SRC}"
patch -p1 --dry-run < \
  "${SGL_KERNEL_NPU}/kernel_patches/openeuler-6.6/0001-a5-ubus-route-override.patch"
patch -p1 < \
  "${SGL_KERNEL_NPU}/kernel_patches/openeuler-6.6/0001-a5-ubus-route-override.patch"
```

产物的 `vermagic` 必须以
`6.6.0-159.4.13.167.oe2403sp4.aarch64` 开头，且构建不能存在 modpost
`undefined symbol`/CRC 错误。只有 `modules_prepare` 而没有原始完整
`Module.symvers` 不够。

### 4.4 在宿主机以可回滚方式部署

这是宿主机内核变更，必须安排维护窗口并确保具有串口/BMC/GRUB 回退能力。`ubus`
引用计数为 17，不能在业务运行期间直接 `rmmod ubus`；应安装模块、重建 initramfs 后重启。

先把构建产物复制到宿主机临时目录，例如：

```text
/root/a5-ubus-test/ubus.ko
```

随后在宿主机执行。以下步骤会改变宿主机启动模块，执行前先确认目标路径和维护窗口：

```bash
A5_KERNEL_RELEASE=$(uname -r)
A5_NEW_UBUS=/root/a5-ubus-test/ubus.ko
A5_MODULE_DIR="/lib/modules/${A5_KERNEL_RELEASE}/updates/a5-explicit-relay"
A5_BACKUP_DIR="/root/a5-ubus-backup-${A5_KERNEL_RELEASE}"

test -f "${A5_NEW_UBUS}"
test "$(modinfo -F vermagic "${A5_NEW_UBUS}" | awk '{print $1}')" = "${A5_KERNEL_RELEASE}"

mkdir -p "${A5_BACKUP_DIR}"
cp -a "$(modinfo -n ubus)" "${A5_BACKUP_DIR}/"
cp -a "/boot/initramfs-${A5_KERNEL_RELEASE}.img" "${A5_BACKUP_DIR}/"

mkdir -p "${A5_MODULE_DIR}"
install -m 0644 "${A5_NEW_UBUS}" "${A5_MODULE_DIR}/ubus.ko"
depmod -a "${A5_KERNEL_RELEASE}"

modinfo -n ubus
modinfo -F vermagic ubus
```

此时 `modinfo -n ubus` 必须指向 `updates/a5-explicit-relay/ubus.ko`。如果没有指向新模块，
停止，不要重建 initramfs或重启，先检查 `modules.dep`/`modules.alias` 的模块优先级。

确认路径正确后重建该内核的 initramfs：

```bash
dracut --force "/boot/initramfs-${A5_KERNEL_RELEASE}.img" "${A5_KERNEL_RELEASE}"
lsinitrd "/boot/initramfs-${A5_KERNEL_RELEASE}.img" | grep '/ubus\.ko'
```

若内核强制模块签名，必须先用该机器信任的证书签名；`CONFIG_MODULE_SIG=y` 本身不等于
强制签名，最终以 `sig_enforce`、Secure Boot/lockdown 和实际加载策略为准。签名不满足时
不要重启进入实验环境。

重启后在宿主机验证：

```bash
uname -r
modinfo -n ubus
dmesg -T | grep -Ei 'ubus|module verification|invalid module|unknown symbol' | tail -n 100
find /sys/bus/ub/devices -name route_override -print
```

只有 `route_override` 属性出现，才能进入第 5 节。若启动或模块加载失败，通过 GRUB/BMC
回退原内核/initramfs，或从救援环境删除 `updates/a5-explicit-relay` 并恢复备份后执行
`depmod`、`dracut`。

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
