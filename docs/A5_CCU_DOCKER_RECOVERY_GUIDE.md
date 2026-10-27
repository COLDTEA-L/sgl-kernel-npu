# A5 CCU 实验 Docker 恢复与依赖安装指南

适用分支：`feature/a5-multipath-physical-contention`。目标是恢复原生 HCCL
AllToAll、全 CCU baseline 多路径算子及 physical contention 测试环境。
这里的“重建 Docker”指恢复镜像、重建实验容器，不是重装 Docker 服务或宿主机驱动。

本指南不删除旧容器/镜像，不复位设备、不修改驱动/固件/ko/全局路由，也不安装实验版
`libhccl.so`。所有设备测试必须先确认管理员授权、设备健康、没有第三方任务。
命令分为**宿主机执行**与**容器内执行**，不要混用。需要 `set -e` 的操作放在括号
子 shell 内，失败不会退出当前 Docker 交互 shell。任一步失败都先解决，不继续整套粘贴。

**本次恢复的固定目标：** 宿主机归档
`/home/l00934901/sglang_yuanwen_old_transfer_20261016.tar`，新容器名和 hostname
都为 `sglang_yuanwen_old`。镜像名与容器名不是一回事；第 3 节根据实际 load 结果
确认镜像，不假定归档里的镜像仍叫 `sglang:a5_0623`。本机已确认文件后缀为 `.tar`，
不要改名，不需要 RAR 解压工具；是否为合格的 Docker 镜像归档仍由第 3.2 节检查。

本次从头恢复按顺序执行：**3.1 检查 → 3.2 检查 tar/load → 3.3 创建/进入容器 →
第 4 节检查并补依赖 → 第 5 节更新源码 → 6.2 编译安装 route → 6.3 编译安装
DeepEP → 第 7 节预检 → 第 8 节短测**。第 3 节在宿主机执行，进入容器后再做第 4～8 节。
已有匹配二进制时可以用 6.1 替代两次编译；不能只编译 DeepEP 而漏装 route。
若同名旧容器还在，先看第 2 节，不删除它来腾名字。

**当前继续位置：重新执行修正后的第 3.1 节。** 此前使用错误文件名的 3.1 在第一条
检查处退出，没有加载镜像或创建容器；检查显示 `HOST_CHECK_DONE` 后再执行 3.2。
3.2 成功显示 `IMAGE_LOAD_DONE` 后执行 3.3，不重复 load。

## 1. 恢复顺序与需要保留的东西

| 层次 | 内容 | 缺失时的处理 |
|---|---|---|
| 宿主机 | 正常的 NPU 驱动、设备节点、驱动拓扑 | 交管理员处理；容器内编译不能修复 |
| 基础镜像 | CPU 架构/系统、Python、原配 T560 CANN、HCOMM/HCCL/URMA | 恢复同一套已验证镜像或配套离线包，不能任意升级 |
| Python 环境 | 配套 torch、torch_npu、编译依赖 | 安装到实际运行实验的同一个解释器 |
| 源码 | sgl-kernel-npu 当前分支、匹配的 HCCL 构建源码与第三方依赖 | 从宿主机工作区或备份恢复 |
| 自定义 route 包 | `liba5_ccu_urma_route_probe.so` 及包内头文件 | 安装匹配的 `.run`，或重编 route 包 |
| 自定义 DeepEP | `deep_ep`、`deep_ep_cpp*.so`、包内 vendor OPP | 安装匹配 wheel，或重编 DeepEP |
| 实验结果 | 日志、JSON、profiling 原始目录 | 放在宿主机挂载目录，不能仅保存在容器可写层 |

**拉取 Git 源码不等于安装二进制。** 执行
`A5_CCU_MULTIPATH_PHYSICAL_CONTENTION_GUIDE.md` 第 2 节，不能修复旧镜像中的
`stale DeepEP extension`，也不会生成缺失的 route `.so`。

只有两份自定义组件需要我们自行构建：**route package + DeepEP wheel**。
HCCL 源码树在这里提供 custom-op 构建框架；不是让你编译替换系统 HCCL。
当前 contention 不需要另外构建 URMA trace 库、source-route provider 或 SGLang。

以下沿用实验路径，实际不同必须相应修改：

```text
宿主机镜像归档：   /home/l00934901/sglang_yuanwen_old_transfer_20261016.tar
新容器/hostname：  sglang_yuanwen_old
恢复用本地镜像名： sglang_yuanwen_old:recovery_20261016（第 3.2 节创建）
宿主机/容器工作区： /home/l00934901
算子仓库：         /home/l00934901/sgl-kernel-npu
HCCL 构建源码：    /home/l00934901/hccl
CANN：             /usr/local/Ascend/cann-9.1.T560
恢复材料：         /home/l00934901/a5_recovery
持久化结果：       /home/l00934901/profiling
```

## 2. 宿主机：先看旧容器是否还能直接使用

```bash
docker version
docker ps -a --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}'
docker image ls
npu-smi info
npu-smi info -m
```

若只是容器停止、容器仍存在，优先启动它，不需要重新 load/run 或重编：

```bash
# 替换成 docker ps -a 中确认属于自己的旧实验容器名。
docker start sglang_yuanwen_old
docker exec -it sglang_yuanwen_old /bin/bash
```

如果旧容器仍在、确需创建新容器，先在宿主机保存配置，特别是 Mounts、Devices、
DeviceRequests、网络、共享内存、Runtime 和镜像 ID：

```bash
(
set -e
umask 077
RECOVERY_OLD_CONTAINER=sglang_yuanwen_old
RECOVERY_RECORD_DIR=$(mktemp -d /home/l00934901/a5_docker_record.XXXXXX)
docker inspect "${RECOVERY_OLD_CONTAINER}" > "${RECOVERY_RECORD_DIR}/container.inspect.json"
docker inspect --format '{{.Image}}' "${RECOVERY_OLD_CONTAINER}"
echo "容器配置保存在：${RECOVERY_RECORD_DIR}"
)
```

`inspect` 可能包含凭证、代理等敏感环境变量，只本地保存，不整份发到公共对话。
如果 Docker 服务本身不可用，交管理员恢复，本文不操作服务、不清理 Docker 数据目录。

## 3. 宿主机：加载镜像、创建新容器

### 3.1 确认 tar 归档、工具、磁盘和同名容器

以下在**存放 tar 的那台机器的宿主机**执行，只检查，不 load、不占用 NPU。
开始和成功都有提示，缺文件/工具会打印具体原因，不再静默退出：

```bash
(
set -e
trap 'echo "[ERROR] 3.1 检查失败：${BASH_COMMAND}；先处理错误，不执行 3.2。" >&2' ERR
RECOVERY_ARCHIVE=/home/l00934901/sglang_yuanwen_old_transfer_20261016.tar
echo "[START] 3.1 宿主机检查，归档：${RECOVERY_ARCHIVE}"
test -f "${RECOVERY_ARCHIVE}" || {
    echo "[ERROR] 找不到归档文件：${RECOVERY_ARCHIVE}；请核对所在机器、路径和文件名。" >&2
    exit 1
}
test -r "${RECOVERY_ARCHIVE}" || {
    echo "[ERROR] 没有归档读取权限：${RECOVERY_ARCHIVE}" >&2
    exit 1
}
for RECOVERY_TOOL in file python3 tar docker; do
    command -v "${RECOVERY_TOOL}" || {
        echo "[ERROR] 缺少工具：${RECOVERY_TOOL}" >&2
        exit 1
    }
done
docker version
file "${RECOVERY_ARCHIVE}"
ls -lh "${RECOVERY_ARCHIVE}"
df -h /home/l00934901
docker info --format 'DockerRootDir={{.DockerRootDir}}'
docker ps -a --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}'
if docker container inspect sglang_yuanwen_old >/dev/null 2>&1; then
    echo '[ERROR] 同名容器 sglang_yuanwen_old 已存在；先按第 2 节检查复用，不重复 load/run。' >&2
    exit 1
fi
echo 'HOST_CHECK_DONE：检查命令已通过；确认下述磁盘空间要求后执行第 3.2 节。'
)
```

另查上面 `DockerRootDir` 所在磁盘的可用空间；3.1 只打印目录，不自动判断可用空间
是否足够。Docker load 需要展开镜像层，归档大小不等于最终占用。**不需要先把这份
tar 解压到工作区**，也不额外复制一份镜像 tar。空间不足先协调，本文不清理其他镜像。

**若 `sglang_yuanwen_old` 已存在，停止新建，先按第 2 节确认是否直接 start/exec。**
不要删除旧容器来绕过重名。只使用自己可信的备份，不运行归档里的未知脚本。
若宿主机缺工具，由管理员从允许的软件源/离线包补齐；不要在共享宿主机照抄
第 4 节容器内的依赖安装命令。本次 tar 恢复不需要安装解压 RAR 的工具。

### 3.2 检查这份 tar 是 Docker 镜像归档，然后直接 load

本次直接使用 `/home/l00934901/sglang_yuanwen_old_transfer_20261016.tar`。Docker load
读取镜像 tar，不需要先 `tar -xf`；后缀为 tar 仍不等于内容一定是 Docker 镜像，所以
先检查归档布局。检查失败就停止，不自动改用其他解压/导入方式。
[Docker load 说明](https://docs.docker.com/reference/cli/docker/image/load/)

下面只在全新的记录目录保存 tar 文件清单和 load 日志，不解压 tar 到宿主机工作区，
不覆盖原归档；Docker 自己会在其数据目录展开镜像层。可整段执行：

```bash
(
set -eo pipefail
trap 'echo "[ERROR] 3.2 镜像检查/加载失败：${BASH_COMMAND}；保留日志，先处理错误。" >&2' ERR
umask 077
RECOVERY_ARCHIVE=/home/l00934901/sglang_yuanwen_old_transfer_20261016.tar
RECOVERY_IMAGE_ALIAS=sglang_yuanwen_old:recovery_20261016
echo "[START] 3.2 检查并加载：${RECOVERY_ARCHIVE}"
test -f "${RECOVERY_ARCHIVE}" && test -r "${RECOVERY_ARCHIVE}" || {
    echo "[ERROR] 归档不存在或不可读：${RECOVERY_ARCHIVE}" >&2
    exit 1
}
# 防止重复执行时覆盖上一次恢复用的镜像标签；已有该标签时先核对并复用。
if docker image inspect "${RECOVERY_IMAGE_ALIAS}" >/dev/null 2>&1; then
    echo "恢复镜像标签已存在，先核对；不要重复 load 或覆盖：${RECOVERY_IMAGE_ALIAS}" >&2
    exit 1
fi
mkdir -p /home/l00934901/a5_recovery
RECOVERY_LOAD_RECORD=$(mktemp -d /home/l00934901/a5_recovery/sglang_yuanwen_old_load.XXXXXX)
echo "恢复记录目录：${RECOVERY_LOAD_RECORD}"

# 防止把 docker export 的 rootfs 或普通源码压缩包当成 docker save 镜像。
echo '[CHECK] 正在检查 tar 目录；大归档可能需要等待。'
tar -tf "${RECOVERY_ARCHIVE}" >"${RECOVERY_LOAD_RECORD}/image-tar.list" \
    2>"${RECOVERY_LOAD_RECORD}/tar-check.log" || {
    echo "[ERROR] tar 读取失败，检查是否损坏/传输完整；日志：${RECOVERY_LOAD_RECORD}/tar-check.log" >&2
    exit 1
}
python3 - "${RECOVERY_LOAD_RECORD}/image-tar.list" <<'PY'
from pathlib import Path, PurePosixPath
import sys
names = {str(PurePosixPath(line)) for line in Path(sys.argv[1]).read_text().splitlines()}
if 'manifest.json' not in names and not {'oci-layout', 'index.json'} <= names:
    raise SystemExit('不是可识别的 Docker/OCI 镜像归档；停止，不自动改用 docker import。')
print('IMAGE_ARCHIVE_LAYOUT_OK')
PY

# load 可能重设归档内原有标签的指向。保留原启动命令使用的旧标签，避免丢失引用。
docker image ls --no-trunc >"${RECOVERY_LOAD_RECORD}/images-before-load.txt"
if docker image inspect sglang:a5_0623 >/dev/null 2>&1; then
    RECOVERY_OLD_IMAGE=$(docker image inspect --format '{{.Id}}' sglang:a5_0623)
    docker image tag "${RECOVERY_OLD_IMAGE}" \
        "sglang_yuanwen_old_before_load:$(basename "${RECOVERY_LOAD_RECORD}")"
fi
echo "[LOAD] 准备加载：${RECOVERY_ARCHIVE}"
docker load --input "${RECOVERY_ARCHIVE}" 2>&1 | tee "${RECOVERY_LOAD_RECORD}/docker-load.log"
python3 - "${RECOVERY_LOAD_RECORD}/docker-load.log" <<'PY' >"${RECOVERY_LOAD_RECORD}/loaded-refs.txt"
from pathlib import Path
import sys
refs = set()
for line in Path(sys.argv[1]).read_text().splitlines():
    for prefix in ('Loaded image: ', 'Loaded image ID: '):
        if line.startswith(prefix):
            refs.add(line[len(prefix):].strip())
if not refs:
    raise SystemExit('未从 load 日志识别镜像，请人工查看日志，不使用猜测的旧 tag。')
print('\n'.join(sorted(refs)))
PY
while IFS= read -r RECOVERY_REF; do
    docker image inspect --format '{{.Id}}' "${RECOVERY_REF}"
done <"${RECOVERY_LOAD_RECORD}/loaded-refs.txt" | sort -u >"${RECOVERY_LOAD_RECORD}/loaded-ids.txt"
test "$(wc -l <"${RECOVERY_LOAD_RECORD}/loaded-ids.txt")" -eq 1 || {
    echo '归档加载了多个不同镜像；先按日志选择正确镜像，不自动建容器。' >&2
    exit 1
}
RECOVERY_IMAGE_ID=$(<"${RECOVERY_LOAD_RECORD}/loaded-ids.txt")
docker image tag "${RECOVERY_IMAGE_ID}" "${RECOVERY_IMAGE_ALIAS}"
docker image inspect --format 'ID={{.Id}} OS={{.Os}} ARCH={{.Architecture}}' "${RECOVERY_IMAGE_ALIAS}"
echo "第 3.3 节使用镜像：${RECOVERY_IMAGE_ALIAS}"
echo "保留恢复记录：${RECOVERY_LOAD_RECORD}"
echo 'IMAGE_LOAD_DONE：镜像已加载并设置恢复别名；接下来执行第 3.3 节。'
)
```

注意：

- 大镜像的 tar 列表检查和 load 都可能耗时较长，这是 Host 磁盘工作，
  **尚未下发通信算子**，不要因等待同时再开一份恢复任务。
- 若文件清单只有另一份镜像 tar、源码或普通数据，没有顶层镜像 manifest，检查会停止。
  先看记录目录里的 `image-tar.list`，确认是否拿错了归档；不递归解压/批量 load，
  不选择 `layer.tar` 当整份镜像，也不把整个工作区重新打包后盲目 load。
- 仅有 `bin/`、`etc/`、`usr/` 等 rootfs 而没有镜像 manifest，可能是 `docker export`
  备份。它与 `docker save` 不同，不能按原镜像完整恢复配置；应优先取得 save 归档，
  不自动用 `docker import`，以免丢失原 ENV/USER/CMD 等镜像配置。
- 这里新增的本地镜像别名固定为 `sglang_yuanwen_old:recovery_20261016`；它不要求
  原归档的 tag 是 `sglang:a5_0623`。若归档带有其他正在使用的同名 tag，load 前应先
  与使用者协调并保留那些 tag 的原 ID，不能只凭容器名推断镜像归属。
- 镜像恢复不包含宿主机绑定目录，也不重建旧容器的挂载/权限；旧镜像没有最新
  DeepEP/route 包时，需要后面两步编译安装，不能直接认为 load 完就恢复好了。

### 3.3 按原挂载方式创建 `sglang_yuanwen_old`

下面参考提供的旧命令：保留 host 网络、特权模式、hixlep/route/hccn、工作区、NPU
日志、dcmi、driver 和 firmware 的挂载，**容器名和 hostname 改为 `sglang_yuanwen_old`**。
为防止误写宿主机，配置、工具、driver/firmware 改为只读挂载；工作区、`/mnt` 和日志
仍可写。`--mount` 在源路径缺失时会报错，不像 `-v` 自动创建同名空目录。

**仅在管理员已允许原特权运行方式、镜像可信时执行。** `--privileged` 提供很大的宿主机
权限，并不限制到 2/3/4/5 卡；只读挂载也不是特权容器的安全隔离保证。不得在容器内
安装/修改宿主机驱动、固件、路由或复位公共设备。本文沿用原配权限不是要求把已有的
最小权限环境升级成特权；设备隔离策略不同的新机器应先由管理员确认。
[Docker 权限说明](https://docs.docker.com/engine/containers/run/#runtime-privilege-and-linux-capabilities)

```bash
(
set -e
RECOVERY_CONTAINER=sglang_yuanwen_old
RECOVERY_IMAGE=sglang_yuanwen_old:recovery_20261016
docker image inspect "${RECOVERY_IMAGE}" >/dev/null
if docker container inspect "${RECOVERY_CONTAINER}" >/dev/null 2>&1; then
    echo "同名容器已存在，停止；先检查并按第 2 节复用：${RECOVERY_CONTAINER}" >&2
    exit 1
fi
RECOVERY_IMAGE_PLATFORM=$(docker image inspect --format '{{.Os}}/{{.Architecture}}' "${RECOVERY_IMAGE}")
case "$(uname -m):${RECOVERY_IMAGE_PLATFORM}" in
    aarch64:linux/arm64|x86_64:linux/amd64) ;;
    *) echo "镜像平台与本机不匹配：$(uname -m) / ${RECOVERY_IMAGE_PLATFORM}" >&2; exit 1 ;;
esac
for RECOVERY_DIR in /etc/hixlep /mnt /home/l00934901 /var/log/npu \
    /usr/local/dcmi /usr/local/Ascend/driver /usr/local/Ascend/firmware; do
    test -d "${RECOVERY_DIR}" || { echo "缺少挂载目录：${RECOVERY_DIR}" >&2; exit 1; }
done
for RECOVERY_FILE in /lib/route.conf /etc/hccn.conf /etc/ascend_install.info \
    /usr/local/Ascend/driver/topo/950/atlas_950_1.json; do
    test -f "${RECOVERY_FILE}" || { echo "缺少配置/拓扑：${RECOVERY_FILE}" >&2; exit 1; }
done
# 两个 npu-smi 路径只挂载本机真实存在且可执行的；至少需要一个。
RECOVERY_NPU_SMI_MOUNTS=()
for RECOVERY_BIN in /usr/local/bin/npu-smi /usr/local/sbin/npu-smi; do
    if test -x "${RECOVERY_BIN}"; then
        RECOVERY_NPU_SMI_MOUNTS+=(--mount "type=bind,src=${RECOVERY_BIN},dst=${RECOVERY_BIN},readonly")
    fi
done
test "${#RECOVERY_NPU_SMI_MOUNTS[@]}" -gt 0 || { echo '没有找到可挂载的 npu-smi' >&2; exit 1; }
for RECOVERY_NODE in /dev/ummu /dev/uburma; do
    test -e "${RECOVERY_NODE}" || { echo "缺少 A5 设备节点：${RECOVERY_NODE}" >&2; exit 1; }
done
mkdir -p /home/l00934901/a5_recovery /home/l00934901/profiling

docker run --pull=never \
    --name "${RECOVERY_CONTAINER}" \
    --hostname "${RECOVERY_CONTAINER}" \
    --entrypoint /bin/bash \
    --network host \
    --privileged \
    --ipc private --shm-size 16g --ulimit memlock=-1:-1 \
    --mount type=bind,src=/etc/hixlep,dst=/etc/hixlep,readonly \
    --mount type=bind,src=/mnt,dst=/mnt \
    --mount type=bind,src=/home/l00934901,dst=/home/l00934901 \
    --mount type=bind,src=/var/log/npu,dst=/usr/slog \
    --mount type=bind,src=/lib/route.conf,dst=/lib/route.conf,readonly \
    --mount type=bind,src=/etc/hccn.conf,dst=/etc/hccn.conf,readonly \
    --mount type=bind,src=/etc/ascend_install.info,dst=/etc/ascend_install.info,readonly \
    "${RECOVERY_NPU_SMI_MOUNTS[@]}" \
    --mount type=bind,src=/usr/local/dcmi,dst=/usr/local/dcmi,readonly \
    --mount type=bind,src=/usr/local/Ascend/driver,dst=/usr/local/Ascend/driver,readonly \
    --mount type=bind,src=/usr/local/Ascend/firmware,dst=/usr/local/Ascend/firmware,readonly \
    --workdir /home/l00934901 \
    -itd "${RECOVERY_IMAGE}" --noprofile --norc

docker inspect --format '{{range .Mounts}}{{println .Source "->" .Destination "RW=" .RW}}{{end}}' \
    "${RECOVERY_CONTAINER}"
echo '已创建。下一步在宿主机执行 docker exec，进入后做第 4 节。'
)
```

```bash
docker exec -it sglang_yuanwen_old /bin/bash
```

说明：

- 创建时的 `--noprofile --norc` 防止直接执行镜像里旧 shell 启动脚本；进入新 shell 后
  仍按第 4 节显式选择 Python、source T560，不依赖旧环境变量碰巧正确。
- 缺少上述配置文件/驱动目录时先联系管理员核对新机器的真实路径，不创建空文件凑数，
  不把其他机器的 route/hccn 配置直接覆盖过来。只读配置若与原安装器不兼容，应先确认
  具体写入目标，不为绕过错误把共享 driver/firmware 改成可写。
- 工作区用**同一次父目录挂载**，使两个源码仓可进行硬链接。route 构建脚本的 `cp -al`
  不能跨文件系统/独立挂载边界；两个仓本身也必须在同一文件系统，不要额外分别挂载。
- `/home/l00934901` 和 `/mnt` 的挂载会遮住镜像里同路径的旧文件。若原 HCCL 源码、
  wheel 或工具只在镜像内这些位置，load 不会把它们合并到宿主机；先恢复各自工作区备份。
  不在目标目录已有文件时盲目 docker cp 覆盖。
- 不挂载整个 `/usr/local/Ascend`，CANN 由镜像提供；仅 driver/firmware 来自宿主机。
  950 还需要 UB/UMMU 设备节点与匹配用户态库，特权模式不能修复缺失的宿主机组件。
  [Ascend 官方 950 容器示例](https://github.com/Ascend/cann-container-image/blob/main/OVERVIEW.md)
- 不加 `--rm`，退出交互 shell 不删除容器；不停止或重命名原来的 `sglang_yuanwen`。
  私有 IPC 的共享内存上限设为 16GiB（不是启动就占用 16GiB），不共享他人的 IPC。
- 能看到所有设备不等于有权使用所有设备。后续 `--p-devices`/`--q-devices` 和 relay
  只能填写管理员授权且健康的卡；`ASCEND_RT_VISIBLE_DEVICES` 也不是独占资源保证。

## 4. 容器内：检查基础环境，不立即占卡

### 4.1 设备、目录和 CANN

```bash
id
uname -m
cat /etc/os-release
npu-smi info
npu-smi info -m
ls -ld /dev/ummu /dev/uburma
ls -l /usr/local/Ascend/driver/topo/950/atlas_950_1.json
```

宿主机正常而容器不正常，优先查挂载、权限和运行库，不要重装驱动。
有卡显示 Critical/Alarm 时，不用该卡或经过它的 relay 路径做测试，先交管理员定位。
安装到容器自己的 `/usr/local/Ascend/cann-9.1.T560` 需要相应写权限；若镜像默认不是
root 用户，由管理员确认后在宿主机用 `docker exec -it -u 0 sglang_yuanwen_old /bin/bash`
进入，再激活原 Python 环境。不要改宿主机目录权限来绕过安装错误。
以下是加载库/查看版本，不创建 tensor 或通信域：

```bash
(
set -e
test -f /usr/local/Ascend/cann-9.1.T560/set_env.sh
source /usr/local/Ascend/cann-9.1.T560/set_env.sh
echo "ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
test "$(readlink -f "${ASCEND_HOME_PATH}")" = "$(readlink -f /usr/local/Ascend/cann-9.1.T560)"
test -f "${ASCEND_HOME_PATH}/lib64/libascendcl.so"
test -f "${ASCEND_HOME_PATH}/lib64/libhccl.so"
test -f "${ASCEND_HOME_PATH}/lib64/libhcomm.so"
test -f "${ASCEND_HOME_PATH}/include/hcomm/hcomm_res.h"
find "${ASCEND_HOME_PATH}" -name ASCConfig.cmake -print
find "${ASCEND_HOME_PATH}" -name ascend_toolkit_install.info -print
python3 - <<'PY'
import ctypes
for name in ('libascendcl.so', 'libhccl.so', 'libhcomm.so', 'liburma.so'):
    ctypes.CDLL(name, mode=ctypes.RTLD_GLOBAL)
    print('LOAD_OK', name)
PY
)
```

`ASCConfig.cmake` 是 DeepEP AscendC 构建依赖；有运行库不代表 SDK/开发文件完整。
部分版本库使用不同的 SONAME，`liburma.so` 检查失败时还应检查实际安装文件和 HCOMM
依赖，不能创建不明软链接强行通过。`urma_admin show` 可补充确认 URMA 设备是否可见。

**缺 CANN/HCOMM/URMA 怎么办：** 优先恢复相同 T560 镜像/离线软件包及原来的依赖配置，
不是安装 Python 包。普通公开版 CANN 9.1/9.2 不等于本实验的 T560。
如需从安装包重装，先确认原配 Toolkit、950 ops/HCOMM、URMA 用户态包、CPU 架构及
驱动配套，不缺哪个都装一遍。确认是原配 `.run` 后，可按包的 `--help` 核对安装参数：

```bash
(
set -e
RECOVERY_TOOLKIT_RUN=/home/l00934901/a5_recovery/packages/REPLACE_WITH_T560_TOOLKIT.run
test -f "${RECOVERY_TOOLKIT_RUN}"
bash "${RECOVERY_TOOLKIT_RUN}" --help
)
```

确认该包支持且接受目标布局后，Toolkit 的常规安装形式是：

```bash
# 替换为上一段核对过的文件；仅在新容器内的非共享 CANN 安装位置执行。
bash /home/l00934901/a5_recovery/packages/REPLACE_WITH_T560_TOOLKIT.run \
    --install --install-path=/usr/local/Ascend/cann-9.1.T560
```

安装器可能受已有安装登记影响，或生成额外子目录。安装后重新执行 4.1，实际
`set_env.sh`、SDK、运行库必须指向同一棵完整安装树；不要只改一个环境变量。
其他配套包按其原配说明安装；没有安装包/版本记录时停止，向管理员取得材料。
**不在此猜测 URMA RPM 名称、不重装内核驱动。**
[CANN Toolkit/ops 安装说明](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/softwareinst/instg/instg_0054.html)

### 4.2 选择原来的 Python 环境，检查依赖

优先激活镜像中原来能跑实验的环境。若它确实是以前的 Conda 环境，可使用：

```bash
# 仅当此环境确实存在时执行；没有 Conda 不需要额外安装 Conda。
source /opt/conda/bin/activate cam_py311_pt28
```

环境名不是版本证据。后续构建和测试都用同一个 `python3`；不要一会儿系统 Python、
一会儿 Conda。每次新开 shell 都要恢复 CANN 和 Python 环境。

```bash
source /usr/local/Ascend/cann-9.1.T560/set_env.sh
command -v python3
python3 --version
python3 -m pip --version
pip3 --version
python3 - <<'PY'
import importlib
import importlib.metadata as metadata
import pathlib
import sys
import sysconfig

print('PYTHON', sys.executable)
print('Python.h', pathlib.Path(sysconfig.get_path('include')) / 'Python.h')
failed = []
# 先导入 torch，再导入 torch_npu；不调用 set_device/is_available/通信初始化。
for module_name, dist_name in (
    ('torch', 'torch'), ('torch_npu', 'torch-npu'), ('numpy', 'numpy'),
    ('packaging', 'packaging'), ('yaml', 'PyYAML'), ('pybind11', 'pybind11'),
):
    try:
        module = importlib.import_module(module_name)
        print('IMPORT_OK', dist_name, metadata.version(dist_name), module.__file__)
    except Exception as error:
        failed.append(module_name)
        print('IMPORT_FAIL', module_name, type(error).__name__, str(error))
for dist_name in ('pip', 'setuptools', 'wheel'):
    try:
        print('PACKAGE_OK', dist_name, metadata.version(dist_name))
    except metadata.PackageNotFoundError:
        failed.append(dist_name)
        print('PACKAGE_MISSING', dist_name)
if not (pathlib.Path(sysconfig.get_path('include')) / 'Python.h').is_file():
    failed.append('Python development headers')
if failed:
    raise SystemExit('需要补齐/修复：' + ', '.join(failed))
PY
```

区别 `ModuleNotFoundError` 与 `undefined symbol`/动态库缺失：前者可能缺包，后者
通常要先核对软件配套和库路径，不能靠反复安装一个“最新版”解决。
构建脚本内部会使用 `pip3`，其 Python 版本也必须与 `python3 -m pip` 一致。

### 4.3 缺 Python 包：从原配离线 wheelhouse 安装

推荐把通过验证的 wheel 及其传递依赖保存在
`/home/l00934901/a5_recovery/wheelhouse`，并保留版本记录。
不要直接复制另一 Python 版本/架构的 `site-packages`，也不要把开发机器上的 `.so`
拷过来假装安装成功。

如果缺的是 torch/torch_npu，选择**原来验证过的准确文件**一起恢复；文件名要匹配
当前 Python ABI（如 cp311）、CPU 架构和 T560 软件栈。以下占位文件名必须替换：

```bash
(
set -e
RECOVERY_WHEELHOUSE=/home/l00934901/a5_recovery/wheelhouse
RECOVERY_TORCH_WHEEL="${RECOVERY_WHEELHOUSE}/REPLACE_WITH_MATCHING_torch.whl"
RECOVERY_TORCH_NPU_WHEEL="${RECOVERY_WHEELHOUSE}/REPLACE_WITH_MATCHING_torch_npu.whl"
test -f "${RECOVERY_TORCH_WHEEL}"
test -f "${RECOVERY_TORCH_NPU_WHEEL}"
python3 -m pip install --no-index --find-links "${RECOVERY_WHEELHOUSE}" \
    "${RECOVERY_TORCH_WHEEL}" "${RECOVERY_TORCH_NPU_WHEEL}"
)
```

该安装会解析依赖，wheelhouse 必须包含它们；如缺依赖应补离线材料，不临时切公共源。
若已安装的框架版本不匹配，应先确认准确目标文件，再对上面的安装加
`--force-reinstall`；不要无依据先卸掉整个环境。

若 torch/torch_npu 配套且可导入，只缺编译辅助包，可从审核过的 wheelhouse 补齐：

```bash
python3 -m pip install --no-index \
  --find-links /home/l00934901/a5_recovery/wheelhouse \
  pybind11 setuptools wheel==0.45.1 numpy packaging PyYAML
```

除 wheel 外，具体版本由恢复材料中的已验证包集合约束；不要把任意最新版堆进目录。
`wheel==0.45.1` 与本仓当前构建脚本的缺包回退一致。最好预先安装，避免构建时联网。
**不使用 `pip install deep_ep` 从公共源找同名包**；我们的接口来自本仓自定义 wheel。
这几个独立实验不要求安装完整 SGLang、torchair 或 pytest。

### 4.4 缺系统编译工具

先检查，不修改宿主机：

```bash
for RECOVERY_TOOL in bash gcc g++ make cmake git nm readelf file patch tar gzip unzip zip timeout; do
    command -v "${RECOVERY_TOOL}" || true
done
cmake --version
gcc --version
```

当前顶层项目要求 CMake **至少 3.20**。缺工具时，只在新容器内、使用单位允许的软件源
安装对应发行版软件包，优先恢复原来验证过的工具链：

```bash
# Ubuntu/Debian 容器，按需执行；离线环境使用管理员提供的镜像源/离线包。
apt-get update
apt-get install -y build-essential cmake git binutils file patch tar gzip unzip zip \
  util-linux procps python3-dev
```

```bash
# openEuler/RHEL 系容器，按需执行；不要与上面的 apt 命令混用。
dnf install -y gcc gcc-c++ make cmake git binutils file patch tar gzip unzip zip \
  util-linux procps-ng python3-devel
```

开发头文件必须对应实际 Python；系统 `python3-dev/devel` 不一定能补齐另一个 Conda
解释器的 `Python.h`。OS 源中的 CMake 若过旧，使用管理员认可的对应版本离线包。

## 5. 容器内：恢复并确认源码

两个仓库通常来自宿主机挂载，不随镜像恢复回旧版本：

```bash
cd /home/l00934901/sgl-kernel-npu
git branch --show-current
git rev-parse HEAD
git status --short
test -x /home/l00934901/hccl/build.sh
test -f examples/a5_ccu_urma_route_probe/CMakeLists.txt
test -f scripts/build_a5_ccu_urma_route_probe.sh
test -f tests/python/deepep/a5_ccu_test_runtime.py
test -f docs/topology/a5_hccn_device_topology_raw.txt
```

若算子仓库已存在，先检查并更新当前实验分支。下面只更新干净的 tracked 工作区，
发现本地修改或快进失败就停止；不覆盖、不清理用户修改。不因重装镜像改切旧 overlap 分支：

```bash
(
set -e
cd /home/l00934901/sgl-kernel-npu
git status --short
git diff --quiet
git diff --cached --quiet
RECOVERY_BRANCH=feature/a5-multipath-physical-contention
git fetch origin "${RECOVERY_BRANCH}"
git switch "${RECOVERY_BRANCH}"
git merge --ff-only "origin/${RECOVERY_BRANCH}"
test -f scripts/build_a5_ccu_urma_route_probe.sh
test -f tests/python/deepep/a5_ccu_test_runtime.py
)
```

如果挂载目录里根本没有算子仓库，**不执行上面的已有仓库更新块**，改用首次克隆：

```bash
(
set -e
test ! -e /home/l00934901/sgl-kernel-npu
git clone --branch feature/a5-multipath-physical-contention --single-branch \
  https://github.com/COLDTEA-L/sgl-kernel-npu.git \
  /home/l00934901/sgl-kernel-npu
)
```

HCCL 源码及 `cann-cmake` 等第三方依赖应恢复**原来配套的源码树/缓存**；不要用未知
最新版代替。没有该树但有已验证 route `.run` 时，可先走 6.1，不需要为了安装重新构建。
如果两者都没有，应先取得原配材料，而不是在共享机器上编译安装新的系统 HCCL。

## 6. 容器内：恢复两份自定义二进制

### 6.1 快速路径：已有匹配安装包

停止自己旧库对应的实验后安装。若 CANN/Python 是共享挂载，安装会影响其他使用者，
必须先协调；不能因为在 Docker 内就假设文件更新完全隔离。

```bash
(
set -e
cd /home/l00934901/sgl-kernel-npu
source /usr/local/Ascend/cann-9.1.T560/set_env.sh
unset LD_PRELOAD
RECOVERY_ROUTE_RUN=/home/l00934901/a5_recovery/packages/REPLACE_WITH_MATCHING_ROUTE.run
RECOVERY_DEEPEP_WHEEL=/home/l00934901/a5_recovery/wheelhouse/REPLACE_WITH_MATCHING_deep_ep.whl
test -f "${RECOVERY_ROUTE_RUN}"
test -f "${RECOVERY_DEEPEP_WHEEL}"
env -u ASCEND_CUSTOM_OPP_PATH -u ASCEND_OPP_PATH \
    bash "${RECOVERY_ROUTE_RUN}" --quiet --install \
    --install-path=/usr/local/Ascend/cann-9.1.T560
python3 -m pip install --force-reinstall --no-cache-dir --no-deps "${RECOVERY_DEEPEP_WHEEL}"
)
```

前提是来源、版本与框架/CANN 配套均明确。安装后必须做第 7 节真实加载检查；安装器
返回成功不等于实际加载的 `.so` 已更新。缺包或不配套时走 6.2/6.3。

### 6.2 从头编译安装第一份组件：route 包

本次需要从源码恢复时先执行本节，再执行 6.3。先确认第 4 节的 Python 环境已激活、
第 5 节两个仓库都在、没有自己的旧实验仍占用这些安装文件。构建日志保存在挂载工作区，
失败时从日志找具体缺失依赖，不继续装另一份不配套的旧产物：

```bash
(
set -e
cd /home/l00934901/sgl-kernel-npu
test "$(git branch --show-current)" = feature/a5-multipath-physical-contention
source /usr/local/Ascend/cann-9.1.T560/set_env.sh
unset LD_PRELOAD A5_CCU_PREPARED_ALLOW_LAZY_STREAM A5_URMA_TP_TRACE_PREFIX
set -o pipefail
# route builder 会优先 source 这两个别名中的第一个；提前阻止其切到其他 CANN。
for RECOVERY_CANN_ALIAS in /usr/local/Ascend/cann/set_env.sh \
    /usr/local/Ascend/ascend-toolkit/set_env.sh; do
    if test -f "${RECOVERY_CANN_ALIAS}"; then
        RECOVERY_ALIAS_HOME=$(bash -ec 'source "$1" >/dev/null; readlink -f "$ASCEND_HOME_PATH"' \
            _ "${RECOVERY_CANN_ALIAS}")
        test "${RECOVERY_ALIAS_HOME}" = "$(readlink -f /usr/local/Ascend/cann-9.1.T560)" || {
            echo "构建入口 ${RECOVERY_CANN_ALIAS} 指向其他 CANN：${RECOVERY_ALIAS_HOME}" >&2
            exit 1
        }
        break
    fi
done
test -x /home/l00934901/hccl/build.sh
mkdir -p /home/l00934901/a5_recovery/build_logs
RECOVERY_ROUTE_LOG=$(mktemp /home/l00934901/a5_recovery/build_logs/route.XXXXXX.log)
echo "route 编译日志：${RECOVERY_ROUTE_LOG}"
HCCL_REPO=/home/l00934901/hccl \
bash scripts/build_a5_ccu_urma_route_probe.sh \
    --install --install-path /usr/local/Ascend/cann-9.1.T560 2>&1 | tee "${RECOVERY_ROUTE_LOG}"
test -f /usr/local/Ascend/cann-9.1.T560/opp/vendors/cust/lib64/liba5_ccu_urma_route_probe.so
echo 'ROUTE_BUILD_AND_INSTALL_DONE；仍需第 7 节检查实际 ABI'
)
```

注意编译日志的 `CANN home`，必须与实际运行的 T560 相同。构建脚本会尝试 source
`/usr/local/Ascend/cann/set_env.sh` 或旧 toolkit 入口；如果这些别名指向另一版本，应先
纠正容器内的环境配置，不能用两个版本混合编译。不要修改宿主机共享软链接。
编译脚本会使用 HCCL 源码目录及缓存，但不会要求安装 patched `libhccl.so`。

### 6.3 从头编译安装第二份组件：DeepEP wheel

用同一个已激活的 Python 环境执行，**仅构建 A5 DeepEP 所需的多路径单算子**，不运行
默认全模块构建，也不需要再额外编译 SGLang 或替换系统 HCCL：

```bash
(
set -e
cd /home/l00934901/sgl-kernel-npu
test "$(git branch --show-current)" = feature/a5-multipath-physical-contention
source /usr/local/Ascend/cann-9.1.T560/set_env.sh
unset LD_PRELOAD
set -o pipefail
mkdir -p /home/l00934901/a5_recovery/build_logs
RECOVERY_DEEPEP_LOG=$(mktemp /home/l00934901/a5_recovery/build_logs/deepep.XXXXXX.log)
RECOVERY_BUILD_STARTED_NS=$(python3 -c 'import time; print(time.time_ns())')
echo "DeepEP 编译日志：${RECOVERY_DEEPEP_LOG}"
DEEPEP_SINGLE_OP=explicit_multipath_all2all_ccu \
bash build.sh -a deepep Ascend950 2>&1 | tee "${RECOVERY_DEEPEP_LOG}"
RECOVERY_BUILT_WHEEL=$(python3 - "${RECOVERY_BUILD_STARTED_NS}" <<'PY'
from pathlib import Path
import sys
wheels = [p for p in Path('output').glob('deep_ep-*.whl')
          if p.stat().st_mtime_ns >= int(sys.argv[1])]
if not wheels:
    raise SystemExit('没有找到本次新生成的 DeepEP wheel；不安装遗留旧包')
print(max(wheels, key=lambda path: path.stat().st_mtime).resolve())
PY
)
test -f "${RECOVERY_BUILT_WHEEL}"
python3 -m pip install --force-reinstall --no-cache-dir --no-deps "${RECOVERY_BUILT_WHEEL}"
python3 -m pip show deep-ep
echo 'DEEPEP_BUILD_AND_INSTALL_DONE；继续第 7 节实际加载预检'
)
```

必须先确认构建成功，再选择本次产物，不能在失败后继续安装 output 中遗留的旧包。
脚本会重建仓库的生成目录（包括 `build`），先保存需要保留的历史产物，不清理源码。
默认编译并行度会使用较多 Host CPU/内存，公共机器上需协调；编译和 ABI 预检不运行
四卡通信实验。已经有匹配 wheel 时，无需为了每次恢复再次编译。

## 7. 容器内：真实加载、ABI 和 profiling 检查

### 7.1 两份二进制必须同时通过

```bash
(
set -e
cd /home/l00934901/sgl-kernel-npu
source /usr/local/Ascend/cann-9.1.T560/set_env.sh
unset LD_PRELOAD
export A5_CCU_ROUTE_PROBE_LIB=/usr/local/Ascend/cann-9.1.T560/opp/vendors/cust/lib64/liba5_ccu_urma_route_probe.so
test -f "${A5_CCU_ROUTE_PROBE_LIB}"
ldd "${A5_CCU_ROUTE_PROBE_LIB}"
python3 tests/python/deepep/test_a5_ccu_urma_multiroute_all2all.py \
    --implementation prepared --self-copy-mode ccu \
    --channel-sync-mode baseline --runtime-check
)
```

必须看到 `MULTIPATH_RUNTIME_VERIFIED`，并满足：

| 库 | 检查项 | 最低要求 |
|---|---|---:|
| DeepEP | `A5DeepEpExplicitMultipathAttrAbiVersion` | 12 |
| route | `A5CcuUrmaPreparedPlanAbiVersion` | 16 |
| route | `A5CcuLocalRemoteAbiVersion` | 2 |
| route | `A5CcuChannelSyncAbiVersion` | 1 |

这些是本项目自定义功能 ABI，不是 torch 的 C++ ABI。不能编辑返回值或删除检查绕过。
关注 `CASE_TEST_EXTENSION`、`CASE_TEST_RUNTIME`、`CASE_DEEP_EP_ABI`、
`CASE_PREPARED_PLAN_ABI`、`CASE_ROUTE_FEATURE_ABI` 中的**实际路径与版本**。
`--runtime-check` 不执行 `set_device` 或创建通信域；它只证明加载与功能版本通过，
不能替代实际设备正确性检查。

```bash
cd /home/l00934901/sgl-kernel-npu
python3 tests/python/deepep/test_a5_ccu_path_contention_cpu.py
python3 tests/python/deepep/test_a5_ccu_path_contention_analyzer_cpu.py
python3 tests/python/deepep/test_a5_ccu_test_runtime_cpu.py
```

### 7.2 确认 profiler 组件存在

```bash
source /usr/local/Ascend/cann-9.1.T560/set_env.sh
python3 - <<'PY'
import torch_npu
import importlib.util
print('torch_npu:', torch_npu.__version__)
assert hasattr(torch_npu.profiler, 'profile')
assert hasattr(torch_npu.profiler, 'tensorboard_trace_handler')
for name in ('msprof', 'msprof_analyze'):
    spec = importlib.util.find_spec(name)
    print(name, spec.origin if spec is not None else '未安装为 Python 模块；以当前 CANN 配套方式检查')
print('PROFILER_API_PRESENT；尚未证明设备采集/导出成功')
PY
command -v msprof || true
```

正式 runner 使用 torch_npu profiler；CLI `msprof` 是否在 PATH 不是唯一判据。
API 存在也不证明 collector/导出依赖完整。profiling 缺组件时，恢复与该 torch_npu/CANN
匹配的 profiler 安装包，不盲装公共源最新分析器；最终以第 8 节的小规模采集成功为准。

## 8. 容器内：短测通过后再跑正式测试

从这里开始才实际使用设备。先查 `npu-smi info` 并协调独占窗口；下面只示例 2、3、4、5，
4 卡若仍异常必须改用管理员授权的健康卡及对应 relay，不能照抄。
结果放在挂载的 `/home/l00934901/profiling` 内。

### 8.1 不采 profiling 的 16KiB 短测

```bash
cd /home/l00934901/sgl-kernel-npu
bash scripts/run_a5_ccu_multipath_contention.sh \
  --cann-root /usr/local/Ascend/cann-9.1.T560 \
  --experiment contention \
  --p-devices 2,3 --p-relay 4 --q-devices 4,5 --q-relay 2 \
  --sizes 16384 --warmup 2 --iters 5 --repeats 1 \
  --no-mindstudio-profile \
  --output-root /home/l00934901/profiling/recovery_smoke
```

确认所有 rank 数据校验通过、进程正常退出、报告完整。短测迭代少不等于启动快：建链、
CCU 注册与清理仍有固定成本，不要因等待而重复启动多份。

### 8.2 单尺寸小规模 profiling 验证

```bash
cd /home/l00934901/sgl-kernel-npu
bash scripts/run_a5_ccu_multipath_contention.sh \
  --cann-root /usr/local/Ascend/cann-9.1.T560 \
  --experiment contention \
  --p-devices 2,3 --p-relay 4 --q-devices 4,5 --q-relay 2 \
  --sizes 16384 --warmup 2 --iters 5 --repeats 1 \
  --output-root /home/l00934901/profiling/recovery_profile_smoke
```

确认 baseline P、baseline Q、concurrent P/Q 的采集结果与 timeline 均可读，才回到
正式 `--warmup 100 --iters 100`。环境曾在多尺寸连续执行时失败，恢复验收和正式测试
都优先**每次一个数据量**。预检通过、短测成功和 profiler 成功是三个不同验收点。

## 9. 容器恢复后的常见错误

| 现象 | 优先检查 | 处理 |
|---|---|---|
| `stale DeepEP extension` / attr ABI 太低 | 实际 extension 路径、Python 解释器 | 安装本仓匹配 wheel；必要时 6.3 重编，不能只拉 Git |
| route `.so` 不存在 / prepared ABI 太低 | `A5_CCU_ROUTE_PROBE_LIB` 指向谁、CANN 路径 | 6.1/6.2 恢复 route 包，DeepEP wheel 不包含它 |
| 安装了仍加载旧 DeepEP | 系统 site-packages、user site、源码目录多份安装 | 按预检打印的实际路径定位，使用同一 Python 的 pip；不手删全部 site-packages |
| `No module named deep_ep` | 当前 Python 是否装 wheel | 用 `python3 -m pip` 安装；正式入口使用共享 bootstrap，不靠随意加 PYTHONPATH |
| `undefined symbol` / `GLIBCXX` / `libhcomm.so` 等加载失败 | CANN/torch/torch_npu/编译器/架构与库路径 | 恢复配套依赖，在当前环境重编；不复制不明 `.so`、不改 ABI 数字 |
| `Invalid cross-device link` | 两个源码仓是否在同一文件系统和挂载边界 | 恢复单次父目录挂载；不要在报错后手工绕过构建脚本预检 |
| 找不到 `cann-cmake` / `prepare.cmake` | HCCL 源码的 third_party 或缓存 | 恢复原配依赖；同机有完整缓存时构建脚本会尝试复用 |
| 缺 `ASCConfig.cmake`、AscendC 开发文件、`Python.h` | SDK/开发头文件是否齐全 | 补原配开发组件，不用 runtime-only 镜像冒充构建环境 |
| 缺 driver topology JSON | 宿主机是否存在、driver 挂载是否完整 | 恢复挂载；宿主机缺失由管理员处理，不伪造拓扑 |
| 能看到 NPU，但 URMA 建链失败 | UB/UMMU 节点权限、用户态库、原配置 | 比对旧容器配置与配套包；不动全局路由或安装内核模块 |
| `Health=Critical` / 算子执行报硬件错误 | 具体错误码与管理员检查 | 停止涉及异常卡的实验，不自动 reset，不让 relay 绕经它 |
| profiling API 存在但采集/导出失败 | CANN profiler 与 torch_npu 依赖 | 恢复对应配套组件；只做第 8 节小采集，不直接扫全部尺寸 |

## 10. 通过验收后：制作下一次恢复材料

### 10.1 容器内记录版本，备份安装产物

只在完全通过第 7/8 节后执行。下面目录在宿主机挂载工作区内，不随容器删除而消失：

```bash
(
set -e
umask 077
cd /home/l00934901/sgl-kernel-npu
source /usr/local/Ascend/cann-9.1.T560/set_env.sh
RECOVERY_SNAPSHOT=$(mktemp -d /home/l00934901/a5_recovery/validated.XXXXXX)
python3 --version > "${RECOVERY_SNAPSHOT}/python-version.txt" 2>&1
python3 -m pip freeze --all > "${RECOVERY_SNAPSHOT}/pip-freeze.txt"
git rev-parse HEAD > "${RECOVERY_SNAPSHOT}/sgl-commit.txt"
git status --short > "${RECOVERY_SNAPSHOT}/sgl-worktree-status.txt"
git -C /home/l00934901/hccl rev-parse HEAD > "${RECOVERY_SNAPSHOT}/hccl-commit.txt"
uname -m > "${RECOVERY_SNAPSHOT}/architecture.txt"
cp /etc/os-release "${RECOVERY_SNAPSHOT}/os-release.txt"
find /usr/local/Ascend/cann-9.1.T560 -name ascend_toolkit_install.info -print \
  > "${RECOVERY_SNAPSHOT}/cann-version-file-locations.txt"
sha256sum /usr/local/Ascend/cann-9.1.T560/opp/vendors/cust/lib64/liba5_ccu_urma_route_probe.so \
  > "${RECOVERY_SNAPSHOT}/route-library.sha256"
echo "版本记录：${RECOVERY_SNAPSHOT}"
)
```

还需将**对应的** DeepEP wheel、route `.run`、框架及依赖 wheel、T560 原配安装包放入
恢复材料目录，保存各自 SHA256 与版本/架构记录。route `.run` 通常在
`/home/l00934901/hccl/build_out`，DeepEP wheel 在仓库 `output`。
不要只备份 `.so`，更不要只备份旧镜像。

`pip freeze` 是版本线索，不是可直接在离线机重放的完整材料：它可能含本地路径、URL，
也不包含 wheel 内容或系统依赖。应由允许联网的环境按确认过的版本收集完整离线
wheelhouse，不能在实验机临时找公共源最新版。检查本地路径/URL 是否含敏感信息，
不要把记录未经处理地公开。CANN 版本文件的内容、HCCL third_party/缓存、实际 Python
路径、包安装路径以及容器启动参数也要保存；只记录 Git commit 不足以复现脏工作区。

### 10.2 宿主机保存已验证镜像

先让自己该容器内的任务全部正常结束。`docker commit` 默认会短暂暂停容器，不能
对正在执行实验或他人共用的容器直接操作。[Docker commit 说明](https://docs.docker.com/reference/cli/docker/container/commit/)

```bash
(
set -e
umask 077
# 只能对本次自己的、已经完全验收且没有运行任务的容器做快照。
RECOVERY_CONTAINER=sglang_yuanwen_old
RECOVERY_IMAGE="a5-ccu-validated:$(date +%Y%m%d_%H%M%S)"
RECOVERY_HOST_SNAPSHOT=$(mktemp -d /home/l00934901/a5_recovery/image_snapshot.XXXXXX)
docker inspect "${RECOVERY_CONTAINER}" > "${RECOVERY_HOST_SNAPSHOT}/container.inspect.json"
docker commit "${RECOVERY_CONTAINER}" "${RECOVERY_IMAGE}"
docker save --output "${RECOVERY_HOST_SNAPSHOT}/image.tar" "${RECOVERY_IMAGE}"
sha256sum "${RECOVERY_HOST_SNAPSHOT}/image.tar" > "${RECOVERY_HOST_SNAPSHOT}/image.tar.sha256"
printf '%s\n' "${RECOVERY_IMAGE}" > "${RECOVERY_HOST_SNAPSHOT}/image-tag.txt"
echo "镜像备份：${RECOVERY_HOST_SNAPSHOT}"
)
```

**commit 不包含挂载卷/绑定目录的数据。** 镜像归档也不会替你重建设备权限和挂载。
工作区、实验结果和恢复材料仍需单独保存到可靠存储；本机磁盘故障时，同盘备份也会丢。
共享挂载的 CANN/Python 不会被 commit 收进去；镜像可能包含凭证和内部软件，仅按单位
规定本地保存和传递。[绑定挂载说明](https://docs.docker.com/engine/storage/bind-mounts/)

## 11. 最终验收清单

- [ ] 原配镜像/CPU 架构、CANN T560、torch、torch_npu 及 URMA 配套一致。
- [ ] 容器与宿主机设备映射明确，UB/UMMU/driver/topology 可见；异常卡未被使用。
- [ ] 两源码仓及结果在持久化工作区，HCCL 构建依赖齐全且可做硬链接。
- [ ] `python3`、pip、构建与运行使用同一环境，缺包/动态库错误已解决。
- [ ] route 包与 DeepEP wheel 都安装，真实预检达到 12 / 16 / 2 / 1 最低 ABI 要求。
- [ ] 三项 CPU 回归通过，不占卡的检查没有被误当作设备正确性证明。
- [ ] 单尺寸短测、进程退出、单尺寸 profiling 分别通过。
- [ ] 记录镜像/启动参数/版本/产物及校验值，再进入正式 100 次实验。

下一次已有合格镜像时，优先顺序是：**start/exec 或 load/run → 基础检查 → ABI 预检
→ 单尺寸短测**；缺哪层补哪层，不必每次重编所有组件。
