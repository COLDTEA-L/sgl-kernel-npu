# A5 CCU 显式多路径 AllToAll 新机器部署指南

适用代码分支：`feature/a5-ccu-copy-comm-overlap`。编写日期：2026年10月19日。

目标是在另一台 Ascend 950 机器上，重新部署两卡或四卡显式 direct/relay AllToAll。
不能把旧服务器的 CANN 路径、容器、二进制、EID 和拓扑直接搬过来就认为环境等价。
部署顺序是：**硬件与容器可用 → 软件版本配套 → 旧对象式 CCU 接口可用 → 编译两份自有库
→ direct-only → 显式 relay → 可选入图验证 → 性能测量**。任一层失败，先解决该层。

本指南以当前 `--self-copy-mode ccu --channel-sync-mode baseline` 为稳定起点。
`packed`、`packed-ready` 在已有反馈中没有收益且波动较大，不作为新环境验收配置。

## 1 哪些安装是必要的

| 层次 | 新环境需要什么 | 本算子是否修改它 |
|---|---|---|
| 宿主机 | 与950硬件配套的驱动、固件、正常工作的UB通信基础设施 | 不修改，不加载自定义ko |
| 容器 | 正确透传NPU、通信设备、驱动运行库和必要拓扑文件 | 只使用管理员批准的透传配置 |
| CANN | 对应硬件的Toolkit、算子包及HCCL/HCOMM开发和运行接口 | 不替换系统libhccl/libhcomm |
| PyTorch | 相互配套的Python、torch、torch_npu | 选择并固定一套版本 |
| route自定义包 | `liba5_ccu_urma_route_probe.so`及头文件 | 需要从本分支编译安装 |
| DeepEP wheel | 包含peer-plan和多类型桥接接口的`deep_ep_cpp` | 首次部署需要编译安装 |
| HCCL源码仓 | custom-op构建脚手架和对应构建依赖 | 仅用于构建route包，不安装实验版HCCL |
| 新机器拓扑 | 本机driver topology JSON和本机EID清单 | 只读采集，不改硬件路由表 |

当前实现不是靠AIV直驱URMA的专用NIT模块运行，也不要求安装我们修改过的`ubus.ko`。
不需要为本算子安装openEuler完整内核源码、kernel-devel或Module.symvers。
不要求另行拉取HIXL或从源码编译UMDK；URMA运行库/provider使用目标平台配套版本，
不要用通用开源包覆盖厂商驱动配套的通信库。
如果平台原本的UB驱动、固件、网络不正常，应由管理员按对应硬件版本处理，不能用算子包修复。

## 2 旧环境中不能照抄的内容

| 内容 | 旧环境示例或历史信息 | 新环境必须核对的内容 |
|---|---|---|
| 硬件和主机架构 | Ascend 950、aarch64、一机8卡 | 确认确实是950；其他芯片或不同互连不在已验证范围内 |
| 驱动和内核 | openEuler及某个6.6内核 | 按硬件/CANN配套，不要求复制旧内核编号 |
| CANN | 有卡机`cann-9.1.T560`；开发机曾用`9.1.0-beta.1`无卡编译 | 版本名相近不代表CCU接口相同；先检查头文件、符号和链接 |
| Python与框架 | 旧Python3.11环境及其torch/torch_npu | 按官方配套选择；不能安装任意最新版组合 |
| 文件路径 | `/home/l00934901/...`、`/usr/local/Ascend/cann-9.1.T560` | 全部改为新机器实际路径 |
| wheel和SO | 旧aarch64、特定Python/CANN编译产物 | 优先本地重编；不能跨Python ABI、架构或未知SDK直接复制 |
| 计算卡和relay | 旧例子计算6、7，relay0～5 | 确认存在、允许使用且拓扑满足要求，不假设这些ID仍有效 |
| 可用卡集合N | 旧例子N=8 | 传实际可借用集合；不是盲目传0～7，更不是进程可见卡数 |
| EID与UDMAC | 仓库`docs/topology/a5_hccn_device_topology_raw.txt` | 该文件是旧机器快照，relay测试必须替换为新机器采集结果 |
| driver topology | `atlas_950_1.json` | 文件名、JSON格式、physical ID映射和实际接线都要核对 |
| direct-route编号 | 旧实验candidate0是direct | 编号是当前CommLink枚举序号，不保证新环境0一定是direct |
| profiling和性能 | 旧机器几十微秒的结果 | 重新测；CPU、NUMA、频率、共享负载及relay使用情况都会不同 |

先从管理员处取得硬件对应的驱动/固件、CANN、Python和框架版本组合。
官方[PyTorch配套表](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.md)
要求TorchNPU与PyTorch、CANN、Python及驱动固件配套；该表不能替代本项目的旧CCU接口检查。

## 3 准备宿主机和独立容器

### 3.1 宿主机只读检查

在宿主机执行，保留输出。`npu-smi`不可用时先找管理员确认工具路径与权限；
不能仅凭“找不到命令”认定卡坏了。也不能仅凭`npu-smi`成功认定URMA/CCU可用。

```bash
uname -a
uname -m
cat /etc/os-release
command -v npu-smi || true
npu-smi info
lsmod | grep -E 'ubus|ubcore|uburma|udma|ubagg' || true
ls -l /dev/davinci* /dev/uburma* /dev/ummu* /dev/hisi_hdc* 2>/dev/null
```

模块名、设备节点和透传方式可能随驱动改变，以上是核对线索，不是要求所有名字必须出现。
容器内`modinfo ubus`找不到文件，但`lsmod`显示模块已加载，可能只是容器没有宿主机的
`/lib/modules`；这不是要求重新安装或卸载UBUS的理由。

### 3.2 容器和挂载

优先使用管理员提供、已与宿主驱动匹配的950开发镜像。若从零创建，按
[官方950容器部署说明](https://github.com/Ascend/cann-container-image/blob/main/OVERVIEW.md#run-a-cann-container-on-950-series-aarch64-products)
核对NPU、管理和UB通信设备透传，不照抄A2/A3的`docker run`模板。
镜像版本还必须通过第5节的旧CCU接口检查，不能因为镜像更新就认为一定兼容。

容器外检查管理员创建的实际配置：

```bash
docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}'
```

确定容器名后，使用实际名称：

```bash
read -r -p '输入新环境容器名: ' A5_CONTAINER
docker inspect "${A5_CONTAINER}" --format '{{json .HostConfig.Devices}}'
docker inspect "${A5_CONTAINER}" --format '{{json .Mounts}}'
docker exec -it "${A5_CONTAINER}" bash
```

重点确认：

- NPU及HCOMM/URMA所需设备可访问；不能只有计算设备而缺少通信设备。
- driver运行库、拓扑目录可读；读挂载足够的目录不要申请写权限。
- 编译仓和输出目录可写，磁盘、共享内存及权限满足框架运行要求。
- CANN和Python安装目录是否为宿主机bind mount或共享卷。**Docker内安装不等于不影响宿主机**。
- 不要默认使用`--privileged`、挂载整个`/dev`，或为relay暴露不必要的计算设备。具体透传由管理员确认。

本指南不自动安装/升级宿主驱动或固件，不reload模块，不reset设备，不修改全局UB路由。
共享机器若不允许更新公共CANN目录，应先建立获准的私有镜像/安装前缀，而不是覆盖公共文件。

## 4 配置目录和Python环境

以下步骤在**目标容器**执行。所有后续代码块默认沿用本节变量；新开shell后要重新配置。
不要把含`exit`或`set -e`的构建片段直接source进交互shell，避免失败后退出容器会话。

先列出本机CANN环境脚本，再输入真正使用的版本根目录，不填旧服务器路径：

```bash
find /usr/local/Ascend -maxdepth 4 -name set_env.sh -type f -print 2>/dev/null
read -r -p '输入CANN根目录，目录下应有set_env.sh: ' CANN_ROOT
export CANN_ROOT
test -f "${CANN_ROOT}/set_env.sh" && source "${CANN_ROOT}/set_env.sh"

export DEPLOY_ROOT="${HOME}/a5-multipath-deploy"
export SGL_REPO="${DEPLOY_ROOT}/sgl-kernel-npu"
export HCCL_REPO="${DEPLOY_ROOT}/hccl"
export OUTPUT_ROOT="${DEPLOY_ROOT}/profiling"
export EVIDENCE_ROOT="${DEPLOY_ROOT}/environment"
mkdir -p "${DEPLOY_ROOT}" "${OUTPUT_ROOT}" "${EVIDENCE_ROOT}"

printf 'CANN_ROOT=%s\nASCEND_HOME_PATH=%s\nASCEND_TOOLKIT_HOME=%s\n' \
  "${CANN_ROOT}" "${ASCEND_HOME_PATH:-}" "${ASCEND_TOOLKIT_HOME:-}"
```

如果`test`失败，不继续；选择包含正确`set_env.sh`的根目录。只有一套预期CANN生效后再编译。

Python可复用管理员提供的专用conda环境；不要把旧conda环境名当新机器的固定要求。
如果需要从零建环境，可在选定Python下创建独立venv：

```bash
python3 --version
python3 -m venv "${DEPLOY_ROOT}/venv"
source "${DEPLOY_ROOT}/venv/bin/activate"
python3 -m pip --version
command -v python3
command -v pip3
```

已经激活合适conda环境时跳过venv，不再嵌套创建另一层环境。
由第2节配套表确定torch/torch_npu的**具体安装包**。如果基底镜像已包含正确组合，不重新安装。
离线部署可用准备好的wheel文件；以下通过交互选择文件，避免误复制占位路径：

```bash
read -r -p '输入已核对版本和架构的torch wheel绝对路径: ' TORCH_WHL
read -r -p '输入与它配套的torch_npu wheel绝对路径: ' TORCH_NPU_WHL
test -f "${TORCH_WHL}" && test -f "${TORCH_NPU_WHL}" && \
  python3 -m pip install "${TORCH_WHL}" "${TORCH_NPU_WHL}"
```

pip仍可能解析其他依赖；内网需要把依赖也放到批准的wheelhouse，使用
`--no-index --find-links /实际wheelhouse目录`，不能认为两个wheel包含全部依赖。
不要为解决安装冲突而随意升级CANN或torch，也不要下载另一个环境的`deep_ep_cpp.so`强行覆盖。

构建工具至少检查Git、C/C++编译器、Make、CMake、binutils及Python开发头文件。
本仓顶层要求CMake>=3.20，DeepEP host代码使用C++17；还需与编译器配套的Python开发包。
Python构建依赖包含pybind11、setuptools、wheel；预装wheel0.45.1可避免当前脚本在线下载该依赖。
由管理员或批准的软件源安装，不在公共宿主机盲目执行全量升级。
新venv没有这些构建依赖时，使用选定开发镜像/平台的锁版本requirements文件安装后再继续；
不要把下面的`pip show`误当安装命令。本仓尚未为每个可能的新SDK提供一份通用依赖锁文件。

```bash
git --version
g++ --version
cmake --version
make --version
nm --version
python3 -m pip show torch torch-npu pybind11 setuptools wheel
python3 - <<'PY'
import sys, sysconfig, platform
from pathlib import Path
import torch, torch_npu
print('python:', sys.executable, sys.version)
print('arch:', platform.machine())
print('torch:', torch.__version__, torch.__file__)
print('torch_npu:', torch_npu.__version__, torch_npu.__file__)
print('torch C++11 ABI:', torch.compiled_with_cxx11_abi())
header = Path(sysconfig.get_path('include')) / 'Python.h'
print('Python.h:', header, header.is_file())
PY
```

当前route库使用旧HCOMM对象式C++接口，其CMake显式使用`_GLIBCXX_USE_CXX11_ABI=0`；
DeepEP则跟随torch的ABI配置。这是不同链接边界，不能通过改一个宏就认定兼容。
SDK换版本、出现`std::__cxx11`或C++未定义符号时，需要核对真实头文件和库的ABI。

## 5 检查CANN是否支持当前CCU实现

### 5.1 清理旧实验选择并核对开发文件

只清理本项目旧实验相关变量，不粗暴清空整个`LD_LIBRARY_PATH`，后者可能破坏驱动依赖：

```bash
unset LD_PRELOAD A5_URMA_TP_TRACE_PREFIX A5_CCU_PREPARED_ALLOW_LAZY_STREAM
unset A5_CCU_SELF_COPY_MODE A5_CCU_CHANNEL_SYNC_MODE
unset A5_CCU_ROUTE_INDEX A5_CCU_ROUTE_INDICES A5_CCU_PATH_UIDS A5_CCU_PATH_WEIGHTS
unset A5_CCU_SYNTHETIC_SRC_EID A5_CCU_SYNTHETIC_DST_EID A5_CCU_SYNTHETIC_ROUTE_MANIFEST
unset HCCL_RUNTIME

find "${CANN_ROOT}" -name ccu_kernel.h -o -name hccl_ccu_res.h \
  -o -name libhcomm.so -o -name libhccl.so -o -name ASCConfig.cmake
printf '%s\n' "${LD_LIBRARY_PATH:-}"
```

应能找到`hcomm/ccu/ccu_kernel.h`、`hcomm/ccu/hccl_ccu_res.h`等开发头文件及实际运行库。
只有runtime而没有开发包，不能直接编译。头文件可能在`pkg_inc`或架构子目录中。
不要链接`stub`目录的库后宣称设备运行接口可用。

### 5.2 旧对象接口的运行库检查

下面不初始化卡，不创建通信域。它加载指定CANN目录中的实际HCOMM，而不是随便查找系统同名库：

```bash
python3 - <<'PY'
import ctypes, os, platform
from pathlib import Path
root = Path(os.environ['CANN_ROOT']).resolve()
arch = platform.machine()
paths = [root / 'lib64/libhcomm.so', root / f'{arch}-linux/lib64/libhcomm.so']
path = next((p.resolve() for p in paths if p.is_file()), None)
if path is None:
    raise SystemExit('libhcomm.so不存在，请核对CANN_ROOT和安装包: ' + str(paths))
lib = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
required = ['HcclChannelAcquire', 'HcclThreadAcquireWithStream',
    'HcclCcuKernelRegister', 'HcclCcuKernelRegisterFinish', 'HcclCcuKernelLaunch',
    'HcommChannelGetStatus', 'HcommThreadNotifyRecordOnThread', 'HcommThreadNotifyWaitOnThread']
missing = []
print('loaded HCOMM:', path)
for name in required:
    found = hasattr(lib, name)
    print('FOUND' if found else 'MISSING', name)
    if not found:
        missing.append(name)
if missing:
    raise SystemExit('旧CCU接口不完整，停止当前部署路线: ' + ','.join(missing))
print('C接口符号检查PASS；还需要C++接口链接和有卡执行验证')
PY
```

还需要C++的`hcomm::CcuKernel`、`CcuRep`、NotifyRecord/NotifyWait、WriteNb、LocalCopyNb、
连续资源分配和LoopGroup接口。可用`nm -DC --defined-only`检查上一段打印的实际HCOMM文件；
最终由第7节的真实编译、链接和加载检查。

这里检查的是已经使用的 **HcclCcuKernelRegister / RegisterFinish / Launch**，
不是之前失败路线里的`HcommCcuKernelRegisterStart`、`CcuWriteMemToMem`等新接口。
不要把“缺少那批新接口”误判成旧对象路线必然不能用；反过来，找到同名符号也不代表
接口结构ABI、固件能力或graph capture一定兼容。

**停止条件：**旧对象头文件/符号缺失或不匹配时，不能靠复制旧`libhcomm.so`、替换新机器
系统`libhccl.so`或改我们的ABI版本返回值绕过。应选取获准的兼容SDK镜像，或先做新接口适配。

## 6 拉取两个仓和构建依赖

当前代码仓及用途：

| 仓 | 分支 | 说明 |
|---|---|---|
| `COLDTEA-L/sgl-kernel-npu` | `feature/a5-ccu-copy-comm-overlap` | 当前算子、桥接、测试和文档 |
| `yuanwenliu/hccl` | `feature/a5-ccu-explicit-multipath-alltoall` | 复用现有custom-op构建框架 |

HCCL分支保留早期实验补丁，但本部署只调用其custom-op构建流程，**不编译安装完整HCCL**。
不要再设置指向旧patched HCCL产物的`--hccl-lib-dir`，不要执行旧原生HCCL改造文档中的覆盖步骤。

全新目录执行：

```bash
(
  set -eo pipefail
  git clone --branch feature/a5-ccu-copy-comm-overlap \
    https://github.com/COLDTEA-L/sgl-kernel-npu.git "${SGL_REPO}"
  git clone --branch feature/a5-ccu-explicit-multipath-alltoall \
    https://gitcode.com/yuanwenliu/hccl.git "${HCCL_REPO}"
  git -C "${SGL_REPO}" rev-parse HEAD
  git -C "${HCCL_REPO}" rev-parse HEAD
  stat -c '%d %n' "${SGL_REPO}" "${HCCL_REPO}"
)
```

两个`stat`设备号必须一致：当前route构建脚本用硬链接把自定义源码临时放到HCCL树内。
仓位于不同文件系统会失败；不要把它误判成CANN错误。
路径已存在时不执行clone覆盖，先检查远端、分支、提交和本地差异，再`fetch`和`pull --ff-only`；
不自动reset、不删除用户文件。两个Git提交号都要随部署记录保存。

HCCL构建依赖包括`cann-cmake`。脚本可复用：

```text
$HCCL_REPO/third_party/cann-cmake/function/prepare.cmake
$HCCL_REPO/build_device/_deps/cann-cmake-src/function/prepare.cmake
$HCCL_REPO/build/_deps/cann-cmake-src/function/prepare.cmake
```

干净新仓不一定包含这些缓存，内网也不一定能在线获取依赖。
应提前准备与所选HCCL构建版本配套的完整依赖源码/缓存和Git提交记录，按它的构建说明放置。
只复制某个`build/`目录或改名一个空文件夹，不能替代依赖安装；未知版本的`cann-cmake`也不能随便混用。
本项目single-op构建跳过部分无关算子准备，不代表所有第三方依赖都消失。

## 7 编译和安装两份自有产物

### 7.1 安装前检查影响范围

先停止本用户正在使用旧route/DeepEP的测试进程；不要停止其他用户进程。
检查目标CANN的`opp/vendors/cust`：当前安装器可能先卸载同vendor的旧自定义包。
如果该目录还装有其他公共算子，先请管理员确认隔离/备份方案，**不要直接继续安装**。
优先使用独立容器层或获准的私有CANN前缀。

route构建脚本会尝试加载`/usr/local/Ascend/cann/set_env.sh`或
`/usr/local/Ascend/ascend-toolkit/set_env.sh`。多版本环境中，这可能覆盖刚才选定的环境。
`--install-path`只是安装目标，不能据此保证编译也用了该SDK。
必须检查构建输出的`CANN home`及CMake路径；若不是预期版本，停止，先调整构建脚本的选版本逻辑，
不要修改公共`latest`软链接，也不要把另一版本编译产物硬装进目标CANN。

### 7.2 编译并确认SDK后安装route包

先只编译打包，不安装：

```bash
(
  set -eo pipefail
  cd "${SGL_REPO}"
  source "${CANN_ROOT}/set_env.sh"
  unset LD_PRELOAD A5_URMA_TP_TRACE_PREFIX
  HCCL_REPO="${HCCL_REPO}" \
    bash scripts/build_a5_ccu_urma_route_probe.sh \
      2>&1 | tee "${EVIDENCE_ROOT}/build_route.log"
)
```

检查日志实际选择的SDK，再决定是否安装：

```bash
grep -nE 'CANN home|ASCEND_CANN_PACKAGE_PATH|CUSTOM_CANN_PACKAGE_PATH|Build succeeded|Verified' \
  "${EVIDENCE_ROOT}/build_route.log"
```

确认编译SDK、打包ABI和计划安装路径一致，且第7.1节的公共目录影响检查已完成后，执行：

```bash
(
  set -eo pipefail
  cd "${SGL_REPO}"
  source "${CANN_ROOT}/set_env.sh"
  HCCL_REPO="${HCCL_REPO}" \
    bash scripts/build_a5_ccu_urma_route_probe.sh \
      --install --install-path "${CANN_ROOT}" \
      2>&1 | tee "${EVIDENCE_ROOT}/install_route.log"
)
```

这会再次调用构建再安装；期间不要切换CANN环境或公共软链接。
预期安装的是：

```text
$CANN_ROOT/opp/vendors/cust/lib64/liba5_ccu_urma_route_probe.so
$CANN_ROOT/opp/vendors/cust/include/a5_ccu_urma_route_probe.h
```

必须同时看到打包库和安装库验证通过。当前预期prepared ABI=16、peer ABI=5、Channel-sync ABI=1；
`ccu`模式还需local+remote ABI>=2。它们是**我们自有库的版本标志**，不是HCCL系统版本。
脚本失败、没有生成run包或安装库仍旧时，不进入算子测试。

### 7.3 首次编译安装DeepEP wheel

新环境首次部署应本地编译。只有已经安装匹配本机Python/torch/CANN、且通过后续ABI检查的
多类型wheel时才跳过；“另一个容器里装过”不等于当前解释器已安装。

```bash
(
  set -eo pipefail
  cd "${SGL_REPO}"
  source "${CANN_ROOT}/set_env.sh"
  export ASCEND_HOME_PATH="${CANN_ROOT}"
  export ASCEND_TOOLKIT_HOME="${CANN_ROOT}"
  DEEPEP_SINGLE_OP=explicit_multipath_all2all_ccu \
    bash build.sh -a deepep Ascend950 \
    2>&1 | tee "${EVIDENCE_ROOT}/build_deepep.log"
  BUILT_SO=$(find output/lib -maxdepth 1 -type f -name 'deep_ep_cpp*.so' -print -quit)
  test -n "${BUILT_SO}"
  nm -D --defined-only "${BUILT_SO}" | grep ' A5DeepEpPeerPlanAbiVersion$'
  WHEEL=$(ls -t output/deep_ep-*.whl | head -1)
  test -f "${WHEEL}"
  printf 'Installing wheel: %s\n' "${WHEEL}"
  python3 -m pip install --force-reinstall --no-cache-dir --no-deps "${WHEEL}"
)
```

新clone没有历史产物，以上wheel选择清晰；若复用旧仓，需先核对wheel时间和本次日志，
不能在编译失败后从历史output里随便选一个wheel。
构建过程会生成源码树内的配置/产物，不需要将这些文件提交回内网远端。
检查`python3 -m pip --version`与构建用Python一致；脚本内部也用`python3`/`pip3`，两者应属于同一环境。

### 7.4 用真实测试入口检查加载和ABI

```bash
(
  set -eo pipefail
  cd "${SGL_REPO}"
  source "${CANN_ROOT}/set_env.sh"
  export A5_CCU_ROUTE_PROBE_LIB="${CANN_ROOT}/opp/vendors/cust/lib64/liba5_ccu_urma_route_probe.so"
  source scripts/a5_ccu_test_env.sh
  a5_ccu_prepare_test_env "${CANN_ROOT}" "${SGL_REPO}"
  python3 tests/python/deepep/test_a5_ccu_peer_plan_alltoall.py \
    --runtime-check --self-copy-mode ccu --channel-sync-mode baseline
)
```

预期出现`CASE_TEST_RUNTIME`、实际route/extension路径，以及`PEER_RUNTIME_VERIFIED`。
DeepEP需要`A5DeepEpPeerPlanAbiVersion>=2`；旧两卡桥接的attr ABI>=12。
`runtime-check`不建通信域、不打流，只证明库加载、桥接API与自有版本标志可用。
旧wheel存在`deep_ep_cpp`顶层导入与包内导入两种布局，当前入口使用别名兼容加载；
不要为解决`No module named deep_ep_cpp`复制另一个SO到site-packages根目录。

## 8 先检查direct-only

### 8.1 选择真实可用的两张计算卡

输入物理ID，不是torch逻辑rank。先确认独占或获准使用，再运行；脚本不会自动判断卡是否空闲。

```bash
read -r -p '输入两张计算卡物理ID，逗号分隔: ' TEST_DEVICES
read -r -p '输入允许纳入本次计划的全部物理ID集合，逗号分隔: ' AVAILABLE_PHYS
read -r -p '输入已确认的direct CommLink枚举序号，首次可检查0: ' DIRECT_ROUTE
export TEST_DEVICES AVAILABLE_PHYS DIRECT_ROUTE
```

只向这两张计算卡启动rank；其他卡是否作为relay稍后配置。
框架中的rank0/1通常不是物理卡0/1，`ASCEND_RT_VISIBLE_DEVICES`改变可见性后必须核对日志映射。

### 8.2 原生HCCL短测

```bash
(
  set -eo pipefail
  cd "${SGL_REPO}"
  bash scripts/run_a5_ccu_native_alltoall_baseline.sh \
    --devices "${TEST_DEVICES}" \
    --bytes 4194304 --warmup 1 --iters 3 --repeats 1 \
    --sync-mode batch --timeout-seconds 300 \
    --cann-root "${CANN_ROOT}" --output-root "${OUTPUT_ROOT}"
)
```

这是`dist.all_to_all_single`原生通信，不验证显式EID合成能力。
该脚本要求CCU_SCHED，但是否实际执行CCU仍以运行日志和后续profiling为准，不能只看环境变量。

### 8.3 自定义direct-only短测

```bash
(
  set -eo pipefail
  cd "${SGL_REPO}"
  bash scripts/run_a5_ccu_peer_plan_alltoall.sh \
    --devices "${TEST_DEVICES}" --available-phys "${AVAILABLE_PHYS}" \
    --direct-route "${DIRECT_ROUTE}" \
    --self-copy-mode ccu --channel-sync-mode baseline \
    --sync-mode batch --graph-backend none \
    --bytes 4194304 --warmup 1 --iters 3 --repeats 1 \
    --timeout-seconds 300 --cann-root "${CANN_ROOT}" \
    --output-root "${OUTPUT_ROOT}"
)
```

不传`--relay-phys`和`--relay-map`就是direct-only，这一步不需要旧EID快照。
应同时通过changed-payload检查、批量排队逐次snapshot检查和进程退出验收。
若提示`base must be a direct CommLink`，先检查新通信域的CommLink目录和hop，
不能强行把candidate2或hop=2改名为direct。
需要目录日志时，可使用已有route probe的`--channel-only`并开启`A5_CCU_TRACE_LINK=1`；
它是额外诊断，不代替本节的数据正确性验收。

## 9 为新机器重新采集拓扑和EID

### 9.1 拓扑JSON与采集范围

```bash
find /usr/local/Ascend/driver/topo -type f -name '*.json' -print 2>/dev/null
read -r -p '输入本机实际950拓扑JSON绝对路径: ' TOPOLOGY_JSON
export TOPOLOGY_JSON
test -f "${TOPOLOGY_JSON}"
```

核对`peer_list/edge_list`、`peer_count/edge_count`、物理ID对应关系、端点Die/Port和`UB_CTP`协议。
当前解析器针对已用过的950 JSON格式；文件结构或ID语义变化时，需要先适配解析器，
不能把新数据硬改成旧物理ID。超节点内多个节点/不同网层也不能当旧一机8卡fullmesh。

### 9.2 只读采集本机端口和EID清单

在允许访问这些查询的容器中执行；否则由管理员在宿主机采集后，把文本和JSON只读带入容器。

```bash
(
  set -eo pipefail
  HCCN_TOOL=$(command -v hccn_tool || true)
  if test -z "${HCCN_TOOL}"; then HCCN_TOOL=/usr/local/Ascend/driver/tools/hccn_tool; fi
  test -x "${HCCN_TOOL}"
  RAW="${EVIDENCE_ROOT}/hccn_device_topology_raw.txt"
  IFS=',' read -r -a PHYSICAL_IDS <<< "${AVAILABLE_PHYS}"
  for dev in "${PHYSICAL_IDS[@]}"; do
    printf '===== Physical Device %s =====\n' "${dev}"
    env -u ASCEND_RT_VISIBLE_DEVICES -u ASCEND_VISIBLE_DEVICES \
      "${HCCN_TOOL}" -g -dev_info -i "${dev}"
  done > "${RAW}"
  printf 'Saved: %s\n' "${RAW}"
)
```

该命令格式来自当前A5脚本。**新驱动若不支持，或输出没有UDMAC/EID表，不能继续当成有效清单。**
按新工具的帮助查找只读EID查询，保存其原始输出；当前解析器需要每张物理卡的
`udmac...`名称、EID index、完整128-bit EID及卡分隔标记，格式改变须先适配解析器。
不能使用设置EID/路由/端口的命令来补齐查询结果。

检查解析器是否真正读到了新机器EID：

```bash
export TOPOLOGY_RAW="${EVIDENCE_ROOT}/hccn_device_topology_raw.txt"
(
  set -e
  cd "${SGL_REPO}"
  python3 - <<'PY'
import collections, os, sys
from pathlib import Path
sys.path.insert(0, 'scripts')
from resolve_a5_synthetic_relay_eids import load_entries, load_topology
rows = load_entries(Path(os.environ['TOPOLOGY_RAW']))
known, edges = load_topology(Path(os.environ['TOPOLOGY_JSON']))
print('EID rows per physical device:', dict(collections.Counter(r['device'] for r in rows)))
print('topology peers:', len(known), 'edges:', len(edges))
required = {int(x) for x in os.environ['AVAILABLE_PHYS'].split(',')}
if not rows or not required <= {r['device'] for r in rows} or not required <= known:
    raise SystemExit('采集/解析不完整或物理ID映射不同，停止relay部署')
PY
)
```

当前resolver还含有旧机器观察到的EID-port编码假设，例如从EID特定位推导port key。
这不是跨版本通用协议保证。即便能解析出行，也要与本机Die/Port、链路UP状态和实际拓扑核对。
EID中的局部数字不能直接当relay卡ID；不得复用旧`resolved_eid_pairs.tsv`或旧manifest。

### 9.3 显式选一张relay并预生成计划

```bash
read -r -p '输入获准使用的一张relay物理ID，必须不在计算卡集合内: ' RELAY_PHYS
export RELAY_PHYS
(
  set -eo pipefail
  cd "${SGL_REPO}"
  python3 scripts/prepare_a5_ccu_peer_plan.py \
    --devices "${TEST_DEVICES}" --available-phys "${AVAILABLE_PHYS}" \
    --relay-phys "${RELAY_PHYS}" --direct-route "${DIRECT_ROUTE}" \
    --topology "${TOPOLOGY_RAW}" --topology-json "${TOPOLOGY_JSON}" \
    --output-dir "${EVIDENCE_ROOT}/first_relay_plan"
  column -s $'\t' -t "${EVIDENCE_ROOT}/first_relay_plan/peer_plan.tsv"
)
```

检查src/dst物理卡、relay、EID、两跳edge端口、每rank本地die是否合理。
relay入口和出口须在同一个relay die，每rank所有Channel也须满足当前单本地die约束。
计划生成成功只是控制参数合法，不是物理relay已被证明。

## 10 显式relay和四卡验收

### 10.1 两卡加一张指定relay

```bash
(
  set -eo pipefail
  cd "${SGL_REPO}"
  bash scripts/run_a5_ccu_peer_plan_alltoall.sh \
    --devices "${TEST_DEVICES}" --available-phys "${AVAILABLE_PHYS}" \
    --relay-phys "${RELAY_PHYS}" --direct-route "${DIRECT_ROUTE}" \
    --topology "${TOPOLOGY_RAW}" --topology-json "${TOPOLOGY_JSON}" \
    --self-copy-mode ccu --channel-sync-mode baseline \
    --sync-mode batch --graph-backend none \
    --bytes 4194304 --warmup 1 --iters 3 --repeats 1 \
    --timeout-seconds 300 --cann-root "${CANN_ROOT}" --output-root "${OUTPUT_ROOT}"
)
```

通过后，把`RELAY_PHYS`改成新的、明确指定的逗号分隔集合，逐步扩展。
不要求relay数量为偶数；但须满足实际可用集合、拓扑、单die和资源预算约束。
不能为了凑6条relay借用未经授权或忙碌卡。

物理路径验收不能只看数据PASS或ChannelHandle数字。
要用本机拓扑预期的`src出口 → relay入口 → relay出口 → dst入口`，核对HCCN计数增量。
新版A5端口查询是`hccn_tool -g -stat -i DEV -u UDIE -p PORT`，参数均来自本机拓扑。
背景流量会影响计数，需要对照/重复采集，不能把别人的流量算成自己的relay。
查询不支持只说明该观测面不可用，不证明relay不存在，也不能据此声称物理路径已确认。

### 10.2 四卡必须重新制定逐对relay策略

四卡用`--devices`的实际四个物理ID，relay从通信域之外选。
每对peer的最大relay数为`(N-k)//(k-1)`；N指本次允许参与路径计划的总物理卡数。
还要满足当前实现的“不允许同一源rank在不同peer上复用同一relay”约束。

仓库`docs/topology/a5_peer_plan_4rank_example.json`只适合其写明的物理ID组合；
新环境必须根据本机拓扑编辑一个新的JSON，不直接使用该示例。
格式例如`{"0,1": [4], ...}`，key为升序物理卡对，不是逻辑rank。
未指定relay的卡对保留direct，不能把通信域内卡当relay。

先确认选出的四张卡全可用，再输入真实配置路径：

```bash
read -r -p '输入四张计算卡物理ID，逗号分隔: ' FOUR_DEVICES
read -r -p '输入已按新拓扑编写的四卡relay JSON绝对路径: ' RELAY_MAP
export FOUR_DEVICES RELAY_MAP
(
  set -eo pipefail
  test -f "${RELAY_MAP}"
  cd "${SGL_REPO}"
  bash scripts/run_a5_ccu_peer_plan_alltoall.sh \
    --devices "${FOUR_DEVICES}" --available-phys "${AVAILABLE_PHYS}" \
    --relay-map "${RELAY_MAP}" --direct-route "${DIRECT_ROUTE}" \
    --topology "${TOPOLOGY_RAW}" --topology-json "${TOPOLOGY_JSON}" \
    --self-copy-mode ccu --channel-sync-mode baseline \
    --sync-mode batch --graph-backend none \
    --bytes 4194304 --warmup 1 --iters 3 --repeats 1 \
    --timeout-seconds 600 --cann-root "${CANN_ROOT}" --output-root "${OUTPUT_ROOT}"
)
```

当前只支持2/4 rank，不能把设备列表扩成8卡就宣称支持8-rank AllToAll。
`ccu`模式9个固定参数加每Channel3个参数，总量最多48，因此最多13个Channel/rank，
不是仅靠拓扑公式就能无限增加relay。两卡direct+6 relay为7个，四卡每peer direct+1 relay为6个。

## 11 多类型 入图和正式性能

### 11.1 多类型

支持FP16、BF16、FP32、INT8、INT32，算子读取输入dtype；无需分别实现一套搬运算法。
将第8/10节命令增加`--dtype float16`、`bfloat16`、`float32`、`int8`或`int32`逐项短测。
所有rank必须使用一致dtype、shape、weights；不要一次跳过正确性就测全部性能。

`--bytes 4194304`始终是**每个目的rank 4MiB，包含self**。
两卡tensor8MiB，四卡tensor16MiB；四卡每rank网络发送量12MiB，另有self4MiB。
direct : 每一条relay默认2:1，不是direct : 所有relay合计2:1。
共享环境脚本还默认设置`HCCL_BUFFSIZE=2300`；这是通信缓冲区配置，不是本次payload字节数。
新机器要核对它与可用内存及平台设置是否合适，不把旧值当硬件要求，也不要为排错盲目调大。

### 11.2 可选ACLGraph

需要跑模型图时，对已通过的第8/10节短测把参数换成：

```text
--sync-mode per-call --graph-backend aclgraph --warmup 1 --iters 3
```

其余设备、拓扑和路径配置保持不变。不使用`aclgraphs`、`-aclgraph`或把
`batch`和当前受限的graph测试强行组合。
必须验证capture、首次执行和多次改变输入后的replay正确；Meta dispatch或能编译不等于设备入图通过。
plan准备/bind在图外，先释放图再销毁资源。标准Ascend C/ACLNN外壳不是当前已交付接口的前提，
本部署使用已实现的torch自定义算子及prepared/peer-plan运行时。

### 11.3 性能和MindStudio

部署验收后再测性能。对已通过的相同命令改为：

```text
--sync-mode batch --graph-backend none --warmup 100 --iters 20 --repeats 1 --profile
```

100次warmup后再开始profiling，正式20次为20次独立算子调用，循环外统一synchronize。
不在性能循环里做正确性检查，不把初始化、装包或capture开销算进算子设备时长。
保持`--self-copy-mode ccu --channel-sync-mode baseline`。
原生baseline也用相同设备和bytes、warmup100/正式20/一轮/batch，不能拿旧机器或不同负载比较。

MindStudio打开本次结果目录的`r1/profiling/rank*/*_ascend_pt`。
`host_avg_us`包含host提交和同步，不是纯设备CCU耗时；CCU任务时长又可能包含peer等待。
还要看两侧同一轮的开始/结束时间，不能只比较单rank最大值。
`size(Byte)=4294967295`等通信profiling占位值不能当真实payload，用命令和plan中的bytes解释。

## 12 常见故障和停止条件

| 现象 | 首先核对什么 | 不应做什么 |
|---|---|---|
| Git认证失败 | 新机器网络、凭据和批准的SSH/PAT配置；内网可用经校验的源码离线包 | 不复制旧机器私钥、旧代理地址，不全局关闭SSL验证 |
| Git切分支冲突/对象损坏 | 分支、diff、构建生成文件；损坏仓先保留备份，优先干净clone | 不盲目reset或删除整个工作目录 |
| 编译找不到cann-cmake | 是否真有完整依赖、离线缓存是否配套 | 不用空目录冒充，也不临时改公共网络环境 |
| 找不到CCU头文件 | 开发包是否完整、CMake是否选对SDK | 不从另一CANN版本复制单个头文件凑编译 |
| undefined symbol | 实际加载SO、依赖路径、旧对象接口及C++ ABI | 不改ABI返回值跳过检查，不替换系统HCOMM碰运气 |
| 只有Meta通过 | 尚未完成设备执行/graph replay | 不当作可跑模型的最终验收 |
| route或DeepEP自有ABI过旧 | 本次是否成功安装、Python/site-packages是否同一环境 | 不把编译目录SO和旧安装wheel混用 |
| 原生HCCL也失败 | 容器透传、驱动、软件配套、卡权限和通信基础设施 | 不先修改算子路径逻辑 |
| direct-only成功，relay失败 | 本机EID/拓扑、local ID、port编码、同die与Channel状态 | 不重用旧EID，不自动换relay或改路由表 |
| 旧普通URMA探针显示完整EID NOT_VISIBLE | 它测试的是普通URMA本地查询/绑定面，不是HCOMM建链 | 不把这个结果当作本路线不可用的直接证据 |
| status124/137 | timeout输出、最后阶段、OOM/外部信号证据 | 137不自动等于OOM；有TERM后kill记录时优先查超时 |
| HcclCcuKernelRegister status7 | 接口返回值定义、所选资源配置、SDK与固件限制 | 不自动增大slice/MS数，更不reset公共卡 |
| 进程退出0但报告不完整 | 每rank结果文件、correctness、lifecycle和日志是否齐全 | 不只看shell退出码判PASS |
| packed波动或无收益 | 先回baseline，仅保留诊断对照 | 不把实验优化当部署默认 |

自定义包安装只更新指定文件，不会主动退出整个Docker；交互shell继承`set -e`、直接source
含`exit`的脚本、连接断开、容器主进程退出等都要分别排查。
先看`docker ps -a`，区分“exec会话结束”和“容器停止”，不要一概归因于算子。

## 13 部署交接记录

至少保存下列内容到独立、可回收的部署目录，不把敏感凭据放入归档：

- 宿主芯片/架构、驱动固件、OS、容器镜像tag及digest、设备透传和mount清单。
- 实际CANN根目录/版本、torch/torch_npu/Python版本、编译器/CMake和C++ ABI信息。
- 两个源码仓提交、cann-cmake等构建依赖提交；两份构建日志、包SHA256及安装路径。
- 新机器driver topology JSON、EID原始清单和解析结果；测试使用的物理ID和relay策略。
- 运行时ABI/路径检查、direct-only、relay、各dtype和按需graph验收结果。
- 性能使用的bytes、权重、self-copy-mode、channel-sync-mode、sync-mode及原始profiling。

正式交付条件是按需求完成上述逐层验收，而不是“编译通过”或“旧环境测过”。
新环境版本不同但以上接口、ABI、数据正确性和路径验证均通过时，可以形成新的部署基线；
若旧对象CCU接口缺失，则属于适配工作，不能当普通重装包继续执行。

相关实现和已有专项测试：

- [copy和通信调度指南](A5_CCU_COPY_COMM_OVERLAP_GUIDE.md)
- [两卡和四卡peer-plan指南](A5_CCU_PEER_PLAN_ALLTOALL_GUIDE.md)
- [多数据类型指南](A5_CCU_MULTIDTYPE_ALLTOALL_GUIDE.md)
- [官方框架配套表](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.md)
- [官方950容器部署说明](https://github.com/Ascend/cann-container-image/blob/main/OVERVIEW.md#run-a-cann-container-on-950-series-aarch64-products)
