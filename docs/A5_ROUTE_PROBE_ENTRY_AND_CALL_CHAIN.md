# A5 Route Probe：从 Shell 脚本到 C/C++ 的调用链

## 1. 文档目的

本文对应分支：`feature/a5-urma-explicit-relay-provider`。

本文只解释以下两套probe是怎样启动和调用的：

1. CCU+URMA route通信probe；
2. 完整EID用户态可见性probe。

重点是 `.sh -> 可执行文件 -> main -> 动态库/API -> host实现 -> CCU kernel` 的文件关系。RankGraph内部如何枚举
layer/link不在本文展开。

## 2. 两套probe总览

| Probe | 启动脚本 | main源文件 | 生成的可执行文件 | 主要目标 |
|---|---|---|---|---|
| CCU+URMA route probe | `scripts/run_a5_ccu_urma_route_probe.sh` | `examples/a5_ccu_urma_route_probe/testcase/main.cc` | `a5_ccu_urma_route_probe_test` | 选择已有route、建立CCU channel并执行`WriteNb` |
| 完整EID probe | `scripts/run_a5_urma_full_eid_route_validation.sh` | `examples/a5_urma_full_eid_probe/resolve_full_eid.c` | `resolve_full_eid` | 检查RankGraph日志中的完整EID能否被用户态URMA精确绑定 |

两个probe不是同一个程序：前者链接ACL、HCCL和自定义route probe库；后者直接链接`liburma.so`。

## 3. CCU+URMA route probe

### 3.1 用户执行的Shell脚本

入口：

```text
scripts/run_a5_ccu_urma_route_probe.sh
```

典型命令：

```bash
bash scripts/run_a5_ccu_urma_route_probe.sh \
  --devices 6,7 \
  --route-index 2 \
  --bytes 2097152 \
  --warmup 10 \
  --iters 100
```

脚本主要完成：

1. 设置`ASCEND_RT_VISIBLE_DEVICES=6,7`等环境；
2. 编译testcase可执行文件；
3. 创建临时目录和`root_info.bin`；
4. 分别构造rank0和rank1命令；
5. 后台启动两个独立进程并等待结束；
6. 按选项采集msprof或HCCN统计。

脚本构造的命令形态为：

```text
a5_ccu_urma_route_probe_test \
  --bytes ... \
  --warmup ... \
  --iters ... \
  --route-index 2 \
  --worker-rank 0 \
  --root-info-file /tmp/.../root_info.bin

a5_ccu_urma_route_probe_test \
  --bytes ... \
  --warmup ... \
  --iters ... \
  --route-index 2 \
  --worker-rank 1 \
  --root-info-file /tmp/.../root_info.bin
```

因此它不是由`torchrun`拉起，而是脚本使用`&`启动两个相同的C++程序。每个进程负责一个HCCL rank。

### 3.2 testcase如何编译

Makefile：

```text
examples/a5_ccu_urma_route_probe/testcase/Makefile
```

源文件：

```makefile
SOURCES = main.cc
```

输出：

```makefile
TARGET = a5_ccu_urma_route_probe_test
```

关键链接库：

```makefile
-lacl_rt
-la5_ccu_urma_route_probe
```

其中`liba5_ccu_urma_route_probe.so`是前面已经编译并安装到CANN vendor目录的自定义库：

```text
$ASCEND_HOME_PATH/opp/vendors/cust/lib64/liba5_ccu_urma_route_probe.so
```

对应头文件安装在：

```text
$ASCEND_HOME_PATH/opp/vendors/cust/include/a5_ccu_urma_route_probe.h
```

### 3.3 C++ main入口

入口文件：

```text
examples/a5_ccu_urma_route_probe/testcase/main.cc
```

入口函数：

```cpp
int main(int argc, char **argv)
```

`main()`依次完成：

```text
ParseOptions
  -> 设置A5_CCU_ROUTE_INDEX/A5_CCU_ROUTE_INDICES
  -> aclInit
  -> 检查可见设备数为2
  -> rank0生成并写出HcclRootInfo，rank1读取HcclRootInfo
  -> 构造ThreadContext
  -> RunRank
```

这里的`workerRank`是进程参数，不是`main.cc`在一个进程内再创建两个rank线程。

### 3.4 RunRank创建通信域与设备资源

`main.cc::RunRank()`执行：

```text
aclrtSetDevice(workerRank)
  -> HcclCommInitRootInfo
  -> aclrtCreateStream
  -> aclrtMalloc(sendBuf/recvBuf)
  -> 初始化输入buffer
```

随后在warmup和正式计时循环里调用：

```cpp
HcclCcuUrmaRouteProbe(
    sendBuf,
    recvBuf,
    sendCount,
    HCCL_DATA_TYPE_FP32,
    comm,
    stream);

aclrtSynchronizeStream(stream);
```

计时结束后，`RunRank()`将`recvBuf`复制回host，检查来自两个source rank的数据，并输出平均值、P50和P95。

### 3.5 公开C API与host实现

API声明：

```text
examples/a5_ccu_urma_route_probe/inc/a5_ccu_urma_route_probe.h
```

入口声明：

```cpp
HcclResult HcclCcuUrmaRouteProbe(
    void *sendBuf,
    void *recvBuf,
    uint64_t sendCount,
    HcclDataType dataType,
    HcclComm comm,
    aclrtStream stream);
```

host实现：

```text
examples/a5_ccu_urma_route_probe/op_host/route_probe.cc
```

当前`HcclCcuUrmaRouteProbe()`直接转发给：

```cpp
HcclCcuUrmaMultiRouteWrite(...);
```

后者首先调用：

```cpp
GetRouteResources(
    comm,
    stream,
    RouteKernelKind::ROUTE_WRITE,
    &resources);
```

### 3.6 GetRouteResources负责选路与建链

实现文件：

```text
examples/a5_ccu_urma_route_probe/op_host/utils.cc
```

核心流程：

```text
GetCcuRouteIndices
  -> SelectRoute
  -> 取得所选已有CommLink并生成HcclChannelDesc
  -> HcclThreadAcquireWithStream(COMM_ENGINE_CCU)
  -> HcclChannelAcquire
  -> HcommChannelGetStatus
  -> 注册CCU kernel
  -> 返回RouteResources
```

本文不展开`SelectRoute`内部的RankGraph layer/link枚举。只需注意：`route-index=2`是在这里变成一份
`HcclChannelDesc`，再交给`HcclChannelAcquire`。

`RouteResources`中保存后续执行需要的：

```text
rank/rankSize
routeIndices
weights
dieId
mainThread/routeThread
ChannelHandle列表
CcuKernelHandle
```

同一个`comm + stream + routeKey + kernelKind`的资源会被cache，warmup和正式迭代不会每次重新建channel。

### 3.7 host侧启动CCU kernel

`route_probe.cc`根据各路径权重切分bytes，构造：

```cpp
RouteTaskArg taskArg(...);
```

然后启动已注册的kernel：

```cpp
HcclCcuKernelLaunch(
    comm,
    resources.routeThread,
    resources.kernel,
    &taskArg);
```

### 3.8 CCU kernel执行数据传输

实现文件：

```text
examples/a5_ccu_urma_route_probe/op_kernel_ccu/route_kernel.cc
```

入口不是普通CPU `main()`，而是由HCOMM注册和`HcclCcuKernelLaunch`启动的CCU kernel对象：

```text
CreateRouteKernel
  -> RouteKernel::Algorithm
```

核心数据面调用：

```cpp
WriteNb(channels_[i], destination, source, pathBytes[i], events.back());
```

执行顺序为：

```text
交换远端地址/token
  -> 对每条channel提交WriteNb
  -> WaitEvent
  -> completion notify同步
```

### 3.9 route probe完整文件链

```text
scripts/run_a5_ccu_urma_route_probe.sh
  -> examples/a5_ccu_urma_route_probe/testcase/Makefile
  -> examples/a5_ccu_urma_route_probe/testcase/main.cc::main
  -> main.cc::RunRank
  -> HcclCcuUrmaRouteProbe（liba5_ccu_urma_route_probe.so导出API）
  -> op_host/route_probe.cc::HcclCcuUrmaMultiRouteWrite
  -> op_host/utils.cc::GetRouteResources
  -> op_host/utils.cc::SelectRoute
  -> HcclChannelAcquire/HcommChannelGetStatus
  -> HcclCcuKernelLaunch
  -> op_kernel_ccu/route_kernel.cc::RouteKernel::Algorithm
  -> WriteNb/WaitEvent
```

## 4. 完整EID用户态可见性probe

### 4.1 用户执行的Shell脚本

入口：

```text
scripts/run_a5_urma_full_eid_route_validation.sh
```

典型命令：

```bash
bash scripts/run_a5_urma_full_eid_route_validation.sh \
  --src-phy 6 \
  --dst-phy 7 \
  --urma-include /usr/include/ub/umdk/urma \
  --urma-lib-dir /lib64 \
  --output-root /home/l00934901/profiling
```

该脚本内部包含两个阶段。

### 4.2 阶段A：借用route probe生成完整EID清单

如果没有传入`--route-log`，脚本先执行：

```bash
bash scripts/run_a5_ccu_urma_route_probe.sh \
  --devices SRC,DST \
  --route-index 0 \
  --bytes 1024 \
  --warmup 0 \
  --iters 1 \
  --channel-only
```

虽然参数选择`route0`，route probe仍会打印该卡对的全部候选route。脚本解析rank0日志中的：

```text
route
hop
src_phy/dst_phy
src_addr/dst_addr
```

并生成：

```text
route_eid_manifest.tsv
```

这里复用了前一个C++ route probe；EID probe本身不调用RankGraph API。

### 4.3 resolve_full_eid如何编译

Makefile：

```text
examples/a5_urma_full_eid_probe/Makefile
```

源文件及输出：

```makefile
resolve_full_eid: resolve_full_eid.c
```

链接：

```makefile
-lurma -lpthread
```

它不链接自定义route probe库，也不注册CCU kernel。

### 4.4 C main入口

源文件：

```text
examples/a5_urma_full_eid_probe/resolve_full_eid.c
```

入口：

```c
int main(int argc, char **argv)
```

脚本针对manifest里的每一个完整EID单独执行一次：

```bash
resolve_full_eid "${eid}"
```

### 4.5 C main执行的URMA调用

```text
urma_str_to_eid
  -> urma_init
  -> urma_get_device_by_eid
  -> urma_get_eid_list
  -> 查找完整EID对应eid_index
  -> urma_create_context
  -> 比较ctx->eid与请求EID
  -> urma_delete_context
  -> urma_free_eid_list
  -> urma_uninit
```

它只验证完整EID能否被普通用户态URMA精确解析和绑定，不执行：

```text
HcclChannelAcquire
HcclCcuKernelLaunch
WriteNb
实际数据传输
relay/source-route安装
```

输出会汇总到：

```text
full_eid_visibility.tsv
```

若任一内部route EID不可见，脚本输出：

```text
EID_RESULT status=NOT_VISIBLE
SCHEME_A_GATE=FAIL
```

### 4.6 EID probe完整文件链

```text
scripts/run_a5_urma_full_eid_route_validation.sh
  ├── 阶段A
  │   -> scripts/run_a5_ccu_urma_route_probe.sh
  │   -> a5_ccu_urma_route_probe_test
  │   -> route日志
  │   -> route_eid_manifest.tsv
  │
  └── 阶段B
      -> examples/a5_urma_full_eid_probe/Makefile
      -> examples/a5_urma_full_eid_probe/resolve_full_eid.c::main
      -> liburma.so
      -> full_eid_visibility.tsv
```

## 5. 两套main函数的职责区别

| 对比项 | `testcase/main.cc` | `resolve_full_eid.c` |
|---|---|---|
| 进程数 | Shell启动两个进程，每个进程一个rank | 每个EID启动一个短进程 |
| 初始化ACL | 是 | 否 |
| 初始化HCCL通信域 | 是 | 否 |
| 直接调用RankGraph | 通过动态库内`SelectRoute`间接调用 | 否，只解析route probe日志 |
| 建立channel | 是 | 否 |
| 注册/启动CCU kernel | 是 | 否 |
| 实际发送数据 | 是 | 否 |
| 调用公开URMA EID API | 不由main直接调用 | 是 |
| 产出 | 正确性、时延、带宽及route日志 | EID可见性和context精确绑定结果 |

## 6. 与Python算子测试的区别

常规DeepEP/AllToAll测试通常是：

```text
Python测试脚本
  -> Python binding
  -> deep_ep_cpp.so
  -> aclnn/custom-op host API
  -> device/CCU kernel
```

当前route probe则是：

```text
Shell脚本
  -> C++ testcase main
  -> liba5_ccu_urma_route_probe.so
  -> HCCL/HCOMM channel API
  -> CCU kernel
```

之所以使用独立C++ main，是为了直接控制：

- `HcclComm`的创建和销毁；
- rank0/rank1 root info交换；
- `HcclChannelAcquire`；
- route/channel诊断日志；
- CCU kernel注册和启动；
- 在不依赖PyTorch/DeepEP binding的情况下隔离控制面问题。
