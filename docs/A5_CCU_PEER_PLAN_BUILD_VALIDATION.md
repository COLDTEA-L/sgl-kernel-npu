# peer-plan 初版开发验证记录

分支 `feature/a5-ccu-peer-plan-alltoall-4rank`；2026-10-08。
开发服务器 `8.146.225.22`，容器 `cam_lyw_dev_91`，无 NPU。
实际 CANN 为 `/usr/local/Ascend/cann-9.1.0-beta.1`（`/usr/local/Ascend/cann` 指向它），
Python 环境为 `cam_py311_pt28`。这不是有卡机器的 T560 运行验证。

已完成：

- 新 CCU runtime CMake `-Werror` 编译、安装到容器 `/tmp/a5-peer-plan-install`。
- `ctypes.CDLL` 加载新 runtime，独立 `A5CcuPeerPlanAbiVersion()==1`，三个新 C API 存在。
- DeepEP 完整 `build.sh -a deepep Ascend950` 构建、wheel 打包及 pip 安装。
- 安装后独立 `A5DeepEpPeerPlanAbiVersion()==1`，Buffer 方法存在。
- 两卡 / 四卡 Meta 与 `torch.compile(..., fullgraph=True)` 回归通过。
  测试发现并修正 FakeTensor symbolic shape 上直接调用 `numel()` 的问题；新增 Meta 检查使用 SymInt。
- CPU 布局：两卡 / 四卡 direct-only、relay 分片精确覆盖、两卡六条 relay、四卡限制、
  禁止域内 relay/跨 peer 复用 relay、溢出、零权重、小分片、缺少 peer 检查通过。
- Python policy/resolver 测试：拓扑两条真实边与端点 EID 联接、奇数 relay 数、
  非顺序 physical-rank 映射、完整/中断日志分析通过。
- 源码 Python 语法、shell `bash -n` 与 Git whitespace 检查通过。
- wheel 安装命令之后仍能执行容器内的 `echo`/`hostname`，未出现安装主动退出容器。

复现 runtime 编译（无需改 libhccl）：

```bash
source /opt/conda/bin/activate cam_py311_pt28
source /usr/local/Ascend/cann/set_env.sh
cd /home/liuyuanwen/sgl-kernel-npu-discovered-path
cmake -S examples/a5_ccu_urma_route_probe -B /tmp/a5-peer-plan-build \
  -DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann \
  -DCUSTOM_OPS_OPP_LIB_PATH=/tmp/a5-peer-plan-install/lib \
  -DCUSTOM_OPS_OPP_INC_PATH=/tmp/a5-peer-plan-install/include
cmake --build /tmp/a5-peer-plan-build -j8
cmake --install /tmp/a5-peer-plan-build
```

**未验证**：真实四卡 ChannelAcquire、硬件接收数据、指定物理 relay 的 HCCN 链、
四卡 ACL graph 首次 capture/repeated replay、性能收益。新指南第 4 节先测 direct-only，
第 5 节再测显式 relay，第 7 节单独验证 profiling/graph。
Meta/fullgraph PASS 只是软件图接口验证，不等于这些硬件项目已经完成。
