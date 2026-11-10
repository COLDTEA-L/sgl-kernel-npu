# 新机容器 rootinfo 来源修复

适用 `feature/a5-ccu-8rank-direct-alltoallv`，当前实例为宿主机
`liteserver-hps-tokens-001-00002`、容器 `sglang_yuanwen_old`。

## 1. 已定位的问题

2026-10-27 用户提供：

- 宿主机 `/etc/hccl_rootinfo.json` SHA256：
  `2e0254a1a271c850429fed35b31cfc2347caca8348701c53c176d65c89da1b04`。
- 容器同名文件 SHA256：
  `cae8f1905ca309224bccddc94065d2b206d4066f351fd824463271f4cffa8c8e`。
- Docker 显式挂载缺少 rootinfo，driver/topology 目录已挂载。
- 容器 rootinfo 中的四个 EID 与旧机快照逐条相同，当前实时 EID 属于另一套地址。

因此已确认容器使用了旧机地址配置；原 Docker 恢复命令漏挂 rootinfo。
显式 Mounts 不能排除 runtime 自动挂载，所以修复脚本还检查实际 mountinfo。
宿主文件是否正确仍需和当前设备核对，不能仅依据文件名或哈希不同就直接安装。
本问题不等于普通占卡，也不能通过对 EID 某些字段加常数/取模修复。

## 2. 更新代码后，在宿主机运行修复脚本

先按 [分支指南第 1 步](A5_CCU_ALLTOALLV_BRANCH_QUICKSTART.md#1-拉取并切换分支)
更新本分支，保证包含 plane 修正和 `scripts/fix_a5_ccu_container_rootinfo.sh`。
以下命令在 **宿主机** 执行，不能在 `root@sglang_yuanwen_old` 容器 shell 中执行。

仅检查：

```bash
cd /home/l00934901/sgl-kernel-npu
bash scripts/fix_a5_ccu_container_rootinfo.sh \
  --container sglang_yuanwen_old \
  --root-info /etc/hccl_rootinfo.json \
  --src-phy 2 --dst-phy 3 --relay-phy 4 --relay-die 1
```

用户要求修复当前容器时，执行带 `--apply` 的版本即可；它也会先完整检查：

```bash
cd /home/l00934901/sgl-kernel-npu
bash scripts/fix_a5_ccu_container_rootinfo.sh \
  --container sglang_yuanwen_old \
  --root-info /etc/hccl_rootinfo.json \
  --src-phy 2 --dst-phy 3 --relay-phy 4 --relay-die 1 \
  --apply
```

脚本执行顺序：

1. 将宿主文件复制到新建的共享目录
   `/home/l00934901/a5-rootinfo-fix.XXXXXX/host-rootinfo.json`，不修改宿主原件。
2. 在容器内加载该副本，并读取容器中的 Atlas，实时查询本次涉及卡的 hccn EID/端口状态。
   检查 `2→4→3` 至少存在一个 relay die 1、四个端点 EID 都匹配且端口 UP 的组合。
   不创建 HCCL 通信域、不分配 NPU tensor、不启动通信算子。
3. 检查失败则停止，容器 `/etc` 不变，保留诊断目录。成功输出
   `HOST_ROOTINFO_CHECK_PASS`；这只证明对应的地址/端口来源一致。
4. `--apply` 模式检查容器内实际挂载：目标文件或整个 `/etc` 若已挂载则停止，
   避免写穿宿主配置。应在重建容器时修正该挂载来源。
5. 备份容器内已有的两种大小写 rootinfo 到同一个共享目录；不存在的名字记录 `.absent`。
6. 用经过检查的宿主副本，逐文件原子替换容器中的
   `/etc/hccl_rootinfo.json` 和 `/etc/hccl_rootInfo.json`，两者内容相同。
   这样既避免自动发现冲突，也避免不同消费者读到旧大小写文件。
7. 比较两个容器文件与已校验副本的 SHA256，一致才输出 `ROOTINFO_APPLY_PASS`。

脚本不会 stop/restart 容器，不复位设备，不修改宿主 rootinfo/驱动/固件/路由。
它替换的是容器文件，**没有给运行中的容器新增 Docker bind mount**。
已有测试进程可能已经缓存通信配置，后续需要启动新的测试进程；不要求重启整机。
若要回退，先按打印的证据目录核对 `.before` 文件及 `.absent` 记录，再恢复原容器文件。

## 3. 重新做选路预检

修复成功后进入容器，按 [分支指南第 4 步](A5_CCU_ALLTOALLV_BRANCH_QUICKSTART.md#4-新机器-rootinfoatlas实时-eid-选路预检)
重测。该机给出的 Atlas 为 `2:0/6 ↔ 4:1/0`、`3:0/3 ↔ 4:1/3`，relay die 是 1。
跨卡两端的 plane_0/plane_1 标签不必相同；这个解析修复也必须已更新。

```bash
docker exec -it sglang_yuanwen_old bash
```

```bash
cd /home/l00934901/sgl-kernel-npu
python3 scripts/resolve_a5_explicit_multirelay_eids.py \
  --src-phy 2 --dst-phy 3 --relay-phys 4 --relay-planes 1 \
  --diagnostics-json /tmp/a5-rootinfo-fixed-check.json
```

成功后继续构建/ABI/设备短测的相应步骤。选路通过不证明此前 EI0014 已解决，
也不证明 ChannelAcquire 或 NPU 数据正确性通过；仍以新测试进程的实际结果为准。
本次只有 Python/脚本/文档变化，不需要因此重编 route/DeepEP。

## 4. 持久修复：创建容器时只读挂载本机 rootinfo

不能用 `docker start` 或 `docker exec` 给现有容器追加 Docker 挂载。
当前容器可先按第 2 节修复；将来协调创建替代容器时，保留实际所需设备和现有挂载，
在原 `docker run` 中增加下面两项，源文件必须已通过本机地址校验：

```bash
--mount type=bind,src=/etc/hccl_rootinfo.json,dst=/etc/hccl_rootinfo.json,readonly \
--mount type=bind,src=/etc/hccl_rootinfo.json,dst=/etc/hccl_rootInfo.json,readonly \
```

这是增加到已有命令的参数片段，不是独立执行的 shell 命令。
如果宿主机真正的文件名是另一种大小写，两项 src 都改成实际路径。
两个 dst 使用同一源文件，防止镜像中另一种大小写的旧文件被误读。
不要覆盖宿主文件，不要删除正在使用的容器来强行重建；保留现有容器及可写层工作。
新建后再核对 hashes、实际 mountinfo，并重复第 2 节的只读检查和第 3 节选路。

## 5. 开发验证范围

脚本已通过 bash -n、内嵌 Python AST 检查，以及开发容器 `cam_lyw_dev_91`
中的 fixture 集成检查：正确来源的只读检查通过；错误 EID 来源在 `--apply` 模式
被提前拒绝，并验证容器实际 rootinfo 内容/存在状态未改变。
未对实验机执行安装，未在开发容器安装 fixture rootinfo；真实 `--apply` 后的
文件哈希和后续设备测试仍以用户执行结果为准。
日志位于开发仓 `build/ep8_validation/rootinfo_fix_check.log`、`rootinfo_fix_reject.log`。
