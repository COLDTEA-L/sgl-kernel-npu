# A5 CCU discovered-path AllToAll 实验指导

## 目标与边界

本分支把 HCCL RankGraph 已发现的 `CommLink` 当作可选择的 path object，创建对应 CCU Channel，并把对端 slice 按权重切给多个 Channel 并发 `WriteNb`。

它不会修改 `ubus.ko` 或全局 UB route table，也不宣称能指定物理 relay 卡。`path_uid` 只标识本次 HCCL 通信会话中已经发现的 CommLink；最终物理出口必须由 HCCN counter 实验确认。

注意：旧参数 `route-index=0/2` 只是枚举序号，既不是 `route_addr_idx`，也不是 UBUS route table index 或 TP path ID。

## 1. 拉取分支

```bash
cd /home/l00934901/sgl-kernel-npu
git fetch origin
git switch -C feature/a5-ccu-discovered-path-alltoall \
  --track origin/feature/a5-ccu-discovered-path-alltoall
git rev-parse --short HEAD
```

## 2. 编译、安装并刷新 DeepEP

```bash
cd /home/l00934901/sgl-kernel-npu
source /usr/local/Ascend/cann-9.1.T560/set_env.sh

bash scripts/build_a5_ccu_urma_route_probe.sh --install

cd python/deep_ep
python3 setup.py bdist_wheel
LATEST_WHEEL=$(ls -1t dist/*.whl | head -n 1)
python3 -m pip install --force-reinstall --no-deps "${LATEST_WHEEL}"
cd ../..

source python/deep_ep/deep_ep/vendors/hwcomputing/bin/set_env.bash
python3 - <<'PY'
import deep_ep, deep_ep_cpp
print(deep_ep.__file__)
print(deep_ep_cpp.__file__)
PY
```

每次 C++ binding 或自定义 CCU 库变化后都要重新编译安装。不要用目录中任意一个旧 wheel；命令必须选择修改时间最新的包。

## 3. 枚举本次会话的 path catalog

以下示例使用物理卡 4、5；换卡只需修改 `--devices` 以及 `ASCEND_RT_VISIBLE_DEVICES`。

```bash
mkdir -p /home/l00934901/profiling

bash scripts/run_a5_ccu_urma_route_probe.sh \
  --devices 4,5 \
  --route-index 0 \
  --bytes 2097152 \
  --warmup 1 \
  --iters 1 \
  --channel-only 2>&1 | tee /home/l00934901/profiling/path_catalog_4_5.log

python3 scripts/parse_a5_ccu_path_catalog.py \
  /home/l00934901/profiling/path_catalog_4_5.log \
  --output /home/l00934901/profiling/path_catalog_4_5.tsv
```

日志中的记录格式为：

```text
PATH_CATALOG rank=0 peer=1 path_uid=... ordinal=... layer=... hop=... src_addr=... dst_addr=...
```

只有同时出现在 rank0、rank1 的 UID 才会进入 TSV。UID 由规范化后的两端地址、协议、hop 和 layer 计算；它用于本次拓扑/软件版本下的选择，升级 CANN、改变卡对或重建拓扑后应重新枚举。

## 4. 建立 CommLink 到物理出口 footprint

对 TSV 中每个 UID 对应的 `rank0_ordinal` 单独打流。当前 HCCN 采集脚本仍用 ordinal，是因为它和同一次 catalog 日志一一对应；最终 AllToAll 使用 UID。

```bash
bash scripts/run_a5_ccu_urma_route_probe.sh \
  --devices 4,5 \
  --route-index 0 \
  --bytes 4194304 \
  --warmup 100 \
  --iters 100 \
  --remote-only \
  --hccn-stat \
  --hccn-devices 0,1,2,3,4,5,6,7 \
  --hccn-stat-root /home/l00934901/profiling
```

对其他 ordinal 重复。需要同时保存空载 counter，避免把后台业务流量误判为 relay。建议至少测 1/2/4 MiB、每个大小重复 3 次，并比较随 payload 增长的 counter 斜率：

- 只在端点直连出口显著增长：direct footprint；
- 中间卡对应 ingress/egress 同时、稳定且按 payload 增长：forwarding footprint；
- 直连与 forwarding 端口同时增长：aggregate footprint；
- 只有全机背景波动：UNKNOWN，不能据此命名 relay。

## 5. 运行显式 path UID AllToAll

从 `path_catalog_4_5.tsv` 选择两个 UID：

```bash
PATH_A=<第一个path_uid>
PATH_B=<第二个path_uid>

bash scripts/run_a5_ccu_discovered_path_alltoall.sh \
  --devices 4,5 \
  --path-uids "${PATH_A},${PATH_B}" \
  --path-weights 2,1 \
  --bytes 2097152 \
  --warmup 100 \
  --iters 100 \
  --output-root /home/l00934901/profiling
```

脚本在同一配置下依次运行：native HCCL、A-only、B-only、A+B serial、A+B concurrent，并保存 `summary.json`。`2,1` 表示 peer slice 的 2/3 给 A、1/3 给 B；不是给每条路径各发送完整 2 MiB。

并发实验的最低判据：

1. correctness PASS；
2. `T_concurrent < T_serial`；
3. aggregate bandwidth 优于最佳单路径；
4. 同一测量窗口内两个预期 footprint 都有显著 counter 增量。

`T_concurrent ≈ max(T_A_chunk,T_B_chunk)` 只是理想参考，不作为硬性 PASS。若只有前两项而没有物理 counter 证据，只能说 CCU 命令并发提交，不能说物理链路并发。

## 6. 单独调用 Python 接口

```bash
export ASCEND_RT_VISIBLE_DEVICES=4,5
export HCCL_OP_EXPANSION_MODE=CCU_SCHED
export HCCL_BUFFSIZE=2300

python3 -m torch.distributed.run --standalone --nproc-per-node=2 \
  tests/python/deepep/test_a5_ccu_urma_multiroute_all2all.py \
  --implementation multiroute \
  --path-uids "${PATH_A},${PATH_B}" \
  --path-weights 2,1 \
  --schedule concurrent \
  --bytes 2097152 --warmup 100 --iters 100
```

故障判断：`path_uid is not in this session catalog` 表示 UID 已过期或卡对不一致；`selected routes span die` 表示当前实现不能把跨 die path 放进同一个 CCU kernel，需要按 die 分组，不能强行运行。

