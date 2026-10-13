# Atlas 950 两卡 1KiB H2D/D2H 主机 I/O 实验

分支：`feature/a5-h2d-io-contention`。本实验只使用两张计算卡，默认物理卡
`2、3`，默认固定每次拷贝为 `1024 Byte`，测量单卡和双卡同时进行
pinned-host H2D/D2H 拷贝时的完成时间和膨胀比。这个尺寸用于近似未来
Host 调度器更新 HBM 路径控制块的开销。

这是独立的主机 I/O 诊断，不初始化 HCCL communicator，不加载 DeepEP 或自定义
route library，不执行 CCU/UB relay，不修改系统路由、不加载 ko。它回答的是卡2、3
是否竞争主机内存、PCIe/主机互联、DMA或HBM侧资源，不能替代多路径算子的拥塞验证。

## 1. 测量内容

每张卡由一个独立torchrun进程控制，使用一块pinned CPU tensor和一块NPU `uint8`
tensor。所有计时拷贝均设置`non_blocking=True`，若host tensor没有真正pinned，测试
直接失败，不回退到pageable同步拷贝。

- H2D：host tensor → NPU tensor；主机侧读、设备HBM侧写。
- D2H：NPU tensor → host tensor；设备HBM侧读、主机侧写。
- solo：只有一张卡提交拷贝，另一进程只参与同步。
- both：卡2、3同时执行同一方向，用于测同向竞争。
- mixed：一张卡H2D、另一张卡D2H，是全双工方向对照，不代替same-direction测试。

默认case：

| case | rank0 / 物理卡2 | rank1 / 物理卡3 |
|---|---|---|
| `h2d_solo_rank0` | H2D | idle |
| `h2d_solo_rank1` | idle | H2D |
| `h2d_both` | H2D | H2D |
| `d2h_solo_rank0` | D2H | idle |
| `d2h_solo_rank1` | idle | D2H |
| `d2h_both` | D2H | D2H |
| `mixed_rank0_h2d_rank1_d2h` | H2D | D2H |
| `mixed_rank0_d2h_rank1_h2d` | D2H | H2D |

每个batch同时记录：

1. NPU event包围的copy-stream时间；
2. host提交至`synchronize`返回的端到端时间；
3. 每次拷贝GB/s；
4. 两进程起跑偏差和实际执行区间重叠比例；
5. 并发时间 / 同卡同方向solo时间的膨胀倍数。

正式数据使用NPU event时间为主，同时保留host时间。小数据容易由host提交和event开销
主导，因此必须同时查看绝对微秒数，不能只看百分比。

`1024 Byte` 在本实验中对应 `1024` 个 `uint8`，不表示 1024 个文字。
每个计时 batch 默认连续提交 1000 次拷贝，只在 batch 末尾同步一次，
最后用总时间除以 1000 得到单次平均时间。这样能降低 event 分辨率和同步开销对
1KiB 结果的扰动，但它仍是连续 copy 的均摊延迟，不是“每次 H2D 后立即启动算子”
的端到端延迟。

## 2. 有卡机器拉取

有卡机器只拉取和运行实验。切换前要求tracked工作区干净；如果检测到本地修改就立即
停止，不在共享机器上自动备份或清理。未知untracked文件保留原位；若它与新分支同名
导致`git switch`拒绝，停止并由机器管理者处理，不要运行`git clean`。

```bash
(
set -e
cd /home/l00934901/sgl-kernel-npu
git status --short
git diff --quiet
git diff --cached --quiet
BRANCH=feature/a5-h2d-io-contention
git fetch origin "${BRANCH}"
git switch "${BRANCH}"
git merge --ff-only "origin/${BRANCH}"
git branch --show-current
git log -1 --oneline
test -f scripts/run_a5_h2d_io_contention.sh
test -f tests/python/deepep/test_a5_h2d_io_contention.py
)
```

本实验没有C++或自定义算子改动：**无需重新编译route package、DeepEP wheel或HCCL，
也不需要pip install**。只需当前Python中的PyTorch/Torch-NPU与CANN环境可用。

## 3. CPU侧入口检查

不初始化NPU：

```bash
cd /home/l00934901/sgl-kernel-npu
python3 tests/python/deepep/test_a5_h2d_io_contention_cpu.py
bash scripts/run_a5_h2d_io_contention.sh --help
```

CPU测试只检查参数、case方向、带宽和区间计算，不代表pinned allocator或NPU DMA通过。

## 4. 首次短测

先确认物理卡2、3在当前共享机器上确实空闲。短测仍只测 1KiB，
但减少迭代次数：

```bash
cd /home/l00934901/sgl-kernel-npu
bash scripts/run_a5_h2d_io_contention.sh \
  --devices 2,3 \
  --sizes 1024 \
  --warmup 10 --iters 100 --repeats 1 \
  --output-root /home/l00934901/profiling
```

脚本先打印唯一的`Result directory`。成功后报告应显示：

- `complete: True`；
- 两个rank都使用pinned host memory；
- 每个size的H2D、D2H correctness gate均为PASS；
- solo、same-direction和mixed case都有记录；
- 无`H2D_IO_FAILURE`，进程退出状态为0。

若pinned allocation失败，保留日志并停止，不要删除严格检查或改成普通CPU tensor继续测；
pageable拷贝会引入隐式staging/同步，结果不再是同一实验。

## 5. 1KiB 正式测试

默认命令仅使用卡2、3，每张卡只分配 1KiB pinned host buffer 和 1KiB NPU buffer：

```bash
cd /home/l00934901/sgl-kernel-npu
bash scripts/run_a5_h2d_io_contention.sh \
  --devices 2,3 \
  --sizes 1024 \
  --warmup 100 --iters 1000 --repeats 5 \
  --output-root /home/l00934901/profiling
```

不带参数直接运行也是同一组默认值：

```bash
cd /home/l00934901/sgl-kernel-npu
bash scripts/run_a5_h2d_io_contention.sh
```

默认在同一对torchrun进程中完成全部case，避免重复初始化runtime，
也减少共享机器占用时间。warmup、correctness和case间barrier均不计入正式copy时间。

如果需要确认 1KiB 附近的拐点，可以补充一组小控制块尺寸：

```bash
cd /home/l00934901/sgl-kernel-npu
bash scripts/run_a5_h2d_io_contention.sh \
  --devices 2,3 \
  --sizes 64,128,256,512,1024,2048,4096 \
  --warmup 100 --iters 1000 --repeats 5 \
  --output-root /home/l00934901/profiling
```

这组扫描研究的是小控制块延迟，不用它推断大块 H2D/D2H 带宽平台。

## 6. 输出与判读

运行目录包含：

```text
run_settings.json             # 设备、size、case和迭代设置
cases/h2d_io.log              # 两个rank的完整日志
cases/h2d_io.status           # torchrun退出状态
results/rank0.json            # 卡2原子结果
results/rank1.json            # 卡3原子结果
h2d_io_records.csv            # 全部solo/concurrent原始样本
h2d_io_inflation.csv          # 每repeat相对同卡solo的膨胀
h2d_io_summary.json           # 机器可读汇总
h2d_io_report.md              # 中位数报告
```

关键比较：

```text
h2d_both(rank0) / h2d_solo_rank0
h2d_both(rank1) / h2d_solo_rank1
d2h_both(rank0) / d2h_solo_rank0
d2h_both(rank1) / d2h_solo_rank1
```

`device_time_inflation=1.25`表示相同卡、相同方向和相同数据量下，并发时间比solo
增加25%。同时看`solo_device_avg_us`、`concurrent_device_avg_us`和GB/s；小数据的高
百分比可能只有很小的绝对微秒差。

mixed case用于验证方向性：如果H2D+H2D明显膨胀，而H2D+D2H基本不膨胀，结果支持
同向共享带宽竞争；如果mixed同样严重，需要考虑主机内存、root complex、DMA队列、
NUMA或HBM等并非全双工链路方向独立的共享资源。

`overlap_shorter_fraction`应接近1，表示较短作业基本处于并发窗口内；还需查看
`overlap_longer_fraction`和`start_skew_us`。窗口严重错位的样本不应用于拥塞结论。

## 7. 实验边界

- 测量单位是每张卡、每次copy的bytes，不是两卡合计。
- NPU event时间反映同一NPU stream上的copy区间，不是PCIe线速的独立硬件计数。
- `GB/s`使用十进制10^9 bytes/s，由有效payload除以时间得到。
- correctness gate在计时外执行一次精确常数模式检查；性能batch复用buffer，未逐次保存snapshot。
- 本实验不采HCCN flit，因为H2D/D2H不是UB relay流量。
- 测试不绑定CPU NUMA。若两卡膨胀明显，下一步再加入CPU/NUMA绑定对照，不在第一版中混入额外变量。
- 共享机器存在其他CPU内存或PCIe流量时，先换空闲窗口复测，不能直接归因于卡2、3互相竞争。
