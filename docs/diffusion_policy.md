# 双 Franka + 双 Wuji：ROS2 bag → Diffusion Policy

本实现对接同级 `diffusion_policy` 仓库的 `ReplayBuffer`、`SequenceSampler`、
`BaseImageDataset`、`DiffusionUnetImagePolicy` 和 `TrainDiffusionUnetImageWorkspace`。
所有新增代码、依赖、缓存、转换结果和训练输出都放在 `Convert_data` 内。源 bag 使用 SQLite 只读连接。
这是图像版 Diffusion Policy 的 RGB-D 扩展，与原有 `scripts/dp3` 点云格式分别使用。

## 1. 数据与默认约定

已读取 `company-remote/tomato/episode60` 的 metadata、collection_state 和实际 CDR 消息：

| 输入 | 实际话题 / 类型 | 输出 |
|---|---|---|
| 左/右臂位置 | `/left/franka/joint_states`、`/right/franka/joint_states`，各 7 维 | `joint_pos` 的前 14 维 |
| 左/右手位置 | `/teleop/wuji/{left,right}/joint_states`，各 20 维 | `joint_pos` 的后 40 维 |
| 臂目标 | `/teleop/validated_arm_commands` | `action` 的前 14 维 |
| 手目标 | `/teleop/wuji/{left,right}/command` | `action` 的后 40 维 |
| 头部 RGB | `/cam0/color/image_raw`，RGB8，640×400 | `head_rgb` |
| 左/右腕 RGB | `/cam1/color/image_raw`、`/cam2/color/image_raw`，RGB8，640×480 | `left_wrist_rgb`、`right_wrist_rgb` |
| 头部深度 | `/cam0/depth/image_raw`，16UC1，640×400 | `head_depth` |

- 状态和动作均为 **54 维弧度绝对关节位置**：左臂 7 → 右臂 7 → 左手 20 → 右手 20。
  不用下一帧实测状态替代动作，不自动转成增量动作。
- 严格按 `schemas/gello_contract.yaml` 中的关节名重排；臂的 measured 与 validated 名称分别匹配。
  手部关节集合不一致会拒绝转换。维度从 contract 推导，更换手部时必须提供正确映射并修改训练 shape_meta。
- 30 Hz 整数纳秒时间网格，使用 collection_state 的已验证 source effective 窗口，不重复裁首尾 1 秒。
  不使用 bag 接收时间代替 source header 时间，也不导出 milestone 重叠前缀。
- 各流取当前时刻或之前的最后一个样本；状态最大年龄 75 ms、图像 100 ms、状态标志与动作 150 ms。
  手部健康标志、臂 faults 也参与筛选。缺帧、过期或故障会切段，默认丢弃不足 16 帧的片段。
- 未接管的关节保持退出接管时的最后目标；若此前没有命令，则固定使用退出接管时已有的实测位置。
  此时命令年龄可超过 150 ms；动作来源和时间均记录。没有历史测量时拒绝该时间步，不取未来数据填充。
- RGB 等比例补黑边后缩放为 **320×240**，uint8；头部上下补边，腕部不改变宽高比。
- 深度单独存为 **float32 米**，无效/非正值为 0。16UC1 默认乘 `depth_scale: 0.001`，沿用此采集的配置约定；
  更换深度驱动时须确认尺度。32FC1 按米读取。最近邻缩放避免混合边缘深度。
- 深度与 RGB 分别送入编码器；不做像素级 RGB-depth 注册、不投影点云，不需要外参。
- 全黑 RGB、整帧无效深度、非有限关节值和连续片段内单关节变化超过 1 rad 会让该 bag 转换失败。

## 2. 安装与转换

### 一键批量转换（30 Hz）

新增 `scripts/dp_batch`：`--all`（默认）递归发现输入根目录内所有 episode，
`--episodes episode60 episode62 ...` 一次选择多个 episode，`--episode-list` 从 TXT 选择。
三种选择方式互斥，统一按 **30 Hz** 转换，
自动跳过已提交的 bag、恢复中断转换，并在结束时全量校验输出。
所有选中的 episode 合并到一个数据集，`meta/episode_ends` 保留各连续片段的边界。
`--workers N` 控制同时转换的 episode 数，默认 1；例如 `--workers 4` 启动最多 4 个转换进程。
每个进程独立读取、解码、对齐和压缩一个 bag，主进程按输入顺序提交到同一个 Zarr。
图像提交直接复制已压缩的逐帧数据块，不重复解码/压缩；episode 边界和提交记录保持一致。
临时结果在输出目录的 `conversion/staging` 内，同时最多保留 N 个待提交任务，正常结束后自动清理。
多进程会增加内存和临时磁盘占用；实际提速取决于 CPU、磁盘和网络读取带宽。
续跑时可以修改 workers 数量，它不改变数据格式、30 Hz 帧率或配置指纹。

```bash
cd /home/descfly/lcl_work/Convert_data
# 首次安装；已完成安装的机器可省略。
bash scripts/setup.sh

# 默认批量转换 ../company-remote/tomato 到 outputs/dp/tomato_30hz。
scripts/dp_batch --all --workers 4

# 一次指定多个 episode，写入同一个 replay_buffer.zarr。
scripts/dp_batch --episodes episode60 episode62 episode63 --workers 3 \
  --output outputs/dp/selected_30hz

# 自定义输入、输出；允许明确跳过采集状态不合格的 bag。
scripts/dp_batch --source-root /home/descfly/lcl_work/company-remote/tomato \
  --output outputs/dp/tomato_30hz --skip-ineligible

# 按 TXT 清单批量转换。
scripts/dp_batch --episode-list lists/high_quality.txt --workers 4 \
  --output outputs/dp/tomato_high_30hz
```

该脚本固定使用 30 Hz，即使 `--config` 指向其他帧率的配置，也会将本次输出帧率设为 30。
动作、状态和图像共享 30 Hz 时间网格；原始 20 Hz 头部图像采用过去帧保持，部分相邻输出帧会相同。
时间步长按整数纳秒交替取 33,333,333 / 33,333,334 ns，避免累计舍入误差。
旧 20 Hz 数据集不能直接追加 30 Hz 数据；请使用新的 `_30hz` 输出目录。
训练默认路径也已改为 `outputs/dp/tomato_30hz`。

### 单个 bag 与底层命令

```bash
cd /home/descfly/lcl_work/Convert_data
bash scripts/setup.sh

# 单个完整 bag，30 Hz 使用新输出目录。
scripts/dp convert --source-root ../company-remote/tomato/episode60 \
  --output outputs/dp/episode60_30hz --resume
scripts/dp verify outputs/dp/episode60_30hz

# 整个 source_root 下的 bag，默认配置指向 ../company-remote/tomato。
scripts/dp convert --output outputs/dp/tomato_30hz --resume
scripts/dp verify outputs/dp/tomato_30hz

# 或按已有质量清单选择，范围/单个编号语法沿用现有列表工具。
scripts/dp convert --episode-list lists/high_quality.txt \
  --output outputs/dp/tomato_high_30hz --resume

# 切换输入根目录；--limit 仅用于选前 N 个 bag 的试运行。
scripts/dp convert --source-root /path/to/bags --limit 2 \
  --output outputs/dp/another_sample --resume
```

默认对采集状态不合格的 bag 报错；明确允许跳过时加 `--skip-ineligible`，原因保存到 selection.json。
转换过程中的解码/对齐/质量错误会逐 bag 记录、回滚，并继续其他 bag；命令返回非零退出码。
源文件 SHA-256 计算会增加一次源数据读取。大批量转换应预留磁盘空间，默认保留 5 GB。

每个 bag 完整写入后才产生提交记录；`--resume` 根据记录截断未提交的尾部并跳过已完成 bag。
输出配置/contract/消息 schema 不一致或已提交源文件 size/mtime 改变时拒绝续跑。
输出目录限制在 `Convert_data` 下，且不能与输入重叠。转换期间有文件锁防止两个转换进程同时写入。
**先完成转换及 verify，再启动训练**；训练期间不要向同一个数据集追加。

## 3. Zarr v2 结构

```text
outputs/dp/episode60_30hz/
  replay_buffer.zarr/
    data/
      joint_pos        float32 [T,54]
      action           float32 [T,54]
      head_rgb         uint8   [T,240,320,3]
      left_wrist_rgb   uint8   [T,240,320,3]
      right_wrist_rgb  uint8   [T,240,320,3]
      head_depth       float32 [T,240,320,1]  # metres; 0 = invalid
    meta/episode_ends  int64   [E]            # cumulative, exclusive end indices
    provenance/
      timestamp_ns       int64 [T]
      source_time_ns     int64 [T,S]
      receive_time_ns    int64 [T,S]
      engaged            uint8 [T,4]
      action_from_state  uint8 [T,4]
  shape_meta.yaml
  conversion_report.json
  verification.json
  conversion/
    manifest.json        # immutable config, contract, fingerprint
    selection.json       # selected/skipped input bags
    records/000000.json  # bag SHA-256, source snapshot, alignment, segment mapping, inspection
```

Zarr attrs 保存 `stream_keys`（时间矩阵列序）、状态/动作关节名、组顺序、fps、动作语义、深度尺度与图像变换。
仅 `data` 中的数组参与 ReplayBuffer 的训练采样。图像按帧分块、Zstd 无损压缩，转换和训练都不把全量图像放进内存。
`verify` 会读遍所有图像，并检查数组类型、维度、episode 边界、连续时间网格、因果性、时效与关节跳变。

## 4. 官方 DP 训练接入

现成的 `RealPushTImageDataset` 将动作限定为特定任务的 2/6 维，其默认图像编码器也不接受 depth 类型。
这里用 `convest.dp.dataset.FrankaImageDataset` 和 `convest.dp.encoder.RGBDObsEncoder` 接入官方训练工作区。
编码器为各路 RGB 和单通道深度分别配置 ResNet18 + GroupNorm，权重从头训练。
官方仓库无需复制或修改任何文件。

**转换环境**是本目录 `.runtime`（Python ≥3.11）；**训练环境**使用已有官方 DP 环境。
适配模块通过 `PYTHONPATH` 读取，不向 DP 环境安装转换包；本机已在 `robodiff5090` 环境实际验证。

```bash
cd /home/descfly/lcl_work/Convert_data
export DP_ROOT=/home/descfly/lcl_work/diffusion_policy
export DP_DATASET=/home/descfly/lcl_work/Convert_data/outputs/dp/episode60_30hz

# 读取实际输出 shape_meta，测试采样、loss、梯度和预测；CPU、小 U-Net、两步采样。
/home/descfly/miniforge3/envs/robodiff5090/bin/python -B \
  scripts/check_dp_training.py "$DP_DATASET" \
  --report reports/dp_episode60_training.json

# 正式训练：DP_DATASET 改成已转换完的多 bag 数据集。
/home/descfly/miniforge3/envs/robodiff5090/bin/python -B scripts/train_dp.py
```

其他机器替换 Python 环境路径即可。训练配置为 `configs/train_franka_dp.yaml`，任务配置为
`configs/task/franka_rgbd_image.yaml`。如修改转换分辨率/关节数，必须同步 task 的 shape_meta，
可参考输出目录的 `shape_meta.yaml`；loader 会拒绝不匹配的配置。

训练输入：`obs` 中关节为 `[B,2,54]`，RGB 为 `[B,2,3,240,320]`，深度为 `[B,2,1,240,320]`；
`action` 为 `[B,16,54]`。loader 将 RGB /255，深度按 `depth_max_m=2.0` 截断后除以该值；
policy normalizer 再映射图像到 [-1,1]。存储的米单位深度不受训练截断影响。
部署时必须复用这些预处理与关节顺序；预测返回弧度绝对目标，默认执行 chunk 长度 8，对应约 0.267 秒（8/30）。

- 训练/验证按原始 `source_recording_id` 分组，同 bag 的不同连续片段不会泄漏到两侧。
- 关节 normalizer 只用训练集统计。只有一个源 bag 时验证集为空，不伪造验证数据。
- 默认 10% 原始采集用于验证，batch size 8，300 epochs；显存不足可减小 batch size。
- 默认 WandB offline，输出在 `Convert_data/outputs/dp_training/`；没有真机 rollout。
  默认 checkpoint 按 `train_loss` 保存；多 bag 有验证集时可改为 `val_loss`。
- 本代码提供离线训练接口；实际机器人执行器、控制延迟测定与闭环成功率验证需在部署系统中完成。

## 5. 验证记录

当前 30 Hz 示例 episode60：**1951 帧、1 个连续片段、54 维动作**，通过批量入口的完整转换、全量校验和重复续跑检查。
结果在 `outputs/dp/episode60_30hz`；见 `reports/dp_episode60_30hz_verification.json`、`reports/dp_episode60_30hz_resume.txt`。

历史 20 Hz 示例 episode60：**1301 帧、1 个连续片段、54 维动作**（原输出 `outputs/dp/episode60` 保留）。
见 `reports/dp_episode60_inspection.json`、`reports/dp_episode60_verification.json`、
`reports/dp_episode60_resume.txt`、`reports/dp_episode60_training.json`。
官方训练工作区单步优化和 checkpoint 保存记录见 `reports/dp_workspace_smoke.json`。
转换测试 58 项通过；DP 环境下的数据集专项测试 4 项通过。
单样例仅验证数据/代码链路，尚未批量转换全部源 bag，也未进行收敛训练。

```bash
# 避免系统 ROS 的 pytest 自动插件污染本地虚拟环境。
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
  .runtime/bin/python -m pytest -q

# DP 训练环境下验证分组划分、训练统计、边界 padding 和深度缩放。
PYTHONPATH=src:../diffusion_policy PYTHONDONTWRITEBYTECODE=1 \
  /home/descfly/miniforge3/envs/robodiff5090/bin/python -B \
  -m unittest discover -s tests -p test_dp_dataset.py -v
```

### 全量转换前的本机盘点

本次扫描 `company-remote/tomato` 共 262 个 episode，224 个通过采集校验，38 个未通过
（未完成、已丢弃或缺少 collection_state.json）。逐项原因见 `reports/dp_batch_inventory.json`。
按 episode60 的实测压缩比和各 bag 的有效时长估算，224 个合格 bag 的 30 Hz 输出约 280 GB，
扫描时本地可用空间约 74 GB；估算会随图像内容和实际保留片段变化。
全量转换需先准备足够空间；仅转换通过采集校验的 bag 可使用 `--all --skip-ineligible`。

### 并行实现测试

`tests/test_dp_workers.py` 使用临时生成的微型 ROS2 CDR/SQLite 数据：验证多进程并发、
并行与串行结果逐数组一致、跨 episode 的索引偏移、失败回滚、变更 worker 数量后的续跑、
压缩图像块直接复制和临时文件清理。没有使用用户的真实 episode 进行本轮转换。
