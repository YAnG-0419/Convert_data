# 机器人数据转换工作区

目前支持 **Gello / 双 FR3 + Wuji 的 ROS2 sqlite3 bag → Pi05 LeRobot v2.1 / ACT HDF5 / DP3 未裁剪 Zarr**。Pi05/ACT 使用 `scripts/convest`，DP3 质量分组转换使用 `scripts/dp3`。

所有代码、依赖、缓存、日志、转换结果均放在本目录。源 bag 使用 SQLite 只读连接；不安装 ROS，不修改 Pi05、gello-retarget、系统 Python 或它们的环境，不上传数据。

## DP3 未裁剪 Zarr 转换

实现遵循 `ROSBAG_TO_DP3_CONVERSION.md`：按 header 时间因果对齐到 10 Hz，按关节名称生成 54 维 state/action，用 cam0 深度和对应 CameraInfo 反投影，应用配置中的固定外参，输出变长、未裁剪、未体素化、未 FPS 的点云。后续点云裁剪和固定到 1024 点不属于本脚本。

正式转换前必须编辑 `configs/gello_dp3.yaml`，填写真实的 `point_frame`、`T_point_from_depth_camera` 和 `calibration_version`。默认占位外参会被拒绝，避免把相机坐标系误标成公共工位坐标系。

```bash
cd /home/descfly/Convert_data
bash scripts/setup.sh

# 三档分别转换；已有输出必须带 --resume，追加 TXT 后也使用同一命令。
scripts/dp3 convert --quality high --resume
scripts/dp3 convert --quality standard --skip-ineligible --resume
scripts/dp3 convert --quality low --skip-ineligible --resume

# 或一次按配置中的三个 TXT 依次转换。
scripts/dp3 convert --quality all --skip-ineligible --resume

# 每档独立校验。
scripts/dp3 verify --quality all
```

三档对应 `lists/high_quality.txt`、`lists/standard_quality.txt`、`lists/low_quality.txt`，分别写入 `outputs/dp3_uncropped/high`、`standard`、`low`。质量档允许重叠；采集状态不合格默认拒绝，只有显式 `--skip-ineligible` 才跳过并记录。转换中断后，`--resume` 会按提交日志截断未完成数组并继续，不重复已提交 bag。

数据部给出的 `episode51–53` 可用开发配置检查完整链路；该配置故意保留相机坐标系，不能作为正式坐标转换交付：

```bash
scripts/dp3 convert --config configs/gello_dp3_sample.yaml --quality sample --resume
scripts/dp3 verify --config configs/gello_dp3_sample.yaml --quality sample
```

每档输出一个 `dataset_uncropped.zarr`，其中 `point_cloud_xyz [P,3]` 与 `point_cloud_offsets [T+1]` 保存变长点云，`state/action [T,54]` 顺序为左臂、左手、右臂、右手；同时生成 `manifest.json`、`episode_manifest.json`、`conversion_report.json`、`invalid_segments.json` 和 `conversion/` 续跑记录。源文件 SHA-256、源/接收时间、外参和关节名称均随交付保存。

## ACT HDF5 转换

对应 [ACT 数据规范](ROSBag_to_ACT_HDF5_Conversion_Spec.md)，配置在 [configs/gello_act.yaml](configs/gello_act.yaml)。扩展设计和接口见 [转换包架构](docs/conversion_architecture.md)。下面后续的 Pi05 专属章节仍描述原格式。

默认 ACT 输出：20 Hz、54 维 qpos/action、qvel、三路 RGB uint8 320×240，相机名 `head` / `left_wrist` / `right_wrist`。每个连续片段写一个 `episode_0.hdf5` 等文件；断档会切分。关节顺序固定为左臂、右臂、左手、右手。源头部图片先上下补边再缩放。速度缺失则在连续片段内差分。

```bash
cd /home/user/lpy/convert_data

# 独立的 5 bag 测试清单，不改变已有质量分组。
scripts/convest convert --config configs/gello_act.yaml \
  --episode-list lists/act_smoke.txt \
  --output outputs/act/tomato_test --repo-id act/tomato_test \
  --resume --workers 4
scripts/convest verify outputs/act/tomato_test

# 将你当前的标准组清单转换为 ACT；与 Pi05 分开输出。
scripts/convest convert --config configs/gello_act.yaml \
  --episode-list lists/standard_quality.txt \
  --output outputs/act/tomato_standard_quality --repo-id act/tomato_standard_quality \
  --resume --workers 2
scripts/convest verify outputs/act/tomato_standard_quality
```

如清单包含采集不合格的 bag，默认在写入前拒绝；只有显式加 `--skip-ineligible` 才跳过。更换任务请先复制配置并修改 `task`。只控制双臂时把新配置的 `state_dim` 改为 `14`，输出到新目录。

首尾裁剪以原始公共源时间窗口为基准，与采集验证区间取交集，避免重复裁掉已验证区间的 1 秒。ACT 使用最近邻，图像与手部最大时间差 50 ms、手臂 20 ms；旧 Pi05 的过去帧保持策略不受影响。严格阈值可能让部分 bag 只保留若干连续片段，详情见 `conversion/records/*.json`。

`verify` 自动识别格式。ACT 会全量读取三路图像并检查维度、类型、NaN/inf、全零动作、黑帧、连续时间戳、固定关节顺序与跳变阈值。HDF5 文件使用可直接读取的 RGB 数组，默认 LZF 无损压缩。任一片段出现相邻帧单关节 qpos/action 变化超过 1 rad 时，该 bag 转换失败；阈值位于配置中，修改后需写新数据集。一个 bag 的所有片段都成功后才提交，失败不会保留部分训练文件。

输出目录也包含 `conversion/`，保存配置指纹、原始 bag 快照、输出映射、质量记录和校验报告。TXT 追加、`--resume`、多进程文件锁与 Pi05 共用实现。HDF5 数据可直接用 `h5py.File(..., "r")` 读取；ACT loader 及训练 loss 仍需在训练端验证。

本机 ACT 所需 `h5py` 仅安装于当前工作区 `.runtime`。新机器使用 `bash scripts/setup.sh`；通过 Python 包安装时可选择 `.[act]`。

## Pi05 v2.1 完成标记

Pi05 配置默认使用 `segments: all`。没有完成标记的 bag 仍输出一条完整 episode；
`collection_state.json` 中有 `milestones` 时，每个标记输出一条从有效开始到标记的前缀，
最后再输出完整 episode。前缀和完整段有意共享开头的数据，每条输出都重新从 `timestamp=0`
和 `frame_index=0` 开始。标记使用 ROS source header 的时钟，前缀最后一帧不会晚于标记。

可用 `--segments full` 只导出完整段，或用 `--segments milestones` 只导出标记前缀。
`segments: milestones` 遇到没有标记的 bag 会报错。标记必须是严格递增的整数纳秒、使用
`clock: ros` 并落在验证后的有效区间内；带标记的数据若在对齐后存在断档也会拒绝转换，
避免前缀 episode 跨越无效数据。

清洗后压缩时间轴的数据可使用 `--segments first-milestone-or-full`：有标记时只输出从
有效起点到第一个标记的前缀，没有标记时输出完整 episode。若清洗目录保留
`source_collection_state.json` 和 `trim_report.json`，读取器会在内存中校正有效窗口、标记
以及未被清洗器改写的自定义状态消息时间戳；源 `.db3` 不会被修改。不要把
`source_collection_state.json` 直接改名为 `collection_state.json`，否则会混用压缩前后的时间轴。

例如转换清洗后的 `datasets/B`：

```bash
scripts/convest convert \
  --source-root datasets/B --segments first-milestone-or-full \
  --output outputs/fr3_wuji/tomato_B_cleaned \
  --repo-id fr3_wuji/tomato_B_cleaned \
  --resume --workers 8
scripts/convest verify outputs/fr3_wuji/tomato_B_cleaned
```

`segment_tasks` 可为不同阶段设置任务。当前番茄配置将 `milestone_1` 标为
`Pick up a tomato truss with the right hand.`，完整段继续使用顶层 `task`。未配置的其他
milestone 使用其 ID 作为占位任务名。输出的 `meta/episodes.jsonl` 和转换记录包含
`segment_id`、`source_recording_id`、精确标记时间及各段源时间范围；训练/验证划分应按
`source_recording_id` 分组，防止同一采集的重叠前缀落入不同集合。

单独转换 episode476 到新数据集：

```bash
cd /home/user/lpy/convert_data
scripts/convest convert \
  --source-root /home/user/franka_teleop_data/bags/gello \
  --episode episode476 --segments all \
  --output outputs/fr3_wuji/tomato_episode476 \
  --repo-id fr3_wuji/tomato_episode476 --resume
scripts/convest verify outputs/fr3_wuji/tomato_episode476 --full-video
```

标记配置和转换器版本参与数据集指纹，因此启用该能力后应使用新输出目录；旧数据集不能用
`--resume` 直接混入新分段规则。

## 用 TXT 分组并追加转换

原始 bag 保持在 `/home/user/franka_teleop_data/bags/gello`。目前已经创建三份清单：

| 分组 | 序号清单 | 输出目录 | repo_id |
|---|---|---|---|
| 优秀 | `lists/high_quality.txt`：18、19、20、22、23 | `outputs/fr3_wuji/tomato_high_quality` | `fr3_wuji/tomato_high_quality` |
| 标准 | `lists/standard_quality.txt`：32、64 | `outputs/fr3_wuji/tomato_standard_quality` | `fr3_wuji/tomato_standard_quality` |
| 低质量 | `lists/low_quality.txt`：232、238 | `outputs/fr3_wuji/tomato_low_quality` | `fr3_wuji/tomato_low_quality` |

**episode64 当前采集状态为 incomplete：手臂状态消息断档 209.9 ms，超过采集规则的 150 ms。** 它保留在标准清单里；严格转换会拒绝整份清单，在显式指定 `--skip-ineligible` 时才跳过 64 并转换 32。质量评级不会覆盖采集完整性检查。

**episode238 当前采集状态为 incomplete：`arm_command_status` 源时间断档 419.8 ms，超过采集规则的 150 ms。** 它保留在低质量清单里；严格转换会拒绝整份清单，在显式指定 `--skip-ineligible` 时才跳过 238 并转换 232。

```bash
cd /home/user/lpy/convert_data

# 只读检查序号和采集状态；不创建数据集，不写报告。存在不合格项时返回码为 2。
scripts/convest check-list lists/high_quality.txt
scripts/convest check-list lists/standard_quality.txt
scripts/convest check-list lists/low_quality.txt

# 优秀组。--resume 可以从首次创建一直使用到后续追加。
scripts/convest convert \
  --episode-list lists/high_quality.txt \
  --output outputs/fr3_wuji/tomato_high_quality \
  --repo-id fr3_wuji/tomato_high_quality \
  --resume --workers 4

# 标准组。显式跳过不合格的 64；不会放宽任何采集/时序限制。
scripts/convest convert \
  --episode-list lists/standard_quality.txt \
  --output outputs/fr3_wuji/tomato_standard_quality \
  --repo-id fr3_wuji/tomato_standard_quality \
  --skip-ineligible --resume --workers 4

# 低质量组。显式跳过不合格的 238；不会放宽任何采集/时序限制。
scripts/convest convert \
  --episode-list lists/low_quality.txt \
  --output outputs/fr3_wuji/tomato_low_quality \
  --repo-id fr3_wuji/tomato_low_quality \
  --skip-ineligible --resume --workers 4

# 转换结束后校验。追加后需要再次运行。
scripts/convest verify outputs/fr3_wuji/tomato_high_quality
scripts/convest verify outputs/fr3_wuji/tomato_standard_quality
scripts/convest verify outputs/fr3_wuji/tomato_low_quality
```

### 自己创建新的组

新建任意 TXT，例如 `lists/my_group.txt`，每行一个原始 episode 序号或连续范围：

```text
# 这是备注
18
19
23
34-39
```

支持 `18` / `episode18` 两种单序号写法，以及 `34-39` 这样的闭区间范围（包含 34 和 39，共 6 个序号）；也可写成 `episode34-episode39`。范围使用英文短横线 `-`，两侧不要加空格，起始序号不能大于结束序号。可混合使用单序号和范围，支持逗号、中文逗号和空白分隔；空行及 `#` 后的备注会被忽略。展开后发现重复会报错，例如 `18` 与 `episode018`，或 `34-39` 与 `38`。

接着选择一个新的 `repo_id`（如 `fr3_wuji/tomato_my_group`），将 `--output` 设为 `outputs/<repo_id>`，更换上面命令的三个参数即可。输出目录可自动创建，也可由你提前创建为空目录。为配合 Pi05 的数据查找，推荐始终保持这个目录对应关系。`--output` 仍限于当前工作区内部。

输入根目录默认来自 `configs/gello_pi05.yaml`，需要时可使用 `--source-root /path/to/bags`。多个输入目录存在相同 episode 编号时会报歧义，需缩小输入根目录。缺失序号和重复序号始终报错，`--skip-ineligible` 只允许跳过采集不合格项。

### 向现有数据集增加序号

在原 TXT 中加入新序号，再运行原来的带 `--resume` 命令。已经转换的 bag 不重复编码，新 bag 的输出编号接着往后排；即使追加的原始序号更小，也不会重排已有输出。TXT 的路径、注释和内容可以调整，不计入数据格式指纹；`repo_id`、任务、采样参数和 schema 不能在续跑时改变。

从 TXT 删除序号不会删除历史输出，日志和 `conversion/selection.json` 会明确列出保留的历史 bag。如要移除数据或将某条数据在优秀/标准/低质量组之间移动，应生成新的目标数据集。不同组的清单允许重叠；程序不会自动将多份数据集当作互斥集合，需由你维护分组。

从 TXT 创建的数据集，续跑时必须继续提供 `--episode-list`，防止漏写参数后误转整个原始目录。已有的非空目录必须有本工具的 manifest，且使用 `--resume` 才能写入。多个进程同时写同一输出会被文件锁阻止。

**追加后重新校验，并重新计算该组的 Pi05 归一化统计；不要与训练同时续写数据。** 每次清单及跳过原因都会保存在 `conversion/selections/`，最新一次在 `conversion/selection.json`，实际包含的 bag 以 `conversion/records/` 为准。

首次迁移机器时运行 `bash scripts/setup.sh` 创建独立环境；当前机器已准备好。可用 `scripts/convest inspect` 生成全量采集状态清单，或 `--episode episode19` / `--limit 1` 做单独试跑。这两个选择器与 `--episode-list` 互斥。不传任何选择器时仍为旧的全量转换行为，因此分组转换请保留完整命令。

新任务先复制 `configs/gello_pi05.yaml`，修改 `task` 和需要的 `segment_tasks`，再传 `--config`。**默认英文任务沿用 Pi05 现有番茄任务，是本批数据的暂定标签；不同任务必须修改配置后写入新输出目录。** 同一任务只按质量分组时，无需改变 task。

## 数据约定

| 输出 | 内容 |
|---|---|
| `observation.state` | float32，108 维：左臂位置 7、左臂速度 7、右臂位置 7、右臂速度 7、左手位置 20、左手速度 20、右手位置 20、右手速度 20 |
| `action` | float32，54 维：左臂 7、右臂 7、左手 20、右手 20；绝对关节目标，单位 rad |
| `observation.engaged` | 左臂、右臂、左手、右手的接管状态 |
| `observation.images.cam0` | 头部 RGB 相机 |
| `observation.images.cam1` | 左腕 RGB 相机 |
| `observation.images.cam2` | 右腕 RGB 相机 |
| `source_timestamp_ns` | 每行对应的原始绝对源时间，int64 纳秒，可用于审计 |

Pi05 的 `Fr3WujiInputs` 会从 108 维状态提取位置，并将状态/动作重排成 `[左臂、左手、右臂、右手]`。这里保留采集契约顺序，避免重复重排。手臂 delta-action 也由 Pi05 训练变换处理，转换器不预先相减。

三路 RGB 保留原分辨率，编码为 H264/yuv420p。输出视频统一 **30 fps，第一帧 PTS=0**；Parquet 第 i 行对应视频第 i 帧。cam0 的原始 20 Hz 图像通过过去帧保持升到 30 Hz，其他相机同样按源时间对齐。视频是有损编码，像素不会逐位相同。

本目标格式只输出 Pi05 使用的 RGB，不复制深度、相机内参或遥测包到训练特征；完整原始信息仍保留在源 bag。

## 筛选和时间对齐

1. 仅处理 `collection_state.json` 中 `state=finalized`、`finalized=true`、`failures` 为空的 bag。缺失状态、丢弃、中断和采集验证失败会列入跳过报告。
2. 使用采集报告的 `source_effective_start_time_ns` 到 `source_effective_end_time_ns`，这已包含原采集程序的公共时间区间和边缘裁剪；**不再次裁剪，也不擅自扩展到整个 bag 的起止时间**。部分 bag 的有效窗口会明显短于录制时长，详情见 inventory 与来源记录。
3. 以消息 `header.stamp` 建立整数纳秒时间轴；收到时间不替代源时间。对到达乱序做稳定排序，相同源时间使用最后收到的消息。零时间戳、错误序列化、缺失/部分关节、缺少速度或非有限值直接报错，不填零。
4. 对状态、动作、接管状态和图像使用过去最近一条消息，默认最大年龄 150 ms，不取未来帧，不做线性插值。
5. 明确停用的分组保持停用瞬间的最后目标；从停用状态开始则用首个实测位置初始化。接管状态本身必须新鲜。重新接管要求相应新命令，允许状态与命令最多 20 ms 的发布偏差。
6. 无效行会被排除，剩余连续区间拆为独立 episode；不足 2 帧的短区间丢弃。每段从零重新计时，保留 `source_timestamp_ns`。因此输出 episode 数可能大于有效 bag 数，动作序列不会跨越断档。

## 输出结构

```text
outputs/fr3_wuji/tomato/
  meta/
    info.json                  # codebase_version: v2.1
    tasks.jsonl
    episodes.jsonl
    episodes_stats.jsonl       # v2.1 每 episode 统计，包含 RGB 统计
  data/chunk-000/episode_000000.parquet
  videos/chunk-000/observation.images.cam0/episode_000000.mp4
  videos/chunk-000/observation.images.cam1/episode_000000.mp4
  videos/chunk-000/observation.images.cam2/episode_000000.mp4
  conversion/
    manifest.json              # 配置和 schema 指纹
    discovery.json             # 全部 bag 的采集状态
    selection.json             # 最近使用的 TXT、选择/跳过/保留的 bag
    selections/                # 每次运行的清单历史
    records/000000.json        # 每个 bag 的来源快照、分段映射、时序质量、文件路径
    errors.json                # 本次运行的转换失败，续跑会重新尝试
    summary.json               # 本次运行结束后的汇总
    verification.json
    pi05_verification.json
```

按 bag 暂存，全部成功后提交来源记录，再更新标准元数据。中断后 `--resume` 清理未提交的项目输出并重做对应 bag。处理其他 bag 的异常会记录并继续；存在失败时命令返回非零。磁盘剩余空间低于配置保留量时停止。

转换过程中可看 `meta/info.json` 中的已提交帧数和 episode 数。`summary.json`、验证报告反映最近一次完成的运行；全量转换完成后应重新验证，**不要一边续写一边训练**。

## 在 Pi05 中读取

无需改 Pi05 源代码，也无需复制数据到旧数据集目录。对训练进程设置独立的数据根目录即可，原来的 `fr3_wuji/tomato` repo id 对应本目录下的新输出。

```bash
export HF_LEROBOT_HOME=/home/user/lpy/convert_data/outputs
```

本机 Pi05 安装的 TorchCodec 缺少 FFmpeg 动态库，自动选择它会报 `Could not load libtorchcodec`。已经验证它现有的 PyAV 可以正常读取。工作区提供的 `scripts/pi05` 只在本次进程中选择 PyAV，并设置本工作区的数据/缓存路径；不会修改 Pi05 文件、卸载包或安装系统库。

检查现有 LeRobot loader 和 `Fr3WujiInputs`，包括 50 帧动作窗口以及 episode 末尾 padding：

```bash
cd /home/user/lpy/convert_data
scripts/pi05 check outputs/fr3_wuji/tomato_high_quality
scripts/pi05 check outputs/fr3_wuji/tomato_standard_quality
scripts/pi05 check outputs/fr3_wuji/tomato_low_quality
```

检查脚本只读取 Pi05 的解释器和代码，缓存及报告写在当前工作区。数据本身无需这里的代码即可被标准 v2.1 loader 读取。

**新训练需要重新计算该批数据的 Pi05 normalization stats，不能直接沿用旧 65 条数据的统计。** LeRobot 的 `episodes_stats.jsonl` 是原始训练特征的统计；Pi05 的 `norm_stats.json` 还涉及在线重排、50 帧动作窗口和手臂 delta，二者用途不同。

```bash
# 全量转换结束后执行，直接读取向量，无需解码 RGB
scripts/pi05 norm outputs/fr3_wuji/tomato_high_quality
scripts/pi05 norm outputs/fr3_wuji/tomato_standard_quality
scripts/pi05 norm outputs/fr3_wuji/tomato_low_quality
# 自动读取各组 manifest 的 repo_id，分别保存到：
# outputs/pi05_assets/pi05_fr3_wuji/<repo_id>/norm_stats.json

# 确认数据、标签与归一化统计后，可用此命令启动训练（本次任务不自动训练）
scripts/pi05 train --repo-id fr3_wuji/tomato_high_quality --exp-name tomato_high_quality
```

归一化脚本读取本机 `pi05_fr3_wuji` 的动作 horizon/delta 设置，使用 Pi05 原有重排、DeltaActions 和 RunningStats，对全部样本计算统计。训练入口显式将 assets/checkpoints 指向工作区的 `outputs/`，避免覆写 Pi05 原有资产。归一化脚本用文件锁阻止与转换同时运行。检查和归一化脚本自动读取目标数据集的 repo_id；训练入口的 `--repo-id` 会传给 Pi05 的 `--data.repo-id`。仍使用 `pi05_fr3_wuji` 的机器人和模型配置，不修改 Pi05 源文件。

## 代码结构与扩展规划

```text
lists/                      # 用户维护的质量分组 TXT
configs/                    # 输入、目标、任务及采样配置
schemas/                    # 采集契约和自定义 ROS 消息的本地快照
src/convest/
  sources/gello_rosbag2.py   # bag 发现、只读输入、反序列化、按名称提取
  selection.py              # TXT 解析、去歧义、采集状态预检
  align.py                  # 时间对齐及连续 episode 表达
  formats/lerobot_v21.py    # 输出 schema、Parquet、视频、统计
  pipeline.py               # 筛选、并行、顺序提交、续跑及来源记录
  verify.py                 # 独立的输出结构与媒体检查
  cli.py                    # inspect / check-list / convert / verify
scripts/                    # 安装、入口与 Pi05 兼容检查
tests/                      # 时间语义、异常输入、写出往返和并行索引测试
reports/                    # 本次数据清点与运行记录
```

后续增加输入格式时，在 `sources/` 实现源适配器，复用对齐/分段逻辑；增加 LeRobot v3、RLDS、HDF5 等输出时，在 `formats/` 增加 writer，复用已规范化的状态、动作、时间线和图像访问接口。当前 CLI 对未实现的格式明确拒绝，不通过修改版本号冒充兼容。

依赖锁定在 `requirements-lock.txt`。迁移机器可设置 `PYTHON_BIN=/path/to/python3.11 bash scripts/setup.sh`，无需依赖 Pi05 的 Python 包。`.runtime` 通过标准 venv 复用所选 Python 的基础运行时，但不修改该基础运行时。

```bash
env -u PYTHONPATH PYTHONNOUSERSITE=1 .runtime/bin/python -m pytest -q --basetemp=.cache/pytest-tmp
```

来源契约与消息定义复制自本机 `gello-retarget`，许可见 `schemas/LICENSE.gello-retarget`；Pi05 格式依据本机安装的 LeRobot v2.1 与 `examples/fr3_wuji`、`fr3_wuji_policy.py` 核对。

## 主要代码设计、改动与数据转换方法

### 设计思路及与 Pi05 原有转换的关系

代码分成输入读取、清单筛选、时间对齐、格式写出和任务调度几部分。`sources/gello_rosbag2.py` 负责从 ROS2 bag 提取消息，`align.py` 生成对齐后的状态、动作、图像索引及连续片段，`formats/lerobot_v21.py` 负责写成训练数据，`pipeline.py` 负责并行执行、编号、提交和续跑。

```text
原始 ROS2 bag + 采集状态 + 用户 TXT 清单
  → 检查序号和采集完整性
  → 读取消息、按关节名提取数值、建立源时间索引
  → 按统一时间轴采样、处理停用状态和断档
  → 写出 Parquet、三路 MP4、LeRobot v2.1 元数据
  → 输出校验 / 原始数据抽样比对 / Pi05 读取检查
```

参考了 gello-retarget 的采集契约、消息定义和动作语义，以及 Pi05 的 `examples/fr3_wuji/convert_dataset.py`、`fr3_wuji_policy.py` 和本机 LeRobot v2.1 loader。Pi05 原脚本的输入是已经生成的单 episode 旧 LeRobot 目录，主要负责合并与视频重新对齐；本工具直接从 rosbag 生成 v2.1，不需要先生成旧 LeRobot 数据，也不调用旧目录中的 `dataloader.py`。

### 实际使用的数据转换方法

| 环节 | 使用的方法 | 作用及输出 |
|---|---|---|
| ROS2 消息读取 | Python `sqlite3` 只读连接 + `rosbags` 的 CDR 反序列化 | 支持未压缩的 ROS2 sqlite3 bag；标准消息使用 ROS2 Humble 类型库，自定义状态消息使用 `schemas/` 中的本地定义，不需要启动 ROS 节点。 |
| 关节映射与数值转换 | 按 `JointState.name` 查找契约中的关节，而非直接使用消息数组顺序；转成 NumPy `float32` | 生成契约顺序的 108 维状态和 54 维绝对动作，保留原始弧度数值。重复关节、部分关节组、缺失速度或非有限数值会报错，不补零。 |
| 多路时间对齐 | 按 `header.stamp` 建立整数纳秒时间轴；稳定排序、相同源时间保留最后收到的样本；使用 `searchsorted(..., side="right") - 1` 查找过去最近样本 | 默认统一到 30 Hz。这是因果采样保持（零阶保持，ZOH），不会读取未来帧，也不做线性插值。使用采集报告已验证的起止区间，不重复进行边缘裁剪。 |
| 接管状态与动作保持 | 读取四个分组的接管状态；接管时检查命令是否新鲜，停用时保持停用瞬间的最后目标 | 起始停用且没有历史目标时，使用实测位置初始化；重新接管时要求匹配新命令，允许状态与命令有最多 20 ms 的发布偏差。接管状态本身也必须有效、新鲜。 |
| 数据断档处理 | 默认 150 ms 新鲜度检查 + 连续有效区间分段 | 无效采样行会切断 episode，避免动作窗口跨越断档。每段时间重新从零开始，保留绝对 `source_timestamp_ns`；不足 2 帧的短片段丢弃。 |
| 图像与视频转换 | 根据原图 `encoding` 和行步长恢复 RGB，通过 PyAV 调用 `libx264`，编码为 MP4/H264、`yuv420p` | 保留原分辨率，默认 CRF 23、`veryfast`。每路视频与数据行使用相同帧率和帧数，第 i 行对应第 i 帧；20 Hz 的 cam0 通过保持上一张图像对齐到 30 Hz。H264 是有损编码，不保证像素逐位相同。 |
| 表格及元数据写出 | PyArrow 固定长度向量列 + Parquet/Zstandard 压缩，JSON/JSONL 保存元数据 | 输出状态、动作、接管状态、时间及连续索引；生成 `info.json`、`tasks.jsonl`、`episodes.jsonl`、`episodes_stats.jsonl`，版本为 v2.1，每 1000 个 episode 一个 chunk。 |
| 数据集统计 | 对向量计算 min、max、mean、std、count；RGB 使用均匀抽帧及空间下采样统计 | RGB 每段每路最多抽 100 帧，统计值缩放到 [0,1]，形状遵循 v2.1 的通道统计约定。它是抽样统计，不是对全部原始视频像素做精确遍历。 |
| Pi05 训练归一化 | 直接读取 Parquet 向量，调用 Pi05 的重排函数、`DeltaActions` 和 `RunningStats` | 根据训练配置生成动作窗口，窗口在 episode 末尾重复最后一个动作；手臂转为相对当前状态的 delta，手部保持绝对目标，再计算训练用 `norm_stats.json`。这个步骤单独执行，不会改写数据集里的绝对动作。 |

这里没有自动评判“优秀/标准/低质量”的视觉或任务成功率算法：质量分组由用户的 TXT 决定，采集完整性检查独立执行。深度、相机内参及完整遥测消息未写入这个 Pi05 训练格式，仍保留在原始 bag 中。当前没有实现 LeRobot v3、RLDS 或 HDF5 转换器；后续可增加相应格式的 writer。

### 主要实现与已完成的功能改动

- **清单分组和独立命名**：新增 [selection.py](src/convest/selection.py)，在 [cli.py](src/convest/cli.py) 中增加 `check-list`、`--episode-list`、`--repo-id`、`--source-root` 和 `--skip-ineligible`。支持数字、`episode18`、`34-39` 闭区间、注释、UTF-8 BOM 和中英文逗号；范围先展开再统一检查重复，反向范围、重复、缺失及有歧义的序号会拒绝处理。不合格数据默认阻止转换，只有明确指定跳过选项才会略过并记录原因。
- **追加转换和来源记录**：在 [pipeline.py](src/convest/pipeline.py) 中将清单选择与不可变的转换配置分开。TXT 可增减序号，续跑依据已提交的来源路径跳过旧 bag，只给新增数据分配后续编号；改变 `repo_id`、任务、采样参数或 schema 会因指纹不符而拒绝续跑。来源文件记录大小和修改时间快照，清单另存 SHA-256 及每次运行的选择记录。清单删行不会删除历史数据。
- **有限并行和可恢复写出**：使用 `ProcessPoolExecutor` 按 bag 并行，完成后按确定的顺序提交，避免工作进程完成先后影响编号。`--workers 4` 表示最多 4 个进程同时处理不同 bag；每个视频编码器默认 2 个线程。图像只先建立消息头及位置索引，再按需读取和编码，数值时间序列按 bag 放入内存，不将整批原始视频展开到内存或 PNG 目录。
- **目录与中断保护**：支持自动创建输出目录或使用用户预建的空目录。对已有非空数据集要求本工具的 manifest 和 `--resume`，通过文件锁阻止同一数据集被多个转换任务同时写入。每个 bag 先写入暂存目录，完成后提交产物及来源记录；恢复时仅清理该输出中未提交的产物。源 bag 始终以只读方式访问。
- **分组数据接入 Pi05**：更新 [check_pi05.py](scripts/check_pi05.py)、[compute_pi05_norm.py](scripts/compute_pi05_norm.py) 和 [pi05_entry.py](scripts/pi05_entry.py)。检查与统计脚本从目标 manifest 读取 `repo_id`，各组归一化统计保存到独立目录；训练入口将用户指定的 `repo_id` 传给 Pi05。针对本机 TorchCodec 动态库问题，入口只在当前进程内选择已可用的 PyAV。

### 已完成的验证

清单与追加功能实现时，共 **26 项测试通过**，覆盖关节映射、时间对齐、动作保持、断档分段、视频写出、TXT 解析、异常输入、追加和重复续跑等行为。优秀组 5 个 episode、标准组实际导出的 1 个 episode，均通过了完整视频解码与索引校验、Pi05 读取及输入变换检查，并对原始 bag 做了抽样比对。重复执行 `--resume` 后，已有 Parquet 和 MP4 的 SHA-256 与修改时间保持一致。

这些是该次生成结果的验证记录；以后追加数据仍需重新校验和计算对应组的训练归一化统计。报告见 [group_creation_report.json](reports/group_creation_report.json) 和 [group_resume_verification.json](reports/group_resume_verification.json)。
