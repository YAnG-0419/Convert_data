# 转换包架构与扩展约定

本工作区作为独立的离线数据转换包维护。输入 bag 只读，转换结果、测试临时目录、依赖和缓存都放在本工作区；不依赖 ROS 进程，不安装或修改训练工作区。

## 已实现的分层

```text
CLI + YAML 配置 + TXT episode 清单
                 ↓
registry.py：source_format / target_format 注册表
                 ↓
sources/：bag 发现、采集状态、只读 CDR 解码、按名称归一化关节
                 ↓
recipes/：选择时间窗口、时间对齐、速度处理、连续分段
                 ↓
formats/：训练格式写入、输出编号、格式元数据、校验
                 ↓
pipeline.py：多进程调度、暂存、来源记录、提交、恢复、追加
```

目前输入为 `gello_rosbag2`：ROS2 sqlite3、无 bag 压缩、RGB Image + JointState、本地采集状态与消息 schema。ROS1、MCAP、其他机器人还没有实现，后续通过来源适配器新增。

| 目标 | 采样策略 | 状态/动作 | 图片 | 输出 |
|---|---|---|---|---|
| `lerobot_v21` | 原 Pi05 过去帧保持，30 Hz，150 ms | 108/54 | 原尺寸 H264 | Parquet + MP4 |
| `act_hdf5` | 最近邻，20 Hz，图像/手 50 ms、臂 20 ms | 54/54 或 14/14 | RGB 320×240，HDF5 无损压缩 | episode_N.hdf5 |

Pi05 完成标记支持对应 `converter_version=0.2.0`；有标记的 bag 可输出重叠前缀和完整段。旧命令仍默认 Pi05，但旧数据集不能以 `--resume` 混入新的分段规则。ACT 有独立格式版本 `act-1`，配置/schema 改变必须写入新输出。

## 下一种需求如何添加

1. **先确定数据契约**：目标文件结构、关节顺序、动作语义、图像处理、时间策略、缺失值规则、读入端约束。
2. 如果现有源读取可复用，直接使用 `sources/gello_rosbag2.py`；新类型输入则在 `SOURCES` 注册模块，实现发现、快照、Bag 只读访问和采样策略需要的流读取接口。现阶段来源接口是显式 Python 模块协议，不提供自动插件扫描。
3. 在 `recipes/` 新建模块，实现 `validate_config(config)`、`validate_contract(contract, config)`、`prepare(source, bag, item, config, contract)`。`prepare` 返回 `(streams, aligned)`；`aligned` 至少提供 `timeline`、`segments`、`report`，其他数组由该目标 writer 约定。需要从一条来源生成重叠逻辑 episode 时，可另实现 `output_segments(item, aligned, config)`。
4. 在 `formats/` 新建 writer，实现 `write_segment(root, bag, streams, aligned, bounds, ep_index, global_index, config, contract, segment=None)`、`reindex_segment(stage, episode, episode_index, global_index, task_index=0)`、`write_metadata(root, records, config)`。episode 记录必须包含 `episode_index`、`length`、`features`、`paths`、源起止时间。`features` 必须能够作跨 episode 的严格 schema 比较。
5. `write_segment` 只能写其暂存目录；`reindex_segment` 将最终文件放到 `stage/final/<relative path>` 并更新 episode 记录。流水线统一搬移文件，提交 `conversion/records/*.json` 后才算成功。不要让 writer 自行提交或覆盖历史输出。
6. 实现 `verify(root, full_video=False)`，注册 `TARGETS` 中的 recipe、writer、verifier、自有文件恢复匹配规则及格式版本。恢复匹配规则只匹配该格式自己的输出。
7. 新增 YAML 配置、合成 CDR/SQLite 测试、实际少量 bag 验证记录和使用说明。新的格式依赖优先放入 pyproject 的可选依赖；完整工作区 setup 的锁文件同步更新。

流水线不需要为新增目标增加 if/else，也不复制 TXT 筛选、续跑和锁逻辑。新的语义变化应新增/更新 recipe 版本及测试；仅新增目标不能改变旧目标的采样结果。

## ACT 的关键决策

- 固定顺序：左臂 7、右臂 7、左手 20、右手 20。读取 `schemas/gello_contract.yaml` 的固定名称，逐条消息重排；不会从任意一帧猜测手部顺序。14 维模式无需手部 topic。
- 动作是关节绝对目标，单位 rad，按最近邻直接采样。不会应用 Pi05 的停用目标保持或依赖其状态 topic；没有足够新鲜命令时，该行丢弃。这可能使部分 bag 分成多个输出 episode。
- 时间采用消息 header 的整数纳秒值。等距时选较早消息；超过各组阈值即丢弃。相同源时间保留最后收到的消息，沿用源适配器策略。
- 裁剪依据采集报告的 `source_common_start/end_time_ns`，首尾各裁 1 秒后与已验证的 effective 窗口取交集。缺少 common 字段时，使用必需流的公共区间。已经裁掉的边缘不重复裁剪，也不会扩展采集验证区间。具体边界记录到来源报告。
- 断档切分，短于 `min_segment_frames` 的片段丢弃。默认 2 帧是格式最低要求，正式训练可在新配置中提高以保证 episode 长度。
- qvel 优先用有效源速度；缺失/非有限速度按连续片段内 qpos 差分估计，端点用单边差分。数组长度错误仍报错。显式配置 `qvel_policy: zeros` 才全部填零。
- 三路图片先居中补黑边到目标比例，再 area resize。640×400 头部图补成 640×480 后缩到 320×240；640×480 腕部图直接缩放。源解码处理 rgb8/bgr8 和 step 行填充，写入始终 RGB。
- HDF5 `compress=False` 表示没有 JPEG 编码；`compression: lzf/gzip` 是 HDF5 透明无损压缩，图像仍是 `(T,H,W,3)` uint8。
- 非有限低维值、整段全零动作、全黑帧、分段内分辨率变化会失败。默认 `max_joint_step_rad: 1.0` 是可配置的质量筛选阈值，检查相邻 qpos/action 的单关节变化，并非机械臂安全限位；实际任务需要调整时复制配置并生成新数据集。
- 缺失文件、字段/维度/关节名称不一致、时间断档在 verify 中报错；全量读取图像采用小批次。花屏、视觉语义及动作是否符合示教意图仍需要人工视频检查，自动校验不等价于训练验证。

## 验证与边界

测试覆盖 Pi05 完成标记前缀、重叠内容、分段任务和非法标记，以及 ACT 最近邻与阈值、断档差分、命名关节重排、图像补边和颜色、14/54 维真实 CDR fixture、并行提交、追加编号、续跑字节与时间不变、源只读及异常拒绝。

执行全部测试时隔离终端继承的 ROS pytest 插件，并将临时文件放在当前目录：

```bash
PYTHONPATH="$PWD/src" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .runtime/bin/python -m pytest -q \
  --basetemp=.cache/pytest-conversion
```

输出不自带 ACT 训练代码，也不启动训练；ACT dataloader、相机名配置、action chunk 和 loss 下降需在执行端另行验证。

### 本次实现验证（2026-09-10）

- 全部 41 项测试通过，其中原有 Pi05 测试 33 项、ACT 新增测试 8 项。
- `lists/act_smoke.txt` 中的 episode18、19、20、22、103 转为 5 个连续 HDF5 episode，共 8,442 帧，约 5.14 GB；目录为 `outputs/act/tomato_test`。
- [HDF5 全量校验](../outputs/act/tomato_test/conversion/verification.json)：全部三路图像、低维字段和连续时间戳通过。
- [源数据比对](../reports/act_source_audit.json)：全部低维数组与重建对齐结果一致；每个 episode 首、中、末帧三路 RGB 像素与源图片按规定处理后完全一致；源文件快照未变。
- [续跑验证](../reports/act_resume_verification.json)：重复运行不增加 bag，全部 HDF5 文件的 SHA-256、大小和修改时间不变。
- [episode103 三路并排视频](../reports/act_episode103_preview.mp4) / [预览图](../reports/act_episode103_preview.png)，从左到右为 head、left_wrist、right_wrist。抽样预览已查看，补边与颜色正常；不替代完整示教动作审阅。
- 历史 Pi05 数据指纹算法与新注册接口一致。其中 `outputs/fr3_wuji/tomato` 的历史 manifest 仍记录旧工作区路径 `/home/user/lpy/convest_data`，本次只使用当前 schema 副本核对算法，未更改该 manifest；它的路径迁移问题不由本次 ACT 改动解决。
