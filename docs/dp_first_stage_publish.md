# 全量首阶段数据转换与上传

## 数据规则

- 源目录：`/home/descfly/lcl_work/company-remote/tomato`，只读。
- 上传目录：`/home/descfly/lcl_work/company-remote/tomato-dp`。
- `collection_state.json` 必须标记 `finalized=true`、`state=finalized`，无采集失败，且存在有效的源时钟窗口。
- 无 `milestones`：保留完整的**采集校验有效窗口**，继续使用原有首尾质量裁剪。
- 有一个 ROS 时钟记录点：保留有效窗口起点至 `milestone.timestamp_ns - 1`，记录点及之后的数据全部排除。
- 非 ROS 时钟、无效记录点或多于一个记录点拒绝转换，写入跳过清单。
- 本批 262 条中 223 条符合要求：130 条单阶段、93 条双阶段前缀。38 条未通过采集校验或缺失状态文件；另有 episode146 的有效窗口仅 4 帧（约 0.13 秒），不足 16 帧，列入跳过清单。

30 Hz 因果对齐；三路 RGB + 头部深度 + 54 维关节位置和绝对目标动作。RGB 为 320×240 uint8，
深度为 320×240×1 float32 米。时间缺口会形成独立片段，训练不会跨越片段或记录点。

## 执行与恢复

```bash
cd /home/descfly/lcl_work/Convert_data
scripts/dp_publish --dry-run
scripts/dp_publish --workers 2
```

同一命令可以恢复；已上传的源数据、配置与归档大小必须匹配。
运行期间不要启动第二个任务，本机锁会拒绝并发发布。
`--workers` 只影响并行度，内存/磁盘预算会进一步限制同时在处理的 bag。
转换使用本地临时缓存；每条输出完整读取图像校验，再归档；上传后的文件逐字节读取并核对 SHA-256。
只有远程归档和发布清单均成功提交后，才删除本次任务产生的本机临时目录。原始 bag 不执行写操作。

可单独转换到本机普通 Zarr 目录：

```bash
scripts/dp_batch --all --skip-ineligible --segment-mode first_stage \
  --workers 2 --cache-source --output outputs/dp/local_first_stage
```

本机磁盘空间不足以保存全量未归档输出，实际执行采用 `dp_publish` 的逐条上传流程。

## 上传结构

```text
tomato-dp/
  dataset_manifest.json        # 整批索引、校验和、来源、边界、完成状态
  inventory.json               # 223 条选择依据 + 39 条跳过原因
  episodes/
    episode57/
      replay_buffer.zarr.zip   # 标准 Zarr v2 ZipStore，图像块已用 zstd 压缩
      record.json              # 来源 SHA-256、原始与保留窗口、片段边界
      verification.json        # 完整图像/动作/时间检查结果
    ...
  training_adapter/
    convest/dp/...
  README.md
```

按 episode 归档避免通过 SSHFS 逐个上传大量小文件。每个 ZIP 内含标准 `data/*`、`meta/episode_ends`
及 `provenance/*` 数组，可用 `zarr.ZipStore(path, mode='r')` 打开；没有再压缩已经压缩的图像块。
全量训练通过 `FrankaImageDataset` 自动识别 `dataset_manifest.json`，懒加载这些 ZIP，按原始采集划分
训练/验证，使用官方 DP 的 `SequenceSampler`，无需先解压或将所有图像加载进内存。
发布清单状态必须为 `complete` 才允许整批训练，避免误用尚未传完的数据。

## 在已有官方 DP 环境中使用

当前主机：

```bash
cd /home/descfly/lcl_work/Convert_data
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export DP_DATASET=/home/descfly/lcl_work/company-remote/tomato-dp
export PYTHONPATH=/home/descfly/lcl_work/Convert_data/src:/home/descfly/lcl_work/diffusion_policy
/home/descfly/miniforge3/envs/robodiff5090/bin/python -B scripts/train_dp.py --config-name train_franka_dp
```

这次任务只执行转换和上传，不启动新的训练。
服务器训练时，将 `tomato-dp/training_adapter` 加入 `PYTHONPATH`；数据集 `_target_` 为
`convest.dp.dataset.FrankaImageDataset`，`dataset_path` 设置为服务器上 `tomato-dp` 的实际路径。
相机与动作 `shape_meta` 见发布清单中 `parts[0].attrs.shape_meta`，或随附的训练配置。

## 查看状态

```bash
.runtime/bin/python -B scripts/dp_publish_status.py
```

打开 <http://127.0.0.1:8766>。完整日志在 `outputs/dp_first_stage_publish/publish.log`，
本机状态在同目录 `progress.json`，远程提交状态在目标目录 `dataset_manifest.json`。
