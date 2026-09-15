# 三批优秀数据的 DP 训练与离线演示

本次试验从 `company-remote/tomato` 的 262 个 episode 中筛选，先要求采集状态完整，
再采用已有 `lists/high_quality.txt` 人工优秀清单。32 个候选按采集顺序分为三段，
每段按 source header 连续性和遥测异常排序，选出 4 个。
技术分数不代表任务成功率，任务质量以已有人工标注为依据。

| 批次 | 训练 episode | 验证 episode |
|---|---|---|
| 1 | 404、409、417 | 419 |
| 2 | 439、444、456 | 457 |
| 3 | 459、469、474 | 483 |

每条完整采集只属于一个划分；任何切分片段都随原 `source_recording_id` 一起分组。
选择依据、各话题时间间隔、分数和名单在 `outputs/dp_pilot_20260915/selection.json`。

## 转换与训练

- 转换：30 Hz，54 维关节位置/绝对目标动作，三路 RGB + 米单位深度。
- 输出图像：320×240；训练编码器缩放到 160×120，RGB 双线性、深度最近邻。
- 远程输入通过 `--cache-source` 顺序复制到 `Convert_data` 内的临时目录后再本地读取，
  SHA-256 在缓存文件上计算；复制前后及提交前检查原始快照，提交后清理缓存。
- 同级官方 DP 的 `TrainDiffusionUnetImageWorkspace` 与 `DiffusionUnetImagePolicy` 执行训练。
- 4 路独立 ResNet18 + GroupNorm，从头训练；U-Net 通道数 128/256/512。
- 观测窗口 2 帧、动作训练窗口 16 帧、预测执行段 8 帧。
- 初始实验 30 epochs，每个 epoch 最多 50 次更新，共 1500 次优化；batch size 16。
- 首轮动作误差较大，随后恢复模型与优化器，再训练 20 epochs，每轮 500 次更新；累计 11500 次优化。
- 首轮结果单独保存在 `outputs/dp_pilot_20260915/stage_1500`，最终结果位于上一级目录。
- 学习率 1e-4，100 步 warmup，之后保持常数；每 5 个 epoch 全量计算三个留出采集的验证 loss。
- 推理使用完整 100 步 DDPM 去噪。安装的旧版调度器按 t-1 更新，不能直接跳到每隔 5 步；首轮 20 步结果仅保留作问题诊断。
- EMA 模型用于离线预测；训练日志中的验证 loss 由官方工作区的普通训练模型计算。
- 验证 loss 最优的两个 checkpoint、latest 和 final 均保存。本轮不是收敛或真机成功率认证。

官方工作区在截断每轮步数时会额外递增 epoch 边界的 global_step；页面与导出曲线按唯一训练
batch 记录统计实际优化次数，最终报告同时保留 trainer global_step 和 optimizer_updates。

## 可复现入口

```bash
cd /home/descfly/lcl_work/Convert_data
.runtime/bin/python -m pip install --cache-dir .cache/pip -r requirements-pilot.txt

# 一次完成筛选、转换、校验、预览、训练、评估和视频输出。
DP_WORKERS=4 \
DP_PYTHON=/home/descfly/miniforge3/envs/robodiff5090/bin/python \
  bash scripts/run_dp_pilot.sh

# 转换已完成时从校验和训练开始；已有训练完成标记时复用 final checkpoint。
bash scripts/run_dp_pilot.sh --train-only
```

首轮完成后，复现本次额外 10000 次更新（直接训练入口会执行新训练；重复运行此段会继续增加更新）：

```bash
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export CONVERT_ROOT=/home/descfly/lcl_work/Convert_data
export DP_DATASET="$CONVERT_ROOT/outputs/dp/pilot_high_30hz"
export PYTHONPATH="$CONVERT_ROOT/src:$CONVERT_ROOT/../diffusion_policy"
export XDG_CACHE_HOME="$CONVERT_ROOT/.cache"
export TORCH_HOME="$CONVERT_ROOT/.cache/torch" CUDA_CACHE_PATH="$CONVERT_ROOT/.cache/cuda"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
/home/descfly/miniforge3/envs/robodiff5090/bin/python -B scripts/train_dp.py \
  --config-name train_franka_pilot training.num_epochs=20 training.max_train_steps=500 \
  "+resume_checkpoint=$CONVERT_ROOT/outputs/dp_training/pilot_high_30hz/checkpoints/final.ckpt"
# 已有完成标记，以下命令复用模型并重新生成评估与演示。
bash scripts/run_dp_pilot.sh --train-only
```

数据集：`outputs/dp/pilot_high_30hz`。
模型与逐步 loss：`outputs/dp_training/pilot_high_30hz`。
展示文件：`outputs/dp_pilot_20260915`。

## 过程页面

```bash
.runtime/bin/python -B scripts/dp_pilot_dashboard.py
```

打开 <http://127.0.0.1:8765>，可查看筛选名单、缓存/转换进度、训练曲线、数据预览和预测视频。
服务只绑定本机回环地址，不对外发布数据。

## 演示如何解读

`data_preview.mp4` 是原始录制相机的片段回放。
`prediction_demo.mp4` 上方为留出 episode 的真实录制画面，下方为同一观察时刻模型预测的
8 步动作和录制目标动作对比。上下内容均有明确标签；视频按抽取的验证上下文播放，
屏幕上的时间是原录制时间，不等于视频播放时间。

评估在每条留出采集的整个有效窗口均匀取最多 96 个上下文，计算 54 维动作 MAE/RMSE（弧度），
同时比较“保持当前关节位置”和“训练集平均动作”两个基线。验证集也参与训练过程的模型选择，
因此这里称为留出验证评估，不称为独立测试。离线误差不能推导真机闭环任务成功率。
