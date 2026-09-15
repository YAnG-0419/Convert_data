#!/usr/bin/env python3
"""Write the pilot report from completed conversion, training and evaluation artifacts."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'outputs/dp_pilot_20260915'
TRAIN=ROOT/'outputs/dp_training/pilot_high_30hz'


def read(path):
    return json.loads(path.read_text())


def main():
    done=read(TRAIN/'training_complete.json')
    result=read(ART/'evaluation.json')
    if result['checkpoint_global_step']!=done['global_step']:
        raise SystemExit('Evaluation does not match the final checkpoint')
    first=read(ART/'stage_1500/inference_100/evaluation.json')
    verified=read(ART/'verification.json')
    logs=[json.loads(x) for x in (TRAIN/'logs.json.txt').read_text().splitlines()]
    vals=[r for r in logs if 'val_loss' in r]
    mae=result['summary']['diffusion_policy']['all_joints']['mae_rad']
    old=first['summary']['diffusion_policy']['all_joints']['mae_rad']
    hold=result['summary']['hold_current_position']['all_joints']['mae_rad']
    avg=result['summary']['training_mean_action']['all_joints']['mae_rad']
    plan=read(ART/'experiment.json')
    plan.update(status='complete',optimizer_updates=done['optimizer_updates'],trainer_global_step=done['global_step'],
                final_action_mae_rad=mae,last_validation_loss=vals[-1]['val_loss'],
                last_validation_epoch=vals[-1]['epoch'],final_checkpoint=result['checkpoint'])
    (ART/'experiment.json').write_text(json.dumps(plan,indent=2)+'\n')
    text=f'''# Franka 双臂 + Wuji 双手：DP 训练实验结果

## 已完成

从 262 条采集中，按采集校验、已有人工优秀清单及时间连续性筛选三批，共 12 条。
三批合并训练一个模型，固定留出每批一条验证。

| 批次 | 训练 episode | 验证 episode |
|---|---|---|
| 1 | 404、409、417 | 419 |
| 2 | 439、444、456 | 457 |
| 3 | 459、469、474 | 483 |

- 转换完成：{verified['bags']} 个原始采集、{verified['frames']:,} 帧、30 Hz；完整图像读取校验通过。
- 观测：三路 RGB、头部深度、54 维关节位置；动作：54 维绝对关节目标，单位弧度。
- 图像保存为 320×240，训练编码器输入缩放到 160×120。
- 训练使用同级官方 diffusion_policy 的工作区与策略；RTX 5090；9 条训练、3 条验证。
- 首轮 1,500 次更新，随后恢复模型及优化器，追加 10,000 次；总计 **{done['optimizer_updates']:,} 次更新**。
- 4 个独立 ResNet18 编码器、GroupNorm、U-Net 通道 128/256/512、batch size 16、学习率 1e-4、EMA。
- 完整数据、代码、模型和输出均在 `Convert_data` 中；官方 DP 仓库和原始采集保持原状。

## 训练与动作评估

验证噪声预测 MSE 从 **{vals[0]['val_loss']:.4f}** 降至最近一次 **{vals[-1]['val_loss']:.4f}**
（epoch {vals[-1]['epoch']}；每 5 个 epoch 验证一次，最终 epoch 为 {done['epoch']-1}）。
该 loss 来自普通训练模型；以下动作评估来自最终 EMA 模型。

在 3 条验证采集中均匀抽取 {result['contexts']} 个完整上下文，每次使用 2 帧观测，
预测 8 帧目标动作（30 Hz，约 0.267 秒），使用 {result['inference_steps']} 步去噪，固定种子 {result['seed']}。

| 方法 | 全部 54 关节 MAE（rad，越小越好） |
|---|---:|
| 首轮 DP，1,500 次更新，100 步去噪 | {old:.5f} |
| 最终 DP，{done['optimizer_updates']:,} 次更新 | {mae:.5f} |
| 保持当前关节位置 | {hold:.5f} |
| 训练集平均动作 | {avg:.5f} |

最终 DP 相比首轮的 MAE 下降 **{(1-mae/old)*100:.1f}%**。
'''
    if mae>hold:
        text+='但动作误差仍高于“保持当前位置”基线，当前模型尚未展示优于短时间动作惯性的预测能力。\n'
    else:
        text+='动作误差低于“保持当前位置”基线；仍需更多场景与闭环执行评估。\n'
    text+='\n| 部位 | DP MAE（rad） | 保持当前位置 MAE（rad） |\n|---|---:|---:|\n'
    for key,label in [('left_arm','左臂'),('right_arm','右臂'),('left_hand','左手'),('right_hand','右手')]:
        text+=f"| {label} | {result['summary']['diffusion_policy'][key]['mae_rad']:.5f} | {result['summary']['hold_current_position'][key]['mae_rad']:.5f} |\n"
    diagnostic=ART/'stage_11500_steps20/evaluation.json'
    if diagnostic.exists():
        diag=read(diagnostic)
        text+=f"\n额外诊断：同一最终模型原先采用 20 步去噪时，MAE 为 {diag['summary']['diffusion_policy']['all_joints']['mae_rad']:.5f} rad；详见 [原 20 步评估](stage_11500_steps20/evaluation.json)。当前环境旧版 DDPM 按 t-1 计算，跳步不匹配；主表和最终视频已修正为完整 100 步配置，模型权重未变。\n"
    text+='''
## 查看过程与演示

- [训练和演示页面](http://127.0.0.1:8765)
- [最终预测演示视频](prediction_demo.mp4)
- [原始数据预览视频](data_preview.mp4)
- [训练曲线](training_curves.png)
- [分部位动作误差图](evaluation_errors.png)
- [首轮原始 20 步演示，仅作问题诊断](stage_1500/prediction_demo.mp4)
- [逐条采集评估指标](evaluation.json)
- [选择依据与时间连续性评分](selection.json)

视频上方是录制相机画面，下方是模型动作预测与录制目标的对照；抽样播放时间与原录制时间不同，
以画面中 recorded t 标签为准。视频不包含机器人执行预测动作的真机闭环演示。
验证数据也用于训练过程监测，因此这些指标不是独立测试集成绩，不能换算为任务成功率。

## 文件与复现

- 数据集：`outputs/dp/pilot_high_30hz/replay_buffer.zarr`
- 最终模型：`outputs/dp_training/pilot_high_30hz/checkpoints/final.ckpt`
- 首轮模型：`outputs/dp_training/pilot_high_30hz/checkpoints/final_1500.ckpt`
- 全部逐步日志：`outputs/dp_training/pilot_high_30hz/logs.json.txt`
- 批处理与训练入口：`scripts/run_dp_pilot.sh`
- 参数与继续训练说明：`docs/dp_pilot_experiment.md`

以上相对路径均相对于 `/home/descfly/lcl_work/Convert_data`。
'''
    (ART/'report.md').write_text(text)
    print(text)


if __name__=='__main__':main()
