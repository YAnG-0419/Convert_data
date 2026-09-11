# ROS Bag 转 ACT HDF5 数据格式说明

本文档用于把番茄采摘任务的 ROS2 bag 原始数据转换为当前 ACT 训练代码可读取的 HDF5 episode 数据。

## 目标目录结构

转换完成后，数据集目录建议如下：

```text
tomato_act_dataset/
  episode_0.hdf5
  episode_1.hdf5
  episode_2.hdf5
  ...
```

每个 `.hdf5` 文件对应一次完整示教 episode。

## HDF5 内部结构

每个 episode 文件应包含：

```text
episode_0.hdf5
  attrs:
    sim = False

  /observations/qpos
    shape: (T, state_dim)
    dtype: float32 或 float64

  /observations/qvel
    shape: (T, state_dim)
    dtype: float32 或 float64

  /observations/images/head
    shape: (T, 240, 320, 3)
    dtype: uint8
    color: RGB

  /observations/images/left_wrist
    shape: (T, 240, 320, 3)
    dtype: uint8
    color: RGB

  /observations/images/right_wrist
    shape: (T, 240, 320, 3)
    dtype: uint8
    color: RGB

  /action
    shape: (T, action_dim)
    dtype: float32 或 float64
```

## 推荐转换参数

```text
采样频率: 20Hz
图像尺寸: 320x240
图像格式: RGB uint8
相机数量: 3 路
裁剪: 每个 rosbag 首尾各裁掉 1 秒
```

图像处理建议：

- `cam0/head`: 原始 `640x400`，先上下 padding 到 `640x480`，再 resize 到 `320x240`。
- `cam1/left_wrist`: 原始 `640x480`，直接 resize 到 `320x240`。
- `cam2/right_wrist`: 原始 `640x480`，直接 resize 到 `320x240`。
- 原始图像编码为 `rgb8` 时，不需要 BGR/RGB 通道交换。
- 三路相机最终尺寸必须一致，否则 ACT dataloader 无法 stack 多相机图像。

## ROS Topic 映射

根据当前 `episode103` 数据，建议使用以下映射：

```yaml
images:
  head: /cam0/color/image_raw
  left_wrist: /cam1/color/image_raw
  right_wrist: /cam2/color/image_raw

arm_state:
  left: /left/franka/joint_states
  right: /right/franka/joint_states

arm_action:
  topic: /teleop/validated_arm_commands

hand_state:
  left: /teleop/wuji/left/joint_states
  right: /teleop/wuji/right/joint_states

hand_action:
  left: /teleop/wuji/left/command
  right: /teleop/wuji/right/command
```

## 状态和动作维度

如果最终机器人需要控制双 Franka 机械臂和双手，推荐使用 54 维状态与 54 维动作。

```text
qpos = left_arm_state(7) + right_arm_state(7) + left_hand_state(20) + right_hand_state(20)
state_dim = 54

action = validated_arm_commands(14) + left_hand_command(20) + right_hand_command(20)
action_dim = 54
```

如果最终执行端只控制双臂，不控制手部，则可以先使用：

```text
qpos = left_arm_state(7) + right_arm_state(7)
state_dim = 14

action = validated_arm_commands(14)
action_dim = 14
```

但对于“识别番茄、抓起采摘、放到盘子里”的任务，只使用 14 维双臂动作大概率不足，因为抓取和采摘动作需要手部或夹爪控制信号。

## 固定关节顺序

不能每帧直接按 ROS `JointState.name` 的原始顺序拼接，因为不同 topic 或不同帧里的 name 顺序可能不一致。转换时必须按照固定顺序重排。

推荐双臂顺序：

```text
left_arm:
  left_fr3_joint1
  left_fr3_joint2
  left_fr3_joint3
  left_fr3_joint4
  left_fr3_joint5
  left_fr3_joint6
  left_fr3_joint7

right_arm:
  right_fr3_joint1
  right_fr3_joint2
  right_fr3_joint3
  right_fr3_joint4
  right_fr3_joint5
  right_fr3_joint6
  right_fr3_joint7
```

`/teleop/validated_arm_commands` 中的 action 名称为：

```text
left_fr3v2_joint1
left_fr3v2_joint2
left_fr3v2_joint3
left_fr3v2_joint4
left_fr3v2_joint5
left_fr3v2_joint6
left_fr3v2_joint7
right_fr3v2_joint1
right_fr3v2_joint2
right_fr3v2_joint3
right_fr3v2_joint4
right_fr3v2_joint5
right_fr3v2_joint6
right_fr3v2_joint7
```

手部 20 维可按消息里的 `name` 顺序固定保存，但必须保证所有 episode 使用完全相同的顺序。

## 时间对齐要求

每个训练 timestep 应满足：

```text
time_t:
  image_t
  qpos_t
  action_t
```

推荐做法：

1. 建立统一的 `20Hz` 时间轴。
2. 对每个时间点，查找距离该时间点最近的相机帧、关节状态、动作命令。
3. 超过最大允许时间差的 timestep 直接丢弃。
4. 每个 episode 最好保留连续片段，避免中间出现大断点。

推荐最大时间差：

```text
图像: <= 50ms
arm state/action: <= 20ms
hand state/action: <= 50ms
```

如果一个 episode 中间出现超过阈值的大空洞，建议切分 episode 或丢弃异常片段。

## qvel 处理

`/observations/qvel` 必须存在。

推荐优先级：

1. 如果 ROS `JointState.velocity` 有有效值，按固定关节顺序保存。
2. 如果 velocity 缺失或为空，用 qpos 按时间差做有限差分估计。
3. 如果暂时不使用速度，也可以填全 0，但 shape 必须是 `(T, state_dim)`。

当前 ACT 训练代码主要使用 `qpos` 和 `action`，但 `qvel` 在 HDF5 schema 中仍建议保留。

## 必须检查项

每个生成的 HDF5 至少确认：

```python
qpos.shape == (T, 54)
qvel.shape == (T, 54)
action.shape == (T, 54)

images/head.shape == (T, 240, 320, 3)
images/left_wrist.shape == (T, 240, 320, 3)
images/right_wrist.shape == (T, 240, 320, 3)

images/head.dtype == uint8
images/left_wrist.dtype == uint8
images/right_wrist.dtype == uint8

root.attrs["sim"] == False
```

如果先做 14 维双臂版本，则检查：

```python
qpos.shape == (T, 14)
qvel.shape == (T, 14)
action.shape == (T, 14)
```

低维数据不能出现：

```text
NaN
inf
突然的大跳变
全 0 action
左右臂顺序混乱
手部顺序混乱
```

图像数据不能出现：

```text
黑屏
花屏
RGB/BGR 颜色反转
尺寸不一致
帧与动作明显错位
```

## 建议先转换的小规模测试集

正式全量转换前，建议先转换 5-10 个 episode：

```text
tomato_act_dataset_test/
  episode_0.hdf5
  episode_1.hdf5
  ...
```

确认以下事项后再全量转换：

- ACT dataloader 能读取。
- 可视化视频正常。
- qpos/action 曲线连续。
- 训练 loss 能正常下降。
- 手部/夹爪动作与画面中的抓取动作对齐。

## 最终推荐

优先推荐：

```text
3 路相机
320x240
20Hz
首尾各裁掉 1 秒
state_dim = 54
action_dim = 54
```

这个配置在数据量、训练速度、视觉细节和任务完整性之间比较平衡，适合“识别番茄、抓起采摘、放到盘子里”的目标。
