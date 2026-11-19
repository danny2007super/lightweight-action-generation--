# 轻量动作生成模型：ACT 与 Flow Matching 单臂对照

在 Dobot 双机械臂平台上做单臂试管抓取、放置任务。实体左臂执行动作，另一臂保持固定。使用同一批示教、两路 RGB 相机和相同的 30 步动作窗口，对比原版 ACT 架构与轻量 Flow Matching 实现。

## 当前结果

| 方法 | 真机成功次数 | 观察 |
| --- | ---: | --- |
| ACT | 6/10 | 在这 10 次试验中较稳定 |
| Flow Matching | 4/10 | 曾出现采样抖动及间歇停顿 |

成功次数由现场操作者报告。10 次试验样本很小；这里不据此断言模型的普遍优劣。两种方法的控制、采样和平滑设置并不完全相同，因此这也是系统级对照，而非只改变网络结构的严格消融实验。

## 数据与模型

- 原始示教为 HDF5，共 44 条；人工剔除 episode 21、29，余 42 条，按 episode 分成 34 条训练、8 条验证。
- 观测为 `left`、`top` 两路 480×640 RGB 图像与 14 维状态；动作也是 14 维。前 7 维对应固定侧臂，后 7 维对应任务左臂的 6 个关节和夹爪。夹爪约定 0 为闭合、1 为张开。约 15 Hz。
- ACT 使用原版 CVAE/Transformer 策略，ResNet18 视觉骨干，20,000 次训练更新；保存的最佳检查点来自第 17,750 次更新。FM 使用高斯噪声到动作的线性路径、速度场回归与 Euler 积分，并沿用 ACT 的示教划分和归一化统计。FM 训练至 20,000 次更新，验证指标最低的检查点在第 5,000 次更新。
- ACT 的 L1+KL 损失与 FM 的 flow MSE 定义不同，数值不可直接比较。真机成功率是更贴近任务的指标。

## 真机控制与卡顿记录

`ACT真机评估准备/start_act_live.py` 是共享的实时控制入口：每个控制周期读取最新双相机画面和机器人状态，预测 30 步动作块，按时间对重叠动作块加权，再用 `ServoJ` 下发当前目标。启动窗口按 `S` 开始，`Q` 停止；抓取后松开可自动收尾。`ACT_POLICY=act` 或 `ACT_POLICY=fm` 选择模型。

FM 的采样从噪声开始。此前每次推理重新抽样，现场观察到抖动；当前实现默认固定 `FLOW_SEED=0`，并使用动作块时间加权、目标 EMA（默认 α=0.7）。设置 `FLOW_SEED=-1` 可恢复随机采样。固定种子只能去掉一类采样波动，并不能保证策略轨迹正确。

此前间歇停顿与控制周期超出 `ServoJ` 插值窗口有关。当前代码为两路相机各设后台读取线程，控制循环只取最新帧，并根据实际控制周期调整 `ServoJ` 插值时间。**现场报告的 4/10 成功率及卡顿发生在调试过程中；相机线程修改后的完整成功率和延迟分布尚未在这里重新统计。**

## 目录

| 目录 | 内容 |
| --- | --- |
| `ACT训练准备/` | ACT 数据适配、训练和原版源码；`upstream/` 固定到原版 ACT 提交 `742c753` |
| `FlowMatching准备/` | FM 模型、训练和纯推理接口 |
| `ACT真机评估准备/` | ACT 纯推理与 ACT/FM 共用的真机控制入口 |
| `robot_bridge/` | 真机入口需要的 Dobot/Feetech 驱动最小子集 |

## 运行前准备

本仓库只发布源码、配置和少量归一化统计。原始 HDF5、相机画面、真机日志、Python 环境和模型权重均未上传。ACT `best.pt` 约 960 MB，超出普通 GitHub 单文件限制；需要由数据与模型持有人自行放到 `ACT训练准备/runs/run_001/best.pt`。FM `best.pt` 放到 `FlowMatching准备/runs/run_fullres_resume_002/best.pt`。权重与本仓库的配置及归一化统计必须成套使用。

Python 3.11 环境需要 PyTorch、torchvision、NumPy、h5py、einops、IPython、OpenCV、pyrealsense2、pyserial。GPU 训练和推理记录使用 PyTorch 2.7.1 + CUDA 12.8、torchvision 0.22.1。安装命令应按目标机器的 CUDA 与系统版本选用官方对应包。

真机入口还需设置相机序列号：`ACT_LEFT_CAMERA_SERIAL`、`ACT_TOP_CAMERA_SERIAL`。可按机器设置 `DOBOT_LEFT_IP`、`DOBOT_RIGHT_IP`、`DOBOT_LEFT_GRIPPER_PORT`；默认控制器地址为 `192.168.5.2`、`192.168.5.1`，左夹爪端口为 `COM5`。配置应核对相机方向、关节顺序、夹爪映射、急停与工作区后再运行。`启动ACT真机.cmd` 和 `启动FM真机.cmd` 调用当前环境的 `python`。

## 来源与许可

- ACT 原版：[tonyzhaozh/act](https://github.com/tonyzhaozh/act)，固定提交 `742c753c0d4a5d87076c8f69e5628c79a8cc5488`，原许可证见 `ACT训练准备/upstream/LICENSE`。
- `robot_bridge/` 来源于 Dobot 代码，MIT 许可及第三方声明见该目录的 `LICENSE`、`THIRD-PARTY-LICENSES`。
- 本目录的 FM 是围绕当前 RGB HDF5 数据实现的轻量对照模型，不是 FlowPolicy 原版模型。
