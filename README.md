# StreamVGGT__mixed

本仓库是基于 StreamVGGT 的 RGB+Event 训练分支，当前已支持：
- Event token 融合（cross-attention）
- DL3DV Screen+Event 训练
- random / sequence / mixed 三种采样模式
- 混合采样比例可配（`sequence_ratio`）
- 多卡分布式训练（Accelerate）

## 目录概览

- `src/train.py`：主训练入口（Hydra + Accelerate）
- `config/train_dl3dv.yaml`：DL3DV 训练配置
- `src/dust3r/datasets/dl3dv.py`：`DL3DV_ScreenEvent_Multi` 数据集实现
- `src/inference_with_event.py`：RGB+Event 推理脚本
- `demo_gradio.py`：Gradio Demo
- `ckpt/`：默认权重目录
- `checkpoints/`：训练输出目录

## 环境安装

在仓库根目录执行：

```bash
pip install -r requirements.txt
```

## 数据格式（DL3DV_ScreenEvent_Multi）

每个序列目录需要至少包含：

```text
<SEQ_ROOT>/
  images/
    000000.png
    000001.png
    ...
  events/
    000000.pt
    000001.pt
    ...
```

要求：
- `images` 与 `events` 通过同名 stem 对齐
- `events/*.pt` 通道数应匹配 `event_in_chans`（默认 8）

## 训练配置要点

`config/train_dl3dv.yaml` 中常用字段：
- `fusion`, `use_event`, `event_in_chans`
- `num_views`, `num_views_min`, `num_views_max`
- `sample_mode`: `sequence | random | mixed`
- `sequence_ratio`: 在 `mixed` 模式下采到 sequence 的概率

混合比例计算：
- `sequence_ratio = p(sequence)`
- `random_ratio = 1 - sequence_ratio`

示例：
- `sample_mode=mixed sequence_ratio=0.7` 表示约 70% sequence + 30% random

## 训练命令

先进入源码目录：

```bash
cd /share/magic_group/aigc/fcr/EventVGGT/data/StreamVGGT__mixed/src
```

单卡：

```bash
python ./train.py --config-name train_dl3dv
```

4 卡（4,5,6,7）：

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 HYDRA_FULL_ERROR=1 \
accelerate launch --multi_gpu --num_processes 4 --main_process_port 29667 \
./train.py --config-name train_dl3dv
```

4 卡 + 混合采样：

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 HYDRA_FULL_ERROR=1 \
accelerate launch --multi_gpu --num_processes 4 --main_process_port 29667 \
./train.py --config-name train_dl3dv sample_mode=mixed sequence_ratio=0.7
```

## 推理命令（RGB+Event）

```bash
python /share/magic_group/aigc/fcr/EventVGGT/data/StreamVGGT__mixed/src/inference_with_event.py \
  --checkpoint /share/magic_group/aigc/fcr/EventVGGT/data/StreamVGGT__mixed/checkpoints/vggt_train_dl3dv_2026_03_14_dino/checkpoint-last.pth \
  --data_root <sequence_root_with_images_and_events> \
  --output <output_dir> \
  --fusion crossattn \
  --event_in_chans 8
```

## 训练日志与可视化

- tqdm 进度条显示在主进程终端
- TensorBoard 日志默认写入 `${output_dir}/logs`

## 常见排查

- 训练卡在 DataLoader：先用 `num_workers=0` 验证数据可读
- 多卡无负载：确认 `CUDA_VISIBLE_DEVICES` 与 `--num_processes` 匹配
- Event 通道报错：检查 `event_in_chans` 与 `.pt` 实际通道一致
