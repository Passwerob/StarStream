# StreamVGGT — RGB + Event Fusion (FSDP_inject Branch)

本分支在 [StreamVGGT](https://github.com/wzzheng/StreamVGGT) 基础上引入 **事件相机 (Event Camera)** 数据融合，实现 RGB+Event 联合 4D 视觉几何感知。核心改动包括：Event Patch Embedding、Cross-Attention 融合模块、混合采样策略、课程学习 (Curriculum Learning)、以及基于 **FSDP (Fully Sharded Data Parallel)** 的分布式训练。

---

## 目录

- [项目概览](#项目概览)
- [目录结构](#目录结构)
- [环境安装](#环境安装)
- [模型架构](#模型架构)
- [数据准备](#数据准备)
- [训练配置详解](#训练配置详解)
- [训练命令](#训练命令)
- [推理命令](#推理命令)
- [评测](#评测)
- [Demo](#demo)
- [常见问题排查](#常见问题排查)
- [致谢](#致谢)

---

## 项目概览

**StreamVGGT** 是一种因果 Transformer 架构，用于实时流式 4D 视觉几何感知。与需要重新处理整个序列的离线模型不同，StreamVGGT 利用时序因果注意力和缓存 Memory Token 支持高效的增量在线重建。

本分支 (`FSDP_inject`) 在此基础上新增以下能力：

| 特性 | 说明 |
|------|------|
| **Event 融合** | 通过 `EventPatchEmbed` + `CrossAttnFuse` 将事件体素 (event voxel) 注入 RGB 特征流 |
| **融合模式** | `none`（纯 RGB）/ `crossattn`（跨注意力融合） |
| **混合采样** | `sequence` / `random` / `mixed` 三种帧采样模式，`mixed` 模式下可通过 `sequence_ratio` 控制比例 |
| **课程学习** | `CurriculumMixDataset` 支持模拟+真实数据渐进式混合训练 |
| **FSDP 训练** | 强制使用 Accelerate FSDP + BF16 混合精度 |
| **知识蒸馏** | 以 VGGT 为教师模型，`DistillLoss` 对齐学生输出（相机位姿、深度、点云） |
| **多数据集支持** | DL3DV（模拟 Screen+Event）、M3ED（真实事件数据）、以及 15+ 原始 StreamVGGT 数据集 |

---

## 目录结构

```
StreamVGGT_mixed_FSDP_inject/
├── README.md                          # 本文件
├── requirements.txt                   # 训练依赖
├── requirements_demo.txt              # Demo 依赖
├── demo_gradio.py                     # Gradio Demo（纯 RGB）
├── LICENSE.txt
│
├── config/                            # Hydra 训练配置
│   ├── train.yaml                     #   原始多数据集蒸馏配置（无 Event）
│   ├── finetune.yaml                  #   监督微调配置（ConfLoss + FinetuneLoss）
│   ├── train_dl3dv.yaml               #   DL3DV 单数据集，可训练 DINO
│   ├── train_dl3dv_fsdp.yaml          #   DL3DV + FSDP，固定帧长
│   ├── train_M3ed_fsdp.yaml           #   M3ED 单数据集
│   ├── train_M3ed_fsdp_test.yaml      #   M3ED 冒烟测试（极小数据量）
│   ├── train_M3ed_curriculum.yaml     #   DL3DV+M3ED 课程学习 v1
│   └── train_M3ed_curriculum_v2.yaml  #   DL3DV+M3ED 课程学习 v2（推荐）
│
├── ckpt/                              # 预训练/教师权重目录
│   └── model.pt                       #   VGGT 预训练 checkpoint
│
├── checkpoints/                       # 训练产物（自动生成）
│
├── src/
│   ├── train.py                       # 主训练入口（Hydra + Accelerate FSDP）
│   ├── train.sh                       # 训练启动脚本
│   ├── finetune.py                    # 微调入口
│   ├── distill.py                     # 蒸馏训练入口
│   ├── inference_with_event.py        # RGB+Event 推理脚本
│   ├── inference.sh                   # 推理 + 点云可视化一键脚本
│   │
│   ├── streamvggt/                    # StreamVGGT 模型实现
│   │   └── models/
│   │       ├── streamvggt.py          #   模型封装（Aggregator + heads）
│   │       └── aggregator.py          #   核心：EventPatchEmbed, CrossAttnFuse, 交替注意力
│   │
│   ├── vggt/                          # VGGT 教师模型
│   │   └── models/
│   │       └── vggt.py
│   │
│   ├── dust3r/                        # 数据集、损失、推理工具
│   │   ├── datasets/
│   │   │   ├── dl3dv.py               #   DL3DV_Multi, DL3DV_ScreenEvent_Multi
│   │   │   └── base/
│   │   │       └── easy_dataset.py    #   CurriculumMixDataset 等基础设施
│   │   ├── losses.py                  #   DistillLoss, ConfLoss, FinetuneLoss
│   │   └── inference.py               #   loss_of_one_batch（学生-教师对比）
│   │
│   ├── croco/                         # 预训练 backbone 工具
│   │
│   ├── eval/                          # 评测脚本
│   │   ├── monodepth/                 #   单目深度评测
│   │   ├── video_depth/               #   视频深度评测
│   │   ├── mv_recon/                  #   多视图重建评测
│   │   └── pose_evaluation/           #   相机位姿评测（Co3D 等）
│   │
│   ├── eval_benchmark/                # 事件/RGB 重建 benchmark 评测 CLI
│   │
│   ├── visual_util.py                 # 可视化工具
│   ├── see_pointcloud.py              # 点云查看器
│   └── web_viewer/                    # Web 3D 查看器
│
├── assets/                            # 图片资源
├── cloud_opt/                         # 点云优化
├── datasets_preprocess/               # 数据预处理脚本
├── examples/                          # 示例数据
└── lib/                               # 第三方库
```

---

## 环境安装

### 1. 创建 conda 环境

```bash
conda create -n StreamVGGT python=3.11 cmake=3.14.0
conda activate StreamVGGT
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
conda install 'llvm-openmp<16'
```

核心依赖版本：

| 包 | 版本 |
|---|------|
| torch | 2.3.1 |
| torchvision | 0.18.1 |
| numpy | 1.26.1 |
| accelerate | latest |
| hydra-core | latest |
| transformers | latest |

### 3. 下载预训练权重

将 VGGT 预训练模型放到 `ckpt/model.pt`。

- 教师模型：[VGGT 预训练权重](https://github.com/facebookresearch/vggt)
- StreamVGGT 权重：见 [Hugging Face](https://huggingface.co/) 或 [清华云盘](https://cloud.tsinghua.edu.cn/)

---

## 模型架构

### 整体流程

```
输入序列: [RGB 图像 + Event Voxel] × N 帧
         ↓
   ┌─────────────────────────────────────┐
   │        DINOv2 Patch Embed           │  ← RGB 图像 patch 化
   │        (dinov2_vitl14_reg)          │
   └──────────────┬──────────────────────┘
                  │
   ┌──────────────▼──────────────────────┐
   │      EventPatchEmbed (Conv)         │  ← Event Voxel patch 化
   │      event_in_chans → embed_dim     │
   └──────────────┬──────────────────────┘
                  │
   ┌──────────────▼──────────────────────┐
   │        CrossAttnFuse (MHA)          │  ← Q=RGB tokens, K/V=Event tokens
   │   output = RGB + Residual(Attn)     │     残差连接回 RGB patch tokens
   └──────────────┬──────────────────────┘
                  │
   ┌──────────────▼──────────────────────┐
   │     Frame/Global 交替注意力          │  ← VGGT 式因果 Transformer
   │     (带 Camera/Register Token)      │
   └──────────────┬──────────────────────┘
                  │
         ┌───────┼───────┬───────┐
         ↓       ↓       ↓       ↓
      Camera   Depth   Pts3D   Track
       Head    Head    Head    Head
```

### Event 融合模块详解

| 组件 | 说明 |
|------|------|
| **`EventPatchEmbed`** | 将 event voxel（默认 8 通道）通过 Conv2d 映射为 patch tokens，后接 LayerNorm |
| **`CrossAttnFuse`** | 多头交叉注意力：RGB patch tokens 作为 Query，Event tokens 作为 Key/Value，输出通过残差加回 RGB |
| **`event_proj`** | 可选的线性投影层，将 Event token 投影到与 RGB token 相同的维度 |

融合方式由 `fusion` 参数控制：
- **`none`**：不使用 Event 数据，等价于原始 StreamVGGT
- **`crossattn`**：启用 Cross-Attention 融合（推荐）

### 训练范式：知识蒸馏

- **学生模型**：`StreamVGGT`（带 Event 融合分支）
- **教师模型**：`VGGT`（冻结，纯 RGB，不读 Event）
- **损失函数**：`DistillLoss`，对齐学生与教师的 `camera_pose`、`depth`、`pts3d`、`confidence` 等输出

### 参数冻结策略

| 参数 | 是否冻结 | 说明 |
|------|----------|------|
| `aggregator.patch_embed` (DINOv2) | 默认冻结 | 设置 `train_dino=True` 可解冻（以 `dino_lr_scale` 缩放学习率） |
| `aggregator.camera_token` | 冻结 | — |
| `aggregator.register_token` | 冻结 | — |
| Event 融合模块 | 可训练 | `EventPatchEmbed`, `CrossAttnFuse`, `event_proj` |
| 各预测 Head | 可训练 | Camera, Depth, Pts3D, Track heads |

---

## 数据准备

### DL3DV_ScreenEvent_Multi 格式

训练使用的核心数据集为 `DL3DV_ScreenEvent_Multi`，每个序列目录结构如下：

```
<SEQUENCE_ROOT>/
├── images/           # RGB 图像
│   ├── 000000.png
│   ├── 000001.png
│   └── ...
└── events/           # 事件体素（与图像同名对齐）
    ├── 000000.pt     # torch.Tensor, shape = (event_in_chans, H, W)
    ├── 000001.pt
    └── ...
```

**要求**：
- `images/` 与 `events/` 通过文件名 stem（不含扩展名）一一对齐
- `events/*.pt` 的通道数必须与配置中 `event_in_chans` 一致（默认 8）
- 图像支持 `.png` / `.jpg` / `.jpeg` 格式
- 事件体素如果空间分辨率与图像不同，推理/训练时会自动双线性插值对齐

**Event Voxel 预处理**（训练和推理时均会执行）：
1. 将 NaN / Inf 替换为 0
2. `clamp(-50, 50)` 截断极端值
3. 逐帧标准化：`(evt - mean) / std`（若 `std < 1e-6` 则置零）

### 支持的数据集

| 数据源 | 类型 | 对应配置 |
|--------|------|----------|
| **DL3DV** | 模拟 Screen+Event | `train_dl3dv.yaml`, `train_dl3dv_fsdp.yaml` |
| **M3ED** | 真实事件相机 | `train_M3ed_fsdp.yaml` |
| **DL3DV + M3ED** | 课程混合 | `train_M3ed_curriculum.yaml`, `train_M3ed_curriculum_v2.yaml` |
| Co3D, ScanNet, ARKitScenes, ... | 原始 StreamVGGT 数据 | `train.yaml` |

---

## 训练配置详解

所有配置文件位于 `config/` 目录，使用 Hydra 管理。以下逐一说明关键配置。

### 核心参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `fusion` | `none` / `crossattn` | Event 融合方式 |
| `use_event` | bool | 是否启用 Event 输入 |
| `event_in_chans` | int | Event voxel 通道数（默认 8） |
| `event_patch_size` | int / Null | Event patch 大小，默认同 `patch_size`（14） |
| `train_dino` | bool | 是否解冻 DINOv2 patch_embed |
| `dino_lr_scale` | float | DINOv2 参数学习率缩放因子（默认 0.05） |

### 帧采样参数

| 参数 | 说明 |
|------|------|
| `num_views` | 每 batch 的最大视图数 |
| `num_views_min` / `num_views_max` | 动态视图数范围（`fixed_length=False` 时生效） |
| `sample_mode` | `sequence`（按序连续帧）/ `random`（随机帧）/ `mixed`（混合） |
| `sequence_ratio` | `mixed` 模式下选择 sequence 采样的概率 |
| `min_interval` / `max_interval` | 连续帧采样时的帧间隔范围 |
| `fixed_length` | 是否固定每个 batch 的帧数 |

**混合采样计算**：
- `sequence_ratio = p(sequence)`
- `random_ratio = 1 - sequence_ratio`
- 例如 `sample_mode=mixed sequence_ratio=0.75` 表示约 75% sequence + 25% random

### 课程学习参数（CurriculumMixDataset）

| 参数 | 说明 |
|------|------|
| `dataset_a` / `dataset_b` | A = 模拟数据（DL3DV），B = 真实数据（M3ED） |
| `curriculum_total_size` | 每 epoch 的总采样数 |
| `curriculum_warmup_epochs` | warmup 阶段的 epoch 数 |
| `curriculum_initial_ratio_b` | warmup 阶段 B 的初始比例（v2 新增，默认 0.05） |
| `curriculum_final_ratio_b` | 训练结束时 B 的目标比例（默认 0.8） |
| `curriculum_min_ratio_a` | A 的最低比例下限（默认 0.2） |

**课程策略**：
1. Warmup 阶段（epoch < `warmup_epochs`）：B 比例从 `initial_ratio_b` 线性增至起始值
2. Warmup 后：B 比例线性增长至 `final_ratio_b`，A 比例不低于 `min_ratio_a`

### 训练超参

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `lr` | 3e-6 ~ 1e-5 | 基础学习率 |
| `min_lr` | 1e-7 | 最低学习率 |
| `weight_decay` | 0.05 | AdamW 权重衰减 |
| `warmup_epochs` | 0.5 ~ 1 | 学习率 warmup epoch 数 |
| `epochs` | 10 | 总训练 epoch |
| `batch_size` | 1 | 每卡 batch size |
| `gradient_checkpointing` | True | 梯度检查点（节省显存） |
| `amp` | 1 | 混合精度（BF16） |

### 各配置文件速查

| 配置文件 | 数据源 | 关键特点 |
|----------|--------|----------|
| `train.yaml` | 15 个多数据集加权混合 | 原始 StreamVGGT 蒸馏，**无 Event** |
| `finetune.yaml` | 同 train.yaml | 监督微调，`ConfLoss + FinetuneLoss` |
| `train_dl3dv.yaml` | DL3DV 单数据集 | `train_dino=True`，变长帧数，`interval=1` |
| `train_dl3dv_fsdp.yaml` | DL3DV 单数据集 | 固定帧长 15，`interval=1~3`，`gradient_checkpointing` |
| `train_M3ed_fsdp.yaml` | M3ED 单数据集 | 固定帧长 15，真实事件数据 |
| `train_M3ed_fsdp_test.yaml` | M3ED 小子集 | 50 train / 5 test，冒烟调试用 |
| `train_M3ed_curriculum.yaml` | DL3DV + M3ED | 课程学习 v1，`warmup_epochs=2` |
| **`train_M3ed_curriculum_v2.yaml`** | DL3DV + M3ED | **课程学习 v2（推荐）**，`lr=3e-6`，v2 带 `initial_ratio_b` |

---

## 训练命令

> **本分支训练脚本强制要求 FSDP**，需通过 `accelerate launch --use_fsdp` 启动。

### 使用启动脚本（推荐）

```bash
cd src/
bash train.sh
```

`train.sh` 默认使用 4 卡（`CUDA_VISIBLE_DEVICES=4,5,6,7`）和 `train_M3ed_curriculum_v2` 配置。

### 手动启动

**4 卡 FSDP 训练（推荐配置：课程学习 v2）**：

```bash
cd src/

CUDA_VISIBLE_DEVICES=4,5,6,7 HYDRA_FULL_ERROR=1 \
accelerate launch --use_fsdp --num_processes 4 --main_process_port 29500 \
./train.py --config-name train_M3ed_curriculum_v2
```

**DL3DV 单数据集训练**：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 HYDRA_FULL_ERROR=1 \
accelerate launch --use_fsdp --num_processes 4 --main_process_port 29500 \
./train.py --config-name train_dl3dv_fsdp
```

**覆盖配置参数**（Hydra CLI 覆盖）：

```bash
accelerate launch --use_fsdp --num_processes 4 --main_process_port 29500 \
./train.py --config-name train_dl3dv_fsdp \
  sample_mode=mixed \
  sequence_ratio=0.7 \
  lr=5e-6 \
  epochs=20
```

**冒烟测试（验证环境可用）**：

```bash
CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 \
accelerate launch --use_fsdp --num_processes 1 --main_process_port 29500 \
./train.py --config-name train_M3ed_fsdp_test
```

### 断点续训

训练脚本支持自动恢复：若 `output_dir` 中存在 `checkpoint-last.pth`，会自动加载并续训。也可手动指定：

```bash
./train.py --config-name train_M3ed_curriculum_v2 resume=path/to/checkpoint.pth
```

### 训练日志

- **终端**：主进程显示 tqdm 进度条，包含 loss 和 lr
- **TensorBoard**：日志写入 `${save_dir}/${exp_name}/logs/`
  ```bash
  tensorboard --logdir checkpoints/vggt_train_M3ed_curriculum_v2/logs
  ```
- **文本日志**：`${output_dir}/log.txt`（每 epoch 写入 JSON 格式统计）

---

## 推理命令

### 使用一键脚本

```bash
cd src/
bash inference.sh <checkpoint_path> <frame_num> [port]
```

**示例**：

```bash
bash inference.sh \
  ../checkpoints/vggt_train_M3ed_curriculum_v2/checkpoint-last.pth \
  30 \
  8001
```

该脚本会：
1. 运行 `inference_with_event.py` 进行 RGB+Event 推理
2. 将结果保存到 `output/<checkpoint_name>/`
3. 自动启动点云查看器（`see_pointcloud.py`）

### 手动运行推理

```bash
python src/inference_with_event.py \
  --checkpoint <checkpoint_path> \
  --data_root <sequence_root> \
  --output <output_dir> \
  --fusion crossattn \
  --event_in_chans 8 \
  --max_frames 50
```

**完整参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--checkpoint` | （必填） | 模型 checkpoint 路径 |
| `--data_root` | （必填） | 序列根目录（含 `images/` 和 `events/`） |
| `--output` | （必填） | 输出目录 |
| `--fusion` | `crossattn` | 融合模式：`none` / `crossattn` |
| `--event_in_chans` | 8 | Event 通道数 |
| `--img_size` | 518 | 图像尺寸 |
| `--patch_size` | 14 | Patch 大小 |
| `--embed_dim` | 1024 | 嵌入维度 |
| `--strict_load` | False | 是否严格加载 checkpoint |
| `--autocast` | `auto` | 混合精度模式：`off` / `auto` / `bf16` / `fp16` |
| `--conf_threshold` | 0.0 | 点云置信度阈值（低于此值的点被过滤） |
| `--max_frames` | -1 | 最大推理帧数（-1 表示全部） |
| `--resize_hw` | -1 -1 | 指定推理分辨率（H W），-1 表示自动适配 |

### 推理输出结构

```
<output_dir>/
├── images/                        # 输入 RGB 图像副本
│   ├── frame_000000.png
│   └── ...
├── depth/                         # 深度估计
│   ├── frame_000000.npy           #   原始深度（float32）
│   ├── frame_000000.png           #   归一化深度（uint16）
│   ├── frame_000000_vis.png       #   Turbo colormap 可视化
│   └── frame_000000_meta.json     #   深度范围 {min, max}
├── cameras/                       # 相机参数
│   ├── frame_000000_extrinsic.txt #   4×4 外参矩阵
│   ├── frame_000000_intrinsic.txt #   3×3 内参矩阵
│   ├── cameras.npz                #   所有帧的相机参数（npz 格式）
│   └── colmap/                    #   COLMAP 格式输出
│       ├── cameras.txt
│       ├── images.txt
│       └── points3D.txt
├── point_cloud/                   # 点云
│   ├── frame_000000.ply           #   单帧点云
│   └── merged.ply                 #   全序列合并点云
└── transforms.json                # NeRF 兼容格式（OPENCV 相机模型）
```

---

## 评测

### 单目深度评测

```bash
cd src/
bash eval/monodepth/run.sh
```

结果保存在 `eval_results/monodepth/${data}_${model_name}/metric.json`。

### 视频深度评测

```bash
cd src/
bash eval/video_depth/run.sh
```

结果保存在 `eval_results/video_depth/${data}_${model_name}/result_scale.json`。

### 多视图重建评测

```bash
cd src/
bash eval/mv_recon/run.sh
```

结果保存在 `eval_results/mv_recon/${model_name}_${ckpt_name}/logs_all.txt`。

### 相机位姿评测

```bash
# 安装额外依赖
pip install pycolmap==3.10.0 pyceres==2.3
git clone https://github.com/cvg/LightGlue.git && cd LightGlue && pip install -e . && cd ..

# 运行评测
cd src/
python eval/pose_evaluation/test_co3d.py \
  --co3d_dir /YOUR/CO3D/PATH \
  --co3d_anno_dir /YOUR/CO3D/ANNO/PATH \
  --seed 0
```

### Event/RGB 重建 Benchmark

```bash
cd src/
python -m eval_benchmark \
  --root <reconstruction_output_dir> \
  --gt_root <ground_truth_dir>
```

支持 LPIPS、深度尺度评测、RPE（相对位姿误差）等指标。

---

## Demo

基于 Gradio 的在线 Demo（纯 RGB 模式）：

```bash
pip install -r requirements_demo.txt
python demo_gradio.py
```

> 注意：Demo 使用的是不带 Event 融合的 StreamVGGT，如需 RGB+Event 推理请使用 `inference_with_event.py`。

---

## 常见问题排查

| 问题 | 排查方法 |
|------|----------|
| **训练启动报 FSDP 错误** | 确保使用 `accelerate launch --use_fsdp` 启动，而非直接 `python train.py` |
| **训练卡在 DataLoader** | 先用 `num_workers=0` 验证数据可读 |
| **多卡无负载** | 确认 `CUDA_VISIBLE_DEVICES` 与 `--num_processes` 数量匹配 |
| **Event 通道报错** | 检查 `event_in_chans` 与 `.pt` 文件实际通道一致 |
| **NaN loss** | 训练脚本内置 NaN 检测与跳过机制，超过 100 次跳过会提前结束 epoch；检查数据是否含异常值 |
| **显存不足** | 启用 `gradient_checkpointing=True`，减少 `num_views`，或使用更多卡 |
| **加载 checkpoint 报 key mismatch** | 新增的 Event 融合模块在旧权重中不存在，使用 `strict_load=False` 或 `pretrained_strict=False` |
| **推理深度全为 NaN** | 检查推理日志中的 `[DIAG]` 信息，确认 Event 数据正常加载 |

---

## 致谢

本项目基于以下工作：

- [StreamVGGT](https://github.com/wzzheng/StreamVGGT) — 流式 4D 视觉几何 Transformer
- [VGGT](https://github.com/facebookresearch/vggt) — 视觉几何基础模型
- [DUSt3R](https://github.com/naver/dust3r) — 稠密 3D 重建
- [MonST3R](https://github.com/Junyi42/monst3r) — 单目场景 3D 重建
- [Spann3R](https://github.com/HengyiWang/spann3r) — 空间记忆 3D 重建
- [CUT3R](https://github.com/CUT3R/CUT3R) — 数据处理与评测流程
- [Point3R](https://github.com/wzzheng/Point3R) — 流式 3D 重建

## 引用

```bibtex
@article{streamVGGT,
    title={Streaming 4D Visual Geometry Transformer},
    author={Dong Zhuo and Wenzhao Zheng and Jiahe Guo and Yuqi Wu and Jie Zhou and Jiwen Lu},
    journal={arXiv preprint arXiv:2507.11539},
    year={2025}
}
```
