# StarStream / StreamVGGT 训练策略详解 (train.md)

> **基准配置**：`config/train_local_test_0417.yaml`（2026-04-17）
> **启动方式**：`src/train.sh` + `src/train.py`
> **后续训练策略（不论数据量大小）均遵循本文约定**——若要放大到 full-scale，只需按 §12 的方式等比例扩 `curriculum_total_size` / `num_views` / `epochs`，其余（loss、融合、课程分段、监督源）保持不变。

---

## 0. TL;DR —— 最快跑通

```bash
cd StarStream_clean/src

# 默认 GPU 4,5，用最新基准配置
bash train.sh 4,5 train_local_test_0417

# 多卡
bash train.sh 0,1,2,3 train_local_test_0417

# 启用 wandb
WANDB_MODE=online bash train.sh 0,1,2,3 train_local_test_0417
```

日志：`checkpoints/local_test_0417/logs/train_YYYYMMDD_HHMMSS.log`
权重：`checkpoints/local_test_0417/{checkpoint-last.pth, checkpoint-best.pth, checkpoint-final.pth, model_weights.pth}`

---

## 1. 训练总览（一图看懂设计）

本仓库训练一个 **RGB + Event** 多模态 4D 流式重建网络 `StreamVGGT`。策略要点：

| 维度 | 设计 |
|------|------|
| **骨干** | `StreamVGGT`（因果 Transformer + 多任务头：camera / depth / pmap / track） |
| **教师** | 原版 `VGGT`（冻结、`eval()`，始终读**清洁 RGB**）提供伪 GT |
| **监督** | **GtPoseDistillLoss**：dataset GT **优先**，teacher 蒸馏**兜底**，再加两项物理自监督 |
| **模态** | RGB + Event Voxel（8 通道），`CrossAttnFuse` 注入 aggregator |
| **质量门控** | **SNR × Event-Density 先验**（`use_fuse_prior=True`）驱动融合门控 |
| **数据** | **三个数据集**：DL3DV (sim) + M3ED (real day+night) + DSEC (real day) |
| **课程** | **三阶段 `CurriculumMixDataset`**：清洁 → 低光逐级加深 → 真实事件数据混入 |
| **分布式** | Accelerate + **FSDP FULL_SHARD** + BF16（`train.sh` 强制） |
| **正则** | AdamW + weight decay 分组 + grad clip 1.0 + gradient_checkpointing + NaN/Inf 保护 |
| **评估** | 每 epoch 末一次：validation loss + NVS 渲染 PSNR / SSIM / LPIPS |
| **Preflight** | 训练真正开始前跑 2 个 val batch 走完整 pipeline，出任何 NaN/Inf 立即 abort |

---

## 2. 代码与数据布局

```
StarStream_clean/
├── config/
│   └── train_local_test_0417.yaml   # ★ 当前策略基准      
├── ckpt/
│   └── model.pt                      # VGGT 权重（同时作 teacher 与 student 初始化）
├── checkpoints/
│   └── local_test_0417/              # exp_name 决定输出目录
│       ├── checkpoint-last.pth       # 自动 resume 点
│       ├── checkpoint-best.pth       # 最佳 val_loss
│       ├── checkpoint-final.pth
│       ├── model_weights.pth         # bare state_dict (for inference)
│       ├── code/<mm_dd-HH:MM:SS>/    # 训练时的代码快照
│       └── logs/                     # TB + 文本日志
└── src/
    ├── train.sh             # ★ 启动脚本（FSDP, accelerate launch）
    ├── train_resume.sh      # 断点续训脚本
    ├── train.py             # 主训练逻辑（Hydra 入口）
    ├── dust3r/              # 数据集、loss、metrics、渲染
    └── streamvggt/          # 主网络（aggregator + heads）
```

### 数据根路径（配置写死，需保证存在）

| 数据集 | 路径 | 角色 | 关键属性 |
|--------|------|------|----------|
| **DL3DV** (11 scenes) | `/data/fcr/data/train_data/DL3DV` | `dataset_a` 仿真基底 | GT pose ✅ / DA3 depth ✅ / 仿真 event ✅ / `images_low_light_{1..4}/` ✅ |
| **M3ED** (26 scenes: 13 day + 13 night) | `/data/fcr/data/train_data/M3ED` | `dataset_b` 真实事件 | GT pose ❌（在 `meta/poses_selected.npz`，走 teacher 兜底）/ DA3 depth ✅ / **真实 event** ✅ |
| **DSEC** (13 scenes, day) | `/data/fcr/data/train_data/DSEC` | `dataset_b` 真实事件 | GT pose ❌ / DA3 depth ✅ / **真实 event** ✅ |

> `dataset_b = dataset_real_m3ed + dataset_real_dsec`（字符串拼接的 `+` 由 `EasyDataset` 重载成 `CatDataset`）。

---

## 3. 启动流程（`train.sh` 做了什么）

```bash
bash train.sh [GPU_IDS] [CONFIG_NAME]
# GPU_IDS      默认 4,5
# CONFIG_NAME  默认 train_M3ed_curriculum_0416_8gpu（请显式传 train_local_test_0417）
```

脚本依次：

1. **Conda**：`source /data/fcr/miniconda3/etc/profile.d/conda.sh && conda activate StreamVGGT`
2. **cuDNN 修复**：把 torch 捆绑的 `nvidia.cudnn/lib` 前置到 `LD_LIBRARY_PATH`，避免系统 8.0.4 段错误
3. **WandB**：默认 `WANDB_MODE=disabled`；打开方式：`WANDB_MODE=online bash train.sh ...`，项目 `StarStream`，run name = `CONFIG_NAME`
4. **GPU 可见**：`CUDA_VISIBLE_DEVICES=${GPU_IDS}`，`NUM_GPUS` 自动统计
5. **FSDP 环境变量**（见 §6）
6. **杂项**：`OMP_NUM_THREADS=8`、`NCCL_P2P_DISABLE=0`、`TORCH_NCCL_BLOCKING_WAIT=1`、`HYDRA_FULL_ERROR=1`
7. **Accelerate 启动**：
    ```bash
    accelerate launch \
        --num_processes ${NUM_GPUS} \
        --num_machines 1 \
        --mixed_precision bf16 \
        --main_process_port 29501 \
        train.py --config-name "${CONFIG_NAME}"
    ```

> `train.py` 强制要求 `ACCELERATE_USE_FSDP=true`。不要用 `python train.py` 直接跑。

---

## 4. 配置详解（以 `train_local_test_0417.yaml` 为准）

### 4.1 路径与实验标识

```yaml
save_dir:   '../checkpoints'
exp_name:   'local_test_0417'
output_dir: ${save_dir}/${exp_name}/
logdir:     ${save_dir}/${exp_name}/logs
```

### 4.2 模型 / 初始化 / 融合

```yaml
teacher:           ../ckpt/model.pt
pretrained:        ../ckpt/model.pt
load_only_encoder: False
long_context:      False
fixed_length:      True
resume:            Null

fusion:            crossattn      # 或 'none'（ablation）
use_event:         True
event_in_chans:    8
event_patch_size:  Null           # 让 patch embed 自适应
inject_interval:   4              # 每 4 个 aggregator block 注入一次 adapter
use_fuse_prior:    True           # ★ SNR × event-density 质量感知先验
```

**`use_fuse_prior=True` 的作用**：在 `CrossAttnFuse` 内部根据两个**数据驱动** prior 调制 fusion gate：
- RGB SNR 代理（过曝/欠曝区 RGB 不可靠 → 降低 RGB 权重）
- Event density（无事件的静止区 → 降低 event 权重）
不启用时退化为单纯的可学习 gate（baseline ablation）。

### 4.3 DINO patch embed 策略

```yaml
train_dino:    False
dino_lr_scale: 0.05
```

- `train_dino=False`（默认）：`aggregator.patch_embed` **完全冻结**。
- 恒定冻结：`aggregator.camera_token`、`aggregator.register_token`（这两个 token 影响任务头行为，不可训练）。
- 若要训 DINO：改 `train_dino=True`，代码会把 `patch_embed.*` 参数以 `lr_scale=0.05` 独立分组放入 AdamW。

### 4.4 视图采样

```yaml
num_views:       2      # fixed_length=True 时的标准帧数
num_views_min:   2      # min_view_size（动态可变帧范围）
num_views_max:   3      # max_view_size
num_test_views:  2
allow_repeat:    False
sample_mode:     mixed  # sequence / random / mixed
sequence_ratio:  0.75   # mixed 模式下 sequence 抽样占比
n_corres_train:  0
n_corres_test:   0
```

> 本基准配置之所以 `num_views` 很小，是为了**快速迭代验证流程**；放大到 full-scale 训练时按 §12 提升。

### 4.5 损失（混合监督 + 自监督）

```yaml
loss_weight_camera:  1.0
loss_weight_depth:   0.5
loss_weight_pmap:    0.1
loss_weight_track:   0.5
loss_weight_pose_sc: 0.3
loss_weight_evt:     0.15
warmup_pose_sc:      2       # 前 2 个 epoch 不算自监督 pose SC
warmup_evt:          2       # 前 2 个 epoch 不算 event 物理 loss

train_criterion: "GtPoseDistillLoss(
  lambda_camera=${loss_weight_camera},
  lambda_depth=${loss_weight_depth},
  lambda_pmap=${loss_weight_pmap},
  lambda_track=${loss_weight_track},
  lambda_pose_sc=${loss_weight_pose_sc},
  lambda_evt=${loss_weight_evt},
  warmup_pose_sc=${warmup_pose_sc},
  warmup_evt=${warmup_evt})"
test_criterion:   <同上>
```

`GtPoseDistillLoss` 的监督源优先级：

| 项 | 数据**有** GT 时 | 数据**无** GT 时 | 权重 λ |
|----|-----------------|-----------------|------|
| **Camera** | `camera_pose_gt` + `camera_intrinsics_gt`（w2c 首帧相对化 → 9D 编码） | teacher `camera_pose` | 1.0 |
| **Depth** | `da3_disparity` + `da3_valid_mask`（disparity 空间，尺度+偏移不变 loss） | teacher `depth` | 0.5 |
| **Pmap** | 无 GT | teacher `pts3d_in_other_view` | 0.1 |
| **Track** | 无 GT | teacher `track` | 0.5 |
| **Pose SC** | 自监督：三元组 SE(3) 链式一致性（需 `is_video=True` 且 V≥3） | — | 0.3，`warmup_pose_sc=2` |
| **Evt** | 自监督：RGB 亮度恒常方程 ↔ event 极性物理约束 | — | 0.15，`warmup_evt=2` |

**数据集层面的 GT 命中矩阵**：

| 数据集 | `camera_pose_gt` | `da3_disparity` | 结果 |
|--------|------------------|-----------------|------|
| DL3DV | ✅ | ✅ | Camera / Depth 全走 GT |
| M3ED | ❌（pose 在 `meta/poses_selected.npz`，不导出 GT 字段） | ✅ | Camera 走 teacher，Depth 走 DA3 |
| DSEC | ❌ | ✅ | 同 M3ED |

> 所以"混合监督"本质上是 **逐 view 逐字段**决定监督源，DL3DV 走更强的 GT，M3ED/DSEC 自动切 teacher 兜底，不需要改代码。

### 4.6 数据集声明

三个数据集都用同一个 `DL3DV_ScreenEvent_Multi` 类（它能统一处理 DL3DV / M3ED / DSEC 的目录结构），参数共享：

```yaml
resolution:     (518, 392)          # ★ 单一分辨率（基准配置）
transform:      SeqColorJitter
min_interval:   1
max_interval:   2                   # 小数据下帧间隔更紧凑
sample_mode:    mixed
sequence_ratio: 0.75
event_in_chans: 8
```

> **放大到 full-scale 训练时**：将 `resolution` 改为多宽高比池（见 §12），`min_interval/max_interval` 提到 1/3，保持三数据集**参数一致**。

### 4.7 课程学习（★ 训练策略核心）

```yaml
curriculum_total_size:        500   # 每 epoch 样本数
curriculum_warmup_epochs:     1     # （legacy 参数，真实阶段由下面三个字段控制）
curriculum_final_ratio_b:     0.5   # dataset_b 比例上限
curriculum_min_ratio_a:       0.5   # dataset_a 比例下限
curriculum_initial_ratio_b:   0.0

curriculum_dark_start_epoch:  1     # 从第 1 epoch 开始给学生看低光
curriculum_real_start_epoch:  3     # 从第 3 epoch 开始混入真实数据
curriculum_dark_level_schedule: [2, 3, 4, 4]
#   level 1 ≈ 0.40  brightness  (mild)
#   level 2 ≈ 0.27              (medium)
#   level 3 ≈ 0.16              (dark)
#   level 4 ≈ 0.05              (extreme)
```

**五阶段 schedule（epoch 从 0 计，`epochs=5`）：**

| epoch | 学生看到 | dataset_b 比例 | `dark_level_max` |
|-------|----------|----------------|------------------|
| **0** | DL3DV 清洁 RGB | 0% | 0（无暗化） |
| **1** | DL3DV 低光（level ≤ 2，覆盖 8/11 scenes） | 0% | 2 |
| **2** | DL3DV 低光（level ≤ 3，覆盖 11/11 scenes） | 0% | 3 |
| **3** | DL3DV 低光（level ≤ 4，extreme）+ M3ED+DSEC 真实事件混入 | 线性 ramp 起点 | 4 |
| **4** | 同 3，real ratio 继续上升到 `final_ratio_b=0.5` | ramp 终点 | 4 |

关键规则：

1. **Teacher 始终读清洁 RGB**（view 里的 `teacher_img`），**学生才看暗化图像**——蒸馏目标质量不被破坏。
2. 每个 epoch 开头 `CurriculumMixDataset.set_epoch(epoch)` 会：
   - 递归调用所有叶子 dataset 的 `set_dark_schedule(use_low_light, dark_level_max)`
   - 按 `_get_ratio_b(epoch)` 重新生成 `(n_a, n_b)` 的混合索引
   - 打印 `[CurriculumMix] epoch=<e>, ratio_b=<%>, n_a=<>, n_b=<>, low_light=<>, dark_level_max=<>`
3. 每个 scene 在启用低光后，从 `images_low_light_{1..dark_level_max}/` 里**随机选一个可用级别**，clip 内所有视图共用该级别（一致的亮度）。缺失的级别会自动回退。
4. `ratio_b(epoch)` 从 `real_start_epoch` 开始，在剩余 epoch 内从 `initial_ratio_b=0.0` 线性升到 `final_ratio_b=0.5`，并保证 `ratio_a ≥ min_ratio_a=0.5`。

### 4.8 测试集

```yaml
test_dataset: 100 @ DL3DV_ScreenEvent_Multi(
  split='train',
  ROOT='/data/fcr/data/train_data/DL3DV',
  resolution=(518, 392),
  num_views=${num_test_views},
  seed=42,              # 固定种子 → 每次 eval 的样本稳定可比
  min_interval=1,
  max_interval=2,
  sample_mode='mixed',
  sequence_ratio=0.5,
  event_in_chans=8)
```

- 固定在 **DL3DV 清洁** 上评估，seed=42，取前 100 样本。`validate()` 对 test loader 始终调用 `set_epoch(0)`（不随训练 epoch 变），保证**每次 eval 的子集完全相同**，PSNR/SSIM/LPIPS 曲线可横向比较。
- 验证集不进课程（不暗化），反映的是"模型在干净目标域上的回退能力"，与训练域漂移解耦。

### 4.9 优化器 / 调度 / 训练超参

```yaml
seed:                   0
batch_size:             1            # 每卡 batch；effective batch = NUM_GPUS × accum_iter
accum_iter:             1
gradient_checkpointing: True
epochs:                 5
start_epoch:            0
start_step:             0
weight_decay:           0.05
lr:                     5e-6
min_lr:                 1e-7
warmup_epochs:          1
amp:                    1            # 启用 bf16 AMP
num_workers:            4
```

| 内容 | 说明 |
|------|------|
| **优化器** | AdamW，β=(0.9, 0.95) |
| **参数分组** | `no_weight_decay` (1D 张量、bias、layernorm) 不衰减；其余按 `weight_decay`。`train_dino=True` 时 `patch_embed.*` 独立分组按 `dino_lr_scale` |
| **LR schedule** | per-iter 线性 warmup（1 个 epoch）→ cosine decay → `min_lr=1e-7` |
| **Grad clip** | `NativeScaler` 统一做 `clip_grad=1.0` |
| **Grad checkpoint** | 模型侧开启，显存显著下降 |
| **混合精度** | `accelerate --mixed_precision bf16`，FSDP 参数也跑 bf16 |

### 4.10 评估 / 保存 / 日志

```yaml
eval_freq:        1            # 每 1 epoch 评估一次
max_val_batches:  20           # 评估最多 20 个 batch
save_freq:        0.5          # 每 50% epoch 写一次 checkpoint-last
max_checkpoints:  5
keep_freq:        1            # 每 epoch 归档 checkpoint-<epoch>.pth
print_freq:       5
print_img_freq:   50000000     # 关闭图像打印（太耗时）
num_imgs_vis:     4
```

---

## 5. 训练主循环（`train.py::train` 做了什么）

1. **Accelerator**：FSDP + BF16 + `InitProcessGroupKwargs(timeout=6000s)`
2. **代码快照**：`save_current_code(output_dir)` → `code/<mm_dd-HH:MM:SS>/`
3. **自动 resume**：`output_dir/checkpoint-last.pth` 存在就自动加载（见 §8）
4. **种子**：`seed + process_index`
5. **数据构建**：
   - 读取 `use_sim / use_m3ed / use_dsec`；三者都开（基准配置）→ 直接使用 YAML 里的 `train_dataset` 字符串
   - 若某个开关关闭，代码会在 runtime 按开关动态拼新的 `CurriculumMixDataset(...)`（只剩一种数据时自动跳过课程退化成 `total_size @ dataset`）
6. **模型**：
   - 学生 `StreamVGGT(fusion=crossattn, use_event=True, inject_interval=4, use_fuse_prior=True)`
   - 教师 `VGGT()`，`strict=True` 加载 `teacher`（`../ckpt/model.pt`），全部参数 `requires_grad=False`，`eval()`
   - 学生初始化 `pretrained`（`strict=False`，允许 missing / unexpected 键，打印前 10 个）
7. **冻结策略**：
   - 默认冻结 `aggregator.patch_embed.*`（`train_dino=False`）
   - 恒定冻结 `aggregator.camera_token`、`aggregator.register_token`
   - 打印 `Frozen X / Total Y parameters (Z%)`
8. **参数分组 → AdamW**（β=(0.9, 0.95)）+ `NativeScaler`
9. **Resume 加载**：`misc.load_model` → FSDP wrap → `misc.load_optimizer_after_fsdp`（FSDP 下 optimizer state 在 shard 之后重新 re-split）
10. **Wandb** 初始化（仅主进程，`resume="allow"`）
11. **★ Preflight 验证**：跑 2 个 val batch 走完整 pipeline（forward → NVS render → PSNR/SSIM/LPIPS），**要求每个 test dataset 同时产出 loss + psnr + ssim + lpips 四项**，缺任一项或出现 NaN/Inf 立即 `RuntimeError`
12. **Epoch 循环** `for epoch in [start_epoch, epochs]`：
    - 每 `save_freq` 写 `checkpoint-last.pth`
    - 每 `keep_freq` 归档 `checkpoint-<epoch>.pth`
    - `epoch == epochs` 时写 `checkpoint-final.pth` + `model_weights.pth` 后 break
    - `train_one_epoch(...)`
    - `validate(...)` → 若 `avg(val_loss) < best_so_far` 保存 `checkpoint-best.pth`

### 5.1 `train_one_epoch` 关键行为

- `data_loader.dataset.set_epoch(epoch)` 触发 `CurriculumMixDataset` 更新索引和暗化
- `criterion.set_epoch(epoch)` 更新自监督项 warmup gating
- `img` 从 `[-1, 1]` 归一化到 `[0, 1]`
- per-iter `misc.adjust_learning_rate(optimizer, epoch_f, args)`
- **NaN/Inf 保护**：
  - **非有限 loss**：跨进程 gather 判断；出现则 `zero_grad() + continue`；**累计 100 次**中止 epoch，日志打印 debug 信息（包括 event_voxel 的 shape/min/max/std、是否 nan/inf、dataset/label/instance）
  - **NaN/Inf grad norm**：**连续 50 次**视为发散，中止 epoch
- **Adapter gate logging**：每 `print_freq * accum_iter` 步把 `aggregator.inject_adapters[i].scale.tanh()` 写入 `adapter_gate/{i}`（TB + Wandb），用于监控融合门控是否健康
- **intra-epoch checkpoint**：每 `save_freq * len(loader)` 步写一次 `checkpoint-last.pth`，保存**真实 `best_so_far`**（不是 `inf`）和当前 epoch 号
- **断点续训 step 跳过**：`start_step > 0` 时跳过**已完成的 step（`<=`）**，避免重复 gradient update

### 5.2 `validate` 关键行为

- 复用训练 forward：`loss_of_one_batch(..., teacher=teacher, inference=False, symmetrize_batch=False, use_amp=bool(amp))`，同时返回 teacher 的 `gts`（9D pose encoding、`pts3d_in_other_view`、`depth` 等）
- **NVS 伪 GT 构造**（`_build_nvs_gt_views`）：我们的数据集不提供 `camera_pose` (4×4) / `camera_intrinsics` / `pts3d` 世界系点云字段，因此用 teacher 输出构造：
  - `camera_pose = eye(4)`（VGGT 首帧 = 世界系，`get_render_results` 只用 `gts[0]["camera_pose"]`）
  - `camera_intrinsics`：由 `pose_encoding_to_extri_intri` 从 9D `camera_pose` 解码（`[B, 3, 3]`）
  - `pts3d = pts3d_in_other_view`（teacher 世界系点云）
  - `img` 保留原视图（eval 集是清洁 DL3DV，teacher/student 看同一张）
- 传入 `get_render_results(gt_views, preds)` → NVS 渲染 → `compute_psnr` / `SSIMMetric` / `LPIPSMetric`
- 含义：**val PSNR/SSIM/LPIPS 衡量 student 与 teacher 在 NVS 空间的几何一致性**，与蒸馏目标一致
- 聚合后以 `val_<ds_name>/<metric>` 写 TB + Wandb（`step = (epoch+1) * len(train_loader)`）
- `avg(val_loss)` 触发 `checkpoint-best.pth`

---

## 6. FSDP 配置

`train.sh` 导出的全部 FSDP env：

```bash
export ACCELERATE_USE_FSDP=true
export FSDP_AUTO_WRAP_POLICY=TRANSFORMER_BASED_WRAP
export FSDP_TRANSFORMER_CLS_TO_WRAP=Block        # 按 aggregator Block 粒度 wrap
export FSDP_BACKWARD_PREFETCH=BACKWARD_PRE
export FSDP_SHARDING_STRATEGY=FULL_SHARD         # 参数 / 梯度 / 优化器全分片
export FSDP_STATE_DICT_TYPE=FULL_STATE_DICT      # 保存 full ckpt（rank0_only）
export FSDP_OFFLOAD_PARAMS=false                 # 不 offload，保持 GPU 速度
export FSDP_SYNC_MODULE_STATES=true
export FSDP_USE_ORIG_PARAMS=true                 # 使 param_groups 工作
export FSDP_CPU_RAM_EFFICIENT_LOADING=false
export FSDP_FORWARD_PREFETCH=false
export FSDP_ACTIVATION_CHECKPOINTING=false       # 由 model.gradient_checkpointing 处理
```

其他 env：`OMP_NUM_THREADS=8`、`NCCL_P2P_DISABLE=0`、`TORCH_NCCL_BLOCKING_WAIT=1`、`HYDRA_FULL_ERROR=1`。

**保存**：`save_final_model` 用 `FullStateDictConfig(offload_to_cpu=True, rank0_only=True)` + `misc.save_on_master`，保证只在 rank0 写盘、不 OOM。

---

## 7. 数据流水线

单 batch 是 `List[Dict]`（每个 view 一个 dict），关键字段：

| 字段 | 形状 / 类型 | 说明 |
|------|-------------|------|
| `img` | `[B, 3, H, W]`（训练内归一到 `[0, 1]`） | **学生输入**（可能被暗化） |
| `teacher_img` | 同上，**清洁 RGB** | 只给 teacher |
| `event_voxel` | `[B, 8, H, W]`（标准化后） | 学生事件输入 |
| `event_voxel_raw` | 同上，**保留极性原值** | `Levt` 物理 loss 用 |
| `camera_pose_gt` / `camera_intrinsics_gt` | `[B, 4, 4]` / `[B, 3, 3]`（仅 DL3DV 有） | 相机 GT |
| `da3_disparity` / `da3_valid_mask` | `[B, H, W]` / `[B, H, W]` bool | 深度伪 GT（3 个数据集都有） |
| `valid_mask` | `[B, H, W]` bool | 通用有效像素 |
| `is_video` / `is_metric` | bool | 门控 pose SC、metric 标记 |
| `label` / `instance` / `dataset` | meta | 日志 / debug |

> 基准配置下分辨率统一 `(518, 392)`。full-scale 训练时请启用多分辨率池（详见 §12）。

---

## 8. 断点续训

### 8.1 自动续训（推荐）

只要 `output_dir` 下有 `checkpoint-last.pth`，**重跑 `train.sh` 即可**，`train.py` 会自动加载：
- 模型、优化器、loss scaler 状态
- `epoch`（继续从下一 epoch）
- `start_step`（epoch 内对齐到未完成的 step）
- `best_so_far`

### 8.2 指定 checkpoint 续训

使用 `src/train_resume.sh`：
- 额外导出 `WANDB_RESUME=must` + `WANDB_RUN_ID=<id>` 复用 wandb run
- Hydra override：`resume=<绝对路径>`（写在脚本里的 `RESUME_CKPT`）

### 8.3 续训注意事项

1. **不能中途修改**：`epochs`、`curriculum_*`、`curriculum_total_size`、`num_views`—— 会破坏课程时间轴对齐。
2. 从**不同 exp_name** 恢复时：把源 `checkpoint-last.pth` 复制或软链到目标 `output_dir`。
3. `start_step` 会被保存到 ckpt，重启后在当前 epoch 跳过已完成 step。

---

## 9. Preflight 与监控

### 9.1 Preflight 验证（强制）

训练真正开始前自动执行 2-batch 的完整 val：
- 若任一 metric 非 finite → 立即 abort
- 若所有 test dataset 都返回空 metrics（通常是路径错）→ 立即 abort
- 通过后日志打印：
```
============================================================
  PREFLIGHT VALIDATION PASSED
============================================================
```

### 9.2 训练期指标（TB + Wandb）

- `train_loss`、`train_lr`
- **Loss 分项**：`train_Lcamera`、`train_Ldepth`、`train_Lpmap`、`train_Ltrack`、`train_Lpose_sc`、`train_Levt`、`train_total`
- **监督命中数**：`train_n_gt_pose_views`、`train_n_da3_views`（每个 batch 里走 GT 的 view 数）
- **融合门控**：`adapter_gate/{i}`——健康值前期接近 0，中后期随数据缓慢增加；长期全 0 表示事件对融合无贡献
- **每 epoch 开头**：`[CurriculumMix] epoch=.., ratio_b=.., n_a=.., n_b=.., low_light=.., dark_level_max=..`

### 9.3 验证期指标

- `val_<ds>/loss`、`val_<ds>/psnr`、`val_<ds>/ssim`、`val_<ds>/lpips`
- `avg(val_loss)` 触发 `checkpoint-best.pth`

---

## 10. 常见排错

| 现象 | 原因 / 处理 |
|------|-----------|
| `RuntimeError: This training script is FSDP-only` | 没用 `train.sh`（或没导 `ACCELERATE_USE_FSDP=true`） |
| Preflight `returned no metrics` | `test_dataset` 的 ROOT 不存在 / resolution 与实际不符 |
| Preflight 出 NaN | teacher ckpt 损坏、event voxel 全零 / 全 NaN、`da3_disparity` 异常 |
| `Non-finite loss` 反复触发 | 优先看打印里的 `event_info`（shape/has_nan/has_inf/min/max/std）；检查数据预处理 |
| 50 次 `NaN grad norm` 强制 break | lr 过大 / 数据风格剧烈切换；降 `lr` 或延长 `warmup_epochs` |
| OOM | 降 `num_views` / `num_views_max`；确认 `gradient_checkpointing=True`、`fixed_length=True`；必要时 `FSDP_OFFLOAD_PARAMS=true`（速度显著下降） |
| 每 epoch 开头卡 dataloader | `num_workers` 过大 / 磁盘慢；`CurriculumMixDataset.set_epoch` 会重新 permutation，几秒内属正常 |
| `adapter_gate` 长期 0 | `use_event=False` / `fusion=none` ablation；或 event voxel 读入全 0 |
| 续训后 curriculum 错位 | 中途改了 `epochs` / `curriculum_*_epoch` / `curriculum_total_size` |
| 续训后 `checkpoint-best` 被更差模型覆盖 | **已修复**。原因是 intra-epoch ckpt 曾把 `best_so_far=inf` 写入，resume 后任何 val loss 都会"更好"。现已改为保存真实 `best_so_far` |
| Wandb 401 / 超时 | 用 `WANDB_MODE=disabled` 暂避；或配对 `WANDB_API_KEY` / `WANDB_BASE_URL` |
| cuDNN 段错误 | `train.sh` 已前置 `nvidia.cudnn/lib`；若仍失败检查 conda env 是否真的装了 torch 捆绑 cuDNN |
| `[val] NVS rendering failed: 'camera_pose'` | **已修复**。原因：`get_render_results` 期望 `gts[i]` 含 `camera_pose` (4×4) / `camera_intrinsics` / `pts3d`，但数据集只给 `camera_pose_gt` / `camera_intrinsics_gt` / `da3_disparity`（无世界系点云）。修复：`loss_of_one_batch` 多返回 teacher 的 `gts`；`validate` 调用 `_build_nvs_gt_views` 用 teacher 输出构造伪 GT（`camera_pose=eye(4)`、`camera_intrinsics` 由 9D pose encoding 解码、`pts3d=pts3d_in_other_view`）。val 的 PSNR/SSIM/LPIPS 语义变为 **student NVS ↔ teacher NVS 一致性**，与蒸馏目标一致 |
| 每个 epoch 切换时"看似卡住" | 不是真卡：epoch 末要写 `checkpoint-last.pth`（FSDP full state dict + CPU offload + rank0_only，约 1–2 分钟）+ 可能同时写 `checkpoint-best.pth`；下一 epoch 开头还要触发 `CurriculumMixDataset.set_epoch` 重 permutation + 探测低光目录 + DataLoader workers cold start。等 1~3 分钟确认是否推进；如嫌慢可调大 `save_freq`（或设 0）减少中途落盘次数 |

---

## 11. 调参指引（保持基准策略骨架）

**策略不变**（融合、监督规则、课程三阶段结构、损失组合、自监督 warmup）——以下建议只调"数量级"：

| 目的 | 建议改动 |
|------|----------|
| **提高真实域适应** | `curriculum_final_ratio_b: 0.6~0.7`，相应延长 `epochs`；`min_ratio_a` 跟着降 |
| **更温和暗化** | `dark_level_schedule: [1, 2, 2, 3]`，或推迟 `dark_start_epoch` |
| **推迟真实数据** | `real_start_epoch` 再加 1~2（需保证 `real_start_epoch < epochs - 1` 才有 ramp 空间） |
| **消融 fuse prior** | `use_fuse_prior: False`（其他不变），直接做对比实验 |
| **消融 event 融合** | `fusion: none` 或 `use_event: False` |
| **打开 DINO 微调** | `train_dino: True` + `dino_lr_scale: 0.02~0.05` |
| **增强自监督** | 降 `warmup_pose_sc`、`warmup_evt` 到 1；或提高 `loss_weight_pose_sc`、`loss_weight_evt`（建议不超过 2× 默认） |
| **加训练稳定性** | `lr: 3e-6`、`warmup_epochs: 2` |

---

## 12. 从 0417 基准放大到 full-scale 训练

基准配置是 **local small-scale**（50 scenes、`total_size=500`、`num_views=2`、单一分辨率）。要放到正式训练，**同策略只扩规模**：

| 项 | 基准值 (0417) | Full-scale 建议 | 备注 |
|----|---------------|-----------------|------|
| `curriculum_total_size` | 500 | 5000 ~ 10000 | 直接乘 10~20，保证每 epoch 足够 step |
| `num_views` | 2 | 15 ~ 23 | 时序上下文变长，aggregator 能力更强 |
| `num_views_min/max` | 2/3 | 13/15 或 19/25 | 与 `num_views` 匹配 |
| `num_test_views` | 2 | 4 ~ 6 | 评测 NVS 更丰富 |
| `max_interval` | 2 | 3 | 帧间距更大，覆盖更广的相机运动 |
| `resolution` | `(518, 392)` | 多宽高比池（下方） | 提升视角鲁棒性 |
| `epochs` | 5 | 7 ~ 10 | 课程需要足够空间展开 |
| `curriculum_dark_level_schedule` | `[2, 3, 4, 4]` | `[2, 3, 4, 4, 4]` 等长 | 配合更长 epochs |
| `curriculum_dark_start_epoch` | 1 | 2 | 先让学生稳一阵子 |
| `curriculum_real_start_epoch` | 3 | 4 | 让暗化阶段先充分学习 |
| `max_val_batches` | 20 | 50 | 更稳的 val 曲线 |
| `save_freq` | 0.5 | 0.1 | 大 epoch 下更频繁落盘 |
| `num_workers` | 4 | 8 ~ 16 | 与 CPU 核数 / 磁盘吞吐匹配 |
| GPU | 1~2 | 4~8 | FSDP 线性扩展 |

**多宽高比 resolution 池**（full-scale 推荐，三个数据集共用）：
```yaml
resolution: [
  (518, 392), (518, 336), (518, 294), (518, 266), (518, 210), (518, 154),
  (392, 518), (336, 518), (294, 518), (266, 518)
]
```

**保持不变的策略骨架**（这是"和基准保持一致"的本质）：

1. `teacher = pretrained = ../ckpt/model.pt`
2. `fusion=crossattn`、`use_event=True`、`event_in_chans=8`、`inject_interval=4`、`use_fuse_prior=True`
3. `train_dino=False`（默认）；如需开启仅调 `dino_lr_scale`
4. `GtPoseDistillLoss` 的六项权重维持：`camera=1.0 / depth=0.5 / pmap=0.1 / track=0.5 / pose_sc=0.3 / evt=0.15`，以及 `warmup_pose_sc=2 / warmup_evt=2`（如总 epochs 明显变长可同比放大到 3~5）
5. **三阶段课程**：`clean → dark ramp → real inject`，三阶段的**存在**是核心，具体 epoch 分界按总长度等比例映射
6. 三数据集**全部启用**：`use_sim=use_m3ed=use_dsec=True`；`dataset_b = M3ED + DSEC`
7. **DL3DV 走 GT pose + DA3 depth；M3ED/DSEC 自动走 teacher pose + DA3 depth**——由 loss 的 GT 命中自动决定
8. **Teacher 始终读清洁 RGB，学生读暗化 RGB**（`teacher_img` vs `img`）
9. `FSDP FULL_SHARD` + `USE_ORIG_PARAMS=true` + BF16，grad checkpoint 开
10. `NativeScaler` + `clip_grad=1.0` + NaN/Inf 保护 100/50

> 不要在放大时"改 loss 公式 / 换融合策略 / 去掉课程"，否则就不是同一个策略了。

---

## 13. 推荐工作流

1. **小规模冒烟**：`bash train.sh 0,1 train_local_test_0417` → 通过 preflight → 跑完 1 个 epoch 确认 `[CurriculumMix]` 日志符合预期
2. **策略一致地扩量**：复制 `train_local_test_0417.yaml` → 按 §12 改规模参数 → 新建 `train_XXX.yaml` → `bash train.sh 0,1,2,3,4,5,6,7 train_XXX`
3. **监控**：
   - Wandb：`train_loss`、`train_Lcamera/Ldepth/Lpmap/Ltrack/Lpose_sc/Levt`、`train_n_gt_pose_views`、`train_n_da3_views`、`adapter_gate/*`、`val_<ds>/{psnr,ssim,lpips,loss}`
   - TB：同上，另看 `train_lr` 曲线（cosine 应平滑）
4. **中断恢复**：直接重跑 `train.sh`（自动 resume）；或 `train_resume.sh` 带 wandb run id
5. **产出**：`checkpoints/<exp>/{checkpoint-final.pth, model_weights.pth, checkpoint-best.pth}`
6. **下游**：`model_weights.pth` 用于 `src/inference.sh` / `src/run_eval_benchmark.sh`

---

## 附录 A. 关键代码入口速查

| 功能 | 位置 |
|------|------|
| 启动脚本 | `src/train.sh` |
| 断点续训 | `src/train_resume.sh` |
| 训练 main | `src/train.py::train` |
| 每 epoch 循环 | `src/train.py::train_one_epoch` |
| 验证 | `src/train.py::validate` |
| 最终保存 | `src/train.py::save_final_model` |
| 课程学习 | `src/dust3r/datasets/base/easy_dataset.py::CurriculumMixDataset` |
| 暗化 schedule 传播 | `CurriculumMixDataset._propagate_dark_schedule` |
| 数据集（三合一） | `src/dust3r/datasets/dl3dv.py::DL3DV_ScreenEvent_Multi` |
| 混合监督 Loss | `src/dust3r/losses.py::GtPoseDistillLoss` |
| Event 物理 loss | `GtPoseDistillLoss._evt_loss` |
| Pose 自一致 loss | `GtPoseDistillLoss._pose_sc_loss` |
| 主网络 | `src/streamvggt/models/streamvggt.py::StreamVGGT` |
| Aggregator / 融合 | `src/streamvggt/models/aggregator.py::{Aggregator, CrossAttnFuse}` |
| 质量 prior | `aggregator.py::{RGBReliabilityPrior, EventDensityPrior}` |
| Teacher | `src/vggt/models/vggt.py::VGGT` |
| NVS / 指标 | `src/dust3r/val_metrics.py`、`src/dust3r/utils/render.py::get_render_results` |

## 附录 B. 基准配置速览（复制即用）

```yaml
# 模型 / 融合
fusion: crossattn
use_event: True
event_in_chans: 8
inject_interval: 4
use_fuse_prior: True
train_dino: False
dino_lr_scale: 0.05

# 损失权重（保持不变）
loss_weight_camera: 1.0
loss_weight_depth:  0.5
loss_weight_pmap:   0.1
loss_weight_track:  0.5
loss_weight_pose_sc: 0.3
loss_weight_evt:     0.15
warmup_pose_sc: 2
warmup_evt:     2

# 数据三合一（全部启用）
use_sim:  True
use_m3ed: True
use_dsec: True
# dataset_b = dataset_real_m3ed + dataset_real_dsec

# 课程三阶段（epoch 0: 清洁 → 1..N-2: 暗化 ramp → N-2..N-1: real inject）
curriculum_final_ratio_b:     0.5
curriculum_min_ratio_a:       0.5
curriculum_initial_ratio_b:   0.0
curriculum_dark_start_epoch:  1          # 放大训练时建议 2
curriculum_real_start_epoch:  3          # 放大训练时建议 4
curriculum_dark_level_schedule: [2, 3, 4, 4]   # 放大可扩到 [2,3,4,4,4]

# 优化器
lr:            5e-6
min_lr:        1e-7
weight_decay:  0.05
warmup_epochs: 1
amp:           1
gradient_checkpointing: True

# 监督优先级（代码内自动）
# camera: camera_pose_gt  →  teacher.camera_pose
# depth : da3_disparity   →  teacher.depth
# pmap  : teacher.pts3d_in_other_view   (总是 teacher)
# track : teacher.track                 (总是 teacher)
# pose_sc / evt: 自监督，warmup 后启用
```

> 把基准策略视为**七个不变量**：①teacher 始终清洁；②学生走课程；③DL3DV GT pose + 三家 DA3 depth；④M3ED/DSEC pose 走 teacher；⑤CrossAttn + 质量 prior；⑥六项 loss 组合；⑦FSDP + BF16。放大时保留这七条即可。
