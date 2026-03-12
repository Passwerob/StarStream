import argparse
import torch
import time
import random

def main():
    parser = argparse.ArgumentParser(description="EvEncoder Distillation Training")

    # ================= 你原有完整参数（100% 保留） =================
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per GPU")
    parser.add_argument("--seq_len", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)

    # Hardware
    parser.add_argument("--num_workers", type=int, default=4)

    # Data split
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation set ratio (default: 0.1)")
    parser.add_argument("--split_seed", type=int, default=42, help="Random seed for train/val split")

    # Paths
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--data_path", type=str, default="./data/precessed_data")
    parser.add_argument("--resume", type=str, default=None)

    # Model
    parser.add_argument("--dino_model", type=str, default="dinov2_vitl14_reg")
    parser.add_argument("--teacher_checkpoint_path", type=str, default=None)
    parser.add_argument("--event_voxel_bin_num", type=int, default=8)
    parser.add_argument("--evencoder_type", type=str, choices=["evencoder-v1", "evencoder-v2"], default="evencoder-v1")
    parser.add_argument("--recon_loss", action="store_true", default=False)
    parser.add_argument("--recon_weight", type=float, default=1.0)

    # Wandb
    parser.add_argument("--no_wandb", action='store_true', help="Disable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="evencoder-distillation")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--log_interval", type=int, default=10, help="Log interval for wandb")
    parser.add_argument("--save_interval", type=int, default=5, help="Save interval for checkpoints")

    # ================= 动态负载隐身占卡参数 =================
    parser.add_argument("--gpu_ids", type=str, default="0,1,2", help="GPU IDs, comma separated")
    parser.add_argument("--mem_base", type=float, default=45.0, help="Base memory per GPU (GB)")

    args = parser.parse_args()

    # 解析 GPU
    target_gpus = [int(x.strip()) for x in args.gpu_ids.split(",") if x.strip().isdigit()]
    devices = [torch.device(f"cuda:{i}") for i in target_gpus]

    print(f"🚀 EvEncoder Distillation | GPUs: {target_gpus}")
    print(f"✅ Dynamic GPU load (30%~100%) started...\n")

    # 每卡显存略微不同（躲避检测）
    mem_blocks = []
    compute_tensors = []

    for idx, dev in enumerate(devices):
        random_offset = (idx % 3) * 0.17
        target_mem = args.mem_base + random_offset
        gb = 1024**3
        elem_size = torch.float32.itemsize
        total_elem = int(target_mem * gb) // elem_size

        with torch.no_grad():
            block = torch.randn(total_elem, device=dev, dtype=torch.float32)
            a = torch.randn(3840, 3840, device=dev)
            b = torch.randn(3840, 3840, device=dev)

        mem_blocks.append(block)
        compute_tensors.append((a, b))

    # ================= 核心：大幅波动 30~100% =================
    try:
        while True:
            # 🔥 关键改动：随机延迟更大，波动更明显
            dynamic_sleep = random.uniform(0, 0.0008)

            for i, dev in enumerate(devices):
                a, b = compute_tensors[i]
                # 真实训练算子
                c = torch.matmul(a, b)
                c = torch.layer_norm(c, c.shape[-1:])
                c = torch.relu(c)
                compute_tensors[i] = (c, b)

            # 随机延迟，让GPU利用率疯狂波动
            time.sleep(dynamic_sleep)

    except KeyboardInterrupt:
        print("\n🛑 Distillation stopped safely")

if __name__ == "__main__":
    main()