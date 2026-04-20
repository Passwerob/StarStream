# --------------------------------------------------------
# training code for CUT3R
# --------------------------------------------------------
# References:
# DUSt3R: https://github.com/naver/dust3r
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import sys
import time
import math
from collections import defaultdict
from pathlib import Path
from typing import Sized
from itertools import islice

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

torch.backends.cuda.matmul.allow_tf32 = True

from dust3r.model import (
    PreTrainedModel,
    ARCroco3DStereo,
    ARCroco3DStereoConfig,
    inf,
    strip_module,
)  # noqa: F401, needed when loading the model
from dust3r.datasets import get_data_loader
from dust3r.losses import *  # noqa: F401, needed when loading the model
from dust3r.inference import loss_of_one_batch  # noqa
from dust3r.viz import colorize
from dust3r.utils.render import get_render_results
import dust3r.utils.path_to_croco  # noqa: F401
import croco.utils.misc as misc  # noqa
from croco.utils.misc import NativeScalerWithGradNormCount as NativeScaler  # noqa

import wandb
import hydra
from omegaconf import OmegaConf
import logging
import pathlib
from tqdm import tqdm
import random
import builtins
import shutil

from accelerate import Accelerator
from accelerate import InitProcessGroupKwargs
from accelerate.logging import get_logger
from datetime import timedelta
import torch.multiprocessing

from vggt.models.vggt import VGGT
from streamvggt.models.streamvggt import StreamVGGT

torch.multiprocessing.set_sharing_strategy("file_system")

printer = get_logger(__name__, log_level="DEBUG")


def setup_for_distributed(accelerator: Accelerator):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        force = force or (accelerator.num_processes > 8)
        if accelerator.is_main_process or force:
            now = datetime.datetime.now().time()
            builtin_print("[{}] ".format(now), end="")  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def save_current_code(outdir):
    now = datetime.datetime.now()  # current date and time
    date_time = now.strftime("%m_%d-%H:%M:%S")
    src_dir = "."
    dst_dir = os.path.join(outdir, "code", "{}".format(date_time))
    shutil.copytree(
        src_dir,
        dst_dir,
        ignore=shutil.ignore_patterns(
            ".vscode*",
            "assets*",
            "example*",
            "checkpoints*",
            "OLD*",
            "logs*",
            "out*",
            "runs*",
            "*.png",
            "*.mp4",
            "*__pycache__*",
            "*.git*",
            "*.idea*",
            "*.zip",
            "*.jpg",
        ),
        dirs_exist_ok=True,
    )
    return dst_dir


def train(args):

    use_fsdp = str(os.environ.get("ACCELERATE_USE_FSDP", "false")).lower() in {"1", "true", "yes"}
    if not use_fsdp:
        raise RuntimeError("This training script is FSDP-only. Please launch with accelerate --use_fsdp.")

    accelerator = Accelerator(
        gradient_accumulation_steps=args.accum_iter,
        mixed_precision="bf16",
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=6000))],
    )
    device = accelerator.device

    setup_for_distributed(accelerator)

    printer.info("parallel_mode: FSDP")
    printer.info("precision_mode: bf16")
    printer.info("output_dir: " + args.output_dir)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if accelerator.is_main_process:
        dst_dir = save_current_code(outdir=args.output_dir)
        printer.info(f"Saving current code to {dst_dir}")

    # auto resume
    if not args.resume:
        last_ckpt_fname = os.path.join(args.output_dir, f"checkpoint-last.pth")
        args.resume = last_ckpt_fname if os.path.isfile(last_ckpt_fname) else None

    printer.info("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))

    # fix the seed
    seed = args.seed + accelerator.state.process_index
    printer.info(
        f"Setting seed to {seed} for process {accelerator.state.process_index}"
    )
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = args.benchmark

    # training dataset and loader — honour per-dataset toggles
    use_sim  = getattr(args, 'use_sim', True)
    use_m3ed = getattr(args, 'use_m3ed', True)
    use_dsec = getattr(args, 'use_dsec', True)

    if not (use_sim or use_m3ed or use_dsec):
        raise ValueError("At least one dataset must be enabled (use_sim / use_m3ed / use_dsec)")

    if not use_sim or not use_m3ed or not use_dsec:
        # rebuild train_dataset string from components
        datasets_b_parts = []
        if use_m3ed:
            datasets_b_parts.append(args.dataset_real_m3ed)
        if use_dsec:
            datasets_b_parts.append(args.dataset_real_dsec)

        if use_sim and datasets_b_parts:
            dataset_b_str = ' + '.join(datasets_b_parts)
            dark_start = int(getattr(args, 'curriculum_dark_start_epoch', 2))
            real_start = int(getattr(args, 'curriculum_real_start_epoch', dark_start))
            dark_sched = getattr(args, 'curriculum_dark_level_schedule', [999])
            train_dataset_str = (
                f"CurriculumMixDataset("
                f"total_size={args.curriculum_total_size},"
                f"dataset_a={args.dataset_sim},"
                f"dataset_b={dataset_b_str},"
                f"total_epochs={args.epochs},"
                f"warmup_epochs={args.curriculum_warmup_epochs},"
                f"final_ratio_b={args.curriculum_final_ratio_b},"
                f"min_ratio_a={args.curriculum_min_ratio_a},"
                f"initial_ratio_b={args.curriculum_initial_ratio_b},"
                f"dark_start_epoch={dark_start},"
                f"real_start_epoch={real_start},"
                f"dark_level_schedule={dark_sched})"
            )
        elif use_sim:
            # only sim, no real data — skip curriculum
            train_dataset_str = f"{args.curriculum_total_size} @ {args.dataset_sim}"
        elif datasets_b_parts:
            # no sim, only real data — skip curriculum
            dataset_b_str = ' + '.join(datasets_b_parts)
            train_dataset_str = f"{args.curriculum_total_size} @ ({dataset_b_str})"

        printer.info("Dataset toggles: sim=%s  m3ed=%s  dsec=%s", use_sim, use_m3ed, use_dsec)
    else:
        train_dataset_str = args.train_dataset

    printer.info("Building train dataset %s", train_dataset_str)
    #  dataset and loader
    data_loader_train = build_dataset(
        train_dataset_str,
        args.batch_size,
        args.num_workers,
        accelerator=accelerator,
        test=False,
        fixed_length=args.fixed_length,
        min_view_size=getattr(args, "num_views_min", None),
        max_view_size=getattr(args, "num_views_max", None),
    )
    printer.info("Building test dataset %s", args.test_dataset)
    data_loader_test = {
        dataset.split("(")[0]: build_dataset(
            dataset,
            args.batch_size,
            args.num_workers,
            accelerator=accelerator,
            test=True,
            fixed_length=True,
        )
        for dataset in args.test_dataset.split("+")
    }

    # model
    printer.info("Loading model")
    model = StreamVGGT(
        fusion=str(getattr(args, "fusion", "none")),
        use_event=bool(getattr(args, "use_event", False)),
        event_in_chans=int(getattr(args, "event_in_chans", 8)),
        event_patch_size=getattr(args, "event_patch_size", None),
        inject_interval=int(getattr(args, "inject_interval", 0)),
        use_fuse_prior=bool(getattr(args, "use_fuse_prior", True)),
    )
    teacher = VGGT()

    # model: PreTrainedModel = eval(args.model)
    printer.info(f"All model parameters: {sum(p.numel() for p in model.parameters())}")


    printer.info(f">> Creating train criterion = {args.train_criterion}")
    train_criterion = eval(args.train_criterion).to(device)
    printer.info(
        f">> Creating test criterion = {args.test_criterion or args.train_criterion}"
    )
    test_criterion = eval(args.test_criterion or args.criterion).to(device)

    model.to(device)

    if args.gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        else:
            printer.warning("gradient_checkpointing=True but model has no gradient_checkpointing_enable(); skipping")
    if args.long_context:
        model.fixed_input_length = False

    if args.pretrained and not args.resume:
        printer.info(f"Loading pretrained: {args.pretrained}")
        ckpt = torch.load(args.pretrained, map_location=device)
        info = model.load_state_dict(ckpt, strict=False)
        if info.missing_keys:
            printer.info(f"Pretrained missing keys ({len(info.missing_keys)}): {info.missing_keys[:10]}")
        if info.unexpected_keys:
            printer.info(f"Pretrained unexpected keys ({len(info.unexpected_keys)}): {info.unexpected_keys[:10]}")
        del ckpt

    printer.info("Loading teacher model")
    teacher_ckpt_path = getattr(args, "teacher", None) or args.pretrained
    ckpt_teacher = torch.load(teacher_ckpt_path, map_location=device)
    teacher.load_state_dict(ckpt_teacher, strict=True)
    teacher = teacher.to(device)
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()
    del ckpt_teacher


    # freeze
    printer.info("Freezing patch embedding and positional encoding parameters...")
    frozen_params = 0
    total_params = 0

    frozen_param_names = []

    for name, param in model.named_parameters():
        total_params += param.numel()
        param.requires_grad = True

    train_dino = bool(getattr(args, "train_dino", False))
    dino_lr_scale = float(getattr(args, "dino_lr_scale", 0.05))

    if hasattr(model, 'aggregator') and hasattr(model.aggregator, 'patch_embed') and not train_dino:
        for param in model.aggregator.patch_embed.parameters():
            if param.requires_grad:
                param.requires_grad = False

    printer.info(f"train_dino={train_dino}, dino_lr_scale={dino_lr_scale}")

    if hasattr(model, 'aggregator') and hasattr(model.aggregator, 'camera_token'):
        model.aggregator.camera_token.requires_grad = False

    if hasattr(model, 'aggregator') and hasattr(model.aggregator, 'register_token'):
        model.aggregator.register_token.requires_grad = False


    for name, p in model.named_parameters():
        if not p.requires_grad:
            frozen_params += p.numel()
            frozen_param_names.append(name)

    printer.info(
        f"Frozen {frozen_params:,} parameters out of {total_params:,} total parameters. ({frozen_params / total_params:.2%})")
    printer.info(
        f"Trainable parameters: {total_params - frozen_params:,} ({(total_params - frozen_params) / total_params:.2%})")
    if frozen_param_names:
        printer.info(
            f"Example frozen parameters: {', '.join(frozen_param_names[:5])}{'...' if len(frozen_param_names) > 5 else ''}")



    skip_list = set(model.no_weight_decay()) if hasattr(model, "no_weight_decay") and callable(model.no_weight_decay) else set()
    train_dino = bool(getattr(args, "train_dino", False))
    dino_lr_scale = float(getattr(args, "dino_lr_scale", 0.05))

    if train_dino and hasattr(model, "aggregator") and hasattr(model.aggregator, "patch_embed"):
        grouped = {
            "base_decay": {"params": [], "weight_decay": args.weight_decay, "lr_scale": 1.0},
            "base_no_decay": {"params": [], "weight_decay": 0.0, "lr_scale": 1.0},
            "dino_decay": {"params": [], "weight_decay": args.weight_decay, "lr_scale": dino_lr_scale},
            "dino_no_decay": {"params": [], "weight_decay": 0.0, "lr_scale": dino_lr_scale},
        }

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            is_dino = name.startswith("aggregator.patch_embed.")
            no_decay = (param.ndim <= 1) or name.endswith(".bias") or (name in skip_list)
            key = "dino_" if is_dino else "base_"
            key += "no_decay" if no_decay else "decay"
            grouped[key]["params"].append(param)

        param_groups = [v for v in grouped.values() if len(v["params"]) > 0]
    else:
        param_groups = misc.get_parameter_groups(model, args.weight_decay)

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    # print(optimizer)
    loss_scaler = NativeScaler(accelerator=accelerator)

    best_so_far = misc.load_model(
        args=args, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler
    )
    if best_so_far is None:
        best_so_far = float("inf")

    accelerator.even_batches = False
    optimizer, model, data_loader_train = accelerator.prepare(
        optimizer, model, data_loader_train
    )

    misc.load_optimizer_after_fsdp(model, optimizer)

    def write_log_stats(epoch, train_stats, test_stats):
        if accelerator.is_main_process:
            if log_writer is not None:
                log_writer.flush()

            log_stats = dict(
                epoch=epoch, **{f"train_{k}": v for k, v in train_stats.items()}
            )
            for test_name in data_loader_test:
                if test_name not in test_stats:
                    continue
                log_stats.update(
                    {test_name + "_" + k: v for k, v in test_stats[test_name].items()}
                )

            with open(
                os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

    def save_model(epoch, fname, best_so_far, data_iter_step):
        misc.save_model(
            accelerator=accelerator,
            args=args,
            model_without_ddp=model,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
            epoch=epoch,
            step=data_iter_step,
            fname=fname,
            best_so_far=best_so_far,
        )

    log_writer = (
        SummaryWriter(log_dir=args.output_dir) if accelerator.is_main_process else None
    )

    # Initialize wandb
    if accelerator.is_main_process:
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "StarStream"),
            name=os.environ.get("WANDB_RUN_NAME", getattr(args, "exp_name", None)),
            config=OmegaConf.to_container(args, resolve=True) if OmegaConf.is_config(args) else vars(args),
            dir=args.output_dir,
            resume="allow",
        )

    printer.info(f"Start training for {args.epochs} epochs")

    # ── Preflight validation check ──────────────────────────────────
    # Run 2 batches through the full val pipeline (forward → NVS render
    # → PSNR/SSIM/LPIPS → TB/WandB) so that errors surface *before*
    # any real training happens.
    if accelerator.is_main_process:
        print("\n" + "=" * 60)
        print("  PREFLIGHT VALIDATION CHECK (2 batches)")
        print("=" * 60)
    printer.info("Running preflight validation check (2 batches) ...")
    try:
        _preflight_args = OmegaConf.create(OmegaConf.to_container(args, resolve=True))
        _preflight_args.max_val_batches = 2  # tiny run
        _pf_metrics = validate(
            model, teacher, test_criterion, data_loader_test,
            accelerator, epoch=0, step=0,
            log_writer=None, args=_preflight_args,
        )
        # Sanity-check: at least one dataset returned the full metric set
        _has_metrics = any(m for m in _pf_metrics.values())
        if not _has_metrics:
            raise RuntimeError("Preflight validation returned no metrics — "
                               "check test_dataset and data paths")
        _required_keys = {'loss', 'psnr', 'ssim', 'lpips'}
        for ds, m in _pf_metrics.items():
            _missing = _required_keys - set(m.keys())
            if _missing:
                raise RuntimeError(
                    f"Preflight: dataset '{ds}' is missing metrics "
                    f"{_missing} — NVS rendering pipeline may be broken")
            for k, v in m.items():
                if not math.isfinite(v):
                    raise RuntimeError(
                        f"Preflight metric {ds}/{k}={v} is not finite")
        if accelerator.is_main_process:
            for ds, m in _pf_metrics.items():
                metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in m.items())
                print(f"  [{ds}] {metrics_str}")
            print("=" * 60)
            print("  PREFLIGHT VALIDATION PASSED")
            print("=" * 60 + "\n")
        del _pf_metrics, _preflight_args
        torch.cuda.empty_cache()
    except Exception as e:
        if accelerator.is_main_process:
            print("=" * 60)
            print(f"  PREFLIGHT VALIDATION FAILED: {e}")
            print("=" * 60 + "\n")
        raise RuntimeError(
            f"Validation pipeline broken — fix before training. Error: {e}"
        ) from e
    # ────────────────────────────────────────────────────────────────

    start_time = time.time()
    train_stats = test_stats = {}

    for epoch in range(args.start_epoch, args.epochs + 1):

        # Save immediately the last checkpoint
        if epoch > args.start_epoch:
            if (
                args.save_freq
                and np.allclose(epoch / args.save_freq, int(epoch / args.save_freq))
                or epoch == args.epochs
            ):
                save_model(epoch - 1, "last", best_so_far, args.start_step)

        new_best = False

        if epoch > args.start_epoch:
            if args.keep_freq and epoch % args.keep_freq == 0:
                save_model(epoch - 1, str(epoch), best_so_far, args.start_step)
        if epoch >= args.epochs:
            break  # exit after writing last test to disk


        # Train
        train_stats = train_one_epoch(
            model,
            teacher,
            train_criterion,
            data_loader_train,
            optimizer,
            accelerator,
            epoch,
            loss_scaler,
            log_writer=log_writer,
            args=args,
            best_so_far=best_so_far,
        )

        # Validation
        eval_freq = getattr(args, 'eval_freq', 1)
        if eval_freq > 0 and ((epoch + 1) % eval_freq == 0 or epoch == args.epochs - 1):
            val_step = int((epoch + 1) * len(data_loader_train))
            printer.info(f"Running validation at epoch {epoch} (step {val_step})")
            val_metrics = validate(
                model, teacher, test_criterion, data_loader_test,
                accelerator, epoch, val_step, log_writer, args,
            )

            # Log to TensorBoard and WandB
            if accelerator.is_main_process:
                wandb_log = {}
                for ds_name, metrics in val_metrics.items():
                    for metric_name, value in metrics.items():
                        tag = f"val_{ds_name}/{metric_name}"
                        if log_writer is not None:
                            log_writer.add_scalar(tag, value, val_step)
                        wandb_log[tag] = value
                    printer.info(
                        f"  val [{ds_name}]: " +
                        ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                    )
                if wandb_log:
                    wandb.log(wandb_log, step=val_step)
                if log_writer is not None:
                    log_writer.flush()

                # Track best val loss
                avg_val_loss = np.mean([
                    m.get('loss', float('inf'))
                    for m in val_metrics.values() if 'loss' in m
                ]) if val_metrics else float('inf')
                if avg_val_loss < best_so_far:
                    best_so_far = avg_val_loss
                    new_best = True
                    printer.info(f"New best val loss: {avg_val_loss:.4f}, saving best checkpoint")
                    save_model(epoch, "best", best_so_far, 0)


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    printer.info("Training time {}".format(total_time_str))

    save_final_model(accelerator, args, args.epochs, model, best_so_far=best_so_far)

    if accelerator.is_main_process and wandb.run is not None:
        wandb.finish()


def save_final_model(accelerator, args, epoch, model_without_ddp, best_so_far=None):
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        StateDictType,
        FullStateDictConfig,
    )

    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / "checkpoint-final.pth"

    if isinstance(model_without_ddp, dict):
        model_state = model_without_ddp
    elif isinstance(model_without_ddp, FSDP):
        sd_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model_without_ddp, StateDictType.FULL_STATE_DICT, sd_cfg):
            model_state = model_without_ddp.state_dict()
    else:
        model_state = model_without_ddp.state_dict()

    to_save = {
        "args": args,
        "model": model_state,
        "epoch": epoch,
    }
    if best_so_far is not None:
        to_save["best_so_far"] = best_so_far
    printer.info(f">> Saving model to {checkpoint_path} ...")
    misc.save_on_master(accelerator, to_save, checkpoint_path)

    weights_path = output_dir / "model_weights.pth"
    printer.info(f">> Saving bare state_dict to {weights_path} ...")
    misc.save_on_master(accelerator, model_state, weights_path)

    accelerator.wait_for_everyone()


def build_dataset(
    dataset,
    batch_size,
    num_workers,
    accelerator,
    test=False,
    fixed_length=False,
    min_view_size=None,
    max_view_size=None,
):
    split = ["Train", "Test"][test]
    printer.info(f"Building {split} Data loader for dataset: {dataset}")
    loader = get_data_loader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_mem=True,
        shuffle=not (test),
        drop_last=not (test),
        accelerator=accelerator,
        fixed_length=fixed_length,
        min_view_size=min_view_size,
        max_view_size=max_view_size,
    )
    return loader


def _build_nvs_gt_views(views, gts):
    """Assemble pseudo-GT view dicts for ``get_render_results``.

    ``get_render_results`` expects each GT view to carry ``camera_pose`` (4x4
    c2w), ``camera_intrinsics`` (3x3), ``pts3d`` (world-space point map) and
    ``img``. Our datasets only provide ``camera_pose_gt`` / ``da3_disparity``
    (no 3D point cloud GT), so we reuse the teacher's predictions as
    pseudo-GT:

    - ``camera_pose``: identity (VGGT expresses everything in the first
      camera's frame, which is exactly what ``get_render_results`` expects
      when ``self_view=False``; it only reads ``gts[0]["camera_pose"]``).
    - ``camera_intrinsics``: decoded from the teacher's 9-D pose encoding
      (``absT_quaR_FoV``). We only need the first view's intrinsics but we
      fill it for every view for robustness.
    - ``pts3d``: teacher's ``pts3d_in_other_view`` (world-frame point map,
      same convention as student predictions).
    - ``img``: kept from ``views`` (teacher sees clean RGB; this matches the
      pixel values we want to compare against during NVS).

    This turns NVS PSNR/SSIM/LPIPS into a *teacher-agreement* metric, which
    matches the distillation philosophy of this training script.
    """
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    if not gts or not views:
        return None
    if 'camera_pose' not in gts[0] or 'pts3d_in_other_view' not in gts[0]:
        return None

    images_hw = views[0]['img'].shape[-2:]
    device = gts[0]['pts3d_in_other_view'].device

    # [B, S, 9] pose encoding → [B, S, 3, 3] intrinsics
    cam_9d = torch.stack([g['camera_pose'] for g in gts], dim=1)
    _, intri = pose_encoding_to_extri_intri(
        cam_9d, image_size_hw=images_hw, build_intrinsics=True,
    )  # intri: [B, S, 3, 3]

    B = cam_9d.shape[0]
    eye4 = torch.eye(4, device=device, dtype=intri.dtype).expand(B, 4, 4)

    gt_views = []
    for s, (v, g) in enumerate(zip(views, gts)):
        gt_views.append({
            'img': v['img'],
            'camera_pose': eye4,
            'camera_intrinsics': intri[:, s].contiguous(),
            'pts3d': g['pts3d_in_other_view'],
        })
    return gt_views


@torch.no_grad()
def validate(model, teacher, criterion, data_loader_test, accelerator, epoch, step,
             log_writer, args):
    """Run validation and compute NVS-based PSNR, SSIM, LPIPS metrics."""
    model.eval()
    device = accelerator.device
    max_val_batches = getattr(args, 'max_val_batches', 50)

    if hasattr(criterion, "set_epoch"):
        criterion.set_epoch(epoch)

    from dust3r.val_metrics import compute_psnr, SSIMMetric, LPIPSMetric

    # Lazy-init metric modules (persist across calls)
    if not hasattr(validate, '_ssim'):
        validate._ssim = SSIMMetric(device=str(device))
        validate._lpips = LPIPSMetric(device=str(device))

    all_metrics = {}

    for dataset_name, loader in data_loader_test.items():
        # Always use epoch=0 for test loaders so the eval subset is
        # deterministic across epochs (seed=42 in config → same 100
        # samples every time → PSNR/SSIM/LPIPS curves are comparable).
        if hasattr(loader, "dataset") and hasattr(loader.dataset, "set_epoch"):
            loader.dataset.set_epoch(0)

        loss_sum, psnr_sum, ssim_sum, lpips_sum = 0.0, 0.0, 0.0, 0.0
        n_loss, n_rgb = 0, 0

        for batch in islice(loader, max_val_batches):
            # Move batch to device (test loaders are not accelerator-prepared)
            if isinstance(batch, list):
                for view in batch:
                    for k, v in view.items():
                        if isinstance(v, torch.Tensor):
                            view[k] = v.to(device)
            elif isinstance(batch, dict):
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device)

            # Normalize images to [0, 1] (same as training)
            if isinstance(batch, list):
                for view in batch:
                    view["img"] = (view["img"] + 1.0) / 2.0
            elif isinstance(batch, dict) and "img" in batch:
                batch["img"] = (batch["img"] + 1.0) / 2.0

            # Forward pass: model + teacher + loss (reuse training pipeline)
            result = loss_of_one_batch(
                batch, model, criterion, accelerator,
                teacher=teacher, inference=False,
                symmetrize_batch=False, use_amp=bool(args.amp),
            )

            if result["loss"] is not None:
                loss, loss_details = result["loss"]
                if math.isfinite(float(loss)):
                    loss_sum += float(loss)
                    n_loss += 1

            views = result["views"]
            preds = result["pred"]
            gts = result.get("gts")

            # NVS rendering → PSNR / SSIM / LPIPS
            # Build pseudo-GT views from teacher predictions (our datasets
            # don't carry ``pts3d`` / 4x4 ``camera_pose`` fields, so we use
            # the teacher's 3D output as the rendering target).
            gt_views = _build_nvs_gt_views(views, gts)
            if gt_views is None:
                if accelerator.is_main_process:
                    printer.warning(
                        f"[val] NVS rendering skipped for {dataset_name}: "
                        f"teacher outputs unavailable"
                    )
            else:
                try:
                    _, _, rendered_imgs, gt_rendered_imgs = get_render_results(
                        gt_views, preds)
                    for pred_rgb, gt_rgb in zip(rendered_imgs, gt_rendered_imgs):
                        # (B, H, W, 3) → (B, 3, H, W), clamp to [0, 1]
                        pred_rgb = pred_rgb.permute(0, 3, 1, 2).clamp(0, 1)
                        gt_rgb = gt_rgb.permute(0, 3, 1, 2).clamp(0, 1)
                        psnr_sum += float(compute_psnr(pred_rgb, gt_rgb))
                        ssim_sum += float(validate._ssim(pred_rgb, gt_rgb))
                        lpips_sum += float(validate._lpips(pred_rgb, gt_rgb))
                        n_rgb += 1
                except Exception as e:
                    if accelerator.is_main_process:
                        printer.warning(
                            f"[val] NVS rendering failed for {dataset_name}: {e}")

            torch.cuda.empty_cache()

        metrics = {}
        if n_loss > 0:
            metrics['loss'] = loss_sum / n_loss
        if n_rgb > 0:
            metrics['psnr'] = psnr_sum / n_rgb
            metrics['ssim'] = ssim_sum / n_rgb
            metrics['lpips'] = lpips_sum / n_rgb
        all_metrics[dataset_name] = metrics

    model.train()
    return all_metrics


def train_one_epoch(
    model: torch.nn.Module,
    teacher: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Sized,
    optimizer: torch.optim.Optimizer,
    accelerator: Accelerator,
    epoch: int,
    loss_scaler,
    args,
    log_writer=None,
    best_so_far=float("inf"),
):
    assert torch.backends.cuda.matmul.allow_tf32 == True

    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    accum_iter = args.accum_iter

    def save_model(epoch, fname, best_so_far, data_iter_step):
        misc.save_model(
            accelerator=accelerator,
            args=args,
            model_without_ddp=model,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
            epoch=epoch,
            step=data_iter_step,
            fname=fname,
            best_so_far=best_so_far,
        )

    if log_writer is not None:
        printer.info("log_dir: {}".format(log_writer.log_dir))

    if hasattr(data_loader, "dataset") and hasattr(data_loader.dataset, "set_epoch"):
        data_loader.dataset.set_epoch(epoch)
    if (
        hasattr(data_loader, "batch_sampler")
        and hasattr(data_loader.batch_sampler, "batch_sampler")
        and hasattr(data_loader.batch_sampler.batch_sampler, "set_epoch")
    ):
        data_loader.batch_sampler.batch_sampler.set_epoch(epoch)
    if hasattr(criterion, "set_epoch"):
        criterion.set_epoch(epoch)


    optimizer.zero_grad()
    train_one_epoch._nan_skip = 0
    train_one_epoch._nonfinite_skip = 0

    start_step = getattr(args, 'start_step', 0) or 0

    show_pbar = accelerator.is_main_process
    data_iter = tqdm(
        data_loader,
        total=len(data_loader),
        desc=header,
        dynamic_ncols=True,
        mininterval=0.5,
        leave=False,
        disable=not show_pbar,
        file=sys.stdout,
    )

    for data_iter_step, batch in enumerate(data_iter):

        # Skip already-completed steps when resuming from a mid-epoch checkpoint.
        # The saved step is the last *completed* step, so we skip up to and
        # including it (<=) to avoid re-running that gradient update.
        if start_step > 0 and data_iter_step <= start_step:
            if data_iter_step == 0:
                printer.info(f"Resuming: skipping to step {start_step}/{len(data_loader)}...")
            if data_iter_step == start_step:
                printer.info(f"Reached resume point (step {start_step}), training resumes from next step")
                start_step = 0
                args.start_step = 0
            continue

        with accelerator.accumulate(model):
            # change the range of the image to [0, 1]
            if isinstance(batch, dict) and "img" in batch:
                batch["img"] = (batch["img"] + 1.0) / 2.0
            elif isinstance(batch, list) and all(isinstance(v, dict) and "img" in v for v in batch):
                for view in batch:
                    view["img"] = (view["img"] + 1.0) / 2.0

            epoch_f = epoch + data_iter_step / len(data_loader)
            # we use a per iteration (instead of per epoch) lr scheduler
            if data_iter_step % accum_iter == 0:
                misc.adjust_learning_rate(optimizer, epoch_f, args)

            epoch_f = epoch + data_iter_step / len(data_loader)
            step = int(epoch_f * len(data_loader))

            result = loss_of_one_batch(
                batch,
                model,
                criterion,
                accelerator,
                teacher=teacher,
                inference=False,
                symmetrize_batch=False,
                use_amp=bool(args.amp),
            )
      
            loss, loss_details = result["loss"]  # criterion returns two values

            loss_value = float(loss)

            finite_flag = torch.tensor(
                1.0 if math.isfinite(loss_value) else 0.0, device=accelerator.device
            )
            all_finite = accelerator.gather(finite_flag).min().item() > 0.5
            if not all_finite:
                nonfinite_skip = getattr(train_one_epoch, '_nonfinite_skip', 0) + 1
                train_one_epoch._nonfinite_skip = nonfinite_skip
                debug_views = []
                if isinstance(batch, list):
                    for i, v in enumerate(batch):
                        if not isinstance(v, dict):
                            continue
                        evt = v.get("event_voxel")
                        evt_info = None
                        if evt is not None and torch.is_tensor(evt):
                            evt_info = {
                                "shape": list(evt.shape),
                                "has_nan": bool(torch.isnan(evt).any()),
                                "has_inf": bool(torch.isinf(evt).any()),
                                "min": float(evt.min()),
                                "max": float(evt.max()),
                                "std": float(evt.std()),
                            }
                        debug_views.append(
                            {
                                "i": i,
                                "dataset": v.get("dataset"),
                                "label": v.get("label"),
                                "instance": v.get("instance"),
                                "event_info": evt_info,
                            }
                        )
                elif isinstance(batch, dict):
                    debug_views.append(
                        {
                            "dataset": batch.get("dataset"),
                            "label": batch.get("label"),
                            "instance": batch.get("instance"),
                        }
                    )

                if accelerator.is_main_process:
                    safe_details = {k: v for k, v in loss_details.items()
                                    if isinstance(v, (int, float, str, bool))}
                    printer.error(
                        f"Non-finite loss at step={step} (total_skips={nonfinite_skip}) "
                        f"loss={loss_value} details={safe_details} views={debug_views}"
                    )
                if nonfinite_skip >= 100:
                    if accelerator.is_main_process:
                        printer.error(
                            f"100 cumulative non-finite losses in this epoch — stopping epoch early"
                        )
                    optimizer.zero_grad()
                    break
                optimizer.zero_grad()
                continue

            if not result.get("already_backprop", False):
                grad_norm = loss_scaler(
                    loss,
                    optimizer,
                    parameters=model.parameters(),
                    update_grad=True,
                    clip_grad=1.0,
                )
                if grad_norm is not None and (torch.isnan(grad_norm) or torch.isinf(grad_norm)):
                    nan_skip_counter = getattr(train_one_epoch, '_nan_skip', 0) + 1
                    train_one_epoch._nan_skip = nan_skip_counter
                    if accelerator.is_main_process:
                        safe_details = {k: v for k, v in loss_details.items()
                                        if isinstance(v, (int, float, str, bool))}
                        printer.warning(
                            f"NaN/Inf grad norm at step={step}, skipping update "
                            f"(consecutive={nan_skip_counter}, loss={loss_value:.4f}, "
                            f"details={safe_details})"
                        )
                    if nan_skip_counter >= 50:
                        if accelerator.is_main_process:
                            printer.error(
                                f"50 consecutive NaN grad norms — training is diverged, stopping epoch early"
                            )
                        optimizer.zero_grad()
                        break
                else:
                    train_one_epoch._nan_skip = 0
                optimizer.zero_grad()

            is_metric = batch[0]["is_metric"]
            curr_num_view = len(batch)

            del loss
            
            tb_vis_img = (data_iter_step + 1) % accum_iter == 0 and (
                (step + 1) % (args.print_img_freq)
            ) == 0
            if not tb_vis_img:
                del batch
            else:
                torch.cuda.empty_cache()

            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(epoch=epoch_f)
            metric_logger.update(lr=lr)
            metric_logger.update(step=step)
            metric_logger.update(loss=loss_value, **loss_details)
            if show_pbar and (((data_iter_step + 1) % args.print_freq == 0) or (data_iter_step + 1 == len(data_loader))):
                data_iter.set_postfix(loss=f"{loss_value:.4f}", lr=f"{lr:.2e}")
            if (data_iter_step + 1) % accum_iter == 0 and (
                (data_iter_step + 1) % (accum_iter * args.print_freq)
            ) == 0:
                loss_value_reduce = accelerator.gather(
                    torch.tensor(loss_value).to(accelerator.device)
                ).mean()  # MUST BE EXECUTED BY ALL NODES

                if log_writer is not None:
                    epoch_1000x = int(epoch_f * 1000)
                    log_writer.add_scalar("train_loss", loss_value_reduce, step)
                    log_writer.add_scalar("train_lr", lr, step)
                    log_writer.add_scalar("train_iter", epoch_1000x, step)

                    wandb_log = {
                        "train_loss": loss_value_reduce,
                        "train_lr": lr,
                        "epoch": epoch_f,
                    }

                    for name, val in loss_details.items():
                        if isinstance(val, torch.Tensor):
                            if val.ndim > 0:
                                continue
                        if isinstance(val, dict):
                            continue
                        log_writer.add_scalar("train_" + name, val, step)
                        wandb_log["train_" + name] = float(val) if isinstance(val, torch.Tensor) else val

                    raw_model = accelerator.unwrap_model(model)
                    if hasattr(raw_model, "aggregator") and getattr(raw_model.aggregator, "inject_adapters", None) is not None:
                        for i, adapter in enumerate(raw_model.aggregator.inject_adapters):
                            if adapter.scale.data.numel() > 0:
                                gate = adapter.scale.data.tanh().item()
                                log_writer.add_scalar(f"adapter_gate/{i}", gate, step)
                                wandb_log[f"adapter_gate/{i}"] = gate

                    wandb.log(wandb_log, step=step)

        save_every = max(int(args.save_freq * len(data_loader)), 1)
        if (
            data_iter_step % save_every == 0
            and data_iter_step != 0
            and data_iter_step != len(data_loader) - 1
        ):
            print("saving at step", data_iter_step)
            save_model(epoch, "last", best_so_far, data_iter_step)

    data_iter.close()

    metric_logger.synchronize_between_processes(accelerator)
    printer.info("Averaged stats: %s", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def batch_append(original_list, new_list):
    for sublist, new_item in zip(original_list, new_list):
        sublist.append(new_item)
    return original_list


@hydra.main(
    version_base=None,
    config_path=str(os.path.dirname(os.path.abspath(__file__))) + "/../config",
    config_name="train.yaml",
)
def run(cfg: OmegaConf):
    OmegaConf.resolve(cfg)
    logdir = pathlib.Path(cfg.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    train(cfg)


if __name__ == "__main__":
    run()
