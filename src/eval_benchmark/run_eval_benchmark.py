#!/usr/bin/env python3
"""
CLI entry point for the event-based reconstruction evaluation benchmark.

Usage:
    python -m eval_benchmark --root /path/to/output [OPTIONS]

Examples:
    # Full evaluation with GT
    python -m eval_benchmark \\
        --root output/checkpoint-2 \\
        --gt_root /path/to/gt \\
        --device cuda \\
        --lpips_backbone alex \\
        --depth_scale_mode both

    # Evaluate all checkpoints in an output directory
    python -m eval_benchmark \\
        --root output/ \\
        --gt_root /path/to/gt

    # Use VGG backbone for LPIPS, custom RPE delta
    python -m eval_benchmark \\
        --root output/checkpoint-2 \\
        --gt_root /path/to/gt \\
        --lpips_backbone vgg \\
        --rpe_delta 5
"""

import argparse
import sys
import time


def parse_args():
    p = argparse.ArgumentParser(
        description="Event-based RGB Reconstruction Evaluation Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--root", type=str, required=True,
        help="Root directory to scan for predicted outputs.",
    )
    p.add_argument(
        "--gt_root", type=str, default=None,
        help="Root directory containing ground truth data.",
    )
    p.add_argument(
        "--output", type=str, default=None,
        help="Directory to save results (JSON + CSV). Default: <root>/eval_results/",
    )
    p.add_argument(
        "--device", type=str, default=None, choices=["cpu", "cuda"],
        help="Device for LPIPS. Default: auto-detect.",
    )
    p.add_argument(
        "--lpips_backbone", type=str, default="alex", choices=["alex", "vgg"],
        help="LPIPS backbone network. Default: alex.",
    )
    p.add_argument(
        "--depth_scale_mode", type=str, default="both",
        choices=["raw", "median", "both"],
        help="Depth evaluation mode: 'raw' (no scaling), 'median' (median-scaled), "
             "or 'both' (report both). Default: both.",
    )
    p.add_argument(
        "--rpe_delta", type=int, default=1,
        help="Frame delta for Relative Pose Error. Default: 1.",
    )
    p.add_argument(
        "--event_threshold", type=float, default=0.01,
        help="Polarity threshold τ for ternary event quantization. Default: 0.01.",
    )
    p.add_argument(
        "--images_subdir", type=str, default="images",
        help="Subdirectory name containing predicted images. Default: images. "
             "Use 'rendered_images' for 3D-rendered outputs.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    from .evaluator import BenchmarkEvaluator

    print("=" * 64)
    print("  Event-Based RGB Reconstruction Evaluation Benchmark")
    print("=" * 64)
    print(f"  Root:             {args.root}")
    print(f"  GT Root:          {args.gt_root or '(none)'}")
    print(f"  Output:           {args.output or '<root>/eval_results/'}")
    print(f"  Device:           {args.device or 'auto'}")
    print(f"  LPIPS backbone:   {args.lpips_backbone}")
    print(f"  Depth scale mode: {args.depth_scale_mode}")
    print(f"  RPE delta:        {args.rpe_delta}")
    print(f"  Event threshold:  {args.event_threshold}")
    print(f"  Images subdir:    {args.images_subdir}")
    print("=" * 64)

    t0 = time.time()

    evaluator = BenchmarkEvaluator(
        root=args.root,
        gt_root=args.gt_root,
        output_dir=args.output,
        device=args.device,
        lpips_backbone=args.lpips_backbone,
        depth_scale_mode=args.depth_scale_mode,
        rpe_delta=args.rpe_delta,
        event_threshold=args.event_threshold,
        images_subdir=args.images_subdir,
    )
    results = evaluator.run()

    elapsed = time.time() - t0
    print(f"\n[Benchmark] Total time: {elapsed:.1f}s")

    if not results:
        print("[Benchmark] No results produced. Check your data paths.")
        sys.exit(1)

    return results


if __name__ == "__main__":
    main()
