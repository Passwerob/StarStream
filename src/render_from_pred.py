#!/usr/bin/env python3
"""
Render images from predicted 3D point cloud.

For each frame, back-projects all OTHER frames' pixels to 3D using their
predicted depth + camera params, then renders the aggregated point cloud
from the target frame's camera. This evaluates both depth and pose quality.

Usage:
    python render_from_pred.py --pred_root <inference_output_dir> [--splat_radius 2]
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Render images from predicted 3D")
    p.add_argument("--pred_root", type=str, required=True,
                    help="Inference output directory with transforms.json, depth/, images/")
    p.add_argument("--splat_radius", type=int, default=2,
                    help="Splatting radius in pixels for each projected point")
    p.add_argument("--depth_clip_percentile", type=float, default=99.0,
                    help="Clip depth above this percentile to remove outliers")
    p.add_argument("--output_subdir", type=str, default="rendered_images",
                    help="Subdirectory name for rendered output images")
    return p.parse_args()


def load_transforms(path: Path):
    """Load camera params from transforms.json."""
    with open(path, "r") as f:
        data = json.load(f)
    frames = []
    for fr in data["frames"]:
        c2w = np.array(fr["transform_matrix"], dtype=np.float64)
        w2c = np.linalg.inv(c2w)
        K = np.array([
            [fr["fl_x"], 0, fr["cx"]],
            [0, fr["fl_y"], fr["cy"]],
            [0, 0, 1],
        ], dtype=np.float64)
        frames.append({
            "file_path": fr["file_path"],
            "c2w": c2w,
            "w2c": w2c,
            "K": K,
            "w": fr["w"],
            "h": fr["h"],
        })
    return frames


def backproject_frame(depth: np.ndarray, color: np.ndarray, K: np.ndarray,
                      c2w: np.ndarray, clip_pct: float = 99.0):
    """Back-project a single frame's pixels to 3D world coordinates.

    Returns: points_world (N,3), colors (N,3) both float64/float32.
    """
    H, W = depth.shape
    valid = (depth > 1e-4) & np.isfinite(depth)

    if clip_pct < 100:
        dmax = np.percentile(depth[valid], clip_pct)
        valid &= (depth <= dmax)

    vs, us = np.where(valid)
    zs = depth[vs, us].astype(np.float64)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    xs = (us.astype(np.float64) - cx) * zs / fx
    ys = (vs.astype(np.float64) - cy) * zs / fy

    pts_cam = np.stack([xs, ys, zs], axis=-1)  # (N,3)

    R = c2w[:3, :3]
    t = c2w[:3, 3]
    pts_world = (R @ pts_cam.T).T + t  # (N,3)

    colors_out = color[vs, us].astype(np.float32)
    if colors_out.max() > 1.5:
        colors_out = colors_out / 255.0

    return pts_world, colors_out


def zbuffer_splat(proj_uv, proj_z, colors, H, W, radius):
    """Z-buffer splatting with numpy. Sort by depth (far-to-near) so closer
    points overwrite farther ones, then splat each point as a small patch."""
    rendered = np.zeros((H, W, 3), dtype=np.float32)
    zbuf = np.full((H, W), 1e10, dtype=np.float64)

    order = np.argsort(-proj_z)
    proj_uv = proj_uv[order]
    proj_z = proj_z[order]
    colors = colors[order]

    N = len(proj_z)
    if radius == 0:
        u_arr = proj_uv[:, 0]
        v_arr = proj_uv[:, 1]
        valid = (u_arr >= 0) & (u_arr < W) & (v_arr >= 0) & (v_arr < H)
        for i in np.where(valid)[0]:
            u, v = u_arr[i], v_arr[i]
            zbuf[v, u] = proj_z[i]
            rendered[v, u] = colors[i]
        return rendered, zbuf

    for i in range(N):
        u0, v0 = proj_uv[i, 0], proj_uv[i, 1]
        z = proj_z[i]

        v_lo = max(0, v0 - radius)
        v_hi = min(H, v0 + radius + 1)
        u_lo = max(0, u0 - radius)
        u_hi = min(W, u0 + radius + 1)

        if v_lo >= v_hi or u_lo >= u_hi:
            continue

        patch = zbuf[v_lo:v_hi, u_lo:u_hi]
        mask = z < patch
        if mask.any():
            zbuf[v_lo:v_hi, u_lo:u_hi][mask] = z
            for c in range(3):
                rendered[v_lo:v_hi, u_lo:u_hi, c][mask] = colors[i, c]

    return rendered, zbuf


def render_pointcloud_to_camera(pts_world, colors, w2c, K, H, W, radius=2):
    """Project global point cloud onto a camera and render via z-buffer splatting."""
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    pts_cam = (R @ pts_world.T).T + t  # (N,3)

    mask_front = pts_cam[:, 2] > 1e-4
    pts_cam = pts_cam[mask_front]
    colors_f = colors[mask_front]

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    us = (pts_cam[:, 0] * fx / pts_cam[:, 2] + cx)
    vs = (pts_cam[:, 1] * fy / pts_cam[:, 2] + cy)

    margin = radius + 1
    in_bounds = (us >= -margin) & (us < W + margin) & (vs >= -margin) & (vs < H + margin)
    us = us[in_bounds]
    vs = vs[in_bounds]
    zs = pts_cam[in_bounds, 2]
    cs = colors_f[in_bounds]

    proj_uv = np.stack([us.astype(np.int32), vs.astype(np.int32)], axis=-1)

    rendered, zbuf = zbuffer_splat(proj_uv, zs, cs.astype(np.float32), H, W, radius)
    return rendered, zbuf


def fill_holes(rendered, zbuf, kernel_size=5):
    """Simple inpainting for holes (pixels with no depth)."""
    mask = (zbuf >= 1e9).astype(np.uint8)
    if mask.sum() == 0:
        return rendered

    img_u8 = np.clip(rendered * 255, 0, 255).astype(np.uint8)
    filled = cv2.inpaint(img_u8, mask, kernel_size, cv2.INPAINT_TELEA)
    return filled.astype(np.float32) / 255.0


def main():
    args = parse_args()
    pred_root = Path(args.pred_root)

    tf_path = pred_root / "transforms.json"
    if not tf_path.exists():
        raise FileNotFoundError(f"transforms.json not found in {pred_root}")

    frames = load_transforms(tf_path)
    n_frames = len(frames)
    print(f"[render] Loaded {n_frames} frames from {tf_path}")

    H, W = frames[0]["h"], frames[0]["w"]
    print(f"[render] Image size: {W}x{H}")

    all_pts = []
    all_colors = []

    for i, fr in enumerate(frames):
        img_path = pred_root / fr["file_path"]
        stem = Path(fr["file_path"]).stem  # e.g. frame_000000

        depth_npy = pred_root / "depth" / f"{stem}.npy"
        if not depth_npy.exists():
            print(f"  [WARN] Depth not found: {depth_npy}, skipping frame {i}")
            continue

        depth = np.load(str(depth_npy)).astype(np.float64)
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  [WARN] Image not found: {img_path}, skipping frame {i}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if img.shape[:2] != depth.shape[:2]:
            depth = cv2.resize(depth.astype(np.float32),
                               (img.shape[1], img.shape[0]),
                               interpolation=cv2.INTER_NEAREST).astype(np.float64)

        pts, cols = backproject_frame(depth, img, fr["K"], fr["c2w"],
                                      clip_pct=args.depth_clip_percentile)
        all_pts.append(pts)
        all_colors.append(cols)
        print(f"  Frame {i}: {len(pts)} points back-projected")

    global_pts = np.concatenate(all_pts, axis=0)
    global_colors = np.concatenate(all_colors, axis=0)
    print(f"[render] Global point cloud: {len(global_pts)} points total")

    out_dir = pred_root / args.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[render] Rendering {n_frames} views (splat_radius={args.splat_radius})...")
    for i, fr in enumerate(frames):
        stem = Path(fr["file_path"]).stem

        src_mask = np.ones(len(global_pts), dtype=bool)
        offset = 0
        for j in range(len(all_pts)):
            nj = len(all_pts[j])
            if j == i:
                src_mask[offset:offset + nj] = False
            offset += nj

        other_pts = global_pts[src_mask]
        other_colors = global_colors[src_mask]

        rendered, zbuf = render_pointcloud_to_camera(
            other_pts, other_colors, fr["w2c"], fr["K"], H, W,
            radius=args.splat_radius,
        )

        rendered_filled = fill_holes(rendered, zbuf, kernel_size=7)

        out_path = out_dir / f"{stem}.png"
        out_bgr = cv2.cvtColor(
            np.clip(rendered_filled * 255, 0, 255).astype(np.uint8),
            cv2.COLOR_RGB2BGR,
        )
        cv2.imwrite(str(out_path), out_bgr)

        coverage = 1.0 - np.mean(zbuf >= 1e9)
        print(f"  Frame {i}: coverage={coverage:.1%}, saved {out_path.name}")

    print(f"\n[render] Done. Rendered images saved to: {out_dir}")


if __name__ == "__main__":
    main()
