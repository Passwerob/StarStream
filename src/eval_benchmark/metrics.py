"""
Evaluation metrics for event-based RGB reconstruction.

Research-grade, physically consistent metrics covering:
  - perceptual (LPIPS with configurable backbone)
  - structural (component-wise gradient error, edge F1)
  - temporal consistency (grayscale temporal diff)
  - event alignment (unified signed-map + ternary polarity)
  - depth accuracy (raw & median-scaled dual-mode)
  - camera pose error (raw + aligned + RPE with configurable delta)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Lazy-loaded heavy imports
# ---------------------------------------------------------------------------
_lpips_models: Dict[str, object] = {}
_torch = None


def _get_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


def _get_lpips_model(backbone: str = "alex", device=None):
    """Lazy-load LPIPS model, cached per backbone."""
    global _lpips_models
    key = f"{backbone}_{device}"
    if key not in _lpips_models:
        torch = _get_torch()
        import lpips
        model = lpips.LPIPS(net=backbone, verbose=False)
        dev = torch.device(device) if device else torch.device("cpu")
        model = model.to(dev).eval()
        _lpips_models[key] = model
    return _lpips_models[key]


def _rgb_to_gray(img: np.ndarray) -> np.ndarray:
    """ITU-R BT.601 luma: Y = 0.299R + 0.587G + 0.114B.

    Args:
        img: [H, W, 3] float32 in [0, 1], RGB order.

    Returns:
        [H, W] float64 grayscale.
    """
    return (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]).astype(np.float64)


# ============================================================================
# I. RGB / STRUCTURE METRICS
# ============================================================================

def compute_lpips(
    pred: np.ndarray,
    gt: np.ndarray,
    device: Optional[str] = None,
    backbone: str = "alex",
) -> float:
    """
    Perceptual similarity via LPIPS (lower is better).

    Args:
        pred: [H, W, 3] float32 in [0, 1]
        gt:   [H, W, 3] float32 in [0, 1]
        device: 'cuda' or 'cpu'
        backbone: 'alex' or 'vgg'

    Returns:
        LPIPS score (scalar).
    """
    torch = _get_torch()
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = _get_lpips_model(backbone, dev)

    def _to_tensor(img):
        t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
        return (t * 2.0 - 1.0).to(dev)

    with torch.no_grad():
        score = model(_to_tensor(pred), _to_tensor(gt))
    return float(score.item())


def compute_gradient_error(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Component-wise gradient error using Sobel filters.

    Computes: mean L1(Gx_pred - Gx_gt) + mean L1(Gy_pred - Gy_gt)

    This is the physically correct formulation — gradient *direction*
    information is preserved, unlike magnitude-only comparison.

    Args:
        pred: [H, W, 3] float32 in [0, 1]
        gt:   [H, W, 3] float32 in [0, 1]

    Returns:
        Component-wise gradient L1 error (scalar).
    """
    pred_gray = _rgb_to_gray(pred).astype(np.float32)
    gt_gray = _rgb_to_gray(gt).astype(np.float32)

    gx_pred = cv2.Sobel(pred_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy_pred = cv2.Sobel(pred_gray, cv2.CV_32F, 0, 1, ksize=3)
    gx_gt = cv2.Sobel(gt_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy_gt = cv2.Sobel(gt_gray, cv2.CV_32F, 0, 1, ksize=3)

    error_x = float(np.mean(np.abs(gx_pred - gx_gt)))
    error_y = float(np.mean(np.abs(gy_pred - gy_gt)))
    return error_x + error_y


def compute_edge_f1(
    pred: np.ndarray,
    gt: np.ndarray,
    low_thresh: int = 100,
    high_thresh: int = 200,
    tolerance: int = 2,
) -> Dict[str, float]:
    """
    Edge F1 score with strict matching protocol.

    - Fixed Canny thresholds: (100, 200)
    - Matching tolerance: radius = 2 pixels
    - Precision: fraction of predicted edges within tolerance of a GT edge
    - Recall: fraction of GT edges within tolerance of a predicted edge
    - F1: harmonic mean

    Args:
        pred: [H, W, 3] float32 in [0, 1]
        gt:   [H, W, 3] float32 in [0, 1]
        low_thresh, high_thresh: Canny thresholds
        tolerance: pixel radius for edge matching

    Returns:
        dict with 'precision', 'recall', 'f1'.
    """
    pred_gray = np.clip(pred * 255, 0, 255).astype(np.uint8)
    gt_gray = np.clip(gt * 255, 0, 255).astype(np.uint8)
    if pred_gray.ndim == 3:
        pred_gray = cv2.cvtColor(pred_gray, cv2.COLOR_RGB2GRAY)
    if gt_gray.ndim == 3:
        gt_gray = cv2.cvtColor(gt_gray, cv2.COLOR_RGB2GRAY)

    edges_pred = cv2.Canny(pred_gray, low_thresh, high_thresh)
    edges_gt = cv2.Canny(gt_gray, low_thresh, high_thresh)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tolerance + 1, 2 * tolerance + 1))
    gt_dilated = cv2.dilate(edges_gt, kernel)
    pred_dilated = cv2.dilate(edges_pred, kernel)

    pred_pts = edges_pred > 0
    gt_pts = edges_gt > 0

    n_pred = max(int(np.sum(pred_pts)), 1)
    n_gt = max(int(np.sum(gt_pts)), 1)

    precision = float(np.sum(pred_pts & (gt_dilated > 0))) / n_pred
    recall = float(np.sum(gt_pts & (pred_dilated > 0))) / n_gt
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)

    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def compute_temporal_error(
    pred_curr: np.ndarray,
    pred_prev: np.ndarray,
    gt_curr: np.ndarray,
    gt_prev: np.ndarray,
) -> float:
    """
    Temporal difference error in *grayscale* (BT.601 luma).

    ΔI_pred = Y(I_pred[t]) - Y(I_pred[t-1])
    ΔI_gt   = Y(I_gt[t])   - Y(I_gt[t-1])
    error   = mean |ΔI_pred - ΔI_gt|

    Args:
        pred_curr, pred_prev: [H, W, 3] float32 in [0, 1]
        gt_curr, gt_prev:     [H, W, 3] float32 in [0, 1]

    Returns:
        Mean absolute temporal difference error.
    """
    delta_pred = _rgb_to_gray(pred_curr) - _rgb_to_gray(pred_prev)
    delta_gt = _rgb_to_gray(gt_curr) - _rgb_to_gray(gt_prev)
    return float(np.mean(np.abs(delta_pred - delta_gt)))


# ============================================================================
# II. EVENT CONSISTENCY METRICS
# ============================================================================

def _event_to_signed_map(event: np.ndarray) -> np.ndarray:
    """Convert raw event tensor to a single-channel signed map.

    For multi-channel voxel grids (typical: positive/negative bins),
    collapse to: (#positive) - (#negative).
    For 2-channel data: channel 0 = positive, channel 1 = negative.
    For N-channel voxel grids: sum of all channels as signed accumulation.

    Args:
        event: [C, H, W] or [H, W] float32

    Returns:
        [H, W] float64 signed event map.
    """
    if event.ndim == 2:
        return event.astype(np.float64)

    C = event.shape[0]
    evt = event.astype(np.float64)

    if C == 2:
        return evt[0] - evt[1]

    return np.sum(evt, axis=0)


def compute_event_consistency(
    pred_curr: np.ndarray,
    pred_prev: np.ndarray,
    event: np.ndarray,
    eps: float = 1e-3,
    polarity_threshold: float = 0.01,
) -> Dict[str, float]:
    """
    Physically-consistent event alignment metrics.

    Protocol:
      1. Convert real event → signed map: E_real = Σ(positive) - Σ(negative)
      2. Pseudo-event: E_hat = log(Y(I_t) + eps) - log(Y(I_{t-1}) + eps)
      3. Ternary polarity: {+1, 0, -1} with threshold τ

    Metrics:
      - event_l1: mean |E_hat - E_real| on signed maps
      - event_corr: Pearson correlation on signed maps
      - polarity_acc: accuracy on ternary polarity maps

    Args:
        pred_curr: [H, W, 3] float32 in [0, 1]
        pred_prev: [H, W, 3] float32 in [0, 1]
        event: [C, H, W] or [H, W] float32 raw event data
        eps: log stability constant
        polarity_threshold: τ for ternary quantization

    Returns:
        dict with event_l1, event_corr, polarity_acc.
    """
    gray_curr = _rgb_to_gray(pred_curr)
    gray_prev = _rgb_to_gray(pred_prev)
    pseudo_event = np.log(gray_curr + eps) - np.log(gray_prev + eps)

    event_signed = _event_to_signed_map(event)

    if pseudo_event.shape != event_signed.shape:
        pseudo_event = cv2.resize(
            pseudo_event.astype(np.float32),
            (event_signed.shape[1], event_signed.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float64)

    results: Dict[str, float] = {}

    # L1 on signed maps
    results["event_l1"] = float(np.mean(np.abs(pseudo_event - event_signed)))

    # Pearson correlation on signed maps
    pe_flat = pseudo_event.ravel()
    ev_flat = event_signed.ravel()
    pe_c = pe_flat - pe_flat.mean()
    ev_c = ev_flat - ev_flat.mean()
    denom = np.sqrt(np.dot(pe_c, pe_c) * np.dot(ev_c, ev_c))
    results["event_corr"] = float(np.dot(pe_c, ev_c) / max(denom, 1e-12))

    # Ternary polarity accuracy
    tau = polarity_threshold

    def _ternary(x):
        out = np.zeros_like(x, dtype=np.int8)
        out[x > tau] = 1
        out[x < -tau] = -1
        return out

    pol_pred = _ternary(pseudo_event)
    pol_real = _ternary(event_signed)
    active = (pol_pred != 0) | (pol_real != 0)
    n_active = int(np.sum(active))
    if n_active > 0:
        results["polarity_acc"] = float(np.sum((pol_pred == pol_real) & active) / n_active)
    else:
        results["polarity_acc"] = 1.0

    return results


# ============================================================================
# III. DEPTH METRICS
# ============================================================================

def _depth_core(
    p: np.ndarray,
    g: np.ndarray,
) -> Dict[str, float]:
    """Compute depth metrics on pre-filtered, pre-scaled arrays."""
    ratio = np.maximum(p / g, g / p)
    return {
        "delta1": float(np.mean(ratio < 1.25)),
        "delta2": float(np.mean(ratio < 1.25 ** 2)),
        "delta3": float(np.mean(ratio < 1.25 ** 3)),
        "abs_rel": float(np.mean(np.abs(p - g) / g)),
        "rmse_log": float(np.sqrt(np.mean((np.log(p) - np.log(g)) ** 2))),
    }


def compute_depth_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    min_depth: float = 1e-3,
    max_depth: float = 80.0,
    scale_mode: str = "both",
) -> Dict[str, float]:
    """
    Depth evaluation in dual mode: raw and median-scaled.

    Args:
        pred: [H, W] float32 predicted depth
        gt:   [H, W] float32 ground truth depth
        min_depth, max_depth: valid depth range
        scale_mode: 'raw', 'median', or 'both'

    Returns:
        dict of metric_name -> value.
        Median-scaled metrics are prefixed with 'ms_'.
    """
    valid = (
        (gt > min_depth) & (gt < max_depth)
        & (pred > min_depth) & (pred < max_depth)
        & np.isfinite(gt) & np.isfinite(pred)
    )
    if np.sum(valid) < 10:
        return {}

    p_raw = pred[valid].astype(np.float64)
    g = gt[valid].astype(np.float64)

    results: Dict[str, float] = {}

    if scale_mode in ("raw", "both"):
        raw = _depth_core(p_raw, g)
        for k, v in raw.items():
            results[k] = v

    if scale_mode in ("median", "both"):
        scale = np.median(g) / max(np.median(p_raw), 1e-8)
        p_scaled = p_raw * scale
        ms = _depth_core(p_scaled, g)
        for k, v in ms.items():
            results[f"ms_{k}"] = v

    return results


# ============================================================================
# IV. CAMERA POSE METRICS
# ============================================================================

def _umeyama_alignment(
    src: np.ndarray, dst: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Umeyama alignment: dst ≈ s * R @ src + t."""
    mx, my = src.mean(0), dst.mean(0)
    Xc, Yc = src - mx, dst - my
    S = (Xc.T @ Yc) / len(src)
    U, D, Vt = np.linalg.svd(S)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    var = (Xc ** 2).sum() / len(src)
    s = np.trace(np.diag(D)) / max(var, 1e-12)
    t = my - s * (R @ mx)
    return float(s), R, t


def _rotation_error_deg(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    """Angular error in degrees between two rotation matrices."""
    R_diff = R_pred @ R_gt.T
    cos_angle = np.clip((np.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def compute_pose_metrics(
    pred_poses: List[Tuple[int, np.ndarray]],
    gt_poses: List[Tuple[int, np.ndarray]],
    rpe_delta: int = 1,
) -> Dict[str, float]:
    """
    Full camera pose evaluation: raw + Umeyama-aligned + RPE.

    Args:
        pred_poses: list of (frame_index, 4x4 extrinsic matrix)
        gt_poses:   list of (frame_index, 4x4 extrinsic matrix)
        rpe_delta: frame gap for Relative Pose Error (default 1)

    Returns:
        dict with pose_t_raw, pose_ate, pose_r_deg, rpe_trans, rpe_rot_deg,
        and matched_frames.
    """
    pred_map = {idx: T for idx, T in pred_poses}
    gt_map = {idx: T for idx, T in gt_poses}
    common = sorted(set(pred_map.keys()) & set(gt_map.keys()))

    if len(common) < 2:
        return {}

    Tp = np.stack([pred_map[i] for i in common])
    Tg = np.stack([gt_map[i] for i in common])

    cp = Tp[:, :3, 3].copy()
    cg = Tg[:, :3, 3].copy()

    # --- Raw translation error (before alignment) ---
    raw_t_errors = np.linalg.norm(cp - cg, axis=1)
    pose_t_raw = float(np.mean(raw_t_errors))

    # --- Umeyama-aligned ATE ---
    s, Ra, ta = _umeyama_alignment(cp, cg)
    cp_aligned = (s * (Ra @ cp.T)).T + ta
    ate_rmse = float(np.sqrt(np.mean(np.sum((cp_aligned - cg) ** 2, axis=1))))

    # --- Per-frame rotation error (after alignment) ---
    rot_errors = []
    for i in range(len(common)):
        Rp = Ra @ Tp[i, :3, :3]
        Rg = Tg[i, :3, :3]
        rot_errors.append(_rotation_error_deg(Rp, Rg))
    pose_r_deg = float(np.mean(rot_errors))

    # --- RPE with configurable delta ---
    rpe_trans, rpe_rot = [], []
    for i in range(len(common) - rpe_delta):
        j = i + rpe_delta
        Tpi = Tp[i].copy()
        Tpj = Tp[j].copy()
        Tpi[:3, :3] = Ra @ Tpi[:3, :3]
        Tpj[:3, :3] = Ra @ Tpj[:3, :3]
        Tpi[:3, 3] = s * (Ra @ Tpi[:3, 3]) + ta
        Tpj[:3, 3] = s * (Ra @ Tpj[:3, 3]) + ta

        rel_pred = np.linalg.inv(Tpi) @ Tpj
        rel_gt = np.linalg.inv(Tg[i]) @ Tg[j]
        E = np.linalg.inv(rel_gt) @ rel_pred
        rpe_trans.append(float(np.linalg.norm(E[:3, 3])))
        rpe_rot.append(_rotation_error_deg(E[:3, :3], np.eye(3)))

    results: Dict[str, float] = {
        "pose_t_raw": pose_t_raw,
        "pose_ate": ate_rmse,
        "pose_r_deg": pose_r_deg,
        "matched_frames": float(len(common)),
    }
    if rpe_trans:
        results["rpe_trans"] = float(np.mean(rpe_trans))
        results["rpe_rot_deg"] = float(np.mean(rpe_rot))

    return results
