"""
Main evaluator: orchestrates data loading, metric computation, and result output.

Research-grade pipeline with configurable LPIPS backbone, dual-mode depth
evaluation, physically-consistent event comparison, and configurable RPE delta.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .data_discovery import (
    SequenceData,
    SequenceDiscoverer,
    load_depth,
    load_event,
    load_image,
)
from .metrics import (
    compute_depth_metrics,
    compute_edge_f1,
    compute_event_consistency,
    compute_gradient_error,
    compute_lpips,
    compute_pose_metrics,
    compute_temporal_error,
)


def _tqdm_wrap(iterable, **kwargs):
    try:
        from tqdm import tqdm
        return tqdm(iterable, **kwargs)
    except ImportError:
        return iterable


class BenchmarkEvaluator:
    """
    End-to-end evaluator for event-based RGB reconstruction.

    Handles missing modalities gracefully: only computes metrics for which
    data is available.
    """

    def __init__(
        self,
        root: str,
        gt_root: Optional[str] = None,
        output_dir: Optional[str] = None,
        device: Optional[str] = None,
        lpips_backbone: str = "alex",
        depth_scale_mode: str = "both",
        rpe_delta: int = 1,
        event_threshold: float = 0.01,
        images_subdir: str = "images",
    ):
        self.root = Path(root)
        self.gt_root = Path(gt_root) if gt_root else None
        self.output_dir = Path(output_dir) if output_dir else self.root / "eval_results"
        self.device = device
        self.lpips_backbone = lpips_backbone
        self.depth_scale_mode = depth_scale_mode
        self.rpe_delta = rpe_delta
        self.event_threshold = event_threshold
        self.images_subdir = images_subdir

    def run(self) -> Dict[str, Any]:
        """
        Execute the full evaluation pipeline.

        Returns:
            Nested dict: { sequence_name: { metric: value }, ..., "mean": {...} }
        """
        print(f"[Benchmark] Scanning: {self.root}")
        discoverer = SequenceDiscoverer(self.root, self.gt_root,
                                        images_subdir=self.images_subdir)
        sequences = discoverer.discover()

        if not sequences:
            print("[Benchmark] No valid sequences found.")
            return {}

        print(f"[Benchmark] Found {len(sequences)} sequence(s):")
        for seq in sequences:
            print(seq.summary())
            print()

        all_results: Dict[str, Dict[str, Any]] = {}

        for seq in _tqdm_wrap(sequences, desc="Evaluating sequences"):
            print(f"\n{'='*60}")
            print(f"  Evaluating: {seq.name}")
            print(f"{'='*60}")
            result = self._evaluate_sequence(seq)
            all_results[seq.name] = result
            self._print_sequence_result(seq.name, result)

        mean_result = self._compute_mean(all_results)
        all_results["mean"] = mean_result

        print(f"\n{'='*60}")
        print("  GLOBAL MEAN")
        print(f"{'='*60}")
        self._print_sequence_result("mean", mean_result)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._save_json(all_results)
        self._save_csv(all_results)

        return all_results

    def _evaluate_sequence(self, seq: SequenceData) -> Dict[str, Any]:
        result: Dict[str, Any] = {}

        if seq.has_rgb_pairs:
            result.update(self._eval_rgb_metrics(seq))

        if seq.has_events and seq.num_rgb_pairs >= 2:
            result.update(self._eval_event_consistency(seq))

        if seq.has_depth_pairs:
            result.update(self._eval_depth_metrics(seq))

        if seq.has_pose_pairs:
            result.update(self._eval_pose_metrics(seq))

        return result

    # -----------------------------------------------------------------
    # RGB / Structure
    # -----------------------------------------------------------------
    def _eval_rgb_metrics(self, seq: SequenceData) -> Dict[str, Any]:
        pairs = [(f.index, f.rgb_pred, f.rgb_gt) for f in seq.frames if f.rgb_pred and f.rgb_gt]
        if not pairs:
            return {}

        lpips_scores: List[float] = []
        grad_errors: List[float] = []
        edge_f1_scores: List[float] = []
        temporal_errors: List[float] = []

        prev_pred = None
        prev_gt = None

        for idx, pred_path, gt_path in _tqdm_wrap(pairs, desc="  RGB metrics", leave=False):
            try:
                pred_img = load_image(pred_path)
                gt_img = load_image(gt_path)
            except Exception as e:
                print(f"  [WARN] Failed to load frame {idx}: {e}")
                continue

            if pred_img.shape != gt_img.shape:
                h = min(pred_img.shape[0], gt_img.shape[0])
                w = min(pred_img.shape[1], gt_img.shape[1])
                pred_img = pred_img[:h, :w]
                gt_img = gt_img[:h, :w]

            try:
                score = compute_lpips(pred_img, gt_img, device=self.device, backbone=self.lpips_backbone)
                lpips_scores.append(score)
            except Exception as e:
                print(f"  [WARN] LPIPS failed for frame {idx}: {e}")

            grad_errors.append(compute_gradient_error(pred_img, gt_img))

            ef = compute_edge_f1(pred_img, gt_img)
            edge_f1_scores.append(ef["f1"])

            if prev_pred is not None and prev_gt is not None:
                temporal_errors.append(compute_temporal_error(pred_img, prev_pred, gt_img, prev_gt))

            prev_pred = pred_img
            prev_gt = gt_img

        result: Dict[str, Any] = {}
        bb = self.lpips_backbone
        if lpips_scores:
            result[f"lpips_{bb}"] = float(np.mean(lpips_scores))
        if grad_errors:
            result["grad_error"] = float(np.mean(grad_errors))
        if edge_f1_scores:
            result["edge_f1"] = float(np.mean(edge_f1_scores))
        if temporal_errors:
            result["temporal_error"] = float(np.mean(temporal_errors))
        result["num_rgb_frames"] = len(pairs)
        result["_lpips_backbone"] = bb
        return result

    # -----------------------------------------------------------------
    # Event consistency
    # -----------------------------------------------------------------
    def _eval_event_consistency(self, seq: SequenceData) -> Dict[str, Any]:
        frames_with_events = [f for f in seq.frames if f.event and f.rgb_pred]
        if len(frames_with_events) < 2:
            return {}

        event_scores: Dict[str, List[float]] = {}
        prev_pred = None
        prev_idx = None

        for frame in _tqdm_wrap(frames_with_events, desc="  Event consistency", leave=False):
            try:
                pred_img = load_image(frame.rgb_pred)
                event_data = load_event(frame.event)
            except Exception as e:
                print(f"  [WARN] Event load failed for frame {frame.index}: {e}")
                prev_pred = None
                continue

            if prev_pred is not None:
                try:
                    scores = compute_event_consistency(
                        pred_img, prev_pred, event_data,
                        polarity_threshold=self.event_threshold,
                    )
                    for k, v in scores.items():
                        event_scores.setdefault(k, []).append(v)
                except Exception as e:
                    print(f"  [WARN] Event consistency failed {prev_idx}->{frame.index}: {e}")

            prev_pred = pred_img
            prev_idx = frame.index

        result: Dict[str, Any] = {}
        for k, vals in event_scores.items():
            result[k] = float(np.mean(vals))
        if event_scores:
            result["num_event_pairs"] = len(next(iter(event_scores.values())))
        return result

    # -----------------------------------------------------------------
    # Depth (dual mode)
    # -----------------------------------------------------------------
    def _eval_depth_metrics(self, seq: SequenceData) -> Dict[str, Any]:
        pairs = [(f.index, f.depth_pred, f.depth_gt) for f in seq.frames if f.depth_pred and f.depth_gt]
        if not pairs:
            return {}

        all_metrics: Dict[str, List[float]] = {}

        for idx, pred_path, gt_path in _tqdm_wrap(pairs, desc="  Depth metrics", leave=False):
            try:
                meta_path = pred_path.parent / f"{pred_path.stem}_meta.json"
                d_pred = load_depth(pred_path, meta_path if meta_path.exists() else None)
                d_gt = load_depth(gt_path)
            except Exception as e:
                print(f"  [WARN] Depth load failed for frame {idx}: {e}")
                continue

            if d_pred.shape != d_gt.shape:
                h = min(d_pred.shape[0], d_gt.shape[0])
                w = min(d_pred.shape[1], d_gt.shape[1])
                d_pred = d_pred[:h, :w]
                d_gt = d_gt[:h, :w]

            metrics = compute_depth_metrics(d_pred, d_gt, scale_mode=self.depth_scale_mode)
            for k, v in metrics.items():
                all_metrics.setdefault(k, []).append(v)

        result: Dict[str, Any] = {}
        for k, vals in all_metrics.items():
            result[k] = float(np.mean(vals))
        if all_metrics:
            result["num_depth_frames"] = len(pairs)
        return result

    # -----------------------------------------------------------------
    # Pose
    # -----------------------------------------------------------------
    def _eval_pose_metrics(self, seq: SequenceData) -> Dict[str, Any]:
        pred_list = [(p.index, p.extrinsic) for p in seq.poses_pred]
        gt_list = [(p.index, p.extrinsic) for p in seq.poses_gt]
        return compute_pose_metrics(pred_list, gt_list, rpe_delta=self.rpe_delta)

    # -----------------------------------------------------------------
    # Aggregation & output
    # -----------------------------------------------------------------
    @staticmethod
    def _compute_mean(all_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        metric_accum: Dict[str, List[float]] = {}
        skip_prefixes = ("num_", "_", "matched_frames")
        for result in all_results.values():
            for k, v in result.items():
                if isinstance(v, (int, float)) and not any(k.startswith(p) for p in skip_prefixes):
                    metric_accum.setdefault(k, []).append(float(v))
        return {k: float(np.mean(vals)) for k, vals in metric_accum.items()}

    @staticmethod
    def _print_sequence_result(name: str, result: Dict[str, Any]):
        if not result:
            print(f"  [{name}] No metrics computed.")
            return

        rgb_keys = [k for k in result if k.startswith("lpips_")] + [
            "grad_error", "edge_f1", "temporal_error", "num_rgb_frames",
        ]
        event_keys = ["event_l1", "event_corr", "polarity_acc", "num_event_pairs"]
        depth_keys = [
            "delta1", "delta2", "delta3", "abs_rel", "rmse_log",
            "ms_delta1", "ms_delta2", "ms_delta3", "ms_abs_rel", "ms_rmse_log",
            "num_depth_frames",
        ]
        pose_keys = [
            "pose_t_raw", "pose_ate", "pose_r_deg",
            "rpe_trans", "rpe_rot_deg", "matched_frames",
        ]

        def _print_group(title, keys):
            available = {k: result[k] for k in keys if k in result}
            if not available:
                return
            print(f"\n  {title}:")
            for k, v in available.items():
                if isinstance(v, float):
                    print(f"    {k:30s} = {v:.6f}")
                else:
                    print(f"    {k:30s} = {v}")

        _print_group("RGB / Perceptual", rgb_keys)
        _print_group("Event Consistency", event_keys)
        _print_group("Depth — Raw", [k for k in depth_keys if not k.startswith("ms_")])
        _print_group("Depth — Median-Scaled", [k for k in depth_keys if k.startswith("ms_")])
        _print_group("Camera Pose", pose_keys)

        printed = set(rgb_keys + event_keys + depth_keys + pose_keys)
        remaining = {k: v for k, v in result.items() if k not in printed and not k.startswith("_")}
        if remaining:
            _print_group("Other", list(remaining.keys()))

    def _save_json(self, results: Dict[str, Any]):
        out_path = self.output_dir / "eval_results.json"
        clean = _strip_internal_keys(results)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(clean, f, indent=2, default=_json_serializer)
        print(f"\n[Benchmark] JSON saved: {out_path}")

    def _save_csv(self, results: Dict[str, Any]):
        out_path = self.output_dir / "eval_results.csv"
        clean = _strip_internal_keys(results)
        all_keys = set()
        for r in clean.values():
            all_keys.update(r.keys())
        all_keys = sorted(all_keys)

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["sequence"] + all_keys)
            for seq_name, r in clean.items():
                row = [seq_name]
                for k in all_keys:
                    v = r.get(k, "")
                    if isinstance(v, float):
                        row.append(f"{v:.6f}")
                    else:
                        row.append(str(v) if v != "" else "")
                writer.writerow(row)
        print(f"[Benchmark] CSV saved: {out_path}")


def _strip_internal_keys(results: Dict[str, Any]) -> Dict[str, Any]:
    """Remove keys starting with '_' from output."""
    out = {}
    for seq, metrics in results.items():
        if isinstance(metrics, dict):
            out[seq] = {k: v for k, v in metrics.items() if not k.startswith("_")}
        else:
            out[seq] = metrics
    return out


def _json_serializer(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)
