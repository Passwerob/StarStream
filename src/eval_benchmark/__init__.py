"""
Event-based RGB Reconstruction Evaluation Benchmark.

Research-grade metrics tailored for event camera systems under
low light, motion blur, and high dynamic range.
"""

from .data_discovery import SequenceDiscoverer, SequenceData
from .metrics import (
    compute_lpips,
    compute_gradient_error,
    compute_edge_f1,
    compute_temporal_error,
    compute_event_consistency,
    compute_depth_metrics,
    compute_pose_metrics,
)
from .evaluator import BenchmarkEvaluator

__all__ = [
    "SequenceDiscoverer",
    "SequenceData",
    "BenchmarkEvaluator",
    "compute_lpips",
    "compute_gradient_error",
    "compute_edge_f1",
    "compute_temporal_error",
    "compute_event_consistency",
    "compute_depth_metrics",
    "compute_pose_metrics",
]
