"""
Automatic data discovery for event-based reconstruction evaluation.

Recursively scans directories, infers file structure and modalities,
and groups data into sequences aligned by frame index.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Naming / extension patterns
# ---------------------------------------------------------------------------

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
DEPTH_EXTS = {".npy", ".npz", ".png", ".pfm", ".exr"}
EVENT_EXTS = {".pt", ".pth", ".npy", ".npz", ".png"}
POSE_EXTS = {".txt", ".npy", ".npz", ".json"}

PRED_KEYWORDS = {"pred", "predicted", "output", "result", "recon", "infer", "est"}
GT_KEYWORDS = {"gt", "ground_truth", "groundtruth", "ref", "reference", "target", "label"}

RGB_DIR_KEYWORDS = {"images", "image", "rgb", "color", "frames", "frame"}
DEPTH_DIR_KEYWORDS = {"depth", "depths", "disparity", "disp"}
EVENT_DIR_KEYWORDS = {"events", "event", "voxel", "voxels", "event_frames", "ev"}
POSE_DIR_KEYWORDS = {"cameras", "camera", "pose", "poses", "cam", "cams", "extrinsics"}

EXCLUDED_DIR_KEYWORDS = {"vis", "viz", "debug", "overlay", "preview", "cache", "tmp"}
MIN_LEAF_IMAGES = 10

_IDX_RE = re.compile(r"(\d+)")


def _extract_index(filename: str) -> Optional[int]:
    """Extract the first numeric index from a filename."""
    m = _IDX_RE.search(Path(filename).stem)
    return int(m.group(1)) if m else None


def _classify_role(path: Path) -> str:
    """Classify whether a path belongs to 'pred' or 'gt' based on ancestor names."""
    parts = [p.lower() for p in path.parts]
    for p in parts:
        if any(kw in p for kw in PRED_KEYWORDS):
            return "pred"
        if any(kw in p for kw in GT_KEYWORDS):
            return "gt"
    return "unknown"


def _dir_matches_keywords(dirname: str, keywords: set) -> bool:
    lower = dirname.lower()
    return any(kw == lower or kw in lower for kw in keywords)


def _is_excluded_dir(path: Path) -> bool:
    """Check if any ancestor or the directory itself matches exclusion keywords."""
    for part in path.parts:
        if any(kw == part.lower() or kw in part.lower() for kw in EXCLUDED_DIR_KEYWORDS):
            return True
    return False


def _count_image_files(d: Path) -> int:
    """Count image files in a directory (non-recursive)."""
    return sum(1 for f in d.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class FrameData:
    """Aligned data for a single frame index."""
    index: int
    rgb_pred: Optional[Path] = None
    rgb_gt: Optional[Path] = None
    depth_pred: Optional[Path] = None
    depth_gt: Optional[Path] = None
    event: Optional[Path] = None


@dataclass
class PoseData:
    """Camera pose for a single frame."""
    index: int
    extrinsic: np.ndarray  # 4x4
    intrinsic: Optional[np.ndarray] = None  # 3x3


@dataclass
class SequenceData:
    """All data discovered for a single sequence."""
    name: str
    root: Path
    frames: List[FrameData] = field(default_factory=list)
    poses_pred: List[PoseData] = field(default_factory=list)
    poses_gt: List[PoseData] = field(default_factory=list)

    @property
    def has_rgb_pairs(self) -> bool:
        return any(f.rgb_pred and f.rgb_gt for f in self.frames)

    @property
    def has_depth_pairs(self) -> bool:
        return any(f.depth_pred and f.depth_gt for f in self.frames)

    @property
    def has_events(self) -> bool:
        return any(f.event is not None for f in self.frames)

    @property
    def has_pose_pairs(self) -> bool:
        return len(self.poses_pred) > 0 and len(self.poses_gt) > 0

    @property
    def num_rgb_pairs(self) -> int:
        return sum(1 for f in self.frames if f.rgb_pred and f.rgb_gt)

    def summary(self) -> str:
        lines = [
            f"Sequence: {self.name}",
            f"  Root: {self.root}",
            f"  Frames: {len(self.frames)}",
            f"  RGB pairs: {self.num_rgb_pairs}",
            f"  Depth pairs: {sum(1 for f in self.frames if f.depth_pred and f.depth_gt)}",
            f"  Event frames: {sum(1 for f in self.frames if f.event)}",
            f"  Pred poses: {len(self.poses_pred)}",
            f"  GT poses: {len(self.poses_gt)}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# File loaders
# ---------------------------------------------------------------------------

def load_image(path: Path) -> np.ndarray:
    """Load image as float32 [H,W,3] in [0,1] range, RGB order."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Cannot read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return img


def load_depth(path: Path, meta_path: Optional[Path] = None) -> np.ndarray:
    """Load depth map as float32 [H,W]. Handles .npy, .npz, .png, .pfm."""
    suffix = path.suffix.lower()
    if suffix == ".npy":
        d = np.load(str(path)).astype(np.float32)
    elif suffix == ".npz":
        z = np.load(str(path))
        keys = list(z.keys())
        depth_key = None
        for k in ["depth", "arr_0"] + keys:
            if k in z:
                depth_key = k
                break
        d = z[depth_key].astype(np.float32)
    elif suffix == ".png":
        d = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if d is None:
            raise IOError(f"Cannot read depth: {path}")
        d = d.astype(np.float32)
        if d.max() > 256:
            # 16-bit depth: check if we have meta for denormalization
            if meta_path and meta_path.exists():
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                dmin, dmax = meta["min"], meta["max"]
                d = d / 65535.0 * (dmax - dmin) + dmin
            else:
                d = d / 1000.0  # assume mm -> m
    elif suffix == ".pfm":
        d = _read_pfm(path)
    else:
        raise ValueError(f"Unsupported depth format: {suffix}")

    if d.ndim == 3:
        d = d[:, :, 0]

    mask_invalid = np.isinf(d) | np.isnan(d)
    d[mask_invalid] = 0.0
    return d


def _read_pfm(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        header = f.readline().rstrip().decode("ascii")
        if header == "PF":
            channels = 3
        elif header == "Pf":
            channels = 1
        else:
            raise ValueError(f"Not a PFM file: {path}")
        dims = f.readline().rstrip().decode("ascii")
        w, h = map(int, dims.split())
        scale = float(f.readline().rstrip().decode("ascii"))
        endian = "<" if scale < 0 else ">"
        data = np.frombuffer(f.read(), endian + "f")
        shape = (h, w, channels) if channels > 1 else (h, w)
        data = np.reshape(data, shape)
        data = np.flipud(data).astype(np.float32)
    return data


def load_event(path: Path) -> np.ndarray:
    """Load event data as float32 [C,H,W] or [H,W]."""
    import torch

    suffix = path.suffix.lower()
    if suffix in (".pt", ".pth"):
        evt = torch.load(str(path), map_location="cpu")
        if not isinstance(evt, (torch.Tensor, np.ndarray)):
            evt = np.array(evt, dtype=np.float32)
        if isinstance(evt, torch.Tensor):
            evt = evt.numpy()
        evt = evt.astype(np.float32)
    elif suffix == ".npy":
        evt = np.load(str(path)).astype(np.float32)
    elif suffix == ".npz":
        z = np.load(str(path))
        keys = list(z.keys())
        k = "event" if "event" in z else keys[0]
        evt = z[k].astype(np.float32)
    elif suffix == ".png":
        evt = cv2.imread(str(path), cv2.IMREAD_UNCHANGED).astype(np.float32)
        if evt.ndim == 3:
            evt = evt.transpose(2, 0, 1)
    else:
        raise ValueError(f"Unsupported event format: {suffix}")
    return evt


def load_pose_txt(path: Path) -> np.ndarray:
    """Load 4x4 pose matrix from a text file."""
    return np.loadtxt(str(path), dtype=np.float64).reshape(4, 4)


def load_poses_from_transforms_json(path: Path) -> List[PoseData]:
    """Load poses from NeRF-style transforms.json."""
    with open(path, "r") as f:
        data = json.load(f)
    poses = []
    for frame in data.get("frames", []):
        idx = _extract_index(frame.get("file_path", ""))
        if idx is None:
            continue
        T = np.array(frame["transform_matrix"], dtype=np.float64)
        K = None
        fl_x = frame.get("fl_x", data.get("fl_x"))
        fl_y = frame.get("fl_y", data.get("fl_y"))
        cx = frame.get("cx", data.get("cx"))
        cy = frame.get("cy", data.get("cy"))
        if fl_x is not None:
            K = np.array([
                [float(fl_x), 0, float(cx or 0)],
                [0, float(fl_y or fl_x), float(cy or 0)],
                [0, 0, 1],
            ], dtype=np.float64)
        ext = np.linalg.inv(T) if np.abs(np.linalg.det(T[:3, :3]) - 1.0) < 0.1 else T
        poses.append(PoseData(index=idx, extrinsic=ext, intrinsic=K))
    return poses


def load_poses_from_npz(path: Path) -> List[PoseData]:
    """Load poses from cameras.npz (extrinsics, intrinsics, frame_names)."""
    z = np.load(str(path), allow_pickle=True)
    ext = z.get("extrinsics", z.get("poses"))
    intr = z.get("intrinsics", None)
    names = z.get("frame_names", None)
    poses = []
    for i in range(len(ext)):
        idx = i
        if names is not None:
            parsed = _extract_index(str(names[i]))
            if parsed is not None:
                idx = parsed
        K = intr[i] if intr is not None and i < len(intr) else None
        poses.append(PoseData(index=idx, extrinsic=ext[i].astype(np.float64), intrinsic=K))
    return poses


def load_poses_from_selected_npz(path: Path) -> List[PoseData]:
    """Load poses from Eventbenchmark-style poses_selected.npz.

    Expected keys: 'k' (frame indices), 'pose' (N,4,4 extrinsics).
    Some screens only store index mappings without pose matrices.
    """
    z = np.load(str(path), allow_pickle=True)
    if "pose" not in z:
        return []
    poses_arr = z["pose"]
    if poses_arr.ndim != 3 or poses_arr.shape[1:] != (4, 4):
        return []
    indices = z["k"] if "k" in z else np.arange(len(poses_arr))
    poses = []
    for i in range(len(indices)):
        ext = poses_arr[i].astype(np.float64)
        if ext.shape == (3, 4):
            e4 = np.eye(4, dtype=np.float64)
            e4[:3, :4] = ext
            ext = e4
        poses.append(PoseData(index=int(indices[i]), extrinsic=ext))
    return poses


def load_gt_poses_from_cam_dir(cam_dir: Path) -> List[PoseData]:
    """Load GT poses from DL3DV-style .npz per-frame files (00001.npz etc.)."""
    npz_files = sorted(cam_dir.glob("*.npz"))
    poses = []
    for fp in npz_files:
        m = re.search(r"(\d+)", fp.stem)
        if not m:
            continue
        idx = int(m.group(1)) - 1
        z = np.load(str(fp))
        if "pose" in z:
            ext = z["pose"].astype(np.float64)
            K = z["intrinsic"].astype(np.float64) if "intrinsic" in z else None
            if ext.shape == (3, 4):
                e4 = np.eye(4, dtype=np.float64)
                e4[:3, :4] = ext
                ext = e4
            poses.append(PoseData(index=idx, extrinsic=ext, intrinsic=K))
    return poses


# ---------------------------------------------------------------------------
# Sequence discovery
# ---------------------------------------------------------------------------

class SequenceDiscoverer:
    """
    Recursively scan a root directory and discover evaluation sequences.

    Strategy:
    1. Look for checkpoint-style output dirs (transforms.json marker)
    2. Look for pred/gt paired directories
    3. Look for directories with images/ and events/ subdirs
    4. Fallback: treat each leaf directory with images as a sequence
    """

    def __init__(self, root: str | Path, gt_root: Optional[str | Path] = None,
                 images_subdir: str = "images"):
        self.root = Path(root)
        self.gt_root = Path(gt_root) if gt_root else None
        self.images_subdir = images_subdir

    @staticmethod
    def _is_eventbenchmark_screen(d: Path) -> bool:
        """Detect Eventbenchmark screen directory by its signature files."""
        return (
            (d / "meta.json").is_file()
            and (d / "cameras.yaml").is_file()
            and (d / "images").is_dir()
            and (d / "events").is_dir()
        )

    def _parse_eventbenchmark_screen(self, d: Path) -> Optional[SequenceData]:
        """Parse a single Eventbenchmark screen directory.

        Layout:
            screen-XXXXXX/
              meta.json, cameras.yaml
              images/000000.png ... 000048.png   (GT RGB)
              events/000000.pt  ... 000048.pt   (event voxels)
              meta/poses_selected.npz            (GT poses: keys 'k', 'pose')
        """
        category = d.parent.name if d.parent != self.root else ""
        name = f"{category}/{d.name}" if category else d.name
        seq = SequenceData(name=name, root=d)
        frame_map: Dict[int, FrameData] = {}

        img_dir = d / "images"
        for f in sorted(img_dir.iterdir()):
            if f.suffix.lower() in IMAGE_EXTS:
                idx = _extract_index(f.name)
                if idx is not None:
                    fd = FrameData(index=idx, rgb_gt=f)
                    frame_map[idx] = fd

        evt_dir = d / "events"
        if evt_dir.is_dir():
            for f in sorted(evt_dir.iterdir()):
                if f.suffix.lower() in EVENT_EXTS:
                    idx = _extract_index(f.name)
                    if idx is not None:
                        if idx in frame_map:
                            frame_map[idx].event = f
                        else:
                            frame_map[idx] = FrameData(index=idx, event=f)

        seq.frames = sorted(frame_map.values(), key=lambda fd: fd.index)

        poses_npz = d / "meta" / "poses_selected.npz"
        if poses_npz.is_file():
            try:
                seq.poses_gt = load_poses_from_selected_npz(poses_npz)
            except Exception as e:
                print(f"  [WARN] Failed to load poses from {poses_npz}: {e}")

        return seq if seq.frames else None

    def discover(self) -> List[SequenceData]:
        sequences = []

        # Strategy 0: Eventbenchmark screen directories
        #   Single screen dir (root itself is a screen)
        if self._is_eventbenchmark_screen(self.root):
            seq = self._parse_eventbenchmark_screen(self.root)
            if seq and seq.frames:
                sequences.append(seq)
        else:
            # Multiple screens: scan for screen-* children recursively
            for meta_json in self.root.rglob("meta.json"):
                candidate = meta_json.parent
                if self._is_eventbenchmark_screen(candidate):
                    seq = self._parse_eventbenchmark_screen(candidate)
                    if seq and seq.frames:
                        sequences.append(seq)

        if sequences:
            if self.gt_root:
                for seq in sequences:
                    self._attach_gt(seq)
            return self._deduplicate(sequences)

        # Strategy 1: checkpoint output directories with transforms.json
        for tj in self.root.rglob("transforms.json"):
            seq = self._parse_checkpoint_dir(tj.parent)
            if seq and seq.frames:
                sequences.append(seq)

        if sequences:
            if self.gt_root:
                for seq in sequences:
                    self._attach_gt(seq)
            return self._deduplicate(sequences)

        # Strategy 2: look for pred/gt paired directories
        sequences = self._discover_pred_gt_pairs()
        if sequences:
            return self._deduplicate(sequences)

        # Strategy 3: directories with images/ subdirectory
        for img_dir in self.root.rglob("images"):
            if img_dir.is_dir():
                seq = self._parse_generic_dir(img_dir.parent)
                if seq and seq.frames:
                    sequences.append(seq)

        if not sequences:
            # Strategy 4: leaf directories with ≥MIN_LEAF_IMAGES images
            seen_dirs: set = set()
            for ext in IMAGE_EXTS:
                for img_file in self.root.rglob(f"*{ext}"):
                    d = img_file.parent
                    if d in seen_dirs:
                        continue
                    seen_dirs.add(d)
                    if _is_excluded_dir(d):
                        continue
                    seq = self._parse_leaf_dir(d)
                    if seq and seq.frames:
                        sequences.append(seq)

        if self.gt_root:
            for seq in sequences:
                self._attach_gt(seq)

        return self._deduplicate(sequences)

    def _parse_checkpoint_dir(self, d: Path) -> Optional[SequenceData]:
        """Parse a checkpoint-style output directory."""
        seq = SequenceData(name=d.name, root=d)

        img_dir = d / self.images_subdir
        if not img_dir.is_dir():
            img_dir = d / "images"
        image_files = {}
        if img_dir.is_dir():
            for f in sorted(img_dir.iterdir()):
                if f.suffix.lower() in IMAGE_EXTS:
                    idx = _extract_index(f.name)
                    if idx is not None:
                        image_files[idx] = f

        # Load depth
        depth_dir = d / "depth"
        depth_files = {}
        if depth_dir.is_dir():
            for f in sorted(depth_dir.iterdir()):
                if f.suffix.lower() == ".npy" and "_meta" not in f.stem and "_vis" not in f.stem:
                    idx = _extract_index(f.name)
                    if idx is not None:
                        depth_files[idx] = f

        # Load events (look in data_root if known)
        event_files: Dict[int, Path] = {}

        # Build frames
        all_indices = sorted(set(image_files.keys()) | set(depth_files.keys()))
        for idx in all_indices:
            fd = FrameData(index=idx)
            fd.rgb_pred = image_files.get(idx)
            fd.depth_pred = depth_files.get(idx)
            fd.event = event_files.get(idx)
            seq.frames.append(fd)

        # Load poses from transforms.json
        tf_path = d / "transforms.json"
        if tf_path.exists():
            seq.poses_pred = load_poses_from_transforms_json(tf_path)

        # Also try cameras.npz
        cam_npz = d / "cameras" / "cameras.npz"
        if cam_npz.exists() and not seq.poses_pred:
            seq.poses_pred = load_poses_from_npz(cam_npz)

        return seq

    def _parse_generic_dir(self, d: Path) -> Optional[SequenceData]:
        """Parse a generic sequence directory."""
        seq = SequenceData(name=d.name, root=d)

        for subdir in d.iterdir():
            if not subdir.is_dir():
                continue
            dirname = subdir.name.lower()

            if _dir_matches_keywords(dirname, RGB_DIR_KEYWORDS):
                role = _classify_role(subdir)
                self._scan_images(subdir, seq, role)
            elif _dir_matches_keywords(dirname, DEPTH_DIR_KEYWORDS):
                role = _classify_role(subdir)
                self._scan_depths(subdir, seq, role)
            elif _dir_matches_keywords(dirname, EVENT_DIR_KEYWORDS):
                self._scan_events(subdir, seq)
            elif _dir_matches_keywords(dirname, POSE_DIR_KEYWORDS):
                role = _classify_role(subdir)
                self._scan_poses(subdir, seq, role)

        # Try transforms.json in root
        tf = d / "transforms.json"
        if tf.exists() and not seq.poses_pred:
            seq.poses_pred = load_poses_from_transforms_json(tf)

        return seq if seq.frames else None

    def _parse_leaf_dir(self, d: Path) -> Optional[SequenceData]:
        """Parse a leaf directory containing image files.

        STRICT: requires ≥MIN_LEAF_IMAGES valid images and excludes
        directories whose names match EXCLUDED_DIR_KEYWORDS.
        """
        if _is_excluded_dir(d):
            return None
        if _count_image_files(d) < MIN_LEAF_IMAGES:
            return None
        seq = SequenceData(name=d.name, root=d)
        role = _classify_role(d)
        self._scan_images(d, seq, role if role != "unknown" else "pred")
        return seq if seq.frames else None

    def _discover_pred_gt_pairs(self) -> List[SequenceData]:
        """Find directories that appear as pred/gt pairs."""
        sequences = []
        pred_dirs = []
        gt_dirs = []

        for d in self.root.rglob("*"):
            if not d.is_dir():
                continue
            role = _classify_role(d)
            if role == "pred":
                pred_dirs.append(d)
            elif role == "gt":
                gt_dirs.append(d)

        for pd in pred_dirs:
            seq = self._parse_generic_dir(pd)
            if seq:
                for gd in gt_dirs:
                    if gd.parent == pd.parent or self.gt_root:
                        self._attach_gt_from_dir(seq, gd)
                sequences.append(seq)
        return sequences

    def _scan_images(self, d: Path, seq: SequenceData, role: str):
        frame_map = {f.index: f for f in seq.frames}
        for f in sorted(d.iterdir()):
            if f.suffix.lower() not in IMAGE_EXTS:
                continue
            idx = _extract_index(f.name)
            if idx is None:
                continue
            if idx not in frame_map:
                frame_map[idx] = FrameData(index=idx)
                seq.frames.append(frame_map[idx])
            if role == "gt":
                frame_map[idx].rgb_gt = f
            else:
                frame_map[idx].rgb_pred = f

    def _scan_depths(self, d: Path, seq: SequenceData, role: str):
        frame_map = {f.index: f for f in seq.frames}
        for f in sorted(d.iterdir()):
            if f.suffix.lower() not in DEPTH_EXTS:
                continue
            if "_vis" in f.stem or "_meta" in f.stem:
                continue
            idx = _extract_index(f.name)
            if idx is None:
                continue
            if idx not in frame_map:
                frame_map[idx] = FrameData(index=idx)
                seq.frames.append(frame_map[idx])
            if role == "gt":
                frame_map[idx].depth_gt = f
            else:
                frame_map[idx].depth_pred = f

    def _scan_events(self, d: Path, seq: SequenceData):
        frame_map = {f.index: f for f in seq.frames}
        for f in sorted(d.iterdir()):
            if f.suffix.lower() not in EVENT_EXTS:
                continue
            idx = _extract_index(f.name)
            if idx is None:
                continue
            if idx not in frame_map:
                frame_map[idx] = FrameData(index=idx)
                seq.frames.append(frame_map[idx])
            frame_map[idx].event = f

    def _scan_poses(self, d: Path, seq: SequenceData, role: str):
        # transforms.json
        tf = d / "transforms.json"
        if tf.exists():
            poses = load_poses_from_transforms_json(tf)
            if role == "gt":
                seq.poses_gt = poses
            else:
                seq.poses_pred = poses
            return

        # cameras.npz
        npz = d / "cameras.npz"
        if npz.exists():
            poses = load_poses_from_npz(npz)
            if role == "gt":
                seq.poses_gt = poses
            else:
                seq.poses_pred = poses
            return

        # Per-frame .npz (DL3DV style)
        npz_files = sorted(d.glob("*.npz"))
        if npz_files:
            poses = load_gt_poses_from_cam_dir(d)
            if poses:
                if role == "gt":
                    seq.poses_gt = poses
                else:
                    seq.poses_pred = poses
                return

        # Per-frame extrinsic txt files
        ext_files = sorted(d.glob("*extrinsic*"))
        if ext_files:
            poses = []
            for ef in ext_files:
                idx = _extract_index(ef.name)
                if idx is not None:
                    try:
                        T = load_pose_txt(ef)
                        # Try to find matching intrinsic
                        K = None
                        intr_f = ef.parent / ef.name.replace("extrinsic", "intrinsic")
                        if intr_f.exists():
                            K = np.loadtxt(str(intr_f), dtype=np.float64).reshape(3, 3)
                        poses.append(PoseData(index=idx, extrinsic=T, intrinsic=K))
                    except Exception:
                        pass
            if role == "gt":
                seq.poses_gt = poses
            else:
                seq.poses_pred = poses

    def _attach_gt(self, seq: SequenceData):
        """Attach GT data from gt_root to an existing sequence."""
        if not self.gt_root or not self.gt_root.exists():
            return

        candidates = [
            self.gt_root,
            self.gt_root / seq.name,
        ]
        for cand in candidates:
            if cand.is_dir():
                if self._is_eventbenchmark_screen(cand):
                    self._attach_gt_from_eventbenchmark(seq, cand)
                else:
                    self._attach_gt_from_dir(seq, cand)
                if seq.has_rgb_pairs or seq.has_depth_pairs or seq.has_pose_pairs:
                    break

    def _attach_gt_from_eventbenchmark(self, seq: SequenceData, screen_dir: Path):
        """Attach GT from an Eventbenchmark screen directory."""
        frame_map = {f.index: f for f in seq.frames}

        img_dir = screen_dir / "images"
        if img_dir.is_dir():
            for f in sorted(img_dir.iterdir()):
                if f.suffix.lower() in IMAGE_EXTS:
                    idx = _extract_index(f.name)
                    if idx is not None and idx in frame_map:
                        frame_map[idx].rgb_gt = f

        evt_dir = screen_dir / "events"
        if evt_dir.is_dir():
            for f in sorted(evt_dir.iterdir()):
                if f.suffix.lower() in EVENT_EXTS:
                    idx = _extract_index(f.name)
                    if idx is not None and idx in frame_map:
                        frame_map[idx].event = f

        poses_npz = screen_dir / "meta" / "poses_selected.npz"
        if poses_npz.is_file():
            try:
                seq.poses_gt = load_poses_from_selected_npz(poses_npz)
            except Exception as e:
                print(f"  [WARN] Failed to load GT poses: {e}")

    def _attach_gt_from_dir(self, seq: SequenceData, gt_dir: Path):
        """Attach GT modalities from a GT directory."""
        frame_map = {f.index: f for f in seq.frames}

        # GT images
        for subdir_name in ["images", "image", "rgb", "color", "frames"]:
            gt_img_dir = gt_dir / subdir_name
            if gt_img_dir.is_dir():
                for f in sorted(gt_img_dir.iterdir()):
                    if f.suffix.lower() in IMAGE_EXTS:
                        idx = _extract_index(f.name)
                        if idx is not None and idx in frame_map:
                            frame_map[idx].rgb_gt = f
                break

        # GT depth
        for subdir_name in ["depth", "depths"]:
            gt_depth_dir = gt_dir / subdir_name
            if gt_depth_dir.is_dir():
                for f in sorted(gt_depth_dir.iterdir()):
                    if f.suffix.lower() in DEPTH_EXTS and "_vis" not in f.stem and "_meta" not in f.stem:
                        idx = _extract_index(f.name)
                        if idx is not None and idx in frame_map:
                            frame_map[idx].depth_gt = f
                break

        # GT events
        for subdir_name in ["events", "event", "voxel", "voxels"]:
            gt_evt_dir = gt_dir / subdir_name
            if gt_evt_dir.is_dir():
                for f in sorted(gt_evt_dir.iterdir()):
                    if f.suffix.lower() in EVENT_EXTS:
                        idx = _extract_index(f.name)
                        if idx is not None and idx in frame_map:
                            frame_map[idx].event = f
                break

        # GT poses: first check Eventbenchmark-style meta/poses_selected.npz
        poses_npz = gt_dir / "meta" / "poses_selected.npz"
        if poses_npz.is_file():
            try:
                seq.poses_gt = load_poses_from_selected_npz(poses_npz)
            except Exception:
                pass
        else:
            for subdir_name in ["cam", "cameras", "camera", "poses", "dense/cam"]:
                gt_pose_dir = gt_dir / subdir_name
                if gt_pose_dir.is_dir():
                    gt_npz = sorted(gt_pose_dir.glob("*.npz"))
                    if gt_npz:
                        seq.poses_gt = load_gt_poses_from_cam_dir(gt_pose_dir)
                    else:
                        self._scan_poses(gt_pose_dir, seq, "gt")
                    break

    @staticmethod
    def _deduplicate(sequences: List[SequenceData]) -> List[SequenceData]:
        seen = set()
        result = []
        for seq in sequences:
            key = str(seq.root)
            if key not in seen:
                seen.add(key)
                seq.frames.sort(key=lambda f: f.index)
                result.append(seq)
        return result
