import os.path as osp
import os
import sys
import itertools
import json
import hashlib
import time

import PIL.Image
import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(osp.join(osp.dirname(__file__), "..", ".."))
import cv2

from dust3r.datasets.base.base_multiview_dataset import BaseMultiViewDataset
from dust3r.datasets.base.easy_dataset import EasyDataset
from dust3r.datasets.utils.transforms import SeqColorJitter
from dust3r.utils.image import ImgNorm, imread_cv2


class DL3DV_Multi(BaseMultiViewDataset):
    def __init__(
        self,
        *args,
        split,
        ROOT,
        min_interval=1,
        max_interval=20,
        video_prob=0.5,
        fix_interval_prob=0.5,
        block_shuffle=25,
        **kwargs,
    ):
        self.ROOT = ROOT
        self.video = True
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.video_prob = video_prob
        self.fix_interval_prob = fix_interval_prob
        self.block_shuffle = block_shuffle
        self.is_metric = False
        super().__init__(*args, **kwargs)

        self.loaded_data = self._load_data()

    def _load_data(self):
        self.all_scenes = sorted(
            [f for f in os.listdir(self.ROOT) if os.path.isdir(osp.join(self.ROOT, f))]
        )
        subscenes = []
        for scene in self.all_scenes:
            # not empty
            subscenes.extend(
                [
                    osp.join(scene, f)
                    for f in os.listdir(osp.join(self.ROOT, scene))
                    if os.path.isdir(osp.join(self.ROOT, scene, f))
                    and len(os.listdir(osp.join(self.ROOT, scene, f))) > 0
                ]
            )

        offset = 0
        scenes = []
        sceneids = []
        images = []
        scene_img_list = []
        start_img_ids = []
        j = 0

        for scene_idx, scene in enumerate(subscenes):
            scene_dir = osp.join(self.ROOT, scene, "dense")
            rgb_paths = sorted(
                [
                    f
                    for f in os.listdir(os.path.join(scene_dir, "rgb"))
                    if f.endswith(".png")
                ]
            )
            assert len(rgb_paths) > 0, f"{scene_dir} is empty."
            num_imgs = len(rgb_paths)
            cut_off = (
                self.num_views if not self.allow_repeat else max(self.num_views // 3, 3)
            )

            if num_imgs < cut_off:
                print(f"Skipping {scene}")
                continue

            img_ids = list(np.arange(num_imgs) + offset)
            start_img_ids_ = img_ids[: num_imgs - cut_off + 1]

            scenes.append(scene)
            scene_img_list.append(img_ids)
            sceneids.extend([j] * num_imgs)
            images.extend(rgb_paths)
            start_img_ids.extend(start_img_ids_)
            offset += num_imgs
            j += 1

        self.scenes = scenes
        self.sceneids = sceneids
        self.images = images
        self.start_img_ids = start_img_ids
        self.scene_img_list = scene_img_list

    def __len__(self):
        return len(self.start_img_ids)

    def get_image_num(self):
        return len(self.images)

    def _get_views(self, idx, resolution, rng, num_views):
        start_id = self.start_img_ids[idx]
        scene_id = self.sceneids[start_id]
        all_image_ids = self.scene_img_list[scene_id]
        pos, ordered_video = self.get_seq_from_start_id(
            num_views,
            start_id,
            all_image_ids,
            rng,
            min_interval=self.min_interval,
            max_interval=self.max_interval,
            video_prob=self.video_prob,
            fix_interval_prob=self.fix_interval_prob,
            block_shuffle=self.block_shuffle,
        )
        image_idxs = np.array(all_image_ids)[pos]

        views = []
        for view_idx in image_idxs:
            scene_id = self.sceneids[view_idx]
            scene_dir = osp.join(self.ROOT, self.scenes[scene_id], "dense")

            rgb_path = self.images[view_idx]
            basename = rgb_path[:-4]

            rgb_image = imread_cv2(
                osp.join(scene_dir, "rgb", rgb_path), cv2.IMREAD_COLOR
            )
            depthmap = np.load(osp.join(scene_dir, "depth", basename + ".npy")).astype(
                np.float32
            )
            depthmap[~np.isfinite(depthmap)] = 0  # invalid
            cam_file = np.load(osp.join(scene_dir, "cam", basename + ".npz"))
            sky_mask = (
                cv2.imread(
                    osp.join(scene_dir, "sky_mask", rgb_path), cv2.IMREAD_UNCHANGED
                )
                >= 127
            )
            outlier_mask = cv2.imread(
                osp.join(scene_dir, "outlier_mask", rgb_path), cv2.IMREAD_UNCHANGED
            )
            depthmap[sky_mask] = -1.0
            depthmap[outlier_mask >= 127] = 0.0
            depthmap = np.nan_to_num(depthmap, nan=0, posinf=0, neginf=0)
            threshold = (
                np.percentile(depthmap[depthmap > 0], 98)
                if depthmap[depthmap > 0].size > 0
                else 0
            )
            depthmap[depthmap > threshold] = 0.0

            intrinsics = cam_file["intrinsic"].astype(np.float32)
            camera_pose = cam_file["pose"].astype(np.float32)

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=view_idx
            )

            views.append(
                dict(
                    img=rgb_image,
                    depthmap=depthmap.astype(np.float32),
                    camera_pose=camera_pose.astype(np.float32),
                    camera_intrinsics=intrinsics.astype(np.float32),
                    dataset="dl3dv",
                    label=self.scenes[scene_id] + "_" + rgb_path,
                    instance=osp.join(scene_dir, "rgb", rgb_path),
                    is_metric=self.is_metric,
                    is_video=ordered_video,
                    quantile=np.array(0.9, dtype=np.float32),
                    img_mask=True,
                    ray_mask=False,
                    camera_only=False,
                    depth_only=False,
                    single_view=False,
                    reset=False,
                )
            )
        return views


class DL3DV_ScreenEvent_Multi(EasyDataset):
    def __init__(
        self,
        *args,
        split=None,
        ROOT=None,
        resolution=(518, 392),
        transform=ImgNorm,
        num_views=10,
        seed=0,
        event_in_chans=None,
        min_interval=1,
        max_interval=1,
        sample_mode="sequence",
        sequence_ratio=0.5,
        exclude_keywords=None,
        low_light_prefix="images_low_light",
        **kwargs,
    ):
        self.ROOT = ROOT
        self.split = split
        self.num_views = int(num_views)
        self.seed = seed
        self.event_in_chans = event_in_chans
        self.min_interval = int(min_interval)
        self.max_interval = int(max_interval)
        self.sample_mode = str(sample_mode).lower()
        self.sequence_ratio = float(sequence_ratio)
        self.exclude_keywords = [k.lower() for k in exclude_keywords] if exclude_keywords else []
        if self.sample_mode not in ("sequence", "random", "mixed"):
            raise ValueError(f"Invalid sample_mode={self.sample_mode}, expected one of ['sequence','random','mixed']")
        if not (0.0 <= self.sequence_ratio <= 1.0):
            raise ValueError(f"Invalid sequence_ratio={self.sequence_ratio}, expected in [0, 1]")
        self.is_metric = False
        self._set_resolutions(resolution)

        self.seq_transform = transform is SeqColorJitter
        self.transform = transform

        # Low-light curriculum state (mutated by set_dark_schedule).
        # Naming convention: ``{prefix}_{level}/`` e.g. images_low_light_1/,
        # images_low_light_2/, ... where higher level = darker.
        # Legacy single directory ``{prefix}/`` (no suffix) is treated as
        # level 1 when no numbered directories exist.
        self._low_light_prefix = str(low_light_prefix)
        self._use_low_light = False
        self._dark_level_max = 0

        if self.ROOT is None:
            raise ValueError("ROOT must be provided")
        self._seqs = self._index_sequences()
        if len(self._seqs) == 0:
            raise RuntimeError(f"No valid sequences found under {self.ROOT}")

        self._pose_cache = {}
        # Per-scene cache of available dark levels: {seq_name: [1, 2, 3, ...]}
        self._dark_dirs_cache = {}

    def set_dark_schedule(self, use_low_light: bool, dark_level_max: int = 0):
        """Called by CurriculumMixDataset.set_epoch() to control low-light curriculum.

        Args:
            use_low_light: If True, student reads from low-light directories
                instead of ``images/`` (teacher always gets clean via
                ``teacher_img``).
            dark_level_max: Maximum dark level allowed this epoch. The dataset
                randomly samples a level from 1..dark_level_max for each
                scene (only among levels whose directories exist on disk).
        """
        self._use_low_light = bool(use_low_light)
        self._dark_level_max = int(dark_level_max)

    def _get_available_dark_levels(self, seq_dir):
        """Return sorted list of dark levels whose directories exist for this scene.

        Probes ``{prefix}_1/``, ``{prefix}_2/``, ... up to level 9, plus the
        legacy un-suffixed ``{prefix}/`` (mapped to level 1).
        """
        cached = self._dark_dirs_cache.get(seq_dir)
        if cached is not None:
            return cached

        levels = []
        # Numbered directories: images_low_light_1, _2, _3, ...
        for lvl in range(1, 10):
            d = osp.join(seq_dir, f"{self._low_light_prefix}_{lvl}")
            if osp.isdir(d):
                levels.append(lvl)
        # Legacy single directory (no suffix) → treat as level 1.
        if not levels:
            d = osp.join(seq_dir, self._low_light_prefix)
            if osp.isdir(d):
                levels.append(1)

        self._dark_dirs_cache[seq_dir] = levels
        return levels

    def _set_resolutions(self, resolutions):
        if not isinstance(resolutions, list):
            resolutions = [resolutions]
        self._resolutions = []
        for resolution in resolutions:
            if isinstance(resolution, int):
                width = height = resolution
            else:
                width, height = resolution
            self._resolutions.append((int(width), int(height)))

    def _cache_path(self):
        key = json.dumps({
            "root": self.ROOT,
            "exclude": self.exclude_keywords,
        }, sort_keys=True)
        h = hashlib.md5(key.encode()).hexdigest()[:12]
        return osp.join(self.ROOT, f".seq_index_cache_{h}.json")

    def _index_sequences(self):
        cache_file = self._cache_path()
        if osp.isfile(cache_file):
            try:
                with open(cache_file, "r") as f:
                    cached = json.load(f)
                seqs = [(e["name"], e["img_dir"], e["evt_dir"], e["stems"]) for e in cached]
                print(f"[DL3DV_ScreenEvent_Multi] Loaded {len(seqs)} sequences from cache {cache_file}")
                return seqs
            except Exception as exc:
                print(f"[DL3DV_ScreenEvent_Multi] Cache read failed ({exc}), re-scanning...")

        t0 = time.time()
        seqs = []
        n_excluded = 0
        for name in sorted(os.listdir(self.ROOT)):
            seq_dir = osp.join(self.ROOT, name)
            if not osp.isdir(seq_dir):
                continue
            if self.exclude_keywords and any(kw in name.lower() for kw in self.exclude_keywords):
                n_excluded += 1
                continue
            img_dir = osp.join(seq_dir, "images")
            evt_dir = osp.join(seq_dir, "events")
            if not (osp.isdir(img_dir) and osp.isdir(evt_dir)):
                continue
            img_stems = {f[:-4] for f in os.listdir(img_dir) if f.endswith(".png")}
            evt_stems = {f[:-3] for f in os.listdir(evt_dir) if f.endswith(".pt")}
            stems = sorted(img_stems & evt_stems)
            if len(stems) < 4:
                continue
            seqs.append((name, img_dir, evt_dir, stems))

        elapsed = time.time() - t0
        if n_excluded > 0:
            print(f"[DL3DV_ScreenEvent_Multi] Excluded {n_excluded} scenes "
                  f"matching keywords {self.exclude_keywords}, kept {len(seqs)} scenes")
        print(f"[DL3DV_ScreenEvent_Multi] Indexed {len(seqs)} sequences in {elapsed:.1f}s from {self.ROOT}")

        try:
            to_save = [{"name": s[0], "img_dir": s[1], "evt_dir": s[2], "stems": s[3]} for s in seqs]
            with open(cache_file, "w") as f:
                json.dump(to_save, f)
            print(f"[DL3DV_ScreenEvent_Multi] Saved index cache to {cache_file}")
        except Exception as exc:
            print(f"[DL3DV_ScreenEvent_Multi] Warning: failed to save cache ({exc})")

        return seqs

    def __len__(self):
        return len(self._seqs)

    def _get_pose_entry(self, seq_name, seq_dir, num_stems):
        """Lazy-load and validate per-scene GT pose + intrinsics.

        Returns a dict ``{poses, K_orig, orig_w, orig_h}`` on success or
        ``None`` if transforms.json is missing, malformed, or its frame count
        does not match ``num_stems`` (in which case we cannot safely align
        pose index to stems index and fall back to teacher distillation).
        """
        cached = self._pose_cache.get(seq_name, "missing")
        if cached != "missing":
            return cached

        entry = None
        tfm_path = osp.join(seq_dir, "transforms.json")
        if osp.isfile(tfm_path):
            try:
                with open(tfm_path, "r") as f:
                    data = json.load(f)
                frames = data.get("frames", []) or []
                if frames:
                    # Align with ``stems`` which is sorted by filename.
                    # Pipeline convention: colmap_im_id monotonically maps to
                    # renamed files in the same order.
                    try:
                        frames = sorted(
                            frames, key=lambda fr: int(fr.get("colmap_im_id", 0))
                        )
                    except Exception:
                        frames = sorted(
                            frames, key=lambda fr: str(fr.get("file_path", ""))
                        )
                    poses = []
                    for fr in frames:
                        tm = fr.get("transform_matrix")
                        if tm is None:
                            poses = []
                            break
                        tm = np.asarray(tm, dtype=np.float32)
                        if tm.shape != (4, 4):
                            poses = []
                            break
                        poses.append(tm)
                    fl_x = float(data.get("fl_x", 0.0))
                    fl_y = float(data.get("fl_y", 0.0))
                    cx = float(data.get("cx", 0.0))
                    cy = float(data.get("cy", 0.0))
                    orig_w = float(data.get("w", 0.0))
                    orig_h = float(data.get("h", 0.0))
                    if (
                        poses
                        and fl_x > 0 and fl_y > 0
                        and orig_w > 0 and orig_h > 0
                        and len(poses) == int(num_stems)
                    ):
                        K_orig = np.array(
                            [
                                [fl_x, 0.0, cx],
                                [0.0, fl_y, cy],
                                [0.0, 0.0, 1.0],
                            ],
                            dtype=np.float32,
                        )
                        entry = dict(
                            poses=poses,
                            K_orig=K_orig,
                            orig_w=orig_w,
                            orig_h=orig_h,
                        )
            except Exception:
                entry = None

        self._pose_cache[seq_name] = entry
        return entry

    @staticmethod
    def _rescale_intrinsics(K_orig, orig_w, orig_h, new_w, new_h):
        sx = float(new_w) / float(orig_w)
        sy = float(new_h) / float(orig_h)
        K = K_orig.copy()
        K[0, 0] *= sx
        K[1, 1] *= sy
        K[0, 2] *= sx
        K[1, 2] *= sy
        return K

    def _sample_sequence_ids(self, rng, stems_len, num_views):
        interval = self.min_interval if self.min_interval == self.max_interval else int(
            rng.integers(self.min_interval, self.max_interval + 1)
        )
        max_start = stems_len - 1 - (num_views - 1) * interval
        if max_start < 0:
            interval = 1
            max_start = max(0, stems_len - num_views)
        start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
        return [start + i * interval for i in range(num_views)]

    def _sample_random_ids(self, rng, stems_len, num_views):
        replace = stems_len < num_views
        frame_ids = rng.choice(stems_len, size=num_views, replace=replace).tolist()
        return sorted(int(fid) for fid in frame_ids)

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            idx, res_idx, view_count = idx
            resolution = self._resolutions[int(res_idx)]
            num_views = int(view_count)
        else:
            resolution = self._resolutions[0]
            num_views = self.num_views

        for _ in range(10):
            try:
                return self._getitem_inner(idx, resolution, num_views)
            except (FileNotFoundError, OSError):
                idx = np.random.randint(len(self))
        return self._getitem_inner(idx, resolution, num_views)

    def _getitem_inner(self, idx, resolution, num_views):
        rng = np.random.default_rng()
        seq_name, img_dir, evt_dir, stems = self._seqs[int(idx) % len(self._seqs)]

        if self.sample_mode == "mixed":
            use_sequence = bool(rng.random() < self.sequence_ratio)
        else:
            use_sequence = self.sample_mode == "sequence"

        if use_sequence:
            frame_ids = self._sample_sequence_ids(rng, len(stems), num_views)
        else:
            frame_ids = self._sample_random_ids(rng, len(stems), num_views)

        tfm = SeqColorJitter() if self.seq_transform else self.transform
        W, H = resolution

        # Scene-level lazy loads: GT pose metadata + DA3 disparity directory.
        seq_dir = osp.dirname(img_dir)
        pose_entry = self._get_pose_entry(seq_name, seq_dir, num_stems=len(stems))
        da3_dir = osp.join(seq_dir, "depth_da3")
        has_da3_dir = osp.isdir(da3_dir)

        # Decide whether to feed the student low-light images for this scene.
        # Available levels are probed lazily and cached per scene directory.
        available_levels = (
            self._get_available_dark_levels(seq_dir) if self._use_low_light else []
        )
        # Filter to levels <= dark_level_max for this epoch.
        eligible_levels = [l for l in available_levels if l <= self._dark_level_max]
        serve_low_light = len(eligible_levels) > 0

        # Pick one dark level for the entire clip (all views share the same
        # level so the brightness shift is consistent within a sequence).
        chosen_level = rng.choice(eligible_levels) if serve_low_light else 0
        if chosen_level > 0:
            # Numbered directory first, fall back to legacy un-suffixed.
            ll_dir_name = f"{self._low_light_prefix}_{chosen_level}"
            ll_dir_candidate = osp.join(seq_dir, ll_dir_name)
            if not osp.isdir(ll_dir_candidate):
                ll_dir_candidate = osp.join(seq_dir, self._low_light_prefix)
        else:
            ll_dir_candidate = None

        views = []
        for i, fid in enumerate(frame_ids):
            stem = stems[int(fid)]
            clean_img_path = osp.join(img_dir, stem + ".png")
            evt_path = osp.join(evt_dir, stem + ".pt")

            clean_img = PIL.Image.open(clean_img_path).convert("RGB")
            clean_img = clean_img.resize((W, H), PIL.Image.BICUBIC)
            clean_img_t = tfm(clean_img)

            if serve_low_light and ll_dir_candidate is not None:
                ll_path = osp.join(ll_dir_candidate, stem + ".png")
                if osp.isfile(ll_path):
                    student_img = PIL.Image.open(ll_path).convert("RGB")
                    student_img = student_img.resize((W, H), PIL.Image.BICUBIC)
                    img_t = tfm(student_img)
                else:
                    img_t = clean_img_t
            else:
                img_t = clean_img_t

            evt = torch.load(evt_path, map_location="cpu")
            if not isinstance(evt, torch.Tensor):
                evt = torch.as_tensor(evt)
            evt = evt.float()
            if evt.ndim == 2:
                evt = evt.unsqueeze(0)
            if self.event_in_chans is not None and int(evt.shape[0]) != int(
                self.event_in_chans
            ):
                raise ValueError(
                    f"Bad event channels for {evt_path}: got {evt.shape[0]} expected {self.event_in_chans}"
                )
            if tuple(evt.shape[-2:]) != (H, W):
                evt = F.interpolate(
                    evt.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False
                ).squeeze(0)

            evt = torch.nan_to_num(evt, nan=0.0, posinf=0.0, neginf=0.0)
            evt = evt.clamp(-50.0, 50.0)
            evt_raw = evt.clone()
            evt_std = evt.std()
            if evt_std < 1e-6:
                evt = torch.zeros_like(evt)
            else:
                evt_mean = evt.mean()
                evt = (evt - evt_mean) / evt_std

            valid_mask = np.ones((H, W), dtype=bool)

            view = dict(
                img=img_t,
                teacher_img=clean_img_t,
                event_voxel=evt,
                event_voxel_raw=evt_raw,
                valid_mask=valid_mask,
                dataset="dl3dv",
                label=f"{seq_name}/{stem}",
                instance=clean_img_path,
                is_metric=self.is_metric,
                is_video=bool(use_sequence),
                reset=False,
            )

            # --- Optional: GT camera pose + intrinsics from transforms.json.
            # Silently skipped when unavailable so the same dataset class can
            # serve both sim (has COLMAP pose) and real (no pose) datasets.
            if pose_entry is not None:
                fid_int = int(fid)
                if 0 <= fid_int < len(pose_entry["poses"]):
                    view["camera_pose_gt"] = torch.from_numpy(
                        pose_entry["poses"][fid_int].copy()
                    ).float()
                    K_scaled = self._rescale_intrinsics(
                        pose_entry["K_orig"],
                        pose_entry["orig_w"],
                        pose_entry["orig_h"],
                        W,
                        H,
                    )
                    view["camera_intrinsics_gt"] = torch.from_numpy(K_scaled).float()

            # --- Optional: DA3 disparity pseudo-GT.
            if has_da3_dir:
                disp_path = osp.join(da3_dir, stem + ".pt")
                if osp.isfile(disp_path):
                    try:
                        disp = torch.load(disp_path, map_location="cpu")
                        if not torch.is_tensor(disp):
                            disp = torch.as_tensor(disp)
                        disp = disp.float()
                        # Accept [H, W] / [1, H, W] / [H, W, 1] shapes.
                        if disp.ndim == 3:
                            if disp.shape[0] == 1:
                                disp = disp.squeeze(0)
                            elif disp.shape[-1] == 1:
                                disp = disp.squeeze(-1)
                        if disp.ndim != 2:
                            raise ValueError(
                                f"Unexpected da3 disp shape {tuple(disp.shape)}"
                            )
                        valid = torch.isfinite(disp) & (disp > 0)
                        if tuple(disp.shape) != (H, W):
                            disp = F.interpolate(
                                disp.unsqueeze(0).unsqueeze(0),
                                size=(H, W),
                                mode="bilinear",
                                align_corners=False,
                            ).squeeze(0).squeeze(0)
                            valid = F.interpolate(
                                valid.float().unsqueeze(0).unsqueeze(0),
                                size=(H, W),
                                mode="nearest",
                            ).squeeze(0).squeeze(0) > 0.5
                        disp = torch.nan_to_num(
                            disp, nan=0.0, posinf=0.0, neginf=0.0
                        )
                        view["da3_disparity"] = disp
                        view["da3_valid_mask"] = valid
                    except Exception:
                        # Malformed / missing file: skip silently; loss will
                        # fall back to teacher distillation for this view.
                        pass

            views.append(view)
        return views
