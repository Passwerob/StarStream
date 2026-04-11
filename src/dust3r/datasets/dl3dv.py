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

        if self.ROOT is None:
            raise ValueError("ROOT must be provided")
        self._seqs = self._index_sequences()
        if len(self._seqs) == 0:
            raise RuntimeError(f"No valid sequences found under {self.ROOT}")

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

        views = []
        for i, fid in enumerate(frame_ids):
            stem = stems[int(fid)]
            img_path = osp.join(img_dir, stem + ".png")
            evt_path = osp.join(evt_dir, stem + ".pt")

            img = PIL.Image.open(img_path).convert("RGB")
            img = img.resize((W, H), PIL.Image.BICUBIC)
            img_t = tfm(img)

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
            evt_std = evt.std()
            if evt_std < 1e-6:
                evt = torch.zeros_like(evt)
            else:
                evt_mean = evt.mean()
                evt = (evt - evt_mean) / evt_std

            valid_mask = np.ones((H, W), dtype=bool)

            views.append(
                dict(
                    img=img_t,
                    event_voxel=evt,
                    valid_mask=valid_mask,
                    dataset="dl3dv",
                    label=f"{seq_name}/{stem}",
                    instance=img_path,
                    is_metric=self.is_metric,
                    is_video=bool(use_sequence),
                    reset=False,
                )
            )
        return views
