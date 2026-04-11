# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# modified from DUSt3R

import numpy as np
from dust3r.datasets.base.batched_sampler import (
    BatchedRandomSampler,
    CustomRandomSampler,
)
import torch


class EasyDataset:
    """a dataset that you can easily resize and combine.
    Examples:
    ---------
        2 * dataset ==> duplicate each element 2x

        10 @ dataset ==> set the size to 10 (random sampling, duplicates if necessary)

        dataset1 + dataset2 ==> concatenate datasets
    """

    def __add__(self, other):
        return CatDataset([self, other])

    def __rmul__(self, factor):
        return MulDataset(factor, self)

    def __rmatmul__(self, factor):
        return ResizedDataset(factor, self)

    def set_epoch(self, epoch):
        pass  # nothing to do by default

    def make_sampler(
        self,
        batch_size,
        shuffle=True,
        drop_last=True,
        world_size=1,
        rank=0,
        fixed_length=False,
        min_view_size=None,
        max_view_size=None,
    ):
        if not (shuffle):
            raise NotImplementedError()  # cannot deal yet
        num_of_aspect_ratios = len(self._resolutions)
        num_of_views = self.num_views

        if fixed_length:
            min_num_views = num_of_views
            max_num_views = num_of_views
        else:
            min_num_views = 4 if min_view_size is None else int(min_view_size)
            max_num_views = num_of_views if max_view_size is None else int(max_view_size)
            min_num_views = max(1, min_num_views)
            max_num_views = min(num_of_views, max_num_views)
            if min_num_views > max_num_views:
                raise ValueError(
                    f"Invalid view range: min_view_size={min_num_views}, max_view_size={max_num_views}, num_views={num_of_views}"
                )

        sampler = CustomRandomSampler(
            self,
            batch_size,
            num_of_aspect_ratios,
            min_num_views,
            max_num_views,
            world_size,
            warmup=1,
            drop_last=drop_last,
        )
        return BatchedRandomSampler(sampler, batch_size, drop_last)


class MulDataset(EasyDataset):
    """Artifically augmenting the size of a dataset."""

    multiplicator: int

    def __init__(self, multiplicator, dataset):
        assert isinstance(multiplicator, int) and multiplicator > 0
        self.multiplicator = multiplicator
        self.dataset = dataset

    def __len__(self):
        return self.multiplicator * len(self.dataset)

    def __repr__(self):
        return f"{self.multiplicator}*{repr(self.dataset)}"

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            idx, other, another = idx
            return self.dataset[idx // self.multiplicator, other, another]
        else:
            return self.dataset[idx // self.multiplicator]

    @property
    def _resolutions(self):
        return self.dataset._resolutions

    @property
    def num_views(self):
        return self.dataset.num_views


class ResizedDataset(EasyDataset):
    """Artifically changing the size of a dataset."""

    new_size: int

    def __init__(self, new_size, dataset):
        assert isinstance(new_size, int) and new_size > 0
        self.new_size = new_size
        self.dataset = dataset

    def __len__(self):
        return self.new_size

    def __repr__(self):
        size_str = str(self.new_size)
        for i in range((len(size_str) - 1) // 3):
            sep = -4 * i - 3
            size_str = size_str[:sep] + "_" + size_str[sep:]
        return f"{size_str} @ {repr(self.dataset)}"

    def set_epoch(self, epoch):
        # this random shuffle only depends on the epoch
        rng = np.random.default_rng(seed=epoch + 777)

        # shuffle all indices
        perm = rng.permutation(len(self.dataset))

        # rotary extension until target size is met
        shuffled_idxs = np.concatenate(
            [perm] * (1 + (len(self) - 1) // len(self.dataset))
        )
        self._idxs_mapping = shuffled_idxs[: self.new_size]

        assert len(self._idxs_mapping) == self.new_size

    def __getitem__(self, idx):
        assert hasattr(
            self, "_idxs_mapping"
        ), "You need to call dataset.set_epoch() to use ResizedDataset.__getitem__()"
        if isinstance(idx, tuple):
            idx, other, another = idx
            return self.dataset[self._idxs_mapping[idx], other, another]
        else:
            return self.dataset[self._idxs_mapping[idx]]

    @property
    def _resolutions(self):
        return self.dataset._resolutions

    @property
    def num_views(self):
        return self.dataset.num_views


class CatDataset(EasyDataset):
    """Concatenation of several datasets"""

    def __init__(self, datasets):
        for dataset in datasets:
            assert isinstance(dataset, EasyDataset)
        self.datasets = datasets
        self._cum_sizes = np.cumsum([len(dataset) for dataset in datasets])

    def __len__(self):
        return self._cum_sizes[-1]

    def __repr__(self):
        # remove uselessly long transform
        return " + ".join(
            repr(dataset).replace(
                ",transform=Compose( ToTensor() Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))",
                "",
            )
            for dataset in self.datasets
        )

    def set_epoch(self, epoch):
        for dataset in self.datasets:
            dataset.set_epoch(epoch)

    def __getitem__(self, idx):
        other = None
        if isinstance(idx, tuple):
            idx, other, another = idx

        if not (0 <= idx < len(self)):
            raise IndexError()

        db_idx = np.searchsorted(self._cum_sizes, idx, "right")
        dataset = self.datasets[db_idx]
        new_idx = idx - (self._cum_sizes[db_idx - 1] if db_idx > 0 else 0)

        if other is not None and another is not None:
            new_idx = (new_idx, other, another)
        return dataset[new_idx]

    @property
    def _resolutions(self):
        resolutions = self.datasets[0]._resolutions
        for dataset in self.datasets[1:]:
            assert tuple(dataset._resolutions) == tuple(resolutions)
        return resolutions

    @property
    def num_views(self):
        num_views = self.datasets[0].num_views
        for dataset in self.datasets[1:]:
            assert dataset.num_views == num_views
        return num_views


class CurriculumMixDataset(EasyDataset):
    """Mix two datasets with an epoch-dependent ratio (curriculum learning).

    During warmup epochs, only dataset_a is used.  After warmup, dataset_b's
    share increases linearly until it reaches ``final_ratio_b``, while
    dataset_a never drops below ``min_ratio_a``.

    Both datasets must share the same resolutions and num_views.

    Usage (in yaml, via eval):
        CurriculumMixDataset(
            total_size=10000,
            dataset_a=DL3DV_ScreenEvent_Multi(...),   # simulated
            dataset_b=DL3DV_ScreenEvent_Multi(...),   # real
            total_epochs=10,
            warmup_epochs=2,
            final_ratio_b=0.8,
            min_ratio_a=0.2,
        )
    """

    def __init__(
        self,
        total_size,
        dataset_a,
        dataset_b,
        total_epochs=10,
        warmup_epochs=2,
        final_ratio_b=0.8,
        min_ratio_a=0.2,
        initial_ratio_b=0.0,
    ):
        assert isinstance(total_size, int) and total_size > 0
        assert 0.0 <= final_ratio_b <= 1.0
        assert 0.0 <= min_ratio_a <= 1.0
        assert 0.0 <= initial_ratio_b <= final_ratio_b
        self.total_size = total_size
        self.dataset_a = dataset_a
        self.dataset_b = dataset_b
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.final_ratio_b = min(final_ratio_b, 1.0 - min_ratio_a)
        self.min_ratio_a = min_ratio_a
        self.initial_ratio_b = initial_ratio_b

    def __len__(self):
        return self.total_size

    def __repr__(self):
        return (
            f"CurriculumMix(total={self.total_size}, "
            f"warmup={self.warmup_epochs}, final_b={self.final_ratio_b:.2f}, "
            f"a={repr(self.dataset_a)}, b={repr(self.dataset_b)})"
        )

    def _get_ratio_b(self, epoch):
        if epoch < self.warmup_epochs:
            if self.initial_ratio_b > 0 and self.warmup_epochs > 0:
                warmup_progress = epoch / self.warmup_epochs
                return self.initial_ratio_b * warmup_progress
            return self.initial_ratio_b
        ramp_length = max(self.total_epochs - self.warmup_epochs - 1, 1)
        progress = min((epoch - self.warmup_epochs) / ramp_length, 1.0)
        return self.initial_ratio_b + progress * (self.final_ratio_b - self.initial_ratio_b)

    def set_epoch(self, epoch):
        ratio_b = self._get_ratio_b(epoch)
        n_b = int(self.total_size * ratio_b)
        n_a = self.total_size - n_b

        rng = np.random.default_rng(seed=epoch + 777)

        perm_a = rng.permutation(len(self.dataset_a))
        idxs_a = np.concatenate(
            [perm_a] * (1 + (n_a - 1) // max(len(self.dataset_a), 1))
        )[:n_a]

        if n_b > 0:
            perm_b = rng.permutation(len(self.dataset_b))
            idxs_b = np.concatenate(
                [perm_b] * (1 + (n_b - 1) // max(len(self.dataset_b), 1))
            )[:n_b]
        else:
            idxs_b = np.array([], dtype=np.intp)

        sources = np.concatenate([
            np.zeros(n_a, dtype=np.intp),
            np.ones(n_b, dtype=np.intp),
        ])
        local_idxs = np.concatenate([idxs_a, idxs_b])

        shuffle_order = rng.permutation(self.total_size)
        self._sources = sources[shuffle_order]
        self._local_idxs = local_idxs[shuffle_order]

        print(f"[CurriculumMix] epoch={epoch}, "
              f"ratio_b={ratio_b:.2%}, n_a={n_a}, n_b={n_b}")

    def __getitem__(self, idx):
        assert hasattr(self, "_sources"), (
            "Call set_epoch() before __getitem__()"
        )
        if isinstance(idx, tuple):
            idx, other, another = idx
            ds = self.dataset_a if self._sources[idx] == 0 else self.dataset_b
            return ds[int(self._local_idxs[idx]), other, another]
        ds = self.dataset_a if self._sources[idx] == 0 else self.dataset_b
        return ds[int(self._local_idxs[idx])]

    @property
    def _resolutions(self):
        return self.dataset_a._resolutions

    @property
    def num_views(self):
        return self.dataset_a.num_views
