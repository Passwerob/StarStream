"""Validation metrics for StarStream training: PSNR, SSIM, LPIPS."""

import torch
import torch.nn.functional as F
from dust3r.losses import SSIM


def compute_psnr(pred: torch.Tensor, gt: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """Compute PSNR between predicted and ground truth images.

    Args:
        pred: (B, C, H, W) or (B, H, W, C) in [0, max_val]
        gt:   same shape as pred
        max_val: maximum pixel value
    Returns:
        Scalar PSNR in dB.
    """
    mse = F.mse_loss(pred.float(), gt.float())
    if mse == 0:
        return torch.tensor(float('inf'), device=pred.device)
    return 20.0 * torch.log10(torch.tensor(max_val, device=pred.device)) - 10.0 * torch.log10(mse)


class SSIMMetric:
    """Wraps the existing SSIM loss class to return SSIM value (higher=better)."""

    def __init__(self, device='cuda'):
        self._ssim = SSIM().to(device).eval()

    @torch.no_grad()
    def __call__(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """Compute SSIM between pred and gt.

        Args:
            pred: (B, C, H, W) in [0, 1]
            gt:   (B, C, H, W) in [0, 1]
        Returns:
            Scalar SSIM value (higher is better).
        """
        # SSIM class returns (1 - SSIM) / 2, so SSIM = 1 - 2 * loss
        loss = self._ssim(pred.float(), gt.float())
        return 1.0 - 2.0 * loss.mean()


class LPIPSMetric:
    """Batched, torch-native LPIPS wrapper for use during training validation."""

    def __init__(self, backbone='alex', device='cuda'):
        import lpips
        self._model = lpips.LPIPS(net=backbone, verbose=False).to(device).eval()
        for p in self._model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def __call__(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """Compute LPIPS between pred and gt.

        Args:
            pred: (B, C, H, W) in [0, 1]
            gt:   (B, C, H, W) in [0, 1]
        Returns:
            Scalar LPIPS score (lower is better).
        """
        # LPIPS expects [-1, 1]
        return self._model(pred.float() * 2.0 - 1.0, gt.float() * 2.0 - 1.0).mean()
