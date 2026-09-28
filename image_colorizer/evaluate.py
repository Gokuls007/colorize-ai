"""
Evaluation module for the ImageColorization project.
"""

import os
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.color import lab2rgb
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch import nn
from tqdm import tqdm

from image_colorizer.inference import _resolve_device


def _lab_to_rgb_batch(L: torch.Tensor, ab: torch.Tensor) -> np.ndarray:
    L_denorm = (L + 1.0) * 50.0
    ab_denorm = ab * 110.0
    lab = torch.cat([L_denorm, ab_denorm], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_images = []
    for img_lab in lab:
        rgb = lab2rgb(img_lab)
        rgb_images.append(rgb)
    return np.stack(rgb_images, axis=0)


def evaluate_model(
    model: nn.Module,
    val_dataloader: torch.utils.data.DataLoader,
    device: str = "cuda",
) -> Dict[str, float]:
    device = _resolve_device(device)
    model = model.to(device)
    model.net_G.eval()

    psnr_scores = []
    ssim_scores = []
    l1_losses = []
    l1_criterion = nn.L1Loss()

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Evaluating", unit="batch"):
            L = batch["L"].to(device)
            ab_true = batch["ab"].to(device)
            ab_pred = model.net_G(L)
            l1_loss = l1_criterion(ab_pred, ab_true).item()
            l1_losses.append(l1_loss)
            rgb_pred = _lab_to_rgb_batch(L, ab_pred)
            rgb_true = _lab_to_rgb_batch(L, ab_true)
            for i in range(rgb_pred.shape[0]):
                pred_img = np.clip(rgb_pred[i], 0, 1)
                true_img = np.clip(rgb_true[i], 0, 1)
                psnr = peak_signal_noise_ratio(true_img, pred_img, data_range=1.0)
                psnr_scores.append(psnr)
                ssim = structural_similarity(true_img, pred_img, data_range=1.0, channel_axis=2)
                ssim_scores.append(ssim)

    model.net_G.train()
    metrics = {
        "avg_psnr": float(np.mean(psnr_scores)) if psnr_scores else 0.0,
        "avg_ssim": float(np.mean(ssim_scores)) if ssim_scores else 0.0,
        "min_psnr": float(np.min(psnr_scores)) if psnr_scores else 0.0,
        "max_psnr": float(np.max(psnr_scores)) if psnr_scores else 0.0,
        "avg_l1_loss": float(np.mean(l1_losses)) if l1_losses else 0.0,
    }
    return metrics


def print_metrics(metrics: Dict[str, float]) -> None:
    print("\nEvaluation Metrics")
    print("="*40)
    print(f"  Avg PSNR:      {metrics.get('avg_psnr', 0):.2f} dB")
    print(f"  Min PSNR:      {metrics.get('min_psnr', 0):.2f} dB")
    print(f"  Max PSNR:      {metrics.get('max_psnr', 0):.2f} dB")
    print(f"  Avg SSIM:      {metrics.get('avg_ssim', 0):.4f}")
    print(f"  Avg L1 Loss:   {metrics.get('avg_l1_loss', 0):.4f}")
    print("="*40 + "\n")


def save_comparison_images(
    model: nn.Module,
    sample_batch: Dict[str, torch.Tensor],
    output_dir: str,
    device: str = "cuda",
    num_images: int = 5,
    filename: str = "comparison.png",
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    device = _resolve_device(device)
    model = model.to(device)
    model.net_G.eval()
    L = sample_batch["L"].to(device)
    ab_true = sample_batch["ab"].to(device)
    with torch.no_grad():
        ab_pred = model.net_G(L)
    model.net_G.train()
    fake_imgs = _lab_to_rgb_batch(L, ab_pred)
    real_imgs = _lab_to_rgb_batch(L, ab_true)
    num_images = min(num_images, L.shape[0])
    fig, axes = plt.subplots(3, num_images, figsize=(3 * num_images, 9))
    row_labels = ["Grayscale (Input)", "Colorized (Model)", "Ground Truth"]
    for i in range(num_images):
        ax = axes[0, i] if num_images > 1 else axes[0]
        ax.imshow(L[i][0].cpu().numpy(), cmap="gray")
        ax.axis("off")
        if i == 0: ax.set_ylabel(row_labels[0], fontsize=12, fontweight="bold")
        ax = axes[1, i] if num_images > 1 else axes[1]
        ax.imshow(np.clip(fake_imgs[i], 0, 1))
        ax.axis("off")
        if i == 0: ax.set_ylabel(row_labels[1], fontsize=12, fontweight="bold")
        ax = axes[2, i] if num_images > 1 else axes[2]
        ax.imshow(np.clip(real_imgs[i], 0, 1))
        ax.axis("off")
        if i == 0: ax.set_ylabel(row_labels[2], fontsize=12, fontweight="bold")
    plt.suptitle("Image Colorization Results", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    output_path = os.path.join(output_dir, filename)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Comparison image saved to: {output_path}")
    return output_path
