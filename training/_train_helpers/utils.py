"""
Utility functions for rawgen project.

This module contains common utility functions for DDP setup, image processing,
metrics calculation, visualization, and model management.
"""

import os
import random
from pathlib import Path
from typing import Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb

# DDP related imports
import torch.distributed as dist

# torchmetrics import
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


# -------------------------
# DDP Utilities
# -------------------------

def setup_ddp():
    """Initialize DDP process group.

    Supports both torchrun (LOCAL_RANK/RANK) and srun (SLURM_LOCALID/SLURM_PROCID).
    """
    # Map SLURM env vars to PyTorch expected vars if not already set (srun without torchrun)
    if "LOCAL_RANK" not in os.environ and "SLURM_LOCALID" in os.environ:
        os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
    if "RANK" not in os.environ and "SLURM_PROCID" in os.environ:
        os.environ["RANK"] = os.environ["SLURM_PROCID"]
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def cleanup_ddp():
    """Clean up DDP process group."""
    dist.destroy_process_group()


def is_main_process():
    """Check if current process is main process (rank 0)."""
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True  # If DDP not initialized, assume main process


# -------------------------
# Seed Utilities
# -------------------------

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------
# Image Processing Utilities
# -------------------------

def ensure_multiple_of(x: int, base=8) -> int:
    """Ensure x is a multiple of base."""
    return x - (x % base)


def srgb01_to_m11(t: torch.Tensor) -> torch.Tensor:
    """Convert sRGB from [0,1] to [-1,1] range."""
    return t * 2.0 - 1.0


def m11_to_01(t: torch.Tensor) -> torch.Tensor:
    """Convert from [-1,1] to [0,1] range."""
    return (t.clamp(-1, 1) + 1.0) / 2.0


# -------------------------
# Metrics Utilities
# -------------------------

def compute_metrics_unit01(gt01: np.ndarray, pred01: np.ndarray) -> Tuple[float, float]:
    """Compute PSNR and SSIM metrics for images in [0,1] range using torchmetrics."""
    # Convert numpy arrays to torch tensors (HWC -> CHW)
    # Ensure tensors are created on CPU, as numpy arrays reside on CPU.
    gt_tensor = torch.from_numpy(gt01).permute(2, 0, 1).unsqueeze(0).to("cpu")
    pred_tensor = torch.from_numpy(pred01).permute(2, 0, 1).unsqueeze(0).to("cpu")
    
    # Directly create metric instances
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to("cpu")
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to("cpu")
    
    # Compute metrics.
    psnr_val = psnr_metric(pred_tensor, gt_tensor)
    ssim_val = ssim_metric(pred_tensor, gt_tensor)
    
    return float(psnr_val.item()), float(ssim_val.item())


# -------------------------
# Visualization Utilities
# -------------------------

def visualize_results(x_target, x_pred, filename, epoch, save_dir, use_wandb=False, log_prefix: str = "val"):
    """Visualize training/validation results and save as image."""
    x_target_01, x_pred_01 = m11_to_01(x_target), m11_to_01(x_pred)
    x_target_hwc = x_target_01.float().permute(1, 2, 0).detach().cpu().numpy()
    x_pred_hwc = x_pred_01.detach().float().permute(1, 2, 0).cpu().numpy()

    # Use torchmetrics for per-image metrics on the plot title
    psnr, ssim_val = compute_metrics_unit01(x_target_hwc, x_pred_hwc)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(x_target_hwc); axes[0].set_title('Target XYZ', fontsize=12); axes[0].axis('off')
    axes[1].imshow(x_pred_hwc); axes[1].set_title(f'Predicted XYZ\nPSNR: {psnr:.2f}, SSIM: {ssim_val:.4f}', fontsize=12); axes[1].axis('off')
    plt.suptitle(f'Epoch {epoch} - {filename}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    stem = Path(filename).stem
    save_filename = f"epoch_{epoch}_{stem}_psnr_{psnr:.2f}_ssim_{ssim_val:.4f}.png"
    save_path = save_dir / save_filename
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print message (works in both DDP and single process environments)
    try:
        if is_main_process():
            print(f"Visualization saved: {save_path}")
    except ValueError:
        # DDP not initialized, just print directly
        print(f"Visualization saved: {save_path}")
    
    # Wandb logging if enabled: log the saved combined figure
    if use_wandb and wandb.run is not None:
        try:
            if is_main_process():
                wandb.log({
                    f"{log_prefix}/viz": wandb.Image(str(save_path), caption=f"Epoch {epoch} - {filename} | PSNR: {psnr:.2f}, SSIM: {ssim_val:.4f}")
                })
        except ValueError:
            # DDP not initialized, just log directly
            wandb.log({
                f"{log_prefix}/viz": wandb.Image(str(save_path), caption=f"Epoch {epoch} - {filename} | PSNR: {psnr:.2f}, SSIM: {ssim_val:.4f}")
            })
    
    return psnr, ssim_val


def visualize_diffusion_results(condition_srgb_input, anchor_srgb_gt, anchor_srgb_pred, filename, epoch, save_dir):
    """
    Visualize diffusion validation results and save as image.
    Args:
        anchor_srgb_gt: Ground truth sRGB image tensor in [-1, 1] (C, H, W)
        anchor_srgb_pred: Predicted sRGB image tensor in [-1, 1] (C, H, W)
        filename: Basename for saving
        epoch: Current epoch number
        save_dir: Directory to save visualization
    Returns:
        psnr: PSNR value
        ssim_val: SSIM value
        save_path: Path to saved image file
    """
    # Convert to [0, 1] and HWC for metric computation and plotting
    anchor_gt_01 = m11_to_01(anchor_srgb_gt).float().permute(1, 2, 0).detach().cpu().numpy()
    anchor_pred_01 = m11_to_01(anchor_srgb_pred).float().permute(1, 2, 0).detach().cpu().numpy()
    psnr, ssim_val = compute_metrics_unit01(anchor_gt_01, anchor_pred_01)

    import matplotlib.pyplot as plt  # local import to keep module lightweight if headless

    cond_01 = m11_to_01(condition_srgb_input).float().permute(1, 2, 0).detach().cpu().numpy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(cond_01)
    axes[0].set_title('Input Variant sRGB', fontsize=12)
    axes[0].axis('off')
    axes[1].imshow(anchor_gt_01)
    axes[1].set_title('GT Anchor sRGB', fontsize=12)
    axes[1].axis('off')
    axes[2].imshow(anchor_pred_01)
    axes[2].set_title(f'Predicted Anchor sRGB\nPSNR: {psnr:.2f}, SSIM: {ssim_val:.4f}', fontsize=12)
    axes[2].axis('off')
    if epoch is not None:
        plt.suptitle(f'Epoch {epoch} - {filename}', fontsize=14, fontweight='bold')
    else:
        plt.suptitle(f'{filename}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    stem = Path(filename).stem
    if epoch is not None:
        save_filename = f"epoch_{epoch}_{stem}_psnr_{psnr:.2f}_ssim_{ssim_val:.4f}.png"
    else:
        save_filename = f"{stem}_psnr_{psnr:.2f}_ssim_{ssim_val:.4f}.png"
    save_path = save_dir / save_filename
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    try:
        if is_main_process():
            print(f"Visualization saved: {save_path}")
    except:
        pass
    
    return psnr, ssim_val, save_path


# -------------------------
# Model Utilities
# -------------------------

def save_decoder(vae_ddp, path: Path):
    """Save decoder state dict from DDP wrapped VAE."""
    path.parent.mkdir(parents=True, exist_ok=True)
    vae_original = vae_ddp.module
    sd = {"decoder": vae_original.decoder.state_dict(), "config": vae_original.config}
    torch.save(sd, path)
    print(f"Saved: {str(path)}")


# -------------------------
# Learning Rate Scheduler Utilities
# -------------------------

def get_scheduler(optimizer, cfg, total_steps_per_epoch):
    """
    Create learning rate scheduler with various types.
    
    Supported scheduler types:
    - "cosine": Cosine decay with linear warmup (step-based)
    - "linear": Linear decay with linear warmup (step-based)
    - "none": No scheduler

    Args:
        optimizer: PyTorch optimizer
        cfg: Configuration object with scheduler parameters
        total_steps_per_epoch: Number of training steps per epoch

    Returns:
        tuple: (scheduler, step_type) where step_type is one of:
            - "step": Update per training step
            - None: No scheduler
    """
    from torch.optim.lr_scheduler import LambdaLR
    import math

    if not cfg.use_scheduler or cfg.scheduler_type == "none":
        return None, None

    # LambdaLR-based schedulers with warmup (step-based)
    warmup_steps = cfg.warmup_steps  # Use fixed warmup steps directly
    total_steps = cfg.max_epochs * total_steps_per_epoch

    def lr_lambda(current_step):
        # Linear warmup
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))

        # After warmup
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))

        if cfg.scheduler_type == "cosine":
            # Cosine decay
            min_lr_ratio = cfg.min_lr / cfg.lr
            return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
        elif cfg.scheduler_type == "linear":
            # Linear decay
            return max(0.0, 1.0 - progress)
        else:
            return 1.0

    scheduler = LambdaLR(optimizer, lr_lambda)
    return scheduler, "step"
