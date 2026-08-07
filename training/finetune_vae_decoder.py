#!/usr/bin/env python3
"""Fine-tune VAE decoder for XYZ image reconstruction."""

import os
import random
import argparse
from pathlib import Path
from typing import Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm

from _train_helpers.dataset import load_manifest_pairs, srgb01_to_m11, ensure_multiple_of
from _train_helpers import (
    setup_ddp, cleanup_ddp, is_main_process, set_seed,
    visualize_results, save_decoder, m11_to_01, get_scheduler
)
from _train_helpers.logging_utils import init_logger
from _train_helpers.vae_model_utils import load_vae_for_training

# torchmetrics import
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.collections import MetricCollection


# -------------------------
# Dataset
# -------------------------

class XYZLatentDataset(Dataset):
    """
    Dataset for loading XYZ latents and GT XYZ images.
    
    Loads pre-encoded XYZ latents from .pt files and corresponding GT XYZ
    images from PNG files. Latents are unscaled (divided by scaling_factor)
    before being returned since they were saved with scaling already applied.
    """
    
    def __init__(
        self,
        manifest_path: Path,
        root_key: str = "root",
        split: str = "train",
        dataset_subsets: Optional[List[str]] = None,
        latent_dir_name: str = "latents",
        gt_xyz_dir_name: str = "anchor-variants-imgs",
        image_size: Optional[int] = None,
        scaling_factor: Optional[float] = None,
        shift_factor: Optional[float] = None,
    ):
        self.split = split
        if self.split not in {"train", "val", "test", "all"}:
            raise ValueError("split must be one of {'train','val','test','all'}")
        
        self.latent_dir_name = latent_dir_name
        self.gt_xyz_dir_name = gt_xyz_dir_name
        self.image_size = image_size
        self.scaling_factor = scaling_factor
        self.shift_factor = shift_factor
        
        if scaling_factor is None:
            raise ValueError("scaling_factor must be provided. Get it from VAE config: vae.config.scaling_factor")
        
        # Load pairs from manifest
        pairs, meta = load_manifest_pairs(manifest_path, dataset_subsets, split=self.split, root_key=root_key)
        self.manifest_root = Path(meta.get("root", "."))
        
        # Build sample list: find matching XYZ latent and GT XYZ image files
        samples = []
        skipped = 0
        
        for srgb_path, xyz_path in pairs:
            basename = xyz_path.stem.replace("_xyz", "")
            dataset_type = xyz_path.parts[-3] if len(xyz_path.parts) >= 3 else "unknown"
            
            # Construct paths
            latent_path = self.manifest_root / dataset_type / latent_dir_name / f"{basename}_anchor_xyz.pt"
            gt_xyz_path = self.manifest_root / dataset_type / gt_xyz_dir_name / f"{basename}_anchor_xyz.png"
            
            # Check if both files exist
            if latent_path.exists() and gt_xyz_path.exists():
                samples.append({
                    "basename": basename,
                    "dataset_type": dataset_type,
                    "latent_path": latent_path,
                    "gt_xyz_path": gt_xyz_path,
                })
            else:
                skipped += 1
        
        if skipped > 0:
            print(f"[WARN] {skipped} samples skipped due to missing files in split '{self.split}'")
        
        if not samples:
            raise ValueError(f"No valid samples found for split '{self.split}'")
        
        self.samples = samples
        
        # Resize target size
        self.target_hw = None
        if image_size:
            img_size = ensure_multiple_of(image_size, 8)
            self.target_hw = (img_size, img_size)  # (H, W)
        
        print(f"Loaded {len(self.samples)} XYZ latent-image pairs for {split} split")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        sample = self.samples[idx]

        latent_data = torch.load(sample["latent_path"], map_location="cpu")
        if isinstance(latent_data, dict):
            if "latent" in latent_data:
                latent = latent_data["latent"].float()
            else:
                # Try to find tensor in dict
                tensors = [v for v in latent_data.values() if isinstance(v, torch.Tensor)]
                if tensors:
                    latent = tensors[0].float()
                else:
                    raise ValueError(f"Could not find latent tensor in {sample['latent_path']}")
        elif isinstance(latent_data, torch.Tensor):
            latent = latent_data.float()
        else:
            raise ValueError(f"Unexpected latent format in {sample['latent_path']}")
        
        # Latents are cached pre-scaled; undo it here.
        latent = latent / self.scaling_factor
        if self.shift_factor is not None:
            latent = latent + self.shift_factor

        xyz_img = cv2.imread(str(sample["gt_xyz_path"]), cv2.IMREAD_UNCHANGED)
        if xyz_img is None:
            raise ValueError(f"Failed to load XYZ image: {sample['gt_xyz_path']}")
        
        # Handle different image formats
        if xyz_img.ndim == 2:
            xyz_img = np.stack([xyz_img, xyz_img, xyz_img], axis=-1)
        if xyz_img.ndim == 3 and xyz_img.shape[2] == 4:
            xyz_img = xyz_img[:, :, :3]
        
        if xyz_img.dtype == np.uint8:
            xyz01 = (xyz_img.astype(np.float32) / 255.0).astype(np.float32)
        elif xyz_img.dtype == np.uint16:
            xyz01 = (xyz_img.astype(np.float32) / 65535.0).astype(np.float32)
        else:
            xyz01 = xyz_img.astype(np.float32)

        xyz01 = np.ascontiguousarray(xyz01[:, :, ::-1])

        # Resize XYZ image if needed (for supervision resolution)
        # Latents are kept at their encoded resolution - VAE decoder handles upscaling
        if self.target_hw is not None:
            h, w = self.target_hw
            if xyz01.shape[0] != h or xyz01.shape[1] != w:
                xyz01 = cv2.resize(xyz01, (w, h), interpolation=cv2.INTER_LANCZOS4)

        if not xyz01.flags['C_CONTIGUOUS']:
            xyz01 = np.ascontiguousarray(xyz01)

        xyz_t01 = torch.from_numpy(xyz01.transpose(2, 0, 1)).to(torch.float32)  # [C, H, W]
        xyz_t = srgb01_to_m11(xyz_t01)

        return latent, xyz_t, sample["basename"]


# -------------------------
# Validation for Latent-based Decoder Fine-tuning
# -------------------------

class DecoderForDDP(nn.Module):
    """Exposes vae.decode as forward so DDP synchronizes gradients.

    Training must call this wrapper, not vae.decode directly.
    """

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    @property
    def decoder(self):
        return self.vae.decoder

    @property
    def config(self):
        return self.vae.config

    def forward(self, z):
        return self.vae.decode(z).sample


def run_validation_latent_based(
    vae_ddp,
    dl_val, 
    metrics: MetricCollection, 
    local_rank: int, 
    epoch: int, 
    global_step: int, 
    cfg,
    dtype,
    logger = None,
    use_wandb=False
):
    """Decode cached latents and score them against GT XYZ.

    Returns the mean L1 loss used for best-checkpoint selection.
    """
    was_training = vae_ddp.training
    vae_ddp.eval()
    
    # Reset metrics state at the beginning of validation
    metrics.reset()
    
    # Reservoir sampling for visualization on rank 0
    rng = random.Random(cfg.seed + epoch)
    viz_capacity = cfg.num_val_visualize
    viz_buffer = []
    processed_samples = 0
    
    loss_sum = 0.0
    num_batches = 0

    desc = f"Validation Epoch {epoch}"
    pbar = tqdm(dl_val, desc=desc, disable=not is_main_process())
    
    with torch.no_grad():
        for z, x_gt, filenames in pbar:
            z = z.to(local_rank, non_blocking=True)
            x_gt = x_gt.to(local_rank, non_blocking=True)
            
            # Decode latent directly (no encoding needed)
            with torch.amp.autocast("cuda", enabled=cfg.amp, dtype=dtype):
                x_pred = vae_ddp.module(z)

            # Guard against non-finite values to avoid NaN metrics
            finite_pred = torch.isfinite(x_pred)
            finite_gt = torch.isfinite(x_gt)
            if not (finite_pred.all() and finite_gt.all()):
                if is_main_process():
                    pred_bad = finite_pred.numel() - finite_pred.sum().item()
                    gt_bad = finite_gt.numel() - finite_gt.sum().item()
                    pred_min = x_pred.min().item()
                    pred_max = x_pred.max().item()
                    gt_min = x_gt.min().item()
                    gt_max = x_gt.max().item()
                    print(
                        f"[WARN] Non-finite in val batch. "
                        f"pred_bad={pred_bad} gt_bad={gt_bad} "
                        f"pred_min={pred_min:.4f} pred_max={pred_max:.4f} "
                        f"gt_min={gt_min:.4f} gt_max={gt_max:.4f}"
                    )
                continue
            
            x_pred_01, x_gt_01 = m11_to_01(x_pred), m11_to_01(x_gt)
            
            # Update PSNR/SSIM metrics on the current batch (runs on GPU)
            metrics.update(x_pred_01, x_gt_01)
            
            loss_sum += F.l1_loss(x_pred, x_gt).item()
            num_batches += 1
            
            # Reservoir sampling logic
            if is_main_process():
                for i in range(z.size(0)):
                    item = (x_gt[i].cpu(), x_pred[i].cpu(), filenames[i])
                    if len(viz_buffer) < viz_capacity:
                        viz_buffer.append(item)
                    else:
                        j = rng.randint(0, processed_samples)
                        if j < viz_capacity:
                            viz_buffer[j] = item
                    processed_samples += 1
    
    # .compute() handles DDP synchronization and aggregation automatically
    final_metrics = metrics.compute()
    
    val_loss = loss_sum / num_batches if num_batches > 0 else 0.0

    # Synchronize val_loss across processes for DDP
    if dist.is_initialized():
        val_loss_tensor = torch.tensor([val_loss], device=local_rank, dtype=torch.float32)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
        val_loss = val_loss_tensor.item()
    
    if is_main_process():
        avg_psnr = final_metrics['psnr'].item()
        avg_ssim = final_metrics['ssim'].item()
        
        # Print validation results
        print(f"\n[Validation] Epoch {epoch}:")
        print(f"  PSNR={avg_psnr:.3f}, SSIM={avg_ssim:.4f}, L1={val_loss:.6f}")

        # Log to wandb/logger
        log_dict = {
            "val/psnr": avg_psnr,
            "val/ssim": avg_ssim,
            "val/loss": val_loss,
            "val/epoch": epoch,
            "val/step": global_step
        }
        if logger is not None:
            logger.log(log_dict, step=global_step)
        
        # Visualize sampled items
        save_dir = cfg.visualize_dir / f"val_epoch_{epoch}"
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving {len(viz_buffer)} validation visualizations to {save_dir}...")
        for x_gt_one, x_pr_one, fname in viz_buffer:
            psnr, ssim_val = visualize_results(
                x_gt_one, x_pr_one, fname, epoch, save_dir, use_wandb, log_prefix="val"
            )
    
    if was_training:
        vae_ddp.train()
    
    return val_loss


# -------------------------
# Config
# -------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune VAE Decoder for XYZ Reconstruction")
    
    # Model & Data
    parser.add_argument("--model-id", type=str, default="black-forest-labs/FLUX.1-dev",
                        choices=[
                            "black-forest-labs/FLUX.1-dev",
                        ],
                        help="Model ID for VAE loading")
    parser.add_argument("--trainval-json", type=str, default="../trainval.json",
                        help="Path to trainval.json manifest")
    parser.add_argument("--root-key", type=str, default="root",
                        help="Root key in manifest (e.g., 'root' or 'root_aws')")
    parser.add_argument("--datasets", type=str, nargs="+", default=["all"],
                        help="Dataset subsets to include")
    parser.add_argument("--latent-dir-name", type=str, default="latents",
                        help="Subdirectory name containing XYZ latents")
    parser.add_argument("--gt-xyz-dir-name", type=str, default="anchor-variants-imgs",
                        help="Subdirectory name containing GT XYZ images")
    parser.add_argument("--image-size", type=int, default=1024,
                        help="Target image size for supervision (GT XYZ will be resized to this)")
    parser.add_argument("--use-shift-factor", action="store_true", default=False,
                        help="Whether to apply shift factor to latents")

    # Training
    parser.add_argument("--freeze-modules", type=str, nargs="+", default=[],
                        help="Decoder module name substrings to freeze")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--max-epochs", type=int, default=150,
                        help="Maximum number of epochs")
    parser.add_argument("--grad-accum-steps", type=int, default=2,
                        help="Gradient accumulation steps")
    parser.add_argument("--amp", action="store_true", default=False,
                        help="Enable autocast for mixed precision")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"],
                        help="Compute dtype under autocast (default: bfloat16)")
    parser.add_argument("--param-dtype", type=str, default=None,
                        choices=["float32", "float16", "bfloat16"],
                        help="dtype for trainable params. Use float32 with --amp --dtype bfloat16 "
                             "for fp32 master weights.")
    parser.add_argument("--clip-grad-norm", type=float, default=1.0,
                        help="Gradient clipping norm")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay for AdamW")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # Learning Rate Scheduler
    parser.add_argument("--use-scheduler", action="store_true", default=True,
                        help="Use learning rate scheduler")
    parser.add_argument("--warmup-steps", type=int, default=1000,
                        help="Number of warmup steps")
    parser.add_argument("--scheduler-type", type=str, default="cosine",
                        choices=["cosine", "linear", "none"],
                        help="Type of LR scheduler after warmup")
    parser.add_argument("--min-lr", type=float, default=1e-7,
                        help="Minimum learning rate for cosine decay")
    
    # Dataloader
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Number of dataloader workers")
    parser.add_argument("--prefetch-factor", type=int, default=3,
                        help="Dataloader prefetch factor")
    parser.add_argument("--pin-memory", action="store_true", default=True,
                        help="Pin memory for dataloader")
    parser.add_argument("--persistent-workers", action="store_true", default=True,
                        help="Use persistent workers")
    
    # Validation & Logging
    parser.add_argument("--valid-every-n-epochs", type=int, default=10,
                        help="Run validation every N epochs")
    parser.add_argument("--visualize-per-n-epochs", type=int, default=10,
                        help="Visualize results every N epochs")
    parser.add_argument("--save-every", type=int, default=10,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--log-every", type=int, default=100,
                        help="Log metrics every N steps")
    parser.add_argument("--val-batch-size", type=int, default=2,
                        help="Validation batch size")
    parser.add_argument("--num-val-visualize", type=int, default=5,
                        help="Number of validation samples to visualize")
    
    # Wandb
    parser.add_argument("--use-wandb", action="store_true", default=False,
                        help="Use wandb for logging")
    parser.add_argument("--wandb-project", type=str, default="vae-decoder-xyz-finetune",
                        help="Wandb project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="Wandb entity")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                        help="Wandb run name")
    
    # TensorBoard
    parser.add_argument("--use-tb", action="store_true", default=True,
                        help="Use TensorBoard for logging")
    parser.add_argument("--tb-root-dir", type=str, nargs='+', default=['./tensorboard'],
                        help="TensorBoard root directory (can specify multiple paths)")
    
    # Results directory
    parser.add_argument("--results-root", type=str, default="./results/",
                        help="Root directory for saving experiment results")
    parser.add_argument("--run-name-prefix", type=str, default="finetune",
                    help="Prefix for run names in results directory")
    
    args = parser.parse_args()
    
    # Convert string paths to Path objects
    args.trainval_json = Path(args.trainval_json)
    args.out_dir = Path("./vae_decoder_xyz_ckpt_finetune")
    
    return args


# -------------------------
# Training
# -------------------------

def main():
    args = parse_args()
    setup_ddp()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.backends.cudnn.benchmark = True

    # Initialize logger
    logger = init_logger(args, default_run_name_prefix=args.run_name_prefix)
    
    base_dir = logger.get_base_dir(args.results_root)
    args.out_dir = base_dir / "vae_decoder_xyz_ckpt_finetune"
    args.visualize_dir = base_dir / "visualizations"
    if is_main_process():
        args.out_dir.mkdir(parents=True, exist_ok=True)
        args.visualize_dir.mkdir(parents=True, exist_ok=True)
    
    set_seed(args.seed)

    # Load VAE model using utility function
    if is_main_process():
        print(f"Loading VAE model: {args.model_id}")
    
    # Determine dtype
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]
    param_dtype = dtype_map[args.param_dtype] if args.param_dtype else dtype
    if is_main_process():
        print(f"Parameter dtype: {param_dtype} | autocast compute dtype: {dtype} (amp={args.amp})")

    vae = load_vae_for_training(
        model_id=args.model_id,
        device=torch.device(f"cuda:{local_rank}"),
        freeze_encoder=True,
        dtype=param_dtype
    )
    
    # Get scaling factor from VAE config
    scaling_factor = getattr(vae.config, 'scaling_factor', 0.18215)  # Default SD value
    shift_factor = getattr(vae.config, 'shift_factor', None)  # Default SD value
    shift_factor = shift_factor if args.use_shift_factor else None
    if is_main_process():
        print(f"Using scaling factor: {scaling_factor}")
        if shift_factor is not None:
            print(f"Using shift factor: {shift_factor}")
    
    # Delete encoder to save memory (we only use decoder for this task)
    if is_main_process():
        print("Deleting encoder to save memory...")
    del vae.encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Freeze selected decoder modules in full mode
    if args.freeze_modules:
        if is_main_process():
            print(f"Freezing decoder modules matching: {args.freeze_modules}")
        frozen = []
        for module_name, module in vae.decoder.named_modules():
            if any(key in module_name for key in args.freeze_modules):
                for param in module.parameters():
                    param.requires_grad = False
                frozen.append(module_name)
        if is_main_process():
            print(f"Frozen module count: {len(frozen)}")
    
    # Wrap with DDP — see DecoderForDDP.
    vae = DDP(DecoderForDDP(vae), device_ids=[local_rank], find_unused_parameters=False)

    if is_main_process():
        trainable_params = sum(p.numel() for p in vae.module.decoder.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in vae.module.decoder.parameters())
        print(f"Trainable params (decoder): {trainable_params:,}")
        print(f"Total params (decoder): {total_params:,}")
        if total_params > 0:
            print(f"Trainable ratio: {100.0 * trainable_params / total_params:.2f}%")
        logger.log_at_step_0({
            "model/trainable_params": trainable_params,
            "model/total_params": total_params,
        })

    if is_main_process():
        print(f"Loading manifest: {str(args.trainval_json)}")
    
    # Create datasets
    ds_train = XYZLatentDataset(
        manifest_path=args.trainval_json,
        root_key=args.root_key,
        split="train",
        dataset_subsets=args.datasets,
        latent_dir_name=args.latent_dir_name,
        gt_xyz_dir_name=args.gt_xyz_dir_name,
        image_size=args.image_size,
        scaling_factor=scaling_factor,
        shift_factor=shift_factor,
    )
    
    ds_val = XYZLatentDataset(
        manifest_path=args.trainval_json,
        root_key=args.root_key,
        split="val",
        dataset_subsets=args.datasets,
        latent_dir_name=args.latent_dir_name,
        gt_xyz_dir_name=args.gt_xyz_dir_name,
        image_size=args.image_size,
        scaling_factor=scaling_factor,
        shift_factor=shift_factor,
    )
    
    global_rank = dist.get_rank()
    sampler_train = DistributedSampler(ds_train, num_replicas=world_size, rank=global_rank, shuffle=True)
    sampler_val = DistributedSampler(ds_val, num_replicas=world_size, rank=global_rank, shuffle=False)
    
    dl_args = {
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "persistent_workers": args.persistent_workers,
        "prefetch_factor": args.prefetch_factor
    }
    dl = DataLoader(ds_train, batch_size=args.batch_size, sampler=sampler_train, shuffle=False, **dl_args)
    dl_val = DataLoader(ds_val, batch_size=args.val_batch_size, sampler=sampler_val, shuffle=False, **dl_args)

    # Optimizer and scheduler
    # Get trainable parameters (respect requires_grad set by module freezing)
    trainable_params = [p for p in vae.module.decoder.parameters() if p.requires_grad]
    
    opt = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay
    )
    
    # Create scheduler
    total_steps_per_epoch = len(dl) // args.grad_accum_steps
    scheduler, _ = get_scheduler(opt, args, total_steps_per_epoch)
    
    # Validation metrics
    val_metrics = MetricCollection({
        'psnr': PeakSignalNoiseRatio(data_range=1.0),
        'ssim': StructuralSimilarityIndexMeasure(data_range=1.0)
    }).to(local_rank)

    global_step = 0
    best_val_loss = float('inf')
    vae.train()
    
    for epoch in range(1, args.max_epochs + 1):
        sampler_train.set_epoch(epoch)
        pbar = tqdm(dl, desc=f"Epoch {epoch}/{args.max_epochs}", dynamic_ncols=True, disable=not is_main_process())
        
        running_loss = 0.0
        last_batch_data = None
        
        for i, (z, x_gt, filenames) in enumerate(pbar):
            z = z.to(local_rank, non_blocking=True)
            x_gt = x_gt.to(local_rank, non_blocking=True)
            
            with torch.amp.autocast("cuda", enabled=args.amp, dtype=dtype):
                # Decode latent to XYZ, through DDP so gradients are all-reduced
                x_pred = vae(z)
                loss = F.l1_loss(x_pred, x_gt)

            (loss / args.grad_accum_steps).backward()

            if (i + 1) % args.grad_accum_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(vae.module.decoder.parameters(), args.clip_grad_norm)
                opt.step()
                opt.zero_grad(set_to_none=True)
                
                if scheduler is not None:
                    scheduler.step()
                
                if is_main_process() and global_step % args.log_every == 0:
                    log_dict = {
                        "train/grad_norm": grad_norm.item(),
                        "train/step": global_step
                    }
                    if scheduler is not None:
                        log_dict["train/learning_rate"] = scheduler.get_last_lr()[0]
                    logger.log(log_dict, step=global_step)

                global_step += 1

            running_loss += loss.item()

            if is_main_process() and global_step % args.log_every == 0:
                avg_loss = running_loss / (i + 1)
                pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
                logger.log({
                    "train/loss": avg_loss,
                    "train/learning_rate": opt.param_groups[0]['lr'],
                    "train/epoch": epoch,
                    "train/step": global_step
                }, step=global_step)

            last_batch_data = (z, x_gt, x_pred, filenames)

        if is_main_process():
            logger.log({
                "train/epoch_loss": running_loss / len(dl),
                "train/epoch": epoch,
                "train/step": global_step
            }, step=global_step)
        
        if is_main_process() and epoch % args.visualize_per_n_epochs == 0 and last_batch_data:
            print(f"\nVisualizing training results for epoch {epoch}...")
            z, x_gt, x_pred, filenames = last_batch_data
            psnr, ssim_val = visualize_results(
                x_gt[0], x_pred[0], filenames[0],
                epoch, args.visualize_dir, False, log_prefix="train"
            )
            logger.log({
                "train/vis_psnr": psnr,
                "train/vis_ssim": ssim_val,
                "train/epoch": epoch,
                "train/step": global_step
            }, step=global_step)
        
        # Run validation and get validation loss
        if epoch % args.valid_every_n_epochs == 0:
            val_loss = run_validation_latent_based(
                vae, dl_val, val_metrics, local_rank, epoch, global_step, args, dtype,
                logger=logger, use_wandb=args.use_wandb
            )
            
            # Save best checkpoint based on validation loss
            if is_main_process() and val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"New best val loss: {val_loss:.6f} (epoch {epoch})")
                save_decoder(vae, args.out_dir / "vae_decoder_xyz_best.pt")
                print(f"Saved best decoder checkpoint")
                
                # Log best val loss
                logger.log({
                    "val/best_loss": best_val_loss,
                    "val/best_epoch": epoch,
                    "val/step": global_step
                }, step=global_step)

        if is_main_process() and epoch % args.save_every == 0:
            # Save full decoder
            save_decoder(vae, args.out_dir / f"vae_decoder_xyz_epoch{epoch}.pt")

    if is_main_process():
        # Save full decoder
        save_decoder(vae, args.out_dir / f"vae_decoder_xyz_final.pt")

        print(f"\nTraining complete!")
        print(f"Best validation loss: {best_val_loss:.6f}")
        logger.close()
    cleanup_ddp()


if __name__ == "__main__":
    main()

