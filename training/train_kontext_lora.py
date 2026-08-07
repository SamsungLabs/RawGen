"""LoRA fine-tuning of the FLUX.1-Kontext transformer (variant -> anchor).

The variant latent conditions the model by sequence concatenation rather than
by widening the input channels.
"""

import os
# Disable tokenizer parallelism to avoid deadlocks in DDP
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from diffusers import FluxPipeline
from tqdm import tqdm

from peft import LoraConfig, get_peft_model
from peft.tuners.lora.config import LoraRuntimeConfig
from peft.utils.peft_types import PeftType

from _train_helpers import VariantAnchorLatentDataset
from _train_helpers import (
    setup_ddp, cleanup_ddp, is_main_process, set_seed, get_scheduler,
    visualize_diffusion_results
)
from _train_helpers.logging_utils import init_logger


# -------------------------
# Model Modification Functions
# -------------------------

def apply_lora_to_transformer(transformer: nn.Module, lora_config: LoraConfig) -> nn.Module:
    """Apply LoRA adapters to the FLUX transformer."""
    print("Applying LoRA to transformer...")
    
    # Apply LoRA
    model_with_lora = get_peft_model(transformer, lora_config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model_with_lora.parameters())
    trainable_params = sum(p.numel() for p in model_with_lora.parameters() if p.requires_grad)
    
    print(f"LoRA applied:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Trainable ratio: {trainable_params/total_params*100:.2f}%")
    
    return model_with_lora


def extract_adapter_state_dict(model: nn.Module) -> dict:
    """Return only the LoRA adapter tensors of a PEFT-wrapped model."""
    return {k: v for k, v in model.state_dict().items() if ".lora_" in k}


# -------------------------
# Config
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="FLUX.1 Kontext Diffusion Transformer Training (LoRA)")
    
    # Model & Data
    parser.add_argument("--model-id", type=str, default="black-forest-labs/FLUX.1-Kontext-dev",
                        help="FLUX.1 Kontext model ID")
    parser.add_argument("--datasets", type=str, nargs='+', required=True,
                        help="One or more dataset names, or 'all' (e.g., a5k raise)")
    parser.add_argument("--latent-dir-name", type=str, required=True,
                        help="Subdirectory under dataset_type containing anchor/variant latents")
    parser.add_argument("--trainval-json", type=str, default="../trainval.json",
                        help="Path to trainval.json manifest")
    parser.add_argument("--root-key", type=str, default="root",
                        help="Root key in manifest for dataset subsets (e.g., 'root' or 'root_aws')")
    parser.add_argument("--num-variations", type=int, default=5,
                        help="Number of variants per anchor")

    # LoRA parameters
    parser.add_argument("--lora-rank", type=int, default=64,
                        help="LoRA rank (higher = more capacity, more memory)")
    parser.add_argument("--lora-alpha", type=int, default=64,
                        help="LoRA alpha (scaling factor)")
    parser.add_argument("--lora-dropout", type=float, default=0.0,
                        help="LoRA dropout")
    parser.add_argument("--lora-target-modules", nargs="+", 
                        default=["to_q", "to_k", "to_v", "to_out.0"],
                        help="Target modules for LoRA")
    
    # Training
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Training batch size per GPU")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--max-epochs", type=int, default=50,
                        help="Maximum number of epochs")
    parser.add_argument("--grad-accum-steps", type=int, default=2,
                        help="Gradient accumulation steps")
    parser.add_argument("--clip-grad-norm", type=float, default=1.0,
                        help="Gradient clipping norm")
    parser.add_argument("--amp", action="store_true", default=False,
                        help="Enable autocast for mixed precision")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"],
                        help="Training dtype (default: bfloat16)")
    parser.add_argument("--param-dtype", type=str, default=None,
                        choices=["float32", "float16", "bfloat16"],
                        help="dtype for trainable LoRA params. Defaults to --dtype.")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=False,
                        help="Enable gradient checkpointing to save memory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay for AdamW")
    parser.add_argument("--use-shift-factor", action="store_true", default=False,
                        help="Apply vae.config.shift_factor when decoding latents in visualization. "
                             "Off by default to match original training convention.")

    # Learning Rate Scheduler
    parser.add_argument("--use-scheduler", action="store_true", default=True,
                        help="Use learning rate scheduler")
    parser.add_argument("--warmup-steps", type=int, default=500,
                        help="Number of warmup steps")
    parser.add_argument("--scheduler-type", type=str, default="cosine",
                        choices=["cosine", "linear", "none"],
                        help="Type of LR scheduler after warmup")
    parser.add_argument("--min-lr", type=float, default=1e-7,
                        help="Minimum learning rate for cosine decay")
    
    # Dataloader
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Number of dataloader workers")
    parser.add_argument("--pin-memory", action="store_true", default=True,
                        help="Pin memory for dataloader")
    parser.add_argument("--persistent-workers", action="store_true", default=True,
                        help="Use persistent workers")
    
    # Validation & Logging
    parser.add_argument("--valid-every-n-epochs", type=int, default=10,
                        help="Run validation every N epochs")
    parser.add_argument("--save-every", type=int, default=20,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--log-every", type=int, default=50,
                        help="Log metrics every N steps")
    parser.add_argument("--val-batch-size", type=int, default=4,
                        help="Validation batch size")
    
    # Wandb
    parser.add_argument("--use-wandb", action="store_true", default=False,
                        help="Use wandb for logging")
    parser.add_argument("--wandb-project", type=str, default="rawgen-kontext-diffusion-lora",
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
    
    # Resume
    parser.add_argument("--auto-resume", action="store_true", default=False,
                        help="Auto-resume from latest checkpoint in results dir")
    parser.add_argument("--resume-path", type=str, default=None,
                        help="Explicit checkpoint path (overrides auto-resume)")

    # Results directory
    parser.add_argument("--results-root", type=str, default="./results/",
                        help="Root directory for saving experiment results")
    parser.add_argument("--run-name-prefix", type=str, default="kontext-lora",
                        help="Prefix for run names in results directory")
    
    # Architecture
    parser.add_argument("--latent-channels", type=int, default=16,
                        help="Number of latent channels (FLUX default: 16)")
    
    args = parser.parse_args()
    
    # Convert string paths to Path objects
    args.trainval_json = Path(args.trainval_json)

    return args


def create_null_text_embeddings(
    text_encoder,
    text_encoder_2,
    tokenizer,
    tokenizer_2,
    device,
    dtype
):
    """Build encoder + pooled embeddings for the empty prompt ''."""
    prompt = ""

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    
    text_inputs_2 = tokenizer_2(
        prompt,
        padding="max_length",
        max_length=tokenizer_2.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    
    with torch.no_grad():
        # CLIP (text_encoder) - used for pooled projection only
        prompt_embeds = text_encoder(
            text_inputs.input_ids.to(device),
            output_hidden_states=True
        )
        pooled_prompt_embeds = prompt_embeds.pooler_output  # CLIP pooled output
        
        # T5 (text_encoder_2) - used for encoder hidden states
        prompt_embeds_2 = text_encoder_2(
            text_inputs_2.input_ids.to(device),
            output_hidden_states=True
        )
        # Use T5's last hidden state as encoder_hidden_states
        # FLUX primarily uses T5 for text conditioning
        encoder_hidden_states = prompt_embeds_2.hidden_states[-1]
        
        # For pooled projection, use only CLIP's pooled output
        pooled_projections = pooled_prompt_embeds
    
    return encoder_hidden_states.to(dtype), pooled_projections.to(dtype)


def prepare_latent_image_ids(num_patches_h, num_patches_w, device, dtype):
    """Build the (N, 3) latent image ids tensor used by FLUX RoPE."""
    # Create 2D grid of positional IDs for patches
    latent_image_ids = torch.zeros(num_patches_h, num_patches_w, 3, device=device, dtype=dtype)
    latent_image_ids[..., 1] = torch.arange(num_patches_h, device=device, dtype=dtype)[:, None]
    latent_image_ids[..., 2] = torch.arange(num_patches_w, device=device, dtype=dtype)[None, :]
    
    # Flatten to 2D (no batch dimension)
    latent_image_ids = latent_image_ids.reshape(-1, 3)
    
    return latent_image_ids


def prepare_text_ids(encoder_hidden_states, device, dtype):
    """(seq_len, 3) text ids: column 0 = arange(seq_len), columns 1-2 = 0."""
    if encoder_hidden_states.ndim == 3:
        seq_len = encoder_hidden_states.shape[1]
    else:
        seq_len = encoder_hidden_states.shape[0]
    
    # Sequential IDs along column 0; columns 1-2 stay zero.
    text_ids = torch.zeros(seq_len, 3, device=device, dtype=dtype)
    text_ids[:, 0] = torch.arange(seq_len, device=device, dtype=dtype)

    return text_ids


# -------------------------
# Training
# -------------------------
def main():
    args = parse_args()
    setup_ddp()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.backends.cudnn.benchmark = True

    # Auto-resume: find latest run folder with checkpoints
    resume_ckpt_path = None
    if args.resume_path:
        resume_ckpt_path = Path(args.resume_path)
    elif args.auto_resume and is_main_process():
        results_root = Path(args.results_root)
        if results_root.exists():
            # Find folders matching run_name_prefix pattern
            candidates = sorted(
                [d for d in results_root.iterdir()
                 if d.is_dir() and args.run_name_prefix in d.name],
                key=lambda d: d.name
            )
            for cand in reversed(candidates):  # most recent first
                ckpt_dir = cand / "kontext_lora_ckpt"
                if ckpt_dir.exists():
                    ckpts = sorted(ckpt_dir.glob("kontext_lora_epoch*.pt"))
                    if ckpts:
                        resume_ckpt_path = ckpts[-1]  # latest epoch
                        # Reuse this run folder: extract run_name from folder name
                        # folder name format: YYMMDD-{run_name}
                        folder_name = cand.name
                        parts = folder_name.split("-", 1)
                        if len(parts) == 2:
                            args.wandb_run_name = parts[1]
                        print(f"Auto-resume found: {resume_ckpt_path}")
                        break

    # Broadcast resume_ckpt_path to all processes
    resume_info = [str(resume_ckpt_path) if resume_ckpt_path else None]
    if dist.is_initialized():
        dist.broadcast_object_list(resume_info, src=0)
    if resume_info[0] is not None:
        resume_ckpt_path = Path(resume_info[0])

    # Initialize logger (handles wandb and TensorBoard)
    logger = init_logger(args, default_run_name_prefix=args.run_name_prefix)
    
    base_dir = logger.get_base_dir(args.results_root)
    args.out_dir = base_dir / "kontext_lora_ckpt"
    args.visualize_dir = base_dir / "visualizations"
    if is_main_process():
        args.out_dir.mkdir(parents=True, exist_ok=True)
        args.visualize_dir.mkdir(parents=True, exist_ok=True)
    
    # Rank-invariant seed for model init; per-rank offset applied after (see below).
    set_seed(args.seed)

    # --- MODEL LOADING ---
    if is_main_process():
        print(f"Loading FLUX.1 Kontext model: {args.model_id}")
        print(f"Precision: {'bfloat16 (AMP enabled)' if args.amp else 'bfloat16'}")
        print(f"Gradient checkpointing: {'enabled' if args.gradient_checkpointing else 'disabled'}")
        print(f"LoRA rank: {args.lora_rank}, alpha: {args.lora_alpha}")
    
    # Determine dtype
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]
    
    # Main process downloads first; others load from cache.
    if is_main_process():
        print("Downloading model (main process only)...")
        pipe = FluxPipeline.from_pretrained(
            args.model_id,
            torch_dtype=dtype
        )
    
    # Synchronize: wait for main process to finish downloading
    if world_size > 1:
        dist.barrier()
    
    # Other processes load from cache
    if not is_main_process():
        pipe = FluxPipeline.from_pretrained(
            args.model_id,
            torch_dtype=dtype
        )
    
    transformer = pipe.transformer
    scheduler = pipe.scheduler
    text_encoder = pipe.text_encoder
    text_encoder_2 = pipe.text_encoder_2
    tokenizer = pipe.tokenizer
    tokenizer_2 = pipe.tokenizer_2
    
    # Check if transformer uses guidance embeddings
    use_guidance_embed = getattr(transformer.config, 'guidance_embeds', False)

    if is_main_process():
        print("Applying LoRA adapters...")
    
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
    )

    transformer = apply_lora_to_transformer(transformer, lora_config)

    # fp32 master weights for the adapter; must precede .to()/DDP/optimizer.
    if args.param_dtype is not None:
        _pdt = dtype_map[args.param_dtype]
        for p in transformer.parameters():
            if p.requires_grad:
                p.data = p.data.to(_pdt)

    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        if is_main_process():
            print("Enabling gradient checkpointing...")
        if hasattr(transformer, 'enable_gradient_checkpointing'):
            transformer.enable_gradient_checkpointing()
        elif hasattr(transformer, 'base_model') and hasattr(transformer.base_model.model, 'enable_gradient_checkpointing'):
            transformer.base_model.model.enable_gradient_checkpointing()
        else:
            if is_main_process():
                print("Warning: Could not enable gradient checkpointing")
    
    transformer = transformer.to(local_rank)
    text_encoder = text_encoder.to(local_rank)
    text_encoder_2 = text_encoder_2.to(local_rank)

    for param in text_encoder.parameters():
        param.requires_grad = False
    for param in text_encoder_2.parameters():
        param.requires_grad = False
    
    # Create null text embeddings for unconditional generation
    if is_main_process():
        print("Creating null text embeddings...")
    encoder_hidden_states_null, pooled_projections_null = create_null_text_embeddings(
        text_encoder, text_encoder_2, tokenizer, tokenizer_2,
        device=local_rank, dtype=dtype
    )
    
    # Create text IDs for null embeddings (required by FLUX transformer)
    txt_ids_null = prepare_text_ids(encoder_hidden_states_null, device=local_rank, dtype=dtype)
    
    # Store vae_scale_factor before deleting pipe (needed for packing/unpacking)
    vae_scale_factor = pipe.vae_scale_factor
    
    # Free pipeline memory
    del pipe
    torch.cuda.empty_cache()
    
    # Wrap with DDP
    transformer = DDP(transformer, device_ids=[local_rank], find_unused_parameters=False)

    # Per-rank RNG offset so each rank draws its own (timestep, noise).
    # Must come after model init to keep LoRA injection rank-invariant.
    _rank_seed = args.seed + dist.get_rank()
    torch.manual_seed(_rank_seed)
    torch.cuda.manual_seed_all(_rank_seed)


    # Count trainable parameters
    trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    
    if is_main_process():
        total_params = sum(p.numel() for p in transformer.parameters())
        log_dict = {
            "model/trainable_params": trainable_params,
            "model/total_params": total_params,
            "config/lora_rank": args.lora_rank,
            "config/lora_alpha": args.lora_alpha,
            "config/gradient_checkpointing": args.gradient_checkpointing,
            "config/amp": args.amp,
            "config/architecture": "kontext_sequence_concat",
        }
        logger.log_at_step_0(log_dict)
    
    # --- DATASET LOADING ---
    # Accept multiple datasets via nargs '+'
    if isinstance(args.datasets, list):
        if len(args.datasets) == 1 and args.datasets[0].lower() == "all":
            dataset_list = "all"
        else:
            dataset_list = args.datasets
    else:
        dataset_list = "all" if str(args.datasets).lower() == "all" else [args.datasets]
    if is_main_process():
        print(f"Loading datasets: datasets={args.datasets} -> parsed={dataset_list}, latent_dir_name={args.latent_dir_name}")
    
    # Use VariantAnchorLatentDataset for variant → anchor mapping
    ds_train = VariantAnchorLatentDataset(
        args.trainval_json,
        root_key=args.root_key,
        split="train",
        datasets=dataset_list,
        num_variations=args.num_variations,
        latent_dir_name=args.latent_dir_name,
    )
    ds_val = VariantAnchorLatentDataset(
        args.trainval_json,
        root_key=args.root_key,
        split="val",
        datasets=dataset_list,
        num_variations=args.num_variations,
        latent_dir_name=args.latent_dir_name,
    )

    if is_main_process():
        print(f"Dataset loaded -> train: {len(ds_train)}, val: {len(ds_val)}")
    
    global_rank = dist.get_rank()
    sampler_train = DistributedSampler(ds_train, num_replicas=world_size, rank=global_rank, shuffle=True)
    sampler_val = DistributedSampler(ds_val, num_replicas=world_size, rank=global_rank, shuffle=False)
    
    dl = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        sampler=sampler_train,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers if args.num_workers > 0 else False
    )
    
    dl_val = DataLoader(
        ds_val,
        batch_size=args.val_batch_size,
        sampler=sampler_val,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers if args.num_workers > 0 else False
    )
    
    # --- OPTIMIZER & SCHEDULER ---
    optimizer_params = [p for p in transformer.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        optimizer_params,
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay
    )
    
    # Create scheduler
    total_steps_per_epoch = len(dl) // args.grad_accum_steps
    scheduler_lr, _ = get_scheduler(opt, args, total_steps_per_epoch)
    
    # --- TRAINING LOOP ---
    if is_main_process():
        print("Starting training...")
        print(f"Total epochs: {args.max_epochs}")
        print(f"Steps per epoch: {len(dl)}")
        print(f"Effective steps per epoch: {total_steps_per_epoch}")
    
    # Cache params_to_clip to avoid recreating list every step
    params_to_clip = optimizer_params

    global_step = 0
    start_epoch = 1

    # Resume from checkpoint
    if resume_ckpt_path is not None and resume_ckpt_path.exists():
        if is_main_process():
            print(f"Loading checkpoint: {resume_ckpt_path}")
        # The checkpoint stores a peft LoraConfig object, so it needs an allowlist.
        with torch.serialization.safe_globals([LoraConfig, PeftType, LoraRuntimeConfig]):
            ckpt = torch.load(
                resume_ckpt_path, map_location=f"cuda:{local_rank}", weights_only=True
            )
        missing, unexpected = transformer.module.load_state_dict(
            ckpt["lora_weights"], strict=False
        )
        stale = unexpected + [k for k in missing if ".lora_" in k]
        if stale:
            raise ValueError(
                f"{resume_ckpt_path}: adapter tensors do not match this model "
                f"({len(stale)} mismatched, e.g. {stale[:3]})"
            )
        opt.load_state_dict(ckpt["optimizer"])
        if scheduler_lr is not None and ckpt.get("scheduler") is not None:
            scheduler_lr.load_state_dict(ckpt["scheduler"])
        global_step = ckpt.get("global_step", 0)
        start_epoch = ckpt["config"]["epoch"] + 1
        if is_main_process():
            print(f"Resumed from epoch {ckpt['config']['epoch']}, global_step {global_step}")
            print(f"Continuing from epoch {start_epoch}")
        del ckpt

    transformer.train()

    for epoch in range(start_epoch, args.max_epochs + 1):
        sampler_train.set_epoch(epoch)
        pbar = tqdm(dl, desc=f"Epoch {epoch}/{args.max_epochs}", disable=not is_main_process())
        running_loss = 0.0
        
        for i, batch_data in enumerate(pbar):
            # (anchor_latent, variant_latent, basename, variation_idx)
            anchor_latent, variant_latent, basename, variation_idx = batch_data
            anchor_latent = anchor_latent.to(local_rank, dtype=dtype)
            variant_latent = variant_latent.to(local_rank, dtype=dtype)
            
            B, C, H, W = anchor_latent.shape
            
            # Sample random timesteps (continuous for flow matching).
            # fp32 required: bf16 torch.rand has too few distinct values and can return exactly 0.
            timesteps = torch.rand(B, device=local_rank, dtype=torch.float32)

            noise = torch.randn_like(anchor_latent)

            # Flow matching: interpolate in fp32, cast back to model dtype.
            _t = timesteps.view(B, 1, 1, 1)
            noisy_latent = ((1 - _t) * anchor_latent + _t * noise).to(dtype)

            velocity_target = noise - anchor_latent
            
            # Pack latents to transformer format [B, num_patches, 64]
            # FLUX packing: [B, C, H, W] -> [B, (H//2)*(W//2), C*4] where patch_size=2
            num_patches_h = H // 2
            num_patches_w = W // 2
            num_patches = num_patches_h * num_patches_w
            
            # Pack latents - variant doesn't need gradients (context only)
            with torch.no_grad():
                variant_latent_packed = FluxPipeline._pack_latents(variant_latent, B, C, H, W)  # [B, num_patches, 64] - context
            noisy_latent_packed = FluxPipeline._pack_latents(noisy_latent, B, C, H, W)  # [B, num_patches, 64] - target
            velocity_target_packed = FluxPipeline._pack_latents(velocity_target, B, C, H, W)  # [B, num_patches, 64]
            
            # Sequence concatenation [context, target]
            # [B, num_patches, 64] + [B, num_patches, 64] -> [B, num_patches*2, 64]
            model_input = torch.cat([variant_latent_packed, noisy_latent_packed], dim=1)  # [B, num_patches*2, 64]

            encoder_hidden_states_batch = encoder_hidden_states_null.expand(B, -1, -1)
            txt_ids_batch = txt_ids_null  # Use pre-created 2D tensor

            pooled_projections_batch = pooled_projections_null.expand(B, -1)
            
            # Image IDs for the doubled sequence [context, target]; both halves share the same ids
            img_ids_context = prepare_latent_image_ids(num_patches_h, num_patches_w, local_rank, dtype)  # [num_patches, 3]
            img_ids_target = prepare_latent_image_ids(num_patches_h, num_patches_w, local_rank, dtype)  # [num_patches, 3]
            img_ids_batch = torch.cat([img_ids_context, img_ids_target], dim=0)  # [num_patches*2, 3]
            
            # Forward pass with autocast
            with torch.amp.autocast("cuda", enabled=args.amp, dtype=dtype):
                # Prepare forward arguments
                forward_kwargs = {
                    "hidden_states": model_input,  # [B, num_patches*2, 64]
                    "timestep": timesteps,
                    "encoder_hidden_states": encoder_hidden_states_batch,
                    "pooled_projections": pooled_projections_batch,
                    "txt_ids": txt_ids_batch,
                    "img_ids": img_ids_batch,  # [num_patches*2, 3]
                    "return_dict": False
                }
                # Add guidance only if model uses guidance embeddings
                if use_guidance_embed:
                    forward_kwargs["guidance"] = torch.full((B,), 3.5, device=local_rank, dtype=dtype)
                
                velocity_pred_full = transformer(**forward_kwargs)[0]  # [B, num_patches*2, 64]
                
                # KONTEXT: Extract target portion (second half) from output — target is last in sequence
                velocity_pred_packed = velocity_pred_full[:, num_patches:, :]  # [B, num_patches, 64]

                # Compute MSE loss on target portion only
                loss = F.mse_loss(velocity_pred_packed, velocity_target_packed)
            
            # Backward pass with gradient accumulation
            (loss / args.grad_accum_steps).backward()
            
            if (i + 1) % args.grad_accum_steps == 0:
                # Clip gradients (use cached params_to_clip)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    params_to_clip,
                    args.clip_grad_norm
                )

                opt.step()
                opt.zero_grad(set_to_none=True)
                
                # Update learning rate
                if scheduler_lr is not None:
                    scheduler_lr.step()
                
                # Log to wandb and TensorBoard
                if is_main_process() and global_step % args.log_every == 0:
                    log_dict = {
                        "train/grad_norm": grad_norm.item(),
                        "train/step": global_step
                    }
                    if scheduler_lr is not None:
                        log_dict["train/learning_rate"] = scheduler_lr.get_last_lr()[0]
                    logger.log(log_dict, step=global_step)
                
                global_step += 1
            
            running_loss += loss.item()
            
            # Update progress bar
            if is_main_process() and (i + 1) % args.log_every == 0:
                avg_loss = running_loss / (i + 1)
                pbar.set_postfix(loss=f"{avg_loss:.6f}")
                logger.log({
                    "train/loss": avg_loss,
                    "train/learning_rate": opt.param_groups[0]['lr'],
                    "train/epoch": epoch,
                    "train/step": global_step
                }, step=global_step)
        
        # Log epoch metrics
        if is_main_process():
            logger.log({
                "train/epoch_loss": running_loss / len(dl),
                "train/epoch": epoch,
                "train/step": global_step
            }, step=global_step)
        
        # Validation
        if epoch % args.valid_every_n_epochs == 0:
            if is_main_process():
                print(f"\n[Validation] Epoch {epoch}...")
            
            # Load VAE for visualization (main process only, not full pipeline)
            if is_main_process():
                print("Loading VAE for visualization...")
                from diffusers import AutoencoderKL

                vae = AutoencoderKL.from_pretrained(
                    args.model_id,
                    subfolder="vae",
                    torch_dtype=dtype
                ).to(local_rank)

                vae.eval()
                # Enable VAE memory-saving features
                if hasattr(vae, "enable_slicing"):
                    vae.enable_slicing()
                if hasattr(vae, "enable_tiling"):
                    vae.enable_tiling()

            transformer.eval()
            val_loss = 0.0
            val_count = 0
            
            # Collect samples for visualization
            vis_samples = []
            num_val_visualize = 5
            
            with torch.no_grad():
                for batch_data in tqdm(
                    dl_val, desc="Validation", disable=not is_main_process()
                ):
                    # (anchor_latent, variant_latent, basename, _)
                    anchor_latent, variant_latent, basename, _ = batch_data
                    anchor_latent = anchor_latent.to(local_rank, dtype=dtype)
                    variant_latent = variant_latent.to(local_rank, dtype=dtype)
                    
                    B, C, H, W = anchor_latent.shape
                    num_patches_h = H // 2
                    num_patches_w = W // 2
                    num_patches = num_patches_h * num_patches_w
                    
                    # Collect samples for visualization (first 5)
                    # Move to CPU to free GPU memory
                    if len(vis_samples) < num_val_visualize:
                        for b_idx in range(B):
                            if len(vis_samples) < num_val_visualize:
                                vis_samples.append((
                                    anchor_latent[b_idx:b_idx+1].cpu(),
                                    variant_latent[b_idx:b_idx+1].cpu(),
                                    basename[b_idx] if isinstance(basename, list) else basename
                                ))

                    # Dedicated generator: validation draws stay fixed across epochs.
                    val_gen = torch.Generator(device=local_rank).manual_seed(
                        args.seed + 10_000 + val_count
                    )
                    timesteps = torch.rand(B, device=local_rank, dtype=torch.float32, generator=val_gen)
                    noise = torch.randn(
                        anchor_latent.shape, device=local_rank, dtype=dtype, generator=val_gen
                    )
                    _t = timesteps.view(B, 1, 1, 1)
                    noisy_latent = ((1 - _t) * anchor_latent + _t * noise).to(dtype)
                    velocity_target = noise - anchor_latent
                    
                    # Pack latents
                    variant_latent_packed = FluxPipeline._pack_latents(variant_latent, B, C, H, W)
                    noisy_latent_packed = FluxPipeline._pack_latents(noisy_latent, B, C, H, W)
                    velocity_target_packed = FluxPipeline._pack_latents(velocity_target, B, C, H, W)
                    
                    # Sequence concatenation [context, target]
                    model_input = torch.cat([variant_latent_packed, noisy_latent_packed], dim=1)

                    encoder_hidden_states_val = encoder_hidden_states_null.expand(B, -1, -1)
                    txt_ids_val = txt_ids_null  # Use pre-created 2D tensor

                    pooled_projections_batch = pooled_projections_null.expand(B, -1)
                    
                    # Image IDs for the doubled sequence [context, target]
                    img_ids_context = prepare_latent_image_ids(num_patches_h, num_patches_w, local_rank, dtype)
                    img_ids_target = prepare_latent_image_ids(num_patches_h, num_patches_w, local_rank, dtype)
                    img_ids_batch = torch.cat([img_ids_context, img_ids_target], dim=0)
                    
                    with torch.amp.autocast("cuda", enabled=args.amp, dtype=dtype):
                        # Prepare forward arguments
                        forward_kwargs = {
                            "hidden_states": model_input,
                            "timestep": timesteps,
                            "encoder_hidden_states": encoder_hidden_states_val,
                            "pooled_projections": pooled_projections_batch,
                            "txt_ids": txt_ids_val,
                            "img_ids": img_ids_batch,
                            "return_dict": False
                        }
                        # Add guidance only if model uses guidance embeddings
                        if use_guidance_embed:
                            forward_kwargs["guidance"] = torch.full((B,), 3.5, device=local_rank, dtype=dtype)

                        velocity_pred_full = transformer(**forward_kwargs)[0]

                        # Extract target portion (second half) — target is last in sequence
                        velocity_pred_packed = velocity_pred_full[:, num_patches:, :]

                        loss = F.mse_loss(velocity_pred_packed, velocity_target_packed)

                    val_loss += loss.item()
                    val_count += 1
            
            # All-reduce validation loss across all GPUs for stable measurement
            val_loss_tensor = torch.tensor([val_loss, val_count], device=local_rank, dtype=torch.float64)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
            avg_val_loss = (val_loss_tensor[0] / val_loss_tensor[1]).item() if val_loss_tensor[1] > 0 else 0.0

            if is_main_process():
                print(f"[Validation] Epoch {epoch}: Loss={avg_val_loss:.6f}")
                logger.log({
                    "val/loss": avg_val_loss,
                    "val/epoch": epoch,
                    "val/step": global_step
                }, step=global_step)
            
            # Run multi-step inference and visualize
            if is_main_process() and len(vis_samples) > 0:
                print(f"\n[Visualization] Running inference on {len(vis_samples)} samples...")
                # Determine number of inference steps (6 for schnell, 30 for dev)
                num_inference_steps = 6 if "schnell" in args.model_id.lower() else 30
                vis_gen = torch.Generator(device=local_rank).manual_seed(
                    args.seed + 10_000 + val_count
                )
                # Inference mode to reduce memory
                with torch.inference_mode():
                    for anchor_latent_single, variant_latent_single, base_single in vis_samples:
                        # Move samples back to GPU for inference (they were stored on CPU)
                        anchor_latent_single = anchor_latent_single.to(local_rank, dtype=dtype)
                        variant_latent_single = variant_latent_single.to(local_rank, dtype=dtype)

                        # Multi-step denoising inference
                        B_single = 1
                        C_single, H_single, W_single = anchor_latent_single.shape[1], anchor_latent_single.shape[2], anchor_latent_single.shape[3]
                        num_patches_h_single = H_single // 2
                        num_patches_w_single = W_single // 2
                        num_patches_single = num_patches_h_single * num_patches_w_single
                        
                        # Start from pure noise
                        latent = torch.randn(
                            anchor_latent_single.shape, device=local_rank, dtype=dtype,
                            generator=vis_gen
                        )

                        # mu shifts the sigma schedule (use_dynamic_shifting=True).
                        scheduler.set_timesteps(num_inference_steps, device=local_rank, mu=1.0)
                        
                        # Pack variant latent (context) - stays constant throughout inference
                        variant_latent_packed_single = FluxPipeline._pack_latents(
                            variant_latent_single, B_single, C_single, H_single, W_single
                        )
                        
                        # IDs for a single sample, [context, target]
                        img_ids_context_single = prepare_latent_image_ids(num_patches_h_single, num_patches_w_single, local_rank, dtype)
                        img_ids_target_single = prepare_latent_image_ids(num_patches_h_single, num_patches_w_single, local_rank, dtype)
                        img_ids_single = torch.cat([img_ids_context_single, img_ids_target_single], dim=0)
                        txt_ids_single = txt_ids_null

                        # Prepare text embeddings for single sample
                        encoder_hidden_states_single = encoder_hidden_states_null.expand(B_single, -1, -1)
                        pooled_projections_single = pooled_projections_null.expand(B_single, -1)
                        
                        # Denoising loop
                        for t_idx, t in enumerate(scheduler.timesteps):
                            # Pack current latent (target)
                            latent_packed = FluxPipeline._pack_latents(latent, B_single, C_single, H_single, W_single)
                            
                            # Sequence concatenation [context, target]
                            model_input_single = torch.cat([variant_latent_packed_single, latent_packed], dim=1)  # [1, num_patches*2, 64]
                            
                            # Prepare timestep (FLUX expects timestep in [0, 1] range, divide by 1000)
                            timestep_single = (t.expand(B_single) / 1000.0).to(dtype)
                            
                            # Predict velocity
                            with torch.amp.autocast("cuda", enabled=args.amp, dtype=dtype):
                                # Prepare forward arguments
                                forward_kwargs = {
                                    "hidden_states": model_input_single,
                                    "timestep": timestep_single,
                                    "encoder_hidden_states": encoder_hidden_states_single,
                                    "pooled_projections": pooled_projections_single,
                                    "txt_ids": txt_ids_single,
                                    "img_ids": img_ids_single,
                                    "return_dict": False
                                }
                                # Add guidance only if model uses guidance embeddings
                                if use_guidance_embed:
                                    forward_kwargs["guidance"] = torch.full((B_single,), 3.5, device=local_rank, dtype=dtype)
                                
                                velocity_pred_full_single = transformer(**forward_kwargs)[0]  # [1, num_patches*2, 64]
                            
                            # KONTEXT: Extract target portion (second half) — target is last in sequence
                            velocity_pred_packed_single = velocity_pred_full_single[:, num_patches_single:, :]  # [1, num_patches, 64]
                            
                            # Unpack velocity prediction (transformer outputs packed format)
                            # _unpack_latents expects pixel space resolution (latent_size * vae_scale_factor)
                            image_height = H_single * vae_scale_factor
                            image_width = W_single * vae_scale_factor
                            velocity_pred_single = FluxPipeline._unpack_latents(
                                velocity_pred_packed_single, image_height, image_width, vae_scale_factor
                            )
                            
                            # Update latent using scheduler
                            latent = scheduler.step(velocity_pred_single, t, latent, return_dict=False)[0]
                        
                        # Decode both GT and predicted latents
                        shift = vae.config.shift_factor if args.use_shift_factor else 0.0
                        with torch.amp.autocast("cuda", enabled=args.amp, dtype=dtype):
                            anchor_srgb_gt = vae.decode(anchor_latent_single / vae.config.scaling_factor + shift).sample
                            anchor_srgb_pred = vae.decode(latent / vae.config.scaling_factor + shift).sample
                            cond_srgb = vae.decode(variant_latent_single / vae.config.scaling_factor + shift).sample
                        
                        # Visualize and save (outside autocast for better metric accuracy)
                        psnr, ssim_val, save_path = visualize_diffusion_results(
                            cond_srgb[0],
                            anchor_srgb_gt[0],
                            anchor_srgb_pred[0],
                            base_single,
                            epoch,
                            args.visualize_dir
                        )
                        
                        print(f"  {base_single}: PSNR={psnr:.2f}, SSIM={ssim_val:.4f}")
                        
                        # Log to wandb and TensorBoard
                        logger.log({
                            f"val/vis_psnr_{base_single}": psnr,
                            f"val/vis_ssim_{base_single}": ssim_val,
                            "val/epoch": epoch,
                            "val/step": global_step
                        }, step=global_step)
                        # Log image to wandb and TensorBoard
                        caption = f"Epoch {epoch} - {base_single} | PSNR: {psnr:.2f}, SSIM: {ssim_val:.4f}"
                        logger.log_image(f"val/viz_{base_single}", save_path, global_step, caption=caption)
                        # Free per-sample tensors to reduce peak memory
                        del latent, anchor_srgb_gt, anchor_srgb_pred, cond_srgb, variant_latent_packed_single, img_ids_single
            
            # Unload VAE to free memory
            if is_main_process():
                del vae
                torch.cuda.empty_cache()
                print("VAE unloaded.\n")

            transformer.train()
        
        # Save checkpoint
        if is_main_process() and epoch % args.save_every == 0:
            checkpoint_path = args.out_dir / f"kontext_lora_epoch{epoch}.pt"
            
            # Adapter tensors only; FLUX base weights come from the pretrained model.
            lora_state_dict = extract_adapter_state_dict(transformer.module)
            
            checkpoint = {
                "lora_weights": lora_state_dict,
                "lora_config": lora_config,
                "config": {
                    "model_id": args.model_id,
                    "architecture": "kontext_sequence_concat",
                    "prediction_type": "velocity",
                    "latent_channels": args.latent_channels,
                    "dtype": args.dtype,
                    "gradient_checkpointing": args.gradient_checkpointing,
                    "lora_rank": args.lora_rank,
                    "lora_alpha": args.lora_alpha,
                    "lora_target_modules": args.lora_target_modules,
                    "epoch": epoch
                },
                "optimizer": opt.state_dict(),
                "scheduler": scheduler_lr.state_dict() if scheduler_lr is not None else None,
                "global_step": global_step
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    # Save final checkpoint
    if is_main_process():
        final_path = args.out_dir / "kontext_lora_final.pt"
        
        lora_state_dict = extract_adapter_state_dict(transformer.module)
        
        checkpoint = {
            "lora_weights": lora_state_dict,
            "lora_config": lora_config,
            "config": {
                "model_id": args.model_id,
                "architecture": "kontext_sequence_concat",
                "prediction_type": "velocity",
                "latent_channels": args.latent_channels,
                "dtype": args.dtype,
                "gradient_checkpointing": args.gradient_checkpointing,
                "lora_rank": args.lora_rank,
                "lora_alpha": args.lora_alpha,
                "lora_target_modules": args.lora_target_modules,
                "epoch": args.max_epochs
            },
            "optimizer": opt.state_dict(),
            "scheduler": scheduler_lr.state_dict() if scheduler_lr is not None else None,
            "global_step": global_step
        }
        torch.save(checkpoint, final_path)
        print(f"Final checkpoint saved: {final_path}")
        print("Training complete!")
        logger.close()
    
    cleanup_ddp()


if __name__ == "__main__":
    main()

