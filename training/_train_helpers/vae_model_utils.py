"""
Utilities for loading VAE models for training.

This module provides functions to load the FLUX VAE (FLUX.1-dev / FLUX.1-Kontext-dev)
for fine-tuning purposes.
"""

import torch
import torch.nn as nn
from diffusers import FluxPipeline


def load_vae_for_training(
    model_id: str,
    device: torch.device,
    freeze_encoder: bool = True,
    dtype: torch.dtype = torch.float32
) -> nn.Module:
    """
    Load VAE model for training with encoder freezing option.

    Supports FLUX models:
    - FLUX-dev: black-forest-labs/FLUX.1-dev
    - FLUX-Kontext-dev: black-forest-labs/FLUX.1-Kontext-dev

    Args:
        model_id: Model ID string (e.g., "black-forest-labs/FLUX.1-dev")
        device: Target device for the model
        freeze_encoder: If True, freeze encoder parameters and make decoder trainable
        dtype: Data type for the model (default: float32 for training stability)

    Returns:
        VAE model with encoder frozen (if requested) and decoder trainable
    """
    model_id_lower = model_id.lower()

    # Detect model type and load accordingly
    if "flux" in model_id_lower:
        # FLUX models: Load full pipeline, extract VAE
        if torch.cuda.is_available():
            print(f"Loading FLUX pipeline to extract VAE: {model_id}")
        pipeline = FluxPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=None
        )
        vae = pipeline.vae.to(device)
        # Free up memory by deleting the pipeline
        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    else:
        raise ValueError(f"Unknown model type: {model_id}. Supported models: FLUX-dev, FLUX-Kontext-dev")
    
    # Set VAE to eval mode initially
    vae.eval()
    
    # Freeze encoder and make decoder trainable if requested
    if freeze_encoder:
        for p in vae.encoder.parameters():
            p.requires_grad = False
        for p in vae.decoder.parameters():
            p.requires_grad = True
    
    return vae

