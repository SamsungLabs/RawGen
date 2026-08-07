"""Load fine-tuned Kontext LoRA transformer + fine-tuned VAE decoder state-dict."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from peft.tuners.lora.config import LoraRuntimeConfig
from peft.utils.peft_types import PeftType


def load_lora_transformer(
    transformer: nn.Module,
    ckpt_path: Path | str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> nn.Module:
    """Load a Kontext LoRA `.pt` checkpoint produced by training/train_kontext_lora.py.

    Checkpoint dict must contain keys 'lora_config' (dict or LoraConfig) and
    'lora_weights' (state dict). The released checkpoint stores *only* the LoRA
    adapter tensors — the frozen FLUX base weights come from the pretrained
    model, so base keys are expected to be reported missing and are ignored.

    Raises if any adapter tensor fails to bind.
    """
    ckpt_path = Path(ckpt_path)
    # 'lora_config' is a pickled peft object, so it needs an explicit allowlist
    # under weights_only=True.
    with torch.serialization.safe_globals([LoraConfig, PeftType, LoraRuntimeConfig]):
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    if "lora_weights" not in ckpt or "lora_config" not in ckpt:
        raise ValueError(
            f"{ckpt_path}: missing 'lora_weights'/'lora_config' keys "
            f"(found: {list(ckpt.keys())})"
        )
    lcfg = ckpt["lora_config"]
    if isinstance(lcfg, dict):
        lcfg = LoraConfig(**lcfg)
    peft_transformer = get_peft_model(transformer, lcfg)
    missing, unexpected = peft_transformer.load_state_dict(ckpt["lora_weights"], strict=False)

    if unexpected:
        raise ValueError(
            f"{ckpt_path}: {len(unexpected)} tensor(s) in the checkpoint do not exist "
            f"in the LoRA-wrapped transformer, e.g. {unexpected[:3]}. "
            f"The checkpoint and the base model disagree."
        )
    missing_adapter = [k for k in missing if ".lora_" in k]
    if missing_adapter:
        raise ValueError(
            f"{ckpt_path}: {len(missing_adapter)} adapter tensor(s) were not provided "
            f"by the checkpoint, e.g. {missing_adapter[:3]}."
        )
    n_adapter = sum(1 for k in ckpt["lora_weights"] if ".lora_" in k)
    if n_adapter == 0:
        raise ValueError(f"{ckpt_path}: contains no LoRA adapter tensors.")
    print(f"[LoRA] applied {n_adapter} adapter tensors ({len(missing)} base keys from pretrained)")

    peft_transformer = peft_transformer.to(device=device, dtype=dtype)
    peft_transformer.eval()
    return peft_transformer


def load_finetuned_vae_decoder(vae: nn.Module, ckpt_path: Path | str) -> None:
    """Load a fine-tuned VAE decoder state dict from a single `.pt` file.

    Expects a `.pt` file containing a dict with key ``"decoder"`` whose value
    is a full `vae.decoder.state_dict()`, as produced by the full-fine-tune
    mode of `training/finetune_vae_decoder.py`. Mutates ``vae.decoder`` in place.
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        raise ValueError(f"{ckpt_path}: expected a .pt file")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    if not isinstance(ckpt, dict) or "decoder" not in ckpt:
        raise ValueError(
            f"{ckpt_path}: expected dict with key 'decoder' "
            f"(found: {list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt).__name__})"
        )
    missing, unexpected = vae.decoder.load_state_dict(ckpt["decoder"], strict=False)
    if missing or unexpected:
        raise ValueError(
            f"{ckpt_path}: decoder state dict does not match this VAE — "
            f"missing={len(missing)} unexpected={len(unexpected)} "
            f"(missing e.g. {missing[:3]}, unexpected e.g. {unexpected[:3]})"
        )
