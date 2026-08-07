"""FLUX.1-Kontext helpers and the variant->anchor denoise loop.

Single-GPU, bfloat16.
"""
from __future__ import annotations

import torch
from diffusers import FluxPipeline


def create_null_text_embeddings(
    text_encoder, text_encoder_2, tokenizer, tokenizer_2,
    device: torch.device, dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build encoder + pooled embeddings for the empty prompt ''."""
    prompt = ""

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    text_inputs_2 = tokenizer_2(
        prompt,
        padding="max_length",
        max_length=tokenizer_2.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        # CLIP (text_encoder) — used for pooled projection only
        prompt_embeds = text_encoder(
            text_inputs.input_ids.to(device),
            output_hidden_states=True,
        )
        pooled_projections = prompt_embeds.pooler_output

        # T5 (text_encoder_2) — used for encoder hidden states
        prompt_embeds_2 = text_encoder_2(
            text_inputs_2.input_ids.to(device),
            output_hidden_states=True,
        )
        encoder_hidden_states = prompt_embeds_2.hidden_states[-1]

    return encoder_hidden_states.to(dtype), pooled_projections.to(dtype)


def prepare_latent_image_ids(
    num_patches_h: int, num_patches_w: int,
    device, dtype: torch.dtype,
) -> torch.Tensor:
    """Build the (N, 3) latent image ids tensor used by FLUX RoPE."""
    latent_image_ids = torch.zeros(num_patches_h, num_patches_w, 3, device=device, dtype=dtype)
    latent_image_ids[..., 1] = torch.arange(num_patches_h, device=device, dtype=dtype)[:, None]
    latent_image_ids[..., 2] = torch.arange(num_patches_w, device=device, dtype=dtype)[None, :]
    return latent_image_ids.reshape(-1, 3)


def prepare_text_ids(encoder_hidden_states: torch.Tensor, device, dtype) -> torch.Tensor:
    """(seq_len, 3) text ids: column 0 = arange(seq_len), columns 1-2 = 0."""
    if encoder_hidden_states.ndim == 3:
        seq_len = encoder_hidden_states.shape[1]
    else:
        seq_len = encoder_hidden_states.shape[0]
    text_ids = torch.zeros(seq_len, 3, device=device, dtype=dtype)
    text_ids[:, 0] = torch.arange(seq_len, device=device, dtype=dtype)
    return text_ids


@torch.no_grad()
def denoise_variant_to_anchor(
    *,
    transformer,
    scheduler,
    variant_latent: torch.Tensor,   # [B, C, H, W]
    num_steps: int,
    enc_null: torch.Tensor,         # [1, seq_t, dim_t]
    pooled_null: torch.Tensor,      # [1, dim_p]
    txt_ids_null: torch.Tensor,     # [seq_t, 3]
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
    guidance_scale: float = 3.5,
) -> torch.Tensor:
    """Single-GPU bfloat16 denoise loop. Returns the anchor latent."""
    B, C, H, W = variant_latent.shape
    latent = torch.randn(B, C, H, W, generator=generator, device=device, dtype=dtype)

    if hasattr(scheduler.config, "use_dynamic_shifting") and scheduler.config.use_dynamic_shifting:
        scheduler.set_timesteps(num_steps, device=device, mu=1.0)
    else:
        scheduler.set_timesteps(num_steps, device=device)

    nph, npw = H // 2, W // 2
    num_patches = nph * npw

    ctx_packed = FluxPipeline._pack_latents(variant_latent, B, C, H, W)
    img_ids_ctx = prepare_latent_image_ids(nph, npw, device, dtype)
    img_ids_tgt = prepare_latent_image_ids(nph, npw, device, dtype)
    img_ids = torch.cat([img_ids_ctx, img_ids_tgt], dim=0)

    enc_batch = enc_null.expand(B, -1, -1)
    pooled_batch = pooled_null.expand(B, -1)

    cfg = transformer.module.config if hasattr(transformer, "module") else transformer.config
    use_guidance_embed = bool(getattr(cfg, "guidance_embeds", False))

    vae_spatial_sf = 8  # FLUX VAE scale factor

    for t in scheduler.timesteps:
        lat_packed = FluxPipeline._pack_latents(latent, B, C, H, W)
        model_input = torch.cat([ctx_packed, lat_packed], dim=1)
        ts = (t.expand(B) / 1000.0).to(dtype)

        fwd = dict(
            hidden_states=model_input,
            timestep=ts,
            encoder_hidden_states=enc_batch,
            pooled_projections=pooled_batch,
            txt_ids=txt_ids_null,
            img_ids=img_ids,
            return_dict=False,
        )
        if use_guidance_embed:
            fwd["guidance"] = torch.full((B,), guidance_scale, device=device, dtype=dtype)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda"), dtype=dtype):
            vel_full = transformer(**fwd)[0]

        vel_packed = vel_full[:, num_patches:, :]
        vel = FluxPipeline._unpack_latents(
            vel_packed, H * vae_spatial_sf, W * vae_spatial_sf, vae_spatial_sf
        )
        latent = scheduler.step(vel, t, latent, return_dict=False)[0]

    return latent
