#!/usr/bin/env python3
"""Text-to-RAW: prompt → FLUX.1-dev T2I latent → Kontext denoise → XYZ → 9 PNG + 2 DNG.

Usage:
    python text_to_raw.py \\
        --prompts-file samples/prompts.txt \\
        --output-dir out/text_to_raw/ \\
        --num-illum 2
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import random
import re
from pathlib import Path

import numpy as np
import torch
from diffusers import FluxPipeline
from tqdm import tqdm

from rawgen_inference.checkpoints import load_lora_transformer, load_finetuned_vae_decoder
from rawgen_inference.dng_packager import package_raw_to_dng
from rawgen_inference.flux_kontext import (
    create_null_text_embeddings, prepare_text_ids, denoise_variant_to_anchor,
)
from rawgen_inference.illumination import (
    load_illumination_data, sample_illuminations, load_camera_profiles,
)
from rawgen_inference.io import save_png16, m11_to_01
from rawgen_inference.xyz_to_raw import xyz_to_camera_raw


NUS_CAMERAS = [
    "Canon1DsMkIII", "Canon600D", "NikonD40", "NikonD5200",
    "OlympusEPL6", "PanasonicGX1", "SamsungNX2000", "SonyA57", "FujifilmXM1",
]
DNG_CAMERAS = ["S20", "S25U"]
DNG_CONTAINER_NAME = {
    "S20":  "SamsungS20FE.dng",
    "S25U": "S25U_ProRAW_main_cam.dng",
}

# Training and evaluation only ever ran at 1024**2. Other resolutions silently
# shift the sampler schedule and cross the VAE tiling threshold.
SIZE = 1024


def slugify(text: str, max_len: int = 64) -> str:
    t = re.sub(r"[^a-z0-9\-_. ]+", "", text.lower()).strip().replace(" ", "_")
    return t[:max_len] or "prompt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--prompts-file", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--transformer-ckpt", type=Path,
                   default=Path("checkpoints/kontext_lora.pt"))
    p.add_argument("--vae-decoder-ckpt", type=Path,
                   default=Path("checkpoints/vae_decoder_xyz.pt"))
    p.add_argument("--illum-json", type=Path, default=Path("samples/illuminations.json"))
    p.add_argument("--dng-dir", type=Path, default=Path("samples/dng_profiles"))
    p.add_argument("--num-illum", type=int, default=2)
    p.add_argument("--t2i-steps", type=int, default=30)
    p.add_argument("--kontext-steps", type=int, default=30)
    p.add_argument("--guidance-scale", type=float, default=3.5,
                   help="Guidance for the FLUX.1-dev T2I stage only.")
    p.add_argument("--kontext-guidance", type=float, default=3.5,
                   help="Guidance embed for the Kontext denoise stage "
                        "(the trained default is 3.5; decoupled from --guidance-scale).")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--t2i-model-id", type=str, default="black-forest-labs/FLUX.1-dev")
    p.add_argument("--kontext-model-id", type=str, default="black-forest-labs/FLUX.1-Kontext-dev")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    rng = random.Random(args.seed)
    gen = torch.Generator(device=device).manual_seed(args.seed)

    prompts = [ln.strip() for ln in args.prompts_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not prompts:
        raise SystemExit(f"no prompts in {args.prompts_file}")

    out = args.output_dir
    (out / "xyz").mkdir(parents=True, exist_ok=True)
    (out / "png").mkdir(parents=True, exist_ok=True)
    (out / "dng").mkdir(parents=True, exist_ok=True)
    (out / "text_prompt").mkdir(parents=True, exist_ok=True)

    # === Stage 1: T2I → packed latents ===
    print(f"Stage 1: loading {args.t2i_model_id}")
    pipe_t2i = FluxPipeline.from_pretrained(args.t2i_model_id, torch_dtype=dtype, low_cpu_mem_usage=True).to(device)
    vae_spatial_sf = pipe_t2i.vae_scale_factor
    # FLUX hands back latents in its own (z - shift) * scale convention, while the
    # LoRA was trained on caches built as mode() * scale. Undo the shift so both
    # entry points feed the adapter the same normalisation.
    latent_shift = pipe_t2i.vae.config.shift_factor * pipe_t2i.vae.config.scaling_factor

    all_t2i_latents: list[torch.Tensor] = []
    B = max(1, args.batch_size)
    for bi in tqdm(range(math.ceil(len(prompts) / B)), desc="[Stage 1] T2I"):
        batch = prompts[bi * B:(bi + 1) * B]
        with torch.no_grad():
            packed = pipe_t2i(
                batch, height=SIZE, width=SIZE,
                num_inference_steps=args.t2i_steps,
                guidance_scale=args.guidance_scale,
                generator=gen, output_type="latent", return_dict=True,
            ).images
            lat = FluxPipeline._unpack_latents(packed, SIZE, SIZE, vae_spatial_sf).to(device, dtype=dtype)
            lat = lat + latent_shift
        all_t2i_latents.append(lat.cpu())
    all_t2i_latents = torch.cat(all_t2i_latents, dim=0)

    for i, prompt in enumerate(prompts):
        (out / "text_prompt" / f"{i:04d}_{slugify(prompt)}.txt").write_text(prompt, encoding="utf-8")

    del pipe_t2i
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # === Stage 2: Kontext denoise → XYZ → 9 PNG + 2 DNG ===
    print(f"Stage 2: loading {args.kontext_model_id}")
    pipe = FluxPipeline.from_pretrained(args.kontext_model_id, torch_dtype=dtype, low_cpu_mem_usage=True).to(device)
    vae = pipe.vae
    transformer = load_lora_transformer(pipe.transformer, args.transformer_ckpt, device=device, dtype=dtype)
    load_finetuned_vae_decoder(vae, args.vae_decoder_ckpt)
    vae = vae.to(device=device, dtype=dtype).eval()

    enc_null, pooled_null = create_null_text_embeddings(
        pipe.text_encoder, pipe.text_encoder_2, pipe.tokenizer, pipe.tokenizer_2,
        device, dtype,
    )
    txt_ids_null = prepare_text_ids(enc_null, device=device, dtype=dtype)

    illum_all = load_illumination_data(args.illum_json)
    nus_profiles = load_camera_profiles(args.dng_dir, NUS_CAMERAS)
    dng_profiles = load_camera_profiles(args.dng_dir, DNG_CAMERAS)
    dng_container = {cam: args.dng_dir / name for cam, name in DNG_CONTAINER_NAME.items()}

    metadata = {
        "t2i_model_id": args.t2i_model_id,
        "kontext_model_id": args.kontext_model_id,
        "transformer_ckpt": str(args.transformer_ckpt),
        "vae_decoder_ckpt": str(args.vae_decoder_ckpt),
        "seed": args.seed, "size": SIZE,
        "t2i_steps": args.t2i_steps, "kontext_steps": args.kontext_steps,
        "num_illum": args.num_illum,
        "samples": [],
    }

    # Each batch is written out before the next one is decoded, so peak memory is
    # O(batch) rather than O(len(prompts)). `rng` is still consumed in prompt
    # order, which is what keeps the illuminant draws reproducible.
    for bi in tqdm(range(math.ceil(len(prompts) / B)), desc="[Stage 2] denoise→RAW"):
        bs, be = bi * B, min((bi + 1) * B, len(prompts))
        ctx = all_t2i_latents[bs:be].to(device, dtype=dtype)
        anchor = denoise_variant_to_anchor(
            transformer=transformer, scheduler=pipe.scheduler,
            variant_latent=ctx, num_steps=args.kontext_steps,
            enc_null=enc_null, pooled_null=pooled_null, txt_ids_null=txt_ids_null,
            generator=gen, device=device, dtype=dtype,
            guidance_scale=args.kontext_guidance,
        )
        with torch.no_grad():
            decoded = vae.decode(anchor / pipe.vae.config.scaling_factor).sample
        xyz_batch = m11_to_01(decoded).float().cpu()

        for k in range(be - bs):
            i = bs + k
            prompt = prompts[i]
            slug = slugify(prompt)
            xyz_hwc = xyz_batch[k].numpy().transpose(1, 2, 0).astype(np.float32)
            save_png16(xyz_hwc, out / "xyz" / f"{i:04d}_{slug}_xyz.png")

            sampled = sample_illuminations(illum_all, args.num_illum, rng)
            sample_meta = {"index": i, "prompt": prompt, "illuminations": []}

            for ii, illum in enumerate(sampled):
                illum_meta = {"index": ii, "illum_xyz": illum["illum_xyz"],
                              "illum_cct": illum["illum_cct"],
                              "gt_illum_per_camera": {}}

                for cam in NUS_CAMERAS:
                    raw_u16, gt = xyz_to_camera_raw(xyz_hwc, illum, nus_profiles[cam], gamma_decode=True)
                    bl = nus_profiles[cam]["black_level"]
                    wl = nus_profiles[cam]["white_level"]
                    raw_norm = np.clip((raw_u16.astype(np.float32) - bl) / max(wl - bl, 1), 0, 1)
                    save_png16(raw_norm,
                               out / "png" / f"{i:04d}_{slug}_{cam}_illum{ii}.png",
                               encode=True)
                    illum_meta["gt_illum_per_camera"][cam] = gt

                for cam in DNG_CAMERAS:
                    raw_u16, gt = xyz_to_camera_raw(xyz_hwc, illum, dng_profiles[cam], gamma_decode=True)
                    package_raw_to_dng(
                        raw_rgb_uint16=raw_u16, wb_vec=gt, camera_id=cam,
                        container_dng_path=dng_container[cam],
                        output_path=out / "dng" / f"{i:04d}_{slug}_{cam}_illum{ii}.dng",
                    )
                    illum_meta["gt_illum_per_camera"][cam] = gt

                sample_meta["illuminations"].append(illum_meta)
            metadata["samples"].append(sample_meta)

    del pipe, transformer, vae, all_t2i_latents
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    with open(out / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Done -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
