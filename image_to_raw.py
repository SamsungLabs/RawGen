#!/usr/bin/env python3
"""Image-to-RAW: sRGB image(s) → cross-camera RAW PNG (9 NUS) + Samsung S20/S25U DNG.

Usage:
    python image_to_raw.py \\
        --input samples/inputs/ \\
        --output-dir out/image_to_raw/ \\
        --num-illum 2 \\
        --num-steps 30 \\
        --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
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
from rawgen_inference.io import (
    save_png16, save_jpg, srgb01_to_m11, m11_to_01, srgb_gamma_encode,
)
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True,
                   help="An sRGB image file or a directory of them.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--transformer-ckpt", type=Path,
                   default=Path("checkpoints/kontext_lora.pt"))
    p.add_argument("--vae-decoder-ckpt", type=Path,
                   default=Path("checkpoints/vae_decoder_xyz.pt"))
    p.add_argument("--illum-json", type=Path,
                   default=Path("samples/illuminations.json"))
    p.add_argument("--dng-dir", type=Path,
                   default=Path("samples/dng_profiles"))
    p.add_argument("--num-illum", type=int, default=2)
    p.add_argument("--num-steps", type=int, default=30)
    p.add_argument("--kontext-guidance", type=float, default=3.5,
                   help="Guidance scale for the Kontext variant->anchor denoise "
                        "(the trained default is 3.5).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model-id", type=str,
                   default="black-forest-labs/FLUX.1-Kontext-dev")
    p.add_argument("--save-jpg", action="store_true")
    return p.parse_args()


def _list_inputs(p: Path) -> list[Path]:
    if p.is_file():
        return [p]
    return sorted(
        q for q in p.rglob("*")
        if q.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    )


def _load_srgb_as_m11(path: Path, device, dtype) -> torch.Tensor:
    # IMREAD_COLOR forces 3 channels (gray and RGBA would break cvtColor),
    # IMREAD_ANYDEPTH keeps 16-bit inputs at 16 bits.
    img = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
    if img is None:
        raise IOError(f"could not read {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_LANCZOS4)
    img = img.astype(np.float32) / (65535.0 if img.dtype == np.uint16 else 255.0)
    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    t = srgb01_to_m11(t).to(device=device, dtype=dtype)
    return t


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    rng = random.Random(args.seed)
    gen = torch.Generator(device=device).manual_seed(args.seed)

    print(f"Loading {args.model_id} ...")
    pipe = FluxPipeline.from_pretrained(args.model_id, torch_dtype=dtype, low_cpu_mem_usage=True)
    pipe = pipe.to(device)
    vae = pipe.vae
    vae_latent_sf = pipe.vae.config.scaling_factor

    transformer = load_lora_transformer(
        pipe.transformer, args.transformer_ckpt, device=device, dtype=dtype,
    )
    load_finetuned_vae_decoder(vae, args.vae_decoder_ckpt)
    vae = vae.to(device=device, dtype=dtype).eval()

    enc_null, pooled_null = create_null_text_embeddings(
        pipe.text_encoder, pipe.text_encoder_2,
        pipe.tokenizer, pipe.tokenizer_2,
        device, dtype,
    )
    txt_ids_null = prepare_text_ids(enc_null, device=device, dtype=dtype)

    print("Loading illumination JSON + camera profiles ...")
    illum_all = load_illumination_data(args.illum_json)
    nus_profiles = load_camera_profiles(args.dng_dir, NUS_CAMERAS)
    dng_profiles = load_camera_profiles(args.dng_dir, DNG_CAMERAS)
    dng_container = {cam: args.dng_dir / name for cam, name in DNG_CONTAINER_NAME.items()}

    inputs = _list_inputs(args.input)
    if not inputs:
        raise SystemExit(f"no input images under {args.input}")

    out = args.output_dir
    (out / "xyz").mkdir(parents=True, exist_ok=True)
    (out / "png").mkdir(parents=True, exist_ok=True)
    (out / "dng").mkdir(parents=True, exist_ok=True)

    metadata = {
        "model_id": args.model_id,
        "transformer_ckpt": str(args.transformer_ckpt),
        "vae_decoder_ckpt": str(args.vae_decoder_ckpt),
        "seed": args.seed, "num_inference_steps": args.num_steps,
        "kontext_guidance": args.kontext_guidance,
        "num_illum": args.num_illum, "size": SIZE,
        "samples": [],
    }

    for in_path in tqdm(inputs, desc="image_to_raw"):
        basename = in_path.stem
        x_m11 = _load_srgb_as_m11(in_path, device, dtype)

        with torch.no_grad():
            # .mode() matches the latent cache the LoRA was trained on.
            variant_latent = vae.encode(x_m11).latent_dist.mode() * vae_latent_sf

        anchor_latent = denoise_variant_to_anchor(
            transformer=transformer, scheduler=pipe.scheduler,
            variant_latent=variant_latent, num_steps=args.num_steps,
            enc_null=enc_null, pooled_null=pooled_null, txt_ids_null=txt_ids_null,
            guidance_scale=args.kontext_guidance,
            generator=gen, device=device, dtype=dtype,
        )

        with torch.no_grad():
            decoded = vae.decode(anchor_latent / vae_latent_sf).sample
        xyz_01 = m11_to_01(decoded)[0].float().cpu().numpy().transpose(1, 2, 0)

        save_png16(xyz_01, out / "xyz" / f"{basename}_xyz.png")

        sampled = sample_illuminations(illum_all, args.num_illum, rng)
        sample_meta = {"basename": basename, "illuminations": []}

        for ii, illum in enumerate(sampled):
            illum_meta = {"index": ii, "illum_xyz": illum["illum_xyz"],
                          "illum_cct": illum["illum_cct"], "gt_illum_per_camera": {}}

            for cam in NUS_CAMERAS:
                raw_u16, gt = xyz_to_camera_raw(
                    xyz_01, illum, nus_profiles[cam], gamma_decode=True,
                )
                bl = nus_profiles[cam]["black_level"]
                wl = nus_profiles[cam]["white_level"]
                raw_norm = np.clip(
                    (raw_u16.astype(np.float32) - bl) / max(wl - bl, 1), 0, 1,
                )
                save_png16(
                    raw_norm, out / "png" / f"{basename}_{cam}_illum{ii}.png",
                    encode=True,
                )
                if args.save_jpg:
                    save_jpg(
                        srgb_gamma_encode(raw_norm),
                        out / "png" / f"{basename}_{cam}_illum{ii}.jpg",
                    )
                illum_meta["gt_illum_per_camera"][cam] = gt

            for cam in DNG_CAMERAS:
                raw_u16, gt = xyz_to_camera_raw(
                    xyz_01, illum, dng_profiles[cam], gamma_decode=True,
                )
                package_raw_to_dng(
                    raw_rgb_uint16=raw_u16,
                    wb_vec=gt,
                    camera_id=cam,
                    container_dng_path=dng_container[cam],
                    output_path=out / "dng" / f"{basename}_{cam}_illum{ii}.dng",
                )
                illum_meta["gt_illum_per_camera"][cam] = gt

            sample_meta["illuminations"].append(illum_meta)
        metadata["samples"].append(sample_meta)

    with open(out / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Done -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
