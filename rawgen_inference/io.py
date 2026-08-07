"""sRGB / linear / gamma helpers, range converters, PNG / JPG savers."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import colour


def m11_to_01(t: torch.Tensor) -> torch.Tensor:
    """Map [-1, 1] → [0, 1] (input clamped)."""
    return (t.clamp(-1.0, 1.0) + 1.0) / 2.0


def srgb01_to_m11(t: torch.Tensor) -> torch.Tensor:
    """Map [0, 1] → [-1, 1]."""
    return t * 2.0 - 1.0


def srgb_gamma_encode(x: np.ndarray) -> np.ndarray:
    """Apply sRGB OETF (linear → gamma-encoded), clamp to [0,1], float32."""
    return colour.cctf_encoding(np.clip(x, 0.0, 1.0), function="sRGB").astype(np.float32)


def srgb_gamma_decode(x: np.ndarray) -> np.ndarray:
    """Apply sRGB EOTF (gamma-encoded → linear), clamp to [0,1], float32."""
    return colour.cctf_decoding(np.clip(x, 0.0, 1.0), function="sRGB").astype(np.float32)


def save_png16(img_float01: np.ndarray, path: Path, *, encode: bool = False) -> None:
    """Save HWC [0,1] float image as 16-bit PNG. encode=True applies the sRGB OETF first."""
    img = np.clip(img_float01, 0.0, 1.0).astype(np.float32)
    if encode:
        img = srgb_gamma_encode(img)
    out = np.rint(np.clip(img, 0.0, 1.0) * 65535.0).astype(np.uint16)
    cv2.imwrite(str(path), out[:, :, ::-1])  # RGB→BGR for cv2


def save_jpg(img_float01: np.ndarray, path: Path, *, quality: int = 95) -> None:
    """Save HWC [0,1] float image as 8-bit JPG. No gamma conversion."""
    out = np.rint(np.clip(img_float01, 0.0, 1.0) * 255.0).astype(np.uint8)
    cv2.imwrite(str(path), out[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
