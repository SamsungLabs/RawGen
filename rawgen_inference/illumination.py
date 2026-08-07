"""Illumination JSON loading + sampling, target-camera DNG profile loading."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import rawpy
from exiftool import ExifToolHelper


def load_illumination_data(json_path: Path | str) -> list[dict]:
    """Load the cross-camera illumination JSON file.

    Expects a JSON array of dicts each with keys ``illum_xyz`` (3-vector) and
    ``illum_cct`` (float).
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{json_path}: expected a JSON array of illumination entries")
    return data


def sample_illuminations(data: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Sample without replacement when possible, with replacement otherwise.

    Uses the supplied Random instance so the choice is reproducible.
    """
    if n <= len(data):
        return rng.sample(data, n)
    return [rng.choice(data) for _ in range(n)]


def _exif_matrix_to_array(s: str) -> np.ndarray:
    vals = np.array([float(x) for x in s.split(" ")], dtype=np.float64)
    if vals.size == 9:
        return vals.reshape(3, 3)
    if vals.size == 3:
        return np.diag(vals)
    raise ValueError(f"unexpected matrix length {vals.size}")


def _find_dng(camera: str, dng_dir: Path) -> Path:
    matches = sorted(p for p in dng_dir.glob("*.dng") if camera.lower() in p.name.lower())
    if not matches:
        raise FileNotFoundError(f"no DNG matching '{camera}' under {dng_dir}")
    if len(matches) > 1:
        raise ValueError(
            f"'{camera}' matches {len(matches)} DNGs under {dng_dir}: "
            f"{[p.name for p in matches]}. Keep exactly one profile DNG per camera."
        )
    return matches[0]


def _extract_profile(dng_path: Path) -> dict:
    """Return {ColorMatrix1/2, ForwardMatrix1/2, CalibrationIlluminant1/2, black_level, white_level}."""
    if "S25U" in dng_path.name:
        # Samsung ProRAW is JPEG-XL-compressed DNG 1.7; LibRaw cannot open it.
        # Values from the container's IFD0 tags.
        black_level, white_level = 0, 4095
    else:
        raw = rawpy.imread(str(dng_path))
        black_level = int(raw.black_level_per_channel[0])
        white_level = int(raw.white_level)
        del raw
    with ExifToolHelper() as et:
        m = et.get_metadata(str(dng_path))[0]
    return {
        "ColorMatrix1": _exif_matrix_to_array(m["EXIF:ColorMatrix1"]).tolist(),
        "ColorMatrix2": _exif_matrix_to_array(m["EXIF:ColorMatrix2"]).tolist(),
        "ForwardMatrix1": _exif_matrix_to_array(m["EXIF:ForwardMatrix1"]).tolist(),
        "ForwardMatrix2": _exif_matrix_to_array(m["EXIF:ForwardMatrix2"]).tolist(),
        "CalibrationIlluminant1": m.get("EXIF:CalibrationIlluminant1"),
        "CalibrationIlluminant2": m.get("EXIF:CalibrationIlluminant2"),
        "black_level": black_level,
        "white_level": white_level,
    }


def load_camera_profiles(dng_dir: Path | str, cameras: Iterable[str]) -> dict[str, dict]:
    """Load DNG color-science profiles for each named camera.

    Returns a dict mapping camera name → {ColorMatrix1, ColorMatrix2,
    ForwardMatrix1, ForwardMatrix2, CalibrationIlluminant1,
    CalibrationIlluminant2, black_level, white_level}.
    """
    dng_dir = Path(dng_dir)
    if not dng_dir.is_dir():
        raise FileNotFoundError(f"DNG dir not found: {dng_dir}")
    return {cam: _extract_profile(_find_dng(cam, dng_dir)) for cam in cameras}
