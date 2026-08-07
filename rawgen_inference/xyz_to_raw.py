"""Linear XYZ → target-camera RAW RGB (uint16). Adapted from
the RawGen pipeline."""
from __future__ import annotations

import numpy as np
import colour


# EXIF LightSource code → CCT in K, as used by the DNG CalibrationIlluminant
# tags. https://exiftool.org/TagNames/EXIF.html#LightSource
_LIGHT_SOURCE_CCT = {
    17: 2856.0,  # Standard light A
    18: 4874.0,  # Standard light B
    19: 6774.0,  # Standard light C
    20: 5503.0,  # D55
    21: 6504.0,  # D65
    22: 7504.0,  # D75
    23: 5003.0,  # D50
}


def _calibration_cct(illuminant, default: float) -> float:
    """CCT of a CalibrationIlluminant tag, or `default` if it is absent."""
    try:
        return _LIGHT_SOURCE_CCT[int(illuminant)]
    except (TypeError, ValueError, KeyError):
        return default


def interpolate_calibration_matrix(
    profile: dict, color_temp: float
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Linear blend between (CM1, FM1) and (CM2, FM2) by reciprocal CCT.

    Which of the two calibrations is the warm one is per-camera — the NUS
    bodies tag illuminant 1 as A and 2 as D65, the Samsung phones do the
    reverse — so the blend follows the container's CalibrationIlluminant tags.
    """
    cm1 = np.array(profile["ColorMatrix1"], dtype=np.float64)
    cm2 = np.array(profile["ColorMatrix2"], dtype=np.float64)
    fm1 = np.array(profile["ForwardMatrix1"], dtype=np.float64)
    fm2 = np.array(profile["ForwardMatrix2"], dtype=np.float64)
    t1 = _calibration_cct(profile.get("CalibrationIlluminant1"), 6504.0)
    t2 = _calibration_cct(profile.get("CalibrationIlluminant2"), 2856.0)
    if t1 == t2:
        g = 1.0
    else:
        g = (1.0 / color_temp - 1.0 / t2) / (1.0 / t1 - 1.0 / t2)
    g = float(np.clip(g, 0.0, 1.0))
    cm = g * cm1 + (1.0 - g) * cm2
    fm = g * fm1 + (1.0 - g) * fm2
    return cm, fm, int(profile["black_level"]), int(profile["white_level"])


def xyz_to_camera_raw(
    xyz_image: np.ndarray,
    illum_entry: dict,
    camera_profile: dict,
    *,
    gamma_decode: bool = True,
) -> tuple[np.ndarray, list[float]]:
    """Convert linear (or gamma-encoded) XYZ image to target-camera RAW RGB.

    Returns (raw_rgb_uint16 [H,W,3], illum_rgb_g_normalized [R/G, 1, B/G]).
    """
    img = xyz_image
    if gamma_decode:
        img = colour.cctf_decoding(img, function="sRGB")
    img = img.astype(np.float64)

    illum_xyz = np.asarray(illum_entry["illum_xyz"], dtype=np.float64)
    illum_cct = float(illum_entry["illum_cct"])

    cm, fm, black_level, white_level = interpolate_calibration_matrix(
        camera_profile, illum_cct
    )

    # XYZ → white-balanced camera RGB
    wb_rgb = np.clip(img @ np.linalg.inv(fm).T, 0.0, 1.0)

    # Illum CCT → camera RGB, G-normalized
    illum_rgb = cm @ illum_xyz
    gnorm = illum_rgb[1] if illum_rgb[1] != 0 else 1e-6
    illum_rgb = illum_rgb / gnorm

    # Apply illuminant, clamp, scale to camera range
    raw = np.clip(wb_rgb * illum_rgb[None, None, :], 0.0, 1.0)
    raw_scaled = raw * (white_level - black_level) + black_level
    raw_uint16 = np.clip(raw_scaled, 0.0, 65535.0).astype(np.uint16)

    return raw_uint16, illum_rgb.tolist()
