"""Package linear camera RAW RGB into a Bayer-mosaiced .dng for S20 / S25U.

One image at a time.
"""
from __future__ import annotations

import os
import struct
from binascii import hexlify, unhexlify
from pathlib import Path
from typing import Literal

import numpy as np

from ._dng_helpers.dng_utils import RGB2bayer, update_wb_values, update_hex_image
from ._dng_helpers.ssdng import save_to_dng


# CROP_*_OFF: byte offset of the DefaultCropOrigin / DefaultCropSize value pair
# (two little-endian uint32) in the container. CROP_*_ORIG: the value shipped there.
_S20 = {
    "W": 4032, "H": 3024,
    "BL": 64, "WL": 1023,
    "WB_START": 50816,
    "IMAGE_START": 59288,
    "CROP_ORIGIN_OFF": 25104, "CROP_ORIGIN_ORIG": (8, 8),
    "CROP_SIZE_OFF": 25112, "CROP_SIZE_ORIG": (4016, 3008),
}
_S25U = {
    "W": 4000, "H": 3000,
    "WB_START": 2729 - 1,  # 2728
    "CROP_ORIGIN_OFF": 1060, "CROP_ORIGIN_ORIG": (0, 0),
    "CROP_SIZE_OFF": 1068, "CROP_SIZE_ORIG": (4000, 3000),
}


def _assert_container_layout(container_dng_path: Path, expected_image_start_hex: int) -> None:
    """Fail loudly if the container's image strip is not where we expect it.

    `IMAGE_START` / `WB_START` above are byte offsets baked into the container
    files as shipped.
    """
    import logging

    import tifffile

    expected = expected_image_start_hex // 2
    try:
        # Silence tifffile's warning on the S20 container's Orientation tag.
        tiff_log = logging.getLogger("tifffile")
        prev_level = tiff_log.level
        tiff_log.setLevel(logging.ERROR)
        try:
            with tifffile.TiffFile(str(container_dng_path)) as tf:
                actual = tf.pages[0].dataoffsets[0]
        finally:
            tiff_log.setLevel(prev_level)
    except Exception as exc:  # noqa: BLE001 - any parse failure is disqualifying
        raise ValueError(
            f"{container_dng_path.name}: cannot read the image strip offset "
            f"({type(exc).__name__}: {exc}). The packager is hard-coded for byte "
            f"{expected}; supply the container as shipped."
        ) from exc
    if actual != expected:
        raise ValueError(
            f"{container_dng_path.name}: image strip starts at byte {actual}, but the "
            f"packager is hard-coded for {expected}. The container's metadata has been "
            f"rewritten — regenerate it, or update IMAGE_START to {actual * 2}."
        )


def _center_pad(
    img: np.ndarray, canvas_h: int, canvas_w: int
) -> tuple[np.ndarray, int, int]:
    """Center `img` [h,w,3] into a zero canvas [canvas_h,canvas_w,3].

    Offsets are forced even so the Bayer G-R-B-G phase of the padded canvas is
    unchanged by the pad. Returns (canvas, left, top).
    """
    h, w = img.shape[:2]
    if h > canvas_h or w > canvas_w:
        raise ValueError(
            f"image {w}x{h} larger than sensor canvas {canvas_w}x{canvas_h}"
        )
    top = ((canvas_h - h) // 2) & ~1
    left = ((canvas_w - w) // 2) & ~1
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=img.dtype)
    canvas[top:top + h, left:left + w, :] = img
    return canvas, left, top


def _patch_default_crop(
    dng_path: Path, spec: dict, left: int, top: int, w: int, h: int
) -> None:
    """Point DefaultCropOrigin / DefaultCropSize at the padded window.

    Without this a reader renders the whole sensor canvas, of which only the
    centered window carries image. Length-preserving byte overwrite; the values
    currently at those offsets are checked first, so a container whose layout
    differs fails here instead of being patched at the wrong bytes.
    """
    with open(dng_path, "r+b") as f:
        for key, new in (("CROP_ORIGIN", (left, top)), ("CROP_SIZE", (w, h))):
            off = spec[f"{key}_OFF"]
            f.seek(off)
            found = struct.unpack("<II", f.read(8))
            if found != spec[f"{key}_ORIG"]:
                raise ValueError(
                    f"{dng_path.name}: expected {key} {spec[f'{key}_ORIG']} at byte "
                    f"{off}, found {found}. The container layout has changed."
                )
            f.seek(off)
            f.write(struct.pack("<II", *new))


def _package_s20(
    raw_rgb_uint16: np.ndarray,
    wb_vec: list[float],
    container_dng_path: Path,
    output_path: Path,
) -> None:
    _assert_container_layout(container_dng_path, _S20["IMAGE_START"])
    BL, WL = _S20["BL"], _S20["WL"]

    # Input is scaled to [BL, WL]. Normalize to [0,1] so the zero pad lands on BL.
    img01 = (raw_rgb_uint16.astype(np.float32) - BL) / max(WL - BL, 1)
    big, left, top = _center_pad(img01, _S20["H"], _S20["W"])

    bayer = RGB2bayer(big)
    bayer = bayer * (WL - BL) + BL
    bayer = np.clip(bayer, 0, WL)

    # update_* operate on a hex-encoded copy, not raw bytes.
    with open(container_dng_path, "rb") as f:
        myhex = bytearray(hexlify(f.read()))

    myhex = update_wb_values(myhex, wb_vec, _S20["WB_START"])
    myhex = update_hex_image(myhex, bayer, _S20["IMAGE_START"])

    with open(output_path, "wb") as f:
        f.write(unhexlify(myhex))

    h, w = raw_rgb_uint16.shape[:2]
    _patch_default_crop(output_path, _S20, left, top, w, h)


def _package_s25u(
    raw_rgb_uint16: np.ndarray,
    wb_vec: list[float],
    container_dng_path: Path,
    output_path: Path,
) -> None:
    # `xyz_to_camera_raw` scaled the RGB to the S25U camera range [0, 4095]
    # (matching the container's WhiteLevel tag). Center-pad the uint16 image into
    # the 4000x3000 sensor canvas (border 0 == BlackLevel).
    bayer_input, left, top = _center_pad(
        raw_rgb_uint16.astype(np.uint16), _S25U["H"], _S25U["W"]
    )

    # Pass 1: patch WB bytes in a hex-encoded copy of the container, write temp file.
    with open(container_dng_path, "rb") as f:
        myhex = bytearray(hexlify(f.read()))
    myhex = update_wb_values(myhex, wb_vec, _S25U["WB_START"])

    tmp = str(output_path) + ".tmp"
    with open(tmp, "wb") as f:
        f.write(unhexlify(myhex))

    try:
        # Pass 2: replace the JPEG-XL-compressed raw plane.
        save_to_dng(
            bayer_input,
            tmp,
            str(output_path),
            bitspersample=16,
            compression_type="jpeg",
            recon_compression="jpegxl",
        )
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

    h, w = raw_rgb_uint16.shape[:2]
    _patch_default_crop(output_path, _S25U, left, top, w, h)


def package_raw_to_dng(
    *,
    raw_rgb_uint16: np.ndarray,
    wb_vec: list[float],
    camera_id: Literal["S20", "S25U"],
    container_dng_path: "Path | str",
    output_path: "Path | str",
) -> None:
    """Write a camera RAW DNG for the given camera_id.

    raw_rgb_uint16: [H, W, 3] camera RGB as uint16.
    wb_vec: length-3 G-normalised illumination vector [R/G, 1, B/G].
    camera_id: 'S20' or 'S25U'.
    container_dng_path: SamsungS20FE.dng or S25U_ProRAW_main_cam.dng.
    output_path: where to write the resulting .dng.
    """
    container_dng_path = Path(container_dng_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if camera_id == "S20":
        _package_s20(raw_rgb_uint16, wb_vec, container_dng_path, output_path)
    elif camera_id == "S25U":
        _package_s25u(raw_rgb_uint16, wb_vec, container_dng_path, output_path)
    else:
        raise ValueError(f"unsupported camera_id: {camera_id!r}")
