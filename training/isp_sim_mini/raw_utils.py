"""
Vendored from ISP_Sim_mini (src/raw_utils.py).

Author(s):
Abdelrahman Abdelhamed (a.abdelhamed@samsung.com)

Upstream: https://github.com/AbdoKamel/simple-camera-pipeline — MIT License,
Copyright (c) [2019] [Abdelrahman Abdelhamed]. The MIT notice is retained per
its terms. Modifications for RawGen are CC BY-NC-SA 4.0; see ../../LICENSE.md.
"""


import numpy as np


def RGGB2Bayer(im, _cfa_pattern=[0,1,1,2]):
    # add 1 to the second-green and the blue channel (e.g., [0, 1, 1, 2] will be [0, 1, 2, 3])
    _cfa_pattern_arr=np.asarray(_cfa_pattern)
    _cfa_pattern_arr[_cfa_pattern_arr == 2] += 1
    _cfa_pattern_arr[2:][_cfa_pattern_arr[2:] == 1] += 1
    # convert RGGB stacked image to one channel Bayer
    bayer = np.zeros((im.shape[0] * 2, im.shape[1] * 2))
    bayer[0::2, 0::2] = im[:, :, _cfa_pattern_arr[0]]
    bayer[0::2, 1::2] = im[:, :, _cfa_pattern_arr[1]]
    bayer[1::2, 0::2] = im[:, :, _cfa_pattern_arr[2]]
    bayer[1::2, 1::2] = im[:, :, _cfa_pattern_arr[3]]
    return bayer


def stack_rggb_channels(raw_image, bayer_pattern=None):
    """
    Stack the four channels of a CFA/Bayer raw image along a third dimension.
    """
    if bayer_pattern is None:
        bayer_pattern = [0, 1, 1, 2]
    height, width = raw_image.shape
    channels = []
    pattern = np.array(bayer_pattern)
    # add 1 to the second-green and the blue channel (e.g., [0, 1, 1, 2] will be [0, 1, 2, 3])
    pattern[pattern == 2] += 1
    pattern[2:][pattern[2:] == 1] += 1
    idx = [[0, 0], [0, 1], [1, 0], [1, 1]]
    for c in pattern:
        raw_image_c = raw_image[idx[c][0]:height:2, idx[c][1]:width:2].copy()
        channels.append(raw_image_c)

    # special case: channels re-ordered to [B G2 G1 R] instead of [R G1 G2 B] when
    # bayer_pattern==[G B R G]; need to flip it back.
    if bayer_pattern == [1, 2, 0, 1]:
        channels.reverse()
    channels = np.stack(channels, axis=-1)
    return channels


def stack_rgb_channels(raw_image, bayer_pattern):
    """
    Stack the four channels in a CFA/Bayer image into 3 RGB channels, averaging the two G channels.
    """
    rggb = stack_rggb_channels(raw_image, bayer_pattern)
    rgb = np.zeros((rggb.shape[0], rggb.shape[1], 3), dtype=np.float32)
    rgb[:, :, 0] = rggb[:, :, 0]
    rgb[:, :, 1] = (rggb[:, :, 1] + rggb[:, :, 2]) / 2.0
    rgb[:, :, 2] = rggb[:, :, 3]
    return rgb


def rggb_to_rgb(image_4ch, bayer_pattern):
    bayer_pattern = list(bayer_pattern)
    g1_idx = bayer_pattern.index(1)
    g2_idx = 3 if g1_idx == 0 else 2
    r_idx = bayer_pattern.index(0)
    b_idx = bayer_pattern.index(2)
    g = np.mean([image_4ch[:, :, g1_idx], image_4ch[:, :, g2_idx]], axis=0)
    rgb = np.stack([image_4ch[:, :, r_idx], g, image_4ch[:, :, b_idx]], axis=-1)
    return rgb
