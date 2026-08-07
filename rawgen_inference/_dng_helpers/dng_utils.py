"""
Copyright (c) 2023 Samsung Electronics Co., Ltd.

Author(s):
Abhijith Punnappurath (abhijith.p@samsung.com)
Luxi Zhao (lucy.zhao@samsung.com)

Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License, (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc/4.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
For conditions of distribution and use, see the accompanying LICENSE.md file.

"""
import numpy as np
import cv2
from skimage import color

from .pipeline_utils import (
    get_image_tags, get_values, get_image_ifds, ratios2floats,
    raw_rgb_to_cct, interpolate_cst,
)
from exifread.utils import Ratio
from fractions import Fraction
from binascii import hexlify
from scipy.spatial import Delaunay


def compute_ang_error(source, target):
    source = np.reshape(source, [-1, 3]).astype(np.float32)
    target = np.reshape(target, [-1, 3]).astype(np.float32)
    source_norm = np.sqrt(np.sum(np.power(source, 2), 1))
    target_norm = np.sqrt(np.sum(np.power(target, 2), 1))
    norm = source_norm * target_norm
    L = np.shape(norm)[0]
    inds = norm != 0
    angles = np.sum(source[inds, :] * target[inds, :], 1) / norm[inds]
    angles[angles > 1] = 1
    f = np.arccos(angles)
    f[np.isnan(f)] = 0
    f = f * 180 / np.pi
    return sum(f) / L


def imresize(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    return cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)


def get_camera_calibration_matrices(tags, ifds):
    possible_keys_1 = ['Image Tag 0xC623', 'CameraCalibration1', 'Image CameraCalibration1']
    camera_calibration_matrix_1 = get_values(tags, possible_keys_1)
    possible_keys_2 = ['Image Tag 0xC624', 'CameraCalibration2', 'Image CameraCalibration2']
    camera_calibration_matrix_2 = get_values(tags, possible_keys_2)
    return camera_calibration_matrix_1, camera_calibration_matrix_2


def get_forward_matrices(tags, ifds):
    possible_keys_1 = ['Image Tag 0xC714', 'ForwardMatrix1', 'Image ForwardMatrix1']
    forward_matrix_1 = get_values(tags, possible_keys_1)
    possible_keys_2 = ['Image Tag 0xC715', 'ForwardMatrix2', 'Image ForwardMatrix2']
    forward_matrix_2 = get_values(tags, possible_keys_2)
    return forward_matrix_1, forward_matrix_2


def get_extra_tags(image_path, metadata):
    ## inputs are path to dng file, and current metadata from pipeline
    tags = get_image_tags(image_path)
    ifds = get_image_ifds(image_path)

    camera_calibration_matrix_1, camera_calibration_matrix_2 = get_camera_calibration_matrices(tags, ifds)
    metadata['camera_calibration_matrix_1'] = camera_calibration_matrix_1
    metadata['camera_calibration_matrix_2'] = camera_calibration_matrix_2

    forward_matrix_1, forward_matrix_2 = get_forward_matrices(tags, ifds)
    metadata['forward_matrix_1'] = forward_matrix_1
    metadata['forward_matrix_2'] = forward_matrix_2

    return metadata


def white_balance_3channel(normalized_image, as_shot_neutral, clip=True):
    white_balanced_image = normalized_image / as_shot_neutral
    if clip:
        white_balanced_image = np.clip(white_balanced_image, 0.0, 1.0)
    return white_balanced_image


def get_wb_as_matrix(as_shot_neutral):
    if type(as_shot_neutral[0]) is Ratio:
        as_shot_neutral = ratios2floats(as_shot_neutral)
    return np.diag(1 / np.asarray(as_shot_neutral))


def get_cst_matrix(color_matrix_1, color_matrix_2, illuminant):
    if type(color_matrix_1[0]) is Ratio:
        color_matrix_1 = ratios2floats(color_matrix_1)
    if type(color_matrix_2[0]) is Ratio:
        color_matrix_2 = ratios2floats(color_matrix_2)
    xyz2cam1 = np.reshape(np.asarray(color_matrix_1), (3, 3))
    xyz2cam2 = np.reshape(np.asarray(color_matrix_2), (3, 3))
    # normalize rows (needed?)
    xyz2cam1 = xyz2cam1 / np.sum(xyz2cam1, axis=1, keepdims=True)
    xyz2cam2 = xyz2cam2 / np.sum(xyz2cam2, axis=1, keepdims=True)

    # interpolate between CSTs based on illuminant
    cct = raw_rgb_to_cct(illuminant, xyz2cam1, xyz2cam2)
    # print(cct)
    xyz2cam_interp = interpolate_cst(xyz2cam1, xyz2cam2, cct)
    xyz2cam_interp = xyz2cam_interp / np.sum(xyz2cam_interp, axis=1, keepdims=True)
    cam2xyz_interp = np.linalg.inv(xyz2cam_interp)
    return cam2xyz_interp


def get_xyz_to_srgb_mat():
    xyz2srgb = np.array([[3.2404542, -1.5371385, -0.4985314],
                         [-0.9692660, 1.8760108, 0.0415560],
                         [0.0556434, -0.2040259, 1.0572252]])

    xyz2srgb = xyz2srgb / np.sum(xyz2srgb, axis=-1, keepdims=True)

    return xyz2srgb


def apply_combined_mat(image, mat):
    outimg = mat[np.newaxis, np.newaxis, :, :] * image[:, :, np.newaxis, :]
    outimg = np.sum(outimg, axis=-1)
    outimg = np.clip(outimg, 0.0, 1.0)
    return outimg


def RGB2bayer(img_rgb):
    h, w = img_rgb.shape[:2]
    bayer = np.empty((h, w), dtype=np.float32)

    (R, G, B) = cv2.split(img_rgb)

    # G R B G
    bayer[0::2, 0::2] = G[0::2, 0::2]  # top left
    bayer[0::2, 1::2] = R[0::2, 1::2]  # top right
    bayer[1::2, 0::2] = B[1::2, 0::2]  # bottom left
    bayer[1::2, 1::2] = G[1::2, 1::2]  # bottom right

    return bayer


def wb_float_to_rat(wb_values):
    wb_frac_R = Fraction(wb_values[0]).limit_denominator(max_denominator=2 ** 16 - 1)
    wb_frac_B = Fraction(wb_values[2]).limit_denominator(max_denominator=2 ** 16 - 1)
    numer = np.array([wb_frac_R.numerator, 1, wb_frac_B.numerator]).astype(np.uint32)
    denom = np.array([wb_frac_R.denominator, 1, wb_frac_B.denominator]).astype(np.uint32)
    return numer, denom


def update_wb_values(myhex, wb_values, wb_start):
    if not isinstance(wb_values[0], Ratio):
        numer, denom = wb_float_to_rat(wb_values)
    else:
        numer = np.array([wb_values[0].numerator, 1, wb_values[2].numerator]).astype(np.uint32)
        denom = np.array([wb_values[0].denominator, 1, wb_values[2].denominator]).astype(np.uint32)

    mystart = wb_start
    myend = mystart + 8
    for i in range(3):
        myhex[mystart:myend] = hexlify(numer[i].tobytes())  # byteswap()
        # myhex[mystart:myend] = hexlify(swap32(numer[i]).to_bytes(4, byteorder='big'))
        mystart = myend
        myend = mystart + 8

        myhex[mystart:myend] = hexlify(denom[i].tobytes())  # byteswap()
        # myhex[mystart:myend] = hexlify(swap32(denom[i]).to_bytes(4, byteorder='big'))
        mystart = myend
        myend = mystart + 8

    return myhex


def update_hex_image(myhex, bayer, image_start):
    myhex[image_start:image_start + bayer.size * 4] = hexlify(bayer.astype(np.uint16).flatten().tobytes())
    return myhex


def get_illum_normalized_by_g(illum_in_arr):
    return illum_in_arr[:, 0] / illum_in_arr[:, 1], illum_in_arr[:, 1] / illum_in_arr[:, 1], illum_in_arr[:,
                                                                                             2] / illum_in_arr[:, 1]


def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`
    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p) >= 0


def array_float_to_rat(array_values):
    numer = []
    denom = []
    for i in range(len(array_values)):
        frac = Fraction(array_values[i]).limit_denominator(max_denominator=2 ** 16 - 1)
        numer.append(np.array([frac.numerator]).astype(np.int32))
        denom.append(np.array([frac.denominator]).astype(np.int32))
    return numer, denom


def update_colormatrix1_values(myhex, colormatrix1, colormatrix1_start):
    numer, denom = array_float_to_rat(colormatrix1)

    mystart = colormatrix1_start
    myend = mystart + 8
    for i in range(9):
        myhex[mystart:myend] = hexlify(numer[i].tobytes())  # byteswap()
        # myhex[mystart:myend] = hexlify(swap32(numer[i]).to_bytes(4, byteorder='big'))
        mystart = myend
        myend = mystart + 8

        myhex[mystart:myend] = hexlify(denom[i].tobytes())  # byteswap()
        # myhex[mystart:myend] = hexlify(swap32(denom[i]).to_bytes(4, byteorder='big'))
        mystart = myend
        myend = mystart + 8

    return myhex


def calc_deltaE_rgb(source, target):
    """
    :param source: rgb, 8 bit
    :param target: rgb, 8 bit
    :return:
    """
    source = color.rgb2lab(source)
    target = color.rgb2lab(target)
    source = np.reshape(source, [-1, 3]).astype(np.float32)
    target = np.reshape(target, [-1, 3]).astype(np.float32)
    delta_e = np.sqrt(np.sum(np.power(source - target, 2), 1))
    delta_e = sum(delta_e) / (np.shape(delta_e)[0])
    return delta_e


def safe_invert_gains(image, wb_vec, rgb_gain=1.0):
    """
    Inverts gains while safely handling saturated pixels.
    Source: numpy version of
    https://github.com/timothybrooks/unprocessing/blob/master/unprocess.py#L92
    """
    gains = np.array(wb_vec) * rgb_gain
    gains = gains[np.newaxis, np.newaxis, :]

    # Prevents dimming of saturated pixels by smoothly masking gains near white.
    gray = np.mean(image, axis=-1, keepdims=True)
    inflection = 0.9

    mask = (np.maximum(gray - inflection, 0.0) / (1.0 - inflection)) ** 2.0
    safe_gains = np.maximum(mask + (1.0 - mask) * gains, gains)
    image = image * safe_gains
    image = np.clip(image, 0.0, 1.0)
    return image
