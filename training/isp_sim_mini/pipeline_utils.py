"""
Vendored from ISP_Sim_mini (src/pipeline_utils.py).

Author(s):
Abdelrahman Abdelhamed (a.abdelhamed@samsung.com)

Upstream: https://github.com/AbdoKamel/simple-camera-pipeline — MIT License,
Copyright (c) [2019] [Abdelrahman Abdelhamed]. The MIT notice is retained per
its terms. Modifications for RawGen are CC BY-NC-SA 4.0; see ../../LICENSE.md.

Camera pipeline utilities.
"""

import os
import cv2
import numpy as np
import exifread
import rawpy
from .cct_utils import raw_rgb_to_cct, interpolate_cst, raw_rgb_to_cct_fm
from fractions import Fraction
from exifread.utils import Ratio
from scipy.io import loadmat
from colour_demosaicing import demosaicing_CFA_Bayer_Menon2007
from .exif_utils import parse_exif, get_tag_values_from_ifds
from .dng_opcode import parse_opcode_lists, parse_profile_gain_table_map
from .raw_utils import stack_rgb_channels
import colour


def get_visible_raw_image(image_path):
    try:
        raw_image = rawpy.imread(image_path).raw_image_visible.copy()
    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        raise
    return raw_image


def save_image_stage(image, out_path, stage, save_as, bit_depth=8, stage_id=''):
    if stage_id:
        stage_id = f'{stage_id}_'

    if bit_depth != 8 and save_as == 'jpg':
        save_as = 'png'

    if os.path.isdir(out_path):
        output_image_path = os.path.join(out_path, 'merged_{}{}.{}'.format(stage_id, stage, save_as))
    else:
        out_path_no_ext, _ = os.path.splitext(out_path)
        output_image_path = '{}_{}{}.{}'.format(out_path_no_ext, stage_id, stage, save_as)
    print(output_image_path)
    output_image = (image * (2 ** bit_depth - 1)).astype('uint' + str(bit_depth))

    if len(output_image.shape) > 2:
        output_image = output_image[:, :, ::-1]
    if save_as == 'jpg':
        cv2.imwrite(output_image_path, output_image, [cv2.IMWRITE_JPEG_QUALITY, 100])
    else:
        cv2.imwrite(output_image_path, output_image)


def get_image_tags(image_path):
    with open(image_path, 'rb') as f:
        tags = exifread.process_file(f)
    return tags


def get_image_ifds(image_path):
    ifds = parse_exif(image_path, verbose=False)
    return ifds


def get_metadata(image_path):
    metadata = {}
    tags = get_image_tags(image_path)
    ifds = get_image_ifds(image_path)
    metadata['active_area'] = get_active_area(tags, ifds)
    metadata['linearization_table'] = get_linearization_table(tags, ifds)
    metadata['black_level'] = get_black_level(tags, ifds)
    metadata['white_level'] = get_white_level(tags, ifds)
    metadata['cfa_pattern'] = get_cfa_pattern(tags, ifds)
    metadata['as_shot_neutral'] = get_as_shot_neutral(tags, ifds)
    metadata['hsv_lut'] = get_hsv_luts(tags, ifds)
    metadata['profile_lut'] = get_profile_luts(tags, ifds)
    calib_illum_1, calib_illum_2 = get_calibration_illums(tags, ifds)
    metadata['calib_illum_1'] = calib_illum_1
    metadata['calib_illum_2'] = calib_illum_2
    color_matrix_1, color_matrix_2 = get_color_matrices(tags, ifds)
    metadata['color_matrix_1'] = color_matrix_1
    metadata['color_matrix_2'] = color_matrix_2
    forward_matrix_1, forward_matrix_2 = get_forward_matrices(tags, ifds)
    metadata['forward_matrix_1'] = forward_matrix_1
    metadata['forward_matrix_2'] = forward_matrix_2
    metadata['orientation'] = get_orientation(tags, ifds)
    metadata['noise_profile'] = get_noise_profile(tags, ifds)
    metadata['iso'] = get_iso(ifds)
    metadata['exposure_time'] = get_exposure_time(ifds)
    metadata['baseline_exposure'] = get_baseline_exposure(tags, ifds)
    metadata['profile_gain_table_dict'] = get_profile_gain_table_dict(tags, ifds)
    metadata['default_crop_origin'] = get_default_crop_origin(ifds)
    metadata['default_crop_size'] = get_default_crop_size(ifds)
    metadata['black_level_delta_h'] = get_black_level_delta_h(ifds)
    metadata['black_level_delta_v'] = get_black_level_delta_v(ifds)
    metadata['make'] = get_make(ifds)
    metadata['model'] = get_model(ifds)
    metadata['bits_per_sample'] = get_bits_per_sample(ifds)
    metadata['compression'] = get_compression(ifds)
    # ...

    # opcode lists
    metadata['opcode_lists'] = parse_opcode_lists(ifds)

    # fall back to default values, if necessary
    if metadata['black_level'] is None:
        metadata['black_level'] = 0
        print("Black level is None; using 0.")
    if metadata['white_level'] is None:
        metadata['white_level'] = 2 ** 16
        print("White level is None; using 2 ** 16.")
    if metadata['cfa_pattern'] is None:
        metadata['cfa_pattern'] = [0, 1, 1, 2]
        # print("CFAPattern is None; using [0, 1, 1, 2] (RGGB)")
    # if metadata['as_shot_neutral'] is None:
        # metadata['as_shot_neutral'] = [1, 1, 1]
        # print("AsShotNeutral is None; using [1, 1, 1]")
    if metadata['color_matrix_1'] is None:
        metadata['color_matrix_1'] = [1] * 9
        print("ColorMatrix1 is None; using [1, 1, 1, 1, 1, 1, 1, 1, 1]")
    if metadata['color_matrix_2'] is None:
        metadata['color_matrix_2'] = [1] * 9
        print("ColorMatrix2 is None; using [1, 1, 1, 1, 1, 1, 1, 1, 1]")
    if metadata['orientation'] is None:
        metadata['orientation'] = 0
        print("Orientation is None; using 0.")
    # ...
    return metadata


def get_compression(ifds):
    compression = get_tag_values_from_ifds(259, ifds)
    return compression[0]


def get_bits_per_sample(ifds):
    bits = get_tag_values_from_ifds(258, ifds)
    return bits


def get_active_area(tags, ifds):
    possible_keys = ['Image Tag 0xC68D', 'Image Tag 50829', 'ActiveArea', 'Image ActiveArea']
    vals = get_values(tags, possible_keys)
    if vals is None:
        vals = get_tag_values_from_ifds(50829, ifds)
    return vals


def get_linearization_table(tags, ifds):
    possible_keys = ['Image Tag 0xC618', 'Image Tag 50712', 'LinearizationTable', 'Image LinearizationTable']
    return get_values(tags, possible_keys)


def get_black_level(tags, ifds):
    possible_keys = ['Image Tag 0xC61A', 'Image Tag 50714', 'BlackLevel', 'Image BlackLevel']
    vals = get_values(tags, possible_keys)
    if vals is None:
        # print("Black level not found in exifread tags. Searching IFDs.")
        vals = get_tag_values_from_ifds(50714, ifds)
    return vals


def get_white_level(tags, ifds):
    possible_keys = ['Image Tag 0xC61D', 'Image Tag 50717', 'WhiteLevel', 'Image WhiteLevel']
    vals = get_values(tags, possible_keys)
    if vals is None:
        # print("White level not found in exifread tags. Searching IFDs.")
        vals = get_tag_values_from_ifds(50717, ifds)
    return vals


def get_cfa_pattern(tags, ifds):
    possible_keys = ['CFAPattern', 'Image CFAPattern']
    vals = get_values(tags, possible_keys)
    if vals is None:
        # print("CFAPattern not found in exifread tags. Searching IFDs.")
        vals = get_tag_values_from_ifds(33422, ifds)
    return vals


def get_as_shot_neutral(tags, ifds):
    possible_keys = ['Image Tag 0xC628', 'Image Tag 50728', 'AsShotNeutral', 'Image AsShotNeutral']
    return get_values(tags, possible_keys)


def get_baseline_exposure(tags, ifds):
    possible_keys = ['Image Tag 0xC62A', 'Image Tag 50730', 'BaselineExposure', 'Image BaselineExposure']
    exposure = int(get_values(tags, possible_keys)[0])
    return exposure


def get_hsv_luts(tags, ifds):
    possible_keys_1 = ['Image Tag 0xC6F9', 'Image Tag 50937', 'ProfileHueSatMapDims', 'Image ProfileHueSatMapDims']
    hue_sat_map_dims = get_values(tags, possible_keys_1)
    if hue_sat_map_dims is None:
        hue_sat_map_dims = get_tag_values_from_ifds(50937, ifds)
    possible_keys_2 = ['Image Tag 0xC6FA', 'Image Tag 50938', 'ProfileHueSatMapData1', 'Image ProfileHueSatMapData1']
    hsv_lut_1 = None
    if hsv_lut_1 is None:
        hsv_lut_1 = get_tag_values_from_ifds(50939, ifds)

    hsv_lut = None
    if hue_sat_map_dims is not None and hsv_lut_1 is not None:
        hue_sat_map_dims.append(3)
        hsv_lut = np.reshape(hsv_lut_1, newshape=hue_sat_map_dims)

    return hsv_lut


def get_profile_luts(tags, ifds):
    possible_keys_1 = ['Image Tag 0xC725', 'Image Tag 50981', 'ProfileLookTableDims', 'Image ProfileLookTableDims']
    profile_dims = get_values(tags, possible_keys_1)
    if profile_dims is None:
        profile_dims = get_tag_values_from_ifds(50981, ifds)
    possible_keys_2 = ['Image Tag 0xC726', 'Image Tag 50982', 'ProfileLookTableData', 'Image ProfileLookTableData']
    profile_lut = None
    if profile_lut is None:
        profile_lut = get_tag_values_from_ifds(50982, ifds)

    if profile_dims is not None and profile_lut is not None:
        profile_dims.append(3)
        profile_lut = np.reshape(profile_lut, newshape=profile_dims)

    return profile_lut


def get_profile_gain_table_dict(tags, ifds):
    possible_keys = ['Image Tag 0xCD2D', 'Image Tag 52525', 'ProfileGainTableMap']
    # can also inspect keys from Tiff container using `image.series[0].keyframe.tags._dict`
    values = get_values(tags, possible_keys)
    if not values or values is None:
        values = get_tag_values_from_ifds(52525, ifds)
    if values is None:
        # No profile gain table
        return None
    values = bytearray(values)
    profile_gain_table_dict = parse_profile_gain_table_map(values)
    return profile_gain_table_dict


def get_calibration_illums(tags, ifds):
    possible_keys_1 = ['Image Tag 0xC65A', 'Image Tag 50778', 'CalibrationIlluminant1', 'Image CalibrationIlluminant1']
    calib_illum_1 = get_values(tags, possible_keys_1)[0]
    possible_keys_2 = ['Image Tag 0xC65B', 'Image Tag 50779', 'CalibrationIlluminant2', 'Image CalibrationIlluminant2']
    calib_illum_2 = get_values(tags, possible_keys_2)[0]
    return calib_illum_1, calib_illum_2


def get_color_matrices(tags, ifds):
    possible_keys_1 = ['Image Tag 0xC621', 'Image Tag 50721', 'ColorMatrix1', 'Image ColorMatrix1']
    color_matrix_1 = get_values(tags, possible_keys_1)
    possible_keys_2 = ['Image Tag 0xC622', 'Image Tag 50722', 'ColorMatrix2', 'Image ColorMatrix2']
    color_matrix_2 = get_values(tags, possible_keys_2)
    return color_matrix_1, color_matrix_2


def get_forward_matrices(tags, ifds):
    possible_keys_1 = ['Image Tag 0xC714', 'Image Tag 50964', 'ForwardMatrix1', 'Image ForwardMatrix1']
    forward_matrix_1 = get_values(tags, possible_keys_1)
    possible_keys_2 = ['Image Tag 0xC715', 'Image Tag 50965', 'ForwardMatrix2', 'Image ForwardMatrix2']
    forward_matrix_2 = get_values(tags, possible_keys_2)
    return forward_matrix_1, forward_matrix_2


def get_orientation(tags, ifds):
    possible_tags = ['Orientation', 'Image Orientation']
    return get_values(tags, possible_tags)


def get_noise_profile(tags, ifds):
    possible_keys = ['Image Tag 0xC761', 'Image Tag 51041', 'NoiseProfile', 'Image NoiseProfile']
    vals = get_values(tags, possible_keys)
    if vals is None:
        # print("Noise profile not found in exifread tags. Searching IFDs.")
        vals = get_tag_values_from_ifds(51041, ifds)
    return vals


def get_iso(ifds):
    # 0x8827	34855
    return get_tag_values_from_ifds(34855, ifds)


def get_exposure_time(ifds):
    # 0x829a	33434
    exposure_time = get_tag_values_from_ifds(33434, ifds)[0]
    return float(exposure_time.numerator) / float(exposure_time.denominator)


def get_default_crop_origin(ifds):
    return get_tag_values_from_ifds(50719, ifds)


def get_default_crop_size(ifds):
    return get_tag_values_from_ifds(50720, ifds)


def get_black_level_delta_h(ifds):
    bldh = get_tag_values_from_ifds(50715, ifds)
    if bldh and type(bldh[0]) is Fraction:
        bldh = [float(bldh[i].numerator) / bldh[i].denominator for i in range(len(bldh))]
    return bldh


def get_black_level_delta_v(ifds):
    bldv = get_tag_values_from_ifds(50716, ifds)
    if bldv and type(bldv[0]) is Fraction:
        bldv = [float(bldv[i].numerator) / bldv[i].denominator for i in range(len(bldv))]
    return bldv


def get_make(ifds):
    make = get_tag_values_from_ifds(271, ifds)
    if make:
        make = b''.join(make[:-1]).decode()
    return make


def get_model(ifds):
    model = get_tag_values_from_ifds(272, ifds)
    if model:
        model = b''.join(model[:-1]).decode()
    return model


def get_values(tags, possible_keys):
    values = None
    for key in possible_keys:
        if key in tags.keys():
            values = tags[key].values
    return values


def active_area_cropping(image, active_area):
    if active_area is not None and active_area != [0, 0, image.shape[0], image.shape[1]]:
        image = image[active_area[0]: active_area[2], active_area[1]:active_area[3]]

    return image


def default_cropping(image, default_crop_origin, default_crop_size):
    if default_crop_origin is None or default_crop_size is None:
        return image
    
    if type(default_crop_origin[0]) is Fraction:
        default_crop_origin = [round(float(x.numerator) / float(x.denominator)) for x in default_crop_origin]
    if type(default_crop_size[0]) is Fraction:
        default_crop_size = [round(float(x.numerator) / float(x.denominator)) for x in default_crop_size]
    
    img_height, img_width = image.shape[:2]
    
    crop_x = max(0, int(default_crop_origin[0]))
    crop_y = max(0, int(default_crop_origin[1]))
    
    crop_width = min(int(default_crop_size[0]), img_width - crop_x)
    crop_height = min(int(default_crop_size[1]), img_height - crop_y)
    
    if crop_width <= 0 or crop_height <= 0:
        print(f"Warning: Invalid crop region. Using full image. Origin: {default_crop_origin}, Size: {default_crop_size}")
        return image
    
    if np.any([x != int(x) for x in default_crop_origin]):
        xs, ys = np.meshgrid(np.arange(crop_width) + default_crop_origin[0],
                             np.arange(crop_height) + default_crop_origin[1])
        xs = xs.astype(np.float32)
        ys = ys.astype(np.float32)
        
        xs = np.clip(xs, 0, img_width - 1)
        ys = np.clip(ys, 0, img_height - 1)
        
        image = cv2.remap(image, xs, ys, cv2.INTER_LINEAR)
    else:
        image = image[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]
    
    return image


def normalize(raw_image, black_level, white_level, black_level_delta_h=None, black_level_delta_v=None, clip=True):
    h, w = raw_image.shape[:2]

    if type(black_level) is list and len(black_level) == 1:
        black_level = float(black_level[0])
    if type(white_level) is list and len(white_level) == 1:
        white_level = float(white_level[0])
    black_level_mask = black_level
    if type(black_level) is list and len(black_level) == 4:
        if type(black_level[0]) is Ratio:
            black_level = ratios2floats(black_level)
        black_level_mask = np.zeros(raw_image.shape)
        idx2by2 = [[0, 0], [0, 1], [1, 0], [1, 1]]
        step2 = 2
        for i, idx in enumerate(idx2by2):
            black_level_mask[idx[0]::step2, idx[1]::step2] = black_level[i]

    # add black level delta h, v
    if black_level_delta_h:
        black_level_mask += np.tile(np.array(black_level_delta_h)[np.newaxis, :], [h, 1])
    if black_level_delta_v:
        delta_v = np.array(black_level_delta_v)
        if len(delta_v) != h:
            if len(delta_v) == h + 1:
                delta_v = delta_v[:-1]
            elif len(delta_v) == h - 1:
                delta_v = np.append(delta_v, delta_v[-1])
            else:
                delta_v = np.interp(np.arange(h), np.arange(len(delta_v)), delta_v)
        black_level_mask += np.tile(delta_v[:, np.newaxis], [1, w])

    normalized_image = raw_image.astype(np.float32) - black_level_mask
    # if some values were smaller than black level
    if clip:
        normalized_image[normalized_image < 0] = 0
    normalized_image = normalized_image / (white_level - black_level_mask)
    return normalized_image


def denormalize(raw_image, black_level, white_level):
    if type(black_level) is list and len(black_level) == 1:
        black_level = float(black_level[0])
    if type(white_level) is list and len(white_level) == 1:
        white_level = float(white_level[0])
    black_level_mask = black_level
    if type(black_level) is list and len(black_level) == 4:
        if type(black_level[0]) is Ratio:
            black_level = ratios2floats(black_level)
        black_level_mask = np.zeros(raw_image.shape)
        idx2by2 = [[0, 0], [0, 1], [1, 0], [1, 1]]
        step2 = 2
        for i, idx in enumerate(idx2by2):
            black_level_mask[idx[0]::step2, idx[1]::step2] = black_level[i]
    denormalized_image = raw_image.astype(np.float32) * (white_level - black_level_mask) + black_level_mask
    denormalized_image[denormalized_image < black_level] = black_level
    denormalized_image[denormalized_image > white_level] = white_level
    return denormalized_image


def ratios2floats(ratios):
    floats = []
    for ratio in ratios:
        floats.append(float(ratio.num) / ratio.den)
    return floats


def calc_gw(image, cfa_pattern):
    if len(image.shape) != 3:
        # bayer image
        image = stack_rgb_channels(image, cfa_pattern)

    gw_wb_val = np.reshape(image, (-1, 3)).mean(axis=0)
    illuminant = gw_wb_val / gw_wb_val[1]
    return illuminant


def white_balance(normalized_image, as_shot_neutral, cfa_pattern, clip=True, alg_type='default'):
    """
    :param normalized_image:
    :param as_shot_neutral: required when alg_type=='default', not required when alg_type=='gw'
    :param cfa_pattern: required when image is bayer, not required wh
    :param clip:
    :param alg_type:
    :return:
    """
    if alg_type == 'gw':
        illuminant = calc_gw(normalized_image, cfa_pattern)
    else:
        if type(as_shot_neutral[0]) is Ratio:
            as_shot_neutral = ratios2floats(as_shot_neutral)
        illuminant = as_shot_neutral

    if len(normalized_image.shape) == 3:
        return white_balance_rgb(normalized_image, illuminant, clip)

    idx2by2 = [[0, 0], [0, 1], [1, 0], [1, 1]]
    step2 = 2
    white_balanced_image = np.zeros(normalized_image.shape)
    for i, idx in enumerate(idx2by2):
        idx_y = idx[0]
        idx_x = idx[1]
        white_balanced_image[idx_y::step2, idx_x::step2] = \
            normalized_image[idx_y::step2, idx_x::step2] / illuminant[cfa_pattern[i]]
    if clip:
        white_balanced_image = np.clip(white_balanced_image, 0.0, 1.0)
    return white_balanced_image


def white_balance_rgb(image_rgb, illuminant, clip=True):
    illuminant_ = illuminant.copy()
    if type(illuminant_[0]) is Ratio:
        illuminant_ = ratios2floats(illuminant_)
    illuminant_ = np.reshape(illuminant_, (1, 1, 3))
    white_balanced_image = image_rgb / illuminant_
    if clip:
        white_balanced_image = np.clip(white_balanced_image, 0.0, 1.0)
    return white_balanced_image


def lens_shading_correction(raw_image, gain_map_opcode, bayer_pattern, gain_map=None, clip=True):
    """
    Apply lens shading correction map.
    :param raw_image: Input normalized (in [0, 1]) raw image.
    :param gain_map_opcode: Gain map opcode.
    :param bayer_pattern: Bayer pattern (RGGB, GRBG, ...).
    :param gain_map: Optional gain map to replace gain_map_opcode. 1 or 4 channels in order: R, Gr, Gb, and B.
    :param clip: Whether to clip result image to [0, 1].
    :return: Image with gain map applied; lens shading corrected.
    """
    if gain_map is not None:
        # resize gain map
        gain_map = cv2.resize(gain_map, dsize=(raw_image.shape[1] // 2, raw_image.shape[0] // 2),
                              interpolation=cv2.INTER_LINEAR)
    elif len(gain_map_opcode) == 1:
        gain_map = gain_map_opcode[0].data['map_gain_2d']
        gain_map = cv2.resize(gain_map, dsize=(raw_image.shape[1] // 2, raw_image.shape[0] // 2),
                              interpolation=cv2.INTER_LINEAR)
    elif len(gain_map_opcode) == 0:
        # If gain_map_opcode is empty, return the original image
        return raw_image
    else:
        # The 4 gain maps should come in the order of R G G B (verified by cfa_pattern and the top/left values)
        resized_gain_maps = [
            cv2.resize(opcode_data.data['map_gain_2d'], dsize=(raw_image.shape[1] // 2, raw_image.shape[0] // 2),
                       interpolation=cv2.INTER_LINEAR) for opcode_data in gain_map_opcode]
        gain_map = np.dstack(resized_gain_maps)

    if len(gain_map.shape) == 2:
        # make the gain map 4 channels, if needed
        gain_map = np.tile(gain_map[..., np.newaxis], [1, 1, 4])

    # if gain_map_opcode:
    #     # TODO: consider other parameters
    #
    #     top = gain_map_opcode.data['top']
    #     left = gain_map_opcode.data['left']
    #     bottom = gain_map_opcode.data['bottom']
    #     right = gain_map_opcode.data['right']
    #     rp = gain_map_opcode.data['row_pitch']
    #     cp = gain_map_opcode.data['col_pitch']
    #
    #     gm_w = right - left
    #     gm_h = bottom - top

    # gain_map = cv2.resize(gain_map, dsize=(gm_w, gm_h), interpolation=cv2.INTER_LINEAR)

    # TODO
    # if top > 0:
    #     pass
    # elif left > 0:
    #     left_col = gain_map[:, 0:1]
    #     rep_left_col = np.tile(left_col, [1, left])
    #     gain_map = np.concatenate([rep_left_col, gain_map], axis=1)
    # elif bottom < raw_image.shape[0]:
    #     pass
    # elif right < raw_image.shape[1]:
    #     pass

    result_image = raw_image.copy()

    # one channel
    # result_image[::rp, ::cp] *= gain_map[::rp, ::cp]

    # per bayer channel
    upper_left_idx = [[0, 0], [0, 1], [1, 0], [1, 1]]
    bayer_pattern_idx = np.array(bayer_pattern)
    # blue channel index --> 3
    bayer_pattern_idx[bayer_pattern_idx == 2] = 3
    # second green channel index --> 2
    if bayer_pattern_idx[3] == 1:
        bayer_pattern_idx[3] = 2
    else:
        bayer_pattern_idx[2] = 2
    for c in range(4):
        i0 = upper_left_idx[c][0]
        j0 = upper_left_idx[c][1]
        result_image[i0::2, j0::2] *= gain_map[:, :, bayer_pattern_idx[c]]

    if clip:
        result_image = np.clip(result_image, 0.0, 1.0)

    return result_image


def get_lut_with_most_highlight_suppression(profile_gain_table):
    """
    Find the LUT inside the LUT table that dims the highlights the most 
    -- has the smallest average gains for the highlights.
    The selected tone curve gain the shadow tones less than other tone curves in the same table only a 
    little bit.
    """
    h, w, num_tones = profile_gain_table.shape
    highlight_thresh = 0.7
    highlight_tone = np.round(num_tones * highlight_thresh).astype(np.uint16)  # treat tones above this as highlights

    profile_gain_table = profile_gain_table.reshape(-1, num_tones)
    highlight_gain_avgs = profile_gain_table[:, highlight_tone:].mean(axis=-1)  # (9 * 12, x) -> (9 * 12)
    min_lut_index = np.argmin(highlight_gain_avgs)
    min_lut = profile_gain_table[min_lut_index]
    return min_lut
    

def apply_profile_gain_table_map(image, profile_gain_table_dict):
    """
    Take only 1 LUT and apply it as a global tone curve as a quick
    workaround to avoid halo artifacts from local tone curve interpolation.

    Simplified implementation of profile gain table. Does not match with
    Adobe SDK, see RefBaselineProfileGainTableMap in dng_reference.cpp
    and DNG spec 1.7.1.0 ProfileGainTableMap.
    :param image: Input normalized (in [0, 1]) ??? image.
    :param profile_gain_table_dict: profile_gain_table_dict
    :param clip: Whether to clip result image to [0, 1].
    :return: Image with gain map applied; 

    'map_origin_v', 'map_origin_h' are not being used currently
    """
    profile_gain_table = profile_gain_table_dict['profile_gain_table']
    profile_gain_table_1d = get_lut_with_most_highlight_suppression(profile_gain_table)
    
    map_input_weights = profile_gain_table_dict['map_input_weights']
    weights = np.array(map_input_weights[:-2])
    table_indices = np.clip(np.sum(image * weights, axis=-1), 0, 1)  # weights are [1/3, 1/3, 1/3] for S25U; Use the average intensity as index to the table of gains  
    table_indices = np.round(table_indices * (profile_gain_table.shape[-1] - 1)).astype(np.uint16)

    gain_map = profile_gain_table_1d[table_indices]
    out_image = image * gain_map[..., np.newaxis]  # DNG 1.7 recommends NOT to clip
    return out_image


def get_opencv_demsaic_flag(cfa_pattern, output_channel_order, alg_type='VNG'):
    # using opencv edge-aware demosaicing
    if alg_type != '':
        alg_type = '_' + alg_type
    if output_channel_order == 'BGR':
        if cfa_pattern == [0, 1, 1, 2]:  # RGGB
            opencv_demosaic_flag = eval('cv2.COLOR_BAYER_BG2BGR' + alg_type)
        elif cfa_pattern == [2, 1, 1, 0]:  # BGGR
            opencv_demosaic_flag = eval('cv2.COLOR_BAYER_RG2BGR' + alg_type)
        elif cfa_pattern == [1, 0, 2, 1]:  # GRBG
            opencv_demosaic_flag = eval('cv2.COLOR_BAYER_GB2BGR' + alg_type)
        elif cfa_pattern == [1, 2, 0, 1]:  # GBRG
            opencv_demosaic_flag = eval('cv2.COLOR_BAYER_GR2BGR' + alg_type)
        else:
            opencv_demosaic_flag = eval('cv2.COLOR_BAYER_BG2BGR' + alg_type)
            #print("CFA pattern not identified.")
    else:  # RGB
        if cfa_pattern == [0, 1, 1, 2]:  # RGGB
            opencv_demosaic_flag = eval('cv2.COLOR_BAYER_BG2RGB' + alg_type)
        elif cfa_pattern == [2, 1, 1, 0]:  # BGGR
            opencv_demosaic_flag = eval('cv2.COLOR_BAYER_RG2RGB' + alg_type)
        elif cfa_pattern == [1, 0, 2, 1]:  # GRBG
            opencv_demosaic_flag = eval('cv2.COLOR_BAYER_GB2RGB' + alg_type)
        elif cfa_pattern == [1, 2, 0, 1]:  # GBRG
            opencv_demosaic_flag = eval('cv2.COLOR_BAYER_GR2RGB' + alg_type)
        else:
            opencv_demosaic_flag = eval('cv2.COLOR_BAYER_BG2RGB' + alg_type)
            #print("CFA pattern not identified.")
    return opencv_demosaic_flag


def denoise(image, alg_type='fgs', cfa_pattern=None):
    if alg_type == 'fgs':
        guide = (image * 255).astype(dtype=np.uint8)
        denoised_image = cv2.ximgproc.fastGlobalSmootherFilter(guide, image, 100, 0.75)
    else:
        denoised_image = image

    return denoised_image


def demosaic(bayer_image, cfa_pattern, output_channel_order='RGB', alg_type='VNG', clip=True):
    """
    Demosaic a Bayer image.
    :param bayer_image: Image in Bayer format, single channel.
    :param cfa_pattern: Bayer/CFA pattern.
    :param output_channel_order: Either RGB or BGR.
    :param alg_type: algorithm type. options: '', 'EA' for edge-aware, 'VNG' for variable number of gradients
    :param clip: Whether to clip values to [0, 1].
    :return: Demosaiced image.
    """
    if alg_type == 'VNG':
        max_val = 255
        wb_image = (bayer_image * max_val).astype(dtype=np.uint8)
    else:
        max_val = 16383
        wb_image = (bayer_image * max_val).astype(dtype=np.uint16)

    if alg_type in ['', 'EA', 'VNG']:
        opencv_demosaic_flag = get_opencv_demsaic_flag(cfa_pattern, output_channel_order, alg_type=alg_type)
        demosaiced_image = cv2.cvtColor(wb_image, opencv_demosaic_flag)
    elif alg_type == 'menon2007':
        cfa_pattern_str = "".join(["RGB"[i] for i in cfa_pattern])
        demosaiced_image = demosaicing_CFA_Bayer_Menon2007(wb_image, pattern=cfa_pattern_str)
    else:
        raise ValueError('Unsupported demosaicing algorithm, alg_type = {}'.format(alg_type))

    demosaiced_image = demosaiced_image.astype(dtype=np.float32) / max_val

    if clip:
        demosaiced_image = np.clip(demosaiced_image, 0, 1)

    return demosaiced_image


def apply_color_space_transform(image, color_matrix_1, color_matrix_2, forward_matrix_1, forward_matrix_2,
                                illuminant=None, interpolate_csts=True,
                                calib_illum_1=21, calib_illum_2=17, mode='fm'):
    """
    :param mode: cm | fm
    cm - use color matrices to compute cst
    fm - use forward matrices to compute cst
    :return:
    """
    if mode == 'fm':
        return apply_color_space_transform_fm(image, forward_matrix_1, forward_matrix_2, illuminant,
                                              interpolate_csts=interpolate_csts,
                                              calib_illum_1=calib_illum_1,
                                              calib_illum_2=calib_illum_2,
                                              color_matrix_1=color_matrix_1, 
                                              color_matrix_2=color_matrix_2)
    elif mode == 'cm':
        return apply_color_space_transform_cm(image, color_matrix_1, color_matrix_2, illuminant,
                                              calib_illum_1=calib_illum_1,
                                              calib_illum_2=calib_illum_2)
    else:
        raise Exception('Unsupported mode: ', mode)


def apply_color_space_transform_cm(image, color_matrix_1, color_matrix_2, illuminant=None,
                                calib_illum_1=21, calib_illum_2=17):
    """
    Implements "If the ForwardMatrix tags are not included in the camera profile" in DNG spec 1.4.
    Performs color space transform to CIE XYZ and performs WB in CIE XYZ.
    Do not use the `white_balance` stage with this function because this function
    performs WB along with CST. 

    :param image: RAW image, not white balanced
    :param color_matrix_1: ColorMatrix1 tag from dng
    :param color_matrix_2: ColorMatrix2 tag from dng
    :param illuminant: AsShotNeutral tag from dng
    :param calib_illum_1: CalibrationIlluminant1
    :param calib_illum_2: CalibrationIlluminant2
    """
    bradford = np.array([
        [0.8951000,  0.2664000, -0.1614000],
        [-0.7502000,  1.7135000,  0.0367000],
        [0.0389000, -0.0685000,  1.0296000],
    ])

    if type(color_matrix_1[0]) is Ratio:
        color_matrix_1 = ratios2floats(color_matrix_1)
    if type(color_matrix_2[0]) is Ratio:
        color_matrix_2 = ratios2floats(color_matrix_2)
    if type(illuminant[0]) is Ratio:
        illuminant = ratios2floats(illuminant)
    xyz2cam1 = np.reshape(np.asarray(color_matrix_1), (3, 3))
    xyz2cam2 = np.reshape(np.asarray(color_matrix_2), (3, 3))

    cct = raw_rgb_to_cct(illuminant, xyz2cam1, xyz2cam2, calib_illum_1, calib_illum_2)
    xyz2cam_interp = interpolate_cst(xyz2cam1, xyz2cam2, cct, calib_illum_1, calib_illum_2)
    cam2xyz = np.linalg.inv(xyz2cam_interp)
    M_A = bradford
    M_A_inverse = np.linalg.inv(bradford)
    
    src_illum_xyz = cam2xyz @ illuminant
    src_illum_lms = M_A @ src_illum_xyz
    tgt_illum_xyz = np.array([0.9642, 1.0, 0.8249])  # D50
    tgt_illum_lms = M_A @ tgt_illum_xyz

    scaling_factors_lms = tgt_illum_lms / src_illum_lms

    # RAW -> (cam2xyz) -> XYZ -> (M_A) -> LMS -> (scaling_factors_lms) -> WB-ed LMS -> (M_A_inverse) -> WB-ed XYZ
    mat = M_A_inverse @ np.diag(scaling_factors_lms) @ M_A @ cam2xyz
    xyz_image_CA = image @ mat.T

    xyz_image_CA = np.clip(xyz_image_CA, 0.0, 1.0)
    return xyz_image_CA


def apply_color_space_transform_fm(image, forward_matrix_1, forward_matrix_2, illuminant=None, interpolate_csts=True,
                                color_matrix_1=None, color_matrix_2=None,
                                calib_illum_1=21, calib_illum_2=17):
    """
    To be used with xyz2srgb_d50
    :param image:
    :param forward_matrix_1:
    :param forward_matrix_2:
    :param illuminant:
    :param interpolate_csts:
    :param calib_illum_1:
    :param calib_illum_2:
    :return:
    """
    if type(forward_matrix_1[0]) is Ratio:
        forward_matrix_1 = ratios2floats(forward_matrix_1)
    if type(forward_matrix_2[0]) is Ratio:
        forward_matrix_2 = ratios2floats(forward_matrix_2)
    if type(illuminant[0]) is Ratio:
        illuminant = ratios2floats(illuminant)

    cam2xyz1 = np.reshape(np.asarray(forward_matrix_1), (3, 3))
    cam2xyz2 = np.reshape(np.asarray(forward_matrix_2), (3, 3))

    if type(color_matrix_1[0]) is Ratio:
        color_matrix_1 = ratios2floats(color_matrix_1)
    if type(color_matrix_2[0]) is Ratio:
        color_matrix_2 = ratios2floats(color_matrix_2)
    if type(illuminant[0]) is Ratio:
        illuminant = ratios2floats(illuminant)
    
    xyz2cam1 = np.reshape(np.asarray(color_matrix_1), (3, 3))
    xyz2cam2 = np.reshape(np.asarray(color_matrix_2), (3, 3))

    if interpolate_csts and illuminant is not None:
        # interpolate between CSTs based on illuminant
        # Adobe SDK uses ColorMatrices to compute the CCT even when ForwardMatrices are present
        cct = raw_rgb_to_cct(illuminant, xyz2cam1, xyz2cam2, calib_illum_1, calib_illum_2)
        xyz2cam_interp = interpolate_cst(cam2xyz1, cam2xyz2, cct, calib_illum_1, calib_illum_2)
        xyz_image = image @ xyz2cam_interp.T
    else:
        # for now, use one matrix
        # simplified matrix multiplication
        xyz_image = image @ cam2xyz1.T

    xyz_image = np.clip(xyz_image, 0.0, 1.0)
    return xyz_image


def transform_xyz_to_prophoto(xyz_image):
    xyz2pp = np.array([[1.3459433, -0.2556075, -0.0511118],
                       [-0.5445989, 1.5081673, 0.0205351],
                       [0.0000000, 0.0000000, 1.2118128]])
    xyz2pp = xyz2pp / np.sum(xyz2pp, axis=-1, keepdims=True)
    pp_image = np.matmul(xyz_image, xyz2pp.T)
    pp_image = np.clip(pp_image, 0.0, 1.0)
    return pp_image


def transform_prophoto_to_srgb(pp_image):
    # Solved from ppn2srgbn = xyz2srgbn @ xyz2ppn.inv
    pp2srgb = np.array([[1.84770997, -0.53597631, -0.31173367],
                        [-0.25511275, 1.24992018, 0.00519257],
                        [-0.0164456, -0.14912261, 1.1655682]])
    srgb_image = np.matmul(pp_image, pp2srgb.T)
    srgb_image = np.clip(srgb_image, 0.0, 1.0)
    return srgb_image


def transform_xyz_to_srgb(xyz_image, white_point='d50'):
    if white_point == 'd50':
        xyz2srgb = np.array([[3.1338561, -1.6168667, -0.4906146],
                             [-0.9787684, 1.9161415, 0.0334540],
                             [0.0719453, -0.2289914, 1.4052427]])  # d50
    else:
        raise Exception('Unsupported white point: ', white_point)

    srgb_image = xyz_image @ xyz2srgb.T
    srgb_image = np.clip(srgb_image, 0.0, 1.0)
    return srgb_image


def fix_orientation(image, orientation):
    # 1 = Horizontal(normal)
    # 2 = Mirror horizontal
    # 3 = Rotate 180
    # 4 = Mirror vertical
    # 5 = Mirror horizontal and rotate 270 CW
    # 6 = Rotate 90 CW
    # 7 = Mirror horizontal and rotate 90 CW
    # 8 = Rotate 270 CW

    if type(orientation) is list:
        orientation = orientation[0]

    if orientation == 1:
        pass
    elif orientation == 2:
        image = cv2.flip(image, 0)
    elif orientation == 3:
        image = cv2.rotate(image, cv2.ROTATE_180)
    elif orientation == 4:
        image = cv2.flip(image, 1)
    elif orientation == 5:
        image = cv2.flip(image, 0)
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif orientation == 6 or orientation == 9:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif orientation == 7:
        image = cv2.flip(image, 0)
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif orientation == 8:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return image


def reverse_orientation(image, orientation):
    # 1 = Horizontal(normal)
    # 2 = Mirror horizontal
    # 3 = Rotate 180
    # 4 = Mirror vertical
    # 5 = Mirror horizontal and rotate 270 CW
    # 6 = Rotate 90 CW
    # 7 = Mirror horizontal and rotate 90 CW
    # 8 = Rotate 270 CW
    rev_orientations = np.array([1, 2, 3, 4, 5, 8, 7, 6])
    return fix_orientation(image, rev_orientations[orientation - 1])


def apply_gamma(x, mode='srgb'):
    if mode == '2.2':
        return x ** (1.0 / 2.2)
    elif mode == 'srgb':
        x = colour.cctf_encoding(x, str='sRGB')
        return x
    else:
        raise Exception(f'Unrecognized mode: {mode}.')


def apply_tone_map(x, tone_curve='simple-s-curve'):
    if tone_curve == 'simple-s-curve':
        tone_mapped_image = 3 * x ** 2 - 2 * x ** 3
    else:
        # tone_curve = loadmat('tone_curve.mat')
        tone_curve = loadmat(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'tone_curve.mat'))
        tone_curve = tone_curve['tc']
        x = np.round(x * (len(tone_curve) - 1)).astype(int)
        tone_mapped_image = np.squeeze(tone_curve[x])
    return tone_mapped_image


def fix_missing_params(params):
    """
    Fix params dictionary by filling missing parameters with default values.
    :param params: Input params dictionary.
    :return: Fixed params dictionary.
    """
    params_fixed = params.copy()
    default_params = {
        'input_stage': 'raw',
        'output_stage': 'tone',
        'save_as': 'jpg',
        'white_balancer': 'default',
        'demosaicer': '',
        'denoiser': 'fgs',
        'tone_curve': 'simple-s-curve',
        'xyz2srgb_white_point': 'd50',
        'cst_mode': 'fm',
        'gamma_mode': 'srgb',
    }
    for key in default_params.keys():
        if key not in params_fixed:
            params_fixed[key] = default_params[key]
    return params_fixed
