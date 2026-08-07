"""
Vendored from ISP_Sim_mini (src/pipeline.py).

Author(s):
Abdelrahman Abdelhamed (a.abdelhamed@samsung.com)

Upstream: https://github.com/AbdoKamel/simple-camera-pipeline — MIT License,
Copyright (c) [2019] [Abdelrahman Abdelhamed]. The MIT notice is retained per
its terms. Modifications for RawGen are CC BY-NC-SA 4.0; see ../../LICENSE.md.
"""


from .pipeline_utils import get_visible_raw_image, get_metadata, normalize, white_balance, demosaic, \
    denoise, apply_color_space_transform, transform_xyz_to_srgb, apply_gamma, apply_tone_map, fix_orientation, \
    fix_missing_params, lens_shading_correction, active_area_cropping, default_cropping, \
    transform_xyz_to_prophoto, transform_prophoto_to_srgb, apply_profile_gain_table_map


def run_module(image, module, built_in_function, built_in_args):
    if type(module) is list and len(module) == 2:
        image = module[0](image, **module[1])
    elif type(module) is str:
        image = built_in_function(image, **built_in_args)
    else:
        raise ValueError('Invalid input module.')
    return image


def lens_shading_correction_stage(current_image, metadata, clip):
    gain_map_opcode = None
    if 'opcode_lists' in metadata:
        if 51009 in metadata['opcode_lists']:
            opcode_list_2 = metadata['opcode_lists'][51009]
            gain_map_opcode = [opcode_list_2[opcode_id] for opcode_id in sorted(opcode_list_2.keys())
                               if 9 <= opcode_id <= 9.3]

    lsc_map = None
    if 'lsc_map' in metadata:
        lsc_map = metadata['lsc_map']

    if gain_map_opcode is not None:
        current_image = lens_shading_correction(current_image, gain_map_opcode=gain_map_opcode,
                                                bayer_pattern=metadata['cfa_pattern'], clip=clip)
    elif lsc_map is not None:
        current_image = lens_shading_correction(current_image, gain_map_opcode=None,
                                                bayer_pattern=metadata['cfa_pattern'], gain_map=metadata['lsc_map'],
                                                clip=clip)
    return current_image


def run_pipeline(image_or_path, params=None, metadata=None, stages=None, clip=True):
    if type(image_or_path) == str:
        image_path = image_or_path
        # raw image data
        raw_image = get_visible_raw_image(image_path)
        # metadata
        metadata = get_metadata(image_path)
    else:
        raw_image = image_or_path.copy()
        # must provide metadata
        if metadata is None:
            raise ValueError("Must provide metadata when providing image data in first argument.")

    # take a deep copy of params as it will be modified below
    params = params.copy()

    # fill any missing parameters with default values
    params = fix_missing_params(params)

    '''
    Function performed at each stage. Follows this format:
    * {'stage_name': [function_name, function_params]} 
    * Assumes the function takes in `current_image` as the first parameter. 
    '''
    operation_by_stage = {
        'active_area_cropping': [active_area_cropping, {'active_area': metadata['active_area']}],
        'default_cropping': [default_cropping, {'default_crop_origin': metadata['default_crop_origin'],
                                                'default_crop_size': metadata['default_crop_size']}],
        'normal': [normalize, {
            'black_level': metadata['black_level'],
            'white_level': metadata['white_level'],
            'black_level_delta_h': metadata['black_level_delta_h'],
            'black_level_delta_v': metadata['black_level_delta_v'],
            'clip': clip}],
        'lens_shading_correction': [lens_shading_correction_stage, {'metadata': metadata, 'clip': clip}],
        'white_balance': [run_module, {
            'module': params['white_balancer'],
            'built_in_function': white_balance,
            'built_in_args': {
                'alg_type': params['white_balancer'],
                'as_shot_neutral': metadata['as_shot_neutral'],
                'cfa_pattern': metadata['cfa_pattern'],
                'clip': clip
            }
        }],
        'demosaic': [run_module, {
            'module': params['demosaicer'],
            'built_in_function': demosaic,
            'built_in_args': {
                'cfa_pattern': metadata['cfa_pattern'],
                'output_channel_order': 'RGB',
                'alg_type': params['demosaicer'],
            }
        }],
        'denoise': [run_module, {
            'module': params['denoiser'],
            'built_in_function': denoise,
            'built_in_args': {
                'alg_type': params['denoiser'],
                'cfa_pattern': metadata['cfa_pattern']
            }
        }],
        'xyz': [apply_color_space_transform, {
            'color_matrix_1': metadata['color_matrix_1'],
            'color_matrix_2': metadata['color_matrix_2'],
            'forward_matrix_1': metadata['forward_matrix_1'],
            'forward_matrix_2': metadata['forward_matrix_2'],
            'illuminant': metadata['as_shot_neutral'],
            'calib_illum_1': metadata['calib_illum_1'],
            'calib_illum_2': metadata['calib_illum_2'],
            'mode': params['cst_mode']
        }],
        'prophoto': [transform_xyz_to_prophoto, {}],
        'profile_gain_table': [apply_profile_gain_table_map, {'profile_gain_table_dict': metadata['profile_gain_table_dict']}],
        'pp2srgb': [transform_prophoto_to_srgb, {}],
        'xyz2srgb': [transform_xyz_to_srgb, {
            'white_point': params['xyz2srgb_white_point']
        }],
        'fix_orient': [fix_orientation, {'orientation': metadata['orientation']}],
        'gamma': [apply_gamma, {
            'mode': params['gamma_mode']
        }],
        'tone': [run_module, {
            'module': params['tone_curve'],
            'built_in_function': apply_tone_map,
            'built_in_args': {
                'tone_curve': params['tone_curve']
            }
        }],
    }

    if not stages:
        stages = ['raw', 'active_area_cropping', 'normal', 'lens_shading_correction', 'white_balance',
                  'demosaic', 'denoise', 'xyz', 'prophoto', 'profile_gain_table', 'srgb', 'fix_orient', 'gamma',
                  'tone', 'default_cropping']

    input_stage = params['input_stage']
    output_stage = params['output_stage']
    if input_stage not in stages \
            or output_stage not in stages \
            or stages.index(input_stage) > stages.index(output_stage):
        raise ValueError('Invalid input/output stage: input_stage = {}, output_stage = {}'.format(input_stage,
                                                                                                  output_stage))
    # Handle transforming directly from xyz to srgb
    srgb_idx = stages.index('srgb') if 'srgb' in stages else -1
    if srgb_idx > 0: # if srgb is one of the stages 
        # replace the "srgb" stage with either pp2srgb or xyz2srgb
        stages[srgb_idx] = 'pp2srgb' if 'prophoto' in stages else 'xyz2srgb'

    if output_stage == 'srgb':
        output_stage = stages[srgb_idx]

    input_idx = stages.index(input_stage)
    output_idx = stages.index(output_stage)
    current_image = raw_image

    for stage in stages[input_idx + 1:output_idx + 1]:
        operation = operation_by_stage[stage]
        current_image = operation[0](current_image, **operation[1])
    return current_image
