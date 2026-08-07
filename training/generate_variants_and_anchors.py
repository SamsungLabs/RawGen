#!/usr/bin/env python3
"""
Generate ISP variant images and anchor images from DNG files (Step 1).

This script generates training data by creating multiple ISP-processed variant images
from DNG files, along with anchor sRGB and optionally anchor XYZ images.

Uses the vendored mini ISP pipeline with random parameters to simulate
different photo-finishing pipelines.

Output:
    - Variant images saved in {dataset}/anchor-variants-imgs/ directory
    - Metadata JSON with processing statistics

Usage:
    python generate_variants_and_anchors.py \
        --trainval-json ../trainval.json \
        --split train \
        --output-dir ./output \
        --num-variations 5 \
        --cpu-workers 8
"""

import argparse
import json
import zlib
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from multiprocessing import Pool

import numpy as np
import cv2
from tqdm import tqdm

# Import ISP pipeline utilities
from isp_sim_mini.pipeline_utils import get_visible_raw_image, get_metadata
from isp_sim_mini.pipeline import run_pipeline
import colour

# Import project utilities
from _train_helpers import load_manifest_pairs


class ISPAugmenter:
    """
    ISP-based augmentation for generating varying sRGB images from DNG files.
    
    Uses the vendored mini ISP pipeline with a tone-mapping formula
    and random ISP parameters to simulate different photo-finishing pipelines.
    """
    
    def __init__(
        self,
        augmentation_method: str = "SimISP",
        color_grading_range: Tuple[float, float, float, float] = (0.7, 1.3, 0.7, 1.3),
        contrast_range: Tuple[float, float] = (0.8, 1.2),
        brightness_range: Optional[Tuple[float, float]] = None,
        tone_map_beta_mean: float = 0.6,
        tone_map_beta_std: float = 0.1,
        tone_map_gamma_mean: float = 0.9,
        tone_map_gamma_std: float = 0.1,
        seed: Optional[int] = None
    ):
        self.augmentation_method = augmentation_method
        self.color_grading_range = color_grading_range  # (r_min, r_max, b_min, b_max)
        self.contrast_range = contrast_range
        self.brightness_range = brightness_range
        self.tone_map_beta_mean = tone_map_beta_mean
        self.tone_map_beta_std = tone_map_beta_std
        self.tone_map_gamma_mean = tone_map_gamma_mean
        self.tone_map_gamma_std = tone_map_gamma_std
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def dng_to_anchor_srgb(self, dng_path: Path) -> np.ndarray:
        """Generate anchor sRGB image in [0,1] from DNG using the default ISP pipeline."""
        if not dng_path.exists():
            raise FileNotFoundError(f"DNG file not found: {dng_path}")
        
        try:
            raw_image = get_visible_raw_image(str(dng_path))
            metadata = get_metadata(str(dng_path))
        except Exception as e:
            raise RuntimeError(f"Failed to load DNG {dng_path}: {e}")
        
        # Define ISP pipeline parameters (default, no augmentation)
        isp_params = {
            'input_stage': 'raw',
            'output_stage': 'gamma',
            'cst_mode': 'fm',  # Forward matrix
            'white_balancer': 'default',
            'demosaicer': 'EA',  # Edge-Aware demosaicing
            'xyz2srgb_white_point': 'd50',
            'save_as': 'png'
        }
        
        # Define pipeline stages
        stages = ['raw', 'normal', 'lens_shading_correction',
                 'white_balance', 'demosaic', 'xyz', 'srgb', 'fix_orient', 'gamma']
        
        # Run ISP pipeline (no augmentation)
        srgb_img_float = run_pipeline(
            raw_image, 
            metadata=metadata, 
            params=isp_params, 
            stages=stages
        )
        
        srgb_img_float = np.clip(srgb_img_float, 0.0, 1.0)

        return srgb_img_float

    def dng_to_varying_srgb(
        self,
        dng_path: Path,
        num_variations: int
    ) -> Tuple[List[np.ndarray], List[Dict]]:
        """Generate `num_variations` sRGB images in [0,1] from DNG, each with random ISP parameters."""
        if not dng_path.exists():
            raise FileNotFoundError(f"DNG file not found: {dng_path}")
        
        try:
            raw_image = get_visible_raw_image(str(dng_path))
            metadata = get_metadata(str(dng_path))
        except Exception as e:
            raise RuntimeError(f"Failed to load DNG {dng_path}: {e}")
            
        varying_srgb_list = []
        isp_params_list = []
        
        for i in range(num_variations):
            # Generate random ISP parameters
            isp_params = self._generate_random_isp_params()
            isp_params_list.append(isp_params)
            
            # Apply ISP augmentation using real pipeline
            if self.augmentation_method == "SimISP":
                varying_srgb = self._apply_real_isp_pipeline(
                    raw_image, metadata, isp_params
                )
            else:
                raise ValueError(f"Unknown augmentation method: {self.augmentation_method}")
            
            varying_srgb_list.append(varying_srgb)
        
        return varying_srgb_list, isp_params_list
    
    def _generate_random_isp_params(self) -> Dict:
        """Generate random ISP parameters for augmentation."""
        # Color grading: sample R and B multipliers independently (G=1.0 fixed)
        r_min, r_max, b_min, b_max = self.color_grading_range
        r_multiplier = random.uniform(r_min, r_max)
        b_multiplier = random.uniform(b_min, b_max)
        # Create RGB multipliers for ISP pipeline (G=1.0 fixed)
        wb_rgb = np.array([r_multiplier, 1.0, b_multiplier])
        
        # Contrast
        contrast = random.uniform(self.contrast_range[0], self.contrast_range[1])
        
        # Brightness (optional)
        brightness = None
        if self.brightness_range is not None:
            # Sample additive brightness offset directly in sRGB domain
            brightness = random.uniform(self.brightness_range[0], self.brightness_range[1])
        
        # Tone mapping parameters
        beta = np.random.normal(self.tone_map_beta_mean, self.tone_map_beta_std)
        gamma = np.random.normal(self.tone_map_gamma_mean, self.tone_map_gamma_std)
        
        # Clamp parameters to reasonable ranges
        beta = np.clip(beta, 0.1, 2.0)
        gamma = np.clip(gamma, 0.5, 1.5)
        
        return {
            "r_multiplier": r_multiplier,
            "b_multiplier": b_multiplier,
            "wb_rgb": wb_rgb,  # Internal use for ISP pipeline
            "contrast": contrast,
            "brightness": brightness,
            "beta": beta,
            "gamma": gamma
        }
    
    def _apply_real_isp_pipeline(
        self, 
        raw_image: np.ndarray, 
        metadata: Dict, 
        params: Dict
    ) -> np.ndarray:
        """Run the vendored mini ISP pipeline with the given random photo-finishing parameters."""
        try:
            # Modify metadata with random parameters
            modified_metadata = metadata.copy()
            
            # Apply random white balance
            wb_rgb = params["wb_rgb"]
            if 'as_shot_neutral' in modified_metadata and modified_metadata['as_shot_neutral'] is not None:
                # Modify as_shot_neutral for white balance variation
                original_neutral = np.array(modified_metadata['as_shot_neutral'])
                modified_neutral = original_neutral * wb_rgb
                modified_metadata['as_shot_neutral'] = modified_neutral.tolist()
            
            # Define ISP pipeline parameters
            isp_params = {
                'input_stage': 'raw',
                'output_stage': 'gamma',
                'cst_mode': 'fm',  # Forward matrix
                'white_balancer': 'default',
                'demosaicer': 'EA',  # Edge-Aware demosaicing
                'xyz2srgb_white_point': 'd50',
                'save_as': 'png'
            }
            
            # Define pipeline stages
            stages = ['raw', 'normal', 'lens_shading_correction',
                     'white_balance', 'demosaic', 'xyz', 'srgb', 'fix_orient', 'gamma']
            
            # Run ISP pipeline
            srgb_img_float = run_pipeline(
                raw_image, 
                metadata=modified_metadata, 
                params=isp_params, 
                stages=stages
            )
            
            srgb_img_float = np.clip(srgb_img_float, 0.0, 1.0)

            # Tone mapping runs in the linear domain
            linear_rgb = self._srgb_to_linear_rgb(srgb_img_float)
            linear_rgb = np.clip(linear_rgb, 0.0, 1.0)

            beta = params["beta"]
            gamma = params["gamma"]
            
            E_i = np.clip(linear_rgb, 0.0, 1.0)
            E_i_gamma = np.power(E_i, gamma)
            
            numerator = (1 + beta) * E_i_gamma
            denominator = beta + E_i_gamma
            denominator = np.maximum(denominator, 1e-8)
            
            tone_mapped = numerator / denominator
            
            # Convert tone-mapped linear RGB to sRGB for display-domain adjustments
            srgb_after_tm = self._linear_to_srgb(tone_mapped)
            srgb_after_tm = np.clip(srgb_after_tm, 0.0, 1.0)
            
            # Apply brightness (additive) and contrast (pivot at 0.5) in sRGB domain
            if params.get("contrast") is not None:
                srgb_after_tm = (srgb_after_tm - 0.5) * params["contrast"] + 0.5
            if params.get("brightness") is not None:
                srgb_after_tm = srgb_after_tm + params["brightness"]
            
            # Clamp to valid [0,1] sRGB range
            final_srgb = np.clip(srgb_after_tm, 0.0, 1.0)
            
            return final_srgb
            
        except Exception as e:
            raise RuntimeError(f"ISP pipeline failed: {e}")
    
    def _srgb_to_linear_rgb(self, srgb: np.ndarray) -> np.ndarray:
        """Convert sRGB to linear RGB using colour library cctf_decoding."""
        return colour.cctf_decoding(srgb, function='sRGB')
    
    def _linear_to_srgb(self, linear_rgb: np.ndarray) -> np.ndarray:
        """Convert linear RGB to sRGB using colour library cctf_encoding."""
        return colour.cctf_encoding(linear_rgb, function='sRGB')

    def dng_to_anchor_xyz(self, dng_path: Path, apply_gamma: bool = True) -> np.ndarray:
        """Generate anchor XYZ image in [0,1] from DNG using the default ISP pipeline."""
        if not dng_path.exists():
            raise FileNotFoundError(f"DNG file not found: {dng_path}")
        
        try:
            raw_image = get_visible_raw_image(str(dng_path))
            metadata = get_metadata(str(dng_path))
        except Exception as e:
            raise RuntimeError(f"Failed to load DNG {dng_path}: {e}")
        
        # Define ISP pipeline parameters (default, no augmentation)
        isp_params = {
            'input_stage': 'raw',
            'output_stage': 'gamma' if apply_gamma else 'fix_orient',
            'cst_mode': 'fm',  # Forward matrix
            'white_balancer': 'default',
            'demosaicer': 'EA',  # Edge-Aware demosaicing
            'xyz2srgb_white_point': 'd50',
            'save_as': 'png'
        }
        
        # Define pipeline stages based on gamma option
        if apply_gamma:
            stages = ['raw', 'normal', 'lens_shading_correction',
                     'white_balance', 'demosaic', 'xyz', 'fix_orient', 'gamma']
        else:
            stages = ['raw', 'normal', 'lens_shading_correction',
                     'white_balance', 'demosaic', 'xyz', 'fix_orient']
        
        # Run ISP pipeline
        xyz_img_float = run_pipeline(
            raw_image, 
            metadata=metadata, 
            params=isp_params, 
            stages=stages
        )
        
        xyz_img_float = np.clip(xyz_img_float, 0.0, 1.0)

        return xyz_img_float


def resize_image_with_mode(img: np.ndarray, target_size: int, mode: str = "crop") -> np.ndarray:
    """Resize to a square `target_size`, either center-cropping first ("crop") or stretching."""
    if mode == "crop":
        # Center-crop to largest square, then resize
        h, w = img.shape[:2]
        if h == w and h == target_size:
            return img
        
        # Crop to largest centered square
        size = min(h, w)
        y_start = (h - size) // 2
        x_start = (w - size) // 2
        cropped = img[y_start:y_start + size, x_start:x_start + size]
        
        # Resize to target size
        if size != target_size:
            resized = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
            return resized
        return cropped
    
    elif mode == "stretch":
        # Direct resize (may break aspect ratio)
        if img.shape[0] == target_size and img.shape[1] == target_size:
            return img
        resized = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
        return resized
    
    else:
        raise ValueError(f"Unknown resize mode: {mode}")


def save_image_as_png(img: np.ndarray, path: Path, bit_depth: int = 8):
    """Save a [0,1] float image as an 8- or 16-bit PNG."""
    img_clamped = np.clip(img, 0.0, 1.0)

    if bit_depth == 8:
        # Convert to uint8
        img_out = (img_clamped * 255).astype(np.uint8)
    elif bit_depth == 16:
        # Convert to uint16
        img_out = (img_clamped * 65535).astype(np.uint16)
    else:
        raise ValueError(f"Unsupported bit depth: {bit_depth}")
    
    # Convert RGB to BGR for OpenCV
    img_bgr = img_out[:, :, ::-1]
    cv2.imwrite(str(path), img_bgr)


def process_single_pair_for_isp(pair_data):
    """
    Process a single pair for ISP variant generation.
    This function must be at module level for multiprocessing to work.
    """
    srgb_path, xyz_path, args, isp_params = pair_data
    
    try:
        # Extract basename and paths
        basename = xyz_path.stem.replace("_xyz", "")
        dng_path = xyz_path.parent.parent / "DNG" / f"{basename}.dng"
        variants_dir = xyz_path.parent.parent / args.output_dir
        
        # Skip if DNG doesn't exist
        if not dng_path.exists():
            return {"success": False, "basename": basename, "error": f"DNG not found: {dng_path}"}
        
        # Create variants directory
        variants_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if variants already exist
        anchor_srgb_path = variants_dir / f"{basename}_anchor_srgb.png"
        anchor_xyz_path = variants_dir / f"{basename}_anchor_xyz.png"
        var_paths = [variants_dir / f"{basename}_var_{i:02d}_srgb.png" for i in range(args.num_variations)] if args.num_variations > 0 else []
        
        # If regenerate_anchor_srgb_only mode, only check anchor sRGB
        if args.regenerate_anchor_srgb_only:
            # Always regenerate anchor sRGB in this mode (don't skip)
            pass
        else:
            # Check if all required files exist
            files_to_check = [anchor_srgb_path]
            if args.num_variations > 0:
                files_to_check.extend(var_paths)
            if args.generate_xyz_anchor:
                files_to_check.append(anchor_xyz_path)
            
            if all(p.exists() for p in files_to_check):
                return {
                    "success": True,
                    "basename": basename,
                    "skipped": True,
                    "reason": "already exists",
                    "variants_dir": str(variants_dir)
                }
        
        # Initialize ISP augmenter for this process
        isp_params_for_scene = isp_params.copy()
        
        # Generate unique seed per scene (unless --use-shared-seed is set)
        if not args.use_shared_seed and isp_params_for_scene.get("seed") is not None:
            base_seed = isp_params_for_scene["seed"]
            # crc32, not hash(): CPython randomises str hashing per process
            # (PYTHONHASHSEED), which would make --seed non-reproducible.
            scene_seed = (base_seed + zlib.crc32(basename.encode())) % (2**32)
            isp_params_for_scene["seed"] = scene_seed
        
        isp_augmenter = ISPAugmenter(**isp_params_for_scene)
        
        # If regenerate_anchor_srgb_only mode, only generate and save anchor sRGB
        if args.regenerate_anchor_srgb_only:
            # Generate anchor sRGB image from DNG using ISP pipeline
            anchor_srgb_img = isp_augmenter.dng_to_anchor_srgb(dng_path)
            anchor_srgb_img = resize_image_with_mode(anchor_srgb_img, args.resize_res_for_batch_process, args.resize_mode)

            # Save anchor sRGB image (8-bit)
            save_image_as_png(anchor_srgb_img, anchor_srgb_path, bit_depth=8)
            
            return {
                "success": True,
                "basename": basename,
                "skipped": False,
                "variants_dir": str(variants_dir)
            }
        
        # Normal mode: generate all images
        # Generate anchor sRGB image from DNG using ISP pipeline
        anchor_srgb_img = isp_augmenter.dng_to_anchor_srgb(dng_path)
        anchor_srgb_img = resize_image_with_mode(anchor_srgb_img, args.resize_res_for_batch_process, args.resize_mode)

        # Generate anchor XYZ image if enabled
        if args.generate_xyz_anchor:
            anchor_xyz_img = isp_augmenter.dng_to_anchor_xyz(dng_path, apply_gamma=args.xyz_anchor_gamma)
            anchor_xyz_img = resize_image_with_mode(anchor_xyz_img, args.resize_res_for_batch_process, args.resize_mode)
        
        # Generate varying images from DNG (only if num_variations > 0)
        if args.num_variations > 0:
            varying_images, isp_params_list = isp_augmenter.dng_to_varying_srgb(dng_path, args.num_variations)
            
            # Resize varying images
            resized_varying_images = []
            for img in varying_images:
                resized_img = resize_image_with_mode(img, args.resize_res_for_batch_process, args.resize_mode)
                resized_varying_images.append(resized_img)
        else:
            isp_params_list = []
            resized_varying_images = []
        
        # Save anchor sRGB image (8-bit)
        save_image_as_png(anchor_srgb_img, anchor_srgb_path, bit_depth=8)
        
        # Save anchor XYZ image (16-bit)
        if args.generate_xyz_anchor:
            save_image_as_png(anchor_xyz_img, anchor_xyz_path, bit_depth=16)
        
        # Save varying images (8-bit) - only if num_variations > 0
        if args.num_variations > 0:
            for i, img in enumerate(resized_varying_images):
                save_image_as_png(img, var_paths[i], bit_depth=8)
        
        # Convert parameters to JSON-serializable format
        variant_params = {}
        for i, params in enumerate(isp_params_list):
            var_key = f"var_{i:02d}"
            variant_params[var_key] = {
                "r_multiplier": float(params["r_multiplier"]),
                "b_multiplier": float(params["b_multiplier"]),
                "contrast": float(params["contrast"]),
                "brightness": float(params["brightness"]) if params["brightness"] is not None else None,
                "beta": float(params["beta"]),
                "gamma": float(params["gamma"])
            }
        
        return {
            "success": True,
            "basename": basename,
            "skipped": False,
            "variant_params": variant_params,
            "variants_dir": str(variants_dir)
        }
        
    except Exception as e:
        try:
            basename_for_error = basename
        except NameError:
            basename_for_error = xyz_path.stem.replace("_xyz", "")
        return {"success": False, "basename": basename_for_error, "error": str(e)}


def create_parameter_ranges_from_args(args, isp_params: Dict) -> Dict:
    """
    Collect the sampling ranges declared on the command line (not the sampled values).

    Stored in meta.json so downstream code can normalize ISP parameters.
    """
    param_ranges = {}
    
    # r_multiplier and b_multiplier: use color_grading_range from args
    if hasattr(args, 'color_grading_range') and args.color_grading_range:
        color_grading_range = tuple(args.color_grading_range)
    else:
        color_grading_range = isp_params.get("color_grading_range", (0.7, 1.3, 0.7, 1.3))
    # color_grading_range is (r_min, r_max, b_min, b_max)
    param_ranges["r_multiplier"] = {
        "min": float(color_grading_range[0]),
        "max": float(color_grading_range[1])
    }
    param_ranges["b_multiplier"] = {
        "min": float(color_grading_range[2]),
        "max": float(color_grading_range[3])
    }
    
    # contrast: use contrast_range from args
    if hasattr(args, 'contrast_range') and args.contrast_range:
        contrast_range = tuple(args.contrast_range)
    else:
        contrast_range = isp_params.get("contrast_range", (0.7, 1.3))
    param_ranges["contrast"] = {
        "min": float(contrast_range[0]),
        "max": float(contrast_range[1])
    }
    
    # brightness: use brightness_range from args (only if not None)
    brightness_range = None
    if hasattr(args, 'brightness_range') and args.brightness_range is not None:
        brightness_range = tuple(args.brightness_range)
    elif isp_params.get("brightness_range") is not None:
        brightness_range = tuple(isp_params["brightness_range"])
    
    if brightness_range is not None:
        param_ranges["brightness"] = {
            "min": float(brightness_range[0]),
            "max": float(brightness_range[1])
        }
    
    # beta: use tone_map_beta_mean ± tone_map_beta_std (clamped to [0.1, 2.0])
    if hasattr(args, 'tone_map_beta_mean'):
        beta_mean = args.tone_map_beta_mean
    else:
        beta_mean = isp_params.get("tone_map_beta_mean", 0.6)
    
    if hasattr(args, 'tone_map_beta_std'):
        beta_std = args.tone_map_beta_std
    else:
        beta_std = isp_params.get("tone_map_beta_std", 0.1)
    
    # Use mean ± 3*std to cover most of the range, then clamp to actual limits [0.1, 2.0]
    beta_min = max(0.1, beta_mean - 3 * beta_std)
    beta_max = min(2.0, beta_mean + 3 * beta_std)
    param_ranges["beta"] = {
        "min": float(beta_min),
        "max": float(beta_max)
    }
    
    # gamma: use tone_map_gamma_mean ± tone_map_gamma_std (clamped to [0.5, 1.5])
    if hasattr(args, 'tone_map_gamma_mean'):
        gamma_mean = args.tone_map_gamma_mean
    else:
        gamma_mean = isp_params.get("tone_map_gamma_mean", 0.9)
    
    if hasattr(args, 'tone_map_gamma_std'):
        gamma_std = args.tone_map_gamma_std
    else:
        gamma_std = isp_params.get("tone_map_gamma_std", 0.1)
    
    # Use mean ± 3*std to cover most of the range, then clamp to actual limits [0.5, 1.5]
    gamma_min = max(0.5, gamma_mean - 3 * gamma_std)
    gamma_max = min(1.5, gamma_mean + 3 * gamma_std)
    param_ranges["gamma"] = {
        "min": float(gamma_min),
        "max": float(gamma_max)
    }
    
    return param_ranges


def generate_isp_variants(pairs: List[Tuple[Path, Path]], args, isp_params: Dict) -> Dict:
    """Generate ISP variant images from DNG files, save them, and return generation statistics."""
    print("Generating ISP variant images...")
    
    # Prepare data for multiprocessing
    pair_data_list = [(srgb_path, xyz_path, args, isp_params) for srgb_path, xyz_path in pairs]
    
    # Process all pairs
    results = []
    processed_count = 0
    skipped_count = 0
    failed_count = 0
    
    with Pool(processes=args.cpu_workers) as pool:
        with tqdm(total=len(pairs), desc="Generating ISP Variants", leave=True) as pbar:
            for result in pool.imap(process_single_pair_for_isp, pair_data_list):
                results.append(result)
                
                if result["success"]:
                    if result.get("skipped", False):
                        skipped_count += 1
                    else:
                        processed_count += 1
                else:
                    failed_count += 1
                
                pbar.update(1)
                pbar.set_postfix({
                    'processed': processed_count,
                    'skipped': skipped_count,
                    'failed': failed_count
                })
    
    print(f"\nGeneration Complete:")
    print(f"  Processed: {processed_count}")
    print(f"  Skipped (already exist): {skipped_count}")
    print(f"  Failed: {failed_count}")
    
    # Collect variant parameters and save to meta.json files
    # Group by variants_dir since different datasets may have different directories
    params_by_dir = {}
    
    for result in results:
        if result["success"] and not result.get("skipped", False):
            variants_dir_str = result.get("variants_dir")
            basename = result.get("basename")
            variant_params = result.get("variant_params")
            
            if variants_dir_str and basename and variant_params:
                if variants_dir_str not in params_by_dir:
                    params_by_dir[variants_dir_str] = {}
                params_by_dir[variants_dir_str][basename] = variant_params
    
    # Save meta.json files for each directory
    meta_files_saved = 0
    for variants_dir_str, params_dict in params_by_dir.items():
        variants_dir = Path(variants_dir_str)
        meta_json_path = variants_dir / "meta.json"
        
        # Load existing meta.json if it exists
        existing_meta = {}
        if meta_json_path.exists():
            try:
                with open(meta_json_path, 'r') as f:
                    existing_meta = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load existing meta.json at {meta_json_path}: {e}")
                existing_meta = {}
        
        # Extract existing data
        existing_params = {}
        
        # Keep only basename entries (not metadata like _param_ranges)
        for key, value in existing_meta.items():
            if not key.startswith("_"):
                existing_params[key] = value
        
        # Merge existing parameters with new ones (new ones take precedence)
        merged_params = {**existing_params, **params_dict}
        
        # Create parameter ranges from argparse arguments (not from sampled values)
        param_ranges = create_parameter_ranges_from_args(args, isp_params)
        
        # Create final metadata structure
        final_meta = {
            "_param_ranges": param_ranges,
            **merged_params
        }
        
        # Save merged parameters with ranges
        try:
            with open(meta_json_path, 'w') as f:
                json.dump(final_meta, f, indent=2)
            meta_files_saved += 1
            print(f"Saved meta.json to {meta_json_path} ({len(merged_params)} entries)")
        except Exception as e:
            print(f"Error: Failed to save meta.json to {meta_json_path}: {e}")
    
    if meta_files_saved > 0:
        print(f"Saved {meta_files_saved} meta.json file(s)")
    
    return {
        "pairs_processed": processed_count,
        "pairs_skipped": skipped_count,
        "pairs_failed": failed_count,
        "results": results
    }


def build_argparser() -> argparse.ArgumentParser:
    """Build command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate ISP variant images and anchor images from DNG files (Step 1)"
    )
    
    # Model and data parameters
    parser.add_argument("--trainval-json", type=Path, default="../trainval.json",
                       help="Path to trainval.json manifest")
    parser.add_argument("--split", type=str, default="all", choices=["train", "val", "test", "all"],
                       help="Dataset split to process")
    parser.add_argument("--datasets", type=str, nargs="+", default=None,
                       help="Dataset subsets to include (e.g., a5k raise nus, or 'all')")
    parser.add_argument("--root-key", type=str, default="root",
                       help="Root key in manifest for dataset subsets (e.g., 'root' or 'root_aws')")
    parser.add_argument("--output-dir", type=str, default="anchor-variants-imgs", # variants-N-anchor-pairs-wo-brightness
                       help="Subdirectory name to save generated images next to dataset root")
    parser.add_argument("--num-variations", type=int, default=5,
                       help="Number of varying sRGB per image")
    
    # ISP augmentation parameters
    parser.add_argument("--augmentation-method", type=str, default="SimISP",
                       help="Augmentation method")
    parser.add_argument("--color-grading-range", type=float, nargs=4, default=[0.7, 1.3, 0.7, 1.3],
                       help="Color grading range for R and B channels: r_min r_max b_min b_max (G=1.0 fixed)")
    parser.add_argument("--contrast-range", type=float, nargs=2, default=[0.7, 1.3],
                       help="Contrast multiplier range (applied in sRGB around pivot 0.5)")
    parser.add_argument("--brightness-range", type=float, nargs=2, default=None,
                       help="Brightness additive offset in sRGB (e.g., --brightness-range -0.3 0.3, omit to disable)")
    parser.add_argument("--tone-map-beta-mean", type=float, default=0.6,
                       help="Beta parameter mean for tone mapping")
    parser.add_argument("--tone-map-beta-std", type=float, default=0.1,
                       help="Beta parameter std for tone mapping")
    parser.add_argument("--tone-map-gamma-mean", type=float, default=0.9,
                       help="Gamma parameter mean for tone mapping")
    parser.add_argument("--tone-map-gamma-std", type=float, default=0.1,
                       help="Gamma parameter std for tone mapping")
    
    # Processing parameters
    parser.add_argument("--cpu-workers", type=int, default=4,
                    help="Number of CPU workers for ISP variant generation")
    parser.add_argument("--resize-res-for-batch-process", type=int, default=1024,
                    help="Resize resolution for batch processing (default: 1024)")
    parser.add_argument("--resize-mode", type=str, default="crop", choices=["crop", "stretch"],
                       help="Resize mode: 'crop' (center-crop then resize) or 'stretch' (direct resize)")
    parser.add_argument("--seed", type=int, default=42,
                    help="Random seed")
    parser.add_argument("--use-shared-seed", action="store_true",
                       help="Use same seed for all scenes (variants with same index will have identical ISP params across scenes)")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum number of samples to process (for testing)")
    
    # XYZ anchor generation
    parser.add_argument("--generate-xyz-anchor", action="store_true",
                       help="Generate XYZ anchor images in addition to sRGB anchors")
    parser.add_argument("--xyz-anchor-gamma", action=argparse.BooleanOptionalAction, default=True,
                       help="Encode the XYZ anchor with the sRGB OETF (default: on). "
                            "xyz_to_raw.py assumes this encoding.")

    # Anchor sRGB regeneration options
    parser.add_argument("--regenerate-anchor-srgb-only", action="store_true",
                       help="Regenerate only anchor sRGB images (skip variants and XYZ)")

    return parser


def main():
    """Main processing function."""
    args = build_argparser().parse_args()
    
    # Validate arguments
    if not args.trainval_json.exists():
        raise FileNotFoundError(f"Manifest file not found: {args.trainval_json}")
    
    if args.num_variations < 0:
        raise ValueError("num_variations must be >= 0")
    
    if args.cpu_workers < 1:
        raise ValueError("cpu_workers must be >= 1")
    
    print(f"Processing split: {args.split}")
    print(f"Number of variations per image: {args.num_variations}")
    print(f"CPU workers for ISP generation: {args.cpu_workers}")
    print(f"Resize resolution for batch processing: {args.resize_res_for_batch_process}")
    print(f"Resize mode: {args.resize_mode}")
    print(f"Augmentation method: {args.augmentation_method}")
    print(f"Generate XYZ anchor: {args.generate_xyz_anchor}")
    print(f"Regenerate anchor sRGB only: {args.regenerate_anchor_srgb_only}")
    print(f"Use shared seed across scenes: {args.use_shared_seed}")
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Load manifest and get sRGB-XYZ pairs
    print(f"Loading manifest from {args.trainval_json}...")
    try:
        pairs, meta = load_manifest_pairs(
            args.trainval_json,
            dataset_subsets=args.datasets,
            split=args.split,
            root_key=args.root_key,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load manifest: {e}")
    
    if not pairs:
        raise ValueError(f"No image pairs found for split '{args.split}'")
    
    # Sort pairs by filename (basename) for consistent ordering
    pairs = sorted(pairs, key=lambda x: x[1].stem)
    print(f"Sorted {len(pairs)} pairs by filename")
    
    if args.max_samples is not None:
        pairs = pairs[:args.max_samples]
        print(f"Limited to {len(pairs)} samples for testing")
    
    print(f"Found {len(pairs)} sRGB-XYZ pairs")
    
    # No output directory creation here; images are saved next to dataset under 'anchor-variants-imgs'
    
    # Prepare ISP parameters
    isp_params = {
        "augmentation_method": args.augmentation_method,
        "color_grading_range": tuple(args.color_grading_range),
        "contrast_range": tuple(args.contrast_range),
        "brightness_range": tuple(args.brightness_range) if args.brightness_range and len(args.brightness_range) == 2 else None,
        "tone_map_beta_mean": args.tone_map_beta_mean,
        "tone_map_beta_std": args.tone_map_beta_std,
        "tone_map_gamma_mean": args.tone_map_gamma_mean,
        "tone_map_gamma_std": args.tone_map_gamma_std,
        "seed": args.seed
    }
    
    # Generate ISP variants
    print("=" * 80)
    stats = generate_isp_variants(pairs, args, isp_params)
    
    # Final summary
    print(f"\nProcessing complete!")
    print(f"  Processed: {stats['pairs_processed']}")
    print(f"  Skipped: {stats['pairs_skipped']}")
    print(f"  Failed: {stats['pairs_failed']}")
    print(f"  Total variant images: {stats['pairs_processed'] * args.num_variations}")
    print(f"  Total anchor sRGB: {stats['pairs_processed']}")
    if args.generate_xyz_anchor:
        print(f"  Total anchor XYZ: {stats['pairs_processed']}")
    print(f"Manifest root: {meta.get('root', 'N/A')}")


if __name__ == "__main__":
    main()

