"""
Dataset classes and utilities for sRGB-XYZ image pairs.

This module provides a single unified dataset class with support for dataset
subset selection, split management (train/val/test), random cropping (train
only), resizing, and normalization modes. It returns tensors in the [-1,1]
range for both sRGB and XYZ to keep downstream code simple and consistent.
"""

import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import torch
import numpy as np

# Re-exported from utils.py for the training scripts.
from .utils import srgb01_to_m11, m11_to_01


def ensure_multiple_of(x: int, base=8) -> int:
    """Ensure x is a multiple of base."""
    return x - (x % base)


def load_manifest_pairs(
    manifest_path: Path, 
    dataset_subsets: Optional[List[str]] = None,
    split: str = "val",
    root_key: str = "root"
) -> Tuple[List[Tuple[Path, Path]], Dict[str, str]]:
    """
    Load image pairs from manifest with optional dataset subset filtering.
    Now supports selecting a split among {"train","val","test","all"}.
    When split="all", combines train+val+test splits.
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    
    # Validate split parameter
    if split not in ["train", "val", "test", "all"]:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', 'test', or 'all'")
    
    # Handle "all" split by combining train, val, and test
    if split == "all":
        items = []
        for split_name in ["train", "val", "test"]:
            items.extend(manifest.get(split_name, []))
    else:
        items = manifest.get(split, [])
    meta_root = Path(manifest.get("meta", {}).get(root_key, "."))
    srgb_suffix = manifest.get("meta", {}).get("srgb_suffix", "_srgb.png")
    xyz_suffix = manifest.get("meta", {}).get("xyz_suffix", "_xyz.png")
    
    # Get available dataset types from manifest
    available_types = set()
    for item in items:
        if "dataset_type" in item:
            available_types.add(item["dataset_type"])
    
    # Determine which dataset types to include
    if dataset_subsets is None or "all" in dataset_subsets:
        selected_types = available_types
    else:
        selected_types = set(dataset_subsets)
        invalid_types = selected_types - available_types
        if invalid_types:
            raise ValueError(f"Invalid dataset types: {invalid_types}. Available types: {sorted(available_types)}")
    
    pairs: List[Tuple[Path, Path]] = []
    for item in items:
        if "dataset_type" in item and item["dataset_type"] not in selected_types:
            continue
        if "srgb" in item and "xyz" in item:
            s_path = Path(item["srgb"])
            x_path = Path(item["xyz"])
        else:
            ds_type = item["dataset_type"]
            base = item["basename"]
            s_path = meta_root / ds_type / "sRGB" / f"{base}{srgb_suffix}"
            x_path = meta_root / ds_type / "XYZ" / f"{base}{xyz_suffix}"
        pairs.append((s_path, x_path))
    
    meta = {
        "root": str(meta_root),
        "srgb_suffix": srgb_suffix,
        "xyz_suffix": xyz_suffix,
    }
    
    return pairs, meta
