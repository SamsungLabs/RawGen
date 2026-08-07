#!/usr/bin/env python3
"""Split sRGB/XYZ paired datasets into train/val and write a JSON manifest.

Dataset layout:
  <root>/{dataset_type}/{sRGB,XYZ}/filename{_srgb.png,_xyz.png}

For each dataset_type, we collect pairs by basename (filename without the
suffix), shuffle deterministically with a given seed, split by train_ratio,
and aggregate across all dataset types into a single JSON file with metadata.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random


@dataclass
class PairItem:
    dataset_type: str
    basename: str
    srgb: Path
    xyz: Path

    def to_json_dict(self) -> Dict[str, str]:
        # Store only minimal info; consumers can reconstruct full paths
        return {
            "dataset_type": self.dataset_type,
            "basename": self.basename,
        }


def discover_dataset_types(root: Path, specified_types: Optional[List[str]], quiet: bool) -> List[str]:
    if specified_types:
        dataset_types: List[str] = []
        for dataset_type in specified_types:
            ds_dir = root / dataset_type
            if not ds_dir.is_dir():
                if not quiet:
                    print(f"[WARN] dataset_type '{dataset_type}' not found under {root}", file=sys.stderr)
                continue
            dataset_types.append(dataset_type)
        return dataset_types

    # Auto-discover: any immediate subdirectory of root that contains sRGB and XYZ subdirs
    discovered: List[str] = []
    if not root.is_dir():
        raise FileNotFoundError(f"Root directory not found: {root}")
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        if (child / "sRGB").is_dir() and (child / "XYZ").is_dir():
            discovered.append(child.name)
        else:
            # Skip silently in quiet mode
            if not quiet:
                print(f"[INFO] Skipping '{child.name}' (missing 'sRGB' or 'XYZ' dir)")
    return discovered


def collect_pairs_for_dataset(
    root: Path,
    dataset_type: str,
    srgb_suffix: str,
    xyz_suffix: str,
    quiet: bool,
) -> Tuple[List[PairItem], Dict[str, int]]:
    srgb_dir = root / dataset_type / "sRGB"
    xyz_dir = root / dataset_type / "XYZ"

    if not srgb_dir.is_dir() or not xyz_dir.is_dir():
        if not quiet:
            print(f"[WARN] Dataset '{dataset_type}' missing 'sRGB' or 'XYZ' directory; skipping.", file=sys.stderr)
        return [], {"total_srgb": 0, "total_xyz": 0, "matched": 0, "unmatched_srgb": 0, "unmatched_xyz": 0}

    # Index XYZ files by basename
    xyz_index: Dict[str, Path] = {}
    total_xyz = 0
    for path in sorted(xyz_dir.glob(f"*{xyz_suffix}")):
        total_xyz += 1
        base = path.name[: -len(xyz_suffix)] if path.name.endswith(xyz_suffix) else None
        if base is None:
            continue
        xyz_index[base] = path

    # Walk sRGB and match
    pairs: List[PairItem] = []
    unmatched_srgb = 0
    total_srgb = 0
    for path in sorted(srgb_dir.glob(f"*{srgb_suffix}")):
        total_srgb += 1
        if not path.name.endswith(srgb_suffix):
            unmatched_srgb += 1
            continue
        base = path.name[: -len(srgb_suffix)]
        xyz_path = xyz_index.get(base)
        if xyz_path is None:
            unmatched_srgb += 1
            if not quiet:
                print(f"[WARN] No XYZ match for sRGB file: {path.name}", file=sys.stderr)
            continue
        pairs.append(PairItem(dataset_type=dataset_type, basename=base, srgb=path.resolve(), xyz=xyz_path.resolve()))

    # Also count XYZ files without sRGB matches for reporting
    unmatched_xyz = 0
    srgb_bases = {p.basename for p in pairs}
    for base in xyz_index.keys():
        if base not in srgb_bases:
            unmatched_xyz += 1

    stats = {
        "total_srgb": total_srgb,
        "total_xyz": total_xyz,
        "matched": len(pairs),
        "unmatched_srgb": unmatched_srgb,
        "unmatched_xyz": unmatched_xyz,
    }

    if not quiet:
        print(
            f"[INFO] Dataset '{dataset_type}': sRGB={total_srgb}, XYZ={total_xyz}, matched={len(pairs)}, "
            f"unmatched_srgb={unmatched_srgb}, unmatched_xyz={unmatched_xyz}"
        )

    return pairs, stats


def split_pairs(pairs: List[PairItem], train_ratio: float, val_ratio: float, seed: int) -> Tuple[List[PairItem], List[PairItem], List[PairItem]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1 (exclusive)")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1 (exclusive)")
    if train_ratio + val_ratio > 1.0:
        raise ValueError("train_ratio + val_ratio must not exceed 1.0")
    
    rng = random.Random(seed)
    shuffled = list(pairs)
    rng.shuffle(shuffled)
    
    train_count = int(len(shuffled) * train_ratio)
    val_count = int(len(shuffled) * val_ratio)
    
    train_list = shuffled[:train_count]
    val_list = shuffled[train_count:train_count + val_count]
    test_list = shuffled[train_count + val_count:]
    
    return train_list, val_list, test_list


def load_existing_json(json_path: Path) -> Tuple[Dict[str, object], List[PairItem], List[PairItem], List[PairItem]]:
    """Load existing JSON file and return metadata, train, val, and test lists."""
    if not json_path.exists():
        return {}, [], [], []
    
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    meta = data.get("meta", {})
    train_data = data.get("train", [])
    val_data = data.get("val", [])
    test_data = data.get("test", [])
    
    # Convert dict data back to PairItem objects
    train_items = []
    val_items = []
    test_items = []
    
    for item_dict in train_data:
        train_items.append(PairItem(
            dataset_type=item_dict["dataset_type"],
            basename=item_dict["basename"],
            srgb=Path(),  # Will be reconstructed when needed
            xyz=Path()    # Will be reconstructed when needed
        ))
    
    for item_dict in val_data:
        val_items.append(PairItem(
            dataset_type=item_dict["dataset_type"],
            basename=item_dict["basename"],
            srgb=Path(),  # Will be reconstructed when needed
            xyz=Path()    # Will be reconstructed when needed
        ))
    
    for item_dict in test_data:
        test_items.append(PairItem(
            dataset_type=item_dict["dataset_type"],
            basename=item_dict["basename"],
            srgb=Path(),  # Will be reconstructed when needed
            xyz=Path()    # Will be reconstructed when needed
        ))
    
    return meta, train_items, val_items, test_items


def merge_split_data(
    existing_train: List[PairItem],
    existing_val: List[PairItem],
    existing_test: List[PairItem],
    new_train: List[PairItem],
    new_val: List[PairItem],
    new_test: List[PairItem],
    target_dataset_types: List[str],
) -> Tuple[List[PairItem], List[PairItem], List[PairItem]]:
    """Merge existing data with new data, overwriting specified dataset types."""
    # Filter out existing data from target dataset types
    filtered_existing_train = [item for item in existing_train if item.dataset_type not in target_dataset_types]
    filtered_existing_val = [item for item in existing_val if item.dataset_type not in target_dataset_types]
    filtered_existing_test = [item for item in existing_test if item.dataset_type not in target_dataset_types]
    
    # Combine filtered existing data with new data
    merged_train = filtered_existing_train + new_train
    merged_val = filtered_existing_val + new_val
    merged_test = filtered_existing_test + new_test
    
    return merged_train, merged_val, merged_test


def write_split_json(
    output_path: Path,
    meta: Dict[str, object],
    train_list: List[PairItem],
    val_list: List[PairItem],
    test_list: List[PairItem],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": meta,
        "train": [p.to_json_dict() for p in train_list],
        "val": [p.to_json_dict() for p in val_list],
        "test": [p.to_json_dict() for p in test_list],
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Split sRGB/XYZ paired datasets into train/val JSON manifest")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("./data"),
        help="Root path of dataset directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("../trainval.json"),
        help="Output JSON path (default: ../trainval.json)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train ratio in (0,1)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation ratio in (0,1)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic split")
    parser.add_argument(
        "--dataset-types",
        nargs="*",
        default=None,
        help="Optional list of dataset types to include (default: auto-discover)",
    )
    parser.add_argument("--srgb-suffix", type=str, default="_srgb.png", help="Suffix for sRGB files")
    parser.add_argument("--xyz-suffix", type=str, default="_xyz.png", help="Suffix for XYZ files")
    parser.add_argument("--quiet", action="store_true", help="Reduce logging output")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_argparser().parse_args(argv)

    root: Path = args.root
    output_path: Path = args.output or (root / "trainval.json")
    train_ratio: float = args.train_ratio
    val_ratio: float = args.val_ratio
    seed: int = args.seed
    dataset_types: Optional[List[str]] = args.dataset_types
    srgb_suffix: str = args.srgb_suffix
    xyz_suffix: str = args.xyz_suffix
    quiet: bool = args.quiet

    if not 0.0 < train_ratio < 1.0:
        print("[ERROR] --train-ratio must be in (0,1)", file=sys.stderr)
        return 2
    if not 0.0 < val_ratio < 1.0:
        print("[ERROR] --val-ratio must be in (0,1)", file=sys.stderr)
        return 2
    if train_ratio + val_ratio > 1.0:
        print("[ERROR] --train-ratio + --val-ratio must not exceed 1.0", file=sys.stderr)
        return 2

    # Load existing JSON if it exists
    existing_meta, existing_train, existing_val, existing_test = load_existing_json(output_path)
    
    try:
        types = discover_dataset_types(root, dataset_types, quiet)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 2

    if not types:
        print(f"[ERROR] No dataset types found under {root}", file=sys.stderr)
        return 2

    # Process new datasets
    new_train: List[PairItem] = []
    new_val: List[PairItem] = []
    new_test: List[PairItem] = []
    per_type_counts: Dict[str, Dict[str, int]] = {}
    per_type_total_pairs: Dict[str, int] = {}

    for dataset_type in types:
        pairs, stats = collect_pairs_for_dataset(root, dataset_type, srgb_suffix, xyz_suffix, quiet)
        total_pairs = len(pairs)
        per_type_total_pairs[dataset_type] = total_pairs
        if total_pairs == 0:
            if not quiet:
                print(f"[WARN] No matched pairs for dataset '{dataset_type}'. Skipping split.", file=sys.stderr)
            per_type_counts[dataset_type] = {"total": 0, "train": 0, "val": 0, "test": 0}
            continue

        train_list, val_list, test_list = split_pairs(pairs, train_ratio, val_ratio, seed)
        new_train.extend(train_list)
        new_val.extend(val_list)
        new_test.extend(test_list)
        per_type_counts[dataset_type] = {
            "total": total_pairs,
            "train": len(train_list),
            "val": len(val_list),
            "test": len(test_list),
        }

    if not new_train and not new_val:
        print("[ERROR] No pairs found in any dataset. Nothing to write.", file=sys.stderr)
        return 1

    # Merge with existing data
    all_train, all_val, all_test = merge_split_data(existing_train, existing_val, existing_test, new_train, new_val, new_test, types)

    # Update metadata
    created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    
    # Get existing dataset types and counts
    existing_dataset_types = existing_meta.get("dataset_types", [])
    existing_counts = existing_meta.get("counts", {})
    
    # Update dataset types (remove duplicates, preserve order)
    all_dataset_types = []
    for dt in existing_dataset_types:
        if dt not in types:  # Keep existing types that are not being updated
            all_dataset_types.append(dt)
    all_dataset_types.extend(types)  # Add new/updated types
    
    # Update counts
    updated_counts = {}
    for dt, counts in existing_counts.items():
        if dt != "all" and dt not in types:  # Keep existing counts for non-updated datasets
            updated_counts[dt] = counts
    
    # Add new/updated counts
    updated_counts.update(per_type_counts)
    
    # Calculate new totals
    counts_all = {
        "total": sum(c.get("total", 0) for c in updated_counts.values() if isinstance(c, dict)),
        "train": sum(c.get("train", 0) for c in updated_counts.values() if isinstance(c, dict)),
        "val": sum(c.get("val", 0) for c in updated_counts.values() if isinstance(c, dict)),
        "test": sum(c.get("test", 0) for c in updated_counts.values() if isinstance(c, dict)),
    }
    updated_counts["all"] = counts_all

    meta: Dict[str, object] = {
        "root": str(root.resolve()),
        "dataset_types": all_dataset_types,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "seed": seed,
        "srgb_suffix": srgb_suffix,
        "xyz_suffix": xyz_suffix,
        "created_at": created_at,
        "counts": updated_counts,
    }

    write_split_json(output_path, meta, all_train, all_val, all_test)

    if not quiet:
        print(
            f"[INFO] Wrote split JSON to: {output_path} (train={len(all_train)}, val={len(all_val)}, test={len(all_test)}, total={len(all_train)+len(all_val)+len(all_test)})"
        )
        if existing_train or existing_val or existing_test:
            print(f"[INFO] Updated datasets: {types}")
            print(f"[INFO] Preserved datasets: {[dt for dt in existing_dataset_types if dt not in types]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
