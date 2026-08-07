"""
Logging utilities for wandb and TensorBoard.

This module provides unified logging functions that handle both wandb and TensorBoard
logging, reducing code duplication across training scripts.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

import numpy as np
from PIL import Image
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Import is_main_process - handle both direct import and relative import
try:
    from .utils import is_main_process
except ImportError:
    import torch.distributed as dist
    def is_main_process():
        """Check if current process is main process (rank 0)."""
        if dist.is_initialized():
            return dist.get_rank() == 0
        return True  # If DDP not initialized, assume main process


def _is_numeric(value: Any) -> bool:
    """Check if a value is numeric and can be logged to TensorBoard."""
    return isinstance(value, (int, float, complex)) and not isinstance(value, bool)


class Logger:
    """Unified logger for wandb and TensorBoard."""
    
    def __init__(self, args, default_run_name_prefix: str = "run"):
        """
        Initialize logger with wandb and TensorBoard support.
        
        Args:
            args: Argument parser object with logging configuration
            default_run_name_prefix: Default prefix for run name if wandb is not used
        """
        self.args = args
        self.default_run_name_prefix = default_run_name_prefix
        self.tb_writers: List[SummaryWriter] = []
        self.run_name: Optional[str] = None
        self.yymmdd: Optional[str] = None
        
        # Initialize wandb if requested
        if is_main_process() and args.use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_run_name,
                config=vars(args),
            )
            if wandb.run:
                print(f"wandb initialized: {wandb.run.name}")
        
        # Generate run name
        self._init_run_name()
    
    def _init_run_name(self):
        """Initialize run name and broadcast to all processes."""
        yymmdd, run_name = None, None
        if is_main_process():
            yymmdd = time.strftime("%y%m%d")
            if self.args.wandb_run_name:
                run_name = str(self.args.wandb_run_name)
            elif self.args.use_wandb and WANDB_AVAILABLE and wandb.run:
                run_name = wandb.run.name
            else:
                run_name = f"{self.default_run_name_prefix}-{time.strftime('%H%M%S')}"
        
        # Broadcast to all processes
        obj_list = [yymmdd, run_name]
        if dist.is_initialized():
            dist.broadcast_object_list(obj_list, src=0)
        yymmdd, run_name = obj_list
        
        self.yymmdd = yymmdd
        self.run_name = run_name
        
        # Initialize TensorBoard writers after run_name is determined
        if is_main_process() and self.args.use_tb:
            for tb_dir in self.args.tb_root_dir:
                tb_path = Path(tb_dir) / run_name
                tb_path.mkdir(parents=True, exist_ok=True)
                writer = SummaryWriter(str(tb_path))
                self.tb_writers.append(writer)
                print(f"TensorBoard writer initialized: {tb_path}")
    
    def log(self, log_dict: Dict[str, Any], step: Optional[int] = None, exclude_keys: Optional[List[str]] = None):
        """
        Log metrics to both wandb and TensorBoard.
        
        Args:
            log_dict: Dictionary of metric names to values
            step: Global step (if None, will try to extract from log_dict["train/step"] or "val/step")
            exclude_keys: Keys to exclude from TensorBoard logging (e.g., "train/step")
        """
        if not is_main_process():
            return
        
        if exclude_keys is None:
            exclude_keys = ["train/step", "val/step", "train/epoch", "val/epoch"]
        
        # Determine step from log_dict if not provided
        if step is None:
            step = log_dict.get("train/step") or log_dict.get("val/step")
        
        # Log to wandb
        if self.args.use_wandb and WANDB_AVAILABLE:
            wandb.log(log_dict)
        
        # Log to TensorBoard (only numeric values)
        if self.args.use_tb and step is not None:
            for writer in self.tb_writers:
                for key, value in log_dict.items():
                    if key not in exclude_keys and _is_numeric(value):
                        writer.add_scalar(key, value, step)
                # Flush to ensure logs are written to disk
                writer.flush()
    
    def log_at_step_0(self, log_dict: Dict[str, Any]):
        """Log metrics at step 0 (typically for model/config info)."""
        if not is_main_process():
            return
        
        # Log to wandb
        if self.args.use_wandb and WANDB_AVAILABLE:
            wandb.log(log_dict)
        
        # Log to TensorBoard (only numeric values)
        if self.args.use_tb:
            for writer in self.tb_writers:
                for key, value in log_dict.items():
                    if _is_numeric(value):
                        writer.add_scalar(key, value, 0)
                # Flush to ensure logs are written to disk
                writer.flush()
    
    def close(self):
        """Close all loggers."""
        if not is_main_process():
            return
        
        if self.args.use_wandb and WANDB_AVAILABLE:
            wandb.finish()
        
        if self.args.use_tb:
            for writer in self.tb_writers:
                writer.close()
            if self.tb_writers:
                print("TensorBoard writers closed.")
    
    def log_image(self, tag: str, image_path: Union[str, Path], step: int, caption: Optional[str] = None):
        """
        Log image to both wandb and TensorBoard.
        
        Args:
            tag: Tag/name for the image (e.g., "val/viz", "train/viz")
            image_path: Path to the image file
            step: Global step
            caption: Optional caption for the image
        """
        if not is_main_process():
            return
        
        image_path = Path(image_path)
        if not image_path.exists():
            print(f"[WARN] Image file not found: {image_path}")
            return
        
        # Log to wandb
        if self.args.use_wandb and WANDB_AVAILABLE:
            try:
                if caption:
                    wandb.log({tag: wandb.Image(str(image_path), caption=caption)})
                else:
                    wandb.log({tag: wandb.Image(str(image_path))})
            except Exception as e:
                print(f"[WARN] Failed to log image to wandb: {e}")
        
        # Log to TensorBoard
        if self.args.use_tb:
            try:
                # Load image and convert to numpy array
                pil_image = Image.open(image_path)
                # Convert to RGB if necessary
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                img_array = np.array(pil_image)
                # TensorBoard expects [H, W, C] format with values in [0, 255]
                # Ensure values are in valid range
                if img_array.dtype != np.uint8:
                    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
                
                for writer in self.tb_writers:
                    writer.add_image(tag, img_array, step, dataformats='HWC')
                # Flush to ensure logs are written to disk
                for writer in self.tb_writers:
                    writer.flush()
            except Exception as e:
                print(f"[WARN] Failed to log image to TensorBoard: {e}")
    
    def get_base_dir(self, results_dir: str = None) -> Path:
        """
        Get base directory for results: {results_dir}/{yymmdd}-{run_name}.
        
        Args:
            results_dir: Root directory for results. If None, uses args.results_root if available,
                        otherwise defaults to "../results"
        """
        if results_dir is None:
            # Try to get from args if available
            if hasattr(self.args, 'results_root'):
                results_dir = self.args.results_root
            else:
                results_dir = "../results"
        return Path(results_dir) / f"{self.yymmdd}-{self.run_name}"


def init_logger(args, default_run_name_prefix: str = "run") -> Logger:
    """
    Initialize and return a Logger instance.
    
    Args:
        args: Argument parser object with logging configuration
        default_run_name_prefix: Default prefix for run name if wandb is not used
    
    Returns:
        Logger instance
    """
    return Logger(args, default_run_name_prefix)

