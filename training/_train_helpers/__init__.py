"""Training-only helper modules (datasets, DDP and logging utilities)."""
# utils namespace
from .utils import *  # noqa: F401,F403

# data namespace
from .dataset import *  # noqa: F401,F403


# Lazy import: keeps latent datasets off the import path unless used.
def __getattr__(name):
    if name == "VariantAnchorLatentDataset":
        from .latent_dataset import VariantAnchorLatentDataset
        return VariantAnchorLatentDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
