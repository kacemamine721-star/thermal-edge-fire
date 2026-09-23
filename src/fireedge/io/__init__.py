"""I/O package for thermal-edge-fire datasets."""
from .activefire import index_patches, read_image, read_mask, image_meta
from .tssatfire import list_geotiffs, describe, read_decimated
from .thraws import read_raw_granule, describe_thraws

__all__ = [
    "index_patches",
    "read_image",
    "read_mask",
    "image_meta",
    "list_geotiffs",
    "describe",
    "read_decimated",
    "read_raw_granule",
    "describe_thraws",
]
