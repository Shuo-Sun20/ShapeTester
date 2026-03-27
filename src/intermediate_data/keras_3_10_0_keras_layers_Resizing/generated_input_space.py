import keras
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Union

# 1. Valid test case
valid_test_case = {
    "height": 32,
    "width": 32,
    "inputs": np.random.rand(2, 64, 64, 3).astype(np.float32),
    "interpolation": "bilinear",
    "crop_to_aspect_ratio": False,
    "pad_to_aspect_ratio": False,
    "fill_mode": "constant",
    "fill_value": 0.0,
    "antialias": False,
    "data_format": None,
    "name": None,
    "dtype": None,
    "trainable": None
}

# 2 & 3. Parameters affecting output tensor shape (excluding "inputs")
# Height and width: continuous integer parameters
# data_format: discrete parameter with 2 values
# crop_to_aspect_ratio and pad_to_aspect_ratio: boolean parameters that can affect content but not final dimensions
# Note: While crop_to_aspect_ratio and pad_to_aspect_ratio don't change output dimensions,
# they affect how the image content is transformed to fit those dimensions.

# 4. InputSpace dataclass
@dataclass
class InputSpace:
    """Dataclass containing all parameters that affect the output tensor shape."""
    
    # Height parameter: integer, affects output height dimension
    # Typical values including boundaries and intermediate values
    height: List[int] = None
    
    # Width parameter: integer, affects output width dimension  
    # Typical values including boundaries and intermediate values
    width: List[int] = None
    
    # data_format: string, affects dimension ordering
    # Discrete parameter with 2 possible values
    data_format: List[Optional[str]] = None
    
    def __post_init__(self):
        # Set default discretized value ranges if not provided
        if self.height is None:
            # Include boundary values (1, very small) and typical values
            # 1 is minimum valid size, 224 is common in ImageNet models
            # 299 for Inception, 32/64 for CIFAR, 512 for higher res
            self.height = [1, 32, 64, 128, 224, 299, 512, 1024]
        
        if self.width is None:
            # Same discretization as height for consistency
            self.width = [1, 32, 64, 128, 224, 299, 512, 1024]
        
        if self.data_format is None:
            # Only two valid values according to documentation
            self.data_format = [None, "channels_last", "channels_first"]