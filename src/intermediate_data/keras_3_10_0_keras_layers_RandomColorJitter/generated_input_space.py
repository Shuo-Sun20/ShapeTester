import keras
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, Union, List

# 1. Define valid_test_case
valid_test_case = {
    'value_range': (0, 255),
    'brightness_factor': 0.2,
    'contrast_factor': 0.2,
    'saturation_factor': 0.2,
    'hue_factor': 0.2,
    'seed': 42,
    'data_format': 'channels_last',
    'inputs': np.random.rand(2, 224, 224, 3).astype(np.float32) * 255.0
}

@dataclass
class InputSpace:
    data_format: List[Optional[str]] = field(
        default_factory=lambda: ['channels_last', 'channels_first', None]
    )