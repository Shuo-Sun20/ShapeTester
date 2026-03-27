import keras
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union, List

# 1. Define valid_test_case
valid_test_case = {
    "inputs": keras.random.uniform((32, 224, 224, 3), 0, 255),
    "value_range": (0, 255),
    "num_ops": 2,
    "factor": 0.5,
    "interpolation": "bilinear",
    "seed": None,
    "data_format": None
}

# 2. Identify shape-affecting parameters
# Based on analysis: None of the parameters in call_func (except "inputs") affect output tensor shape.
# RandAugment performs pixel-level and geometric transformations that preserve spatial dimensions.

# 3. Value space construction
# Since no shape-affecting parameters exist, we define InputSpace as an empty class.

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    pass