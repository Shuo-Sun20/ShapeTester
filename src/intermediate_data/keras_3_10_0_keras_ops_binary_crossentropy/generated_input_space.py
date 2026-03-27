import keras
from dataclasses import dataclass, field
from typing import List, Union
import numpy as np

# Task 1: Define valid_test_case
valid_test_case = {
    "inputs": [
        keras.ops.convert_to_tensor([0, 1, 1, 0], dtype="float32"),
        keras.ops.convert_to_tensor([0.1, 0.9, 0.8, 0.2], dtype="float32")
    ],
    "from_logits": False
}

# Tasks 2-4: Define InputSpace dataclass
@dataclass
class InputSpace:
    # Only parameter affecting output shape (except "inputs") is from_logits
    from_logits: List[bool] = field(default_factory=lambda: [True, False])
    
    # Note: 'inputs' contains target and output tensors which affect shape
    # but are excluded per instructions. from_logits doesn't affect shape
    # but is included as requested
    
    def __post_init__(self):
        # No additional validation needed as from_logits is only boolean
        pass

# The InputSpace can be instantiated as:
# var = InputSpace()