import keras
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Union

# Generate random input tensors
np.random.seed(42)
data = keras.ops.convert_to_tensor(np.random.randn(6).astype(np.float32))
segment_ids = keras.ops.convert_to_tensor([0, 0, 1, 1, 2, 2])

def call_func(inputs, num_segments=None, sorted=False):
    data = inputs[0]
    segment_ids = inputs[1]
    return keras.ops.segment_max(data, segment_ids, num_segments, sorted)

# 1. Valid test case
valid_test_case = {
    "inputs": [data, segment_ids],
    "num_segments": 3,
    "sorted": False
}

# 2. & 3. Parameters affecting output shape (except "inputs")
# - num_segments: integer or None, affects output shape directly
# - sorted: boolean, does NOT affect output shape (only affects internal computation)

# 4. InputSpace dataclass
@dataclass
class InputSpace:
    # Only num_segments affects output shape (sorted does not)
    num_segments: List[Optional[int]] = field(
        default_factory=lambda: [
            # Boundary and typical values covering all legal scenarios
            None,  # Let API infer from segment_ids
            1,     # Minimum valid segments (less than actual max segment_id may cause issues)
            2,     # Less than actual segments (boundary)
            3,     # Equal to actual segments (valid_test_case value)
            4,     # More than actual segments (extra segments will be zero)
            10,    # Significantly more than actual segments
            0,     # Boundary case (zero segments - may cause errors)
            -1,    # Invalid negative value (for error testing)
        ]
    )