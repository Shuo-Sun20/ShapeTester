import keras
import numpy as np
from dataclasses import dataclass, field
from typing import Any, List

valid_test_case = {
    "inputs": np.random.randn(3, 4).astype(np.float32),
    "decimals": 2
}

@dataclass
class InputSpace:
    # The only parameter (other than "inputs") in call_func() is "decimals"
    decimals: List[Any] = field(
        default_factory=lambda: [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    )
    # Note: "decimals" is an integer parameter that does NOT affect the shape of the output tensor.
    # The output shape is always identical to the input shape for keras.ops.round.
    # However, per the instructions, we list all parameters except "inputs".