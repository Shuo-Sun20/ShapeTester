import keras
from dataclasses import dataclass, field
from typing import List, Tuple, Union

# 1. Valid test case
x = keras.random.normal((3,))
valid_test_case = {
    "inputs": [x],
    "shape": (2, 3)
}

# 2-4. InputSpace dataclass
@dataclass
class InputSpace:
    # The only parameter affecting output shape (besides inputs) is 'shape'
    shape: List[Union[int, Tuple[int, ...]]] = field(
        default_factory=lambda: [
            # 1D shapes
            3,
            (3,),
            # 2D shapes
            (1, 3),
            (2, 3),
            (3, 3)
        ]
    )