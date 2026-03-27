import keras
import numpy as np
from dataclasses import dataclass, field
from typing import List

valid_test_case = {'inputs': keras.ops.convert_to_tensor(5)}

@dataclass
class InputSpace:
    inputs: List[int] = field(default_factory=lambda: [1, 2, 5, 10, 20])