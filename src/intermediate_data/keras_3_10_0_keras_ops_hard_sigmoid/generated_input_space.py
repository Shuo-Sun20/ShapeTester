import numpy as np
from dataclasses import dataclass
from typing import List, Union

valid_test_case = {"inputs": np.array([-1.0, 0.0, 1.0])}

@dataclass
class InputSpace:
    # Note: For keras.ops.hard_sigmoid, only the 'inputs' parameter affects output shape,
    # but the instruction explicitly asks to exclude 'inputs'. 
    # There are no other parameters that affect output shape.
    # Therefore, this class contains no parameters.
    pass