import keras
from dataclasses import dataclass
from typing import List, Tuple

# Valid test case dictionary
valid_test_case = {
    "inputs": [keras.random.uniform(shape=(3, 4))]
}

@dataclass
class InputSpace:
    """
    Dataclass containing all parameters that can affect the shape of the output tensor
    when calling call_func(), excluding 'inputs' parameter.
    """
    # Since call_func() only has 'inputs' parameter and we're excluding it,
    # there are no other parameters that affect output shape
    pass