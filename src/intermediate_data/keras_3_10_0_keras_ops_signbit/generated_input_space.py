import keras
from dataclasses import dataclass, field
from typing import List

# Step 1: Define a valid test case
x = keras.ops.convert_to_tensor([[-0.5, 0.0, 0.5], [1.0, -1.0, 0.0], [0.3, -0.3, 0.0]])
valid_test_case = {"inputs": [x]}

# Steps 2-4: Define the InputSpace dataclass
@dataclass
class InputSpace:
    # There are no parameters in call_func (other than 'inputs') that affect the output shape.
    # The output shape of keras.ops.signbit(x) is solely determined by the shape of the input tensor x,
    # which is passed via the 'inputs' parameter. Since 'inputs' is excluded per the task instructions,
    # no additional parameters remain. Therefore, InputSpace is an empty dataclass.
    pass