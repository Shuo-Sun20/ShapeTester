import keras
from dataclasses import dataclass, field
import numpy as np

# 1. Define valid_test_case
abs_tensor = keras.random.normal((3, 4))
angle_tensor = keras.random.normal((3, 4))
valid_test_case = {"inputs": [abs_tensor, angle_tensor]}

# 2. Parameters affecting output shape: The list 'inputs' contains two tensors
# 3. Value space analysis for tensor shapes
# Discretized shapes covering various dimensions/scenarios
shapes = [
    (0,),           # 0-D scalar
    (1,),           # 1-D size 1
    (5,),           # 1-D typical
    (1, 1),         # 2-D size 1x1
    (2, 3),         # 2-D typical
    (1, 1, 1),      # 3-D size 1x1x1
    (4, 5, 6),      # 3-D typical
    (2, 3, 4, 5),   # 4-D
    (3, 4)          # The example shape from valid_test_case
]

# Generate all possible input pairs for the discretized shapes
value_space_for_inputs = [valid_test_case["inputs"]]  # Start with the valid test case

for shape in shapes:
    if shape != (3, 4):  # Avoid duplicate of the valid_test_case shape
        abs_tensor = keras.random.normal(shape)
        angle_tensor = keras.random.normal(shape)
        value_space_for_inputs.append([abs_tensor, angle_tensor])

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    inputs: list = field(default_factory=lambda: value_space_for_inputs)