import keras
import numpy as np
from dataclasses import dataclass, field

def call_func(target_shape, inputs):
    layer = keras.layers.Reshape(target_shape)
    return layer(inputs)

# Generate random input tensor with batch dimension
batch_size = 2
input_tensor = keras.random.normal((batch_size, 12))

# 1. valid_test_case definition
valid_test_case = {
    "target_shape": (3, 4),
    "inputs": input_tensor
}

# 2 & 3. Parameter analysis and discretization
# Only "target_shape" affects the output shape (besides inputs)

# 4. InputSpace definition
@dataclass
class InputSpace:
    # target_shape must have product = 12 for valid reshapes (batch 2, features 12)
    # Discretized to 5 representative values
    target_shape: list = field(default_factory=lambda: [
        (12,),           # 1D
        (3, 4),          # 2D - example from docs
        (2, 6),          # 2D alternative
        (2, 2, 3),       # 3D
        (-1, 2, 2)       # With inferred dimension
    ])