import keras
import numpy as np

def call_func(inputs, axisa=-1, axisb=-1, axisc=-1, axis=None):
    x1, x2 = inputs
    return keras.ops.cross(x1, x2, axisa=axisa, axisb=axisb, axisc=axisc, axis=axis)

# Generate random input tensors
np.random.seed(42)
x1_tensor = keras.ops.convert_to_tensor(np.random.randn(3, 4, 3).astype(np.float32))
x2_tensor = keras.ops.convert_to_tensor(np.random.randn(3, 4, 3).astype(np.float32))

# Call function and save output
example_output = call_func([x1_tensor, x2_tensor], axis=-1)