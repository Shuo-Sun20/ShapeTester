import keras
import keras.ops as ops
import numpy as np

def call_func(inputs):
    return ops.slogdet(inputs[0])

# Generate random 3x3 matrix as input
x = ops.convert_to_tensor(np.random.randn(3, 3).astype(np.float32))
example_output = list(call_func([x]))