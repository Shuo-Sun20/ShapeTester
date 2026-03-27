import keras
import numpy as np

def call_func(inputs, indexing="xy"):
    return keras.ops.meshgrid(*inputs, indexing=indexing)

# Generate random 1D tensors
x = keras.ops.convert_to_tensor(np.random.randn(3))
y = keras.ops.convert_to_tensor(np.random.randn(4))
z = keras.ops.convert_to_tensor(np.random.randn(5))

example_output = call_func(inputs=[x, y, z], indexing="ij")