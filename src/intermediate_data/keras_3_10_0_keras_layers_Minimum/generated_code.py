import keras
import numpy as np

def call_func(inputs, minimum_kwargs=None):
    if minimum_kwargs is None:
        minimum_kwargs = {}
    layer = keras.layers.Minimum(**minimum_kwargs)
    return layer(inputs)

# Generate random input tensors
tensor1 = np.random.rand(2, 3, 4).astype(np.float32)
tensor2 = np.random.rand(2, 3, 4).astype(np.float32)
tensor_list = [tensor1, tensor2]

# Call the function
example_output = call_func(tensor_list)