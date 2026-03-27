import keras
import numpy as np

def call_func(inputs, bins=10, range=None):
    x = inputs[0]
    return keras.ops.histogram(x, bins=bins, range=range)

# Generate random input tensor
x = keras.ops.convert_to_tensor(np.random.rand(8))
# Call the function and store output as list of tensors
example_output = list(call_func(inputs=[x]))