import keras
import numpy as np

def call_func(inputs, axis=None, keepdims=False):
    return keras.ops.amin(inputs[0], axis=axis, keepdims=keepdims)

# Generate random tensor
random_tensor = keras.ops.convert_to_tensor(np.random.randn(3, 4))

# Call function and save output
example_output = call_func(inputs=[random_tensor], axis=1, keepdims=False)