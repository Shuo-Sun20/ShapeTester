import keras

def call_func(inputs, axis=None, keepdims=False):
    # keras.ops.mean is a function, not a class
    # Directly call the API with the provided parameters
    # 'inputs' corresponds to the 'x' parameter of keras.ops.mean
    return keras.ops.mean(x=inputs, axis=axis, keepdims=keepdims)

# Generate random input tensor
import numpy as np
np.random.seed(42)  # For reproducibility
random_tensor = keras.ops.convert_to_tensor(np.random.randn(3, 4, 5))

# Call function with example parameters
example_output = call_func(inputs=random_tensor, axis=1, keepdims=True)