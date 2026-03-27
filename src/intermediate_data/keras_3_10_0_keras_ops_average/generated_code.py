import keras
import numpy as np

def call_func(inputs, axis=None):
    if isinstance(inputs, list):
        if len(inputs) == 1:
            x = inputs[0]
            weights = None
        elif len(inputs) == 2:
            x, weights = inputs
        else:
            raise ValueError("Inputs list must contain 1 or 2 tensors")
    else:
        x = inputs
        weights = None
    
    return keras.ops.average(x=x, axis=axis, weights=weights)

# Create random input tensors
np.random.seed(42)
x_data = np.random.randn(3, 4).astype(np.float32)
weights_data = np.random.randn(4).astype(np.float32)

# Call with both x and weights
example_output = call_func(inputs=[x_data, weights_data], axis=1)