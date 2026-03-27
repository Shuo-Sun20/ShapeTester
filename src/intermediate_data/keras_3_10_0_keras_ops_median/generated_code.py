import keras
import numpy as np

def call_func(inputs, axis=None, keepdims=False):
    return keras.ops.median(x=inputs, axis=axis, keepdims=keepdims)

# Generate random tensor
np.random.seed(42)
random_tensor = np.random.randn(3, 4, 5).astype(np.float32)
# Call function
example_output = call_func(inputs=random_tensor, axis=1, keepdims=True)