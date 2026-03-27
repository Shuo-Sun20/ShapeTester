import keras
import numpy as np

def call_func(inputs):
    return keras.ops.tanh_shrink(inputs)

# Generate a random tensor
np.random.seed(42)
random_tensor = np.random.randn(4, 3).astype(np.float32)

# Call the function and save the output
example_output = call_func(random_tensor)