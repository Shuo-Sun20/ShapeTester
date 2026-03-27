import keras
import numpy as np

def call_func(inputs, k=1, axes=(0, 1)):
    return keras.ops.rot90(array=inputs, k=k, axes=axes)

# Generate random tensor as input
random_tensor = np.random.rand(3, 4, 5).astype(np.float32)
example_output = call_func(random_tensor, k=2, axes=(1, 2))