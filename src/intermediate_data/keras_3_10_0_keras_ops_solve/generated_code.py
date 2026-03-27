import keras
import numpy as np

def call_func(inputs):
    a, b = inputs[0], inputs[1]
    return keras.ops.solve(a, b)

# Generate random valid inputs
np.random.seed(42)
batch_shape = (2, 3, 4)  # Example batch dimensions
M = 5  # Matrix dimension M
N = 2  # For the case when b is (..., M, N)

# Generate random tensor a with shape (..., M, M)
a_shape = batch_shape + (M, M)
a = np.random.randn(*a_shape).astype('float32')

# Generate random tensor b with shape (..., M) (or can be (..., M, N))
b_shape = batch_shape + (M,)
b = np.random.randn(*b_shape).astype('float32')

example_output = call_func([a, b])