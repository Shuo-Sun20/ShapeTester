import keras
import numpy as np

def call_func(inputs):
    return keras.ops.imag(inputs)

# Create random complex tensor using numpy and convert to keras tensor
real_part = np.random.randn(3, 4).astype(np.float32)
imag_part = np.random.randn(3, 4).astype(np.float32)
complex_np = real_part + 1j * imag_part
complex_tensor = keras.ops.convert_to_tensor(complex_np)

# Call function and save output
example_output = call_func(complex_tensor)