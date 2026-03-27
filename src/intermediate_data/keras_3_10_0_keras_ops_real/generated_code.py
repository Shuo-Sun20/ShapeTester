import keras
import numpy as np

def call_func(inputs):
    return keras.ops.real(inputs)

# Generate random complex tensor input
real_part = np.random.randn(3, 4).astype(np.float32)
imag_part = np.random.randn(3, 4).astype(np.float32)
complex_tensor = keras.ops.convert_to_tensor(real_part + 1j * imag_part)

example_output = call_func(complex_tensor)