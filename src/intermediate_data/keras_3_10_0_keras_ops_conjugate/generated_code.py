import keras
import numpy as np

def call_func(inputs):
    return keras.ops.conjugate(inputs[0])

# Generate random complex tensor
real_part = keras.random.normal(shape=(3, 4))
imag_part = keras.random.normal(shape=(3, 4))
complex_np = np.array(real_part) + 1j * np.array(imag_part)
complex_tensor = keras.ops.convert_to_tensor(complex_np)

example_output = call_func([complex_tensor])