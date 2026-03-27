import keras
import numpy as np

def call_func(inputs):
    return keras.ops.view_as_complex(inputs)

real_imag = np.random.randn(3, 2)
example_output = call_func(real_imag)