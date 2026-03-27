import keras
import numpy as np

def call_func(inputs, pad_width, mode="constant", constant_values=None):
    return keras.ops.pad(x=inputs, pad_width=pad_width, mode=mode, constant_values=constant_values)

example_output = call_func(
    inputs=keras.ops.convert_to_tensor(np.random.randn(3, 4, 5)),
    pad_width=((1, 2), (3, 4), (5, 6)),
    mode="constant",
    constant_values=0.5
)