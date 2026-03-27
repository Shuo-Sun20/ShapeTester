import keras
import numpy as np

def call_func(inputs, indices_or_sections, axis=0):
    return keras.ops.split(x=inputs, indices_or_sections=indices_or_sections, axis=axis)

example_output = call_func(
    inputs=keras.ops.convert_to_tensor(np.random.randn(12, 3)),
    indices_or_sections=[3, 6, 9],
    axis=0
)