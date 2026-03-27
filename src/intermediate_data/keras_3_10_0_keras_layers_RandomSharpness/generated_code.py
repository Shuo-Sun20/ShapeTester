import keras
import numpy as np

def call_func(inputs, factor, value_range=(0, 255), data_format=None, seed=None):
    layer = keras.layers.RandomSharpness(
        factor=factor,
        value_range=value_range,
        data_format=data_format,
        seed=seed
    )
    return layer(inputs)

# Generate random input tensor
input_tensor = np.random.rand(2, 64, 64, 3) * 255.0
example_output = call_func(input_tensor, factor=0.5)