import keras
import numpy as np

def call_func(factor, inputs, value_range=(0, 255), data_format=None, seed=None):
    layer = keras.layers.RandomPosterization(
        factor=factor,
        value_range=value_range,
        data_format=data_format,
        seed=seed
    )
    if isinstance(inputs, list):
        output = layer(*inputs)
    else:
        output = layer(inputs)
    return output

input_tensor = np.random.uniform(0, 255, size=(2, 32, 32, 3)).astype(np.float32)
example_output = call_func(factor=4, inputs=input_tensor)