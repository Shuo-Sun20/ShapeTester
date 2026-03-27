import keras
import numpy as np

def call_func(factor, inputs, value_range=(0, 255), data_format=None, seed=None):
    layer = keras.layers.RandomColorDegeneration(
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

random_input = np.random.rand(2, 224, 224, 3).astype(np.float32)
example_output = call_func(factor=(0.2, 0.8), inputs=random_input)