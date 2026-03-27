import keras
import numpy as np

def call_func(
    inputs,
    factor,
    value_range=(0, 255),
    data_format=None,
    seed=None
):
    layer = keras.layers.RandomHue(
        factor=factor,
        value_range=value_range,
        data_format=data_format,
        seed=seed
    )
    return layer(inputs)

np.random.seed(42)
example_input = np.random.rand(8, 32, 32, 3).astype('float32')
example_output = call_func(
    inputs=example_input,
    factor=0.5,
    value_range=[0, 1]
)