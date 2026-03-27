import keras
import numpy as np

def call_func(inputs, factor=1.0, kernel_size=3, sigma=1.0, value_range=(0, 255), data_format=None, seed=None):
    layer = keras.layers.RandomGaussianBlur(
        factor=factor,
        kernel_size=kernel_size,
        sigma=sigma,
        value_range=value_range,
        data_format=data_format,
        seed=seed
    )
    return layer(inputs)

input_tensor = np.random.uniform(0, 255, size=(4, 32, 32, 3)).astype('float32')
example_output = call_func(
    inputs=input_tensor,
    factor=1.0,
    kernel_size=3,
    sigma=1.0,
    value_range=(0, 255),
    data_format=None,
    seed=None
)