import keras
import numpy as np

def call_func(
    pool_size,
    inputs,
    strides=None,
    padding="valid",
    data_format=None,
    name=None
):
    layer = keras.layers.AveragePooling1D(
        pool_size=pool_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        name=name
    )
    output = layer(inputs)
    return output

input_tensor = np.random.rand(2, 10, 3).astype('float32')
example_output = call_func(pool_size=2, inputs=input_tensor)