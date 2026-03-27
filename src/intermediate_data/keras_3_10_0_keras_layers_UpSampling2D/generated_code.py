import numpy as np
import keras

def call_func(size=(2, 2), data_format=None, interpolation="nearest", inputs=None):
    layer = keras.layers.UpSampling2D(
        size=size,
        data_format=data_format,
        interpolation=interpolation
    )
    output = layer(inputs)
    return output

input_tensor = np.random.randn(2, 4, 4, 3).astype(np.float32)
example_output = call_func(size=(2, 2), interpolation="bilinear", inputs=input_tensor)