import keras
import numpy as np

def call_func(
    inputs,
    size,
    strides=None,
    dilation_rate=1,
    padding="valid",
    data_format=None
):
    return keras.ops.image.extract_patches(
        images=inputs,
        size=size,
        strides=strides,
        dilation_rate=dilation_rate,
        padding=padding,
        data_format=data_format
    )

example_input = np.random.random((2, 20, 20, 3)).astype("float32")
example_output = call_func(inputs=example_input, size=(5, 5))