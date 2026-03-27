import keras
import numpy as np

def call_func(
    pool_size,
    strides=None,
    padding="valid",
    data_format=None,
    name=None,
    kwargs=None,
    inputs=None
):
    if kwargs is None:
        kwargs = {}
    layer = keras.layers.AveragePooling2D(
        pool_size=pool_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        name=name,
        **kwargs
    )
    if isinstance(inputs, list):
        output = layer(*inputs)
    else:
        output = layer(inputs)
    return output

# Generate random input tensor (batch_size=2, height=8, width=8, channels=3)
input_tensor = np.random.randn(2, 8, 8, 3).astype(np.float32)
# Call function with example parameters
example_output = call_func(
    pool_size=(2, 2),
    strides=(2, 2),
    padding="valid",
    inputs=input_tensor
)